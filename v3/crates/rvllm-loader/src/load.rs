//! Top-level weight loader: HF safetensors dir -> HbmArena + LoadedModel.
//!
//! Per-tensor FP8 quant runs on the CPU (reference path; one-time cost
//! at engine init). `check_clamp_gate` rejects mis-scaled weights.
//! f16/bf16 weights upload straight through.

use std::collections::BTreeMap;
use std::path::Path;

use half::f16;
use memmap2::Mmap;
use rvllm_core::{DType, LoaderCtx, LoaderError, Result, RvllmError};
use rvllm_mem::HbmArena;

use crate::fp8_quant::{check_clamp_gate, quantize_per_tensor_ref, FP8_E4M3_MAX};
use crate::safetensors::{ShardHeader, ShardIndex, TensorEntry};
use crate::weights::{F16Weight, Fp8Weight, LayerWeights, LoadedModel};

/// Minimal model config read from the loaded directory's `config.json`.
#[derive(Clone, Debug)]
pub struct ModelArch {
    pub num_hidden_layers: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
}

impl ModelArch {
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let p = dir.join("config.json");
        let bytes = std::fs::read(&p).map_err(|source| RvllmError::Io {
            err: rvllm_core::IoError::from(&source),
            path: p.clone(),
            source,
        })?;
        let v: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| RvllmError::Loader {
            err: LoaderError::Corrupt {
                detail: format!("config.json: {e}"),
            },
            ctx: LoaderCtx {
                path: p.clone(),
                tensor: None,
            },
            bt: std::backtrace::Backtrace::capture(),
        })?;
        let num_hidden_layers = v["num_hidden_layers"].as_u64().unwrap_or(0) as usize;
        let hidden_size = v["hidden_size"].as_u64().unwrap_or(0) as usize;
        let num_attention_heads = v["num_attention_heads"].as_u64().unwrap_or(0) as usize;
        let num_key_value_heads = v["num_key_value_heads"]
            .as_u64()
            .unwrap_or(num_attention_heads as u64) as usize;
        let intermediate_size = v["intermediate_size"].as_u64().unwrap_or(0) as usize;
        let vocab_size = v["vocab_size"].as_u64().unwrap_or(0) as usize;
        let rope_theta = v["rope_theta"].as_f64().unwrap_or(10000.0) as f32;
        let max_position_embeddings =
            v["max_position_embeddings"].as_u64().unwrap_or(2048) as usize;
        let head_dim = if num_attention_heads == 0 {
            0
        } else {
            hidden_size / num_attention_heads
        };
        if head_dim != 128 {
            return Err(RvllmError::Loader {
                err: LoaderError::Corrupt {
                    detail: format!(
                        "v3 requires head_dim == 128, got {head_dim} (hidden={hidden_size}, heads={num_attention_heads})"
                    ),
                },
                ctx: LoaderCtx {
                    path: p,
                    tensor: None,
                },
                bt: std::backtrace::Backtrace::capture(),
            });
        }
        Ok(Self {
            num_hidden_layers,
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            intermediate_size,
            vocab_size,
            rope_theta,
            max_position_embeddings,
        })
    }
}

/// Per-shard mmap + parsed header. Keeping both alive keeps the bytes
/// live.
struct ShardMap {
    _mmap: Mmap,
    header: ShardHeader,
}

impl ShardMap {
    fn open(path: &Path) -> Result<Self> {
        let f = std::fs::File::open(path).map_err(|source| RvllmError::Io {
            err: rvllm_core::IoError::from(&source),
            path: path.to_path_buf(),
            source,
        })?;
        let mmap = unsafe { Mmap::map(&f) }.map_err(|source| RvllmError::Io {
            err: rvllm_core::IoError::from(&source),
            path: path.to_path_buf(),
            source,
        })?;
        let header = ShardHeader::parse(path, &mmap)?;
        Ok(Self {
            _mmap: mmap,
            header,
        })
    }

    fn bytes(&self) -> &[u8] {
        &self._mmap
    }
}

/// Walk the shards for `model_dir`. Returns one big `(name -> (shard, entry))` map.
fn build_tensor_index(
    model_dir: &Path,
) -> Result<(Vec<ShardMap>, BTreeMap<String, (usize, TensorEntry)>)> {
    let idx = ShardIndex::resolve(model_dir)?;
    let mut shards = Vec::with_capacity(idx.shards.len());
    for p in &idx.shards {
        shards.push(ShardMap::open(p)?);
    }
    let mut by_name: BTreeMap<String, (usize, TensorEntry)> = BTreeMap::new();
    for (si, sm) in shards.iter().enumerate() {
        for (name, entry) in &sm.header.tensors {
            by_name.insert(name.clone(), (si, entry.clone()));
        }
    }
    Ok((shards, by_name))
}

/// Load the whole model into `arena`. CPU-path FP8 quantization; one
/// sync cuMemcpyHtoD per tensor. Call once at engine init.
pub fn load_model(model_dir: &Path, arena: &HbmArena, arch: &ModelArch) -> Result<LoadedModel> {
    let (shards, tensors) = build_tensor_index(model_dir)?;

    let bytes_of = |si: usize, e: &TensorEntry| -> &[u8] {
        let s = &shards[si].bytes();
        let start = e.file_offset as usize;
        &s[start..start + e.nbytes as usize]
    };

    let get_tensor = |prefix: &str| -> Option<(usize, TensorEntry)> {
        tensors.get(prefix).cloned()
    };

    let must_get = |name: &str| -> Result<(usize, TensorEntry)> {
        get_tensor(name).ok_or_else(|| RvllmError::Loader {
            err: LoaderError::MissingTensor {
                name: name.to_string(),
            },
            ctx: LoaderCtx {
                path: model_dir.to_path_buf(),
                tensor: Some(name.to_string()),
            },
            bt: std::backtrace::Backtrace::capture(),
        })
    };

    // --- f16 weights (direct upload) --------------------------------------
    let upload_f16 = |name: &'static str, hf_name: &str| -> Result<F16Weight> {
        let (si, e) = must_get(hf_name)?;
        let bytes = bytes_of(si, &e);
        let buf = tensor_to_f16_bytes(&e, bytes, model_dir)?;
        let region = arena.region(name, buf.len(), 16)?;
        unsafe { region.copy_from_host(&buf)? };
        Ok(F16Weight {
            offset_bytes: region.device_ptr() - arena_base(arena),
            shape: e.shape.clone(),
        })
    };

    let embedding = upload_f16("embedding", "model.embed_tokens.weight")?;
    let final_norm = upload_f16("final_norm", "model.norm.weight")?;
    // LM head as FP8 per-tensor, so the final GEMM matches the layer path.
    let lm_head_fp8 = upload_fp8_from(
        arena,
        "lm_head",
        &must_get("lm_head.weight")?,
        &shards,
        model_dir,
    )?;

    // RoPE cos/sin table -- precompute at load time.
    let (cos_bytes, sin_bytes) = rope_cos_sin_bytes(arch);
    let rope_cos = {
        let r = arena.region("rope_cos", cos_bytes.len(), 16)?;
        unsafe { r.copy_from_host(&cos_bytes)? };
        F16Weight {
            offset_bytes: r.device_ptr() - arena_base(arena),
            shape: vec![arch.max_position_embeddings, arch.head_dim / 2],
        }
    };
    let rope_sin = {
        let r = arena.region("rope_sin", sin_bytes.len(), 16)?;
        unsafe { r.copy_from_host(&sin_bytes)? };
        F16Weight {
            offset_bytes: r.device_ptr() - arena_base(arena),
            shape: vec![arch.max_position_embeddings, arch.head_dim / 2],
        }
    };

    // --- per-layer --------------------------------------------------------
    let mut layers = Vec::with_capacity(arch.num_hidden_layers);
    for l in 0..arch.num_hidden_layers {
        let ln = |s: &str| format!("model.layers.{l}.{s}");

        let qkv_f16_bytes = concat_qkv(
            &must_get(&ln("self_attn.q_proj.weight"))?,
            &must_get(&ln("self_attn.k_proj.weight"))?,
            &must_get(&ln("self_attn.v_proj.weight"))?,
            &shards,
            model_dir,
        )?;
        let qkv_rows = (arch.num_attention_heads + 2 * arch.num_key_value_heads) * arch.head_dim;
        let qkv_cols = arch.hidden_size;
        let qkv = upload_fp8(
            arena,
            "qkv",
            &qkv_f16_bytes,
            &[qkv_rows, qkv_cols],
            &ln("self_attn.qkv.weight"),
            model_dir,
        )?;

        let o_proj = upload_fp8_from(
            arena,
            "o_proj",
            &must_get(&ln("self_attn.o_proj.weight"))?,
            &shards,
            model_dir,
        )?;

        let gate_up_f16 = concat_gate_up(
            &must_get(&ln("mlp.gate_proj.weight"))?,
            &must_get(&ln("mlp.up_proj.weight"))?,
            &shards,
            model_dir,
        )?;
        let gate_up_rows = 2 * arch.intermediate_size;
        let gate_up_cols = arch.hidden_size;
        let gate_up = upload_fp8(
            arena,
            "gate_up",
            &gate_up_f16,
            &[gate_up_rows, gate_up_cols],
            &ln("mlp.gate_up.weight"),
            model_dir,
        )?;

        let down_proj = upload_fp8_from(
            arena,
            "down_proj",
            &must_get(&ln("mlp.down_proj.weight"))?,
            &shards,
            model_dir,
        )?;

        let input_layernorm = {
            let (si, e) = must_get(&ln("input_layernorm.weight"))?;
            let buf = tensor_to_f16_bytes(&e, bytes_of(si, &e), model_dir)?;
            let r = arena.region("input_ln", buf.len(), 16)?;
            unsafe { r.copy_from_host(&buf)? };
            F16Weight {
                offset_bytes: r.device_ptr() - arena_base(arena),
                shape: e.shape.clone(),
            }
        };
        let post_attention_layernorm = {
            let (si, e) = must_get(&ln("post_attention_layernorm.weight"))?;
            let buf = tensor_to_f16_bytes(&e, bytes_of(si, &e), model_dir)?;
            let r = arena.region("post_attn_ln", buf.len(), 16)?;
            unsafe { r.copy_from_host(&buf)? };
            F16Weight {
                offset_bytes: r.device_ptr() - arena_base(arena),
                shape: e.shape.clone(),
            }
        };

        layers.push(LayerWeights {
            qkv,
            gate_up,
            o_proj,
            down_proj,
            input_layernorm,
            post_attention_layernorm,
        });
    }

    Ok(LoadedModel {
        embedding,
        lm_head_fp8,
        final_norm,
        rope_cos,
        rope_sin,
        layers,
    })
}

fn arena_base(arena: &HbmArena) -> u64 {
    // Device pointer of a zero-length region at the current top is the
    // base-ish; we use the arena's first region to get the start. Since
    // we don't have a direct getter, we return 0 and rely on offsets
    // being byte offsets from a known anchor the caller already has.
    let _ = arena;
    0
}

fn tensor_to_f16_bytes(
    e: &TensorEntry,
    raw: &[u8],
    model_dir: &Path,
) -> Result<Vec<u8>> {
    match e.dtype {
        DType::F16 => Ok(raw.to_vec()),
        DType::Bf16 => Ok(bf16_bytes_to_f16_bytes(raw)),
        DType::F32 => Ok(f32_bytes_to_f16_bytes(raw)),
        _ => Err(RvllmError::Loader {
            err: LoaderError::DtypeMismatch {
                tensor: e.name.clone(),
                expected: DType::F16,
                got: e.dtype,
            },
            ctx: LoaderCtx {
                path: model_dir.to_path_buf(),
                tensor: Some(e.name.clone()),
            },
            bt: std::backtrace::Backtrace::capture(),
        }),
    }
}

fn bf16_bytes_to_f16_bytes(raw: &[u8]) -> Vec<u8> {
    // bf16 -> f32 -> f16. Two bytes per input.
    let n = raw.len() / 2;
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n {
        let lo = raw[2 * i];
        let hi = raw[2 * i + 1];
        let as_f32 = f32::from_bits(u32::from_le_bytes([0, 0, lo, hi]));
        out.extend_from_slice(&f16::from_f32(as_f32).to_le_bytes());
    }
    out
}

fn f32_bytes_to_f16_bytes(raw: &[u8]) -> Vec<u8> {
    let n = raw.len() / 4;
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n {
        let v = f32::from_le_bytes(raw[4 * i..4 * i + 4].try_into().unwrap());
        out.extend_from_slice(&f16::from_f32(v).to_le_bytes());
    }
    out
}

fn f16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    let n = bytes.len() / 2;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let v = f16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]);
        out.push(v.to_f32());
    }
    out
}

fn quantize_to_fp8_bytes(f32_vals: &[f32], scale: f32) -> Vec<u8> {
    let inv = 1.0 / scale;
    let mut out = Vec::with_capacity(f32_vals.len());
    for v in f32_vals {
        let q = (*v * inv).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX);
        out.push(fp8_e4m3_encode(q));
    }
    out
}

// Minimal reference E4M3 encode. FP8 E4M3: 1 sign, 4 exp, 3 mantissa,
// bias 7, finite range [-448, 448]. For v3 engine init we only need
// correctness, not speed (CPU one-time path).
fn fp8_e4m3_encode(v: f32) -> u8 {
    if v.is_nan() {
        return 0x7f; // S=0, E=1111, M=111 is NaN in E4M3FN (or similar).
    }
    let s: u8 = if v < 0.0 { 0x80 } else { 0 };
    let a = v.abs();
    if a == 0.0 {
        return s;
    }
    if a > FP8_E4M3_MAX {
        return s | 0x7e; // max finite
    }
    let bits = a.to_bits();
    let exp32 = ((bits >> 23) & 0xff) as i32 - 127;
    let mant32 = bits & 0x7f_ffff;
    let exp8 = exp32 + 7;
    if exp8 <= 0 {
        // subnormal in E4M3: exp = 0, mantissa includes hidden bit
        let shift = 1 - exp8;
        let m = (mant32 | (1 << 23)) >> (21 + shift);
        return s | (m as u8 & 0x07);
    }
    if exp8 >= 0xf {
        return s | 0x7e;
    }
    let m = (mant32 >> 20) as u8 & 0x07;
    s | ((exp8 as u8 & 0x0f) << 3) | m
}

fn upload_fp8_from(
    arena: &HbmArena,
    region_name: &'static str,
    (si, entry): &(usize, TensorEntry),
    shards: &[ShardMap],
    model_dir: &Path,
) -> Result<Fp8Weight> {
    let raw = {
        let s = shards[*si].bytes();
        let start = entry.file_offset as usize;
        &s[start..start + entry.nbytes as usize]
    };
    let f16_bytes = tensor_to_f16_bytes(entry, raw, model_dir)?;
    upload_fp8(
        arena,
        region_name,
        &f16_bytes,
        &entry.shape,
        &entry.name,
        model_dir,
    )
}

fn upload_fp8(
    arena: &HbmArena,
    region_name: &'static str,
    f16_bytes: &[u8],
    shape: &[usize],
    tensor_name: &str,
    model_dir: &Path,
) -> Result<Fp8Weight> {
    let f32_vals = f16_bytes_to_f32(f16_bytes);
    let q = quantize_per_tensor_ref(&f32_vals);
    check_clamp_gate(tensor_name, q.clamp_ppm, model_dir)?;
    let fp8 = quantize_to_fp8_bytes(&f32_vals, q.scale);
    let region = arena.region(region_name, fp8.len(), 16)?;
    unsafe { region.copy_from_host(&fp8)? };
    // Also stage the per-tensor scale as a 4-byte device scalar.
    let scale_region = arena.region("fp8_scale", 4, 4)?;
    let scale_bytes = q.scale.to_le_bytes();
    unsafe { scale_region.copy_from_host(&scale_bytes)? };
    Ok(Fp8Weight {
        offset_bytes: region.device_ptr() - arena_base(arena),
        scale_ptr: scale_region.device_ptr(),
        shape: shape.to_vec(),
        scale: q.scale,
        clamp_ppm: q.clamp_ppm,
        dtype: DType::Fp8E4M3,
    })
}

fn concat_qkv(
    q: &(usize, TensorEntry),
    k: &(usize, TensorEntry),
    v: &(usize, TensorEntry),
    shards: &[ShardMap],
    model_dir: &Path,
) -> Result<Vec<u8>> {
    let qb = tensor_to_f16_bytes(
        &q.1,
        &shards[q.0].bytes()[q.1.file_offset as usize..(q.1.file_offset + q.1.nbytes) as usize],
        model_dir,
    )?;
    let kb = tensor_to_f16_bytes(
        &k.1,
        &shards[k.0].bytes()[k.1.file_offset as usize..(k.1.file_offset + k.1.nbytes) as usize],
        model_dir,
    )?;
    let vb = tensor_to_f16_bytes(
        &v.1,
        &shards[v.0].bytes()[v.1.file_offset as usize..(v.1.file_offset + v.1.nbytes) as usize],
        model_dir,
    )?;
    let mut out = Vec::with_capacity(qb.len() + kb.len() + vb.len());
    out.extend_from_slice(&qb);
    out.extend_from_slice(&kb);
    out.extend_from_slice(&vb);
    Ok(out)
}

fn concat_gate_up(
    g: &(usize, TensorEntry),
    u: &(usize, TensorEntry),
    shards: &[ShardMap],
    model_dir: &Path,
) -> Result<Vec<u8>> {
    let gb = tensor_to_f16_bytes(
        &g.1,
        &shards[g.0].bytes()[g.1.file_offset as usize..(g.1.file_offset + g.1.nbytes) as usize],
        model_dir,
    )?;
    let ub = tensor_to_f16_bytes(
        &u.1,
        &shards[u.0].bytes()[u.1.file_offset as usize..(u.1.file_offset + u.1.nbytes) as usize],
        model_dir,
    )?;
    let mut out = Vec::with_capacity(gb.len() + ub.len());
    out.extend_from_slice(&gb);
    out.extend_from_slice(&ub);
    Ok(out)
}

/// Precompute RoPE cos/sin tables as f32 bytes (v2's fused_rope_cache
/// kernel takes `const float*` tables).
fn rope_cos_sin_bytes(arch: &ModelArch) -> (Vec<u8>, Vec<u8>) {
    let half = arch.head_dim / 2;
    let mut cos = Vec::with_capacity(arch.max_position_embeddings * half * 4);
    let mut sin = Vec::with_capacity(arch.max_position_embeddings * half * 4);
    let inv_theta: Vec<f32> = (0..half)
        .map(|i| 1.0 / arch.rope_theta.powf(2.0 * i as f32 / arch.head_dim as f32))
        .collect();
    for pos in 0..arch.max_position_embeddings {
        for &freq in &inv_theta {
            let angle = pos as f32 * freq;
            cos.extend_from_slice(&angle.cos().to_le_bytes());
            sin.extend_from_slice(&angle.sin().to_le_bytes());
        }
    }
    let _ = f16::from_f32(0.0);
    (cos, sin)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp8_encode_zero_is_zero() {
        assert_eq!(fp8_e4m3_encode(0.0), 0);
        assert_eq!(fp8_e4m3_encode(-0.0), 0);
    }

    #[test]
    fn fp8_encode_clamps_at_max() {
        assert_eq!(fp8_e4m3_encode(10_000.0), 0x7e);
        assert_eq!(fp8_e4m3_encode(-10_000.0), 0xfe);
    }

    #[test]
    fn fp8_encode_small_values_preserve_sign() {
        assert!(fp8_e4m3_encode(1.0) & 0x80 == 0);
        assert!(fp8_e4m3_encode(-1.0) & 0x80 == 0x80);
    }

    #[test]
    fn rope_tables_size_correct() {
        let a = ModelArch {
            num_hidden_layers: 1,
            hidden_size: 128,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 128,
            intermediate_size: 256,
            vocab_size: 32,
            rope_theta: 10000.0,
            max_position_embeddings: 4,
        };
        let (cos, sin) = rope_cos_sin_bytes(&a);
        // 4 positions * 64 half * 2 bytes = 512
        assert_eq!(cos.len(), 512);
        assert_eq!(sin.len(), 512);
    }
}
