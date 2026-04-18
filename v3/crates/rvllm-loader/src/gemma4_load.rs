//! Gemma 4 weight loader.
//!
//! Handles: different weight prefixes (model.language_model.layers.*),
//! tied embeddings (lm_head = embed_tokens), 4 norms per layer,
//! QK-norm weights, and per-layer KV head variation.

use std::collections::BTreeMap;
use std::path::Path;

use half::f16;
use rvllm_core::{DType, LoaderCtx, LoaderError, Result, RvllmError};
use rvllm_mem::HbmArena;

use crate::fp8_quant::{check_clamp_gate, quantize_per_tensor_ref, FP8_E4M3_MAX};
use crate::gemma4_arch::Gemma4Arch;
use crate::gemma4_weights::{Gemma4LayerWeights, Gemma4LoadedModel};
use crate::safetensors::{ShardHeader, ShardIndex, TensorEntry};
use crate::weights::{F16Weight, Fp8Weight};

struct ShardMap {
    _mmap: memmap2::Mmap,
    header: ShardHeader,
}

impl ShardMap {
    fn open(path: &Path) -> Result<Self> {
        let f = std::fs::File::open(path).map_err(|source| RvllmError::Io {
            err: rvllm_core::IoError::from(&source),
            path: path.to_path_buf(),
            source,
        })?;
        let mmap = unsafe { memmap2::Mmap::map(&f) }.map_err(|source| RvllmError::Io {
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

pub fn load_gemma4_model(
    model_dir: &Path,
    arena: &HbmArena,
    arch: &Gemma4Arch,
) -> Result<Gemma4LoadedModel> {
    let idx = ShardIndex::resolve(model_dir)?;
    let mut shards = Vec::with_capacity(idx.shards.len());
    for p in &idx.shards {
        shards.push(ShardMap::open(p)?);
    }
    let mut tensors: BTreeMap<String, (usize, TensorEntry)> = BTreeMap::new();
    for (si, sm) in shards.iter().enumerate() {
        for (name, entry) in &sm.header.tensors {
            tensors.insert(name.clone(), (si, entry.clone()));
        }
    }

    let bytes_of = |si: usize, e: &TensorEntry| -> &[u8] {
        let s = shards[si].bytes();
        let start = e.file_offset as usize;
        &s[start..start + e.nbytes as usize]
    };

    let prefix = &arch.weight_prefix;

    let get_tensor = |name: &str| -> Option<(usize, TensorEntry)> {
        tensors.get(name).cloned()
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

    let upload_f16 = |name: &'static str, hf_name: &str| -> Result<F16Weight> {
        let (si, e) = must_get(hf_name)?;
        let buf = tensor_to_f16_bytes(&e, bytes_of(si, &e), model_dir)?;
        let region = arena.region(name, buf.len(), 16)?;
        unsafe { region.copy_from_host(&buf)? };
        Ok(F16Weight {
            offset_bytes: region.device_ptr(),
            shape: e.shape.clone(),
        })
    };

    // Gemma uses (1 + gamma) for RMSNorm weights. Pre-add 1.0 at load time
    // so the kernel can use raw multiplication (same as Llama/Qwen path).
    let upload_f16_gemma_norm = |name: &'static str, hf_name: &str| -> Result<F16Weight> {
        let (si, e) = must_get(hf_name)?;
        let mut buf = tensor_to_f16_bytes(&e, bytes_of(si, &e), model_dir)?;
        let n = buf.len() / 2;
        for i in 0..n {
            let lo = buf[2 * i];
            let hi = buf[2 * i + 1];
            let bits = u16::from_le_bytes([lo, hi]);
            let v = f16::from_bits(bits);
            let adjusted = f16::from_f32(v.to_f32() + 1.0);
            let out = adjusted.to_le_bytes();
            buf[2 * i] = out[0];
            buf[2 * i + 1] = out[1];
        }
        let region = arena.region(name, buf.len(), 16)?;
        unsafe { region.copy_from_host(&buf)? };
        Ok(F16Weight {
            offset_bytes: region.device_ptr(),
            shape: e.shape.clone(),
        })
    };

    let embed_name = format!("{prefix}.embed_tokens.weight");
    // Gemma models scale embeddings by sqrt(hidden_size) after lookup.
    // Pre-scale at load time so the embedding_gather kernel doesn't need modification.
    let embedding = {
        let (si, e) = must_get(&embed_name)?;
        let mut buf = tensor_to_f16_bytes(&e, bytes_of(si, &e), model_dir)?;
        let scale = (arch.hidden_size as f32).sqrt();
        eprintln!("[loader] Gemma embedding scale: sqrt({}) = {:.2}", arch.hidden_size, scale);
        let n = buf.len() / 2;
        for i in 0..n {
            let bits = u16::from_le_bytes([buf[2*i], buf[2*i+1]]);
            let v = f16::from_bits(bits);
            let scaled = f16::from_f32(v.to_f32() * scale);
            let out = scaled.to_le_bytes();
            buf[2*i] = out[0];
            buf[2*i+1] = out[1];
        }
        let region = arena.region("embedding", buf.len(), 16)?;
        unsafe { region.copy_from_host(&buf)? };
        F16Weight {
            offset_bytes: region.device_ptr(),
            shape: e.shape.clone(),
        }
    };

    let norm_name = format!("{prefix}.norm.weight");
    let final_norm = upload_f16("final_norm", &norm_name)?;

    // Detect pre-quantized FP8 weights (e.g. RedHatAI/gemma-4-31B-it-FP8-Dynamic).
    // These have F8_E4M3 linear weights + per-channel BF16 weight_scale tensors.
    let probe_name = format!("{prefix}.layers.0.self_attn.q_proj.weight");
    let fp8_prequant = get_tensor(&probe_name)
        .map(|(_, e)| e.dtype == DType::Fp8E4M3)
        .unwrap_or(false);
    if fp8_prequant {
        eprintln!("[loader] Gemma 4 FP8 pre-quantized mode: uploading weights directly with per-channel scale unification");
    } else {
        eprintln!("[loader] Gemma 4 BF16 mode: CPU-quantizing to FP8 at load time");
    }

    let lm_head_fp8 = if let Some((si, e)) = get_tensor("lm_head.weight") {
        if e.dtype == DType::Fp8E4M3 {
            let scale_entry = get_tensor("lm_head.weight_scale");
            upload_fp8_direct_channelscale(
                arena, "lm_head", &(si, e), scale_entry.as_ref(), &shards,
            )?
        } else {
            upload_fp8(
                arena,
                "lm_head",
                &tensor_to_f16_bytes(&e, bytes_of(si, &e), model_dir)?,
                &e.shape,
                "lm_head.weight",
                model_dir,
            )?
        }
    } else {
        // Tied embeddings: embed_tokens is always BF16. CPU-quantize to FP8.
        let (si, e) = must_get(&embed_name)?;
        eprintln!("[loader] tied embeddings: CPU-quantizing BF16 embed_tokens ({} elements) to FP8 for lm_head",
            e.shape.iter().product::<usize>());
        let buf = tensor_to_f16_bytes(&e, bytes_of(si, &e), model_dir)?;
        upload_fp8(
            arena,
            "lm_head",
            &buf,
            &e.shape,
            "lm_head(tied_embed)",
            model_dir,
        )?
    };

    // Sliding RoPE: theta=10000, full rotation of head_dim_sliding (256)
    let sliding_rotary_dim = arch.head_dim_sliding;
    let (cos_s, sin_s) = rope_cos_sin_bytes(
        arch.head_dim_sliding,
        arch.max_position_embeddings,
        arch.rope_theta_sliding,
        sliding_rotary_dim,
    );
    // Global RoPE: theta=1M, partial rotation (0.25 * head_dim_global = 128 of 512)
    let global_rotary_dim = arch.rotary_dim_for_layer(
        arch.layer_types.iter().position(|t| *t == crate::gemma4_arch::Gemma4LayerType::GlobalAttention).unwrap_or(0)
    );
    let (cos_g, sin_g) = rope_cos_sin_bytes(
        arch.head_dim_global,
        arch.max_position_embeddings,
        arch.rope_theta_global,
        global_rotary_dim,
    );

    let rope_cos_sliding = upload_rope(arena, "rope_cos_sliding", &cos_s)?;
    let rope_sin_sliding = upload_rope(arena, "rope_sin_sliding", &sin_s)?;
    let rope_cos_global = upload_rope(arena, "rope_cos_global", &cos_g)?;
    let rope_sin_global = upload_rope(arena, "rope_sin_global", &sin_g)?;

    // Per-layer weight shapes differ between sliding and global layers:
    //   Sliding: q=[8192,5376] k=[4096,5376] v=[4096,5376] o=[5376,8192]
    //   Global:  q=[16384,5376] k=[2048,5376] NO v_proj    o=[5376,16384]
    // Global layers have attention_k_eq_v=true: K weight serves as both K and V.

    let mut layers = Vec::with_capacity(arch.num_hidden_layers);
    for l in 0..arch.num_hidden_layers {
        let ln = |s: &str| format!("{prefix}.layers.{l}.{s}");

        let is_global = arch.layer_types[l] == crate::gemma4_arch::Gemma4LayerType::GlobalAttention;
        let layer_hd = arch.head_dim_for_layer(l);
        let layer_nkvh = arch.num_kv_heads_for_layer(l);
        let layer_q_dim = arch.num_attention_heads * layer_hd;
        let layer_kv_dim = layer_nkvh * layer_hd;

        let q_tensor = must_get(&ln("self_attn.q_proj.weight"))?;
        let k_tensor = must_get(&ln("self_attn.k_proj.weight"))?;
        let has_v = get_tensor(&ln("self_attn.v_proj.weight")).is_some();
        let v_tensor = if has_v {
            must_get(&ln("self_attn.v_proj.weight"))?
        } else {
            k_tensor.clone()
        };
        let qkv_rows = layer_q_dim + 2 * layer_kv_dim;

        let (qkv, o_proj, gate_up, down_proj) = if fp8_prequant {
            let q_scale = get_tensor(&ln("self_attn.q_proj.weight_scale"));
            let k_scale = get_tensor(&ln("self_attn.k_proj.weight_scale"));
            let v_scale = if has_v {
                get_tensor(&ln("self_attn.v_proj.weight_scale"))
            } else {
                k_scale.clone()
            };
            let qkv = fuse_fp8_direct_channelscale(
                arena, "qkv",
                &[&q_tensor, &k_tensor, &v_tensor],
                &[q_scale.as_ref(), k_scale.as_ref(), v_scale.as_ref()],
                &shards,
                &[qkv_rows, arch.hidden_size],
            )?;

            let o_entry = must_get(&ln("self_attn.o_proj.weight"))?;
            let o_scale = get_tensor(&ln("self_attn.o_proj.weight_scale"));
            let o_proj = upload_fp8_direct_channelscale(
                arena, "o_proj", &o_entry, o_scale.as_ref(), &shards,
            )?;

            let gate_entry = must_get(&ln("mlp.gate_proj.weight"))?;
            let up_entry = must_get(&ln("mlp.up_proj.weight"))?;
            let gate_scale = get_tensor(&ln("mlp.gate_proj.weight_scale"));
            let up_scale = get_tensor(&ln("mlp.up_proj.weight_scale"));
            let gate_up = fuse_fp8_direct_channelscale(
                arena, "gate_up",
                &[&gate_entry, &up_entry],
                &[gate_scale.as_ref(), up_scale.as_ref()],
                &shards,
                &[2 * arch.intermediate_size, arch.hidden_size],
            )?;

            let down_entry = must_get(&ln("mlp.down_proj.weight"))?;
            let down_scale = get_tensor(&ln("mlp.down_proj.weight_scale"));
            let down_proj = upload_fp8_direct_channelscale(
                arena, "down_proj", &down_entry, down_scale.as_ref(), &shards,
            )?;

            (qkv, o_proj, gate_up, down_proj)
        } else {
            let qkv_f16 = concat_tensors(
                &[&q_tensor, &k_tensor, &v_tensor],
                &shards,
                model_dir,
            )?;
            let qkv = upload_fp8(
                arena,
                "qkv",
                &qkv_f16,
                &[qkv_rows, arch.hidden_size],
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

            let gate_up_f16 = concat_tensors(
                &[
                    &must_get(&ln("mlp.gate_proj.weight"))?,
                    &must_get(&ln("mlp.up_proj.weight"))?,
                ],
                &shards,
                model_dir,
            )?;
            let gate_up = upload_fp8(
                arena,
                "gate_up",
                &gate_up_f16,
                &[2 * arch.intermediate_size, arch.hidden_size],
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

            (qkv, o_proj, gate_up, down_proj)
        };

        let input_layernorm =
            upload_f16("input_ln", &ln("input_layernorm.weight"))?;
        let post_attention_layernorm =
            upload_f16("post_attn_ln", &ln("post_attention_layernorm.weight"))?;
        let pre_feedforward_layernorm =
            upload_f16("pre_ff_ln", &ln("pre_feedforward_layernorm.weight"))?;
        let post_feedforward_layernorm =
            upload_f16("post_ff_ln", &ln("post_feedforward_layernorm.weight"))?;

        let q_norm = upload_f16("q_norm", &ln("self_attn.q_norm.weight"))?;
        let k_norm = upload_f16("k_norm", &ln("self_attn.k_norm.weight"))?;

        let layer_scalar = upload_f16("layer_scalar", &ln("layer_scalar"))?;

        if l < 2 {
            eprintln!(
                "[loader] layer {l} FP8: qkv_scale={:.6e} o={:.6e} gate_up={:.6e} down={:.6e}",
                qkv.scale, o_proj.scale, gate_up.scale, down_proj.scale,
            );
        }

        layers.push(Gemma4LayerWeights {
            qkv,
            o_proj,
            gate_up,
            down_proj,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            q_norm,
            k_norm,
            layer_scalar,
        });
    }

    Ok(Gemma4LoadedModel {
        embedding,
        lm_head_fp8,
        final_norm,
        rope_cos_sliding,
        rope_sin_sliding,
        rope_cos_global,
        rope_sin_global,
        layers,
    })
}

fn upload_rope(arena: &HbmArena, name: &'static str, data: &[u8]) -> Result<F16Weight> {
    let r = arena.region(name, data.len(), 16)?;
    unsafe { r.copy_from_host(data)? };
    Ok(F16Weight {
        offset_bytes: r.device_ptr(),
        shape: vec![data.len() / 2],
    })
}

fn rope_cos_sin_bytes(
    head_dim: usize,
    max_pos: usize,
    theta: f32,
    rotary_dim: usize,
) -> (Vec<u8>, Vec<u8>) {
    let half = rotary_dim / 2;
    let mut cos = Vec::with_capacity(max_pos * half * 2);
    let mut sin = Vec::with_capacity(max_pos * half * 2);
    let inv_theta: Vec<f32> = (0..half)
        .map(|i| 1.0 / theta.powf(2.0 * i as f32 / rotary_dim as f32))
        .collect();
    let _ = head_dim;
    for pos in 0..max_pos {
        for &freq in &inv_theta {
            let angle = pos as f32 * freq;
            cos.extend_from_slice(&f16::from_f32(angle.cos()).to_le_bytes());
            sin.extend_from_slice(&f16::from_f32(angle.sin()).to_le_bytes());
        }
    }
    (cos, sin)
}

fn tensor_to_f16_bytes(e: &TensorEntry, raw: &[u8], model_dir: &Path) -> Result<Vec<u8>> {
    match e.dtype {
        DType::F16 => Ok(raw.to_vec()),
        DType::Bf16 => Ok(bf16_to_f16(raw)),
        DType::F32 => Ok(f32_to_f16(raw)),
        DType::Fp8E4M3 => Ok(fp8e4m3_to_f16(raw)),
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

fn fp8e4m3_to_f16(raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(raw.len() * 2);
    for &b in raw {
        out.extend_from_slice(&f16::from_f32(fp8_e4m3_to_f32(b)).to_le_bytes());
    }
    out
}

fn bf16_to_f16(raw: &[u8]) -> Vec<u8> {
    let n = raw.len() / 2;
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n {
        let as_f32 = f32::from_bits(u32::from_le_bytes([0, 0, raw[2 * i], raw[2 * i + 1]]));
        out.extend_from_slice(&f16::from_f32(as_f32).to_le_bytes());
    }
    out
}

fn f32_to_f16(raw: &[u8]) -> Vec<u8> {
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

fn concat_tensors(
    entries: &[&(usize, TensorEntry)],
    shards: &[ShardMap],
    model_dir: &Path,
) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    for &&(si, ref e) in entries {
        let raw = &shards[si].bytes()[e.file_offset as usize..(e.file_offset + e.nbytes) as usize];
        let buf = tensor_to_f16_bytes(e, raw, model_dir)?;
        out.extend_from_slice(&buf);
    }
    Ok(out)
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
    let scale_region = arena.region("fp8_scale", 4, 4)?;
    unsafe { scale_region.copy_from_host(&q.scale.to_le_bytes())? };
    Ok(Fp8Weight {
        offset_bytes: region.device_ptr(),
        scale_ptr: scale_region.device_ptr(),
        shape: shape.to_vec(),
        scale: q.scale,
        clamp_ppm: q.clamp_ppm,
        dtype: DType::Fp8E4M3,
    })
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

/// Read per-channel BF16 scales [rows, 1] into an f32 vec.
fn read_channelscale_bf16(
    scale_entry: &(usize, TensorEntry),
    shards: &[ShardMap],
) -> Vec<f32> {
    let (si, e) = scale_entry;
    let raw = &shards[*si].bytes()[e.file_offset as usize..(e.file_offset + e.nbytes) as usize];
    let n = raw.len() / 2;
    let mut scales = Vec::with_capacity(n);
    for i in 0..n {
        let as_f32 = f32::from_bits(u32::from_le_bytes([0, 0, raw[2 * i], raw[2 * i + 1]]));
        scales.push(as_f32);
    }
    scales
}

/// Upload a pre-quantized FP8 weight with per-channel BF16 scales.
///
/// Converts per-channel scales to a single per-tensor scale by taking the
/// max across channels, then rescales any rows whose scale differs.
fn upload_fp8_direct_channelscale(
    arena: &HbmArena,
    region_name: &'static str,
    (si, entry): &(usize, TensorEntry),
    scale_entry: Option<&(usize, TensorEntry)>,
    shards: &[ShardMap],
) -> Result<Fp8Weight> {
    let raw = {
        let s = shards[*si].bytes();
        let start = entry.file_offset as usize;
        &s[start..start + entry.nbytes as usize]
    };
    let rows = entry.shape[0];
    let cols = entry.nbytes as usize / rows;
    let (unified_scale, fp8_bytes) = if let Some(se) = scale_entry {
        let ch_scales = read_channelscale_bf16(se, shards);
        let max_scale = ch_scales.iter().copied().fold(0.0f32, f32::max);
        let mut out = raw.to_vec();
        for r in 0..rows {
            let rs = ch_scales[r];
            if (rs - max_scale).abs() > 1e-12 {
                let ratio = rs / max_scale;
                let row_start = r * cols;
                for c in 0..cols {
                    let f = fp8_e4m3_to_f32(out[row_start + c]) * ratio;
                    out[row_start + c] = fp8_e4m3_encode(f);
                }
            }
        }
        (max_scale, out)
    } else {
        (1.0 / 448.0, raw.to_vec())
    };
    let region = arena.region(region_name, fp8_bytes.len(), 16)?;
    unsafe { region.copy_from_host(&fp8_bytes)? };
    let scale_region = arena.region("fp8_scale", 4, 4)?;
    unsafe { scale_region.copy_from_host(&unified_scale.to_le_bytes())? };
    Ok(Fp8Weight {
        offset_bytes: region.device_ptr(),
        scale_ptr: scale_region.device_ptr(),
        shape: entry.shape.clone(),
        scale: unified_scale,
        clamp_ppm: 0.0,
        dtype: DType::Fp8E4M3,
    })
}

/// Fuse multiple pre-quantized FP8 tensors (e.g. QKV, gate+up) with
/// per-channel BF16 scales into one contiguous FP8 region with a
/// single per-tensor scale.
fn fuse_fp8_direct_channelscale(
    arena: &HbmArena,
    region_name: &'static str,
    parts: &[&(usize, TensorEntry)],
    scale_entries: &[Option<&(usize, TensorEntry)>],
    shards: &[ShardMap],
    fused_shape: &[usize],
) -> Result<Fp8Weight> {
    // Collect all per-channel scales, find global max.
    let mut all_ch_scales: Vec<Vec<f32>> = Vec::new();
    let mut global_max = 0.0f32;
    for se in scale_entries {
        if let Some(s) = se {
            let ch = read_channelscale_bf16(s, shards);
            let m = ch.iter().copied().fold(0.0f32, f32::max);
            if m > global_max { global_max = m; }
            all_ch_scales.push(ch);
        } else {
            all_ch_scales.push(vec![]);
        }
    }
    if global_max == 0.0 {
        global_max = 1.0 / 448.0;
    }

    let mut fused = Vec::new();
    for (i, &(si, ref entry)) in parts.iter().enumerate() {
        let raw = &shards[*si].bytes()[entry.file_offset as usize..(entry.file_offset + entry.nbytes) as usize];
        let rows = entry.shape[0];
        let cols = entry.nbytes as usize / rows;
        let ch_scales = &all_ch_scales[i];
        if ch_scales.is_empty() {
            fused.extend_from_slice(raw);
        } else {
            for r in 0..rows {
                let rs = ch_scales[r];
                let row_start = r * cols;
                if (rs - global_max).abs() < 1e-12 {
                    fused.extend_from_slice(&raw[row_start..row_start + cols]);
                } else {
                    let ratio = rs / global_max;
                    for c in 0..cols {
                        let f = fp8_e4m3_to_f32(raw[row_start + c]) * ratio;
                        fused.push(fp8_e4m3_encode(f));
                    }
                }
            }
        }
    }
    let region = arena.region(region_name, fused.len(), 16)?;
    unsafe { region.copy_from_host(&fused)? };
    let scale_region = arena.region("fp8_scale", 4, 4)?;
    unsafe { scale_region.copy_from_host(&global_max.to_le_bytes())? };
    Ok(Fp8Weight {
        offset_bytes: region.device_ptr(),
        scale_ptr: scale_region.device_ptr(),
        shape: fused_shape.to_vec(),
        scale: global_max,
        clamp_ppm: 0.0,
        dtype: DType::Fp8E4M3,
    })
}

fn quantize_to_fp8_bytes(f32_vals: &[f32], scale: f32) -> Vec<u8> {
    let inv = 1.0 / scale;
    f32_vals
        .iter()
        .map(|v| fp8_e4m3_encode((*v * inv).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)))
        .collect()
}

fn fp8_e4m3_encode(v: f32) -> u8 {
    if v.is_nan() {
        return 0x7f;
    }
    let s: u8 = if v < 0.0 { 0x80 } else { 0 };
    let a = v.abs();
    if a == 0.0 {
        return s;
    }
    if a > FP8_E4M3_MAX {
        return s | 0x7e;
    }
    let bits = a.to_bits();
    let exp32 = ((bits >> 23) & 0xff) as i32 - 127;
    let mant32 = bits & 0x7f_ffff;
    let exp8 = exp32 + 7;
    if exp8 <= 0 {
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

fn fp8_e4m3_to_f32(b: u8) -> f32 {
    let s = (b >> 7) & 1;
    let e = (b >> 3) & 0xF;
    let m = b & 0x7;
    let val = if e == 0 {
        if m == 0 { 0.0f32 } else { (m as f32) * (1.0 / 512.0) }
    } else if e == 15 && m == 7 {
        return f32::NAN;
    } else {
        f32::from_bits(((e as u32 + 120) << 23) | ((m as u32) << 20))
    };
    if s != 0 { -val } else { val }
}
