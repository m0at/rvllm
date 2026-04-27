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

/// Per-layer attention type.
/// - Full: standard causal attention over entire context
/// - SlidingAttention: local sliding window (Gemma 4)
/// - Linear: GDN linear attention (Qwen3.5/3.6)
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LayerAttnType {
    Full,
    SlidingAttention,
    Linear,
}

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
    pub attention_bias: bool,
    pub rms_norm_eps: f32,
    pub layer_types: Vec<LayerAttnType>,
    // -- Gemma 4 fields (None / 0 for non-Gemma models) --
    pub global_head_dim: Option<usize>,
    pub num_global_key_value_heads: Option<usize>,
    pub global_rope_theta: Option<f32>,
    pub partial_rotary_factor: Option<f32>,
    pub sliding_window: Option<usize>,
    pub final_logit_softcapping: Option<f32>,
    pub hidden_activation: Option<String>,
    pub tie_word_embeddings: bool,
    pub attention_k_eq_v: bool,
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
        // Gemma 3/4: text model params nested under text_config.
        let has_text_config = v["text_config"]["hidden_size"].is_u64();
        let tc = if has_text_config { &v["text_config"] } else { &v };
        let num_hidden_layers = tc["num_hidden_layers"].as_u64().unwrap_or(0) as usize;
        let hidden_size = tc["hidden_size"].as_u64().unwrap_or(0) as usize;
        let num_attention_heads = tc["num_attention_heads"].as_u64().unwrap_or(0) as usize;
        let num_key_value_heads = tc["num_key_value_heads"]
            .as_u64()
            .unwrap_or(num_attention_heads as u64) as usize;
        let intermediate_size = tc["intermediate_size"].as_u64().unwrap_or(0) as usize;
        let vocab_size = tc["vocab_size"].as_u64().unwrap_or(0) as usize;
        // Gemma 4 nests rope_theta under rope_parameters.sliding_attention.
        let rope_theta = tc["rope_parameters"]["sliding_attention"]["rope_theta"]
            .as_f64()
            .or_else(|| tc["rope_theta"].as_f64())
            .unwrap_or(10000.0) as f32;
        let max_position_embeddings =
            tc["max_position_embeddings"].as_u64().unwrap_or(2048) as usize;
        let attention_bias = tc["attention_bias"].as_bool().unwrap_or(false);
        let rms_norm_eps = tc["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32;
        let layer_types_val = tc["layer_types"].as_array()
            .or_else(|| v["layer_types"].as_array());
        let layer_types: Vec<LayerAttnType> = match layer_types_val {
            Some(arr) => arr.iter().map(|t| {
                match t.as_str().unwrap_or("full_attention") {
                    "sliding_attention" => LayerAttnType::SlidingAttention,
                    "linear_attention" => LayerAttnType::Linear,
                    _ => LayerAttnType::Full,
                }
            }).collect(),
            None => vec![LayerAttnType::Full; num_hidden_layers],
        };
        let head_dim = tc["head_dim"]
            .as_u64()
            .map(|d| d as usize)
            .unwrap_or_else(|| if num_attention_heads == 0 { 0 } else { hidden_size / num_attention_heads });
        // Accept head_dim 128 (Llama/Qwen/Mistral) and 256 (Gemma 4).
        if head_dim != 128 && head_dim != 256 {
            return Err(RvllmError::Loader {
                err: LoaderError::Corrupt {
                    detail: format!(
                        "v3 requires head_dim in {{128, 256}}, got {head_dim} (hidden={hidden_size}, heads={num_attention_heads})"
                    ),
                },
                ctx: LoaderCtx {
                    path: p.clone(),
                    tensor: None,
                },
                bt: std::backtrace::Backtrace::capture(),
            });
        }
        let hidden_activation = tc["hidden_act"].as_str()
            .or_else(|| tc["hidden_activation"].as_str())
            .map(|s| s.to_string());
        let global_head_dim = tc["global_head_dim"].as_u64().map(|d| d as usize);
        let num_global_key_value_heads = tc["num_global_key_value_heads"].as_u64().map(|d| d as usize);
        let global_rope_theta = tc.get("rope_parameters")
            .and_then(|rp| rp["full_attention"]["rope_theta"].as_f64())
            .or_else(|| tc["rope_theta_global"].as_f64())
            .map(|t| t as f32);
        let partial_rotary_factor = tc.get("rope_parameters")
            .and_then(|rp| rp["full_attention"]["partial_rotary_factor"].as_f64())
            .or_else(|| tc["partial_rotary_factor"].as_f64())
            .map(|f| f as f32);
        let sliding_window = tc["sliding_window"].as_u64()
            .or_else(|| tc["sliding_window_size"].as_u64())
            .map(|s| s as usize);
        let final_logit_softcapping = tc["final_logit_softcapping"].as_f64()
            .or_else(|| tc["logit_softcapping"].as_f64())
            .map(|s| s as f32);
        let tie_word_embeddings = tc["tie_word_embeddings"].as_bool()
            .or_else(|| v["tie_word_embeddings"].as_bool())
            .unwrap_or(false);
        let attention_k_eq_v = tc["attention_k_eq_v"].as_bool().unwrap_or(false);

        if global_head_dim.is_some() {
            let n_sliding = layer_types.iter().filter(|t| **t == LayerAttnType::SlidingAttention).count();
            let n_full = layer_types.iter().filter(|t| **t == LayerAttnType::Full).count();
            eprintln!(
                "[loader] Gemma 4: {num_hidden_layers} layers ({n_sliding} sliding + {n_full} full), \
                 head_dim={head_dim}/{}, kv_heads={num_key_value_heads}/{}, \
                 rope={rope_theta}/{:?}, partial_rot={:?}, sw={:?}, \
                 softcap={:?}, act={:?}, tied={}",
                global_head_dim.unwrap_or(0),
                num_global_key_value_heads.unwrap_or(0),
                global_rope_theta, partial_rotary_factor,
                sliding_window, final_logit_softcapping,
                hidden_activation, tie_word_embeddings,
            );
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
            attention_bias,
            rms_norm_eps,
            layer_types,
            global_head_dim,
            num_global_key_value_heads,
            global_rope_theta,
            partial_rotary_factor,
            sliding_window,
            final_logit_softcapping,
            hidden_activation,
            tie_word_embeddings,
            attention_k_eq_v,
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
    // Cycle 47 step 6a parallel guard (codex review of e0cf73b).
    if let Some(awq) = crate::compressed_tensors::read_awq_config_from_dir(model_dir)
        .map_err(|e| RvllmError::Loader {
            err: LoaderError::Corrupt { detail: format!("quantization_config: {e}") },
            ctx: LoaderCtx { path: model_dir.to_path_buf(), tensor: None },
            bt: std::backtrace::Backtrace::capture(),
        })?
    {
        return Err(RvllmError::Loader {
            err: LoaderError::UnsupportedQuantization {
                detail: format!(
                    "AWQ {:?} W{} detected; load_multiformat::load_model does not implement AWQ — use load_gemma4_model after cycle 48 wiring.",
                    awq.format, awq.scheme.num_bits
                ),
            },
            ctx: LoaderCtx { path: model_dir.to_path_buf(), tensor: None },
            bt: std::backtrace::Backtrace::capture(),
        });
    }
    let (shards, tensors) = build_tensor_index(model_dir)?;

    let wprefix: &str = if tensors.contains_key("model.embed_tokens.weight") {
        "model"
    } else if tensors.contains_key("language_model.model.embed_tokens.weight") {
        eprintln!("[loader] detected language_model.model.* weight prefix");
        "language_model.model"
    } else {
        "model"
    };
    let tie_embeddings = !tensors.contains_key("lm_head.weight")
        && !tensors.contains_key(&format!("{wprefix}.lm_head.weight"));

    let bytes_of = |si: usize, e: &TensorEntry| -> &[u8] {
        let s = &shards[si].bytes();
        let start = e.file_offset as usize;
        &s[start..start + e.nbytes as usize]
    };

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

    let embed_name = format!("{wprefix}.embed_tokens.weight");
    let norm_name = format!("{wprefix}.norm.weight");
    let embedding = upload_f16("embedding", &embed_name)?;
    let final_norm = upload_f16("final_norm", &norm_name)?;
    let lm_head_fp8 = if tie_embeddings {
        eprintln!("[loader] tied embeddings -> reusing embed_tokens as lm_head");
        let (si, e) = must_get(&embed_name)?;
        upload_fp8_from(arena, "lm_head", &(si, e), &shards, model_dir)?
    } else {
        upload_fp8_from(arena, "lm_head", &must_get("lm_head.weight")?, &shards, model_dir)?
    };
    eprintln!(
        "[loader] lm_head FP8 scale={:.6e} clamp_ppm={:.1}",
        lm_head_fp8.scale, lm_head_fp8.clamp_ppm,
    );

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
    // Detect pre-quantized FP8 models (e.g. neuralmagic) by checking
    // the dtype of the first q_proj weight tensor.
    let weights_are_fp8 = get_tensor(&format!("{wprefix}.layers.0.self_attn.q_proj.weight"))
        .map_or(false, |(_, e)| e.dtype == DType::Fp8E4M3);
    if weights_are_fp8 {
        eprintln!("[loader] detected pre-quantized FP8 weights, using direct upload");
    }

    let mut layers = Vec::with_capacity(arch.num_hidden_layers);
    for l in 0..arch.num_hidden_layers {
        let ln = |s: &str| format!("{wprefix}.layers.{l}.{s}");

        let qkv_rows = (arch.num_attention_heads + 2 * arch.num_key_value_heads) * arch.head_dim;
        let qkv_cols = arch.hidden_size;
        let qkv = if weights_are_fp8 {
            upload_fp8_fused_direct(
                arena, "qkv",
                &[
                    must_get(&ln("self_attn.q_proj.weight"))?,
                    must_get(&ln("self_attn.k_proj.weight"))?,
                    must_get(&ln("self_attn.v_proj.weight"))?,
                ],
                &[
                    get_tensor(&ln("self_attn.q_proj.weight_scale")),
                    get_tensor(&ln("self_attn.k_proj.weight_scale")),
                    get_tensor(&ln("self_attn.v_proj.weight_scale")),
                ],
                &shards,
                &[qkv_rows, qkv_cols],
            )?
        } else {
            let qkv_f16_bytes = concat_qkv(
                &must_get(&ln("self_attn.q_proj.weight"))?,
                &must_get(&ln("self_attn.k_proj.weight"))?,
                &must_get(&ln("self_attn.v_proj.weight"))?,
                &shards,
                model_dir,
            )?;
            upload_fp8(
                arena, "qkv", &qkv_f16_bytes,
                &[qkv_rows, qkv_cols],
                &ln("self_attn.qkv.weight"), model_dir,
            )?
        };

        let qkv_bias = if arch.attention_bias {
            let qkv_bias_bytes = concat_qkv_bias(
                &must_get(&ln("self_attn.q_proj.bias"))?,
                &must_get(&ln("self_attn.k_proj.bias"))?,
                &must_get(&ln("self_attn.v_proj.bias"))?,
                &shards,
                model_dir,
            )?;
            let r = arena.region("qkv_bias", qkv_bias_bytes.len(), 16)?;
            unsafe { r.copy_from_host(&qkv_bias_bytes)? };
            Some(F16Weight {
                offset_bytes: r.device_ptr() - arena_base(arena),
                shape: vec![qkv_rows],
            })
        } else {
            None
        };

        let o_proj = if weights_are_fp8 {
            upload_fp8_direct(
                arena, "o_proj",
                &must_get(&ln("self_attn.o_proj.weight"))?,
                get_tensor(&ln("self_attn.o_proj.weight_scale")),
                &shards,
            )?
        } else {
            upload_fp8_from(arena, "o_proj",
                &must_get(&ln("self_attn.o_proj.weight"))?, &shards, model_dir)?
        };

        let gate_up_rows = 2 * arch.intermediate_size;
        let gate_up_cols = arch.hidden_size;
        let gate_up = if weights_are_fp8 {
            upload_fp8_fused_direct(
                arena, "gate_up",
                &[
                    must_get(&ln("mlp.gate_proj.weight"))?,
                    must_get(&ln("mlp.up_proj.weight"))?,
                ],
                &[
                    get_tensor(&ln("mlp.gate_proj.weight_scale")),
                    get_tensor(&ln("mlp.up_proj.weight_scale")),
                ],
                &shards,
                &[gate_up_rows, gate_up_cols],
            )?
        } else {
            let gate_up_f16 = concat_gate_up(
                &must_get(&ln("mlp.gate_proj.weight"))?,
                &must_get(&ln("mlp.up_proj.weight"))?,
                &shards, model_dir,
            )?;
            upload_fp8(arena, "gate_up", &gate_up_f16,
                &[gate_up_rows, gate_up_cols], &ln("mlp.gate_up.weight"), model_dir)?
        };

        let down_proj = if weights_are_fp8 {
            upload_fp8_direct(
                arena, "down_proj",
                &must_get(&ln("mlp.down_proj.weight"))?,
                get_tensor(&ln("mlp.down_proj.weight_scale")),
                &shards,
            )?
        } else {
            upload_fp8_from(arena, "down_proj",
                &must_get(&ln("mlp.down_proj.weight"))?, &shards, model_dir)?
        };

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

        if l < 2 {
            eprintln!(
                "[loader] layer {l} FP8 scales: qkv={:.6e} o={:.6e} gate_up={:.6e} down={:.6e} | clamp_ppm: qkv={:.1} o={:.1} gu={:.1} dn={:.1}",
                qkv.scale, o_proj.scale, gate_up.scale, down_proj.scale,
                qkv.clamp_ppm, o_proj.clamp_ppm, gate_up.clamp_ppm, down_proj.clamp_ppm,
            );
        }
        layers.push(LayerWeights {
            qkv,
            qkv_bias,
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
        DType::Fp8E4M3 => Ok(fp8e4m3_bytes_to_f16_bytes(raw)),
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

fn fp8e4m3_bytes_to_f16_bytes(raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(raw.len() * 2);
    for &b in raw {
        let s = (b >> 7) & 1;
        let e = (b >> 3) & 0xF;
        let m = b & 0x7;
        let val = if e == 0 {
            if m == 0 { 0.0f32 } else { (m as f32) * (1.0 / 512.0) * if s != 0 { -1.0 } else { 1.0 } }
        } else if e == 15 && m == 7 {
            f32::NAN
        } else {
            let v = f32::from_bits(((e as u32 + 120) << 23) | ((m as u32) << 20));
            if s != 0 { -v } else { v }
        };
        out.extend_from_slice(&f16::from_f32(val).to_le_bytes());
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
    use rayon::prelude::*;
    bytes
        .par_chunks_exact(2)
        .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect()
}

fn quantize_to_fp8_bytes(f32_vals: &[f32], scale: f32) -> Vec<u8> {
    use rayon::prelude::*;
    let inv = 1.0 / scale;
    f32_vals
        .par_iter()
        .map(|v| fp8_e4m3_encode((*v * inv).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)))
        .collect()
}

// Minimal reference E4M3 encode with round-to-nearest-even (matches NVIDIA hw).
// FP8 E4M3FN: 1 sign, 4 exp, 3 mantissa, bias 7, finite range [-448, 448].
fn fp8_e4m3_encode(v: f32) -> u8 {
    if v.is_nan() {
        return 0x7f;
    }
    let s: u8 = if v.to_bits() >> 31 != 0 { 0x80 } else { 0 };
    let a = v.abs();
    // Canonicalise signed zero: both +0.0 and -0.0 encode to 0x00. FP8
    // E4M3FN technically has a -0 (0x80) but the quantiser treats zero
    // as sign-less — downstream attention / GEMM paths hit a dequant
    // "zero + tiny scale → subnormal f16" edge if -0 leaks through.
    if a == 0.0 {
        return 0;
    }
    if a > FP8_E4M3_MAX {
        return s | 0x7e;
    }
    let bits = a.to_bits();
    let exp32 = ((bits >> 23) & 0xff) as i32 - 127;
    let mant32 = bits & 0x7f_ffff;
    let mut exp8 = exp32 + 7;
    if exp8 <= 0 {
        let shift = 1 - exp8;
        let full = (mant32 | (1 << 23)) as u32;
        let rshift = (20 + shift) as u32;
        let mut m = full >> rshift;
        let round_bit = if rshift > 0 { (full >> (rshift - 1)) & 1 } else { 0 };
        let sticky = if rshift > 1 { (full & ((1 << (rshift - 1)) - 1) != 0) as u32 } else { 0 };
        m += round_bit & (sticky | (m & 1));
        if m >= 8 {
            return s | 0x08;
        }
        return s | (m as u8 & 0x07);
    }
    let trunc = mant32 >> 20;
    let round_bit = (mant32 >> 19) & 1;
    let sticky = (mant32 & 0x7_ffff) != 0;
    let m = trunc + (round_bit & (sticky as u32 | (trunc & 1)));
    if m >= 8 {
        exp8 += 1;
        if exp8 > 15 {
            return s | 0x7e;
        }
        return s | ((exp8 as u8 & 0x0f) << 3);
    }
    if exp8 > 15 {
        return s | 0x7e;
    }
    s | ((exp8 as u8 & 0x0f) << 3) | (m as u8 & 0x07)
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
        channelscale_ptr: None,
        blockscale_ptr: None,
        blockscale_n_blocks: 0,
        blockscale_k_blocks: 0,
    })
}

fn upload_fp8_direct(
    arena: &HbmArena,
    region_name: &'static str,
    (si, entry): &(usize, TensorEntry),
    scale_tensor: Option<(usize, TensorEntry)>,
    shards: &[ShardMap],
) -> Result<Fp8Weight> {
    let raw = {
        let s = shards[*si].bytes();
        let start = entry.file_offset as usize;
        &s[start..start + entry.nbytes as usize]
    };
    let region = arena.region(region_name, raw.len(), 16)?;
    unsafe { region.copy_from_host(raw)? };
    let scale = if let Some((ssi, se)) = scale_tensor {
        let sb = &shards[ssi].bytes()[se.file_offset as usize..se.file_offset as usize + 4];
        f32::from_le_bytes([sb[0], sb[1], sb[2], sb[3]])
    } else {
        1.0 / 448.0
    };
    let scale_region = arena.region("fp8_scale", 4, 4)?;
    unsafe { scale_region.copy_from_host(&scale.to_le_bytes())? };
    eprintln!("[loader] {region_name} FP8 direct: scale={scale:.6e}");
    Ok(Fp8Weight {
        offset_bytes: region.device_ptr() - arena_base(arena),
        scale_ptr: scale_region.device_ptr(),
        shape: entry.shape.clone(),
        scale,
        clamp_ppm: 0.0,
        dtype: DType::Fp8E4M3,
        channelscale_ptr: None,
        blockscale_ptr: None,
        blockscale_n_blocks: 0,
        blockscale_k_blocks: 0,
    })
}

fn upload_fp8_fused_direct(
    arena: &HbmArena,
    region_name: &'static str,
    parts: &[(usize, TensorEntry)],
    scale_tensors: &[Option<(usize, TensorEntry)>],
    shards: &[ShardMap],
    shape: &[usize],
) -> Result<Fp8Weight> {
    let mut scales: Vec<f32> = Vec::new();
    for st in scale_tensors {
        if let Some((ssi, se)) = st {
            let sb = &shards[*ssi].bytes()[se.file_offset as usize..se.file_offset as usize + 4];
            scales.push(f32::from_le_bytes([sb[0], sb[1], sb[2], sb[3]]));
        } else {
            scales.push(1.0 / 448.0);
        }
    }
    let max_scale = scales.iter().copied().fold(0.0f32, f32::max);
    let mut fused = Vec::new();
    for (i, (si, entry)) in parts.iter().enumerate() {
        let raw = {
            let s = shards[*si].bytes();
            let start = entry.file_offset as usize;
            &s[start..start + entry.nbytes as usize]
        };
        if (scales[i] - max_scale).abs() < 1e-12 {
            fused.extend_from_slice(raw);
        } else {
            let ratio = scales[i] / max_scale;
            for &b in raw {
                let f = fp8_e4m3_to_f32(b) * ratio;
                fused.push(fp8_e4m3_encode(f));
            }
        }
    }
    let region = arena.region(region_name, fused.len(), 16)?;
    unsafe { region.copy_from_host(&fused)? };
    let scale_region = arena.region("fp8_scale", 4, 4)?;
    unsafe { scale_region.copy_from_host(&max_scale.to_le_bytes())? };
    eprintln!("[loader] {region_name} FP8 fused: scales={scales:?} -> unified={max_scale:.6e}");
    Ok(Fp8Weight {
        offset_bytes: region.device_ptr() - arena_base(arena),
        scale_ptr: scale_region.device_ptr(),
        shape: shape.to_vec(),
        scale: max_scale,
        clamp_ppm: 0.0,
        dtype: DType::Fp8E4M3,
        channelscale_ptr: None,
        blockscale_ptr: None,
        blockscale_n_blocks: 0,
        blockscale_k_blocks: 0,
    })
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

fn concat_qkv_bias(
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

/// Precompute RoPE cos/sin tables as f16 bytes. The v3 fused_rope_cache
/// variant takes __half cos/sin, not float — v3 keeps zero f32
/// activations/constants in the decode path.
fn rope_cos_sin_bytes(arch: &ModelArch) -> (Vec<u8>, Vec<u8>) {
    let half = arch.head_dim / 2;
    let mut cos = Vec::with_capacity(arch.max_position_embeddings * half * 2);
    let mut sin = Vec::with_capacity(arch.max_position_embeddings * half * 2);
    let inv_theta: Vec<f32> = (0..half)
        .map(|i| 1.0 / arch.rope_theta.powf(2.0 * i as f32 / arch.head_dim as f32))
        .collect();
    for pos in 0..arch.max_position_embeddings {
        for &freq in &inv_theta {
            let angle = pos as f32 * freq;
            cos.extend_from_slice(&f16::from_f32(angle.cos()).to_le_bytes());
            sin.extend_from_slice(&f16::from_f32(angle.sin()).to_le_bytes());
        }
    }
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
            attention_bias: false,
            rms_norm_eps: 1e-6,
            layer_types: vec![LayerAttnType::Full],
            global_head_dim: None,
            num_global_key_value_heads: None,
            global_rope_theta: None,
            partial_rotary_factor: None,
            sliding_window: None,
            final_logit_softcapping: None,
            hidden_activation: None,
            tie_word_embeddings: false,
            attention_k_eq_v: false,
        };
        let (cos, sin) = rope_cos_sin_bytes(&a);
        // 4 positions * 64 half * 2 bytes = 512
        assert_eq!(cos.len(), 512);
        assert_eq!(sin.len(), 512);
    }
}
