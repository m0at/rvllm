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
    // Cycle 47 step 6a + cycle 48 step 7b: detect AWQ checkpoints at
    // entry. parse_awq_config malformed → Corrupt. Valid AWQ →
    // (cycle 48 step 7b) walk the per-layer tensor shapes against the
    // shard index using `validate_awq_linear`, then fail with a
    // detailed "validated, upload pending" message. Cycle 48 step 7c
    // replaces this with the actual upload + layer.awq population.
    let awq_config = crate::compressed_tensors::read_awq_config_from_dir(model_dir)
        .map_err(|e| RvllmError::Loader {
            err: LoaderError::Corrupt {
                detail: format!("quantization_config: {e}"),
            },
            ctx: LoaderCtx { path: model_dir.to_path_buf(), tensor: None },
            bt: std::backtrace::Backtrace::capture(),
        })?;

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

    if let Some(awq) = &awq_config {
        eprintln!(
            "[loader] AWQ-quantized checkpoint detected (format={:?}, num_bits={}, group_size={}, ignore={:?})",
            awq.format, awq.scheme.num_bits, awq.scheme.group_size, awq.ignore
        );
        validate_gemma4_awq_layout(arch, awq, &tensors).map_err(|e| RvllmError::Loader {
            err: LoaderError::UnsupportedQuantization {
                detail: format!("AWQ layout validation failed: {e}"),
            },
            ctx: LoaderCtx { path: model_dir.to_path_buf(), tensor: None },
            bt: std::backtrace::Backtrace::capture(),
        })?;
        // (validate_gemma4_awq_layout already logs the per-linear total)
        return load_gemma4_awq_model_inner(
            model_dir, arena, arch, awq, &shards, &tensors,
        );
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
        {
            let first4: Vec<f32> = (0..4).map(|i| {
                let bits = u16::from_le_bytes([buf[2*i], buf[2*i+1]]);
                f16::from_bits(bits).to_f32()
            }).collect();
            eprintln!("[loader] embed after sqrt(H) scale: first4={:.4?}", first4);
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
        eprintln!("[loader] Gemma 4 FP8 pre-quantized mode: uploading weights directly with cuBLASLt per-channel scales");
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

    let lm_head_f16 = {
        let (si, e) = if let Some(t) = get_tensor("lm_head.weight") { t } else { must_get(&embed_name)? };
        let buf = tensor_to_f16_bytes(&e, bytes_of(si, &e), model_dir)?;
        eprintln!("[loader] lm_head_f16: {} elements ({:.1} MB)", e.shape.iter().product::<usize>(), buf.len() as f64 / 1e6);
        let region = arena.region("lm_head_f16", buf.len(), 16)?;
        unsafe { region.copy_from_host(&buf)? };
        F16Weight { offset_bytes: region.device_ptr(), shape: e.shape.clone() }
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

    let load_max_layers = std::env::var("RVLLM_MAX_LAYERS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .map(|v| v.min(arch.num_hidden_layers))
        .unwrap_or(arch.num_hidden_layers);
    if load_max_layers < arch.num_hidden_layers {
        eprintln!(
            "[loader] RVLLM_MAX_LAYERS={load_max_layers}: loading only first {load_max_layers} of {} layers",
            arch.num_hidden_layers
        );
    }

    let mut layers = Vec::with_capacity(load_max_layers);
    for l in 0..load_max_layers {
        let ln = |s: &str| format!("{prefix}.layers.{l}.{s}");

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
            let f16_only = std::env::var("RVLLM_F16_ONLY").map_or(false, |v| v == "1");

            if f16_only {
                let dummy = Fp8Weight {
                    offset_bytes: 0, scale_ptr: 0, shape: vec![],
                    scale: 0.0, clamp_ppm: 0.0, dtype: DType::Fp8E4M3,
                    channelscale_ptr: None,
                    blockscale_ptr: None,
                    blockscale_n_blocks: 0,
                    blockscale_k_blocks: 0,
                };
                (dummy.clone(), dummy.clone(), dummy.clone(), dummy.clone())
            } else {
                let split_fp8 = std::env::var("RVLLM_SPLIT_QKV").map_or(true, |v| v != "0");

                if split_fp8 {
                    // Split quantization: Q, K, V get separate per-tensor FP8 scales,
                    // then concatenate bytes + build a per-row channelscale vector.
                    let q_f16 = tensor_to_f16_bytes(&q_tensor.1, bytes_of(q_tensor.0, &q_tensor.1), model_dir)?;
                    let k_f16 = tensor_to_f16_bytes(&k_tensor.1, bytes_of(k_tensor.0, &k_tensor.1), model_dir)?;
                    let v_f16 = tensor_to_f16_bytes(&v_tensor.1, bytes_of(v_tensor.0, &v_tensor.1), model_dir)?;

                    let q_f32 = f16_bytes_to_f32(&q_f16);
                    let k_f32 = f16_bytes_to_f32(&k_f16);
                    let v_f32 = f16_bytes_to_f32(&v_f16);

                    let q_q = quantize_per_tensor_ref(&q_f32);
                    let k_q = quantize_per_tensor_ref(&k_f32);
                    let v_q = quantize_per_tensor_ref(&v_f32);

                    if l == 0 {
                        eprintln!("[loader] split QKV scales: q={:.6e} k={:.6e} v={:.6e}",
                            q_q.scale, k_q.scale, v_q.scale);
                    }

                    let q_rows = q_tensor.1.shape[0];
                    let k_rows = k_tensor.1.shape[0];
                    let v_rows = v_tensor.1.shape[0];

                    let q_fp8 = quantize_to_fp8_bytes(&q_f32, q_q.scale);
                    let k_fp8 = quantize_to_fp8_bytes(&k_f32, k_q.scale);
                    let v_fp8 = quantize_to_fp8_bytes(&v_f32, v_q.scale);

                    let mut fused_bytes = Vec::with_capacity(q_fp8.len() + k_fp8.len() + v_fp8.len());
                    fused_bytes.extend_from_slice(&q_fp8);
                    fused_bytes.extend_from_slice(&k_fp8);
                    fused_bytes.extend_from_slice(&v_fp8);

                    let region = arena.region("qkv", fused_bytes.len(), 16)?;
                    unsafe { region.copy_from_host(&fused_bytes)? };

                    // Per-row channelscale: each row gets its sub-matrix's scale
                    let mut chscales: Vec<f32> = Vec::with_capacity(q_rows + k_rows + v_rows);
                    chscales.extend(std::iter::repeat(q_q.scale).take(q_rows));
                    chscales.extend(std::iter::repeat(k_q.scale).take(k_rows));
                    chscales.extend(std::iter::repeat(v_q.scale).take(v_rows));
                    let cs_bytes: Vec<u8> = chscales.iter().flat_map(|s| s.to_le_bytes()).collect();
                    let cs_r = arena.region("qkv_chscale", cs_bytes.len(), 16)?;
                    unsafe { cs_r.copy_from_host(&cs_bytes)? };

                    let one = 1.0f32;
                    let one_r = arena.region("qkv_scale_one", 4, 4)?;
                    unsafe { one_r.copy_from_host(&one.to_le_bytes())? };

                    let qkv = Fp8Weight {
                        offset_bytes: region.device_ptr(),
                        scale_ptr: one_r.device_ptr(),
                        shape: vec![qkv_rows, arch.hidden_size],
                        scale: 1.0,
                        clamp_ppm: 0.0,
                        dtype: DType::Fp8E4M3,
                        channelscale_ptr: Some(cs_r.device_ptr()),
                        blockscale_ptr: None,
                        blockscale_n_blocks: 0,
                        blockscale_k_blocks: 0,
                    };

                    // gate_up: same split treatment
                    let gate_entry = must_get(&ln("mlp.gate_proj.weight"))?;
                    let up_entry = must_get(&ln("mlp.up_proj.weight"))?;
                    let gate_f16 = tensor_to_f16_bytes(&gate_entry.1, bytes_of(gate_entry.0, &gate_entry.1), model_dir)?;
                    let up_f16 = tensor_to_f16_bytes(&up_entry.1, bytes_of(up_entry.0, &up_entry.1), model_dir)?;
                    let gate_f32 = f16_bytes_to_f32(&gate_f16);
                    let up_f32 = f16_bytes_to_f32(&up_f16);
                    let gate_qq = quantize_per_tensor_ref(&gate_f32);
                    let up_qq = quantize_per_tensor_ref(&up_f32);
                    let gate_rows = gate_entry.1.shape[0];
                    let up_rows = up_entry.1.shape[0];
                    let gate_fp8 = quantize_to_fp8_bytes(&gate_f32, gate_qq.scale);
                    let up_fp8_bytes = quantize_to_fp8_bytes(&up_f32, up_qq.scale);
                    let mut gu_bytes = Vec::with_capacity(gate_fp8.len() + up_fp8_bytes.len());
                    gu_bytes.extend_from_slice(&gate_fp8);
                    gu_bytes.extend_from_slice(&up_fp8_bytes);
                    let gu_r = arena.region("gate_up", gu_bytes.len(), 16)?;
                    unsafe { gu_r.copy_from_host(&gu_bytes)? };
                    let mut gu_scales: Vec<f32> = Vec::with_capacity(gate_rows + up_rows);
                    gu_scales.extend(std::iter::repeat(gate_qq.scale).take(gate_rows));
                    gu_scales.extend(std::iter::repeat(up_qq.scale).take(up_rows));
                    let gus_bytes: Vec<u8> = gu_scales.iter().flat_map(|s| s.to_le_bytes()).collect();
                    let gus_r = arena.region("gu_chscale", gus_bytes.len(), 16)?;
                    unsafe { gus_r.copy_from_host(&gus_bytes)? };
                    let gu_one_r = arena.region("gu_scale_one", 4, 4)?;
                    unsafe { gu_one_r.copy_from_host(&one.to_le_bytes())? };
                    let gate_up = Fp8Weight {
                        offset_bytes: gu_r.device_ptr(),
                        scale_ptr: gu_one_r.device_ptr(),
                        shape: vec![2 * arch.intermediate_size, arch.hidden_size],
                        scale: 1.0, clamp_ppm: 0.0, dtype: DType::Fp8E4M3,
                        channelscale_ptr: Some(gus_r.device_ptr()),
                        blockscale_ptr: None,
                        blockscale_n_blocks: 0,
                        blockscale_k_blocks: 0,
                    };

                    // O-proj and down-proj: single matrix, per-tensor is fine
                    let o_proj = upload_fp8_from(arena, "o_proj", &must_get(&ln("self_attn.o_proj.weight"))?, &shards, model_dir)?;
                    let down_proj = upload_fp8_from(arena, "down_proj", &must_get(&ln("mlp.down_proj.weight"))?, &shards, model_dir)?;

                    (qkv, o_proj, gate_up, down_proj)
                } else {
                    // Original fused path
                    let qkv_f16_bytes = concat_tensors(&[&q_tensor, &k_tensor, &v_tensor], &shards, model_dir)?;
                    let qkv = upload_fp8(arena, "qkv", &qkv_f16_bytes, &[qkv_rows, arch.hidden_size], &ln("self_attn.qkv.weight"), model_dir)?;
                    let o_proj = upload_fp8_from(arena, "o_proj", &must_get(&ln("self_attn.o_proj.weight"))?, &shards, model_dir)?;
                    let gate_up_f16_bytes = concat_tensors(&[&must_get(&ln("mlp.gate_proj.weight"))?, &must_get(&ln("mlp.up_proj.weight"))?], &shards, model_dir)?;
                    let gate_up = upload_fp8(arena, "gate_up", &gate_up_f16_bytes, &[2 * arch.intermediate_size, arch.hidden_size], &ln("mlp.gate_up.weight"), model_dir)?;
                    let down_proj = upload_fp8_from(arena, "down_proj", &must_get(&ln("mlp.down_proj.weight"))?, &shards, model_dir)?;
                    (qkv, o_proj, gate_up, down_proj)
                }
            }
        };

        let f16_max = if std::env::var("RVLLM_F16_ONLY").map_or(false, |v| v == "1") {
            arch.num_hidden_layers
        } else {
            std::env::var("RVLLM_F16_LAYERS")
                .ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(0)
        };
        let f16_only = std::env::var("RVLLM_F16_ONLY").map_or(false, |v| v == "1");
        let (qkv_f16_w, o_proj_f16_w, gate_up_f16_w, down_proj_f16_w) = if l < f16_max {
            // Upload F16 weights directly (BF16->F16 for BF16 checkpoints, FP8->dequant->F16 for FP8)
            let upload_concat_f16 = |parts: &[&(usize, TensorEntry)], name: &str, shape: Vec<usize>| -> Result<F16Weight> {
                if f16_only || !fp8_prequant {
                    let buf = concat_tensors(parts, &shards, model_dir)?;
                    let r = arena.region(Box::leak(format!("{name}_L{l}").into_boxed_str()), buf.len(), 16)?;
                    unsafe { r.copy_from_host(&buf)? };
                    Ok(F16Weight { offset_bytes: r.device_ptr(), shape })
                } else {
                    // FP8 dequant with per-channel scales
                    let mut out = Vec::new();
                    for &(si, ref entry) in parts.iter() {
                        let raw = &shards[*si].bytes()[entry.file_offset as usize..(entry.file_offset + entry.nbytes) as usize];
                        let rows = entry.shape[0];
                        let cols = entry.nbytes as usize / rows;
                        let scale_name = format!("{}_scale", entry.name);
                        let ch_scales = if let Some(se) = get_tensor(&scale_name) {
                            read_channelscale_bf16(&se, &shards, rows)
                        } else {
                            vec![1.0 / 448.0; rows]
                        };
                        for r in 0..rows {
                            let rs = ch_scales[r];
                            for c in 0..cols {
                                out.extend_from_slice(&f16::from_f32(fp8_e4m3_to_f32(raw[r * cols + c]) * rs).to_le_bytes());
                            }
                        }
                    }
                    let r = arena.region(Box::leak(format!("{name}_L{l}").into_boxed_str()), out.len(), 16)?;
                    unsafe { r.copy_from_host(&out)? };
                    Ok(F16Weight { offset_bytes: r.device_ptr(), shape })
                }
            };

            let qkv_w = upload_concat_f16(
                &[&q_tensor, &k_tensor, &v_tensor], "qkv_f16",
                vec![qkv_rows, arch.hidden_size],
            )?;
            let o_entry = must_get(&ln("self_attn.o_proj.weight"))?;
            let o_w = upload_concat_f16(&[&o_entry], "o_f16", o_entry.1.shape.clone())?;
            let gate_e = must_get(&ln("mlp.gate_proj.weight"))?;
            let up_e = must_get(&ln("mlp.up_proj.weight"))?;
            let gu_w = upload_concat_f16(
                &[&gate_e, &up_e], "gu_f16",
                vec![2 * arch.intermediate_size, arch.hidden_size],
            )?;
            let d_entry = must_get(&ln("mlp.down_proj.weight"))?;
            let d_w = upload_concat_f16(&[&d_entry], "d_f16", d_entry.1.shape.clone())?;

            if l == 0 { eprintln!("[loader] F16 weights for layers 0..{f16_max}"); }
            (Some(qkv_w), Some(o_w), Some(gu_w), Some(d_w))
        } else {
            (None, None, None, None)
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
            qkv: Some(qkv),
            o_proj: Some(o_proj),
            gate_up: Some(gate_up),
            down_proj: Some(down_proj),
            qkv_f16: qkv_f16_w,
            o_proj_f16: o_proj_f16_w,
            gate_up_f16: gate_up_f16_w,
            down_proj_f16: down_proj_f16_w,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            q_norm,
            k_norm,
            layer_scalar,
            // Cycle 46 step 5c: AWQ load path is wired in cycle 47.
            // For now every checkpoint falls through the FP8 cascade
            // and `Gemma4AwqLayerPtrs` defaults all-zero in bring-up.
            awq: None,
        });
    }

    Ok(Gemma4LoadedModel {
        embedding,
        lm_head_fp8,
        lm_head_f16,
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
    // Proportional RoPE: frequencies use head_dim as divisor, not rotary_dim.
    // Only `half` frequencies are computed (partial rotation), but each
    // frequency value is spaced as if the full head_dim were rotated.
    let inv_theta: Vec<f32> = (0..half)
        .map(|i| 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32))
        .collect();
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
    use rayon::prelude::*;
    bytes
        .par_chunks_exact(2)
        .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect()
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
        channelscale_ptr: None,
        blockscale_ptr: None,
        blockscale_n_blocks: 0,
        blockscale_k_blocks: 0,
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

/// Read an FP8 weight's BF16 scale entry into a per-row f32 vector of
/// length `rows`. Two on-disk layouts are recognized:
///
///   * **Per-channel:** `shape = [rows]` or `[rows, 1]`. Returned flat.
///   * **Blockwise:** `shape = [rows_blocks, cols_blocks]` where
///     `rows_blocks * 128 >= rows`. Expanded to a per-row vector by
///     broadcasting each row-block's first column-block scale over 128
///     rows. This loses the column-block variation — the resulting
///     vector is a compatibility shim for the cuBLASLt
///     `OUTER_VEC_32F` path, which only consumes per-row scales.
///     Full-fidelity consumption of blockwise scales requires a
///     block-scale GEMV launcher (see `rvllm-kernels::gb10_dispatch` +
///     `fp8_gemv_blockwise_wpr_*_kernel`), which is not wired into the
///     runtime's launch path yet.
///
/// Panics on any other layout — better a clear panic at load-time than
/// silent miscalibration at inference.
///
/// **Use `read_blockscale_bf16` instead** when the consumer's kernel
/// ABI expects a 2-D blockscale tensor (e.g. `Fp8GemvF16InLaunch`).
/// This function collapses the 2-D source to a per-row projection
/// which is the wrong shape for those kernels; calling them with a
/// projected vector walks off the end of the buffer into neighbouring
/// weights' memory.
fn read_channelscale_bf16(
    scale_entry: &(usize, TensorEntry),
    shards: &[ShardMap],
    rows: usize,
) -> Vec<f32> {
    let (si, e) = scale_entry;
    let raw = &shards[*si].bytes()[e.file_offset as usize..(e.file_offset + e.nbytes) as usize];
    let n = raw.len() / 2;
    // bf16 → f32: place bf16 in upper half of u32, zero lower.
    let bf16_le_to_f32 = |lo: u8, hi: u8| f32::from_bits(u32::from_le_bytes([0, 0, lo, hi]));

    // Case 1: per-channel (1-D or trivially [rows, 1]).
    if n == rows && (e.shape.len() == 1 || (e.shape.len() == 2 && e.shape[1] == 1)) {
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push(bf16_le_to_f32(raw[2 * i], raw[2 * i + 1]));
        }
        return out;
    }

    // Case 2: 2-D block-scale `[rows_blocks, cols_blocks]`.
    if e.shape.len() == 2 {
        let rows_blocks = e.shape[0];
        let cols_blocks = e.shape[1];
        const BLOCK: usize = 128;
        assert_eq!(
            n,
            rows_blocks * cols_blocks,
            "block-scale flat count {n} != rows_blocks {rows_blocks} * cols_blocks {cols_blocks}",
        );
        assert!(
            rows_blocks * BLOCK >= rows,
            "block-scale rows_blocks {rows_blocks} * {BLOCK} < weight rows {rows}",
        );
        let mut out = Vec::with_capacity(rows);
        for r in 0..rows {
            let rb = r / BLOCK;
            // First column-block's scale for this row-block.
            let idx = rb * cols_blocks;
            out.push(bf16_le_to_f32(raw[2 * idx], raw[2 * idx + 1]));
        }
        return out;
    }

    panic!(
        "unrecognized FP8 scale layout for {:?}: shape={:?}, flat_count={n}, expected rows={rows}",
        e.name, e.shape
    );
}

/// Pure-function core of `read_blockscale_bf16`: bf16-LE bytes →
/// `Vec<f32>`. Lifted out so we can unit-test the decode path
/// without constructing a `ShardMap` (which needs a real mmap'd
/// file). The outer `read_blockscale_bf16` wrapper handles the
/// shape dispatch + ShardMap slicing.
fn decode_blockscale_bytes(raw: &[u8], rows_blocks: usize, cols_blocks: usize) -> Option<Vec<f32>> {
    let n = raw.len() / 2;
    if n != rows_blocks * cols_blocks {
        return None;
    }
    let bf16_le_to_f32 = |lo: u8, hi: u8| f32::from_bits(u32::from_le_bytes([0, 0, lo, hi]));
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(bf16_le_to_f32(raw[2 * i], raw[2 * i + 1]));
    }
    Some(out)
}

/// Read a 2-D blockwise FP8 scale tensor into a raw f32 buffer
/// preserving its `[rows_blocks, cols_blocks]` shape. Returns
/// `None` when the source is per-channel 1-D (not blockscale).
///
/// Use for consumers whose kernel ABI is `[N_blocks, K_blocks]`
/// (e.g. CUTLASS SFB, `fp8_gemv_blockwise_wpr_*`). For the
/// compatibility per-row projection, call `read_channelscale_bf16`.
fn read_blockscale_bf16(
    scale_entry: &(usize, TensorEntry),
    shards: &[ShardMap],
) -> Option<(Vec<f32>, usize, usize)> {
    let (si, e) = scale_entry;
    if e.shape.len() != 2 {
        return None;
    }
    let rows_blocks = e.shape[0];
    let cols_blocks = e.shape[1];
    let raw = &shards[*si].bytes()[e.file_offset as usize..(e.file_offset + e.nbytes) as usize];
    let decoded = decode_blockscale_bytes(raw, rows_blocks, cols_blocks)?;
    Some((decoded, rows_blocks, cols_blocks))
}

#[cfg(test)]
mod blockscale_tests {
    use super::*;

    // bf16 = upper 16 bits of f32.
    fn f32_to_bf16_le(x: f32) -> [u8; 2] {
        let bits = x.to_bits();
        let hi = ((bits >> 16) & 0xFFFF) as u16;
        hi.to_le_bytes()
    }

    #[test]
    fn decode_blockscale_roundtrips_representative_values() {
        // Two row-blocks × three col-blocks = 6 bf16 scales. Values
        // chosen so they survive bf16 round-trip exactly (powers of 2
        // + simple fractions).
        let src = [0.25_f32, 0.5, 1.0, 2.0, 4.0, 0.125];
        let mut raw = Vec::with_capacity(src.len() * 2);
        for v in &src { raw.extend_from_slice(&f32_to_bf16_le(*v)); }

        let out = decode_blockscale_bytes(&raw, 2, 3).expect("shape fits");
        assert_eq!(out, src);
    }

    #[test]
    fn decode_blockscale_rejects_shape_mismatch() {
        // 8 bytes = 4 bf16 values, but caller claims 2x3 = 6. Should
        // reject rather than silently read garbage.
        let raw = vec![0u8; 8];
        assert!(decode_blockscale_bytes(&raw, 2, 3).is_none());
    }

    #[test]
    fn decode_blockscale_preserves_layout_order() {
        // Regression guard: read order must be row-major, matching the
        // on-disk safetensors layout. This is the exact bug the PR
        // reviewer flagged — `channelscale_ptr` projected the 2-D
        // tensor to per-row via `rb * cols_blocks` indexing. Check
        // the full blockscale preserves column ordering too.
        let src: Vec<f32> = (0..6).map(|i| (i as f32) * 0.5).collect();
        let raw: Vec<u8> = src.iter().flat_map(|v| f32_to_bf16_le(*v)).collect();
        let out = decode_blockscale_bytes(&raw, 2, 3).unwrap();
        // Expect row 0: [0.0, 0.5, 1.0], row 1: [1.5, 2.0, 2.5].
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 0.5);
        assert_eq!(out[2], 1.0);
        assert_eq!(out[3], 1.5);
        assert_eq!(out[4], 2.0);
        assert_eq!(out[5], 2.5);
    }
}

/// Upload pre-quantized FP8 weight with per-channel BF16 scales.
/// Raw FP8 bytes go straight to GPU. Per-channel scales uploaded as f32
/// vector. Weight scalar scale set to 1.0 -- channelscale applied post-GEMM.
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
    let region = arena.region(region_name, raw.len(), 16)?;
    unsafe { region.copy_from_host(raw)? };
    if let Some(se) = scale_entry {
        // Legacy per-row channelscale vector (compat path for cuBLASLt
        // OUTER_VEC_32F + any consumer that only knows per-row).
        let ch_scales = read_channelscale_bf16(se, shards, rows);
        let scale_bytes: Vec<u8> = ch_scales.iter().flat_map(|s| s.to_le_bytes()).collect();
        let cs_r = arena.region("fp8_chscale", scale_bytes.len(), 16)?;
        unsafe { cs_r.copy_from_host(&scale_bytes)? };
        // Parallel full 2-D blockscale tensor when the source is
        // shaped [N_blocks, K_blocks]. Consumed by kernels whose ABI
        // expects the full 2-D shape (e.g. Fp8GemvBlockwiseF16InLaunch
        // on the sm_121 fast path). `None` when the source is per-row
        // 1-D — those weights stay on the channelscale path.
        let (block_ptr, n_blocks, k_blocks) =
            if let Some((bs, rb, cb)) = read_blockscale_bf16(se, shards) {
                let bs_bytes: Vec<u8> = bs.iter().flat_map(|s| s.to_le_bytes()).collect();
                let bs_r = arena.region("fp8_blockscale", bs_bytes.len(), 16)?;
                unsafe { bs_r.copy_from_host(&bs_bytes)? };
                (Some(bs_r.device_ptr()), rb as u32, cb as u32)
            } else {
                (None, 0u32, 0u32)
            };
        let one = 1.0f32;
        let one_r = arena.region("fp8_scale", 4, 4)?;
        unsafe { one_r.copy_from_host(&one.to_le_bytes())? };
        Ok(Fp8Weight {
            offset_bytes: region.device_ptr(),
            scale_ptr: one_r.device_ptr(),
            shape: entry.shape.clone(),
            scale: 1.0,
            clamp_ppm: 0.0,
            dtype: DType::Fp8E4M3,
            channelscale_ptr: Some(cs_r.device_ptr()),
            blockscale_ptr: block_ptr,
            blockscale_n_blocks: n_blocks,
            blockscale_k_blocks: k_blocks,
        })
    } else {
        let fallback = 1.0f32 / 448.0;
        let sr = arena.region("fp8_scale", 4, 4)?;
        unsafe { sr.copy_from_host(&fallback.to_le_bytes())? };
        Ok(Fp8Weight {
            offset_bytes: region.device_ptr(),
            scale_ptr: sr.device_ptr(),
            shape: entry.shape.clone(),
            scale: fallback,
            clamp_ppm: 0.0,
            dtype: DType::Fp8E4M3,
            channelscale_ptr: None,
            blockscale_ptr: None,
            blockscale_n_blocks: 0,
            blockscale_k_blocks: 0,
        })
    }
}

/// Fuse multiple pre-quantized FP8 tensors (QKV, gate+up) with per-channel
/// scales. Raw FP8 bytes concatenated, per-channel scale vectors concatenated.
/// Weight scalar scale = 1.0, channelscale applied post-GEMM.
fn fuse_fp8_direct_channelscale(
    arena: &HbmArena,
    region_name: &'static str,
    parts: &[&(usize, TensorEntry)],
    scale_entries: &[Option<&(usize, TensorEntry)>],
    shards: &[ShardMap],
    fused_shape: &[usize],
) -> Result<Fp8Weight> {
    let mut fused_bytes = Vec::new();
    let mut fused_scales: Vec<f32> = Vec::new();
    let mut has_scales = false;

    for (i, &(si, ref entry)) in parts.iter().enumerate() {
        let raw = &shards[*si].bytes()[entry.file_offset as usize..(entry.file_offset + entry.nbytes) as usize];
        fused_bytes.extend_from_slice(raw);
        let rows = entry.shape[0];
        if let Some(se) = scale_entries.get(i).and_then(|x| x.as_ref()) {
            // `read_channelscale_bf16` guarantees `len() == rows` for
            // both per-channel and blockwise layouts — see its docs.
            let ch = read_channelscale_bf16(se, shards, rows);
            fused_scales.extend_from_slice(&ch);
            has_scales = true;
        } else {
            fused_scales.extend(std::iter::repeat(1.0 / 448.0).take(rows));
        }
    }

    let region = arena.region(region_name, fused_bytes.len(), 16)?;
    unsafe { region.copy_from_host(&fused_bytes)? };

    let (scale_ptr, channelscale_ptr) = if has_scales {
        let scale_bytes: Vec<u8> = fused_scales.iter().flat_map(|s| s.to_le_bytes()).collect();
        let cs_r = arena.region("fp8_chscale", scale_bytes.len(), 16)?;
        unsafe { cs_r.copy_from_host(&scale_bytes)? };
        let one = 1.0f32;
        let one_r = arena.region("fp8_scale", 4, 4)?;
        unsafe { one_r.copy_from_host(&one.to_le_bytes())? };
        (one_r.device_ptr(), Some(cs_r.device_ptr()))
    } else {
        let fallback = 1.0f32 / 448.0;
        let sr = arena.region("fp8_scale", 4, 4)?;
        unsafe { sr.copy_from_host(&fallback.to_le_bytes())? };
        (sr.device_ptr(), None)
    };

    Ok(Fp8Weight {
        offset_bytes: region.device_ptr(),
        scale_ptr,
        shape: fused_shape.to_vec(),
        scale: 1.0,
        clamp_ppm: 0.0,
        dtype: DType::Fp8E4M3,
        channelscale_ptr,
        // Fused qkv / gate_up synthesis collapses per-part scales into
        // one concatenated per-row vector; 2-D blockscale reconstruction
        // across parts isn't well-defined (different parts ship with
        // different block alignments), so these weights never take
        // the blockscale fast path. Synthesised → `blockscale_ptr =
        // None`, any GEMM path that reads `blockscale_ptr` must fall
        // back to the channelscale-preserving path.
        blockscale_ptr: None,
        blockscale_n_blocks: 0,
        blockscale_k_blocks: 0,
    })
}

fn quantize_to_fp8_bytes(f32_vals: &[f32], scale: f32) -> Vec<u8> {
    use rayon::prelude::*;
    let inv = 1.0 / scale;
    f32_vals
        .par_iter()
        .map(|v| fp8_e4m3_encode((*v * inv).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)))
        .collect()
}

fn fp8_e4m3_encode(v: f32) -> u8 {
    if v.is_nan() {
        return 0x7f;
    }
    let s: u8 = if v.to_bits() >> 31 != 0 { 0x80 } else { 0 };
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
    let mut exp8 = exp32 + 7;
    if exp8 <= 0 {
        let shift = 1 - exp8;
        if shift >= 12 {
            return s;
        }
        let full = (mant32 | (1 << 23)) as u32;
        let rshift = (20 + shift) as u32;
        let mut m = full >> rshift;
        let round_bit = if rshift > 0 { (full >> (rshift - 1)) & 1 } else { 0 };
        let sticky = if rshift > 1 { (full & ((1 << (rshift - 1)) - 1) != 0) as u32 } else { 0 };
        m += round_bit & (sticky | (m & 1));
        if m >= 8 {
            return s | 0x08; // overflow to smallest normal: exp=1, m=0
        }
        return s | (m as u8 & 0x07);
    }
    // round-to-nearest-even: 20 bits dropped from f32 mantissa
    let trunc = mant32 >> 20;
    let round_bit = (mant32 >> 19) & 1;
    let sticky = (mant32 & 0x7_ffff) != 0;
    let m = trunc + (round_bit & (sticky as u32 | (trunc & 1)));
    if m >= 8 {
        exp8 += 1;
        // E4M3FN: exp=15 is valid (max finite=0x7e=448), only 0x7f is NaN
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

/// Cycle 49 step 8a: AWQ-only Gemma 4 load path. Implements the
/// non-quantized preludes (embedding sqrt(H) pre-scale, final_norm,
/// lm_head, RoPE tables) and stubs the per-layer body so cycle 49
/// step 8b can land per-layer norms, 8c can land the AWQ upload, and
/// 8d can land debug-probe guards + smoke.
///
/// This is structurally a near-twin of `load_gemma4_model`'s FP8 path
/// (lines ~140-250 above) but tailored for AWQ checkpoints:
/// - lm_head + embedding stay BF16-source per AwqConfig.ignore (no
///   FP8E4M3 weight tensor exists in AWQ checkpoints).
/// - per-layer FP8 fields all populate as `None`; AWQ tensors land
///   in `layer.awq` instead.
fn load_gemma4_awq_model_inner(
    model_dir: &Path,
    arena: &HbmArena,
    arch: &crate::gemma4_arch::Gemma4Arch,
    awq: &crate::compressed_tensors::AwqConfig,
    shards: &[ShardMap],
    tensors: &BTreeMap<String, (usize, TensorEntry)>,
) -> Result<Gemma4LoadedModel> {
    let prefix = &arch.weight_prefix;

    let bytes_of = |si: usize, e: &TensorEntry| -> &[u8] {
        let s = shards[si].bytes();
        let start = e.file_offset as usize;
        &s[start..start + e.nbytes as usize]
    };
    let must_get = |name: &str| -> Result<(usize, TensorEntry)> {
        tensors.get(name).cloned().ok_or_else(|| RvllmError::Loader {
            err: LoaderError::MissingTensor { name: name.to_string() },
            ctx: LoaderCtx {
                path: model_dir.to_path_buf(),
                tensor: Some(name.to_string()),
            },
            bt: std::backtrace::Backtrace::capture(),
        })
    };
    let get_tensor = |name: &str| tensors.get(name).cloned();

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

    // ----- embedding (sqrt(H) pre-scale, identical to FP8 path) -----
    let embed_name = format!("{prefix}.embed_tokens.weight");
    let embedding = {
        let (si, e) = must_get(&embed_name)?;
        let mut buf = tensor_to_f16_bytes(&e, bytes_of(si, &e), model_dir)?;
        let scale = (arch.hidden_size as f32).sqrt();
        eprintln!("[awq-loader] embedding sqrt({}) = {:.2}", arch.hidden_size, scale);
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
        F16Weight { offset_bytes: region.device_ptr(), shape: e.shape.clone() }
    };

    // ----- final_norm (F16) -----
    let final_norm = {
        let nm = format!("{prefix}.norm.weight");
        let (si, e) = must_get(&nm)?;
        let buf = tensor_to_f16_bytes(&e, bytes_of(si, &e), model_dir)?;
        let region = arena.region("final_norm", buf.len(), 16)?;
        unsafe { region.copy_from_host(&buf)? };
        F16Weight { offset_bytes: region.device_ptr(), shape: e.shape.clone() }
    };

    // ----- lm_head (typically in AwqConfig.ignore → BF16 source) -----
    // AwqConfig.ignore lists "lm_head" → it's stored as BF16 in the
    // safetensors. Same code paths as the FP8-from-BF16 branch in
    // load_gemma4_model. fp8 form for the existing decode path; f16
    // form for the f16 logit projection.
    let lm_head_fp8 = if let Some((si, e)) = get_tensor("lm_head.weight") {
        if e.dtype == DType::Fp8E4M3 {
            let scale_entry = get_tensor("lm_head.weight_scale");
            upload_fp8_direct_channelscale(arena, "lm_head", &(si, e), scale_entry.as_ref(), shards)?
        } else {
            upload_fp8(arena, "lm_head",
                &tensor_to_f16_bytes(&e, bytes_of(si, &e), model_dir)?,
                &e.shape, "lm_head.weight", model_dir)?
        }
    } else {
        let (si, e) = must_get(&embed_name)?;
        let buf = tensor_to_f16_bytes(&e, bytes_of(si, &e), model_dir)?;
        upload_fp8(arena, "lm_head", &buf, &e.shape, "lm_head(tied_embed)", model_dir)?
    };
    let lm_head_f16 = {
        let (si, e) = if let Some(t) = get_tensor("lm_head.weight") { t } else { must_get(&embed_name)? };
        let buf = tensor_to_f16_bytes(&e, bytes_of(si, &e), model_dir)?;
        let region = arena.region("lm_head_f16", buf.len(), 16)?;
        unsafe { region.copy_from_host(&buf)? };
        F16Weight { offset_bytes: region.device_ptr(), shape: e.shape.clone() }
    };

    // ----- RoPE tables (sliding + global, identical to FP8 path) -----
    let sliding_rotary_dim = arch.head_dim_sliding;
    let (cos_s, sin_s) = rope_cos_sin_bytes(
        arch.head_dim_sliding, arch.max_position_embeddings,
        arch.rope_theta_sliding, sliding_rotary_dim,
    );
    let global_rotary_dim = arch.rotary_dim_for_layer(
        arch.layer_types.iter().position(|t| *t == crate::gemma4_arch::Gemma4LayerType::GlobalAttention).unwrap_or(0)
    );
    let (cos_g, sin_g) = rope_cos_sin_bytes(
        arch.head_dim_global, arch.max_position_embeddings,
        arch.rope_theta_global, global_rotary_dim,
    );
    let rope_cos_sliding = upload_rope(arena, "rope_cos_sliding", &cos_s)?;
    let rope_sin_sliding = upload_rope(arena, "rope_sin_sliding", &sin_s)?;
    let rope_cos_global  = upload_rope(arena, "rope_cos_global",  &cos_g)?;
    let rope_sin_global  = upload_rope(arena, "rope_sin_global",  &sin_g)?;

    // ----- per-layer (cycle 49 step 8b: norms uploaded; 8c stubs AWQ) -----
    let load_max_layers = std::env::var("RVLLM_MAX_LAYERS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .map(|v| v.min(arch.num_hidden_layers))
        .unwrap_or(arch.num_hidden_layers);
    if load_max_layers < arch.num_hidden_layers {
        eprintln!(
            "[awq-loader] RVLLM_MAX_LAYERS={load_max_layers}: loading only first {load_max_layers} of {} layers",
            arch.num_hidden_layers
        );
    }

    let mut layers: Vec<crate::gemma4_weights::Gemma4LayerWeights> = Vec::with_capacity(load_max_layers);
    for l in 0..load_max_layers {
        let ln = |s: &str| format!("{prefix}.layers.{l}.{s}");

        // F16 norms (identical names + shapes to FP8 path).
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

        // Cycle 49 step 8c: AWQ tensor upload via upload_gemma4_awq_layer.
        // Build a byte_lookup closure that maps safetensors entry name
        // to (DType, shape, owned bytes Vec). For global layers
        // (attention_k_eq_v=true), self_attn.v_proj.* aliases to
        // self_attn.k_proj.* — runtime sees v_packed/scale/zero
        // pointers identical to k_packed/scale/zero, the AWQ kernel
        // happily produces the V output by re-running on K's weights.
        let (q_out, kv_out, o_in, has_v_proj) = awq_layer_shape_for(arch, l);
        let layer_prefix = format!("{prefix}.layers.{l}");
        let v_alias_prefix = format!("{layer_prefix}.self_attn.v_proj.");
        let k_alias_prefix = format!("{layer_prefix}.self_attn.k_proj.");
        let byte_lookup = |name: &str| -> Option<(rvllm_core::DType, Vec<usize>, Vec<u8>)> {
            // Global-layer alias: redirect any v_proj.* lookup to k_proj.*
            // so upload_gemma4_awq_layer's "all 7 linears" contract
            // succeeds without a synthetic v_proj tensor.
            let aliased: String;
            let resolved = if !has_v_proj && name.starts_with(&v_alias_prefix) {
                aliased = format!("{}{}", k_alias_prefix, &name[v_alias_prefix.len()..]);
                aliased.as_str()
            } else {
                name
            };
            let (si, e) = tensors.get(resolved)?.clone();
            let raw = bytes_of(si, &e).to_vec();
            Some((e.dtype, e.shape.clone(), raw))
        };
        // Reuse awq_layer_shape_for outputs (codex review of 92cd9e0)
        // — keeps the validator and the upload path in lock-step on
        // global v_proj / heterogeneous head-dim handling.
        let geom = crate::compressed_tensors::AwqLayerShapes {
            q_out, kv_out,
            o_out:            arch.hidden_size,
            o_in,
            mlp_intermediate: arch.intermediate_size,
            hidden:           arch.hidden_size,
        };
        let awq_layer = crate::compressed_tensors::upload_gemma4_awq_layer(
            arena, &layer_prefix, &geom, &awq.scheme, &byte_lookup, has_v_proj,
        ).map_err(|e| RvllmError::Loader {
            err: LoaderError::Corrupt {
                detail: format!("layer {l} AWQ upload: {e}"),
            },
            ctx: LoaderCtx {
                path: model_dir.to_path_buf(),
                tensor: Some(format!("{layer_prefix}.self_attn.*")),
            },
            bt: std::backtrace::Backtrace::capture(),
        })?;

        // Cycle 49 step 8e: upload_gemma4_awq_layer now handles the V/K
        // alias internally (skips the v_proj arena region for globals
        // and clones K's AwqLinearWeight into the v_proj slot), so no
        // post-upload alias step is needed here.

        if l == 0 || l == load_max_layers - 1 {
            eprintln!(
                "[awq-loader] layer {l} ({:?}) AWQ uploaded: q_packed=0x{:x} (group_size={}{})",
                arch.layer_types[l], awq_layer.q_proj.packed_offset_bytes, awq_layer.q_proj.group_size,
                if has_v_proj { "" } else { ", v=k alias" },
            );
        }

        layers.push(crate::gemma4_weights::Gemma4LayerWeights {
            qkv:       None,  // AWQ replaces FP8 fused QKV
            o_proj:    None,
            gate_up:   None,
            down_proj: None,
            qkv_f16:        None,
            o_proj_f16:     None,
            gate_up_f16:    None,
            down_proj_f16:  None,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            q_norm,
            k_norm,
            layer_scalar,
            awq: Some(awq_layer),
        });
    }

    eprintln!(
        "[awq-loader] all {} layers uploaded; building Gemma4LoadedModel",
        layers.len()
    );

    Ok(Gemma4LoadedModel {
        embedding,
        lm_head_fp8,
        lm_head_f16,
        final_norm,
        rope_cos_sliding,
        rope_sin_sliding,
        rope_cos_global,
        rope_sin_global,
        layers,
    })
}

/// Cycle 48 step 7c: per-layer AWQ geometry derived from
/// `Gemma4Arch.layer_types[i]`. Shared between the validator
/// (`validate_gemma4_awq_layout`) and the future upload path so they
/// can't drift.
///
/// Returns `(q_out, kv_out, o_in, has_v_proj)`:
///   Sliding: q = num_attention_heads * head_dim_sliding, kv similar;
///            has_v_proj = true (all 7 linears in the checkpoint).
///   Global:  q = num_attention_heads * head_dim_global,  kv similar;
///            has_v_proj = false (`attention_k_eq_v=true`, K reused as V).
pub(crate) fn awq_layer_shape_for(
    arch: &crate::gemma4_arch::Gemma4Arch,
    layer_idx: usize,
) -> (usize, usize, usize, bool) {
    match arch.layer_types[layer_idx] {
        crate::gemma4_arch::Gemma4LayerType::SlidingAttention => {
            let q = arch.num_attention_heads * arch.head_dim_sliding;
            let kv = arch.num_kv_heads_sliding * arch.head_dim_sliding;
            (q, kv, q, true)
        }
        crate::gemma4_arch::Gemma4LayerType::GlobalAttention => {
            let q = arch.num_attention_heads * arch.head_dim_global;
            let kv = arch.num_kv_heads_global * arch.head_dim_global;
            (q, kv, q, false)
        }
    }
}

/// Cycle 48 step 7c roadmap (documented here so cycle 49 can pick up
/// without re-deriving the plan from scratch):
///
/// 1. Drop the `UnsupportedQuantization` early-return at the
///    validator-pass site in `load_gemma4_model`.
///
/// 2. Branch into a separate `load_gemma4_awq_model_inner` function
///    that is structurally a near-twin of the FP8 path:
///      a. embedding (sqrt(H) pre-scale, F16 upload)
///      b. final_norm (F16 upload)
///      c. lm_head: AwqConfig.ignore typically lists "lm_head" → loaded
///         as-is via the existing CPU-quantize-to-FP8 fallback (the
///         BF16 source matches the FP8 path's existing input shape)
///      d. RoPE tables (sliding + global, identical to FP8 path)
///      e. per-layer:
///         * f16 norms (input/post_attn/pre_ff/post_ff_layernorm,
///           q_norm, k_norm, layer_scalar) — identical to FP8 path
///         * Build a byte_lookup closure over &shards + &tensors that
///           returns (DType, Vec<usize>, Vec<u8>) per entry name. The
///           bytes Vec is owned (memcpy from mmap) so the closure
///           doesn't borrow shards across `upload_gemma4_awq_layer`'s
///           internal allocations.
///         * Derive per-layer geometry via `awq_layer_shape_for(arch, l)`.
///           Build `AwqLayerShapes { q_out, kv_out, o_in, o_out: hidden,
///           mlp_intermediate: arch.intermediate_size, hidden:
///           arch.hidden_size }`.
///         * Call `upload_gemma4_awq_layer(arena, &layer_prefix, &geom,
///           &awq.scheme, &byte_lookup)` — gets back AwqLayerWeights.
///         * For global layers, splice K's AwqLinearWeight into V's slot
///           in the returned struct (mirrors the FP8 path's K-as-V
///           reuse). Or: have a `upload_gemma4_awq_layer_global` that
///           skips v_proj and copies q_proj-style — TBD in cycle 49.
///         * Push `Gemma4LayerWeights { qkv: None, o_proj: None,
///           gate_up: None, down_proj: None, qkv_f16: None,
///           o_proj_f16: None, gate_up_f16: None, down_proj_f16: None,
///           input_layernorm, ..., awq: Some(awq_layer_weights) }`.
///      f. Build `Gemma4LoadedModel { embedding, lm_head_fp8,
///         lm_head_f16, final_norm, rope_*, layers }`.
///
/// 3. Codex finding from cycle 48 step 7a: guard CUDA debug probes in
///    `gemma4_layer_exec.rs` that unconditionally read `weights.qkv_scale`
///    + `weights.qkv_fp8`. Add `if weights.qkv_fp8 != 0` around them
///    so AWQ-only layers (qkv_fp8 = 0) don't trigger SIGSEGV in
///    debug mode.
///
/// 4. Smoke test: load `ebircak/gemma-4-31B-it-4bit-W4A16-AWQ`, fire
///    the same prompts the FP8 production path serves
///    (`v3/scripts/bench_sm121.sh 1 1` + ZeroClaw weather query),
///    confirm decoding works, measure tok/s. Target: 6-10 tok/s vs
///    FP8 production 3.3-4.0.
///
/// Walk every Gemma 4 layer's expected AWQ tensor
/// names against the shard index, validate dtype + shape per linear
/// via `validate_awq_linear`. Sliding layers (`head_dim=256, kv=16`)
/// have different N for q/k/v/o vs global layers
/// (`head_dim=512, kv=4`); per-layer geometry is derived from
/// `arch.layer_types[i]` so heterogeneous Gemma 4 attention is handled
/// the same way the bring-up reads it.
///
/// Returns `Ok(())` when every linear in every non-ignored layer
/// validates cleanly. Returns a descriptive `Err(String)` on the
/// first failure so the caller can attribute the abort to one
/// specific tensor.
fn validate_gemma4_awq_layout(
    arch: &crate::gemma4_arch::Gemma4Arch,
    awq: &crate::compressed_tensors::AwqConfig,
    tensors: &BTreeMap<String, (usize, TensorEntry)>,
) -> std::result::Result<(), String> {
    let lookup = |key: &str| -> Option<(rvllm_core::DType, Vec<usize>)> {
        tensors.get(key).map(|(_, e)| (e.dtype, e.shape.clone()))
    };
    let prefix = &arch.weight_prefix;
    let mut total_validated = 0usize;
    for (i, lt) in arch.layer_types.iter().enumerate() {
        // Per-layer geometry via the shared helper so the validator
        // and the cycle-49 upload path can't drift on global v_proj
        // handling.
        let (q_out, kv_out, o_in, has_v_proj) = awq_layer_shape_for(arch, i);
        let layer_prefix = format!("{prefix}.layers.{i}");
        let mut layer_validated = 0usize;
        let mut lin = |name: &str, dense: [usize; 2]| -> std::result::Result<(), String> {
            let full = format!("{layer_prefix}.{name}");
            if awq.is_ignored(&full) {
                return Ok(()); // not AWQ-quantized; FP8 path will load it
            }
            crate::compressed_tensors::validate_awq_linear(
                &full, dense, &awq.scheme, &lookup,
            ).map(|_| { layer_validated += 1; })
              .map_err(|e| format!("layer {i} ({lt:?}) {name}: {e}"))
        };
        lin("self_attn.q_proj",  [q_out, arch.hidden_size])?;
        lin("self_attn.k_proj",  [kv_out, arch.hidden_size])?;
        if has_v_proj {
            lin("self_attn.v_proj",  [kv_out, arch.hidden_size])?;
        }
        lin("self_attn.o_proj",  [arch.hidden_size, o_in])?;
        lin("mlp.gate_proj",     [arch.intermediate_size, arch.hidden_size])?;
        lin("mlp.up_proj",       [arch.intermediate_size, arch.hidden_size])?;
        lin("mlp.down_proj",     [arch.hidden_size, arch.intermediate_size])?;
        total_validated += layer_validated;
    }
    eprintln!(
        "[loader] AWQ layout validated: {} layer-linear entries across {} layers",
        total_validated, arch.num_hidden_layers
    );
    Ok(())
}

#[cfg(test)]
mod fp8_tests {
    use super::*;

    fn all_fp8_values() -> Vec<(u8, f32)> {
        (0..=255u8)
            .filter_map(|b| {
                let v = fp8_e4m3_to_f32(b);
                if v.is_nan() { None } else { Some((b, v)) }
            })
            .collect()
    }

    fn brute_nearest_fp8(x: f32) -> u8 {
        if x.is_nan() { return 0x7f; }
        let vals = all_fp8_values();
        let mut best_byte = 0u8;
        let mut best_dist = f64::MAX;
        let mut best_val = 0.0f64;
        for &(b, fv) in &vals {
            let d = (x as f64 - fv as f64).abs();
            if d < best_dist || (d == best_dist && {
                let bm = b & 0x07;
                let prev_m = best_byte & 0x07;
                (bm % 2 == 0) && (prev_m % 2 != 0)
            }) {
                best_dist = d;
                best_byte = b;
                best_val = fv as f64;
            }
        }
        let _ = best_val;
        best_byte
    }

    #[test]
    fn roundtrip_all_256_bytes() {
        let mut fails = Vec::new();
        for b in 0..=255u8 {
            let v = fp8_e4m3_to_f32(b);
            if v.is_nan() { continue; }
            let re = fp8_e4m3_encode(v);
            if re != b {
                fails.push((b, v, re));
            }
        }
        if !fails.is_empty() {
            for (b, v, re) in &fails {
                eprintln!("ROUNDTRIP FAIL: byte 0x{b:02x}({b}) -> f32={v} -> encode=0x{re:02x}({re})");
            }
            panic!("{} of 255 roundtrips failed", fails.len());
        }
    }

    #[test]
    fn midpoints_bankers_rounding() {
        let vals = all_fp8_values();
        let positives: Vec<(u8, f32)> = vals.iter()
            .filter(|(_, v)| *v > 0.0)
            .copied()
            .collect();
        let mut fails = Vec::new();
        for w in positives.windows(2) {
            let (b_lo, v_lo) = w[0];
            let (b_hi, v_hi) = w[1];
            let mid = (v_lo as f64 + v_hi as f64) / 2.0;
            let mid_f32 = mid as f32;
            if mid_f32 as f64 != mid { continue; }
            let m_lo = b_lo & 0x07;
            let m_hi = b_hi & 0x07;
            let expected = if m_lo % 2 == 0 { b_lo } else { b_hi };
            let got = fp8_e4m3_encode(mid_f32);
            if got != expected {
                fails.push((mid_f32, b_lo, b_hi, expected, got));
            }
        }
        if !fails.is_empty() {
            for (mid, lo, hi, exp, got) in &fails {
                eprintln!("MIDPOINT FAIL: {mid} between 0x{lo:02x}({lo}) and 0x{hi:02x}({hi}): expected 0x{exp:02x} got 0x{got:02x}");
            }
            panic!("{} midpoint rounding failures", fails.len());
        }
    }

    #[test]
    fn sweep_all_f32_in_fp8_range() {
        let pos_vals: Vec<(u8, f32)> = all_fp8_values().into_iter()
            .filter(|(_, v)| *v >= 0.0)
            .collect();
        let mut boundaries: Vec<(f32, u8)> = Vec::new();
        for w in pos_vals.windows(2) {
            let (b_lo, v_lo) = w[0];
            let (b_hi, v_hi) = w[1];
            let mid = ((v_lo as f64 + v_hi as f64) / 2.0) as f32;
            let m_lo = b_lo & 0x07;
            let rte_byte = if m_lo % 2 == 0 { b_lo } else { b_hi };
            boundaries.push((mid, rte_byte));
        }

        let expected_for = |v: f32| -> u8 {
            if v == 0.0 { return 0; }
            if v > 448.0 { return 0x7e; }
            for w in pos_vals.windows(2) {
                let (b_lo, v_lo) = w[0];
                let (b_hi, v_hi) = w[1];
                if v >= v_lo && v <= v_hi {
                    if v == v_lo { return b_lo; }
                    if v == v_hi { return b_hi; }
                    let d_lo = (v as f64 - v_lo as f64).abs();
                    let d_hi = (v as f64 - v_hi as f64).abs();
                    if d_lo < d_hi { return b_lo; }
                    if d_hi < d_lo { return b_hi; }
                    let m_lo = b_lo & 0x07;
                    return if m_lo % 2 == 0 { b_lo } else { b_hi };
                }
            }
            if v <= pos_vals[0].1 {
                let d = (v as f64 - pos_vals[0].1 as f64).abs();
                if d < pos_vals[0].1 as f64 / 2.0 { return pos_vals[0].0; }
                return 0;
            }
            0x7e
        };

        let mut total = 0u64;
        let mut fails = 0u64;
        let mut first_fails: Vec<(f32, u8, u8)> = Vec::new();
        let max_bits = 448.0f32.to_bits();
        for bits in 0..=max_bits {
            let v = f32::from_bits(bits);
            if v.is_nan() || v.is_infinite() || v < 0.0 { continue; }
            total += 1;
            let got = fp8_e4m3_encode(v);
            let exp = expected_for(v);
            if got != exp {
                fails += 1;
                if first_fails.len() < 20 {
                    first_fails.push((v, exp, got));
                }
            }
        }
        if !first_fails.is_empty() {
            eprintln!("\n=== FP8 ENCODER MISMATCHES ({fails}/{total}) ===");
            for (v, exp, got) in &first_fails {
                let exp_v = fp8_e4m3_to_f32(*exp);
                let got_v = fp8_e4m3_to_f32(*got);
                eprintln!("  {v:.10} (0x{:08x}): expected 0x{exp:02x}={exp_v} got 0x{got:02x}={got_v}",
                    v.to_bits());
            }
            panic!("{fails} of {total} positive f32 values encoded wrong");
        }
        eprintln!("PASS: all {total} positive f32 values in [0, 448] encode correctly");
    }

    #[test]
    fn test_specific_mismatch_values() {
        let vals: &[f32] = &[-10.071, -80.569, 9.352, -74.814, -63.304, -25.897, -4.316, -20.142];
        let nvidia: &[u8] = &[210, 234, 81, 233, 232, 221, 201, 218];
        let mut fails = Vec::new();
        for (i, &v) in vals.iter().enumerate() {
            let rust_byte = fp8_e4m3_encode(v);
            let brute_byte = brute_nearest_fp8(v);
            let nv = nvidia[i];
            eprintln!("  {v:8.3}: rust=0x{rust_byte:02x}({rust_byte:3}) brute=0x{brute_byte:02x}({brute_byte:3}) nvidia=0x{nv:02x}({nv:3}) \
                rust_val={} brute_val={} nv_val={}",
                fp8_e4m3_to_f32(rust_byte), fp8_e4m3_to_f32(brute_byte), fp8_e4m3_to_f32(nv));
            if rust_byte != brute_byte {
                fails.push((v, rust_byte, brute_byte, nv));
            }
        }
        if !fails.is_empty() {
            panic!("{} values: Rust encoder disagrees with brute-force nearest", fails.len());
        }
    }
}
