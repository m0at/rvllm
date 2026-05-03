//! Qwen 3.6 35B-A3B weight loader.
//!
//! Phase 1 (`load_qwen36_outside`): the three outside-the-stack tensors
//! (embedding, final RMSNorm, lm_head) — bf16 → f16 upload.
//!
//! Phase 2a (`load_qwen36_model`): outside + per-layer attention tensors
//! for the 10 full-attention layers. Per-layer linear-attention state
//! matrices and the 256-expert MoE block are NOT yet uploaded; they
//! land in Phase 2b/3.
//!
//! Tensor naming: Qwen 3.6 uses the `model.language_model.*` prefix.
//! FP8 weights carry a `weight_scale_inv` companion (BF16, shape
//! `[N/128, K/128]`, blockwise [128, 128] scales — same layout the
//! sm_120 CUTLASS blockwise FP8 GEMM consumes for Gemma 4).

use std::collections::BTreeMap;
use std::path::Path;

use half::f16;
use memmap2::Mmap;
use rvllm_core::{DType, LoaderCtx, LoaderError, Result, RvllmError};
use rvllm_mem::HbmArena;

use crate::load::LayerAttnType;
use crate::qwen36_weights::{Qwen36FullAttnLayer, Qwen36LoadedModel, Qwen36LoadedOutside};
use crate::safetensors::{ShardHeader, ShardIndex, TensorEntry};
use crate::weights::{F16Weight, Fp8Weight};

const QWEN36_PREFIX: &str = "model.language_model";

/// Internal mmap wrapper. Mirrors the private `ShardMap` in
/// `gemma4_load.rs` — kept duplicate here to avoid widening the public
/// API surface for a Phase-1/2 loader.
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

/// Per-load context: opened shards + tensor index. Shared between the
/// outside-tensor pass and the per-layer pass so we mmap each shard
/// exactly once.
struct LoadCtx<'a> {
    shards: Vec<ShardMap>,
    tensors: BTreeMap<String, (usize, TensorEntry)>,
    model_dir: &'a Path,
    arena: &'a HbmArena<'a>,
}

impl<'a> LoadCtx<'a> {
    fn new(model_dir: &'a Path, arena: &'a HbmArena<'a>) -> Result<Self> {
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
        Ok(Self {
            shards,
            tensors,
            model_dir,
            arena,
        })
    }

    fn bytes_of(&self, si: usize, e: &TensorEntry) -> &[u8] {
        let s = self.shards[si].bytes();
        let start = e.file_offset as usize;
        &s[start..start + e.nbytes as usize]
    }

    fn must_get(&self, name: &str) -> Result<(usize, TensorEntry)> {
        self.tensors
            .get(name)
            .cloned()
            .ok_or_else(|| RvllmError::Loader {
                err: LoaderError::MissingTensor {
                    name: name.to_string(),
                },
                ctx: LoaderCtx {
                    path: self.model_dir.to_path_buf(),
                    tensor: Some(name.to_string()),
                },
                bt: std::backtrace::Backtrace::capture(),
            })
    }

    fn upload_f16(&self, region_name: &'static str, hf_name: &str) -> Result<(F16Weight, u64)> {
        let (si, e) = self.must_get(hf_name)?;
        let buf = tensor_to_f16_bytes(&e, self.bytes_of(si, &e), self.model_dir)?;
        let region = self.arena.region(region_name, buf.len(), 16)?;
        unsafe { region.copy_from_host(&buf)? };
        Ok((
            F16Weight {
                offset_bytes: region.device_ptr(),
                shape: e.shape.clone(),
            },
            buf.len() as u64,
        ))
    }

    /// Upload an FP8 weight + its blockwise `weight_scale_inv` companion.
    /// Qwen 3.6 ships FP8 e4m3 weights with bf16 [N/128, K/128] scales.
    /// We:
    ///   1. Upload the FP8 bytes verbatim.
    ///   2. Convert the bf16 scale tensor to f32 and upload as the
    ///      blockscale arena region (the sm_120 CUTLASS path expects
    ///      f32 scales here, mirroring the Gemma 4 fp8 blockwise pipe).
    ///   3. Set `scale = 1.0` (per-tensor scale unused on the blockwise
    ///      path) + `channelscale_ptr = None` (Qwen has no per-row
    ///      channelscale).
    fn upload_fp8_blockwise(
        &self,
        region_name: &'static str,
        weight_name: &str,
    ) -> Result<Fp8Weight> {
        let (wsi, we) = self.must_get(weight_name)?;
        if we.dtype != DType::Fp8E4M3 {
            return Err(RvllmError::Loader {
                err: LoaderError::DtypeMismatch {
                    tensor: we.name.clone(),
                    expected: DType::Fp8E4M3,
                    got: we.dtype,
                },
                ctx: LoaderCtx {
                    path: self.model_dir.to_path_buf(),
                    tensor: Some(we.name.clone()),
                },
                bt: std::backtrace::Backtrace::capture(),
            });
        }
        let raw = self.bytes_of(wsi, &we);
        let region = self.arena.region(region_name, raw.len(), 16)?;
        unsafe { region.copy_from_host(raw)? };

        let scale_name = format!("{weight_name}_scale_inv");
        let (ssi, se) = self.must_get(&scale_name)?;
        // Qwen ships bf16 scales — convert to f32 for the kernel.
        let scale_bytes = match se.dtype {
            DType::Bf16 => bf16_bytes_to_f32_bytes(self.bytes_of(ssi, &se)),
            DType::F32 => self.bytes_of(ssi, &se).to_vec(),
            _ => {
                return Err(RvllmError::Loader {
                    err: LoaderError::DtypeMismatch {
                        tensor: se.name.clone(),
                        expected: DType::F32,
                        got: se.dtype,
                    },
                    ctx: LoaderCtx {
                        path: self.model_dir.to_path_buf(),
                        tensor: Some(se.name.clone()),
                    },
                    bt: std::backtrace::Backtrace::capture(),
                });
            }
        };
        let bs_region = self.arena.region("qwen36_fp8_blockscale", scale_bytes.len(), 16)?;
        unsafe { bs_region.copy_from_host(&scale_bytes)? };

        // Per-tensor scalar `1.0` placeholder (the blockwise path drives
        // dequant via the 2-D `blockscale_ptr`; per-tensor `scale` is
        // unused but the Fp8Weight contract requires a non-null pointer).
        let one = 1.0f32;
        let one_r = self.arena.region("qwen36_fp8_scale", 4, 4)?;
        unsafe { one_r.copy_from_host(&one.to_le_bytes())? };

        let weight_n = we.shape[0];
        let weight_k = if we.shape.len() >= 2 { we.shape[1] } else { 0 };
        let n_blocks = (weight_n + 127) / 128;
        let k_blocks = (weight_k + 127) / 128;

        Ok(Fp8Weight {
            offset_bytes: region.device_ptr(),
            scale_ptr: one_r.device_ptr(),
            shape: we.shape.clone(),
            scale: 1.0,
            clamp_ppm: 0.0,
            dtype: DType::Fp8E4M3,
            channelscale_ptr: None,
            blockscale_ptr: Some(bs_region.device_ptr()),
            blockscale_n_blocks: n_blocks as u32,
            blockscale_k_blocks: k_blocks as u32,
        })
    }
}

/// Phase 1 entry point: outside tensors only.
pub fn load_qwen36_outside(
    model_dir: &Path,
    arena: &HbmArena,
) -> Result<Qwen36LoadedOutside> {
    let ctx = LoadCtx::new(model_dir, arena)?;
    load_outside_via_ctx(&ctx)
}

/// Phase 2a entry point: outside tensors + per-layer attention weights
/// for the 10 full-attention layers. Linear-attention layers + MoE
/// blocks are still TODO; their corresponding `Option` slots are `None`.
pub fn load_qwen36_model(
    model_dir: &Path,
    arena: &HbmArena,
    layer_types: &[LayerAttnType],
) -> Result<Qwen36LoadedModel> {
    let ctx = LoadCtx::new(model_dir, arena)?;
    let outside = load_outside_via_ctx(&ctx)?;

    let mut full_attn_layers: Vec<Option<Qwen36FullAttnLayer>> = Vec::with_capacity(layer_types.len());
    let mut full_loaded = 0usize;
    let mut full_skipped = 0usize;

    for (l, ty) in layer_types.iter().enumerate() {
        match ty {
            LayerAttnType::Full => {
                let layer = load_full_attn_layer(&ctx, l)?;
                full_attn_layers.push(Some(layer));
                full_loaded += 1;
            }
            LayerAttnType::Linear | LayerAttnType::SlidingAttention => {
                full_attn_layers.push(None);
                full_skipped += 1;
            }
        }
    }

    eprintln!(
        "[qwen36-loader] per-layer attention upload complete: \
         {full_loaded} full-attn layers loaded, {full_skipped} non-full \
         layers skipped (linear-attn + MoE land in Phase 2b/3)."
    );

    Ok(Qwen36LoadedModel {
        outside,
        full_attn_layers,
    })
}

fn load_outside_via_ctx(ctx: &LoadCtx) -> Result<Qwen36LoadedOutside> {
    let embed_name = format!("{QWEN36_PREFIX}.embed_tokens.weight");
    let norm_name = format!("{QWEN36_PREFIX}.norm.weight");
    let lm_head_name = "lm_head.weight";

    let (embed_tokens, embed_tokens_bytes) = ctx.upload_f16("qwen36_embedding", &embed_name)?;
    let (final_norm, final_norm_bytes) = ctx.upload_f16("qwen36_final_norm", &norm_name)?;
    let (lm_head, lm_head_bytes) = ctx.upload_f16("qwen36_lm_head", lm_head_name)?;

    eprintln!(
        "[qwen36-loader] outside tensors uploaded: \
         embed_tokens {:?} ({:.1} MiB), \
         final_norm {:?} ({:.2} KiB), \
         lm_head {:?} ({:.1} MiB)",
        embed_tokens.shape,
        embed_tokens_bytes as f64 / (1024.0 * 1024.0),
        final_norm.shape,
        final_norm_bytes as f64 / 1024.0,
        lm_head.shape,
        lm_head_bytes as f64 / (1024.0 * 1024.0),
    );

    Ok(Qwen36LoadedOutside {
        embed_tokens,
        final_norm,
        lm_head,
        embed_tokens_bytes,
        final_norm_bytes,
        lm_head_bytes,
    })
}

fn load_full_attn_layer(ctx: &LoadCtx, layer_idx: usize) -> Result<Qwen36FullAttnLayer> {
    let ln = |s: &str| format!("{QWEN36_PREFIX}.layers.{layer_idx}.{s}");

    let (input_layernorm, _) = ctx.upload_f16("qwen36_input_ln", &ln("input_layernorm.weight"))?;
    let (post_attention_layernorm, _) =
        ctx.upload_f16("qwen36_post_attn_ln", &ln("post_attention_layernorm.weight"))?;
    let (q_norm, _) = ctx.upload_f16("qwen36_q_norm", &ln("self_attn.q_norm.weight"))?;
    let (k_norm, _) = ctx.upload_f16("qwen36_k_norm", &ln("self_attn.k_norm.weight"))?;

    let q_proj = ctx.upload_fp8_blockwise("qwen36_q_proj", &ln("self_attn.q_proj.weight"))?;
    let k_proj = ctx.upload_fp8_blockwise("qwen36_k_proj", &ln("self_attn.k_proj.weight"))?;
    let v_proj = ctx.upload_fp8_blockwise("qwen36_v_proj", &ln("self_attn.v_proj.weight"))?;
    let o_proj = ctx.upload_fp8_blockwise("qwen36_o_proj", &ln("self_attn.o_proj.weight"))?;

    if layer_idx <= 7 {
        eprintln!(
            "[qwen36-loader] layer {layer_idx} full-attn: q={:?} k={:?} v={:?} o={:?}",
            q_proj.shape, k_proj.shape, v_proj.shape, o_proj.shape,
        );
    }

    Ok(Qwen36FullAttnLayer {
        input_layernorm,
        post_attention_layernorm,
        q_norm,
        k_norm,
        q_proj,
        k_proj,
        v_proj,
        o_proj,
    })
}

fn tensor_to_f16_bytes(e: &TensorEntry, raw: &[u8], model_dir: &Path) -> Result<Vec<u8>> {
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

fn bf16_bytes_to_f32_bytes(raw: &[u8]) -> Vec<u8> {
    let n = raw.len() / 2;
    let mut out = Vec::with_capacity(n * 4);
    for i in 0..n {
        let lo = raw[2 * i];
        let hi = raw[2 * i + 1];
        let as_f32 = f32::from_bits(u32::from_le_bytes([0, 0, lo, hi]));
        out.extend_from_slice(&as_f32.to_le_bytes());
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
