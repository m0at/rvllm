//! Qwen 3.6 35B-A3B outside-layer weight loader (Phase 1).
//!
//! Loads the three "outside-the-stack" tensors (embedding, final RMSNorm,
//! lm_head) into the device arena as f16. Per-layer tensors (attention,
//! MoE experts, linear-attn state matrices) are Phase 2/3.
//!
//! Tensor naming: Qwen 3.6 uses the `model.language_model.*` prefix (same
//! shape as Gemma 4 vision-conditional). FP8 weights carry a
//! `weight_scale_inv` companion (NOT `weight_scale` like Gemma 4) — Phase 2
//! handles per-layer FP8 with that suffix.
//!
//! Phase 1 only touches bf16 tensors; the FP8 distinction does not yet
//! matter here.

use std::collections::BTreeMap;
use std::path::Path;

use half::f16;
use memmap2::Mmap;
use rvllm_core::{DType, LoaderCtx, LoaderError, Result, RvllmError};
use rvllm_mem::HbmArena;

use crate::safetensors::{ShardHeader, ShardIndex, TensorEntry};
use crate::weights::F16Weight;
use crate::qwen36_weights::Qwen36LoadedOutside;

const QWEN36_PREFIX: &str = "model.language_model";

/// Internal mmap wrapper. Mirrors the private `ShardMap` in
/// `gemma4_load.rs` — kept duplicate here to avoid widening the public
/// API surface for a Phase-1 loader.
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

/// Load + upload the three outside-layer weights for Qwen 3.6.
///
/// Returns the f16 device offsets + byte counts, ready for the
/// embedding-gather + RMSNorm + lm_head matmul kernels in Phase 2.
pub fn load_qwen36_outside(
    model_dir: &Path,
    arena: &HbmArena,
) -> Result<Qwen36LoadedOutside> {
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

    let must_get = |name: &str| -> Result<(usize, TensorEntry)> {
        tensors.get(name).cloned().ok_or_else(|| RvllmError::Loader {
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

    let upload_f16 = |region_name: &'static str, hf_name: &str| -> Result<(F16Weight, u64)> {
        let (si, e) = must_get(hf_name)?;
        let buf = tensor_to_f16_bytes(&e, bytes_of(si, &e), model_dir)?;
        let region = arena.region(region_name, buf.len(), 16)?;
        unsafe { region.copy_from_host(&buf)? };
        Ok((
            F16Weight {
                offset_bytes: region.device_ptr(),
                shape: e.shape.clone(),
            },
            buf.len() as u64,
        ))
    };

    let embed_name = format!("{QWEN36_PREFIX}.embed_tokens.weight");
    let norm_name = format!("{QWEN36_PREFIX}.norm.weight");
    let lm_head_name = "lm_head.weight";

    let (embed_tokens, embed_tokens_bytes) = upload_f16("qwen36_embedding", &embed_name)?;
    let (final_norm, final_norm_bytes) = upload_f16("qwen36_final_norm", &norm_name)?;
    let (lm_head, lm_head_bytes) = upload_f16("qwen36_lm_head", lm_head_name)?;

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

fn f32_bytes_to_f16_bytes(raw: &[u8]) -> Vec<u8> {
    let n = raw.len() / 4;
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n {
        let v = f32::from_le_bytes(raw[4 * i..4 * i + 4].try_into().unwrap());
        out.extend_from_slice(&f16::from_f32(v).to_le_bytes());
    }
    out
}
