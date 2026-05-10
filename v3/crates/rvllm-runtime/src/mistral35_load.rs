//! Mistral 3.5 NVFP4 weight upload pass.
//!
//! Splits responsibilities across crates per the rvllm DAG:
//!
//! - `rvllm-loader` owns the struct definitions
//!   (`Mistral35LoadedModel`, `Mistral35LayerLoaded`,
//!   `Nvfp4LinearLoaded`) — see `mistral35_weights.rs`.
//! - This file (`rvllm-runtime`) owns the orchestration: it mmaps
//!   safetensors shards, allocates persistent arena regions for
//!   the device-resident weights, runs the per-projection
//!   `launch_nvfp4_sfb_transform` (CUTLASS — only reachable from
//!   this layer of the DAG), and assembles the
//!   `Mistral35LoadedModel`.
//!
//! Per-projection upload sequence (codex-reviewed strategy):
//!
//! 1. mmap the source shard (kernel pages-in lazily).
//! 2. Persistent device buffer for `weight_packed` (`U8 [N, K/2]`)
//!    via `arena.region(...) + copy_from_host`.
//! 3. Transient scratch region for natural-layout `weight_scale`
//!    (`E4M3 [N, K/16]`); upload, then run
//!    `launch_nvfp4_sfb_transform(natural -> CUTLASS-interleaved)`
//!    into a separate persistent region.
//! 4. Persistent device scalar for `weight_global_scale` (`F32 [1]`).
//! 5. Drop the natural scratch reference (arena bump-allocator
//!    keeps the high-water but the next projection reuses the same
//!    end of the arena via `arena.checkpoint() + restore` the
//!    bring-up does at outer-scope).
//!
//! The orchestration is intentionally synchronous + on-stream: no
//! pipelining of projection uploads today. 88 layers × 7 linears =
//! 616 transform launches at load-time; each is a few µs, total
//! upload dominated by the actual H→D byte copies.

use std::collections::BTreeMap;
use std::path::Path;

use rvllm_core::{DType, LoaderCtx, LoaderError, Result, RvllmError};
use rvllm_loader::mistral35_arch::Mistral35Arch;
use rvllm_loader::mistral35_weights::{
    Mistral35LayerLoaded, Mistral35LinearKind, Mistral35LoadedModel, Mistral35Outside,
    Nvfp4LinearLoaded, Nvfp4LinearShape,
};
use rvllm_loader::safetensors::{ShardHeader, ShardIndex, TensorEntry};
use rvllm_loader::weights::F16Weight;
use rvllm_mem::HbmArena;

use crate::mistral35_bring_up::Mistral35EnginePaths;

/// Mmap'd shard pool — keeps every safetensors file mapped in for
/// the duration of the upload pass. `BTreeMap<name, (shard_idx,
/// entry)>` lets us locate any tensor without a per-projection
/// re-scan.
struct ShardPool {
    mmaps: Vec<memmap2::Mmap>,
    paths: Vec<std::path::PathBuf>,
    tensors: BTreeMap<String, (usize, TensorEntry)>,
    model_dir: std::path::PathBuf,
}

impl ShardPool {
    fn open(model_dir: &Path) -> Result<Self> {
        let idx = ShardIndex::resolve(model_dir)?;
        let mut mmaps = Vec::with_capacity(idx.shards.len());
        let mut paths = Vec::with_capacity(idx.shards.len());
        let mut tensors: BTreeMap<String, (usize, TensorEntry)> = BTreeMap::new();
        for (shard_idx, shard_path) in idx.shards.iter().enumerate() {
            let f = std::fs::File::open(shard_path).map_err(|source| RvllmError::Io {
                err: rvllm_core::IoError::from(&source),
                path: shard_path.clone(),
                source,
            })?;
            let mmap = unsafe { memmap2::Mmap::map(&f) }
                .map_err(|source| RvllmError::Io {
                    err: rvllm_core::IoError::from(&source),
                    path: shard_path.clone(),
                    source,
                })?;
            let header = ShardHeader::parse(shard_path, &mmap)?;
            for (name, entry) in header.tensors.into_iter() {
                tensors.insert(name, (shard_idx, entry));
            }
            mmaps.push(mmap);
            paths.push(shard_path.clone());
        }
        Ok(Self { mmaps, paths, tensors, model_dir: model_dir.to_path_buf() })
    }

    fn must_get(&self, name: &str) -> Result<(usize, &TensorEntry)> {
        match self.tensors.get(name) {
            Some((si, e)) => Ok((*si, e)),
            None => Err(RvllmError::Loader {
                err: LoaderError::MissingTensor { name: name.to_string() },
                ctx: LoaderCtx {
                    path: self.model_dir.clone(),
                    tensor: Some(name.to_string()),
                },
                bt: std::backtrace::Backtrace::capture(),
            }),
        }
    }

    fn bytes_of(&self, si: usize, e: &TensorEntry) -> &[u8] {
        let mm = &self.mmaps[si];
        let start = e.file_offset as usize;
        &mm[start..start + e.nbytes as usize]
    }
}

/// Upload a BF16 tensor with full dtype + shape validation.
///
/// Round-9 #2 fix: the previous `upload_tensor_verbatim` accepted any
/// dtype and any shape, so a quantized or wrong-rank tensor named
/// `lm_head.weight` (or any of the BF16-class tensors — embeddings,
/// final norm, layernorms, vision, projector) could be uploaded
/// silently and then reinterpreted as BF16 by the kernel side,
/// triggering OOB reads in the GEMM. This helper enforces:
///  * dtype matches the caller's `expected_dtype` (BF16 in practice).
///  * shape matches `expected_shape` exactly.
///  * mmap byte length equals `product(shape) * bytes_per_elem`.
///
/// `expected_shape = None` keeps the loader flexible for tensors
/// whose first dim depends on the checkpoint (e.g. `[vocab, hidden]`
/// when the loader has not threaded `vocab` to this call site yet).
/// In that case only dtype + a `nbytes % bytes_per_elem == 0` check
/// runs — still strictly stronger than the verbatim fallback.
fn upload_typed_tensor(
    arena: &HbmArena<'_>,
    pool: &ShardPool,
    region_name: &'static str,
    tensor_name: &str,
    expected_dtype: DType,
    expected_shape: Option<&[usize]>,
) -> Result<F16Weight> {
    let (si, e) = pool.must_get(tensor_name)?;
    let bytes_per_elem: usize = match expected_dtype {
        DType::Bf16 | DType::F16 => 2,
        DType::F32 => 4,
        DType::Fp8E4M3 | DType::U8 => 1,
        other => return Err(RvllmError::Loader {
            err: LoaderError::Corrupt {
                detail: format!(
                    "upload_typed_tensor: dtype {:?} not supported for {tensor_name}", other,
                ),
            },
            ctx: LoaderCtx { path: pool.model_dir.clone(), tensor: Some(tensor_name.to_string()) },
            bt: std::backtrace::Backtrace::capture(),
        }),
    };
    if e.dtype != expected_dtype {
        return Err(RvllmError::Loader {
            err: LoaderError::Corrupt {
                detail: format!(
                    "Mistral 3.5 tensor {tensor_name}: dtype={:?} but \
                     loader expected {:?}",
                    e.dtype, expected_dtype,
                ),
            },
            ctx: LoaderCtx { path: pool.model_dir.clone(), tensor: Some(tensor_name.to_string()) },
            bt: std::backtrace::Backtrace::capture(),
        });
    }
    if let Some(want) = expected_shape {
        if e.shape != want {
            return Err(RvllmError::Loader {
                err: LoaderError::Corrupt {
                    detail: format!(
                        "Mistral 3.5 tensor {tensor_name}: shape={:?} but \
                         loader expected {:?}",
                        e.shape, want,
                    ),
                },
                ctx: LoaderCtx { path: pool.model_dir.clone(), tensor: Some(tensor_name.to_string()) },
                bt: std::backtrace::Backtrace::capture(),
            });
        }
    }
    let elem_count: usize = e.shape.iter().product();
    let expect_bytes = elem_count * bytes_per_elem;
    let raw = pool.bytes_of(si, e);
    if raw.len() != expect_bytes {
        return Err(RvllmError::Loader {
            err: LoaderError::Corrupt {
                detail: format!(
                    "Mistral 3.5 tensor {tensor_name}: mmap len={} but \
                     shape={:?} × {} = {} bytes",
                    raw.len(), e.shape, bytes_per_elem, expect_bytes,
                ),
            },
            ctx: LoaderCtx { path: pool.model_dir.clone(), tensor: Some(tensor_name.to_string()) },
            bt: std::backtrace::Backtrace::capture(),
        });
    }
    let region = arena.region(region_name, raw.len(), 16)?;
    unsafe { region.copy_from_host(raw)? };
    Ok(F16Weight {
        offset_bytes: region.device_ptr(),
        shape: e.shape.clone(),
    })
}

/// Round-12 phase 5b: read several BF16 tensors, validate dtype +
/// per-tensor shape, concat them along the OUTER axis (axis 0), and
/// upload as one device region. Reserved for a future fused-QKV /
/// fused-gate-up GEMM in the Pixtral vision tower.
///
/// All input tensors must have identical shape `[rows_per, cols]`.
/// Output device buffer has shape `[count * rows_per, cols]`.
///
/// Currently unused: cuBLAS's stride-batched output for the
/// per-block forward would need to land in [3*N, v_hidden] head-major
/// (= 3 vertically-stacked [N, v_hidden] slabs) to keep the
/// downstream RoPE / attention kernels stride-correct, but the
/// natural cuBLAS row-major output is [N, 3*v_hidden] interleaved.
/// Using this helper would require either (a) a transpose kernel
/// after the fused GEMM (defeats purpose), (b) a stride_a=0
/// broadcast batched-strided variant whose support on this driver
/// is unverified, or (c) RoPE/attention kernel rewrites with a
/// stride parameter. Tracked in MISTRAL35_VISION_TODO.md under
/// "Performance — batched Q/K/V GEMMs".
#[allow(dead_code)]
fn upload_bf16_concat_axis0(
    arena: &HbmArena<'_>,
    pool: &ShardPool,
    region_name: &'static str,
    tensor_names: &[String],
    rows_per: usize,
    cols: usize,
) -> Result<F16Weight> {
    let bytes_per_elem = 2;
    let single_bytes = rows_per * cols * bytes_per_elem;
    let total_bytes = single_bytes * tensor_names.len();
    let mut staging = vec![0u8; total_bytes];
    for (i, name) in tensor_names.iter().enumerate() {
        let (si, e) = pool.must_get(name)?;
        if e.dtype != DType::Bf16 {
            return Err(RvllmError::Loader {
                err: LoaderError::Corrupt {
                    detail: format!(
                        "concat tensor {name}: dtype={:?} but Bf16 required",
                        e.dtype),
                },
                ctx: LoaderCtx { path: pool.model_dir.clone(), tensor: Some(name.clone()) },
                bt: std::backtrace::Backtrace::capture(),
            });
        }
        if e.shape != vec![rows_per, cols] {
            return Err(RvllmError::Loader {
                err: LoaderError::Corrupt {
                    detail: format!(
                        "concat tensor {name}: shape={:?} but [{rows_per}, {cols}] required",
                        e.shape),
                },
                ctx: LoaderCtx { path: pool.model_dir.clone(), tensor: Some(name.clone()) },
                bt: std::backtrace::Backtrace::capture(),
            });
        }
        let raw = pool.bytes_of(si, e);
        if raw.len() != single_bytes {
            return Err(RvllmError::Loader {
                err: LoaderError::Corrupt {
                    detail: format!(
                        "concat tensor {name}: raw bytes {} != {}",
                        raw.len(), single_bytes),
                },
                ctx: LoaderCtx { path: pool.model_dir.clone(), tensor: Some(name.clone()) },
                bt: std::backtrace::Backtrace::capture(),
            });
        }
        staging[i * single_bytes..(i + 1) * single_bytes].copy_from_slice(raw);
    }
    let region = arena.region(region_name, total_bytes, 16)?;
    unsafe { region.copy_from_host(&staging)? };
    Ok(F16Weight {
        offset_bytes: region.device_ptr(),
        shape: vec![tensor_names.len() * rows_per, cols],
    })
}

/// Convenience wrapper: BF16 only.
fn upload_bf16_tensor(
    arena: &HbmArena<'_>,
    pool: &ShardPool,
    region_name: &'static str,
    tensor_name: &str,
    expected_shape: Option<&[usize]>,
) -> Result<F16Weight> {
    upload_typed_tensor(
        arena, pool, region_name, tensor_name, DType::Bf16, expected_shape,
    )
}

/// Round-12 phase 3a: upload a `[O, C, H, W]` BF16 conv weight as
/// `[O, H, W, C]` row-major.
///
/// Pixtral's `patch_conv.weight` ships in PyTorch's natural Conv2d
/// layout (`[out_c, in_c, kh, kw]` = `[1664, 3, 14, 14]`). The
/// per-image patches our preprocessor produces are HWC-flattened
/// (`[N, kh*kw*in_c]` with the inner row laid out
/// `[ip, jp, c]`). To make the patch_conv a plain GEMM
/// `[N, C*H*W] @ [O, C*H*W]^T` we permute the conv weight to
/// `[O, kh, kw, in_c]` once at load time so the inner orderings
/// match.
///
/// This avoids a per-request permute at forward time (cheaper) and
/// keeps the GPU forward purely a single bf16_gemm_f32 launch.
#[cfg(feature = "cuda")]
fn upload_bf16_conv_weight_chw_to_hwc(
    arena: &HbmArena<'_>,
    pool: &ShardPool,
    region_name: &'static str,
    tensor_name: &str,
    expected_shape: &[usize; 4],
) -> Result<F16Weight> {
    let (si, e) = pool.must_get(tensor_name)?;
    if e.dtype != DType::Bf16 {
        return Err(RvllmError::Loader {
            err: LoaderError::Corrupt {
                detail: format!(
                    "patch_conv {tensor_name}: dtype={:?} but expected Bf16",
                    e.dtype),
            },
            ctx: LoaderCtx { path: pool.model_dir.clone(), tensor: Some(tensor_name.to_string()) },
            bt: std::backtrace::Backtrace::capture(),
        });
    }
    if e.shape != expected_shape {
        return Err(RvllmError::Loader {
            err: LoaderError::Corrupt {
                detail: format!(
                    "patch_conv {tensor_name}: shape={:?} but expected {:?}",
                    e.shape, expected_shape),
            },
            ctx: LoaderCtx { path: pool.model_dir.clone(), tensor: Some(tensor_name.to_string()) },
            bt: std::backtrace::Backtrace::capture(),
        });
    }
    let [o, c, h, w] = *expected_shape;
    let total = o * c * h * w;
    let raw = pool.bytes_of(si, e);
    if raw.len() != total * 2 {
        return Err(RvllmError::Loader {
            err: LoaderError::Corrupt {
                detail: format!(
                    "patch_conv {tensor_name}: raw bytes {} ≠ {} * 2",
                    raw.len(), total),
            },
            ctx: LoaderCtx { path: pool.model_dir.clone(), tensor: Some(tensor_name.to_string()) },
            bt: std::backtrace::Backtrace::capture(),
        });
    }
    // Permute on host: src[o, c, h, w] -> dst[o, h, w, c].
    let mut permuted = vec![0u8; raw.len()];
    let chw = c * h * w;
    for oi in 0..o {
        let src_o = oi * chw * 2;
        let dst_o = oi * chw * 2;
        for hi in 0..h {
            for wi in 0..w {
                for ci in 0..c {
                    let src_off = src_o + (((ci * h) + hi) * w + wi) * 2;
                    let dst_off = dst_o + (((hi * w) + wi) * c + ci) * 2;
                    permuted[dst_off]     = raw[src_off];
                    permuted[dst_off + 1] = raw[src_off + 1];
                }
            }
        }
    }
    let region = arena.region(region_name, permuted.len(), 16)?;
    unsafe { region.copy_from_host(&permuted)? };
    // Report the *permuted* shape so the GEMM caller sees a flat
    // `[O, H*W*C]` weight: stash as `[O, H, W, C]`.
    Ok(F16Weight {
        offset_bytes: region.device_ptr(),
        shape: vec![o, h, w, c],
    })
}

/// Upload one Mistral 3.5 NVFP4 linear, run the SFB transform, and
/// produce the device-resident `Nvfp4LinearLoaded` triple.
///
/// Stream + cutlass backend semantics (codex review #2):
/// `tile_atom_to_shape_SFB` consumes only `(N, K, L)`, so the
/// transform runs once per linear and the result is persistent.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn upload_nvfp4_linear(
    arena: &HbmArena<'_>,
    pool: &ShardPool,
    backend: &rvllm_cutlass::lib_so::CutlassBackend,
    stream: u64,
    base: &str,
    shape: Nvfp4LinearShape,
) -> Result<Nvfp4LinearLoaded> {
    // (1) weight_packed
    let packed_name = format!("{base}.weight_packed");
    let (psi, pe) = pool.must_get(&packed_name)?;
    let packed_raw = pool.bytes_of(psi, pe);
    let packed_region = arena.region("mistral35_w_packed", packed_raw.len(), 16)?;
    unsafe { packed_region.copy_from_host(packed_raw)? };
    let packed_ptr = packed_region.device_ptr();
    let packed_bytes = packed_raw.len();

    // (2) weight_scale (natural [N, K/16] E4M3) -> transient scratch.
    //     The natural region's lifetime ends after the transform
    //     completes; we keep the arena `Region` alive via the local
    //     binding only until that point.
    let scale_name = format!("{base}.weight_scale");
    let (ssi, se) = pool.must_get(&scale_name)?;
    let scale_raw = pool.bytes_of(ssi, se);
    // Round-9 #1 fix: derive the natural [N, K/16] e4m3 size directly from
    // the Mistral shape. Previously we asked the CUTLASS backend for it,
    // which returns 0 when `CutlassBackend::Absent` (or when libcutlass is
    // built without NVFP4 symbols), making the *default* fused W4A16 GEMV
    // path falsely depend on CUTLASS. The fused kernel never reads any
    // CUTLASS-interleaved layout — it consumes the natural scale bytes
    // verbatim — so the validation must work without a live backend.
    if shape.k % 16 != 0 {
        return Err(RvllmError::Loader {
            err: LoaderError::Corrupt {
                detail: format!(
                    "Mistral 3.5 weight_scale {scale_name}: shape K={} is not a \
                     multiple of the NVFP4 group size (16)",
                    shape.k,
                ),
            },
            ctx: LoaderCtx {
                path: pool.model_dir.clone(),
                tensor: Some(scale_name.clone()),
            },
            bt: std::backtrace::Backtrace::capture(),
        });
    }
    let nat_bytes = (shape.n as usize) * (shape.k as usize / 16); // e4m3 = 1 byte
    if nat_bytes != scale_raw.len() {
        return Err(RvllmError::Loader {
            err: LoaderError::Corrupt {
                detail: format!(
                    "Mistral 3.5 weight_scale {scale_name}: mmap len={} \
                     but expected n*(k/16) = {}*{} = {}",
                    scale_raw.len(), shape.n, shape.k / 16, nat_bytes,
                ),
            },
            ctx: LoaderCtx {
                path: pool.model_dir.clone(),
                tensor: Some(scale_name.clone()),
            },
            bt: std::backtrace::Backtrace::capture(),
        });
    }
    let nat_region = arena.region("mistral35_w_sfb_natural", scale_raw.len(), 16)?;
    unsafe { nat_region.copy_from_host(scale_raw)? };
    let sfb_natural_ptr = nat_region.device_ptr();

    // (3) Persistent CUTLASS-interleaved SFB target — only built for
    // the legacy W4A4 NVFP4 tensor-core GEMM path. The default fused
    // W4A16 GEMV (`RVLLM_W4A16_GEMV` ≠ "0") consumes
    // `sfb_natural_ptr` directly and never reads `sfb_cutlass_ptr`, so
    // building it would waste ~10 MiB × 7 × 88 ≈ 6 GiB of arena and
    // 616 transform launches per bring-up.
    let want_legacy_cutlass = std::env::var("RVLLM_W4A16_GEMV")
        .map(|s| matches!(s.as_str(), "0" | "false" | "FALSE")).unwrap_or(false);
    let (sfb_cutlass_ptr, sfb_bytes) = if want_legacy_cutlass {
        let sfb_bytes = backend.nvfp4_sfb_bytes(shape.n as i32, shape.k as i32);
        if sfb_bytes == 0 {
            return Err(RvllmError::Loader {
                err: LoaderError::Corrupt {
                    detail: format!(
                        "cutlass nvfp4_sfb_bytes(n={}, k={}) = 0 — backend not \
                         active or shape rejected",
                        shape.n, shape.k,
                    ),
                },
                ctx: LoaderCtx {
                    path: pool.model_dir.clone(),
                    tensor: Some(scale_name.clone()),
                },
                bt: std::backtrace::Backtrace::capture(),
            });
        }
        let sfb_region = arena.region("mistral35_w_sfb_cutlass", sfb_bytes, 16)?;
        unsafe {
            backend.launch_nvfp4_sfb_transform(
                nat_region.device_ptr(),
                sfb_region.device_ptr(),
                shape.n as i32,
                shape.k as i32,
                stream,
            )?;
        }
        (sfb_region.device_ptr(), sfb_bytes)
    } else {
        (0u64, 0usize)
    };

    // (4) weight_global_scale  F32 [1]
    let gs_name = format!("{base}.weight_global_scale");
    let (gsi, ge) = pool.must_get(&gs_name)?;
    let gs_raw = pool.bytes_of(gsi, ge);
    if gs_raw.len() != 4 {
        return Err(RvllmError::Loader {
            err: LoaderError::Corrupt {
                detail: format!(
                    "Mistral 3.5 {gs_name}: expected 4 bytes (F32 [1]), got {}",
                    gs_raw.len(),
                ),
            },
            ctx: LoaderCtx {
                path: pool.model_dir.clone(),
                tensor: Some(gs_name.clone()),
            },
            bt: std::backtrace::Backtrace::capture(),
        });
    }
    // LLMCompressor / compressed-tensors NVFP4 stores weight_global_scale
    // as the encode scale `FP8_E4M3_MAX * FP4_E2M1_MAX / weight_amax`.
    // Its per-block E4M3 scales are generated as `(block_amax / 6) *
    // weight_global_scale`, while dequantization divides that local scale
    // by weight_global_scale.
    //
    // CUTLASS Sm120 blockscaled MMA multiplies by SFA and SFB as supplied,
    // and its epilogue `alpha_ptr` is a plain multiplicative output scalar.
    // Therefore the device scalar passed to CUTLASS must be the decode
    // scale, `1 / weight_global_scale`, not the checkpoint value.
    let gs_f32: f32 = f32::from_le_bytes([gs_raw[0], gs_raw[1], gs_raw[2], gs_raw[3]]);
    // F4#3 fix: `RVLLM_NVFP4_ALPHA_MULT` was an unguarded debug knob
    // that silently scaled every NVFP4 weight by `alpha_mult` if set,
    // which would corrupt all 616 projections at once. Now gated
    // behind `RVLLM_DEBUG_NVFP4=1` and loudly warned at startup so
    // an operator can't accidentally inherit it from a stale shell.
    let alpha_mult: f32 = match (
        std::env::var("RVLLM_NVFP4_ALPHA_MULT").ok().and_then(|s| s.parse::<f32>().ok()),
        std::env::var_os("RVLLM_DEBUG_NVFP4").is_some(),
    ) {
        (Some(m), true) if (m - 1.0).abs() > f32::EPSILON => {
            // Print only once per process by stamping a static flag
            // via the load path (this fn runs N=616 times per startup,
            // so guard the eprint).
            use std::sync::atomic::{AtomicBool, Ordering};
            static WARNED: AtomicBool = AtomicBool::new(false);
            if !WARNED.swap(true, Ordering::SeqCst) {
                eprintln!(
                    "[mistral35-load] WARNING: RVLLM_NVFP4_ALPHA_MULT={} \
                     × every weight scale (RVLLM_DEBUG_NVFP4 set). \
                     Use only for one-shot quantization-noise probes; \
                     unset for production.",
                    m,
                );
            }
            m
        }
        (Some(m), false) if (m - 1.0).abs() > f32::EPSILON => {
            return Err(RvllmError::Loader {
                err: LoaderError::Corrupt {
                    detail: format!(
                        "RVLLM_NVFP4_ALPHA_MULT={} silently rescales every \
                         NVFP4 weight; refusing to load. Set \
                         RVLLM_DEBUG_NVFP4=1 to opt in to the diagnostic, \
                         or unset RVLLM_NVFP4_ALPHA_MULT for production.",
                        m
                    ),
                },
                ctx: LoaderCtx { path: pool.model_dir.clone(), tensor: None },
                bt: std::backtrace::Backtrace::capture(),
            });
        }
        _ => 1.0,
    };
    let alpha_f32 = if gs_f32.is_finite() && gs_f32 != 0.0 {
        alpha_mult / gs_f32
    } else {
        gs_f32
    };
    let gs_region = arena.region("mistral35_w_global_scale", 4, 4)?;
    unsafe { gs_region.copy_from_host(&alpha_f32.to_le_bytes())? };
    let global_scale_ptr = gs_region.device_ptr();

    Ok(Nvfp4LinearLoaded {
        shape,
        packed_ptr,
        sfb_natural_ptr,
        sfb_cutlass_ptr,
        global_scale_ptr,
        packed_bytes,
        sfb_bytes,
        bf16_ptr: 0,
    })
}

/// Synchronously upload all Mistral 3.5 weights into `arena` and
/// run the SFB transform per projection.
///
/// Caller must hold the `cutlass_backend` for the lifetime of the
/// returned `Mistral35LoadedModel` (the SFB pointers are owned by
/// `arena`, but the launch backend handle is needed for forward).
#[cfg(feature = "cuda")]
pub fn load_mistral35_model(
    paths: &Mistral35EnginePaths,
    arch: &Mistral35Arch,
    arena: &HbmArena<'_>,
    cutlass_backend: &rvllm_cutlass::lib_so::CutlassBackend,
    stream: u64,
) -> Result<Mistral35LoadedModel> {
    // F2#1 fix: only the legacy W4A4 NVFP4 tensor-core path needs the
    // CUTLASS NVFP4 entry-point set. The default fused W4A16 GEMV
    // never calls those symbols, so a stale / missing
    // libcutlass_sm120.so should not block startup. Validate the
    // weight_scale shape downstream regardless (per-projection size
    // check is in upload_nvfp4_linear).
    let want_legacy_cutlass = std::env::var("RVLLM_W4A16_GEMV")
        .map(|s| matches!(s.as_str(), "0" | "false" | "FALSE")).unwrap_or(false);
    if want_legacy_cutlass {
        cutlass_backend.require_nvfp4()?;
    }
    let pool = ShardPool::open(&paths.model_dir)?;

    let prefix = arch.weight_prefix.as_str();
    eprintln!("[mistral35-load] uploading outside tensors (embed + norm + lm_head)…");
    let hidden = arch.text.hidden_size;
    let vocab = arch.text.vocab_size;
    let outside = Mistral35Outside {
        embed_tokens: upload_bf16_tensor(
            arena, &pool, "mistral35_embed",
            &format!("{prefix}.embed_tokens.weight"),
            Some(&[vocab, hidden]),
        )?,
        final_norm: upload_bf16_tensor(
            arena, &pool, "mistral35_final_norm",
            &format!("{prefix}.norm.weight"),
            Some(&[hidden]),
        )?,
        // Mistral 3.5: tie_word_embeddings = false → separate
        // lm_head tensor at top level.
        lm_head: upload_bf16_tensor(
            arena, &pool, "mistral35_lm_head", "lm_head.weight",
            Some(&[vocab, hidden]),
        )?,
    };
    eprintln!(
        "[mistral35-load]   embed: {:?}  final_norm: {:?}  lm_head: {:?}",
        outside.embed_tokens.shape, outside.final_norm.shape, outside.lm_head.shape,
    );

    eprintln!(
        "[mistral35-load] uploading {} decoder layers (norms + 7 NVFP4 linears each)…",
        arch.text.num_hidden_layers,
    );
    let mut layers = Vec::with_capacity(arch.text.num_hidden_layers);
    let layer_log_step = (arch.text.num_hidden_layers / 8).max(1);
    for li in 0..arch.text.num_hidden_layers {
        let layer = upload_one_layer(arena, &pool, cutlass_backend, stream, prefix, li, arch)?;
        if li % layer_log_step == 0 || li + 1 == arch.text.num_hidden_layers {
            eprintln!("[mistral35-load]   layer {li}/{} done", arch.text.num_hidden_layers);
        }
        layers.push(layer);
    }
    eprintln!("[mistral35-load] all language weights resident on device");

    // Round-10 #1 fix: default OFF for Mistral.
    //
    // The Pixtral splice / forward path is not wired into the Mistral
    // decoder yet; vision-bearing requests are rejected at admission
    // in v3/crates/rvllm-serve/src/openai/handlers.rs. Loading the
    // vision tower (~2 GiB BF16 + projector) is therefore pure
    // arena/VRAM waste and slows startup by tens of seconds. Operators
    // wiring up the splice can opt in with RVLLM_LOAD_VISION=1.
    //
    // Accepted values: "1" / "true" / "TRUE"  → load
    //                  unset / anything else  → skip
    let load_vision = std::env::var("RVLLM_LOAD_VISION")
        .map(|s| matches!(s.as_str(), "1" | "true" | "TRUE"))
        .unwrap_or(false);
    let vision = if load_vision {
        eprintln!(
            "[mistral35-load] RVLLM_LOAD_VISION=1: loading Pixtral tower \
             (HTTP admission still rejects vision requests until splice \
             is wired)"
        );
        Some(upload_mistral35_vision(arena, &pool, arch)?)
    } else {
        eprintln!(
            "[mistral35-load] vision tower skipped (default — set \
             RVLLM_LOAD_VISION=1 once Pixtral splice is wired)"
        );
        None
    };

    Ok(Mistral35LoadedModel { outside, layers, vision })
}

#[cfg(feature = "cuda")]
fn upload_mistral35_vision(
    arena: &HbmArena<'_>,
    pool: &ShardPool,
    arch: &Mistral35Arch,
) -> Result<rvllm_loader::mistral35_weights::Mistral35Vision> {
    use rvllm_loader::mistral35_weights::{Mistral35Vision, VisionLayerLoaded};

    eprintln!(
        "[mistral35-load] uploading vision tower (Pixtral, {} layers)…",
        arch.vision.num_hidden_layers,
    );

    // The checkpoint stores vision tensors under the *direct*
    // `model.vision_tower.*` prefix (not nested under language_model).
    // Mistral checkpoints sometimes use `model.vision_tower.` and
    // older variants used bare `vision_tower.`; the inventory
    // validator already accepts both. The shard pool's
    // BTreeMap doesn't care — names are exact.
    let vt = "model.vision_tower";
    let mmp = "model.multi_modal_projector";

    // Round-12 #3a: permute patch_conv from CHW inner to HWC inner so
    // it can multiply our HWC-flattened patches as a plain GEMM. The
    // expected on-disk shape is [v_hidden, num_channels, patch, patch]
    // = [1664, 3, 14, 14] for the canonical Mistral 3.5 / Pixtral.
    let v_hidden = arch.vision.hidden_size;
    let p = arch.vision.patch_size;
    let nc = arch.vision.num_channels;
    let patch_conv = upload_bf16_conv_weight_chw_to_hwc(
        arena, pool, "mistral35_v_patch_conv",
        &format!("{vt}.patch_conv.weight"),
        &[v_hidden, nc, p, p],
    )?;
    let ln_pre = upload_bf16_tensor(
        arena, pool, "mistral35_v_ln_pre",
        &format!("{vt}.ln_pre.weight"), None,
    )?;

    let mut vlayers = Vec::with_capacity(arch.vision.num_hidden_layers);
    let log_step = (arch.vision.num_hidden_layers / 6).max(1);
    for li in 0..arch.vision.num_hidden_layers {
        let lb = format!("{vt}.transformer.layers.{li}");
        let layer = VisionLayerLoaded {
            attention_norm: upload_bf16_tensor(arena, pool, "v_attn_norm",
                &format!("{lb}.attention_norm.weight"), None)?,
            q_proj: upload_bf16_tensor(arena, pool, "v_q",
                &format!("{lb}.attention.q_proj.weight"), None)?,
            k_proj: upload_bf16_tensor(arena, pool, "v_k",
                &format!("{lb}.attention.k_proj.weight"), None)?,
            v_proj: upload_bf16_tensor(arena, pool, "v_v",
                &format!("{lb}.attention.v_proj.weight"), None)?,
            o_proj: upload_bf16_tensor(arena, pool, "v_o",
                &format!("{lb}.attention.o_proj.weight"), None)?,
            ffn_norm: upload_bf16_tensor(arena, pool, "v_ffn_norm",
                &format!("{lb}.ffn_norm.weight"), None)?,
            gate_proj: upload_bf16_tensor(arena, pool, "v_gate",
                &format!("{lb}.feed_forward.gate_proj.weight"), None)?,
            up_proj: upload_bf16_tensor(arena, pool, "v_up",
                &format!("{lb}.feed_forward.up_proj.weight"), None)?,
            down_proj: upload_bf16_tensor(arena, pool, "v_down",
                &format!("{lb}.feed_forward.down_proj.weight"), None)?,
        };
        vlayers.push(layer);
        if li % log_step == 0 || li + 1 == arch.vision.num_hidden_layers {
            eprintln!("[mistral35-load]   vision layer {li}/{} done",
                arch.vision.num_hidden_layers);
        }
    }

    let projector_norm = upload_bf16_tensor(
        arena, pool, "v_proj_norm", &format!("{mmp}.norm.weight"), None)?;
    let projector_patch_merger = upload_bf16_tensor(
        arena, pool, "v_patch_merger",
        &format!("{mmp}.patch_merger.merging_layer.weight"), None)?;
    let projector_linear_1 = upload_bf16_tensor(
        arena, pool, "v_proj_l1", &format!("{mmp}.linear_1.weight"), None)?;
    let projector_linear_2 = upload_bf16_tensor(
        arena, pool, "v_proj_l2", &format!("{mmp}.linear_2.weight"), None)?;

    eprintln!(
        "[mistral35-load] vision tower uploaded: patch_conv={:?} \
         ln_pre={:?} layers={} projector(norm/merger/l1/l2)={:?}/{:?}/{:?}/{:?}",
        patch_conv.shape, ln_pre.shape, vlayers.len(),
        projector_norm.shape, projector_patch_merger.shape,
        projector_linear_1.shape, projector_linear_2.shape,
    );

    Ok(Mistral35Vision {
        patch_conv, ln_pre, layers: vlayers,
        projector_norm, projector_patch_merger,
        projector_linear_1, projector_linear_2,
    })
}

#[cfg(feature = "cuda")]
fn upload_one_layer(
    arena: &HbmArena<'_>,
    pool: &ShardPool,
    backend: &rvllm_cutlass::lib_so::CutlassBackend,
    stream: u64,
    prefix: &str,
    li: usize,
    arch: &Mistral35Arch,
) -> Result<Mistral35LayerLoaded> {
    let layer_base = format!("{prefix}.layers.{li}");
    // Norms — BF16 [hidden] with full dtype + shape validation.
    let hidden = arch.text.hidden_size;
    let input_layernorm = upload_bf16_tensor(
        arena, pool, "mistral35_in_norm",
        &format!("{layer_base}.input_layernorm.weight"),
        Some(&[hidden]),
    )?;
    let post_attention_layernorm = upload_bf16_tensor(
        arena, pool, "mistral35_post_norm",
        &format!("{layer_base}.post_attention_layernorm.weight"),
        Some(&[hidden]),
    )?;

    // Resolve the seven NVFP4 linears.
    let mut loaded: [Option<Nvfp4LinearLoaded>; 7] = Default::default();
    for (idx, kind) in Mistral35LinearKind::ALL.iter().enumerate() {
        let shape = kind.shape_for(arch);
        let base = format!("{layer_base}.{}", kind.name());
        loaded[idx] = Some(upload_nvfp4_linear(arena, pool, backend, stream, &base, shape)?);
    }
    // SAFETY: the loop above populates every slot; we unwrap each
    // exactly once.
    let q_proj = loaded[0].take().unwrap();
    let k_proj = loaded[1].take().unwrap();
    let v_proj = loaded[2].take().unwrap();
    let o_proj = loaded[3].take().unwrap();
    let gate_proj = loaded[4].take().unwrap();
    let up_proj = loaded[5].take().unwrap();
    let down_proj = loaded[6].take().unwrap();

    Ok(Mistral35LayerLoaded {
        input_layernorm,
        post_attention_layernorm,
        q_proj, k_proj, v_proj, o_proj,
        gate_proj, up_proj, down_proj,
    })
}
