//! rvllm-attention: FA3 SM90 paged decode + prefill.
//!
//! Two kernels only: `paged_decode` and `paged_prefill`. Both live in
//! `libfa3_kernels.so` which is built from the FlashAttention-3 Hopper
//! source at deploy time. No PTX fallback: engine refuses to start if
//! the `.so` is missing or not in the manifest.
//!
//! The invariants:
//! - `head_dim == 128` hard gate at construction (`AttentionError::UnsupportedHeadDim`)
//! - GQA ratio sanity (`num_heads` divisible by `num_kv_heads`)
//! - context_lens[i] == 0 valid padded-slot marker; kernel must predicate

pub mod decode;
pub mod prefill;

pub use decode::{PagedDecodeFp8Launcher, PagedDecodeLauncher, PagedDecodeParams};
pub use prefill::{PagedPrefillFp8Launcher, PagedPrefillLauncher, PagedPrefillParams};

use rvllm_core::{AttentionError, AttnCtx, Result, RvllmError};

/// Runtime-constructed wrapper around `libfa3_kernels.so`. The wrapper
/// refuses to exist if the .so is missing or its manifest-verified
/// exports don't include the entry points. Callers obtain launchers
/// from the wrapper.
/// Function pointer types for FA3 .so exports.
#[cfg(feature = "cuda")]
pub(crate) type WorkspaceSizeFn = unsafe extern "C" fn(
    batch_size: i32,
    num_heads: i32,
    max_num_splits: i32,
) -> i32;

#[cfg(feature = "cuda")]
#[allow(clippy::type_complexity)]
pub(crate) type PagedDecodeFn = unsafe extern "C" fn(
    q_ptr: *mut std::ffi::c_void,
    k_cache_ptr: *mut std::ffi::c_void,
    v_cache_ptr: *mut std::ffi::c_void,
    o_ptr: *mut std::ffi::c_void,
    block_tables_ptr: *mut std::ffi::c_void,
    context_lens_ptr: *mut std::ffi::c_void,
    workspace_ptr: *mut std::ffi::c_void,
    scale: f32,
    batch_size: i32,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    block_size: i32,
    max_blocks_per_seq: i32,
    num_blocks_total: i32,
    stream: *mut std::ffi::c_void,
) -> i32;

// FP8 E4M3 paged decode: Q / K cache / V cache are FP8 (1 byte/elem).
// q_descale / k_descale / v_descale point at f32 per-tensor scale scalars
// on the device. O is fp16.
#[cfg(feature = "cuda")]
#[allow(clippy::type_complexity)]
pub(crate) type PagedDecodeFp8Fn = unsafe extern "C" fn(
    q_fp8_ptr: *mut std::ffi::c_void,
    k_cache_fp8_ptr: *mut std::ffi::c_void,
    v_cache_fp8_ptr: *mut std::ffi::c_void,
    o_f16_ptr: *mut std::ffi::c_void,
    block_tables_ptr: *mut std::ffi::c_void,
    context_lens_ptr: *mut std::ffi::c_void,
    workspace_ptr: *mut std::ffi::c_void,
    q_descale_ptr: *mut f32,
    k_descale_ptr: *mut f32,
    v_descale_ptr: *mut f32,
    scale: f32,
    batch_size: i32,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    block_size: i32,
    max_blocks_per_seq: i32,
    num_blocks_total: i32,
    stream: *mut std::ffi::c_void,
) -> i32;

// FP8 E4M3 paged PREFILL: multi-query causal self-attention. Q layout is
// [total_q, num_heads, head_dim] indexed via cu_seqlens_q. K / V cache
// are paged FP8. Causal mask applied per-seq.
#[cfg(feature = "cuda")]
#[allow(clippy::type_complexity)]
pub(crate) type PagedPrefillFp8Fn = unsafe extern "C" fn(
    q_fp8_ptr: *mut std::ffi::c_void,
    k_cache_fp8_ptr: *mut std::ffi::c_void,
    v_cache_fp8_ptr: *mut std::ffi::c_void,
    o_f16_ptr: *mut std::ffi::c_void,
    block_tables_ptr: *mut std::ffi::c_void,
    context_lens_ptr: *mut std::ffi::c_void,
    cu_seqlens_q_ptr: *mut std::ffi::c_void,
    workspace_ptr: *mut std::ffi::c_void,
    q_descale_ptr: *mut f32,
    k_descale_ptr: *mut f32,
    v_descale_ptr: *mut f32,
    scale: f32,
    total_q: i32,
    max_seqlen_q: i32,
    batch_size: i32,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    block_size: i32,
    max_blocks_per_seq: i32,
    num_blocks_total: i32,
    stream: *mut std::ffi::c_void,
) -> i32;

#[derive(Debug)]
pub struct Fa3Kernels {
    pub so_path: std::path::PathBuf,
    #[cfg(feature = "cuda")]
    _lib: libloading::Library,
    #[cfg(feature = "cuda")]
    pub(crate) fn_workspace_size: WorkspaceSizeFn,
    #[cfg(feature = "cuda")]
    pub(crate) fn_paged_decode: PagedDecodeFn,
    #[cfg(feature = "cuda")]
    pub(crate) fn_paged_decode_fp8: PagedDecodeFp8Fn,
    /// Optional because older libfa3_kernels.so builds don't export it.
    /// Binaries using only decode can load against either .so; prefill
    /// callers must check is_some() and error gracefully otherwise.
    #[cfg(feature = "cuda")]
    pub(crate) fn_paged_prefill_fp8: Option<PagedPrefillFp8Fn>,
}

impl Fa3Kernels {
    /// Load the FA3 .so. Called once at engine init from a
    /// `KernelLoader`-produced path. Returns `Err` with explicit
    /// `AttentionError::Fa3SoMissing` if the path does not exist.
    pub fn load(path: std::path::PathBuf, head_dim: u32) -> Result<Self> {
        if !path.exists() {
            return Err(RvllmError::Attention {
                err: AttentionError::Fa3SoMissing { path: path.clone() },
                ctx: AttnCtx {
                    op: "Fa3Kernels::load",
                    stream: 0,
                    num_seqs: 0,
                    head_dim,
                },
                bt: std::backtrace::Backtrace::capture(),
            });
        }
        if head_dim != 128 {
            return Err(RvllmError::Attention {
                err: AttentionError::UnsupportedHeadDim {
                    got: head_dim,
                    required: 128,
                },
                ctx: AttnCtx {
                    op: "Fa3Kernels::load",
                    stream: 0,
                    num_seqs: 0,
                    head_dim,
                },
                bt: std::backtrace::Backtrace::capture(),
            });
        }

        #[cfg(feature = "cuda")]
        {
            unsafe {
                let _lib = libloading::Library::new(&path).map_err(|_e| RvllmError::Attention {
                    err: AttentionError::Fa3SoMissing { path: path.clone() },
                    ctx: AttnCtx {
                        op: "dlopen",
                        stream: 0,
                        num_seqs: 0,
                        head_dim,
                    },
                    bt: std::backtrace::Backtrace::capture(),
                })?;
                let ws_sym: libloading::Symbol<WorkspaceSizeFn> = _lib
                    .get(b"fa3_sm90_workspace_size\0")
                    .map_err(|_e| RvllmError::Attention {
                        err: AttentionError::Fa3SoMissing { path: path.clone() },
                        ctx: AttnCtx {
                            op: "dlsym:fa3_sm90_workspace_size",
                            stream: 0,
                            num_seqs: 0,
                            head_dim,
                        },
                        bt: std::backtrace::Backtrace::capture(),
                    })?;
                let dec_sym: libloading::Symbol<PagedDecodeFn> = _lib
                    .get(b"fa3_sm90_paged_decode\0")
                    .map_err(|_e| RvllmError::Attention {
                        err: AttentionError::Fa3SoMissing { path: path.clone() },
                        ctx: AttnCtx {
                            op: "dlsym:fa3_sm90_paged_decode",
                            stream: 0,
                            num_seqs: 0,
                            head_dim,
                        },
                        bt: std::backtrace::Backtrace::capture(),
                    })?;
                let dec_fp8_sym: libloading::Symbol<PagedDecodeFp8Fn> = _lib
                    .get(b"fa3_sm90_paged_decode_fp8\0")
                    .map_err(|_e| RvllmError::Attention {
                        err: AttentionError::Fa3SoMissing { path: path.clone() },
                        ctx: AttnCtx {
                            op: "dlsym:fa3_sm90_paged_decode_fp8",
                            stream: 0,
                            num_seqs: 0,
                            head_dim,
                        },
                        bt: std::backtrace::Backtrace::capture(),
                    })?;
                // Optional: old .so builds may not export this symbol.
                let fn_paged_prefill_fp8: Option<PagedPrefillFp8Fn> = _lib
                    .get::<PagedPrefillFp8Fn>(b"fa3_sm90_paged_prefill_fp8\0")
                    .ok()
                    .map(|s| *s);
                let fn_workspace_size = *ws_sym;
                let fn_paged_decode = *dec_sym;
                let fn_paged_decode_fp8 = *dec_fp8_sym;
                return Ok(Self {
                    so_path: path,
                    _lib,
                    fn_workspace_size,
                    fn_paged_decode,
                    fn_paged_decode_fp8,
                    fn_paged_prefill_fp8,
                });
            }
        }
        #[cfg(not(feature = "cuda"))]
        Ok(Self { so_path: path })
    }

    /// Minimum workspace size in bytes for the given batch + heads.
    pub fn workspace_size(&self, batch_size: i32, num_heads: i32) -> usize {
        #[cfg(feature = "cuda")]
        unsafe {
            let s = (self.fn_workspace_size)(batch_size, num_heads, 128);
            return s as usize;
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (batch_size, num_heads);
            0
        }
    }
}

// libloading::Library holds an unowned dlopen handle; safe to send per v2.
#[cfg(feature = "cuda")]
unsafe impl Send for Fa3Kernels {}
#[cfg(feature = "cuda")]
unsafe impl Sync for Fa3Kernels {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_so_rejected_at_load() {
        let err = Fa3Kernels::load("/nonexistent/libfa3_kernels.so".into(), 128).unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("Fa3SoMissing"));
    }

    #[test]
    fn non_128_head_dim_rejected() {
        // use a real-ish path so the missing-so check doesn't fire first
        let tmp = std::env::temp_dir().join("fa3-fake.so");
        std::fs::write(&tmp, b"fake").unwrap();
        let err = Fa3Kernels::load(tmp.clone(), 64).unwrap_err();
        std::fs::remove_file(&tmp).ok();
        let s = format!("{err}");
        assert!(s.contains("UnsupportedHeadDim"));
    }
}
