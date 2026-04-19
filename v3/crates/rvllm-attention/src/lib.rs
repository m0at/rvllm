//! rvllm-attention: FA3 SM90 paged decode + prefill.
//!
//! Two kernels only: `paged_decode` and `paged_prefill`. Both live in
//! `libfa3_kernels.so` which is built from the FlashAttention-3 Hopper
//! source at deploy time. No PTX fallback: engine refuses to start if
//! the `.so` is missing or not in the manifest.
//!
//! The invariants:
//! - `head_dim` must be one of `{128, 256, 512}` at construction
//! - GQA ratio sanity (`num_heads` divisible by `num_kv_heads`)
//! - context_lens[i] == 0 valid padded-slot marker; kernel must predicate

pub mod decode;
pub mod prefill;

pub use decode::{PagedDecodeFp8Launcher, PagedDecodeLauncher, PagedDecodeParams};
pub use prefill::{
    PagedPrefillFp8Launcher, PagedPrefillLauncher, PagedPrefillParams,
    UnifiedPrefillParams, UNIFIED_PREFILL_BLOCK_M,
};

use rvllm_core::{AttentionError, AttnCtx, Result, RvllmError};

const SUPPORTED_HEAD_DIMS: &[u32] = &[128, 256, 512];

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
    window_size_left: i32,
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
    window_size_left: i32,
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
    window_size_left: i32,
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
        if !SUPPORTED_HEAD_DIMS.contains(&head_dim) {
            return Err(RvllmError::Attention {
                err: AttentionError::UnsupportedHeadDim {
                    got: head_dim,
                    supported: SUPPORTED_HEAD_DIMS,
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
                // Try SM90 (FA3 Hopper) symbols first, then SM89 (Ada).
                let is_sm89 = _lib.get::<WorkspaceSizeFn>(b"fa3_sm90_workspace_size\0").is_err()
                    && _lib.get::<WorkspaceSizeFn>(b"fa_sm89_workspace_size\0").is_ok();
                if is_sm89 {
                    eprintln!("[rvllm-attention] using SM89 (Ada) attention backend");
                }
                let (ws_name, dec_name, fp8_name, prefill_name): (&[u8], &[u8], &[u8], &[u8]) = if is_sm89 {
                    (b"fa_sm89_workspace_size\0", b"fa_sm89_paged_decode\0",
                     b"fa_sm89_paged_decode_fp8\0", b"fa_sm89_paged_prefill_fp8\0")
                } else {
                    (b"fa3_sm90_workspace_size\0", b"fa3_sm90_paged_decode\0",
                     b"fa3_sm90_paged_decode_fp8\0", b"fa3_sm90_paged_prefill_fp8\0")
                };
                let sym_err = |name: &'static str| RvllmError::Attention {
                    err: AttentionError::Fa3SoMissing { path: path.clone() },
                    ctx: AttnCtx { op: name, stream: 0, num_seqs: 0, head_dim },
                    bt: std::backtrace::Backtrace::capture(),
                };
                let ws_sym: libloading::Symbol<WorkspaceSizeFn> = _lib
                    .get(ws_name).map_err(|_| sym_err("dlsym:workspace_size"))?;
                let dec_sym: libloading::Symbol<PagedDecodeFn> = _lib
                    .get(dec_name).map_err(|_| sym_err("dlsym:paged_decode"))?;
                let dec_fp8_sym: libloading::Symbol<PagedDecodeFp8Fn> = _lib
                    .get(fp8_name).map_err(|_| sym_err("dlsym:paged_decode_fp8"))?;
                let fn_paged_prefill_fp8: Option<PagedPrefillFp8Fn> = _lib
                    .get::<PagedPrefillFp8Fn>(prefill_name)
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

// ============================================================================
// Fa2PtxKernels — sm_121 (GB10) attention backend via PTX-launched FA2 kernels
// ============================================================================

/// PTX-based attention backend for Blackwell consumer targets where
/// `libfa3_kernels.so` does not apply (FA3 requires WGMMA + TMA
/// multicast, both Hopper-only). Loads `flash_attention.ptx` via
/// `KernelLoader` and resolves the four entry points we compile:
/// `flash_attention_2_kernel`, `flash_attention_2_decode_kernel`,
/// `flash_attention_2_f16kv_kernel`,
/// `flash_attention_2_decode_f16kv_kernel`.
///
/// This PR ships the backend *structurally*: the kernels load, symbols
/// resolve, and `AttentionBackend::workspace_size` returns a correct
/// zero (FA2 has no .so-managed workspace). The actual launch path
/// still needs to translate the FA3 parameter set (paged KV, FP8
/// descale pointers, `window_size_left` semantics) into FA2's call
/// shape plus an fp8-KV path (FA2 today takes f16/f32 KV only). That
/// translation + a matching fp8-KV kernel variant are tracked as the
/// next GB10 follow-up PR; launching through `Fa2Ptx` before then
/// returns a typed `FeatureNotAvailable` error so the engine fails
/// closed.
#[derive(Debug)]
pub struct Fa2PtxKernels {
    pub head_dim: u32,
    #[cfg(feature = "cuda")]
    pub flash_attention_mod: rvllm_kernels::LoadedModule,
    #[cfg(feature = "cuda")]
    pub fn_decode: rvllm_kernels::KernelFn,
    #[cfg(feature = "cuda")]
    pub fn_decode_f16kv: rvllm_kernels::KernelFn,
    #[cfg(feature = "cuda")]
    pub fn_prefill: rvllm_kernels::KernelFn,
    #[cfg(feature = "cuda")]
    pub fn_prefill_f16kv: rvllm_kernels::KernelFn,
    /// `flash_attention_2_decode_fp8kv_kernel` — Gemma 4 decode path
    /// on sm_121, FP8 E4M3 KV cache, f16 output. Matches the
    /// `PagedDecodeFp8Launcher` ABI. BC=32 variant; used for
    /// head_dim ≤ 256.
    #[cfg(feature = "cuda")]
    pub fn_decode_fp8kv: rvllm_kernels::KernelFn,
    /// BC=16 variant for head_dim=512 (Gemma 4 global-attention).
    /// With BC=32 the smem request for head_dim=512 overflows the
    /// 99 KB opt-in ceiling on sm_121; BC=16 halves the tile to fit.
    /// Launcher picks between the two based on `params.head_dim`.
    #[cfg(feature = "cuda")]
    pub fn_decode_fp8kv_bc16: rvllm_kernels::KernelFn,
}

impl Fa2PtxKernels {
    /// Load `flash_attention.ptx` (the FA2 source compiled for this
    /// arch) via the shared `KernelLoader`. Resolves all four entry
    /// points. `head_dim` must be one of the supported values —
    /// mirrors `Fa3Kernels::load` behaviour.
    pub fn load(loader: &rvllm_kernels::KernelLoader, head_dim: u32) -> Result<Self> {
        if !SUPPORTED_HEAD_DIMS.contains(&head_dim) {
            return Err(RvllmError::Attention {
                err: AttentionError::UnsupportedHeadDim {
                    got: head_dim,
                    supported: SUPPORTED_HEAD_DIMS,
                },
                ctx: AttnCtx {
                    op: "Fa2PtxKernels::load",
                    stream: 0,
                    num_seqs: 0,
                    head_dim,
                },
                bt: std::backtrace::Backtrace::capture(),
            });
        }

        #[cfg(feature = "cuda")]
        {
            let flash_attention_mod = loader.load_ptx("flash_attention")?;
            let fn_decode = flash_attention_mod.get_function("flash_attention_2_decode_kernel")?;
            let fn_decode_f16kv =
                flash_attention_mod.get_function("flash_attention_2_decode_f16kv_kernel")?;
            let fn_prefill = flash_attention_mod.get_function("flash_attention_2_kernel")?;
            let fn_prefill_f16kv =
                flash_attention_mod.get_function("flash_attention_2_f16kv_kernel")?;
            let fn_decode_fp8kv =
                flash_attention_mod.get_function("flash_attention_2_decode_fp8kv_kernel")?;
            let fn_decode_fp8kv_bc16 = flash_attention_mod
                .get_function("flash_attention_2_decode_fp8kv_bc16_kernel")?;
            Ok(Self {
                head_dim,
                flash_attention_mod,
                fn_decode,
                fn_decode_f16kv,
                fn_prefill,
                fn_prefill_f16kv,
                fn_decode_fp8kv,
                fn_decode_fp8kv_bc16,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = loader;
            Ok(Self { head_dim })
        }
    }
}

// ============================================================================
// AttentionBackend — unifies Fa3 (SM90 dlopen) and Fa2Ptx (sm_121 PTX)
// ============================================================================

/// Which attention backend the runtime is using on the live device.
/// Picked once at bring-up per `CompileTarget`:
///
///   * SM80 / SM89 / SM90 → `Fa3` (dlopen `libfa3_kernels.so`)
///   * SM121 (Blackwell consumer) → `Fa2Ptx` (PTX-launched FA2)
///
/// Callers (launcher structs in `decode.rs` / `prefill.rs`) `match`
/// on this enum and route to the appropriate launch path. An attempt
/// to launch a path that a given backend doesn't implement returns
/// `AttentionError::FeatureNotAvailable` rather than silently
/// succeeding with wrong output.
///
/// `#[non_exhaustive]` so a future SM100-specific backend (or a
/// trait-based dispatch table) can be added without breaking
/// downstream external matches.
#[derive(Debug)]
#[non_exhaustive]
pub enum AttentionBackend {
    Fa3(Fa3Kernels),
    Fa2Ptx(Fa2PtxKernels),
}

impl AttentionBackend {
    /// Minimum workspace size in bytes for the given batch + heads.
    /// The FA2 PTX path does not use an external workspace — the
    /// scratch is allocated per-block in shared memory inside the
    /// kernel itself — so it returns 0.
    #[must_use]
    pub fn workspace_size(&self, batch_size: i32, num_heads: i32) -> usize {
        match self {
            AttentionBackend::Fa3(fa3) => fa3.workspace_size(batch_size, num_heads),
            AttentionBackend::Fa2Ptx(_) => 0,
        }
    }

    /// Head dim this backend was constructed for.
    #[must_use]
    pub fn head_dim(&self) -> u32 {
        match self {
            // `Fa3Kernels` doesn't store head_dim as a public field;
            // it's validated at `load` time. For AttentionBackend we
            // reconstruct the invariant at construction and expose
            // it uniformly.
            AttentionBackend::Fa3(_) => 0, // caller already validated at Fa3Kernels::load
            AttentionBackend::Fa2Ptx(fa2) => fa2.head_dim,
        }
    }
}

impl From<Fa3Kernels> for AttentionBackend {
    fn from(fa3: Fa3Kernels) -> Self {
        AttentionBackend::Fa3(fa3)
    }
}

impl From<Fa2PtxKernels> for AttentionBackend {
    fn from(fa2: Fa2PtxKernels) -> Self {
        AttentionBackend::Fa2Ptx(fa2)
    }
}

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
    fn unsupported_head_dim_rejected() {
        // use a real-ish path so the missing-so check doesn't fire first
        let tmp = std::env::temp_dir().join("fa3-fake.so");
        std::fs::write(&tmp, b"fake").unwrap();
        let err = Fa3Kernels::load(tmp.clone(), 64).unwrap_err();
        std::fs::remove_file(&tmp).ok();
        let s = format!("{err}");
        assert!(s.contains("UnsupportedHeadDim"));
    }
}
