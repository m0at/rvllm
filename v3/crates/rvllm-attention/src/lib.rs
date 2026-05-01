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

pub use decode::{
    PagedDecodeFp8Launcher, PagedDecodeLauncher, PagedDecodeNvfp4Launcher,
    PagedDecodeParams,
};
pub use prefill::{
    PagedPrefillFp8Launcher, PagedPrefillLauncher, PagedPrefillNvfp4Launcher,
    PagedPrefillParams, UnifiedPrefillParams, UNIFIED_PREFILL_BLOCK_M,
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
    /// `flash_attention_2_decode_f16io_kernel` — f16 I/O decode against
    /// an f16 paged KV cache. Matches the `PagedDecodeLauncher` ABI so
    /// sm_121 can serve `RVLLM_F16_KV=1` without f32<->f16 scratch.
    /// Head-dim is capped at 256 by the smem budget (BC=32 with
    /// head_dim=512 overflows the 99 KB opt-in ceiling); head_dim > 256
    /// returns `FeatureNotAvailable` — that arm is Gemma 4 global
    /// attention which must use FP8 KV today.
    #[cfg(feature = "cuda")]
    pub fn_decode_f16io: rvllm_kernels::KernelFn,
    #[cfg(feature = "cuda")]
    pub fn_prefill: rvllm_kernels::KernelFn,
    #[cfg(feature = "cuda")]
    pub fn_prefill_f16kv: rvllm_kernels::KernelFn,
    /// `flash_attention_2_decode_fp8kv_kernel` — Gemma 4 decode path
    /// on sm_121, FP8 E4M3 KV cache, f16 output. Matches the
    /// `PagedDecodeFp8Launcher` ABI. BC=16 (see flash_attention.cu);
    /// head_dim=512 could not fit BC=32 on sm_121`s 99 KB opt-in
    /// smem cap, and head_dim=256 measurably benefits from BC=16
    /// occupancy (2+ blocks/SM vs 1 at BC=32), so sm_121 converged
    /// on BC=16 for all head_dims.
    #[cfg(feature = "cuda")]
    pub fn_decode_fp8kv: rvllm_kernels::KernelFn,
    /// GQA-grouped FP8 decode — CTA per (seq, kv_head). Optional;
    /// absent on older PTX trees. Dispatched when
    /// `num_heads > num_kv_heads` and GQA ≤ MAX_GQA_DECODE (=4).
    #[cfg(feature = "cuda")]
    pub fn_decode_fp8kv_gqa: Option<rvllm_kernels::KernelFn>,
    /// `flash_attention_2_prefill_fp8kv_unified_kernel` — multi-query
    /// FP8-KV prefill. Port of vLLM's
    /// `kernel_unified_attention_2d`; see
    /// `v3/UNIFIED_PREFILL_SPEC.md`. Replaces the per-token decode
    /// loop currently used by
    /// `gemma4_layer_exec::Gemma4Phase::Prefill`, which is the
    /// dominant TTFT cost on sm_121 (≈30× slower than vLLM on a
    /// 1836-token prompt).
    ///
    /// Loaded from a separate PTX module
    /// (`flash_attention_unified_prefill.ptx`) so we can iterate on
    /// the kernel without reflashing the full
    /// `flash_attention.ptx`. Optional until the kernel body lands
    /// — `None` routes callers to the per-qi decode fallback.
    #[cfg(feature = "cuda")]
    pub fn_prefill_fp8kv_unified: Option<rvllm_kernels::KernelFn>,
    /// The owning PTX module for `fn_prefill_fp8kv_unified`. Kept
    /// alive alongside the function handle so `cuModuleUnload` only
    /// fires at drop.
    #[cfg(feature = "cuda")]
    pub unified_prefill_mod: Option<rvllm_kernels::LoadedModule>,

    // ---- NVFP4 KV cache path (rusty_sm121_nvfp4 branch) ----
    //
    // The NVFP4 kernels live in a separate PTX module
    // (`flash_attention_nvfp4kv.ptx`) so the FP8 path's compile surface
    // stays untouched. `fused_rope_partial_nvfp4kv_kernel` similarly
    // lives in its own PTX. Both modules are loaded alongside the main
    // `flash_attention.ptx` — the module handles sit on this struct so
    // the KernelFn pointers stay valid across calls.
    /// `flash_attention_nvfp4kv.ptx` — owner of the four NVFP4 FA2
    /// entry points below. Lifetime anchored here for KernelFn
    /// validity across calls.
    #[cfg(feature = "cuda")]
    pub flash_attention_nvfp4kv_mod: Option<rvllm_kernels::LoadedModule>,
    /// `fused_rope_partial_nvfp4kv.ptx` — owner of the single
    /// `fused_rope_partial_nvfp4kv_kernel` entry point.
    #[cfg(feature = "cuda")]
    pub fused_rope_nvfp4kv_mod: Option<rvllm_kernels::LoadedModule>,
    /// Cycle 55 step 8 (Phase B): bf16-input sibling of
    /// `fused_rope_nvfp4kv_mod`. None if PTX absent on this build.
    pub fused_rope_nvfp4kv_bf16in_mod: Option<rvllm_kernels::LoadedModule>,
    /// Cycle 55 step 9 (Phase F): bf16-OUTPUT siblings of the 6
    /// NVFP4 attention kernels. Loaded as a single PTX module sharing
    /// the same source-of-truth math; only the output dtype flips
    /// f16 → bf16 across all 6 kernel entries.
    pub flash_attention_nvfp4kv_bf16out_mod: Option<rvllm_kernels::LoadedModule>,
    /// Cycle 55 step 10: bf16-output unified prefill module.
    pub unified_prefill_nvfp4kv_bf16out_mod: Option<rvllm_kernels::LoadedModule>,
    /// Cycle 55 step 10: bf16-output split-decode + reducer module.
    pub split_decode_nvfp4kv_bf16out_mod: Option<rvllm_kernels::LoadedModule>,
    /// `flash_attention_2_decode_nvfp4kv_kernel` — NVFP4 KV decode,
    /// BC=32 variant for head_dim ≤ 256.
    #[cfg(feature = "cuda")]
    pub fn_decode_nvfp4kv: Option<rvllm_kernels::KernelFn>,
    /// BC=16 decode variant for head_dim=512 (global-attn smem fit).
    #[cfg(feature = "cuda")]
    pub fn_decode_nvfp4kv_bc16: Option<rvllm_kernels::KernelFn>,
    /// GQA-grouped decode variants — one CTA per (seq, kv_head)
    /// handling all `num_heads / num_kv_heads` queries sharing that
    /// KV slice. Halves K/V dequant + smem traffic at decode for
    /// models with GQA > 1 (Gemma 4: ratio 2). Dispatched when
    /// `num_heads > num_kv_heads` and GQA ≤ MAX_GQA_DECODE (=4).
    #[cfg(feature = "cuda")]
    pub fn_decode_nvfp4kv_gqa: Option<rvllm_kernels::KernelFn>,
    #[cfg(feature = "cuda")]
    pub fn_decode_nvfp4kv_gqa_bc16: Option<rvllm_kernels::KernelFn>,
    /// `flash_attention_2_prefill_nvfp4kv_kernel` — NVFP4 KV prefill,
    /// BC=32 variant.
    #[cfg(feature = "cuda")]
    pub fn_prefill_nvfp4kv: Option<rvllm_kernels::KernelFn>,
    /// BC=16 prefill variant for head_dim=512.
    #[cfg(feature = "cuda")]
    pub fn_prefill_nvfp4kv_bc16: Option<rvllm_kernels::KernelFn>,
    /// `flash_attention_2_prefill_nvfp4kv_unified_kernel` — multi-query
    /// NVFP4 prefill using `m16n8k16` f16 MMA (dequant NVFP4 → f16
    /// smem via `cvt.rn.f16x2.e2m1x2`, then standard SM80-era f16 MMA
    /// driven through `f16_mma_frag_pack.cuh`). Mirror of
    /// `fn_prefill_fp8kv_unified`; targets the ~10× NVFP4-vs-FP8
    /// batch-prefill gap (memory 22222222aa010020). Owner module is
    /// `unified_prefill_nvfp4kv_mod` below. Optional — missing when
    /// the kernel tree predates Phase 2b of aa01001nvf4f16mma.
    #[cfg(feature = "cuda")]
    pub fn_prefill_nvfp4kv_unified: Option<rvllm_kernels::KernelFn>,
    /// Owner PTX module for `fn_prefill_nvfp4kv_unified`.
    #[cfg(feature = "cuda")]
    pub unified_prefill_nvfp4kv_mod: Option<rvllm_kernels::LoadedModule>,
    /// `fused_rope_partial_nvfp4kv_kernel` — RoPE + NVFP4 paged-KV
    /// cache write (layer-exec uses this in the decode/prefill hot
    /// path when `kv_dtype == Nvfp4`).
    #[cfg(feature = "cuda")]
    pub fn_rope_nvfp4kv: Option<rvllm_kernels::KernelFn>,
    /// Cycle 55 step 8: bf16-input sibling of fn_rope_nvfp4kv. Same
    /// launch ABI; Q/K/V activation inputs flip f16 → bf16. cos/sin
    /// tables stay f16 (Phase D revisits).
    pub fn_rope_nvfp4kv_bf16in: Option<rvllm_kernels::KernelFn>,
    /// Cycle 55 step 9 (Phase F): bf16-output siblings of the NVFP4
    /// attention decode + prefill kernels. Six entries cover the
    /// canonical decode (BC=32 + BC=16 variants), GQA (BC=32 + BC=16),
    /// and unified prefill (BC=32 + BC=16). Same launch ABI as their
    /// f16-output counterparts; only the output dtype is bf16 so the
    /// downstream O-projection can consume it without a runtime
    /// f16↔bf16 conversion.
    pub fn_decode_nvfp4kv_bf16out: Option<rvllm_kernels::KernelFn>,
    pub fn_decode_nvfp4kv_bc16_bf16out: Option<rvllm_kernels::KernelFn>,
    pub fn_decode_nvfp4kv_gqa_bf16out: Option<rvllm_kernels::KernelFn>,
    pub fn_decode_nvfp4kv_gqa_bc16_bf16out: Option<rvllm_kernels::KernelFn>,
    pub fn_prefill_nvfp4kv_bf16out: Option<rvllm_kernels::KernelFn>,
    pub fn_prefill_nvfp4kv_bc16_bf16out: Option<rvllm_kernels::KernelFn>,
    /// Cycle 55 step 10: bf16-output unified-prefill kernel handle.
    pub fn_prefill_nvfp4kv_unified_bf16out: Option<rvllm_kernels::KernelFn>,
    /// Cycle 55 step 10: bf16-output split-decode + reducer handles.
    pub fn_decode_nvfp4kv_split_bf16out: Option<rvllm_kernels::KernelFn>,
    pub fn_decode_nvfp4kv_split_bc16_bf16out: Option<rvllm_kernels::KernelFn>,
    pub fn_paged_attn_reduce_bf16: Option<rvllm_kernels::KernelFn>,

    // ---- Split-KV decode path (paged_attention_v2-style) ----
    //
    // Partitions the KV sequence into `PARTITION_SIZE`-token tiles,
    // launches one CTA per (seq, head, partition) in phase 1, then a
    // reduce kernel combines partial outputs in phase 2. Unblocks
    // SM-occupancy at long-context / bs=1 decode — at 15k ctx the
    // per-head kernel only launches 32 CTAs on GB10's 108 SMs (see
    // harness `fa2_nvfp4_split_decode_check.py` for protocol).
    /// Owner PTX module — `flash_attention_split_decode_nvfp4kv.ptx`.
    #[cfg(feature = "cuda")]
    pub split_decode_nvfp4kv_mod: Option<rvllm_kernels::LoadedModule>,
    /// `flash_attention_2_decode_nvfp4kv_split_kernel` (BC=32).
    #[cfg(feature = "cuda")]
    pub fn_decode_nvfp4kv_split: Option<rvllm_kernels::KernelFn>,
    /// `flash_attention_2_decode_nvfp4kv_split_bc16_kernel`.
    #[cfg(feature = "cuda")]
    pub fn_decode_nvfp4kv_split_bc16: Option<rvllm_kernels::KernelFn>,
    /// `paged_attention_reduce_f16_kernel` — combines partial outputs.
    /// Head-dtype-agnostic (f16 in / out). Shared by FP8 split path
    /// when that's added.
    #[cfg(feature = "cuda")]
    pub fn_paged_attn_reduce_f16: Option<rvllm_kernels::KernelFn>,
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
            let fn_decode_f16io =
                flash_attention_mod.get_function("flash_attention_2_decode_f16io_kernel")?;
            let fn_prefill = flash_attention_mod.get_function("flash_attention_2_kernel")?;
            let fn_prefill_f16kv =
                flash_attention_mod.get_function("flash_attention_2_f16kv_kernel")?;
            let fn_decode_fp8kv =
                flash_attention_mod.get_function("flash_attention_2_decode_fp8kv_kernel")?;
            let fn_decode_fp8kv_gqa = flash_attention_mod
                .get_function("flash_attention_2_decode_fp8kv_gqa_kernel")
                .ok();

            // Optional: the unified prefill PTX module is added in
            // Phase A of `UNIFIED_PREFILL_SPEC.md` and its body lands
            // in Phase B. Treat a missing module / missing symbol as
            // "not available" rather than fatal so bring-up survives
            // until the body compiles.
            let (unified_prefill_mod, fn_prefill_fp8kv_unified) =
                match loader.load_ptx("flash_attention_unified_prefill") {
                    Ok(m) => {
                        let f = m
                            .get_function("flash_attention_2_prefill_fp8kv_unified_kernel")
                            .ok();
                        (Some(m), f)
                    }
                    Err(_) => (None, None),
                };

            // NVFP4 modules — best-effort load. A pre-rusty_sm121_nvfp4
            // kernels/ tree won't have these PTX files; the runtime
            // falls back to Absent on that branch, so returning None
            // here keeps the main load path compatible. Actual dispatch
            // only consults these when `RVLLM_NVFP4_KV=1` is set.
            let (
                flash_attention_nvfp4kv_mod,
                fn_decode_nvfp4kv,
                fn_decode_nvfp4kv_bc16,
                fn_prefill_nvfp4kv,
                fn_prefill_nvfp4kv_bc16,
                fn_decode_nvfp4kv_gqa,
                fn_decode_nvfp4kv_gqa_bc16,
            ) = match loader.load_ptx("flash_attention_nvfp4kv") {
                Ok(m) => {
                    let d    = m.get_function("flash_attention_2_decode_nvfp4kv_kernel").ok();
                    let d16  = m.get_function("flash_attention_2_decode_nvfp4kv_bc16_kernel").ok();
                    let p    = m.get_function("flash_attention_2_prefill_nvfp4kv_kernel").ok();
                    let p16  = m.get_function("flash_attention_2_prefill_nvfp4kv_bc16_kernel").ok();
                    let dg   = m.get_function("flash_attention_2_decode_nvfp4kv_gqa_kernel").ok();
                    let dg16 = m.get_function("flash_attention_2_decode_nvfp4kv_gqa_bc16_kernel").ok();
                    (Some(m), d, d16, p, p16, dg, dg16)
                }
                Err(_) => (None, None, None, None, None, None, None),
            };
            let (fused_rope_nvfp4kv_mod, fn_rope_nvfp4kv) =
                match loader.load_ptx("fused_rope_partial_nvfp4kv") {
                    Ok(m) => {
                        let f = m.get_function("fused_rope_partial_nvfp4kv_kernel").ok();
                        (Some(m), f)
                    }
                    Err(_) => (None, None),
                };
            // Cycle 55 step 8 (Phase B): bf16-input sibling. Separate
            // PTX module so older kernel trees still bring up; runtime
            // dispatch gates on `is_some()`.
            let (fused_rope_nvfp4kv_bf16in_mod, fn_rope_nvfp4kv_bf16in) =
                match loader.load_ptx("fused_rope_partial_nvfp4kv_bf16in") {
                    Ok(m) => {
                        let f = m.get_function("fused_rope_partial_nvfp4kv_bf16in_kernel").ok();
                        (Some(m), f)
                    }
                    Err(_) => (None, None),
                };
            // Cycle 55 step 9 (Phase F): bf16-OUTPUT NVFP4 attention
            // kernels. All 6 entries (decode + decode_bc16 + decode_gqa
            // + decode_gqa_bc16 + prefill + prefill_bc16) live in
            // `flash_attention_nvfp4kv_bf16out`. Selected when the
            // downstream O-projection consumes bf16 (eventually the
            // only path once Phase B/F complete).
            let (
                flash_attention_nvfp4kv_bf16out_mod,
                fn_decode_nvfp4kv_bf16out,
                fn_decode_nvfp4kv_bc16_bf16out,
                fn_prefill_nvfp4kv_bf16out,
                fn_prefill_nvfp4kv_bc16_bf16out,
                fn_decode_nvfp4kv_gqa_bf16out,
                fn_decode_nvfp4kv_gqa_bc16_bf16out,
            ) = match loader.load_ptx("flash_attention_nvfp4kv_bf16out") {
                Ok(m) => {
                    let d    = m.get_function("flash_attention_2_decode_nvfp4kv_bf16out_kernel").ok();
                    let d16  = m.get_function("flash_attention_2_decode_nvfp4kv_bc16_bf16out_kernel").ok();
                    let p    = m.get_function("flash_attention_2_prefill_nvfp4kv_bf16out_kernel").ok();
                    let p16  = m.get_function("flash_attention_2_prefill_nvfp4kv_bc16_bf16out_kernel").ok();
                    let dg   = m.get_function("flash_attention_2_decode_nvfp4kv_gqa_bf16out_kernel").ok();
                    let dg16 = m.get_function("flash_attention_2_decode_nvfp4kv_gqa_bc16_bf16out_kernel").ok();
                    (Some(m), d, d16, p, p16, dg, dg16)
                }
                Err(_) => (None, None, None, None, None, None, None),
            };
            // Cycle 55 step 10: split-decode + unified-prefill
            // bf16-output siblings. Phase-1 split kernels write f32
            // tmp_out (cycle-21 fix) and are dtype-agnostic; only their
            // symbols are renamed. The Phase-2 reducer flips f16→bf16.
            let (
                unified_prefill_nvfp4kv_bf16out_mod,
                fn_prefill_nvfp4kv_unified_bf16out,
            ) = match loader.load_ptx("flash_attention_unified_prefill_nvfp4kv_bf16out") {
                Ok(m) => {
                    let f = m.get_function("flash_attention_2_prefill_nvfp4kv_unified_bf16out_kernel").ok();
                    (Some(m), f)
                }
                Err(_) => (None, None),
            };
            let (
                split_decode_nvfp4kv_bf16out_mod,
                fn_decode_nvfp4kv_split_bf16out,
                fn_decode_nvfp4kv_split_bc16_bf16out,
                fn_paged_attn_reduce_bf16,
            ) = match loader.load_ptx("flash_attention_split_decode_nvfp4kv_bf16out") {
                Ok(m) => {
                    let s32 = m.get_function("flash_attention_2_decode_nvfp4kv_split_bf16out_kernel").ok();
                    let s16 = m.get_function("flash_attention_2_decode_nvfp4kv_split_bc16_bf16out_kernel").ok();
                    let r   = m.get_function("paged_attention_reduce_bf16_kernel").ok();
                    (Some(m), s32, s16, r)
                }
                Err(_) => (None, None, None, None),
            };

            // Unified NVFP4 prefill — Phase 2b follow-up. Separate PTX
            // so old kernel trees can still bring up (runtime gates
            // dispatch on `is_some()`).
            let (unified_prefill_nvfp4kv_mod, fn_prefill_nvfp4kv_unified) =
                match loader.load_ptx("flash_attention_unified_prefill_nvfp4kv") {
                    Ok(m) => {
                        let f = m
                            .get_function(
                                "flash_attention_2_prefill_nvfp4kv_unified_kernel",
                            )
                            .ok();
                        (Some(m), f)
                    }
                    Err(_) => (None, None),
                };

            // Split-KV decode module — optional. Missing on PTX trees
            // predating the paged_attention_v2-style split kernel.
            let (
                split_decode_nvfp4kv_mod,
                fn_decode_nvfp4kv_split,
                fn_decode_nvfp4kv_split_bc16,
                fn_paged_attn_reduce_f16,
            ) = match loader.load_ptx("flash_attention_split_decode_nvfp4kv") {
                Ok(m) => {
                    let s32 = m.get_function(
                        "flash_attention_2_decode_nvfp4kv_split_kernel").ok();
                    let s16 = m.get_function(
                        "flash_attention_2_decode_nvfp4kv_split_bc16_kernel").ok();
                    let r   = m.get_function(
                        "paged_attention_reduce_f16_kernel").ok();
                    (Some(m), s32, s16, r)
                }
                Err(_) => (None, None, None, None),
            };

            Ok(Self {
                head_dim,
                flash_attention_mod,
                fn_decode,
                fn_decode_f16io,
                fn_prefill,
                fn_prefill_f16kv,
                fn_decode_fp8kv,
                fn_decode_fp8kv_gqa,
                fn_prefill_fp8kv_unified,
                unified_prefill_mod,
                flash_attention_nvfp4kv_mod,
                fused_rope_nvfp4kv_mod,
                fused_rope_nvfp4kv_bf16in_mod,
                flash_attention_nvfp4kv_bf16out_mod,
                unified_prefill_nvfp4kv_bf16out_mod,
                split_decode_nvfp4kv_bf16out_mod,
                fn_decode_nvfp4kv,
                fn_decode_nvfp4kv_bc16,
                fn_decode_nvfp4kv_gqa,
                fn_decode_nvfp4kv_gqa_bc16,
                fn_prefill_nvfp4kv,
                fn_prefill_nvfp4kv_bc16,
                fn_prefill_nvfp4kv_unified,
                unified_prefill_nvfp4kv_mod,
                fn_rope_nvfp4kv,
                fn_rope_nvfp4kv_bf16in,
                fn_decode_nvfp4kv_bf16out,
                fn_decode_nvfp4kv_bc16_bf16out,
                fn_decode_nvfp4kv_gqa_bf16out,
                fn_decode_nvfp4kv_gqa_bc16_bf16out,
                fn_prefill_nvfp4kv_bf16out,
                fn_prefill_nvfp4kv_bc16_bf16out,
                fn_prefill_nvfp4kv_unified_bf16out,
                fn_decode_nvfp4kv_split_bf16out,
                fn_decode_nvfp4kv_split_bc16_bf16out,
                fn_paged_attn_reduce_bf16,
                split_decode_nvfp4kv_mod,
                fn_decode_nvfp4kv_split,
                fn_decode_nvfp4kv_split_bc16,
                fn_paged_attn_reduce_f16,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            // Non-CUDA build (mock worker, laptop integration tests).
            // The bf16in / bf16out NVFP4 fields are declared without a
            // `#[cfg(feature = "cuda")]` gate, so the struct shape is
            // identical across feature flags — initialising them all
            // to `None` keeps `cargo check --workspace` and
            // `--no-default-features` building. Without these explicit
            // None fields, this branch fails compile with E0063
            // "missing fields …" the moment a new bf16 kernel slot
            // is added to the struct.
            let _ = loader;
            Ok(Self {
                head_dim,
                fused_rope_nvfp4kv_bf16in_mod: None,
                flash_attention_nvfp4kv_bf16out_mod: None,
                unified_prefill_nvfp4kv_bf16out_mod: None,
                split_decode_nvfp4kv_bf16out_mod: None,
                fn_rope_nvfp4kv_bf16in: None,
                fn_decode_nvfp4kv_bf16out: None,
                fn_decode_nvfp4kv_bc16_bf16out: None,
                fn_decode_nvfp4kv_gqa_bf16out: None,
                fn_decode_nvfp4kv_gqa_bc16_bf16out: None,
                fn_prefill_nvfp4kv_bf16out: None,
                fn_prefill_nvfp4kv_bc16_bf16out: None,
                fn_prefill_nvfp4kv_unified_bf16out: None,
                fn_decode_nvfp4kv_split_bf16out: None,
                fn_decode_nvfp4kv_split_bc16_bf16out: None,
                fn_paged_attn_reduce_bf16: None,
            })
        }
    }

    /// Convenience: is the unified multi-Q FP8 prefill kernel
    /// available for this head_dim? Returns `false` when the PTX
    /// module is missing (Phase A/B not built yet) or the function
    /// symbol didn't resolve. Callers that can fall back to the
    /// decode-per-qi loop gate on this.
    pub fn has_unified_prefill(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            self.fn_prefill_fp8kv_unified.is_some()
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }

    /// NVFP4 analogue: is the f16-MMA unified NVFP4 prefill kernel
    /// loaded? Dispatched from `gemma4_layer_exec.rs` when
    /// `kv_dtype == Nvfp4 && num_tokens > 1`.
    pub fn has_unified_prefill_nvfp4(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            self.fn_prefill_nvfp4kv_unified.is_some()
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
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
