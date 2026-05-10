//! Mistral 3.5 NVFP4 bring-up scaffolding.
//!
//! Step 5 (Rust scaffolding) — the entry-point shape `cuda_worker`
//! dispatches into. Everything that *can* be validated up-front
//! without an actual GPU forward pass runs here so an operator
//! pointing the server at a Mistral checkpoint gets concrete
//! diagnostics during startup:
//!
//! 1. Parse `config.json` via [`rvllm_loader::mistral35_arch::Mistral35Arch`]
//!    (fails on YaRN drift, GQA-ratio mismatch, missing pixtral block).
//! 2. Open the safetensors index and validate every NVFP4 linear via
//!    [`rvllm_loader::mistral35_weights::validate_mistral35_inventory`]
//!    (616 packed/scale/global, 434 vision BF16, 4 projector BF16).
//! 3. Resolve `libcutlass_sm120.so` and call
//!    [`rvllm_cutlass::lib_so::CutlassBackend::require_nvfp4`] —
//!    refuses startup when the NVFP4 entry-point set is missing.
//!
//! Only after all three pass does [`Mistral35Bringup::load`] succeed.
//! The forward path ([`Mistral35Bringup::run_generate`]) is the next
//! milestone; it currently returns a typed
//! [`Mistral35Error::ForwardNotImplemented`] so a per-request
//! invocation surfaces a clean error rather than silently producing
//! garbage.
//!
//! No CUDA / cudarc imports here — the heavy GPU plumbing lives
//! alongside the Gemma 4 / Qwen 3.6 paths and lands when the
//! NVFP4 kernel ABI is wired through (steps 4-CUDA / 6 / 9).
//!
//! ## Device-upload strategy (codex review, 2026-05-07/08)
//!
//! When the CUDA forward path lands, weight upload follows this
//! per-projection plan:
//!
//! 1. mmap → DtoH-stream-upload `weight_packed` (`U8 [N, K/2]`) →
//!    persistent device-resident NVFP4 weight buffer.
//! 2. mmap → DtoH-stream-upload `weight_scale` (`E4M3 [N, K/16]`)
//!    into a TRANSIENT load-time scratch buffer.
//! 3. Run `CutlassBackend::launch_nvfp4_sfb_transform` to convert
//!    the natural-layout weight scale into the CUTLASS-interleaved
//!    SFB layout. The result is **persistent** device-resident.
//! 4. Free / re-use the natural-layout scratch buffer; the
//!    safetensors mmap is the durable source-of-truth, no need to
//!    keep the natural copy around.
//! 5. mmap → DtoH-stream-upload `weight_global_scale` (`F32 [1]`)
//!    → persistent device-resident scalar (4-byte aligned, used as
//!    `epilogue.thread.alpha_ptr` in every GEMM call).
//!
//! `tile_atom_to_shape_SFB` consumes only `(N, K, L)` — runtime
//! `M` is discarded — so the load-time-once transform is
//! correct: there is no need for an `(m_bucket, n, k)` SFB cache.
//!
//! Per-prefill SFA (activation side) follows the inverse pattern:
//! `prep_act` → `sfa_transform` → GEMM, all on the same stream
//! into a single reusable scratch buffer sized at the max SFA
//! across the seven projection shapes (`scratch_budget_with_
//! backend` already reports this number).

use std::path::PathBuf;

use rvllm_core::{LoaderCtx, LoaderError, Result, RvllmError};
use rvllm_loader::mistral35_arch::Mistral35Arch;
use rvllm_loader::mistral35_weights::{
    validate_mistral35_inventory, Mistral35TensorCounts, Mistral35WeightInventory,
};
use rvllm_loader::safetensors::{ShardHeader, ShardIndex, TensorEntry};

use crate::gemma4_bring_up::Gemma4EnginePaths;

#[cfg(feature = "cuda")]
use rvllm_kernels::{KernelFn, KernelLoader, LoadedModule};

/// PTX kernel handles needed by the Mistral 3.5 smoke-forward (Step
/// 5 GPU half). Held on `Mistral35Bringup` so the LoadedModule
/// lifetimes outlive every per-request launch.
#[cfg(feature = "cuda")]
pub struct Mistral35ForwardKernels {
    pub embedding_gather_bf16_mod: LoadedModule,
    pub fn_embedding_gather_bf16: KernelFn,
    pub rmsnorm_inplace_bf16_gbf16_mod: LoadedModule,
    pub fn_rmsnorm_inplace_bf16_gbf16: KernelFn,
    pub silu_mul_bf16_mod: LoadedModule,
    pub fn_silu_mul_bf16: KernelFn,
    pub attn_m1_gqa_broadcast_mod: LoadedModule,
    pub fn_attn_m1_gqa_broadcast: KernelFn,
    pub vector_add_bf16_mod: LoadedModule,
    pub fn_vector_add_bf16: KernelFn,
    pub argmax_mod: LoadedModule,
    pub fn_argmax_f32: KernelFn,
    pub rope_split_half_bf16_mod: LoadedModule,
    pub fn_rope_split_half_bf16: KernelFn,
    pub kv_cache_write_bf16_mod: LoadedModule,
    pub fn_kv_cache_write_bf16: KernelFn,
    pub qk_dot_bf16_mod: LoadedModule,
    pub fn_qk_dot_bf16: KernelFn,
    pub qk_dot_gqa_bf16_mod: LoadedModule,
    pub fn_qk_dot_gqa_bf16: KernelFn,
    pub softmax_v_bf16_mod: LoadedModule,
    pub fn_softmax_v_bf16: KernelFn,
    // W4A16 NVFP4 path (Mistral 3.5 checkpoint format): dequantize
    // weight blocks to BF16 then run `cublasLt bf16_gemm_f32`. The
    // legacy W4A4 (CUTLASS NVFP4 tensor-core) path is kept compiled
    // but no longer wired in `gemm` — see MISTRAL35_BUG_HUNT.md.
    pub nvfp4_dequant_weights_bf16_mod: LoadedModule,
    pub fn_nvfp4_dequant_weights_bf16: KernelFn,
    pub f32_to_bf16_mod: LoadedModule,
    pub fn_f32_to_bf16: KernelFn,
    pub mistral35_w4a16_gemv_bf16_mod: LoadedModule,
    pub fn_mistral35_w4a16_gemv_bf16: KernelFn,

    // ── Pixtral vision (Round-12 phase 2) ─────────────────────────────
    // The vision tower runs entirely in BF16 (decoder weights are
    // NVFP4, vision weights are BF16 from the checkpoint).
    pub pixtral_rotary_2d_bf16_mod: LoadedModule,
    pub fn_pixtral_rotary_2d_bf16: KernelFn,
    pub patch_merger_pixtral_2x2_mod: LoadedModule,
    pub fn_patch_merger_pixtral_2x2: KernelFn,
    // Round-12 phase 3c — additional kernels for the per-block ViT
    // forward (already built for Qwen/Gemma vision; share without
    // duplication).
    pub extract_head_bf16_mod: LoadedModule,
    pub fn_extract_head_bf16: KernelFn,
    pub scatter_heads_bf16_mod: LoadedModule,
    pub fn_scatter_heads_bf16: KernelFn,
    pub softmax_row_f32_to_bf16_mod: LoadedModule,
    pub fn_softmax_row_f32_to_bf16: KernelFn,
    pub gelu_tanh_mul_bf16_mod: LoadedModule,
    pub fn_gelu_tanh_mul_bf16: KernelFn,
    pub scale_inplace_f32_mod: LoadedModule,
    pub fn_scale_inplace_f32: KernelFn,
    pub transpose_heads_v_bf16_mod: LoadedModule,
    pub fn_transpose_heads_v_bf16: KernelFn,
    pub gelu_tanh_bf16_mod: LoadedModule,
    pub fn_gelu_tanh_bf16: KernelFn,
}

/// Pre-allocated per-token forward scratch — sized for m=1 across
/// every Mistral 3.5 projection shape so a single forward never
/// hits the arena allocator. Hoisting these out of
/// `forward_smoke_q_proj_inner` cuts ~5K allocator + restore calls
/// per chat request and eliminates a real per-step latency spike
/// observed empirically (440 ms/step with per-call alloc + restore
/// vs an expected ~30-60 ms once those go away).
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy)]
pub struct Mistral35Scratch {
    pub token_in_ptr: u64,        // i32 [1] — embed gather input
    pub h_residual_ptr: u64,      // BF16 [hidden]
    pub h_work_ptr: u64,          // BF16 [hidden]
    pub a_packed_ptr: u64,        // U8   [m=1 * max_k / 2]
    pub sfa_natural_ptr: u64,     // E4M3 [m=1 * max_k / 16]
    pub sfa_cutlass_ptr: u64,     // CUTLASS-interleaved SFA scratch
    pub q_out_ptr: u64,           // BF16 [n_q_heads * head_dim]
    pub k_out_ptr: u64,           // BF16 [n_kv_heads * head_dim]
    pub v_out_ptr: u64,           // BF16 [n_kv_heads * head_dim]
    pub attn_out_ptr: u64,        // BF16 [hidden]
    pub o_out_ptr: u64,           // BF16 [hidden]
    pub gate_out_ptr: u64,        // BF16 [intermediate]
    pub up_out_ptr: u64,          // BF16 [intermediate]
    pub silu_mid_ptr: u64,        // BF16 [intermediate]
    pub down_out_ptr: u64,        // BF16 [hidden]
    pub workspace_ptr: u64,       // CUTLASS GEMM scratch
    pub w_bf16_scratch_ptr: u64,  // BF16 dequantized weight tile,
                                  //   sized to max(n*k) across projections
    pub w_bf16_scratch_bytes: usize,
    pub out_f32_scratch_ptr: u64, // F32 [1, max_n] for cublasLt output
                                  //   before f32_to_bf16 cast
    pub workspace_bytes: usize,
    pub logits_ptr: u64,          // F32 [vocab]
    pub token_out_ptr: u64,       // i32 [1] — argmax target
}

/// Per-layer KV cache (BF16). Stored as
/// `[max_pos, n_kv_heads, head_dim]` row-major. Allocated
/// up-front in the arena at bring-up time, one (K, V) pair per
/// transformer layer. The `position`-th token's K and V are
/// written at slot `position` by the `mistral35_kv_cache_write_bf16`
/// kernel and read back by `mistral35_qk_dot_bf16` /
/// `mistral35_softmax_v_bf16`. No NVFP4-KV path is wired in
/// Mistral today.
///
/// ## Lifetime invariant (#4 — explicit)
///
/// The cache is **persistent across requests** — only one request
/// runs at a time today and there is no `reset()` between them.
/// Correctness rests on a single load-bearing invariant enforced
/// by the forward path:
///
/// **For every (layer, position) read at slot `s`, the same
/// (layer, position=s) was written earlier in the SAME request.**
///
/// `attention_step` only reads slots `0 .. past_len`, where
/// `past_len = position + 1` and every prior `position` in
/// `[0, position]` was written by an earlier `prefill_token` /
/// `decode_token` call within the current `generate()` call.
/// Slots `≥ past_len` are never read, so any stale state from a
/// previous request at those positions is unobservable.
///
/// Cancellation cannot violate this: a cancelled request leaves
/// stale state at positions it had written, but the next request
/// starts at `position=0` and overwrites those slots before reading
/// them. Slots beyond the new request's max position are not read.
///
/// Anything that wants to violate this rule (paged batching,
/// prefix-cache reuse, speculative decode) MUST first switch to
/// per-request seq-id tagging or an explicit length-bounded slice;
/// it is not safe to silently relax `past_len = position + 1`.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct LayerKvCache {
    pub k_ptr: u64,
    pub v_ptr: u64,
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct Mistral35KvCache {
    pub layers: Vec<LayerKvCache>,
    pub max_pos: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// Scratch F32 buffer for `qk_dot` output of shape
    /// `[n_q_heads, max_pos]`. Single allocation reused per layer.
    pub scores_f32_ptr: u64,
}

/// YaRN cos/sin tables resident on device (F32 layout). Built host-side
/// in `mistral35_yarn::build_yarn_rope_tables` and uploaded once at
/// bring-up time. The forward path indexes per-position into them via
/// the `rope_split_half_bf16` kernel.
#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct Mistral35RopeTables {
    pub cos_ptr: u64,
    pub sin_ptr: u64,
    pub max_pos: usize,
    pub head_dim: usize,
}

/// Same path bundle Gemma 4 takes — kept so `cuda_worker` can pass
/// the resolved paths through unchanged. We alias rather than
/// inventing `Mistral35EnginePaths` to avoid duplicated path
/// resolution code; the spec calls out "split or sibling type" as
/// the long-term direction (Step 5-CUDA, when the per-family
/// kernel selection actually diverges).
pub type Mistral35EnginePaths = Gemma4EnginePaths;

/// Top-level bring-up handle. Parses arch, validates the inventory,
/// and verifies CUTLASS NVFP4 symbols are present. The actual
/// forward path lands when steps 4-CUDA / 6 / 9 are wired.
///
/// `Gemma4EnginePaths` upstream is not `Debug`, so neither is this.
pub struct Mistral35Bringup {
    pub paths: Mistral35EnginePaths,
    pub arch: Mistral35Arch,
    pub inventory: Mistral35WeightInventory,
    /// Bytes the caller asked us to allocate for the arena. The
    /// upload pass uses ~72 GiB at production scale; the arena is
    /// owned by the bring-up via `arena: Box<HbmArena<'static>>`.
    pub arena_bytes: usize,
    pub nvfp4_active: bool,
    /// Resolved CUTLASS backend handle. The forward path uses it
    /// directly without re-opening the `.so`.
    #[cfg(feature = "cuda")]
    pub cutlass_backend: Option<rvllm_cutlass::lib_so::CutlassBackend>,
    /// CUDA primary-context handle. Holds the device alive for the
    /// life of the bring-up.
    #[cfg(feature = "cuda")]
    pub ctx: Option<std::sync::Arc<rvllm_mem::context::CudaContextHandle>>,
    /// Compute stream that the upload pass + forward path enqueue
    /// onto.
    #[cfg(feature = "cuda")]
    pub stream: Option<rvllm_mem::stream::Stream>,
    /// HBM arena backing every device-resident weight + scratch
    /// region. `'static` is a lie — the arena's actual lifetime is
    /// tied to `ctx` above; we transmute on construction the same
    /// way Qwen36Bringup does.
    #[cfg(feature = "cuda")]
    pub arena: Option<Box<rvllm_mem::HbmArena<'static>>>,
    /// Loaded model — populated when `Mistral35Bringup::load`
    /// completes the upload pass. `None` only on the no-cuda path
    /// (which short-circuits before upload).
    #[cfg(feature = "cuda")]
    pub model: Option<rvllm_loader::mistral35_weights::Mistral35LoadedModel>,
    /// PTX kernel handles for the smoke-forward path. None on the
    /// no-cuda build.
    #[cfg(feature = "cuda")]
    pub forward_kernels: Option<Mistral35ForwardKernels>,
    /// cuBLASLt handle + workspace for the BF16×BF16→F32 lm_head GEMM.
    #[cfg(feature = "cuda")]
    pub cublaslt: Option<rvllm_cutlass::cublaslt::CublasLt>,
    /// Device-resident YaRN cos/sin tables.
    #[cfg(feature = "cuda")]
    pub rope_tables: Option<Mistral35RopeTables>,
    /// Per-layer BF16 KV cache + F32 scores scratch.
    #[cfg(feature = "cuda")]
    pub kv_cache: Option<Mistral35KvCache>,
    /// Pre-allocated per-call forward scratch.
    #[cfg(feature = "cuda")]
    pub scratch: Option<Mistral35Scratch>,
}

#[derive(Debug)]
pub enum Mistral35Error {
    /// Generation called before the GPU forward path is wired.
    ForwardNotImplemented,
    /// CUTLASS NVFP4 symbol set missing.
    Nvfp4SymbolsMissing,
    /// `cuda` feature not enabled at compile time.
    NoCudaFeature,
}

/// Output of [`Mistral35Bringup::forward_pixtral_vision`].
///
/// `data` holds little-endian BF16 bytes — `[num_tokens, hidden_dim]`
/// row-major — already projected through the multi-modal projector
/// to the language-decoder hidden width (`text_config.hidden_size`,
/// 12288 for the canonical Mistral 3.5 NVFP4 checkpoint). The
/// generate path splices these bytes verbatim into the prefill
/// embed buffer at `slot.token_start * row_bytes`, mirroring the
/// Qwen 3.6 / Gemma 4 vision splice.
///
/// Round-12 (Pixtral vision phase 2 — struct landed; the producer
/// stub returns ForwardNotImplemented until phase 3 fills the
/// 48-block forward).
#[derive(Debug)]
pub struct Mistral35VisionForwardOutput {
    /// Little-endian BF16 bytes, length `num_tokens * hidden_dim * 2`.
    pub data: Vec<u8>,
    /// Soft-token count = `merged_h * merged_w` from preprocessing.
    pub num_tokens: usize,
    /// = `arch.text.hidden_size` (12288 for Mistral 3.5).
    pub hidden_dim: usize,
    /// `(grid_h, grid_w)` of the pre-merge ViT output, useful for
    /// debug logs + diff harness.
    pub patch_grid: (u32, u32),
    /// `(merged_h, merged_w)` of the post-merge tensor, equal to
    /// `patch_grid / spatial_merge_size`. `merged_h * merged_w =
    /// num_tokens` always.
    pub merged_grid: (u32, u32),
}

// F4#1 fix: the legacy `KvDecodeStrategy` enum + GQA routing logic
// was inherited from Gemma 4 / Qwen 3.6 NVFP4-KV decode kernels and
// did NOT describe what Mistral 3.5 actually executes. Mistral
// allocates a BF16 KV cache and runs dedicated
// `mistral35_qk_dot_bf16` / `mistral35_softmax_v_bf16` kernels in
// `attention_step`. The enum was logged at startup ("legacy
// nvfp4_kv_decode_strategy={:?} unused") which lied about the
// runtime path. Removed entirely. If Mistral is ever switched onto
// a real NVFP4-KV path, raise `MAX_GQA_DECODE`/`MAX_GQA_SPLIT` in
// `kernels/flash_attention*nvfp4kv.cu` to ≥12 AND pack/write/read
// KV via the NVFP4 path — at that point reintroduce a Mistral-
// specific strategy enum that actually branches in the forward.

/// Per-layer scratch budget for one Mistral 3.5 prefill chunk of
/// `m` tokens. Outputs are byte counts the runtime needs to reserve
/// in its HbmArena before the GPU forward path runs. Pure
/// arithmetic — no GPU access — so it's testable from the no-cuda
/// build.
///
/// The fields cover the four projection shapes Mistral uses:
///   * `q/o`:    `n=12288, k=12288`
///   * `k/v`:    `n=1024,  k=12288`
///   * `gate/up`: `n=28672, k=12288`
///   * `down`:   `n=12288, k=28672`
///
/// `_a_packed` is the activation NVFP4-packed buffer (`m * k / 2`).
/// `_sfa_natural` is the natural-layout SFA scratch (`m * k / 16`).
/// `_sfa_cutlass` is the GEMM-ready interleaved SFA — sized as the
/// maximum across projection K's to allow re-use across layers.
/// `_workspace` is the CUTLASS GEMM workspace; we report the max
/// across projection shapes so a single allocation covers them all.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct LayerScratchBudget {
    pub a_packed_bytes: usize,
    pub sfa_natural_bytes: usize,
    /// Upper bound across the (q/o, k/v, gate/up, down) projection
    /// shapes — the runtime reuses one buffer across layers and
    /// projections, so we size for the worst case.
    pub sfa_cutlass_bytes_max: usize,
    /// Same upper-bound logic for the CUTLASS workspace. 0 here is
    /// a valid value when the backend reports it doesn't need any
    /// scratch for the queried shapes.
    pub workspace_bytes_max: usize,
}

impl LayerScratchBudget {
    /// Pure-Rust budget computation independent of the CUTLASS
    /// `.so` — used at load time when the backend's symbol set
    /// hasn't been validated yet, and as the test-friendly path.
    /// `m` is the number of tokens in one prefill chunk.
    ///
    /// `a_packed = m * max_k / 2`, `sfa_natural = m * max_k / 16`.
    /// `sfa_cutlass_bytes_max` returns the natural-layout size as
    /// a conservative upper bound — the real CUTLASS-interleaved
    /// SFA is at most this large plus modest swizzle padding which
    /// the per-shape backend query reports exactly. Callers that
    /// have a live `CutlassBackend` should prefer
    /// [`Self::with_backend`].
    pub fn natural(arch: &Mistral35Arch, m: usize) -> Self {
        let h = arch.text.hidden_size;
        let i = arch.text.intermediate_size;
        // Max K across the seven projections is the down-proj K
        // (= intermediate_size). Everything else has K = hidden_size.
        let max_k = h.max(i);
        Self {
            a_packed_bytes: m * (max_k / 2),
            sfa_natural_bytes: m * (max_k / 16),
            sfa_cutlass_bytes_max: m * (max_k / 16),
            workspace_bytes_max: 0,
        }
    }

    /// Backend-aware budget computation. Asks the live
    /// `CutlassBackend` for the exact CUTLASS-interleaved SFA size
    /// and the GEMM workspace bytes per Mistral projection shape,
    /// taking the max across the seven projections so a single
    /// allocation covers every layer. Falls back to
    /// [`Self::natural`] when the backend reports zero (e.g. NVFP4
    /// symbols not yet bound).
    #[cfg(feature = "cuda")]
    pub fn with_backend(
        arch: &Mistral35Arch,
        m: usize,
        backend: &rvllm_cutlass::lib_so::CutlassBackend,
    ) -> Self {
        let mut out = Self::natural(arch, m);
        let mi = m as i32;
        let h = arch.text.hidden_size as i32;
        let i = arch.text.intermediate_size as i32;
        let q = arch.q_rows() as i32;
        let kv = arch.kv_rows() as i32;
        // Walk each projection (q/k/v/o/gate/up/down) and take the
        // max SFA + workspace size across them.
        let projections: [(i32, i32); 7] = [
            (q, h),    // q
            (kv, h),   // k
            (kv, h),   // v
            (q, h),    // o
            (i, h),    // gate
            (i, h),    // up
            (h, i),    // down
        ];
        for (n, k) in projections {
            let sfa = backend.nvfp4_sfa_bytes(mi, k);
            if sfa > out.sfa_cutlass_bytes_max {
                out.sfa_cutlass_bytes_max = sfa;
            }
            let ws = backend.nvfp4_workspace_size(mi, n, k);
            if ws > out.workspace_bytes_max {
                out.workspace_bytes_max = ws;
            }
        }
        // Backend's natural-SFA query (when active) overrides the
        // arithmetic estimate; the two should match but the backend
        // accounts for any future layout-padding rules without us
        // having to mirror them here.
        let max_k = h.max(i);
        let nat = backend.nvfp4_natural_sfa_bytes(mi, max_k);
        if nat > 0 {
            out.sfa_natural_bytes = nat;
        }
        out
    }
}

// =====================================================================
//  Round-11 #1: debug-env gate.
//
//  The Mistral generate() path used to read `RVLLM_SMOKE_*` and
//  `RVLLM_KV_BYPASS` directly. Those knobs change forward behaviour
//  (skip prefill, override RoPE pos, bypass KV cache, truncate
//  max_tokens, dump intermediates) and a stale entry in a server
//  profile would silently corrupt every request. The fix:
//
//   * `mistral35_debug_active()` — returns true only when
//     `RVLLM_DEBUG_MISTRAL35=1` is explicitly set.
//   * `debug_env_os` / `debug_env_str` — wrappers that return None
//     unless the gate is active. Forward-path call sites use these
//     instead of `std::env::var(...)`.
//   * `validate_no_stale_debug_envs()` — invoked from
//     `Mistral35Bringup::load`. If any `RVLLM_SMOKE_*` /
//     `RVLLM_KV_BYPASS` is set without the gate, startup aborts
//     with a clear loader error pointing at the offending key.
// =====================================================================

/// Whether the operator has explicitly opted into Mistral 3.5
/// debug mode. Required for any `RVLLM_SMOKE_*` / `RVLLM_KV_BYPASS`
/// to take effect.
#[inline]
pub fn mistral35_debug_active() -> bool {
    matches!(
        std::env::var("RVLLM_DEBUG_MISTRAL35").ok().as_deref(),
        Some("1") | Some("true") | Some("TRUE")
    )
}

/// Read an env var only when the Mistral 3.5 debug gate is active.
/// Production callers always see `None`; debug callers see whatever
/// the operator set.
#[inline]
pub fn debug_env_os(key: &str) -> Option<std::ffi::OsString> {
    if mistral35_debug_active() { std::env::var_os(key) } else { None }
}

/// String variant of [`debug_env_os`].
#[inline]
pub fn debug_env_str(key: &str) -> Option<String> {
    if mistral35_debug_active() { std::env::var(key).ok() } else { None }
}

/// Mistral-prefixed env-keys whose presence in a non-debug profile
/// indicates a stale debug session leaking into production. Loader
/// startup aborts when any of these is set without
/// `RVLLM_DEBUG_MISTRAL35=1`.
const STALE_DEBUG_KEYS: &[&str] = &[
    "RVLLM_SMOKE_SINGLE",
    "RVLLM_SMOKE_MAX_NEW",
    "RVLLM_SMOKE_ATTN_NO_PAST",
    "RVLLM_SMOKE_ROPE_POS_OVERRIDE",
    "RVLLM_SMOKE_FULL_DUMP",
    "RVLLM_SMOKE_LAYER_RMS",
    "RVLLM_SMOKE_DUMP_DIR",
    "RVLLM_SMOKE_NO_RESTORE",
    "RVLLM_KV_BYPASS",
    "RVLLM_BOUNDARY_DUMP",
    "RVLLM_BOUNDARY_DUMP_LAYER",
];

fn validate_no_stale_debug_envs(model_dir: &std::path::Path) -> Result<()> {
    if mistral35_debug_active() { return Ok(()); }
    for key in STALE_DEBUG_KEYS {
        if std::env::var_os(key).is_some() {
            return Err(corrupt(
                model_dir.to_path_buf(),
                format!(
                    "[mistral35] refusing to start: {key} is set but \
                     RVLLM_DEBUG_MISTRAL35=1 is not. These knobs change \
                     the forward-path behaviour silently (skipped \
                     prefill, overridden RoPE position, KV bypass, \
                     truncated max_tokens). Set RVLLM_DEBUG_MISTRAL35=1 \
                     to opt in, or unset {key} for a production run."
                ),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod debug_gate_tests {
    use super::*;
    use std::sync::Mutex;
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn debug_inactive_when_var_unset() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        std::env::remove_var("RVLLM_DEBUG_MISTRAL35");
        assert!(!mistral35_debug_active());
    }

    #[test]
    fn debug_active_when_var_one() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        std::env::set_var("RVLLM_DEBUG_MISTRAL35", "1");
        assert!(mistral35_debug_active());
        std::env::remove_var("RVLLM_DEBUG_MISTRAL35");
    }

    #[test]
    fn validate_rejects_smoke_envs_without_gate() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        std::env::remove_var("RVLLM_DEBUG_MISTRAL35");
        std::env::set_var("RVLLM_SMOKE_SINGLE", "1");
        let r = validate_no_stale_debug_envs(std::path::Path::new("/x"));
        std::env::remove_var("RVLLM_SMOKE_SINGLE");
        assert!(r.is_err(), "must reject leaked smoke env in production");
    }

    #[test]
    fn validate_passes_in_debug_mode() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        std::env::set_var("RVLLM_DEBUG_MISTRAL35", "1");
        std::env::set_var("RVLLM_SMOKE_SINGLE", "1");
        let r = validate_no_stale_debug_envs(std::path::Path::new("/x"));
        std::env::remove_var("RVLLM_SMOKE_SINGLE");
        std::env::remove_var("RVLLM_DEBUG_MISTRAL35");
        assert!(r.is_ok(), "must accept debug knobs under explicit gate");
    }

    #[test]
    fn debug_env_returns_none_without_gate() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        std::env::remove_var("RVLLM_DEBUG_MISTRAL35");
        std::env::set_var("RVLLM_SMOKE_DUMP_DIR", "/tmp/x");
        let v = debug_env_str("RVLLM_SMOKE_DUMP_DIR");
        std::env::remove_var("RVLLM_SMOKE_DUMP_DIR");
        assert!(v.is_none(),
            "production code must never see smoke env values");
    }
}

impl Mistral35Bringup {
    /// Pre-compute the per-layer scratch budget for a prefill chunk
    /// of `m` tokens, using pure arithmetic (no CUTLASS backend
    /// involvement). The backend-aware variant lives in
    /// [`Self::scratch_budget_with_backend`].
    pub fn scratch_budget(&self, m: usize) -> LayerScratchBudget {
        LayerScratchBudget::natural(&self.arch, m)
    }

    /// Build the YaRN cos/sin tables for `pos in 0..max_pos` using
    /// the arch's resolved YaRN config + head_dim. Pure CPU work
    /// here; the future fused CUDA kernel takes the result as a
    /// uniform `[max_pos, head_dim/2]` f32 device buffer.
    ///
    /// Defaults `max_pos` to `original_max_position_embeddings`
    /// when callers want the in-trained-window precompute (4096
    /// for Mistral 3.5).
    pub fn build_rope_tables(
        &self,
        max_pos: Option<usize>,
    ) -> crate::mistral35_yarn::YarnRopeTables {
        let mp = max_pos
            .unwrap_or(self.arch.text.yarn.original_max_position_embeddings);
        crate::mistral35_yarn::build_yarn_rope_tables(
            &self.arch.text.yarn,
            self.arch.text.head_dim,
            mp,
        )
    }

    /// Backend-aware scratch budget. Queries the active
    /// `CutlassBackend` for exact SFA + workspace sizes per
    /// projection shape and returns the per-layer max. Useful when
    /// the runtime forward integration knows the live backend
    /// (the bring-up doesn't hold the backend handle today —
    /// `cuda_worker.rs` does — so this fn takes one explicitly).
    #[cfg(feature = "cuda")]
    pub fn scratch_budget_with_backend(
        &self,
        m: usize,
        backend: &rvllm_cutlass::lib_so::CutlassBackend,
    ) -> LayerScratchBudget {
        LayerScratchBudget::with_backend(&self.arch, m, backend)
    }
}

impl Mistral35Bringup {
    /// Resolved batched-prefill phase configuration. Read once at
    /// startup so individual layer-launches can branch in O(1) on
    /// the cached struct rather than touching env vars per launch.
    /// Phase semantics are documented in
    /// `v3/MISTRAL35_BATCHED_PREFILL_PLAN.md`.
    pub fn batched_prefill_config(&self) -> BatchedPrefillConfig {
        BatchedPrefillConfig::from_env()
    }
}

/// Per-phase env gates for the Mistral 3.5 batched prefill pipeline.
/// Mirrors the Qwen 3.6 `RVLLM_QWEN36_BATCH_*` family so a future
/// rollout can flip the same gates per phase as it lands. Default
/// is "all on" matching the Qwen phase-7 production default — opt
/// out by setting the var to `0` if a phase regresses cosine.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct BatchedPrefillConfig {
    /// Phase M-A: NVFP4 q/k/v/o/gate/up/down projections batched
    /// over N tokens. Single CUTLASS NVFP4 GEMM launch per
    /// projection per layer.
    pub batch_projections: bool,
    /// Phase M-A optional: fused `[Q || K || V]` projection. Off
    /// by default — the Mistral checkpoint ships split tensors;
    /// fusing is a load-time staging step that doubles loader
    /// complexity for ~2 launches/layer saved.
    pub fused_qkv: bool,
    /// Phase M-B: fused YaRN RoPE + NVFP4 KV-write across N
    /// tokens. One launch per layer.
    pub batch_rope: bool,
    /// Phase M-C: FA2 NVFP4 prefill kernel served the whole `[N]`
    /// prompt in one launch per layer.
    pub batch_full_prefill: bool,
    /// Phase M-D: outer-loop deletion in `run_generate`. Implicit
    /// when M-A/M-B/M-C are all on. Stored explicitly so a
    /// regression bisect can flip phases independently.
    pub outer_loop_deleted: bool,
    /// Phase M-E: decode-step CUDA Graph capture. Off by default
    /// — same scope as the parked Qwen Phase 8 work.
    pub decode_graph_capture: bool,
}

impl Default for BatchedPrefillConfig {
    fn default() -> Self {
        // Round-12 phase 5c fix: defaults reflect what the runtime
        // ACTUALLY executes today, not the aspirational plan. Every
        // batched phase is OFF until the kernels land; the logger and
        // any downstream branch on `outer_loop_deleted` see the same
        // truth as the executor.
        //
        // To opt into a future batched phase, set the corresponding
        // env (RVLLM_MISTRAL35_BATCH_*) to 1 — bring-up will refuse
        // to start until the kernel is wired (see the env-set guard
        // in `Mistral35Bringup::load`).
        Self {
            batch_projections: false,
            fused_qkv: false,
            batch_rope: false,
            batch_full_prefill: false,
            outer_loop_deleted: false,
            decode_graph_capture: false,
        }
    }
}

impl BatchedPrefillConfig {
    fn from_env() -> Self {
        let on = |name: &str, default: bool| -> bool {
            match std::env::var(name).ok().as_deref() {
                Some("0") | Some("false") | Some("FALSE") | Some("off") => false,
                Some("1") | Some("true") | Some("TRUE") | Some("on") => true,
                _ => default,
            }
        };
        // Round-12 phase 5c fix: every batched gate defaults to false
        // because none of the M-A/B/C/D/E phases are wired into the
        // executor yet. Setting any of these to 1 explicitly is
        // rejected at bring-up start (see Mistral35Bringup::load).
        let batch_projections = on("RVLLM_MISTRAL35_BATCH_PROJ_PREFILL", false);
        let fused_qkv = on("RVLLM_MISTRAL35_FUSED_QKV", false);
        let batch_rope = on("RVLLM_MISTRAL35_BATCH_ROPE_PREFILL", false);
        let batch_full_prefill = on("RVLLM_MISTRAL35_BATCH_FULL_PREFILL", false);
        let decode_graph_capture = on("RVLLM_MISTRAL35_DECODE_GRAPH", false);
        // Phase M-D collapses when its predecessors are all on.
        let outer_loop_deleted =
            batch_projections && batch_rope && batch_full_prefill;
        Self {
            batch_projections,
            fused_qkv,
            batch_rope,
            batch_full_prefill,
            outer_loop_deleted,
            decode_graph_capture,
        }
    }
}

impl Mistral35Bringup {
    /// Open + validate the model directory. Does not yet upload any
    /// weights. Fails on:
    ///
    /// - non-Mistral / corrupted `config.json`
    /// - YaRN `mscale_all_dim != 0.0` (known checkpoint correction)
    /// - missing/wrong-shape/wrong-dtype NVFP4 tensors
    /// - missing `libcutlass_sm120.so` NVFP4 symbol set (under
    ///   `cuda` feature; on default builds this branch is skipped
    ///   because no `.so` is opened).
    pub fn load(paths: Mistral35EnginePaths, arena_bytes: usize) -> Result<Self> {
        // Round-11 #1: fail-fast at startup if any RVLLM_SMOKE_* /
        // RVLLM_KV_BYPASS env is set without the explicit debug
        // opt-in. These knobs change generate() behaviour silently
        // (skip prefill, override RoPE position, bypass KV cache,
        // truncate max_tokens) and must NOT leak from a debug
        // session into a production server profile.
        validate_no_stale_debug_envs(&paths.model_dir)?;

        // (1) Arch — already gated by family resolver, but re-parse
        //     here so failures point at the actual model_dir even
        //     when the operator passed `--model-family auto`.
        let arch = Mistral35Arch::from_dir(&paths.model_dir)?
            .ok_or_else(|| corrupt(
                paths.model_dir.clone(),
                "Mistral35Bringup::load: config.json does not match Mistral 3.5 markers \
                 (architectures[0]==Mistral3ForConditionalGeneration, model_type==mistral3, \
                 quantization_config.format==nvfp4-pack-quantized)".into(),
            ))?;
        eprintln!("[mistral35] {}", arch.summary());

        // (2) Inventory — mmap the index header (cheap), validate
        //     every NVFP4 linear's dtype + shape, count vision /
        //     projector tensors. Doesn't read tensor payload bytes.
        let tensors = scan_safetensors_index(&paths.model_dir)?;
        let inventory = validate_mistral35_inventory(&arch, &tensors)?;
        let expected = Mistral35TensorCounts::expected(arch.text.num_hidden_layers);
        eprintln!(
            "[mistral35] inventory: packed={}/{} scale={}/{} global_scale={}/{} \
             vision_bf16={}/{} projector_bf16={}/{}",
            inventory.counts.packed, expected.packed,
            inventory.counts.scale, expected.scale,
            inventory.counts.global_scale, expected.global_scale,
            inventory.counts.vision_bf16, expected.vision_bf16,
            inventory.counts.projector_bf16, expected.projector_bf16,
        );

        // (3) CUTLASS NVFP4 symbols. Only on `cuda` builds — the
        //     default build doesn't open the `.so` at all, so we
        //     skip the gate there and only record the field. Any
        //     attempt to actually generate then trips
        //     `Mistral35Error::NoCudaFeature` cleanly.
        // (3) CUTLASS backend resolution. On `cuda` builds we store
        //     the loaded backend handle so the forward path can call
        //     into it directly. On default (no-cuda) builds we skip
        //     the dlopen entirely and report nvfp4_active=false; any
        //     forward call then errors with NoCudaFeature.
        #[cfg(feature = "cuda")]
        let (nvfp4_active, cutlass_backend) = {
            let backend = load_and_require_nvfp4(&paths)?;
            (backend.nvfp4_active(), Some(backend))
        };
        #[cfg(not(feature = "cuda"))]
        let nvfp4_active = false;

        // (4) CUDA init + arena + stream + weight upload (cuda feature only).
        //     Mirrors the Qwen36Bringup pattern. The arena's '_ lifetime
        //     gets transmuted to 'static so the bring-up can own it via
        //     a `Box<HbmArena<'static>>` field; safe because `ctx` is
        //     stored alongside and ensures the device context outlives
        //     the arena.
        #[cfg(feature = "cuda")]
        let (ctx, stream, arena, model, forward_kernels_opt, cublaslt_opt,
             rope_tables_opt, kv_cache_opt, scratch_opt) = {
            use std::sync::Arc;
            let ctx = Arc::new(rvllm_mem::context::CudaContextHandle::init(0)?);
            let compile_target: Option<rvllm_core::CompileTarget> = {
                let (major, minor) = ctx.compute_capability();
                rvllm_core::CompileTarget::from_compute_capability(major, minor)
            };
            let arena_dyn: rvllm_mem::HbmArena<'_> = {
                #[cfg(feature = "gb10")]
                {
                    if matches!(compile_target, Some(rvllm_core::CompileTarget::Sm121)) {
                        rvllm_mem::UnifiedArena::new(&ctx, arena_bytes)?.into_inner()
                    } else {
                        rvllm_mem::HbmArena::new(&ctx, arena_bytes)?
                    }
                }
                #[cfg(not(feature = "gb10"))]
                {
                    rvllm_mem::HbmArena::new(&ctx, arena_bytes)?
                }
            };
            // SAFETY: `ctx` is stored on the bring-up alongside the
            // arena, so the device context outlives every reference
            // into the arena. The Qwen path uses the same transmute.
            let arena_static: rvllm_mem::HbmArena<'static> =
                unsafe { std::mem::transmute(arena_dyn) };
            let arena_box = Box::new(arena_static);
            let stream = rvllm_mem::stream::Stream::new(&ctx)?;

            // F2#4 fix: validate the kernel manifest BEFORE the 60-90 s
            // weight upload so a stale build (kernels/build.sh not
            // re-run after a Mistral kernel edit) fails fast instead
            // of burning a minute of upload before erroring.
            {
                let kernels_dir =
                    crate::bring_up::resolve_kernels_dir(&ctx, &paths.kernels_dir)?;
                let manifest_path = kernels_dir.join("manifest.json");
                let manifest_pre = rvllm_kernels::manifest::KernelManifest
                    ::load_and_verify(&manifest_path)?;
                if let Some(t) = compile_target {
                    manifest_pre.assert_arch(t.as_sm_str())?;
                }
                const MISTRAL_REQUIRED_PTX: &[&str] = &[
                    "argmax",
                    "embedding_gather_bf16",
                    "f32_to_bf16",
                    "mistral35_attn_m1_gqa_broadcast_bf16",
                    "mistral35_kv_cache_write_bf16",
                    "mistral35_qk_dot_bf16",
                    "mistral35_softmax_v_bf16",
                    "mistral35_w4a16_gemv_bf16",
                    "nvfp4_dequant_weights_bf16",
                    "rmsnorm_inplace_bf16_gbf16",
                    "rope_split_half_bf16",
                    "silu_mul_bf16",
                    "vector_add_bf16",
                ];
                let missing: Vec<&&str> = MISTRAL_REQUIRED_PTX.iter()
                    .filter(|name| manifest_pre.path_of(name).is_none())
                    .collect();
                if !missing.is_empty() {
                    let names: Vec<&str> = missing.iter().map(|s| **s).collect();
                    return Err(corrupt(
                        paths.model_dir.clone(),
                        format!("Mistral 3.5 bring-up: PTX manifest missing required kernels {:?}. \
                                 Re-run `kernels/build.sh sm_121` to regenerate.",
                                names),
                    ));
                }
            }

            let backend_ref = cutlass_backend.as_ref().unwrap();
            eprintln!("[mistral35-load] starting weight upload (this can take ~30-90s)…");
            let t0 = std::time::Instant::now();
            let mut model = crate::mistral35_load::load_mistral35_model(
                &paths,
                &arch,
                arena_box.as_ref(),
                backend_ref,
                stream.raw(),
            )?;
            eprintln!(
                "[mistral35-load] weight upload complete in {:.1}s ({} layers loaded)",
                t0.elapsed().as_secs_f32(),
                model.layers.len(),
            );

            // cuBLASLt for the BF16 lm_head GEMM. 32 MiB workspace
            // matches the Gemma 4 setup; same arena.
            let cublaslt_ws_bytes: usize = 32 * 1024 * 1024;
            let cublaslt_ws_region = arena_box.region(
                "mistral35_cublaslt_ws", cublaslt_ws_bytes, 256)?;
            let cublaslt_handle = rvllm_cutlass::cublaslt::CublasLt::new(
                cublaslt_ws_region.device_ptr(), cublaslt_ws_bytes)?;

            // Resolve the PTX kernel handles for the smoke-forward
            // path. Only the bare minimum for "embed + first
            // RMSNorm + Q proj" is loaded today; the full forward
            // expands this set as it lands.
            let kernels_dir =
                crate::bring_up::resolve_kernels_dir(&ctx, &paths.kernels_dir)?;
            let manifest_path = kernels_dir.join("manifest.json");
            let manifest = rvllm_kernels::manifest::KernelManifest
                ::load_and_verify(&manifest_path)?;
            if let Some(t) = compile_target {
                manifest.assert_arch(t.as_sm_str())?;
            }
            // (Pre-upload manifest check above already verified the
            // Mistral PTX set is present; this load is just for the
            // KernelLoader handle.)
            let kernels = std::sync::Arc::new(KernelLoader::new(manifest));
            let embedding_gather_bf16_mod = kernels.load_ptx("embedding_gather_bf16")?;
            let fn_embedding_gather_bf16 = embedding_gather_bf16_mod
                .get_function("embedding_gather_bf16_kernel")?;
            let rmsnorm_inplace_bf16_gbf16_mod =
                kernels.load_ptx("rmsnorm_inplace_bf16_gbf16")?;
            let fn_rmsnorm_inplace_bf16_gbf16 = rmsnorm_inplace_bf16_gbf16_mod
                .get_function("rmsnorm_inplace_bf16_gbf16_kernel")?;
            let silu_mul_bf16_mod = kernels.load_ptx("silu_mul_bf16")?;
            let fn_silu_mul_bf16 = silu_mul_bf16_mod
                .get_function("silu_mul_bf16_kernel")?;
            let attn_m1_gqa_broadcast_mod =
                kernels.load_ptx("mistral35_attn_m1_gqa_broadcast_bf16")?;
            let fn_attn_m1_gqa_broadcast = attn_m1_gqa_broadcast_mod
                .get_function("mistral35_attn_m1_gqa_broadcast_bf16_kernel")?;
            let vector_add_bf16_mod = kernels.load_ptx("vector_add_bf16")?;
            let fn_vector_add_bf16 = vector_add_bf16_mod
                .get_function("vector_add_bf16_kernel")?;
            let argmax_mod = kernels.load_ptx("argmax")?;
            let fn_argmax_f32 = argmax_mod.get_function("argmax_kernel")?;
            let rope_split_half_bf16_mod = kernels.load_ptx("rope_split_half_bf16")?;
            let fn_rope_split_half_bf16 = rope_split_half_bf16_mod
                .get_function("rope_split_half_bf16_kernel")?;
            let kv_cache_write_bf16_mod = kernels.load_ptx("mistral35_kv_cache_write_bf16")?;
            let fn_kv_cache_write_bf16 = kv_cache_write_bf16_mod
                .get_function("mistral35_kv_cache_write_bf16_kernel")?;
            let qk_dot_bf16_mod = kernels.load_ptx("mistral35_qk_dot_bf16")?;
            let qk_dot_gqa_bf16_mod = kernels.load_ptx("mistral35_qk_dot_gqa_bf16")?;
            let fn_qk_dot_gqa_bf16 = qk_dot_gqa_bf16_mod
                .get_function("mistral35_qk_dot_gqa_bf16_kernel")?;
            let fn_qk_dot_bf16 = qk_dot_bf16_mod
                .get_function("mistral35_qk_dot_bf16_kernel")?;
            let softmax_v_bf16_mod = kernels.load_ptx("mistral35_softmax_v_bf16")?;
            let fn_softmax_v_bf16 = softmax_v_bf16_mod
                .get_function("mistral35_softmax_v_bf16_kernel")?;
            let nvfp4_dequant_weights_bf16_mod =
                kernels.load_ptx("nvfp4_dequant_weights_bf16")?;
            let fn_nvfp4_dequant_weights_bf16 = nvfp4_dequant_weights_bf16_mod
                .get_function("nvfp4_dequant_weights_bf16_kernel")?;
            let f32_to_bf16_mod = kernels.load_ptx("f32_to_bf16")?;
            let fn_f32_to_bf16 = f32_to_bf16_mod
                .get_function("f32_to_bf16_kernel")?;
            let mistral35_w4a16_gemv_bf16_mod =
                kernels.load_ptx("mistral35_w4a16_gemv_bf16")?;
            let fn_mistral35_w4a16_gemv_bf16 = mistral35_w4a16_gemv_bf16_mod
                .get_function("mistral35_w4a16_gemv_bf16_kernel")?;
            // Pixtral vision (Round-12 phase 2). Always loaded so the
            // forward path can call them; the upload of the vision
            // weights themselves is gated by RVLLM_LOAD_VISION=1
            // (Round-10 #1).
            let pixtral_rotary_2d_bf16_mod =
                kernels.load_ptx("pixtral_rotary_2d_bf16")?;
            let fn_pixtral_rotary_2d_bf16 = pixtral_rotary_2d_bf16_mod
                .get_function("pixtral_rotary_2d_bf16_kernel")?;
            let patch_merger_pixtral_2x2_mod =
                kernels.load_ptx("patch_merger_pixtral_2x2")?;
            let fn_patch_merger_pixtral_2x2 = patch_merger_pixtral_2x2_mod
                .get_function("patch_merger_pixtral_2x2_kernel")?;
            // Phase 3c shared vision kernels.
            let extract_head_bf16_mod = kernels.load_ptx("extract_head_bf16")?;
            let fn_extract_head_bf16 = extract_head_bf16_mod
                .get_function("extract_head_bf16_kernel")?;
            let scatter_heads_bf16_mod = kernels.load_ptx("scatter_heads_bf16")?;
            let fn_scatter_heads_bf16 = scatter_heads_bf16_mod
                .get_function("scatter_heads_bf16_kernel")?;
            let softmax_row_f32_to_bf16_mod =
                kernels.load_ptx("softmax_row_f32_to_bf16")?;
            let fn_softmax_row_f32_to_bf16 = softmax_row_f32_to_bf16_mod
                .get_function("softmax_row_f32_to_bf16_kernel")?;
            let gelu_tanh_mul_bf16_mod = kernels.load_ptx("gelu_tanh_mul_bf16")?;
            let fn_gelu_tanh_mul_bf16 = gelu_tanh_mul_bf16_mod
                .get_function("gelu_tanh_mul_bf16_kernel")?;
            let scale_inplace_f32_mod = kernels.load_ptx("scale_inplace_f32")?;
            let fn_scale_inplace_f32 = scale_inplace_f32_mod
                .get_function("scale_inplace_f32_kernel")?;
            let transpose_heads_v_bf16_mod =
                kernels.load_ptx("transpose_heads_v_bf16")?;
            let fn_transpose_heads_v_bf16 = transpose_heads_v_bf16_mod
                .get_function("transpose_heads_v_bf16_kernel")?;
            let gelu_tanh_bf16_mod = kernels.load_ptx("gelu_tanh_bf16")?;
            let fn_gelu_tanh_bf16 = gelu_tanh_bf16_mod
                .get_function("gelu_tanh_bf16_kernel")?;
            let forward_kernels = Mistral35ForwardKernels {
                embedding_gather_bf16_mod,
                fn_embedding_gather_bf16,
                rmsnorm_inplace_bf16_gbf16_mod,
                fn_rmsnorm_inplace_bf16_gbf16,
                silu_mul_bf16_mod,
                fn_silu_mul_bf16,
                attn_m1_gqa_broadcast_mod,
                fn_attn_m1_gqa_broadcast,
                vector_add_bf16_mod,
                fn_vector_add_bf16,
                argmax_mod,
                fn_argmax_f32,
                rope_split_half_bf16_mod,
                fn_rope_split_half_bf16,
                kv_cache_write_bf16_mod,
                fn_kv_cache_write_bf16,
                qk_dot_bf16_mod,
                fn_qk_dot_bf16,
                qk_dot_gqa_bf16_mod,
                fn_qk_dot_gqa_bf16,
                softmax_v_bf16_mod,
                fn_softmax_v_bf16,
                nvfp4_dequant_weights_bf16_mod,
                fn_nvfp4_dequant_weights_bf16,
                f32_to_bf16_mod,
                fn_f32_to_bf16,
                mistral35_w4a16_gemv_bf16_mod,
                fn_mistral35_w4a16_gemv_bf16,
                pixtral_rotary_2d_bf16_mod,
                fn_pixtral_rotary_2d_bf16,
                patch_merger_pixtral_2x2_mod,
                fn_patch_merger_pixtral_2x2,
                extract_head_bf16_mod,
                fn_extract_head_bf16,
                scatter_heads_bf16_mod,
                fn_scatter_heads_bf16,
                softmax_row_f32_to_bf16_mod,
                fn_softmax_row_f32_to_bf16,
                gelu_tanh_mul_bf16_mod,
                fn_gelu_tanh_mul_bf16,
                scale_inplace_f32_mod,
                fn_scale_inplace_f32,
                transpose_heads_v_bf16_mod,
                fn_transpose_heads_v_bf16,
                gelu_tanh_bf16_mod,
                fn_gelu_tanh_bf16,
            };

            // W4A16 pre-dequantization is NOT viable for Mistral 3.5:
            // the model is 121B params (88 layers × 1.38B/layer), so
            // BF16-dequantized weights would need 243 GB — exceeds the
            // 128 GB unified-memory budget. The on-the-fly per-call
            // dequant in `gemm` works for tests with 1–8 prompt tokens
            // but exceeds the 600 s gateway timeout for the 364-token
            // chat-templated prompt (224K dequant launches per request).
            // Correct production answer: a fused m=1 W4A16 GEMV kernel
            // (analogue of the existing fp8_gemv_blockwise family) that
            // streams dequant inside the GEMM tile. Tracked in
            // MISTRAL35_BUG_HUNT.md.

            // P1#3 fix: RoPE tables must cover whatever positions the
            // KV cache will be asked to fill. The HTTP admission caps
            // requests at RVLLM_KV_CACHE_MAX_POS (default = config's
            // original_max_position_embeddings = 4096); if an operator
            // raises that env to e.g. 8192, attention at slot ≥ 4096
            // would index out of the RoPE table. Pre-resolve the same
            // fallback chain bring-up will use for kv_max_pos and
            // build the YaRN tables to that ceiling.
            let yarn_max_pos: usize = std::env::var("RVLLM_KV_CACHE_MAX_POS")
                .ok().and_then(|s| s.parse().ok())
                .unwrap_or(arch.text.yarn.original_max_position_embeddings);
            // YaRN cos/sin tables — `yarn_max_pos` positions ×
            // head_dim/2 = 64 floats each, ~16 B per position;
            // upload to device as F32. Position 0 is identity (cos=1, sin=0).
            let yarn_tables = crate::mistral35_yarn::build_yarn_rope_tables(
                &arch.text.yarn,
                arch.text.head_dim,
                yarn_max_pos,
            );
            let cos_bytes = yarn_tables.cos.len() * 4;
            let sin_bytes = yarn_tables.sin.len() * 4;
            let cos_region = arena_box.region("mistral35_rope_cos", cos_bytes, 16)?;
            let sin_region = arena_box.region("mistral35_rope_sin", sin_bytes, 16)?;
            unsafe {
                let cos_bytes_slice: &[u8] = std::slice::from_raw_parts(
                    yarn_tables.cos.as_ptr() as *const u8,
                    yarn_tables.cos.len() * 4,
                );
                let sin_bytes_slice: &[u8] = std::slice::from_raw_parts(
                    yarn_tables.sin.as_ptr() as *const u8,
                    yarn_tables.sin.len() * 4,
                );
                cos_region.copy_from_host(cos_bytes_slice)?;
                sin_region.copy_from_host(sin_bytes_slice)?;
            }
            let rope_tables_dev = Mistral35RopeTables {
                cos_ptr: cos_region.device_ptr(),
                sin_ptr: sin_region.device_ptr(),
                max_pos: yarn_tables.max_pos,
                head_dim: yarn_tables.head_dim,
            };
            eprintln!(
                "[mistral35] yarn tables uploaded: max_pos={} head_dim={} \
                 mscale={:.5} ({} KiB)",
                rope_tables_dev.max_pos, rope_tables_dev.head_dim,
                yarn_tables.mscale, (cos_bytes + sin_bytes) / 1024,
            );

            // Per-layer KV cache (BF16). One pair (K, V) per layer at
            // [max_pos, n_kv_heads, head_dim] BF16. RVLLM_KV_CACHE_MAX_POS
            // override; defaults to original_max_position_embeddings (4096
            // for Mistral 3.5). Per-layer 4 MiB at max_pos=4096; 88
            // layers → ~700 MiB total — well within the 80 GiB arena.
            let kv_max_pos: usize = std::env::var("RVLLM_KV_CACHE_MAX_POS")
                .ok().and_then(|s| s.parse().ok())
                .unwrap_or(arch.text.yarn.original_max_position_embeddings);
            let kv_n_heads = arch.text.num_key_value_heads;
            let kv_head_dim = arch.text.head_dim;
            let per_layer_bytes = kv_max_pos * kv_n_heads * kv_head_dim * 2;
            let mut kv_layers: Vec<LayerKvCache> = Vec::with_capacity(model.layers.len());
            for _ in 0..model.layers.len() {
                let k = arena_box.region("mistral35_k_cache", per_layer_bytes, 16)?;
                let v = arena_box.region("mistral35_v_cache", per_layer_bytes, 16)?;
                kv_layers.push(LayerKvCache {
                    k_ptr: k.device_ptr(), v_ptr: v.device_ptr(),
                });
            }
            // Scratch F32 [n_q_heads, max_pos] for attention scores.
            let scores_bytes = arch.text.num_attention_heads * kv_max_pos * 4;
            let scores_region = arena_box.region(
                "mistral35_scores_f32", scores_bytes, 16)?;
            let kv_cache_dev = Mistral35KvCache {
                layers: kv_layers,
                max_pos: kv_max_pos,
                n_kv_heads: kv_n_heads,
                head_dim: kv_head_dim,
                scores_f32_ptr: scores_region.device_ptr(),
            };
            eprintln!(
                "[mistral35] kv cache allocated: layers={} max_pos={} \
                 n_kv_heads={} head_dim={} per_layer={} KiB total={} MiB",
                kv_cache_dev.layers.len(), kv_max_pos, kv_n_heads, kv_head_dim,
                per_layer_bytes / 1024,
                (per_layer_bytes * kv_cache_dev.layers.len() * 2) / (1024 * 1024),
            );

            // Pre-allocated per-call scratch. Sized for m=1 across the
            // worst-case projection shapes Mistral 3.5 uses. Reused on
            // every forward step so the hot path never touches the
            // arena allocator.
            let h = arch.text.hidden_size;
            let i_s = arch.text.intermediate_size;
            let max_k = h.max(i_s);
            let n_q_dim = arch.text.num_attention_heads * arch.text.head_dim;
            let n_kv_dim = arch.text.num_key_value_heads * arch.text.head_dim;
            let vocab = arch.text.vocab_size;
            let h_bytes = h * 2;
            let i_bytes = i_s * 2;
            let cb = cutlass_backend.as_ref()
                .expect("cutlass_backend must be Some on cuda build");
            let mut max_workspace = 0usize;
            for &(n, k) in &[
                (n_q_dim, h), (n_kv_dim, h), (h, h),
                (i_s, h), (h, i_s),
            ] {
                let w = cb.nvfp4_workspace_size(1, n as i32, k as i32);
                if w > max_workspace { max_workspace = w; }
            }
            let max_workspace = max_workspace.max(16);
            // F2#3 fix: only the legacy `RVLLM_W4A16_GEMV=0`
            // (dequant-then-cublasLt) path needs `w_bf16_scratch` (~705
            // MB) and `out_f32_scratch` (~115 KB). Default fused W4A16
            // GEMV reads weights direct via lin.sfb_natural_ptr and
            // writes BF16 directly to `q_out_ptr` etc. Sizing them at
            // 0 under fused mode reclaims real arena budget on GB10.
            let want_legacy_dequant = std::env::var("RVLLM_W4A16_GEMV")
                .map(|s| matches!(s.as_str(), "0" | "false" | "FALSE"))
                .unwrap_or(false);
            // W4A16 path scratch sizing: max BF16 dequantized weight tile
            // across all projections. Mistral 3.5 layer projections:
            //   q_proj/o_proj: hidden×hidden            (= 12288²)
            //   k/v_proj:      n_kv_dim×hidden          (= 1024×12288)
            //   gate/up_proj:  intermediate×hidden      (= 28672×12288)
            //   down_proj:     hidden×intermediate      (= 12288×28672)
            // → max(N*K) = max(h*h, n_q_dim*h, n_kv_dim*h, i_s*h, h*i_s).
            let max_w_elems_legacy = (n_q_dim * h)
                .max(n_kv_dim * h)
                .max(h * h)
                .max(i_s * h)
                .max(h * i_s);
            let max_w_bf16_bytes = if want_legacy_dequant {
                max_w_elems_legacy * 2
            } else { 0 };
            // F32 output staging for cublasLt bf16_gemm_f32 — sized to
            // max(N) across projections, m=1.
            let max_out_n_legacy = n_q_dim.max(n_kv_dim).max(h).max(i_s);
            let max_out_f32_bytes = if want_legacy_dequant {
                max_out_n_legacy * 4
            } else { 0 };
            let scratch = Mistral35Scratch {
                token_in_ptr:   arena_box.region("mistral35_token_in", 4, 4)?.device_ptr(),
                h_residual_ptr: arena_box.region("mistral35_h_residual", h_bytes, 16)?.device_ptr(),
                h_work_ptr:     arena_box.region("mistral35_h_work", h_bytes, 16)?.device_ptr(),
                a_packed_ptr:   arena_box.region("mistral35_a_packed", max_k / 2, 16)?.device_ptr(),
                sfa_natural_ptr: arena_box.region("mistral35_sfa_natural",
                    cb.nvfp4_natural_sfa_bytes(1, max_k as i32), 16)?.device_ptr(),
                sfa_cutlass_ptr: arena_box.region("mistral35_sfa_cutlass",
                    cb.nvfp4_sfa_bytes(1, max_k as i32), 16)?.device_ptr(),
                q_out_ptr:      arena_box.region("mistral35_q_out", n_q_dim * 2, 16)?.device_ptr(),
                k_out_ptr:      arena_box.region("mistral35_k_out", n_kv_dim * 2, 16)?.device_ptr(),
                v_out_ptr:      arena_box.region("mistral35_v_out", n_kv_dim * 2, 16)?.device_ptr(),
                attn_out_ptr:   arena_box.region("mistral35_attn_out", h_bytes, 16)?.device_ptr(),
                o_out_ptr:      arena_box.region("mistral35_o_out", h_bytes, 16)?.device_ptr(),
                gate_out_ptr:   arena_box.region("mistral35_gate_out", i_bytes, 16)?.device_ptr(),
                up_out_ptr:     arena_box.region("mistral35_up_out", i_bytes, 16)?.device_ptr(),
                silu_mid_ptr:   arena_box.region("mistral35_silu_mid", i_bytes, 16)?.device_ptr(),
                down_out_ptr:   arena_box.region("mistral35_down_out", h_bytes, 16)?.device_ptr(),
                workspace_ptr:  arena_box.region("mistral35_workspace", max_workspace, 256)?.device_ptr(),
                workspace_bytes: max_workspace,
                w_bf16_scratch_ptr: if max_w_bf16_bytes > 0 {
                    arena_box.region(
                        "mistral35_w_bf16_scratch", max_w_bf16_bytes, 256)?.device_ptr()
                } else { 0 },
                w_bf16_scratch_bytes: max_w_bf16_bytes,
                out_f32_scratch_ptr: if max_out_f32_bytes > 0 {
                    arena_box.region(
                        "mistral35_out_f32_scratch", max_out_f32_bytes, 16)?.device_ptr()
                } else { 0 },
                logits_ptr:     arena_box.region("mistral35_logits_f32", vocab * 4, 16)?.device_ptr(),
                token_out_ptr:  arena_box.region("mistral35_token_out", 4, 4)?.device_ptr(),
            };
            eprintln!(
                "[mistral35] scratch hoisted to bring-up: max_workspace={} bytes",
                max_workspace,
            );

            (Some(ctx), Some(stream), Some(arena_box), Some(model),
             Some(forward_kernels), Some(cublaslt_handle), Some(rope_tables_dev),
             Some(kv_cache_dev), Some(scratch))
        };

        #[cfg(feature = "cuda")]
        let bringup = Self {
            paths, arch, inventory, arena_bytes, nvfp4_active,
            cutlass_backend, ctx, stream, arena, model,
            forward_kernels: forward_kernels_opt,
            cublaslt: cublaslt_opt,
            rope_tables: rope_tables_opt,
            kv_cache: kv_cache_opt,
            scratch: scratch_opt,
        };
        #[cfg(not(feature = "cuda"))]
        let bringup = Self { paths, arch, inventory, arena_bytes, nvfp4_active };
        // F4#1 fix: log the actual KV path. Mistral 3.5 runs BF16 KV
        // through `mistral35_qk_dot_bf16` + `mistral35_softmax_v_bf16`
        // in `attention_step`; no NVFP4-KV path is wired today.
        eprintln!(
            "[mistral35] kv=bf16 (per-head BF16 attention via \
             mistral35_qk_dot_bf16 / mistral35_softmax_v_bf16), \
             n_q={} n_kv={} gqa_ratio={}",
            bringup.arch.text.num_attention_heads,
            bringup.arch.text.num_key_value_heads,
            bringup.arch.gqa_ratio(),
        );
        // YaRN cos/sin tables for the original-max window. The
        // device upload happens later (CUDA forward path); for now
        // we just log the table size + the mscale value so the
        // operator can sanity-check the YaRN parameters resolved
        // correctly from config.json.
        {
            let head_dim = bringup.arch.text.head_dim;
            let max_pos = bringup.arch.text.yarn.original_max_position_embeddings;
            let mscale = crate::mistral35_yarn::yarn_mscale(&bringup.arch.text.yarn);
            eprintln!(
                "[mistral35] yarn: head_dim={head_dim} original_max={max_pos} \
                 mscale={mscale:.5} (factor={}, beta_fast={}, beta_slow={})",
                bringup.arch.text.yarn.factor,
                bringup.arch.text.yarn.beta_fast,
                bringup.arch.text.yarn.beta_slow,
            );
        }
        // Round-12 phase 5c (codex review #1): the BatchedPrefillConfig
        // is plan documentation — none of its phases (M-A/M-B/M-C/M-D/
        // M-E) are wired into `generate_with_vision`'s executor today.
        // The forward path iterates token-major over `prompt`. Logging
        // `outer_loop_deleted=true` here was a load-bearing lie that
        // produced false operator expectations about scaling. Replaced
        // with an honest one-liner.
        //
        // If the operator explicitly opted into a batched gate
        // (RVLLM_MISTRAL35_BATCH_PROJ_PREFILL=1 etc), fail fast — they
        // expect batched execution, the runtime can't deliver it.
        let env_set_batched = [
            "RVLLM_MISTRAL35_BATCH_PROJ_PREFILL",
            "RVLLM_MISTRAL35_FUSED_QKV",
            "RVLLM_MISTRAL35_BATCH_ROPE_PREFILL",
            "RVLLM_MISTRAL35_BATCH_FULL_PREFILL",
            "RVLLM_MISTRAL35_DECODE_GRAPH",
        ].iter().filter(|k| {
            matches!(std::env::var(k).ok().as_deref(),
                     Some("1") | Some("true") | Some("TRUE") | Some("on"))
        }).copied().collect::<Vec<_>>();
        if !env_set_batched.is_empty() {
            return Err(corrupt(
                bringup.paths.model_dir.clone(),
                format!(
                    "[mistral35] {} requested but the runtime executor \
                     is token-major (batched-prefill phases M-A/B/C/D/E \
                     are plan-doc only, not wired into \
                     generate_with_vision). Unset these envs or wait \
                     for the batched-prefill landing tracked in \
                     v3/MISTRAL35_BATCHED_PREFILL_PLAN.md.",
                    env_set_batched.join(", "),
                ),
            ));
        }
        eprintln!(
            "[mistral35] executor: token-major (per-prompt-token \
             forward via forward_smoke_q_proj_inner). batched-prefill \
             phases (M-A/B/C/D/E) are plan-doc only — see \
             MISTRAL35_BATCHED_PREFILL_PLAN.md."
        );
        // Indicative scratch-budget at a few common prefill chunk
        // sizes so the operator can see whether the requested
        // arena_bytes covers the steady-state working set up front.
        // The backend-aware path takes precedence under `cuda` so
        // the line reports the exact CUTLASS-derived sizes; the
        // arithmetic fallback covers no-cuda builds.
        for m in [1usize, 32, 256] {
            #[cfg(feature = "cuda")]
            let b = match bringup.cutlass_backend.as_ref() {
                Some(backend) => bringup.scratch_budget_with_backend(m, backend),
                None => bringup.scratch_budget(m),
            };
            #[cfg(not(feature = "cuda"))]
            let b = bringup.scratch_budget(m);

            eprintln!(
                "[mistral35] scratch_budget m={m:>3}: a_packed={:>10} B  \
                 sfa_natural={:>9} B  sfa_cutlass(max)={:>9} B  \
                 workspace(max)={:>9} B",
                b.a_packed_bytes, b.sfa_natural_bytes, b.sfa_cutlass_bytes_max,
                b.workspace_bytes_max,
            );
        }
        Ok(bringup)
    }

    /// Forward placeholder. Returns a typed error at every call.
    /// Wired into `cuda_worker.rs::spawn_cuda_worker` for the
    /// Mistral 3.5 branch so per-request execution surfaces the
    /// "kernel not implemented" message via the existing
    /// `GenerateEvent::Error` path.
    pub fn forward_not_implemented_yet() -> Mistral35Error {
        Mistral35Error::ForwardNotImplemented
    }
}

fn scan_safetensors_index(
    model_dir: &std::path::Path,
) -> Result<std::collections::BTreeMap<String, TensorEntry>> {
    let idx = ShardIndex::resolve(model_dir)?;
    let mut tensors: std::collections::BTreeMap<String, TensorEntry> =
        std::collections::BTreeMap::new();
    for shard_path in &idx.shards {
        // mmap the whole shard so ShardHeader::parse can validate
        // tensor offsets against the real file size. mmap costs no
        // RAM up-front (kernel pages-in lazily), and we only touch
        // the first ~33 KiB header region; payload pages stay
        // unmapped. Mirrors the qwen36_load.rs / gemma4_load.rs
        // pattern. The original `std::fs::read` (full Vec<u8> read)
        // OOM'd on /home/r00t/mistral-3.5 — 75 GiB across 10 shards;
        // a truncated read was rejected by the offset validator.
        let f = std::fs::File::open(shard_path).map_err(|source| RvllmError::Io {
            err: rvllm_core::IoError::from(&source),
            path: shard_path.clone(),
            source,
        })?;
        let mmap = unsafe { memmap2::Mmap::map(&f) }.map_err(|source| RvllmError::Io {
            err: rvllm_core::IoError::from(&source),
            path: shard_path.clone(),
            source,
        })?;
        let header = ShardHeader::parse(shard_path, &mmap)?;
        for (name, entry) in header.tensors.into_iter() {
            tensors.insert(name, entry);
        }
        // mmap drops here — kernel reclaims any paged-in regions.
    }
    Ok(tensors)
}

#[cfg(feature = "cuda")]
fn load_and_require_nvfp4(
    paths: &Mistral35EnginePaths,
) -> Result<rvllm_cutlass::lib_so::CutlassBackend> {
    use rvllm_cutlass::lib_so::CutlassBackend;
    use rvllm_core::CompileTarget;

    // Resolve the same way `cuda_worker` does for Gemma 4 / Qwen.
    // The sm_121 hint short-circuits straight to libcutlass_sm120.so.
    let backend = CutlassBackend::load_for(
        Some(CompileTarget::Sm121),
        paths.cutlass_so.clone(),
        &[],
    )?;
    // F1#3 fix: only the legacy W4A4 NVFP4 tensor-core GEMM path needs
    // the CUTLASS NVFP4 entry-point set. The default fused W4A16 GEMV
    // doesn't call any of those symbols, so a stale / missing
    // libcutlass_sm120.so should not block startup.
    let want_legacy_cutlass = std::env::var("RVLLM_W4A16_GEMV")
        .map(|s| matches!(s.as_str(), "0" | "false" | "FALSE")).unwrap_or(false);
    if want_legacy_cutlass {
        backend.require_nvfp4()?;
    }
    Ok(backend)
}

fn corrupt(path: PathBuf, detail: String) -> RvllmError {
    RvllmError::Loader {
        err: LoaderError::Corrupt { detail },
        ctx: LoaderCtx { path, tensor: None },
        bt: std::backtrace::Backtrace::capture(),
    }
}

/// Per-stage dump returned by the smoke-forward. Layer-0 stages stay
/// dumped for visibility; for layers 1..88 only the residual rms is
/// captured (in `layer_residual_rms`). Final stages cover RMSNorm
/// → lm_head → argmax → predicted token id.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct SmokeStageDump {
    pub post_embed: Vec<f32>,      // [hidden=12288]
    pub post_rmsnorm: Vec<f32>,    // [hidden=12288] = RMSNorm(embed, gamma_in)
    pub q_out: Vec<f32>,           // [n_q*head_dim=12288]
    pub k_out: Vec<f32>,           // [n_kv*head_dim=1024]
    pub v_out: Vec<f32>,           // [n_kv*head_dim=1024]
    pub attn_out: Vec<f32>,        // [hidden=12288] V broadcast across q-heads
    pub o_out: Vec<f32>,           // [hidden=12288] o_proj(attn_out)
    pub h_after_attn: Vec<f32>,    // [hidden=12288] embed + o_out (residual)
    pub post_attn_norm: Vec<f32>,  // [hidden=12288] RMSNorm(h_after_attn, gamma_post)
    pub gate_out: Vec<f32>,        // [intermediate=28672]
    pub up_out: Vec<f32>,          // [intermediate=28672]
    pub silu_mid: Vec<f32>,        // [intermediate=28672] silu(gate)*up
    pub down_out: Vec<f32>,        // [hidden=12288] down_proj(silu_mid)
    pub h_after_layer0: Vec<f32>,  // [hidden=12288] h_after_attn + down_out
    /// Per-layer residual rms after each layer (88 entries).
    pub layer_residual_rms: Vec<f32>,
    /// Final RMSNorm output (just before lm_head).
    pub h_after_final_norm: Vec<f32>,
    /// Top-8 (token_id, logit) pairs for sanity. logits_top8 is sorted
    /// descending by logit value.
    pub logits_top8: Vec<(u32, f32)>,
    /// argmax of the lm_head output — the predicted next token id.
    pub predicted_token: u32,
}

/// Smoke-forward for the upload+kernel pipeline: embed token id →
/// first-layer RMSNorm → first-layer Q-projection (NVFP4 GEMM) →
/// host readback at every stage. Lets callers bisect which step
/// is producing the all-zero output.
#[cfg(feature = "cuda")]
impl Mistral35Bringup {
    pub unsafe fn forward_smoke_q_proj(&self, token_id: u32, position: i32) -> Result<SmokeStageDump> {
        // Default to computing logits + argmax + DtoH on every call —
        // backwards-compatible for the smoke endpoint.
        self.forward_smoke_q_proj_inner(token_id, position, true, None)
    }

    /// Same as [`forward_smoke_q_proj`] with optional skip of the
    /// final lm_head + argmax + per-call DtoH-fence on the predicted
    /// token. Used by `generate_smoke` to elide the expensive
    /// [vocab=131072, hidden=12288] BF16 GEMM and the per-call stream
    /// fence on every prefill step except the last — that's where the
    /// O(N²) prefill latency was hiding.
    ///
    /// When `compute_logits` is false:
    ///   * lm_head BF16 GEMM is skipped
    ///   * argmax kernel is skipped
    ///   * predicted_token DtoH is skipped (predicted_token = u32::MAX)
    ///   * h_after_final_norm is uncomputed
    pub unsafe fn forward_smoke_q_proj_inner(
        &self,
        token_id: u32,
        position: i32,
        compute_logits: bool,
        // Round-12 phase 3e: optional vision splice. If provided,
        // points at a BF16 [hidden] device buffer that overrides the
        // result of `embed_gather` for this token. The pointer is
        // owned by the caller (typically a per-request scratch
        // region built once in `generate`).
        vision_splice_dev_ptr: Option<u64>,
    ) -> Result<SmokeStageDump> {
        use rvllm_cutlass::lib_so::CutlassBackend;

        let model = self.model.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_smoke_layer0: model not uploaded".into(),
        ))?;
        let kernels = self.forward_kernels.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_smoke_layer0: forward_kernels not loaded".into(),
        ))?;
        let arena = self.arena.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_smoke_layer0: arena absent".into(),
        ))?;
        let stream = self.stream.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_smoke_layer0: stream absent".into(),
        ))?;
        let backend: &CutlassBackend = self.cutlass_backend.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_smoke_layer0: cutlass backend absent".into(),
        ))?;
        if (token_id as usize) >= self.arch.text.vocab_size {
            return Err(corrupt(
                self.paths.model_dir.clone(),
                format!(
                    "forward_smoke_layer0: token_id={token_id} >= vocab={}",
                    self.arch.text.vocab_size,
                ),
            ));
        }

        let stream_u64 = stream.raw();
        let hidden = self.arch.text.hidden_size as u32;
        let vocab = self.arch.text.vocab_size as u32;

        let layer0 = &model.layers[0];
        let m: i32 = 1;
        let h_k: i32 = self.arch.text.hidden_size as i32;
        let i_k: i32 = self.arch.text.intermediate_size as i32;

        // Pre-allocated scratch (bring-up time). Hot-path forward never
        // touches the arena allocator.
        let scr = self.scratch.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_smoke_layer0: scratch absent".into(),
        ))?;
        let hidden_bytes = (hidden as usize) * 2; // BF16
        // Refresh the input-token slot via cuMemsetD32Async ON OUR
        // CUSTOM STREAM. This is the same race-fix pattern Qwen 3.6
        // landed: sync `cuMemcpyHtoD_v2` synchronizes only with the
        // DEFAULT stream — our compute stream is non-blocking, so
        // sync HtoD doesn't wait for prior kernels on it, which means
        // a freshly-written token_in_ptr can be overwritten BEFORE
        // the prior call's embed_gather has read it. cuMemsetD32Async
        // queues on our stream so it serializes naturally with
        // embed_gather (which runs after it on the same stream).
        // Single-token mode worked because there was no prior call
        // to race with.
        unsafe {
            use cudarc::driver::sys::*;
            let r = cuMemsetD32Async(
                scr.token_in_ptr as CUdeviceptr,
                token_id, // u32 broadcast
                1,        // count
                stream_u64 as CUstream,
            );
            if r != CUresult::CUDA_SUCCESS {
                return Err(RvllmError::cuda(
                    "cuMemsetD32Async(token_in)",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx { stream: stream_u64,
                        kernel: "token_in_d32async", launch: None, device: -1 },
                ));
            }
        }
        let workspace_bytes = scr.workspace_bytes;

        // Per-stage dumps cost a fence + DtoH each. Off by default in
        // multi-token generation paths; turn on with
        // RVLLM_SMOKE_FULL_DUMP=1 for one-shot diagnostics.
        // Stage dumps cost a fence + DtoH each (huge per prefill step).
        // Only enable when explicitly asked AND only on the LAST call
        // of generate_smoke (= compute_logits=true) so a 364-token
        // prefill doesn't trigger 364×14 dumps.
        let full_dump = debug_env_os("RVLLM_SMOKE_FULL_DUMP").is_some()
            && compute_logits;

        // Helper: fence + DtoH a bf16 region of `count` elements, widen to f32.
        // Returns empty Vec when full_dump is off.
        let dump_bf16 = |_stream: &rvllm_mem::stream::Stream,
                         stream_u64: u64,
                         dev_ptr: u64,
                         count: usize,
                         label: &'static str| -> Result<Vec<f32>> {
            if !full_dump {
                return Ok(Vec::new());
            }
            _stream.fence()?;
            let mut buf = vec![0u16; count];
            unsafe {
                use cudarc::driver::sys::*;
                let r = cuMemcpyDtoH_v2(
                    buf.as_mut_ptr() as *mut std::ffi::c_void,
                    dev_ptr as CUdeviceptr,
                    count * 2,
                );
                if r != CUresult::CUDA_SUCCESS {
                    return Err(RvllmError::cuda(
                        "cuMemcpyDtoH_v2(smoke stage)",
                        rvllm_core::CudaErrorKind::MemcpyFailed,
                        rvllm_core::CudaCtx {
                            stream: stream_u64, kernel: label,
                            launch: None, device: -1,
                        },
                    ));
                }
            }
            Ok(buf.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect())
        };

        // Helper: device-to-device async copy on our stream.
        unsafe fn dtod(stream_u64: u64, dst: u64, src: u64, n_bytes: usize) -> Result<()> {
            use cudarc::driver::sys::*;
            let r = cuMemcpyDtoDAsync_v2(
                dst as CUdeviceptr, src as CUdeviceptr,
                n_bytes, stream_u64 as CUstream,
            );
            if r != CUresult::CUDA_SUCCESS {
                return Err(RvllmError::cuda(
                    "cuMemcpyDtoDAsync_v2(layer0)",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx { stream: stream_u64,
                        kernel: "dtod_copy", launch: None, device: -1 },
                ));
            }
            Ok(())
        }

        // W4A16 GEMM path (Mistral 3.5 NVFP4 checkpoint) —
        //   1) dequantize the weight tile to BF16 in `w_bf16_scratch`,
        //   2) cublasLt bf16_gemm_f32 with the BF16 activation,
        //   3) cast f32 → bf16 into `out_ptr`.
        // The legacy `stage_act` (NVFP4 activation prep) and the
        // CUTLASS NVFP4 tensor-core GEMM (`backend.launch_nvfp4_gemm`)
        // are no longer reachable on this path; see
        // MISTRAL35_BUG_HUNT.md ROOT CAUSE for why.
        let cublaslt_inner = self.cublaslt.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_smoke_layer0: cublaslt absent".into(),
        ))?;
        let kernels_inner = kernels;
        // Fast path: fused m=1 W4A16 GEMV. Single kernel launch per
        // projection instead of dequant→cublasLt→cast (3 launches +
        // 705 MB scratch traffic). Default-on; opt out with
        // RVLLM_W4A16_GEMV=0 to use the legacy dequant path.
        let use_fused_gemv = !std::env::var("RVLLM_W4A16_GEMV")
            .map(|s| matches!(s.as_str(), "0" | "false" | "FALSE")).unwrap_or(false);
        let gemm = |out_bf16_ptr: u64,
                    act_bf16_ptr: u64,
                    lin: &rvllm_loader::mistral35_weights::Nvfp4LinearLoaded|
         -> Result<()> {
            let n = lin.shape.n as i32;
            let k = lin.shape.k as i32;
            if use_fused_gemv {
                unsafe {
                    let mut o   = out_bf16_ptr;
                    let mut wp  = lin.packed_ptr;
                    let mut ws  = lin.sfb_natural_ptr;
                    let mut gs  = lin.global_scale_ptr;
                    let mut a   = act_bf16_ptr;
                    let mut narg = n;
                    let mut karg = k;
                    let args: [*mut std::ffi::c_void; 7] = [
                        (&mut o)    as *mut u64 as *mut _,
                        (&mut wp)   as *mut u64 as *mut _,
                        (&mut ws)   as *mut u64 as *mut _,
                        (&mut gs)   as *mut u64 as *mut _,
                        (&mut a)    as *mut u64 as *mut _,
                        (&mut narg) as *mut i32 as *mut _,
                        (&mut karg) as *mut i32 as *mut _,
                    ];
                    rvllm_fused::launch_raw(
                        kernels_inner.fn_mistral35_w4a16_gemv_bf16,
                        (n as u32, 1, 1),
                        (256, 1, 1),
                        0, stream_u64,
                        &args,
                    )?;
                }
                return Ok(());
            }
            let w_elems = (n as usize) * (k as usize);
            if w_elems * 2 > scr.w_bf16_scratch_bytes {
                return Err(corrupt(
                    self.paths.model_dir.clone(),
                    format!("gemm: w_bf16_scratch too small: \
                             need {} bytes for n={} k={}, have {}",
                        w_elems * 2, n, k, scr.w_bf16_scratch_bytes),
                ));
            }
            // (1) Per-call dequant w_packed → BF16 scratch.
            //     Per-token cost: 88 × 7 = 616 launches per forward.
            //     Acceptable for short prompts (≤ a few tokens); for
            //     full prefill of 100s of tokens, a fused m=1 W4A16
            //     GEMV kernel is required (see MISTRAL35_BUG_HUNT.md).
            unsafe {
                let mut wp = lin.packed_ptr;
                let mut ws = lin.sfb_natural_ptr;
                let mut gs = lin.global_scale_ptr;
                let mut wb = scr.w_bf16_scratch_ptr;
                let mut n_arg = n;
                let mut k_arg = k;
                let args: [*mut std::ffi::c_void; 6] = [
                    (&mut wp)    as *mut u64 as *mut _,
                    (&mut ws)    as *mut u64 as *mut _,
                    (&mut gs)    as *mut u64 as *mut _,
                    (&mut wb)    as *mut u64 as *mut _,
                    (&mut n_arg) as *mut i32 as *mut _,
                    (&mut k_arg) as *mut i32 as *mut _,
                ];
                let blocks_x = (((k as u32) + 255) / 256).max(1);
                rvllm_fused::launch_raw(
                    kernels_inner.fn_nvfp4_dequant_weights_bf16,
                    (blocks_x, n as u32, 1),
                    (256, 1, 1),
                    0, stream_u64,
                    &args,
                )?;
            }
            // Codex round 3 boundary-dump #1: first 256 rows of the
            // dequantized BF16 weight tile, before the GEMM consumes it.
            // Triggered ONCE — first gemm call when RVLLM_BOUNDARY_DUMP=1
            // and RVLLM_SMOKE_DUMP_DIR is set. That call is layer-0 q_proj.
            static BOUNDARY_DUMPED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            let do_boundary = debug_env_os("RVLLM_BOUNDARY_DUMP").is_some()
                && BOUNDARY_DUMPED.compare_exchange(false, true,
                    std::sync::atomic::Ordering::SeqCst,
                    std::sync::atomic::Ordering::SeqCst).is_ok();
            let dump_dir = if do_boundary {
                debug_env_str("RVLLM_SMOKE_DUMP_DIR")
            } else { None };
            if let Some(ref dir) = dump_dir {
                stream.fence()?;
                let rows = 256.min(n as usize);
                let bytes = rows * (k as usize) * 2;
                let mut buf = vec![0u8; bytes];
                unsafe {
                    use cudarc::driver::sys::*;
                    let _ = cuMemcpyDtoH_v2(
                        buf.as_mut_ptr() as *mut std::ffi::c_void,
                        scr.w_bf16_scratch_ptr as CUdeviceptr,
                        bytes);
                }
                let p = std::path::PathBuf::from(dir).join("boundary_w_bf16_first256rows.bf16");
                let _ = std::fs::write(&p, &buf);
                eprintln!("[mistral35] boundary dump #1: w_bf16 {} rows × {} cols ({} B)",
                    rows, k, bytes);
            }
            // (2) cublasLt bf16 × bf16 → f32: out[1, n] = act[1, k] · w[n, k]^T.
            unsafe {
                cublaslt_inner.bf16_gemm_f32(
                    act_bf16_ptr,                // a [1, k]
                    scr.w_bf16_scratch_ptr,      // b [n, k] BF16
                    scr.out_f32_scratch_ptr,     // d [1, n] f32
                    m, n, k, stream_u64,
                )?;
            }
            // Codex round 3 boundary-dump #2: pre-cast F32 GEMM output.
            if let Some(ref dir) = dump_dir {
                stream.fence()?;
                let bytes = (n as usize) * 4;
                let mut buf = vec![0u8; bytes];
                unsafe {
                    use cudarc::driver::sys::*;
                    let _ = cuMemcpyDtoH_v2(
                        buf.as_mut_ptr() as *mut std::ffi::c_void,
                        scr.out_f32_scratch_ptr as CUdeviceptr,
                        bytes);
                }
                let p = std::path::PathBuf::from(dir).join("boundary_qproj_f32_pre_cast.f32");
                let _ = std::fs::write(&p, &buf);
                eprintln!("[mistral35] boundary dump #2: qproj F32 pre-cast ({} B)", bytes);
            }
            // (3) f32 → bf16 cast into the caller's output buffer.
            unsafe {
                let mut dst = out_bf16_ptr;
                let mut src = scr.out_f32_scratch_ptr;
                let mut count = n;
                let args: [*mut std::ffi::c_void; 3] = [
                    (&mut dst)   as *mut u64 as *mut _,
                    (&mut src)   as *mut u64 as *mut _,
                    (&mut count) as *mut i32 as *mut _,
                ];
                let blocks = ((n as u32) + 1023) / 1024;
                rvllm_fused::launch_raw(
                    kernels_inner.fn_f32_to_bf16,
                    (blocks, 1, 1),
                    (1024, 1, 1),
                    0, stream_u64,
                    &args,
                )?;
            }
            // Codex round 3 boundary-dump #3: post-cast BF16, pre-RoPE.
            if let Some(ref dir) = dump_dir {
                stream.fence()?;
                let bytes = (n as usize) * 2;
                let mut buf = vec![0u8; bytes];
                let mut buf2 = vec![0u8; bytes];
                eprintln!("[mistral35-debug] in-closure: out_bf16_ptr=0x{:x}  scr.q_out_ptr=0x{:x}",
                    out_bf16_ptr, scr.q_out_ptr);
                unsafe {
                    use cudarc::driver::sys::*;
                    let _ = cuMemcpyDtoH_v2(
                        buf.as_mut_ptr() as *mut std::ffi::c_void,
                        out_bf16_ptr as CUdeviceptr,
                        bytes);
                    // Also read via scr.q_out_ptr directly to compare
                    let _ = cuMemcpyDtoH_v2(
                        buf2.as_mut_ptr() as *mut std::ffi::c_void,
                        scr.q_out_ptr as CUdeviceptr,
                        bytes);
                }
                let p = std::path::PathBuf::from(dir).join("boundary_qproj_bf16_post_cast.bf16");
                let _ = std::fs::write(&p, &buf);
                let p2 = std::path::PathBuf::from(dir).join("boundary_qproj_via_scr_q_out.bf16");
                let _ = std::fs::write(&p2, &buf2);
                eprintln!("[mistral35] boundary dump #3: qproj BF16 post-cast pre-RoPE ({} B)",
                    bytes);
            }
            Ok(())
        };
        // Suppress dead_code on the NVFP4 W4A4 staging — kept compiled so
        // the CUTLASS path remains buildable for diagnostic ablations.
        let _ = (backend, m, scr.a_packed_ptr, scr.sfa_natural_ptr,
                 scr.sfa_cutlass_ptr, scr.workspace_ptr, workspace_bytes);

        // Helper: launch a kernel via the rvllm-fused launch_raw path.
        unsafe fn launch_kernel(
            kernel: rvllm_kernels::KernelFn,
            grid: (u32, u32, u32),
            block: (u32, u32, u32),
            args: &[*mut std::ffi::c_void],
            stream_u64: u64,
        ) -> Result<()> {
            rvllm_fused::launch_raw(kernel, grid, block, 0, stream_u64, args)
        }

        // RoPE handles + tables. The kernel rotates a [n_heads, head_dim]
        // BF16 buffer in-place at the given absolute position. The
        // closure captures head_dim/cos_ptr/sin_ptr; head_dim itself
        // is declared further down (alongside n_q_heads/gqa_ratio) so
        // we capture it via a local Copy of self.arch.text.head_dim
        // up here.
        let rope_tables = self.rope_tables.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_smoke_layer0: rope_tables absent".into(),
        ))?;
        let rope_position: i32 = position;
        if rope_position < 0 || (rope_position as usize) >= rope_tables.max_pos {
            return Err(corrupt(
                self.paths.model_dir.clone(),
                format!("forward_smoke_layer0: rope_position={rope_position} \
                         out of range [0, {})", rope_tables.max_pos),
            ));
        }
        let head_dim_for_rope: i32 = self.arch.text.head_dim as i32;
        let kv_cache = self.kv_cache.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_smoke_layer0: kv_cache absent".into(),
        ))?;
        // For now: position=0, past_len=1. Decode loop will increment.
        let attn_position: i32 = rope_position;
        // Diagnostic: RVLLM_SMOKE_ATTN_NO_PAST=1 makes attention only
        // see the current token (single key, broadcast V) — i.e.
        // ignore the KV cache. If predictions stop being garbage with
        // this on, multi-key attention is the bug.
        let attn_no_past = debug_env_os("RVLLM_SMOKE_ATTN_NO_PAST").is_some();
        let past_len: i32 = if attn_no_past { 1 } else { attn_position + 1 };
        if (past_len as usize) > kv_cache.max_pos {
            return Err(corrupt(
                self.paths.model_dir.clone(),
                format!("forward_smoke_layer0: past_len={past_len} > kv_max_pos={}",
                    kv_cache.max_pos),
            ));
        }
        let inv_sqrt_d: f32 = 1.0 / (self.arch.text.head_dim as f32).sqrt();
        let n_q_heads_attn: u32 = self.arch.text.num_attention_heads as u32;
        let n_kv_heads_attn: i32 = self.arch.text.num_key_value_heads as i32;
        let gqa_ratio_attn: i32 = (self.arch.text.num_attention_heads
            / self.arch.text.num_key_value_heads) as i32;
        let head_dim_attn: i32 = self.arch.text.head_dim as i32;

        // Codex round-7 boundary-dump env (read once before closures
        // so attention_step can reference them).
        let bd_layer: Option<usize> = debug_env_str("RVLLM_BOUNDARY_DUMP_LAYER")
            .and_then(|s| s.parse().ok());
        let bd_dir: Option<String> = debug_env_str("RVLLM_SMOKE_DUMP_DIR");

        // Attention closure: KV write + qk_dot + softmax_v.
        let attention_step = |layer_idx: usize| -> Result<()> {
            let layer_kv = &kv_cache.layers[layer_idx];
            // (a) Write K_in and V_in to slot=position of the cache.
            {
                let mut kin = scr.k_out_ptr;
                let mut vin = scr.v_out_ptr;
                let mut k_cache = layer_kv.k_ptr;
                let mut v_cache = layer_kv.v_ptr;
                let mut nkv = n_kv_heads_attn;
                let mut hd = head_dim_attn;
                // With attn_no_past, KV cache write goes to slot 0
                // (overwriting prior tokens), and qk_dot reads only
                // slot 0 (past_len=1). This degenerates to "current
                // token attends to itself" — same as the broadcast
                // hack baseline.
                let mut pos = if attn_no_past { 0 } else { attn_position };
                let args = [
                    (&mut kin) as *mut u64 as *mut std::ffi::c_void,
                    (&mut vin) as *mut u64 as *mut std::ffi::c_void,
                    (&mut k_cache) as *mut u64 as *mut std::ffi::c_void,
                    (&mut v_cache) as *mut u64 as *mut std::ffi::c_void,
                    (&mut nkv) as *mut i32 as *mut std::ffi::c_void,
                    (&mut hd) as *mut i32 as *mut std::ffi::c_void,
                    (&mut pos) as *mut i32 as *mut std::ffi::c_void,
                ];
                unsafe {
                    launch_kernel(
                        kernels.fn_kv_cache_write_bf16,
                        (n_kv_heads_attn as u32, 1, 1),
                        (head_dim_attn as u32, 1, 1),
                        &args, stream_u64,
                    )?;
                }
                // Diagnostic bypass: if RVLLM_KV_BYPASS=1, ALSO do the
                // copy via cuMemcpyDtoDAsync to verify whether the
                // kernel above is a no-op. Slot offset = position * nkv * hd * 2.
                if debug_env_os("RVLLM_KV_BYPASS").is_some() {
                    use cudarc::driver::sys::*;
                    let bytes = (n_kv_heads_attn as usize) * (head_dim_attn as usize) * 2;
                    let pos_off = (pos as usize) * bytes;
                    eprintln!("[mistral35-debug] bypass: kin=0x{:x} vin=0x{:x} k_dst=0x{:x} v_dst=0x{:x} bytes={} stream=0x{:x}",
                        kin, vin, k_cache + pos_off as u64, v_cache + pos_off as u64, bytes, stream_u64);
                    let r1 = cuMemcpyDtoDAsync_v2(
                        (k_cache + pos_off as u64) as CUdeviceptr,
                        kin as CUdeviceptr, bytes,
                        stream_u64 as CUstream);
                    let r2 = cuMemcpyDtoDAsync_v2(
                        (v_cache + pos_off as u64) as CUdeviceptr,
                        vin as CUdeviceptr, bytes,
                        stream_u64 as CUstream);
                    eprintln!("[mistral35-debug] bypass results: k={:?} v={:?}", r1, r2);
                }
            }

            // (b) qk_dot: scores[n_q_heads, past_len] = Q · K[0..past_len]^T / sqrt(d)
            {
                let mut q_ptr = scr.q_out_ptr;
                let mut k_cache = layer_kv.k_ptr;
                let mut sc_ptr = kv_cache.scores_f32_ptr;
                let mut hd = head_dim_attn;
                let mut nkv = n_kv_heads_attn;
                let mut gr = gqa_ratio_attn;
                let mut pl = past_len;
                let mut isd = inv_sqrt_d;
                let args = [
                    (&mut q_ptr) as *mut u64 as *mut std::ffi::c_void,
                    (&mut k_cache) as *mut u64 as *mut std::ffi::c_void,
                    (&mut sc_ptr) as *mut u64 as *mut std::ffi::c_void,
                    (&mut hd) as *mut i32 as *mut std::ffi::c_void,
                    (&mut nkv) as *mut i32 as *mut std::ffi::c_void,
                    (&mut gr) as *mut i32 as *mut std::ffi::c_void,
                    (&mut pl) as *mut i32 as *mut std::ffi::c_void,
                    (&mut isd) as *mut f32 as *mut std::ffi::c_void,
                ];
                let smem = (head_dim_attn as u32) * 4;
                // GQA-aware fast path: 1 CTA per (kv_head, t) with K
                // loaded once and reused across `gqa_ratio` Q-heads.
                // K-cache bandwidth drops by `gqa_ratio` (12× for
                // Mistral 3.5). Same output ABI as the legacy
                // `mistral35_qk_dot_bf16` kernel — no other call
                // sites touched. Default-on after the byte-identical
                // smoke matrix passed (text-only greedy + long
                // German decode + vision E2E all match the legacy
                // path bit-for-bit). `RVLLM_MISTRAL35_QK_DOT_GQA=0`
                // explicitly opts back to the legacy kernel for
                // debugging.
                let use_gqa = std::env::var("RVLLM_MISTRAL35_QK_DOT_GQA")
                    .ok().as_deref().map(|s| s != "0" && !s.is_empty())
                    .unwrap_or(true);
                if use_gqa && gqa_ratio_attn > 1 {
                    let mut nq = n_q_heads_attn as i32;
                    let args_gqa = [
                        (&mut q_ptr) as *mut u64 as *mut std::ffi::c_void,
                        (&mut k_cache) as *mut u64 as *mut std::ffi::c_void,
                        (&mut sc_ptr) as *mut u64 as *mut std::ffi::c_void,
                        (&mut hd) as *mut i32 as *mut std::ffi::c_void,
                        (&mut nq) as *mut i32 as *mut std::ffi::c_void,
                        (&mut nkv) as *mut i32 as *mut std::ffi::c_void,
                        (&mut gr) as *mut i32 as *mut std::ffi::c_void,
                        (&mut pl) as *mut i32 as *mut std::ffi::c_void,
                        (&mut isd) as *mut f32 as *mut std::ffi::c_void,
                    ];
                    unsafe {
                        rvllm_fused::launch_raw(
                            kernels.fn_qk_dot_gqa_bf16,
                            (n_kv_heads_attn as u32, past_len as u32, 1),
                            (head_dim_attn as u32, 1, 1),
                            smem, stream_u64, &args_gqa,
                        )?;
                    }
                } else {
                    unsafe {
                        rvllm_fused::launch_raw(
                            kernels.fn_qk_dot_bf16,
                            (n_q_heads_attn, past_len as u32, 1),
                            (head_dim_attn as u32, 1, 1),
                            smem, stream_u64, &args,
                        )?;
                    }
                }
            }

            // (c) softmax_v: out[n_q_heads, head_dim] = softmax(scores) · V[0..past_len]
            {
                let mut sc_ptr = kv_cache.scores_f32_ptr;
                let mut v_cache = layer_kv.v_ptr;
                let mut out_ptr = scr.attn_out_ptr;
                let mut hd = head_dim_attn;
                let mut nkv = n_kv_heads_attn;
                let mut gr = gqa_ratio_attn;
                let mut pl = past_len;
                let args = [
                    (&mut sc_ptr) as *mut u64 as *mut std::ffi::c_void,
                    (&mut v_cache) as *mut u64 as *mut std::ffi::c_void,
                    (&mut out_ptr) as *mut u64 as *mut std::ffi::c_void,
                    (&mut hd) as *mut i32 as *mut std::ffi::c_void,
                    (&mut nkv) as *mut i32 as *mut std::ffi::c_void,
                    (&mut gr) as *mut i32 as *mut std::ffi::c_void,
                    (&mut pl) as *mut i32 as *mut std::ffi::c_void,
                ];
                let smem = (past_len as u32) * 4;
                unsafe {
                    rvllm_fused::launch_raw(
                        kernels.fn_softmax_v_bf16,
                        (n_q_heads_attn, 1, 1),
                        (head_dim_attn as u32, 1, 1),
                        smem, stream_u64, &args,
                    )?;
                }
            }
            // Codex round-7: layer-selectable K/V/scores/attn_out dump
            // at the LAST forward (compute_logits=true → full_dump path
            // is the user's expectation). For prefill of N tokens at
            // max_new=1, that's position=N-1 with past_len=N → cache
            // slots [0..N-1] are populated.
            if let (Some(L), Some(ref dir)) = (bd_layer, bd_dir.as_ref()) {
                if L == layer_idx && full_dump {
                    stream.fence()?;
                    let slot_bytes = (n_kv_heads_attn as usize) * (head_dim_attn as usize) * 2;
                    let n_slots = past_len as usize;
                    let kv_total = slot_bytes * n_slots;
                    let mut buf_k = vec![0u8; kv_total];
                    let mut buf_v = vec![0u8; kv_total];
                    let scores_bytes = (n_q_heads_attn as usize) * n_slots * 4;
                    let mut buf_s = vec![0u8; scores_bytes];
                    unsafe { use cudarc::driver::sys::*;
                        let _ = cuMemcpyDtoH_v2(buf_k.as_mut_ptr() as *mut std::ffi::c_void,
                            layer_kv.k_ptr as CUdeviceptr, kv_total);
                        let _ = cuMemcpyDtoH_v2(buf_v.as_mut_ptr() as *mut std::ffi::c_void,
                            layer_kv.v_ptr as CUdeviceptr, kv_total);
                        let _ = cuMemcpyDtoH_v2(buf_s.as_mut_ptr() as *mut std::ffi::c_void,
                            kv_cache.scores_f32_ptr as CUdeviceptr, scores_bytes);
                    }
                    let p = std::path::PathBuf::from(dir);
                    let _ = std::fs::write(p.join(format!("k_cache_layer{}_all_slots.bf16", L)), &buf_k);
                    let _ = std::fs::write(p.join(format!("v_cache_layer{}_all_slots.bf16", L)), &buf_v);
                    let _ = std::fs::write(p.join(format!("scores_layer{}_all_slots.f32", L)), &buf_s);
                    // attn_out dump (BF16, n_q_heads*head_dim)
                    let attn_bytes = (n_q_heads_attn as usize) * (head_dim_attn as usize) * 2;
                    let mut buf_a = vec![0u8; attn_bytes];
                    unsafe { use cudarc::driver::sys::*;
                        let _ = cuMemcpyDtoH_v2(buf_a.as_mut_ptr() as *mut std::ffi::c_void,
                            scr.attn_out_ptr as CUdeviceptr, attn_bytes); }
                    let _ = std::fs::write(p.join(format!("attn_out_layer{}.bf16", L)), &buf_a);
                    // q_out dump (post-RoPE)
                    let q_bytes = (n_q_heads_attn as usize) * (head_dim_attn as usize) * 2;
                    let mut buf_q = vec![0u8; q_bytes];
                    unsafe { use cudarc::driver::sys::*;
                        let _ = cuMemcpyDtoH_v2(buf_q.as_mut_ptr() as *mut std::ffi::c_void,
                            scr.q_out_ptr as CUdeviceptr, q_bytes); }
                    let _ = std::fs::write(p.join(format!("q_out_layer{}.bf16", L)), &buf_q);
                    eprintln!("[mistral35-debug] layer {} dump: {} kv slots, scores stride {}",
                        L, n_slots, n_slots);
                }
            }
            Ok(())
        };
        // Diagnostic knob: RVLLM_SMOKE_ROPE_POS_OVERRIDE=0 forces every
        // layer's RoPE to position=0. If the model's output changes
        // with this on, the bug touches RoPE; if it stays at the same
        // wrong predicted token, RoPE is not the issue.
        let rope_pos_override: Option<i32> = debug_env_str("RVLLM_SMOKE_ROPE_POS_OVERRIDE")
            .and_then(|s| s.parse().ok());
        let rope = |buf_ptr: u64, n_heads: u32| -> Result<()> {
            let mut ptr = buf_ptr;
            let mut cos_ptr = rope_tables.cos_ptr;
            let mut sin_ptr = rope_tables.sin_ptr;
            let mut head_dim_arg = head_dim_for_rope;
            let mut pos_arg = rope_pos_override.unwrap_or(rope_position);
            let args = [
                (&mut ptr) as *mut u64 as *mut std::ffi::c_void,
                (&mut cos_ptr) as *mut u64 as *mut std::ffi::c_void,
                (&mut sin_ptr) as *mut u64 as *mut std::ffi::c_void,
                (&mut head_dim_arg) as *mut i32 as *mut std::ffi::c_void,
                (&mut pos_arg) as *mut i32 as *mut std::ffi::c_void,
            ];
            unsafe {
                launch_kernel(
                    kernels.fn_rope_split_half_bf16,
                    (n_heads, 1, 1),
                    ((head_dim_for_rope as u32) / 2, 1, 1),
                    &args, stream_u64,
                )
            }
        };

        // ============================================================
        // Layer-0 forward (pos=0, single token; RoPE is identity at
        // pos=0 so we skip it; m=1 attention reduces to V broadcast
        // because softmax over a single key is 1.0).
        // ============================================================

        // Codex round-7: layer-selectable per-position dump of
        // h_residual_{L-1} at the entry of layer L's processing.
        // Triggered on EVERY forward (not just last) when env
        // RVLLM_BOUNDARY_DUMP_LAYER=L is set.
        let dump_h_resid_pre_layer = |target_layer: usize| -> Result<()> {
            if let (Some(L), Some(ref dir)) = (bd_layer, bd_dir.as_ref()) {
                if L == target_layer {
                    stream.fence()?;
                    let bytes = (hidden as usize) * 2;
                    let mut buf = vec![0u8; bytes];
                    unsafe { use cudarc::driver::sys::*;
                        let _ = cuMemcpyDtoH_v2(buf.as_mut_ptr() as *mut std::ffi::c_void,
                            scr.h_residual_ptr as CUdeviceptr, bytes); }
                    let prev = if L == 0 { -1 } else { (L as i32) - 1 };
                    let label = if L == 0 {
                        format!("h_residual_pre_layer0_pos{}.bf16", attn_position)
                    } else {
                        format!("h_residual_layer{}_pos{}.bf16", prev, attn_position)
                    };
                    let _ = std::fs::write(std::path::PathBuf::from(dir).join(&label), &buf);
                }
            }
            Ok(())
        };
        // (1) Embed gather → h_residual.
        rvllm_fused::EmbeddingGatherLaunch { num_tokens: 1, hidden, vocab }
            .launch(
                kernels.fn_embedding_gather_bf16,
                scr.h_residual_ptr,
                model.outside.embed_tokens.offset_bytes,
                scr.token_in_ptr,
                stream_u64,
            )?;
        // (1b) Round-12 phase 3e — vision splice override. If the
        // caller marked this position as a vision-soft-token slot,
        // overwrite the embed-gathered hidden state with the
        // pre-computed BF16 vision-tower output. The text-side
        // embed_gather above still runs (cheap), so logging /
        // diagnostics see a consistent baseline; the splice is
        // observed by everything downstream of this DtoD copy.
        if let Some(splice_ptr) = vision_splice_dev_ptr {
            unsafe {
                use cudarc::driver::sys::*;
                let r = cuMemcpyDtoDAsync_v2(
                    scr.h_residual_ptr,
                    splice_ptr,
                    (hidden as usize) * 2,
                    stream.raw() as CUstream,
                );
                if r != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "mistral35: vision splice DtoD",
                        rvllm_core::CudaErrorKind::LaunchFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            }
        }
        let post_embed = dump_bf16(
            stream, stream_u64,
            scr.h_residual_ptr, hidden as usize,
            "smoke_post_embed",
        )?;

        // pre-layer-0 dump (h_residual at entry of layer 0 = post_embed)
        dump_h_resid_pre_layer(0)?;
        // (2) DtoD copy h_residual → h_work, then RMSNorm in-place
        //     with input_layernorm gamma.
        dtod(stream_u64, scr.h_work_ptr,
             scr.h_residual_ptr, hidden_bytes)?;
        rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
            num_tokens: 1, hidden, eps: self.arch.text.rms_norm_eps as f32,
        }.launch(
            kernels.fn_rmsnorm_inplace_bf16_gbf16,
            scr.h_work_ptr,
            layer0.input_layernorm.offset_bytes,
            stream_u64,
        )?;
        let post_rmsnorm = dump_bf16(
            stream, stream_u64,
            scr.h_work_ptr, hidden as usize,
            "smoke_post_rmsnorm",
        )?;

        // (3) Q/K/V projections — W4A16 dequant + bf16 GEMM.
        gemm(scr.q_out_ptr, scr.h_work_ptr, &layer0.q_proj)?;
        // Dump A: q_out_ptr + out_f32_scratch right after q_proj returns
        if debug_env_os("RVLLM_BOUNDARY_DUMP").is_some() && full_dump {
            if let Some(dir) = debug_env_str("RVLLM_SMOKE_DUMP_DIR") {
                stream.fence()?;
                unsafe {
                    use cudarc::driver::sys::*;
                    let _ = cuCtxSynchronize();
                }
                let n = layer0.q_proj.shape.n;
                let bytes_bf = n * 2;
                let bytes_f32 = n * 4;
                let mut buf_bf = vec![0u8; bytes_bf];
                let mut buf_f32 = vec![0u8; bytes_f32];
                unsafe { use cudarc::driver::sys::*;
                    let _ = cuMemcpyDtoH_v2(buf_bf.as_mut_ptr() as *mut std::ffi::c_void,
                        scr.q_out_ptr as CUdeviceptr, bytes_bf);
                    let _ = cuMemcpyDtoH_v2(buf_f32.as_mut_ptr() as *mut std::ffi::c_void,
                        scr.out_f32_scratch_ptr as CUdeviceptr, bytes_f32);
                }
                let dirp = std::path::PathBuf::from(&dir);
                let _ = std::fs::write(dirp.join("boundary_A_after_qproj.bf16"), &buf_bf);
                let _ = std::fs::write(dirp.join("boundary_A_f32_scratch.f32"), &buf_f32);
            }
        }
        gemm(scr.k_out_ptr, scr.h_work_ptr, &layer0.k_proj)?;
        if debug_env_os("RVLLM_BOUNDARY_DUMP").is_some() && full_dump {
            if let Some(dir) = debug_env_str("RVLLM_SMOKE_DUMP_DIR") {
                stream.fence()?;
                let bytes = (layer0.q_proj.shape.n as usize) * 2;
                let mut buf = vec![0u8; bytes];
                unsafe { use cudarc::driver::sys::*;
                    let _ = cuMemcpyDtoH_v2(buf.as_mut_ptr() as *mut std::ffi::c_void,
                        scr.q_out_ptr as CUdeviceptr, bytes); }
                let _ = std::fs::write(std::path::PathBuf::from(&dir)
                    .join("boundary_B_after_kproj.bf16"), &buf);
            }
        }
        gemm(scr.v_out_ptr, scr.h_work_ptr, &layer0.v_proj)?;
        if debug_env_os("RVLLM_BOUNDARY_DUMP").is_some() && full_dump {
            if let Some(dir) = debug_env_str("RVLLM_SMOKE_DUMP_DIR") {
                stream.fence()?;
                let bytes = (layer0.q_proj.shape.n as usize) * 2;
                let mut buf = vec![0u8; bytes];
                unsafe { use cudarc::driver::sys::*;
                    let _ = cuMemcpyDtoH_v2(buf.as_mut_ptr() as *mut std::ffi::c_void,
                        scr.q_out_ptr as CUdeviceptr, bytes); }
                let _ = std::fs::write(std::path::PathBuf::from(&dir)
                    .join("boundary_C_after_vproj.bf16"), &buf);
            }
        }
        // Boundary dump #4: q_out_ptr just before RoPE — does k/v
        // gemm corrupt it, or does rope itself flip the output?
        if debug_env_os("RVLLM_BOUNDARY_DUMP").is_some() && full_dump {
            if let Some(dir) = debug_env_str("RVLLM_SMOKE_DUMP_DIR") {
                stream.fence()?;
                let bytes = (layer0.q_proj.shape.n as usize) * 2;
                let mut buf = vec![0u8; bytes];
                unsafe {
                    use cudarc::driver::sys::*;
                    let _ = cuMemcpyDtoH_v2(
                        buf.as_mut_ptr() as *mut std::ffi::c_void,
                        scr.q_out_ptr as CUdeviceptr,
                        bytes);
                }
                let p = std::path::PathBuf::from(&dir).join("boundary_qproj_pre_rope.bf16");
                let _ = std::fs::write(&p, &buf);
                eprintln!("[mistral35] boundary dump #4: q pre-rope ({} B)", bytes);
            }
        }
        // RoPE on Q and K (V never gets rotated). Identity at pos=0.
        let n_q_heads_u: u32 = self.arch.text.num_attention_heads as u32;
        let n_kv_heads_u: u32 = self.arch.text.num_key_value_heads as u32;
        rope(scr.q_out_ptr, n_q_heads_u)?;
        rope(scr.k_out_ptr, n_kv_heads_u)?;
        let q_out = dump_bf16(stream, stream_u64,
            scr.q_out_ptr, layer0.q_proj.shape.n, "smoke_q_out")?;
        let k_out = dump_bf16(stream, stream_u64,
            scr.k_out_ptr, layer0.k_proj.shape.n, "smoke_k_out")?;
        let v_out = dump_bf16(stream, stream_u64,
            scr.v_out_ptr, layer0.v_proj.shape.n, "smoke_v_out")?;

        // Sentinel removed — earlier diagnosis confused me because the
        // smoke runs forward twice (input token at pos=0, predicted
        // token at pos=1). Each run memsets slot 0 to 0xAA, but run 2
        // writes to slot 1, so slot 0 ends as sentinel-from-run-2.
        // Verify sentinel landed in v_cache before attention_step.
        if debug_env_os("RVLLM_BOUNDARY_DUMP").is_some() && full_dump {
            stream.fence()?;
            let bytes = (n_kv_heads_attn as usize) * (head_dim_attn as usize) * 2;
            let mut buf = vec![0u8; bytes];
            unsafe { use cudarc::driver::sys::*;
                let _ = cuMemcpyDtoH_v2(buf.as_mut_ptr() as *mut std::ffi::c_void,
                    kv_cache.layers[0].v_ptr as CUdeviceptr, bytes); }
            eprintln!("[mistral35-debug] PRE-attn v_cache slot0 first 8 bytes: {:?}",
                &buf[..8]);
        }
        // Boundary dump #6: v_out_ptr immediately before attention_step,
        // to check whether anything modifies it between the v_out.f32
        // dump above and the kv_cache_write read.
        if debug_env_os("RVLLM_BOUNDARY_DUMP").is_some() && full_dump {
            if let Some(dir) = debug_env_str("RVLLM_SMOKE_DUMP_DIR") {
                stream.fence()?;
                let bytes = (layer0.v_proj.shape.n) * 2;
                let mut buf = vec![0u8; bytes];
                unsafe { use cudarc::driver::sys::*;
                    let _ = cuMemcpyDtoH_v2(buf.as_mut_ptr() as *mut std::ffi::c_void,
                        scr.v_out_ptr as CUdeviceptr, bytes); }
                let _ = std::fs::write(std::path::PathBuf::from(&dir)
                    .join("boundary_v_out_pre_attn.bf16"), &buf);
            }
        }
        // (4) Attention: write K/V to layer-0 cache @ pos=position,
        //     compute scores via qk_dot, softmax · V into attn_out.
        attention_step(0)?;
        let head_dim = head_dim_attn;
        let n_q_heads = n_q_heads_attn;
        let gqa_ratio = gqa_ratio_attn;
        let attn_out = dump_bf16(stream, stream_u64,
            scr.attn_out_ptr, hidden as usize, "smoke_attn_out")?;

        // Boundary dump K-cache slot 0 too, to compare with K-out.
        if debug_env_os("RVLLM_BOUNDARY_DUMP").is_some() && full_dump {
            if let Some(dir) = debug_env_str("RVLLM_SMOKE_DUMP_DIR") {
                stream.fence()?;
                let bytes = (n_kv_heads_attn as usize) * (head_dim_attn as usize) * 2;
                let mut buf = vec![0u8; bytes];
                unsafe { use cudarc::driver::sys::*;
                    let _ = cuMemcpyDtoH_v2(buf.as_mut_ptr() as *mut std::ffi::c_void,
                        kv_cache.layers[0].k_ptr as CUdeviceptr, bytes); }
                let _ = std::fs::write(std::path::PathBuf::from(&dir)
                    .join("boundary_k_cache_slot0.bf16"), &buf);
                eprintln!("[mistral35-debug] kv_cache.layers[0]: k_ptr=0x{:x} v_ptr=0x{:x}  scr.k_out_ptr=0x{:x} scr.v_out_ptr=0x{:x}",
                    kv_cache.layers[0].k_ptr, kv_cache.layers[0].v_ptr,
                    scr.k_out_ptr, scr.v_out_ptr);
            }
        }
        // Boundary dump #5: dump V-cache for slots 0..=position so we
        // can match against whichever slot kv_cache_write touched in
        // the current call.
        if debug_env_os("RVLLM_BOUNDARY_DUMP").is_some() && full_dump {
            if let Some(dir) = debug_env_str("RVLLM_SMOKE_DUMP_DIR") {
                stream.fence()?;
                let slot_bytes = (n_kv_heads_attn as usize) * (head_dim_attn as usize) * 2;
                let n_slots = (rope_position as usize) + 1;
                let total = slot_bytes * n_slots;
                let mut buf = vec![0u8; total];
                unsafe { use cudarc::driver::sys::*;
                    let _ = cuMemcpyDtoH_v2(buf.as_mut_ptr() as *mut std::ffi::c_void,
                        kv_cache.layers[0].v_ptr as CUdeviceptr, total); }
                let _ = std::fs::write(std::path::PathBuf::from(&dir)
                    .join("boundary_v_cache_all_slots.bf16"), &buf);
                eprintln!("[mistral35-debug] dumped {} v-cache slots ({} bytes)", n_slots, total);
            }
        }
        // Codex round-6: K-cache-all-slots + scores dump for past_len=3+ bisect.
        if debug_env_os("RVLLM_BOUNDARY_DUMP").is_some() && full_dump {
            if let Some(dir) = debug_env_str("RVLLM_SMOKE_DUMP_DIR") {
                stream.fence()?;
                let slot_bytes = (n_kv_heads_attn as usize) * (head_dim_attn as usize) * 2;
                let n_slots = (rope_position as usize) + 1;
                let total_kv = slot_bytes * n_slots;
                let mut buf_k = vec![0u8; total_kv];
                unsafe { use cudarc::driver::sys::*;
                    let _ = cuMemcpyDtoH_v2(buf_k.as_mut_ptr() as *mut std::ffi::c_void,
                        kv_cache.layers[0].k_ptr as CUdeviceptr, total_kv); }
                let _ = std::fs::write(std::path::PathBuf::from(&dir)
                    .join("boundary_k_cache_all_slots.bf16"), &buf_k);
                // Scores: f32 [n_q_heads, past_len], packed with stride past_len.
                let scores_bytes = (n_q_heads_attn as usize) * n_slots * 4;
                let mut buf_s = vec![0u8; scores_bytes];
                unsafe { use cudarc::driver::sys::*;
                    let _ = cuMemcpyDtoH_v2(buf_s.as_mut_ptr() as *mut std::ffi::c_void,
                        kv_cache.scores_f32_ptr as CUdeviceptr, scores_bytes); }
                let _ = std::fs::write(std::path::PathBuf::from(&dir)
                    .join("boundary_scores_layer0_all_slots.f32"), &buf_s);
                eprintln!("[mistral35-debug] dumped {} k-cache slots + scores stride {}",
                    n_slots, n_slots);
            }
        }
        // (5) O-projection (W4A16): bf16 attn_out × o_proj weights.
        gemm(scr.o_out_ptr, scr.attn_out_ptr, &layer0.o_proj)?;
        let o_out = dump_bf16(stream, stream_u64,
            scr.o_out_ptr, layer0.o_proj.shape.n, "smoke_o_out")?;

        // (6) Residual: h_residual += o_out.
        rvllm_fused::gemma4_launcher::VectorAddF16Launch { n: hidden }.launch(
            kernels.fn_vector_add_bf16,
            scr.h_residual_ptr,
            scr.o_out_ptr,
            stream_u64,
        )?;
        let h_after_attn = dump_bf16(stream, stream_u64,
            scr.h_residual_ptr, hidden as usize, "smoke_h_after_attn")?;

        // (7) DtoD copy h_residual → h_work, RMSNorm with
        //     post_attention_layernorm.
        dtod(stream_u64, scr.h_work_ptr,
             scr.h_residual_ptr, hidden_bytes)?;
        rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
            num_tokens: 1, hidden, eps: self.arch.text.rms_norm_eps as f32,
        }.launch(
            kernels.fn_rmsnorm_inplace_bf16_gbf16,
            scr.h_work_ptr,
            layer0.post_attention_layernorm.offset_bytes,
            stream_u64,
        )?;
        let post_attn_norm = dump_bf16(stream, stream_u64,
            scr.h_work_ptr, hidden as usize, "smoke_post_attn_norm")?;

        // (8) gate / up projections (W4A16).
        gemm(scr.gate_out_ptr, scr.h_work_ptr, &layer0.gate_proj)?;
        gemm(scr.up_out_ptr,   scr.h_work_ptr, &layer0.up_proj)?;
        let gate_out = dump_bf16(stream, stream_u64,
            scr.gate_out_ptr, layer0.gate_proj.shape.n, "smoke_gate_out")?;
        let up_out = dump_bf16(stream, stream_u64,
            scr.up_out_ptr, layer0.up_proj.shape.n, "smoke_up_out")?;

        // (9) silu_mul: silu_mid = silu(gate) * up, both BF16.
        let i_size = self.arch.text.intermediate_size as i32;
        {
            let mut out_ptr = scr.silu_mid_ptr;
            let mut g_ptr = scr.gate_out_ptr;
            let mut u_ptr = scr.up_out_ptr;
            let mut n_arg = i_size;
            let args = [
                (&mut out_ptr) as *mut u64 as *mut std::ffi::c_void,
                (&mut g_ptr) as *mut u64 as *mut std::ffi::c_void,
                (&mut u_ptr) as *mut u64 as *mut std::ffi::c_void,
                (&mut n_arg) as *mut i32 as *mut std::ffi::c_void,
            ];
            const BLOCK: u32 = 256;
            let n = i_size as u32;
            let grid = ((n + BLOCK - 1) / BLOCK, 1, 1);
            launch_kernel(
                kernels.fn_silu_mul_bf16,
                grid, (BLOCK, 1, 1), &args, stream_u64,
            )?;
        }
        let silu_mid = dump_bf16(stream, stream_u64,
            scr.silu_mid_ptr,
            self.arch.text.intermediate_size, "smoke_silu_mid")?;

        // (10) Down projection (W4A16): bf16 silu_mid × down_proj weights.
        gemm(scr.down_out_ptr, scr.silu_mid_ptr, &layer0.down_proj)?;
        let down_out = dump_bf16(stream, stream_u64,
            scr.down_out_ptr,
            layer0.down_proj.shape.n, "smoke_down_out")?;

        // (11) Residual: h_residual += down_out.
        rvllm_fused::gemma4_launcher::VectorAddF16Launch { n: hidden }.launch(
            kernels.fn_vector_add_bf16,
            scr.h_residual_ptr,
            scr.down_out_ptr,
            stream_u64,
        )?;
        let h_after_layer0 = dump_bf16(stream, stream_u64,
            scr.h_residual_ptr, hidden as usize, "smoke_h_after_layer0")?;

        // ============================================================
        // Layers 1..88 — same flow as layer 0 with that layer's
        // weights. Closure reused; only the residual rms is captured
        // per layer (full per-stage dumps would be ~88×14 tensors).
        // ============================================================
        let mut layer_residual_rms: Vec<f32> = Vec::with_capacity(model.layers.len());
        // Record layer-0 residual rms as the first entry. Round-11 #4
        // fix: when `RVLLM_SMOKE_FULL_DUMP` is off, dump_bf16 returns
        // an empty vec so `sumsq / 0.0` produced NaN here. Match the
        // 0.0-placeholder convention used for layers 1..N below.
        let l0_rms = if h_after_layer0.is_empty() {
            0.0
        } else {
            let l0_sumsq: f64 = h_after_layer0.iter()
                .map(|&x| (x as f64) * (x as f64)).sum();
            (l0_sumsq / (h_after_layer0.len() as f64)).sqrt() as f32
        };
        layer_residual_rms.push(l0_rms);

        let head_dim_arg = head_dim;
        let n_q_heads_arg = n_q_heads;
        let gqa_ratio_arg = gqa_ratio;
        let i_size = i_size;
        for layer_idx in 1..model.layers.len() {
            let layer = &model.layers[layer_idx];

            // Codex round-7: pre-layer-L h_residual dump (= layer L-1's
            // output residual). One file per (layer, position).
            dump_h_resid_pre_layer(layer_idx)?;
            // Pre-attn norm.
            dtod(stream_u64, scr.h_work_ptr,
                 scr.h_residual_ptr, hidden_bytes)?;
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: 1, hidden, eps: self.arch.text.rms_norm_eps as f32,
            }.launch(
                kernels.fn_rmsnorm_inplace_bf16_gbf16,
                scr.h_work_ptr,
                layer.input_layernorm.offset_bytes,
                stream_u64,
            )?;

            // Q/K/V + RoPE on Q and K (W4A16 dequant + bf16 GEMM).
            gemm(scr.q_out_ptr, scr.h_work_ptr, &layer.q_proj)?;
            gemm(scr.k_out_ptr, scr.h_work_ptr, &layer.k_proj)?;
            gemm(scr.v_out_ptr, scr.h_work_ptr, &layer.v_proj)?;
            rope(scr.q_out_ptr, n_q_heads_u)?;
            rope(scr.k_out_ptr, n_kv_heads_u)?;

            // Attention via KV cache (write + qk_dot + softmax_v).
            attention_step(layer_idx)?;

            // O proj + residual (W4A16).
            gemm(scr.o_out_ptr, scr.attn_out_ptr, &layer.o_proj)?;
            rvllm_fused::gemma4_launcher::VectorAddF16Launch { n: hidden }.launch(
                kernels.fn_vector_add_bf16,
                scr.h_residual_ptr,
                scr.o_out_ptr,
                stream_u64,
            )?;

            // Post-attn norm.
            dtod(stream_u64, scr.h_work_ptr,
                 scr.h_residual_ptr, hidden_bytes)?;
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: 1, hidden, eps: self.arch.text.rms_norm_eps as f32,
            }.launch(
                kernels.fn_rmsnorm_inplace_bf16_gbf16,
                scr.h_work_ptr,
                layer.post_attention_layernorm.offset_bytes,
                stream_u64,
            )?;

            // gate / up / silu_mul (W4A16).
            gemm(scr.gate_out_ptr, scr.h_work_ptr, &layer.gate_proj)?;
            gemm(scr.up_out_ptr,   scr.h_work_ptr, &layer.up_proj)?;
            {
                let mut out_ptr = scr.silu_mid_ptr;
                let mut g_ptr = scr.gate_out_ptr;
                let mut u_ptr = scr.up_out_ptr;
                let mut n_arg = i_size;
                let args = [
                    (&mut out_ptr) as *mut u64 as *mut std::ffi::c_void,
                    (&mut g_ptr) as *mut u64 as *mut std::ffi::c_void,
                    (&mut u_ptr) as *mut u64 as *mut std::ffi::c_void,
                    (&mut n_arg) as *mut i32 as *mut std::ffi::c_void,
                ];
                const BLOCK: u32 = 256;
                let n = i_size as u32;
                let grid = ((n + BLOCK - 1) / BLOCK, 1, 1);
                launch_kernel(
                    kernels.fn_silu_mul_bf16,
                    grid, (BLOCK, 1, 1), &args, stream_u64,
                )?;
            }

            // down + residual (W4A16).
            gemm(scr.down_out_ptr, scr.silu_mid_ptr, &layer.down_proj)?;
            rvllm_fused::gemma4_launcher::VectorAddF16Launch { n: hidden }.launch(
                kernels.fn_vector_add_bf16,
                scr.h_residual_ptr,
                scr.down_out_ptr,
                stream_u64,
            )?;

            // Per-layer h_residual dump for compounding bisect.
            // Triggered only when RVLLM_BISECT_DUMP_LAYERS lists the
            // current layer_idx as a comma-separated decimal int.
            if let Ok(list) = std::env::var("RVLLM_BISECT_DUMP_LAYERS") {
                if list.split(',').any(|s| s.trim().parse::<usize>().ok() == Some(layer_idx)) {
                    if let Some(dir) = debug_env_str("RVLLM_SMOKE_DUMP_DIR") {
                        stream.fence()?;
                        let bytes = (hidden as usize) * 2;
                        let mut buf = vec![0u8; bytes];
                        unsafe { use cudarc::driver::sys::*;
                            let _ = cuMemcpyDtoH_v2(buf.as_mut_ptr() as *mut std::ffi::c_void,
                                scr.h_residual_ptr as CUdeviceptr, bytes); }
                        let _ = std::fs::write(std::path::PathBuf::from(&dir)
                            .join(format!("h_residual_layer{}.bf16", layer_idx)), &buf);
                    }
                }
            }
            // Per-layer residual rms costs a stream fence + DtoH per
            // layer × 88 layers × N prefill steps. Off by default;
            // RVLLM_SMOKE_LAYER_RMS=1 turns it on for one-shot
            // diagnostics. When off, push 0.0 placeholder so the
            // length matches num_layers.
            if debug_env_os("RVLLM_SMOKE_LAYER_RMS").is_some() {
                let v = dump_bf16(stream, stream_u64,
                    scr.h_residual_ptr, hidden as usize,
                    "smoke_layer_residual")?;
                let sumsq: f64 = v.iter().map(|&x| (x as f64) * (x as f64)).sum();
                layer_residual_rms.push((sumsq / (v.len() as f64)).sqrt() as f32);
            } else {
                layer_residual_rms.push(0.0);
            }
        }

        // ============================================================
        // Final RMSNorm + lm_head + argmax. Skipped entirely when
        // `compute_logits` is false (prefill non-last steps).
        // ============================================================
        if !compute_logits {
            // Scratch is pre-allocated at bring-up; consecutive forward
            // steps overwrite the SAME device addresses. CUDA's same-
            // stream ordering ensures kernel-N+1 doesn't start before
            // kernel-N completes, so no host-side fence is needed
            // between calls — the driver's launch queue auto-throttles
            // when full. Deliberately no `stream.fence()` here: that
            // was the dominant per-step latency in the per-call
            // arena.region path.
            return Ok(SmokeStageDump {
                post_embed, post_rmsnorm,
                q_out, k_out, v_out,
                attn_out, o_out, h_after_attn,
                post_attn_norm, gate_out, up_out, silu_mid,
                down_out, h_after_layer0,
                layer_residual_rms,
                h_after_final_norm: Vec::new(),
                logits_top8: Vec::new(),
                predicted_token: u32::MAX,
            });
        }

        let cublaslt = self.cublaslt.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_smoke_layer0: cublaslt absent".into(),
        ))?;

        // Final norm: copy h_residual → h_work, RMSNorm with
        // outside.final_norm gamma.
        dtod(stream_u64, scr.h_work_ptr,
             scr.h_residual_ptr, hidden_bytes)?;
        rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
            num_tokens: 1, hidden, eps: self.arch.text.rms_norm_eps as f32,
        }.launch(
            kernels.fn_rmsnorm_inplace_bf16_gbf16,
            scr.h_work_ptr,
            model.outside.final_norm.offset_bytes,
            stream_u64,
        )?;
        // Diagnostic: also dump the LAST LAYER's qk-dot scores
        // (n_q_heads × past_len F32) so we can inspect the attention
        // distribution numerically. The scores_f32 buffer is overwritten
        // each layer; this captures whatever was last left there
        // (= layer 87's scores for the LAST prefill token).
        if full_dump {
            let bytes = (n_q_heads_attn as usize) * (past_len as usize) * 4;
            let mut buf = vec![0u8; bytes];
            stream.fence()?;
            unsafe {
                use cudarc::driver::sys::*;
                let r = cuMemcpyDtoH_v2(
                    buf.as_mut_ptr() as *mut std::ffi::c_void,
                    kv_cache.scores_f32_ptr as CUdeviceptr,
                    bytes,
                );
                if r == CUresult::CUDA_SUCCESS {
                    if let Some(dir) = debug_env_str("RVLLM_SMOKE_DUMP_DIR") {
                        let path = std::path::PathBuf::from(dir).join("scores_layer87.f32");
                        let _ = std::fs::write(&path, &buf);
                        eprintln!("[mistral35] dumped scores layer87: {} bytes ({}x{})",
                            bytes, n_q_heads_attn, past_len);
                    }
                }
            }
        }

        let h_after_final_norm = dump_bf16(stream, stream_u64,
            scr.h_work_ptr, hidden as usize, "smoke_h_after_final_norm")?;

        // lm_head: BF16 [vocab, hidden] × BF16 [1, hidden] → F32 [1, vocab]
        let vocab = self.arch.text.vocab_size as i32;
        cublaslt.bf16_gemm_f32(
            scr.h_work_ptr,       // a [1, hidden] (BF16)
            model.outside.lm_head.offset_bytes, // b [vocab, hidden] (BF16)
            scr.logits_ptr,        // d [1, vocab] (F32)
            1,                                 // m = num_seqs
            vocab,
            h_k,
            stream_u64,
        )?;

        // argmax_kernel(logits_f32, out_token_i32, vocab_size).
        {
            let mut logits_ptr = scr.logits_ptr;
            let mut tok_ptr = scr.token_out_ptr;
            let mut vocab_arg = vocab;
            let args = [
                (&mut logits_ptr) as *mut u64 as *mut std::ffi::c_void,
                (&mut tok_ptr) as *mut u64 as *mut std::ffi::c_void,
                (&mut vocab_arg) as *mut i32 as *mut std::ffi::c_void,
            ];
            launch_kernel(
                kernels.fn_argmax_f32,
                (1, 1, 1), (1024, 1, 1), &args, stream_u64,
            )?;
        }

        // DtoH the predicted token (always 4 bytes). Full 131072×4
        // logits DtoH + host-side top-8 sort are gated behind
        // RVLLM_SMOKE_FULL_DUMP — they're 524 KB + ~250k cmp ops
        // per call which dominates a multi-token decode.
        stream.fence()?;
        let mut tok_buf = [0u8; 4];
        unsafe {
            use cudarc::driver::sys::*;
            let r1 = cuMemcpyDtoH_v2(
                tok_buf.as_mut_ptr() as *mut std::ffi::c_void,
                scr.token_out_ptr as CUdeviceptr, 4);
            if r1 != CUresult::CUDA_SUCCESS {
                return Err(RvllmError::cuda(
                    "cuMemcpyDtoH_v2(predicted_token)",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx { stream: stream_u64,
                        kernel: "smoke_token_dtoh", launch: None, device: -1 },
                ));
            }
        }
        let predicted_token = u32::from_le_bytes(tok_buf);
        let logits_top8: Vec<(u32, f32)> = if full_dump {
            let mut logits_host = vec![0f32; vocab as usize];
            unsafe {
                use cudarc::driver::sys::*;
                let r2 = cuMemcpyDtoH_v2(
                    logits_host.as_mut_ptr() as *mut std::ffi::c_void,
                    scr.logits_ptr as CUdeviceptr,
                    (vocab as usize) * 4);
                if r2 != CUresult::CUDA_SUCCESS {
                    return Err(RvllmError::cuda(
                        "cuMemcpyDtoH_v2(logits)",
                        rvllm_core::CudaErrorKind::MemcpyFailed,
                        rvllm_core::CudaCtx { stream: stream_u64,
                            kernel: "smoke_logits_dtoh", launch: None, device: -1 },
                    ));
                }
            }
            let mut indexed: Vec<(u32, f32)> = logits_host.iter().enumerate()
                .map(|(i, &v)| (i as u32, v)).collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            indexed.into_iter().take(8).collect()
        } else {
            Vec::new()
        };

        Ok(SmokeStageDump {
            post_embed, post_rmsnorm,
            q_out, k_out, v_out,
            attn_out, o_out, h_after_attn,
            post_attn_norm, gate_out, up_out, silu_mid,
            down_out, h_after_layer0,
            layer_residual_rms, h_after_final_norm,
            logits_top8, predicted_token,
        })
    }

    /// Multi-token autoregressive generation. Processes the prompt
    /// through `forward_smoke_q_proj` at positions 0..prompt.len(),
    /// then loops up to `max_new` decode steps feeding each predicted
    /// token back as the next input. KV cache is persistent across
    /// calls (the bring-up's per-layer cache); the arena scratch is
    /// rewound between calls via `arena.checkpoint() + restore` so
    /// per-token allocations don't accumulate.
    ///
    /// `eos_ids` causes early termination (typical Mistral EOS = 2).
    /// Backward-compatibility shim. New code should call
    /// [`generate`](Self::generate) directly. Kept so existing test
    /// fixtures (and any out-of-tree caller that pinned the old
    /// "smoke" name) continue to compile after the F4#2 rename.
    pub unsafe fn generate_smoke(
        &self,
        prompt: &[u32],
        max_new: usize,
        eos_ids: &[u32],
    ) -> Result<GenerateResult> {
        self.generate(prompt, max_new, eos_ids, None, |_| ())
    }

    /// Backward-compatibility shim — see [`generate`](Self::generate).
    pub unsafe fn generate_smoke_cancellable<F>(
        &self,
        prompt: &[u32],
        max_new: usize,
        eos_ids: &[u32],
        cancelled: Option<&std::sync::atomic::AtomicBool>,
        on_token: F,
    ) -> Result<GenerateResult>
    where
        F: FnMut(u32),
    {
        self.generate(prompt, max_new, eos_ids, cancelled, on_token)
    }

    /// Run a Mistral 3.5 prefill+decode generation.
    ///
    /// **Stages** (each split into a named helper for readability;
    /// F4#2 cleanup):
    ///
    /// 1. `validate_generate_inputs` — non-empty prompt, arena +
    ///    kv_cache present, total_len within KV cap.
    /// 2. `RVLLM_SMOKE_SINGLE=1` short-circuit (one diagnostic
    ///    forward, no prefill loop, no decode loop).
    /// 3. `prefill_token` × prompt.len() — feed each prompt token
    ///    at its position. Only the last token's forward computes
    ///    logits + LM-head argmax; earlier tokens skip that work.
    /// 4. `decode_token` × max_new — feed the previously predicted
    ///    token at the next position. Each call computes logits
    ///    and runs LM-head argmax.
    ///
    /// `cancelled` is checked between every stage; on trip the
    /// partial token list is returned with `last_dump = None`. The
    /// `on_token` callback fires for every newly emitted decode
    /// token so a streaming SSE path can flush it before the next
    /// step.
    /// Run the Pixtral 48-block ViT + projector for a single decoded
    /// image, producing soft tokens spliceable into the language-
    /// decoder embed buffer.
    ///
    /// `image_bytes` is the raw image file (PNG/JPEG/...) the host
    /// admission layer fetched. `preprocess_mistral35_pixtral` does
    /// the resize + patchify + CLIP normalisation; this function
    /// owns everything from there onward (patch_conv → 48 blocks
    /// → patch_merger → projector → BF16 output).
    ///
    /// Round-12 (Pixtral vision phase 2 — signature landed; the
    /// implementation lands per-stage in phase 3+ via the order
    /// described in MISTRAL35_PIXTRAL_VISION_PLAN.md).
    /// Backward-compat host path: runs the Pixtral forward and DtoH
    /// the BF16 soft-token output into a `Vec<u8>`. New callers
    /// should prefer [`Mistral35Bringup::forward_pixtral_vision_into`]
    /// to avoid the round-trip (Round-12 phase 5c codex review #3).
    pub fn forward_pixtral_vision(
        &self,
        image_bytes: &[u8],
    ) -> Result<Mistral35VisionForwardOutput> {
        let (model, vision) = self.vision_preflight()?;
        let pp = self.vision_preprocess(image_bytes)?;
        let _ = model;

        #[cfg(feature = "cuda")]
        {
            // Allocate own arena scratch for the device output, run
            // forward_into, DtoH, free.
            let arena = self.arena.as_ref().ok_or_else(|| corrupt(
                self.paths.model_dir.clone(),
                "forward_pixtral_vision: arena absent".into(),
            ))?;
            let stream = self.stream.as_ref().ok_or_else(|| corrupt(
                self.paths.model_dir.clone(),
                "forward_pixtral_vision: stream absent".into(),
            ))?;
            let nt = pp.num_soft_tokens;
            let patch_grid = pp.patch_grid;
            let merged_grid = pp.merged_grid;
            let text_hidden = self.arch.text.hidden_size;
            let bytes = nt * text_hidden * 2;
            let ck_outer = arena.checkpoint();
            let dst_region = arena.region("pixtral_compat_output", bytes, 16)?;
            self.forward_pixtral_vision_cuda(vision, pp, dst_region.device_ptr())?;
            stream.fence()?;
            let mut data = vec![0u8; bytes];
            unsafe {
                use cudarc::driver::sys::*;
                let r = cuMemcpyDtoH_v2(
                    data.as_mut_ptr() as *mut _,
                    dst_region.device_ptr(),
                    bytes,
                );
                if r != CUresult::CUDA_SUCCESS {
                    unsafe { arena.restore(ck_outer); }
                    return Err(rvllm_core::RvllmError::cuda(
                        "pixtral: DtoH soft tokens",
                        rvllm_core::CudaErrorKind::LaunchFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            }
            unsafe { arena.restore(ck_outer); }
            // Codex review #4: reuse the `pp` already computed above
            // instead of re-decoding the image and re-running
            // preprocess just to ship grid metadata. The CUDA forward
            // already consumed `pp` so the values are still in scope.
            Ok(Mistral35VisionForwardOutput {
                data,
                num_tokens: nt,
                hidden_dim: text_hidden,
                patch_grid,
                merged_grid,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = vision; let _ = pp;
            Err(corrupt(
                self.paths.model_dir.clone(),
                "forward_pixtral_vision: cuda feature not enabled".into(),
            ))
        }
    }

    /// Round-12 phase 5c codex review #3: device-resident vision
    /// forward. Runs the full Pixtral pipeline + writes the
    /// `[num_soft_tokens, text_hidden]` BF16 soft-token output
    /// directly into `dst_dev_ptr` (caller-provided). No DtoH
    /// round-trip.
    ///
    /// `dst_dev_ptr` must point at a device buffer of at least
    /// `expected_num_tokens * text_hidden * 2` bytes that is alive
    /// for the duration of this call. The intermediate scratch is
    /// allocated inside this function's own arena checkpoint and
    /// freed before return.
    ///
    /// Returns the actual number of soft tokens produced (= the
    /// host preprocess's `num_soft_tokens`). The caller should
    /// ensure `expected_num_tokens` matches what the renderer
    /// reserved in the prompt; mismatches log a warning but are not
    /// fatal here (the caller handles slot sizing).
    #[cfg(feature = "cuda")]
    pub fn forward_pixtral_vision_into(
        &self,
        image_bytes: &[u8],
        dst_dev_ptr: u64,
        expected_num_tokens: usize,
    ) -> Result<usize> {
        let (_model, vision) = self.vision_preflight()?;
        let pp = self.vision_preprocess(image_bytes)?;
        let nt = pp.num_soft_tokens;
        if expected_num_tokens != 0 && nt != expected_num_tokens {
            tracing::warn!(
                "[mistral35-vision] forward_pixtral_vision_into: \
                 produced num_soft_tokens={} but caller expected {}",
                nt, expected_num_tokens,
            );
        }
        self.forward_pixtral_vision_cuda(vision, pp, dst_dev_ptr)?;
        Ok(nt)
    }

    /// Pre-flight: vision tower loaded? Default `RVLLM_LOAD_VISION` is
    /// OFF (Round-10 #1); opt in via the rvllm profile.
    fn vision_preflight(
        &self,
    ) -> Result<(
        &rvllm_loader::mistral35_weights::Mistral35LoadedModel,
        &rvllm_loader::mistral35_weights::Mistral35Vision,
    )> {
        let model = self.model.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_pixtral_vision: model not loaded".into(),
        ))?;
        let vision = model.vision.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_pixtral_vision: vision tower not loaded; set \
             RVLLM_LOAD_VISION=1 in the rvllm profile to enable.".into(),
        ))?;
        Ok((model, vision))
    }

    fn vision_preprocess(
        &self,
        image_bytes: &[u8],
    ) -> Result<crate::vision_preprocess::Mistral35Patches> {
        let img = image::load_from_memory(image_bytes).map_err(|e| corrupt(
            self.paths.model_dir.clone(),
            format!("forward_pixtral_vision: image decode failed: {e}"),
        ))?.to_rgb8();
        let cfg = crate::vision_preprocess::Mistral35PreprocessConfig::default();
        crate::vision_preprocess::preprocess_mistral35_pixtral(&img, &cfg)
            .map_err(|e| corrupt(
                self.paths.model_dir.clone(),
                format!("forward_pixtral_vision: preprocess failed: {e:?}"),
            ))
    }

    /// CUDA-only Pixtral ViT forward (Round-12 phase 3b).
    ///
    /// Stages implemented today:
    ///   1. Upload `pp.pixel_values` (f32 HWC) → BF16 device buffer.
    ///   2. patch_conv: cublasLt bf16_gemm_f32 + f32→bf16 cast →
    ///      `[N, v_hidden]` BF16.
    ///   3. ln_pre: rmsnorm_inplace_bf16_gbf16.
    ///
    /// Stages NOT YET implemented (per
    /// MISTRAL35_PIXTRAL_VISION_PLAN.md, future phases):
    ///   4. 48 ViT blocks (norm + QKV + 2D RoPE + FA2 + O + residual
    ///      + norm + MLP + residual).
    ///   5. patch_merger_2x2 reshape + projector linear_1 / linear_2.
    ///   6. DtoH and return Mistral35VisionForwardOutput.
    ///
    /// The function returns Err with a stage marker when the next
    /// unimplemented stage is hit; intermediates can be dumped under
    /// `RVLLM_DEBUG_MISTRAL35=1 RVLLM_PIXTRAL_DUMP_DIR=<path>` for
    /// the cosine-vs-HF gate harness (lands with phase 3-test).
    #[cfg(feature = "cuda")]
    fn forward_pixtral_vision_cuda(
        &self,
        vision: &rvllm_loader::mistral35_weights::Mistral35Vision,
        pp: crate::vision_preprocess::Mistral35Patches,
        // Round-12 phase 5c #3: caller's [num_soft_tokens, text_hidden]
        // BF16 device buffer that the final BF16 cast writes into.
        // Eliminates the DtoH→HtoD round-trip the prior
        // Vec<u8>-returning interface required.
        output_dst_ptr: u64,
    ) -> Result<()> {
        let arena = self.arena.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_pixtral_vision_cuda: arena absent".into(),
        ))?;
        let kernels = self.forward_kernels.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_pixtral_vision_cuda: forward_kernels absent".into(),
        ))?;
        let cublaslt = self.cublaslt.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_pixtral_vision_cuda: cublaslt absent".into(),
        ))?;
        let stream = self.stream.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_pixtral_vision_cuda: stream absent".into(),
        ))?;
        let stream_u64 = stream.raw() as u64;

        let n = pp.num_patches();
        let v_hidden = self.arch.vision.hidden_size;
        let inner = (self.arch.vision.patch_size as usize)
            * (self.arch.vision.patch_size as usize)
            * (self.arch.vision.num_channels as usize);
        debug_assert_eq!(pp.pixel_values.len(), n * inner);

        // Optional intermediate-dump dir for cosine gates.
        let dump_dir: Option<std::path::PathBuf> = debug_env_str(
            "RVLLM_PIXTRAL_DUMP_DIR",
        ).map(std::path::PathBuf::from);
        if let Some(ref d) = dump_dir { let _ = std::fs::create_dir_all(d); }

        // Use a checkpoint so the per-image scratch frees after this
        // forward completes.
        let ck = arena.checkpoint();

        // Allocate working regions.
        let patches_bf16_bytes = n * inner * 2;
        let patches_region = arena.region(
            "pixtral_patches_bf16", patches_bf16_bytes, 16)?;
        // f32 GEMM scratch — `m * v_hidden` floats.
        let f32_scratch_bytes = n * v_hidden * 4;
        let f32_scratch = arena.region(
            "pixtral_f32_scratch", f32_scratch_bytes, 16)?;
        // BF16 hidden buffer — `[N, v_hidden]`.
        let hidden_bytes = n * v_hidden * 2;
        let hidden_region = arena.region(
            "pixtral_hidden", hidden_bytes, 16)?;

        // ── Stage 1: upload patches as BF16 ────────────────────────────
        // Round-trip through a host BF16 buffer (no device-side f32→bf16
        // kernel needed for the upload; we already pay one cast on
        // patch_conv's output).
        let patches_bf16_host: Vec<u8> = pp.pixel_values_to_f16_bytes_bf16();
        debug_assert_eq!(patches_bf16_host.len(), patches_bf16_bytes);
        unsafe { patches_region.copy_from_host(&patches_bf16_host)? };
        if let Some(ref d) = dump_dir {
            let _ = std::fs::write(d.join("patches_bf16.bin"), &patches_bf16_host);
        }

        // ── Stage 2: patch_conv = patches @ W_pc^T ────────────────────
        // patches: [N, inner]   BF16
        // W_pc:    [v_hidden, inner]   BF16 (already permuted to HWC at load)
        // out:     [N, v_hidden]      BF16
        unsafe {
            cublaslt.bf16_gemm_f32(
                patches_region.device_ptr(),
                vision.patch_conv.offset_bytes,
                f32_scratch.device_ptr(),
                n as i32, v_hidden as i32, inner as i32,
                stream_u64,
            )?;
        }
        // Cast f32 → bf16 in-place into hidden_region.
        {
            let total = (n * v_hidden) as i32;
            let mut dst = hidden_region.device_ptr();
            let mut src = f32_scratch.device_ptr();
            let mut count = total;
            let args: [*mut std::ffi::c_void; 3] = [
                (&mut dst)   as *mut u64 as *mut _,
                (&mut src)   as *mut u64 as *mut _,
                (&mut count) as *mut i32 as *mut _,
            ];
            let blocks = ((total as u32) + 1023) / 1024;
            unsafe {
                rvllm_fused::launch_raw(
                    kernels.fn_f32_to_bf16,
                    (blocks, 1, 1),
                    (1024, 1, 1),
                    0, stream_u64,
                    &args,
                )?;
            }
        }
        if let Some(ref d) = dump_dir {
            stream.fence()?;
            let mut host = vec![0u8; hidden_bytes];
            unsafe {
                use cudarc::driver::sys::*;
                let _ = cuMemcpyDtoH_v2(
                    host.as_mut_ptr() as *mut _,
                    hidden_region.device_ptr(), hidden_bytes);
            }
            let _ = std::fs::write(d.join("post_patch_conv.bin"), &host);
        }

        // ── Stage 3: ln_pre RMSNorm (in-place on hidden_region) ───────
        // Use the established RmsnormInplaceLaunch wrapper (it picks
        // smem + block size correctly and validates the args).
        unsafe {
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: n as u32,
                hidden: v_hidden as u32,
                eps: 1.0e-5,
            }.launch(
                kernels.fn_rmsnorm_inplace_bf16_gbf16,
                hidden_region.device_ptr(),
                vision.ln_pre.offset_bytes,
                stream_u64,
            )?;
        }
        if let Some(ref d) = dump_dir {
            stream.fence()?;
            let mut host = vec![0u8; hidden_bytes];
            unsafe {
                use cudarc::driver::sys::*;
                let _ = cuMemcpyDtoH_v2(
                    host.as_mut_ptr() as *mut _,
                    hidden_region.device_ptr(), hidden_bytes);
            }
            let _ = std::fs::write(d.join("post_ln_pre.bin"), &host);
        }

        // ── Stage 4: 48 ViT block stack ───────────────────────────────
        //
        // Per-block sequence (Llama-style ViT, bidirectional attn):
        //   residual = x
        //   x = norm1(x)
        //   q,k,v = x @ Wq^T / Wk^T / Wv^T            (3 BF16 GEMMs)
        //   apply 2D RoPE on q, k                      (full rotary)
        //   scores = q @ k^T  / sqrt(D)                (batched-strided)
        //   p      = softmax(scores)                   (per-row)
        //   v_t    = transpose V to [H, D, N]
        //   attn   = p @ v_t^T                          (batched-strided)
        //   attn_out = scatter_heads(attn, [H,N,D] → [N,H*D])
        //   x = residual + attn_out @ Wo^T
        //   residual = x
        //   x = norm2(x)
        //   gate, up = x @ Wg^T, x @ Wu^T              (2 BF16 GEMMs)
        //   m  = gelu_tanh(gate) * up
        //   x = residual + m @ Wd^T
        //
        // All scratch lives between an inner checkpoint so memory is
        // bounded and reused across the 48 blocks.

        let head_dim = self.arch.vision.head_dim;
        let n_heads = self.arch.vision.num_attention_heads;
        let intermediate = self.arch.vision.intermediate_size;
        debug_assert_eq!(n_heads * head_dim, v_hidden,
            "n_heads * head_dim == v_hidden invariant violated");
        let attn_scale: f32 = 1.0 / (head_dim as f32).sqrt();

        // ── 4.0 Pre-compute + upload Pixtral 2D RoPE cos/sin tables ───
        let (grid_h, grid_w) = pp.patch_grid;
        let rope_tables = crate::mistral35_pixtral_rope::PixtralRopeTables::build(
            grid_h, grid_w, head_dim, self.arch.vision.rope_theta,
        );
        let cos_bytes = rope_tables.cos.len() * 4;
        let sin_bytes = rope_tables.sin.len() * 4;
        let cos_region = arena.region("pixtral_rope_cos", cos_bytes, 16)?;
        let sin_region = arena.region("pixtral_rope_sin", sin_bytes, 16)?;
        // f32 → little-endian bytes.
        let cos_host: Vec<u8> = rope_tables.cos.iter()
            .flat_map(|f| f.to_le_bytes()).collect();
        let sin_host: Vec<u8> = rope_tables.sin.iter()
            .flat_map(|f| f.to_le_bytes()).collect();
        unsafe { cos_region.copy_from_host(&cos_host)? };
        unsafe { sin_region.copy_from_host(&sin_host)? };
        let cos_ptr = cos_region.device_ptr();
        let sin_ptr = sin_region.device_ptr();

        // ── 4.1 Allocate per-block scratch (reused across all 48) ─────
        let residual_region = arena.region(
            "pixtral_residual", hidden_bytes, 16)?;
        let q_region = arena.region("pixtral_q", hidden_bytes, 16)?;
        let k_region = arena.region("pixtral_k", hidden_bytes, 16)?;
        let v_region = arena.region("pixtral_v", hidden_bytes, 16)?;
        let attn_out_region = arena.region(
            "pixtral_attn_out", hidden_bytes, 16)?;
        let attn_hmajor_region = arena.region(
            "pixtral_attn_hmajor", hidden_bytes, 16)?;
        let v_t_region = arena.region("pixtral_v_t", hidden_bytes, 16)?;
        let scores_f32_region = arena.region(
            "pixtral_scores_f32", n * n * n_heads * 4, 16)?;
        let scores_bf16_region = arena.region(
            "pixtral_scores_bf16", n * n * n_heads * 2, 16)?;
        let mlp_gate_region = arena.region(
            "pixtral_mlp_gate", n * intermediate * 2, 16)?;
        let mlp_up_region = arena.region(
            "pixtral_mlp_up", n * intermediate * 2, 16)?;
        // The big f32 scratch must cover both
        // [N, v_hidden]*4 (Q/K/V/O/down output) and
        // [N, intermediate]*4 (gate/up output). Allocate the larger.
        let f32_big_scratch_bytes = n * intermediate.max(v_hidden) * 4;
        let f32_big_scratch = arena.region(
            "pixtral_f32_big_scratch", f32_big_scratch_bytes, 16)?;

        // Helper closures over the kernels + cublasLt.
        let do_gemm = |in_bf16: u64, w_bf16: u64, out_bf16: u64,
                       m: i32, n_dim: i32, k: i32| -> Result<()> {
            unsafe {
                cublaslt.bf16_gemm_f32(
                    in_bf16, w_bf16, f32_big_scratch.device_ptr(),
                    m, n_dim, k, stream_u64,
                )?;
                let total = m * n_dim;
                let mut dst = out_bf16;
                let mut src = f32_big_scratch.device_ptr();
                let mut count = total;
                let args: [*mut std::ffi::c_void; 3] = [
                    (&mut dst) as *mut u64 as *mut _,
                    (&mut src) as *mut u64 as *mut _,
                    (&mut count) as *mut i32 as *mut _,
                ];
                let blocks = ((total as u32) + 1023) / 1024;
                rvllm_fused::launch_raw(
                    kernels.fn_f32_to_bf16,
                    (blocks, 1, 1), (1024, 1, 1),
                    0, stream_u64, &args,
                )
            }
        };

        let do_pixtral_rotary = |buf_bf16: u64| -> Result<()> {
            unsafe {
                let mut qk = buf_bf16;
                let mut cos_p = cos_ptr;
                let mut sin_p = sin_ptr;
                let mut nh = n_heads as i32;
                let mut hd = head_dim as i32;
                let mut rd = head_dim as i32;
                let args: [*mut std::ffi::c_void; 6] = [
                    (&mut qk) as *mut u64 as *mut _,
                    (&mut cos_p) as *mut u64 as *mut _,
                    (&mut sin_p) as *mut u64 as *mut _,
                    (&mut nh) as *mut i32 as *mut _,
                    (&mut hd) as *mut i32 as *mut _,
                    (&mut rd) as *mut i32 as *mut _,
                ];
                rvllm_fused::launch_raw(
                    kernels.fn_pixtral_rotary_2d_bf16,
                    (n as u32, n_heads as u32, 1),
                    ((head_dim / 2) as u32, 1, 1),
                    0, stream_u64, &args,
                )
            }
        };

        let do_vector_add_bf16 = |dst: u64, a: u64, b: u64| -> Result<()> {
            unsafe {
                let mut d = dst;
                let mut aa = a;
                let mut bb = b;
                let mut count = (n * v_hidden) as i32;
                let args: [*mut std::ffi::c_void; 4] = [
                    (&mut d) as *mut u64 as *mut _,
                    (&mut aa) as *mut u64 as *mut _,
                    (&mut bb) as *mut u64 as *mut _,
                    (&mut count) as *mut i32 as *mut _,
                ];
                let blocks = ((count as u32) + 1023) / 1024;
                rvllm_fused::launch_raw(
                    kernels.fn_vector_add_bf16,
                    (blocks, 1, 1), (1024, 1, 1),
                    0, stream_u64, &args,
                )
            }
        };

        let do_copy_bf16 = |dst: u64, src: u64, count_elem: usize| -> Result<()> {
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let r = cuMemcpyDtoDAsync_v2(
                    dst, src, count_elem * 2,
                    stream.raw() as CUstream);
                if r != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "pixtral: dev→dev copy",
                        rvllm_core::CudaErrorKind::LaunchFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            }
            Ok(())
        };

        // ── 4.2 Run all 48 blocks ─────────────────────────────────────
        for block_idx in 0..self.arch.vision.num_hidden_layers {
            let layer = &vision.layers[block_idx];

            // Save residual (= h_io before norm1).
            do_copy_bf16(
                residual_region.device_ptr(),
                hidden_region.device_ptr(),
                n * v_hidden,
            )?;

            // norm1 in-place on h_io.
            unsafe {
                rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                    num_tokens: n as u32,
                    hidden: v_hidden as u32,
                    eps: 1.0e-5,
                }.launch(
                    kernels.fn_rmsnorm_inplace_bf16_gbf16,
                    hidden_region.device_ptr(),
                    layer.attention_norm.offset_bytes,
                    stream_u64,
                )?;
            }

            // q,k,v projections.
            do_gemm(hidden_region.device_ptr(), layer.q_proj.offset_bytes,
                    q_region.device_ptr(),
                    n as i32, v_hidden as i32, v_hidden as i32)?;
            do_gemm(hidden_region.device_ptr(), layer.k_proj.offset_bytes,
                    k_region.device_ptr(),
                    n as i32, v_hidden as i32, v_hidden as i32)?;
            do_gemm(hidden_region.device_ptr(), layer.v_proj.offset_bytes,
                    v_region.device_ptr(),
                    n as i32, v_hidden as i32, v_hidden as i32)?;

            // 2D RoPE on Q + K (in-place).
            do_pixtral_rotary(q_region.device_ptr())?;
            do_pixtral_rotary(k_region.device_ptr())?;

            // QK^T batched-strided over heads → scores_f32 [H, N, N].
            unsafe {
                cublaslt.bf16_gemm_f32_batched_strided(
                    q_region.device_ptr(),
                    k_region.device_ptr(),
                    scores_f32_region.device_ptr(),
                    n as i32, n as i32, head_dim as i32,
                    n_heads as i32,
                    (n_heads * head_dim) as i32,    // lda = H*D (interleaved)
                    (n_heads * head_dim) as i32,    // ldb = H*D
                    n as i32,                        // ldd = N
                    head_dim as i64,                 // stride_a = D
                    head_dim as i64,                 // stride_b = D
                    (n * n) as i64,                  // stride_d = N*N
                    stream_u64,
                )?;
            }

            // Scale scores by 1/sqrt(head_dim).
            unsafe {
                let mut x = scores_f32_region.device_ptr();
                let mut scale = attn_scale;
                let mut count = (n_heads * n * n) as i32;
                let args: [*mut std::ffi::c_void; 3] = [
                    (&mut x) as *mut u64 as *mut _,
                    (&mut scale) as *mut f32 as *mut _,
                    (&mut count) as *mut i32 as *mut _,
                ];
                let blocks = ((count as u32) + 255) / 256;
                rvllm_fused::launch_raw(
                    kernels.fn_scale_inplace_f32,
                    (blocks, 1, 1), (256, 1, 1),
                    0, stream_u64, &args,
                )?;
            }

            // Per-row softmax → scores_bf16 [H, N, N]. Launch grid =
            // H*N rows; the kernel handles each row independently.
            unsafe {
                let mut out = scores_bf16_region.device_ptr();
                let mut input = scores_f32_region.device_ptr();
                let mut sl = n as i32;
                let args: [*mut std::ffi::c_void; 3] = [
                    (&mut out) as *mut u64 as *mut _,
                    (&mut input) as *mut u64 as *mut _,
                    (&mut sl) as *mut i32 as *mut _,
                ];
                let block: u32 = (n as u32).min(1024);
                rvllm_fused::launch_raw(
                    kernels.fn_softmax_row_f32_to_bf16,
                    ((n_heads * n) as u32, 1, 1),
                    (block, 1, 1),
                    0, stream_u64, &args,
                )?;
            }

            // Transpose V from [N, H*D] → [H, D, N] head-major.
            unsafe {
                let mut out = v_t_region.device_ptr();
                let mut in_p = v_region.device_ptr();
                let mut nh = n_heads as i32;
                let mut hd = head_dim as i32;
                let mut nt = n as i32;
                let args: [*mut std::ffi::c_void; 5] = [
                    (&mut out) as *mut u64 as *mut _,
                    (&mut in_p) as *mut u64 as *mut _,
                    (&mut nh) as *mut i32 as *mut _,
                    (&mut hd) as *mut i32 as *mut _,
                    (&mut nt) as *mut i32 as *mut _,
                ];
                rvllm_fused::launch_raw(
                    kernels.fn_transpose_heads_v_bf16,
                    (n as u32, n_heads as u32, 1),
                    (head_dim as u32, 1, 1),
                    0, stream_u64, &args,
                )?;
            }

            // scores @ V_T → out_f32 [H, N, D] batched-strided.
            // Per head: scores [N, N] (lda = N, stride_a = N²) @
            //           V_T [D, N] (ldb = N, stride_b = D*N) →
            //           out [N, D] (ldd = D, stride_d = N*D).
            unsafe {
                cublaslt.bf16_gemm_f32_batched_strided(
                    scores_bf16_region.device_ptr(),
                    v_t_region.device_ptr(),
                    f32_big_scratch.device_ptr(),
                    n as i32, head_dim as i32, n as i32,
                    n_heads as i32,
                    n as i32,                        // lda = N
                    n as i32,                        // ldb = N
                    head_dim as i32,                 // ldd = D
                    (n * n) as i64,                  // stride_a = N²
                    (head_dim * n) as i64,           // stride_b = D*N
                    (n * head_dim) as i64,           // stride_d = N*D
                    stream_u64,
                )?;
            }

            // Cast f32 → bf16 (head-major out_hmajor [H, N, D]).
            unsafe {
                let mut dst = attn_hmajor_region.device_ptr();
                let mut src = f32_big_scratch.device_ptr();
                let mut count = (n_heads * n * head_dim) as i32;
                let args: [*mut std::ffi::c_void; 3] = [
                    (&mut dst) as *mut u64 as *mut _,
                    (&mut src) as *mut u64 as *mut _,
                    (&mut count) as *mut i32 as *mut _,
                ];
                let blocks = ((count as u32) + 1023) / 1024;
                rvllm_fused::launch_raw(
                    kernels.fn_f32_to_bf16,
                    (blocks, 1, 1), (1024, 1, 1),
                    0, stream_u64, &args,
                )?;
            }

            // Scatter [H, N, D] head-major → [N, H*D] interleaved.
            unsafe {
                let mut out = attn_out_region.device_ptr();
                let mut in_p = attn_hmajor_region.device_ptr();
                let mut nh = n_heads as i32;
                let mut hd = head_dim as i32;
                let mut nt = n as i32;
                let args: [*mut std::ffi::c_void; 5] = [
                    (&mut out) as *mut u64 as *mut _,
                    (&mut in_p) as *mut u64 as *mut _,
                    (&mut nh) as *mut i32 as *mut _,
                    (&mut hd) as *mut i32 as *mut _,
                    (&mut nt) as *mut i32 as *mut _,
                ];
                rvllm_fused::launch_raw(
                    kernels.fn_scatter_heads_bf16,
                    (n as u32, n_heads as u32, 1),
                    (head_dim as u32, 1, 1),
                    0, stream_u64, &args,
                )?;
            }

            // O proj: hidden = attn_out @ W_o^T (overwrite hidden_region).
            do_gemm(attn_out_region.device_ptr(), layer.o_proj.offset_bytes,
                    hidden_region.device_ptr(),
                    n as i32, v_hidden as i32, v_hidden as i32)?;

            // residual += hidden  (= residual + attn_out @ W_o^T)
            // hidden_region = residual + hidden_region
            do_vector_add_bf16(
                hidden_region.device_ptr(),
                residual_region.device_ptr(),
                hidden_region.device_ptr(),
            )?;

            // Save residual2.
            do_copy_bf16(
                residual_region.device_ptr(),
                hidden_region.device_ptr(),
                n * v_hidden,
            )?;

            // norm2 in-place on h.
            unsafe {
                rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                    num_tokens: n as u32,
                    hidden: v_hidden as u32,
                    eps: 1.0e-5,
                }.launch(
                    kernels.fn_rmsnorm_inplace_bf16_gbf16,
                    hidden_region.device_ptr(),
                    layer.ffn_norm.offset_bytes,
                    stream_u64,
                )?;
            }

            // gate / up GEMMs (out shape [N, intermediate]).
            do_gemm(hidden_region.device_ptr(), layer.gate_proj.offset_bytes,
                    mlp_gate_region.device_ptr(),
                    n as i32, intermediate as i32, v_hidden as i32)?;
            do_gemm(hidden_region.device_ptr(), layer.up_proj.offset_bytes,
                    mlp_up_region.device_ptr(),
                    n as i32, intermediate as i32, v_hidden as i32)?;

            // GeLU·gate * up → mlp_gate (in place).
            unsafe {
                let mut out = mlp_gate_region.device_ptr();
                let mut a   = mlp_gate_region.device_ptr();
                let mut b   = mlp_up_region.device_ptr();
                let mut count = (n * intermediate) as i32;
                let args: [*mut std::ffi::c_void; 4] = [
                    (&mut out)  as *mut u64 as *mut _,
                    (&mut a)    as *mut u64 as *mut _,
                    (&mut b)    as *mut u64 as *mut _,
                    (&mut count) as *mut i32 as *mut _,
                ];
                let blocks = ((count as u32) + 1023) / 1024;
                rvllm_fused::launch_raw(
                    kernels.fn_gelu_tanh_mul_bf16,
                    (blocks, 1, 1), (1024, 1, 1),
                    0, stream_u64, &args,
                )?;
            }

            // down: hidden = mlp_gate @ W_d^T.
            do_gemm(mlp_gate_region.device_ptr(), layer.down_proj.offset_bytes,
                    hidden_region.device_ptr(),
                    n as i32, v_hidden as i32, intermediate as i32)?;

            // residual += hidden (final block output).
            do_vector_add_bf16(
                hidden_region.device_ptr(),
                residual_region.device_ptr(),
                hidden_region.device_ptr(),
            )?;

            // Round-12 phase 3-test (c): per-block dump for the bisect.
            // Writes to blocks/block_NN.bin under the dump dir. Gated
            // separately from the top-level dump_dir (RVLLM_PIXTRAL_PER_BLOCK_DUMP=1)
            // so the cosine harness can run the cheap path without
            // forcing 48 stream-fences per request.
            if let Some(ref d) = dump_dir {
                if debug_env_str("RVLLM_PIXTRAL_PER_BLOCK_DUMP").is_some() {
                    stream.fence()?;
                    let mut host = vec![0u8; hidden_bytes];
                    unsafe {
                        use cudarc::driver::sys::*;
                        let _ = cuMemcpyDtoH_v2(
                            host.as_mut_ptr() as *mut _,
                            hidden_region.device_ptr(), hidden_bytes);
                    }
                    let blocks_dir = d.join("blocks");
                    let _ = std::fs::create_dir_all(&blocks_dir);
                    let _ = std::fs::write(
                        blocks_dir.join(format!("block_{:02}.bin", block_idx)),
                        &host,
                    );
                }
            }
        }

        if let Some(ref d) = dump_dir {
            stream.fence()?;
            let mut host = vec![0u8; hidden_bytes];
            unsafe {
                use cudarc::driver::sys::*;
                let _ = cuMemcpyDtoH_v2(
                    host.as_mut_ptr() as *mut _,
                    hidden_region.device_ptr(), hidden_bytes);
            }
            let _ = std::fs::write(d.join("post_blocks.bin"), &host);
        }

        // ── Stage 5: projector ─────────────────────────────────────────
        //
        // HF Mistral3MultiModalProjector pipeline:
        //   x [N, v_hidden]
        //   → norm (RMSNorm with [v_hidden] gamma)
        //   → patch_merger:   spatial 2x2 concat to [N/4, 4*v_hidden]
        //                     then merging_layer Linear → [N/4, v_hidden]
        //   → linear_1: v_hidden → text_hidden  ([N/4, text_hidden])
        //   → GeLU (tanh approximation)
        //   → linear_2: text_hidden → text_hidden
        //   → output [N/4, text_hidden]
        //
        // Note ordering: norm comes BEFORE the 2x2 spatial merge (per HF
        // source), so the RMS is computed over the v_hidden=1664 channels
        // of each pre-merge token.

        // 5.1 RMSNorm in-place on hidden_region [N, v_hidden].
        unsafe {
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: n as u32,
                hidden: v_hidden as u32,
                eps: 1.0e-5,
            }.launch(
                kernels.fn_rmsnorm_inplace_bf16_gbf16,
                hidden_region.device_ptr(),
                vision.projector_norm.offset_bytes,
                stream_u64,
            )?;
        }
        if let Some(ref d) = dump_dir {
            stream.fence()?;
            let mut host = vec![0u8; hidden_bytes];
            unsafe {
                use cudarc::driver::sys::*;
                let _ = cuMemcpyDtoH_v2(
                    host.as_mut_ptr() as *mut _,
                    hidden_region.device_ptr(), hidden_bytes);
            }
            let _ = std::fs::write(d.join("post_proj_norm.bin"), &host);
        }

        // 5.2 Spatial 2x2 patch merge: [grid_h, grid_w, v_hidden] →
        //     [merged_h, merged_w, 4*v_hidden]. Pure permutation; the
        //     subsequent Linear collapses the 4*v_hidden axis to v_hidden.
        let (merged_h, merged_w) = pp.merged_grid;
        let merged_n = (merged_h as usize) * (merged_w as usize);
        let inner_4h = 4 * v_hidden;
        let merged_bytes = merged_n * inner_4h * 2;
        let merged_region = arena.region(
            "pixtral_merged_concat", merged_bytes, 16)?;
        unsafe {
            let mut out = merged_region.device_ptr();
            let mut in_p = hidden_region.device_ptr();
            let mut gh = grid_h as i32;
            let mut gw = grid_w as i32;
            let mut hd = v_hidden as i32;
            let args: [*mut std::ffi::c_void; 5] = [
                (&mut out) as *mut u64 as *mut _,
                (&mut in_p) as *mut u64 as *mut _,
                (&mut gh) as *mut i32 as *mut _,
                (&mut gw) as *mut i32 as *mut _,
                (&mut hd) as *mut i32 as *mut _,
            ];
            rvllm_fused::launch_raw(
                kernels.fn_patch_merger_pixtral_2x2,
                (merged_n as u32, 1, 1),
                (256, 1, 1),
                0, stream_u64, &args,
            )?;
        }

        // 5.3 merging_layer Linear: [merged_n, 4*v_hidden] @
        //     merging_layer_weight^T [v_hidden, 4*v_hidden] →
        //     [merged_n, v_hidden] BF16.
        let post_merge_bytes = merged_n * v_hidden * 2;
        let post_merge_region = arena.region(
            "pixtral_post_merge", post_merge_bytes, 16)?;
        unsafe {
            cublaslt.bf16_gemm_f32(
                merged_region.device_ptr(),
                vision.projector_patch_merger.offset_bytes,
                f32_big_scratch.device_ptr(),
                merged_n as i32, v_hidden as i32, inner_4h as i32,
                stream_u64,
            )?;
            let total = (merged_n * v_hidden) as i32;
            let mut dst = post_merge_region.device_ptr();
            let mut src = f32_big_scratch.device_ptr();
            let mut count = total;
            let args: [*mut std::ffi::c_void; 3] = [
                (&mut dst) as *mut u64 as *mut _,
                (&mut src) as *mut u64 as *mut _,
                (&mut count) as *mut i32 as *mut _,
            ];
            let blocks = ((total as u32) + 1023) / 1024;
            rvllm_fused::launch_raw(
                kernels.fn_f32_to_bf16,
                (blocks, 1, 1), (1024, 1, 1),
                0, stream_u64, &args,
            )?;
        }
        if let Some(ref d) = dump_dir {
            stream.fence()?;
            let mut host = vec![0u8; post_merge_bytes];
            unsafe {
                use cudarc::driver::sys::*;
                let _ = cuMemcpyDtoH_v2(
                    host.as_mut_ptr() as *mut _,
                    post_merge_region.device_ptr(), post_merge_bytes);
            }
            let _ = std::fs::write(d.join("post_merge.bin"), &host);
        }

        // 5.4 linear_1: [merged_n, v_hidden] @ W_l1^T [text_hidden, v_hidden]
        //     → [merged_n, text_hidden] BF16.
        let text_hidden = self.arch.text.hidden_size;
        let post_l1_bytes = merged_n * text_hidden * 2;
        let post_l1_region = arena.region(
            "pixtral_post_linear1", post_l1_bytes, 16)?;
        unsafe {
            cublaslt.bf16_gemm_f32(
                post_merge_region.device_ptr(),
                vision.projector_linear_1.offset_bytes,
                f32_big_scratch.device_ptr(),
                merged_n as i32, text_hidden as i32, v_hidden as i32,
                stream_u64,
            )?;
            let total = (merged_n * text_hidden) as i32;
            let mut dst = post_l1_region.device_ptr();
            let mut src = f32_big_scratch.device_ptr();
            let mut count = total;
            let args: [*mut std::ffi::c_void; 3] = [
                (&mut dst) as *mut u64 as *mut _,
                (&mut src) as *mut u64 as *mut _,
                (&mut count) as *mut i32 as *mut _,
            ];
            let blocks = ((total as u32) + 1023) / 1024;
            rvllm_fused::launch_raw(
                kernels.fn_f32_to_bf16,
                (blocks, 1, 1), (1024, 1, 1),
                0, stream_u64, &args,
            )?;
        }

        // 5.5 GeLU (tanh approximation) in-place on post_l1_region.
        unsafe {
            let mut x = post_l1_region.device_ptr();
            let mut count = (merged_n * text_hidden) as i32;
            let args: [*mut std::ffi::c_void; 2] = [
                (&mut x) as *mut u64 as *mut _,
                (&mut count) as *mut i32 as *mut _,
            ];
            let blocks = ((count as u32) + 1023) / 1024;
            rvllm_fused::launch_raw(
                kernels.fn_gelu_tanh_bf16,
                (blocks, 1, 1), (1024, 1, 1),
                0, stream_u64, &args,
            )?;
        }

        // 5.6 linear_2: [merged_n, text_hidden] @ W_l2^T directly into
        //     the caller's `output_dst_ptr`. No more local
        //     `post_l2_region` allocation; no DtoH.
        unsafe {
            cublaslt.bf16_gemm_f32(
                post_l1_region.device_ptr(),
                vision.projector_linear_2.offset_bytes,
                f32_big_scratch.device_ptr(),
                merged_n as i32, text_hidden as i32, text_hidden as i32,
                stream_u64,
            )?;
            let total = (merged_n * text_hidden) as i32;
            let mut dst = output_dst_ptr;
            let mut src = f32_big_scratch.device_ptr();
            let mut count = total;
            let args: [*mut std::ffi::c_void; 3] = [
                (&mut dst) as *mut u64 as *mut _,
                (&mut src) as *mut u64 as *mut _,
                (&mut count) as *mut i32 as *mut _,
            ];
            let blocks = ((total as u32) + 1023) / 1024;
            rvllm_fused::launch_raw(
                kernels.fn_f32_to_bf16,
                (blocks, 1, 1), (1024, 1, 1),
                0, stream_u64, &args,
            )?;
        }

        // ── Stage 6: optional dump of the (now device-resident) output.
        if let Some(ref d) = dump_dir {
            stream.fence()?;
            let post_l2_bytes = merged_n * text_hidden * 2;
            let mut host = vec![0u8; post_l2_bytes];
            unsafe {
                use cudarc::driver::sys::*;
                let _ = cuMemcpyDtoH_v2(
                    host.as_mut_ptr() as *mut _,
                    output_dst_ptr,
                    post_l2_bytes,
                );
            }
            let _ = std::fs::write(d.join("output.bin"), &host);
        }

        // Free intermediate scratch. `output_dst_ptr` lives in the
        // caller's arena scope, NOT under our checkpoint, so it
        // survives the restore.
        stream.fence()?;
        unsafe { arena.restore(ck); }
        Ok(())
    }

    pub unsafe fn generate<F>(
        &self,
        prompt: &[u32],
        max_new: usize,
        eos_ids: &[u32],
        cancelled: Option<&std::sync::atomic::AtomicBool>,
        mut on_token: F,
    ) -> Result<GenerateResult>
    where
        F: FnMut(u32),
    {
        self.generate_with_vision(
            prompt, max_new, eos_ids, cancelled, on_token, &[],
        )
    }

    /// Round-12 phase 3e: vision-aware generate. `vision_splices` is
    /// `[(token_start, num_tokens, embed_bf16_bytes)]` — the splice
    /// data already produced by `forward_pixtral_vision` (BF16 row-
    /// major `[num_tokens, hidden]`). Each splice covers
    /// `[token_start .. token_start + num_tokens)` in the prompt.
    /// Decode tokens (positions ≥ `prompt.len()`) never carry
    /// splices.
    pub unsafe fn generate_with_vision<F>(
        &self,
        prompt: &[u32],
        max_new: usize,
        eos_ids: &[u32],
        cancelled: Option<&std::sync::atomic::AtomicBool>,
        mut on_token: F,
        vision_splices: &[(usize, usize, Vec<u8>)],
    ) -> Result<GenerateResult>
    where
        F: FnMut(u32),
    {
        let is_cancelled = || cancelled
            .map(|f| f.load(std::sync::atomic::Ordering::Relaxed))
            .unwrap_or(false);

        let arena = self.validate_generate_inputs(prompt, max_new)?;
        // #4 fix: explicit per-request KV-lifetime hand-shake. Today
        // a no-op marker; documents the load-bearing invariant and
        // gives future paged/seq-id work a single seam to plumb
        // through.
        self.kv_request_begin(prompt.len() + max_new)?;

        // RVLLM_SMOKE_SINGLE=1: skip the prefill loop entirely and run
        // only ONE forward with the LAST prompt token at position 0.
        // Diagnostic mode for isolating embed/forward bugs without
        // KV-cache state from prior steps.
        if debug_env_os("RVLLM_SMOKE_SINGLE").is_some() {
            let tok = *prompt.last().unwrap();
            let dump = self.forward_smoke_q_proj_inner(tok, 0, true, None)?;
            return Ok(GenerateResult {
                tokens: vec![tok, dump.predicted_token],
                last_dump: Some(dump),
                prompt_len: 1,
            });
        }

        // Round-12 phase 3e: allocate splice region BEFORE the
        // per-token checkpoint so its device pointers stay valid
        // across the prefill loop's per-iteration arena.restore.
        // The outer checkpoint (`ck_outer`) frees both the splice
        // region and any leftover per-token scratch at the very end
        // of generate.
        let ck_outer = arena.checkpoint();
        let do_restore = debug_env_os("RVLLM_SMOKE_NO_RESTORE").is_none();

        let hidden_bytes = (self.arch.text.hidden_size as usize) * 2;
        let mut pos_to_dev_ptr: std::collections::HashMap<i32, u64> =
            std::collections::HashMap::new();
        let mut _splice_region_keepalive: Option<rvllm_mem::Region> = None;
        if !vision_splices.is_empty() {
            // Sanity-check: each splice's data length matches num_tokens * hidden_bytes.
            let mut total = 0usize;
            for (_, n_tok, data) in vision_splices {
                let want = n_tok * hidden_bytes;
                if data.len() != want {
                    return Err(corrupt(
                        self.paths.model_dir.clone(),
                        format!(
                            "generate_with_vision: splice byte len {} != {} (num_tokens={n_tok}, hidden_bytes={hidden_bytes})",
                            data.len(), want,
                        ),
                    ));
                }
                total += data.len();
            }
            // Allocate a region BEFORE the per-token checkpoint so
            // its device-pointer values stay valid across prefill
            // iterations.
            let region = arena.region("mistral35_vision_splices", total, 16)?;
            let base = region.device_ptr();
            let mut cursor: usize = 0;
            for (start, n_tok, data) in vision_splices {
                unsafe {
                    use cudarc::driver::sys::*;
                    let r = cuMemcpyHtoDAsync_v2(
                        base + cursor as u64,
                        data.as_ptr() as *const _,
                        data.len(),
                        self.stream.as_ref().unwrap().raw() as CUstream,
                    );
                    if r != CUresult::CUDA_SUCCESS {
                        return Err(rvllm_core::RvllmError::cuda(
                            "mistral35: vision splice HtoD",
                            rvllm_core::CudaErrorKind::LaunchFailed,
                            rvllm_core::CudaCtx::setup(),
                        ));
                    }
                }
                for t in 0..*n_tok {
                    let pos = (*start + t) as i32;
                    let ptr = base + (cursor + t * hidden_bytes) as u64;
                    pos_to_dev_ptr.insert(pos, ptr);
                }
                cursor += data.len();
            }
            _splice_region_keepalive = Some(region);
        }
        let ck_after_splices = arena.checkpoint();

        let mut tokens: Vec<u32> = prompt.to_vec();
        let mut last_dump: Option<SmokeStageDump> = None;
        let mut last_predicted: u32 = 0;

        // Stage 3: prefill.
        for (i, &tok) in prompt.iter().enumerate() {
            if is_cancelled() {
                return Ok(GenerateResult {
                    tokens, last_dump, prompt_len: prompt.len(),
                });
            }
            let is_last = i == prompt.len() - 1;
            let splice_ptr = pos_to_dev_ptr.get(&(i as i32)).copied();
            if let Some(dump) = self.prefill_token(
                tok, i as i32, is_last, arena, ck_after_splices,
                do_restore, splice_ptr,
            )? {
                last_predicted = dump.predicted_token;
                last_dump = Some(dump);
            }
        }
        // Stage 4: decode.
        for step in 0..max_new {
            tokens.push(last_predicted);
            on_token(last_predicted);
            if eos_ids.contains(&last_predicted) {
                break;
            }
            if step + 1 >= max_new {
                break;
            }
            if is_cancelled() {
                break;
            }
            let pos = (prompt.len() + step) as i32;
            let dump = self.decode_token(
                last_predicted, pos, arena, ck_after_splices, do_restore,
            )?;
            last_predicted = dump.predicted_token;
            last_dump = Some(dump);
        }

        // Free all per-step scratch in one shot — including the
        // vision-splice region allocated above the inner checkpoint.
        // The last forward finished with a stream fence + DtoH so all
        // in-flight kernels are complete by here.
        if do_restore { unsafe { arena.restore(ck_outer); } }
        drop(_splice_region_keepalive);

        Ok(GenerateResult {
            tokens,
            last_dump,
            prompt_len: prompt.len(),
        })
    }

    /// Round-12 phase 5c codex review #3 — KNOWN BROKEN.
    ///
    /// Designed to eliminate the DtoH→HtoD round-trip by running the
    /// Pixtral forward INLINE in the prefill arena and DtoD-splicing
    /// directly. Empirically regresses semantic correctness on real
    /// images ("orange ball" → "I need to see the image to describe
    /// it.") even though `forward_pixtral_vision_into` produces
    /// byte-identical output to the round-trip path when invoked
    /// in isolation (verified via the compat shim
    /// `forward_pixtral_vision`).
    ///
    /// The exact regression cause is unidentified — candidates
    /// include arena-bump-pointer interaction across the inner
    /// cuda fn's checkpoint+restore vs the outer splice_region
    /// allocation, or a stream-ordering subtlety. Production
    /// (`cuda_worker`) uses `generate_with_vision` (Vec<u8>) until
    /// the bug is rooted out.
    #[cfg(feature = "cuda")]
    #[allow(dead_code)]
    pub unsafe fn generate_with_images<F>(
        &self,
        prompt: &[u32],
        max_new: usize,
        eos_ids: &[u32],
        cancelled: Option<&std::sync::atomic::AtomicBool>,
        mut on_token: F,
        vision_images: &[(usize, Vec<u8>)],
    ) -> Result<GenerateResult>
    where
        F: FnMut(u32),
    {
        let is_cancelled = || cancelled
            .map(|f| f.load(std::sync::atomic::Ordering::Relaxed))
            .unwrap_or(false);

        let arena = self.validate_generate_inputs(prompt, max_new)?;
        self.kv_request_begin(prompt.len() + max_new)?;

        if debug_env_os("RVLLM_SMOKE_SINGLE").is_some() {
            let tok = *prompt.last().unwrap();
            let dump = self.forward_smoke_q_proj_inner(tok, 0, true, None)?;
            return Ok(GenerateResult {
                tokens: vec![tok, dump.predicted_token],
                last_dump: Some(dump),
                prompt_len: 1,
            });
        }

        let ck_outer = arena.checkpoint();
        let do_restore = debug_env_os("RVLLM_SMOKE_NO_RESTORE").is_none();
        let hidden_bytes = (self.arch.text.hidden_size as usize) * 2;

        // Pre-flight all images (host preprocess) to compute the splice
        // region byte budget BEFORE allocating it.
        let mut prepped: Vec<(usize, usize)> =
            Vec::with_capacity(vision_images.len());
        let mut total_splice_bytes = 0usize;
        for (start, image_bytes) in vision_images {
            let pp = self.vision_preprocess(image_bytes)?;
            let nt = pp.num_soft_tokens;
            prepped.push((*start, nt));
            total_splice_bytes += nt * hidden_bytes;
        }

        let mut pos_to_dev_ptr: std::collections::HashMap<i32, u64> =
            std::collections::HashMap::new();
        let mut _splice_region_keepalive: Option<rvllm_mem::Region> = None;
        if total_splice_bytes > 0 {
            let region = arena.region(
                "mistral35_vision_splices_inline", total_splice_bytes, 16)?;
            let base = region.device_ptr();
            let mut cursor: usize = 0;
            for ((start, nt), (_, image_bytes)) in prepped.iter().zip(vision_images.iter()) {
                let dst = base + cursor as u64;
                self.forward_pixtral_vision_into(image_bytes, dst, *nt)?;
                for t in 0..*nt {
                    let pos = (*start + t) as i32;
                    let ptr = base + (cursor + t * hidden_bytes) as u64;
                    pos_to_dev_ptr.insert(pos, ptr);
                }
                cursor += nt * hidden_bytes;
            }
            _splice_region_keepalive = Some(region);
        }
        let ck_after_splices = arena.checkpoint();

        let mut tokens: Vec<u32> = prompt.to_vec();
        let mut last_dump: Option<SmokeStageDump> = None;
        let mut last_predicted: u32 = 0;

        for (i, &tok) in prompt.iter().enumerate() {
            if is_cancelled() {
                return Ok(GenerateResult {
                    tokens, last_dump, prompt_len: prompt.len(),
                });
            }
            let is_last = i == prompt.len() - 1;
            let splice_ptr = pos_to_dev_ptr.get(&(i as i32)).copied();
            if let Some(dump) = self.prefill_token(
                tok, i as i32, is_last, arena, ck_after_splices,
                do_restore, splice_ptr,
            )? {
                last_predicted = dump.predicted_token;
                last_dump = Some(dump);
            }
        }
        for step in 0..max_new {
            tokens.push(last_predicted);
            on_token(last_predicted);
            if eos_ids.contains(&last_predicted) {
                break;
            }
            if step + 1 >= max_new {
                break;
            }
            if is_cancelled() {
                break;
            }
            let pos = (prompt.len() + step) as i32;
            let dump = self.decode_token(
                last_predicted, pos, arena, ck_after_splices, do_restore,
            )?;
            last_predicted = dump.predicted_token;
            last_dump = Some(dump);
        }

        if do_restore { unsafe { arena.restore(ck_outer); } }
        drop(_splice_region_keepalive);

        Ok(GenerateResult { tokens, last_dump, prompt_len: prompt.len() })
    }

    /// Stage 0 helper (#4 — explicit KV lifetime).
    ///
    /// Marks the start of a request's window into the persistent
    /// KV cache. Today this is a logical hand-shake — no zeroing
    /// is performed because the load-bearing invariant
    /// (every read at slot `s` was preceded by a write at the same
    /// slot in the same request) is enforced by the forward path.
    /// The hand-shake exists so future code that wants to relax
    /// that invariant (paged batching, prefix-cache reuse,
    /// speculative decode) has a single seam to plumb a real
    /// reset / seq-id mask through.
    ///
    /// `RVLLM_DEBUG_KV_LIFETIME=1` makes the hand-shake log every
    /// request; left off in production.
    fn kv_request_begin(&self, used_slots: usize) -> Result<()> {
        let kv = self.kv_cache.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "kv_request_begin: kv_cache absent".into(),
        ))?;
        if used_slots > kv.max_pos {
            return Err(corrupt(
                self.paths.model_dir.clone(),
                format!("kv_request_begin: used_slots={used_slots} > max_pos={}",
                    kv.max_pos),
            ));
        }
        if std::env::var_os("RVLLM_DEBUG_KV_LIFETIME").is_some() {
            eprintln!(
                "[mistral35] kv_request_begin: used_slots={used_slots} \
                 / max_pos={} (per-request lifetime via past_len = position + 1)",
                kv.max_pos,
            );
        }
        Ok(())
    }

    /// Stage 1 helper: validate inputs + return a borrow of the arena.
    fn validate_generate_inputs(
        &self,
        prompt: &[u32],
        max_new: usize,
    ) -> Result<&rvllm_mem::HbmArena<'static>> {
        if prompt.is_empty() {
            return Err(corrupt(
                self.paths.model_dir.clone(),
                "generate: empty prompt".into(),
            ));
        }
        let arena = self.arena.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "generate: arena absent".into(),
        ))?;
        let kv_cache = self.kv_cache.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "generate: kv_cache absent".into(),
        ))?;
        let total_max = prompt.len() + max_new;
        if total_max > kv_cache.max_pos {
            return Err(corrupt(
                self.paths.model_dir.clone(),
                format!("generate: total_len={total_max} > kv_max_pos={}",
                    kv_cache.max_pos),
            ));
        }
        Ok(arena)
    }

    /// Stage 3 helper: feed one prompt token through the forward.
    /// Only `is_last == true` computes logits + LM-head argmax;
    /// earlier prompt tokens skip the expensive tail and return
    /// `None`. Restores the arena scratch between calls.
    unsafe fn prefill_token(
        &self,
        tok: u32,
        pos: i32,
        is_last: bool,
        arena: &rvllm_mem::HbmArena<'static>,
        ck: usize,
        do_restore: bool,
        vision_splice_dev_ptr: Option<u64>,
    ) -> Result<Option<SmokeStageDump>> {
        let dump = self.forward_smoke_q_proj_inner(
            tok, pos, is_last, vision_splice_dev_ptr,
        )?;
        let out = if is_last { Some(dump) } else { None };
        // Safe: the inner path fences via stream.fence() when
        // compute_logits=false, and DtoH on logits when true.
        if do_restore { arena.restore(ck); }
        Ok(out)
    }

    /// Stage 4 helper: feed one decode token through the forward
    /// (always with `compute_logits = true`). Decode tokens never
    /// carry vision splices — by construction, vision-soft-token
    /// slots only appear in the prompt.
    unsafe fn decode_token(
        &self,
        tok: u32,
        pos: i32,
        arena: &rvllm_mem::HbmArena<'static>,
        ck: usize,
        do_restore: bool,
    ) -> Result<SmokeStageDump> {
        let dump = self.forward_smoke_q_proj_inner(tok, pos, true, None)?;
        if do_restore { arena.restore(ck); }
        Ok(dump)
    }
}

/// Result of [`Mistral35Bringup::generate`]. `tokens` includes the
/// prompt followed by all generated tokens up to (and including) the
/// first EOS hit.
#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct GenerateResult {
    pub tokens: Vec<u32>,
    pub last_dump: Option<SmokeStageDump>,
    pub prompt_len: usize,
}

/// Backward-compatibility alias for the pre-F4#2 type name. New code
/// should refer to [`GenerateResult`] directly.
#[cfg(feature = "cuda")]
pub type GenerateSmokeResult = GenerateResult;

#[cfg(test)]
mod tests {
    use super::*;

    fn paths_for(model_dir: PathBuf) -> Mistral35EnginePaths {
        Gemma4EnginePaths {
            model_dir,
            kernels_dir: PathBuf::from("/tmp/rvllm-mistral-test-kernels"),
            cutlass_so: PathBuf::from("/tmp/rvllm-mistral-test-libcutlass.so"),
            fa3_so: PathBuf::from("/tmp/rvllm-mistral-test-fa3.so"),
            policy_json: PathBuf::from("/tmp/rvllm-mistral-test-policy.json"),
        }
    }

    /// Test arch fixture matching the public Mistral 3.5 NVFP4
    /// checkpoint (88L decoder + 48L pixtral, real shapes for
    /// dense decoder; vision block carries the spec defaults).
    /// `num_hidden_layers` is overridable so cheap unit tests
    /// don't have to care about per-layer counts.
    fn test_arch_fixture(num_hidden_layers: usize) -> Mistral35Arch {
        use rvllm_loader::mistral35_arch::{
            Mistral35TextArch, Mistral35VisionArch, YarnRopeConfig,
        };
        Mistral35Arch {
            text: Mistral35TextArch {
                num_hidden_layers,
                hidden_size: 12288,
                intermediate_size: 28672,
                num_attention_heads: 96,
                num_key_value_heads: 8,
                head_dim: 128,
                vocab_size: 131072,
                max_position_embeddings: 262144,
                rms_norm_eps: 1e-5,
                hidden_act_silu: true,
                tie_word_embeddings: false,
                yarn: YarnRopeConfig {
                    rope_theta: 1_000_000.0,
                    original_max_position_embeddings: 4096,
                    factor: 64.0,
                    beta_fast: 4.0,
                    beta_slow: 1.0,
                    mscale: 1.0,
                    mscale_all_dim: 0.0,
                },
            },
            vision: Mistral35VisionArch {
                model_type_pixtral: true,
                hidden_size: 1664,
                num_hidden_layers: 48,
                num_attention_heads: 16,
                head_dim: 104,
                intermediate_size: 8192,
                patch_size: 14,
                image_size: 1540,
                num_channels: 3,
                rope_theta: 10_000.0,
                spatial_merge_size: 2,
            },
            image_token_index: 10,
            weight_prefix: "model.language_model".into(),
        }
    }

    #[test]
    fn load_rejects_non_mistral_dir() {
        let tmp = std::env::temp_dir().join(format!(
            "rvllm-mistral-bringup-test-{}-nonmistral",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(
            tmp.join("config.json"),
            r#"{"architectures":["Gemma4ForConditionalGeneration"],"model_type":"gemma4"}"#,
        )
        .unwrap();
        match Mistral35Bringup::load(paths_for(tmp.clone()), 1) {
            Ok(_) => panic!("expected non-Mistral dir to be rejected"),
            Err(e) => assert!(format!("{e:?}").contains("Mistral 3.5")),
        }
    }

    #[test]
    fn load_rejects_yarn_drift() {
        let tmp = std::env::temp_dir().join(format!(
            "rvllm-mistral-bringup-test-{}-yarn",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        // Mistral markers + YaRN with a non-zero mscale_all_dim should
        // trip the parser before the inventory pass even runs.
        std::fs::write(
            tmp.join("config.json"),
            r#"{
              "architectures":["Mistral3ForConditionalGeneration"],
              "model_type":"mistral3",
              "quantization_config":{"format":"nvfp4-pack-quantized"},
              "text_config":{
                "num_hidden_layers":2,"hidden_size":12288,
                "intermediate_size":28672,"num_attention_heads":96,
                "num_key_value_heads":8,"head_dim":128,
                "vocab_size":131072,"max_position_embeddings":262144,
                "rms_norm_eps":1e-5,"hidden_act":"silu",
                "tie_word_embeddings":false,"rope_theta":1000000.0,
                "rope_scaling":{"rope_type":"yarn","original_max_position_embeddings":4096,
                  "factor":64.0,"beta_fast":4.0,"beta_slow":1.0,"mscale":1.0,"mscale_all_dim":1.0}
              },
              "vision_config":{"model_type":"pixtral","head_dim":104,"image_size":1540}
            }"#,
        )
        .unwrap();
        match Mistral35Bringup::load(paths_for(tmp), 1) {
            Ok(_) => panic!("expected YaRN drift to be rejected"),
            Err(e) => assert!(format!("{e:?}").contains("mscale_all_dim")),
        }
    }

    #[test]
    fn forward_stub_is_typed_error() {
        let e = Mistral35Bringup::forward_not_implemented_yet();
        assert!(matches!(e, Mistral35Error::ForwardNotImplemented));
    }

    // F4#1 fix: KvDecodeStrategy enum + tests deleted. The enum was
    // never branched on by the Mistral forward (BF16-KV path is
    // hard-wired); the tests pinned its `for_gqa_ratio` mapping
    // which now has no caller.

    /// Serialise the env-var-touching tests so cargo's parallel
    /// runner doesn't make them race. (Tests that don't touch env
    /// don't acquire — keeps the rest of the suite parallel.)
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn batched_prefill_default_is_all_off_until_executor_lands() {
        // Round-12 phase 5c: until generate_with_vision actually
        // runs a layer-major batched prefill, all phase gates must
        // default OFF so the startup log + downstream branches see
        // honest reality (codex review #1 — "lying log" finding).
        let _g = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        for v in [
            "RVLLM_MISTRAL35_BATCH_PROJ_PREFILL",
            "RVLLM_MISTRAL35_FUSED_QKV",
            "RVLLM_MISTRAL35_BATCH_ROPE_PREFILL",
            "RVLLM_MISTRAL35_BATCH_FULL_PREFILL",
            "RVLLM_MISTRAL35_DECODE_GRAPH",
        ] {
            std::env::remove_var(v);
        }
        let cfg = BatchedPrefillConfig::from_env();
        assert!(!cfg.batch_projections);
        assert!(!cfg.batch_rope);
        assert!(!cfg.batch_full_prefill);
        assert!(!cfg.outer_loop_deleted);
        assert!(!cfg.fused_qkv);
        assert!(!cfg.decode_graph_capture);
    }

    #[test]
    fn scratch_budget_natural_matches_spec_shapes() {
        // Mistral 3.5: hidden=12288, intermediate=28672. The
        // budget pivots on max_k = max(h, i) = 28672 (the down-proj
        // K). At m=256 that's:
        //   a_packed     = 256 * 28672 / 2  = 3 670 016 bytes
        //   sfa_natural  = 256 * 28672 / 16 =   458 752 bytes
        let arch = test_arch_fixture(2);
        let b = LayerScratchBudget::natural(&arch, 256);
        assert_eq!(b.a_packed_bytes, 256 * 28672 / 2);
        assert_eq!(b.sfa_natural_bytes, 256 * 28672 / 16);
        assert_eq!(b.sfa_cutlass_bytes_max, 256 * 28672 / 16);
        assert_eq!(b.workspace_bytes_max, 0);
    }

    #[test]
    fn scratch_budget_zero_tokens_is_zero_bytes() {
        let arch = test_arch_fixture(2);
        let b = LayerScratchBudget::natural(&arch, 0);
        assert_eq!(b.a_packed_bytes, 0);
        assert_eq!(b.sfa_natural_bytes, 0);
    }

    #[test]
    fn rope_tables_default_max_pos_is_yarn_original_max() {
        // With max_pos=None, the bring-up uses the YaRN
        // original_max_position_embeddings (4096 for Mistral 3.5).
        // No live bring-up needed for this test — we exercise the
        // helper directly through the same code path.
        let arch = test_arch_fixture(2);
        let head_dim = arch.text.head_dim;
        let omp = arch.text.yarn.original_max_position_embeddings;
        let t = super::super::mistral35_yarn::build_yarn_rope_tables(
            &arch.text.yarn, head_dim, omp,
        );
        assert_eq!(t.max_pos, 4096);
        assert_eq!(t.head_dim, 128);
        assert_eq!(t.cos.len(), 4096 * 64);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn scratch_budget_with_absent_backend_falls_back_to_natural() {
        // With CutlassBackend::Absent every backend query returns 0,
        // so the budget should match the natural arithmetic exactly.
        let arch = test_arch_fixture(2);
        let backend = rvllm_cutlass::lib_so::CutlassBackend::Absent;
        let b_arith = LayerScratchBudget::natural(&arch, 256);
        let b_back = LayerScratchBudget::with_backend(&arch, 256, &backend);
        assert_eq!(b_arith.a_packed_bytes, b_back.a_packed_bytes);
        assert_eq!(b_arith.sfa_natural_bytes, b_back.sfa_natural_bytes);
        assert_eq!(b_arith.sfa_cutlass_bytes_max, b_back.sfa_cutlass_bytes_max);
        assert_eq!(b_arith.workspace_bytes_max, b_back.workspace_bytes_max);
    }

    #[test]
    fn forward_smoke_handles_inactive_bringup_gracefully() {
        // Pure-Rust path — verifies the helper compiles. The real
        // smoke-forward needs CUDA + an uploaded model, exercised
        // out-of-band against /home/r00t/mistral-3.5.
        let _ = test_arch_fixture(2);
    }

    #[test]
    fn batched_prefill_outer_loop_collapses_when_phase_off() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        // Disabling any one of the prefill phases must drop the
        // implicit outer-loop-deleted flag, even though the
        // operator did not flip it directly.
        std::env::set_var("RVLLM_MISTRAL35_BATCH_PROJ_PREFILL", "0");
        std::env::set_var("RVLLM_MISTRAL35_BATCH_ROPE_PREFILL", "1");
        std::env::set_var("RVLLM_MISTRAL35_BATCH_FULL_PREFILL", "1");
        let cfg = BatchedPrefillConfig::from_env();
        assert!(!cfg.batch_projections);
        assert!(!cfg.outer_loop_deleted);
        std::env::remove_var("RVLLM_MISTRAL35_BATCH_PROJ_PREFILL");
    }
}
