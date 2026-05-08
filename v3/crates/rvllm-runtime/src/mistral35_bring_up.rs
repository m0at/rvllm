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

/// Which NVFP4-KV decode kernel can serve a given GQA ratio.
///
/// The existing NVFP4-KV decode kernel set has compile-time caps:
///
/// - `kernels/flash_attention_nvfp4kv.cu::MAX_GQA_DECODE = 4`
///   (also `..._bf16out.cu`) — fused per-`(seq, kv_head)` decode that
///   loads K/V exactly once per tile and computes Q·Kᵀ for all q-heads
///   sharing that kv-head. Gemma 4 sliding (GQA=2) fits.
/// - `kernels/flash_attention_split_decode_nvfp4kv.cu::MAX_GQA_SPLIT = 8`
///   (also `..._bf16out.cu`) — paged_attention_v2-style split decode.
///   Qwen 3.6 (GQA=8) fits.
/// - Per-head fallback (`flash_attention_2_decode_nvfp4kv_kernel`) — one
///   block per `(seq, head)`. Loads K/V once per tile per Q-HEAD
///   instead of per (seq, kv_head); ~`gqa_ratio×` bandwidth waste vs
///   the fused path, but works for any GQA.
///
/// Mistral 3.5 has GQA=12. The kernel-side fix is to raise both
/// constants to ≥12 (q_reg / row_max / acc / s_score allocations
/// scale linearly; sm_121 has the register + smem headroom). Until
/// that lands, Mistral routes through the per-head fallback —
/// correct, just slower. The gate lives here so a future kernel
/// rebuild flips the strategy without changing the runtime.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum KvDecodeStrategy {
    /// `flash_attention_2_decode_nvfp4kv_gqa_kernel` — fused
    /// per-(seq, kv_head), MAX_GQA_DECODE=4. Tightest path.
    FusedGqa,
    /// `paged_attention_v2_split_decode_nvfp4kv_kernel` —
    /// MAX_GQA_SPLIT=8. Used for Qwen 3.6's GQA=8.
    SplitDecode,
    /// `flash_attention_2_decode_nvfp4kv_kernel` — one block per
    /// (seq, head). Works for any GQA at the cost of `gqa_ratio×`
    /// duplicated K/V load. Mistral 3.5 (GQA=12) currently routes
    /// here.
    PerHeadFallback,
}

impl KvDecodeStrategy {
    /// Pick a strategy from the model's GQA ratio. The per-head
    /// fallback is universal; we prefer the fused / split paths
    /// whenever the kernel cap covers the requested ratio.
    pub fn for_gqa_ratio(gqa_ratio: usize) -> Self {
        // Match the kernel-side caps exactly. Bumping these here
        // without bumping MAX_GQA_DECODE / MAX_GQA_SPLIT in the .cu
        // sources would silently route Mistral through a kernel
        // that returns early on `GQA > MAX_*`.
        const MAX_GQA_DECODE_FUSED: usize = 4;
        const MAX_GQA_SPLIT: usize = 8;

        if gqa_ratio == 0 {
            return Self::PerHeadFallback;
        }
        if gqa_ratio <= MAX_GQA_DECODE_FUSED {
            Self::FusedGqa
        } else if gqa_ratio <= MAX_GQA_SPLIT {
            Self::SplitDecode
        } else {
            Self::PerHeadFallback
        }
    }
}

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
    /// Decode-strategy gate. Surfaced at startup so the operator
    /// log line records exactly which path each request will take.
    pub fn kv_decode_strategy(&self) -> KvDecodeStrategy {
        let q = self.arch.text.num_attention_heads;
        let kv = self.arch.text.num_key_value_heads;
        if kv == 0 {
            return KvDecodeStrategy::PerHeadFallback;
        }
        KvDecodeStrategy::for_gqa_ratio(q / kv)
    }

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
        // Production default once the kernels land: every prefill
        // gate ON, decode-graph still off. Until then `load()`
        // ignores the config because `run_generate` is a stub.
        Self {
            batch_projections: true,
            fused_qkv: false,
            batch_rope: true,
            batch_full_prefill: true,
            outer_loop_deleted: true,
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
        let batch_projections = on("RVLLM_MISTRAL35_BATCH_PROJ_PREFILL", true);
        let fused_qkv = on("RVLLM_MISTRAL35_FUSED_QKV", false);
        let batch_rope = on("RVLLM_MISTRAL35_BATCH_ROPE_PREFILL", true);
        let batch_full_prefill = on("RVLLM_MISTRAL35_BATCH_FULL_PREFILL", true);
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
        let (ctx, stream, arena, model, forward_kernels_opt, cublaslt_opt) = {
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

            let backend_ref = cutlass_backend.as_ref().unwrap();
            eprintln!("[mistral35-load] starting weight upload (this can take ~30-90s)…");
            let t0 = std::time::Instant::now();
            let model = crate::mistral35_load::load_mistral35_model(
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
            };

            (Some(ctx), Some(stream), Some(arena_box), Some(model),
             Some(forward_kernels), Some(cublaslt_handle))
        };

        #[cfg(feature = "cuda")]
        let bringup = Self {
            paths, arch, inventory, arena_bytes, nvfp4_active,
            cutlass_backend, ctx, stream, arena, model,
            forward_kernels: forward_kernels_opt,
            cublaslt: cublaslt_opt,
        };
        #[cfg(not(feature = "cuda"))]
        let bringup = Self { paths, arch, inventory, arena_bytes, nvfp4_active };
        let strategy = bringup.kv_decode_strategy();
        eprintln!(
            "[mistral35] kv_decode_strategy={:?} (gqa_ratio={})",
            strategy,
            bringup.arch.gqa_ratio()
        );
        if matches!(strategy, KvDecodeStrategy::PerHeadFallback) {
            eprintln!(
                "[mistral35] note: Mistral 3.5's GQA=12 exceeds the existing \
                 NVFP4-KV fused (MAX_GQA_DECODE=4) and split-decode \
                 (MAX_GQA_SPLIT=8) kernel caps; per-head fallback is \
                 correct but ~12x duplicated K/V load. Raise both .cu \
                 constants to >=12 to flip onto the fused path."
            );
        }
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
        let prefill = bringup.batched_prefill_config();
        eprintln!(
            "[mistral35] batched-prefill: proj={} fused_qkv={} rope={} \
             full_prefill={} outer_loop_deleted={} decode_graph={}",
            prefill.batch_projections,
            prefill.fused_qkv,
            prefill.batch_rope,
            prefill.batch_full_prefill,
            prefill.outer_loop_deleted,
            prefill.decode_graph_capture,
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
    backend.require_nvfp4()?;
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
    pub unsafe fn forward_smoke_q_proj(&self, token_id: u32) -> Result<SmokeStageDump> {
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

        // Scratch allocations for the full layer-0 forward.
        let token_region = arena.region("smoke_token_id", 4, 4)?;
        token_region.copy_from_host(&token_id.to_le_bytes())?;

        let hidden_bytes = (hidden as usize) * 2; // BF16
        // h_residual: persistent residual stream. Receives the embed,
        // then accumulates attn_out and mlp_out via vector_add_bf16.
        let h_residual_region = arena.region("smoke_h_residual", hidden_bytes, 16)?;
        // h_work: scratch where each pre-projection RMSNorm runs.
        // Refilled from h_residual via DtoD copy at each norm site.
        let h_work_region = arena.region("smoke_h_work", hidden_bytes, 16)?;

        // Activation prep scratch — worst-case K = max(hidden, intermediate)
        // (intermediate=28672 for Mistral 3.5 down_proj).
        let max_k = h_k.max(i_k);
        let a_packed_bytes = (m as usize) * (max_k as usize) / 2;
        let a_packed_region = arena.region("smoke_a_packed", a_packed_bytes, 16)?;
        let sfa_natural_bytes = backend.nvfp4_natural_sfa_bytes(m, max_k);
        let sfa_natural_region = arena.region("smoke_sfa_natural", sfa_natural_bytes, 16)?;
        let sfa_cutlass_bytes = backend.nvfp4_sfa_bytes(m, max_k);
        let sfa_cutlass_region = arena.region("smoke_sfa_cutlass", sfa_cutlass_bytes, 16)?;

        // Per-projection output regions. Sized to the projection's N.
        let q_out_region = arena.region("smoke_q_out_bf16",
            (layer0.q_proj.shape.n) * 2, 16)?;
        let k_out_region = arena.region("smoke_k_out_bf16",
            (layer0.k_proj.shape.n) * 2, 16)?;
        let v_out_region = arena.region("smoke_v_out_bf16",
            (layer0.v_proj.shape.n) * 2, 16)?;
        // attn_out: GQA-broadcast V to [hidden=12288] for o_proj input.
        let attn_out_region = arena.region("smoke_attn_out_bf16",
            (hidden as usize) * 2, 16)?;
        let o_out_region = arena.region("smoke_o_out_bf16",
            (layer0.o_proj.shape.n) * 2, 16)?;
        let gate_out_region = arena.region("smoke_gate_out_bf16",
            (layer0.gate_proj.shape.n) * 2, 16)?;
        let up_out_region = arena.region("smoke_up_out_bf16",
            (layer0.up_proj.shape.n) * 2, 16)?;
        // silu_mid: silu(gate)*up, [intermediate=28672].
        let silu_mid_region = arena.region("smoke_silu_mid_bf16",
            (layer0.gate_proj.shape.n) * 2, 16)?;
        let down_out_region = arena.region("smoke_down_out_bf16",
            (layer0.down_proj.shape.n) * 2, 16)?;

        // Workspace — size to the max across all 7 projections.
        let mut workspace_bytes_max = 0usize;
        for lin in layer0.linears() {
            let w = backend.nvfp4_workspace_size(
                m, lin.shape.n as i32, lin.shape.k as i32);
            if w > workspace_bytes_max { workspace_bytes_max = w; }
        }
        let workspace_bytes = workspace_bytes_max.max(16);
        let workspace_region = arena.region("smoke_ws", workspace_bytes, 16)?;

        // Helper: fence + DtoH a bf16 region of `count` elements, widen to f32.
        unsafe fn dump_bf16(
            stream: &rvllm_mem::stream::Stream,
            stream_u64: u64,
            dev_ptr: u64,
            count: usize,
            label: &'static str,
        ) -> Result<Vec<f32>> {
            stream.fence()?;
            let mut buf = vec![0u16; count];
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
            Ok(buf.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect())
        }

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

        // Helper: prep_act + sfa_transform (NVFP4 activation staging).
        let stage_act = |src_bf16_ptr: u64, k_dim: i32| -> Result<()> {
            backend.launch_nvfp4_prep_act(
                src_bf16_ptr,
                a_packed_region.device_ptr(),
                sfa_natural_region.device_ptr(),
                m, k_dim, 1 /* BF16 */, stream_u64,
            )?;
            backend.launch_nvfp4_sfa_transform(
                sfa_natural_region.device_ptr(),
                sfa_cutlass_region.device_ptr(),
                m, k_dim, stream_u64,
            )
        };

        // Helper: NVFP4 GEMM using the staged a_packed + sfa_cutlass.
        let gemm = |out_ptr: u64,
                    lin: &rvllm_loader::mistral35_weights::Nvfp4LinearLoaded|
         -> Result<()> {
            backend.launch_nvfp4_gemm(
                out_ptr,
                a_packed_region.device_ptr(),
                lin.packed_ptr,
                sfa_cutlass_region.device_ptr(),
                lin.sfb_cutlass_ptr,
                lin.global_scale_ptr,
                m, lin.shape.n as i32, lin.shape.k as i32,
                workspace_region.device_ptr(),
                workspace_bytes,
                stream_u64,
            )
        };

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

        // ============================================================
        // Layer-0 forward (pos=0, single token; RoPE is identity at
        // pos=0 so we skip it; m=1 attention reduces to V broadcast
        // because softmax over a single key is 1.0).
        // ============================================================

        // (1) Embed gather → h_residual.
        rvllm_fused::EmbeddingGatherLaunch { num_tokens: 1, hidden, vocab }
            .launch(
                kernels.fn_embedding_gather_bf16,
                h_residual_region.device_ptr(),
                model.outside.embed_tokens.offset_bytes,
                token_region.device_ptr(),
                stream_u64,
            )?;
        let post_embed = dump_bf16(
            stream, stream_u64,
            h_residual_region.device_ptr(), hidden as usize,
            "smoke_post_embed",
        )?;

        // (2) DtoD copy h_residual → h_work, then RMSNorm in-place
        //     with input_layernorm gamma.
        dtod(stream_u64, h_work_region.device_ptr(),
             h_residual_region.device_ptr(), hidden_bytes)?;
        rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
            num_tokens: 1, hidden, eps: self.arch.text.rms_norm_eps as f32,
        }.launch(
            kernels.fn_rmsnorm_inplace_bf16_gbf16,
            h_work_region.device_ptr(),
            layer0.input_layernorm.offset_bytes,
            stream_u64,
        )?;
        let post_rmsnorm = dump_bf16(
            stream, stream_u64,
            h_work_region.device_ptr(), hidden as usize,
            "smoke_post_rmsnorm",
        )?;

        // (3) NVFP4 prep h_work, then Q/K/V projections.
        stage_act(h_work_region.device_ptr(), h_k)?;
        gemm(q_out_region.device_ptr(), &layer0.q_proj)?;
        gemm(k_out_region.device_ptr(), &layer0.k_proj)?;
        gemm(v_out_region.device_ptr(), &layer0.v_proj)?;
        let q_out = dump_bf16(stream, stream_u64,
            q_out_region.device_ptr(), layer0.q_proj.shape.n, "smoke_q_out")?;
        let k_out = dump_bf16(stream, stream_u64,
            k_out_region.device_ptr(), layer0.k_proj.shape.n, "smoke_k_out")?;
        let v_out = dump_bf16(stream, stream_u64,
            v_out_region.device_ptr(), layer0.v_proj.shape.n, "smoke_v_out")?;

        // (4) m=1 GQA-broadcast attention: attn_out[h_q, d] = V[h_kv, d]
        //     where h_kv = h_q / gqa_ratio. Skip RoPE (identity@pos=0).
        let head_dim = self.arch.text.head_dim as i32;
        let n_q_heads = self.arch.text.num_attention_heads as u32;
        let gqa_ratio = (self.arch.text.num_attention_heads
            / self.arch.text.num_key_value_heads) as i32;
        {
            let mut out_ptr = attn_out_region.device_ptr();
            let mut v_ptr = v_out_region.device_ptr();
            let mut head_dim_arg = head_dim;
            let mut gqa_arg = gqa_ratio;
            let args = [
                (&mut out_ptr) as *mut u64 as *mut std::ffi::c_void,
                (&mut v_ptr) as *mut u64 as *mut std::ffi::c_void,
                (&mut head_dim_arg) as *mut i32 as *mut std::ffi::c_void,
                (&mut gqa_arg) as *mut i32 as *mut std::ffi::c_void,
            ];
            launch_kernel(
                kernels.fn_attn_m1_gqa_broadcast,
                (n_q_heads, 1, 1),
                (head_dim as u32, 1, 1),
                &args, stream_u64,
            )?;
        }
        let attn_out = dump_bf16(stream, stream_u64,
            attn_out_region.device_ptr(), hidden as usize, "smoke_attn_out")?;

        // (5) O-projection: NVFP4 prep_act on attn_out, then o_proj.
        stage_act(attn_out_region.device_ptr(), h_k)?;
        gemm(o_out_region.device_ptr(), &layer0.o_proj)?;
        let o_out = dump_bf16(stream, stream_u64,
            o_out_region.device_ptr(), layer0.o_proj.shape.n, "smoke_o_out")?;

        // (6) Residual: h_residual += o_out.
        rvllm_fused::gemma4_launcher::VectorAddF16Launch { n: hidden }.launch(
            kernels.fn_vector_add_bf16,
            h_residual_region.device_ptr(),
            o_out_region.device_ptr(),
            stream_u64,
        )?;
        let h_after_attn = dump_bf16(stream, stream_u64,
            h_residual_region.device_ptr(), hidden as usize, "smoke_h_after_attn")?;

        // (7) DtoD copy h_residual → h_work, RMSNorm with
        //     post_attention_layernorm.
        dtod(stream_u64, h_work_region.device_ptr(),
             h_residual_region.device_ptr(), hidden_bytes)?;
        rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
            num_tokens: 1, hidden, eps: self.arch.text.rms_norm_eps as f32,
        }.launch(
            kernels.fn_rmsnorm_inplace_bf16_gbf16,
            h_work_region.device_ptr(),
            layer0.post_attention_layernorm.offset_bytes,
            stream_u64,
        )?;
        let post_attn_norm = dump_bf16(stream, stream_u64,
            h_work_region.device_ptr(), hidden as usize, "smoke_post_attn_norm")?;

        // (8) NVFP4 prep h_work, then gate / up projections.
        stage_act(h_work_region.device_ptr(), h_k)?;
        gemm(gate_out_region.device_ptr(), &layer0.gate_proj)?;
        gemm(up_out_region.device_ptr(),   &layer0.up_proj)?;
        let gate_out = dump_bf16(stream, stream_u64,
            gate_out_region.device_ptr(), layer0.gate_proj.shape.n, "smoke_gate_out")?;
        let up_out = dump_bf16(stream, stream_u64,
            up_out_region.device_ptr(), layer0.up_proj.shape.n, "smoke_up_out")?;

        // (9) silu_mul: silu_mid = silu(gate) * up, both BF16.
        let i_size = self.arch.text.intermediate_size as i32;
        {
            let mut out_ptr = silu_mid_region.device_ptr();
            let mut g_ptr = gate_out_region.device_ptr();
            let mut u_ptr = up_out_region.device_ptr();
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
            silu_mid_region.device_ptr(),
            self.arch.text.intermediate_size, "smoke_silu_mid")?;

        // (10) Down projection: NVFP4 prep_act on silu_mid (K=intermediate),
        //      then down_proj GEMM.
        stage_act(silu_mid_region.device_ptr(), i_k)?;
        gemm(down_out_region.device_ptr(), &layer0.down_proj)?;
        let down_out = dump_bf16(stream, stream_u64,
            down_out_region.device_ptr(),
            layer0.down_proj.shape.n, "smoke_down_out")?;

        // (11) Residual: h_residual += down_out.
        rvllm_fused::gemma4_launcher::VectorAddF16Launch { n: hidden }.launch(
            kernels.fn_vector_add_bf16,
            h_residual_region.device_ptr(),
            down_out_region.device_ptr(),
            stream_u64,
        )?;
        let h_after_layer0 = dump_bf16(stream, stream_u64,
            h_residual_region.device_ptr(), hidden as usize, "smoke_h_after_layer0")?;

        // ============================================================
        // Layers 1..88 — same flow as layer 0 with that layer's
        // weights. Closure reused; only the residual rms is captured
        // per layer (full per-stage dumps would be ~88×14 tensors).
        // ============================================================
        let mut layer_residual_rms: Vec<f32> = Vec::with_capacity(model.layers.len());
        // Record layer-0 residual rms as the first entry.
        let l0_sumsq: f64 = h_after_layer0.iter()
            .map(|&x| (x as f64) * (x as f64)).sum();
        layer_residual_rms.push(
            (l0_sumsq / (h_after_layer0.len() as f64)).sqrt() as f32);

        let head_dim_arg = head_dim;
        let n_q_heads_arg = n_q_heads;
        let gqa_ratio_arg = gqa_ratio;
        let i_size = i_size;
        for layer_idx in 1..model.layers.len() {
            let layer = &model.layers[layer_idx];

            // Pre-attn norm.
            dtod(stream_u64, h_work_region.device_ptr(),
                 h_residual_region.device_ptr(), hidden_bytes)?;
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: 1, hidden, eps: self.arch.text.rms_norm_eps as f32,
            }.launch(
                kernels.fn_rmsnorm_inplace_bf16_gbf16,
                h_work_region.device_ptr(),
                layer.input_layernorm.offset_bytes,
                stream_u64,
            )?;

            // Q/K/V.
            stage_act(h_work_region.device_ptr(), h_k)?;
            gemm(q_out_region.device_ptr(), &layer.q_proj)?;
            gemm(k_out_region.device_ptr(), &layer.k_proj)?;
            gemm(v_out_region.device_ptr(), &layer.v_proj)?;

            // m=1 GQA-broadcast attention (RoPE@pos=0 = identity).
            {
                let mut out_ptr = attn_out_region.device_ptr();
                let mut v_ptr = v_out_region.device_ptr();
                let mut hd = head_dim_arg;
                let mut g = gqa_ratio_arg;
                let args = [
                    (&mut out_ptr) as *mut u64 as *mut std::ffi::c_void,
                    (&mut v_ptr) as *mut u64 as *mut std::ffi::c_void,
                    (&mut hd) as *mut i32 as *mut std::ffi::c_void,
                    (&mut g) as *mut i32 as *mut std::ffi::c_void,
                ];
                launch_kernel(
                    kernels.fn_attn_m1_gqa_broadcast,
                    (n_q_heads_arg, 1, 1),
                    (head_dim_arg as u32, 1, 1),
                    &args, stream_u64,
                )?;
            }

            // O proj + residual.
            stage_act(attn_out_region.device_ptr(), h_k)?;
            gemm(o_out_region.device_ptr(), &layer.o_proj)?;
            rvllm_fused::gemma4_launcher::VectorAddF16Launch { n: hidden }.launch(
                kernels.fn_vector_add_bf16,
                h_residual_region.device_ptr(),
                o_out_region.device_ptr(),
                stream_u64,
            )?;

            // Post-attn norm.
            dtod(stream_u64, h_work_region.device_ptr(),
                 h_residual_region.device_ptr(), hidden_bytes)?;
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: 1, hidden, eps: self.arch.text.rms_norm_eps as f32,
            }.launch(
                kernels.fn_rmsnorm_inplace_bf16_gbf16,
                h_work_region.device_ptr(),
                layer.post_attention_layernorm.offset_bytes,
                stream_u64,
            )?;

            // gate / up / silu_mul.
            stage_act(h_work_region.device_ptr(), h_k)?;
            gemm(gate_out_region.device_ptr(), &layer.gate_proj)?;
            gemm(up_out_region.device_ptr(),   &layer.up_proj)?;
            {
                let mut out_ptr = silu_mid_region.device_ptr();
                let mut g_ptr = gate_out_region.device_ptr();
                let mut u_ptr = up_out_region.device_ptr();
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

            // down + residual.
            stage_act(silu_mid_region.device_ptr(), i_k)?;
            gemm(down_out_region.device_ptr(), &layer.down_proj)?;
            rvllm_fused::gemma4_launcher::VectorAddF16Launch { n: hidden }.launch(
                kernels.fn_vector_add_bf16,
                h_residual_region.device_ptr(),
                down_out_region.device_ptr(),
                stream_u64,
            )?;

            // Per-layer residual rms (single fence + DtoH every 8 layers
            // is plenty; the per-launch is tiny but the fence cost
            // adds up). For initial smoke we want every layer.
            let v = dump_bf16(stream, stream_u64,
                h_residual_region.device_ptr(), hidden as usize,
                "smoke_layer_residual")?;
            let sumsq: f64 = v.iter().map(|&x| (x as f64) * (x as f64)).sum();
            layer_residual_rms.push((sumsq / (v.len() as f64)).sqrt() as f32);
        }

        // ============================================================
        // Final RMSNorm + lm_head + argmax.
        // ============================================================
        let cublaslt = self.cublaslt.as_ref().ok_or_else(|| corrupt(
            self.paths.model_dir.clone(),
            "forward_smoke_layer0: cublaslt absent".into(),
        ))?;

        // Final norm: copy h_residual → h_work, RMSNorm with
        // outside.final_norm gamma.
        dtod(stream_u64, h_work_region.device_ptr(),
             h_residual_region.device_ptr(), hidden_bytes)?;
        rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
            num_tokens: 1, hidden, eps: self.arch.text.rms_norm_eps as f32,
        }.launch(
            kernels.fn_rmsnorm_inplace_bf16_gbf16,
            h_work_region.device_ptr(),
            model.outside.final_norm.offset_bytes,
            stream_u64,
        )?;
        let h_after_final_norm = dump_bf16(stream, stream_u64,
            h_work_region.device_ptr(), hidden as usize, "smoke_h_after_final_norm")?;

        // lm_head: BF16 [vocab, hidden] × BF16 [1, hidden] → F32 [1, vocab]
        let vocab = self.arch.text.vocab_size as i32;
        let logits_region = arena.region(
            "smoke_logits_f32", (vocab as usize) * 4, 16)?;
        cublaslt.bf16_gemm_f32(
            h_work_region.device_ptr(),       // a [1, hidden] (BF16)
            model.outside.lm_head.offset_bytes, // b [vocab, hidden] (BF16)
            logits_region.device_ptr(),        // d [1, vocab] (F32)
            1,                                 // m = num_seqs
            vocab,
            h_k,
            stream_u64,
        )?;

        // argmax_kernel(logits_f32, out_token_i32, vocab_size).
        let token_out_region = arena.region("smoke_token_out", 4, 4)?;
        {
            let mut logits_ptr = logits_region.device_ptr();
            let mut tok_ptr = token_out_region.device_ptr();
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

        // DtoH the predicted token + the full logits vector for top-8.
        stream.fence()?;
        let mut tok_buf = [0u8; 4];
        let mut logits_host = vec![0f32; vocab as usize];
        unsafe {
            use cudarc::driver::sys::*;
            let r1 = cuMemcpyDtoH_v2(
                tok_buf.as_mut_ptr() as *mut std::ffi::c_void,
                token_out_region.device_ptr() as CUdeviceptr, 4);
            if r1 != CUresult::CUDA_SUCCESS {
                return Err(RvllmError::cuda(
                    "cuMemcpyDtoH_v2(predicted_token)",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx { stream: stream_u64,
                        kernel: "smoke_token_dtoh", launch: None, device: -1 },
                ));
            }
            let r2 = cuMemcpyDtoH_v2(
                logits_host.as_mut_ptr() as *mut std::ffi::c_void,
                logits_region.device_ptr() as CUdeviceptr,
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
        let predicted_token = u32::from_le_bytes(tok_buf);
        // Top-8 by logit value (host-side for diagnostics; small N).
        let mut indexed: Vec<(u32, f32)> = logits_host.iter().enumerate()
            .map(|(i, &v)| (i as u32, v)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let logits_top8: Vec<(u32, f32)> = indexed.into_iter().take(8).collect();

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
}

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

    #[test]
    fn kv_decode_strategy_picks_fused_for_low_gqa() {
        // Gemma 4 sliding has GQA=2; falls under MAX_GQA_DECODE=4.
        assert_eq!(
            KvDecodeStrategy::for_gqa_ratio(2),
            KvDecodeStrategy::FusedGqa
        );
        assert_eq!(
            KvDecodeStrategy::for_gqa_ratio(4),
            KvDecodeStrategy::FusedGqa
        );
    }

    #[test]
    fn kv_decode_strategy_picks_split_for_qwen_ratio() {
        // Qwen 3.6 full attention has GQA=8; below MAX_GQA_SPLIT=8.
        assert_eq!(
            KvDecodeStrategy::for_gqa_ratio(5),
            KvDecodeStrategy::SplitDecode
        );
        assert_eq!(
            KvDecodeStrategy::for_gqa_ratio(8),
            KvDecodeStrategy::SplitDecode
        );
    }

    #[test]
    fn kv_decode_strategy_falls_back_for_mistral() {
        // Mistral 3.5 has GQA=12; both fused and split caps blown.
        assert_eq!(
            KvDecodeStrategy::for_gqa_ratio(12),
            KvDecodeStrategy::PerHeadFallback
        );
        assert_eq!(
            KvDecodeStrategy::for_gqa_ratio(16),
            KvDecodeStrategy::PerHeadFallback
        );
    }

    #[test]
    fn kv_decode_strategy_safe_on_zero_ratio() {
        assert_eq!(
            KvDecodeStrategy::for_gqa_ratio(0),
            KvDecodeStrategy::PerHeadFallback
        );
    }

    /// Serialise the env-var-touching tests so cargo's parallel
    /// runner doesn't make them race. (Tests that don't touch env
    /// don't acquire — keeps the rest of the suite parallel.)
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn batched_prefill_default_is_full_on_except_decode_graph_and_qkv() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        // Ensure no env vars from a previous test leak in.
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
        assert!(cfg.batch_projections);
        assert!(cfg.batch_rope);
        assert!(cfg.batch_full_prefill);
        assert!(cfg.outer_loop_deleted);
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
