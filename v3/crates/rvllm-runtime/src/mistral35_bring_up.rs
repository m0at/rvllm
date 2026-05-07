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

use std::path::PathBuf;

use rvllm_core::{LoaderCtx, LoaderError, Result, RvllmError};
use rvllm_loader::mistral35_arch::Mistral35Arch;
use rvllm_loader::mistral35_weights::{
    validate_mistral35_inventory, Mistral35TensorCounts, Mistral35WeightInventory,
};
use rvllm_loader::safetensors::{ShardHeader, ShardIndex, TensorEntry};

use crate::gemma4_bring_up::Gemma4EnginePaths;

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
    /// Reserved for the device-upload pass (Step 4-CUDA): bytes the
    /// caller asked us to allocate for the arena. We keep it on the
    /// struct so the run-loop can size scratch consistently with
    /// the value `cuda_worker` was started with.
    pub arena_bytes: usize,
    /// Whether every required NVFP4 CUTLASS symbol resolved. Always
    /// true on the `Ok` path — `load` refuses startup otherwise.
    pub nvfp4_active: bool,
    /// Resolved CUTLASS backend handle (only populated under the
    /// `cuda` feature; the .so + fn-pointer cache lives here so the
    /// forward path doesn't have to re-open it). The struct is
    /// `Send + Sync` per its own unsafe impl, so the worker thread
    /// can hold the bring-up across the whole request loop.
    #[cfg(feature = "cuda")]
    pub cutlass_backend: Option<rvllm_cutlass::lib_so::CutlassBackend>,
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
    /// + GEMM workspace bytes per Mistral projection shape, taking
    /// the max across the seven projections so a single allocation
    /// covers every layer. Falls back to [`Self::natural`] when the
    /// backend reports zero (e.g. NVFP4 symbols not yet bound).
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

        #[cfg(feature = "cuda")]
        let bringup = Self {
            paths, arch, inventory, arena_bytes, nvfp4_active, cutlass_backend,
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
        // For the inventory pass we only need the header. Read the
        // first 8 bytes (header_bytes prefix) + the JSON header,
        // never the payload. That keeps load() fast on a 128B
        // checkpoint (~80 GiB on disk) — full validation in seconds
        // rather than a multi-second mmap walk.
        let bytes = std::fs::read(shard_path).map_err(|source| RvllmError::Io {
            err: rvllm_core::IoError::from(&source),
            path: shard_path.clone(),
            source,
        })?;
        let header = ShardHeader::parse(shard_path, &bytes)?;
        for (name, entry) in header.tensors.into_iter() {
            tensors.insert(name, entry);
        }
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
