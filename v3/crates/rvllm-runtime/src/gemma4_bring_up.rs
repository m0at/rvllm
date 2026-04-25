//! Gemma 4 engine bring-up.
//!
//! Parallel to `bring_up.rs` for Llama/Qwen. Assembles every subsystem
//! needed for Gemma 4 inference: variable-head attention, dual RoPE
//! tables, per-layer KV head variation, extra kernel modules.
//!
//! Usage: when `config.json` declares `"Gemma3ForCausalLM"` or similar,
//! the top-level dispatcher constructs `Gemma4Bringup` instead of
//! the regular `Bringup`.

use std::path::PathBuf;
use std::sync::Arc;

use rvllm_attention::{AttentionBackend, Fa3Kernels};
use rvllm_core::Result;
use rvllm_cutlass::{CublasLt, CutlassBackend, Policy};
use rvllm_kernels::{KernelFn, KernelLoader, LoadedModule};
use rvllm_mem::{context::CudaContextHandle, stream::Stream, HbmArena};

use crate::gemma4_layer_exec::Gemma4LayerKernels;

pub use crate::bring_up::HbmArenaCheckpoint;

/// Default per-tensor Q / KV scale for the FP8 E4M3 attention cache.
/// The older 418/448 ≈ 0.933 defaults assume `amax ≈ 418` for the
/// post-QK-norm / post-V-norm activations — an order of magnitude too
/// large for Gemma 4. Empirical calibration sweep (chunk_len=128,
/// English text) found `q=0.1, kv=0.08` minimizes PPL by 4× over the
/// old defaults (PPL 10.2 → 2.3). Both overridable per-run via
/// `RVLLM_Q_SCALE` / `RVLLM_KV_SCALE` env vars for further tuning or
/// per-model calibration.
const DEFAULT_Q_SCALE: f32 = 0.1;
const DEFAULT_KV_SCALE: f32 = 0.08;

pub struct Gemma4EnginePaths {
    pub model_dir: PathBuf,
    pub kernels_dir: PathBuf,
    pub cutlass_so: PathBuf,
    pub fa3_so: PathBuf,
    pub policy_json: PathBuf,
}

pub struct Gemma4FusedModules {
    pub rmsnorm_mod: LoadedModule,
    pub rmsnorm_inplace_mod: LoadedModule,
    pub rope_mod: LoadedModule,
    pub gelu_mod: LoadedModule,
    pub argmax_mod: LoadedModule,
    pub qk_norm_mod: LoadedModule,
    pub softcap_mod: LoadedModule,
    pub residual_scale_mod: LoadedModule,
    pub vnorm_mod: LoadedModule,
    pub vector_add_mod: LoadedModule,
    pub bf16_to_f16_sat_mod: LoadedModule,
    pub rmsnorm_inplace_bf16_mod: LoadedModule,
    pub vector_add_bf16_to_f16_mod: LoadedModule,
    pub f32_to_bf16_mod: LoadedModule,
    pub f32_to_f16_sat_mod: LoadedModule,
    pub scale_cols_f32_mod: LoadedModule,
    pub scale_rows_f32_ratio_mod: LoadedModule,
    pub compute_qkv_scales_mod: LoadedModule,
    pub fused_gelu_mul_f16_mod: LoadedModule,
    pub fused_rope_partial_f16kv_mod: LoadedModule,
    pub fused_norm_add_residual_mod: LoadedModule,
    pub fn_rmsnorm: KernelFn,
    pub fn_rmsnorm_fp8_quant: KernelFn,
    pub fn_quantize: KernelFn,
    pub fn_rope_partial_fp8kv: KernelFn,
    pub fn_gelu_mul: KernelFn,
    pub fn_argmax: KernelFn,
    pub fn_qk_rmsnorm: KernelFn,
    pub fn_softcap: KernelFn,
    pub fn_residual_scale: KernelFn,
    pub fn_vnorm: KernelFn,
    pub fn_vector_add: KernelFn,
    pub fn_bf16_to_f16_sat: KernelFn,
    pub fn_rmsnorm_inplace_bf16: KernelFn,
    pub fn_vector_add_bf16_to_f16: KernelFn,
    pub fn_f32_to_bf16: KernelFn,
    pub fn_f32_to_f16_sat: KernelFn,
    pub fn_scale_cols_f32: KernelFn,
    pub fn_scale_rows_f32_ratio: KernelFn,
    pub fn_compute_qkv_scales: KernelFn,
    pub fn_fused_gelu_mul_f16: KernelFn,
    pub fn_fused_rope_partial_f16kv: KernelFn,
    pub fn_fused_norm_add_residual: KernelFn,
    pub fn_fused_norm_add_residual_f16: KernelFn,
    /// Variant that reads f16 input and skips channelscale; used by the
    /// Sm121 decode fast path after `fp8_gemv_wpr_native_f16in` has
    /// already applied the per-channel scale in the GEMV epilogue.
    pub fn_fused_norm_add_residual_f16in: KernelFn,
    pub fused_norm_add_residual_f16_mod: LoadedModule,
    pub fn_fused_qkv_rmsnorm: KernelFn,
    pub fused_qkv_rmsnorm_mod: LoadedModule,
    pub fn_scale_cols_f16: KernelFn,
    pub scale_cols_f16_mod: LoadedModule,

    // `fp8_gemv.ptx` — GB10 warp-per-row FP8 GEMV kernels. Loaded at
    // bringup so the Sm121 decode fast path (`launch_fp8_gemv_f16in`
    // in `gemma4_layer_exec.rs`) can call it without a per-step
    // module load. Only the f16-input variant is resolved — the
    // other enum variants in `Fp8GemvVariant` document what ships in
    // the PTX but nothing in the runtime path calls them.
    pub fp8_gemv_mod: LoadedModule,
    /// `None` when the live device is not Blackwell (sm_100+) — the
    /// native-CVT entry is gated on `__CUDA_ARCH__ >= 1000` in
    /// `kernels/fp8_gemv.cu`, so the symbol is absent from
    /// pre-Blackwell PTX. `Fp8GemvVariant::available_for(target)` is
    /// the source of truth for this gate. Used by the Sm121 decode
    /// path to run projection GEMMs (QKV / O / gate_up / down)
    /// directly off f16 activations, skipping the FP8 activation-
    /// quant step that cuBLASLt requires.
    pub fn_fp8_gemv_wpr_native_f16in: Option<KernelFn>,
}

/// Session-level prefix cache state. Populated on first `run_generate`
/// call; each subsequent call inspects `last_tokens` for a common
/// prefix with the incoming prompt and, on hit, skips prefill for
/// the matched prefix (the KV entries from the previous request
/// remain valid in the persistent KV region because `kv_cache_ptr`
/// points above the worker's scratch checkpoint).
///
/// MVP of vLLM's block-level prefix caching — no hashing, no
/// reference counting, just a single "last request's prompt" slot.
/// Covers the common zeroclaw pattern (identical 15k-token persona
/// on every request) at a cost of ~100 LOC of plumbing. Full
/// multi-sequence prefix caching is future work.
pub struct PrefixCacheState {
    pub last_tokens: Vec<u32>,
    pub kv_cache_ptr: u64,
    pub kv_cache_bytes: u64,
    pub kv_scale_ptr: u64,
    pub kv_scale_bytes: u64,
    pub kv_dtype: crate::gemma4_layer_exec::KvDtype,
    pub kv_layer_offsets: Vec<u64>,
    pub kv_scale_layer_offsets: Vec<u64>,
    pub num_blocks_total: u32,
    pub block_size: u32,
    /// Length (in tokens) of the prefix from `last_tokens` that
    /// is SAFE to reuse across a subsequent request. Capped at
    /// the last full prefill-chunk boundary so subsequent prompts
    /// that match this prefix are guaranteed to find KV entries
    /// written under the SAME chunk shape as the new request would
    /// use for those positions.
    ///
    /// Without this cap, a short request (e.g. classifier, 3057
    /// tokens prefilled in chunks 2048+1009) leaves slots
    /// [2048..3057) populated under chunk_q=1009. A subsequent
    /// long request (e.g. 15k tokens) would have written those
    /// same slots inside its own first chunk_q=2048. Optimized
    /// NVFP4 kernels are batch-variant; reusing classifier-shape
    /// KV at slots [2048..2922) inside the 15k request produces
    /// catastrophic garbage ("la la la × 1024" repetition collapse,
    /// observed in production via zeroclaw classifier-then-persona
    /// chains).
    ///
    /// Set to `floor(prompt_len / chunk_size) * chunk_size` after
    /// each request completes (or `prompt_len` when chunk_size = 0,
    /// i.e. no chunking).
    pub committed_prefix_len: u32,
    /// Provenance tuple. Cache is INVALIDATED on mismatch — KV
    /// entries written under different policy configuration are
    /// not generally reusable. Cheap to check; protects against
    /// silent miscompare across env-var flips between requests.
    pub provenance: PrefixProvenance,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct PrefixProvenance {
    pub chunk_size: u32,
    pub kv_dtype: crate::gemma4_layer_exec::KvDtype,
    pub hybrid_global_fp8: bool,
    pub scale_policy: u32,        // 0 = amax6, 1 = mse, etc.
    pub batch_prefill: bool,
    pub unified_prefill: bool,
}

impl PrefixProvenance {
    /// Read current env into a provenance tuple.
    pub fn from_env() -> Self {
        let chunk_size: u32 = std::env::var("RVLLM_PREFILL_CHUNK_SIZE")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(0);
        let kv_dtype = crate::gemma4_layer_exec::KvDtype::from_env(false);
        let hybrid_global_fp8 = std::env::var("RVLLM_NVFP4_HYBRID_GLOBAL_FP8")
            .map(|v| v != "0").unwrap_or(false);
        let scale_policy = std::env::var("RVLLM_NVFP4_SCALE_POLICY")
            .ok().and_then(|s| match s.as_str() {
                "amax6" | "0" => Some(0u32),
                "mse" | "1" => Some(1u32),
                _ => None,
            }).unwrap_or(0);
        let batch_prefill = std::env::var_os("RVLLM_BATCH_PREFILL").is_some();
        let unified_prefill = std::env::var_os("RVLLM_UNIFIED_PREFILL").is_some();
        Self { chunk_size, kv_dtype, hybrid_global_fp8, scale_policy,
               batch_prefill, unified_prefill }
    }
}

pub struct Gemma4Bringup {
    pub fused: Gemma4FusedModules,
    pub sliding_attention: AttentionBackend,
    pub global_attention: AttentionBackend,
    pub cutlass: CutlassBackend,
    pub cublaslt: CublasLt,
    pub cublaslt_ws: HbmArenaCheckpoint,
    pub policy: Policy,
    pub arch: rvllm_loader::gemma4_arch::Gemma4Arch,
    pub model: rvllm_loader::gemma4_weights::Gemma4LoadedModel,
    pub kernels: Arc<KernelLoader>,
    pub stream: Stream,
    pub arena: HbmArena<'static>,
    pub ctx: Arc<CudaContextHandle>,
    /// Session-level prefix cache. Populated lazily on first
    /// `run_generate` call; kept across subsequent calls so the
    /// KV cache survives the worker's scratch-checkpoint restore.
    pub prefix_cache: std::sync::Mutex<Option<PrefixCacheState>>,
    // === NVFP4 SHADOW DIAGNOSTIC (remove after collapse locator confirmed) ===
    /// Ground-truth F16 shadow KV region for the instrumented layer set.
    /// Populated on first run_generate when RVLLM_NVFP4_SHADOW_F16=1.
    pub nvfp4_shadow: std::sync::Mutex<Option<NvFp4ShadowAlloc>>,
    /// One-shot latch for first-token dump.
    pub nvfp4_shadow_dumped: std::sync::atomic::AtomicBool,
    // === END NVFP4 SHADOW DIAGNOSTIC ===
}

// === NVFP4 SHADOW DIAGNOSTIC (remove after collapse locator confirmed) ===
/// Parallel to the main KV region but: (a) only the instrumented
/// layers have a slot; (b) every instrumented layer is stored as F16
/// regardless of the primary KV dtype. No scale region needed.
pub struct NvFp4ShadowAlloc {
    pub shadow_ptr: u64,
    pub shadow_bytes: u64,
    /// Per-layer byte offset into `shadow_ptr`. `u64::MAX` sentinel
    /// for layers NOT in the instrumented set.
    pub layer_offsets: Vec<u64>,
    pub layer_indices: Vec<u32>,
    /// Per-instrumented-layer Q snapshot region. Sized for
    /// `num_shadow_layers * num_attention_heads * max_head_dim * 2`
    /// bytes (f16). Populated on decode step 0 only, AFTER the shadow
    /// f16 RoPE (which writes post-RoPE Q into `scratch.q_normed`)
    /// and BEFORE the primary NVFP4 RoPE clobbers it. Per-layer slot
    /// size is uniform (`q_per_layer_bytes`) even when the layer's
    /// head_dim is smaller than max_head_dim — the tail of the slot
    /// is then zero and the Python analyzer truncates using
    /// `head_dim` from meta.json.
    pub shadow_q_ptr: u64,
    pub shadow_q_total_bytes: u64,
    pub shadow_q_per_layer_bytes: u64,
    /// Q throwaway scratch — a single-slot f16 buffer (same size as
    /// one per-layer Q slot) that `rope_f16kv_shadow` targets when we
    /// are NOT capturing (prefill steps, decode step > 0). Keeps
    /// `scratch.q_normed` untouched so the subsequent primary
    /// `rope_nvfp4kv` rotates Q exactly once. Without this, shadow
    /// rope's q_out=q_normed caused double-RoPE on q_fp8 and
    /// corrupted live inference whenever shadow was on.
    pub shadow_q_throwaway_ptr: u64,
}
// === END NVFP4 SHADOW DIAGNOSTIC ===

impl Gemma4Bringup {
    pub fn load(paths: Gemma4EnginePaths, arena_bytes: usize) -> Result<Self> {
        let ctx = Arc::new(CudaContextHandle::init(0)?);
        // Resolve the compile target once per bring-up and thread it
        // through — every call to `ctx.compute_capability()` + the
        // lookup costs nothing individually but spreading it across 5
        // sites means "which CC are we on?" reads inconsistent if a
        // future refactor accidentally shadows `ctx`.
        #[cfg(feature = "cuda")]
        let compile_target: Option<rvllm_core::CompileTarget> = {
            let (major, minor) = ctx.compute_capability();
            rvllm_core::CompileTarget::from_compute_capability(major, minor)
        };
        #[cfg(not(feature = "cuda"))]
        let compile_target: Option<rvllm_core::CompileTarget> = None;

        // Arena backing picked per compute capability — see `Bringup::load`
        // in bring_up.rs for the full rationale (GB10 has no dedicated HBM,
        // cuMemAllocManaged is the right allocator there).
        let arena = {
            #[cfg(feature = "gb10")]
            {
                if matches!(compile_target, Some(rvllm_core::CompileTarget::Sm121)) {
                    rvllm_mem::UnifiedArena::new(&ctx, arena_bytes)?.into_inner()
                } else {
                    HbmArena::new(&ctx, arena_bytes)?
                }
            }
            #[cfg(not(feature = "gb10"))]
            {
                HbmArena::new(&ctx, arena_bytes)?
            }
        };
        let arena: HbmArena<'static> = unsafe { std::mem::transmute(arena) };
        let stream = Stream::new(&ctx)?;

        let arch = rvllm_loader::gemma4_arch::Gemma4Arch::from_dir(&paths.model_dir)?;
        let model = rvllm_loader::gemma4_load::load_gemma4_model(&paths.model_dir, &arena, &arch)?;

        // On sm_121 the arena is `cuMemAllocManaged` pages that fault
        // to the GPU on first touch. After the weight upload the
        // populated region (~30 GiB for Gemma 4 31B) hasn't faulted
        // yet — prefetching it here removes the page-fault storm
        // from the first decode iteration, so first-token latency
        // stops carrying 30 GiB of H→D page migration cost. CUDA 13
        // dropped the single-arg `cuMemPrefetchAsync` in favour of
        // `_v2` with a `CUmemLocation`; cudarc 0.19 only wraps the
        // v2 form for cuda-13. Best-effort: a non-zero RC is logged
        // but doesn't fail bring-up.
        #[cfg(all(feature = "gb10", feature = "cuda"))]
        unsafe {
            if matches!(compile_target, Some(rvllm_core::CompileTarget::Sm121)) {
                let prefetch_bytes = arena.used();
                if prefetch_bytes > 0 {
                    let loc = cudarc::driver::sys::CUmemLocation {
                        type_: cudarc::driver::sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
                        id: 0,
                    };
                    let rc = cudarc::driver::sys::cuMemPrefetchAsync_v2(
                        arena.base_ptr(),
                        prefetch_bytes,
                        loc,
                        0,
                        stream.raw() as _,
                    );
                    if rc != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                        tracing::warn!(
                            "cuMemPrefetchAsync_v2({prefetch_bytes} bytes) rc={rc:?} — first-token latency may spike"
                        );
                    } else {
                        let _ = cudarc::driver::sys::cuStreamSynchronize(
                            stream.raw() as _,
                        );
                    }
                }
            }
        }

        // Per-arch kernel subdirectory resolution — see `resolve_kernels_dir`.
        let kernels_dir = crate::bring_up::resolve_kernels_dir(&ctx, &paths.kernels_dir)?;
        let manifest_path = kernels_dir.join("manifest.json");
        let manifest = rvllm_kernels::manifest::KernelManifest::load_and_verify(&manifest_path)?;
        let kernels = Arc::new(KernelLoader::new(manifest));

        // Attention backend selection. On SM80/SM89/SM90 we stick with
        // the FA3 `.so` (WGMMA + TMA). On sm_121 (GB10) FA3 cannot
        // load — WGMMA doesn't exist on Blackwell consumer silicon —
        // so we route through the PTX-launched FA2 kernels we already
        // compile for every arch. The FA2 launch body is still a
        // follow-up (the decode/prefill launchers return
        // `FeatureNotAvailable` for the `Fa2Ptx` variant), but
        // bring-up now completes on GB10 without a hard fail on
        // `Fa3SoMissing`. See `rvllm_attention::Fa2PtxKernels` docs.
        let (sliding_attention, global_attention) = {
            #[cfg(feature = "gb10")]
            let is_gb10 = matches!(compile_target, Some(rvllm_core::CompileTarget::Sm121));
            #[cfg(not(feature = "gb10"))]
            let is_gb10 = false;
            if is_gb10 {
                let sliding = AttentionBackend::Fa2Ptx(rvllm_attention::Fa2PtxKernels::load(
                    &kernels,
                    arch.head_dim_sliding as u32,
                )?);
                let global = AttentionBackend::Fa2Ptx(rvllm_attention::Fa2PtxKernels::load(
                    &kernels,
                    arch.head_dim_global as u32,
                )?);
                (sliding, global)
            } else {
                // Sliding layers use the FA3 SM90 backend at head_dim=256.
                let sliding = AttentionBackend::Fa3(Fa3Kernels::load(
                    paths.fa3_so.clone(),
                    arch.head_dim_sliding as u32,
                )?);
                // Global layers use the generic fallback paged attention path.
                // Default location is next to the FA3 .so; an explicit override
                // keeps bench/deploy flows flexible while avoiding a new required flag.
                let global_attention_so = std::env::var_os("RVLLM_FA_FALLBACK_SO")
                    .map(PathBuf::from)
                    .unwrap_or_else(|| paths.fa3_so.with_file_name("libfa_sm89_kernels.so"));
                let global = AttentionBackend::Fa3(Fa3Kernels::load(
                    global_attention_so,
                    arch.head_dim_global as u32,
                )?);
                (sliding, global)
            }
        };

        // Sm121 uses CutlassBackend::SoSm120 or Absent; neither consumes
        // the SM90 variant table, so the policy.json can stay unread on
        // that target. Saves a mandatory env var + avoids rejecting a
        // missing-or-placeholder file on GB10 runs.
        let skip_policy =
            matches!(compile_target, Some(rvllm_core::CompileTarget::Sm121));
        let (policy, variants): (Policy, Vec<_>) = if skip_policy {
            let empty = Policy {
                revision: String::new(),
                arch: "sm_121".into(),
                variants: Vec::new(),
                entries: Default::default(),
            };
            (empty, (0..16u32).map(rvllm_cutlass::VariantId).collect())
        } else {
            let policy_bytes = std::fs::read(&paths.policy_json)
                .map_err(|source| rvllm_core::RvllmError::Io {
                    err: rvllm_core::IoError::from(&source),
                    path: paths.policy_json.clone(),
                    source,
                })?;
            let policy: Policy = serde_json::from_slice(&policy_bytes).map_err(|e| {
                rvllm_core::RvllmError::config(
                    rvllm_core::ConfigError::Inconsistent {
                        reasons: vec![format!("policy.json parse: {e}")],
                    },
                    "policy.json",
                )
            })?;
            let mut variants: std::collections::BTreeSet<_> =
                policy.entries.values().map(|e| e.variant).collect();
            for v in 0..16u32 {
                variants.insert(rvllm_cutlass::VariantId(v));
            }
            (policy, variants.into_iter().collect())
        };
        // CUTLASS backend selection — see `bring_up::Bringup::load`
        // for the full rationale (sm_121 has no compatible `.so`).
        let cutlass =
            CutlassBackend::load_for(compile_target, paths.cutlass_so.clone(), &variants)?;

        let cublaslt_ws_bytes: usize = 32 * 1024 * 1024;
        let cublaslt_ws_region = arena.region("cublaslt_ws", cublaslt_ws_bytes, 256)?;
        let cublaslt = CublasLt::new(cublaslt_ws_region.device_ptr(), cublaslt_ws_bytes)?;
        let cublaslt_ws = HbmArenaCheckpoint {
            offset_bytes: 0,
            bytes: cublaslt_ws_bytes,
        };

        // `compile_target` is also what the fused loader uses to gate
        // `Fp8GemvVariant::WprNative` (sm_100+ only). `Some(None)` vs
        // `None` is distinct: it means "probe succeeded but CC isn't
        // in our target matrix", which falls back to `WprLut`.
        let fused = load_gemma4_fused(&kernels, compile_target)?;

        Ok(Self {
            ctx,
            arena,
            stream,
            arch,
            model,
            kernels,
            cutlass,
            cublaslt,
            cublaslt_ws,
            sliding_attention,
            global_attention,
            policy,
            fused,
            prefix_cache: std::sync::Mutex::new(None),
            // NVFP4 shadow diagnostic state (lazy-init in run_generate).
            nvfp4_shadow: std::sync::Mutex::new(None),
            nvfp4_shadow_dumped: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// Allocate the session-level prefix cache's KV cache region.
    /// Must be called BEFORE the cuda worker takes its scratch
    /// checkpoint — the region's device pointer is captured as a
    /// raw u64 and the arena's bump pointer is advanced past it,
    /// so subsequent `arena.restore(scratch_ck)` calls won't clobber
    /// the persistent KV data.
    ///
    /// Safe to call multiple times; becomes a no-op after the first
    /// successful init.
    pub fn init_prefix_cache(&self) -> Result<()> {
        let mut guard = self.prefix_cache.lock().unwrap();
        if guard.is_some() {
            return Ok(());
        }
        let arch = &self.arch;
        let block_size: u32 = 32;
        let num_blocks_total: u32 = std::env::var("RVLLM_NUM_BLOCKS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1024);
        let sliding_blocks = num_blocks_total;
        let kv_dtype = crate::gemma4_layer_exec::KvDtype::from_env(false);

        let mut kv_layer_offsets: Vec<u64> = Vec::with_capacity(arch.num_hidden_layers);
        let mut kv_scale_layer_offsets: Vec<u64> = Vec::with_capacity(arch.num_hidden_layers);
        let mut kv_total_bytes: u64 = 0;
        let mut kv_scale_total_bytes: u64 = 0;
        let mut kv_dtype_per_layer: Vec<crate::gemma4_layer_exec::KvDtype> =
            Vec::with_capacity(arch.num_hidden_layers);
        for l in 0..arch.num_hidden_layers {
            kv_layer_offsets.push(kv_total_bytes);
            kv_scale_layer_offsets.push(kv_scale_total_bytes);
            let is_global = arch.layer_types[l]
                == rvllm_loader::gemma4_arch::Gemma4LayerType::GlobalAttention;
            let layer_blocks = if is_global { num_blocks_total } else { sliding_blocks };
            let nkvh = arch.num_kv_heads_for_layer(l) as u32;
            let hd = arch.head_dim_for_layer(l) as u32;
            let layer_elems =
                2u64 * layer_blocks as u64 * block_size as u64 * nkvh as u64 * hd as u64;
            let kv_dtype_l = crate::gemma4_layer_exec::KvDtype::for_layer_or_env(
                arch.layer_types[l], false);
            kv_dtype_per_layer.push(kv_dtype_l);
            kv_total_bytes += match kv_dtype_l {
                crate::gemma4_layer_exec::KvDtype::F16 => layer_elems * 2,
                crate::gemma4_layer_exec::KvDtype::Fp8 => layer_elems,
                crate::gemma4_layer_exec::KvDtype::Nvfp4 => layer_elems / 2,
            };
            let layer_scale_slots =
                2u64 * layer_blocks as u64 * block_size as u64 * nkvh as u64;
            kv_scale_total_bytes += match kv_dtype_l {
                crate::gemma4_layer_exec::KvDtype::F16 => 0,
                crate::gemma4_layer_exec::KvDtype::Fp8 => layer_scale_slots * 4,
                crate::gemma4_layer_exec::KvDtype::Nvfp4 => layer_elems / 16,
            };
        }

        let kv_region = self.arena.region("persistent_kv", kv_total_bytes as usize, 256)?;
        let kv_cache_ptr = kv_region.device_ptr();
        let kv_scale_bytes_alloc = kv_scale_total_bytes.max(16) as usize;
        let kv_scale_region =
            self.arena.region("persistent_kv_scale", kv_scale_bytes_alloc, 16)?;
        let kv_scale_ptr = kv_scale_region.device_ptr();

        // Leak the Region wrappers by forgetting them. The arena's
        // bump pointer has already advanced past both regions, so
        // their bytes stay reserved for the lifetime of `self.arena`.
        // `arena.restore(scratch_ck)` run by the cuda worker only
        // restores to a checkpoint taken AFTER this call, so these
        // bytes are always above the worker's restore target.
        std::mem::forget(kv_region);
        std::mem::forget(kv_scale_region);

        #[cfg(feature = "cuda")]
        unsafe {
            cudarc::driver::sys::cuMemsetD8_v2(kv_cache_ptr, 0, kv_total_bytes as usize);
            cudarc::driver::sys::cuMemsetD8_v2(kv_scale_ptr, 0, kv_scale_bytes_alloc);
        }

        *guard = Some(PrefixCacheState {
            last_tokens: Vec::new(),
            kv_cache_ptr,
            kv_cache_bytes: kv_total_bytes,
            kv_scale_ptr,
            kv_scale_bytes: kv_scale_total_bytes,
            kv_dtype,
            kv_layer_offsets,
            kv_scale_layer_offsets,
            num_blocks_total,
            block_size,
            committed_prefix_len: 0,
            provenance: PrefixProvenance::from_env(),
        });
        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub unsafe fn run_bench(
        &self,
        num_seqs: u32,
        iters: u32,
        warmup: u32,
    ) -> crate::bring_up::BenchResult {
        use crate::gemma4_layer_exec::*;
        use rvllm_loader::gemma4_arch::Gemma4LayerType;

        let f16_only = false; // bench path always FP8
        let arch = &self.arch;
        let hidden = arch.hidden_size as u32;
        let max_hd = arch.max_head_dim() as u32;
        let max_nkvh = arch.max_kv_heads() as u32;
        let max_q_dim = (arch.num_attention_heads * arch.max_head_dim()) as u32;
        let max_kv_dim = (max_nkvh * max_hd) as u32;
        let max_qkv_rows = max_q_dim + 2 * max_kv_dim;
        let inter = arch.intermediate_size as u32;
        let vocab = arch.vocab_size as u32;
        let stream = self.stream.raw();

        let block_size: u32 = std::env::var("RVLLM_BLOCK_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(32);
        let num_blocks_total: u32 = 1024;
        let max_blocks_per_seq = (num_blocks_total / num_seqs).max(1);

        let arena = &self.arena;
        let hidden_fp8 = arena
            .region("hidden_fp8", (num_seqs * hidden) as usize, 16)
            .unwrap();
        let hidden_scale = arena
            .region("hidden_scale", (num_seqs * 4) as usize, 16)
            .unwrap();
        let qkv_out = arena
            .region("qkv_out", (num_seqs * max_qkv_rows * 2) as usize, 16)
            .unwrap();
        let q_base = qkv_out.device_ptr();
        let q_normed = arena
            .region("q_normed", (num_seqs * max_q_dim * 2) as usize, 16)
            .unwrap();
        let k_normed = arena
            .region("k_normed", (num_seqs * max_kv_dim * 2) as usize, 16)
            .unwrap();
        let v_normed = arena
            .region("v_normed", (num_seqs * max_kv_dim * 2) as usize, 16)
            .unwrap();
        let q_fp8 = arena
            .region("q_fp8", (num_seqs * max_q_dim) as usize, 16)
            .unwrap();
        let attn_out = arena
            .region("attn_out", (num_seqs * max_q_dim * 2) as usize, 16)
            .unwrap();
        let attn_out_fp8 = arena
            .region("attn_out_fp8", (num_seqs * max_q_dim) as usize, 16)
            .unwrap();
        let attn_out_scale = arena
            .region("attn_out_scale", (num_seqs * 4) as usize, 16)
            .unwrap();
        let gate_up_out = arena
            .region("gate_up_out", (num_seqs * 2 * inter * 2) as usize, 16)
            .unwrap();
        let gate_up_fp8 = arena
            .region("gate_up_fp8", (num_seqs * 2 * inter) as usize, 16)
            .unwrap();
        let gate_up_scale = arena
            .region("gate_up_scale", (num_seqs * 4) as usize, 16)
            .unwrap();
        let mlp_out_fp8 = arena
            .region("mlp_out_fp8", (num_seqs * inter) as usize, 16)
            .unwrap();
        let mlp_out_scale = arena
            .region("mlp_out_scale", (num_seqs * 4) as usize, 16)
            .unwrap();
        let delta_f16 = arena
            .region("delta_f16", (num_seqs * hidden * 2) as usize, 16)
            .unwrap();
        let gemm_f32_max_n = std::cmp::max(max_qkv_rows, 2 * inter);
        let gemm_f32_tmp = arena
            .region("gemm_f32_tmp", (num_seqs * gemm_f32_max_n * 4) as usize, 16)
            .unwrap();

        // Bench path KV dtype: env-driven via RVLLM_NVFP4_KV /
        // RVLLM_F16_KV (NVFP4-branch addition). FP8 remains the
        // pre-NVFP4 default.
        let kv_dtype = crate::gemma4_layer_exec::KvDtype::from_env(false);
        // aa01001pftrope0 cliff-fix (F-series): sliding layers need
        // `slot_mapping[t] < sliding_blocks * block_size` at every rope
        // write. The old cap `sliding_window/block_size` broke at
        // prompt_len > sliding_window because slot_mapping is linear
        // 0..prompt_len-1 and ran off the end. Give sliding layers the
        // full pool — ~10 GiB extra at num_blocks_total=1024, fits in
        // the 50+ GiB arena.
        let sliding_blocks = num_blocks_total;

        let mut kv_layer_offsets: Vec<u64> = Vec::with_capacity(arch.num_hidden_layers);
        let mut kv_scale_layer_offsets: Vec<u64> = Vec::with_capacity(arch.num_hidden_layers);
        let mut kv_total_bytes: u64 = 0;
        let mut kv_scale_total_bytes: u64 = 0;
        for l in 0..arch.num_hidden_layers {
            kv_layer_offsets.push(kv_total_bytes);
            kv_scale_layer_offsets.push(kv_scale_total_bytes);
            let layer_blocks = if arch.layer_types[l] == rvllm_loader::gemma4_arch::Gemma4LayerType::GlobalAttention { num_blocks_total } else { sliding_blocks };
            let nkvh_l = arch.num_kv_heads_for_layer(l) as u32;
            let hd_l = arch.head_dim_for_layer(l) as u32;
            let layer_elems = 2u64 * layer_blocks as u64 * block_size as u64 * nkvh_l as u64 * hd_l as u64;
            kv_total_bytes += match kv_dtype {
                crate::gemma4_layer_exec::KvDtype::F16 => layer_elems * 2,
                crate::gemma4_layer_exec::KvDtype::Fp8 => layer_elems,
                crate::gemma4_layer_exec::KvDtype::Nvfp4 => layer_elems / 2, // 2 elems/byte
            };
            // FP8 path: per-slot f32 K/V scales (F-series — one scale per
            // (block, tok, head) for both K and V). NVFP4 path: one E4M3
            // scale per 16 elems. F16 self-scaled so 0 bytes.
            let layer_scale_slots =
                2u64 * layer_blocks as u64 * block_size as u64 * nkvh_l as u64;
            kv_scale_total_bytes += match kv_dtype {
                crate::gemma4_layer_exec::KvDtype::F16 => 0,
                crate::gemma4_layer_exec::KvDtype::Fp8 => layer_scale_slots * 4,
                crate::gemma4_layer_exec::KvDtype::Nvfp4 => layer_elems / 16,
            };
        }
        let kv_cache = arena.region("kv_cache", kv_total_bytes as usize, 256).unwrap();
        let kv_scale_cache = arena.region(
            "kv_scale_cache", kv_scale_total_bytes as usize, 16).unwrap();
        // Per-(seq, head) Q scale scratch, written fresh by rope each
        // forward and consumed by the same step's attention.
        let q_scale_scratch_bytes =
            (num_seqs as u64) * (arch.num_attention_heads as u64) * 4;
        let q_scale_scratch = arena.region(
            "q_scale_scratch", q_scale_scratch_bytes as usize, 16).unwrap();
        // Opt-out for A/B testing: RVLLM_PER_TOKEN_Q_SCALE=0 falls back to
        // the scalar q_scale_ptr (pre-c69f641 behaviour) so PPL can be
        // compared across the two calibration strategies without a rebuild.
        let q_scale_cache_ptr: u64 =
            if std::env::var("RVLLM_PER_TOKEN_Q_SCALE").ok().as_deref() == Some("0") {
                0
            } else {
                q_scale_scratch.device_ptr()
            };
        #[cfg(feature = "cuda")]
        {
            cudarc::driver::sys::cuMemsetD8_v2(kv_cache.device_ptr(), 0, kv_total_bytes as usize);
            cudarc::driver::sys::cuMemsetD8_v2(
                kv_scale_cache.device_ptr(), 0, kv_scale_total_bytes as usize);
            cudarc::driver::sys::cuMemsetD8_v2(
                q_scale_scratch.device_ptr(), 0, q_scale_scratch_bytes as usize);
        }
        // Scale region only exists on the NVFP4 path; keep a 1-byte
        // placeholder region on F16/Fp8 so the Option branches in
        // scratch stay simple (0 pointer = don't touch).
        let kv_cache_scale = if kv_dtype.is_nvfp4() {
            let r = arena.region("kv_cache_scale", kv_scale_total_bytes as usize, 256).unwrap();
            #[cfg(feature = "cuda")]
            {
                cudarc::driver::sys::cuMemsetD8_v2(r.device_ptr(), 0, kv_scale_total_bytes as usize);
            }
            Some(r)
        } else {
            None
        };

        let q_scale_region = arena.region("q_scale", 4, 4).unwrap();
        let kv_scale_region = arena.region("kv_scale", 4, 4).unwrap();
        {
            let q_s: f32 = std::env::var("RVLLM_Q_SCALE")
                .ok().and_then(|v| v.parse().ok()).unwrap_or(DEFAULT_Q_SCALE);
            let kv_s: f32 = std::env::var("RVLLM_KV_SCALE")
                .ok().and_then(|v| v.parse().ok()).unwrap_or(DEFAULT_KV_SCALE);
            q_scale_region.copy_from_host(&q_s.to_le_bytes()).unwrap();
            kv_scale_region.copy_from_host(&kv_s.to_le_bytes()).unwrap();
        }

        let fa3_ws = arena.region("fa3_ws", 16 * 1024 * 1024, 256).unwrap();
        let residual = arena
            .region("residual", (num_seqs * hidden * 2) as usize, 16)
            .unwrap();
        cudarc::driver::sys::cuMemsetD8_v2(
            residual.device_ptr(),
            0,
            (num_seqs * hidden * 2) as usize,
        );

        let positions = arena
            .region("positions", (num_seqs * 4) as usize, 16)
            .unwrap();
        let slot_mapping = arena
            .region("slot_mapping", (num_seqs * 4) as usize, 16)
            .unwrap();
        let context_lens = arena
            .region("context_lens", (num_seqs * 4) as usize, 16)
            .unwrap();
        let block_tables = arena
            .region(
                "block_tables",
                (num_seqs * max_blocks_per_seq * 4) as usize,
                16,
            )
            .unwrap();
        {
            let n = num_seqs as usize;
            let pos: Vec<i32> = (0..n as i32).collect();
            let slot: Vec<i32> = (0..n as i32).collect();
            let ctx: Vec<i32> = vec![1; n];
            let mut bt: Vec<i32> = Vec::with_capacity(n * max_blocks_per_seq as usize);
            for i in 0..n as i32 {
                for b in 0..max_blocks_per_seq as i32 {
                    bt.push(i * max_blocks_per_seq as i32 + b);
                }
            }
            positions.copy_from_host(bytemuck_cast_i32(&pos)).unwrap();
            slot_mapping
                .copy_from_host(bytemuck_cast_i32(&slot))
                .unwrap();
            context_lens
                .copy_from_host(bytemuck_cast_i32(&ctx))
                .unwrap();
            block_tables.copy_from_host(bytemuck_cast_i32(&bt)).unwrap();
        }

        let logits = arena
            .region("logits", (num_seqs * vocab * 2) as usize, 16)
            .unwrap();
        let sampled_tokens = arena
            .region("sampled_tokens", (num_seqs * 4) as usize, 16)
            .unwrap();
        let cutlass_ws_bytes: usize = 16 * 1024 * 1024;
        let cutlass_ws = arena
            .region("cutlass_ws_gemma4", cutlass_ws_bytes, 256)
            .unwrap();
        let residual_ptr = residual.device_ptr();
        let kernels = self.layer_kernels();

        // GEMM plans — uniform shapes across all layers (the sliding/global
        // distinction is a runtime head reshape, weight dims are identical).
        // Use the sliding-layer dims for the plan since those are the common case.
        let q_dim_s = (arch.num_attention_heads * arch.head_dim_sliding) as u32;
        let kv_dim_s = (arch.num_kv_heads_sliding * arch.head_dim_sliding) as u32;
        let qkv_rows_s = q_dim_s + 2 * kv_dim_s;
        use rvllm_cutlass::Fp8GemmPlan;
        let _gemm_plans = Gemma4GemmPlans {
            qkv: Fp8GemmPlan::from_policy(
                &self.policy,
                num_seqs,
                qkv_rows_s,
                hidden,
                rvllm_core::DType::Fp8E4M3,
            )
            .unwrap(),
            o: Fp8GemmPlan::from_policy_residual(
                &self.policy,
                num_seqs,
                hidden,
                q_dim_s,
                rvllm_core::DType::Fp8E4M3,
            )
            .unwrap(),
            gate_up: Fp8GemmPlan::from_policy(
                &self.policy,
                num_seqs,
                2 * inter,
                hidden,
                rvllm_core::DType::Fp8E4M3,
            )
            .unwrap(),
            down: Fp8GemmPlan::from_policy_residual(
                &self.policy,
                num_seqs,
                hidden,
                inter,
                rvllm_core::DType::Fp8E4M3,
            )
            .unwrap(),
        };

        let one_step = || -> rvllm_core::Result<()> {
            for (layer_idx, layer) in self.model.layers.iter().enumerate() {
                let lt = arch.layer_types[layer_idx];
                let hd = arch.head_dim_for_layer(layer_idx) as u32;
                let nkvh = arch.num_kv_heads_for_layer(layer_idx) as u32;
                let q_dim = (arch.num_attention_heads as u32) * hd;
                let kv_dim = nkvh * hd;
                let _qkv_rows = q_dim + 2 * kv_dim;
                let layer_blocks = if lt == Gemma4LayerType::GlobalAttention { num_blocks_total } else { sliding_blocks };

                let dims = Gemma4LayerDims {
                    num_tokens: num_seqs,
                    hidden,
                    num_heads: arch.num_attention_heads as u32,
                    num_kv_heads: nkvh,
                    head_dim: hd,
                    rotary_dim: arch.rotary_dim_for_layer(layer_idx) as u32,
                    intermediate: inter,
                    block_size,
                    max_blocks_per_seq: layer_blocks,
                    num_blocks_total: layer_blocks,
                    attn_scale: 1.0,
                    rms_eps: arch.rms_norm_eps,
                    layer_type: lt,
                    sliding_window: arch.sliding_window_size as u32,
                    f16_kv: kv_dtype.is_f16(),
                    kv_dtype,
                    current_max_context_len: None,
                };

                // Row-major [num_tokens, q_dim+2*kv_dim]: k_out / v_out
                // point at row 0's K / V sub-slice. The rmsnorm kernel
                // applies `src_row_stride` to reach later tokens.
                let k_out = q_base + (q_dim as u64) * 2;
                let v_out = k_out + (kv_dim as u64) * 2;
                let is_global = lt == Gemma4LayerType::GlobalAttention;
                let layer_blocks = if is_global { num_blocks_total } else { sliding_blocks };
                let layer_kv_elems = 2u64 * layer_blocks as u64 * block_size as u64 * nkvh as u64 * hd as u64;
                let kv_layer_bytes = match kv_dtype {
                    crate::gemma4_layer_exec::KvDtype::F16 => layer_kv_elems * 2,
                    crate::gemma4_layer_exec::KvDtype::Fp8 => layer_kv_elems,
                    crate::gemma4_layer_exec::KvDtype::Nvfp4 => layer_kv_elems / 2,
                };
                let layer_kv_base = kv_cache.device_ptr() + kv_layer_offsets[layer_idx];
                // FP8 path (F-series): per-slot f32 K/V scales, always
                // allocated in `kv_scale_cache` (kv_scale_total_bytes=0 on F16).
                let layer_kv_scale_base =
                    kv_scale_cache.device_ptr() + kv_scale_layer_offsets[layer_idx];
                let layer_kv_scale_slots_half =
                    (layer_blocks as u64) * (block_size as u64) * (nkvh as u64);
                // NVFP4 path: K gets the first half of the layer's scale
                // slab, V the second half. layer_kv_elems covers K+V so
                // each half is `layer_kv_elems / 32` bytes (`/2/16`).
                let (k_cache_scale, v_cache_scale) = if kv_dtype
                    == crate::gemma4_layer_exec::KvDtype::Nvfp4
                {
                    let k_scale_bytes = layer_kv_elems / 32;
                    (layer_kv_scale_base, layer_kv_scale_base + k_scale_bytes)
                } else {
                    (0u64, 0u64)
                };

                let (cos, sin) = match lt {
                    Gemma4LayerType::SlidingAttention => (
                        self.model.rope_cos_sliding.offset_bytes,
                        self.model.rope_sin_sliding.offset_bytes,
                    ),
                    Gemma4LayerType::GlobalAttention => (
                        self.model.rope_cos_global.offset_bytes,
                        self.model.rope_sin_global.offset_bytes,
                    ),
                };

                let w = Gemma4LayerWeightPtrs {
                    attn_norm_gamma: layer.input_layernorm.offset_bytes,
                    post_attn_norm_gamma: layer.post_attention_layernorm.offset_bytes,
                    pre_ff_norm_gamma: layer.pre_feedforward_layernorm.offset_bytes,
                    post_ff_norm_gamma: layer.post_feedforward_layernorm.offset_bytes,
                    q_norm_gamma: layer.q_norm.offset_bytes,
                    k_norm_gamma: layer.k_norm.offset_bytes,
                    qkv_fp8: layer.qkv.offset_bytes,
                    qkv_scale: layer.qkv.scale_ptr,
                    o_fp8: layer.o_proj.offset_bytes,
                    o_scale: layer.o_proj.scale_ptr,
                    gate_up_fp8: layer.gate_up.offset_bytes,
                    gate_up_scale: layer.gate_up.scale_ptr,
                    down_fp8: layer.down_proj.offset_bytes,
                    down_scale: layer.down_proj.scale_ptr,
                    layer_scalar_ptr: layer.layer_scalar.offset_bytes,
                    qkv_f16: layer.qkv_f16.as_ref().map_or(0, |w| w.offset_bytes),
                    o_f16: layer.o_proj_f16.as_ref().map_or(0, |w| w.offset_bytes),
                    gate_up_f16: layer.gate_up_f16.as_ref().map_or(0, |w| w.offset_bytes),
                    down_f16: layer.down_proj_f16.as_ref().map_or(0, |w| w.offset_bytes),
                    qkv_chscale: layer.qkv.channelscale_ptr.unwrap_or(0),
                    o_chscale: layer.o_proj.channelscale_ptr.unwrap_or(0),
                    gate_up_chscale: layer.gate_up.channelscale_ptr.unwrap_or(0),
                    down_chscale: layer.down_proj.channelscale_ptr.unwrap_or(0),
                    qkv_blockscale: layer.qkv.blockscale_ptr.unwrap_or(0),
                    o_blockscale: layer.o_proj.blockscale_ptr.unwrap_or(0),
                    gate_up_blockscale: layer.gate_up.blockscale_ptr.unwrap_or(0),
                    down_blockscale: layer.down_proj.blockscale_ptr.unwrap_or(0),
                };

                let scratch = Gemma4LayerScratch {
                    hidden_fp8: hidden_fp8.device_ptr(),
                    hidden_scale: hidden_scale.device_ptr(),
                    q_out: q_base,
                    k_out,
                    v_out,
                    q_normed: q_normed.device_ptr(),
                    k_normed: k_normed.device_ptr(),
                    v_normed: v_normed.device_ptr(),
                    q_fp8: q_fp8.device_ptr(),
                    k_cache: layer_kv_base,
                    v_cache: layer_kv_base + kv_layer_bytes / 2,
                    k_scale_cache: layer_kv_scale_base,
                    v_scale_cache: layer_kv_scale_base + layer_kv_scale_slots_half * 4,
                    q_scale_cache: q_scale_cache_ptr,
                    k_cache_scale,
                    v_cache_scale,
                    q_scale_ptr: q_scale_region.device_ptr(),
                    kv_scale_ptr: kv_scale_region.device_ptr(),
                    attn_out: attn_out.device_ptr(),
                    attn_out_fp8: attn_out_fp8.device_ptr(),
                    attn_out_scale: attn_out_scale.device_ptr(),
                    delta_f16: delta_f16.device_ptr(),
                    gate_up_out: gate_up_out.device_ptr(),
                    gate_up_fp8: gate_up_fp8.device_ptr(),
                    gate_up_scale: gate_up_scale.device_ptr(),
                    mlp_out_fp8: mlp_out_fp8.device_ptr(),
                    mlp_out_scale: mlp_out_scale.device_ptr(),
                    gemm_f32_tmp: gemm_f32_tmp.device_ptr(),
                    cutlass_workspace: cutlass_ws.device_ptr(),
                    cutlass_workspace_bytes: cutlass_ws_bytes,
                    fa3_workspace: fa3_ws.device_ptr(),
                    // NVFP4 shadow diagnostic: default 0 (no shadow).
                    // Overridden in the run_generate decode path when
                    // RVLLM_NVFP4_SHADOW_F16 is on.
                    shadow_k_cache: 0,
                    shadow_v_cache: 0,
                    shadow_q_cache: 0,
                };

                let meta = Gemma4MetadataPtrs {
                    positions: positions.device_ptr(),
                    slot_mapping: slot_mapping.device_ptr(),
                    cos,
                    sin,
                    block_tables: block_tables.device_ptr(),
                    context_lens: context_lens.device_ptr(),
                };

                gemma4_forward(
                    dims,
                    &kernels,
                    &w,
                    &scratch,
                    &meta,
                    &self.cublaslt,
                    &self.cutlass,
                    &self.sliding_attention,
                    &self.global_attention,
                    residual_ptr,
                    stream,
                )?;
            }

            // LM head: final norm + FP8 quant + GEMM + softcap + argmax
            rvllm_fused::FusedRmsnormFp8QuantLaunch {
                num_tokens: num_seqs,
                hidden,
                eps: arch.rms_norm_eps,
            }
            .launch(
                kernels.fused_rmsnorm_fp8_quant,
                hidden_fp8.device_ptr(),
                hidden_scale.device_ptr(),
                residual_ptr,
                self.model.final_norm.offset_bytes,
                stream,
            )?;
            self.cublaslt.fp8_gemm(
                hidden_fp8.device_ptr(),
                self.model.lm_head_fp8.offset_bytes,
                logits.device_ptr(),
                num_seqs as i32,
                vocab as i32,
                hidden as i32,
                hidden_scale.device_ptr(),
                self.model.lm_head_fp8.scale_ptr,
                stream,
            )?;
            logit_softcap(
                self.fused.fn_softcap,
                logits.device_ptr(),
                num_seqs,
                vocab,
                arch.logit_softcap,
                stream,
            )?;
            rvllm_fused::ArgmaxLaunch {
                num_tokens: num_seqs,
                vocab,
            }
            .launch(
                self.fused.fn_argmax,
                logits.device_ptr(),
                sampled_tokens.device_ptr(),
                stream,
            )?;
            Ok(())
        };

        // Warmup
        for _ in 0..warmup {
            one_step().unwrap();
        }
        self.stream.fence().unwrap();

        // Timed
        let no_graph = std::env::var("RVLLM_NO_GRAPH").ok().as_deref() == Some("1");
        let elapsed = if no_graph {
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                one_step().unwrap();
            }
            self.stream.fence().unwrap();
            t0.elapsed()
        } else {
            let graph = rvllm_graph::CapturedGraph::capture(
                num_seqs,
                max_blocks_per_seq,
                rvllm_metadata::MetadataLayout::compute(num_seqs, max_blocks_per_seq).hash(),
                rvllm_graph::GraphFingerprint([0u8; 32]),
                stream,
                || one_step(),
            ).unwrap();
            self.stream.fence().unwrap();
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                graph.replay(stream).unwrap();
            }
            self.stream.fence().unwrap();
            t0.elapsed()
        };

        crate::bring_up::BenchResult {
            ns_per_step: elapsed.as_nanos() / iters.max(1) as u128,
            total_ns: elapsed.as_nanos(),
            iters,
            num_seqs,
            ttft_ns: None,
            ttft_hot_ns: None,
        }
    }

    #[cfg(feature = "cuda")]
    pub unsafe fn run_ppl(
        &self,
        fn_embed: rvllm_kernels::KernelFn,
        token_ids: &[u32],
    ) -> Result<crate::bring_up::PplResult> {
        use crate::bring_up::{dtoh_async_sync, f16_to_f32};
        use crate::gemma4_layer_exec::*;
        use rvllm_loader::gemma4_arch::Gemma4LayerType;

        let arch = &self.arch;
        let hidden = arch.hidden_size as u32;
        let max_hd = arch.max_head_dim() as u32;
        let max_nkvh = arch.max_kv_heads() as u32;
        let max_q_dim = (arch.num_attention_heads * arch.max_head_dim()) as u32;
        let max_kv_dim = (max_nkvh * max_hd) as u32;
        let max_qkv_rows = max_q_dim + 2 * max_kv_dim;
        let inter = arch.intermediate_size as u32;
        let vocab = arch.vocab_size as u32;
        let stream = self.stream.raw();
        let num_seqs: u32 = 1;

        let max_layers: usize = std::env::var("RVLLM_MAX_LAYERS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(arch.num_hidden_layers);
        let skip_softcap = std::env::var("RVLLM_NO_SOFTCAP").map_or(false, |v| v == "1");
        if max_layers < arch.num_hidden_layers {
            eprintln!(
                "[ppl] RVLLM_MAX_LAYERS={max_layers} (of {})",
                arch.num_hidden_layers
            );
        }
        if skip_softcap {
            eprintln!("[ppl] RVLLM_NO_SOFTCAP=1: softcap disabled");
        }
        eprintln!("[ppl] attn_scale=1.0 (Gemma4 QK-norm, no query_pre_attn_scalar)");

        let block_size: u32 = std::env::var("RVLLM_BLOCK_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(32);
        let num_blocks_total: u32 = std::env::var("RVLLM_NUM_BLOCKS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1024);
        let max_blocks_per_seq = (num_blocks_total / num_seqs).max(1);

        let arena = &self.arena;
        let hidden_fp8 = arena.region("hidden_fp8", (num_seqs * hidden) as usize, 16)?;
        let hidden_scale = arena.region("hidden_scale", (num_seqs * 4) as usize, 16)?;
        let qkv_out = arena.region("qkv_out", (num_seqs * max_qkv_rows * 2) as usize, 16)?;
        let q_base = qkv_out.device_ptr();
        let q_normed = arena.region("q_normed", (num_seqs * max_q_dim * 2) as usize, 16)?;
        let k_normed = arena.region("k_normed", (num_seqs * max_kv_dim * 2) as usize, 16)?;
        let v_normed = arena.region("v_normed", (num_seqs * max_kv_dim * 2) as usize, 16)?;
        let q_fp8 = arena.region("q_fp8", (num_seqs * max_q_dim) as usize, 16)?;
        let attn_out = arena.region("attn_out", (num_seqs * max_q_dim * 2) as usize, 16)?;
        let attn_out_fp8 = arena.region("attn_out_fp8", (num_seqs * max_q_dim) as usize, 16)?;
        let attn_out_scale = arena.region("attn_out_scale", (num_seqs * 4) as usize, 16)?;
        let gate_up_out = arena.region("gate_up_out", (num_seqs * 2 * inter * 2) as usize, 16)?;
        let gate_up_fp8 = arena.region("gate_up_fp8", (num_seqs * 2 * inter) as usize, 16)?;
        let gate_up_scale = arena.region("gate_up_scale", (num_seqs * 4) as usize, 16)?;
        let mlp_out_fp8 = arena.region("mlp_out_fp8", (num_seqs * inter) as usize, 16)?;
        let mlp_out_scale = arena.region("mlp_out_scale", (num_seqs * 4) as usize, 16)?;
        let delta_f16 = arena.region("delta_f16_ppl", (num_seqs * hidden * 2) as usize, 16)?;
        let gemm_f32_max_n = std::cmp::max(max_qkv_rows, 2 * inter);
        let gemm_f32_tmp = arena.region(
            "gemm_f32_tmp_ppl",
            (num_seqs * gemm_f32_max_n * 4) as usize,
            16,
        )?;

        let f16_only = std::env::var("RVLLM_F16_ONLY").map_or(false, |v| v == "1");
        let kv_dtype = crate::gemma4_layer_exec::KvDtype::from_env(f16_only);
        let kv_bytes_per_elem_log: u32 = match kv_dtype {
            crate::gemma4_layer_exec::KvDtype::F16 => 2,
            crate::gemma4_layer_exec::KvDtype::Fp8 => 1,
            crate::gemma4_layer_exec::KvDtype::Nvfp4 => 0, // logging placeholder
        };

        // Per-layer KV budget: sliding layers cap at sliding_window/block_size blocks,
        // global layers use full num_blocks_total. Saves ~5x KV memory for long context.
        // aa01001pftrope0 cliff-fix: sliding layers need `slot_mapping[t] < sliding_blocks*block_size`
        // at every t the rope writes; the old cap sliding_blocks = sliding_window/block_size = 32
        // (= 1024 slots for Gemma 4) broke at prompt_len > sliding_window because slot_mapping
        // is linear 0..prompt_len-1 and index 1024+ ran off the end of the sliding KV region.
        // Proper fix is a per-sliding-layer ring buffer (slot = t mod sliding_window) but that
        // needs rope + attention kernel cooperation. For now give sliding layers the full pool —
        // ~10 GiB extra at num_blocks_total=1024, fits in the 50 GiB arena with Gemma 4 31B fp8.
        let sliding_blocks = num_blocks_total;

        let mut kv_layer_offsets: Vec<u64> = Vec::with_capacity(arch.num_hidden_layers);
        let mut kv_scale_layer_offsets: Vec<u64> = Vec::with_capacity(arch.num_hidden_layers);
        let mut kv_total_bytes: u64 = 0;
        let mut kv_scale_total_bytes: u64 = 0;
        for l in 0..arch.num_hidden_layers {
            kv_layer_offsets.push(kv_total_bytes);
            kv_scale_layer_offsets.push(kv_scale_total_bytes);
            let is_global = arch.layer_types[l] == rvllm_loader::gemma4_arch::Gemma4LayerType::GlobalAttention;
            let layer_blocks = if is_global { num_blocks_total } else { sliding_blocks };
            let nkvh = arch.num_kv_heads_for_layer(l) as u32;
            let hd = arch.head_dim_for_layer(l) as u32;
            let layer_elems = 2u64 * layer_blocks as u64 * block_size as u64 * nkvh as u64 * hd as u64;
            kv_total_bytes += match kv_dtype {
                crate::gemma4_layer_exec::KvDtype::F16 => layer_elems * 2,
                crate::gemma4_layer_exec::KvDtype::Fp8 => layer_elems,
                crate::gemma4_layer_exec::KvDtype::Nvfp4 => layer_elems / 2,
            };
            // FP8 path: per-slot f32 K/V scales. NVFP4 path: one E4M3
            // scale per 16 elems. F16: self-scaled (0 bytes).
            let layer_scale_slots =
                2u64 * layer_blocks as u64 * block_size as u64 * nkvh as u64;
            kv_scale_total_bytes += match kv_dtype {
                crate::gemma4_layer_exec::KvDtype::F16 => 0,
                crate::gemma4_layer_exec::KvDtype::Fp8 => layer_scale_slots * 4,
                crate::gemma4_layer_exec::KvDtype::Nvfp4 => layer_elems / 16,
            };
        }
        eprintln!("[ppl] KV cache: {:.1} MB ({:?}, sliding={} blocks, global={} blocks, {} bytes/elem main)",
            kv_total_bytes as f64 / 1e6, kv_dtype, sliding_blocks, num_blocks_total, kv_bytes_per_elem_log);

        let kv_cache = arena.region("kv_cache", kv_total_bytes as usize, 256)?;
        cudarc::driver::sys::cuMemsetD8_v2(kv_cache.device_ptr(), 0, kv_total_bytes as usize);
        // Scale cache shared across FP8 (F-series per-slot f32 scales)
        // and NVFP4 (per-16-elem E4M3 scales). F16 path has 0 scale bytes
        // but we still allocate a placeholder region so region indexing
        // stays uniform.
        let kv_scale_cache =
            arena.region("kv_scale_cache", kv_scale_total_bytes.max(16) as usize, 16)?;
        cudarc::driver::sys::cuMemsetD8_v2(
            kv_scale_cache.device_ptr(), 0, kv_scale_total_bytes as usize);
        let q_scale_scratch_bytes =
            (num_seqs as u64) * (arch.num_attention_heads as u64) * 4;
        let q_scale_scratch = arena.region(
            "q_scale_scratch", q_scale_scratch_bytes as usize, 16)?;
        cudarc::driver::sys::cuMemsetD8_v2(
            q_scale_scratch.device_ptr(), 0, q_scale_scratch_bytes as usize);
        // See run_bench: RVLLM_PER_TOKEN_Q_SCALE=0 opts out.
        let q_scale_cache_ptr: u64 =
            if std::env::var("RVLLM_PER_TOKEN_Q_SCALE").ok().as_deref() == Some("0") {
                0
            } else {
                q_scale_scratch.device_ptr()
            };

        let q_scale_region = arena.region("q_scale", 4, 4)?;
        let kv_scale_region = arena.region("kv_scale", 4, 4)?;
        {
            let q_s: f32 = std::env::var("RVLLM_Q_SCALE")
                .ok().and_then(|v| v.parse().ok()).unwrap_or(DEFAULT_Q_SCALE);
            let kv_s: f32 = std::env::var("RVLLM_KV_SCALE")
                .ok().and_then(|v| v.parse().ok()).unwrap_or(DEFAULT_KV_SCALE);
            q_scale_region.copy_from_host(&q_s.to_le_bytes())?;
            kv_scale_region.copy_from_host(&kv_s.to_le_bytes())?;
        }

        let fa3_ws = arena.region("fa3_ws", 16 * 1024 * 1024, 256)?;
        let cutlass_ws_bytes: usize = 16 * 1024 * 1024;
        let cutlass_ws = arena.region("cutlass_ws_ppl", cutlass_ws_bytes, 256)?;

        let positions = arena.region("positions", (num_seqs * 4) as usize, 16)?;
        let slot_mapping = arena.region("slot_mapping", (num_seqs * 4) as usize, 16)?;
        let context_lens = arena.region("context_lens", (num_seqs * 4) as usize, 16)?;
        let block_tables = arena.region(
            "block_tables",
            (num_seqs * max_blocks_per_seq * 4) as usize,
            16,
        )?;
        {
            let mut bt: Vec<i32> = Vec::with_capacity(max_blocks_per_seq as usize);
            for b in 0..max_blocks_per_seq as i32 {
                bt.push(b);
            }
            block_tables.copy_from_host(bytemuck_cast_i32(&bt))?;
        }

        let residual = arena.region("residual", (num_seqs * hidden * 2) as usize, 16)?;
        let logits = arena.region("logits_ppl", (num_seqs * vocab * 2) as usize, 16)?;
        let logits_f32 = arena.region("logits_f32_ppl", (num_seqs * vocab * 4) as usize, 16)?;
        let token_ids_region = arena.region("token_ids_ppl", (num_seqs * 4) as usize, 16)?;
        let residual_ptr = residual.device_ptr();
        let kernels = self.layer_kernels();

        let q_dim_s = (arch.num_attention_heads * arch.head_dim_sliding) as u32;
        let kv_dim_s = (arch.num_kv_heads_sliding * arch.head_dim_sliding) as u32;
        let qkv_rows_s = q_dim_s + 2 * kv_dim_s;
        use rvllm_cutlass::Fp8GemmPlan;
        let _gemm_plans = Gemma4GemmPlans {
            qkv: Fp8GemmPlan::from_policy(
                &self.policy,
                num_seqs,
                qkv_rows_s,
                hidden,
                rvllm_core::DType::Fp8E4M3,
            )?,
            o: Fp8GemmPlan::from_policy_residual(
                &self.policy,
                num_seqs,
                hidden,
                q_dim_s,
                rvllm_core::DType::Fp8E4M3,
            )?,
            gate_up: Fp8GemmPlan::from_policy(
                &self.policy,
                num_seqs,
                2 * inter,
                hidden,
                rvllm_core::DType::Fp8E4M3,
            )?,
            down: Fp8GemmPlan::from_policy_residual(
                &self.policy,
                num_seqs,
                hidden,
                inter,
                rvllm_core::DType::Fp8E4M3,
            )?,
        };

        let step_counter = std::cell::Cell::new(0u32);
        let one_step = || -> Result<()> {
            for (layer_idx, layer) in self.model.layers.iter().enumerate() {
                if layer_idx >= max_layers {
                    break;
                }
                let lt = arch.layer_types[layer_idx];
                let hd = arch.head_dim_for_layer(layer_idx) as u32;
                let nkvh = arch.num_kv_heads_for_layer(layer_idx) as u32;
                let q_dim = (arch.num_attention_heads as u32) * hd;
                let kv_dim = nkvh * hd;
                let layer_blocks = if lt == Gemma4LayerType::GlobalAttention { num_blocks_total } else { sliding_blocks };

                let dims = Gemma4LayerDims {
                    num_tokens: num_seqs,
                    hidden,
                    num_heads: arch.num_attention_heads as u32,
                    num_kv_heads: nkvh,
                    head_dim: hd,
                    rotary_dim: arch.rotary_dim_for_layer(layer_idx) as u32,
                    intermediate: inter,
                    block_size,
                    max_blocks_per_seq: layer_blocks,
                    num_blocks_total: layer_blocks,
                    attn_scale: 1.0,
                    rms_eps: arch.rms_norm_eps,
                    layer_type: lt,
                    sliding_window: arch.sliding_window_size as u32,
                    f16_kv: kv_dtype.is_f16(),
                    kv_dtype,
                    current_max_context_len: None,
                };

                // Row-major [num_tokens, q_dim+2*kv_dim]: k_out / v_out
                // point at row 0's K / V sub-slice. The rmsnorm kernel
                // applies `src_row_stride` to reach later tokens.
                let k_out = q_base + (q_dim as u64) * 2;
                let v_out = k_out + (kv_dim as u64) * 2;
                let is_global = lt == Gemma4LayerType::GlobalAttention;
                let layer_blocks = if is_global { num_blocks_total } else { sliding_blocks };
                let layer_kv_elems = 2u64 * layer_blocks as u64 * block_size as u64 * nkvh as u64 * hd as u64;
                let kv_layer_bytes = match kv_dtype {
                    crate::gemma4_layer_exec::KvDtype::F16 => layer_kv_elems * 2,
                    crate::gemma4_layer_exec::KvDtype::Fp8 => layer_kv_elems,
                    crate::gemma4_layer_exec::KvDtype::Nvfp4 => layer_kv_elems / 2,
                };
                let layer_kv_base = kv_cache.device_ptr() + kv_layer_offsets[layer_idx];
                let layer_kv_scale_base =
                    kv_scale_cache.device_ptr() + kv_scale_layer_offsets[layer_idx];
                let layer_kv_scale_slots_half =
                    (layer_blocks as u64) * (block_size as u64) * (nkvh as u64);
                let (k_cache_scale, v_cache_scale) = if kv_dtype
                    == crate::gemma4_layer_exec::KvDtype::Nvfp4
                {
                    (layer_kv_scale_base, layer_kv_scale_base + layer_kv_elems / 32)
                } else {
                    (0u64, 0u64)
                };
                let (cos, sin) = match lt {
                    Gemma4LayerType::SlidingAttention => (
                        self.model.rope_cos_sliding.offset_bytes,
                        self.model.rope_sin_sliding.offset_bytes,
                    ),
                    Gemma4LayerType::GlobalAttention => (
                        self.model.rope_cos_global.offset_bytes,
                        self.model.rope_sin_global.offset_bytes,
                    ),
                };

                let w = Gemma4LayerWeightPtrs {
                    attn_norm_gamma: layer.input_layernorm.offset_bytes,
                    post_attn_norm_gamma: layer.post_attention_layernorm.offset_bytes,
                    pre_ff_norm_gamma: layer.pre_feedforward_layernorm.offset_bytes,
                    post_ff_norm_gamma: layer.post_feedforward_layernorm.offset_bytes,
                    q_norm_gamma: layer.q_norm.offset_bytes,
                    k_norm_gamma: layer.k_norm.offset_bytes,
                    qkv_fp8: layer.qkv.offset_bytes,
                    qkv_scale: layer.qkv.scale_ptr,
                    o_fp8: layer.o_proj.offset_bytes,
                    o_scale: layer.o_proj.scale_ptr,
                    gate_up_fp8: layer.gate_up.offset_bytes,
                    gate_up_scale: layer.gate_up.scale_ptr,
                    down_fp8: layer.down_proj.offset_bytes,
                    down_scale: layer.down_proj.scale_ptr,
                    layer_scalar_ptr: layer.layer_scalar.offset_bytes,
                    qkv_f16: layer.qkv_f16.as_ref().map_or(0, |w| w.offset_bytes),
                    o_f16: layer.o_proj_f16.as_ref().map_or(0, |w| w.offset_bytes),
                    gate_up_f16: layer.gate_up_f16.as_ref().map_or(0, |w| w.offset_bytes),
                    down_f16: layer.down_proj_f16.as_ref().map_or(0, |w| w.offset_bytes),
                    qkv_chscale: layer.qkv.channelscale_ptr.unwrap_or(0),
                    o_chscale: layer.o_proj.channelscale_ptr.unwrap_or(0),
                    gate_up_chscale: layer.gate_up.channelscale_ptr.unwrap_or(0),
                    down_chscale: layer.down_proj.channelscale_ptr.unwrap_or(0),
                    qkv_blockscale: layer.qkv.blockscale_ptr.unwrap_or(0),
                    o_blockscale: layer.o_proj.blockscale_ptr.unwrap_or(0),
                    gate_up_blockscale: layer.gate_up.blockscale_ptr.unwrap_or(0),
                    down_blockscale: layer.down_proj.blockscale_ptr.unwrap_or(0),
                };

                let scratch = Gemma4LayerScratch {
                    hidden_fp8: hidden_fp8.device_ptr(),
                    hidden_scale: hidden_scale.device_ptr(),
                    q_out: q_base,
                    k_out,
                    v_out,
                    q_normed: q_normed.device_ptr(),
                    k_normed: k_normed.device_ptr(),
                    v_normed: v_normed.device_ptr(),
                    q_fp8: q_fp8.device_ptr(),
                    k_cache: layer_kv_base,
                    v_cache: layer_kv_base + kv_layer_bytes / 2,
                    k_scale_cache: layer_kv_scale_base,
                    v_scale_cache: layer_kv_scale_base + layer_kv_scale_slots_half * 4,
                    q_scale_cache: q_scale_cache_ptr,
                    k_cache_scale,
                    v_cache_scale,
                    q_scale_ptr: q_scale_region.device_ptr(),
                    kv_scale_ptr: kv_scale_region.device_ptr(),
                    attn_out: attn_out.device_ptr(),
                    attn_out_fp8: attn_out_fp8.device_ptr(),
                    attn_out_scale: attn_out_scale.device_ptr(),
                    delta_f16: delta_f16.device_ptr(),
                    gate_up_out: gate_up_out.device_ptr(),
                    gate_up_fp8: gate_up_fp8.device_ptr(),
                    gate_up_scale: gate_up_scale.device_ptr(),
                    mlp_out_fp8: mlp_out_fp8.device_ptr(),
                    mlp_out_scale: mlp_out_scale.device_ptr(),
                    gemm_f32_tmp: gemm_f32_tmp.device_ptr(),
                    cutlass_workspace: cutlass_ws.device_ptr(),
                    cutlass_workspace_bytes: cutlass_ws_bytes,
                    fa3_workspace: fa3_ws.device_ptr(),
                    // NVFP4 shadow diagnostic: default 0 (no shadow).
                    // Overridden in the run_generate decode path when
                    // RVLLM_NVFP4_SHADOW_F16 is on.
                    shadow_k_cache: 0,
                    shadow_v_cache: 0,
                    shadow_q_cache: 0,
                };

                let meta = Gemma4MetadataPtrs {
                    positions: positions.device_ptr(),
                    slot_mapping: slot_mapping.device_ptr(),
                    cos,
                    sin,
                    block_tables: block_tables.device_ptr(),
                    context_lens: context_lens.device_ptr(),
                };

                gemma4_forward(
                    dims,
                    &kernels,
                    &w,
                    &scratch,
                    &meta,
                    &self.cublaslt,
                    &self.cutlass,
                    &self.sliding_attention,
                    &self.global_attention,
                    residual_ptr,
                    stream,
                )?;

                if step_counter.get() == 0 && layer_idx == 0 {
                    cudarc::driver::sys::cuStreamSynchronize(stream as _);
                    let mut s = [0u16; 4];
                    cudarc::driver::sys::cuMemcpyDtoH_v2(s.as_mut_ptr() as *mut _, residual_ptr, 8);
                    let v: Vec<f32> = s.iter().map(|&x| f16_to_f32(x)).collect();
                    let mut amax = 0f32;
                    let n = hidden as usize;
                    let mut all = vec![0u16; n];
                    cudarc::driver::sys::cuMemcpyDtoH_v2(
                        all.as_mut_ptr() as *mut _,
                        residual_ptr,
                        (n * 2) as _,
                    );
                    for &b in &all {
                        let f = f16_to_f32(b).abs();
                        if f > amax && !f.is_nan() {
                            amax = f;
                        }
                    }
                    eprintln!("  [ppl L0] residual first4={:.6?} amax={:.6}", v, amax);
                    // Check layer_scalar value
                    let mut sc = [0u16; 1];
                    cudarc::driver::sys::cuMemcpyDtoH_v2(
                        sc.as_mut_ptr() as *mut _,
                        layer.layer_scalar.offset_bytes,
                        2,
                    );
                    eprintln!("  [ppl L0] layer_scalar={:.6}", f16_to_f32(sc[0]));
                    // Check norm gamma amax
                    let mut ng = vec![0u16; n];
                    cudarc::driver::sys::cuMemcpyDtoH_v2(
                        ng.as_mut_ptr() as *mut _,
                        layer.input_layernorm.offset_bytes,
                        (n * 2) as _,
                    );
                    let gamma_amax = ng.iter().map(|&b| f16_to_f32(b).abs()).fold(0f32, f32::max);
                    eprintln!("  [ppl L0] input_norm_gamma amax={:.6}", gamma_amax);
                }
                if step_counter.get() == 0 && layer_idx < 3 && layer_idx > 0 {
                    cudarc::driver::sys::cuStreamSynchronize(stream as _);
                    let mut s = [0u16; 4];
                    cudarc::driver::sys::cuMemcpyDtoH_v2(s.as_mut_ptr() as *mut _, residual_ptr, 8);
                    let v: Vec<f32> = s.iter().map(|&x| f16_to_f32(x)).collect();
                    eprintln!("  [ppl L{}] residual={:.4?}", layer_idx, v);
                }
            }

            // LM head: final norm (f16 in-place) + f16 GEMM -> f32 logits
            let dbg_lmhead = step_counter.get() == 0 && std::env::var("RVLLM_DBG_LAYER").is_ok();

            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: num_seqs,
                hidden,
                eps: arch.rms_norm_eps,
            }
            .launch(
                kernels.fused_rmsnorm,
                residual_ptr,
                self.model.final_norm.offset_bytes,
                stream,
            )?;
            if dbg_lmhead {
                cudarc::driver::sys::cuStreamSynchronize(stream as _);
                let mut s = [0u16; 4];
                cudarc::driver::sys::cuMemcpyDtoH_v2(s.as_mut_ptr() as *mut _, residual_ptr, 8);
                let v: Vec<f32> = s.iter().map(|&x| crate::bring_up::f16_to_f32(x)).collect();
                eprintln!("  [lm_head] after rmsnorm_f16: first4={:.4?}", v);
            }
            self.cublaslt.f16_gemm_f32(
                residual_ptr,
                self.model.lm_head_f16.offset_bytes,
                logits_f32.device_ptr(),
                num_seqs as i32,
                vocab as i32,
                hidden as i32,
                stream,
            )?;
            if dbg_lmhead {
                cudarc::driver::sys::cuStreamSynchronize(stream as _);
                let total = (vocab as usize) * (num_seqs as usize);
                let mut buf = vec![0.0f32; total];
                cudarc::driver::sys::cuMemcpyDtoH_v2(
                    buf.as_mut_ptr() as *mut _,
                    logits_f32.device_ptr(),
                    (total * 4) as _,
                );
                let amax = buf.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                eprintln!(
                    "  [lm_head] raw_f32_logits first8={:.4?} amax={:.6e} (n={})",
                    &buf[..8.min(total)],
                    amax,
                    total
                );
            }
            rvllm_fused::gemma4_launcher::Bf16ToF16SatLaunch {
                n: num_seqs * vocab,
            }
            .launch(
                kernels.f32_to_f16_sat,
                logits.device_ptr(),
                logits_f32.device_ptr(),
                stream,
            )?;
            if dbg_lmhead {
                cudarc::driver::sys::cuStreamSynchronize(stream as _);
                let mut s = [0u16; 4];
                cudarc::driver::sys::cuMemcpyDtoH_v2(
                    s.as_mut_ptr() as *mut _,
                    logits.device_ptr(),
                    8,
                );
                let v: Vec<f32> = s.iter().map(|&x| f16_to_f32(x)).collect();
                eprintln!(
                    "  [lm_head] after f32_to_f16_sat: logits_f16 first4={:.4?}",
                    v
                );
            }
            if !skip_softcap {
                logit_softcap(
                    self.fused.fn_softcap,
                    logits.device_ptr(),
                    num_seqs,
                    vocab,
                    arch.logit_softcap,
                    stream,
                )?;
            }
            if dbg_lmhead {
                cudarc::driver::sys::cuStreamSynchronize(stream as _);
                let mut s = [0u16; 4];
                cudarc::driver::sys::cuMemcpyDtoH_v2(
                    s.as_mut_ptr() as *mut _,
                    logits.device_ptr(),
                    8,
                );
                let v: Vec<f32> = s.iter().map(|&x| f16_to_f32(x)).collect();
                eprintln!("  [lm_head] after softcap: logits_f16 first4={:.4?}", v);
            }
            step_counter.set(step_counter.get() + 1);
            Ok(())
        };

        let set_step_meta = |step: i32| -> Result<()> {
            let pos = [step];
            let slot = [step];
            let ctx = [step + 1];
            positions.copy_from_host(bytemuck_cast_i32(&pos))?;
            slot_mapping.copy_from_host(bytemuck_cast_i32(&slot))?;
            context_lens.copy_from_host(bytemuck_cast_i32(&ctx))?;
            Ok(())
        };

        let logits_row_elems = vocab as usize;
        let logits_row_bytes_f32 = logits_row_elems * 4;
        let mut logits_host_f32: Vec<f32> = vec![0.0f32; logits_row_elems];
        let mut total_nll: f64 = 0.0;
        let mut n_evaluated: usize = 0;

        // Build a graph-capturable forward: embed + all layers + lm_head.
        // No debug probes (they break capture).
        let ppl_forward = || -> Result<()> {
            rvllm_fused::EmbeddingGatherLaunch { num_tokens: 1, hidden, vocab }
                .launch(fn_embed, residual_ptr, self.model.embedding.offset_bytes, token_ids_region.device_ptr(), stream)?;
            one_step()
        };

        let use_graph = std::env::var("RVLLM_NO_GRAPH").ok().as_deref() != Some("1");
        let ppl_graph = if use_graph {
            // Dry run to populate KV cache slot 0
            let tok_i32 = [token_ids[0] as i32];
            token_ids_region.copy_from_host(bytemuck_cast_i32(&tok_i32))?;
            set_step_meta(0)?;
            ppl_forward()?;
            self.stream.fence()?;

            let g = rvllm_graph::CapturedGraph::capture(
                num_seqs,
                max_blocks_per_seq,
                rvllm_metadata::MetadataLayout::compute(num_seqs, max_blocks_per_seq).hash(),
                rvllm_graph::GraphFingerprint([0u8; 32]),
                stream,
                || ppl_forward(),
            )?;
            self.stream.fence()?;
            Some(g)
        } else {
            None
        };

        for (t, &tok_id) in token_ids.iter().enumerate() {
            let tok_i32 = [tok_id as i32];
            token_ids_region.copy_from_host(bytemuck_cast_i32(&tok_i32))?;
            set_step_meta(t as i32)?;

            if let Some(ref graph) = ppl_graph {
                graph.replay(stream)?;
            } else {
                ppl_forward()?;
            }

            if t + 1 < token_ids.len() {
                dtoh_async_sync(
                    logits_f32.device_ptr(),
                    logits_host_f32.as_mut_ptr() as *mut i32,
                    logits_row_bytes_f32,
                    stream,
                )?;
                self.stream.fence()?;

                let cap = arch.logit_softcap;
                if !skip_softcap && cap > 0.0 {
                    for x in logits_host_f32.iter_mut() {
                        *x = cap * (*x / cap).tanh();
                    }
                }

                let target = token_ids[t + 1] as usize;
                if t == 0 {
                    let first5: Vec<f32> = logits_host_f32[..5].to_vec();
                    let max_val = logits_host_f32
                        .iter()
                        .copied()
                        .filter(|v| !v.is_nan())
                        .fold(f32::MIN, f32::max);
                    let min_val = logits_host_f32
                        .iter()
                        .copied()
                        .filter(|v| !v.is_nan())
                        .fold(f32::MAX, f32::min);
                    eprintln!(
                        "  [ppl] logits(f32+softcap): first5={:?} min={:.1} max={:.1}",
                        first5, min_val, max_val
                    );
                }
                let nll = crate::bring_up::compute_nll_f32(&logits_host_f32, target);
                total_nll += nll;
                n_evaluated += 1;

                if (t + 1) % 32 == 0 || t + 1 == token_ids.len() - 1 {
                    let running_ppl = (total_nll / n_evaluated as f64).exp();
                    eprintln!(
                        "  step {}/{}: running_ppl={:.4}",
                        t + 1,
                        token_ids.len(),
                        running_ppl
                    );
                }
            } else {
                self.stream.fence()?;
            }
        }

        let ppl = if n_evaluated > 0 {
            (total_nll / n_evaluated as f64).exp()
        } else {
            0.0
        };
        Ok(crate::bring_up::PplResult {
            ppl,
            total_nll,
            n_evaluated,
        })
    }

    #[cfg(feature = "cuda")]
    pub unsafe fn run_generate(
        &self,
        fn_embed: rvllm_kernels::KernelFn,
        fn_argmax: rvllm_kernels::KernelFn,
        prompt_ids: &[u32],
        max_new: usize,
        eos_ids: &[u32],
        // === NVFP4 SHADOW DIAGNOSTIC === per-request opt-in. When
        // false, the shadow allocator and dump hook do nothing
        // regardless of env settings. Gates the feature at the
        // request boundary so upstream-client scaffold calls
        // (zeroclaw classifier, internal probes) cannot accidentally
        // burn the one-shot latch.
        shadow_requested: bool,
    ) -> Result<Vec<u32>> {
        let arch = &self.arch;
        let hidden = arch.hidden_size as u32;
        let vocab = arch.vocab_size as u32;
        let stream = self.stream.raw();

        let block_size: u32 = 32;
        let num_blocks_total: u32 = std::env::var("RVLLM_NUM_BLOCKS")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(1024);

        let arena = &self.arena;
        let max_hd = arch.max_head_dim() as u32;
        let max_nkvh = arch.max_kv_heads() as u32;
        let max_q_dim = (arch.num_attention_heads * arch.max_head_dim()) as u32;
        let max_kv_dim = (max_nkvh * max_hd) as u32;
        let max_qkv_rows = max_q_dim + 2 * max_kv_dim;
        let inter = arch.intermediate_size as u32;
        let max_blocks_per_seq = num_blocks_total;

        let prompt_len = prompt_ids.len() as u32;
        let max_tokens = prompt_len.max(1);

        let hidden_fp8 = arena.region("gen_hidden_fp8", (max_tokens * hidden) as usize, 16)?;
        let hidden_scale = arena.region("gen_hidden_scale", (max_tokens * 4) as usize, 16)?;
        let qkv_out = arena.region("gen_qkv", (max_tokens * max_qkv_rows * 2) as usize, 16)?;
        let q_base = qkv_out.device_ptr();
        let q_normed = arena.region("gen_q_normed", (max_tokens * max_q_dim * 2) as usize, 16)?;
        let k_normed = arena.region("gen_k_normed", (max_tokens * max_kv_dim * 2) as usize, 16)?;
        let v_normed = arena.region("gen_v_normed", (max_tokens * max_kv_dim * 2) as usize, 16)?;
        let q_fp8 = arena.region("gen_q_fp8", (max_tokens * max_q_dim) as usize, 16)?;
        let attn_out = arena.region("gen_attn_out", (max_tokens * max_q_dim * 2) as usize, 16)?;
        let attn_out_fp8 = arena.region("gen_attn_out_fp8", (max_tokens * max_q_dim) as usize, 16)?;
        let attn_out_scale = arena.region("gen_attn_out_scale", (max_tokens * 4) as usize, 16)?;
        let gate_up_out = arena.region("gen_gate_up", (max_tokens * 2 * inter * 2) as usize, 16)?;
        let gate_up_fp8 = arena.region("gen_gate_up_fp8", (max_tokens * 2 * inter) as usize, 16)?;
        let gate_up_scale = arena.region("gen_gate_up_scale", (max_tokens * 4) as usize, 16)?;
        let mlp_out_fp8 = arena.region("gen_mlp_fp8", (max_tokens * inter) as usize, 16)?;
        let mlp_out_scale = arena.region("gen_mlp_scale", (max_tokens * 4) as usize, 16)?;
        let delta_f16 = arena.region("gen_delta", (max_tokens * hidden * 2) as usize, 16)?;
        let gemm_f32_max_n = std::cmp::max(max_qkv_rows, 2 * inter);
        let gemm_f32_tmp = arena.region("gen_gemm_f32", (max_tokens * gemm_f32_max_n * 4) as usize, 16)?;

        // Prefix cache: use the persistent KV region allocated by
        // `init_prefix_cache`. If the cache wasn't pre-initialised
        // (old callers, rvllm-bench / probe), fall back to per-call
        // arena allocation — no cache hit available in that case.
        let sliding_blocks = num_blocks_total;
        let kv_dtype = crate::gemma4_layer_exec::KvDtype::from_env(false);
        let (kv_cache_ptr, kv_scale_ptr, kv_layer_offsets, kv_scale_layer_offsets,
             kv_total_bytes, kv_scale_total_bytes, common_prefix_len_raw) = {
            let guard = self.prefix_cache.lock().unwrap();
            match &*guard {
                Some(pc) => {
                    // Cache hit path: compute the longest common
                    // prefix in raw token ids.
                    let mut prefix = 0usize;
                    while prefix < pc.last_tokens.len()
                        && prefix < prompt_ids.len()
                        && pc.last_tokens[prefix] == prompt_ids[prefix]
                    {
                        prefix += 1;
                    }
                    // Provenance check: invalidate cache entirely
                    // when batch shape / dtype / hybrid / scale
                    // policy / prefill mode differs from when the
                    // KV was written. Optimized NVFP4 kernels are
                    // batch-variant; reusing KV across mismatched
                    // policies produces silent miscompare.
                    let cur_prov = PrefixProvenance::from_env();
                    if cur_prov != pc.provenance {
                        eprintln!(
                            "[prefix-cache] provenance mismatch — invalidating \
                             (was {:?}, now {:?})",
                            pc.provenance, cur_prov
                        );
                        prefix = 0;
                    } else {
                        // Chunk-shape cap: only reuse up to the last
                        // FULLY-WRITTEN chunk boundary of the
                        // previous request. Slots written by a
                        // shorter trailing chunk are unsafe to reuse
                        // because they were quantized under a
                        // different batch shape. Fixes "la la la"
                        // garbage on classifier-then-persona chains.
                        let cap = pc.committed_prefix_len as usize;
                        if cap < prefix {
                            eprintln!(
                                "[prefix-cache] capping reuse {}→{} \
                                 (last committed chunk boundary)",
                                prefix, cap
                            );
                            prefix = cap;
                        }
                    }
                    // Leave at least one token for the prefill to
                    // process (otherwise there's nothing to decode
                    // the last hidden state from).
                    if prefix >= prompt_ids.len() {
                        prefix = prompt_ids.len().saturating_sub(1);
                    }
                    (
                        pc.kv_cache_ptr,
                        pc.kv_scale_ptr,
                        pc.kv_layer_offsets.clone(),
                        pc.kv_scale_layer_offsets.clone(),
                        pc.kv_cache_bytes,
                        pc.kv_scale_bytes,
                        prefix as u32,
                    )
                }
                None => (0, 0, Vec::new(), Vec::new(), 0, 0, 0),
            }
        };

        let use_prefix_cache = kv_cache_ptr != 0;
        // Per-call fallback when cache wasn't initialised.
        let _kv_cache_region;
        let _kv_scale_region;
        let (kv_cache_ptr, kv_scale_ptr,
             kv_layer_offsets, kv_scale_layer_offsets,
             kv_total_bytes, kv_scale_total_bytes) = if use_prefix_cache {
            (kv_cache_ptr, kv_scale_ptr,
             kv_layer_offsets, kv_scale_layer_offsets,
             kv_total_bytes, kv_scale_total_bytes)
        } else {
            let mut kv_layer_offsets: Vec<u64> = Vec::with_capacity(arch.num_hidden_layers);
            let mut kv_scale_layer_offsets: Vec<u64> = Vec::with_capacity(arch.num_hidden_layers);
            let mut kv_total_bytes: u64 = 0;
            let mut kv_scale_total_bytes: u64 = 0;
            for l in 0..arch.num_hidden_layers {
                kv_layer_offsets.push(kv_total_bytes);
                kv_scale_layer_offsets.push(kv_scale_total_bytes);
                let is_global = arch.layer_types[l]
                    == rvllm_loader::gemma4_arch::Gemma4LayerType::GlobalAttention;
                let layer_blocks = if is_global { num_blocks_total } else { sliding_blocks };
                let nkvh = arch.num_kv_heads_for_layer(l) as u32;
                let hd = arch.head_dim_for_layer(l) as u32;
                let layer_elems =
                    2u64 * layer_blocks as u64 * block_size as u64 * nkvh as u64 * hd as u64;
                let kv_dtype_l = crate::gemma4_layer_exec::KvDtype::for_layer_or_env(
                    arch.layer_types[l], false);
                kv_total_bytes += match kv_dtype_l {
                    crate::gemma4_layer_exec::KvDtype::F16 => layer_elems * 2,
                    crate::gemma4_layer_exec::KvDtype::Fp8 => layer_elems,
                    crate::gemma4_layer_exec::KvDtype::Nvfp4 => layer_elems / 2,
                };
                let layer_scale_slots =
                    2u64 * layer_blocks as u64 * block_size as u64 * nkvh as u64;
                kv_scale_total_bytes += match kv_dtype_l {
                    crate::gemma4_layer_exec::KvDtype::F16 => 0,
                    crate::gemma4_layer_exec::KvDtype::Fp8 => layer_scale_slots * 4,
                    crate::gemma4_layer_exec::KvDtype::Nvfp4 => layer_elems / 16,
                };
            }
            let kvr = arena.region("gen_kv", kv_total_bytes as usize, 256)?;
            cudarc::driver::sys::cuMemsetD8_v2(kvr.device_ptr(), 0, kv_total_bytes as usize);
            let kvs = arena.region(
                "gen_kv_scale_cache", kv_scale_total_bytes.max(16) as usize, 16)?;
            cudarc::driver::sys::cuMemsetD8_v2(
                kvs.device_ptr(), 0, kv_scale_total_bytes as usize);
            let kc_ptr = kvr.device_ptr();
            let ks_ptr = kvs.device_ptr();
            _kv_cache_region = kvr;
            _kv_scale_region = kvs;
            (kc_ptr, ks_ptr, kv_layer_offsets, kv_scale_layer_offsets,
             kv_total_bytes, kv_scale_total_bytes)
        };

        // Clamp the prefix-match to the actual KV region size.
        let common_prefix_len: u32 = if use_prefix_cache {
            common_prefix_len_raw
        } else {
            0
        };
        if common_prefix_len > 0 {
            eprintln!(
                "[prefix-cache] hit: reusing {} of {} prompt tokens",
                common_prefix_len, prompt_len
            );
        }
        // Wrap the raw pointers so downstream `.device_ptr()` calls
        // stay source-identical across the cache-hit and fallback
        // paths. This is a 16-byte local, zero runtime cost.
        struct KvHandle(u64);
        impl KvHandle {
            fn device_ptr(&self) -> u64 { self.0 }
        }
        let kv_cache = KvHandle(kv_cache_ptr);
        let kv_scale_cache = KvHandle(kv_scale_ptr);
        let q_scale_scratch_bytes =
            (max_tokens as u64) * (arch.num_attention_heads as u64) * 4;
        let q_scale_scratch = arena.region(
            "gen_q_scale_scratch", q_scale_scratch_bytes as usize, 16)?;
        cudarc::driver::sys::cuMemsetD8_v2(
            q_scale_scratch.device_ptr(), 0, q_scale_scratch_bytes as usize);
        // See run_bench: RVLLM_PER_TOKEN_Q_SCALE=0 opts out.
        let q_scale_cache_ptr: u64 =
            if std::env::var("RVLLM_PER_TOKEN_Q_SCALE").ok().as_deref() == Some("0") {
                0
            } else {
                q_scale_scratch.device_ptr()
            };

        let q_scale_region = arena.region("gen_q_scale", 4, 4)?;
        let kv_scale_region = arena.region("gen_kv_scale", 4, 4)?;
        {
            let q_s: f32 = std::env::var("RVLLM_Q_SCALE")
                .ok().and_then(|v| v.parse().ok()).unwrap_or(DEFAULT_Q_SCALE);
            let kv_s: f32 = std::env::var("RVLLM_KV_SCALE")
                .ok().and_then(|v| v.parse().ok()).unwrap_or(DEFAULT_KV_SCALE);
            q_scale_region.copy_from_host(&q_s.to_le_bytes())?;
            kv_scale_region.copy_from_host(&kv_s.to_le_bytes())?;
        }

        let fa3_ws = arena.region("gen_fa3_ws", 128 * 1024 * 1024, 256)?;
        let cutlass_ws_bytes: usize = 16 * 1024 * 1024;
        let cutlass_ws = arena.region("gen_cutlass_ws", cutlass_ws_bytes, 256)?;

        let positions = arena.region("gen_pos", (max_tokens * 4) as usize, 16)?;
        let slot_mapping = arena.region("gen_slot", (max_tokens * 4) as usize, 16)?;
        let context_lens = arena.region("gen_ctx", 4, 16)?;
        // Sized for max_tokens i32 entries (not just the 2-entry prefix
        // sum): the unified decode-per-qi attention loop reuses this
        // region to stage a per-qi context-lens array `[1, 2, ..., N]`
        // and indexes it by `qi * 4`. With only 8 bytes (old FA2
        // prefill layout) writing beyond entry 1 corrupted adjacent
        // arena regions and degenerated generation quality.
        let cu_seqlens_q = arena.region("gen_cu_seqlens", (max_tokens * 4) as usize, 16)?;
        let block_tables = arena.region("gen_bt", (max_blocks_per_seq * 4) as usize, 16)?;
        {
            let bt: Vec<i32> = (0..max_blocks_per_seq as i32).collect();
            block_tables.copy_from_host(bytemuck_cast_i32(&bt))?;
        }

        let residual = arena.region("gen_residual", (max_tokens * hidden * 2) as usize, 16)?;
        let logits_f32 = arena.region("gen_logits_f32", (vocab * 4) as usize, 16)?;
        let token_ids_region = arena.region("gen_tok_ids", (max_tokens * 4) as usize, 16)?;
        let sampled = arena.region("gen_sampled", 4, 16)?;
        let residual_ptr = residual.device_ptr();
        let kernels = self.layer_kernels();

        use rvllm_loader::gemma4_arch::Gemma4LayerType;
        let max_layers = std::env::var("RVLLM_MAX_LAYERS")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(arch.num_hidden_layers);

        // === NVFP4 SHADOW DIAGNOSTIC (remove after collapse locator confirmed) ===
        // Build (or re-use) the f16 shadow KV region for the instrumented
        // layers. Pure diagnostic — the allocation mirrors the primary
        // allocator but forces F16 for every instrumented layer so the
        // cache is ground-truth. Same layer-blocks / nkvh / hd sizing as
        // the primary path; no scale region (f16 self-scaled).
        // Per-request gate: even with RVLLM_NVFP4_SHADOW_F16=1 in env,
        // skip shadow path unless the operator explicitly opted THIS
        // request in via `X-Rvllm-Shadow: 1`. Closes the
        // upstream-client-scaffold-burns-latch hole.
        let shadow_set: Option<Vec<u32>> = if shadow_requested {
            crate::gemma4_layer_exec::parse_shadow_layers()
        } else {
            None
        };
        // Compute per-layer shadow offsets (populated only for layers
        // in the shadow set; sentinel u64::MAX otherwise). Needed both
        // when we have to allocate the region now AND on subsequent
        // calls when it already exists — cheap to recompute every call.
        let shadow_layer_offsets: Vec<u64> = if let Some(ref lset) = shadow_set {
            let mut offs = vec![u64::MAX; arch.num_hidden_layers];
            let mut cursor: u64 = 0;
            for l in 0..arch.num_hidden_layers {
                if !lset.contains(&(l as u32)) { continue; }
                offs[l] = cursor;
                let is_global = arch.layer_types[l]
                    == rvllm_loader::gemma4_arch::Gemma4LayerType::GlobalAttention;
                let layer_blocks = if is_global { num_blocks_total } else { sliding_blocks };
                let nkvh = arch.num_kv_heads_for_layer(l) as u32;
                let hd = arch.head_dim_for_layer(l) as u32;
                // f16 = 2 bytes/elem; 2× for K and V.
                let layer_bytes =
                    2u64 * (layer_blocks as u64) * (block_size as u64)
                        * (nkvh as u64) * (hd as u64) * 2;
                cursor += layer_bytes;
            }
            offs
        } else {
            Vec::new()
        };
        let shadow_total_bytes: u64 = if let Some(ref lset) = shadow_set {
            let mut sum: u64 = 0;
            for l in 0..arch.num_hidden_layers {
                if !lset.contains(&(l as u32)) { continue; }
                let is_global = arch.layer_types[l]
                    == rvllm_loader::gemma4_arch::Gemma4LayerType::GlobalAttention;
                let layer_blocks = if is_global { num_blocks_total } else { sliding_blocks };
                let nkvh = arch.num_kv_heads_for_layer(l) as u32;
                let hd = arch.head_dim_for_layer(l) as u32;
                sum += 2u64 * (layer_blocks as u64) * (block_size as u64)
                    * (nkvh as u64) * (hd as u64) * 2;
            }
            sum
        } else { 0 };
        // Per-layer Q slot size: num_attention_heads * max_head_dim * 2 (f16).
        // Uniform across layers so indexing is a simple multiply. We dump
        // exactly ONE Q row per layer (decode step 0's Q). Slot is fixed
        // at single-token size regardless of how many tokens prefill
        // wrote — the per-layer slot is only ever populated on decode
        // step 0, when num_tokens == 1.
        let shadow_q_per_layer_bytes: u64 = if shadow_set.is_some() {
            2u64 * (arch.num_attention_heads as u64) * (arch.max_head_dim() as u64)
        } else {
            0
        };
        // Throwaway scratch — must hold the largest single
        // rope_f16kv_shadow Q-output. Decode = 1 token; batch prefill
        // = up to chunk_q tokens (bounded by num_blocks_total *
        // block_size). Size for the prefill upper bound so batch
        // prefill scratch construction can route shadow Q here
        // safely. ~1 GiB worst case on Gemma 4 31B (32768 * 32 *
        // 512 * 2). Trivial vs 128 GiB unified.
        let shadow_q_throwaway_bytes: u64 = if shadow_set.is_some() {
            (num_blocks_total as u64) * (block_size as u64) * shadow_q_per_layer_bytes
        } else {
            0
        };
        let shadow_q_total_bytes: u64 = if let Some(ref lset) = shadow_set {
            shadow_q_per_layer_bytes * (lset.len() as u64)
        } else {
            0
        };
        let (shadow_ptr, shadow_q_ptr, shadow_q_throwaway_ptr): (u64, u64, u64) =
            if let Some(ref lset) = shadow_set {
            let mut guard = self.nvfp4_shadow.lock().unwrap();
            if let Some(ref existing) = *guard {
                (existing.shadow_ptr, existing.shadow_q_ptr, existing.shadow_q_throwaway_ptr)
            } else {
                let bytes = shadow_total_bytes.max(16) as usize;
                let region = arena.region("nvfp4_shadow_kv", bytes, 256)?;
                cudarc::driver::sys::cuMemsetD8_v2(region.device_ptr(), 0, bytes);
                let ptr = region.device_ptr();
                std::mem::forget(region);
                // Per-layer Q snapshot region.
                let q_bytes = shadow_q_total_bytes.max(16) as usize;
                let q_region = arena.region("nvfp4_shadow_q", q_bytes, 256)?;
                cudarc::driver::sys::cuMemsetD8_v2(q_region.device_ptr(), 0, q_bytes);
                let q_ptr = q_region.device_ptr();
                std::mem::forget(q_region);
                // Q throwaway scratch: one slot. Shadow rope targets
                // this when we're not capturing, so q_normed stays
                // untouched and the subsequent primary nvfp4 rope
                // gets pristine pre-RoPE Q as input (exactly-one-RoPE
                // invariant restored).
                let throwaway_bytes = shadow_q_throwaway_bytes.max(16) as usize;
                let throwaway_region = arena.region(
                    "nvfp4_shadow_q_throwaway", throwaway_bytes, 256)?;
                cudarc::driver::sys::cuMemsetD8_v2(
                    throwaway_region.device_ptr(), 0, throwaway_bytes);
                let throwaway_ptr = throwaway_region.device_ptr();
                std::mem::forget(throwaway_region);
                *guard = Some(NvFp4ShadowAlloc {
                    shadow_ptr: ptr,
                    shadow_bytes: shadow_total_bytes,
                    layer_offsets: shadow_layer_offsets.clone(),
                    layer_indices: lset.clone(),
                    shadow_q_ptr: q_ptr,
                    shadow_q_total_bytes,
                    shadow_q_per_layer_bytes,
                    shadow_q_throwaway_ptr: throwaway_ptr,
                });
                eprintln!(
                    "[nvfp4-shadow] allocated {} MiB f16 shadow KV + {} KiB per-layer Q for {} layers: {:?}",
                    shadow_total_bytes / (1024 * 1024),
                    shadow_q_total_bytes / 1024,
                    lset.len(),
                    lset,
                );
                (ptr, q_ptr, throwaway_ptr)
            }
        } else { (0, 0, 0) };
        // === END NVFP4 SHADOW DIAGNOSTIC ===

        // Helper: run one token through all layers (decode path)
        let run_one_token = |tok_id: u32, step: usize| -> Result<()> {
            let tok_i32 = [tok_id as i32];
            token_ids_region.copy_from_host(bytemuck_cast_i32(&tok_i32))?;
            rvllm_fused::EmbeddingGatherLaunch { num_tokens: 1, hidden, vocab }
                .launch(fn_embed, residual_ptr, self.model.embedding.offset_bytes,
                    token_ids_region.device_ptr(), stream)?;

            let pos = [step as i32];
            let slot = [step as i32];
            let ctx = [step as i32 + 1];
            positions.copy_from_host(bytemuck_cast_i32(&pos))?;
            slot_mapping.copy_from_host(bytemuck_cast_i32(&slot))?;
            context_lens.copy_from_host(bytemuck_cast_i32(&ctx))?;

            for (layer_idx, layer) in self.model.layers.iter().enumerate() {
                if layer_idx >= max_layers { break; }
                let lt = arch.layer_types[layer_idx];
                let hd = arch.head_dim_for_layer(layer_idx) as u32;
                let nkvh = arch.num_kv_heads_for_layer(layer_idx) as u32;
                let q_dim = (arch.num_attention_heads as u32) * hd;
                let kv_dim = nkvh * hd;
                let layer_blocks = if lt == Gemma4LayerType::GlobalAttention { num_blocks_total } else { sliding_blocks };
                let layer_kv_elems = 2u64 * layer_blocks as u64 * block_size as u64 * nkvh as u64 * hd as u64;
                let layer_kv_base = kv_cache.device_ptr() + kv_layer_offsets[layer_idx];
                let layer_kv_scale_base =
                    kv_scale_cache.device_ptr() + kv_scale_layer_offsets[layer_idx];
                let layer_kv_scale_slots_half =
                    (layer_blocks as u64) * (block_size as u64) * (nkvh as u64);
                // Per-layer dtype: hybrid mode swaps global layers to FP8,
                // sliding layers stay on env default.
                let kv_dtype = crate::gemma4_layer_exec::KvDtype::for_layer_or_env(lt, false);
                let (k_cache_scale, v_cache_scale) = if kv_dtype
                    == crate::gemma4_layer_exec::KvDtype::Nvfp4
                {
                    (layer_kv_scale_base, layer_kv_scale_base + layer_kv_elems / 32)
                } else {
                    (0u64, 0u64)
                };

                let dims = crate::gemma4_layer_exec::Gemma4LayerDims {
                    num_tokens: 1, hidden,
                    num_heads: arch.num_attention_heads as u32, num_kv_heads: nkvh, head_dim: hd,
                    rotary_dim: arch.rotary_dim_for_layer(layer_idx) as u32,
                    intermediate: inter, block_size,
                    max_blocks_per_seq: layer_blocks, num_blocks_total: layer_blocks,
                    attn_scale: 1.0, rms_eps: arch.rms_norm_eps,
                    layer_type: lt, sliding_window: arch.sliding_window_size as u32,
                    f16_kv: kv_dtype.is_f16(),
                    kv_dtype,
                    // Decode step knows its own ctx CPU-side — `ctx = [step + 1]`
                    // was computed at line ~1843. Pass it so the split-KV
                    // dispatch gates on the current ctx length instead of
                    // the bucket max (avoids dispatching split on short
                    // early-generation turns where one-CTA decode wins).
                    current_max_context_len: Some((step as u32) + 1),
                };
                let w = crate::gemma4_layer_exec::Gemma4LayerWeightPtrs {
                    attn_norm_gamma: layer.input_layernorm.offset_bytes,
                    post_attn_norm_gamma: layer.post_attention_layernorm.offset_bytes,
                    pre_ff_norm_gamma: layer.pre_feedforward_layernorm.offset_bytes,
                    post_ff_norm_gamma: layer.post_feedforward_layernorm.offset_bytes,
                    q_norm_gamma: layer.q_norm.offset_bytes,
                    k_norm_gamma: layer.k_norm.offset_bytes,
                    qkv_fp8: layer.qkv.offset_bytes, qkv_scale: layer.qkv.scale_ptr,
                    o_fp8: layer.o_proj.offset_bytes, o_scale: layer.o_proj.scale_ptr,
                    gate_up_fp8: layer.gate_up.offset_bytes, gate_up_scale: layer.gate_up.scale_ptr,
                    down_fp8: layer.down_proj.offset_bytes, down_scale: layer.down_proj.scale_ptr,
                    layer_scalar_ptr: layer.layer_scalar.offset_bytes,
                    qkv_f16: layer.qkv_f16.as_ref().map_or(0, |w| w.offset_bytes),
                    o_f16: layer.o_proj_f16.as_ref().map_or(0, |w| w.offset_bytes),
                    gate_up_f16: layer.gate_up_f16.as_ref().map_or(0, |w| w.offset_bytes),
                    down_f16: layer.down_proj_f16.as_ref().map_or(0, |w| w.offset_bytes),
                    qkv_chscale: layer.qkv.channelscale_ptr.unwrap_or(0),
                    o_chscale: layer.o_proj.channelscale_ptr.unwrap_or(0),
                    gate_up_chscale: layer.gate_up.channelscale_ptr.unwrap_or(0),
                    down_chscale: layer.down_proj.channelscale_ptr.unwrap_or(0),
                    qkv_blockscale: layer.qkv.blockscale_ptr.unwrap_or(0),
                    o_blockscale: layer.o_proj.blockscale_ptr.unwrap_or(0),
                    gate_up_blockscale: layer.gate_up.blockscale_ptr.unwrap_or(0),
                    down_blockscale: layer.down_proj.blockscale_ptr.unwrap_or(0),
                };
                let k_out = q_base + (q_dim as u64) * 2;
                let v_out = k_out + (kv_dim as u64) * 2;
                let (cos, sin) = match lt {
                    Gemma4LayerType::SlidingAttention => (self.model.rope_cos_sliding.offset_bytes, self.model.rope_sin_sliding.offset_bytes),
                    Gemma4LayerType::GlobalAttention => (self.model.rope_cos_global.offset_bytes, self.model.rope_sin_global.offset_bytes),
                };
                let bytes_per_half_kv = match kv_dtype {
                    crate::gemma4_layer_exec::KvDtype::F16 => layer_kv_elems,
                    crate::gemma4_layer_exec::KvDtype::Fp8 => layer_kv_elems / 2,
                    crate::gemma4_layer_exec::KvDtype::Nvfp4 => layer_kv_elems / 4,
                };
                // === NVFP4 SHADOW DIAGNOSTIC (remove after collapse locator confirmed) ===
                // Populate shadow pointers only for instrumented NVFP4 layers.
                let is_shadow_layer = shadow_ptr != 0
                    && kv_dtype == crate::gemma4_layer_exec::KvDtype::Nvfp4
                    && layer_idx < shadow_layer_offsets.len()
                    && shadow_layer_offsets[layer_idx] != u64::MAX;
                let (shadow_k, shadow_v) = if is_shadow_layer {
                    let base = shadow_ptr + shadow_layer_offsets[layer_idx];
                    // f16 K/V: each is layer_kv_elems/2 elements × 2 bytes = layer_kv_elems bytes.
                    (base, base + layer_kv_elems)
                } else {
                    (0u64, 0u64)
                };
                // Shadow Q target. Two roles:
                //   (a) When the layer is instrumented AND this is the
                //       first decode step (step 0 after prompt), point
                //       at the per-layer Q slot so the Python analyzer
                //       gets post-RoPE f16 Q for logit_err / topk /
                //       out_err.
                //   (b) When the layer is instrumented at any OTHER
                //       step (all prefill steps + decode step >0),
                //       point at the shared throwaway so shadow rope
                //       has a valid q_out target WITHOUT clobbering
                //       `scratch.q_normed`. This is load-bearing: if
                //       q_normed is clobbered, the subsequent primary
                //       rope_nvfp4kv rotates it a second time and
                //       every forward pass is wrong.
                //   (c) When the layer is NOT instrumented, 0 — shadow
                //       rope doesn't run at all.
                let shadow_q = if is_shadow_layer {
                    if step == prompt_ids.len() && shadow_q_ptr != 0 {
                        let pos_in_set = shadow_set
                            .as_ref()
                            .and_then(|s| s.iter().position(|&l| l as usize == layer_idx));
                        match pos_in_set {
                            Some(i) => shadow_q_ptr + (i as u64) * shadow_q_per_layer_bytes,
                            None => shadow_q_throwaway_ptr,
                        }
                    } else {
                        shadow_q_throwaway_ptr
                    }
                } else {
                    0
                };
                // === END NVFP4 SHADOW DIAGNOSTIC ===
                let scratch = crate::gemma4_layer_exec::Gemma4LayerScratch {
                    hidden_fp8: hidden_fp8.device_ptr(), hidden_scale: hidden_scale.device_ptr(),
                    q_out: q_base, k_out, v_out,
                    q_normed: q_normed.device_ptr(), k_normed: k_normed.device_ptr(),
                    v_normed: v_normed.device_ptr(),
                    q_fp8: q_fp8.device_ptr(),
                    k_cache: layer_kv_base,
                    v_cache: layer_kv_base + bytes_per_half_kv,
                    k_cache_scale,
                    v_cache_scale,
                    q_scale_ptr: q_scale_region.device_ptr(), kv_scale_ptr: kv_scale_region.device_ptr(),
                    k_scale_cache: layer_kv_scale_base,
                    v_scale_cache: layer_kv_scale_base + layer_kv_scale_slots_half * 4,
                    q_scale_cache: q_scale_cache_ptr,
                    attn_out: attn_out.device_ptr(), attn_out_fp8: attn_out_fp8.device_ptr(),
                    attn_out_scale: attn_out_scale.device_ptr(), delta_f16: delta_f16.device_ptr(),
                    gate_up_out: gate_up_out.device_ptr(), gate_up_fp8: gate_up_fp8.device_ptr(),
                    gate_up_scale: gate_up_scale.device_ptr(),
                    mlp_out_fp8: mlp_out_fp8.device_ptr(), mlp_out_scale: mlp_out_scale.device_ptr(),
                    gemm_f32_tmp: gemm_f32_tmp.device_ptr(),
                    cutlass_workspace: cutlass_ws.device_ptr(), cutlass_workspace_bytes: cutlass_ws_bytes,
                    fa3_workspace: fa3_ws.device_ptr(),
                    shadow_k_cache: shadow_k,
                    shadow_v_cache: shadow_v,
                    shadow_q_cache: shadow_q,
                };
                let meta = crate::gemma4_layer_exec::Gemma4MetadataPtrs {
                    positions: positions.device_ptr(), slot_mapping: slot_mapping.device_ptr(),
                    cos, sin,
                    block_tables: block_tables.device_ptr(), context_lens: context_lens.device_ptr(),
                };
                crate::gemma4_layer_exec::gemma4_forward(
                    dims, &kernels, &w, &scratch, &meta,
                    &self.cublaslt, &self.cutlass, &self.sliding_attention, &self.global_attention,
                    residual_ptr, stream,
                )?;
            }
            Ok(())
        };

        let t0 = std::time::Instant::now();

        // Phase 1: prompt through per-token decode (default, correct-by-design).
        //
        // On sm_121 with FP8 block-scale weights (Gemma 4 fp8-block), the
        // per-token path uses `fp8_gemv_blockwise_wpr_native_f16in_kernel`
        // which preserves the per-channel weight block-scale. The batch
        // (num_tokens>1) GEMM path goes through
        // `fp8_gemm_channelscale_or_fallback`, which on Blackwell consumer
        // collapses to a scalar weight scale because cuBLASLt's FP8
        // channelscale heuristic `LaunchFailed`s at this arch. That is a
        // genuine numerical difference, not a hidden bug — the two paths
        // are not bit-identical by design at num_tokens<CUTLASS_M_MIN(=128).
        //
        // Path forward for genuine batch-prefill speedup:
        //   * num_tokens >=128 : CUTLASS SM120 blockwise FP8 GEMM (landed;
        //     opt-in via RVLLM_FP8_GEMM_CUTLASS_SM120 + M>=128 gate in
        //     gemma4_layer_exec).  This preserves the per-channel scale
        //     via SFA/SFB prep.
        //   * num_tokens < 128 : per-token loop is optimal (fp8_gemv is
        //     M=1-only; running it T times reads weights T times but each
        //     call is already bandwidth-bound; cost parity with any batched
        //     solution at small M).
        //
        // So we keep the per-token loop as the default for ALL prompt
        // lengths today. RVLLM_BATCH_PREFILL=1 flips to the unified
        // batch path (diagnostic: verifies CUTLASS >=128 correctness,
        // or measures the collapsed-scalar quality floor at <128).
        let skip_decode = std::env::var_os("RVLLM_DIAG_SKIP_DECODE").is_some();
        let use_batch_prefill = std::env::var_os("RVLLM_BATCH_PREFILL").is_some();
        if !skip_decode && !use_batch_prefill {
            // Prefix-cache fast path: if common_prefix_len > 0, the
            // persistent KV region already holds valid entries for
            // slots [0..common_prefix_len). Skip those tokens; the
            // per-token loop picks up at the first new token, attention
            // reads the cached KV for context. Batch-prefill path
            // below doesn't use this shortcut yet (unified kernel
            // would need partial-query support wired through).
            let start = common_prefix_len as usize;
            for (i, &tok) in prompt_ids.iter().enumerate().skip(start) {
                run_one_token(tok, i)?;
            }
        }

        // Optional prefill-vs-decode residual compare (RVLLM_DIAG_COMPARE=1).
        // Captures the last-token residual produced by per-token decode
        // (correct reference), resets KV, re-runs the prompt via batch
        // prefill, captures the same row, and prints the diff. Combine
        // with `RVLLM_MAX_LAYERS=N` to bisect where the two paths
        // diverge. Only fires when prompt_len > 1 (decode==prefill
        // trivially at prompt_len=1).
        let diag_compare =
            std::env::var_os("RVLLM_DIAG_COMPARE").is_some() && !skip_decode;
        let mut decode_ref_last: Vec<u16> = Vec::new();
        let mut decode_ref_first: Vec<u16> = Vec::new();
        if diag_compare && prompt_len > 1 {
            // Already captured: residual_ptr holds LAST token's residual
            // after all prompt tokens were processed sequentially.
            self.stream.fence()?;
            decode_ref_last = vec![0u16; hidden as usize];
            cudarc::driver::sys::cuMemcpyDtoH_v2(
                decode_ref_last.as_mut_ptr() as *mut _,
                residual_ptr,
                (hidden * 2) as _,
            );

            // For FIRST token reference, re-run just token 0 through
            // a fresh KV cache — the residual after that step is what
            // prefill's row 0 should match (no prior context at
            // position 0 in either path).
            cudarc::driver::sys::cuMemsetD8_v2(kv_cache.device_ptr(), 0, kv_total_bytes as usize);
            self.stream.fence()?;
            run_one_token(prompt_ids[0], 0)?;
            self.stream.fence()?;
            decode_ref_first = vec![0u16; hidden as usize];
            cudarc::driver::sys::cuMemcpyDtoH_v2(
                decode_ref_first.as_mut_ptr() as *mut _,
                residual_ptr,
                (hidden * 2) as _,
            );

            // Reset KV again before prefill re-runs the whole prompt.
            cudarc::driver::sys::cuMemsetD8_v2(kv_cache.device_ptr(), 0, kv_total_bytes as usize);
            self.stream.fence()?;
        }

        // Dead prefill block retained behind `if false`; flip to the
        // diag gate to re-run the prompt through batch prefill
        // (correctness is still broken — this path is instrumentation,
        // not production). `skip_decode` takes the prefill path
        // standalone so the existing DBG probes fire on prefill's
        // layer 0 / layer 1 for direct comparison against a normal
        // decode-only run.
        if (diag_compare && prompt_len > 1) || skip_decode || (use_batch_prefill && prompt_len > 1) {
            // Prefix-cache aware batch prefill, OPTIONALLY chunked.
            //
            // When `use_prefix_cache` reports a common prefix of length
            // L, we skip prefill for slots [0..L). The remaining
            // `prompt_len - L` new tokens get processed in chunks of
            // `RVLLM_PREFILL_CHUNK_SIZE` (0 = single chunk / all new
            // tokens at once, matching the pre-chunked path).
            //
            // Each chunk is a "partial query with full-prefix KV
            // history" — the unified kernel handles this natively via
            // `context_lens = chunk_end, cu_seqlens_q = [0, chunk_q],
            // positions = [chunk_start..chunk_end)`. After the chunk
            // runs, its KV is in the persistent cache for the next
            // chunk's attention reads.
            //
            // Diag mode forces L=0 + single chunk so the row-0 /
            // row-(N-1) rel_err comparison stays meaningful.
            let prefix_skip = if diag_compare { 0 } else { common_prefix_len };
            let total_new_q = prompt_len - prefix_skip;
            let chunk_env: u32 = std::env::var("RVLLM_PREFILL_CHUNK_SIZE")
                .ok().and_then(|s| s.parse().ok()).unwrap_or(0);
            let chunk_size_max: u32 = if diag_compare || chunk_env == 0 {
                total_new_q
            } else {
                chunk_env
            };

            if chunk_size_max < total_new_q {
                eprintln!(
                    "[prefill-chunk] total_new_q={} chunk_size_max={} num_chunks={}",
                    total_new_q, chunk_size_max,
                    (total_new_q + chunk_size_max - 1) / chunk_size_max
                );
            }

            // Outer chunk loop. `new_q` at end-of-block holds the LAST
            // chunk's Q length so the downstream last-token-residual
            // extract picks the right row.
            let mut chunk_start_abs: u32 = prefix_skip;
            let mut new_q: u32 = 0;
            let mut chunk_idx: u32 = 0;
            while chunk_start_abs < prompt_len {
                let chunk_end_abs = std::cmp::min(
                    chunk_start_abs + chunk_size_max,
                    prompt_len,
                );
                let chunk_q = chunk_end_abs - chunk_start_abs;
                new_q = chunk_q;
                let tok_ids: Vec<i32> = prompt_ids[
                    chunk_start_abs as usize .. chunk_end_abs as usize
                ].iter().map(|&t| t as i32).collect();
                token_ids_region.copy_from_host(bytemuck_cast_i32(&tok_ids))?;
                if chunk_idx == 0 {
                    self.stream.fence()?;
                    let mut readback = vec![0i32; chunk_q as usize];
                    cudarc::driver::sys::cuMemcpyDtoH_v2(
                        readback.as_mut_ptr() as *mut _,
                        token_ids_region.device_ptr(),
                        (chunk_q * 4) as _,
                    );
                    eprintln!(
                        "[DIAG] batch-prefill prefix_skip={} chunk0_q={} readback[..min(8)]={:?}",
                        prefix_skip, chunk_q, &readback[..readback.len().min(8)]
                    );
                }
                rvllm_fused::EmbeddingGatherLaunch { num_tokens: chunk_q, hidden, vocab }
                    .launch(fn_embed, residual_ptr, self.model.embedding.offset_bytes,
                        token_ids_region.device_ptr(), stream)?;
                if chunk_idx == 0 {
                    self.stream.fence()?;
                    let mut r0 = vec![0u16; 4];
                    let mut r_n_minus_1 = vec![0u16; 4];
                    cudarc::driver::sys::cuMemcpyDtoH_v2(r0.as_mut_ptr() as *mut _, residual_ptr, 8);
                    cudarc::driver::sys::cuMemcpyDtoH_v2(
                        r_n_minus_1.as_mut_ptr() as *mut _,
                        residual_ptr + ((chunk_q - 1) as u64 * hidden as u64 * 2),
                        8,
                    );
                    eprintln!(
                        "[DIAG] post-gather row0[..4]={:?} rowN-1[..4]={:?}",
                        r0.iter().map(|&x| crate::bring_up::f16_to_f32(x)).collect::<Vec<_>>(),
                        r_n_minus_1.iter().map(|&x| crate::bring_up::f16_to_f32(x)).collect::<Vec<_>>(),
                    );
                }

                // positions / slot_mapping for THIS chunk; cu_seq + ctx
                // frame the partial query within the full sequence.
                let pos: Vec<i32> = (chunk_start_abs as i32 .. chunk_end_abs as i32).collect();
                let slot: Vec<i32> = (chunk_start_abs as i32 .. chunk_end_abs as i32).collect();
                let ctx = [chunk_end_abs as i32];
                let cu_seq = [0i32, chunk_q as i32];
                positions.copy_from_host(bytemuck_cast_i32(&pos))?;
                slot_mapping.copy_from_host(bytemuck_cast_i32(&slot))?;
                context_lens.copy_from_host(bytemuck_cast_i32(&ctx))?;
                cu_seqlens_q.copy_from_host(bytemuck_cast_i32(&cu_seq))?;

                let phase = crate::gemma4_layer_exec::Gemma4Phase::Prefill {
                    cu_seqlens_q: cu_seqlens_q.device_ptr(),
                    max_seqlen_q: chunk_q,
                    num_seqs: 1,
                };

                for (layer_idx, layer) in self.model.layers.iter().enumerate() {
                if layer_idx >= max_layers { break; }
                let lt = arch.layer_types[layer_idx];
                let hd = arch.head_dim_for_layer(layer_idx) as u32;
                let nkvh = arch.num_kv_heads_for_layer(layer_idx) as u32;
                let q_dim = (arch.num_attention_heads as u32) * hd;
                let kv_dim = nkvh * hd;
                let layer_blocks = if lt == Gemma4LayerType::GlobalAttention { num_blocks_total } else { sliding_blocks };
                let layer_kv_elems = 2u64 * layer_blocks as u64 * block_size as u64 * nkvh as u64 * hd as u64;
                let layer_kv_base = kv_cache.device_ptr() + kv_layer_offsets[layer_idx];
                let layer_kv_scale_base =
                    kv_scale_cache.device_ptr() + kv_scale_layer_offsets[layer_idx];
                let layer_kv_scale_slots_half =
                    (layer_blocks as u64) * (block_size as u64) * (nkvh as u64);
                // Per-layer dtype: hybrid swaps global to FP8 (sliding stays env default).
                let kv_dtype = crate::gemma4_layer_exec::KvDtype::for_layer_or_env(lt, false);
                // Prefill uses FP8 KV when the ambient dtype is F16
                // (no F16 prefill kernel exists); NVFP4 prefill stays on NVFP4.
                let prefill_kv_dtype = if kv_dtype == crate::gemma4_layer_exec::KvDtype::Nvfp4 {
                    crate::gemma4_layer_exec::KvDtype::Nvfp4
                } else {
                    crate::gemma4_layer_exec::KvDtype::Fp8
                };
                let (k_cache_scale, v_cache_scale) = if prefill_kv_dtype
                    == crate::gemma4_layer_exec::KvDtype::Nvfp4
                {
                    (layer_kv_scale_base, layer_kv_scale_base + layer_kv_elems / 32)
                } else {
                    (0u64, 0u64)
                };

                let dims = crate::gemma4_layer_exec::Gemma4LayerDims {
                    num_tokens: new_q, hidden,
                    num_heads: arch.num_attention_heads as u32, num_kv_heads: nkvh, head_dim: hd,
                    rotary_dim: arch.rotary_dim_for_layer(layer_idx) as u32,
                    intermediate: inter, block_size,
                    max_blocks_per_seq: layer_blocks, num_blocks_total: layer_blocks,
                    attn_scale: 1.0, rms_eps: arch.rms_norm_eps,
                    layer_type: lt, sliding_window: arch.sliding_window_size as u32,
                    f16_kv: false, // prefill uses FP8 KV (no F16 prefill kernel)
                    kv_dtype: prefill_kv_dtype,
                    current_max_context_len: None,  // prefill path — not used by split-KV decode
                };
                let w = crate::gemma4_layer_exec::Gemma4LayerWeightPtrs {
                    attn_norm_gamma: layer.input_layernorm.offset_bytes,
                    post_attn_norm_gamma: layer.post_attention_layernorm.offset_bytes,
                    pre_ff_norm_gamma: layer.pre_feedforward_layernorm.offset_bytes,
                    post_ff_norm_gamma: layer.post_feedforward_layernorm.offset_bytes,
                    q_norm_gamma: layer.q_norm.offset_bytes,
                    k_norm_gamma: layer.k_norm.offset_bytes,
                    qkv_fp8: layer.qkv.offset_bytes, qkv_scale: layer.qkv.scale_ptr,
                    o_fp8: layer.o_proj.offset_bytes, o_scale: layer.o_proj.scale_ptr,
                    gate_up_fp8: layer.gate_up.offset_bytes, gate_up_scale: layer.gate_up.scale_ptr,
                    down_fp8: layer.down_proj.offset_bytes, down_scale: layer.down_proj.scale_ptr,
                    layer_scalar_ptr: layer.layer_scalar.offset_bytes,
                    qkv_f16: layer.qkv_f16.as_ref().map_or(0, |w| w.offset_bytes),
                    o_f16: layer.o_proj_f16.as_ref().map_or(0, |w| w.offset_bytes),
                    gate_up_f16: layer.gate_up_f16.as_ref().map_or(0, |w| w.offset_bytes),
                    down_f16: layer.down_proj_f16.as_ref().map_or(0, |w| w.offset_bytes),
                    qkv_chscale: layer.qkv.channelscale_ptr.unwrap_or(0),
                    o_chscale: layer.o_proj.channelscale_ptr.unwrap_or(0),
                    gate_up_chscale: layer.gate_up.channelscale_ptr.unwrap_or(0),
                    down_chscale: layer.down_proj.channelscale_ptr.unwrap_or(0),
                    qkv_blockscale: layer.qkv.blockscale_ptr.unwrap_or(0),
                    o_blockscale: layer.o_proj.blockscale_ptr.unwrap_or(0),
                    gate_up_blockscale: layer.gate_up.blockscale_ptr.unwrap_or(0),
                    down_blockscale: layer.down_proj.blockscale_ptr.unwrap_or(0),
                };
                // Row-major [num_tokens, q_dim+2*kv_dim]: k_out / v_out
                // point at row 0's K / V sub-slice. The rmsnorm kernel
                // applies `src_row_stride` to reach later tokens — the
                // old `num_tokens * q_dim * 2` formula assumed a
                // columnar "all Q then all K then all V" layout that
                // the cuBLASLt QKV GEMM does NOT produce.
                let k_out = q_base + (q_dim as u64) * 2;
                let v_out = k_out + (kv_dim as u64) * 2;
                let (cos, sin) = match lt {
                    Gemma4LayerType::SlidingAttention => (self.model.rope_cos_sliding.offset_bytes, self.model.rope_sin_sliding.offset_bytes),
                    Gemma4LayerType::GlobalAttention => (self.model.rope_cos_global.offset_bytes, self.model.rope_sin_global.offset_bytes),
                };
                let bytes_per_half_kv = match prefill_kv_dtype {
                    crate::gemma4_layer_exec::KvDtype::F16 => layer_kv_elems,
                    crate::gemma4_layer_exec::KvDtype::Fp8 => layer_kv_elems / 2,
                    crate::gemma4_layer_exec::KvDtype::Nvfp4 => layer_kv_elems / 4,
                };
                // === NVFP4 SHADOW DIAGNOSTIC (remove after collapse locator confirmed) ===
                // Mirror the decode-path scratch population so batch
                // prefill ALSO writes f16 shadow K/V for the prompt
                // tokens. Without this, prefill silently bypasses the
                // shadow hook and the dump on decode step 0 only sees
                // the single new-token write — useless for analysis.
                // Q always routes to the shared throwaway during
                // prefill (we only capture per-layer Q on decode
                // step 0, which goes through run_one_token, not here).
                let prefill_is_shadow_layer = shadow_ptr != 0
                    && prefill_kv_dtype == crate::gemma4_layer_exec::KvDtype::Nvfp4
                    && layer_idx < shadow_layer_offsets.len()
                    && shadow_layer_offsets[layer_idx] != u64::MAX;
                let (prefill_shadow_k, prefill_shadow_v) = if prefill_is_shadow_layer {
                    let base = shadow_ptr + shadow_layer_offsets[layer_idx];
                    (base, base + layer_kv_elems)
                } else {
                    (0u64, 0u64)
                };
                let prefill_shadow_q = if prefill_is_shadow_layer {
                    shadow_q_throwaway_ptr
                } else {
                    0
                };
                // === END NVFP4 SHADOW DIAGNOSTIC ===
                let scratch = crate::gemma4_layer_exec::Gemma4LayerScratch {
                    hidden_fp8: hidden_fp8.device_ptr(), hidden_scale: hidden_scale.device_ptr(),
                    q_out: q_base, k_out, v_out,
                    q_normed: q_normed.device_ptr(), k_normed: k_normed.device_ptr(),
                    v_normed: v_normed.device_ptr(),
                    q_fp8: q_fp8.device_ptr(),
                    k_cache: layer_kv_base,
                    v_cache: layer_kv_base + bytes_per_half_kv,
                    k_cache_scale,
                    v_cache_scale,
                    q_scale_ptr: q_scale_region.device_ptr(), kv_scale_ptr: kv_scale_region.device_ptr(),
                    k_scale_cache: layer_kv_scale_base,
                    v_scale_cache: layer_kv_scale_base + layer_kv_scale_slots_half * 4,
                    q_scale_cache: q_scale_cache_ptr,
                    attn_out: attn_out.device_ptr(), attn_out_fp8: attn_out_fp8.device_ptr(),
                    attn_out_scale: attn_out_scale.device_ptr(), delta_f16: delta_f16.device_ptr(),
                    gate_up_out: gate_up_out.device_ptr(), gate_up_fp8: gate_up_fp8.device_ptr(),
                    gate_up_scale: gate_up_scale.device_ptr(),
                    mlp_out_fp8: mlp_out_fp8.device_ptr(), mlp_out_scale: mlp_out_scale.device_ptr(),
                    gemm_f32_tmp: gemm_f32_tmp.device_ptr(),
                    cutlass_workspace: cutlass_ws.device_ptr(), cutlass_workspace_bytes: cutlass_ws_bytes,
                    fa3_workspace: fa3_ws.device_ptr(),
                    shadow_k_cache: prefill_shadow_k,
                    shadow_v_cache: prefill_shadow_v,
                    shadow_q_cache: prefill_shadow_q,
                };
                let meta = crate::gemma4_layer_exec::Gemma4MetadataPtrs {
                    positions: positions.device_ptr(), slot_mapping: slot_mapping.device_ptr(),
                    cos, sin,
                    block_tables: block_tables.device_ptr(), context_lens: context_lens.device_ptr(),
                };
                crate::gemma4_layer_exec::gemma4_forward_phase(
                    dims, &kernels, &w, &scratch, &meta,
                    &self.cublaslt, &self.cutlass, &self.sliding_attention, &self.global_attention,
                    residual_ptr, stream, phase,
                )?;
            }
                chunk_start_abs = chunk_end_abs;
                chunk_idx += 1;
            } // end chunk loop

            // Diag capture BEFORE the extract-last memcpy: row 0 is
            // the first-token output (no prior context); row
            // new_q-1 is the last-token output the LM head consumes.
            // With chunking, row 0 = first row of the LAST chunk (a
            // mid-prompt position), so the row-0 diag semantics only
            // match the decode reference when diag_compare forces a
            // single chunk.
            self.stream.fence()?;
            let mut prefill_first = vec![0u16; hidden as usize];
            cudarc::driver::sys::cuMemcpyDtoH_v2(
                prefill_first.as_mut_ptr() as *mut _,
                residual_ptr,
                (hidden * 2) as _,
            );
            // The residual buffer holds `new_q` rows after the layer
            // loop (the prefix-cached slots are not re-prefilled).
            // Row `new_q - 1` is the last prompt token regardless of
            // how many tokens were cached.
            let mut prefill_last = vec![0u16; hidden as usize];
            let last_off_diag = (new_q - 1) as u64 * hidden as u64 * 2;
            cudarc::driver::sys::cuMemcpyDtoH_v2(
                prefill_last.as_mut_ptr() as *mut _,
                residual_ptr + last_off_diag,
                (hidden * 2) as _,
            );

            // Extract last token's residual for decode
            if new_q > 1 {
                let last_offset = (new_q - 1) as u64 * hidden as u64 * 2;
                cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                    residual_ptr, residual_ptr + last_offset, (hidden * 2) as usize, stream as _,
                );
            }

            if !diag_compare {
                if skip_decode { return Ok(Vec::new()); }
                // use_batch_prefill: fall through to LM head.
            } else {
                let stats = |label: &str, reference: &[u16], probe: &[u16]| {
                    let mut max_abs = 0f32;
                    let mut sum_sq_diff = 0f64;
                    let mut sum_sq_ref = 0f64;
                    let mut first_diffs: Vec<(f32, f32)> = Vec::new();
                    for i in 0..hidden as usize {
                        let d = crate::bring_up::f16_to_f32(reference[i]);
                        let p = crate::bring_up::f16_to_f32(probe[i]);
                        let diff = (d - p).abs();
                        if diff > max_abs { max_abs = diff; }
                        sum_sq_diff += (diff as f64) * (diff as f64);
                        sum_sq_ref += (d as f64) * (d as f64);
                        if first_diffs.len() < 4 { first_diffs.push((d, p)); }
                    }
                    let rel_err = (sum_sq_diff / sum_sq_ref.max(1e-18)).sqrt();
                    eprintln!(
                        "[DIAG {label}] max_abs={max_abs:.4} rel_err={rel_err:.4e} \
                         first4_ref_probe={first_diffs:?}",
                    );
                };
                eprintln!(
                    "[DIAG] max_layers={} prompt_len={} hidden={}",
                    max_layers, prompt_len, hidden,
                );
                stats("row=0 (first token)", &decode_ref_first, &prefill_first);
                stats("row=N-1 (last token)", &decode_ref_last, &prefill_last);
            }
        }

        // LM head on last prompt token
        rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
            num_tokens: 1, hidden, eps: arch.rms_norm_eps,
        }.launch(kernels.fused_rmsnorm, residual_ptr, self.model.final_norm.offset_bytes, stream)?;
        self.cublaslt.f16_gemm_f32(residual_ptr, self.model.lm_head_f16.offset_bytes,
            logits_f32.device_ptr(), 1, vocab as i32, hidden as i32, stream)?;
        rvllm_fused::ArgmaxLaunch { num_tokens: 1, vocab }
            .launch(fn_argmax, logits_f32.device_ptr(), sampled.device_ptr(), stream)?;

        self.stream.fence()?;
        let mut host_tok = [0i32; 1];
        cudarc::driver::sys::cuMemcpyDtoH_v2(host_tok.as_mut_ptr() as *mut _, sampled.device_ptr(), 4);
        let prefill_ms = t0.elapsed().as_secs_f64() * 1000.0;
        eprintln!("[prefill] {} tokens in {:.1}ms (TTFT={:.1}ms)",
            prompt_ids.len(), prefill_ms, prefill_ms);

        let mut output_ids: Vec<u32> = Vec::with_capacity(max_new);
        output_ids.push(host_tok[0] as u32);
        if eos_ids.contains(&(host_tok[0] as u32)) {
            return Ok(output_ids);
        }

        // Phase 2: Decode new tokens
        for decode_step in 0..max_new - 1 {
            let tok_id = *output_ids.last().unwrap();
            run_one_token(tok_id, prompt_ids.len() + decode_step)?;

            // === NVFP4 SHADOW DIAGNOSTIC (remove after collapse locator confirmed) ===
            // First-token dump: runs exactly once on decode_step == 0
            // when the shadow region is live. After this the latch is
            // set and every subsequent decode step is a no-op.
            if decode_step == 0
                && shadow_ptr != 0
                && !self.nvfp4_shadow_dumped.swap(true, std::sync::atomic::Ordering::Relaxed)
            {
                self.stream.fence()?;
                let dump_dir = std::env::var("RVLLM_NVFP4_SHADOW_DUMP_DIR")
                    .unwrap_or_else(|_| "/tmp/nvfp4_shadow".to_string());
                let _ = std::fs::create_dir_all(&dump_dir);
                let _ctx_now = (prompt_ids.len() + 1) as u32;
                let lset = shadow_set.as_ref().unwrap();
                let first_tok = output_ids[0];
                let bt_entries = max_blocks_per_seq as usize;
                // Host staging for block_tables / context_lens / slot_mapping.
                let mut bt_host = vec![0i32; bt_entries];
                cudarc::driver::sys::cuMemcpyDtoH_v2(
                    bt_host.as_mut_ptr() as *mut _,
                    block_tables.device_ptr(),
                    (bt_entries * 4) as usize,
                );
                let mut ctx_host = [0i32; 1];
                cudarc::driver::sys::cuMemcpyDtoH_v2(
                    ctx_host.as_mut_ptr() as *mut _,
                    context_lens.device_ptr(),
                    4,
                );
                let mut slot_host = [0i32; 1];
                cudarc::driver::sys::cuMemcpyDtoH_v2(
                    slot_host.as_mut_ptr() as *mut _,
                    slot_mapping.device_ptr(),
                    4,
                );
                // Build per-layer metadata + dump bin files.
                let mut layer_meta_json = String::new();
                for &l in lset.iter() {
                    let l = l as usize;
                    if l >= arch.num_hidden_layers { continue; }
                    if l >= shadow_layer_offsets.len() { continue; }
                    if shadow_layer_offsets[l] == u64::MAX { continue; }
                    let lt = arch.layer_types[l];
                    let is_global = lt == rvllm_loader::gemma4_arch::Gemma4LayerType::GlobalAttention;
                    let layer_blocks = if is_global { num_blocks_total } else { sliding_blocks };
                    let nkvh = arch.num_kv_heads_for_layer(l) as u32;
                    let hd = arch.head_dim_for_layer(l) as u32;
                    let layer_elems = 2u64 * (layer_blocks as u64) * (block_size as u64)
                        * (nkvh as u64) * (hd as u64);
                    // Shadow region: f16, layer_elems bytes for K then layer_elems bytes for V.
                    let shadow_base = shadow_ptr + shadow_layer_offsets[l];
                    let shadow_half_bytes = layer_elems; // f16 half-size per K or V
                    let mut k_shadow_host = vec![0u8; shadow_half_bytes as usize];
                    let mut v_shadow_host = vec![0u8; shadow_half_bytes as usize];
                    cudarc::driver::sys::cuMemcpyDtoH_v2(
                        k_shadow_host.as_mut_ptr() as *mut _,
                        shadow_base,
                        shadow_half_bytes as usize,
                    );
                    cudarc::driver::sys::cuMemcpyDtoH_v2(
                        v_shadow_host.as_mut_ptr() as *mut _,
                        shadow_base + shadow_half_bytes,
                        shadow_half_bytes as usize,
                    );
                    let _ = std::fs::write(
                        format!("{}/layer_{}_k_shadow.bin", dump_dir, l),
                        &k_shadow_host,
                    );
                    let _ = std::fs::write(
                        format!("{}/layer_{}_v_shadow.bin", dump_dir, l),
                        &v_shadow_host,
                    );
                    // Primary NVFP4 K/V (packed bytes). The total NVFP4
                    // allocation per layer is `layer_elems / 2` bytes
                    // (because `layer_elems = 2 * X` already counts K+V
                    // and NVFP4 packs 2 elems/byte). Each of K and V is
                    // therefore `layer_elems / 4` bytes within the
                    // layer, matching `bytes_per_half_kv = layer_kv_elems / 4`
                    // used by the rope launcher's `v_cache` offset.
                    // An earlier revision of this dump used
                    // `layer_elems / 2` for the per-side size, which
                    // (a) read past the layer's allocation for V
                    // and (b) made the dumped V file actually contain
                    // the NEXT layer's K data — producing apparent
                    // 100%+ V rel_err in the analyzer when in fact V
                    // was never read from the right offset.
                    let layer_kv_base = kv_cache.device_ptr() + kv_layer_offsets[l];
                    let primary_half_bytes = layer_elems / 4; // K-or-V bytes
                    let mut k_host = vec![0u8; primary_half_bytes as usize];
                    let mut v_host = vec![0u8; primary_half_bytes as usize];
                    cudarc::driver::sys::cuMemcpyDtoH_v2(
                        k_host.as_mut_ptr() as *mut _,
                        layer_kv_base,
                        primary_half_bytes as usize,
                    );
                    cudarc::driver::sys::cuMemcpyDtoH_v2(
                        v_host.as_mut_ptr() as *mut _,
                        layer_kv_base + primary_half_bytes,
                        primary_half_bytes as usize,
                    );
                    let _ = std::fs::write(format!("{}/layer_{}_k.bin", dump_dir, l), &k_host);
                    let _ = std::fs::write(format!("{}/layer_{}_v.bin", dump_dir, l), &v_host);
                    // NVFP4 scale region: E4M3, layer_elems/16 bytes total; first
                    // half for K, second half for V.
                    let layer_kv_scale_base =
                        kv_scale_cache.device_ptr() + kv_scale_layer_offsets[l];
                    let scale_half_bytes = layer_elems / 32; // each of K,V = /32
                    let mut k_scale_host = vec![0u8; scale_half_bytes as usize];
                    let mut v_scale_host = vec![0u8; scale_half_bytes as usize];
                    cudarc::driver::sys::cuMemcpyDtoH_v2(
                        k_scale_host.as_mut_ptr() as *mut _,
                        layer_kv_scale_base,
                        scale_half_bytes as usize,
                    );
                    cudarc::driver::sys::cuMemcpyDtoH_v2(
                        v_scale_host.as_mut_ptr() as *mut _,
                        layer_kv_scale_base + scale_half_bytes,
                        scale_half_bytes as usize,
                    );
                    let _ = std::fs::write(
                        format!("{}/layer_{}_k_scale.bin", dump_dir, l),
                        &k_scale_host,
                    );
                    let _ = std::fs::write(
                        format!("{}/layer_{}_v_scale.bin", dump_dir, l),
                        &v_scale_host,
                    );
                    // Per-layer Q dump (f16, post-RoPE). Snapshot written by
                    // `rope_f16kv_shadow` → memcpy hook in layer_exec.rs on
                    // decode step 0 into a dedicated per-layer slot of size
                    // `shadow_q_per_layer_bytes = num_attention_heads *
                    // max_head_dim * 2`. Tail may be zero when this layer's
                    // head_dim < max_head_dim; Python analyzer truncates
                    // using `head_dim` from meta.json.
                    let pos_in_set = lset.iter().position(|&li| li as usize == l);
                    if let Some(pi) = pos_in_set {
                        if shadow_q_ptr != 0 {
                            let q_slot_base =
                                shadow_q_ptr + (pi as u64) * shadow_q_per_layer_bytes;
                            let mut q_host = vec![0u8; shadow_q_per_layer_bytes as usize];
                            cudarc::driver::sys::cuMemcpyDtoH_v2(
                                q_host.as_mut_ptr() as *mut _,
                                q_slot_base,
                                shadow_q_per_layer_bytes as usize,
                            );
                            let _ = std::fs::write(
                                format!("{}/layer_{}_q.bin", dump_dir, l),
                                &q_host,
                            );
                        }
                    }
                    if !layer_meta_json.is_empty() {
                        layer_meta_json.push_str(",\n");
                    }
                    layer_meta_json.push_str(&format!(
                        "    {{\"layer\": {}, \"layer_type\": \"{:?}\", \"head_dim\": {}, \"num_kv_heads\": {}, \"num_blocks\": {}}}",
                        l, lt, hd, nkvh, layer_blocks,
                    ));
                }
                // Keep legacy single-layer Q dump (last executed layer, FP8)
                // for backward compat with older analyzer runs; per-layer
                // f16 Q files (layer_{L}_q.bin) are the canonical source.
                let q_bytes = (arch.num_attention_heads as u64) * (arch.max_head_dim() as u64);
                let mut q_host = vec![0u8; q_bytes as usize];
                cudarc::driver::sys::cuMemcpyDtoH_v2(
                    q_host.as_mut_ptr() as *mut _,
                    q_fp8.device_ptr(),
                    q_bytes as usize,
                );
                let _ = std::fs::write(format!("{}/q_last_layer.bin", dump_dir), &q_host);
                // meta.json
                let max_head_dim = arch.max_head_dim();
                let meta_json = format!(
                    "{{\n  \"prompt_len\": {},\n  \"num_layers\": {},\n  \"block_size\": {},\n  \"num_heads\": {},\n  \"max_head_dim\": {},\n  \"q_dtype\": \"f16\",\n  \"q_per_layer_bytes\": {},\n  \"context_len\": {},\n  \"slot_mapping\": {},\n  \"first_token_id\": {},\n  \"shadow_layer_indices\": {:?},\n  \"block_table\": {:?},\n  \"layers\": [\n{}\n  ]\n}}\n",
                    prompt_ids.len(),
                    arch.num_hidden_layers,
                    block_size,
                    arch.num_attention_heads,
                    max_head_dim,
                    shadow_q_per_layer_bytes,
                    ctx_host[0],
                    slot_host[0],
                    first_tok,
                    lset,
                    bt_host,
                    layer_meta_json,
                );
                let _ = std::fs::write(format!("{}/meta.json", dump_dir), &meta_json);
                eprintln!(
                    "[nvfp4-shadow] dumped {} instrumented layers to {} (ctx={}, first_tok={})",
                    lset.len(), dump_dir, ctx_host[0], first_tok,
                );
            }
            // === END NVFP4 SHADOW DIAGNOSTIC ===

            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: 1, hidden, eps: arch.rms_norm_eps,
            }.launch(kernels.fused_rmsnorm, residual_ptr, self.model.final_norm.offset_bytes, stream)?;
            self.cublaslt.f16_gemm_f32(residual_ptr, self.model.lm_head_f16.offset_bytes,
                logits_f32.device_ptr(), 1, vocab as i32, hidden as i32, stream)?;
            rvllm_fused::ArgmaxLaunch { num_tokens: 1, vocab }
                .launch(fn_argmax, logits_f32.device_ptr(), sampled.device_ptr(), stream)?;

            self.stream.fence()?;
            cudarc::driver::sys::cuMemcpyDtoH_v2(host_tok.as_mut_ptr() as *mut _, sampled.device_ptr(), 4);
            let next_id = host_tok[0] as u32;
            output_ids.push(next_id);
            if eos_ids.contains(&next_id) { break; }
        }

        let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let decode_ms = total_ms - prefill_ms;
        eprintln!("[generate] {} tokens decoded in {:.1}ms ({:.1} tok/s)",
            output_ids.len(), decode_ms, output_ids.len() as f64 / (decode_ms / 1000.0));

        // Update the prefix cache with this request's prompt so the
        // next request can benefit from a cache hit. We cache ONLY
        // the prompt tokens (not the generated output) — the
        // generated tokens' KV entries are in slots [prompt_len..]
        // and are indeed valid, but zeroclaw typically includes
        // prior assistant responses in the NEXT prompt's history
        // anyway, so persisting generated-token KV here adds
        // complexity without extra benefit.
        if use_prefix_cache {
            if let Ok(mut guard) = self.prefix_cache.lock() {
                if let Some(pc) = guard.as_mut() {
                    pc.last_tokens.clear();
                    pc.last_tokens.extend_from_slice(prompt_ids);
                    // Cap committed prefix at the last full chunk
                    // boundary. Slots written by a trailing partial
                    // chunk are unsafe to reuse — see
                    // PrefixCacheState::committed_prefix_len doc.
                    let chunk_size: u32 = std::env::var("RVLLM_PREFILL_CHUNK_SIZE")
                        .ok().and_then(|s| s.parse().ok()).unwrap_or(0);
                    let prompt_len_u32 = prompt_ids.len() as u32;
                    pc.committed_prefix_len = if chunk_size > 0 && prompt_len_u32 >= chunk_size {
                        (prompt_len_u32 / chunk_size) * chunk_size
                    } else {
                        // No chunking, or prompt fits in a single
                        // chunk → entire prompt was written under
                        // one (unique-shape) launch. Reusing it is
                        // ONLY safe when the next request would
                        // also fit in one chunk; the provenance
                        // chunk_size mismatch will catch that.
                        prompt_len_u32
                    };
                    // Refresh provenance so subsequent provenance
                    // checks compare against the env that ACTUALLY
                    // wrote this KV state.
                    pc.provenance = PrefixProvenance::from_env();
                }
            }
        }
        Ok(output_ids)
    }

    pub fn layer_kernels(&self) -> Gemma4LayerKernels {
        // NVFP4 RoPE kernel handle — `None` on branches without the
        // NVFP4 PTX built into $KERNELS_DIR. Lives on Fa2PtxKernels
        // so the module lifetime outlives the fn handle; extracting
        // via `match` here instead of a helper method to avoid
        // enlarging the AttentionBackend API for a single field.
        #[cfg(feature = "cuda")]
        let fused_rope_partial_nvfp4kv = match &self.sliding_attention {
            rvllm_attention::AttentionBackend::Fa2Ptx(fa2) => fa2.fn_rope_nvfp4kv,
            _ => None,
        };
        #[cfg(not(feature = "cuda"))]
        let fused_rope_partial_nvfp4kv = None;

        Gemma4LayerKernels {
            fused_rmsnorm: self.fused.fn_rmsnorm,
            fused_rmsnorm_fp8_quant: self.fused.fn_rmsnorm_fp8_quant,
            fused_qk_rmsnorm: self.fused.fn_qk_rmsnorm,
            fused_rope_partial_fp8kv: self.fused.fn_rope_partial_fp8kv,
            fused_rope_partial_nvfp4kv,
            fused_gelu_mul: self.fused.fn_gelu_mul,
            quantize_fp8_per_token: self.fused.fn_quantize,
            residual_scale_f16: self.fused.fn_residual_scale,
            vnorm_f16: self.fused.fn_vnorm,
            vector_add_f16: self.fused.fn_vector_add,
            bf16_to_f16_sat: self.fused.fn_bf16_to_f16_sat,
            rmsnorm_inplace_bf16: self.fused.fn_rmsnorm_inplace_bf16,
            vector_add_bf16_to_f16: self.fused.fn_vector_add_bf16_to_f16,
            f32_to_bf16: self.fused.fn_f32_to_bf16,
            f32_to_f16_sat: self.fused.fn_f32_to_f16_sat,
            scale_cols_f32: self.fused.fn_scale_cols_f32,
            scale_rows_f32_ratio: self.fused.fn_scale_rows_f32_ratio,
            compute_qkv_scales: self.fused.fn_compute_qkv_scales,
            fused_gelu_mul_f16: self.fused.fn_fused_gelu_mul_f16,
            fused_rope_partial_f16kv: self.fused.fn_fused_rope_partial_f16kv,
            fused_norm_add_residual: self.fused.fn_fused_norm_add_residual,
            fused_norm_add_residual_f16: self.fused.fn_fused_norm_add_residual_f16,
            fused_norm_add_residual_f16in: self.fused.fn_fused_norm_add_residual_f16in,
            fused_qkv_rmsnorm: self.fused.fn_fused_qkv_rmsnorm,
            scale_cols_f16: self.fused.fn_scale_cols_f16,
            fp8_gemv_wpr_native_f16in: self.fused.fn_fp8_gemv_wpr_native_f16in,
        }
    }
}

fn bytemuck_cast_i32(v: &[i32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

fn load_gemma4_fused(
    loader: &KernelLoader,
    target: Option<rvllm_core::CompileTarget>,
) -> Result<Gemma4FusedModules> {
    let rmsnorm_mod = loader.load_ptx("fused_rmsnorm_fp8_quant")?;
    let rope_mod = loader.load_ptx("fused_rope_partial_fp8kv")?;
    let gelu_mod = loader.load_ptx("fused_gelu_mul_fp8_quant")?;
    let argmax_mod = loader.load_ptx("argmax")?;
    let qk_norm_mod = loader.load_ptx("fused_qk_rmsnorm")?;
    let softcap_mod = loader.load_ptx("logit_softcap")?;
    let residual_scale_mod = loader.load_ptx("residual_scale_f16")?;
    let vnorm_mod = loader.load_ptx("vnorm_f16")?;
    let vector_add_mod = loader.load_ptx("vector_add_f16")?;
    let bf16_to_f16_sat_mod = loader.load_ptx("bf16_to_f16_sat")?;
    let rmsnorm_inplace_bf16_mod = loader.load_ptx("rmsnorm_inplace_bf16")?;
    let vector_add_bf16_to_f16_mod = loader.load_ptx("vector_add_bf16_to_f16")?;
    let f32_to_bf16_mod = loader.load_ptx("f32_to_bf16")?;
    let f32_to_f16_sat_mod = loader.load_ptx("f32_to_f16_sat")?;

    let rmsnorm_inplace_mod = loader.load_ptx("rmsnorm_inplace_f16")?;
    let fn_rmsnorm = rmsnorm_inplace_mod.get_function("rmsnorm_inplace_f16_kernel")?;
    let fn_rmsnorm_fp8_quant = rmsnorm_mod.get_function("fused_rmsnorm_fp8_quant_kernel")?;
    let fn_quantize = rmsnorm_mod.get_function("quantize_fp8_per_token_kernel")?;
    let fn_rope_partial_fp8kv = rope_mod.get_function("fused_rope_partial_fp8kv_kernel")?;
    let fn_gelu_mul = gelu_mod.get_function("fused_gelu_mul_fp8_quant_kernel")?;
    let fn_argmax = argmax_mod.get_function("argmax_kernel")?;
    let fn_qk_rmsnorm = qk_norm_mod.get_function("fused_qk_rmsnorm_kernel")?;
    let fn_softcap = softcap_mod.get_function("logit_softcap_kernel")?;
    let fn_residual_scale = residual_scale_mod.get_function("residual_scale_f16_kernel")?;
    let fn_vnorm = vnorm_mod.get_function("vnorm_f16_kernel")?;
    let fn_vector_add = vector_add_mod.get_function("vector_add_f16_kernel")?;
    let fn_bf16_to_f16_sat = bf16_to_f16_sat_mod.get_function("bf16_to_f16_sat_kernel")?;
    let fn_rmsnorm_inplace_bf16 =
        rmsnorm_inplace_bf16_mod.get_function("rmsnorm_inplace_bf16_kernel")?;
    let fn_vector_add_bf16_to_f16 =
        vector_add_bf16_to_f16_mod.get_function("vector_add_bf16_to_f16_kernel")?;
    let fn_f32_to_bf16 = f32_to_bf16_mod.get_function("f32_to_bf16_kernel")?;
    let fn_f32_to_f16_sat = f32_to_f16_sat_mod.get_function("f32_to_f16_sat_kernel")?;

    let scale_cols_f32_mod = loader.load_ptx("scale_cols_f32")?;
    let fn_scale_cols_f32 = scale_cols_f32_mod.get_function("scale_cols_f32_kernel")?;
    let scale_rows_f32_ratio_mod = loader.load_ptx("scale_rows_f32_ratio")?;
    let fn_scale_rows_f32_ratio =
        scale_rows_f32_ratio_mod.get_function("scale_rows_f32_ratio_kernel")?;

    let compute_qkv_scales_mod = loader.load_ptx("compute_qkv_scales")?;
    let fn_compute_qkv_scales = compute_qkv_scales_mod.get_function("compute_qkv_scales_kernel")?;

    let fused_gelu_mul_f16_mod = loader.load_ptx("fused_gelu_mul_f16")?;
    let fn_fused_gelu_mul_f16 = fused_gelu_mul_f16_mod.get_function("fused_gelu_mul_f16_kernel")?;

    let fused_rope_partial_f16kv_mod = loader.load_ptx("fused_rope_partial_f16kv")?;
    let fn_fused_rope_partial_f16kv =
        fused_rope_partial_f16kv_mod.get_function("fused_rope_partial_f16kv_kernel")?;

    // `fp8_gemv.ptx` — see struct docs. The f16-input native-CVT
    // entry is gated on `__CUDA_ARCH__ >= 1000` in
    // `kernels/fp8_gemv.cu`, so we only resolve it when
    // `Fp8GemvVariant::available_for(target)` says yes.
    let fp8_gemv_mod = loader.load_ptx(rvllm_kernels::FP8_GEMV_PTX_STEM)?;
    let fn_fp8_gemv_wpr_native_f16in = match target {
        Some(t) if rvllm_kernels::Fp8GemvVariant::WprNativeF16In.available_for(t) => Some(
            fp8_gemv_mod
                .get_function(rvllm_kernels::Fp8GemvVariant::WprNativeF16In.entry_point())?,
        ),
        _ => None,
    };

    let fused_norm_add_residual_mod = loader.load_ptx("fused_norm_add_residual")?;
    let fn_fused_norm_add_residual =
        fused_norm_add_residual_mod.get_function("fused_norm_add_residual_kernel")?;

    let fused_norm_add_residual_f16_mod = loader.load_ptx("fused_norm_add_residual_f16")?;
    let fn_fused_norm_add_residual_f16 =
        fused_norm_add_residual_f16_mod.get_function("fused_norm_add_residual_f16_kernel")?;
    let fn_fused_norm_add_residual_f16in =
        fused_norm_add_residual_f16_mod.get_function("fused_norm_add_residual_f16in_kernel")?;

    let fused_qkv_rmsnorm_mod = loader.load_ptx("fused_qkv_rmsnorm")?;
    let fn_fused_qkv_rmsnorm =
        fused_qkv_rmsnorm_mod.get_function("fused_qkv_rmsnorm_kernel")?;

    let scale_cols_f16_mod = loader.load_ptx("scale_cols_f16")?;
    let fn_scale_cols_f16 = scale_cols_f16_mod.get_function("scale_cols_f16_kernel")?;

    Ok(Gemma4FusedModules {
        rmsnorm_mod,
        rmsnorm_inplace_mod,
        rope_mod,
        gelu_mod,
        argmax_mod,
        qk_norm_mod,
        softcap_mod,
        residual_scale_mod,
        vnorm_mod,
        vector_add_mod,
        bf16_to_f16_sat_mod,
        rmsnorm_inplace_bf16_mod,
        vector_add_bf16_to_f16_mod,
        f32_to_bf16_mod,
        f32_to_f16_sat_mod,
        scale_cols_f32_mod,
        scale_rows_f32_ratio_mod,
        compute_qkv_scales_mod,
        fused_gelu_mul_f16_mod,
        fused_rope_partial_f16kv_mod,
        fused_norm_add_residual_mod,
        fn_rmsnorm,
        fn_rmsnorm_fp8_quant,
        fn_quantize,
        fn_rope_partial_fp8kv,
        fn_gelu_mul,
        fn_argmax,
        fn_qk_rmsnorm,
        fn_softcap,
        fn_residual_scale,
        fn_vnorm,
        fn_vector_add,
        fn_bf16_to_f16_sat,
        fn_rmsnorm_inplace_bf16,
        fn_vector_add_bf16_to_f16,
        fn_f32_to_bf16,
        fn_f32_to_f16_sat,
        fn_scale_cols_f32,
        fn_scale_rows_f32_ratio,
        fn_compute_qkv_scales,
        fn_fused_gelu_mul_f16,
        fn_fused_rope_partial_f16kv,
        fn_fused_norm_add_residual,
        fn_fused_norm_add_residual_f16,
        fn_fused_norm_add_residual_f16in,
        fused_norm_add_residual_f16_mod,
        fn_fused_qkv_rmsnorm,
        fused_qkv_rmsnorm_mod,
        fn_scale_cols_f16,
        scale_cols_f16_mod,
        fp8_gemv_mod,
        fn_fp8_gemv_wpr_native_f16in,
    })
}
