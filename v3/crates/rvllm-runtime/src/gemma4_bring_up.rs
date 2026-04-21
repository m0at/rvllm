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
}

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
        let variants: Vec<_> = variants.into_iter().collect();
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
        })
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

        let kv_bytes_per_elem: u32 = 1; // bench path always FP8
        // aa01001pftrope0 cliff-fix: sliding layers need `slot_mapping[t] < sliding_blocks*block_size`
        // at every t the rope writes; the old cap sliding_blocks = sliding_window/block_size = 32
        // (= 1024 slots for Gemma 4) broke at prompt_len > sliding_window because slot_mapping
        // is linear 0..prompt_len-1 and index 1024+ ran off the end of the sliding KV region.
        // Proper fix is a per-sliding-layer ring buffer (slot = t mod sliding_window) but that
        // needs rope + attention kernel cooperation. For now give sliding layers the full pool —
        // ~10 GiB extra at num_blocks_total=1024, fits in the 50 GiB arena with Gemma 4 31B fp8.
        let sliding_blocks = num_blocks_total;

        let mut kv_layer_offsets: Vec<u64> = Vec::with_capacity(arch.num_hidden_layers);
        let mut kv_total_bytes: u64 = 0;
        let mut kv_scale_layer_offsets: Vec<u64> = Vec::with_capacity(arch.num_hidden_layers);
        let mut kv_scale_total_bytes: u64 = 0;
        for l in 0..arch.num_hidden_layers {
            kv_layer_offsets.push(kv_total_bytes);
            kv_scale_layer_offsets.push(kv_scale_total_bytes);
            let layer_blocks = if arch.layer_types[l] == rvllm_loader::gemma4_arch::Gemma4LayerType::GlobalAttention { num_blocks_total } else { sliding_blocks };
            let nkvh_l = arch.num_kv_heads_for_layer(l) as u32;
            let hd_l = arch.head_dim_for_layer(l) as u32;
            let layer_elems = 2u64 * layer_blocks as u64 * block_size as u64 * nkvh_l as u64 * hd_l as u64;
            kv_total_bytes += layer_elems * kv_bytes_per_elem as u64;
            let layer_scale_slots = 2u64 * layer_blocks as u64 * block_size as u64 * nkvh_l as u64;
            kv_scale_total_bytes += layer_scale_slots * 4;
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
                    f16_kv: f16_only || std::env::var("RVLLM_F16_KV").map_or(true, |v| v != "0"),
                };

                // Row-major [num_tokens, q_dim+2*kv_dim]: k_out / v_out
                // point at row 0's K / V sub-slice. The rmsnorm kernel
                // applies `src_row_stride` to reach later tokens.
                let k_out = q_base + (q_dim as u64) * 2;
                let v_out = k_out + (kv_dim as u64) * 2;
                let is_global = lt == Gemma4LayerType::GlobalAttention;
                let layer_blocks = if is_global { num_blocks_total } else { sliding_blocks };
                let layer_kv_elems = 2u64 * layer_blocks as u64 * block_size as u64 * nkvh as u64 * hd as u64;
                let kv_layer_bytes = layer_kv_elems * kv_bytes_per_elem as u64;
                let layer_kv_base = kv_cache.device_ptr() + kv_layer_offsets[layer_idx];
                let layer_kv_scale_base =
                    kv_scale_cache.device_ptr() + kv_scale_layer_offsets[layer_idx];
                let layer_kv_scale_slots_half =
                    (layer_blocks as u64) * (block_size as u64) * (nkvh as u64);

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
        let use_f16_kv = f16_only || std::env::var("RVLLM_F16_KV").map_or(true, |v| v != "0");
        let kv_bytes_per_elem: u32 = if use_f16_kv { 2 } else { 1 };

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
        let mut kv_total_bytes: u64 = 0;
        let mut kv_scale_layer_offsets: Vec<u64> = Vec::with_capacity(arch.num_hidden_layers);
        let mut kv_scale_total_bytes: u64 = 0;
        for l in 0..arch.num_hidden_layers {
            kv_layer_offsets.push(kv_total_bytes);
            kv_scale_layer_offsets.push(kv_scale_total_bytes);
            let is_global = arch.layer_types[l] == rvllm_loader::gemma4_arch::Gemma4LayerType::GlobalAttention;
            let layer_blocks = if is_global { num_blocks_total } else { sliding_blocks };
            let nkvh = arch.num_kv_heads_for_layer(l) as u32;
            let hd = arch.head_dim_for_layer(l) as u32;
            let layer_elems = 2u64 * layer_blocks as u64 * block_size as u64 * nkvh as u64 * hd as u64;
            kv_total_bytes += layer_elems * kv_bytes_per_elem as u64;
            let layer_scale_slots = 2u64 * layer_blocks as u64 * block_size as u64 * nkvh as u64;
            kv_scale_total_bytes += layer_scale_slots * 4;
        }
        eprintln!("[ppl] KV cache: {:.1} MB (sliding={} blocks, global={} blocks, {} bytes/elem)",
            kv_total_bytes as f64 / 1e6, sliding_blocks, num_blocks_total, kv_bytes_per_elem);

        let kv_cache = arena.region("kv_cache", kv_total_bytes as usize, 256)?;
        cudarc::driver::sys::cuMemsetD8_v2(kv_cache.device_ptr(), 0, kv_total_bytes as usize);
        let kv_scale_cache =
            arena.region("kv_scale_cache", kv_scale_total_bytes as usize, 16)?;
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
                    f16_kv: f16_only || std::env::var("RVLLM_F16_KV").map_or(true, |v| v != "0"),
                };

                // Row-major [num_tokens, q_dim+2*kv_dim]: k_out / v_out
                // point at row 0's K / V sub-slice. The rmsnorm kernel
                // applies `src_row_stride` to reach later tokens.
                let k_out = q_base + (q_dim as u64) * 2;
                let v_out = k_out + (kv_dim as u64) * 2;
                let is_global = lt == Gemma4LayerType::GlobalAttention;
                let layer_blocks = if is_global { num_blocks_total } else { sliding_blocks };
                let layer_kv_elems = 2u64 * layer_blocks as u64 * block_size as u64 * nkvh as u64 * hd as u64;
                let kv_layer_bytes = layer_kv_elems * kv_bytes_per_elem as u64;
                let layer_kv_base = kv_cache.device_ptr() + kv_layer_offsets[layer_idx];
                let layer_kv_scale_base =
                    kv_scale_cache.device_ptr() + kv_scale_layer_offsets[layer_idx];
                let layer_kv_scale_slots_half =
                    (layer_blocks as u64) * (block_size as u64) * (nkvh as u64);
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

        // aa01001pftrope0 cliff-fix: sliding layers need `slot_mapping[t] < sliding_blocks*block_size`
        // at every t the rope writes; the old cap sliding_blocks = sliding_window/block_size = 32
        // (= 1024 slots for Gemma 4) broke at prompt_len > sliding_window because slot_mapping
        // is linear 0..prompt_len-1 and index 1024+ ran off the end of the sliding KV region.
        // Proper fix is a per-sliding-layer ring buffer (slot = t mod sliding_window) but that
        // needs rope + attention kernel cooperation. For now give sliding layers the full pool —
        // ~10 GiB extra at num_blocks_total=1024, fits in the 50 GiB arena with Gemma 4 31B fp8.
        let sliding_blocks = num_blocks_total;
        let use_f16_kv = std::env::var("RVLLM_F16_KV").map_or(true, |v| v != "0");
        let kv_bytes_per_elem: u32 = if use_f16_kv { 2 } else { 1 };
        let mut kv_layer_offsets: Vec<u64> = Vec::with_capacity(arch.num_hidden_layers);
        let mut kv_total_bytes: u64 = 0;
        // Per-slot-per-head K/V scale cache offsets (f32 per entry).
        // One entry per (slot, kv_head) — factor `head_dim` smaller
        // than the FP8 KV cache region.
        let mut kv_scale_layer_offsets: Vec<u64> = Vec::with_capacity(arch.num_hidden_layers);
        let mut kv_scale_total_bytes: u64 = 0;
        for l in 0..arch.num_hidden_layers {
            kv_layer_offsets.push(kv_total_bytes);
            kv_scale_layer_offsets.push(kv_scale_total_bytes);
            let is_global = arch.layer_types[l] == rvllm_loader::gemma4_arch::Gemma4LayerType::GlobalAttention;
            let layer_blocks = if is_global { num_blocks_total } else { sliding_blocks };
            let nkvh = arch.num_kv_heads_for_layer(l) as u32;
            let hd = arch.head_dim_for_layer(l) as u32;
            let layer_elems = 2u64 * layer_blocks as u64 * block_size as u64 * nkvh as u64 * hd as u64;
            kv_total_bytes += layer_elems * kv_bytes_per_elem as u64;
            // Scale storage: [2 (K+V)] × num_slots × num_kv_heads × f32.
            let layer_scale_slots = 2u64 * layer_blocks as u64 * block_size as u64 * nkvh as u64;
            kv_scale_total_bytes += layer_scale_slots * 4;
        }
        let kv_cache = arena.region("gen_kv", kv_total_bytes as usize, 256)?;
        cudarc::driver::sys::cuMemsetD8_v2(kv_cache.device_ptr(), 0, kv_total_bytes as usize);
        let kv_scale_cache =
            arena.region("gen_kv_scale_cache", kv_scale_total_bytes as usize, 16)?;
        cudarc::driver::sys::cuMemsetD8_v2(
            kv_scale_cache.device_ptr(), 0, kv_scale_total_bytes as usize);
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

                let dims = crate::gemma4_layer_exec::Gemma4LayerDims {
                    num_tokens: 1, hidden,
                    num_heads: arch.num_attention_heads as u32, num_kv_heads: nkvh, head_dim: hd,
                    rotary_dim: arch.rotary_dim_for_layer(layer_idx) as u32,
                    intermediate: inter, block_size,
                    max_blocks_per_seq: layer_blocks, num_blocks_total: layer_blocks,
                    attn_scale: 1.0, rms_eps: arch.rms_norm_eps,
                    layer_type: lt, sliding_window: arch.sliding_window_size as u32,
                    f16_kv: use_f16_kv,
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
                let scratch = crate::gemma4_layer_exec::Gemma4LayerScratch {
                    hidden_fp8: hidden_fp8.device_ptr(), hidden_scale: hidden_scale.device_ptr(),
                    q_out: q_base, k_out, v_out,
                    q_normed: q_normed.device_ptr(), k_normed: k_normed.device_ptr(),
                    v_normed: v_normed.device_ptr(),
                    q_fp8: q_fp8.device_ptr(),
                    k_cache: layer_kv_base,
                    v_cache: layer_kv_base + (layer_kv_elems / 2) * kv_bytes_per_elem as u64,
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
            for (i, &tok) in prompt_ids.iter().enumerate() {
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
            let tok_ids: Vec<i32> = prompt_ids.iter().map(|&t| t as i32).collect();
            token_ids_region.copy_from_host(bytemuck_cast_i32(&tok_ids))?;
            // Readback sanity-check: confirm the device buffer actually
            // holds the full prompt before the gather kernel runs.
            self.stream.fence()?;
            let mut readback = vec![0i32; prompt_len as usize];
            cudarc::driver::sys::cuMemcpyDtoH_v2(
                readback.as_mut_ptr() as *mut _,
                token_ids_region.device_ptr(),
                (prompt_len * 4) as _,
            );
            eprintln!("[DIAG] token_ids_region readback = {:?}", readback);
            rvllm_fused::EmbeddingGatherLaunch { num_tokens: prompt_len, hidden, vocab }
                .launch(fn_embed, residual_ptr, self.model.embedding.offset_bytes,
                    token_ids_region.device_ptr(), stream)?;
            // Also read back rows 0 and N-1 of residual IMMEDIATELY
            // after the gather, BEFORE any layers run.
            self.stream.fence()?;
            let mut r0 = vec![0u16; 4];
            let mut r_n_minus_1 = vec![0u16; 4];
            cudarc::driver::sys::cuMemcpyDtoH_v2(r0.as_mut_ptr() as *mut _, residual_ptr, 8);
            cudarc::driver::sys::cuMemcpyDtoH_v2(
                r_n_minus_1.as_mut_ptr() as *mut _,
                residual_ptr + ((prompt_len - 1) as u64 * hidden as u64 * 2),
                8,
            );
            eprintln!(
                "[DIAG] post-gather row0[..4]={:?} rowN-1[..4]={:?}",
                r0.iter().map(|&x| crate::bring_up::f16_to_f32(x)).collect::<Vec<_>>(),
                r_n_minus_1.iter().map(|&x| crate::bring_up::f16_to_f32(x)).collect::<Vec<_>>(),
            );

            let pos: Vec<i32> = (0..prompt_len as i32).collect();
            let slot: Vec<i32> = (0..prompt_len as i32).collect();
            let ctx = [prompt_len as i32];
            let cu_seq = [0i32, prompt_len as i32];
            positions.copy_from_host(bytemuck_cast_i32(&pos))?;
            slot_mapping.copy_from_host(bytemuck_cast_i32(&slot))?;
            context_lens.copy_from_host(bytemuck_cast_i32(&ctx))?;
            cu_seqlens_q.copy_from_host(bytemuck_cast_i32(&cu_seq))?;

            let phase = crate::gemma4_layer_exec::Gemma4Phase::Prefill {
                cu_seqlens_q: cu_seqlens_q.device_ptr(),
                max_seqlen_q: prompt_len,
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

                let dims = crate::gemma4_layer_exec::Gemma4LayerDims {
                    num_tokens: prompt_len, hidden,
                    num_heads: arch.num_attention_heads as u32, num_kv_heads: nkvh, head_dim: hd,
                    rotary_dim: arch.rotary_dim_for_layer(layer_idx) as u32,
                    intermediate: inter, block_size,
                    max_blocks_per_seq: layer_blocks, num_blocks_total: layer_blocks,
                    attn_scale: 1.0, rms_eps: arch.rms_norm_eps,
                    layer_type: lt, sliding_window: arch.sliding_window_size as u32,
                    f16_kv: false, // prefill uses FP8 KV (no F16 prefill kernel)
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
                let scratch = crate::gemma4_layer_exec::Gemma4LayerScratch {
                    hidden_fp8: hidden_fp8.device_ptr(), hidden_scale: hidden_scale.device_ptr(),
                    q_out: q_base, k_out, v_out,
                    q_normed: q_normed.device_ptr(), k_normed: k_normed.device_ptr(),
                    v_normed: v_normed.device_ptr(),
                    q_fp8: q_fp8.device_ptr(),
                    k_cache: layer_kv_base,
                    v_cache: layer_kv_base + (layer_kv_elems / 2) * kv_bytes_per_elem as u64,
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

            // Diag capture BEFORE the extract-last memcpy: row 0 is
            // the first-token output (no prior context); row
            // prompt_len-1 is the last-token output the LM head
            // consumes. Without this ordering, the memcpy below
            // would clobber row 0 with row N-1 and the row-0
            // comparison would falsely report a bug.
            self.stream.fence()?;
            let mut prefill_first = vec![0u16; hidden as usize];
            cudarc::driver::sys::cuMemcpyDtoH_v2(
                prefill_first.as_mut_ptr() as *mut _,
                residual_ptr,
                (hidden * 2) as _,
            );
            let mut prefill_last = vec![0u16; hidden as usize];
            let last_off_diag = (prompt_len - 1) as u64 * hidden as u64 * 2;
            cudarc::driver::sys::cuMemcpyDtoH_v2(
                prefill_last.as_mut_ptr() as *mut _,
                residual_ptr + last_off_diag,
                (hidden * 2) as _,
            );

            // Extract last token's residual for decode
            if prompt_len > 1 {
                let last_offset = (prompt_len - 1) as u64 * hidden as u64 * 2;
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
        Ok(output_ids)
    }

    pub fn layer_kernels(&self) -> Gemma4LayerKernels {
        Gemma4LayerKernels {
            fused_rmsnorm: self.fused.fn_rmsnorm,
            fused_rmsnorm_fp8_quant: self.fused.fn_rmsnorm_fp8_quant,
            fused_qk_rmsnorm: self.fused.fn_qk_rmsnorm,
            fused_rope_partial_fp8kv: self.fused.fn_rope_partial_fp8kv,
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
