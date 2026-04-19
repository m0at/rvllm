//! Engine bring-up: assemble every subsystem from paths on disk.
//!
//! This module exists so `main.rs` in the bench + serve binaries can
//! reach for one `bring_up::Engine::load(paths)` call and get a fully
//! wired runtime back. No graph capture here (that's a separate step
//! after weights are loaded and bucket shapes are known).

use std::path::PathBuf;
use std::sync::Arc;

use rvllm_attention::{AttentionBackend, Fa3Kernels};
#[cfg(feature = "cuda")]
use rvllm_core::CompileTarget;
use rvllm_core::{ConfigError, Result, RvllmError};
use rvllm_cutlass::{CublasLt, CutlassBackend, Fp8GemmPlan, Policy};
use rvllm_kernels::{manifest::KernelManifest, KernelFn, KernelLoader, LoadedModule};
use rvllm_loader::{load_model, LoadedModel, ModelArch};
use rvllm_mem::{context::CudaContextHandle, stream::Stream, HbmArena};

/// Paths (and only paths) the engine needs at init. All other config
/// is read from `model_dir/config.json` and `kernels_dir/manifest.json`.
#[derive(Clone, Debug)]
pub struct EnginePaths {
    pub model_dir: PathBuf,
    pub kernels_dir: PathBuf,
    pub cutlass_so: PathBuf,
    pub fa3_so: PathBuf,
    pub policy_json: PathBuf,
}

/// Assembled subsystems.
///
/// Field order matters for Drop: CUDA resources (modules, .so handles,
/// memory) must drop BEFORE the context. Rust drops fields in source
/// order, so everything that touches CUDA comes first and the context
/// is last.
pub struct Bringup {
    pub fused_modules: FusedModules,
    pub fa3: AttentionBackend,
    pub cutlass: CutlassBackend,
    pub cublaslt: CublasLt,
    pub cublaslt_ws: HbmArenaCheckpoint,
    pub policy: Policy,
    pub arch: ModelArch,
    pub model: LoadedModel,
    pub kernels: Arc<KernelLoader>,
    pub stream: Stream,
    pub arena: HbmArena<'static>,
    pub ctx: Arc<CudaContextHandle>,
}

/// Marker: an arena-backed region kept alive by Bringup for the
/// lifetime of the program. cuBLASLt workspace lives here.
pub struct HbmArenaCheckpoint {
    pub offset_bytes: usize,
    pub bytes: usize,
}

/// Loaded CUDA modules + resolved kernel handles. One PTX file per
/// logical group; `fused_rmsnorm_fp8_quant.ptx` holds three kernels
/// (rmsnorm-only, add-rmsnorm, per-token quantize) so one module is
/// reused for three handles.
pub struct FusedModules {
    pub rmsnorm_mod: LoadedModule,
    pub rope_mod: LoadedModule,
    pub silu_mod: LoadedModule,
    pub gelu_mod: Option<LoadedModule>,
    pub argmax_mod: LoadedModule,
    pub add_bias_mod: LoadedModule,
    pub fn_rmsnorm: KernelFn,
    pub fn_add_rmsnorm: KernelFn,
    pub fn_quantize: KernelFn,
    pub fn_rope_cache_fp8kv: KernelFn,
    pub fn_silu_mul: KernelFn,
    pub fn_gelu_mul: Option<KernelFn>,
    pub fn_argmax: KernelFn,
    pub fn_add_bias_f16: KernelFn,
}

impl Bringup {
    pub fn load(paths: EnginePaths, arena_bytes: usize) -> Result<Self> {
        // 1. CUDA context + stream.
        let ctx = Arc::new(CudaContextHandle::init(0)?);

        // Pick arena backing per device compute capability. On GB10
        // (sm_121) — which has no dedicated HBM — route through
        // `UnifiedArena::new` (`cuMemAllocManaged(ATTACH_GLOBAL)` +
        // `cuMemAdvise(SET_PREFERRED_LOCATION, device)`), then unwrap
        // back to `HbmArena` so the storage type on `Bringup` stays
        // uniform. Everywhere else we keep the original
        // `cuMemAlloc_v2` fast path (`HbmArena::new`). Both call
        // `cuMemFree_v2` on Drop, which correctly releases either
        // allocation.
        //
        // Gated on `feature = "gb10"` so pre-Blackwell / non-GB10
        // builds don't pay for managed-memory overhead they don't
        // need.
        let arena = {
            #[cfg(feature = "gb10")]
            {
                let target = rvllm_core::CompileTarget::from_compute_capability(
                    ctx.compute_capability().0,
                    ctx.compute_capability().1,
                );
                if matches!(target, Some(rvllm_core::CompileTarget::Sm121)) {
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
        // SAFETY: arena lifetime 'static via transmute — engine owns it for program lifetime.
        // HbmArena<'ctx> borrows from ctx; the Arc keeps ctx alive. This is sound
        // as long as `ctx` outlives `arena` — which it does (same struct, ctx dropped last).
        let arena: HbmArena<'static> = unsafe { std::mem::transmute(arena) };

        let stream = Stream::new(&ctx)?;

        // 2. Arch + model.
        let arch = ModelArch::from_dir(&paths.model_dir)?;
        let model = load_model(&paths.model_dir, &arena, &arch)?;

        // 3. Kernel manifest -> loader -> modules.
        //    Resolve the per-arch kernel subdirectory from the device's
        //    compute capability. A device we don't build PTX for is a
        //    hard error — no silent fallback to a generic arch.
        let kernels_dir = resolve_kernels_dir(&ctx, &paths.kernels_dir)?;
        let manifest_path = kernels_dir.join("manifest.json");
        let manifest = KernelManifest::load_and_verify(&manifest_path)?;
        let kernels = Arc::new(KernelLoader::new(manifest));
        let fused_modules = load_fused(&kernels)?;

        // 4. Attention backend.
        //    Non-Gemma4 architectures currently always use the FA3 .so.
        //    The Gemma4 bring-up branches on CompileTarget; if you add a
        //    non-Gemma4 sm_121 model, wire the same branch here.
        let fa3 = AttentionBackend::Fa3(Fa3Kernels::load(
            paths.fa3_so.clone(),
            arch.head_dim as u32,
        )?);

        // 5. Policy + CUTLASS .so (resolve every variant referenced in
        //    the policy).
        let policy_bytes = std::fs::read(&paths.policy_json).map_err(|source| RvllmError::Io {
            err: rvllm_core::IoError::from(&source),
            path: paths.policy_json.clone(),
            source,
        })?;
        let policy: Policy = serde_json::from_slice(&policy_bytes).map_err(|e| {
            RvllmError::config(
                ConfigError::Inconsistent {
                    reasons: vec![format!("policy.json parse: {e}")],
                },
                "policy.json",
            )
        })?;
        // Pre-resolve a generous universe of variants so a bench sweep
        // can try any of them without re-bringup. If a symbol is
        // missing from the .so the load path returns typed err — that's
        // expected for a sweep run against a .so without some variant.
        let mut variants: std::collections::BTreeSet<_> =
            policy.entries.values().map(|e| e.variant).collect();
        for v in 0..16u32 {
            variants.insert(rvllm_cutlass::VariantId(v));
        }
        for v in 100..110u32 {
            variants.insert(rvllm_cutlass::VariantId(v));
        }
        let variants: Vec<_> = variants.into_iter().collect();
        // CUTLASS backend selection — sm_121 has no compatible `.so`
        // (CUTLASS SM90 collectives rely on WGMMA + TMA multicast,
        // both Hopper-only). On sm_121 we route through
        // `CutlassBackend::Absent`; FP8 GEMM launches return
        // `CutlassError::FeatureNotAvailable` until a sm_121-native
        // GEMM path lands (next GB10 follow-up).
        let cutlass_target = {
            #[cfg(feature = "cuda")]
            {
                let (maj, min) = ctx.compute_capability();
                rvllm_core::CompileTarget::from_compute_capability(maj, min)
            }
            #[cfg(not(feature = "cuda"))]
            {
                None
            }
        };
        let cutlass =
            CutlassBackend::load_for(cutlass_target, paths.cutlass_so.clone(), &variants)?;

        // cuBLASLt workspace: 32 MiB is recommended for Hopper FP8.
        let cublaslt_ws_bytes: usize = 32 * 1024 * 1024;
        let cublaslt_ws_region = arena.region("cublaslt_ws", cublaslt_ws_bytes, 256)?;
        let cublaslt = CublasLt::new(cublaslt_ws_region.device_ptr(), cublaslt_ws_bytes)?;
        // Keep offset for audit; Region lifetime is tied to arena
        // which lives as long as Bringup.
        let cublaslt_ws = HbmArenaCheckpoint {
            offset_bytes: (cublaslt_ws_region.device_ptr() - cublaslt_ws_region.device_ptr())
                as usize,
            bytes: cublaslt_ws_bytes,
        };

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
            fa3,
            policy,
            fused_modules,
        })
    }

    /// Resolve a GEMM plan for a (M, N, K, dtype) shape. Missing plan
    /// in the policy is a typed AutotuneCacheMiss; the engine refuses
    /// to run that shape.
    pub fn plan(&self, m: u32, n: u32, k: u32) -> Result<Fp8GemmPlan> {
        Fp8GemmPlan::from_policy(&self.policy, m, n, k, rvllm_core::DType::Fp8E4M3)
    }

    /// Run `iters` decode steps against a batch of `num_seqs` fake seqs
    /// (one token each). Returns elapsed nanoseconds.
    ///
    /// The path here is eager (no graph capture): reach for a bench
    /// number, then add the graph capture optimization.
    ///
    /// # Safety
    /// Uses raw device pointers from the arena. The `Bringup` owns the
    /// arena + stream for this function's duration so pointers stay
    /// valid.
    #[cfg(feature = "cuda")]
    pub unsafe fn run_bench(&self, num_seqs: u32, iters: u32, warmup: u32) -> Result<BenchResult> {
        self.run_bench_with_variants(num_seqs, iters, warmup, None, None)
    }

    /// Same as run_bench but with optional variant overrides. When
    /// `nonres_override` is `Some(v)`, every non-residual plan uses
    /// variant v regardless of the policy; same for `res_override` and
    /// residual plans. Lets a caller sweep variants without reloading
    /// weights.
    #[cfg(feature = "cuda")]
    pub unsafe fn run_bench_with_variants(
        &self,
        num_seqs: u32,
        iters: u32,
        warmup: u32,
        nonres_override: Option<u32>,
        res_override: Option<u32>,
    ) -> Result<BenchResult> {
        let _ = (nonres_override, res_override);
        self.run_bench_internal(num_seqs, iters, warmup, nonres_override, res_override)
    }

    #[cfg(feature = "cuda")]
    unsafe fn run_bench_internal(
        &self,
        num_seqs: u32,
        iters: u32,
        warmup: u32,
        nonres_override: Option<u32>,
        res_override: Option<u32>,
    ) -> Result<BenchResult> {
        let skip_lm_head = std::env::var("RVLLM_SKIP_LM_HEAD").ok().as_deref() == Some("1");
        use crate::layer_exec;
        use rvllm_cutlass::Fp8GemmPlan;
        use rvllm_fused::require_multiple;

        let arch = &self.arch;
        let hidden = arch.hidden_size as u32;
        let head_dim = arch.head_dim as u32;
        let nh = arch.num_attention_heads as u32;
        let nkvh = arch.num_key_value_heads as u32;
        let inter = arch.intermediate_size as u32;
        let q_dim = nh * head_dim;
        let kv_dim = nkvh * head_dim;
        let qkv_rows = (nh + 2 * nkvh) * head_dim;
        require_multiple(hidden as usize, 8, "hidden")?;

        // Optional real-prefill phase: when RVLLM_REAL_PREFILL=1 we run
        // one multi-query FA3 prefill over `prefill_len` tokens per seq
        // before the decode loop, instead of 16 eager decode steps.
        // Scratch must fit max(num_seqs, num_seqs * prefill_len) tokens.
        let real_prefill: bool = std::env::var("RVLLM_REAL_PREFILL").ok().as_deref() == Some("1");
        let prefill_len: u32 = std::env::var("RVLLM_PREFILL_LEN")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(16);
        let max_tokens: u32 = if real_prefill {
            num_seqs * prefill_len
        } else {
            num_seqs
        };

        // --- scratch allocations (sized for max_tokens) ------------------
        let arena = &self.arena;
        let hidden_fp8 = arena.region("hidden_fp8", (max_tokens * hidden) as usize, 16)?;
        let hidden_scale = arena.region("hidden_scale", (max_tokens * 4) as usize, 16)?;
        // Packed QKV output. cuBLASLt writes in col-major [N, M] which
        // is physical layout "all Q heads over all M tokens, then all K
        // heads, then all V heads". So k_base = q_base + (num_tokens *
        // q_dim * 2 bytes) and v_base = k_base + (num_tokens * kv_dim *
        // 2 bytes). num_tokens here is max_tokens (the allocation
        // ceiling); the effective per-call offset depends on the phase
        // and is computed by the caller.
        let qkv_out_bytes = (max_tokens * qkv_rows * 2) as usize;
        let qkv_out = arena.region("qkv_out", qkv_out_bytes, 16)?;
        let q_base = qkv_out.device_ptr();
        // For decode and prefill alike, the offsets depend on the
        // GEMM's M dim (num_tokens). At decode num_tokens = num_seqs;
        // at prefill num_tokens = num_seqs * prefill_len. We precompute
        // both sets of offsets so the same scratch region serves both.
        let k_base_decode = q_base + (num_seqs as u64) * (q_dim as u64) * 2;
        let v_base_decode = k_base_decode + (num_seqs as u64) * (kv_dim as u64) * 2;
        let k_base_prefill = q_base + (max_tokens as u64) * (q_dim as u64) * 2;
        let v_base_prefill = k_base_prefill + (max_tokens as u64) * (kv_dim as u64) * 2;
        let attn_out = arena.region("attn_out", (max_tokens * q_dim * 2) as usize, 16)?;
        let attn_out_fp8 = arena.region("attn_out_fp8", (max_tokens * q_dim) as usize, 16)?;
        let attn_out_scale = arena.region("attn_out_scale", (max_tokens * 4) as usize, 16)?;
        let gate_up_out = arena.region("gate_up_out", (max_tokens * 2 * inter * 2) as usize, 16)?;
        let gate_up_fp8 = arena.region("gate_up_fp8", (max_tokens * 2 * inter) as usize, 16)?;
        let gate_up_scale = arena.region("gate_up_scale", (max_tokens * 4) as usize, 16)?;
        let mlp_out_fp8 = arena.region("mlp_out_fp8", (max_tokens * inter) as usize, 16)?;
        let mlp_out_scale = arena.region("mlp_out_scale", (max_tokens * 4) as usize, 16)?;

        let num_blocks_total: u32 = std::env::var("RVLLM_NUM_BLOCKS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1024);
        // FA3 paged_decode block_size (tokens per KV page). Default 64;
        // sweepable via RVLLM_BLOCK_SIZE to test 32 / 128 / 256.
        let block_size: u32 = std::env::var("RVLLM_BLOCK_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(32);
        // max_blocks_per_seq × num_seqs must stay within num_blocks_total
        // so every block_tables entry points at a real page. At 1024
        // pages and N=512, that caps max_blocks_per_seq at 2. Clamp so
        // bigger batches don't overflow and cause FA3 illegal-access.
        let max_blocks_per_seq: u32 = (num_blocks_total / num_seqs).max(1);
        // FP8 E4M3 KV: 1 byte/element (was 2 for f16). Halves KV memory
        // and doubles HBM-bandwidth efficiency on attention reads.
        let kv_per_layer = 2 * num_blocks_total * block_size * nkvh * head_dim;
        let kv_cache = arena.region(
            "kv_cache",
            (arch.num_hidden_layers as u64 * kv_per_layer as u64) as usize,
            256,
        )?;
        // FP8 Q scratch (post-rope) consumed by FA3; f16 Q from QKV GEMM
        // still lives at q_out (2 bytes/elem). Sized by max_tokens so
        // prefill (num_tokens = num_seqs * prefill_len) doesn't overflow.
        let q_fp8 = arena.region("q_fp8", (max_tokens * q_dim) as usize, 16)?;
        // Per-tensor FP8 E4M3 scales for Q and KV quantization in the
        // fused_rope_cache_fp8kv kernel. Convention:
        //   scale = absmax / 448  (the E4M3 representable max)
        //   kernel quantizes:  fp8 = float * (1/scale) = float * (448/absmax)
        //   FA3 dequantizes:   float = fp8 * scale     = fp8 * (absmax/448)
        //
        // The previous placeholder (1/448 = assuming absmax=1.0) clipped any
        // activation outside [-1, 1] — destroying ~80% of dynamic range for
        // typical post-RoPE K/V values (which are in [-8, 8] for Qwen2.5-7B).
        //
        // Calibrated from a real forward pass on Qwen2.5-7B-Instruct (28
        // layers, 128-token prompt). Worst-case per-layer absmax:
        //   K: 418.0 (layer 27)  ->  kv_scale = 418/448 = 0.933
        //   V:  73.5 (layer 27)  ->  v_scale  = 73.5/448 = 0.164
        // Using the K max as the shared scale (the rope kernel quantizes
        // both K and V with the same scale pointer). V loses ~2.5 bits of
        // effective precision but nothing clips. Per-layer or split K/V
        // scales are future work.
        // Override via RVLLM_KV_SCALE_ABSMAX for other models.
        let kv_absmax: f32 = std::env::var("RVLLM_KV_SCALE_ABSMAX")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(418.0f32);
        let q_scale_region = arena.region("q_scale", 4, 4)?;
        let kv_scale_region = arena.region("kv_scale", 4, 4)?;
        {
            let scale: f32 = kv_absmax / 448.0f32;
            q_scale_region.copy_from_host(&scale.to_le_bytes())?;
            kv_scale_region.copy_from_host(&scale.to_le_bytes())?;
        }

        let cutlass_ws_bytes: usize = 16 * 1024 * 1024;
        let cutlass_ws = arena.region("cutlass_ws", cutlass_ws_bytes, 256)?;
        let fa3_ws = arena.region("fa3_ws", 16 * 1024 * 1024, 256)?;

        let residual = arena.region("residual", (max_tokens * hidden * 2) as usize, 16)?;
        // Zero the residual so layer 0's rmsnorm doesn't read stale NaN
        // from a prior process's HBM allocation.
        #[cfg(feature = "cuda")]
        {
            use cudarc::driver::sys::*;
            unsafe {
                cuMemsetD8_v2(residual.device_ptr(), 0, (max_tokens * hidden * 2) as usize);
            }
        }

        // Metadata: populate with valid decode-step values so FA3 walks
        // real KV pages instead of reading garbage.
        //   positions[i]   = i % max_pos
        //   slot_mapping[i]= i                 (writes into first slot)
        //   context_lens[i]= 1                 (one live token)
        //   block_tables[i][b] = i * max_blocks_per_seq + b
        let positions = arena.region("positions", (max_tokens * 4) as usize, 16)?;
        let slot_mapping = arena.region("slot_mapping", (max_tokens * 4) as usize, 16)?;
        let context_lens = arena.region("context_lens", (num_seqs * 4) as usize, 16)?;
        let block_tables = arena.region(
            "block_tables",
            (num_seqs * max_blocks_per_seq * 4) as usize,
            16,
        )?;
        {
            let n = num_seqs as usize;
            let pos_host: Vec<i32> = (0..n as i32).collect();
            let slot_host: Vec<i32> = (0..n as i32).collect();
            let ctx_host: Vec<i32> = vec![1; n];
            let mut bt_host: Vec<i32> = Vec::with_capacity(n * max_blocks_per_seq as usize);
            for i in 0..n as i32 {
                for b in 0..max_blocks_per_seq as i32 {
                    bt_host.push(i * max_blocks_per_seq as i32 + b);
                }
            }
            positions.copy_from_host(bytemuck_cast_i32(&pos_host))?;
            slot_mapping.copy_from_host(bytemuck_cast_i32(&slot_host))?;
            context_lens.copy_from_host(bytemuck_cast_i32(&ctx_host))?;
            block_tables.copy_from_host(bytemuck_cast_i32(&bt_host))?;
        }

        // Plans (from policy) for this specific bucket. Fused QKV uses
        // one GEMM with N = q_dim + 2*kv_dim = (heads + 2*kv_heads)*head_dim.
        let override_nonres = |mut p: Fp8GemmPlan| -> Fp8GemmPlan {
            if let Some(v) = nonres_override {
                p.variant = rvllm_cutlass::VariantId(v);
            }
            p
        };
        let override_res = |mut p: Fp8GemmPlan| -> Fp8GemmPlan {
            if let Some(v) = res_override {
                p.variant = rvllm_cutlass::VariantId(v);
            }
            p
        };
        let plan_qkv = override_nonres(Fp8GemmPlan::from_policy(
            &self.policy,
            num_seqs,
            qkv_rows,
            hidden,
            rvllm_core::DType::Fp8E4M3,
        )?);
        let plan_o = override_res(Fp8GemmPlan::from_policy_residual(
            &self.policy,
            num_seqs,
            hidden,
            q_dim,
            rvllm_core::DType::Fp8E4M3,
        )?);
        let plan_gate_up = override_nonres(Fp8GemmPlan::from_policy(
            &self.policy,
            num_seqs,
            2 * inter,
            hidden,
            rvllm_core::DType::Fp8E4M3,
        )?);
        let plan_down = override_res(Fp8GemmPlan::from_policy_residual(
            &self.policy,
            num_seqs,
            hidden,
            inter,
            rvllm_core::DType::Fp8E4M3,
        )?);
        let vocab = arch.vocab_size as u32;
        let _plan_lm_head = override_nonres(Fp8GemmPlan::from_policy(
            &self.policy,
            num_seqs,
            vocab,
            hidden,
            rvllm_core::DType::Fp8E4M3,
        )?);

        // LM head scratch: batch x vocab f16 logits + batch sampled tokens.
        let logits = arena.region("logits", (num_seqs * vocab * 2) as usize, 16)?;
        let sampled_tokens = arena.region("sampled_tokens", (num_seqs * 4) as usize, 16)?;

        let dims = layer_exec::LayerDims {
            num_tokens: num_seqs,
            hidden,
            num_heads: nh,
            num_kv_heads: nkvh,
            head_dim,
            intermediate: inter,
            block_size,
            max_blocks_per_seq,
            num_blocks_total,
            attn_scale: 1.0 / (head_dim as f32).sqrt(),
            rms_eps: arch.rms_norm_eps,
        };
        let kernels = layer_exec::LayerKernels {
            fused_rmsnorm: self.fused_modules.fn_rmsnorm,
            fused_add_rmsnorm: self.fused_modules.fn_add_rmsnorm,
            fused_rope_cache_fp8kv: self.fused_modules.fn_rope_cache_fp8kv,
            fused_silu_mul: self.fused_modules.fn_silu_mul,
            fused_gelu_mul: self.fused_modules.fn_gelu_mul,
            mlp_activation: self.arch.mlp_activation(),
            quantize_fp8_per_token: self.fused_modules.fn_quantize,
            add_bias_f16: self.fused_modules.fn_add_bias_f16,
        };
        let plans = layer_exec::LayerGemmPlans {
            qkv: plan_qkv,
            o: plan_o,
            gate_up: plan_gate_up,
            down: plan_down,
        };

        let stream = self.stream.raw();
        let residual_ptr = residual.device_ptr();

        let one_step = |phase: layer_exec::LayerPhase| -> Result<()> {
            let (layer_num_tokens, k_base_phase, v_base_phase) = match phase {
                layer_exec::LayerPhase::Decode => (num_seqs, k_base_decode, v_base_decode),
                layer_exec::LayerPhase::Prefill { max_seqlen_q, .. } => {
                    // total_q = num_seqs * max_seqlen_q (uniform prompt length)
                    (num_seqs * max_seqlen_q, k_base_prefill, v_base_prefill)
                }
            };
            let mut phase_dims = dims;
            phase_dims.num_tokens = layer_num_tokens;
            for (layer_idx, layer) in self.model.layers.iter().enumerate() {
                let layer_kv_base =
                    kv_cache.device_ptr() + (layer_idx as u64) * (kv_per_layer as u64);
                let w = layer_exec::LayerWeights {
                    attn_norm_gamma: layer.input_layernorm.offset_bytes,
                    qkv_fp8: layer.qkv.offset_bytes,
                    qkv_scale: layer.qkv.scale_ptr,
                    qkv_bias: layer.qkv_bias.as_ref().map_or(0, |b| b.offset_bytes),
                    o_fp8: layer.o_proj.offset_bytes,
                    o_scale: layer.o_proj.scale_ptr,
                    mlp_norm_gamma: layer.post_attention_layernorm.offset_bytes,
                    gate_up_fp8: layer.gate_up.offset_bytes,
                    gate_up_scale: layer.gate_up.scale_ptr,
                    down_fp8: layer.down_proj.offset_bytes,
                    down_scale: layer.down_proj.scale_ptr,
                };
                let scratch = layer_exec::LayerScratch {
                    hidden_fp8: hidden_fp8.device_ptr(),
                    hidden_scale: hidden_scale.device_ptr(),
                    q_out: q_base,
                    k_out: k_base_phase,
                    v_out: v_base_phase,
                    q_fp8: q_fp8.device_ptr(),
                    k_cache: layer_kv_base,
                    v_cache: layer_kv_base + (kv_per_layer / 2) as u64,
                    q_scale_ptr: q_scale_region.device_ptr(),
                    kv_scale_ptr: kv_scale_region.device_ptr(),
                    attn_out: attn_out.device_ptr(),
                    attn_out_fp8: attn_out_fp8.device_ptr(),
                    attn_out_scale: attn_out_scale.device_ptr(),
                    gate_up_out: gate_up_out.device_ptr(),
                    gate_up_fp8: gate_up_fp8.device_ptr(),
                    gate_up_scale: gate_up_scale.device_ptr(),
                    mlp_out_fp8: mlp_out_fp8.device_ptr(),
                    mlp_out_scale: mlp_out_scale.device_ptr(),
                    cutlass_workspace: cutlass_ws.device_ptr(),
                    cutlass_workspace_bytes: cutlass_ws_bytes,
                    fa3_workspace: fa3_ws.device_ptr(),
                };
                let meta = layer_exec::MetadataPtrs {
                    positions: positions.device_ptr(),
                    slot_mapping: slot_mapping.device_ptr(),
                    cos: self.model.rope_cos.offset_bytes,
                    sin: self.model.rope_sin.offset_bytes,
                    block_tables: block_tables.device_ptr(),
                    context_lens: context_lens.device_ptr(),
                };
                layer_exec::forward_phase(
                    phase_dims,
                    &kernels,
                    &w,
                    &scratch,
                    &meta,
                    &plans,
                    &self.cutlass,
                    &self.cublaslt,
                    &self.fa3,
                    residual_ptr,
                    stream,
                    phase,
                    self.arch.layer_types[layer_idx],
                )?;

                // NaN diagnostic: read 8 f16 values from residual after each layer.
                #[cfg(feature = "cuda")]
                if std::env::var("RVLLM_NAN_CHECK").ok().as_deref() == Some("1") {
                    unsafe {
                        cudarc::driver::sys::cuStreamSynchronize(stream as _);
                        let mut sample = [0u16; 8];
                        cudarc::driver::sys::cuMemcpyDtoH_v2(
                            sample.as_mut_ptr() as *mut _,
                            residual_ptr,
                            16,
                        );
                        let nan_count = sample
                            .iter()
                            .filter(|&&v| (v & 0x7C00) == 0x7C00 && (v & 0x03FF) != 0)
                            .count();
                        let inf_count = sample
                            .iter()
                            .filter(|&&v| v == 0x7C00 || v == 0xFC00)
                            .count();
                        if nan_count > 0 || inf_count > 0 {
                            eprintln!("[NaN] layer {layer_idx}: {nan_count} NaN, {inf_count} Inf in residual[0..8] = {:04X?}", sample);
                        }
                    }
                }
            }
            // Skip LM head during prefill — we only care about first-token
            // sampling after the LAST token of each seq, which is a
            // separate post-prefill step the caller handles.
            let is_prefill = matches!(phase, layer_exec::LayerPhase::Prefill { .. });
            if !skip_lm_head && !is_prefill {
                // LM head tail: fused_rmsnorm_fp8_quant applies the
                // model.norm weight AND produces FP8 hidden in one kernel.
                // Same kernel count as the previous quantize-only step,
                // but includes the final RMSnorm that Qwen2.5 requires —
                // fixes a correctness bug where we previously fed raw
                // residual to lm_head.
                rvllm_fused::FusedRmsnormFp8QuantLaunch {
                    num_tokens: num_seqs,
                    hidden,
                    eps: arch.rms_norm_eps,
                }
                .launch(
                    self.fused_modules.fn_rmsnorm,
                    hidden_fp8.device_ptr(),
                    hidden_scale.device_ptr(),
                    residual_ptr,
                    self.model.final_norm.offset_bytes,
                    stream,
                )?;
                #[cfg(feature = "cuda")]
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
                rvllm_fused::ArgmaxLaunch {
                    num_tokens: num_seqs,
                    vocab,
                }
                .launch(
                    self.fused_modules.fn_argmax,
                    logits.device_ptr(),
                    sampled_tokens.device_ptr(),
                    stream,
                )?;
            }
            Ok(())
        };

        // Eager warmup so any first-run kernel setup lands outside the graph.
        for _ in 0..warmup {
            one_step(layer_exec::LayerPhase::Decode)?;
        }
        self.stream.fence()?;

        // Faux-prefill: eager decode steps with advancing positions that
        // populate KV pages with real forward-pass activations before
        // the timed window. Default 16 matches vLLM's input_len=16.
        // Override via RVLLM_PREFILL to exercise longer-context regimes
        // (FP8 KV HBM-bandwidth win is context-length-sensitive).
        let faux_prefill_steps: i32 = std::env::var("RVLLM_PREFILL")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(16);
        #[allow(non_snake_case)]
        let FAUX_PREFILL_STEPS = faux_prefill_steps;
        let measure_ttft: bool = std::env::var("RVLLM_TTFT").ok().as_deref() == Some("1");

        // Pinned host buffer so we can DtoH the sampled tokens without
        // implicit host-side blocking (needed for a tight TTFT reading).
        let mut ttft_host_buf: rvllm_mem::PinnedBuf<i32> =
            rvllm_mem::PinnedBuf::new(num_seqs as usize)?;
        let n = num_seqs as usize;

        let set_step_meta = |step: i32| -> Result<()> {
            let pos_host: Vec<i32> = (0..n as i32).map(|i| step + i * 32).collect();
            let slot_host: Vec<i32> = (0..n as i32)
                .map(|i| step + i * max_blocks_per_seq as i32 * block_size as i32)
                .collect();
            let ctx_host: Vec<i32> = vec![step + 1; n];
            positions.copy_from_host(bytemuck_cast_i32(&pos_host))?;
            slot_mapping.copy_from_host(bytemuck_cast_i32(&slot_host))?;
            context_lens.copy_from_host(bytemuck_cast_i32(&ctx_host))?;
            Ok(())
        };

        // Real-prefill metadata setup: when RVLLM_REAL_PREFILL=1, we run
        // ONE multi-query FA3 prefill over `prefill_len` tokens per seq
        // (total_q = num_seqs * prefill_len). Populate cu_seqlens_q +
        // per-token positions + per-token slot_mapping + per-seq
        // context_lens = prefill_len.
        let cu_seqlens_q_region = if real_prefill {
            Some(arena.region("cu_seqlens_q", ((num_seqs as usize + 1) * 4) as usize, 16)?)
        } else {
            None
        };

        let run_real_prefill = |ttft_host: &mut rvllm_mem::PinnedBuf<i32>| -> Result<()> {
            let l = prefill_len as i32;
            let total = n * l as usize;
            let pos_host: Vec<i32> = (0..n as i32)
                .flat_map(|_seq| (0..l).collect::<Vec<_>>())
                .collect();
            let slot_host: Vec<i32> = (0..n as i32)
                .flat_map(|seq| {
                    (0..l)
                        .map(|t| seq * max_blocks_per_seq as i32 * block_size as i32 + t)
                        .collect::<Vec<_>>()
                })
                .collect();
            let ctx_host: Vec<i32> = vec![l; n];
            let cu_host: Vec<i32> = (0..=n as i32).map(|i| i * l).collect();
            positions.copy_from_host(bytemuck_cast_i32(&pos_host[..total]))?;
            slot_mapping.copy_from_host(bytemuck_cast_i32(&slot_host[..total]))?;
            context_lens.copy_from_host(bytemuck_cast_i32(&ctx_host))?;
            if let Some(r) = &cu_seqlens_q_region {
                r.copy_from_host(bytemuck_cast_i32(&cu_host))?;
            }
            // One prefill forward pass (all 28 layers).
            let phase = layer_exec::LayerPhase::Prefill {
                cu_seqlens_q: cu_seqlens_q_region
                    .as_ref()
                    .map(|r| r.device_ptr())
                    .unwrap_or(0),
                max_seqlen_q: prefill_len,
            };
            one_step(phase)?;
            // Reset metadata to decode shape for the follow-on decode loop
            // (sequences now have prefill_len tokens cached).
            let _ = ttft_host; // silence on non-ttft path
            Ok(())
        };

        // TTFT: two timed passes.
        //   "cold" = first prefill call from a fresh process. Includes
        //            cuBLASLt heuristic cost for any shape never seen
        //            (prefill M = num_seqs × prompt_len differs from
        //            decode M = num_seqs, so heuristics are cold).
        //   "hot"  = second prefill call (heuristics cached). Represents
        //            per-request TTFT a real deployment would see.
        // When TTFT isn't requested we still need one prefill to populate
        // KV with real activations before the timed decode window.
        let sampled_d_ptr = sampled_tokens.device_ptr();
        let (ttft_ns, ttft_hot_ns): (Option<u128>, Option<u128>) = if measure_ttft {
            // --- cold pass (timed) ---
            self.stream.fence()?;
            let t_cold = std::time::Instant::now();
            if real_prefill {
                run_real_prefill(&mut ttft_host_buf)?;
            } else {
                for step in 0..FAUX_PREFILL_STEPS {
                    set_step_meta(step)?;
                    one_step(layer_exec::LayerPhase::Decode)?;
                }
            }
            self.stream.fence()?;
            dtoh_async_sync(sampled_d_ptr, ttft_host_buf.as_mut_ptr(), n * 4, stream)?;
            self.stream.fence()?;
            let cold_ns = t_cold.elapsed().as_nanos();

            // --- hot pass (timed) — repeat the same prefill, heuristics
            // now cached. Mirrors per-request TTFT under steady load.
            self.stream.fence()?;
            let t_hot = std::time::Instant::now();
            if real_prefill {
                run_real_prefill(&mut ttft_host_buf)?;
            } else {
                for step in 0..FAUX_PREFILL_STEPS {
                    set_step_meta(step)?;
                    one_step(layer_exec::LayerPhase::Decode)?;
                }
            }
            self.stream.fence()?;
            dtoh_async_sync(sampled_d_ptr, ttft_host_buf.as_mut_ptr(), n * 4, stream)?;
            self.stream.fence()?;
            let hot_ns = t_hot.elapsed().as_nanos();

            (Some(cold_ns), Some(hot_ns))
        } else {
            if real_prefill {
                run_real_prefill(&mut ttft_host_buf)?;
            } else {
                for step in 0..FAUX_PREFILL_STEPS {
                    set_step_meta(step)?;
                    one_step(layer_exec::LayerPhase::Decode)?;
                }
            }
            self.stream.fence()?;
            (None, None)
        };

        let one_step = one_step;
        let no_graph = std::env::var("RVLLM_NO_GRAPH").ok().as_deref() == Some("1");

        let elapsed = if no_graph {
            self.stream.fence()?;
            let t0 = std::time::Instant::now();
            for iter in 0..iters as i32 {
                set_step_meta(FAUX_PREFILL_STEPS + iter)?;
                one_step(layer_exec::LayerPhase::Decode)?;
            }
            self.stream.fence()?;
            t0.elapsed()
        } else {
            let graph = rvllm_graph::CapturedGraph::capture(
                num_seqs,
                max_blocks_per_seq,
                rvllm_metadata::MetadataLayout::compute(num_seqs, max_blocks_per_seq).hash(),
                rvllm_graph::GraphFingerprint([0u8; 32]),
                stream,
                || one_step(layer_exec::LayerPhase::Decode),
            )?;
            self.stream.fence()?;
            let t0 = std::time::Instant::now();
            for iter in 0..iters as i32 {
                set_step_meta(FAUX_PREFILL_STEPS + iter)?;
                graph.replay(stream)?;
            }
            self.stream.fence()?;
            t0.elapsed()
        };

        // Debug: dump sampled token IDs to stderr for quality sanity check.
        if std::env::var("RVLLM_DUMP_TOKENS").ok().as_deref() == Some("1") {
            dtoh_async_sync(sampled_d_ptr, ttft_host_buf.as_mut_ptr(), n * 4, stream)?;
            self.stream.fence()?;
            let ids: &[i32] = ttft_host_buf.as_slice();
            let show = ids.len().min(16);
            eprintln!("[TOKENS] sampled_ids[0..{show}] = {:?}", &ids[..show]);
        }
        // Debug: dump logits values.
        if std::env::var("RVLLM_DUMP_LOGITS").ok().as_deref() == Some("1") {
            let logits_elems = (num_seqs * vocab) as usize;
            let logits_bytes = logits_elems * 2;
            let mut logits_host: Vec<u16> = vec![0u16; logits_elems];
            dtoh_async_sync(
                logits.device_ptr(),
                logits_host.as_mut_ptr() as *mut i32,
                logits_bytes,
                stream,
            )?;
            self.stream.fence()?;
            let first5: Vec<f32> = logits_host[..5].iter().map(|&b| f16_to_f32(b)).collect();
            let nan_count = logits_host
                .iter()
                .filter(|&&b| f16_to_f32(b).is_nan())
                .count();
            eprintln!(
                "[LOGITS] nan={}/{} first5={:?}",
                nan_count, logits_elems, &first5
            );
        }

        Ok(BenchResult {
            ns_per_step: elapsed.as_nanos() / iters.max(1) as u128,
            total_ns: elapsed.as_nanos(),
            iters,
            num_seqs,
            ttft_ns,
            ttft_hot_ns,
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub unsafe fn run_bench(&self, num_seqs: u32, iters: u32, _warmup: u32) -> Result<BenchResult> {
        Ok(BenchResult {
            ns_per_step: 0,
            total_ns: 0,
            iters,
            num_seqs,
            ttft_ns: None,
            ttft_hot_ns: None,
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub unsafe fn run_bench_with_variants(
        &self,
        num_seqs: u32,
        iters: u32,
        _warmup: u32,
        _nonres_override: Option<u32>,
        _res_override: Option<u32>,
    ) -> Result<BenchResult> {
        Ok(BenchResult {
            ns_per_step: 0,
            total_ns: 0,
            iters,
            num_seqs,
            ttft_ns: None,
            ttft_hot_ns: None,
        })
    }

    /// Run perplexity evaluation over `token_ids` in decode mode.
    /// Processes one token at a time, runs LM head to get logits,
    /// DtoH one row of logits (vocab x f16) per step, computes
    /// cross-entropy on host. Returns (total_nll, num_evaluated_tokens).
    ///
    /// Uses the same arena allocation order as `run_bench_internal`
    /// so all FA3 pointers land at proven-good addresses.
    #[cfg(feature = "cuda")]
    pub unsafe fn run_ppl(
        &self,
        fn_embed: rvllm_kernels::KernelFn,
        token_ids: &[u32],
    ) -> Result<PplResult> {
        use crate::layer_exec;
        use rvllm_cutlass::Fp8GemmPlan;
        use rvllm_fused::require_multiple;

        let arch = &self.arch;
        let hidden = arch.hidden_size as u32;
        let head_dim = arch.head_dim as u32;
        let nh = arch.num_attention_heads as u32;
        let nkvh = arch.num_key_value_heads as u32;
        let inter = arch.intermediate_size as u32;
        let q_dim = nh * head_dim;
        let kv_dim = nkvh * head_dim;
        let qkv_rows = (nh + 2 * nkvh) * head_dim;
        require_multiple(hidden as usize, 8, "hidden")?;

        let num_seqs: u32 = 1;
        let max_tokens: u32 = num_seqs;
        let block_size: u32 = std::env::var("RVLLM_BLOCK_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(32);
        let num_blocks_total: u32 = std::env::var("RVLLM_NUM_BLOCKS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1024);
        let max_blocks_per_seq: u32 = (num_blocks_total / num_seqs).max(1);
        let kv_per_layer = 2 * num_blocks_total * block_size * nkvh * head_dim;

        // --- scratch allocations (SAME ORDER as run_bench_internal) ---
        let arena = &self.arena;
        let hidden_fp8 = arena.region("hidden_fp8", (max_tokens * hidden) as usize, 16)?;
        let hidden_scale = arena.region("hidden_scale", (max_tokens * 4) as usize, 16)?;
        let qkv_out_bytes = (max_tokens * qkv_rows * 2) as usize;
        let qkv_out = arena.region("qkv_out", qkv_out_bytes, 16)?;
        let q_base = qkv_out.device_ptr();
        let k_base_decode = q_base + (num_seqs as u64) * (q_dim as u64) * 2;
        let v_base_decode = k_base_decode + (num_seqs as u64) * (kv_dim as u64) * 2;
        let attn_out = arena.region("attn_out", (max_tokens * q_dim * 2) as usize, 16)?;
        let attn_out_fp8 = arena.region("attn_out_fp8", (max_tokens * q_dim) as usize, 16)?;
        let attn_out_scale = arena.region("attn_out_scale", (max_tokens * 4) as usize, 16)?;
        let gate_up_out = arena.region("gate_up_out", (max_tokens * 2 * inter * 2) as usize, 16)?;
        let gate_up_fp8 = arena.region("gate_up_fp8", (max_tokens * 2 * inter) as usize, 16)?;
        let gate_up_scale = arena.region("gate_up_scale", (max_tokens * 4) as usize, 16)?;
        let mlp_out_fp8 = arena.region("mlp_out_fp8", (max_tokens * inter) as usize, 16)?;
        let mlp_out_scale = arena.region("mlp_out_scale", (max_tokens * 4) as usize, 16)?;

        let kv_cache = arena.region(
            "kv_cache",
            (arch.num_hidden_layers as u64 * kv_per_layer as u64) as usize,
            256,
        )?;
        let q_fp8 = arena.region("q_fp8", (max_tokens * q_dim) as usize, 16)?;
        let kv_absmax: f32 = std::env::var("RVLLM_KV_SCALE_ABSMAX")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(418.0f32);
        let q_scale_region = arena.region("q_scale", 4, 4)?;
        let kv_scale_region = arena.region("kv_scale", 4, 4)?;
        {
            let scale: f32 = kv_absmax / 448.0f32;
            q_scale_region.copy_from_host(&scale.to_le_bytes())?;
            kv_scale_region.copy_from_host(&scale.to_le_bytes())?;
        }

        let cutlass_ws_bytes: usize = 16 * 1024 * 1024;
        let cutlass_ws = arena.region("cutlass_ws", cutlass_ws_bytes, 256)?;
        let fa3_ws = arena.region("fa3_ws", 16 * 1024 * 1024, 256)?;

        // Metadata BEFORE residual so embedding_gather can't corrupt them.
        let positions = arena.region("positions", (max_tokens * 4) as usize, 16)?;
        let slot_mapping = arena.region("slot_mapping", (max_tokens * 4) as usize, 16)?;
        let context_lens = arena.region("context_lens", (num_seqs * 4) as usize, 16)?;
        let block_tables = arena.region(
            "block_tables",
            (num_seqs * max_blocks_per_seq * 4) as usize,
            16,
        )?;

        let residual = arena.region("residual", (max_tokens * hidden * 2) as usize, 16)?;
        {
            let n = num_seqs as usize;
            let pos_host: Vec<i32> = (0..n as i32).collect();
            let slot_host: Vec<i32> = (0..n as i32).collect();
            let ctx_host: Vec<i32> = vec![1; n];
            let mut bt_host: Vec<i32> = Vec::with_capacity(n * max_blocks_per_seq as usize);
            for i in 0..n as i32 {
                for b in 0..max_blocks_per_seq as i32 {
                    bt_host.push(i * max_blocks_per_seq as i32 + b);
                }
            }
            positions.copy_from_host(bytemuck_cast_i32(&pos_host))?;
            slot_mapping.copy_from_host(bytemuck_cast_i32(&slot_host))?;
            context_lens.copy_from_host(bytemuck_cast_i32(&ctx_host))?;
            block_tables.copy_from_host(bytemuck_cast_i32(&bt_host))?;
        }

        let plan_qkv = Fp8GemmPlan::from_policy(
            &self.policy,
            num_seqs,
            qkv_rows,
            hidden,
            rvllm_core::DType::Fp8E4M3,
        )?;
        let plan_o = Fp8GemmPlan::from_policy_residual(
            &self.policy,
            num_seqs,
            hidden,
            q_dim,
            rvllm_core::DType::Fp8E4M3,
        )?;
        let plan_gate_up = Fp8GemmPlan::from_policy(
            &self.policy,
            num_seqs,
            2 * inter,
            hidden,
            rvllm_core::DType::Fp8E4M3,
        )?;
        let plan_down = Fp8GemmPlan::from_policy_residual(
            &self.policy,
            num_seqs,
            hidden,
            inter,
            rvllm_core::DType::Fp8E4M3,
        )?;
        let vocab = arch.vocab_size as u32;

        let logits = arena.region("logits", (num_seqs * vocab * 2) as usize, 16)?;
        // token_ids upload region
        let token_ids_region = arena.region("token_ids_ppl", (num_seqs * 4) as usize, 16)?;

        let dims = layer_exec::LayerDims {
            num_tokens: num_seqs,
            hidden,
            num_heads: nh,
            num_kv_heads: nkvh,
            head_dim,
            intermediate: inter,
            block_size,
            max_blocks_per_seq,
            num_blocks_total,
            attn_scale: 1.0 / (head_dim as f32).sqrt(),
            rms_eps: arch.rms_norm_eps,
        };
        let kernels = layer_exec::LayerKernels {
            fused_rmsnorm: self.fused_modules.fn_rmsnorm,
            fused_add_rmsnorm: self.fused_modules.fn_add_rmsnorm,
            fused_rope_cache_fp8kv: self.fused_modules.fn_rope_cache_fp8kv,
            fused_silu_mul: self.fused_modules.fn_silu_mul,
            fused_gelu_mul: self.fused_modules.fn_gelu_mul,
            mlp_activation: self.arch.mlp_activation(),
            quantize_fp8_per_token: self.fused_modules.fn_quantize,
            add_bias_f16: self.fused_modules.fn_add_bias_f16,
        };
        let plans = layer_exec::LayerGemmPlans {
            qkv: plan_qkv,
            o: plan_o,
            gate_up: plan_gate_up,
            down: plan_down,
        };

        let stream = self.stream.raw();
        let residual_ptr = residual.device_ptr();
        let step_counter = std::cell::Cell::new(0u32);

        let one_step = |_phase: layer_exec::LayerPhase| -> Result<()> {
            for (layer_idx, layer) in self.model.layers.iter().enumerate() {
                let layer_kv_base =
                    kv_cache.device_ptr() + (layer_idx as u64) * (kv_per_layer as u64);
                let w = layer_exec::LayerWeights {
                    attn_norm_gamma: layer.input_layernorm.offset_bytes,
                    qkv_fp8: layer.qkv.offset_bytes,
                    qkv_scale: layer.qkv.scale_ptr,
                    qkv_bias: layer.qkv_bias.as_ref().map_or(0, |b| b.offset_bytes),
                    o_fp8: layer.o_proj.offset_bytes,
                    o_scale: layer.o_proj.scale_ptr,
                    mlp_norm_gamma: layer.post_attention_layernorm.offset_bytes,
                    gate_up_fp8: layer.gate_up.offset_bytes,
                    gate_up_scale: layer.gate_up.scale_ptr,
                    down_fp8: layer.down_proj.offset_bytes,
                    down_scale: layer.down_proj.scale_ptr,
                };
                let scratch = layer_exec::LayerScratch {
                    hidden_fp8: hidden_fp8.device_ptr(),
                    hidden_scale: hidden_scale.device_ptr(),
                    q_out: q_base,
                    k_out: k_base_decode,
                    v_out: v_base_decode,
                    q_fp8: q_fp8.device_ptr(),
                    k_cache: layer_kv_base,
                    v_cache: layer_kv_base + (kv_per_layer / 2) as u64,
                    q_scale_ptr: q_scale_region.device_ptr(),
                    kv_scale_ptr: kv_scale_region.device_ptr(),
                    attn_out: attn_out.device_ptr(),
                    attn_out_fp8: attn_out_fp8.device_ptr(),
                    attn_out_scale: attn_out_scale.device_ptr(),
                    gate_up_out: gate_up_out.device_ptr(),
                    gate_up_fp8: gate_up_fp8.device_ptr(),
                    gate_up_scale: gate_up_scale.device_ptr(),
                    mlp_out_fp8: mlp_out_fp8.device_ptr(),
                    mlp_out_scale: mlp_out_scale.device_ptr(),
                    cutlass_workspace: cutlass_ws.device_ptr(),
                    cutlass_workspace_bytes: cutlass_ws_bytes,
                    fa3_workspace: fa3_ws.device_ptr(),
                };
                let meta = layer_exec::MetadataPtrs {
                    positions: positions.device_ptr(),
                    slot_mapping: slot_mapping.device_ptr(),
                    cos: self.model.rope_cos.offset_bytes,
                    sin: self.model.rope_sin.offset_bytes,
                    block_tables: block_tables.device_ptr(),
                    context_lens: context_lens.device_ptr(),
                };
                layer_exec::forward_phase(
                    dims,
                    &kernels,
                    &w,
                    &scratch,
                    &meta,
                    &plans,
                    &self.cutlass,
                    &self.cublaslt,
                    &self.fa3,
                    residual_ptr,
                    stream,
                    layer_exec::LayerPhase::Decode,
                    self.arch.layer_types[layer_idx],
                )?;
                #[cfg(feature = "cuda")]
                if step_counter.get() == 0 {
                    cudarc::driver::sys::cuStreamSynchronize(stream as _);
                    let mut s = [0u16; 4];
                    cudarc::driver::sys::cuMemcpyDtoH_v2(s.as_mut_ptr() as *mut _, residual_ptr, 8);
                    let v: Vec<f32> = s.iter().map(|&x| f16_to_f32(x)).collect();
                    eprintln!(
                        "  L{layer_idx}: [{:.4}, {:.4}, {:.4}, {:.4}]",
                        v[0], v[1], v[2], v[3]
                    );
                    if layer_idx == 0 {
                        // Read back the per-token activation scale and weight scales
                        let mut act_scale_val = [0f32; 1];
                        cudarc::driver::sys::cuMemcpyDtoH_v2(
                            act_scale_val.as_mut_ptr() as *mut _,
                            hidden_scale.device_ptr(),
                            4,
                        );
                        let mut wt_qkv_scale = [0f32; 1];
                        cudarc::driver::sys::cuMemcpyDtoH_v2(
                            wt_qkv_scale.as_mut_ptr() as *mut _,
                            layer.qkv.scale_ptr,
                            4,
                        );
                        let mut wt_o_scale = [0f32; 1];
                        cudarc::driver::sys::cuMemcpyDtoH_v2(
                            wt_o_scale.as_mut_ptr() as *mut _,
                            layer.o_proj.scale_ptr,
                            4,
                        );
                        let mut wt_gu_scale = [0f32; 1];
                        cudarc::driver::sys::cuMemcpyDtoH_v2(
                            wt_gu_scale.as_mut_ptr() as *mut _,
                            layer.gate_up.scale_ptr,
                            4,
                        );
                        let mut wt_dn_scale = [0f32; 1];
                        cudarc::driver::sys::cuMemcpyDtoH_v2(
                            wt_dn_scale.as_mut_ptr() as *mut _,
                            layer.down_proj.scale_ptr,
                            4,
                        );
                        eprintln!(
                            "  [DBG L0] act_scale(hidden)={:.6e} wt_qkv={:.6e} wt_o={:.6e} wt_gu={:.6e} wt_dn={:.6e}",
                            act_scale_val[0], wt_qkv_scale[0], wt_o_scale[0], wt_gu_scale[0], wt_dn_scale[0],
                        );
                        eprintln!(
                            "  [DBG L0] rust-side scales: qkv={:.6e} o={:.6e} gu={:.6e} dn={:.6e}",
                            layer.qkv.scale,
                            layer.o_proj.scale,
                            layer.gate_up.scale,
                            layer.down_proj.scale,
                        );
                    }
                }
            }
            step_counter.set(step_counter.get() + 1);
            // LM head (no argmax).
            rvllm_fused::FusedRmsnormFp8QuantLaunch {
                num_tokens: num_seqs,
                hidden,
                eps: 1e-6,
            }
            .launch(
                self.fused_modules.fn_rmsnorm,
                hidden_fp8.device_ptr(),
                hidden_scale.device_ptr(),
                residual_ptr,
                self.model.final_norm.offset_bytes,
                stream,
            )?;
            #[cfg(feature = "cuda")]
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
            Ok(())
        };

        let set_step_meta = |step: i32| -> Result<()> {
            let n = num_seqs as usize;
            let pos_host: Vec<i32> = (0..n as i32).map(|i| step + i * 32).collect();
            let slot_host: Vec<i32> = (0..n as i32)
                .map(|i| step + i * max_blocks_per_seq as i32 * block_size as i32)
                .collect();
            let ctx_host: Vec<i32> = vec![step + 1; n];
            positions.copy_from_host(bytemuck_cast_i32(&pos_host))?;
            slot_mapping.copy_from_host(bytemuck_cast_i32(&slot_host))?;
            context_lens.copy_from_host(bytemuck_cast_i32(&ctx_host))?;
            Ok(())
        };

        // Process token IDs starting at position 0.
        let logits_row_elems = vocab as usize;
        let logits_row_bytes = logits_row_elems * 2;
        let mut logits_host: Vec<u16> = vec![0u16; logits_row_elems];

        let mut total_nll: f64 = 0.0;
        let mut n_evaluated: usize = 0;

        for (t, &tok_id) in token_ids.iter().enumerate() {
            // Embed token.
            let tok_i32 = [tok_id as i32];
            token_ids_region.copy_from_host(bytemuck_cast_i32(&tok_i32))?;
            rvllm_fused::EmbeddingGatherLaunch {
                num_tokens: 1,
                hidden,
                vocab,
            }
            .launch(
                fn_embed,
                residual_ptr,
                self.model.embedding.offset_bytes,
                token_ids_region.device_ptr(),
                stream,
            )?;

            // Probe embedding before layer 0 on first token.
            #[cfg(feature = "cuda")]
            if t == 0 {
                unsafe {
                    cudarc::driver::sys::cuStreamSynchronize(stream as _);
                    let mut emb = [0u16; 8];
                    cudarc::driver::sys::cuMemcpyDtoH_v2(
                        emb.as_mut_ptr() as *mut _,
                        residual_ptr,
                        16,
                    );
                    let vals: Vec<f32> = emb.iter().map(|&x| f16_to_f32(x)).collect();
                    let mut amax = 0f32;
                    let n = hidden as usize;
                    let mut all_emb = vec![0u16; n];
                    cudarc::driver::sys::cuMemcpyDtoH_v2(
                        all_emb.as_mut_ptr() as *mut _,
                        residual_ptr,
                        (n * 2) as _,
                    );
                    for &v in &all_emb {
                        let f = f16_to_f32(v).abs();
                        if f > amax && !f.is_nan() {
                            amax = f;
                        }
                    }
                    eprintln!(
                        "  [DBG t=0] embed first8={:.4?} amax={:.4} expected_scale={:.6e}",
                        &vals[..4],
                        amax,
                        amax / 448.0,
                    );
                }
            }

            // Forward + LM head.
            set_step_meta(t as i32)?;
            one_step(layer_exec::LayerPhase::Decode)?;

            // DtoH logits and compute loss against next token.
            if t + 1 < token_ids.len() {
                dtoh_async_sync(
                    logits.device_ptr(),
                    logits_host.as_mut_ptr() as *mut i32,
                    logits_row_bytes,
                    stream,
                )?;
                self.stream.fence()?;

                let target = token_ids[t + 1] as usize;
                if t == 0 {
                    let first5: Vec<f32> =
                        logits_host[..5].iter().map(|&b| f16_to_f32(b)).collect();
                    let max_val = logits_host
                        .iter()
                        .map(|&b| f16_to_f32(b))
                        .filter(|v| !v.is_nan())
                        .fold(f32::MIN, f32::max);
                    let min_val = logits_host
                        .iter()
                        .map(|&b| f16_to_f32(b))
                        .filter(|v| !v.is_nan())
                        .fold(f32::MAX, f32::min);
                    let target_logit = f16_to_f32(logits_host[token_ids[t + 1] as usize]);
                    eprintln!(
                        "  [DBG] logits: first5={:?} min={:.1} max={:.1} target[{}]={:.1}",
                        first5,
                        min_val,
                        max_val,
                        token_ids[t + 1],
                        target_logit
                    );
                    let nan_count = logits_host
                        .iter()
                        .filter(|&&b| f16_to_f32(b).is_nan())
                        .count();
                    if nan_count > 0 {
                        eprintln!(
                            "  WARNING: {}/{} logits are NaN (first5={:?}). FP8 precision issue.",
                            nan_count, logits_row_elems, &first5
                        );
                    }
                }
                let nll = compute_nll_f16(&logits_host, target);
                total_nll += nll;
                n_evaluated += 1;

                if (t + 1) % 100 == 0 || t + 1 == token_ids.len() - 1 {
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
        Ok(PplResult {
            ppl,
            total_nll,
            n_evaluated,
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub unsafe fn run_ppl(
        &self,
        _fn_embed: rvllm_kernels::KernelFn,
        _token_ids: &[u32],
    ) -> Result<PplResult> {
        Ok(PplResult {
            ppl: 0.0,
            total_nll: 0.0,
            n_evaluated: 0,
        })
    }
}

#[derive(Clone, Debug)]
pub struct PplResult {
    pub ppl: f64,
    pub total_nll: f64,
    pub n_evaluated: usize,
}

#[derive(Copy, Clone, Debug)]
pub struct BenchResult {
    pub ns_per_step: u128,
    pub total_ns: u128,
    pub iters: u32,
    pub num_seqs: u32,
    /// Cold TTFT in ns: time from "prefill starts" → "first sampled
    /// token on host" on the first prefill call in this process.
    /// Includes cuBLASLt per-shape heuristic cost (one-time per engine).
    /// None if TTFT measurement was not requested.
    pub ttft_ns: Option<u128>,
    /// Hot TTFT in ns: same measurement on a second prefill call, with
    /// cuBLASLt algos already cached. Represents per-request TTFT under
    /// steady serving load. None if TTFT measurement was not requested.
    pub ttft_hot_ns: Option<u128>,
}

#[cfg(feature = "cuda")]
fn bytemuck_cast_i32(v: &[i32]) -> &[u8] {
    // SAFETY: i32 has a defined bit layout; we only read these bytes,
    // and the output slice is the same length/alignment.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

/// Async DtoH of `bytes` from device `src` to host `dst` on `stream`.
/// Caller must fence the stream afterwards to ensure the copy completes.
#[cfg(feature = "cuda")]
pub(crate) fn dtoh_async_sync(src: u64, dst: *mut i32, bytes: usize, stream: u64) -> Result<()> {
    use cudarc::driver::sys::*;
    let r = unsafe { cuMemcpyDtoHAsync_v2(dst as *mut _, src, bytes, stream as CUstream) };
    if r != CUresult::CUDA_SUCCESS {
        return Err(rvllm_core::RvllmError::cuda(
            "cuMemcpyDtoHAsync",
            rvllm_core::CudaErrorKind::Other,
            rvllm_core::CudaCtx::setup(),
        ));
    }
    Ok(())
}

#[cfg(feature = "cuda")]
pub(crate) fn compute_nll_f16(logits_f16: &[u16], target: usize) -> f64 {
    let mut max_val: f32 = f32::NEG_INFINITY;
    for &bits in logits_f16.iter() {
        let v = f16_to_f32(bits);
        if v > max_val {
            max_val = v;
        }
    }
    let mut sum_exp: f64 = 0.0;
    for &bits in logits_f16.iter() {
        sum_exp += ((f16_to_f32(bits) - max_val) as f64).exp();
    }
    let log_sum_exp = sum_exp.ln() + max_val as f64;
    let target_logit = f16_to_f32(logits_f16[target]) as f64;
    log_sum_exp - target_logit
}

#[cfg(feature = "cuda")]
pub(crate) fn compute_nll_f32(logits: &[f32], target: usize) -> f64 {
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f64 = logits.iter().map(|&v| ((v - max_val) as f64).exp()).sum();
    let log_sum_exp = sum_exp.ln() + max_val as f64;
    log_sum_exp - logits[target] as f64
}

#[cfg(feature = "cuda")]
#[inline(always)]
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;
    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign << 31);
        }
        let mut m = mant;
        let mut e: i32 = 1;
        while (m & 0x400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3ff;
        let f32_exp = (127 - 15 + e) as u32;
        return f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13));
    }
    if exp == 31 {
        return f32::from_bits((sign << 31) | (0xff_u32 << 23) | (mant << 13));
    }
    let f32_exp = (exp as i32 - 15 + 127) as u32;
    f32::from_bits((sign << 31) | (f32_exp << 23) | (mant << 13))
}

fn load_fused(loader: &KernelLoader) -> Result<FusedModules> {
    let rmsnorm_mod = loader.load_ptx("fused_rmsnorm_fp8_quant")?;
    let rope_mod = loader.load_ptx("fused_rope_cache_fp8kv")?;
    let silu_mod = loader.load_ptx("fused_silu_fp8_quant")?;
    let argmax_mod = loader.load_ptx("argmax")?;
    let add_bias_mod = loader.load_ptx("add_bias_f16")?;

    // GELU kernel is optional -- only needed for Gemma 4 and similar.
    // If the PTX isn't in the manifest, we get None and fall through
    // to a hard error at dispatch time if the model actually needs it.
    let (gelu_mod, fn_gelu_mul) = match loader.load_ptx("fused_gelu_mul_fp8_quant") {
        Ok(m) => {
            let f = m.get_function("fused_gelu_mul_fp8_quant_kernel")?;
            (Some(m), Some(f))
        }
        Err(_) => (None, None),
    };

    let fn_rmsnorm = rmsnorm_mod.get_function("fused_rmsnorm_fp8_quant_kernel")?;
    let fn_add_rmsnorm = rmsnorm_mod.get_function("fused_add_rmsnorm_fp8_quant_kernel")?;
    let fn_quantize = rmsnorm_mod.get_function("quantize_fp8_per_token_kernel")?;
    let fn_rope_cache_fp8kv = rope_mod.get_function("fused_rope_cache_fp8kv_kernel")?;
    let fn_silu_mul = silu_mod.get_function("fused_silu_mul_fp8_quant_kernel")?;
    let fn_argmax = argmax_mod.get_function("argmax_kernel")?;
    let fn_add_bias_f16 = add_bias_mod.get_function("add_bias_f16_kernel")?;

    Ok(FusedModules {
        rmsnorm_mod,
        rope_mod,
        silu_mod,
        gelu_mod,
        argmax_mod,
        add_bias_mod,
        fn_rmsnorm,
        fn_add_rmsnorm,
        fn_quantize,
        fn_rope_cache_fp8kv,
        fn_silu_mul,
        fn_gelu_mul,
        fn_argmax,
        fn_add_bias_f16,
    })
}

/// Resolve `<kernels_root>/<sm_xxx>/` for the CUDA device backing `ctx`.
///
/// Queries the device's compute capability, maps it to a `CompileTarget`,
/// and returns the matching per-arch subdirectory. Rejects devices whose
/// compute capability is not in our PTX build matrix and rejects missing
/// subdirectories (no silent fallback to the legacy top-level layout).
///
/// Under `not(feature = "cuda")` this falls back to `kernels_root` as-is
/// so host-stub builds still compile.
pub fn resolve_kernels_dir(
    ctx: &CudaContextHandle,
    kernels_root: &std::path::Path,
) -> Result<std::path::PathBuf> {
    // Under `not(feature = "cuda")` we have no device to query, so
    // fall through to the root directory unchanged (host-stub builds
    // run invariant-level tests only — they never load kernels).
    #[cfg(not(feature = "cuda"))]
    {
        let _ = ctx;
        return Ok(kernels_root.to_path_buf());
    }

    #[cfg(feature = "cuda")]
    {
        let (major, minor) = ctx.compute_capability();
        let target = CompileTarget::from_compute_capability(major, minor).ok_or_else(|| {
            RvllmError::config(
                ConfigError::InvalidField {
                    name: "compute_capability",
                    reason: format!(
                        "unsupported CUDA compute capability {major}.{minor} \
                         (no PTX build in kernels/). Add a `CompileTarget` \
                         variant and rebuild kernels for this arch."
                    ),
                },
                "compute_capability",
            )
        })?;
        let sub = kernels_root.join(target.as_sm_str());
        if !sub.is_dir() {
            return Err(RvllmError::config(
                ConfigError::InvalidField {
                    name: "kernels_dir",
                    reason: format!(
                        "kernel subdirectory {} for compute capability {major}.{minor} \
                         does not exist; run `kernels/build.sh {}`",
                        sub.display(),
                        target.as_sm_str(),
                    ),
                },
                "kernels_dir",
            ));
        }
        Ok(sub)
    }
}
