//! Engine bring-up: assemble every subsystem from paths on disk.
//!
//! This module exists so `main.rs` in the bench + serve binaries can
//! reach for one `bring_up::Engine::load(paths)` call and get a fully
//! wired runtime back. No graph capture here (that's a separate step
//! after weights are loaded and bucket shapes are known).

use std::path::PathBuf;
use std::sync::Arc;

use rvllm_attention::Fa3Kernels;
use rvllm_core::{ConfigError, Result, RvllmError};
use rvllm_cutlass::{CublasLt, CutlassLib, Fp8GemmPlan, Policy};
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
    pub fa3: Fa3Kernels,
    pub cutlass: CutlassLib,
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
    pub argmax_mod: LoadedModule,
    pub add_bias_mod: LoadedModule,
    pub fn_rmsnorm: KernelFn,
    pub fn_add_rmsnorm: KernelFn,
    pub fn_quantize: KernelFn,
    pub fn_rope_cache_fp8kv: KernelFn,
    pub fn_silu_mul: KernelFn,
    pub fn_argmax: KernelFn,
    pub fn_add_bias_f16: KernelFn,
}

impl Bringup {
    pub fn load(paths: EnginePaths, arena_bytes: usize) -> Result<Self> {
        // 1. CUDA context + stream.
        let ctx = Arc::new(CudaContextHandle::init(0)?);

        // SAFETY: arena lifetime 'static via leak — engine owns it for program lifetime.
        let arena = HbmArena::new(&ctx, arena_bytes)?;
        // The 'static lifetime gymnastics: HbmArena<'ctx> borrows from ctx.
        // The Arc keeps ctx alive. We transmute the lifetime to 'static
        // because Bringup owns both. This is sound as long as `ctx`
        // outlives `arena` — which it does (they live in the same
        // struct and ctx is the last dropped).
        let arena: HbmArena<'static> = unsafe { std::mem::transmute(arena) };

        let stream = Stream::new(&ctx)?;

        // 2. Arch + model.
        let arch = ModelArch::from_dir(&paths.model_dir)?;
        let model = load_model(&paths.model_dir, &arena, &arch)?;

        // 3. Kernel manifest -> loader -> modules.
        let manifest_path = paths.kernels_dir.join("manifest.json");
        let manifest = KernelManifest::load_and_verify(&manifest_path)?;
        let kernels = Arc::new(KernelLoader::new(manifest));
        let fused_modules = load_fused(&kernels)?;

        // 4. FA3 .so.
        let fa3 = Fa3Kernels::load(paths.fa3_so.clone(), arch.head_dim as u32)?;

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
        let mut variants: std::collections::BTreeSet<_> = policy
            .entries
            .values()
            .map(|e| e.variant)
            .collect();
        for v in 0..16u32 {
            variants.insert(rvllm_cutlass::VariantId(v));
        }
        for v in 100..110u32 {
            variants.insert(rvllm_cutlass::VariantId(v));
        }
        let variants: Vec<_> = variants.into_iter().collect();
        let cutlass = CutlassLib::load(paths.cutlass_so.clone(), &variants)?;

        // cuBLASLt workspace: 32 MiB is recommended for Hopper FP8.
        let cublaslt_ws_bytes: usize = 32 * 1024 * 1024;
        let cublaslt_ws_region =
            arena.region("cublaslt_ws", cublaslt_ws_bytes, 256)?;
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
        let real_prefill: bool =
            std::env::var("RVLLM_REAL_PREFILL").ok().as_deref() == Some("1");
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
        let gate_up_out =
            arena.region("gate_up_out", (max_tokens * 2 * inter * 2) as usize, 16)?;
        let gate_up_fp8 = arena.region("gate_up_fp8", (max_tokens * 2 * inter) as usize, 16)?;
        let gate_up_scale = arena.region("gate_up_scale", (max_tokens * 4) as usize, 16)?;
        let mlp_out_fp8 = arena.region("mlp_out_fp8", (max_tokens * inter) as usize, 16)?;
        let mlp_out_scale = arena.region("mlp_out_scale", (max_tokens * 4) as usize, 16)?;

        let num_blocks_total: u32 = 1024;
        // FA3 paged_decode block_size (tokens per KV page). Default 64;
        // sweepable via RVLLM_BLOCK_SIZE to test 32 / 128 / 256.
        let block_size: u32 = std::env::var("RVLLM_BLOCK_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(64);
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
        // Per-tensor FP8 scales shared across the whole engine. Placeholder
        // values (1.0 / 448.0 = ~0.00223) give a clean identity-ish scaling
        // for the garbage-data faux-prefill regime; a real calibration pass
        // computes these once post-load.
        let q_scale_region = arena.region("q_scale", 4, 4)?;
        let kv_scale_region = arena.region("kv_scale", 4, 4)?;
        {
            let seed: f32 = 1.0f32 / 448.0f32;
            q_scale_region.copy_from_host(&seed.to_le_bytes())?;
            kv_scale_region.copy_from_host(&seed.to_le_bytes())?;
        }

        let cutlass_ws_bytes: usize = 16 * 1024 * 1024;
        let cutlass_ws = arena.region("cutlass_ws", cutlass_ws_bytes, 256)?;
        let fa3_ws = arena.region("fa3_ws", 64 * 1024 * 1024, 256)?;

        let residual = arena.region("residual", (num_seqs * hidden * 2) as usize, 16)?;

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
        let plan_lm_head = override_nonres(Fp8GemmPlan::from_policy(
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
            rms_eps: 1e-6,
        };
        let kernels = layer_exec::LayerKernels {
            fused_add_rmsnorm: self.fused_modules.fn_add_rmsnorm,
            fused_rope_cache_fp8kv: self.fused_modules.fn_rope_cache_fp8kv,
            fused_silu_mul: self.fused_modules.fn_silu_mul,
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
                    qkv_bias: layer.qkv_bias.offset_bytes,
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
                )?;
                // Debug: trap exactly which layer the CUDA error appears after.
                // Triggered via RVLLM_DEBUG_LAYERS=1.
                if std::env::var("RVLLM_DEBUG_LAYERS").ok().as_deref() == Some("1") {
                    use cudarc::driver::sys::{cuStreamSynchronize, CUresult};
                    let sync = unsafe { cuStreamSynchronize(stream as *mut _) };
                    eprintln!("DBG layer {layer_idx} phase={phase:?} sync={sync:?}");
                    if sync != CUresult::CUDA_SUCCESS {
                        panic!("layer {layer_idx} failed with {sync:?}");
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
        let measure_ttft: bool =
            std::env::var("RVLLM_TTFT").ok().as_deref() == Some("1");

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
            Some(arena.region(
                "cu_seqlens_q",
                ((num_seqs as usize + 1) * 4) as usize,
                16,
            )?)
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

        // Faux-prefill loop: when not measuring TTFT just run it normally;
        // when measuring TTFT, wrap the whole "prompt ingest -> first token
        // on host" sequence in a timer.
        let sampled_d_ptr = sampled_tokens.device_ptr();
        let ttft_ns: Option<u128> = if measure_ttft {
            self.stream.fence()?;
            let t_ttft_start = std::time::Instant::now();
            if real_prefill {
                run_real_prefill(&mut ttft_host_buf)?;
            } else {
                for step in 0..FAUX_PREFILL_STEPS {
                    set_step_meta(step)?;
                    one_step(layer_exec::LayerPhase::Decode)?;
                }
            }
            // Fence to ensure prefill's sampled_tokens are written before DtoH.
            self.stream.fence()?;
            dtoh_async_sync(sampled_d_ptr, ttft_host_buf.as_mut_ptr(), n * 4, stream)?;
            self.stream.fence()?;
            Some(t_ttft_start.elapsed().as_nanos())
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
            None
        };

        // Capture one decode step into a CUDA graph then replay for the bench.
        let mut one_step = one_step;
        let graph = rvllm_graph::CapturedGraph::capture(
            num_seqs,
            max_blocks_per_seq,
            rvllm_metadata::MetadataLayout::compute(num_seqs, max_blocks_per_seq).hash(),
            rvllm_graph::GraphFingerprint([0u8; 32]),
            stream,
            || one_step(layer_exec::LayerPhase::Decode),
        )?;

        let t0 = std::time::Instant::now();
        for iter in 0..iters as i32 {
            set_step_meta(FAUX_PREFILL_STEPS + iter)?;
            graph.replay(stream)?;
        }
        self.stream.fence()?;
        let elapsed = t0.elapsed();
        Ok(BenchResult {
            ns_per_step: elapsed.as_nanos() / iters.max(1) as u128,
            total_ns: elapsed.as_nanos(),
            iters,
            num_seqs,
            ttft_ns,
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
        })
    }
}

#[derive(Copy, Clone, Debug)]
pub struct BenchResult {
    pub ns_per_step: u128,
    pub total_ns: u128,
    pub iters: u32,
    pub num_seqs: u32,
    /// Time from "prefill starts" to "first sampled token on host" in ns.
    /// None if TTFT measurement was not requested.
    pub ttft_ns: Option<u128>,
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
fn dtoh_async_sync(src: u64, dst: *mut i32, bytes: usize, stream: u64) -> Result<()> {
    use cudarc::driver::sys::*;
    let r = unsafe {
        cuMemcpyDtoHAsync_v2(dst as *mut _, src, bytes, stream as CUstream)
    };
    if r != CUresult::CUDA_SUCCESS {
        return Err(rvllm_core::RvllmError::cuda(
            "cuMemcpyDtoHAsync",
            rvllm_core::CudaErrorKind::Other,
            rvllm_core::CudaCtx::setup(),
        ));
    }
    Ok(())
}

fn load_fused(loader: &KernelLoader) -> Result<FusedModules> {
    let rmsnorm_mod = loader.load_ptx("fused_rmsnorm_fp8_quant")?;
    let rope_mod = loader.load_ptx("fused_rope_cache_fp8kv")?;
    let silu_mod = loader.load_ptx("fused_silu_fp8_quant")?;
    let argmax_mod = loader.load_ptx("argmax")?;
    let add_bias_mod = loader.load_ptx("add_bias_f16")?;

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
        argmax_mod,
        add_bias_mod,
        fn_rmsnorm,
        fn_add_rmsnorm,
        fn_quantize,
        fn_rope_cache_fp8kv,
        fn_silu_mul,
        fn_argmax,
        fn_add_bias_f16,
    })
}
