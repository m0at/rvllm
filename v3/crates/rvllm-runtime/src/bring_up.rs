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

        // --- scratch allocations -----------------------------------------
        let arena = &self.arena;
        let hidden_fp8 = arena.region("hidden_fp8", (num_seqs * hidden) as usize, 16)?;
        let hidden_scale = arena.region("hidden_scale", (num_seqs * 4) as usize, 16)?;
        // One packed QKV output: [num_tokens, q_dim + 2*kv_dim] f16.
        let qkv_out_bytes = (num_seqs * qkv_rows * 2) as usize;
        let qkv_out = arena.region("qkv_out", qkv_out_bytes, 16)?;
        let q_base = qkv_out.device_ptr();
        let k_base = q_base + (num_seqs as u64) * (q_dim as u64) * 2;
        let v_base = k_base + (num_seqs as u64) * (kv_dim as u64) * 2;
        let attn_out = arena.region("attn_out", (num_seqs * q_dim * 2) as usize, 16)?;
        let attn_out_fp8 = arena.region("attn_out_fp8", (num_seqs * q_dim) as usize, 16)?;
        let attn_out_scale = arena.region("attn_out_scale", (num_seqs * 4) as usize, 16)?;
        let gate_up_out =
            arena.region("gate_up_out", (num_seqs * 2 * inter * 2) as usize, 16)?;
        let gate_up_fp8 = arena.region("gate_up_fp8", (num_seqs * 2 * inter) as usize, 16)?;
        let gate_up_scale = arena.region("gate_up_scale", (num_seqs * 4) as usize, 16)?;
        let mlp_out_fp8 = arena.region("mlp_out_fp8", (num_seqs * inter) as usize, 16)?;
        let mlp_out_scale = arena.region("mlp_out_scale", (num_seqs * 4) as usize, 16)?;

        let num_blocks_total: u32 = 1024;
        let block_size: u32 = 64;
        let max_blocks_per_seq: u32 = 128;
        // FP8 E4M3 KV: 1 byte/element (was 2 for f16). Halves KV memory
        // and doubles HBM-bandwidth efficiency on attention reads.
        let kv_per_layer = 2 * num_blocks_total * block_size * nkvh * head_dim;
        let kv_cache = arena.region(
            "kv_cache",
            (arch.num_hidden_layers as u64 * kv_per_layer as u64) as usize,
            256,
        )?;
        // FP8 Q scratch (post-rope) consumed by FA3; f16 Q from QKV GEMM
        // still lives at q_out (2 bytes/elem).
        let q_fp8 = arena.region("q_fp8", (num_seqs * q_dim) as usize, 16)?;
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
        let positions = arena.region("positions", (num_seqs * 4) as usize, 16)?;
        let slot_mapping = arena.region("slot_mapping", (num_seqs * 4) as usize, 16)?;
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

        let one_step = || -> Result<()> {
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
                    k_out: k_base,
                    v_out: v_base,
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
                layer_exec::forward(
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
                )?;
            }
            if !skip_lm_head {
                // LM head: quantize residual -> fp8, GEMM -> logits (f16),
                // argmax over vocab -> sampled_tokens[i32].
                rvllm_fused::QuantizeFp8PerTokenLaunch {
                    num_tokens: num_seqs,
                    dim: hidden,
                }
                .launch(
                    self.fused_modules.fn_quantize,
                    hidden_fp8.device_ptr(),
                    hidden_scale.device_ptr(),
                    residual_ptr,
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
            one_step()?;
        }
        self.stream.fence()?;

        // Faux-prefill: run 16 eager decode steps with advancing positions
        // and slot_mapping, so the paged KV cache actually contains real
        // forward-pass activations before we start timing. vLLM's bench has
        // real KV from real prompt tokens; this is the closest we can get
        // without a separate prefill kernel path.
        const FAUX_PREFILL_STEPS: i32 = 16;
        let n = num_seqs as usize;
        for step in 0..FAUX_PREFILL_STEPS {
            let pos_host: Vec<i32> = (0..n as i32).map(|i| step + i * 32).collect();
            let slot_host: Vec<i32> = (0..n as i32)
                .map(|i| step + i * max_blocks_per_seq as i32 * block_size as i32)
                .collect();
            let ctx_host: Vec<i32> = vec![step + 1; n];
            positions.copy_from_host(bytemuck_cast_i32(&pos_host))?;
            slot_mapping.copy_from_host(bytemuck_cast_i32(&slot_host))?;
            context_lens.copy_from_host(bytemuck_cast_i32(&ctx_host))?;
            one_step()?;
        }
        self.stream.fence()?;

        // Capture one step into a CUDA graph then replay for the bench.
        // Metadata is RE-UPLOADED per iter (not frozen) to match vLLM
        // which reschedules each step; positions / slot_mapping / context_lens
        // advance just like a real decode loop would.
        let mut one_step = one_step;
        let graph = rvllm_graph::CapturedGraph::capture(
            num_seqs,
            max_blocks_per_seq,
            rvllm_metadata::MetadataLayout::compute(num_seqs, max_blocks_per_seq).hash(),
            rvllm_graph::GraphFingerprint([0u8; 32]),
            stream,
            || one_step(),
        )?;

        let t0 = std::time::Instant::now();
        for iter in 0..iters as i32 {
            let pos = FAUX_PREFILL_STEPS + iter;
            let pos_host: Vec<i32> = (0..n as i32).map(|i| pos + i * 32).collect();
            let slot_host: Vec<i32> = (0..n as i32)
                .map(|i| pos + i * max_blocks_per_seq as i32 * block_size as i32)
                .collect();
            let ctx_host: Vec<i32> = vec![pos + 1; n];
            positions.copy_from_host(bytemuck_cast_i32(&pos_host))?;
            slot_mapping.copy_from_host(bytemuck_cast_i32(&slot_host))?;
            context_lens.copy_from_host(bytemuck_cast_i32(&ctx_host))?;
            graph.replay(stream)?;
        }
        self.stream.fence()?;
        let elapsed = t0.elapsed();
        Ok(BenchResult {
            ns_per_step: elapsed.as_nanos() / iters.max(1) as u128,
            total_ns: elapsed.as_nanos(),
            iters,
            num_seqs,
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub unsafe fn run_bench(&self, num_seqs: u32, iters: u32, _warmup: u32) -> Result<BenchResult> {
        Ok(BenchResult {
            ns_per_step: 0,
            total_ns: 0,
            iters,
            num_seqs,
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
        })
    }
}

#[derive(Copy, Clone, Debug)]
pub struct BenchResult {
    pub ns_per_step: u128,
    pub total_ns: u128,
    pub iters: u32,
    pub num_seqs: u32,
}

#[cfg(feature = "cuda")]
fn bytemuck_cast_i32(v: &[i32]) -> &[u8] {
    // SAFETY: i32 has a defined bit layout; we only read these bytes,
    // and the output slice is the same length/alignment.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
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
