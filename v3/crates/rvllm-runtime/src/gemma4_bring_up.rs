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

use rvllm_attention::Fa3Kernels;
use rvllm_core::Result;
use rvllm_cutlass::{CublasLt, CutlassLib, Policy};
use rvllm_kernels::{KernelFn, KernelLoader, LoadedModule};
use rvllm_mem::{context::CudaContextHandle, stream::Stream, HbmArena};

use crate::gemma4_layer_exec::Gemma4LayerKernels;

pub use crate::bring_up::HbmArenaCheckpoint;

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
}

pub struct Gemma4Bringup {
    pub fused: Gemma4FusedModules,
    pub sliding_attention: Fa3Kernels,
    pub global_attention: Fa3Kernels,
    pub cutlass: CutlassLib,
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
        let arena = HbmArena::new(&ctx, arena_bytes)?;
        let arena: HbmArena<'static> = unsafe { std::mem::transmute(arena) };
        let stream = Stream::new(&ctx)?;

        let arch = rvllm_loader::gemma4_arch::Gemma4Arch::from_dir(&paths.model_dir)?;
        let model = rvllm_loader::gemma4_load::load_gemma4_model(
            &paths.model_dir,
            &arena,
            &arch,
        )?;

        let manifest_path = paths.kernels_dir.join("manifest.json");
        let manifest =
            rvllm_kernels::manifest::KernelManifest::load_and_verify(&manifest_path)?;
        let kernels = Arc::new(KernelLoader::new(manifest));

        // Sliding layers use the FA3 SM90 backend at head_dim=256.
        let sliding_attention =
            Fa3Kernels::load(paths.fa3_so.clone(), arch.head_dim_sliding as u32)?;
        // Global layers use the generic fallback paged attention path.
        // Default location is next to the FA3 .so; an explicit override
        // keeps bench/deploy flows flexible while avoiding a new required flag.
        let global_attention_so = std::env::var_os("RVLLM_FA_FALLBACK_SO")
            .map(PathBuf::from)
            .unwrap_or_else(|| paths.fa3_so.with_file_name("libfa_sm89_kernels.so"));
        let global_attention =
            Fa3Kernels::load(global_attention_so, arch.head_dim_global as u32)?;

        let policy_bytes = std::fs::read(&paths.policy_json).map_err(|source| {
            rvllm_core::RvllmError::Io {
                err: rvllm_core::IoError::from(&source),
                path: paths.policy_json.clone(),
                source,
            }
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
        let cutlass = CutlassLib::load(paths.cutlass_so.clone(), &variants)?;

        let cublaslt_ws_bytes: usize = 32 * 1024 * 1024;
        let cublaslt_ws_region =
            arena.region("cublaslt_ws", cublaslt_ws_bytes, 256)?;
        let cublaslt = CublasLt::new(cublaslt_ws_region.device_ptr(), cublaslt_ws_bytes)?;
        let cublaslt_ws = HbmArenaCheckpoint {
            offset_bytes: 0,
            bytes: cublaslt_ws_bytes,
        };

        let fused = load_gemma4_fused(&kernels)?;

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
    pub unsafe fn run_bench(&self, num_seqs: u32, iters: u32, warmup: u32) -> crate::bring_up::BenchResult {
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

        let block_size: u32 = std::env::var("RVLLM_BLOCK_SIZE")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(32);
        let num_blocks_total: u32 = 1024;
        let max_blocks_per_seq = (num_blocks_total / num_seqs).max(1);

        let arena = &self.arena;
        let hidden_fp8 = arena.region("hidden_fp8", (num_seqs * hidden) as usize, 16).unwrap();
        let hidden_scale = arena.region("hidden_scale", (num_seqs * 4) as usize, 16).unwrap();
        let qkv_out = arena.region("qkv_out", (num_seqs * max_qkv_rows * 2) as usize, 16).unwrap();
        let q_base = qkv_out.device_ptr();
        let q_normed = arena.region("q_normed", (num_seqs * max_q_dim * 2) as usize, 16).unwrap();
        let k_normed = arena.region("k_normed", (num_seqs * max_kv_dim * 2) as usize, 16).unwrap();
        let q_fp8 = arena.region("q_fp8", (num_seqs * max_q_dim) as usize, 16).unwrap();
        let attn_out = arena.region("attn_out", (num_seqs * max_q_dim * 2) as usize, 16).unwrap();
        let attn_out_fp8 = arena.region("attn_out_fp8", (num_seqs * max_q_dim) as usize, 16).unwrap();
        let attn_out_scale = arena.region("attn_out_scale", (num_seqs * 4) as usize, 16).unwrap();
        let gate_up_out = arena.region("gate_up_out", (num_seqs * 2 * inter * 2) as usize, 16).unwrap();
        let gate_up_fp8 = arena.region("gate_up_fp8", (num_seqs * 2 * inter) as usize, 16).unwrap();
        let gate_up_scale = arena.region("gate_up_scale", (num_seqs * 4) as usize, 16).unwrap();
        let mlp_out_fp8 = arena.region("mlp_out_fp8", (num_seqs * inter) as usize, 16).unwrap();
        let mlp_out_scale = arena.region("mlp_out_scale", (num_seqs * 4) as usize, 16).unwrap();

        // Uniform KV dim across layer types (sliding 16*256=4096, global 4*512=2048 but
        // k_eq_v doubles it back). Use max for allocation.
        let kv_elem_per_layer = 2 * num_blocks_total * block_size * max_nkvh * max_hd;
        let kv_cache = arena.region(
            "kv_cache",
            (arch.num_hidden_layers as u64 * kv_elem_per_layer as u64) as usize,
            256,
        ).unwrap();
        #[cfg(feature = "cuda")]
        {
            cudarc::driver::sys::cuMemsetD8_v2(
                kv_cache.device_ptr(), 0,
                (arch.num_hidden_layers as u64 * kv_elem_per_layer as u64) as usize,
            );
        }

        let q_scale_region = arena.region("q_scale", 4, 4).unwrap();
        let kv_scale_region = arena.region("kv_scale", 4, 4).unwrap();
        {
            let scale: f32 = 418.0 / 448.0;
            q_scale_region.copy_from_host(&scale.to_le_bytes()).unwrap();
            kv_scale_region.copy_from_host(&scale.to_le_bytes()).unwrap();
        }

        let fa3_ws = arena.region("fa3_ws", 16 * 1024 * 1024, 256).unwrap();
        let residual = arena.region("residual", (num_seqs * hidden * 2) as usize, 16).unwrap();
        cudarc::driver::sys::cuMemsetD8_v2(residual.device_ptr(), 0, (num_seqs * hidden * 2) as usize);

        let positions = arena.region("positions", (num_seqs * 4) as usize, 16).unwrap();
        let slot_mapping = arena.region("slot_mapping", (num_seqs * 4) as usize, 16).unwrap();
        let context_lens = arena.region("context_lens", (num_seqs * 4) as usize, 16).unwrap();
        let block_tables = arena.region("block_tables", (num_seqs * max_blocks_per_seq * 4) as usize, 16).unwrap();
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
            slot_mapping.copy_from_host(bytemuck_cast_i32(&slot)).unwrap();
            context_lens.copy_from_host(bytemuck_cast_i32(&ctx)).unwrap();
            block_tables.copy_from_host(bytemuck_cast_i32(&bt)).unwrap();
        }

        let logits = arena.region("logits", (num_seqs * vocab * 2) as usize, 16).unwrap();
        let sampled_tokens = arena.region("sampled_tokens", (num_seqs * 4) as usize, 16).unwrap();
        let cutlass_ws_bytes: usize = 16 * 1024 * 1024;
        let cutlass_ws = arena.region("cutlass_ws_gemma4", cutlass_ws_bytes, 256).unwrap();
        let residual_ptr = residual.device_ptr();
        let kernels = self.layer_kernels();

        // GEMM plans — uniform shapes across all layers (the sliding/global
        // distinction is a runtime head reshape, weight dims are identical).
        // Use the sliding-layer dims for the plan since those are the common case.
        let q_dim_s = (arch.num_attention_heads * arch.head_dim_sliding) as u32;
        let kv_dim_s = (arch.num_kv_heads_sliding * arch.head_dim_sliding) as u32;
        let qkv_rows_s = q_dim_s + 2 * kv_dim_s;
        use rvllm_cutlass::Fp8GemmPlan;
        let gemm_plans = Gemma4GemmPlans {
            qkv: Fp8GemmPlan::from_policy(&self.policy, num_seqs, qkv_rows_s, hidden, rvllm_core::DType::Fp8E4M3).unwrap(),
            o: Fp8GemmPlan::from_policy_residual(&self.policy, num_seqs, hidden, q_dim_s, rvllm_core::DType::Fp8E4M3).unwrap(),
            gate_up: Fp8GemmPlan::from_policy(&self.policy, num_seqs, 2 * inter, hidden, rvllm_core::DType::Fp8E4M3).unwrap(),
            down: Fp8GemmPlan::from_policy_residual(&self.policy, num_seqs, hidden, inter, rvllm_core::DType::Fp8E4M3).unwrap(),
        };

        let one_step = || -> rvllm_core::Result<()> {
            for (layer_idx, layer) in self.model.layers.iter().enumerate() {
                let lt = arch.layer_types[layer_idx];
                let hd = arch.head_dim_for_layer(layer_idx) as u32;
                let nkvh = arch.num_kv_heads_for_layer(layer_idx) as u32;
                let q_dim = (arch.num_attention_heads as u32) * hd;
                let kv_dim = nkvh * hd;
                let qkv_rows = q_dim + 2 * kv_dim;

                let dims = Gemma4LayerDims {
                    num_tokens: num_seqs,
                    hidden,
                    num_heads: arch.num_attention_heads as u32,
                    num_kv_heads: nkvh,
                    head_dim: hd,
                    rotary_dim: arch.rotary_dim_for_layer(layer_idx) as u32,
                    intermediate: inter,
                    block_size,
                    max_blocks_per_seq,
                    num_blocks_total,
                    attn_scale: 1.0, // Gemma 4 uses QK-norm, not 1/sqrt(d)
                    rms_eps: arch.rms_norm_eps,
                    layer_type: lt,
                    sliding_window: arch.sliding_window_size as u32,
                };

                let k_out = q_base + (num_seqs as u64) * (q_dim as u64) * 2;
                let v_out = k_out + (num_seqs as u64) * (kv_dim as u64) * 2;
                let layer_kv_base = kv_cache.device_ptr()
                    + (layer_idx as u64) * (kv_elem_per_layer as u64);

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
                };

                let scratch = Gemma4LayerScratch {
                    hidden_fp8: hidden_fp8.device_ptr(),
                    hidden_scale: hidden_scale.device_ptr(),
                    q_out: q_base,
                    k_out,
                    v_out,
                    q_normed: q_normed.device_ptr(),
                    k_normed: k_normed.device_ptr(),
                    q_fp8: q_fp8.device_ptr(),
                    k_cache: layer_kv_base,
                    v_cache: layer_kv_base + (kv_elem_per_layer / 2) as u64,
                    q_scale_ptr: q_scale_region.device_ptr(),
                    kv_scale_ptr: kv_scale_region.device_ptr(),
                    attn_out: attn_out.device_ptr(),
                    attn_out_fp8: attn_out_fp8.device_ptr(),
                    attn_out_scale: attn_out_scale.device_ptr(),
                    post_attn_normed: 0, // unused in current forward
                    gate_up_out: gate_up_out.device_ptr(),
                    gate_up_fp8: gate_up_fp8.device_ptr(),
                    gate_up_scale: gate_up_scale.device_ptr(),
                    mlp_out_fp8: mlp_out_fp8.device_ptr(),
                    mlp_out_scale: mlp_out_scale.device_ptr(),
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
                num_seqs, vocab, arch.logit_softcap, stream,
            )?;
            rvllm_fused::ArgmaxLaunch { num_tokens: num_seqs, vocab }
                .launch(self.fused.fn_argmax, logits.device_ptr(),
                    sampled_tokens.device_ptr(), stream)?;
            Ok(())
        };

        // Warmup
        for _ in 0..warmup {
            one_step().unwrap();
        }
        self.stream.fence().unwrap();

        // Timed
        let no_graph = std::env::var("RVLLM_NO_GRAPH").ok().as_deref() == Some("1");
        let t0 = std::time::Instant::now();
        if no_graph {
            for _ in 0..iters {
                one_step().unwrap();
            }
        } else {
            // Graph capture not yet wired for Gemma4 — run eager
            for _ in 0..iters {
                one_step().unwrap();
            }
        }
        self.stream.fence().unwrap();
        let elapsed = t0.elapsed();

        crate::bring_up::BenchResult {
            ns_per_step: elapsed.as_nanos() / iters.max(1) as u128,
            total_ns: elapsed.as_nanos(),
            iters,
            num_seqs,
            ttft_ns: None,
            ttft_hot_ns: None,
        }
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
        }
    }
}

fn bytemuck_cast_i32(v: &[i32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

fn load_gemma4_fused(loader: &KernelLoader) -> Result<Gemma4FusedModules> {
    let rmsnorm_mod = loader.load_ptx("fused_rmsnorm_fp8_quant")?;
    let rope_mod = loader.load_ptx("fused_rope_partial_fp8kv")?;
    let gelu_mod = loader.load_ptx("fused_gelu_mul_fp8_quant")?;
    let argmax_mod = loader.load_ptx("argmax")?;
    let qk_norm_mod = loader.load_ptx("fused_qk_rmsnorm")?;
    let softcap_mod = loader.load_ptx("logit_softcap")?;
    let residual_scale_mod = loader.load_ptx("residual_scale_f16")?;
    let vnorm_mod = loader.load_ptx("vnorm_f16")?;

    let rmsnorm_inplace_mod = loader.load_ptx("rmsnorm_inplace_f16")?;
    let fn_rmsnorm = rmsnorm_inplace_mod.get_function("rmsnorm_inplace_f16_kernel")?;
    let fn_rmsnorm_fp8_quant = rmsnorm_mod.get_function("fused_rmsnorm_fp8_quant_kernel")?;
    let fn_quantize = rmsnorm_mod.get_function("quantize_fp8_per_token_kernel")?;
    let fn_rope_partial_fp8kv =
        rope_mod.get_function("fused_rope_partial_fp8kv_kernel")?;
    let fn_gelu_mul =
        gelu_mod.get_function("fused_gelu_mul_fp8_quant_kernel")?;
    let fn_argmax = argmax_mod.get_function("argmax_kernel")?;
    let fn_qk_rmsnorm =
        qk_norm_mod.get_function("fused_qk_rmsnorm_kernel")?;
    let fn_softcap = softcap_mod.get_function("logit_softcap_kernel")?;
    let fn_residual_scale =
        residual_scale_mod.get_function("residual_scale_f16_kernel")?;
    let fn_vnorm = vnorm_mod.get_function("vnorm_f16_kernel")?;

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
    })
}
