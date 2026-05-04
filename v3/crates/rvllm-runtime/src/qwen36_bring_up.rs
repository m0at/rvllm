//! Qwen 3.6 35B-A3B bring-up (Phase 1: outside-tensor upload).
//!
//! Phase 1 contract:
//!   - Initialize CUDA context, arena, stream.
//!   - Upload the three "outside-the-stack" tensors (embedding, final
//!     RMSNorm, lm_head) via [`rvllm_loader::qwen36_load::load_qwen36_outside`].
//!   - All forward methods (`run_generate`, `run_bench`, `run_ppl`,
//!     `init_prefix_cache`) `unimplemented!()` with phase-pointer
//!     messages — per-layer tensors + forward kernels are Phase 2/3.
//!
//! See `~/.claude/plans/abundant-meandering-sifakis.md` for the
//! phase-list.

use std::path::PathBuf;
use std::sync::Arc;

use rvllm_core::Result;
use rvllm_cutlass::CublasLt;
use rvllm_kernels::{KernelFn, KernelLoader, LoadedModule};
use rvllm_loader::qwen36_weights::Qwen36LoadedModel;
use rvllm_mem::{context::CudaContextHandle, stream::Stream, HbmArena};

use crate::gemma4_bring_up::Gemma4EnginePaths;
use crate::qwen36_arch::Qwen36Arch;

/// Kernel function pointers + their LoadedModule anchors needed for the
/// Qwen 3.6 outside-only forward path (embedding lookup, final RMSNorm,
/// lm_head matmul, argmax).
///
/// `LoadedModule` is RAII — its `Drop` calls `cuModuleUnload`, after
/// which the matching `KernelFn` handles become invalid. Holding the
/// modules alongside the function pointers in this struct keeps the
/// pair alive for the lifetime of the bring-up.
pub struct Qwen36OutsideKernels {
    pub embedding_gather_f16_mod: LoadedModule,
    pub fn_embedding_gather_f16: KernelFn,
    pub rmsnorm_inplace_f16_mod: LoadedModule,
    pub fn_rmsnorm_inplace_f16: KernelFn,
    /// FP8 GEMV for the lm_head matmul. Phase 3d will CPU-quantize the
    /// bf16 lm_head to FP8 at load time so this kernel can be used.
    /// `None` on platforms where the f16-input native-CVT variant
    /// isn't available (gated on `__CUDA_ARCH__ >= 1000`).
    pub fp8_gemv_mod: LoadedModule,
    pub fn_fp8_gemv_wpr_native_f16in: Option<KernelFn>,
    pub argmax_mod: LoadedModule,
    pub fn_argmax: KernelFn,
    /// Phase 3g: fused (final RMSNorm + FP8-quantize) for the lm_head
    /// pre-matmul step. Outputs FP8 hidden + per-token f32 scale that
    /// cuBLASLt's fp8_gemm consumes.
    pub fused_rmsnorm_fp8_quant_mod: LoadedModule,
    pub fn_fused_rmsnorm_fp8_quant: KernelFn,
    /// Phase 4g: partial-rotary RoPE kernel (f16 Q/K/V → f16 q_out
    /// + KV cache write). Qwen 3.6 uses partial_rotary_factor=0.25,
    /// so only `head_dim * 0.25 = 64` of the 256 head_dim is rotated.
    pub fused_rope_partial_f16kv_mod: LoadedModule,
    pub fn_fused_rope_partial_f16kv: KernelFn,
    /// Phase 4h: paged f16 attention decode kernel
    /// (`flash_attention_2_decode_f16io_kernel`). f16 Q/K/V with a
    /// paged f16 KV cache; sliding-window param `< 0` means no window
    /// (Qwen 3.6 full-attn layers don't use sliding).
    pub flash_attention_mod: LoadedModule,
    pub fn_flash_attention_2_decode_f16io: KernelFn,
    /// Phase 4j: Qwen-specific element-wise `sigmoid(gate) * values`
    /// for the `attn_output_gate=true` path. New CUDA kernel added
    /// in this phase — kernels/sigmoid_mul_f16.cu.
    pub sigmoid_mul_f16_mod: LoadedModule,
    pub fn_sigmoid_mul_f16: KernelFn,
    /// Phase 4q: depthwise causal 1D convolution for the
    /// Gated-DeltaNet linear-attn block. New CUDA kernel —
    /// kernels/causal_conv1d_f16.cu.
    pub causal_conv1d_f16_mod: LoadedModule,
    pub fn_causal_conv1d_f16: KernelFn,
}

pub struct Qwen36Bringup {
    pub paths: Gemma4EnginePaths,
    pub arena_bytes: usize,
    pub arch: Qwen36Arch,
    pub ctx: Arc<CudaContextHandle>,
    pub arena: HbmArena<'static>,
    pub stream: Stream,
    pub model: Qwen36LoadedModel,
    pub kernels: Arc<KernelLoader>,
    pub outside_kernels: Qwen36OutsideKernels,
    pub cublaslt: CublasLt,
    /// Phase 4f: precomputed RoPE cos/sin tables for Qwen 3.6
    /// (rope_theta=10M, head_dim=256). Single-axis tables uploaded
    /// at bring-up; MRoPE's section-aware position encoding
    /// (sections [11, 11, 10]) is applied at launch time on top of
    /// these base tables in Phase 4g.
    pub rope_cos: u64,
    pub rope_sin: u64,
    pub rope_max_pos: u32,
}

impl Qwen36Bringup {
    /// Phase 1: CUDA init + arena + outside-tensor upload.
    /// Returns `Err` if `config.json` is missing required Qwen-3.6
    /// markers, or if any of the three outside tensors is missing.
    pub fn load(paths: Gemma4EnginePaths, arena_bytes: usize) -> Result<Self> {
        let arch = match Qwen36Arch::from_dir(&paths.model_dir)? {
            Some(a) => a,
            None => {
                panic!(
                    "Qwen36Bringup::load called for model_dir={:?} but \
                     Qwen36Arch::from_dir returned None — caller dispatched \
                     incorrectly",
                    paths.model_dir
                );
            }
        };
        arch.log_summary();

        let ctx = Arc::new(CudaContextHandle::init(0)?);

        #[cfg(feature = "cuda")]
        let compile_target: Option<rvllm_core::CompileTarget> = {
            let (major, minor) = ctx.compute_capability();
            rvllm_core::CompileTarget::from_compute_capability(major, minor)
        };
        #[cfg(not(feature = "cuda"))]
        let compile_target: Option<rvllm_core::CompileTarget> = None;

        // GB10 (sm_121) has no dedicated HBM — `cuMemAllocManaged` is the
        // right backing. Mirrors the Gemma 4 selection at
        // `gemma4_bring_up.rs::Gemma4Bringup::load`.
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

        let model = rvllm_loader::qwen36_load::load_qwen36_model(
            &paths.model_dir,
            &arena,
            &arch.base.layer_types,
            arch.num_experts,
        )?;

        // Phase 3a: load + verify the PTX kernel manifest the same way
        // Gemma 4 does. The Qwen forward path will need the same
        // model-agnostic kernel set (embedding_gather, fused_rmsnorm,
        // CUTLASS FP8-GEMM, FA2 paged-attn) plus future Qwen-specific
        // ones (qwen_qk_norm, attn_output_gate, linear-attn recurrent,
        // MoE grouped-FP8-GEMM). Initializing the loader here proves
        // the manifest path and arch-pinning still hold for the Qwen
        // bring-up before Phase 3b starts wiring kernel calls.
        let kernels_dir = crate::bring_up::resolve_kernels_dir(&ctx, &paths.kernels_dir)?;
        let manifest_path = kernels_dir.join("manifest.json");
        let manifest =
            rvllm_kernels::manifest::KernelManifest::load_and_verify(&manifest_path)?;
        if let Some(t) = compile_target {
            manifest.assert_arch(t.as_sm_str())?;
        }
        manifest
            .warn_if_revision_drift(rvllm_kernels::manifest::VerifiedManifest::BUILD_REVISION);
        let kernels = Arc::new(KernelLoader::new(manifest));

        // Phase 3b: resolve the outside-path kernel function pointers
        // (embedding_gather + final RMSNorm). The matching modules
        // stay in the bring-up struct so the function handles outlive
        // any per-request scope. lm_head matmul resolution is deferred
        // to Phase 3c — bf16 vs FP8 lm_head dispatch needs an explicit
        // decision (CPU-quantize at load like Gemma's tied path, or
        // route through a bf16 cuBLASLt GEMM).
        let embedding_gather_f16_mod = kernels.load_ptx("embedding_gather_f16")?;
        let fn_embedding_gather_f16 =
            embedding_gather_f16_mod.get_function("embedding_gather_f16_kernel")?;
        let rmsnorm_inplace_f16_mod = kernels.load_ptx("rmsnorm_inplace_f16")?;
        let fn_rmsnorm_inplace_f16 =
            rmsnorm_inplace_f16_mod.get_function("rmsnorm_inplace_f16_kernel")?;
        let fp8_gemv_mod = kernels.load_ptx(rvllm_kernels::FP8_GEMV_PTX_STEM)?;
        let fn_fp8_gemv_wpr_native_f16in = match compile_target {
            Some(t)
                if rvllm_kernels::Fp8GemvVariant::WprNativeF16In.available_for(t) =>
            {
                Some(
                    fp8_gemv_mod.get_function(
                        rvllm_kernels::Fp8GemvVariant::WprNativeF16In.entry_point(),
                    )?,
                )
            }
            _ => None,
        };
        let argmax_mod = kernels.load_ptx("argmax")?;
        let fn_argmax = argmax_mod.get_function("argmax_kernel")?;
        let fused_rmsnorm_fp8_quant_mod = kernels.load_ptx("fused_rmsnorm_fp8_quant")?;
        let fn_fused_rmsnorm_fp8_quant =
            fused_rmsnorm_fp8_quant_mod.get_function("fused_rmsnorm_fp8_quant_kernel")?;
        let fused_rope_partial_f16kv_mod =
            kernels.load_ptx("fused_rope_partial_f16kv")?;
        let fn_fused_rope_partial_f16kv = fused_rope_partial_f16kv_mod
            .get_function("fused_rope_partial_f16kv_kernel")?;
        let flash_attention_mod = kernels.load_ptx("flash_attention")?;
        let fn_flash_attention_2_decode_f16io = flash_attention_mod
            .get_function("flash_attention_2_decode_f16io_kernel")?;
        let sigmoid_mul_f16_mod = kernels.load_ptx("sigmoid_mul_f16")?;
        let fn_sigmoid_mul_f16 =
            sigmoid_mul_f16_mod.get_function("sigmoid_mul_f16_kernel")?;
        let causal_conv1d_f16_mod = kernels.load_ptx("causal_conv1d_f16")?;
        let fn_causal_conv1d_f16 =
            causal_conv1d_f16_mod.get_function("causal_conv1d_f16_kernel")?;
        let outside_kernels = Qwen36OutsideKernels {
            embedding_gather_f16_mod,
            fn_embedding_gather_f16,
            rmsnorm_inplace_f16_mod,
            fn_rmsnorm_inplace_f16,
            fp8_gemv_mod,
            fn_fp8_gemv_wpr_native_f16in,
            argmax_mod,
            fn_argmax,
            fused_rmsnorm_fp8_quant_mod,
            fn_fused_rmsnorm_fp8_quant,
            fused_rope_partial_f16kv_mod,
            fn_fused_rope_partial_f16kv,
            flash_attention_mod,
            fn_flash_attention_2_decode_f16io,
            sigmoid_mul_f16_mod,
            fn_sigmoid_mul_f16,
            causal_conv1d_f16_mod,
            fn_causal_conv1d_f16,
        };
        eprintln!(
            "[qwen36] outside kernels resolved: embedding_gather_f16, \
             rmsnorm_inplace_f16, fp8_gemv (wpr_native_f16in: {}), \
             argmax, fused_rmsnorm_fp8_quant, fused_rope_partial_f16kv, \
             flash_attention_2_decode_f16io, sigmoid_mul_f16, \
             causal_conv1d_f16.",
            outside_kernels.fn_fp8_gemv_wpr_native_f16in.is_some(),
        );

        // Phase 3g: cuBLASLt for the lm_head FP8 GEMM. Same 32 MiB
        // workspace size Gemma 4 uses (gemma4_bring_up.rs:1415).
        let cublaslt_ws_bytes: usize = 32 * 1024 * 1024;
        let cublaslt_ws_region = arena.region("qwen36_cublaslt_ws", cublaslt_ws_bytes, 256)?;
        let cublaslt = CublasLt::new(cublaslt_ws_region.device_ptr(), cublaslt_ws_bytes)?;
        eprintln!(
            "[qwen36] cuBLASLt initialized with {} MiB workspace.",
            cublaslt_ws_bytes / (1024 * 1024)
        );

        // Phase 4f: precompute single-axis RoPE cos/sin tables.
        // Qwen 3.6's `rope_theta` (10_000_000) + `head_dim` (256)
        // give a base table of shape `[max_pos, head_dim/2]` for
        // each of cos and sin. Cap `max_pos` to RVLLM_MAX_TOKENS_CAP
        // (typically 4096) — the model's `max_position_embeddings`
        // is 262144 which would mean 128 MiB of f16 cos/sin tables;
        // capping keeps the bring-up cost reasonable until Phase 4g
        // wires the real RoPE launch and we know the actual decode
        // window. MRoPE section math (sections [11, 11, 10]) is
        // applied on top of these tables at launch time, not baked
        // into the tables themselves (text-only mode, sections
        // collapse to standard RoPE per-position).
        let rope_max_pos = std::env::var("RVLLM_MAX_TOKENS_CAP")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(4096)
            .min(262_144);
        let rope_theta = arch.base.rope_theta;
        let rope_head_dim = arch.base.head_dim as u32;
        let half = (rope_head_dim / 2) as usize;
        let inv_theta: Vec<f32> = (0..half)
            .map(|i| 1.0 / rope_theta.powf(2.0 * i as f32 / rope_head_dim as f32))
            .collect();
        let rope_table_elems = (rope_max_pos as usize) * half;
        let rope_table_bytes = rope_table_elems * 2; // f16
        let mut cos_bytes = Vec::with_capacity(rope_table_bytes);
        let mut sin_bytes = Vec::with_capacity(rope_table_bytes);
        for pos in 0..rope_max_pos as usize {
            for &freq in &inv_theta {
                let angle = pos as f32 * freq;
                let c = angle.cos();
                let s = angle.sin();
                // Inline f32 → f16 (avoid pulling `half` as a runtime dep).
                cos_bytes.extend_from_slice(&f32_to_f16_bits(c).to_le_bytes());
                sin_bytes.extend_from_slice(&f32_to_f16_bits(s).to_le_bytes());
            }
        }
        let rope_cos_region = arena.region("qwen36_rope_cos", rope_table_bytes, 16)?;
        let rope_sin_region = arena.region("qwen36_rope_sin", rope_table_bytes, 16)?;
        unsafe {
            rope_cos_region.copy_from_host(&cos_bytes)?;
            rope_sin_region.copy_from_host(&sin_bytes)?;
        }
        let rope_cos = rope_cos_region.device_ptr();
        let rope_sin = rope_sin_region.device_ptr();
        eprintln!(
            "[qwen36] RoPE tables uploaded: max_pos={rope_max_pos} head_dim={rope_head_dim} \
             theta={rope_theta:.1e} ({:.1} KiB cos + {:.1} KiB sin)",
            rope_table_bytes as f64 / 1024.0,
            rope_table_bytes as f64 / 1024.0,
        );

        let n_full = model.layers.iter().filter(|l| matches!(
            l.attn,
            rvllm_loader::qwen36_weights::Qwen36LayerAttn::Full(_)
        )).count();
        let n_linear = model.layers.iter().filter(|l| matches!(
            l.attn,
            rvllm_loader::qwen36_weights::Qwen36LayerAttn::Linear(_)
        )).count();
        eprintln!(
            "[qwen36] Phase 4q bring-up complete: outside (incl. \
             FP8-quantized lm_head) + {n_full} full-attention + \
             {n_linear} linear-attention + per-layer MoE blocks \
             ({} experts/layer) + KernelLoader + outside kernel \
             pointers + cuBLASLt + outside-only forward smoke \
             (embed → rmsnorm+fp8quant → fp8_gemm → cpu_argmax) \
             validated end-to-end. arena.used()={used:.2} GiB / \
             {total:.2} GiB. Per-layer forward (full-attn + \
             linear-attn + MoE) still TODO. \
             See ~/.claude/plans/abundant-meandering-sifakis.md.",
            arch.num_experts,
            used = arena.used() as f64 / (1024.0 * 1024.0 * 1024.0),
            total = arena_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        );

        let bringup = Self {
            paths,
            arena_bytes,
            arch,
            ctx,
            arena,
            stream,
            model,
            kernels,
            outside_kernels,
            cublaslt,
            rope_cos,
            rope_sin,
            rope_max_pos,
        };
        // Phase 3e smoke: actually launch embedding_gather against the
        // loaded weights. If this throws, the rest of the forward
        // pipeline can't be built on top — fail fast.
        bringup.forward_outside_smoke()?;

        // Phase 4a: verify the reusable `forward_outside_only` API by
        // calling it on a slightly different sequence and logging the
        // argmax token id. Confirms the same kernel chain works
        // through the public method, ready for cuda_worker dispatch
        // in Phase 4b.
        let probe = bringup.forward_outside_only(&[1, 200, 2000, 20_000, 50_000])?;
        eprintln!(
            "[qwen36] forward_outside_only smoke: 5-token input → \
             argmax_token_id={probe} (garbage by design — per-layer \
             forward still TODO)"
        );

        // Phase 4c/4d: per-layer kernel chain. Launches Q+K+V
        // projections against the first full-attention layer's
        // blockwise FP8 weights. Synthetic input — proves all three
        // projection roles work end-to-end through the same kernel.
        bringup.forward_layer3_qkv_probe()?;
        // Phase 4g: partial RoPE launch against the precomputed
        // cos/sin tables. Identity at position 0, sanity-checks
        // ABI + table indexing.
        bringup.forward_layer3_rope_probe()?;
        // Phase 4h: paged f16 attention decode launch with a
        // single-token KV cache stand-in. Validates the FA2
        // decode kernel ABI works for Qwen's heterogeneous
        // (num_heads=16, num_kv_heads=2) shape.
        bringup.forward_layer3_paged_attn_probe()?;
        // Phase 4i: o_proj (closes the attention block).
        bringup.forward_layer3_o_proj_probe()?;
        // Phase 4j: attn_output_gate (sigmoid · attn_out via new
        // sigmoid_mul_f16 kernel).
        bringup.forward_layer3_attn_gate_probe()?;
        // Phase 4k: shared-expert gate_proj (first MoE-block kernel
        // launch). Routed-expert dispatch + bf16 router land in 4l/4m.
        bringup.forward_layer3_moe_shared_probe()?;
        // Phase 4l: router GEMV + top-8 selection (CPU-side smoke).
        bringup.forward_layer3_router_probe()?;
        // Phase 4m: full SwiGLU FFN for routed expert 0
        // (gate + up → silu·mul (host) → down).
        bringup.forward_layer3_routed_expert_probe()?;
        // Phase 4n: complete MoE block — top-8 routed experts (with
        // per-expert offsets) + shared expert (sigmoid-gated) →
        // weighted sum.
        bringup.forward_layer3_full_moe_probe()?;
        // Phase 4o (skeleton): linear-attn weight presence + shape
        // log. Recurrent kernel (Gated-DeltaNet ssm-scan) is real
        // CUDA work TBD.
        bringup.forward_layer0_linear_attn_probe()?;
        // Phase 4p: linear-attn input projections (in_proj_qkv +
        // in_proj_z). Confirms the FP8 GEMV kernel works on
        // linear-attn weight shapes.
        bringup.forward_layer0_linear_in_proj_probe()?;
        // Phase 4q: causal_conv1d_f16 — first piece of the
        // Gated-DeltaNet ssm-scan, runs against real layer-0
        // conv1d weights.
        bringup.forward_layer0_conv1d_probe()?;

        Ok(bringup)
    }

    pub fn kernels_dir(&self) -> &PathBuf {
        &self.paths.kernels_dir
    }

    /// Phase 3e smoke test: launch the embedding-gather kernel against
    /// the loaded `embed_tokens` table for a hardcoded 4-token input,
    /// fence the stream, and DtoH the first 4 hidden-state floats so
    /// we can log them. Validates that:
    ///   - the bring-up's CUDA context + arena + stream + KernelLoader
    ///     are wired correctly
    ///   - the f16 `embed_tokens` upload reaches the device with
    ///     the right layout
    ///   - the EmbeddingGatherLaunch ABI matches Qwen 3.6 dims
    /// without exercising any per-layer math (still Phase 3f+).
    /// Phase 4a: outside-only forward over arbitrary input tokens.
    /// Skips all 40 transformer layers — runs only:
    ///   embed_tokens → final_norm + fp8quant → lm_head fp8_gemm →
    ///   CPU argmax over the LAST token's logits.
    /// Returns the argmax token id of the last input position. Output
    /// will be garbage (no per-layer math), but the kernel pipeline
    /// is exercised end-to-end.
    pub fn forward_outside_only(&self, token_ids: &[i32]) -> Result<i32> {
        if token_ids.is_empty() {
            return Err(rvllm_core::RvllmError::cuda(
                "forward_outside_only: empty token_ids",
                rvllm_core::CudaErrorKind::Other,
                rvllm_core::CudaCtx::setup(),
            ));
        }
        let last_idx = token_ids.len() - 1;
        let hidden = self.arch.base.hidden_size as u32;
        let vocab = self.arch.base.vocab_size as u32;
        let num_tokens = token_ids.len() as u32;

        // Allocate device regions: token IDs (i32) + hidden state (f16).
        let mut token_bytes_owned: Vec<u8> = Vec::with_capacity(token_ids.len() * 4);
        for t in token_ids {
            token_bytes_owned.extend_from_slice(&t.to_le_bytes());
        }
        let tokens_region = self.arena.region(
            "qwen36_outside_tokens",
            token_bytes_owned.len(),
            16,
        )?;
        unsafe { tokens_region.copy_from_host(&token_bytes_owned)? };

        let hidden_bytes = (num_tokens as usize) * (hidden as usize) * 2; // f16
        let hidden_region =
            self.arena.region("qwen36_outside_hidden", hidden_bytes, 16)?;

        unsafe {
            rvllm_fused::EmbeddingGatherLaunch {
                num_tokens,
                hidden,
                vocab,
            }
            .launch(
                self.outside_kernels.fn_embedding_gather_f16,
                hidden_region.device_ptr(),
                self.model.outside.embed_tokens.offset_bytes,
                tokens_region.device_ptr(),
                self.stream.raw() as u64,
            )?;
        }

        let eps = self.arch.base.rms_norm_eps;
        let hidden_fp8_bytes = (num_tokens as usize) * (hidden as usize);
        let hidden_scale_bytes = (num_tokens as usize) * 4;
        let logits_bytes = (num_tokens as usize) * (vocab as usize) * 2;
        let hidden_fp8_region =
            self.arena.region("qwen36_outside_hidden_fp8", hidden_fp8_bytes, 16)?;
        let hidden_scale_region = self.arena.region(
            "qwen36_outside_hidden_scale",
            hidden_scale_bytes,
            16,
        )?;
        let logits_region =
            self.arena.region("qwen36_outside_logits", logits_bytes, 16)?;
        let stream_raw = self.stream.raw() as u64;

        unsafe {
            rvllm_fused::FusedRmsnormFp8QuantLaunch {
                num_tokens,
                hidden,
                eps,
            }
            .launch(
                self.outside_kernels.fn_fused_rmsnorm_fp8_quant,
                hidden_fp8_region.device_ptr(),
                hidden_scale_region.device_ptr(),
                hidden_region.device_ptr(),
                self.model.outside.final_norm.offset_bytes,
                stream_raw,
            )?;
        }

        #[cfg(feature = "cuda")]
        unsafe {
            self.cublaslt.fp8_gemm(
                hidden_fp8_region.device_ptr(),
                self.model.outside.lm_head_fp8.offset_bytes,
                logits_region.device_ptr(),
                num_tokens as i32,
                vocab as i32,
                hidden as i32,
                hidden_scale_region.device_ptr(),
                self.model.outside.lm_head_fp8.scale_ptr,
                stream_raw,
            )?;
        }
        self.stream.fence()?;

        // DtoH only the LAST token's logits row (the argmax output is
        // the prediction for the next token).
        let logits_row_bytes = (vocab as usize) * 2;
        let last_offset = (last_idx as u64) * (logits_row_bytes as u64);
        let mut logits_row_f16 = vec![0u8; logits_row_bytes];
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoH_v2(
                logits_row_f16.as_mut_ptr() as *mut _,
                logits_region.device_ptr() + last_offset,
                logits_row_bytes,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "forward_outside_only DtoH",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }

        let mut best_logit = f32::NEG_INFINITY;
        let mut best_token: i32 = -1;
        for v in 0..vocab as usize {
            let bits = u16::from_le_bytes([
                logits_row_f16[v * 2],
                logits_row_f16[v * 2 + 1],
            ]);
            let l = f16_bits_to_f32(bits);
            if l > best_logit {
                best_logit = l;
                best_token = v as i32;
            }
        }
        Ok(best_token)
    }

    /// Phase 4c/4d probe: chain Q + K + V projection launches against
    /// the first full-attention layer's blockwise FP8 weights. Uses a
    /// synthetic all-ones f16 input row to keep the smoke self-
    /// contained — the kernel chain is real, the values aren't.
    /// Validates:
    ///   - blockwise FP8 GEMV ABI consumes per-layer weight + [N/128,
    ///     K/128] f32 blockscale for all three projection roles.
    ///   - q_proj / k_proj / v_proj device pointers + blockscales are
    ///     populated correctly by the qwen36 loader.
    ///   - shape relations: q_proj outputs `2 * num_heads * head_dim`
    ///     (Q + per-head gate concat for `attn_output_gate=true`);
    ///     k_proj / v_proj output `num_kv_heads * head_dim`.
    pub fn forward_layer3_qkv_probe(&self) -> Result<()> {
        let layer_idx = match self.arch.base.layer_types.iter().position(|t| {
            matches!(t, rvllm_loader::LayerAttnType::Full)
        }) {
            Some(i) => i,
            None => {
                eprintln!("[qwen36] forward_layer3_qkv_probe: no full-attention layer found, skipping");
                return Ok(());
            }
        };
        let layer = match &self.model.layers[layer_idx].attn {
            rvllm_loader::qwen36_weights::Qwen36LayerAttn::Full(l) => l,
            _ => {
                eprintln!("[qwen36] forward_layer3_qkv_probe: layer {layer_idx} is not Full");
                return Ok(());
            }
        };
        let kernel = match self.outside_kernels.fn_fp8_gemv_wpr_native_f16in {
            Some(k) => k,
            None => {
                eprintln!("[qwen36] forward_layer3_qkv_probe: f16in GEMV kernel unavailable on this arch, skipping");
                return Ok(());
            }
        };
        let hidden = self.arch.base.hidden_size as u32;
        let m: u32 = 1;

        // Synthetic f16 all-ones input — h(i) = 1.0 (f16 1.0 = 0x3c00).
        let one_bits = 0x3c00u16.to_le_bytes();
        let mut input_bytes = Vec::with_capacity((hidden as usize) * 2);
        for _ in 0..hidden {
            input_bytes.extend_from_slice(&one_bits);
        }
        let in_region = self
            .arena
            .region("qwen36_l3qkv_in", input_bytes.len(), 16)?;
        unsafe { in_region.copy_from_host(&input_bytes)? };

        // Closure: run one projection role + return the first 4 f16
        // output values as f32 for logging.
        let project = |name: &'static str,
                       region_name: &'static str,
                       weight: &rvllm_loader::weights::Fp8Weight|
         -> Result<[f32; 4]> {
            let blockscale_ptr = match weight.blockscale_ptr {
                Some(p) => p,
                None => {
                    return Err(rvllm_core::RvllmError::cuda(
                        "qwen36_qkv_probe missing blockscale_ptr",
                        rvllm_core::CudaErrorKind::Other,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            };
            let n = weight.shape[0] as u32;
            let k = weight.shape[1] as u32;
            let out_bytes = (m as usize) * (n as usize) * 2;
            let out_region = self.arena.region(region_name, out_bytes, 16)?;
            unsafe {
                rvllm_fused::gemma4_launcher::Fp8GemvF16InLaunch { m, n, k }.launch(
                    kernel,
                    out_region.device_ptr(),
                    weight.offset_bytes,
                    blockscale_ptr,
                    in_region.device_ptr(),
                    self.stream.raw() as u64,
                )?;
            }
            self.stream.fence()?;
            let mut probe = [0u8; 8];
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let rc = cuMemcpyDtoH_v2(
                    probe.as_mut_ptr() as *mut _,
                    out_region.device_ptr(),
                    probe.len(),
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "qwen36_qkv_probe DtoH",
                        rvllm_core::CudaErrorKind::MemcpyFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            }
            let _ = name;
            Ok([
                f16_bits_to_f32(u16::from_le_bytes([probe[0], probe[1]])),
                f16_bits_to_f32(u16::from_le_bytes([probe[2], probe[3]])),
                f16_bits_to_f32(u16::from_le_bytes([probe[4], probe[5]])),
                f16_bits_to_f32(u16::from_le_bytes([probe[6], probe[7]])),
            ])
        };

        let q = project("q", "qwen36_l3q_out", &layer.q_proj)?;
        let k = project("k", "qwen36_l3k_out", &layer.k_proj)?;
        let v = project("v", "qwen36_l3v_out", &layer.v_proj)?;
        eprintln!(
            "[qwen36] forward_layer3_qkv_probe: layer={layer_idx} \
             q={:?} ({:?}) k={:?} ({:?}) v={:?} ({:?})",
            q, layer.q_proj.shape,
            k, layer.k_proj.shape,
            v, layer.v_proj.shape,
        );

        // Phase 4e: Q-Norm + K-Norm. Qwen 3.6 ships per-head RMSNorm
        // weights separately (`q_norm [head_dim]`, `k_norm [head_dim]`)
        // — different from Gemma 4's fused QK-norm. We re-use the
        // generic `rmsnorm_inplace_f16` kernel with `num_tokens =
        // <heads>` rows of `hidden = head_dim`, which is exactly the
        // per-head pattern (each head's [head_dim] vector RMSNorm'd
        // against the same gamma).
        //
        // Smoke uses a fresh synthetic per-head all-twos input rather
        // than chaining off the previous Q/K out regions — keeps the
        // probe self-contained and proves the kernel + per-layer
        // q_norm / k_norm gamma weights work without depending on the
        // qkv-projection layout (which varies with `attn_output_gate`).
        let head_dim = self.arch.base.head_dim as u32;
        let num_heads = self.arch.base.num_attention_heads as u32;
        let num_kv_heads = self.arch.base.num_key_value_heads as u32;
        let eps = self.arch.base.rms_norm_eps;
        let two_bits = 0x4000u16.to_le_bytes(); // f16 2.0
        let mut q_in_bytes = Vec::with_capacity(
            (num_heads as usize) * (head_dim as usize) * 2,
        );
        for _ in 0..(num_heads as usize) * (head_dim as usize) {
            q_in_bytes.extend_from_slice(&two_bits);
        }
        let mut k_in_bytes = Vec::with_capacity(
            (num_kv_heads as usize) * (head_dim as usize) * 2,
        );
        for _ in 0..(num_kv_heads as usize) * (head_dim as usize) {
            k_in_bytes.extend_from_slice(&two_bits);
        }
        let q_norm_in_region =
            self.arena.region("qwen36_l3qnorm_in", q_in_bytes.len(), 16)?;
        let k_norm_in_region =
            self.arena.region("qwen36_l3knorm_in", k_in_bytes.len(), 16)?;
        unsafe {
            q_norm_in_region.copy_from_host(&q_in_bytes)?;
            k_norm_in_region.copy_from_host(&k_in_bytes)?;
        }

        // Per-head RMSNorm: num_tokens = num_heads, hidden = head_dim.
        let launch_norm = |x_ptr: u64,
                           gamma_ptr: u64,
                           heads: u32,
                           label: &'static str|
         -> Result<()> {
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let mut x = x_ptr;
                let mut g = gamma_ptr;
                let mut e = eps;
                let mut h = head_dim as i32;
                let args = [
                    (&mut x) as *mut u64 as *mut core::ffi::c_void,
                    (&mut g) as *mut u64 as *mut core::ffi::c_void,
                    (&mut e) as *mut f32 as *mut core::ffi::c_void,
                    (&mut h) as *mut i32 as *mut core::ffi::c_void,
                ];
                let block = head_dim.min(1024);
                let rc = cuLaunchKernel(
                    self.outside_kernels.fn_rmsnorm_inplace_f16.raw() as CUfunction,
                    heads, 1, 1,
                    block, 1, 1,
                    0,
                    self.stream.raw() as CUstream,
                    args.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        label,
                        rvllm_core::CudaErrorKind::LaunchFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            }
            Ok(())
        };

        launch_norm(
            q_norm_in_region.device_ptr(),
            layer.q_norm.offset_bytes,
            num_heads,
            "qwen36_q_norm",
        )?;
        launch_norm(
            k_norm_in_region.device_ptr(),
            layer.k_norm.offset_bytes,
            num_kv_heads,
            "qwen36_k_norm",
        )?;
        self.stream.fence()?;

        let mut q_probe = [0u8; 8];
        let mut k_probe = [0u8; 8];
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let _ = cuMemcpyDtoH_v2(
                q_probe.as_mut_ptr() as *mut _,
                q_norm_in_region.device_ptr(),
                q_probe.len(),
            );
            let _ = cuMemcpyDtoH_v2(
                k_probe.as_mut_ptr() as *mut _,
                k_norm_in_region.device_ptr(),
                k_probe.len(),
            );
        }
        let qn = [
            f16_bits_to_f32(u16::from_le_bytes([q_probe[0], q_probe[1]])),
            f16_bits_to_f32(u16::from_le_bytes([q_probe[2], q_probe[3]])),
            f16_bits_to_f32(u16::from_le_bytes([q_probe[4], q_probe[5]])),
            f16_bits_to_f32(u16::from_le_bytes([q_probe[6], q_probe[7]])),
        ];
        let kn = [
            f16_bits_to_f32(u16::from_le_bytes([k_probe[0], k_probe[1]])),
            f16_bits_to_f32(u16::from_le_bytes([k_probe[2], k_probe[3]])),
            f16_bits_to_f32(u16::from_le_bytes([k_probe[4], k_probe[5]])),
            f16_bits_to_f32(u16::from_le_bytes([k_probe[6], k_probe[7]])),
        ];
        eprintln!(
            "[qwen36] forward_layer3_qknorm: per-head rmsnorm (head_dim={head_dim}, eps={eps:.0e}) \
             q_norm[head0,0..4]={qn:?} k_norm[head0,0..4]={kn:?}"
        );
        Ok(())
    }

    /// Phase 4g probe: launch `fused_rope_partial_f16kv` against
    /// synthetic Q/K/V buffers and a small KV-cache stand-in. Validates:
    ///   - the partial RoPE kernel ABI (15 args: q_in/k_in/v_in/
    ///     q_out/k_cache/v_cache/cos/sin/positions/slot_mapping +
    ///     5 ints) matches what the qwen36 RoPE wiring will need.
    ///   - the RoPE cos/sin tables uploaded in Phase 4f are usable
    ///     by the kernel (kernel reads cos[pos*hd/2 + i] / sin
    ///     analogues, so a position-0 lookup hits cos=1.0/sin=0.0,
    ///     which means RoPE is identity for position 0 — the
    ///     post-rope Q values should equal the pre-rope Q values).
    /// MRoPE section dispatch (sections [11, 11, 10]) is a no-op
    /// for text-only mode at position 0; Phase 4h+ wires real
    /// positions from the input sequence.
    pub fn forward_layer3_rope_probe(&self) -> Result<()> {
        let head_dim = self.arch.base.head_dim as u32;
        let num_heads = self.arch.base.num_attention_heads as u32;
        let num_kv_heads = self.arch.base.num_key_value_heads as u32;
        // Qwen 3.6 partial_rotary_factor = 0.25 → rotary_dim = 64
        // (only the first 64 of the 256 head_dim values are rotated).
        let rotary_dim = (head_dim as f32 * 0.25) as u32;
        let num_tokens: u32 = 1;

        // Synthetic f16 all-twos for Q/K/V (per-head, all positions).
        let two_bits = 0x4000u16.to_le_bytes();
        let q_elems = (num_tokens as usize) * (num_heads as usize) * (head_dim as usize);
        let kv_elems =
            (num_tokens as usize) * (num_kv_heads as usize) * (head_dim as usize);
        let mut q_in_bytes = Vec::with_capacity(q_elems * 2);
        for _ in 0..q_elems {
            q_in_bytes.extend_from_slice(&two_bits);
        }
        let mut kv_in_bytes = Vec::with_capacity(kv_elems * 2);
        for _ in 0..kv_elems {
            kv_in_bytes.extend_from_slice(&two_bits);
        }

        let q_region = self.arena.region("qwen36_l3rope_q", q_in_bytes.len(), 16)?;
        let k_region = self.arena.region("qwen36_l3rope_k", kv_in_bytes.len(), 16)?;
        let v_region = self.arena.region("qwen36_l3rope_v", kv_in_bytes.len(), 16)?;
        unsafe {
            q_region.copy_from_host(&q_in_bytes)?;
            k_region.copy_from_host(&kv_in_bytes)?;
            v_region.copy_from_host(&kv_in_bytes)?;
        }
        // KV cache stand-in: 1 slot's worth, same byte size as one
        // [num_kv_heads, head_dim] vector. Real KV cache lives in
        // Phase 4h.
        let k_cache_region =
            self.arena.region("qwen36_l3rope_kc", kv_in_bytes.len(), 16)?;
        let v_cache_region =
            self.arena.region("qwen36_l3rope_vc", kv_in_bytes.len(), 16)?;

        // Position [0] + slot_mapping [0] (single-token, slot index 0).
        let zero_i32 = 0i32.to_le_bytes();
        let pos_region = self.arena.region("qwen36_l3rope_pos", 4, 16)?;
        let slot_region = self.arena.region("qwen36_l3rope_slot", 4, 16)?;
        unsafe {
            pos_region.copy_from_host(&zero_i32)?;
            slot_region.copy_from_host(&zero_i32)?;
        }

        // Launch fused_rope_partial_f16kv. Kernel sig (16 args from
        // gemma4_layer_exec.rs::rope_f16kv): q_in, k_in, v_in, q_out
        // (alias=q_in for in-place), k_cache, v_cache, cos, sin,
        // positions, slot_mapping, num_tokens, num_heads, num_kv_heads,
        // head_dim, rotary_dim.
        #[cfg(feature = "cuda")]
        unsafe {
            let mut q_in = q_region.device_ptr();
            let mut k_in = k_region.device_ptr();
            let mut v_in = v_region.device_ptr();
            let mut q_out = q_region.device_ptr(); // in-place
            let mut k_cache = k_cache_region.device_ptr();
            let mut v_cache = v_cache_region.device_ptr();
            let mut cos = self.rope_cos;
            let mut sin = self.rope_sin;
            let mut positions = pos_region.device_ptr();
            let mut slot_mapping = slot_region.device_ptr();
            let mut nt = num_tokens as i32;
            let mut nh = num_heads as i32;
            let mut nkvh = num_kv_heads as i32;
            let mut hd = head_dim as i32;
            let mut rd = rotary_dim as i32;
            let args = [
                (&mut q_in) as *mut u64 as *mut core::ffi::c_void,
                (&mut k_in) as *mut u64 as *mut core::ffi::c_void,
                (&mut v_in) as *mut u64 as *mut core::ffi::c_void,
                (&mut q_out) as *mut u64 as *mut core::ffi::c_void,
                (&mut k_cache) as *mut u64 as *mut core::ffi::c_void,
                (&mut v_cache) as *mut u64 as *mut core::ffi::c_void,
                (&mut cos) as *mut u64 as *mut core::ffi::c_void,
                (&mut sin) as *mut u64 as *mut core::ffi::c_void,
                (&mut positions) as *mut u64 as *mut core::ffi::c_void,
                (&mut slot_mapping) as *mut u64 as *mut core::ffi::c_void,
                (&mut nt) as *mut i32 as *mut core::ffi::c_void,
                (&mut nh) as *mut i32 as *mut core::ffi::c_void,
                (&mut nkvh) as *mut i32 as *mut core::ffi::c_void,
                (&mut hd) as *mut i32 as *mut core::ffi::c_void,
                (&mut rd) as *mut i32 as *mut core::ffi::c_void,
            ];
            let max_heads = num_heads.max(num_kv_heads);
            let grid = (num_tokens, max_heads, 1);
            let block = ((head_dim / 2).max(32), 1, 1);
            rvllm_fused::launch_raw(
                self.outside_kernels.fn_fused_rope_partial_f16kv,
                grid,
                block,
                0,
                self.stream.raw() as u64,
                &args,
            )?;
        }
        self.stream.fence()?;

        // DtoH first 4 f16 of post-rope Q. At position=0 RoPE should
        // be identity (cos=1, sin=0), so q_out[0..4] == 2.0 if the
        // kernel honours the position-0 contract.
        let mut probe = [0u8; 8];
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let _ = cuMemcpyDtoH_v2(
                probe.as_mut_ptr() as *mut _,
                q_region.device_ptr(),
                probe.len(),
            );
        }
        let q0 = f16_bits_to_f32(u16::from_le_bytes([probe[0], probe[1]]));
        let q1 = f16_bits_to_f32(u16::from_le_bytes([probe[2], probe[3]]));
        let q2 = f16_bits_to_f32(u16::from_le_bytes([probe[4], probe[5]]));
        let q3 = f16_bits_to_f32(u16::from_le_bytes([probe[6], probe[7]]));
        eprintln!(
            "[qwen36] forward_layer3_rope_probe: head_dim={head_dim} \
             rotary_dim={rotary_dim} (partial 0.25) pos=0 → \
             q_post_rope[head0,0..4]=[{q0:.3}, {q1:.3}, {q2:.3}, {q3:.3}] \
             (expected ≈ 2.0 at pos=0 — RoPE is identity there)"
        );
        Ok(())
    }

    /// Phase 4h probe: launch `flash_attention_2_decode_f16io_kernel`
    /// against a tiny paged f16 KV cache. Validates the 14-arg ABI of
    /// the f16-IO decode kernel works for Qwen's dims (num_heads=16,
    /// num_kv_heads=2, head_dim=256). Synthetic single-token input
    /// with `context_len = 1` (one slot of KV in the cache); attention
    /// reduces to `softmax(QK^T/sqrt(d))V` over a single key — the
    /// softmax weight is exactly 1.0, so output should equal V.
    pub fn forward_layer3_paged_attn_probe(&self) -> Result<()> {
        let head_dim = self.arch.base.head_dim as u32;
        let num_heads = self.arch.base.num_attention_heads as u32;
        let num_kv_heads = self.arch.base.num_key_value_heads as u32;
        let block_size: u32 = 16;
        let num_blocks: u32 = 1;
        let max_blocks_per_seq: u32 = 1;
        let num_seqs: u32 = 1;

        // Sized buffers (all f16 except block_tables/context_lens int).
        let q_bytes = (num_seqs as usize) * (num_heads as usize) * (head_dim as usize) * 2;
        let kv_cache_bytes =
            (num_blocks as usize) * (block_size as usize) * (num_kv_heads as usize)
                * (head_dim as usize) * 2;
        let out_bytes = q_bytes;

        // Q = all-twos, V = all-threes, K = all-ones (so softmax weight
        // simplifies but the output isn't trivially equal to V — lets
        // us see that the kernel actually composes Q·K^T·V correctly).
        let q_region = self.arena.region("qwen36_l3pa_q", q_bytes, 16)?;
        let k_cache_region =
            self.arena.region("qwen36_l3pa_kc", kv_cache_bytes, 16)?;
        let v_cache_region =
            self.arena.region("qwen36_l3pa_vc", kv_cache_bytes, 16)?;
        let out_region = self.arena.region("qwen36_l3pa_out", out_bytes, 16)?;

        let two_bits = 0x4000u16.to_le_bytes(); // f16 2.0
        let one_bits = 0x3c00u16.to_le_bytes(); // f16 1.0
        let three_bits = 0x4200u16.to_le_bytes(); // f16 3.0
        let mut q_init = Vec::with_capacity(q_bytes);
        for _ in 0..q_bytes / 2 {
            q_init.extend_from_slice(&two_bits);
        }
        let mut k_init = Vec::with_capacity(kv_cache_bytes);
        for _ in 0..kv_cache_bytes / 2 {
            k_init.extend_from_slice(&one_bits);
        }
        let mut v_init = Vec::with_capacity(kv_cache_bytes);
        for _ in 0..kv_cache_bytes / 2 {
            v_init.extend_from_slice(&three_bits);
        }
        unsafe {
            q_region.copy_from_host(&q_init)?;
            k_cache_region.copy_from_host(&k_init)?;
            v_cache_region.copy_from_host(&v_init)?;
        }

        // block_tables = [[0]] (seq 0 → block 0)
        // context_lens = [1] (one valid token in the cache)
        let zero_i32 = 0i32.to_le_bytes();
        let one_i32 = 1i32.to_le_bytes();
        let bt_region = self.arena.region("qwen36_l3pa_bt", 4, 16)?;
        let cl_region = self.arena.region("qwen36_l3pa_cl", 4, 16)?;
        unsafe {
            bt_region.copy_from_host(&zero_i32)?;
            cl_region.copy_from_host(&one_i32)?;
        }

        let scale = 1.0 / (head_dim as f32).sqrt();

        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            // FA2 decode kernel needs dynamic shared memory:
            //   2 * FA2_BC * hd * 4 (K/V tiles) + FA2_BC * 4 (max_logits) + warps * 4
            // Mirrors gemma4 decode.rs:290. For Qwen hd=256 this is
            // ~64 KiB which exceeds the 48 KiB static cap, so the
            // kernel needs `cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE)`
            // before the launch.
            const FA2_THREADS: i32 = 128;
            const FA2_BC: i32 = 32;
            let hd_i = head_dim as i32;
            let smem_bytes =
                2 * FA2_BC * hd_i * 4 + FA2_BC * 4 + (FA2_THREADS / 32) * 4;
            if smem_bytes as u32 >= 48 * 1024 {
                let rc = cuFuncSetAttribute(
                    self.outside_kernels
                        .fn_flash_attention_2_decode_f16io
                        .raw() as CUfunction,
                    CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    smem_bytes,
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "qwen36_l3pa cuFuncSetAttribute",
                        rvllm_core::CudaErrorKind::LaunchFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            }

            let mut output = out_region.device_ptr();
            let mut query = q_region.device_ptr();
            let mut key_cache = k_cache_region.device_ptr();
            let mut value_cache = v_cache_region.device_ptr();
            let mut block_tables = bt_region.device_ptr();
            let mut context_lens = cl_region.device_ptr();
            let mut scale_arg = scale;
            let mut nh = num_heads as i32;
            let mut nkvh = num_kv_heads as i32;
            let mut hd = head_dim as i32;
            let mut bs = block_size as i32;
            let mut mbps = max_blocks_per_seq as i32;
            let mut window: i32 = -1; // no sliding window
            let args = [
                (&mut output) as *mut u64 as *mut core::ffi::c_void,
                (&mut query) as *mut u64 as *mut core::ffi::c_void,
                (&mut key_cache) as *mut u64 as *mut core::ffi::c_void,
                (&mut value_cache) as *mut u64 as *mut core::ffi::c_void,
                (&mut block_tables) as *mut u64 as *mut core::ffi::c_void,
                (&mut context_lens) as *mut u64 as *mut core::ffi::c_void,
                (&mut scale_arg) as *mut f32 as *mut core::ffi::c_void,
                (&mut nh) as *mut i32 as *mut core::ffi::c_void,
                (&mut nkvh) as *mut i32 as *mut core::ffi::c_void,
                (&mut hd) as *mut i32 as *mut core::ffi::c_void,
                (&mut bs) as *mut i32 as *mut core::ffi::c_void,
                (&mut mbps) as *mut i32 as *mut core::ffi::c_void,
                (&mut window) as *mut i32 as *mut core::ffi::c_void,
            ];
            let rc = cuLaunchKernel(
                self.outside_kernels
                    .fn_flash_attention_2_decode_f16io
                    .raw() as CUfunction,
                num_seqs, num_heads, 1,
                FA2_THREADS as u32, 1, 1,
                smem_bytes as u32,
                self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36_l3pa flash_attention_2_decode_f16io",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        self.stream.fence()?;

        let mut probe = [0u8; 8];
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let _ = cuMemcpyDtoH_v2(
                probe.as_mut_ptr() as *mut _,
                out_region.device_ptr(),
                probe.len(),
            );
        }
        let o0 = f16_bits_to_f32(u16::from_le_bytes([probe[0], probe[1]]));
        let o1 = f16_bits_to_f32(u16::from_le_bytes([probe[2], probe[3]]));
        let o2 = f16_bits_to_f32(u16::from_le_bytes([probe[4], probe[5]]));
        let o3 = f16_bits_to_f32(u16::from_le_bytes([probe[6], probe[7]]));
        eprintln!(
            "[qwen36] forward_layer3_paged_attn_probe: heads={num_heads} \
             kv_heads={num_kv_heads} hd={head_dim} block_size={block_size} \
             ctx_len=1 → out[head0,0..4]=[{o0:.3}, {o1:.3}, {o2:.3}, {o3:.3}] \
             (expected ≈ 3.0 — single-key softmax weight is 1.0, output = V = 3.0)"
        );
        Ok(())
    }

    /// Phase 4i probe: launch o_proj (last step of the attention
    /// block). Same blockwise FP8 GEMV kernel as Q/K/V — only the
    /// shape changes: `[hidden=2048, n=num_heads*head_dim=4096]`. The
    /// input is the per-token attention output (concatenated across
    /// heads); the output is the residual contribution that feeds
    /// the post-attention residual + MoE block.
    pub fn forward_layer3_o_proj_probe(&self) -> Result<()> {
        let layer_idx = match self.arch.base.layer_types.iter().position(|t| {
            matches!(t, rvllm_loader::LayerAttnType::Full)
        }) {
            Some(i) => i,
            None => return Ok(()),
        };
        let layer = match &self.model.layers[layer_idx].attn {
            rvllm_loader::qwen36_weights::Qwen36LayerAttn::Full(l) => l,
            _ => return Ok(()),
        };
        let kernel = match self.outside_kernels.fn_fp8_gemv_wpr_native_f16in {
            Some(k) => k,
            None => return Ok(()),
        };
        let n = layer.o_proj.shape[0] as u32;
        let k = layer.o_proj.shape[1] as u32;
        let m: u32 = 1;
        let blockscale_ptr = match layer.o_proj.blockscale_ptr {
            Some(p) => p,
            None => return Ok(()),
        };

        // Synthetic f16 all-twos input matching o_proj's K dim.
        let two_bits = 0x4000u16.to_le_bytes();
        let mut input_bytes = Vec::with_capacity((k as usize) * 2);
        for _ in 0..k {
            input_bytes.extend_from_slice(&two_bits);
        }
        let in_region = self.arena.region("qwen36_l3op_in", input_bytes.len(), 16)?;
        unsafe { in_region.copy_from_host(&input_bytes)? };
        let out_bytes = (m as usize) * (n as usize) * 2;
        let out_region = self.arena.region("qwen36_l3op_out", out_bytes, 16)?;

        unsafe {
            rvllm_fused::gemma4_launcher::Fp8GemvF16InLaunch { m, n, k }.launch(
                kernel,
                out_region.device_ptr(),
                layer.o_proj.offset_bytes,
                blockscale_ptr,
                in_region.device_ptr(),
                self.stream.raw() as u64,
            )?;
        }
        self.stream.fence()?;

        let mut probe = [0u8; 8];
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let _ = cuMemcpyDtoH_v2(
                probe.as_mut_ptr() as *mut _,
                out_region.device_ptr(),
                probe.len(),
            );
        }
        let o0 = f16_bits_to_f32(u16::from_le_bytes([probe[0], probe[1]]));
        let o1 = f16_bits_to_f32(u16::from_le_bytes([probe[2], probe[3]]));
        let o2 = f16_bits_to_f32(u16::from_le_bytes([probe[4], probe[5]]));
        let o3 = f16_bits_to_f32(u16::from_le_bytes([probe[6], probe[7]]));
        eprintln!(
            "[qwen36] forward_layer3_o_proj_probe: o_proj=[{n}, {k}] → \
             out[0..4]=[{o0:.3}, {o1:.3}, {o2:.3}, {o3:.3}] \
             (blockwise FP8 GEMV o_proj against synthetic f16 attn-out)"
        );
        Ok(())
    }

    /// Phase 4j probe: launch the new `sigmoid_mul_f16_kernel` with
    /// gate_logits = 0.0 (sigmoid(0) = 0.5) and values = 4.0, so the
    /// expected output is 2.0 across the buffer. Validates the new
    /// CUDA kernel built + the launch ABI.
    pub fn forward_layer3_attn_gate_probe(&self) -> Result<()> {
        let head_dim = self.arch.base.head_dim as u32;
        let num_heads = self.arch.base.num_attention_heads as u32;
        let n = num_heads * head_dim;

        // values = 4.0 (f16 0x4400), gate_logits = 0.0 (f16 0x0000).
        let four_bits = 0x4400u16.to_le_bytes();
        let zero_bits = 0x0000u16.to_le_bytes();
        let mut v_bytes = Vec::with_capacity((n as usize) * 2);
        let mut g_bytes = Vec::with_capacity((n as usize) * 2);
        for _ in 0..n {
            v_bytes.extend_from_slice(&four_bits);
            g_bytes.extend_from_slice(&zero_bits);
        }
        let v_region = self.arena.region("qwen36_l3sg_v", v_bytes.len(), 16)?;
        let g_region = self.arena.region("qwen36_l3sg_g", g_bytes.len(), 16)?;
        let o_region = self.arena.region("qwen36_l3sg_o", v_bytes.len(), 16)?;
        unsafe {
            v_region.copy_from_host(&v_bytes)?;
            g_region.copy_from_host(&g_bytes)?;
        }

        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let mut output = o_region.device_ptr();
            let mut values = v_region.device_ptr();
            let mut gate = g_region.device_ptr();
            let mut nn = n as i32;
            let args = [
                (&mut output) as *mut u64 as *mut core::ffi::c_void,
                (&mut values) as *mut u64 as *mut core::ffi::c_void,
                (&mut gate) as *mut u64 as *mut core::ffi::c_void,
                (&mut nn) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 256;
            let grid = (n + block - 1) / block;
            let rc = cuLaunchKernel(
                self.outside_kernels.fn_sigmoid_mul_f16.raw() as CUfunction,
                grid, 1, 1,
                block, 1, 1,
                0,
                self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36_l3sg sigmoid_mul_f16",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        self.stream.fence()?;

        let mut probe = [0u8; 8];
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let _ = cuMemcpyDtoH_v2(
                probe.as_mut_ptr() as *mut _,
                o_region.device_ptr(),
                probe.len(),
            );
        }
        let o0 = f16_bits_to_f32(u16::from_le_bytes([probe[0], probe[1]]));
        let o1 = f16_bits_to_f32(u16::from_le_bytes([probe[2], probe[3]]));
        let o2 = f16_bits_to_f32(u16::from_le_bytes([probe[4], probe[5]]));
        let o3 = f16_bits_to_f32(u16::from_le_bytes([probe[6], probe[7]]));
        eprintln!(
            "[qwen36] forward_layer3_attn_gate_probe: n={n} \
             values=4.0 gate_logits=0.0 → out[0..4]=[{o0:.3}, {o1:.3}, \
             {o2:.3}, {o3:.3}] (expected ≈ 2.0 — sigmoid(0)=0.5, \
             4.0×0.5=2.0)"
        );
        Ok(())
    }

    /// Phase 4k probe: launch the SHARED-EXPERT gate_proj of the
    /// per-layer MoE block. Reuses the same blockwise FP8 GEMV kernel
    /// as Q/K/V/o_proj — proves the MoE block's `shared_expert.*`
    /// weight pointers + blockscales are populated correctly by the
    /// Phase-2b loader. The 256 routed experts (top-8 dispatch) and
    /// the bf16 router (`mlp.gate`) are deferred to Phase 4l/4m where
    /// the per-token expert selection + grouped-GEMM dispatch land.
    pub fn forward_layer3_moe_shared_probe(&self) -> Result<()> {
        let layer_idx = match self.arch.base.layer_types.iter().position(|t| {
            matches!(t, rvllm_loader::LayerAttnType::Full)
        }) {
            Some(i) => i,
            None => return Ok(()),
        };
        let moe = &self.model.layers[layer_idx].moe;
        let kernel = match self.outside_kernels.fn_fp8_gemv_wpr_native_f16in {
            Some(k) => k,
            None => return Ok(()),
        };
        let w = &moe.shared_expert_gate_proj;
        let blockscale_ptr = match w.blockscale_ptr {
            Some(p) => p,
            None => return Ok(()),
        };
        let n = w.shape[0] as u32;
        let k = w.shape[1] as u32;
        let m: u32 = 1;

        // Synthetic f16 all-twos input matching gate_proj's K dim
        // (= hidden_size = 2048).
        let two_bits = 0x4000u16.to_le_bytes();
        let mut input_bytes = Vec::with_capacity((k as usize) * 2);
        for _ in 0..k {
            input_bytes.extend_from_slice(&two_bits);
        }
        let in_region =
            self.arena.region("qwen36_l3moe_sh_in", input_bytes.len(), 16)?;
        unsafe { in_region.copy_from_host(&input_bytes)? };
        let out_bytes = (m as usize) * (n as usize) * 2;
        let out_region =
            self.arena.region("qwen36_l3moe_sh_out", out_bytes, 16)?;

        unsafe {
            rvllm_fused::gemma4_launcher::Fp8GemvF16InLaunch { m, n, k }.launch(
                kernel,
                out_region.device_ptr(),
                w.offset_bytes,
                blockscale_ptr,
                in_region.device_ptr(),
                self.stream.raw() as u64,
            )?;
        }
        self.stream.fence()?;

        let mut probe = [0u8; 8];
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let _ = cuMemcpyDtoH_v2(
                probe.as_mut_ptr() as *mut _,
                out_region.device_ptr(),
                probe.len(),
            );
        }
        let o0 = f16_bits_to_f32(u16::from_le_bytes([probe[0], probe[1]]));
        let o1 = f16_bits_to_f32(u16::from_le_bytes([probe[2], probe[3]]));
        let o2 = f16_bits_to_f32(u16::from_le_bytes([probe[4], probe[5]]));
        let o3 = f16_bits_to_f32(u16::from_le_bytes([probe[6], probe[7]]));
        eprintln!(
            "[qwen36] forward_layer3_moe_shared_probe: layer={layer_idx} \
             shared_expert.gate_proj=[{n}, {k}] → \
             out[0..4]=[{o0:.3}, {o1:.3}, {o2:.3}, {o3:.3}] \
             (blockwise FP8 GEMV against per-layer shared-expert weights)"
        );
        Ok(())
    }

    /// Phase 4l probe: router GEMV + top-8 selection on the host
    /// (no GPU launch). The router weight is small — `[256, 2048]`
    /// bf16 = 1 MiB — so a one-shot DtoH + CPU matmul on a synthetic
    /// hidden state is cheap enough for a smoke probe and avoids
    /// committing to a GPU bf16-GEMV kernel before the dispatch
    /// shape is finalized. Phase 4m wires this on-device once the
    /// per-token routing decision drives a real grouped-expert
    /// dispatch.
    ///
    /// The MoE block also stores the router weight as f16 (after the
    /// loader's bf16→f16 upload), so we DtoH the f16 buffer directly.
    pub fn forward_layer3_router_probe(&self) -> Result<()> {
        let layer_idx = match self.arch.base.layer_types.iter().position(|t| {
            matches!(t, rvllm_loader::LayerAttnType::Full)
        }) {
            Some(i) => i,
            None => return Ok(()),
        };
        let moe = &self.model.layers[layer_idx].moe;
        let hidden = self.arch.base.hidden_size as usize;
        let num_experts = self.arch.num_experts;
        let top_k = self.arch.num_experts_per_tok;

        // The router weight is stored as f16 [num_experts, hidden]
        // (loader converts bf16 → f16 in `LoadCtx::upload_f16`).
        let weight_bytes = num_experts * hidden * 2;
        let mut router_f16 = vec![0u8; weight_bytes];
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoH_v2(
                router_f16.as_mut_ptr() as *mut _,
                moe.router.offset_bytes,
                weight_bytes,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36_l3router DtoH",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }

        // Synthetic hidden state h[i] = (i % 8) * 0.125 — small, varied,
        // not constant so the per-expert dot products differentiate.
        let hidden_state: Vec<f32> = (0..hidden)
            .map(|i| ((i % 8) as f32) * 0.125)
            .collect();

        // Host matmul: logits[e] = Σ_k router[e, k] * hidden[k].
        let mut logits = vec![0.0f32; num_experts];
        for e in 0..num_experts {
            let row_off = e * hidden * 2;
            let mut acc = 0.0f32;
            for k in 0..hidden {
                let bits = u16::from_le_bytes([
                    router_f16[row_off + k * 2],
                    router_f16[row_off + k * 2 + 1],
                ]);
                acc += f16_bits_to_f32(bits) * hidden_state[k];
            }
            logits[e] = acc;
        }

        // Top-k selection: partial sort over (logit, expert_idx).
        let mut indexed: Vec<(usize, f32)> =
            logits.iter().enumerate().map(|(i, &l)| (i, l)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top: Vec<(usize, f32)> = indexed.iter().take(top_k).copied().collect();

        // Softmax-normalize the top-k (Qwen's router head_mode).
        let max = top.iter().map(|(_, v)| *v).fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = top.iter().map(|(_, v)| (v - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let weights: Vec<f32> = exps.iter().map(|e| e / sum).collect();

        let pretty: Vec<String> = top
            .iter()
            .zip(weights.iter())
            .map(|((e, l), w)| format!("e{e}({l:.3}/{w:.3})"))
            .collect();
        eprintln!(
            "[qwen36] forward_layer3_router_probe: layer={layer_idx} \
             num_experts={num_experts} top_k={top_k} → \
             selected: {}",
            pretty.join(", ")
        );
        Ok(())
    }

    /// Phase 4m probe: full SwiGLU FFN for ONE routed expert.
    /// Chain: input f16 → expert0.gate_proj → expert0.up_proj →
    ///        host(silu(gate)*up) → expert0.down_proj → out f16.
    ///
    /// The host silu*mul step is a stand-in until a `silu_mul_f16`
    /// device kernel lands (would mirror Phase 4j's `sigmoid_mul_f16`
    /// — both fall in the same "tiny new f16 element-wise" CUDA cost).
    /// Validates the fused-experts arena layout: expert e's weight
    /// slice begins at `experts_*_proj_fused.offset_bytes +
    /// e * per_expert_fp8_bytes` and the blockscale slice at
    /// `blockscale_ptr + e * per_expert_blockscale_f32_bytes`. For
    /// e=0 both offsets are zero, which is the simplest-cut probe.
    pub fn forward_layer3_routed_expert_probe(&self) -> Result<()> {
        let layer_idx = match self.arch.base.layer_types.iter().position(|t| {
            matches!(t, rvllm_loader::LayerAttnType::Full)
        }) {
            Some(i) => i,
            None => return Ok(()),
        };
        let moe = &self.model.layers[layer_idx].moe;
        let kernel = match self.outside_kernels.fn_fp8_gemv_wpr_native_f16in {
            Some(k) => k,
            None => return Ok(()),
        };

        // experts_gate_proj_fused.shape = [num_experts, N, K]
        // For Qwen 3.6: N = moe_intermediate_size = 512, K = hidden = 2048.
        let gate_w = &moe.experts_gate_proj_fused;
        let up_w = &moe.experts_up_proj_fused;
        let down_w = &moe.experts_down_proj_fused;
        let n_int = gate_w.shape[1] as u32; // 512
        let k_in = gate_w.shape[2] as u32; // 2048
        let n_down = down_w.shape[1] as u32; // 2048
        let k_down = down_w.shape[2] as u32; // 512
        let m: u32 = 1;

        // Synthetic f16 all-twos input over the hidden dim.
        let two_bits = 0x4000u16.to_le_bytes();
        let mut input_bytes = Vec::with_capacity((k_in as usize) * 2);
        for _ in 0..k_in {
            input_bytes.extend_from_slice(&two_bits);
        }
        let in_region =
            self.arena.region("qwen36_l3rex_in", input_bytes.len(), 16)?;
        unsafe { in_region.copy_from_host(&input_bytes)? };

        // Outputs of gate / up are size N_int = 512 each (f16).
        let mid_bytes = (n_int as usize) * 2;
        let gate_out_region =
            self.arena.region("qwen36_l3rex_g", mid_bytes, 16)?;
        let up_out_region =
            self.arena.region("qwen36_l3rex_u", mid_bytes, 16)?;
        let silu_mul_region =
            self.arena.region("qwen36_l3rex_silu", mid_bytes, 16)?;
        let down_bytes = (n_down as usize) * 2;
        let down_out_region =
            self.arena.region("qwen36_l3rex_o", down_bytes, 16)?;

        // Expert 0's weight slice begins at the fused region start.
        // For e>0: weight_ptr += e * (N * K), blockscale_ptr +=
        //          e * (N/128 * K/128 * sizeof(f32)).
        let gate_blockscale = match gate_w.blockscale_ptr {
            Some(p) => p,
            None => return Ok(()),
        };
        let up_blockscale = match up_w.blockscale_ptr {
            Some(p) => p,
            None => return Ok(()),
        };
        let down_blockscale = match down_w.blockscale_ptr {
            Some(p) => p,
            None => return Ok(()),
        };

        unsafe {
            rvllm_fused::gemma4_launcher::Fp8GemvF16InLaunch {
                m,
                n: n_int,
                k: k_in,
            }
            .launch(
                kernel,
                gate_out_region.device_ptr(),
                gate_w.offset_bytes,
                gate_blockscale,
                in_region.device_ptr(),
                self.stream.raw() as u64,
            )?;
            rvllm_fused::gemma4_launcher::Fp8GemvF16InLaunch {
                m,
                n: n_int,
                k: k_in,
            }
            .launch(
                kernel,
                up_out_region.device_ptr(),
                up_w.offset_bytes,
                up_blockscale,
                in_region.device_ptr(),
                self.stream.raw() as u64,
            )?;
        }
        self.stream.fence()?;

        // Host-side silu(gate) * up: DtoH both, compute, HtoD result.
        let mut gate_host = vec![0u8; mid_bytes];
        let mut up_host = vec![0u8; mid_bytes];
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let _ = cuMemcpyDtoH_v2(
                gate_host.as_mut_ptr() as *mut _,
                gate_out_region.device_ptr(),
                mid_bytes,
            );
            let _ = cuMemcpyDtoH_v2(
                up_host.as_mut_ptr() as *mut _,
                up_out_region.device_ptr(),
                mid_bytes,
            );
        }
        let mut silu_mul_host = Vec::with_capacity(mid_bytes);
        for i in 0..(n_int as usize) {
            let g = f16_bits_to_f32(u16::from_le_bytes([
                gate_host[i * 2],
                gate_host[i * 2 + 1],
            ]));
            let u = f16_bits_to_f32(u16::from_le_bytes([
                up_host[i * 2],
                up_host[i * 2 + 1],
            ]));
            // SiLU: x · sigmoid(x) = x / (1 + exp(-x))
            let silu_g = g / (1.0f32 + (-g).exp());
            let v = silu_g * u;
            silu_mul_host.extend_from_slice(&f32_to_f16_bits(v).to_le_bytes());
        }
        unsafe { silu_mul_region.copy_from_host(&silu_mul_host)? };

        // down_proj: mid [N=512] → out [hidden=2048]
        unsafe {
            rvllm_fused::gemma4_launcher::Fp8GemvF16InLaunch {
                m,
                n: n_down,
                k: k_down,
            }
            .launch(
                kernel,
                down_out_region.device_ptr(),
                down_w.offset_bytes,
                down_blockscale,
                silu_mul_region.device_ptr(),
                self.stream.raw() as u64,
            )?;
        }
        self.stream.fence()?;

        let mut probe = [0u8; 8];
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let _ = cuMemcpyDtoH_v2(
                probe.as_mut_ptr() as *mut _,
                down_out_region.device_ptr(),
                probe.len(),
            );
        }
        let o0 = f16_bits_to_f32(u16::from_le_bytes([probe[0], probe[1]]));
        let o1 = f16_bits_to_f32(u16::from_le_bytes([probe[2], probe[3]]));
        let o2 = f16_bits_to_f32(u16::from_le_bytes([probe[4], probe[5]]));
        let o3 = f16_bits_to_f32(u16::from_le_bytes([probe[6], probe[7]]));

        // Log a sample of the silu*mul intermediate too, so we see
        // both stages of the SwiGLU FFN.
        let m0 = f16_bits_to_f32(u16::from_le_bytes([silu_mul_host[0], silu_mul_host[1]]));
        let m1 = f16_bits_to_f32(u16::from_le_bytes([silu_mul_host[2], silu_mul_host[3]]));
        let m2 = f16_bits_to_f32(u16::from_le_bytes([silu_mul_host[4], silu_mul_host[5]]));
        let m3 = f16_bits_to_f32(u16::from_le_bytes([silu_mul_host[6], silu_mul_host[7]]));

        eprintln!(
            "[qwen36] forward_layer3_routed_expert_probe: layer={layer_idx} \
             expert=0 gate/up=[{n_int}, {k_in}] down=[{n_down}, {k_down}] \
             silu_mul[0..4]=[{m0:.3}, {m1:.3}, {m2:.3}, {m3:.3}] \
             out[0..4]=[{o0:.3}, {o1:.3}, {o2:.3}, {o3:.3}] \
             (full SwiGLU FFN: gate → up → silu·mul (host) → down)"
        );
        Ok(())
    }

    /// Phase 4n: full MoE block for one token. Combines:
    ///   1. router → top-8 expert indices + softmax weights (CPU,
    ///      same as Phase 4l)
    ///   2. for each chosen expert: full SwiGLU FFN at the expert's
    ///      slice of the fused arena regions (extending Phase 4m to
    ///      non-zero expert offsets)
    ///   3. weighted sum of the 8 routed expert outputs (host f32)
    ///   4. shared expert: full SwiGLU FFN
    ///   5. shared_expert_gate sigmoid (single-element scalar) ·
    ///      shared output
    ///   6. final = routed_sum + gated_shared
    pub fn forward_layer3_full_moe_probe(&self) -> Result<()> {
        let layer_idx = match self.arch.base.layer_types.iter().position(|t| {
            matches!(t, rvllm_loader::LayerAttnType::Full)
        }) {
            Some(i) => i,
            None => return Ok(()),
        };
        let moe = &self.model.layers[layer_idx].moe;
        let kernel = match self.outside_kernels.fn_fp8_gemv_wpr_native_f16in {
            Some(k) => k,
            None => return Ok(()),
        };
        let hidden = self.arch.base.hidden_size as usize;
        let n_int = moe.experts_gate_proj_fused.shape[1] as u32; // 512
        let k_in = moe.experts_gate_proj_fused.shape[2] as u32; // 2048
        let n_down = moe.experts_down_proj_fused.shape[1] as u32; // 2048
        let k_down = moe.experts_down_proj_fused.shape[2] as u32; // 512
        let num_experts = self.arch.num_experts;
        let top_k = self.arch.num_experts_per_tok;
        let m: u32 = 1;

        // ---- 1. router → top-8 ---------------------------------------
        let weight_bytes = num_experts * hidden * 2;
        let mut router_f16 = vec![0u8; weight_bytes];
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let _ = cuMemcpyDtoH_v2(
                router_f16.as_mut_ptr() as *mut _,
                moe.router.offset_bytes,
                weight_bytes,
            );
        }
        let hidden_state: Vec<f32> = (0..hidden)
            .map(|i| ((i % 8) as f32) * 0.125)
            .collect();
        let mut logits = vec![0.0f32; num_experts];
        for e in 0..num_experts {
            let row = e * hidden * 2;
            let mut acc = 0.0f32;
            for k in 0..hidden {
                let bits = u16::from_le_bytes([
                    router_f16[row + k * 2],
                    router_f16[row + k * 2 + 1],
                ]);
                acc += f16_bits_to_f32(bits) * hidden_state[k];
            }
            logits[e] = acc;
        }
        let mut indexed: Vec<(usize, f32)> =
            logits.iter().enumerate().map(|(i, &l)| (i, l)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top: Vec<(usize, f32)> = indexed.iter().take(top_k).copied().collect();
        let max = top.iter().map(|(_, v)| *v).fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = top.iter().map(|(_, v)| (v - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let weights: Vec<f32> = exps.iter().map(|e| e / sum).collect();

        // ---- 2-3. routed experts: FFN at each top-k offset, weighted sum
        // hidden_state f16 input region.
        let mut input_bytes = Vec::with_capacity(hidden * 2);
        for h in &hidden_state {
            input_bytes.extend_from_slice(&f32_to_f16_bits(*h).to_le_bytes());
        }
        let in_region = self.arena.region("qwen36_l3moe_in", input_bytes.len(), 16)?;
        unsafe { in_region.copy_from_host(&input_bytes)? };

        let mid_bytes = (n_int as usize) * 2;
        let down_bytes = (n_down as usize) * 2;
        let gate_region = self.arena.region("qwen36_l3moe_g", mid_bytes, 16)?;
        let up_region = self.arena.region("qwen36_l3moe_u", mid_bytes, 16)?;
        let silu_region = self.arena.region("qwen36_l3moe_s", mid_bytes, 16)?;
        let down_region = self.arena.region("qwen36_l3moe_d", down_bytes, 16)?;

        // Per-expert byte strides into the fused regions.
        let int_per_expert_w = (n_int as u64) * (k_in as u64); // FP8: 1 byte/elem
        let int_per_expert_bs =
            ((n_int as u64) / 128) * ((k_in as u64) / 128) * 4; // f32 blockscale
        let down_per_expert_w = (n_down as u64) * (k_down as u64);
        let down_per_expert_bs =
            ((n_down as u64) / 128) * ((k_down as u64) / 128) * 4;

        let gate_bs = moe.experts_gate_proj_fused.blockscale_ptr.unwrap_or(0);
        let up_bs = moe.experts_up_proj_fused.blockscale_ptr.unwrap_or(0);
        let down_bs = moe.experts_down_proj_fused.blockscale_ptr.unwrap_or(0);
        if gate_bs == 0 || up_bs == 0 || down_bs == 0 {
            return Ok(());
        }

        let mut routed_sum = vec![0.0f32; n_down as usize];

        for ((e_idx, _logit), w) in top.iter().zip(weights.iter()) {
            let e = *e_idx as u64;
            // gate
            unsafe {
                rvllm_fused::gemma4_launcher::Fp8GemvF16InLaunch {
                    m, n: n_int, k: k_in,
                }.launch(
                    kernel,
                    gate_region.device_ptr(),
                    moe.experts_gate_proj_fused.offset_bytes + e * int_per_expert_w,
                    gate_bs + e * int_per_expert_bs,
                    in_region.device_ptr(),
                    self.stream.raw() as u64,
                )?;
                rvllm_fused::gemma4_launcher::Fp8GemvF16InLaunch {
                    m, n: n_int, k: k_in,
                }.launch(
                    kernel,
                    up_region.device_ptr(),
                    moe.experts_up_proj_fused.offset_bytes + e * int_per_expert_w,
                    up_bs + e * int_per_expert_bs,
                    in_region.device_ptr(),
                    self.stream.raw() as u64,
                )?;
            }
            self.stream.fence()?;
            // host silu·mul
            let mut g_host = vec![0u8; mid_bytes];
            let mut u_host = vec![0u8; mid_bytes];
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let _ = cuMemcpyDtoH_v2(g_host.as_mut_ptr() as *mut _, gate_region.device_ptr(), mid_bytes);
                let _ = cuMemcpyDtoH_v2(u_host.as_mut_ptr() as *mut _, up_region.device_ptr(), mid_bytes);
            }
            let mut silu_host = Vec::with_capacity(mid_bytes);
            for i in 0..(n_int as usize) {
                let g = f16_bits_to_f32(u16::from_le_bytes([g_host[i * 2], g_host[i * 2 + 1]]));
                let u = f16_bits_to_f32(u16::from_le_bytes([u_host[i * 2], u_host[i * 2 + 1]]));
                let s = g / (1.0f32 + (-g).exp());
                silu_host.extend_from_slice(&f32_to_f16_bits(s * u).to_le_bytes());
            }
            unsafe { silu_region.copy_from_host(&silu_host)? };

            unsafe {
                rvllm_fused::gemma4_launcher::Fp8GemvF16InLaunch {
                    m, n: n_down, k: k_down,
                }.launch(
                    kernel,
                    down_region.device_ptr(),
                    moe.experts_down_proj_fused.offset_bytes + e * down_per_expert_w,
                    down_bs + e * down_per_expert_bs,
                    silu_region.device_ptr(),
                    self.stream.raw() as u64,
                )?;
            }
            self.stream.fence()?;

            let mut d_host = vec![0u8; down_bytes];
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let _ = cuMemcpyDtoH_v2(d_host.as_mut_ptr() as *mut _, down_region.device_ptr(), down_bytes);
            }
            for i in 0..(n_down as usize) {
                let v = f16_bits_to_f32(u16::from_le_bytes([d_host[i * 2], d_host[i * 2 + 1]]));
                routed_sum[i] += v * w;
            }
        }

        // ---- 4. shared expert: full FFN ----
        let sh_gate_bs = moe.shared_expert_gate_proj.blockscale_ptr.unwrap_or(0);
        let sh_up_bs = moe.shared_expert_up_proj.blockscale_ptr.unwrap_or(0);
        let sh_down_bs = moe.shared_expert_down_proj.blockscale_ptr.unwrap_or(0);
        if sh_gate_bs != 0 && sh_up_bs != 0 && sh_down_bs != 0 {
            unsafe {
                rvllm_fused::gemma4_launcher::Fp8GemvF16InLaunch {
                    m, n: n_int, k: k_in,
                }.launch(
                    kernel,
                    gate_region.device_ptr(),
                    moe.shared_expert_gate_proj.offset_bytes,
                    sh_gate_bs,
                    in_region.device_ptr(),
                    self.stream.raw() as u64,
                )?;
                rvllm_fused::gemma4_launcher::Fp8GemvF16InLaunch {
                    m, n: n_int, k: k_in,
                }.launch(
                    kernel,
                    up_region.device_ptr(),
                    moe.shared_expert_up_proj.offset_bytes,
                    sh_up_bs,
                    in_region.device_ptr(),
                    self.stream.raw() as u64,
                )?;
            }
            self.stream.fence()?;
            let mut g_host = vec![0u8; mid_bytes];
            let mut u_host = vec![0u8; mid_bytes];
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let _ = cuMemcpyDtoH_v2(g_host.as_mut_ptr() as *mut _, gate_region.device_ptr(), mid_bytes);
                let _ = cuMemcpyDtoH_v2(u_host.as_mut_ptr() as *mut _, up_region.device_ptr(), mid_bytes);
            }
            let mut silu_host = Vec::with_capacity(mid_bytes);
            for i in 0..(n_int as usize) {
                let g = f16_bits_to_f32(u16::from_le_bytes([g_host[i * 2], g_host[i * 2 + 1]]));
                let u = f16_bits_to_f32(u16::from_le_bytes([u_host[i * 2], u_host[i * 2 + 1]]));
                let s = g / (1.0f32 + (-g).exp());
                silu_host.extend_from_slice(&f32_to_f16_bits(s * u).to_le_bytes());
            }
            unsafe { silu_region.copy_from_host(&silu_host)? };
            unsafe {
                rvllm_fused::gemma4_launcher::Fp8GemvF16InLaunch {
                    m, n: n_down, k: k_down,
                }.launch(
                    kernel,
                    down_region.device_ptr(),
                    moe.shared_expert_down_proj.offset_bytes,
                    sh_down_bs,
                    silu_region.device_ptr(),
                    self.stream.raw() as u64,
                )?;
            }
            self.stream.fence()?;
            let mut sh_host = vec![0u8; down_bytes];
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let _ = cuMemcpyDtoH_v2(sh_host.as_mut_ptr() as *mut _, down_region.device_ptr(), down_bytes);
            }
            // shared_expert_gate: single-element bf16-as-f16 → sigmoid
            let mut sh_gate_host = [0u8; 2];
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let _ = cuMemcpyDtoH_v2(
                    sh_gate_host.as_mut_ptr() as *mut _,
                    moe.shared_expert_gate_logit.offset_bytes,
                    2,
                );
            }
            let g_logit = f16_bits_to_f32(u16::from_le_bytes(sh_gate_host));
            let g_sigmoid = 1.0f32 / (1.0f32 + (-g_logit).exp());
            for i in 0..(n_down as usize) {
                let v = f16_bits_to_f32(u16::from_le_bytes([sh_host[i * 2], sh_host[i * 2 + 1]]));
                routed_sum[i] += v * g_sigmoid;
            }
        }

        eprintln!(
            "[qwen36] forward_layer3_full_moe_probe: layer={layer_idx} \
             top_k={top_k}/{num_experts} → final[0..4]=[{:.3}, {:.3}, \
             {:.3}, {:.3}] (routed-weighted-sum + sigmoid·shared)",
            routed_sum[0], routed_sum[1], routed_sum[2], routed_sum[3],
        );
        Ok(())
    }

    /// Phase 4q probe: launch the new `causal_conv1d_f16_kernel`
    /// against linear-attn layer 0's `conv1d.weight [8192, 1, 4]`.
    /// Synthetic input chosen so the expected output is closed-form:
    /// input is all-ones in every channel, so out[t, c] = Σ_k w[c, 0, k]
    /// — i.e. each output equals the sum of that channel's 4 conv1d
    /// weight values. Probes the per-channel shape + ABI + first
    /// channel's actual weight sum.
    pub fn forward_layer0_conv1d_probe(&self) -> Result<()> {
        let layer_idx = match self.arch.base.layer_types.iter().position(|t| {
            matches!(t, rvllm_loader::LayerAttnType::Linear)
        }) {
            Some(i) => i,
            None => return Ok(()),
        };
        let la = match &self.model.layers[layer_idx].attn {
            rvllm_loader::qwen36_weights::Qwen36LayerAttn::Linear(l) => l,
            _ => return Ok(()),
        };
        // conv1d.shape = [channels, 1, ks] = [8192, 1, 4]
        let channels = la.conv1d.shape[0] as u32;
        let ks = la.conv1d.shape[2] as u32;
        let seq_len: u32 = 1;

        // Input: (seq_len + ks - 1, channels) all-ones → each output
        // channel = sum of its `ks` weight values.
        let one_bits = 0x3c00u16.to_le_bytes();
        let in_elems = ((seq_len + ks - 1) as usize) * (channels as usize);
        let mut input_bytes = Vec::with_capacity(in_elems * 2);
        for _ in 0..in_elems {
            input_bytes.extend_from_slice(&one_bits);
        }
        let in_region =
            self.arena.region("qwen36_l0c1_in", input_bytes.len(), 16)?;
        unsafe { in_region.copy_from_host(&input_bytes)? };

        let out_bytes = (seq_len as usize) * (channels as usize) * 2;
        let out_region = self.arena.region("qwen36_l0c1_out", out_bytes, 16)?;

        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let mut output = out_region.device_ptr();
            let mut input = in_region.device_ptr();
            let mut weight = la.conv1d.offset_bytes;
            let mut sl = seq_len as i32;
            let mut ch = channels as i32;
            let mut k = ks as i32;
            let args = [
                (&mut output) as *mut u64 as *mut core::ffi::c_void,
                (&mut input) as *mut u64 as *mut core::ffi::c_void,
                (&mut weight) as *mut u64 as *mut core::ffi::c_void,
                (&mut sl) as *mut i32 as *mut core::ffi::c_void,
                (&mut ch) as *mut i32 as *mut core::ffi::c_void,
                (&mut k) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 256;
            let grid_x = (channels + block - 1) / block;
            let rc = cuLaunchKernel(
                self.outside_kernels.fn_causal_conv1d_f16.raw() as CUfunction,
                grid_x, seq_len, 1,
                block, 1, 1,
                0,
                self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36_l0c1 causal_conv1d_f16",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        self.stream.fence()?;

        // DtoH first 4 channels' output + the corresponding 4 weight
        // values for cross-check.
        let mut out_probe = [0u8; 8];
        let mut w_probe = [0u8; (4 * 4) * 2]; // 4 channels × 4 ks × 2 bytes
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let _ = cuMemcpyDtoH_v2(
                out_probe.as_mut_ptr() as *mut _,
                out_region.device_ptr(),
                out_probe.len(),
            );
            let _ = cuMemcpyDtoH_v2(
                w_probe.as_mut_ptr() as *mut _,
                la.conv1d.offset_bytes,
                w_probe.len(),
            );
        }
        let o: [f32; 4] = [
            f16_bits_to_f32(u16::from_le_bytes([out_probe[0], out_probe[1]])),
            f16_bits_to_f32(u16::from_le_bytes([out_probe[2], out_probe[3]])),
            f16_bits_to_f32(u16::from_le_bytes([out_probe[4], out_probe[5]])),
            f16_bits_to_f32(u16::from_le_bytes([out_probe[6], out_probe[7]])),
        ];
        // Expected: out[c] = Σ_k w[c, 0, k] for c=0..3.
        let mut expected = [0.0f32; 4];
        for c in 0..4 {
            let mut s = 0.0f32;
            for k in 0..4 {
                let off = (c * 4 + k) * 2;
                s += f16_bits_to_f32(u16::from_le_bytes([w_probe[off], w_probe[off + 1]]));
            }
            expected[c] = s;
        }
        eprintln!(
            "[qwen36] forward_layer0_conv1d_probe: layer={layer_idx} \
             channels={channels} ks={ks} seq_len={seq_len} \
             out[0..4]={o:?} expected={expected:?}"
        );
        Ok(())
    }

    /// Phase 4p probe: launch the two FP8 input projections of the
    /// linear-attn block (in_proj_qkv + in_proj_z) against the first
    /// linear-attention layer's weights. The blockwise FP8 GEMV
    /// kernel (Phase 4c onwards) is model-agnostic — this probe
    /// confirms it works for the linear-attn projection shapes
    /// `[8192, hidden]` (qkv: 4 × 2048 = q/k/v/extra concat) and
    /// `[4096, hidden]` (z gating stream) which differ from the
    /// full-attention layer's projections.
    ///
    /// The recurrent ssm-scan + conv1d + per-sequence state cache
    /// still need a custom kernel — this only exercises the FP8
    /// matmul entry points to the linear-attn block.
    pub fn forward_layer0_linear_in_proj_probe(&self) -> Result<()> {
        let layer_idx = match self.arch.base.layer_types.iter().position(|t| {
            matches!(t, rvllm_loader::LayerAttnType::Linear)
        }) {
            Some(i) => i,
            None => return Ok(()),
        };
        let la = match &self.model.layers[layer_idx].attn {
            rvllm_loader::qwen36_weights::Qwen36LayerAttn::Linear(l) => l,
            _ => return Ok(()),
        };
        let kernel = match self.outside_kernels.fn_fp8_gemv_wpr_native_f16in {
            Some(k) => k,
            None => return Ok(()),
        };
        let hidden = self.arch.base.hidden_size as u32;
        let m: u32 = 1;

        // Synthetic f16 all-twos input, hidden-sized.
        let two_bits = 0x4000u16.to_le_bytes();
        let mut input_bytes = Vec::with_capacity((hidden as usize) * 2);
        for _ in 0..hidden {
            input_bytes.extend_from_slice(&two_bits);
        }
        let in_region =
            self.arena.region("qwen36_l0lin_in", input_bytes.len(), 16)?;
        unsafe { in_region.copy_from_host(&input_bytes)? };

        // Closure to launch one FP8 projection role.
        let project = |w: &rvllm_loader::weights::Fp8Weight,
                       region_name: &'static str|
         -> Result<[f32; 4]> {
            let n = w.shape[0] as u32;
            let k = w.shape[1] as u32;
            let bs = match w.blockscale_ptr {
                Some(p) => p,
                None => {
                    return Err(rvllm_core::RvllmError::cuda(
                        "qwen36_l0lin missing blockscale",
                        rvllm_core::CudaErrorKind::Other,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            };
            let out_bytes = (m as usize) * (n as usize) * 2;
            let out_region = self.arena.region(region_name, out_bytes, 16)?;
            unsafe {
                rvllm_fused::gemma4_launcher::Fp8GemvF16InLaunch { m, n, k }.launch(
                    kernel,
                    out_region.device_ptr(),
                    w.offset_bytes,
                    bs,
                    in_region.device_ptr(),
                    self.stream.raw() as u64,
                )?;
            }
            self.stream.fence()?;
            let mut probe = [0u8; 8];
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let _ = cuMemcpyDtoH_v2(
                    probe.as_mut_ptr() as *mut _,
                    out_region.device_ptr(),
                    probe.len(),
                );
            }
            Ok([
                f16_bits_to_f32(u16::from_le_bytes([probe[0], probe[1]])),
                f16_bits_to_f32(u16::from_le_bytes([probe[2], probe[3]])),
                f16_bits_to_f32(u16::from_le_bytes([probe[4], probe[5]])),
                f16_bits_to_f32(u16::from_le_bytes([probe[6], probe[7]])),
            ])
        };

        let qkv = project(&la.in_proj_qkv, "qwen36_l0lin_qkv")?;
        let z = project(&la.in_proj_z, "qwen36_l0lin_z")?;
        eprintln!(
            "[qwen36] forward_layer0_linear_in_proj_probe: layer={layer_idx} \
             in_proj_qkv={:?} ({:?}) in_proj_z={:?} ({:?})",
            qkv, la.in_proj_qkv.shape,
            z, la.in_proj_z.shape,
        );
        Ok(())
    }

    /// Phase 4o probe (skeleton): linear-attn weight presence + shape
    /// check. Doesn't launch the recurrent kernel — Gated-DeltaNet
    /// state-space math (A_log + conv1d + dt_bias + ssm-scan) needs
    /// custom CUDA work that's ~500+ LOC of new kernel by itself.
    /// This phase verifies the loader's per-linear-layer pointers are
    /// reachable and shapes match the qwen3-next architecture so the
    /// future kernel implementation can pick up the right weights.
    pub fn forward_layer0_linear_attn_probe(&self) -> Result<()> {
        let layer_idx = match self.arch.base.layer_types.iter().position(|t| {
            matches!(t, rvllm_loader::LayerAttnType::Linear)
        }) {
            Some(i) => i,
            None => return Ok(()),
        };
        let la = match &self.model.layers[layer_idx].attn {
            rvllm_loader::qwen36_weights::Qwen36LayerAttn::Linear(l) => l,
            _ => return Ok(()),
        };
        eprintln!(
            "[qwen36] forward_layer0_linear_attn_probe: layer={layer_idx} \
             A_log={:?} dt_bias={:?} conv1d={:?} in_proj_a={:?} \
             in_proj_b={:?} in_proj_qkv={:?} in_proj_z={:?} norm={:?} \
             out_proj={:?} (Gated-DeltaNet recurrent kernel TODO — \
             needs new CUDA work for ssm-scan + per-sequence state cache)",
            la.a_log.shape, la.dt_bias.shape, la.conv1d.shape,
            la.in_proj_a.shape, la.in_proj_b.shape,
            la.in_proj_qkv.shape, la.in_proj_z.shape,
            la.norm.shape, la.out_proj.shape,
        );
        Ok(())
    }

    pub fn forward_outside_smoke(&self) -> Result<()> {
        let hidden = self.arch.base.hidden_size as u32;
        let vocab = self.arch.base.vocab_size as u32;
        // Hardcoded canary tokens: BOS-likely + a few mid-vocab IDs.
        // Real generation needs the tokenizer; we just want non-zero
        // embeddings for the kernel output.
        let token_ids: [i32; 4] = [1, 100, 1000, 10_000];
        let num_tokens = token_ids.len() as u32;

        // Allocate device regions: token IDs (i32) + hidden state (f16).
        let tokens_region = self.arena.region(
            "qwen36_smoke_tokens",
            std::mem::size_of_val(&token_ids),
            16,
        )?;
        let mut token_bytes = Vec::with_capacity(token_ids.len() * 4);
        for t in &token_ids {
            token_bytes.extend_from_slice(&t.to_le_bytes());
        }
        unsafe { tokens_region.copy_from_host(&token_bytes)? };

        let hidden_bytes = (num_tokens as usize) * (hidden as usize) * 2; // f16
        let hidden_region = self.arena.region("qwen36_smoke_hidden", hidden_bytes, 16)?;

        // Launch embedding_gather_f16 via the standard rvllm-fused
        // ABI used by Gemma 4.
        let launch = rvllm_fused::EmbeddingGatherLaunch {
            num_tokens,
            hidden,
            vocab,
        };
        unsafe {
            launch.launch(
                self.outside_kernels.fn_embedding_gather_f16,
                hidden_region.device_ptr(),
                self.model.outside.embed_tokens.offset_bytes,
                tokens_region.device_ptr(),
                self.stream.raw() as u64,
            )?;
        }

        // Phase 3g: fused (RMSNorm + FP8-quantize) → cuBLASLt
        // FP8 matmul against lm_head → DtoH logits → CPU argmax.
        // Mirrors Gemma 4's outside-only path (gemma4_bring_up.rs:2058).
        let eps = self.arch.base.rms_norm_eps;
        let hidden_fp8_bytes = (num_tokens as usize) * (hidden as usize); // 1 byte/elem
        let hidden_scale_bytes = (num_tokens as usize) * 4; // f32/token
        let logits_bytes = (num_tokens as usize) * (vocab as usize) * 2; // f16
        let hidden_fp8_region =
            self.arena.region("qwen36_smoke_hidden_fp8", hidden_fp8_bytes, 16)?;
        let hidden_scale_region =
            self.arena.region("qwen36_smoke_hidden_scale", hidden_scale_bytes, 16)?;
        let logits_region = self.arena.region("qwen36_smoke_logits", logits_bytes, 16)?;

        let stream_raw = self.stream.raw() as u64;

        unsafe {
            rvllm_fused::FusedRmsnormFp8QuantLaunch {
                num_tokens,
                hidden,
                eps,
            }
            .launch(
                self.outside_kernels.fn_fused_rmsnorm_fp8_quant,
                hidden_fp8_region.device_ptr(),
                hidden_scale_region.device_ptr(),
                hidden_region.device_ptr(),
                self.model.outside.final_norm.offset_bytes,
                stream_raw,
            )?;
        }

        // cuBLASLt fp8_gemm: D = A * B^T.
        //   A = hidden_fp8 [num_tokens, hidden]
        //   B = lm_head_fp8 [vocab, hidden]
        //   D = logits f16 [num_tokens, vocab]
        #[cfg(feature = "cuda")]
        unsafe {
            self.cublaslt.fp8_gemm(
                hidden_fp8_region.device_ptr(),
                self.model.outside.lm_head_fp8.offset_bytes,
                logits_region.device_ptr(),
                num_tokens as i32,
                vocab as i32,
                hidden as i32,
                hidden_scale_region.device_ptr(),
                self.model.outside.lm_head_fp8.scale_ptr,
                stream_raw,
            )?;
        }
        self.stream.fence()?;

        // DtoH the first 4 normalized hidden values for sanity, plus
        // token-0's full logits row for CPU-side argmax (the f16→f32
        // upcast + argmax fits in <10 ms on host for vocab=248k).
        let mut hidden_probe = [0u8; 8];
        let logits_row_bytes = (vocab as usize) * 2;
        let mut logits_row_f16 = vec![0u8; logits_row_bytes];
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let rc1 = cuMemcpyDtoH_v2(
                hidden_probe.as_mut_ptr() as *mut _,
                hidden_region.device_ptr(),
                hidden_probe.len(),
            );
            let rc2 = cuMemcpyDtoH_v2(
                logits_row_f16.as_mut_ptr() as *mut _,
                logits_region.device_ptr(),
                logits_row_bytes,
            );
            if rc1 != CUresult::CUDA_SUCCESS || rc2 != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36_smoke DtoH",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        let f0 = f16_bits_to_f32(u16::from_le_bytes([hidden_probe[0], hidden_probe[1]]));
        let f1 = f16_bits_to_f32(u16::from_le_bytes([hidden_probe[2], hidden_probe[3]]));
        let f2 = f16_bits_to_f32(u16::from_le_bytes([hidden_probe[4], hidden_probe[5]]));
        let f3 = f16_bits_to_f32(u16::from_le_bytes([hidden_probe[6], hidden_probe[7]]));

        let mut best_logit = f32::NEG_INFINITY;
        let mut best_token: i32 = -1;
        for v in 0..vocab as usize {
            let bits = u16::from_le_bytes([
                logits_row_f16[v * 2],
                logits_row_f16[v * 2 + 1],
            ]);
            let l = f16_bits_to_f32(bits);
            if l > best_logit {
                best_logit = l;
                best_token = v as i32;
            }
        }

        eprintln!(
            "[qwen36] forward_outside_smoke: embed → rmsnorm+fp8 → \
             cublaslt.fp8_gemm → cpu_argmax \
             token_ids={token_ids:?} \
             token0_embed[0..4]=[{f0:.4}, {f1:.4}, {f2:.4}, {f3:.4}] \
             token0_argmax_id={best_token} logit={best_logit:.3} \
             eps={eps:.0e}"
        );
        Ok(())
    }

    pub fn run_generate(&self) -> ! {
        unimplemented!(
            "qwen36 phase 3f+ — outside-only forward (rmsnorm + lm_head + argmax) \
             then per-layer math still TODO"
        );
    }

    pub fn run_bench(&self) -> ! {
        unimplemented!("qwen36 phase 2 — bench harness not yet ported");
    }

    pub fn run_ppl(&self) -> ! {
        unimplemented!("qwen36 phase 2 — ppl harness not yet ported");
    }

    pub fn init_prefix_cache(&self) -> ! {
        unimplemented!("qwen36 phase 2 — prefix cache not yet ported");
    }
}

/// IEEE 754 f32 → f16 round-to-nearest-even encode without pulling
/// `half` in as a runtime dep here. Saturates to ±MAX_F16 outside
/// the f16 range; NaN preserved.
fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 31) & 0x1) as u16;
    let exp32 = ((bits >> 23) & 0xff) as i32;
    let mant32 = bits & 0x7f_ffff;
    if exp32 == 0xff {
        // NaN / inf
        let mant16 = if mant32 != 0 { 0x200 } else { 0 };
        return (sign << 15) | (0x1f << 10) | mant16;
    }
    let exp_unbiased = exp32 - 127;
    if exp_unbiased >= 16 {
        // Overflow → ±inf.
        return (sign << 15) | (0x1f << 10);
    }
    if exp_unbiased < -24 {
        // Underflow → ±0.
        return sign << 15;
    }
    if exp_unbiased < -14 {
        // Subnormal in f16.
        let shift = (-14 - exp_unbiased) as u32;
        let mant_full = mant32 | (1 << 23);
        let rshift = 13 + shift;
        let m = mant_full >> rshift;
        let round_bit = (mant_full >> (rshift - 1)) & 1;
        let sticky = mant_full & ((1 << (rshift - 1)) - 1);
        let m = m + (round_bit & ((sticky != 0) as u32 | (m & 1)));
        return (sign << 15) | (m as u16 & 0x3ff);
    }
    let exp16 = (exp_unbiased + 15) as u16;
    let mant16 = (mant32 >> 13) as u16;
    let round_bit = (mant32 >> 12) & 1;
    let sticky = mant32 & 0xfff;
    let m = mant16 + (round_bit as u16 & ((sticky != 0) as u16 | (mant16 & 1)));
    if m & 0x400 != 0 {
        let exp16 = exp16 + 1;
        if exp16 >= 0x1f {
            return (sign << 15) | (0x1f << 10);
        }
        return (sign << 15) | (exp16 << 10);
    }
    (sign << 15) | (exp16 << 10) | (m & 0x3ff)
}

/// Decode a stored f16 bit pattern into f32 without pulling in `half`
/// as an explicit dep of `rvllm-runtime`. Mirrors `half::f16::to_f32`
/// for finite values; specials (NaN/inf) are preserved by IEEE
/// composition.
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 0x1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;
    let f32_bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            // Subnormal — renormalise.
            let mut e: i32 = -14;
            let mut m = mant;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            let m = (m & 0x3ff) << 13;
            (sign << 31) | (((e + 127) as u32) << 23) | m
        }
    } else if exp == 0x1f {
        // NaN or inf.
        (sign << 31) | (0xff << 23) | (mant << 13)
    } else {
        (sign << 31) | (((exp + 112) as u32) << 23) | (mant << 13)
    };
    f32::from_bits(f32_bits)
}
