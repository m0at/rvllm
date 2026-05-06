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
use rvllm_cutlass::{CublasLt, CutlassBackend};
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
    /// f16-input sibling of `fn_argmax` — used by the Qwen36 closer
    /// to argmax over the lm_head's f16 logits row directly on the
    /// GPU. Replaces the previous DtoH-of-full-vocab + host-side
    /// scan; the GPU kernel returns a single i32 token id.
    pub fn_argmax_f16: KernelFn,
    /// Per-token f16→fp8 amax-quantise. Used by
    /// `fp8_proj_dispatch`'s m≥2 branch to feed cuBLASLt
    /// `fp8_gemm` (which expects fp8 input + per-token f32 scale).
    /// At m=1 the GEMV path consumes f16 directly so this kernel
    /// is unused.
    pub fp8_quantize_per_token_f16_mod: LoadedModule,
    pub fn_fp8_quantize_per_token_f16: KernelFn,
    /// Per-token-amax sibling of `fp8_quantize_per_token_f16` (one
    /// f32 per row, vs per-K-block scales). Feeds CUTLASS SM120's
    /// `prep_sfa` entry point on the m≥128 fast path; cuBLASLt
    /// blockwise uses the per-K-block kernel above instead.
    pub fp8_quantize_per_token_amax_f16_mod: LoadedModule,
    pub fn_fp8_quantize_per_token_amax_f16: KernelFn,
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
    /// Qwen-specific partial-NeoX RoPE + KV-cache write. Differs
    /// from the Gemma sibling in pair convention: pairs `(i, i +
    /// rotary_dim/2)` instead of `(i, i + head_dim/2)`. Replaces a
    /// host DtoH→CPU-RoPE→HtoD path that was the dominant per-token
    /// cost in `apply_layer_full_attn` (Phase 4b prep).
    pub fused_rope_qwen_partial_f16kv_mod: LoadedModule,
    pub fn_fused_rope_qwen_partial_f16kv: KernelFn,
    /// Splits q_proj's interleaved `[num_heads, 2*head_dim]` output
    /// into separate q `[num_heads, head_dim]` + gate
    /// `[num_heads, head_dim]` regions. Replaces a host DtoH +
    /// per-head copy_from_slice + HtoD round-trip per token.
    pub split_q_gate_f16_mod: LoadedModule,
    pub fn_split_q_gate_f16: KernelFn,
    /// Conv1d state advance + conv_in assembly for the linear-attn
    /// block. Replaces a 2× DtoH + 2× HtoD pure-shuffle round-trip
    /// with one launch.
    pub conv_state_advance_f16_mod: LoadedModule,
    pub fn_conv_state_advance_f16: KernelFn,
    /// Fused alpha/beta computation for Qwen Gated-DeltaNet
    /// linear-attn. Replaces a host DtoH + f16→f32 + nested CPU
    /// GEMV (130k FLOPs/layer/token) + HtoD round-trip with one
    /// launch. Outputs f32 vectors (matches the existing
    /// alpha_region / beta_region dtype).
    pub qwen_linear_alpha_beta_f16_mod: LoadedModule,
    pub fn_qwen_linear_alpha_beta_f16: KernelFn,
    /// Fused silu + Q/K L2-norm + GQA-expand + V silu-pack for
    /// Qwen Gated-DeltaNet linear-attn. Replaces the host pipeline
    /// (DtoH conv_out + CPU silu/L2/GQA + HtoD q_exp/k_exp/v_pack)
    /// with one launch.
    pub qwen_linear_silu_l2_gqa_f16_mod: LoadedModule,
    pub fn_qwen_linear_silu_l2_gqa_f16: KernelFn,
    /// Per-v-head RMSNormGated with silu(z) gate. Replaces the
    /// host pipeline (DtoH readout/z/gamma + CPU rms + sigmoid·z
    /// + HtoD gated) at the end of `apply_layer_linear_attn`.
    pub qwen_linear_rmsnorm_gated_f16_mod: LoadedModule,
    pub fn_qwen_linear_rmsnorm_gated_f16: KernelFn,
    /// Pointwise SwiGLU activation `out = silu(gate) * up`. Used in
    /// `apply_layer_moe` to replace the per-expert host pipeline
    /// (DtoH gate + DtoH up + CPU silu·mul + HtoD silu) — saves
    /// two DtoH + one HtoD per expert × top_k experts × 30 MoE
    /// layers per token (Phase 4b-prep iter11).
    pub silu_mul_f16_mod: LoadedModule,
    pub fn_silu_mul_f16: KernelFn,
    /// Router-GEMV for the per-layer MoE gate. Reads
    /// `router_weight[num_experts, hidden]` f16 + the rmsnormed
    /// hidden state f16 from device memory and writes f32 logits.
    /// Replaces the host-cached f32 matvec (Phase 4b-prep iter17).
    pub router_gemv_f16_to_f32_mod: LoadedModule,
    pub fn_router_gemv_f16_to_f32: KernelFn,
    /// Per-expert weighted accumulator into f32 routed_sum:
    /// `acc[i] += weight * f16_to_f32(in[i])`. Replaces the host
    /// pipeline (fence + DtoH down + CPU scaled-add) per expert
    /// (Phase 4b-prep iter18, after the fence-before-DtoH bug
    /// from iter13/14 was diagnosed in iter17).
    pub scaled_add_f16_to_f32_mod: LoadedModule,
    pub fn_scaled_add_f16_to_f32: KernelFn,
    /// In-place residual add `inout_f16 += add_f32`. Replaces the
    /// final per-MoE-layer host residual round-trip (DtoH
    /// last_hidden + CPU f16+f32 add + HtoD residual) so the whole
    /// MoE forward stays device-side (Phase 4b-prep iter19).
    pub f16_plus_f32_inplace_f16_mod: LoadedModule,
    pub fn_f16_plus_f32_inplace_f16: KernelFn,
    /// Fused shared-expert gate dot product + sigmoid. Output is a
    /// single f32 scalar on the device, consumed directly by
    /// `scaled_add_f16_to_f32_devw` (Phase 4b-prep iter21).
    pub shared_gate_dot_sigmoid_f16_mod: LoadedModule,
    pub fn_shared_gate_dot_sigmoid_f16: KernelFn,
    /// scaled_add variant that reads the scalar weight from a device
    /// pointer (Phase 4b-prep iter21).
    pub scaled_add_f16_to_f32_devw_mod: LoadedModule,
    pub fn_scaled_add_f16_to_f32_devw: KernelFn,
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
    /// Phase 4r: Gated-DeltaNet per-head delta-rule state update
    /// kernel — the recurrent core of the linear-attn block. New
    /// CUDA kernel — kernels/gated_delta_state_update_f16.cu.
    pub gated_delta_state_update_f16_mod: LoadedModule,
    pub fn_gated_delta_state_update_f16: KernelFn,
    /// Phase 5i: full Gated-DeltaNet decode-step kernel doing forget +
    /// delta correction + state update + readout in one launch.
    /// Replaces the host-side delta-rule loop in apply_layer_linear_attn.
    pub gated_delta_rule_decode_f16_mod: LoadedModule,
    pub fn_gated_delta_rule_decode_f16: KernelFn,
    /// Vision Phase 1: LayerNorm with bias for Qwen ViT (norm1, norm2,
    /// merger.norm). Different from the GemmaRMSNorm-style RMSNorm
    /// used on text-side (additive +1 shift). Kernel:
    /// kernels/layernorm_inplace_f16.cu.
    pub layernorm_inplace_f16_mod: LoadedModule,
    pub fn_layernorm_inplace_f16: KernelFn,
    /// Vision Phase 1: pointwise gelu_pytorch_tanh (Qwen ViT MLP +
    /// Gemma ViT MLP). Kernel: kernels/gelu_tanh_f16.cu.
    pub gelu_tanh_f16_mod: LoadedModule,
    pub fn_gelu_tanh_f16: KernelFn,
    /// Vision Phase 1: row-wise softmax for ViT attention scores.
    /// Kernel: kernels/softmax_row_f16.cu.
    pub softmax_row_f16_mod: LoadedModule,
    pub fn_softmax_row_f16: KernelFn,
    /// Vision Phase 1: 2D-rotary applying to Q/K with per-token
    /// cos/sin tables encoding row+col axes. Kernel:
    /// kernels/vit_rotary_2d_f16.cu.
    pub vit_rotary_2d_f16_mod: LoadedModule,
    pub fn_vit_rotary_2d_f16: KernelFn,
    /// Vision Phase 1: 2D average pooling for Gemma vision tower.
    /// Kernel: kernels/vit_avgpool_f16.cu.
    pub vit_avgpool_f16_mod: LoadedModule,
    pub fn_vit_avgpool_f16: KernelFn,
    /// Vision Phase A2: bilinear interpolation of the learned absolute
    /// pos_embed table (Qwen3-VL specific). Kernel:
    /// kernels/vit_pos_embed_interp_f16.cu.
    pub vit_pos_embed_interp_f16_mod: LoadedModule,
    pub fn_vit_pos_embed_interp_f16: KernelFn,
    /// Vision: scalar in-place scale on f16 (used for the
    /// 1/sqrt(head_dim) attention-score scale).
    pub scale_inplace_f16_mod: LoadedModule,
    pub fn_scale_inplace_f16: KernelFn,
    /// Vision attention: transpose V from [N, head_dim] → [head_dim, N]
    /// so the second GEMM (scores @ V) uses our `input @ weight^T`
    /// helper without computing scores @ V^T by accident.
    pub transpose_2d_f16_mod: LoadedModule,
    pub fn_transpose_2d_f16: KernelFn,
    /// Vision helpers (also used elsewhere in the codebase): in-place
    /// per-row bias add (tensor[t,d] += bias[d]) and f32→f16 cast.
    pub add_bias_f16_mod: LoadedModule,
    pub fn_add_bias_f16: KernelFn,
    pub cast_fp_mod: LoadedModule,
    pub fn_cast_f32_to_f16: KernelFn,
    /// Vision: GPU residual add `dst[i] += src[i]` for the
    /// pre/post-attn and pre/post-MLP residual paths. Replaces the
    /// earlier DtoH-add-HtoD round-trip that synced 27× per image.
    pub vector_add_f16_mod: LoadedModule,
    pub fn_vector_add_f16: KernelFn,
    /// Vision: per-head gather/scatter from `[N, num_heads*head_dim]`
    /// in a single launch. Replaces the per-token DtoD loop that ran
    /// `n_tokens × num_heads × {Q,K,V}` async memcpys per block.
    pub extract_head_f16_mod: LoadedModule,
    pub fn_extract_head_f16: KernelFn,
    pub fn_scatter_head_f16: KernelFn,
}

/// Pre-converted f32 weight caches for one linear-attention layer.
/// Built once at bring-up; consumed by `apply_layer_linear_attn`'s
/// alpha/beta computation in place of per-token DtoH+f16→f32.
#[derive(Debug)]
pub struct Qwen36LinearAttnHostCache {
    pub a_w_f32: Vec<f32>,    // [vus, h_us]  row-major
    pub b_w_f32: Vec<f32>,    // [vus, h_us]  row-major
    pub a_log_f32: Vec<f32>,  // [vus]
    pub dt_bias_f32: Vec<f32>,// [vus]
    pub vus: usize,
    pub h_us: usize,
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
    /// CUTLASS SM120 backend for blockwise FP8 GEMM at m≥128.
    /// Loaded at bring-up; on sm_121 this resolves to
    /// `CutlassBackend::SoSm120` when `libcutlass_sm120.so` is found
    /// (the same .so Gemma uses for its lm_head fast path) and
    /// `CutlassBackend::Absent` otherwise — in which case
    /// `fp8_proj_dispatch` falls back to its looped-GEMV path.
    pub cutlass: CutlassBackend,
    /// Phase 4f: precomputed RoPE cos/sin tables for Qwen 3.6
    /// (rope_theta=10M, head_dim=256). Single-axis tables uploaded
    /// at bring-up; MRoPE's section-aware position encoding
    /// (sections [11, 11, 10]) is applied at launch time on top of
    /// these base tables in Phase 4g.
    pub rope_cos: u64,
    pub rope_sin: u64,
    pub rope_max_pos: u32,
    /// Phase 4t: per-sequence linear-attn state buffer. Sized
    /// `[num_linear_layers, num_heads, d_v, d_k]` f16. Lives ABOVE
    /// the scratch checkpoint so `arena.restore()` between requests
    /// doesn't reclaim it — state must persist across decode steps
    /// for the recurrent Gated-DeltaNet path. Single-sequence pool
    /// for now; multi-sequence batching is Phase 4v+.
    pub linear_state_ptr: u64,
    pub linear_state_bytes: usize,
    pub linear_state_layer_bytes: usize,
    /// Per-linear-attn-layer host-side f32 caches of constant
    /// weights consumed by the Gated-DeltaNet alpha/beta loop. The
    /// pre-Phase-5 implementation re-DtoH-copied these every token
    /// (in_proj_a + in_proj_b ≈ 256 KB per token, plus a_log /
    /// dt_bias) and re-converted f16→f32 in the same loop. Now we
    /// dequantise once at bring-up and cache the f32 vectors here;
    /// the per-token loop reads directly from RAM.
    /// Layout per entry: `a_w[v, k]` row-major `[vus, h_us]`,
    /// same for `b_w`; `a_log[v]` and `dt_bias[v]` are length-vus.
    pub linear_attn_host_cache: Vec<Qwen36LinearAttnHostCache>,
    /// Per-layer host-side f32 cache of the router weight matrix
    /// (`[num_experts, hidden]`). Pre-Phase-4b-prep iter15 the
    /// router GEMV did a fresh DtoH of ~1 MiB f16 weights every
    /// MoE layer × every token, then converted f16→f32 and ran
    /// the host matvec. The weight is constant — caching it as
    /// f32 once at bring-up replaces 30 MiB of per-token DtoH +
    /// 16M f16→f32 conversions with a direct RAM read.
    pub router_host_cache: Vec<Vec<f32>>,
    /// Per-layer host-side f32 cache of the shared-expert gate
    /// weight (`[hidden]`). The pre-iter16 path DtoH'd this every
    /// MoE layer × every token (4 KiB) and converted f16→f32; same
    /// caching pattern as `router_host_cache` (Phase 4b-prep iter16).
    pub shared_gate_host_cache: Vec<Vec<f32>>,
    /// Phase 4u: paged f16 KV cache for the 10 full-attention
    /// layers. Layout `[num_full_layers, 2 (K+V), num_blocks,
    /// block_size, num_kv_heads, head_dim]` f16. Pre-allocated above
    /// the scratch checkpoint so it survives `arena.restore()`
    /// between requests. Reset alongside `reset_linear_state` on
    /// fresh sessions.
    pub kv_cache_ptr: u64,
    pub kv_cache_bytes: usize,
    pub kv_cache_layer_bytes: usize,
    pub kv_cache_num_blocks: u32,
    pub kv_cache_block_size: u32,
    /// Persistent device pointer to the identity block table
    /// `[0, 1, …, kv_cache_num_blocks-1]` i32. The paged-attention
    /// path used to rebuild + re-upload this constant table every
    /// full-attn layer × every token; now we upload once at bring-up
    /// (Phase 4b-prep iter25).
    pub bt_persistent_ptr: u64,
    /// Phase 5f: persistent conv1d state cache for linear-attn layers.
    /// `[num_linear_layers, conv_kernel-1=3, conv_dim=8192]` f16.
    /// Holds the previous (kernel-1) conv-input timesteps so per-token
    /// causal_conv1d can attend to actual prior tokens (not zeros).
    /// Reset alongside reset_linear_state on session boundaries.
    pub conv_state_ptr: u64,
    pub conv_state_bytes: usize,
    pub conv_state_layer_bytes: usize,
}

/// Output of `Qwen36Bringup::forward_qwen_vision`.
pub struct VisionForwardOutput {
    /// Raw little-endian f16 bytes, layout `[num_tokens, hidden_dim]`.
    pub data: Vec<u8>,
    pub num_tokens: usize,
    pub hidden_dim: usize,
    pub grid_thw: [u32; 3],
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
        let fn_argmax_f16 = argmax_mod.get_function("argmax_f16_kernel")?;
        let fp8_quantize_per_token_f16_mod = kernels.load_ptx("fp8_quantize_per_token_f16")?;
        let fn_fp8_quantize_per_token_f16 = fp8_quantize_per_token_f16_mod
            .get_function("fp8_quantize_per_token_f16_kernel")?;
        let fp8_quantize_per_token_amax_f16_mod =
            kernels.load_ptx("fp8_quantize_per_token_amax_f16")?;
        let fn_fp8_quantize_per_token_amax_f16 = fp8_quantize_per_token_amax_f16_mod
            .get_function("fp8_quantize_per_token_amax_f16_kernel")?;
        let fused_rmsnorm_fp8_quant_mod = kernels.load_ptx("fused_rmsnorm_fp8_quant")?;
        let fn_fused_rmsnorm_fp8_quant =
            fused_rmsnorm_fp8_quant_mod.get_function("fused_rmsnorm_fp8_quant_kernel")?;
        let fused_rope_partial_f16kv_mod =
            kernels.load_ptx("fused_rope_partial_f16kv")?;
        let fn_fused_rope_partial_f16kv = fused_rope_partial_f16kv_mod
            .get_function("fused_rope_partial_f16kv_kernel")?;
        let fused_rope_qwen_partial_f16kv_mod =
            kernels.load_ptx("fused_rope_qwen_partial_f16kv")?;
        let fn_fused_rope_qwen_partial_f16kv = fused_rope_qwen_partial_f16kv_mod
            .get_function("fused_rope_qwen_partial_f16kv_kernel")?;
        let split_q_gate_f16_mod = kernels.load_ptx("split_q_gate_f16")?;
        let fn_split_q_gate_f16 = split_q_gate_f16_mod
            .get_function("split_q_gate_f16_kernel")?;
        let conv_state_advance_f16_mod = kernels.load_ptx("conv_state_advance_f16")?;
        let fn_conv_state_advance_f16 = conv_state_advance_f16_mod
            .get_function("conv_state_advance_f16_kernel")?;
        let qwen_linear_alpha_beta_f16_mod =
            kernels.load_ptx("qwen_linear_alpha_beta_f16")?;
        let fn_qwen_linear_alpha_beta_f16 = qwen_linear_alpha_beta_f16_mod
            .get_function("qwen_linear_alpha_beta_f16_kernel")?;
        let qwen_linear_silu_l2_gqa_f16_mod =
            kernels.load_ptx("qwen_linear_silu_l2_gqa_f16")?;
        let fn_qwen_linear_silu_l2_gqa_f16 = qwen_linear_silu_l2_gqa_f16_mod
            .get_function("qwen_linear_silu_l2_gqa_f16_kernel")?;
        let qwen_linear_rmsnorm_gated_f16_mod =
            kernels.load_ptx("qwen_linear_rmsnorm_gated_f16")?;
        let fn_qwen_linear_rmsnorm_gated_f16 = qwen_linear_rmsnorm_gated_f16_mod
            .get_function("qwen_linear_rmsnorm_gated_f16_kernel")?;
        let silu_mul_f16_mod = kernels.load_ptx("silu_mul_f16")?;
        let fn_silu_mul_f16 =
            silu_mul_f16_mod.get_function("silu_mul_f16_kernel")?;
        let router_gemv_f16_to_f32_mod =
            kernels.load_ptx("router_gemv_f16_to_f32")?;
        let fn_router_gemv_f16_to_f32 = router_gemv_f16_to_f32_mod
            .get_function("router_gemv_f16_to_f32_kernel")?;
        let scaled_add_f16_to_f32_mod =
            kernels.load_ptx("scaled_add_f16_to_f32")?;
        let fn_scaled_add_f16_to_f32 = scaled_add_f16_to_f32_mod
            .get_function("scaled_add_f16_to_f32_kernel")?;
        let f16_plus_f32_inplace_f16_mod =
            kernels.load_ptx("f16_plus_f32_inplace_f16")?;
        let fn_f16_plus_f32_inplace_f16 = f16_plus_f32_inplace_f16_mod
            .get_function("f16_plus_f32_inplace_f16_kernel")?;
        let shared_gate_dot_sigmoid_f16_mod =
            kernels.load_ptx("shared_gate_dot_sigmoid_f16")?;
        let fn_shared_gate_dot_sigmoid_f16 = shared_gate_dot_sigmoid_f16_mod
            .get_function("shared_gate_dot_sigmoid_f16_kernel")?;
        let scaled_add_f16_to_f32_devw_mod =
            kernels.load_ptx("scaled_add_f16_to_f32_devw")?;
        let fn_scaled_add_f16_to_f32_devw = scaled_add_f16_to_f32_devw_mod
            .get_function("scaled_add_f16_to_f32_devw_kernel")?;
        let flash_attention_mod = kernels.load_ptx("flash_attention")?;
        let fn_flash_attention_2_decode_f16io = flash_attention_mod
            .get_function("flash_attention_2_decode_f16io_kernel")?;
        let sigmoid_mul_f16_mod = kernels.load_ptx("sigmoid_mul_f16")?;
        let fn_sigmoid_mul_f16 =
            sigmoid_mul_f16_mod.get_function("sigmoid_mul_f16_kernel")?;
        let causal_conv1d_f16_mod = kernels.load_ptx("causal_conv1d_f16")?;
        let fn_causal_conv1d_f16 =
            causal_conv1d_f16_mod.get_function("causal_conv1d_f16_kernel")?;
        let gated_delta_state_update_f16_mod =
            kernels.load_ptx("gated_delta_state_update_f16")?;
        let fn_gated_delta_state_update_f16 = gated_delta_state_update_f16_mod
            .get_function("gated_delta_state_update_f16_kernel")?;
        let gated_delta_rule_decode_f16_mod =
            kernels.load_ptx("gated_delta_rule_decode_f16")?;
        let fn_gated_delta_rule_decode_f16 = gated_delta_rule_decode_f16_mod
            .get_function("gated_delta_rule_decode_f16_kernel")?;
        // Vision-tower kernels (Phase 1 .cu added on rusty_sm121_vision).
        let layernorm_inplace_f16_mod = kernels.load_ptx("layernorm_inplace_f16")?;
        let fn_layernorm_inplace_f16 =
            layernorm_inplace_f16_mod.get_function("layernorm_inplace_f16_kernel")?;
        let gelu_tanh_f16_mod = kernels.load_ptx("gelu_tanh_f16")?;
        let fn_gelu_tanh_f16 = gelu_tanh_f16_mod.get_function("gelu_tanh_f16_kernel")?;
        let softmax_row_f16_mod = kernels.load_ptx("softmax_row_f16")?;
        let fn_softmax_row_f16 = softmax_row_f16_mod.get_function("softmax_row_f16_kernel")?;
        let vit_rotary_2d_f16_mod = kernels.load_ptx("vit_rotary_2d_f16")?;
        let fn_vit_rotary_2d_f16 =
            vit_rotary_2d_f16_mod.get_function("vit_rotary_2d_f16_kernel")?;
        let vit_avgpool_f16_mod = kernels.load_ptx("vit_avgpool_f16")?;
        let fn_vit_avgpool_f16 = vit_avgpool_f16_mod.get_function("vit_avgpool_f16_kernel")?;
        let vit_pos_embed_interp_f16_mod = kernels.load_ptx("vit_pos_embed_interp_f16")?;
        let fn_vit_pos_embed_interp_f16 = vit_pos_embed_interp_f16_mod
            .get_function("vit_pos_embed_interp_f16_kernel")?;
        let scale_inplace_f16_mod = kernels.load_ptx("scale_inplace_f16")?;
        let fn_scale_inplace_f16 =
            scale_inplace_f16_mod.get_function("scale_inplace_f16_kernel")?;
        let transpose_2d_f16_mod = kernels.load_ptx("transpose_2d_f16")?;
        let fn_transpose_2d_f16 =
            transpose_2d_f16_mod.get_function("transpose_2d_f16_kernel")?;
        let add_bias_f16_mod = kernels.load_ptx("add_bias_f16")?;
        let fn_add_bias_f16 = add_bias_f16_mod.get_function("add_bias_f16_kernel")?;
        let cast_fp_mod = kernels.load_ptx("cast_fp")?;
        let fn_cast_f32_to_f16 = cast_fp_mod.get_function("cast_f32_to_f16_kernel")?;
        let vector_add_f16_mod = kernels.load_ptx("vector_add_f16")?;
        let fn_vector_add_f16 =
            vector_add_f16_mod.get_function("vector_add_f16_kernel")?;
        let extract_head_f16_mod = kernels.load_ptx("extract_head_f16")?;
        let fn_extract_head_f16 =
            extract_head_f16_mod.get_function("extract_head_f16_kernel")?;
        let fn_scatter_head_f16 =
            extract_head_f16_mod.get_function("scatter_head_f16_kernel")?;
        let outside_kernels = Qwen36OutsideKernels {
            embedding_gather_f16_mod,
            fn_embedding_gather_f16,
            rmsnorm_inplace_f16_mod,
            fn_rmsnorm_inplace_f16,
            fp8_gemv_mod,
            fn_fp8_gemv_wpr_native_f16in,
            argmax_mod,
            fn_argmax,
            fn_argmax_f16,
            fp8_quantize_per_token_f16_mod,
            fn_fp8_quantize_per_token_f16,
            fp8_quantize_per_token_amax_f16_mod,
            fn_fp8_quantize_per_token_amax_f16,
            fused_rmsnorm_fp8_quant_mod,
            fn_fused_rmsnorm_fp8_quant,
            fused_rope_partial_f16kv_mod,
            fn_fused_rope_partial_f16kv,
            fused_rope_qwen_partial_f16kv_mod,
            fn_fused_rope_qwen_partial_f16kv,
            split_q_gate_f16_mod,
            fn_split_q_gate_f16,
            conv_state_advance_f16_mod,
            fn_conv_state_advance_f16,
            qwen_linear_alpha_beta_f16_mod,
            fn_qwen_linear_alpha_beta_f16,
            qwen_linear_silu_l2_gqa_f16_mod,
            fn_qwen_linear_silu_l2_gqa_f16,
            qwen_linear_rmsnorm_gated_f16_mod,
            fn_qwen_linear_rmsnorm_gated_f16,
            silu_mul_f16_mod,
            fn_silu_mul_f16,
            router_gemv_f16_to_f32_mod,
            fn_router_gemv_f16_to_f32,
            scaled_add_f16_to_f32_mod,
            fn_scaled_add_f16_to_f32,
            f16_plus_f32_inplace_f16_mod,
            fn_f16_plus_f32_inplace_f16,
            shared_gate_dot_sigmoid_f16_mod,
            fn_shared_gate_dot_sigmoid_f16,
            scaled_add_f16_to_f32_devw_mod,
            fn_scaled_add_f16_to_f32_devw,
            flash_attention_mod,
            fn_flash_attention_2_decode_f16io,
            sigmoid_mul_f16_mod,
            fn_sigmoid_mul_f16,
            causal_conv1d_f16_mod,
            fn_causal_conv1d_f16,
            gated_delta_state_update_f16_mod,
            fn_gated_delta_state_update_f16,
            gated_delta_rule_decode_f16_mod,
            fn_gated_delta_rule_decode_f16,
            layernorm_inplace_f16_mod,
            fn_layernorm_inplace_f16,
            gelu_tanh_f16_mod,
            fn_gelu_tanh_f16,
            softmax_row_f16_mod,
            fn_softmax_row_f16,
            vit_rotary_2d_f16_mod,
            fn_vit_rotary_2d_f16,
            vit_avgpool_f16_mod,
            fn_vit_avgpool_f16,
            vit_pos_embed_interp_f16_mod,
            fn_vit_pos_embed_interp_f16,
            scale_inplace_f16_mod,
            fn_scale_inplace_f16,
            transpose_2d_f16_mod,
            fn_transpose_2d_f16,
            add_bias_f16_mod,
            fn_add_bias_f16,
            cast_fp_mod,
            fn_cast_f32_to_f16,
            vector_add_f16_mod,
            fn_vector_add_f16,
            extract_head_f16_mod,
            fn_extract_head_f16,
            fn_scatter_head_f16,
        };
        eprintln!(
            "[qwen36] outside kernels resolved: embedding_gather_f16, \
             rmsnorm_inplace_f16, fp8_gemv (wpr_native_f16in: {}), \
             argmax, fused_rmsnorm_fp8_quant, fused_rope_partial_f16kv, \
             flash_attention_2_decode_f16io, sigmoid_mul_f16, \
             causal_conv1d_f16, gated_delta_state_update_f16.",
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

        // CUTLASS SM120 backend for the m≥128 batched-prefill fast
        // path on sm_121. Same .so Gemma loads. We don't need the
        // policy variant table (sm_121 ships only the blockscale
        // entry point), so pass an empty variants slice.
        let cutlass = CutlassBackend::load_for(
            compile_target,
            paths.cutlass_so.clone(),
            &[],
        )?;
        eprintln!(
            "[qwen36] cutlass backend = {}",
            match &cutlass {
                CutlassBackend::SoSm120(_) => "SoSm120 (sm_121 fast path)",
                CutlassBackend::So(_) => "So",
                CutlassBackend::Absent => "Absent (m≥2 path → looped GEMV fallback)",
                _ => "Other",
            }
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
        // Partial RoPE: rotary_dim = head_dim * 0.25 = 64 for Qwen 3.6.
        // Tables must be stride [rotary_dim/2] per position (matches the
        // fused_rope kernel's `pos * half_rotary + tid` indexing).
        // Frequencies still use head_dim as divisor — proportional-RoPE
        // convention shared with Gemma 4 (gemma4_load.rs:599-604).
        let rotary_dim = (rope_head_dim as f32 * 0.25) as u32;
        let half = (rotary_dim / 2) as usize;
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

        // Phase 4t: per-sequence linear-attn state cache. Persists
        // across decode steps in a single arena region above the
        // scratch checkpoint. Single-sequence layout for now.
        let n_linear_layers = arch
            .base
            .layer_types
            .iter()
            .filter(|t| matches!(t, rvllm_loader::LayerAttnType::Linear))
            .count();
        let num_ssm_heads: usize = 32;
        let d_state: usize = 128;
        let linear_state_layer_bytes =
            num_ssm_heads * d_state * d_state * 2; // f16
        let linear_state_bytes = n_linear_layers * linear_state_layer_bytes;
        let linear_state_region =
            arena.region("qwen36_linear_state", linear_state_bytes, 16)?;
        // Zero the state at bring-up (analogous to a session-start
        // reset). cuMemsetD8 is the fastest path; falls back to a
        // host-side zero buffer if needed.
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemsetD8_v2(
                linear_state_region.device_ptr(),
                0,
                linear_state_bytes,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36_linear_state cuMemsetD8",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        let linear_state_ptr = linear_state_region.device_ptr();
        eprintln!(
            "[qwen36] linear-attn state cache allocated: {} layers × \
             {:.1} MiB = {:.1} MiB total (zero-initialised, persists \
             across decode steps).",
            n_linear_layers,
            linear_state_layer_bytes as f64 / (1024.0 * 1024.0),
            linear_state_bytes as f64 / (1024.0 * 1024.0),
        );

        // Phase 4u: paged f16 KV cache for full-attention layers.
        // Sized for max_tokens_cap context (default 4096) at the
        // current num_kv_heads / head_dim layout. block_size 16 is
        // the standard rvllm paged-attn tile.
        let n_full_layers = arch
            .base
            .layer_types
            .iter()
            .filter(|t| matches!(t, rvllm_loader::LayerAttnType::Full))
            .count();
        let kv_max_tokens = std::env::var("RVLLM_MAX_TOKENS_CAP")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(4096);
        let kv_cache_block_size: u32 = 16;
        let kv_cache_num_blocks = kv_max_tokens.div_ceil(kv_cache_block_size);
        let nkvh = arch.base.num_key_value_heads;
        let hd = arch.base.head_dim;
        // 2 = K + V. Each slot is f16 (2 bytes).
        let kv_cache_layer_bytes = 2usize
            * (kv_cache_num_blocks as usize)
            * (kv_cache_block_size as usize)
            * nkvh
            * hd
            * 2;
        let kv_cache_bytes = n_full_layers * kv_cache_layer_bytes;
        let kv_cache_region =
            arena.region("qwen36_kv_cache", kv_cache_bytes, 16)?;
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemsetD8_v2(
                kv_cache_region.device_ptr(),
                0,
                kv_cache_bytes,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36_kv_cache cuMemsetD8",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        let kv_cache_ptr = kv_cache_region.device_ptr();
        eprintln!(
            "[qwen36] paged KV cache allocated: {n_full_layers} full-attn \
             layers × {:.1} MiB ({} blocks × {} tokens × 2 (K+V) × \
             {nkvh} kv_heads × {hd} hd × f16) = {:.1} MiB total \
             (zero-initialised).",
            kv_cache_layer_bytes as f64 / (1024.0 * 1024.0),
            kv_cache_num_blocks,
            kv_cache_block_size,
            kv_cache_bytes as f64 / (1024.0 * 1024.0),
        );

        // Phase 5f: conv1d state cache. Each linear-attn layer keeps the
        // previous (kernel-1=3) conv-input timesteps so per-token decode
        // sees the real prior context. Layout per layer: [3, 8192] f16.
        let conv_kernel_minus_1: usize = 3;
        let conv_dim: usize = 8192;
        let conv_state_layer_bytes = conv_kernel_minus_1 * conv_dim * 2;
        let conv_state_bytes = n_linear_layers * conv_state_layer_bytes;
        let conv_state_region =
            arena.region("qwen36_conv_state", conv_state_bytes, 16)?;
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemsetD8_v2(conv_state_region.device_ptr(), 0, conv_state_bytes);
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36_conv_state cuMemsetD8",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        let conv_state_ptr = conv_state_region.device_ptr();
        eprintln!(
            "[qwen36] conv1d state cache allocated: {n_linear_layers} \
             linear-attn layers × {:.1} KiB ({conv_kernel_minus_1} \
             timesteps × {conv_dim} channels × f16) = {:.1} MiB total.",
            conv_state_layer_bytes as f64 / 1024.0,
            conv_state_bytes as f64 / (1024.0 * 1024.0),
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
            "[qwen36] Phase 5f bring-up complete: outside (incl. \
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

        let mut bringup = Self {
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
            cutlass,
            rope_cos,
            rope_sin,
            rope_max_pos,
            linear_state_ptr,
            linear_state_bytes,
            linear_state_layer_bytes,
            linear_attn_host_cache: Vec::new(), // populated below
            router_host_cache: Vec::new(),       // populated below
            shared_gate_host_cache: Vec::new(),  // populated below

            kv_cache_ptr,
            kv_cache_bytes,
            kv_cache_layer_bytes,
            kv_cache_num_blocks,
            kv_cache_block_size,
            bt_persistent_ptr: 0, // populated below
            conv_state_ptr,
            conv_state_bytes,
            conv_state_layer_bytes,
        };
        // Phase 4b-prep iter25: upload the constant identity block
        // table once. The paged-attention layer used to rebuild it
        // every full-attn layer × every token.
        {
            let max_blocks = bringup.kv_cache_num_blocks as usize;
            let bt_bytes = max_blocks * 4;
            let bt_region = bringup.arena.region("qwen36_bt_persistent", bt_bytes, 16)?;
            let mut bt_host = Vec::with_capacity(bt_bytes);
            for b in 0..max_blocks {
                bt_host.extend_from_slice(&(b as i32).to_le_bytes());
            }
            unsafe { bt_region.copy_from_host(&bt_host)?; }
            bringup.bt_persistent_ptr = bt_region.device_ptr();
        }
        // Build per-linear-attn-layer host f32 weight caches BEFORE
        // any probe / smoke runs (some of those reach into
        // `apply_layer_linear_attn` and would panic on an empty
        // cache). Pre-Phase-5 code DtoH-copied these weights every
        // token (~256 KB of constants per token); now we dequantise
        // once to f32 and the per-token alpha/beta loop reads
        // straight from RAM.
        {
            let mut linear_caches: Vec<Qwen36LinearAttnHostCache> = Vec::new();
            for layer in bringup.model.layers.iter() {
                if let rvllm_loader::qwen36_weights::Qwen36LayerAttn::Linear(la) = &layer.attn {
                    let vus = la.in_proj_a.shape[0];
                    let h_us = la.in_proj_a.shape[1];
                    let proj_bytes = vus * h_us * 2;
                    let mut a_w = vec![0u8; proj_bytes];
                    let mut b_w = vec![0u8; proj_bytes];
                    let mut a_log_h = vec![0u8; vus * 2];
                    let mut dt_bias_h = vec![0u8; vus * 2];
                    #[cfg(feature = "cuda")]
                    unsafe {
                        use cudarc::driver::sys::*;
                        cuMemcpyDtoH_v2(a_w.as_mut_ptr() as *mut _, la.in_proj_a.offset_bytes, proj_bytes);
                        cuMemcpyDtoH_v2(b_w.as_mut_ptr() as *mut _, la.in_proj_b.offset_bytes, proj_bytes);
                        cuMemcpyDtoH_v2(a_log_h.as_mut_ptr() as *mut _, la.a_log.offset_bytes, a_log_h.len());
                        cuMemcpyDtoH_v2(dt_bias_h.as_mut_ptr() as *mut _, la.dt_bias.offset_bytes, dt_bias_h.len());
                    }
                    let mut a_w_f32 = vec![0.0f32; vus * h_us];
                    let mut b_w_f32 = vec![0.0f32; vus * h_us];
                    for i in 0..(vus * h_us) {
                        a_w_f32[i] = f16_bits_to_f32(u16::from_le_bytes([a_w[i * 2], a_w[i * 2 + 1]]));
                        b_w_f32[i] = f16_bits_to_f32(u16::from_le_bytes([b_w[i * 2], b_w[i * 2 + 1]]));
                    }
                    let mut a_log_f32 = vec![0.0f32; vus];
                    let mut dt_bias_f32 = vec![0.0f32; vus];
                    for v in 0..vus {
                        a_log_f32[v] = f16_bits_to_f32(u16::from_le_bytes([a_log_h[v * 2], a_log_h[v * 2 + 1]]));
                        dt_bias_f32[v] = f16_bits_to_f32(u16::from_le_bytes([dt_bias_h[v * 2], dt_bias_h[v * 2 + 1]]));
                    }
                    linear_caches.push(Qwen36LinearAttnHostCache {
                        a_w_f32, b_w_f32, a_log_f32, dt_bias_f32, vus, h_us,
                    });
                }
            }
            let cache_bytes: usize = linear_caches.iter()
                .map(|c| (c.a_w_f32.len() + c.b_w_f32.len()) * 4
                      + (c.a_log_f32.len() + c.dt_bias_f32.len()) * 4)
                .sum();
            eprintln!(
                "[qwen36] linear_attn host cache: {} layers, {:.1} MiB",
                linear_caches.len(),
                cache_bytes as f64 / (1024.0 * 1024.0)
            );
            bringup.linear_attn_host_cache = linear_caches;
        }

        // Phase 4b-prep iter15: cache the router weight matrices as
        // f32 host vectors so the per-layer router GEMV can skip the
        // 1 MiB DtoH + f16→f32 unpack every token.
        {
            let num_experts = bringup.arch.num_experts;
            let hidden_us = bringup.arch.base.hidden_size as usize;
            let mut caches: Vec<Vec<f32>> = Vec::new();
            for layer in bringup.model.layers.iter() {
                let router_bytes = num_experts * hidden_us * 2;
                let mut router_host = vec![0u8; router_bytes];
                #[cfg(feature = "cuda")]
                unsafe {
                    use cudarc::driver::sys::*;
                    cuMemcpyDtoH_v2(
                        router_host.as_mut_ptr() as *mut _,
                        layer.moe.router.offset_bytes,
                        router_bytes,
                    );
                }
                let mut router_f32 = vec![0.0f32; num_experts * hidden_us];
                for i in 0..(num_experts * hidden_us) {
                    router_f32[i] = f16_bits_to_f32(u16::from_le_bytes([
                        router_host[i * 2], router_host[i * 2 + 1],
                    ]));
                }
                caches.push(router_f32);
            }
            let total_bytes: usize = caches.iter().map(|c| c.len() * 4).sum();
            eprintln!(
                "[qwen36] router host cache: {} layers, {:.1} MiB",
                caches.len(),
                total_bytes as f64 / (1024.0 * 1024.0)
            );
            bringup.router_host_cache = caches;
        }

        // Phase 4b-prep iter16: cache shared-expert gate weights as
        // f32 host vectors. Saves 4 KiB DtoH × num_layers × per token.
        {
            let hidden_us = bringup.arch.base.hidden_size as usize;
            let mut caches: Vec<Vec<f32>> = Vec::new();
            for layer in bringup.model.layers.iter() {
                let sg_bytes = hidden_us * 2;
                let mut sg_host = vec![0u8; sg_bytes];
                #[cfg(feature = "cuda")]
                unsafe {
                    use cudarc::driver::sys::*;
                    cuMemcpyDtoH_v2(
                        sg_host.as_mut_ptr() as *mut _,
                        layer.moe.shared_expert_gate_logit.offset_bytes,
                        sg_bytes,
                    );
                }
                let mut sg_f32 = vec![0.0f32; hidden_us];
                for k in 0..hidden_us {
                    sg_f32[k] = f16_bits_to_f32(u16::from_le_bytes([
                        sg_host[k * 2], sg_host[k * 2 + 1],
                    ]));
                }
                caches.push(sg_f32);
            }
            bringup.shared_gate_host_cache = caches;
        }

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
        // Phase 4r: gated_delta_state_update_f16 — recurrent
        // state-space update, the heart of the linear-attn block.
        bringup.forward_layer0_ssm_state_probe()?;
        // Phase 4s: chain everything for layer 0 linear-attn.
        bringup.forward_layer0_linear_chain_probe()?;
        // Phase 4t: per-sequence state cache (zero-init + reset).
        bringup.linear_state_cache_probe()?;
        // Phase 4u: paged KV cache for full-attn layers.
        bringup.kv_cache_probe()?;

        // Phase 4v: end-to-end smoke that threads real hidden state
        // through layer 0's linear-attn in/out projections + final
        // norm + lm_head + argmax. Token output is garbage by design
        // (degenerate composition between in_proj and out_proj), but
        // proves per-layer FP8 kernels run on the production decode
        // path's actual hidden buffer instead of synthetic inputs.
        // Reset state before this synthetic call so it starts clean
        // (cuda_worker also resets on every request).
        bringup.reset_linear_state()?;
        bringup.reset_kv_cache()?;
        bringup.reset_conv_state()?;
        let probe_5d = bringup.forward_qwen36_decode(&[1, 200, 2000, 20_000, 50_000], 0, &[])?;
        eprintln!(
            "[qwen36] Phase 5d forward_qwen36_decode smoke: 5-token \
             input → argmax_token_id={probe_5d} (linear-attn rewritten \
             to vLLM-correct layout: Q[16,128]+K[16,128]+V[32,128] \
             split, per-head L2-norm on Q/K, in_proj_a/b for α/β, \
             GQA-expanded K/Q for state update, per-v-head readout)"
        );

        // Phase 2b-γ: vision smoke (gated by env). Reads a fixture
        // image from RVLLM_QWEN36_VISION_PROBE_PATH and runs the full
        // ViT forward, dumping output stats. Confirms the 27-block
        // forward + PatchMerger compose without crash and produce
        // sane f16 magnitudes. Quality vs HF reference is Phase 5.
        if let Ok(path) = std::env::var("RVLLM_QWEN36_VISION_PROBE_PATH") {
            match std::fs::read(&path) {
                Ok(bytes) => match bringup.forward_qwen_vision(&bytes) {
                    Ok(out) => {
                        let f32s: Vec<f32> = out
                            .data
                            .chunks_exact(2)
                            .map(|c| {
                                f16_bits_to_f32(u16::from_le_bytes([c[0], c[1]]))
                            })
                            .collect();
                        let l2 = f32s.iter().map(|x| x * x).sum::<f32>().sqrt();
                        let max = f32s.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
                        eprintln!(
                            "[qwen36] vision probe: image={} → tokens={} hidden={} grid_thw={:?} L2={:.3} max={:.3}",
                            path, out.num_tokens, out.hidden_dim, out.grid_thw, l2, max,
                        );
                    }
                    Err(e) => eprintln!("[qwen36] vision probe failed: {e:?}"),
                },
                Err(e) => eprintln!("[qwen36] vision probe: cannot read {path}: {e}"),
            }
        }

        Ok(bringup)
    }

    pub fn kernels_dir(&self) -> &PathBuf {
        &self.paths.kernels_dir
    }

    /// Phase 4u: device pointer for full-attn layer's slice of the
    /// paged KV cache. `layer_seq_idx` is the sequential index of
    /// the full-attn layer (0..num_full_layers), NOT the absolute
    /// model layer index — caller maps via the layer_types array.
    pub fn kv_cache_layer_ptr(&self, layer_seq_idx: u32) -> u64 {
        let off = (layer_seq_idx as usize)
            .saturating_mul(self.kv_cache_layer_bytes);
        if off + self.kv_cache_layer_bytes > self.kv_cache_bytes {
            return 0;
        }
        self.kv_cache_ptr + off as u64
    }

    /// Phase 4u: zero out the paged KV cache (all full-attn layers).
    /// Called on session boundaries alongside `reset_linear_state`.
    pub fn reset_kv_cache(&self) -> Result<()> {
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemsetD8Async(
                self.kv_cache_ptr,
                0,
                self.kv_cache_bytes,
                self.stream.raw() as CUstream,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 reset_kv_cache",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        Ok(())
    }

    /// Phase 4t: zero out the per-sequence linear-attn state cache.
    /// Called on session boundaries (`/new`, fresh request without
    /// session continuity) so a stale state from a prior sequence
    /// doesn't leak into the new one.
    pub fn reset_linear_state(&self) -> Result<()> {
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemsetD8Async(
                self.linear_state_ptr,
                0,
                self.linear_state_bytes,
                self.stream.raw() as CUstream,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 reset_linear_state",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        Ok(())
    }

    /// Phase 5f: zero out the conv1d state cache.
    pub fn reset_conv_state(&self) -> Result<()> {
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemsetD8Async(
                self.conv_state_ptr, 0, self.conv_state_bytes,
                self.stream.raw() as CUstream,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 reset_conv_state",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        Ok(())
    }

    /// Phase 5f: device pointer for layer N's conv-state slice.
    pub fn conv_state_layer_ptr(&self, layer_seq_idx: u32) -> u64 {
        let off = (layer_seq_idx as usize)
            .saturating_mul(self.conv_state_layer_bytes);
        if off + self.conv_state_layer_bytes > self.conv_state_bytes {
            return 0;
        }
        self.conv_state_ptr + off as u64
    }

    /// Phase 4t: device pointer for layer N's slice of the state cache.
    /// Returns 0 if the layer index is out of bounds.
    pub fn linear_state_layer_ptr(&self, layer_seq_idx: u32) -> u64 {
        let off = (layer_seq_idx as usize)
            .saturating_mul(self.linear_state_layer_bytes);
        if off + self.linear_state_layer_bytes > self.linear_state_bytes {
            return 0;
        }
        self.linear_state_ptr + off as u64
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

        // GPU-side argmax over the LAST token's logits row. The
        // argmax_f16_kernel does one block-reduction per row and
        // writes a single i32 — we DtoH 4 bytes instead of the full
        // ~526 KB vocab f16 buffer the previous host-side scan
        // pulled across PCIe every step. (Codex round 16 #3.)
        let logits_row_bytes = (vocab as usize) * 2;
        let last_offset = (last_idx as u64) * (logits_row_bytes as u64);
        let token_region = self.arena.region("qwen36_argmax_tok", 4, 4)?;
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            // Launch with one block per row (here always 1) and up to
            // 1024 threads collaborating on the reduction. Matches
            // the kernel's documented launch config.
            let block_dim: u32 = (vocab as u32).min(1024);
            let mut row_ptr = logits_region.device_ptr() + last_offset;
            let mut out_ptr = token_region.device_ptr();
            let mut vsz: i32 = vocab as i32;
            let args = [
                (&mut row_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut out_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut vsz) as *mut i32 as *mut core::ffi::c_void,
            ];
            let rc = cuLaunchKernel(
                self.outside_kernels.fn_argmax_f16.raw() as CUfunction,
                /*grid*/ 1, 1, 1,
                /*block*/ block_dim, 1, 1,
                /*shared*/ 0,
                self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 argmax_f16 launch",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        self.stream.fence()?;
        let mut tok_buf = [0u8; 4];
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemcpyDtoH_v2(
                tok_buf.as_mut_ptr() as *mut _,
                token_region.device_ptr(),
                4,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 argmax_f16 DtoH(token)",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        Ok(i32::from_le_bytes(tok_buf))
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

    /// Phase 4s probe: chain all linear-attn kernels end-to-end for
    /// layer 0, single timestep. Order:
    ///   1. in_proj_qkv f16 GEMV → conv_input [8192 = 4×2048]
    ///   2. causal_conv1d_f16 (state-cache padded with zeros — no
    ///      history yet, single-step)
    ///   3. host SiLU + split into Q/K/V/extra streams [4 × 2048]
    ///      (qwen3-next exact split is q=k=v=2048 + dt_input=2048
    ///      per the reference; this probe uses that convention)
    ///   4. host: dt = silu(conv_dt) + dt_bias, alpha = exp(-exp(A_log)·dt),
    ///      beta = dt (Mamba-2-style decay+write derivation)
    ///   5. gated_delta_state_update_f16 (state init = 0)
    ///   6. host: read-out per head (Q · state → 32×128 = 4096 dim)
    ///   7. in_proj_z FP8 GEMV → z_logits [4096], host sigmoid · readout
    ///   8. out_proj FP8 GEMV → final delta [hidden=2048]
    ///
    /// Synthetic input. Math approximates qwen3-next without claiming
    /// numerical match against the reference — that's Phase 4u where
    /// a single-token comparison against vLLM nails down any
    /// off-by-one in the dt/alpha/beta computation, the QKV split
    /// order, or the post-SSM normalisation. Goal here: prove every
    /// kernel + host glue step composes without ABI errors.
    pub fn forward_layer0_linear_chain_probe(&self) -> Result<()> {
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
        let kernel_gemv = match self.outside_kernels.fn_fp8_gemv_wpr_native_f16in {
            Some(k) => k,
            None => return Ok(()),
        };
        let hidden = self.arch.base.hidden_size as u32;
        let qkv_n = la.in_proj_qkv.shape[0] as u32; // 8192
        let z_n = la.in_proj_z.shape[0] as u32; // 4096
        let out_n = la.out_proj.shape[0] as u32; // 2048
        let out_k = la.out_proj.shape[1] as u32; // 4096
        let m: u32 = 1;
        let num_heads: u32 = 32;
        let d_state: u32 = 128;
        let head_split: u32 = qkv_n / 4; // 2048 per stream

        // Synthetic hidden input: small varied values.
        let mut input_bytes = Vec::with_capacity((hidden as usize) * 2);
        for i in 0..hidden as usize {
            let v = ((i % 16) as f32) * 0.0625 - 0.5;
            input_bytes.extend_from_slice(&f32_to_f16_bits(v).to_le_bytes());
        }
        let in_region = self.arena.region("qwen36_l0lc_in", input_bytes.len(), 16)?;
        unsafe { in_region.copy_from_host(&input_bytes)? };

        // 1. in_proj_qkv FP8 GEMV → qkv_concat [8192]
        let qkv_bytes = (qkv_n as usize) * 2;
        let qkv_region = self.arena.region("qwen36_l0lc_qkv", qkv_bytes, 16)?;
        let qkv_bs = la.in_proj_qkv.blockscale_ptr.unwrap_or(0);
        if qkv_bs == 0 {
            return Ok(());
        }
        unsafe {
            rvllm_fused::gemma4_launcher::Fp8GemvF16InLaunch {
                m, n: qkv_n, k: hidden,
            }.launch(
                kernel_gemv,
                qkv_region.device_ptr(),
                la.in_proj_qkv.offset_bytes,
                qkv_bs,
                in_region.device_ptr(),
                self.stream.raw() as u64,
            )?;
        }
        self.stream.fence()?;

        // 2. causal_conv1d (state padded with zeros for single-step).
        // conv1d expects [seq+ks-1, channels]. We pad with 3 zero
        // timesteps + 1 real timestep = ks=4 input window.
        let ks: u32 = 4;
        let conv_in_elems = ((1 + ks - 1) as usize) * (qkv_n as usize);
        let conv_in_bytes = conv_in_elems * 2;
        let conv_in_region =
            self.arena.region("qwen36_l0lc_cin", conv_in_bytes, 16)?;
        let conv_out_region =
            self.arena.region("qwen36_l0lc_cout", qkv_bytes, 16)?;
        // Build conv input on host: 3 zero timesteps + qkv_concat.
        let mut conv_in_host = vec![0u8; conv_in_bytes];
        let mut qkv_host = vec![0u8; qkv_bytes];
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let _ = cuMemcpyDtoH_v2(
                qkv_host.as_mut_ptr() as *mut _,
                qkv_region.device_ptr(),
                qkv_bytes,
            );
        }
        // Place qkv_host at last (ks-1=3) timestep position. Earlier
        // positions stay zero.
        let last_off = ((ks - 1) as usize) * (qkv_n as usize) * 2;
        conv_in_host[last_off..last_off + qkv_bytes].copy_from_slice(&qkv_host);
        unsafe { conv_in_region.copy_from_host(&conv_in_host)? };

        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let mut output = conv_out_region.device_ptr();
            let mut input = conv_in_region.device_ptr();
            let mut weight = la.conv1d.offset_bytes;
            let mut sl: i32 = 1;
            let mut ch = qkv_n as i32;
            let mut k_arg = ks as i32;
            let args = [
                (&mut output) as *mut u64 as *mut core::ffi::c_void,
                (&mut input) as *mut u64 as *mut core::ffi::c_void,
                (&mut weight) as *mut u64 as *mut core::ffi::c_void,
                (&mut sl) as *mut i32 as *mut core::ffi::c_void,
                (&mut ch) as *mut i32 as *mut core::ffi::c_void,
                (&mut k_arg) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 256;
            let grid_x = (qkv_n + block - 1) / block;
            let _ = cuLaunchKernel(
                self.outside_kernels.fn_causal_conv1d_f16.raw() as CUfunction,
                grid_x, 1, 1,
                block, 1, 1,
                0,
                self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
        }
        self.stream.fence()?;

        // 3. Host: read conv_out, SiLU, split into Q/K/V/dt streams.
        let mut conv_out_host = vec![0u8; qkv_bytes];
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let _ = cuMemcpyDtoH_v2(
                conv_out_host.as_mut_ptr() as *mut _,
                conv_out_region.device_ptr(),
                qkv_bytes,
            );
        }
        let mut conv_silu_f32 = Vec::with_capacity(qkv_n as usize);
        for i in 0..qkv_n as usize {
            let bits = u16::from_le_bytes([conv_out_host[i * 2], conv_out_host[i * 2 + 1]]);
            let v = f16_bits_to_f32(bits);
            conv_silu_f32.push(v / (1.0f32 + (-v).exp()));
        }
        let q_off = 0usize;
        let k_off = head_split as usize;
        let v_off = (2 * head_split) as usize;
        let dt_off = (3 * head_split) as usize;
        let split_per = head_split as usize;

        // 4. Host: alpha/beta from A_log + dt_bias + dt_input.
        // dt_input is the 4th split = `conv_silu[dt_off..dt_off+split_per]`.
        // For a smoke probe we average dt_input per head into a scalar.
        let mut a_log_host = vec![0u8; (num_heads as usize) * 2];
        let mut dt_bias_host = vec![0u8; (num_heads as usize) * 2];
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let _ = cuMemcpyDtoH_v2(
                a_log_host.as_mut_ptr() as *mut _,
                la.a_log.offset_bytes,
                a_log_host.len(),
            );
            let _ = cuMemcpyDtoH_v2(
                dt_bias_host.as_mut_ptr() as *mut _,
                la.dt_bias.offset_bytes,
                dt_bias_host.len(),
            );
        }
        let per_head = (split_per / num_heads as usize) as usize;
        let mut alpha_f32 = Vec::with_capacity(num_heads as usize);
        let mut beta_f32 = Vec::with_capacity(num_heads as usize);
        for h in 0..num_heads as usize {
            // dt = softplus(dt_input_avg + dt_bias[h])
            let mut dt_avg = 0.0f32;
            for j in 0..per_head {
                dt_avg += conv_silu_f32[dt_off + h * per_head + j];
            }
            dt_avg /= per_head as f32;
            let bias =
                f16_bits_to_f32(u16::from_le_bytes([dt_bias_host[h * 2], dt_bias_host[h * 2 + 1]]));
            let dt = (1.0f32 + (dt_avg + bias).exp()).ln(); // softplus
            let a_log = f16_bits_to_f32(u16::from_le_bytes([
                a_log_host[h * 2], a_log_host[h * 2 + 1],
            ]));
            let a = (-(a_log.exp()) * dt).exp();
            alpha_f32.push(a);
            beta_f32.push(dt);
        }

        // 5. Pack Q/K/V f16 [num_heads, d_state=128] from the splits.
        // (per_head should equal d_state=128 for qwen3-next 35B-A3B.)
        let qkv_per_head_bytes = (num_heads as usize) * (d_state as usize) * 2;
        let mut q_host = Vec::with_capacity(qkv_per_head_bytes);
        let mut k_host = Vec::with_capacity(qkv_per_head_bytes);
        let mut v_host = Vec::with_capacity(qkv_per_head_bytes);
        for h in 0..num_heads as usize {
            for d in 0..d_state as usize {
                let q = conv_silu_f32[q_off + h * per_head + d];
                let k = conv_silu_f32[k_off + h * per_head + d];
                let v = conv_silu_f32[v_off + h * per_head + d];
                q_host.extend_from_slice(&f32_to_f16_bits(q).to_le_bytes());
                k_host.extend_from_slice(&f32_to_f16_bits(k).to_le_bytes());
                v_host.extend_from_slice(&f32_to_f16_bits(v).to_le_bytes());
            }
        }

        // 6. ssm state update.
        let state_bytes =
            (num_heads as usize) * (d_state as usize) * (d_state as usize) * 2;
        let state_region = self.arena.region("qwen36_l0lc_state", state_bytes, 16)?;
        let q_region = self.arena.region("qwen36_l0lc_q", q_host.len(), 16)?;
        let k_region = self.arena.region("qwen36_l0lc_k", k_host.len(), 16)?;
        let v_region = self.arena.region("qwen36_l0lc_v", v_host.len(), 16)?;
        let alpha_bytes: Vec<u8> = alpha_f32.iter().flat_map(|f| f.to_le_bytes()).collect();
        let beta_bytes: Vec<u8> = beta_f32.iter().flat_map(|f| f.to_le_bytes()).collect();
        let alpha_region =
            self.arena.region("qwen36_l0lc_alpha", alpha_bytes.len(), 16)?;
        let beta_region =
            self.arena.region("qwen36_l0lc_beta", beta_bytes.len(), 16)?;
        let zero_state = vec![0u8; state_bytes];
        unsafe {
            state_region.copy_from_host(&zero_state)?;
            q_region.copy_from_host(&q_host)?;
            k_region.copy_from_host(&k_host)?;
            v_region.copy_from_host(&v_host)?;
            alpha_region.copy_from_host(&alpha_bytes)?;
            beta_region.copy_from_host(&beta_bytes)?;
        }
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let mut state = state_region.device_ptr();
            let mut k = k_region.device_ptr();
            let mut v = v_region.device_ptr();
            let mut alpha = alpha_region.device_ptr();
            let mut beta = beta_region.device_ptr();
            let mut nh = num_heads as i32;
            let mut dk = d_state as i32;
            let mut dv = d_state as i32;
            let args = [
                (&mut state) as *mut u64 as *mut core::ffi::c_void,
                (&mut k) as *mut u64 as *mut core::ffi::c_void,
                (&mut v) as *mut u64 as *mut core::ffi::c_void,
                (&mut alpha) as *mut u64 as *mut core::ffi::c_void,
                (&mut beta) as *mut u64 as *mut core::ffi::c_void,
                (&mut nh) as *mut i32 as *mut core::ffi::c_void,
                (&mut dk) as *mut i32 as *mut core::ffi::c_void,
                (&mut dv) as *mut i32 as *mut core::ffi::c_void,
            ];
            let _ = cuLaunchKernel(
                self.outside_kernels.fn_gated_delta_state_update_f16.raw() as CUfunction,
                num_heads, 1, 1,
                16, 16, 1,
                0,
                self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
        }
        self.stream.fence()?;

        // 7. Host: read state, compute Q · S → readout [num_heads, d_state].
        let mut state_host = vec![0u8; state_bytes];
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let _ = cuMemcpyDtoH_v2(
                state_host.as_mut_ptr() as *mut _,
                state_region.device_ptr(),
                state_bytes,
            );
        }
        let mut readout_f32 =
            vec![0.0f32; (num_heads as usize) * (d_state as usize)];
        for h in 0..num_heads as usize {
            for v in 0..d_state as usize {
                let mut acc = 0.0f32;
                for k in 0..d_state as usize {
                    let s_idx =
                        h * (d_state as usize) * (d_state as usize) + v * (d_state as usize) + k;
                    let s = f16_bits_to_f32(u16::from_le_bytes([
                        state_host[s_idx * 2], state_host[s_idx * 2 + 1],
                    ]));
                    let q_idx = h * (d_state as usize) + k;
                    let q = f16_bits_to_f32(u16::from_le_bytes([
                        q_host[q_idx * 2], q_host[q_idx * 2 + 1],
                    ]));
                    acc += s * q;
                }
                readout_f32[h * (d_state as usize) + v] = acc;
            }
        }

        // 8. in_proj_z FP8 GEMV → z_logits, host sigmoid · readout,
        //    then host: per-head norm with linear_attn.norm.weight,
        //    then out_proj FP8 GEMV.
        let z_bytes_dev = (z_n as usize) * 2;
        let z_region = self.arena.region("qwen36_l0lc_z", z_bytes_dev, 16)?;
        let z_bs = la.in_proj_z.blockscale_ptr.unwrap_or(0);
        if z_bs == 0 {
            return Ok(());
        }
        unsafe {
            rvllm_fused::gemma4_launcher::Fp8GemvF16InLaunch {
                m, n: z_n, k: hidden,
            }.launch(
                kernel_gemv,
                z_region.device_ptr(),
                la.in_proj_z.offset_bytes,
                z_bs,
                in_region.device_ptr(),
                self.stream.raw() as u64,
            )?;
        }
        self.stream.fence()?;
        let mut z_host_bytes = vec![0u8; z_bytes_dev];
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let _ = cuMemcpyDtoH_v2(
                z_host_bytes.as_mut_ptr() as *mut _,
                z_region.device_ptr(),
                z_bytes_dev,
            );
        }
        // Per-head RMSNorm on the readout (gamma = norm.weight [128]).
        let mut norm_gamma = vec![0u8; (d_state as usize) * 2];
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let _ = cuMemcpyDtoH_v2(
                norm_gamma.as_mut_ptr() as *mut _,
                la.norm.offset_bytes,
                norm_gamma.len(),
            );
        }
        let mut gated_readout = vec![0u8; (num_heads as usize) * (d_state as usize) * 2];
        for h in 0..num_heads as usize {
            let mut sumsq = 0.0f32;
            for d in 0..d_state as usize {
                let v = readout_f32[h * (d_state as usize) + d];
                sumsq += v * v;
            }
            let rms = (sumsq / d_state as f32 + 1e-6).sqrt();
            for d in 0..d_state as usize {
                let v = readout_f32[h * (d_state as usize) + d] / rms;
                let g = f16_bits_to_f32(u16::from_le_bytes([
                    norm_gamma[d * 2], norm_gamma[d * 2 + 1],
                ]));
                let z_logit = f16_bits_to_f32(u16::from_le_bytes([
                    z_host_bytes[(h * (d_state as usize) + d) * 2],
                    z_host_bytes[(h * (d_state as usize) + d) * 2 + 1],
                ]));
                let sigmoid_z = 1.0f32 / (1.0f32 + (-z_logit).exp());
                let out = v * g * sigmoid_z;
                let bytes = f32_to_f16_bits(out).to_le_bytes();
                gated_readout[(h * (d_state as usize) + d) * 2] = bytes[0];
                gated_readout[(h * (d_state as usize) + d) * 2 + 1] = bytes[1];
            }
        }
        let gated_region =
            self.arena.region("qwen36_l0lc_gated", gated_readout.len(), 16)?;
        unsafe { gated_region.copy_from_host(&gated_readout)? };

        // out_proj: [hidden, num_heads*d_state=4096] FP8 GEMV.
        let out_bytes = (out_n as usize) * 2;
        let out_region = self.arena.region("qwen36_l0lc_out", out_bytes, 16)?;
        let out_bs = la.out_proj.blockscale_ptr.unwrap_or(0);
        if out_bs == 0 {
            return Ok(());
        }
        unsafe {
            rvllm_fused::gemma4_launcher::Fp8GemvF16InLaunch {
                m, n: out_n, k: out_k,
            }.launch(
                kernel_gemv,
                out_region.device_ptr(),
                la.out_proj.offset_bytes,
                out_bs,
                gated_region.device_ptr(),
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
            "[qwen36] forward_layer0_linear_chain_probe: layer={layer_idx} \
             chain in_proj_qkv → conv1d → silu+split → \
             {{α,β from A_log+dt_bias+dt}} → ssm_state_update → \
             Q·S readout → norm + sigmoid·z → out_proj → \
             out[0..4]=[{o0:.4}, {o1:.4}, {o2:.4}, {o3:.4}] \
             (full linear-attn block, single-step, state init=0)"
        );
        Ok(())
    }

    /// Phase 4u probe: verify the paged KV cache is zero-initialised
    /// + per-layer addressable + reset round-trips correctly.
    pub fn kv_cache_probe(&self) -> Result<()> {
        let n_full = self.kv_cache_bytes / self.kv_cache_layer_bytes;
        let l0_ptr = self.kv_cache_layer_ptr(0);
        let last_ptr = self.kv_cache_layer_ptr((n_full - 1) as u32);
        if l0_ptr == 0 || last_ptr == 0 {
            return Ok(());
        }
        let mut probe = [0xFFu8; 16];
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let _ = cuMemcpyDtoH_v2(probe.as_mut_ptr() as *mut _, l0_ptr, 8);
            let _ = cuMemcpyDtoH_v2(probe.as_mut_ptr().add(8) as *mut _, last_ptr, 8);
        }
        let l0_zero = probe[0..8].iter().all(|b| *b == 0);
        let lN_zero = probe[8..16].iter().all(|b| *b == 0);
        eprintln!(
            "[qwen36] kv_cache_probe: {n_full} full-attn slots, \
             layer0 head8b zeroed={l0_zero}, last_layer head8b zeroed={lN_zero}, \
             total={:.1} MiB persistent",
            self.kv_cache_bytes as f64 / (1024.0 * 1024.0),
        );
        Ok(())
    }

    /// Phase 4t probe: verify the persistent linear-attn state cache
    /// is zero-initialised + accessible. After bring-up the cache
    /// must hold all zeros at every layer's slice.
    pub fn linear_state_cache_probe(&self) -> Result<()> {
        let mut probe = [0xFFu8; 16];
        // Sample layer 0 + last linear layer's offsets.
        let n_linear_layers = self.linear_state_bytes / self.linear_state_layer_bytes;
        let last_layer = (n_linear_layers - 1) as u32;
        let l0_ptr = self.linear_state_layer_ptr(0);
        let ln_ptr = self.linear_state_layer_ptr(last_layer);
        if l0_ptr == 0 || ln_ptr == 0 {
            eprintln!("[qwen36] linear_state_cache_probe: pointers invalid, skipping");
            return Ok(());
        }
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let _ = cuMemcpyDtoH_v2(probe.as_mut_ptr() as *mut _, l0_ptr, 8);
            let _ = cuMemcpyDtoH_v2(probe.as_mut_ptr().add(8) as *mut _, ln_ptr, 8);
        }
        let l0_zero = probe[0..8].iter().all(|b| *b == 0);
        let ln_zero = probe[8..16].iter().all(|b| *b == 0);
        eprintln!(
            "[qwen36] linear_state_cache_probe: {n_linear_layers} layer slots, \
             layer0 head8b zeroed={l0_zero}, layer{last_layer} head8b zeroed={ln_zero}, \
             total={:.1} MiB persistent (above scratch checkpoint)",
            self.linear_state_bytes as f64 / (1024.0 * 1024.0),
        );
        // Reset round-trip: dirty layer 0, reset, re-read.
        let dirty: u8 = 0xAA;
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let _ = cuMemsetD8_v2(l0_ptr, dirty, 8);
        }
        self.reset_linear_state()?;
        self.stream.fence()?;
        let mut after = [0xFFu8; 8];
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let _ = cuMemcpyDtoH_v2(after.as_mut_ptr() as *mut _, l0_ptr, 8);
        }
        let reset_ok = after.iter().all(|b| *b == 0);
        eprintln!(
            "[qwen36] linear_state_cache_probe: reset round-trip — \
             dirty(0xAA) → reset_linear_state() → zeroed: {reset_ok}"
        );
        Ok(())
    }

    /// Phase 4r probe: launch the Gated-DeltaNet state-update kernel
    /// against an all-zero starting state, K=V=1.0, alpha=0.5,
    /// beta=2.0. After one step the state should equal `beta * 1.0 *
    /// 1.0 = 2.0` everywhere (decay term contributes 0 because the
    /// initial state is zero). Validates ABI + per-head broadcast +
    /// f16/f32 accumulator math.
    pub fn forward_layer0_ssm_state_probe(&self) -> Result<()> {
        // qwen3-next per-head dims: head_count=32 (A_log shape),
        // d_state = 128 (linear_attn.norm.weight shape).
        let num_heads: u32 = 32;
        let d_k: u32 = 128;
        let d_v: u32 = 128;
        let state_elems = (num_heads as usize) * (d_v as usize) * (d_k as usize);
        let state_bytes = state_elems * 2; // f16

        // K = V = 1.0 (f16 0x3c00); state starts at 0.
        let one_bits = 0x3c00u16.to_le_bytes();
        let zero_bytes = vec![0u8; state_bytes];
        let mut k_bytes = Vec::with_capacity((num_heads as usize) * (d_k as usize) * 2);
        let mut v_bytes = Vec::with_capacity((num_heads as usize) * (d_v as usize) * 2);
        for _ in 0..(num_heads as usize) * (d_k as usize) {
            k_bytes.extend_from_slice(&one_bits);
        }
        for _ in 0..(num_heads as usize) * (d_v as usize) {
            v_bytes.extend_from_slice(&one_bits);
        }
        let alpha_host: Vec<u8> = (0..num_heads)
            .flat_map(|_| 0.5f32.to_le_bytes())
            .collect();
        let beta_host: Vec<u8> = (0..num_heads)
            .flat_map(|_| 2.0f32.to_le_bytes())
            .collect();

        let state_region = self.arena.region("qwen36_l0ssm_s", state_bytes, 16)?;
        let k_region = self.arena.region("qwen36_l0ssm_k", k_bytes.len(), 16)?;
        let v_region = self.arena.region("qwen36_l0ssm_v", v_bytes.len(), 16)?;
        let alpha_region = self.arena.region("qwen36_l0ssm_a", alpha_host.len(), 16)?;
        let beta_region = self.arena.region("qwen36_l0ssm_b", beta_host.len(), 16)?;
        unsafe {
            state_region.copy_from_host(&zero_bytes)?;
            k_region.copy_from_host(&k_bytes)?;
            v_region.copy_from_host(&v_bytes)?;
            alpha_region.copy_from_host(&alpha_host)?;
            beta_region.copy_from_host(&beta_host)?;
        }

        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let mut state = state_region.device_ptr();
            let mut k = k_region.device_ptr();
            let mut v = v_region.device_ptr();
            let mut alpha = alpha_region.device_ptr();
            let mut beta = beta_region.device_ptr();
            let mut nh = num_heads as i32;
            let mut dk = d_k as i32;
            let mut dv = d_v as i32;
            let args = [
                (&mut state) as *mut u64 as *mut core::ffi::c_void,
                (&mut k) as *mut u64 as *mut core::ffi::c_void,
                (&mut v) as *mut u64 as *mut core::ffi::c_void,
                (&mut alpha) as *mut u64 as *mut core::ffi::c_void,
                (&mut beta) as *mut u64 as *mut core::ffi::c_void,
                (&mut nh) as *mut i32 as *mut core::ffi::c_void,
                (&mut dk) as *mut i32 as *mut core::ffi::c_void,
                (&mut dv) as *mut i32 as *mut core::ffi::c_void,
            ];
            // 16x16 = 256 threads per head; each thread strides over
            // its share of the 128*128=16384 state elements.
            let rc = cuLaunchKernel(
                self.outside_kernels.fn_gated_delta_state_update_f16.raw() as CUfunction,
                num_heads, 1, 1,
                16, 16, 1,
                0,
                self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36_l0ssm gated_delta_state_update_f16",
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
                state_region.device_ptr(),
                probe.len(),
            );
        }
        let s0 = f16_bits_to_f32(u16::from_le_bytes([probe[0], probe[1]]));
        let s1 = f16_bits_to_f32(u16::from_le_bytes([probe[2], probe[3]]));
        let s2 = f16_bits_to_f32(u16::from_le_bytes([probe[4], probe[5]]));
        let s3 = f16_bits_to_f32(u16::from_le_bytes([probe[6], probe[7]]));
        eprintln!(
            "[qwen36] forward_layer0_ssm_state_probe: heads={num_heads} \
             d_k={d_k} d_v={d_v} alpha=0.5 beta=2.0 K=V=1.0 init=0 → \
             state[head0,0..4]=[{s0:.3}, {s1:.3}, {s2:.3}, {s3:.3}] \
             (expected ≈ 2.0 — beta·K·V = 2·1·1 = 2; decay·init = 0)"
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

    /// Phase 4v: experimental forward path that threads the embedded
    /// hidden state through ONE real linear-attn layer (layer 0)
    /// before the lm_head closer. First time per-layer kernels run
    /// against the production decode path's actual hidden state
    /// (not synthetic probe input).
    ///
    /// Output is still expected to be wrong: the math approximation
    /// in the linear-attn chain is unverified, only one layer fires
    /// (39 layers skipped), positions aren't threaded, the
    /// state-cache uses the persistent layer-0 slot but never
    /// updates the per-token position. Phase 4w+ tightens against
    /// vLLM. Goal here: prove the per-layer kernels can consume the
    /// real hidden buffer that comes out of `embedding_gather` and
    /// feed `fused_rmsnorm_fp8_quant + cublaslt.fp8_gemm + argmax`.
    /// Phase 4x/5b/5c: full 40-layer decode loop. Linear-attn runs
    /// Gated-DeltaNet against the persistent state cache; full-attn
    /// Phase 2b-γ: full Qwen3-VL vision tower forward.
    ///
    /// Decodes an image (PNG/JPEG/WebP), runs the 27-layer ViT +
    /// PatchMerger natively in Rust+CUDA, returns f16 embeddings ready
    /// for splice into the post-embed text-side hidden buffer.
    ///
    /// Output layout: `[num_merged_tokens, 2048]` f16 (out_hidden_size
    /// == text hidden_size for Qwen 3.6).
    pub fn forward_qwen_vision(
        &self,
        image_bytes: &[u8],
    ) -> Result<VisionForwardOutput> {
        use crate::vision_preprocess::{
            decode_image, preprocess_qwen, QwenPreprocessConfig,
        };
        let vision = self.model.vision.as_ref().ok_or_else(|| {
            rvllm_core::RvllmError::cuda(
                "vision: model.visual not loaded",
                rvllm_core::CudaErrorKind::Other,
                rvllm_core::CudaCtx::setup(),
            )
        })?;

        // ── Step 1: decode + preprocess (CPU). ───────────────────────
        let img = decode_image(image_bytes).map_err(|e| {
            rvllm_core::RvllmError::cuda(
                "vision: image decode failed",
                rvllm_core::CudaErrorKind::Other,
                rvllm_core::CudaCtx::setup(),
            )
        })?;
        let cfg = QwenPreprocessConfig::default();
        let pp = preprocess_qwen(&img, &cfg).map_err(|e| {
            rvllm_core::RvllmError::cuda(
                "vision: preprocess failed",
                rvllm_core::CudaErrorKind::Other,
                rvllm_core::CudaCtx::setup(),
            )
        })?;
        let [grid_t, grid_h, grid_w] = pp.grid_thw;
        let n_tokens = (grid_t as usize) * (grid_h as usize) * (grid_w as usize);
        let patch_dim: usize = 1536;
        let hidden: usize = 1152;
        let num_heads: usize = 16;
        let head_dim: usize = 72;
        let intermediate: usize = 4304;
        let merge: usize = 2;
        let merge_sq = merge * merge;
        let merger_in: usize = hidden * merge_sq; // 4608
        let out_hidden: usize = 2048;
        let n_merged = n_tokens / merge_sq;

        // ── Step 2: upload patches as f16. ──────────────────────────
        let patches_bytes = pp.to_f16_bytes();
        let patches_region =
            self.arena.region("qvis_patches", patches_bytes.len(), 16)?;
        unsafe { patches_region.copy_from_host(&patches_bytes)? };

        let stream_raw = self.stream.raw() as u64;

        // Helper: linear with bias (in_f16 [M,K] @ W^T [N,K] + b[N] →
        // out_f16 [M,N]). Implemented as cuBLASLt f16-GEMM-f32 + cast +
        // bias-add. The work-buffer is allocated by the caller because
        // ranges depend on M, N.
        let linear_with_bias = |in_dev: u64,
                                w_dev: u64,
                                b_dev: u64,
                                out_dev: u64,
                                f32_scratch: u64,
                                m: usize,
                                n: usize,
                                k: usize|
         -> Result<()> {
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                self.cublaslt.f16_gemm_f32(
                    in_dev, w_dev, f32_scratch,
                    m as i32, n as i32, k as i32,
                    stream_raw,
                )?;
                // Cast f32 → f16
                let n_elem = (m * n) as i32;
                let mut out = out_dev;
                let mut input = f32_scratch;
                let mut nn = n_elem;
                let args = [
                    (&mut out) as *mut u64 as *mut core::ffi::c_void,
                    (&mut input) as *mut u64 as *mut core::ffi::c_void,
                    (&mut nn) as *mut i32 as *mut core::ffi::c_void,
                ];
                let block: u32 = 256;
                let grid = ((n_elem as u32 + block - 1) / block, 1u32, 1u32);
                let _ = cuLaunchKernel(
                    self.outside_kernels.fn_cast_f32_to_f16.raw() as CUfunction,
                    grid.0, grid.1, grid.2,
                    block, 1, 1,
                    0, self.stream.raw() as CUstream,
                    args.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                );
                // Add bias in-place
                let mut tensor = out_dev;
                let mut bias = b_dev;
                let mut dim = n as i32;
                let bargs = [
                    (&mut tensor) as *mut u64 as *mut core::ffi::c_void,
                    (&mut bias) as *mut u64 as *mut core::ffi::c_void,
                    (&mut dim) as *mut i32 as *mut core::ffi::c_void,
                ];
                let block_b: u32 = (n as u32).min(1024);
                let _ = cuLaunchKernel(
                    self.outside_kernels.fn_add_bias_f16.raw() as CUfunction,
                    m as u32, 1, 1,
                    block_b, 1, 1,
                    0, self.stream.raw() as CUstream,
                    bargs.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                );
            }
            self.stream.fence()?;
            Ok(())
        };

        // ── Step 3: patch_embed: patches [N, 1536] @ W^T [1152, 1536] + b. ─
        let hidden_bytes = n_tokens * hidden * 2;
        let f32_scratch = self
            .arena
            .region("qvis_f32_scratch", n_tokens * intermediate.max(hidden) * 4, 16)?;
        let hidden_region = self.arena.region("qvis_hidden", hidden_bytes, 16)?;
        linear_with_bias(
            patches_region.device_ptr(),
            vision.patch_embed.proj_weight.offset_bytes,
            vision.patch_embed.proj_bias.offset_bytes,
            hidden_region.device_ptr(),
            f32_scratch.device_ptr(),
            n_tokens, hidden, patch_dim,
        )?;

        // Stage dump: post patch_embed, pre pos_embed.
        if let Ok(path) = std::env::var("RVLLM_QWEN36_VIT_PATCH_EMBED_DUMP") {
            let bytes = n_tokens * hidden * 2;
            let mut host = vec![0u8; bytes];
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let _ = cuMemcpyDtoH_v2(host.as_mut_ptr() as *mut _, hidden_region.device_ptr(), bytes);
            }
            let _ = std::fs::write(&path, &host);
            eprintln!("[qwen36] vit patch_embed dump: {} bytes ([{},{}]) → {}", bytes, n_tokens, hidden, path);
        }

        // ── Step 3.5: add learned absolute pos_embed (bilinear-interp). ─
        // vllm qwen3_vl.py:801: hidden_states += pos_embeds. The pos_embed
        // table is `[2304, 1152]` (num_grid_per_side²); we interpolate it
        // to `[grid_h * grid_w, 1152]` and add to hidden_region in place.
        const QWEN_VIT_NUM_GRID: i32 = 48; // sqrt(num_position_embeddings=2304)
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let mut hs = hidden_region.device_ptr();
            let mut tab = vision.pos_embed.offset_bytes;
            let mut gh = grid_h as i32;
            let mut gw = grid_w as i32;
            let mut ng = QWEN_VIT_NUM_GRID;
            let mut ms = merge as i32;
            let mut hd = hidden as i32;
            let args = [
                (&mut hs) as *mut u64 as *mut core::ffi::c_void,
                (&mut tab) as *mut u64 as *mut core::ffi::c_void,
                (&mut gh) as *mut i32 as *mut core::ffi::c_void,
                (&mut gw) as *mut i32 as *mut core::ffi::c_void,
                (&mut ng) as *mut i32 as *mut core::ffi::c_void,
                (&mut ms) as *mut i32 as *mut core::ffi::c_void,
                (&mut hd) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 256;
            let _ = cuLaunchKernel(
                self.outside_kernels.fn_vit_pos_embed_interp_f16.raw() as CUfunction,
                n_tokens as u32, 1, 1,
                block, 1, 1,
                0, self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
        }
        self.stream.fence()?;

        // Stage dump: post pos_embed (before any block).
        if let Ok(path) = std::env::var("RVLLM_QWEN36_VIT_POSEMB_DUMP") {
            let bytes = n_tokens * hidden * 2;
            let mut host = vec![0u8; bytes];
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let _ = cuMemcpyDtoH_v2(host.as_mut_ptr() as *mut _, hidden_region.device_ptr(), bytes);
            }
            let _ = std::fs::write(&path, &host);
            eprintln!("[qwen36] vit posemb dump: {} bytes → {}", bytes, path);
        }

        // ── Step 4: build per-token cos/sin tables for 2D rotary. ────
        // Matches HF transformers Qwen3VLVisionModel:
        //   freq_table = RotaryEmbedding(head_dim/2)(max_hw)   [max_hw, 18]
        //   embeddings = freq_table[pos_ids]                   [N, 2, 18]
        //   embeddings = embeddings.flatten(1)                 [N, 36]
        //   emb = cat([embeddings, embeddings], dim=-1)        [N, 72] (DUPLICATED)
        //   cos, sin = emb.cos(), emb.sin()                    [N, 72]
        // Rotary then applies rotate_half on the FULL head_dim=72 with these
        // tables. The "partial_rotary" interpretation from vLLM's get_rope
        // call was misleading — the effective rotary_dim is head_dim, not
        // head_dim/2. Within the [N, 72] cos table:
        //   cos[t, 0..18]   = cos(h_pos[t] * inv_freq[k])
        //   cos[t, 18..36]  = cos(w_pos[t] * inv_freq[k])
        //   cos[t, 36..54]  = cos(h_pos[t] * inv_freq[k])  (mirror of 0..18)
        //   cos[t, 54..72]  = cos(w_pos[t] * inv_freq[k])  (mirror of 18..36)
        // inv_freq dim = 18, with theta=10000 over (head_dim/2).
        let rotary_dim = head_dim;            // 72 (full head_dim — see HF cat-trick above)
        let inv_freq_dim = head_dim / 4;      // 18
        let inv_freq_dim_total = head_dim / 2; // 36 — pre-cat embedding width
        let inv_theta: Vec<f32> = (0..inv_freq_dim)
            .map(|k| 1.0 / 10_000.0_f32.powf(2.0 * k as f32 / inv_freq_dim_total as f32))
            .collect();
        let mut cos_table_host = vec![0u8; n_tokens * rotary_dim * 2];
        let mut sin_table_host = vec![0u8; n_tokens * rotary_dim * 2];
        // Determine per-token (h_pos, w_pos) following HF's rot_pos_emb
        // (block-internal merge-aware ordering).
        let mut pos_h = vec![0i32; n_tokens];
        let mut pos_w = vec![0i32; n_tokens];
        {
            let merged_h = (grid_h as usize) / merge;
            let merged_w = (grid_w as usize) / merge;
            let mut idx = 0usize;
            for _t in 0..(grid_t as usize) {
                for bh in 0..merged_h {
                    for bw in 0..merged_w {
                        for ih in 0..merge {
                            for iw in 0..merge {
                                pos_h[idx] = (bh * merge + ih) as i32;
                                pos_w[idx] = (bw * merge + iw) as i32;
                                idx += 1;
                            }
                        }
                    }
                }
            }
        }
        // Build [N, 72] cos/sin tables matching HF cat-of-itself layout:
        //   table[t, k]            = cos/sin(h_pos[t] * inv_freq[k])     for k ∈ [0, 18)
        //   table[t, 18+k]         = cos/sin(w_pos[t] * inv_freq[k])
        //   table[t, 36+k]         = same as table[t, k]                  (cat duplicate)
        //   table[t, 54+k]         = same as table[t, 18+k]               (cat duplicate)
        for t in 0..n_tokens {
            for k in 0..inv_freq_dim {
                let ah = (pos_h[t] as f32) * inv_theta[k];
                let aw = (pos_w[t] as f32) * inv_theta[k];
                let cos_h = half::f16::from_f32(ah.cos()).to_le_bytes();
                let sin_h = half::f16::from_f32(ah.sin()).to_le_bytes();
                let cos_w = half::f16::from_f32(aw.cos()).to_le_bytes();
                let sin_w = half::f16::from_f32(aw.sin()).to_le_bytes();
                let row_base = t * rotary_dim;
                for &mirror in &[0, inv_freq_dim_total] {
                    let off_h = (row_base + mirror + k) * 2;
                    let off_w = (row_base + mirror + inv_freq_dim + k) * 2;
                    cos_table_host[off_h] = cos_h[0]; cos_table_host[off_h + 1] = cos_h[1];
                    sin_table_host[off_h] = sin_h[0]; sin_table_host[off_h + 1] = sin_h[1];
                    cos_table_host[off_w] = cos_w[0]; cos_table_host[off_w + 1] = cos_w[1];
                    sin_table_host[off_w] = sin_w[0]; sin_table_host[off_w + 1] = sin_w[1];
                }
            }
        }
        let cos_region = self.arena.region("qvis_cos", cos_table_host.len(), 16)?;
        let sin_region = self.arena.region("qvis_sin", sin_table_host.len(), 16)?;
        unsafe {
            cos_region.copy_from_host(&cos_table_host)?;
            sin_region.copy_from_host(&sin_table_host)?;
        }

        // ── Step 5: 27-block transformer loop. ──────────────────────
        // Persistent scratch for QKV, attn-out, MLP intermediate, scores.
        let qkv_bytes = n_tokens * 3 * hidden * 2;
        let qkv_region = self.arena.region("qvis_qkv", qkv_bytes, 16)?;
        let q_buf = self.arena.region("qvis_q", n_tokens * hidden * 2, 16)?;
        let k_buf = self.arena.region("qvis_k", n_tokens * hidden * 2, 16)?;
        let v_buf = self.arena.region("qvis_v", n_tokens * hidden * 2, 16)?;
        let attn_out = self.arena.region("qvis_attn_out", n_tokens * hidden * 2, 16)?;
        let mlp_buf = self.arena.region("qvis_mlp", n_tokens * intermediate * 2, 16)?;
        let scores_bytes = n_tokens * n_tokens * 2;
        let scores_buf = self.arena.region("qvis_scores", scores_bytes, 16)?;
        let scores_f32 = self.arena.region("qvis_scores_f32", n_tokens * n_tokens * 4, 16)?;

        let qkv_eps = 1e-6f32;
        let blk_dump_dir = std::env::var("RVLLM_QWEN36_VIT_BLK_DUMP_DIR").ok();
        for (blk_idx, blk) in vision.blocks.iter().enumerate() {
            // ─ pre-attn LayerNorm on a copy ─
            let normed = self.arena.region("qvis_normed", n_tokens * hidden * 2, 16)?;
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let _ = cuMemcpyDtoDAsync_v2(
                    normed.device_ptr(),
                    hidden_region.device_ptr(),
                    n_tokens * hidden * 2,
                    self.stream.raw() as _,
                );
                let mut x = normed.device_ptr();
                let mut g = blk.norm1_w.offset_bytes;
                let mut b = blk.norm1_b.offset_bytes;
                let mut eps = qkv_eps;
                let mut hd_i = hidden as i32;
                let args = [
                    (&mut x) as *mut u64 as *mut core::ffi::c_void,
                    (&mut g) as *mut u64 as *mut core::ffi::c_void,
                    (&mut b) as *mut u64 as *mut core::ffi::c_void,
                    (&mut eps) as *mut f32 as *mut core::ffi::c_void,
                    (&mut hd_i) as *mut i32 as *mut core::ffi::c_void,
                ];
                let block: u32 = (hidden as u32).min(1024);
                let _ = cuLaunchKernel(
                    self.outside_kernels.fn_layernorm_inplace_f16.raw() as CUfunction,
                    n_tokens as u32, 1, 1,
                    block, 1, 1,
                    0, self.stream.raw() as CUstream,
                    args.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                );
            }
            self.stream.fence()?;

            // Block 0 dump: norm1 output (= input to QKV proj).
            if blk_idx == 0 {
                if let Some(dir) = blk_dump_dir.as_deref() {
                    let bytes = n_tokens * hidden * 2;
                    let mut host = vec![0u8; bytes];
                    #[cfg(feature = "cuda")]
                    unsafe {
                        use cudarc::driver::sys::*;
                        let _ = cuMemcpyDtoH_v2(host.as_mut_ptr() as *mut _, normed.device_ptr(), bytes);
                    }
                    let _ = std::fs::write(format!("{dir}/blk0_norm1_out.bin"), &host);
                }
            }

            // ─ QKV proj: normed [N, 1152] @ W^T [3456, 1152] + b. ─
            linear_with_bias(
                normed.device_ptr(),
                blk.qkv_w.offset_bytes,
                blk.qkv_b.offset_bytes,
                qkv_region.device_ptr(),
                f32_scratch.device_ptr(),
                n_tokens, 3 * hidden, hidden,
            )?;

            // ─ Split QKV → Q, K, V (each [N, 1152]). HF lays them out
            //   as [N, 3*hidden] = (Q[N,hidden], K[N,hidden], V[N,hidden]). ─
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let row_bytes = (hidden * 2) as u64;
                for t in 0..n_tokens {
                    let src = qkv_region.device_ptr() + (t as u64) * 3 * row_bytes;
                    let _ = cuMemcpyDtoDAsync_v2(
                        q_buf.device_ptr() + (t as u64) * row_bytes,
                        src,
                        row_bytes as usize,
                        self.stream.raw() as _,
                    );
                    let _ = cuMemcpyDtoDAsync_v2(
                        k_buf.device_ptr() + (t as u64) * row_bytes,
                        src + row_bytes,
                        row_bytes as usize,
                        self.stream.raw() as _,
                    );
                    let _ = cuMemcpyDtoDAsync_v2(
                        v_buf.device_ptr() + (t as u64) * row_bytes,
                        src + 2 * row_bytes,
                        row_bytes as usize,
                        self.stream.raw() as _,
                    );
                }
            }
            self.stream.fence()?;

            // ─ Apply 2D rotary to Q, K. ─
            for &qk_ptr in &[q_buf.device_ptr(), k_buf.device_ptr()] {
                #[cfg(feature = "cuda")]
                unsafe {
                    use cudarc::driver::sys::*;
                    let mut x = qk_ptr;
                    let mut cos = cos_region.device_ptr();
                    let mut sin = sin_region.device_ptr();
                    let mut nh = num_heads as i32;
                    let mut hd_i = head_dim as i32;
                    let mut rd_i = rotary_dim as i32;
                    let args = [
                        (&mut x) as *mut u64 as *mut core::ffi::c_void,
                        (&mut cos) as *mut u64 as *mut core::ffi::c_void,
                        (&mut sin) as *mut u64 as *mut core::ffi::c_void,
                        (&mut nh) as *mut i32 as *mut core::ffi::c_void,
                        (&mut hd_i) as *mut i32 as *mut core::ffi::c_void,
                        (&mut rd_i) as *mut i32 as *mut core::ffi::c_void,
                    ];
                    let _ = cuLaunchKernel(
                        self.outside_kernels.fn_vit_rotary_2d_f16.raw() as CUfunction,
                        n_tokens as u32, num_heads as u32, 1,
                        (rotary_dim / 2) as u32, 1, 1,
                        0, self.stream.raw() as CUstream,
                        args.as_ptr() as *mut *mut core::ffi::c_void,
                        core::ptr::null_mut(),
                    );
                }
            }
            self.stream.fence()?;

            // Block 0: dump Q + K post-rotary (full [N, hidden] f16).
            if blk_idx == 0 {
                if let Some(dir) = blk_dump_dir.as_deref() {
                    let bytes = n_tokens * hidden * 2;
                    let mut q_host = vec![0u8; bytes];
                    let mut k_host = vec![0u8; bytes];
                    let mut v_host = vec![0u8; bytes];
                    #[cfg(feature = "cuda")]
                    unsafe {
                        use cudarc::driver::sys::*;
                        let _ = cuMemcpyDtoH_v2(q_host.as_mut_ptr() as *mut _, q_buf.device_ptr(), bytes);
                        let _ = cuMemcpyDtoH_v2(k_host.as_mut_ptr() as *mut _, k_buf.device_ptr(), bytes);
                        let _ = cuMemcpyDtoH_v2(v_host.as_mut_ptr() as *mut _, v_buf.device_ptr(), bytes);
                    }
                    let _ = std::fs::write(format!("{dir}/blk0_q_postrot.bin"), &q_host);
                    let _ = std::fs::write(format!("{dir}/blk0_k_postrot.bin"), &k_host);
                    let _ = std::fs::write(format!("{dir}/blk0_v.bin"), &v_host);
                }
            }

            // ─ Per-head attention: QK^T → softmax → @V. ─
            // Q, K, V layout is [N, num_heads*head_dim] = [N, 1152]
            // with each token's row containing all heads concatenated.
            // For head h, head data lives at offset h*head_dim within
            // each row, with stride hidden bytes.
            //
            // To get a contiguous [N, head_dim] per head, we copy each
            // head's slice into temporary buffers. For all 16 heads
            // we use a single round-robin scratch (q_h, k_h, v_h, out_h).
            let q_h = self.arena.region("qvis_qh", n_tokens * head_dim * 2, 16)?;
            let k_h = self.arena.region("qvis_kh", n_tokens * head_dim * 2, 16)?;
            let v_h = self.arena.region("qvis_vh", n_tokens * head_dim * 2, 16)?;
            let v_h_t = self.arena.region("qvis_vht", head_dim * n_tokens * 2, 16)?;
            let out_h = self.arena.region("qvis_oh", n_tokens * head_dim * 2, 16)?;
            let scale = 1.0f32 / (head_dim as f32).sqrt();
            // Gather/scatter helpers shared with the per-head attention
            // loop. Replaces the per-token DtoD memcpy schedule (~ N
            // memcpys × 4 directions × num_heads × 27 blocks = O(106k)
            // launches per image at N=196) with a single kernel
            // launch per (head, direction). Codex review #C round 3.
            let extract_head = |dst: u64, src: u64, head_idx: usize| -> Result<()> {
                #[cfg(feature = "cuda")]
                unsafe {
                    use cudarc::driver::sys::*;
                    let mut o = dst;
                    let mut i = src;
                    let mut hi = head_idx as i32;
                    let mut nh = num_heads as i32;
                    let mut hd = head_dim as i32;
                    let args = [
                        (&mut o) as *mut u64 as *mut core::ffi::c_void,
                        (&mut i) as *mut u64 as *mut core::ffi::c_void,
                        (&mut hi) as *mut i32 as *mut core::ffi::c_void,
                        (&mut nh) as *mut i32 as *mut core::ffi::c_void,
                        (&mut hd) as *mut i32 as *mut core::ffi::c_void,
                    ];
                    let rc = cuLaunchKernel(
                        self.outside_kernels.fn_extract_head_f16.raw() as CUfunction,
                        n_tokens as u32, 1, 1,
                        head_dim as u32, 1, 1,
                        0, self.stream.raw() as CUstream,
                        args.as_ptr() as *mut *mut core::ffi::c_void,
                        core::ptr::null_mut(),
                    );
                    if rc != CUresult::CUDA_SUCCESS {
                        return Err(rvllm_core::RvllmError::cuda(
                            "qvis: extract_head launch failed",
                            rvllm_core::CudaErrorKind::LaunchFailed,
                            rvllm_core::CudaCtx::setup(),
                        ));
                    }
                }
                Ok(())
            };
            let scatter_head = |dst: u64, src: u64, head_idx: usize| -> Result<()> {
                #[cfg(feature = "cuda")]
                unsafe {
                    use cudarc::driver::sys::*;
                    let mut o = dst;
                    let mut i = src;
                    let mut hi = head_idx as i32;
                    let mut nh = num_heads as i32;
                    let mut hd = head_dim as i32;
                    let args = [
                        (&mut o) as *mut u64 as *mut core::ffi::c_void,
                        (&mut i) as *mut u64 as *mut core::ffi::c_void,
                        (&mut hi) as *mut i32 as *mut core::ffi::c_void,
                        (&mut nh) as *mut i32 as *mut core::ffi::c_void,
                        (&mut hd) as *mut i32 as *mut core::ffi::c_void,
                    ];
                    let rc = cuLaunchKernel(
                        self.outside_kernels.fn_scatter_head_f16.raw() as CUfunction,
                        n_tokens as u32, 1, 1,
                        head_dim as u32, 1, 1,
                        0, self.stream.raw() as CUstream,
                        args.as_ptr() as *mut *mut core::ffi::c_void,
                        core::ptr::null_mut(),
                    );
                    if rc != CUresult::CUDA_SUCCESS {
                        return Err(rvllm_core::RvllmError::cuda(
                            "qvis: scatter_head launch failed",
                            rvllm_core::CudaErrorKind::LaunchFailed,
                            rvllm_core::CudaCtx::setup(),
                        ));
                    }
                }
                Ok(())
            };
            for h in 0..num_heads {
                extract_head(q_h.device_ptr(), q_buf.device_ptr(), h)?;
                extract_head(k_h.device_ptr(), k_buf.device_ptr(), h)?;
                extract_head(v_h.device_ptr(), v_buf.device_ptr(), h)?;
                self.stream.fence()?;

                // QK^T: [N, head_dim] @ [N, head_dim]^T → [N, N] f32 → cast f16
                // Use cuBLASLt f16_gemm_f32 with N=N, M=N, K=head_dim.
                #[cfg(feature = "cuda")]
                unsafe {
                    self.cublaslt.f16_gemm_f32(
                        q_h.device_ptr(), k_h.device_ptr(),
                        scores_f32.device_ptr(),
                        n_tokens as i32, n_tokens as i32, head_dim as i32,
                        stream_raw,
                    )?;
                    // Apply scale + cast to f16: scores_f16[i] = (scores_f32[i] * scale) → f16
                    // We do this by scaling f32 first (in-place via simple kernel — reuse cast
                    // since we don't have a fused scale_cast: just bake scale into cos/sin? No,
                    // just multiply scale into Q before GEMM. Move scaling there.)
                    let n_elem = (n_tokens * n_tokens) as i32;
                    let mut out = scores_buf.device_ptr();
                    let mut input = scores_f32.device_ptr();
                    let mut nn = n_elem;
                    let args = [
                        (&mut out) as *mut u64 as *mut core::ffi::c_void,
                        (&mut input) as *mut u64 as *mut core::ffi::c_void,
                        (&mut nn) as *mut i32 as *mut core::ffi::c_void,
                    ];
                    let block: u32 = 256;
                    let grid = ((n_elem as u32 + block - 1) / block, 1u32, 1u32);
                    use cudarc::driver::sys::*;
                    let _ = cuLaunchKernel(
                        self.outside_kernels.fn_cast_f32_to_f16.raw() as CUfunction,
                        grid.0, grid.1, grid.2,
                        block, 1, 1,
                        0, self.stream.raw() as CUstream,
                        args.as_ptr() as *mut *mut core::ffi::c_void,
                        core::ptr::null_mut(),
                    );
                }
                self.stream.fence()?;

                // Apply the standard 1/sqrt(head_dim) attention scale
                // to the f16 scores in place before softmax. (Without
                // this, softmax becomes degenerate — one token wins
                // ~all attention — and patches stop mixing, which
                // produces image-content-blind ViT output.)
                #[cfg(feature = "cuda")]
                unsafe {
                    use cudarc::driver::sys::*;
                    let mut x = scores_buf.device_ptr();
                    let mut s = scale;
                    let mut nn = (n_tokens * n_tokens) as i32;
                    let args = [
                        (&mut x) as *mut u64 as *mut core::ffi::c_void,
                        (&mut s) as *mut f32 as *mut core::ffi::c_void,
                        (&mut nn) as *mut i32 as *mut core::ffi::c_void,
                    ];
                    let block: u32 = 256;
                    let grid = ((nn as u32 + block - 1) / block, 1u32, 1u32);
                    let _ = cuLaunchKernel(
                        self.outside_kernels.fn_scale_inplace_f16.raw() as CUfunction,
                        grid.0, grid.1, grid.2,
                        block, 1, 1,
                        0, self.stream.raw() as CUstream,
                        args.as_ptr() as *mut *mut core::ffi::c_void,
                        core::ptr::null_mut(),
                    );
                }
                self.stream.fence()?;

                // Softmax row-wise on scores [N, N]
                #[cfg(feature = "cuda")]
                unsafe {
                    use cudarc::driver::sys::*;
                    let mut x = scores_buf.device_ptr();
                    let mut sl = n_tokens as i32;
                    let args = [
                        (&mut x) as *mut u64 as *mut core::ffi::c_void,
                        (&mut sl) as *mut i32 as *mut core::ffi::c_void,
                    ];
                    let block: u32 = (n_tokens as u32).min(1024);
                    let _ = cuLaunchKernel(
                        self.outside_kernels.fn_softmax_row_f16.raw() as CUfunction,
                        n_tokens as u32, 1, 1,
                        block, 1, 1,
                        0, self.stream.raw() as CUstream,
                        args.as_ptr() as *mut *mut core::ffi::c_void,
                        core::ptr::null_mut(),
                    );
                }
                self.stream.fence()?;

                // scores @ V: [N, N] @ [N, head_dim] → [N, head_dim].
                // f16_gemm_f32 always computes input @ weight^T, so we
                // need V transposed [head_dim, N] for the call to give
                // sum_j scores[r,j] * V[j,c] (instead of scores @ V^T).
                #[cfg(feature = "cuda")]
                unsafe {
                    use cudarc::driver::sys::*;
                    let mut out_p = v_h_t.device_ptr();
                    let mut in_p = v_h.device_ptr();
                    let mut rows = n_tokens as i32;
                    let mut cols = head_dim as i32;
                    let args = [
                        (&mut out_p) as *mut u64 as *mut core::ffi::c_void,
                        (&mut in_p) as *mut u64 as *mut core::ffi::c_void,
                        (&mut rows) as *mut i32 as *mut core::ffi::c_void,
                        (&mut cols) as *mut i32 as *mut core::ffi::c_void,
                    ];
                    let gx = ((cols as u32) + 15) / 16;
                    let gy = ((rows as u32) + 15) / 16;
                    let _ = cuLaunchKernel(
                        self.outside_kernels.fn_transpose_2d_f16.raw() as CUfunction,
                        gx, gy, 1,
                        16, 16, 1,
                        0, self.stream.raw() as CUstream,
                        args.as_ptr() as *mut *mut core::ffi::c_void,
                        core::ptr::null_mut(),
                    );
                }
                self.stream.fence()?;
                #[cfg(feature = "cuda")]
                unsafe {
                    self.cublaslt.f16_gemm_f32(
                        scores_buf.device_ptr(), v_h_t.device_ptr(),
                        scores_f32.device_ptr(),
                        n_tokens as i32, head_dim as i32, n_tokens as i32,
                        stream_raw,
                    )?;
                    // cast f32 → f16 into out_h
                    let n_elem = (n_tokens * head_dim) as i32;
                    let mut out = out_h.device_ptr();
                    let mut input = scores_f32.device_ptr();
                    let mut nn = n_elem;
                    let args = [
                        (&mut out) as *mut u64 as *mut core::ffi::c_void,
                        (&mut input) as *mut u64 as *mut core::ffi::c_void,
                        (&mut nn) as *mut i32 as *mut core::ffi::c_void,
                    ];
                    let block: u32 = 256;
                    let grid = ((n_elem as u32 + block - 1) / block, 1u32, 1u32);
                    use cudarc::driver::sys::*;
                    let _ = cuLaunchKernel(
                        self.outside_kernels.fn_cast_f32_to_f16.raw() as CUfunction,
                        grid.0, grid.1, grid.2,
                        block, 1, 1,
                        0, self.stream.raw() as CUstream,
                        args.as_ptr() as *mut *mut core::ffi::c_void,
                        core::ptr::null_mut(),
                    );
                }
                self.stream.fence()?;

                // Scatter out_h back into attn_out at offset h*head_dim per row.
                scatter_head(attn_out.device_ptr(), out_h.device_ptr(), h)?;
                self.stream.fence()?;
            }

            // ─ O proj + residual: hidden += proj(attn_out). ─
            // Block 0 dump: attn output pre-O-proj.
            if blk_idx == 0 {
                if let Some(dir) = blk_dump_dir.as_deref() {
                    let bytes = n_tokens * hidden * 2;
                    let mut host = vec![0u8; bytes];
                    #[cfg(feature = "cuda")]
                    unsafe {
                        use cudarc::driver::sys::*;
                        let _ = cuMemcpyDtoH_v2(host.as_mut_ptr() as *mut _, attn_out.device_ptr(), bytes);
                    }
                    let _ = std::fs::write(format!("{dir}/blk0_attn_pre_o_proj.bin"), &host);
                }
            }
            let proj_out = self.arena.region("qvis_proj_out", n_tokens * hidden * 2, 16)?;
            linear_with_bias(
                attn_out.device_ptr(),
                blk.proj_w.offset_bytes,
                blk.proj_b.offset_bytes,
                proj_out.device_ptr(),
                f32_scratch.device_ptr(),
                n_tokens, hidden, hidden,
            )?;
            // Residual: hidden += proj_out via GPU vector_add_f16
            // (replaces the earlier DtoH-add-HtoD round-trip per
            // block — Codex review #3 round 4 follow-up).
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let n_elem = (n_tokens * hidden) as i32;
                let mut d = hidden_region.device_ptr();
                let mut s = proj_out.device_ptr();
                let mut nn = n_elem;
                let args = [
                    (&mut d) as *mut u64 as *mut core::ffi::c_void,
                    (&mut s) as *mut u64 as *mut core::ffi::c_void,
                    (&mut nn) as *mut i32 as *mut core::ffi::c_void,
                ];
                let block: u32 = 256;
                let grid = ((n_elem as u32 + block - 1) / block, 1u32, 1u32);
                let rc = cuLaunchKernel(
                    self.outside_kernels.fn_vector_add_f16.raw() as CUfunction,
                    grid.0, grid.1, grid.2,
                    block, 1, 1,
                    0, self.stream.raw() as CUstream,
                    args.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "vision: vector_add (attn residual) launch failed",
                        rvllm_core::CudaErrorKind::LaunchFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            }
            self.stream.fence()?;

            // Block 0 dump: after attention residual (= input + attn_out).
            if blk_idx == 0 {
                if let Some(dir) = blk_dump_dir.as_deref() {
                    let bytes = n_tokens * hidden * 2;
                    let mut host = vec![0u8; bytes];
                    #[cfg(feature = "cuda")]
                    unsafe {
                        use cudarc::driver::sys::*;
                        let _ = cuMemcpyDtoH_v2(host.as_mut_ptr() as *mut _, hidden_region.device_ptr(), bytes);
                    }
                    let _ = std::fs::write(format!("{dir}/blk0_post_attn.bin"), &host);
                }
            }

            // ─ pre-MLP LayerNorm (norm2) on a copy ─
            let normed2 = self.arena.region("qvis_normed2", n_tokens * hidden * 2, 16)?;
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let _ = cuMemcpyDtoDAsync_v2(
                    normed2.device_ptr(),
                    hidden_region.device_ptr(),
                    n_tokens * hidden * 2,
                    self.stream.raw() as _,
                );
                let mut x = normed2.device_ptr();
                let mut g = blk.norm2_w.offset_bytes;
                let mut b = blk.norm2_b.offset_bytes;
                let mut eps = qkv_eps;
                let mut hd_i = hidden as i32;
                let args = [
                    (&mut x) as *mut u64 as *mut core::ffi::c_void,
                    (&mut g) as *mut u64 as *mut core::ffi::c_void,
                    (&mut b) as *mut u64 as *mut core::ffi::c_void,
                    (&mut eps) as *mut f32 as *mut core::ffi::c_void,
                    (&mut hd_i) as *mut i32 as *mut core::ffi::c_void,
                ];
                let block: u32 = (hidden as u32).min(1024);
                let _ = cuLaunchKernel(
                    self.outside_kernels.fn_layernorm_inplace_f16.raw() as CUfunction,
                    n_tokens as u32, 1, 1,
                    block, 1, 1,
                    0, self.stream.raw() as CUstream,
                    args.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                );
            }
            self.stream.fence()?;

            // ─ MLP: fc1 → GELU → fc2 ─
            linear_with_bias(
                normed2.device_ptr(),
                blk.fc1_w.offset_bytes,
                blk.fc1_b.offset_bytes,
                mlp_buf.device_ptr(),
                f32_scratch.device_ptr(),
                n_tokens, intermediate, hidden,
            )?;
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let n_elem = (n_tokens * intermediate) as i32;
                let mut x = mlp_buf.device_ptr();
                let mut nn = n_elem;
                let args = [
                    (&mut x) as *mut u64 as *mut core::ffi::c_void,
                    (&mut nn) as *mut i32 as *mut core::ffi::c_void,
                ];
                let block: u32 = 256;
                let grid = ((n_elem as u32 + block - 1) / block, 1u32, 1u32);
                let _ = cuLaunchKernel(
                    self.outside_kernels.fn_gelu_tanh_f16.raw() as CUfunction,
                    grid.0, grid.1, grid.2,
                    block, 1, 1,
                    0, self.stream.raw() as CUstream,
                    args.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                );
            }
            self.stream.fence()?;
            let mlp_out = self.arena.region("qvis_mlp_out", n_tokens * hidden * 2, 16)?;
            linear_with_bias(
                mlp_buf.device_ptr(),
                blk.fc2_w.offset_bytes,
                blk.fc2_b.offset_bytes,
                mlp_out.device_ptr(),
                f32_scratch.device_ptr(),
                n_tokens, hidden, intermediate,
            )?;
            // Residual: hidden += mlp_out via GPU vector_add_f16.
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let n_elem = (n_tokens * hidden) as i32;
                let mut d = hidden_region.device_ptr();
                let mut s = mlp_out.device_ptr();
                let mut nn = n_elem;
                let args = [
                    (&mut d) as *mut u64 as *mut core::ffi::c_void,
                    (&mut s) as *mut u64 as *mut core::ffi::c_void,
                    (&mut nn) as *mut i32 as *mut core::ffi::c_void,
                ];
                let block: u32 = 256;
                let grid = ((n_elem as u32 + block - 1) / block, 1u32, 1u32);
                let rc = cuLaunchKernel(
                    self.outside_kernels.fn_vector_add_f16.raw() as CUfunction,
                    grid.0, grid.1, grid.2,
                    block, 1, 1,
                    0, self.stream.raw() as CUstream,
                    args.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                );
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(rvllm_core::RvllmError::cuda(
                        "vision: vector_add (mlp residual) launch failed",
                        rvllm_core::CudaErrorKind::LaunchFailed,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
            }
            self.stream.fence()?;

            // Block 0 sub-step dump (post-attn-residual, post-mlp-residual).
            if blk_idx == 0 {
                if let Some(dir) = blk_dump_dir.as_deref() {
                    let bytes = n_tokens * hidden * 2;
                    let mut host = vec![0u8; bytes];
                    #[cfg(feature = "cuda")]
                    unsafe {
                        use cudarc::driver::sys::*;
                        let _ = cuMemcpyDtoH_v2(host.as_mut_ptr() as *mut _, hidden_region.device_ptr(), bytes);
                    }
                    let _ = std::fs::write(format!("{dir}/blk0_post_mlp.bin"), &host);
                }
            }

            // Per-block dump (only blocks 0, 13, 26 to keep io light).
            if let Some(dir) = blk_dump_dir.as_deref() {
                if blk_idx == 0 || blk_idx == 13 || blk_idx == 26 {
                    let bytes = n_tokens * hidden * 2;
                    let mut host = vec![0u8; bytes];
                    #[cfg(feature = "cuda")]
                    unsafe {
                        use cudarc::driver::sys::*;
                        let _ = cuMemcpyDtoH_v2(host.as_mut_ptr() as *mut _, hidden_region.device_ptr(), bytes);
                    }
                    let path = format!("{dir}/blk{blk_idx}.bin");
                    let _ = std::fs::write(&path, &host);
                    eprintln!("[qwen36] vit blk{blk_idx} dump → {path}");
                }
            }
        }

        // Env-gated pre-merger dump for HF reference comparison.
        if let Ok(path) = std::env::var("RVLLM_QWEN36_VISION_PREMERGER_DUMP") {
            let bytes = n_tokens * hidden * 2;
            let mut host = vec![0u8; bytes];
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let _ = cuMemcpyDtoH_v2(host.as_mut_ptr() as *mut _, hidden_region.device_ptr(), bytes);
            }
            let _ = std::fs::write(&path, &host);
            eprintln!(
                "[qwen36] pre-merger dump: {} bytes ([{}, {}] f16) → {}",
                bytes, n_tokens, hidden, path,
            );
        }

        // ── Step 6: PatchMerger. ────────────────────────────────────
        // (a) LayerNorm hidden in-place per-token (gamma/beta are 1152).
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let mut x = hidden_region.device_ptr();
            let mut g = vision.merger.norm_w.offset_bytes;
            let mut b = vision.merger.norm_b.offset_bytes;
            let mut eps = qkv_eps;
            let mut hd_i = hidden as i32;
            let args = [
                (&mut x) as *mut u64 as *mut core::ffi::c_void,
                (&mut g) as *mut u64 as *mut core::ffi::c_void,
                (&mut b) as *mut u64 as *mut core::ffi::c_void,
                (&mut eps) as *mut f32 as *mut core::ffi::c_void,
                (&mut hd_i) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = (hidden as u32).min(1024);
            let _ = cuLaunchKernel(
                self.outside_kernels.fn_layernorm_inplace_f16.raw() as CUfunction,
                n_tokens as u32, 1, 1,
                block, 1, 1,
                0, self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
        }
        self.stream.fence()?;

        // (b) Spatial merge: every 4 spatial-neighbour tokens concat
        // into one row of width merger_in=4608. Token order from
        // pos_h/pos_w: pre-merge tokens are already in
        // (block, intra_h, intra_w) order, so 4 consecutive rows = one
        // 2×2 spatial cluster ⇒ direct concat works.
        let merged_bytes = n_merged * merger_in * 2;
        let merged_region = self.arena.region("qvis_merged", merged_bytes, 16)?;
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let row_bytes = (hidden * 2) as u64;
            for m in 0..n_merged {
                for s in 0..merge_sq {
                    let src = hidden_region.device_ptr() + ((m * merge_sq + s) as u64) * row_bytes;
                    let dst = merged_region.device_ptr()
                        + (m as u64) * (merger_in as u64) * 2
                        + (s as u64) * row_bytes;
                    let _ = cuMemcpyDtoDAsync_v2(dst, src, hidden * 2, self.stream.raw() as _);
                }
            }
        }
        self.stream.fence()?;

        // (c) merger.linear_fc1 → GELU → linear_fc2.
        let merged_out_fc1 = self.arena.region("qvis_mfc1", n_merged * merger_in * 2, 16)?;
        linear_with_bias(
            merged_region.device_ptr(),
            vision.merger.fc1_w.offset_bytes,
            vision.merger.fc1_b.offset_bytes,
            merged_out_fc1.device_ptr(),
            f32_scratch.device_ptr(),
            n_merged, merger_in, merger_in,
        )?;
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let n_elem = (n_merged * merger_in) as i32;
            let mut x = merged_out_fc1.device_ptr();
            let mut nn = n_elem;
            let args = [
                (&mut x) as *mut u64 as *mut core::ffi::c_void,
                (&mut nn) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 256;
            let grid = ((n_elem as u32 + block - 1) / block, 1u32, 1u32);
            let _ = cuLaunchKernel(
                self.outside_kernels.fn_gelu_tanh_f16.raw() as CUfunction,
                grid.0, grid.1, grid.2,
                block, 1, 1,
                0, self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
        }
        self.stream.fence()?;
        let final_region = self.arena.region("qvis_final", n_merged * out_hidden * 2, 16)?;
        linear_with_bias(
            merged_out_fc1.device_ptr(),
            vision.merger.fc2_w.offset_bytes,
            vision.merger.fc2_b.offset_bytes,
            final_region.device_ptr(),
            f32_scratch.device_ptr(),
            n_merged, out_hidden, merger_in,
        )?;

        // ── Step 7: DtoH the final embeddings. ──────────────────────
        let mut out_bytes = vec![0u8; n_merged * out_hidden * 2];
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let _ = cuMemcpyDtoH_v2(
                out_bytes.as_mut_ptr() as *mut _,
                final_region.device_ptr(),
                out_bytes.len(),
            );
        }

        Ok(VisionForwardOutput {
            data: out_bytes,
            num_tokens: n_merged,
            hidden_dim: out_hidden,
            grid_thw: [grid_t, grid_h, grid_w],
        })
    }

    /// Phase 4x/5b/5c: full 40-layer decode loop. Linear-attn runs
    /// Gated-DeltaNet against the persistent state cache; full-attn
    /// runs Q/K/V → q/k_norm → RoPE → FA2 paged decode →
    /// attn_output_gate → o_proj; MoE applies after each attn block.
    ///
    /// `start_position` is the absolute position of token_ids[0] in
    /// the sequence. For prefill: start_position=0, token_ids = full
    /// prompt. For decode-step: start_position=prefill_len, token_ids
    /// = single just-sampled token. State + KV cache are NEVER reset
    /// inside this method — caller (cuda_worker) resets per request.
    pub fn forward_qwen36_decode(
        &self,
        token_ids: &[i32],
        start_position: u32,
        vision_splice: &[(usize, &[u8])],
    ) -> Result<i32> {
        if token_ids.is_empty() {
            return Err(rvllm_core::RvllmError::cuda(
                "forward_qwen36_decode: empty token_ids",
                rvllm_core::CudaErrorKind::Other,
                rvllm_core::CudaCtx::setup(),
            ));
        }
        let hidden = self.arch.base.hidden_size as u32;
        let vocab = self.arch.base.vocab_size as u32;
        let num_tokens = token_ids.len() as u32;
        let last_idx = (num_tokens - 1) as usize;
        let stream_raw = self.stream.raw() as u64;
        let last_hidden_bytes = (hidden as usize) * 2;

        // 1. Token IDs + embed_gather → hidden_region [num_tokens, hidden] f16.
        let mut tok_bytes = Vec::with_capacity(token_ids.len() * 4);
        for t in token_ids {
            tok_bytes.extend_from_slice(&t.to_le_bytes());
        }
        let tok_region =
            self.arena.region("qwen36_pl_tok", tok_bytes.len(), 16)?;
        unsafe { tok_region.copy_from_host(&tok_bytes)? };
        let hidden_bytes = (num_tokens as usize) * (hidden as usize) * 2;
        let hidden_region =
            self.arena.region("qwen36_pl_hidden", hidden_bytes, 16)?;
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
                tok_region.device_ptr(),
                stream_raw,
            )?;
        }
        // No fence: embed_gather wrote hidden_region on stream_raw and
        // every subsequent reader (vision splice HtoDAsync, decode loop
        // kernels) is also on stream_raw — same-stream ordering covers
        // it (Phase 4b-prep iter28).

        // Phase 4d vision splice: overwrite the placeholder-token
        // slots in hidden_region with the per-image vision-tower
        // embeddings. Each entry: (token_start_in_prompt, raw f16
        // bytes for [num_tokens, hidden_dim] = num_tokens * hidden * 2
        // bytes). The token_start is in PROMPT coordinates; we splice
        // only when start_position == 0 (i.e. prefill, when the full
        // prompt is in this hidden_region).
        if !vision_splice.is_empty() && start_position == 0 {
            let row_bytes = (hidden as usize) * 2;
            for (token_start, emb_bytes) in vision_splice {
                let dst_off = (*token_start as u64) * (row_bytes as u64);
                let len = emb_bytes.len();
                if *token_start + len / row_bytes > num_tokens as usize {
                    return Err(rvllm_core::RvllmError::cuda(
                        "vision splice would overrun hidden_region",
                        rvllm_core::CudaErrorKind::Other,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
                #[cfg(feature = "cuda")]
                unsafe {
                    use cudarc::driver::sys::*;
                    let r = cuMemcpyHtoDAsync_v2(
                        hidden_region.device_ptr() + dst_off,
                        emb_bytes.as_ptr() as *const _,
                        len,
                        self.stream.raw() as _,
                    );
                    if r != CUresult::CUDA_SUCCESS {
                        return Err(rvllm_core::RvllmError::cuda(
                            "qwen36 vision splice HtoDAsync",
                            rvllm_core::CudaErrorKind::MemcpyFailed,
                            rvllm_core::CudaCtx::setup(),
                        ));
                    }
                }
            }
            // No fence: vision splice uses cuMemcpyHtoDAsync_v2 on
            // stream_raw and the decode-loop kernels also run on
            // stream_raw, so same-stream ordering is automatic
            // (Phase 4b-prep iter29).
        }

        // 2. Resolve the FP8 GEMV kernel. Hard-fail when missing —
        // the previous silent fallback to `forward_qwen36_outside_
        // closer` skipped ALL 40 transformer layers (only embed →
        // final-norm → lm_head) and produced syntactically plausible
        // but semantically garbage output, indistinguishable from a
        // successful response on the wire. Hard erroring keeps the
        // server honest: an operator with a broken kernel build
        // sees the failure at the first request, not days later
        // when "the model got dumber."
        let kernel_gemv = self.outside_kernels.fn_fp8_gemv_wpr_native_f16in.ok_or_else(|| {
            rvllm_core::RvllmError::cuda(
                "qwen36 forward: fn_fp8_gemv_wpr_native_f16in not loaded — \
                 transformer layers cannot run; refusing to fall back to \
                 embed/final-norm-only path which would produce garbage tokens",
                rvllm_core::CudaErrorKind::Other,
                rvllm_core::CudaCtx::setup(),
            )
        })?;
        // Phase 1 of the batched-prefill plan: the per-token slot
        // is now read/written through `tok_ptr = hidden_region +
        // tok_local × hidden_bytes` directly. The previous
        // `last_hidden_region` scratch buffer + DtoD shuttle in/out
        // are gone; layer functions take `last_hidden_ptr: u64`.

        // Phase 5b: outer per-token loop. For each prompt token in
        // order: pass its slot pointer through all 40 layers
        // (linear-attn updates
        // its persistent state, full-attn writes K/V at slot=position
        // and attends causally over [0..position]), apply MoE, write
        // updated hidden back to hidden_region's slot for the closer.
        // The recurrent linear-attn state + monotonically-growing KV
        // cache mean each token sees full prompt context by the time
        // we hit the last position.
        //
        // KNOWN-PERF: prefill is O(prompt_tokens × layers). The two
        // per-token DtoD-fences (extract + writeback) were dropped
        // — same-stream ordering already guarantees what they were
        // synchronising — but the structural batched-prefill /
        // CUDA-graph path (matching Gemma's `unified_prefill`) is
        // still missing. That requires per-layer kernels that
        // accept a [N, D] hidden region rather than a single
        // [1, D] slot, plus chunked-recurrent linear-attention
        // and batched-causal full-attention variants. Substantial
        // refactor; tracked as Codex round 16 #2 follow-up.
        for tok_local in 0..num_tokens {
            let tok_pos = start_position + tok_local;
            // Phase 1 of the Qwen batched-prefill plan: the layer
            // functions now accept a raw device pointer to the
            // per-token slot in `hidden_region`. The previous
            // DtoD-extract (`hidden_region[off]` → `last_hidden_region`)
            // and DtoD-writeback are gone — `apply_layer_*` reads and
            // writes through `tok_ptr` directly, eliminating two
            // launches per token.
            let tok_ptr = hidden_region.device_ptr() + (tok_local as u64) * (last_hidden_bytes as u64);

            // Phase 5e: optional per-layer activation dump for
            // numerical-correctness audit vs vLLM. Set
            // RVLLM_QWEN36_DUMP_DIR to enable. Dumps last_hidden_region
            // contents (f16, hidden=2048 elems → 4096 bytes) after each
            // layer's attn block and after each layer's MoE block, but
            // ONLY for the last prompt token (tok_local == num_tokens-1
            // && start_position == 0) to avoid runaway disk usage on
            // multi-step decode.
            let dump_dir =
                std::env::var("RVLLM_QWEN36_DUMP_DIR").ok();
            let dump_this_token = dump_dir.is_some()
                && tok_local + 1 == num_tokens
                && start_position == 0;
            if let Some(dir) = dump_dir.as_ref() {
                if dump_this_token {
                    let _ = std::fs::create_dir_all(dir);
                    // Dump the embedding (input to layer 0) once.
                    let mut buf = vec![0u8; last_hidden_bytes];
                    #[cfg(feature = "cuda")]
                    unsafe {
                        use cudarc::driver::sys::*;
                        let _ = cuMemcpyDtoH_v2(
                            buf.as_mut_ptr() as *mut _,
                            tok_ptr,
                            last_hidden_bytes,
                        );
                    }
                    let _ = std::fs::write(format!("{dir}/embed.f16"), &buf);
                }
            }

            let mut linear_seq: u32 = 0;
            let mut full_seq: u32 = 0;
            for layer_idx in 0..self.model.layers.len() {
                let post_attn_norm_ptr = match &self.model.layers[layer_idx].attn {
                    rvllm_loader::qwen36_weights::Qwen36LayerAttn::Linear(la) => {
                        self.apply_layer_linear_attn(
                            la, linear_seq, tok_ptr,
                            kernel_gemv, hidden, last_hidden_bytes,
                        )?;
                        linear_seq += 1;
                        la.post_attention_layernorm.offset_bytes
                    }
                    rvllm_loader::qwen36_weights::Qwen36LayerAttn::Full(fl) => {
                        self.apply_layer_full_attn(
                            fl, full_seq, tok_pos,
                            tok_ptr,
                            kernel_gemv, hidden, last_hidden_bytes,
                        )?;
                        full_seq += 1;
                        fl.post_attention_layernorm.offset_bytes
                    }
                };
                if dump_this_token {
                    let dir = dump_dir.as_ref().unwrap();
                    let mut buf = vec![0u8; last_hidden_bytes];
                    #[cfg(feature = "cuda")]
                    unsafe {
                        use cudarc::driver::sys::*;
                        let _ = cuMemcpyDtoH_v2(
                            buf.as_mut_ptr() as *mut _,
                            tok_ptr,
                            last_hidden_bytes,
                        );
                    }
                    let _ = std::fs::write(
                        format!("{dir}/layer_{layer_idx:02}_attn.f16"), &buf);
                }
                self.apply_layer_moe(
                    &self.model.layers[layer_idx].moe,
                    post_attn_norm_ptr,
                    tok_ptr,
                    kernel_gemv, hidden, last_hidden_bytes,
                    layer_idx,
                )?;
                if dump_this_token {
                    let dir = dump_dir.as_ref().unwrap();
                    let mut buf = vec![0u8; last_hidden_bytes];
                    #[cfg(feature = "cuda")]
                    unsafe {
                        use cudarc::driver::sys::*;
                        let _ = cuMemcpyDtoH_v2(
                            buf.as_mut_ptr() as *mut _,
                            tok_ptr,
                            last_hidden_bytes,
                        );
                    }
                    let _ = std::fs::write(
                        format!("{dir}/layer_{layer_idx:02}_moe.f16"), &buf);
                }
            }
            // No DtoD writeback needed: the layer functions wrote
            // their final residual directly into `tok_ptr`, which
            // already points at the destination slot in hidden_region.
        }
        // Single fence at end of prefill so the closer's read of the
        // last hidden slot sees finalised values (host probes /
        // cublasLt handle synchronisation themselves, but this is
        // the canonical sync point between the per-token chain and
        // the lm_head / argmax stage).
        self.stream.fence()?;

        // 5. Final norm + lm_head + argmax (outside-only closer).
        self.forward_qwen36_outside_closer(
            &hidden_region,
            num_tokens,
            hidden,
            vocab,
            last_idx,
        )
    }

    /// Phase 5d helper: apply one linear-attn layer's Gated-DeltaNet
    /// chain on `last_hidden_region`. Reference: vLLM
    /// `gdn_linear_attn.py::fused_post_conv_prep` and surrounding
    /// per-step decode path.
    ///
    /// Shapes (Qwen 3.6 35B-A3B):
    ///   - num_k_heads = 16, head_k_dim = 128 → key_dim = 2048
    ///   - num_v_heads = 32, head_v_dim = 128 → value_dim = 4096
    ///   - in_proj_qkv [8192, hidden] → conv_dim = 2*key_dim + value_dim = 8192
    ///     contiguous as [Q (key_dim) | K (key_dim) | V (value_dim)]
    ///   - in_proj_z [value_dim, hidden]
    ///   - in_proj_a [num_v_heads, hidden] (bf16)
    ///   - in_proj_b [num_v_heads, hidden] (bf16)
    ///   - A_log [num_v_heads], dt_bias [num_v_heads]
    ///   - norm.weight [head_v_dim] (per-v-head RMSNorm gamma)
    ///
    /// Pipeline:
    ///   1. input_layernorm on copy of last_hidden
    ///   2. in_proj_qkv FP8 GEMV → conv-input [conv_dim]
    ///   3. causal_conv1d → conv_out [conv_dim]
    ///   4. Host: SiLU + split into Q[16,128], K[16,128], V[32,128]
    ///   5. Host: L2-norm Q and K per-head
    ///   6. Host: in_proj_a · normed → a [num_v_heads]
    ///      Host: in_proj_b · normed → b [num_v_heads]
    ///      α[v] = exp(-exp(A_log[v]) * softplus(a[v] + dt_bias[v]))
    ///      β[v] = sigmoid(b[v])
    ///   7. GQA expand: K_exp[v]=K[v/2], Q_exp[v]=Q[v/2] (each k-head
    ///      shared by 2 v-heads since num_v=2*num_k)
    ///   8. ssm state update kernel (32 v-heads, head_v=128, head_k=128)
    ///      against persistent state slice
    ///   9. Q_exp · S → readout[32, 128]
    ///  10. in_proj_z FP8 GEMV → z [value_dim]
    ///  11. Per-v-head RMSNorm with norm.weight + sigmoid(z) gate
    ///  12. out_proj FP8 GEMV → o_buf [hidden]
    ///  13. Host residual sum → write back to last_hidden_region
    #[allow(clippy::too_many_arguments)]
    /// `last_hidden_ptr`: device pointer to the (single-token, for now)
    /// hidden slot this layer reads from and writes back into. Phase 1
    /// (Qwen batched-prefill plan) replaced an `&Region` parameter
    /// here so the prefill loop can pass an offset into the larger
    /// `hidden_region` directly — no per-token DtoD shuttle. See
    /// `v3/QWEN_BATCHED_PREFILL_PLAN.md`.
    fn apply_layer_linear_attn(
        &self,
        la: &rvllm_loader::qwen36_weights::Qwen36LinearAttnLayer,
        linear_seq_idx: u32,
        last_hidden_ptr: u64,
        kernel_gemv: rvllm_kernels::KernelFn,
        hidden: u32,
        last_hidden_bytes: usize,
    ) -> Result<()> {
        let stream_raw = self.stream.raw() as u64;
        let qkv_n = la.in_proj_qkv.shape[0] as u32; // 8192 = conv_dim
        let z_n = la.in_proj_z.shape[0] as u32; // 4096 = value_dim
        let out_n = la.out_proj.shape[0] as u32; // 2048 = hidden
        let out_k = la.out_proj.shape[1] as u32; // 4096 = value_dim
        let m: u32 = 1;
        let qkv_bs = match la.in_proj_qkv.blockscale_ptr { Some(p) => p, None => return Ok(()) };
        let z_bs = match la.in_proj_z.blockscale_ptr { Some(p) => p, None => return Ok(()) };
        let out_bs = match la.out_proj.blockscale_ptr { Some(p) => p, None => return Ok(()) };

        // Hardcoded for Qwen 3.6 35B-A3B; could be plumbed from arch.
        let num_k_heads: u32 = 16;
        let num_v_heads: u32 = 32;
        let head_k_dim: u32 = 128;
        let head_v_dim: u32 = 128;
        let key_dim = num_k_heads * head_k_dim; // 2048
        let value_dim = num_v_heads * head_v_dim; // 4096
        let v_per_k = num_v_heads / num_k_heads; // 2
        let kus = num_k_heads as usize;
        let vus = num_v_heads as usize;
        let hkd = head_k_dim as usize;
        let hvd = head_v_dim as usize;

        // 1. input_layernorm on copy of last_hidden.
        let normed_region =
            self.arena.region("qwen36_pl_normed", last_hidden_bytes, 16)?;
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let _ = cuMemcpyDtoDAsync_v2(
                normed_region.device_ptr(),
                last_hidden_ptr,
                last_hidden_bytes,
                self.stream.raw() as _,
            );
        }
        let eps = self.arch.base.rms_norm_eps;
        unsafe {
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: 1, hidden, eps,
            }.launch(
                self.outside_kernels.fn_rmsnorm_inplace_f16,
                normed_region.device_ptr(),
                la.input_layernorm.offset_bytes,
                stream_raw,
            )?;
        }
        // No fence: in_proj_qkv runs on the same stream and reads
        // normed_region after rmsnorm has written to it; stream
        // ordering guarantees that.

        // 2. in_proj_qkv FP8 GEMV → conv-input. (Phase 4a routing.)
        let qkv_bytes_dev = (qkv_n as usize) * 2;
        let qkv_region = self.arena.region("qwen36_pl_qkv", qkv_bytes_dev, 16)?;
        unsafe {
            self.fp8_proj_dispatch(kernel_gemv, qkv_region.device_ptr(),
                la.in_proj_qkv.offset_bytes, qkv_bs, normed_region.device_ptr(),
                m, qkv_n, hidden, stream_raw)?;
        }
        // No fence: conv_state_advance + conv1d run on the same
        // stream and read qkv after the GEMV has written to it.

        // 3. causal_conv1d. Single-step: prepend the ks-1=3 previous
        //    timesteps from the persistent conv-state cache, append
        //    current qkv = 4 timesteps. After conv1d, update the cache
        //    by shifting (drop oldest, append current).
        let ks: u32 = 4;
        let conv_in_bytes = ((ks as usize) * (qkv_n as usize)) * 2;
        let conv_in_region =
            self.arena.region("qwen36_pl_cin", conv_in_bytes, 16)?;
        let conv_out_region =
            self.arena.region("qwen36_pl_cout", qkv_bytes_dev, 16)?;
        // GPU-side conv_in assembly + state advance. One launch
        // replaces 2× DtoH + 2× HtoD + two CPU vec slicings.
        let conv_state_ptr_layer = self.conv_state_layer_ptr(linear_seq_idx);
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let mut conv_in = conv_in_region.device_ptr();
            let mut state = conv_state_ptr_layer;
            let mut cur = qkv_region.device_ptr();
            let mut ts_i: i32 = qkv_n as i32;
            let args = [
                (&mut conv_in) as *mut u64 as *mut core::ffi::c_void,
                (&mut state) as *mut u64 as *mut core::ffi::c_void,
                (&mut cur) as *mut u64 as *mut core::ffi::c_void,
                (&mut ts_i) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 256;
            let grid: u32 = ((qkv_n + block - 1) / block).max(1);
            let rc = cuLaunchKernel(
                self.outside_kernels.fn_conv_state_advance_f16.raw() as CUfunction,
                grid, 1, 1,
                block, 1, 1,
                0, self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 linear_attn conv_state_advance launch",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let mut output = conv_out_region.device_ptr();
            let mut input = conv_in_region.device_ptr();
            let mut weight = la.conv1d.offset_bytes;
            let mut sl: i32 = 1;
            let mut ch = qkv_n as i32;
            let mut k_arg = ks as i32;
            let args = [
                (&mut output) as *mut u64 as *mut core::ffi::c_void,
                (&mut input) as *mut u64 as *mut core::ffi::c_void,
                (&mut weight) as *mut u64 as *mut core::ffi::c_void,
                (&mut sl) as *mut i32 as *mut core::ffi::c_void,
                (&mut ch) as *mut i32 as *mut core::ffi::c_void,
                (&mut k_arg) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 256;
            let grid_x = (qkv_n + block - 1) / block;
            let _ = cuLaunchKernel(
                self.outside_kernels.fn_causal_conv1d_f16.raw() as CUfunction,
                grid_x, 1, 1,
                block, 1, 1,
                0,
                self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
        }
        // No fence: silu_l2_gqa runs on the same stream after conv1d.

        // 4+5. GPU-side fused silu + Q/K L2-norm + GQA-expand + V silu-pack.
        // Allocates the q_exp / k_exp / v_pack device regions and
        // writes them directly. Replaces the host pipeline (DtoH
        // conv_out + CPU silu + per-k-head L2 + GQA-expand into
        // host bytes + HtoD q/k/v).
        let qk_bytes_pre = vus * hkd * 2;
        let v_bytes_pre  = vus * hvd * 2;
        let q_region = self.arena.region("qwen36_pl_q", qk_bytes_pre, 16)?;
        let k_region = self.arena.region("qwen36_pl_k", qk_bytes_pre, 16)?;
        let v_region = self.arena.region("qwen36_pl_v", v_bytes_pre, 16)?;
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let mut q_out = q_region.device_ptr();
            let mut k_out = k_region.device_ptr();
            let mut v_out = v_region.device_ptr();
            let mut conv_p = conv_out_region.device_ptr();
            let mut vus_i: i32 = vus as i32;
            let mut hkd_i: i32 = hkd as i32;
            let mut hvd_i: i32 = hvd as i32;
            let mut kd_i:  i32 = key_dim as i32;
            let mut nvh:   i32 = num_v_heads as i32;
            let mut vpk:   i32 = v_per_k as i32;
            let args = [
                (&mut q_out) as *mut u64 as *mut core::ffi::c_void,
                (&mut k_out) as *mut u64 as *mut core::ffi::c_void,
                (&mut v_out) as *mut u64 as *mut core::ffi::c_void,
                (&mut conv_p) as *mut u64 as *mut core::ffi::c_void,
                (&mut vus_i) as *mut i32 as *mut core::ffi::c_void,
                (&mut hkd_i) as *mut i32 as *mut core::ffi::c_void,
                (&mut hvd_i) as *mut i32 as *mut core::ffi::c_void,
                (&mut kd_i) as *mut i32 as *mut core::ffi::c_void,
                (&mut nvh) as *mut i32 as *mut core::ffi::c_void,
                (&mut vpk) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = (hkd.max(hvd)) as u32;
            let rc = cuLaunchKernel(
                self.outside_kernels.fn_qwen_linear_silu_l2_gqa_f16.raw() as CUfunction,
                vus as u32, 1, 1,
                block, 1, 1,
                0, self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 linear_attn silu_l2_gqa launch",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }

        // 6. GPU alpha/beta: fused dot-product + softplus / sigmoid
        // launched against the live normed input. Pre-allocate
        // alpha_region and beta_region as f32 device buffers; the
        // kernel writes them directly. Replaces the host pipeline
        // (DtoH input → f16→f32 → nested CPU GEMV → HtoD alpha/beta)
        // with one launch. The bring-up-time host cache from iter3
        // becomes unused on the production path; kept as a fallback
        // reference for diagnostics.
        let h_us = hidden as usize;
        let alpha_region = self.arena.region("qwen36_pl_alpha", vus * 4, 16)?;
        let beta_region  = self.arena.region("qwen36_pl_beta",  vus * 4, 16)?;
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let mut a_out = alpha_region.device_ptr();
            let mut b_out = beta_region.device_ptr();
            let mut a_w_p = la.in_proj_a.offset_bytes;
            let mut b_w_p = la.in_proj_b.offset_bytes;
            let mut a_log_p = la.a_log.offset_bytes;
            let mut dt_bias_p = la.dt_bias.offset_bytes;
            let mut in_p = normed_region.device_ptr();
            let mut vus_i: i32 = vus as i32;
            let mut h_i: i32 = h_us as i32;
            let args = [
                (&mut a_out) as *mut u64 as *mut core::ffi::c_void,
                (&mut b_out) as *mut u64 as *mut core::ffi::c_void,
                (&mut a_w_p) as *mut u64 as *mut core::ffi::c_void,
                (&mut b_w_p) as *mut u64 as *mut core::ffi::c_void,
                (&mut a_log_p) as *mut u64 as *mut core::ffi::c_void,
                (&mut dt_bias_p) as *mut u64 as *mut core::ffi::c_void,
                (&mut in_p) as *mut u64 as *mut core::ffi::c_void,
                (&mut vus_i) as *mut i32 as *mut core::ffi::c_void,
                (&mut h_i) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 256u32.min(h_us as u32).max(1);
            let rc = cuLaunchKernel(
                self.outside_kernels.fn_qwen_linear_alpha_beta_f16.raw() as CUfunction,
                vus as u32, 1, 1,
                block, 1, 1,
                0, self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 linear_attn alpha_beta launch",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }

        // Phase 5i: GPU delta-rule kernel.
        // Build GQA-expanded Q/K (one row per v-head) on host, push to
        // device, alongside V (already per-v-head), alpha, beta. Kernel
        // does forget + delta correction + state update + readout in
        // one launch, writing readout to readout_region. State is
        // updated in-place in the persistent linear-state slice.
        let layer_state_ptr = self.linear_state_layer_ptr(linear_seq_idx);
        let scale = 1.0f32 / (head_k_dim as f32).sqrt();
        let v_bytes = vus * hvd * 2;
        // q_region, k_region, v_region were allocated + filled by
        // the silu_l2_gqa GPU kernel in step 4+5; alpha_region and
        // beta_region by the alpha_beta kernel in step 6. All four
        // are device-resident already — no further HtoD needed.
        let readout_region =
            self.arena.region("qwen36_pl_readout", v_bytes, 16)?;
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let mut state = layer_state_ptr;
            let mut q_ptr = q_region.device_ptr();
            let mut k_ptr = k_region.device_ptr();
            let mut v_ptr = v_region.device_ptr();
            let mut a_ptr = alpha_region.device_ptr();
            let mut b_ptr = beta_region.device_ptr();
            let mut o_ptr = readout_region.device_ptr();
            let mut scale_arg = scale;
            let mut hvd_i = head_v_dim as i32;
            let mut hkd_i = head_k_dim as i32;
            let args = [
                (&mut state) as *mut u64 as *mut core::ffi::c_void,
                (&mut q_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut k_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut v_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut a_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut b_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut o_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut scale_arg) as *mut f32 as *mut core::ffi::c_void,
                (&mut hvd_i) as *mut i32 as *mut core::ffi::c_void,
                (&mut hkd_i) as *mut i32 as *mut core::ffi::c_void,
            ];
            // Block = head_v_dim threads, one per output row.
            // Shared mem = (2*head_k_dim + head_v_dim) * 4 bytes.
            let smem = (2 * head_k_dim + head_v_dim) * 4;
            let _ = cuLaunchKernel(
                self.outside_kernels.fn_gated_delta_rule_decode_f16.raw() as CUfunction,
                num_v_heads, 1, 1,
                head_v_dim, 1, 1,
                smem,
                self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
        }
        // No fence: in_proj_z + rmsnorm_gated run on the same
        // stream after the delta-rule kernel.

        // 10. in_proj_z FP8 GEMV → z [value_dim] (stays on device).
        let z_bytes_dev = (z_n as usize) * 2;
        let z_region = self.arena.region("qwen36_pl_z", z_bytes_dev, 16)?;
        unsafe {
            self.fp8_proj_dispatch(kernel_gemv, z_region.device_ptr(),
                la.in_proj_z.offset_bytes, z_bs, normed_region.device_ptr(),
                m, z_n, hidden, stream_raw)?;
        }

        // 11. GPU per-v-head RMSNormGated + silu(z) gate fused into
        // one launch. Replaces 3× DtoH (readout, z, norm.gamma) +
        // host CPU loop over (vus × hvd) elements + 1× HtoD gated.
        let gated_region = self.arena.region("qwen36_pl_gated", vus * hvd * 2, 16)?;
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let mut g_out = gated_region.device_ptr();
            let mut r_in = readout_region.device_ptr();
            let mut z_in = z_region.device_ptr();
            let mut gamma_p = la.norm.offset_bytes;
            let mut vus_i: i32 = vus as i32;
            let mut hvd_i: i32 = hvd as i32;
            let mut eps_f: f32 = 1e-6;
            let args = [
                (&mut g_out) as *mut u64 as *mut core::ffi::c_void,
                (&mut r_in) as *mut u64 as *mut core::ffi::c_void,
                (&mut z_in) as *mut u64 as *mut core::ffi::c_void,
                (&mut gamma_p) as *mut u64 as *mut core::ffi::c_void,
                (&mut vus_i) as *mut i32 as *mut core::ffi::c_void,
                (&mut hvd_i) as *mut i32 as *mut core::ffi::c_void,
                (&mut eps_f) as *mut f32 as *mut core::ffi::c_void,
            ];
            let block: u32 = hvd as u32;
            let rc = cuLaunchKernel(
                self.outside_kernels.fn_qwen_linear_rmsnorm_gated_f16.raw() as CUfunction,
                vus as u32, 1, 1,
                block, 1, 1,
                0, self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 linear_attn rmsnorm_gated launch",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }

        // 12. out_proj FP8 GEMV → o_buf [hidden].
        let out_region = self.arena.region("qwen36_pl_out", (out_n as usize) * 2, 16)?;
        unsafe {
            self.fp8_proj_dispatch(kernel_gemv, out_region.device_ptr(),
                la.out_proj.offset_bytes, out_bs, gated_region.device_ptr(),
                m, out_n, out_k, stream_raw)?;
        }
        // No fence: residual vector_add runs on the same stream
        // after out_proj.

        // 13. Residual sum: last_hidden_new = last_hidden + o_buf.
        // GPU residual: last_hidden += out_buf via vector_add_f16.
        // Replaces 2× DtoH + CPU loop + 1× HtoD with one launch.
        // Same numerics (`__hadd` is f16 RTNE, matching the previous
        // f16→f32→add→f16-RTNE pipeline byte-for-byte).
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let n_elem = (hidden as usize) * (m as usize);
            let mut dst = last_hidden_ptr;
            let mut src = out_region.device_ptr();
            let mut nn: i32 = n_elem as i32;
            let args = [
                (&mut dst) as *mut u64 as *mut core::ffi::c_void,
                (&mut src) as *mut u64 as *mut core::ffi::c_void,
                (&mut nn) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 1024.min(n_elem as u32).max(1);
            let grid = ((n_elem as u32 + block - 1) / block).max(1);
            let rc = cuLaunchKernel(
                self.outside_kernels.fn_vector_add_f16.raw() as CUfunction,
                grid, 1, 1, block, 1, 1, 0,
                self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 linear_attn residual vector_add_f16",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        // No function-exit fence: the next layer call (or the
        // outer-loop's end-of-prefill fence before lm_head) runs
        // on the same stream and same-stream ordering already
        // guarantees the residual write is visible.
        Ok(())
    }

    /// Phase 4z/5b helper: full-attention layer forward on a single
    /// token at `position` (0-indexed). Composes input_layernorm →
    /// q/k/v_proj → host-deinterleave Q+gate → q/k_norm → RoPE +
    /// KV-cache write at slot=position → paged FA2 decode with
    /// context_len=position+1 (causal) → sigmoid_mul gate → o_proj →
    /// residual sum into last_hidden.
    #[allow(clippy::too_many_arguments)]
    /// `last_hidden_ptr`: see `apply_layer_linear_attn` doc — same
    /// Phase-1 contract.
    fn apply_layer_full_attn(
        &self,
        fl: &rvllm_loader::qwen36_weights::Qwen36FullAttnLayer,
        full_seq_idx: u32,
        position: u32,
        last_hidden_ptr: u64,
        kernel_gemv: rvllm_kernels::KernelFn,
        hidden: u32,
        last_hidden_bytes: usize,
    ) -> Result<()> {
        let stream_raw = self.stream.raw() as u64;
        let head_dim = self.arch.base.head_dim as u32;
        let num_heads = self.arch.base.num_attention_heads as u32; // 16
        let num_kv_heads = self.arch.base.num_key_value_heads as u32; // 2
        let q_n = fl.q_proj.shape[0] as u32; // 8192 = num_heads*head_dim*2
        let k_n = fl.k_proj.shape[0] as u32; // 512
        let v_n = fl.v_proj.shape[0] as u32; // 512
        let o_n = fl.o_proj.shape[0] as u32; // 2048
        let o_k = fl.o_proj.shape[1] as u32; // 4096
        let m: u32 = 1;
        let q_bs = match fl.q_proj.blockscale_ptr { Some(p) => p, None => return Ok(()) };
        let k_bs = match fl.k_proj.blockscale_ptr { Some(p) => p, None => return Ok(()) };
        let v_bs = match fl.v_proj.blockscale_ptr { Some(p) => p, None => return Ok(()) };
        let o_bs = match fl.o_proj.blockscale_ptr { Some(p) => p, None => return Ok(()) };

        // 1. input_layernorm on copy of last_hidden.
        let normed_region =
            self.arena.region("qwen36_pf_normed", last_hidden_bytes, 16)?;
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let _ = cuMemcpyDtoDAsync_v2(
                normed_region.device_ptr(),
                last_hidden_ptr,
                last_hidden_bytes,
                self.stream.raw() as _,
            );
        }
        let eps = self.arch.base.rms_norm_eps;
        unsafe {
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: 1, hidden, eps,
            }.launch(
                self.outside_kernels.fn_rmsnorm_inplace_f16,
                normed_region.device_ptr(),
                fl.input_layernorm.offset_bytes,
                stream_raw,
            )?;
        }
        // No fence: q/k/v projections run on the same stream.

        // 2. q_proj, k_proj, v_proj GEMVs.
        let q_region =
            self.arena.region("qwen36_pf_qg", (q_n as usize) * 2, 16)?;
        let k_region =
            self.arena.region("qwen36_pf_k", (k_n as usize) * 2, 16)?;
        let v_region =
            self.arena.region("qwen36_pf_v", (v_n as usize) * 2, 16)?;
        // Phase 4a: route projections through `fp8_proj_dispatch`.
        // At m=1 (today's caller) this dispatches to the same
        // Fp8GemvF16InLaunch byte-identically. Phase 4b/5/7 will
        // flip m=1 → m=num_tokens and the dispatcher will pick up
        // CUTLASS SM120 (m≥128) automatically — no further edits
        // needed in this function.
        unsafe {
            self.fp8_proj_dispatch(kernel_gemv, q_region.device_ptr(),
                fl.q_proj.offset_bytes, q_bs, normed_region.device_ptr(),
                m, q_n, hidden, stream_raw)?;
            self.fp8_proj_dispatch(kernel_gemv, k_region.device_ptr(),
                fl.k_proj.offset_bytes, k_bs, normed_region.device_ptr(),
                m, k_n, hidden, stream_raw)?;
            self.fp8_proj_dispatch(kernel_gemv, v_region.device_ptr(),
                fl.v_proj.offset_bytes, v_bs, normed_region.device_ptr(),
                m, v_n, hidden, stream_raw)?;
        }
        // No fence: split_q_gate runs on the same stream.

        // 3. GPU split of q_proj output [num_tokens, num_heads, 2*head_dim]
        //    into Q + gate [num_tokens, num_heads, head_dim] each.
        //    Replaces a DtoH + CPU per-head copy_from_slice + HtoD
        //    round-trip per token with one launch.
        let q_size = (num_heads * head_dim) as usize; // 4096
        let hd = head_dim as usize;
        let q_split_region =
            self.arena.region("qwen36_pf_qs", q_size * (m as usize) * 2, 16)?;
        let gate_region =
            self.arena.region("qwen36_pf_gt", q_size * (m as usize) * 2, 16)?;
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let mut qo = q_split_region.device_ptr();
            let mut go = gate_region.device_ptr();
            let mut qi = q_region.device_ptr();
            let mut nh: i32 = num_heads as i32;
            let mut hd_i: i32 = head_dim as i32;
            let args = [
                (&mut qo) as *mut u64 as *mut core::ffi::c_void,
                (&mut go) as *mut u64 as *mut core::ffi::c_void,
                (&mut qi) as *mut u64 as *mut core::ffi::c_void,
                (&mut nh) as *mut i32 as *mut core::ffi::c_void,
                (&mut hd_i) as *mut i32 as *mut core::ffi::c_void,
            ];
            let rc = cuLaunchKernel(
                self.outside_kernels.fn_split_q_gate_f16.raw() as CUfunction,
                num_heads, m, 1,
                head_dim, 1, 1,
                0, self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 full_attn split_q_gate launch",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }

        // 4. q_norm + k_norm (per-head RMSNorm via rmsnorm_inplace
        //    treating heads as "tokens").
        unsafe {
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: num_heads, hidden: head_dim, eps,
            }.launch(
                self.outside_kernels.fn_rmsnorm_inplace_f16,
                q_split_region.device_ptr(),
                fl.q_norm.offset_bytes,
                stream_raw,
            )?;
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: num_kv_heads, hidden: head_dim, eps,
            }.launch(
                self.outside_kernels.fn_rmsnorm_inplace_f16,
                k_region.device_ptr(),
                fl.k_norm.offset_bytes,
                stream_raw,
            )?;
        }
        // No fence: fused_rope runs on the same stream.

        // 5. NeoX-style partial RoPE on GPU + KV-cache write.
        //
        // Replaces the previous host pipeline (DtoH cos/sin, DtoH q/k/v,
        // CPU NeoX rotation, HtoD rotated Q, HtoD K+V to cache slots —
        // 5+ round-trips per token + a 16-head × 32-element CPU loop)
        // with one kernel launch. `fused_rope_qwen_partial_f16kv`
        // pairs `(i, i + rotary_dim/2)` within the first `rotary_dim`
        // elements of each head — Qwen's partial-NeoX convention,
        // distinct from the Gemma kernel's `(i, i + head_dim/2)`
        // pairing.
        let rotary_dim = (head_dim as f32 * 0.25) as u32; // 64
        let kv_layer_ptr = self.kv_cache_layer_ptr(full_seq_idx);
        let half = (self.kv_cache_layer_bytes / 2) as u64;
        let k_cache_layer_ptr = kv_layer_ptr;
        let v_cache_layer_ptr = kv_layer_ptr + half;
        // Phase 4b-prep iter26: pack `position` (used by RoPE as both
        // pos and slot) AND `context_len = position + 1` (used by paged
        // FA2) into ONE 8-byte HtoD per full-attn layer. Replaces two
        // separate 4-byte HtoDs (pos_region + cl_region).
        let context_len_for_pack = (position as i32) + 1;
        let mut pos_cl_bytes = [0u8; 8];
        pos_cl_bytes[..4].copy_from_slice(&(position as i32).to_le_bytes());
        pos_cl_bytes[4..].copy_from_slice(&context_len_for_pack.to_le_bytes());
        let pos_region = self.arena.region("qwen36_pf_pos_cl", 8, 16)?;
        let slot_dev_ptr = pos_region.device_ptr();
        let cl_dev_ptr = pos_region.device_ptr() + 4;
        unsafe {
            pos_region.copy_from_host(&pos_cl_bytes)?;
        }
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let mut q_in_p = q_split_region.device_ptr();
            let mut k_in_p = k_region.device_ptr();
            let mut v_in_p = v_region.device_ptr();
            let mut q_out_p = q_split_region.device_ptr(); // in-place ok
            let mut kc = k_cache_layer_ptr;
            let mut vc = v_cache_layer_ptr;
            let mut cos_p = self.rope_cos;
            let mut sin_p = self.rope_sin;
            let mut pos_p = pos_region.device_ptr();
            let mut slot_p = slot_dev_ptr;
            let mut nt: i32 = m as i32;
            let mut nh: i32 = num_heads as i32;
            let mut nkh: i32 = num_kv_heads as i32;
            let mut hd_i: i32 = head_dim as i32;
            let mut rd: i32 = rotary_dim as i32;
            let args = [
                (&mut q_in_p) as *mut u64 as *mut core::ffi::c_void,
                (&mut k_in_p) as *mut u64 as *mut core::ffi::c_void,
                (&mut v_in_p) as *mut u64 as *mut core::ffi::c_void,
                (&mut q_out_p) as *mut u64 as *mut core::ffi::c_void,
                (&mut kc) as *mut u64 as *mut core::ffi::c_void,
                (&mut vc) as *mut u64 as *mut core::ffi::c_void,
                (&mut cos_p) as *mut u64 as *mut core::ffi::c_void,
                (&mut sin_p) as *mut u64 as *mut core::ffi::c_void,
                (&mut pos_p) as *mut u64 as *mut core::ffi::c_void,
                (&mut slot_p) as *mut u64 as *mut core::ffi::c_void,
                (&mut nt) as *mut i32 as *mut core::ffi::c_void,
                (&mut nh) as *mut i32 as *mut core::ffi::c_void,
                (&mut nkh) as *mut i32 as *mut core::ffi::c_void,
                (&mut hd_i) as *mut i32 as *mut core::ffi::c_void,
                (&mut rd) as *mut i32 as *mut core::ffi::c_void,
            ];
            let max_h = num_heads.max(num_kv_heads);
            let block_x: u32 = (head_dim / 2) as u32;
            let rc = cuLaunchKernel(
                self.outside_kernels.fn_fused_rope_qwen_partial_f16kv.raw() as CUfunction,
                m as u32, max_h, 1,
                block_x, 1, 1,
                0, self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 full_attn fused_rope_qwen launch",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        // No fence: paged FA2 decode runs on the same stream after
        // the fused_rope kernel writes Q (in-place into q_split_region)
        // and K/V into the cache slots.
        // (Phase-4b prep) Q-rotation, K-rotation, KV-cache write all
        // happened in the GPU launch above; the previous host-side
        // pipeline (DtoH cos/sin + DtoH q/k/v + CPU NeoX rotation +
        // HtoD rotated Q + HtoD K, V to cache slots) is gone.

        // 6. Paged FA2 decode. block_tables=identity-mapping
        //    [0, 1, 2, ..., max_blocks_per_seq-1] since each logical
        //    block in our single-sequence cache maps to itself in
        //    physical layout. context_lens=[position+1] (causal —
        //    attend to all prior tokens).
        // Phase 4b-prep iter25/26: identity block table is uploaded
        // once at bring-up; `context_len` was packed with `position`
        // into pos_region above (one combined HtoD per layer).
        let attn_out_region =
            self.arena.region("qwen36_pf_attn_out", q_size * 2, 16)?;
        let scale = 1.0 / (head_dim as f32).sqrt();
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            const FA2_THREADS: i32 = 128;
            const FA2_BC: i32 = 32;
            let hd_i = head_dim as i32;
            let smem_bytes = 2 * FA2_BC * hd_i * 4 + FA2_BC * 4 + (FA2_THREADS / 32) * 4;
            if smem_bytes as u32 >= 48 * 1024 {
                let _ = cuFuncSetAttribute(
                    self.outside_kernels.fn_flash_attention_2_decode_f16io.raw() as CUfunction,
                    CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    smem_bytes,
                );
            }
            let mut output = attn_out_region.device_ptr();
            let mut query = q_split_region.device_ptr();
            let mut key_cache = k_cache_layer_ptr;
            let mut value_cache = v_cache_layer_ptr;
            let mut block_tables = self.bt_persistent_ptr;
            let mut context_lens = cl_dev_ptr;
            let mut scale_arg = scale;
            let mut nh = num_heads as i32;
            let mut nkvh = num_kv_heads as i32;
            let mut hd = head_dim as i32;
            let mut bs = self.kv_cache_block_size as i32;
            let mut mbps = self.kv_cache_num_blocks as i32;
            let mut window: i32 = -1;
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
            let _ = cuLaunchKernel(
                self.outside_kernels.fn_flash_attention_2_decode_f16io.raw() as CUfunction,
                1, num_heads, 1,
                FA2_THREADS as u32, 1, 1,
                smem_bytes as u32,
                self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
        }
        // No fence: attn_output_gate kernel runs on the same stream.

        // 7. attn_output_gate: attn_out * sigmoid(gate).
        let gated_region =
            self.arena.region("qwen36_pf_gated", q_size * 2, 16)?;
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let mut output = gated_region.device_ptr();
            let mut values = attn_out_region.device_ptr();
            let mut gate = gate_region.device_ptr();
            let mut nn = q_size as i32;
            let args = [
                (&mut output) as *mut u64 as *mut core::ffi::c_void,
                (&mut values) as *mut u64 as *mut core::ffi::c_void,
                (&mut gate) as *mut u64 as *mut core::ffi::c_void,
                (&mut nn) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 256;
            let grid = (q_size as u32 + block - 1) / block;
            let _ = cuLaunchKernel(
                self.outside_kernels.fn_sigmoid_mul_f16.raw() as CUfunction,
                grid, 1, 1,
                block, 1, 1,
                0,
                self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
        }
        // No fence: o_proj runs on the same stream after sigmoid_mul.

        // 8. o_proj FP8 GEMV → out_buf [hidden]. (Phase 4a routing.)
        let out_region =
            self.arena.region("qwen36_pf_out", (o_n as usize) * 2, 16)?;
        unsafe {
            self.fp8_proj_dispatch(kernel_gemv, out_region.device_ptr(),
                fl.o_proj.offset_bytes, o_bs, gated_region.device_ptr(),
                m, o_n, o_k, stream_raw)?;
        }
        // No fence: residual vector_add runs on the same stream
        // after o_proj.

        // 9. Residual sum: last_hidden += out_buf, GPU-side.
        // Replaces a 2× DtoH + CPU loop over `hidden` halves + HtoD
        // round-trip with one `vector_add_f16` launch — same numeric
        // result (the CPU loop did f16 → f32 → add → f16 RTNE; the
        // kernel uses `__hadd`, which IS f16 RTNE, so output bytes
        // are identical).
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let n_elem = (hidden as usize) * (m as usize);
            let mut dst = last_hidden_ptr;
            let mut src = out_region.device_ptr();
            let mut nn: i32 = n_elem as i32;
            let args = [
                (&mut dst) as *mut u64 as *mut core::ffi::c_void,
                (&mut src) as *mut u64 as *mut core::ffi::c_void,
                (&mut nn) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 1024.min(n_elem as u32).max(1);
            let grid = ((n_elem as u32 + block - 1) / block).max(1);
            let rc = cuLaunchKernel(
                self.outside_kernels.fn_vector_add_f16.raw() as CUfunction,
                grid, 1, 1, block, 1, 1, 0,
                self.stream.raw() as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 full_attn residual vector_add_f16",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        // No function-exit fence: same-stream ordering covers the
        // next layer's first read of last_hidden_ptr.
        Ok(())
    }

    /// Phase 5a helper: per-layer MoE block forward.
    /// Composes post_attention_layernorm → router top-k softmax →
    /// 8 routed FFNs (gate/up FP8 → host silu·mul → down FP8) +
    /// shared FFN scaled by sigmoid(shared_expert_gate_logit) →
    /// host weighted sum → residual into last_hidden.
    /// post_attn_norm_ptr is the layer's post_attention_layernorm
    /// weight offset (lives on the attn struct, not the moe block).
    #[allow(clippy::too_many_arguments)]
    fn apply_layer_moe(
        &self,
        moe: &rvllm_loader::qwen36_weights::Qwen36MoeBlock,
        post_attn_norm_ptr: u64,
        last_hidden_ptr: u64,
        kernel_gemv: rvllm_kernels::KernelFn,
        hidden: u32,
        last_hidden_bytes: usize,
        layer_idx: usize,
    ) -> Result<()> {
        let stream_raw = self.stream.raw() as u64;
        let n_int = moe.experts_gate_proj_fused.shape[1] as u32; // 512
        let k_in = moe.experts_gate_proj_fused.shape[2] as u32; // 2048
        let n_down = moe.experts_down_proj_fused.shape[1] as u32; // 2048
        let k_down = moe.experts_down_proj_fused.shape[2] as u32; // 512
        let num_experts = self.arch.num_experts;
        let top_k = self.arch.num_experts_per_tok;
        let m: u32 = 1;

        // 1. post_attention_layernorm on copy of last_hidden.
        let normed_region =
            self.arena.region("qwen36_pm_normed", last_hidden_bytes, 16)?;
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let _ = cuMemcpyDtoDAsync_v2(
                normed_region.device_ptr(),
                last_hidden_ptr,
                last_hidden_bytes,
                self.stream.raw() as _,
            );
        }
        let eps = self.arch.base.rms_norm_eps;
        unsafe {
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: 1, hidden, eps,
            }.launch(
                self.outside_kernels.fn_rmsnorm_inplace_f16,
                normed_region.device_ptr(),
                post_attn_norm_ptr,
                stream_raw,
            )?;
        }
        // No fence after rmsnorm: the next op is the GPU router GEMV
        // which runs on the same stream_raw, so ordering is automatic.
        // Removing the per-layer fence saves ~30 fence syscalls / token
        // (Phase 4b-prep iter22).

        // 2. Router top-k. Read normed input + router weights, compute
        //    logits on host, sort top-k, softmax-normalize.
        let hidden_us = hidden as usize;
        // Phase 4b-prep iter21: input_host / input_f32 are no longer
        // needed on the hot path — the shared-expert gate dot product
        // is now a fused GPU kernel (`shared_gate_dot_sigmoid_f16`).
        // The optional debug `normed_l2` rebuilds them lazily.
        let _ = hidden_us; // used by shared-gate kernel grid sizing
        // Phase 4b-prep iter17: GPU router GEMV. Replaces the
        // host-cached f32 matvec (~524k MAC × 30 layers / token)
        // with a single launch reading device-side f16 weights.
        let logits_bytes = num_experts * 4;
        let logits_region =
            self.arena.region("qwen36_pm_logits", logits_bytes, 16)?;
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let mut out = logits_region.device_ptr();
            let mut router_ptr = moe.router.offset_bytes;
            let mut input_ptr = normed_region.device_ptr();
            let mut nx = num_experts as i32;
            let mut hh = hidden_us as i32;
            let args = [
                (&mut out) as *mut u64 as *mut core::ffi::c_void,
                (&mut router_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut input_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut nx) as *mut i32 as *mut core::ffi::c_void,
                (&mut hh) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 256;
            let grid: u32 = num_experts as u32;
            let rc = cuLaunchKernel(
                self.outside_kernels.fn_router_gemv_f16_to_f32.raw() as CUfunction,
                grid, 1, 1,
                block, 1, 1,
                0,
                stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 router_gemv launch",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        // Fence to flush stream_raw before the sync DtoH below: empirically
        // sync `cuMemcpyDtoH_v2` does NOT synchronise non-default streams
        // on this driver/cudarc combination, so without this fence the
        // host reads stale (zero-initialised) logits for some blocks.
        self.stream.fence()?;
        // DtoH the small f32 logits buffer (1 KiB at num_experts=256).
        let mut logits = vec![0.0f32; num_experts];
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let _ = cuMemcpyDtoH_v2(
                logits.as_mut_ptr() as *mut _,
                logits_region.device_ptr(),
                logits_bytes,
            );
        }
        let mut indexed: Vec<(usize, f32)> =
            logits.iter().enumerate().map(|(i, &l)| (i, l)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top: Vec<(usize, f32)> = indexed.iter().take(top_k).copied().collect();
        let max = top.iter().map(|(_, v)| *v).fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = top.iter().map(|(_, v)| (v - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let weights: Vec<f32> = exps.iter().map(|e| e / sum).collect();

        // 3. Routed experts: each runs gate+up FP8 → silu·mul → down FP8.
        let mid_bytes = (n_int as usize) * 2;
        let down_bytes = (n_down as usize) * 2;
        let gate_region = self.arena.region("qwen36_pm_g", mid_bytes, 16)?;
        let up_region = self.arena.region("qwen36_pm_u", mid_bytes, 16)?;
        let silu_region = self.arena.region("qwen36_pm_s", mid_bytes, 16)?;
        let down_region = self.arena.region("qwen36_pm_d", down_bytes, 16)?;
        let int_per_expert_w = (n_int as u64) * (k_in as u64);
        let int_per_expert_bs =
            ((n_int as u64) / 128) * ((k_in as u64) / 128) * 4;
        let down_per_expert_w = (n_down as u64) * (k_down as u64);
        let down_per_expert_bs =
            ((n_down as u64) / 128) * ((k_down as u64) / 128) * 4;
        let gate_bs = match moe.experts_gate_proj_fused.blockscale_ptr { Some(p) => p, None => return Ok(()) };
        let up_bs = match moe.experts_up_proj_fused.blockscale_ptr { Some(p) => p, None => return Ok(()) };
        let down_bs = match moe.experts_down_proj_fused.blockscale_ptr { Some(p) => p, None => return Ok(()) };
        // Phase 4b-prep iter18: keep routed_sum on the GPU for the
        // entire expert loop. f32 accumulator, zeroed once on
        // stream_raw, then DtoH'd ONCE after the shared expert
        // finishes (with a self.stream.fence() before the DtoH —
        // that fence is the iter17-discovered invariant).
        let routed_sum_bytes = (n_down as usize) * 4;
        let routed_sum_region =
            self.arena.region("qwen36_pm_rs", routed_sum_bytes, 16)?;
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let rc = cuMemsetD8Async(
                routed_sum_region.device_ptr(),
                0,
                routed_sum_bytes,
                stream_raw as CUstream,
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 moe routed_sum cuMemsetD8Async",
                    rvllm_core::CudaErrorKind::MemcpyFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        for ((e_idx, _logit), w) in top.iter().zip(weights.iter()) {
            let e = *e_idx as u64;
            unsafe {
                self.fp8_proj_dispatch(kernel_gemv, gate_region.device_ptr(),
                    moe.experts_gate_proj_fused.offset_bytes + e * int_per_expert_w,
                    gate_bs + e * int_per_expert_bs,
                    normed_region.device_ptr(),
                    m, n_int, k_in, stream_raw)?;
                self.fp8_proj_dispatch(kernel_gemv, up_region.device_ptr(),
                    moe.experts_up_proj_fused.offset_bytes + e * int_per_expert_w,
                    up_bs + e * int_per_expert_bs,
                    normed_region.device_ptr(),
                    m, n_int, k_in, stream_raw)?;
            }
            // GPU silu·mul: silu_region = silu(gate_region) * up_region
            // (Phase 4b-prep iter11 — replaces DtoH gate + DtoH up +
            // CPU loop + HtoD silu round-trip per expert).
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let mut out = silu_region.device_ptr();
                let mut g = gate_region.device_ptr();
                let mut u = up_region.device_ptr();
                let mut nn = n_int as i32;
                let args = [
                    (&mut out) as *mut u64 as *mut core::ffi::c_void,
                    (&mut g) as *mut u64 as *mut core::ffi::c_void,
                    (&mut u) as *mut u64 as *mut core::ffi::c_void,
                    (&mut nn) as *mut i32 as *mut core::ffi::c_void,
                ];
                let block: u32 = 256;
                let grid = (n_int as u32 + block - 1) / block;
                let _ = cuLaunchKernel(
                    self.outside_kernels.fn_silu_mul_f16.raw() as CUfunction,
                    grid, 1, 1,
                    block, 1, 1,
                    0,
                    stream_raw as CUstream,
                    args.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                );
            }
            unsafe {
                self.fp8_proj_dispatch(kernel_gemv, down_region.device_ptr(),
                    moe.experts_down_proj_fused.offset_bytes + e * down_per_expert_w,
                    down_bs + e * down_per_expert_bs,
                    silu_region.device_ptr(),
                    m, n_down, k_down, stream_raw)?;
            }
            // GPU scaled_add: routed_sum_region += w * down_region.
            // Same stream as down GEMV → automatic ordering.
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let mut acc = routed_sum_region.device_ptr();
                let mut input = down_region.device_ptr();
                let mut weight = *w;
                let mut nn = n_down as i32;
                let args = [
                    (&mut acc) as *mut u64 as *mut core::ffi::c_void,
                    (&mut input) as *mut u64 as *mut core::ffi::c_void,
                    (&mut weight) as *mut f32 as *mut core::ffi::c_void,
                    (&mut nn) as *mut i32 as *mut core::ffi::c_void,
                ];
                let block: u32 = 256;
                let grid = (n_down as u32 + block - 1) / block;
                let _ = cuLaunchKernel(
                    self.outside_kernels.fn_scaled_add_f16_to_f32.raw() as CUfunction,
                    grid, 1, 1,
                    block, 1, 1,
                    0,
                    stream_raw as CUstream,
                    args.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                );
            }
        }
        // Optional debug L2 of routed-only sum.
        let routed_only_l2: f32 = if std::env::var("RVLLM_QWEN36_DEBUG_MOE").is_ok() {
            self.stream.fence()?;
            let mut rs_host = vec![0.0f32; n_down as usize];
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let _ = cuMemcpyDtoH_v2(
                    rs_host.as_mut_ptr() as *mut _,
                    routed_sum_region.device_ptr(),
                    routed_sum_bytes,
                );
            }
            rs_host.iter().map(|x| x*x).sum::<f32>().sqrt()
        } else { 0.0 };

        // 4. Shared expert FFN scaled by sigmoid(shared_expert_gate_logit).
        let sh_gate_bs = moe.shared_expert_gate_proj.blockscale_ptr.unwrap_or(0);
        let sh_up_bs = moe.shared_expert_up_proj.blockscale_ptr.unwrap_or(0);
        let sh_down_bs = moe.shared_expert_down_proj.blockscale_ptr.unwrap_or(0);
        if sh_gate_bs != 0 && sh_up_bs != 0 && sh_down_bs != 0 {
            unsafe {
                self.fp8_proj_dispatch(kernel_gemv, gate_region.device_ptr(),
                    moe.shared_expert_gate_proj.offset_bytes, sh_gate_bs,
                    normed_region.device_ptr(),
                    m, n_int, k_in, stream_raw)?;
                self.fp8_proj_dispatch(kernel_gemv, up_region.device_ptr(),
                    moe.shared_expert_up_proj.offset_bytes, sh_up_bs,
                    normed_region.device_ptr(),
                    m, n_int, k_in, stream_raw)?;
            }
            // GPU silu·mul on the shared expert (Phase 4b-prep iter11).
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let mut out = silu_region.device_ptr();
                let mut g = gate_region.device_ptr();
                let mut u = up_region.device_ptr();
                let mut nn = n_int as i32;
                let args = [
                    (&mut out) as *mut u64 as *mut core::ffi::c_void,
                    (&mut g) as *mut u64 as *mut core::ffi::c_void,
                    (&mut u) as *mut u64 as *mut core::ffi::c_void,
                    (&mut nn) as *mut i32 as *mut core::ffi::c_void,
                ];
                let block: u32 = 256;
                let grid = (n_int as u32 + block - 1) / block;
                let _ = cuLaunchKernel(
                    self.outside_kernels.fn_silu_mul_f16.raw() as CUfunction,
                    grid, 1, 1,
                    block, 1, 1,
                    0,
                    stream_raw as CUstream,
                    args.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                );
            }
            unsafe {
                self.fp8_proj_dispatch(kernel_gemv, down_region.device_ptr(),
                    moe.shared_expert_down_proj.offset_bytes, sh_down_bs,
                    silu_region.device_ptr(),
                    m, n_down, k_down, stream_raw)?;
            }
            // shared_expert_gate is Linear(hidden→1) per vLLM
            // qwen3_next.py:127-133: gate_logit = weight · normed_hidden
            // (scalar per token), then sigmoid(gate_logit) scales the
            // shared expert output. Weights cached as f32 host vec at
            // bring-up (Phase 4b-prep iter16); per-token path is just
            // a dot-product over RAM.
            // Phase 4b-prep iter21: fused GPU dot+sigmoid + a
            // device-pointer scaled_add kills the per-layer fence +
            // DtoH that an iter20 attempt regressed on. Both kernels
            // run on stream_raw, chained via the device scalar
            // `sg_sigmoid_region`; host never sees the value.
            let sg_sigmoid_region =
                self.arena.region("qwen36_pm_sg_sigmoid", 4, 16)?;
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let mut out = sg_sigmoid_region.device_ptr();
                let mut weight = moe.shared_expert_gate_logit.offset_bytes;
                let mut input_ptr = normed_region.device_ptr();
                let mut hh = hidden_us as i32;
                let args = [
                    (&mut out) as *mut u64 as *mut core::ffi::c_void,
                    (&mut weight) as *mut u64 as *mut core::ffi::c_void,
                    (&mut input_ptr) as *mut u64 as *mut core::ffi::c_void,
                    (&mut hh) as *mut i32 as *mut core::ffi::c_void,
                ];
                let block: u32 = 256;
                let grid: u32 = 1;
                let _ = cuLaunchKernel(
                    self.outside_kernels.fn_shared_gate_dot_sigmoid_f16.raw() as CUfunction,
                    grid, 1, 1,
                    block, 1, 1,
                    0,
                    stream_raw as CUstream,
                    args.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                );
            }
            // GPU scaled_add (devw variant): routed_sum_region +=
            // *sg_sigmoid_region * down_region. Reads the sigmoid
            // from the device scalar produced by the kernel above —
            // stream-ordered, no fence or host round-trip.
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let mut acc = routed_sum_region.device_ptr();
                let mut input = down_region.device_ptr();
                let mut devw = sg_sigmoid_region.device_ptr();
                let mut nn = n_down as i32;
                let args = [
                    (&mut acc) as *mut u64 as *mut core::ffi::c_void,
                    (&mut input) as *mut u64 as *mut core::ffi::c_void,
                    (&mut devw) as *mut u64 as *mut core::ffi::c_void,
                    (&mut nn) as *mut i32 as *mut core::ffi::c_void,
                ];
                let block: u32 = 256;
                let grid = (n_down as u32 + block - 1) / block;
                let _ = cuLaunchKernel(
                    self.outside_kernels.fn_scaled_add_f16_to_f32_devw.raw() as CUfunction,
                    grid, 1, 1,
                    block, 1, 1,
                    0,
                    stream_raw as CUstream,
                    args.as_ptr() as *mut *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                );
            }
        }
        // 5. Residual sum, GPU-side (Phase 4b-prep iter19):
        //    last_hidden[i] = f16(f16_to_f32(last_hidden[i]) + routed_sum[i])
        // Same stream as the per-expert scaled_add and the prior
        // attn writeback into last_hidden_ptr — automatic ordering.
        // Also: with the residual now device-side, the optional debug
        // `RVLLM_QWEN36_DEBUG_MOE` path no longer has free routed_sum
        // on the host. We DtoH it lazily inside the env-gated branch.
        if std::env::var("RVLLM_QWEN36_DEBUG_MOE").is_ok() {
            self.stream.fence()?;
            let mut rs_host = vec![0.0f32; n_down as usize];
            let mut lh_host = vec![0u8; last_hidden_bytes];
            let mut normed_host = vec![0u8; last_hidden_bytes];
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let _ = cuMemcpyDtoH_v2(
                    rs_host.as_mut_ptr() as *mut _,
                    routed_sum_region.device_ptr(),
                    routed_sum_bytes,
                );
                let _ = cuMemcpyDtoH_v2(
                    lh_host.as_mut_ptr() as *mut _,
                    last_hidden_ptr,
                    last_hidden_bytes,
                );
                let _ = cuMemcpyDtoH_v2(
                    normed_host.as_mut_ptr() as *mut _,
                    normed_region.device_ptr(),
                    last_hidden_bytes,
                );
            }
            let total_l2: f32 = rs_host.iter().map(|x| x*x).sum::<f32>().sqrt();
            let shared_l2 = (total_l2*total_l2 - routed_only_l2*routed_only_l2).max(0.0).sqrt();
            let mut normed_l2_sq = 0.0f32;
            for i in 0..hidden_us {
                let v = f16_bits_to_f32(u16::from_le_bytes([
                    normed_host[i * 2], normed_host[i * 2 + 1],
                ]));
                normed_l2_sq += v * v;
            }
            let normed_l2 = normed_l2_sq.sqrt();
            eprintln!("[moe] normed_L2={normed_l2:.2} routed_L2={routed_only_l2:.3} shared_L2~{shared_l2:.3} total_L2={total_l2:.3}");
            let _ = lh_host; // reserved for future per-element debugging
        }
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let mut inout = last_hidden_ptr;
            let mut add = routed_sum_region.device_ptr();
            let mut nn = hidden as i32;
            let args = [
                (&mut inout) as *mut u64 as *mut core::ffi::c_void,
                (&mut add) as *mut u64 as *mut core::ffi::c_void,
                (&mut nn) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 256;
            let grid = (hidden + block - 1) / block;
            let rc = cuLaunchKernel(
                self.outside_kernels.fn_f16_plus_f32_inplace_f16.raw() as CUfunction,
                grid, 1, 1,
                block, 1, 1,
                0,
                stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 moe residual launch",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        // No fence: the residual kernel runs on stream_raw, the next
        // layer's first read of last_hidden_ptr is also on stream_raw,
        // so ordering is automatic.
        Ok(())
    }

    /// Helper: shared closer for `forward_outside_only` and the
    /// Phase-4v experimental path. Takes a `hidden_region` already
    /// populated with f16 hidden state and returns the argmax token
    /// id over the last token's lm_head logits.
    fn forward_qwen36_outside_closer(
        &self,
        hidden_region: &rvllm_mem::Region<'_>,
        num_tokens: u32,
        hidden: u32,
        vocab: u32,
        last_idx: usize,
    ) -> Result<i32> {
        let eps = self.arch.base.rms_norm_eps;
        let hidden_fp8_bytes = (num_tokens as usize) * (hidden as usize);
        let hidden_scale_bytes = (num_tokens as usize) * 4;
        let logits_bytes = (num_tokens as usize) * (vocab as usize) * 2;
        let hidden_fp8_region = self
            .arena
            .region("qwen36_pl_h_fp8", hidden_fp8_bytes, 16)?;
        let hidden_scale_region = self
            .arena
            .region("qwen36_pl_h_scale", hidden_scale_bytes, 16)?;
        let logits_region = self.arena.region("qwen36_pl_logits", logits_bytes, 16)?;
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
        // Phase 4b-prep iter27: GPU argmax over the f16 logits row,
        // replacing a `vocab * 2`-byte DtoH (~524 KiB at vocab=262K)
        // + host max-loop with a single 4-byte device→host copy of
        // the winning token id.
        let logits_row_bytes = (vocab as usize) * 2;
        let last_off = (last_idx as u64) * (logits_row_bytes as u64);
        let token_region = self.arena.region("qwen36_pl_token", 4, 16)?;
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let mut logits_ptr = logits_region.device_ptr() + last_off;
            let mut out_ptr = token_region.device_ptr();
            let mut vs = vocab as i32;
            let args = [
                (&mut logits_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut out_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut vs) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block: u32 = 512;
            let grid: u32 = 1;
            let _ = cuLaunchKernel(
                self.outside_kernels.fn_argmax_f16.raw() as CUfunction,
                grid, 1, 1,
                block, 1, 1,
                0,
                stream_raw as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
        }
        self.stream.fence()?;
        let mut tok_buf = [0i32; 1];
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let _ = cuMemcpyDtoH_v2(
                tok_buf.as_mut_ptr() as *mut _,
                token_region.device_ptr(),
                4,
            );
        }
        Ok(tok_buf[0])
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

    /// Phase 3a (Qwen batched-prefill plan): single dispatch point for
    /// every per-layer projection in `apply_layer_*`. Today every
    /// projection call site instantiates `Fp8GemvF16InLaunch { m, n, k }`
    /// directly; routing them through this method is the prerequisite
    /// for Phase 4, which will batch the per-token loop and pass
    /// `m = num_tokens` instead of `m = 1`.
    ///
    /// Routing:
    /// * **m = 1**: delegates byte-identically to
    ///   `Fp8GemvF16InLaunch { m: 1, n, k }`. Existing tests +
    ///   determinism canaries stay green.
    /// * **m ≥ 2**: returns a typed error pointing at Phase 3b. The
    ///   plan there is to:
    ///     1. quantize the f16 input to fp8 + per-token f32 amax via
    ///        a small `fp8_quantize_per_token_f16` kernel (new),
    ///     2. either pass the existing `[N/128, K/128]` row-major
    ///        weight blockscale straight into `cublaslt.fp8_gemm` (if
    ///        the layout is acceptable to cuBLASLt) or transpose it
    ///        into MN-major like Gemma does for the CUTLASS path,
    ///     3. dispatch to `self.cublaslt.fp8_gemm` for tensor-core
    ///        throughput at large m, then validate per-shape cosine
    ///        ≥ 0.9999 against a reference implementation that loops
    ///        the m=1 GEMV N times.
    ///   The error message names Phase 3b explicitly so a future
    ///   Phase 4 patch that flips the caller to m=N immediately
    ///   surfaces "Phase 3b not done" instead of producing silently
    ///   wrong tokens.
    ///
    /// `kernel_gemv` is the resolved
    /// `fn_fp8_gemv_wpr_native_f16in` handle (the caller is expected
    /// to have already failed-fast if it's not loaded — see
    /// `forward_qwen36_decode`'s ok_or_else).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn fp8_proj_dispatch(
        &self,
        kernel_gemv: rvllm_kernels::KernelFn,
        out_f16: u64,
        weight_fp8: u64,
        b_blockscale: u64,
        input_f16: u64,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        if m == 0 || n == 0 || k == 0 {
            return Err(rvllm_core::RvllmError::cuda(
                "qwen36 fp8_proj_dispatch: zero-size dim",
                rvllm_core::CudaErrorKind::Other,
                rvllm_core::CudaCtx::setup(),
            ));
        }
        if m == 1 {
            // SAFETY: caller-supplied pointers are already-validated
            // device addresses; Fp8GemvF16InLaunch internally re-checks
            // K%8 alignment.
            return rvllm_fused::gemma4_launcher::Fp8GemvF16InLaunch { m, n, k }.launch(
                kernel_gemv,
                out_f16,
                weight_fp8,
                b_blockscale,
                input_f16,
                stream,
            );
        }
        // Phase 3c: when m ≥ 128 AND CutlassBackend::SoSm120 is loaded,
        // dispatch through CUTLASS SM120's blockwise FP8 GEMM (the
        // same .so Gemma uses at lm_head). This is the production
        // fast path on sm_121 (GB10) — cuBLASLt has no blockwise
        // FP8 kernel for that arch; CUTLASS SM120 hard-asserts
        // M≥128 so smaller m falls through to Phase 3b's cuBLASLt
        // try-then-looped-GEMV path.
        #[cfg(feature = "cuda")]
        if m >= 128 {
            if let CutlassBackend::SoSm120(ref lib) = self.cutlass {
                // Per-token amax quantise (CUTLASS prep_sfa expects [M] f32).
                let fp8_bytes = (m as usize) * (k as usize);
                let amax_bytes = (m as usize) * 4;
                let in_fp8 = self
                    .arena
                    .region("qwen36_proj_in_fp8_cutlass", fp8_bytes, 16)?;
                let in_amax = self
                    .arena
                    .region("qwen36_proj_in_amax", amax_bytes, 16)?;
                unsafe {
                    use cudarc::driver::sys::*;
                    let block_dim: u32 = (k as u32).min(1024);
                    let mut o_fp8 = in_fp8.device_ptr();
                    let mut o_amax = in_amax.device_ptr();
                    let mut i_ptr = input_f16;
                    let mut k_i: i32 = k as i32;
                    let args = [
                        (&mut o_fp8) as *mut u64 as *mut core::ffi::c_void,
                        (&mut o_amax) as *mut u64 as *mut core::ffi::c_void,
                        (&mut i_ptr) as *mut u64 as *mut core::ffi::c_void,
                        (&mut k_i) as *mut i32 as *mut core::ffi::c_void,
                    ];
                    let rc = cuLaunchKernel(
                        self.outside_kernels.fn_fp8_quantize_per_token_amax_f16.raw() as CUfunction,
                        m, 1, 1,
                        block_dim, 1, 1,
                        0,
                        stream as CUstream,
                        args.as_ptr() as *mut *mut core::ffi::c_void,
                        core::ptr::null_mut(),
                    );
                    if rc != CUresult::CUDA_SUCCESS {
                        return Err(rvllm_core::RvllmError::cuda(
                            "qwen36 fp8_proj_dispatch: amax-quantise launch (CUTLASS path)",
                            rvllm_core::CudaErrorKind::LaunchFailed,
                            rvllm_core::CudaCtx::setup(),
                        ));
                    }
                }
                // Allocate SFA / SFB / workspace per CUTLASS sizing.
                let sfa_n = lib.sfa_bytes(m as i32, k as i32);
                let sfb_n = lib.sfb_bytes(n as i32, k as i32);
                let ws_n = lib.workspace_size(m as i32, n as i32, k as i32);
                if sfa_n == 0 || sfb_n == 0 {
                    return Err(rvllm_core::RvllmError::cuda(
                        "qwen36 fp8_proj_dispatch: CUTLASS SM120 reported \
                         sfa_bytes/sfb_bytes==0 — legacy .so without these \
                         helpers; rebuild kernels/build_cutlass_sm120_so.sh",
                        rvllm_core::CudaErrorKind::Other,
                        rvllm_core::CudaCtx::setup(),
                    ));
                }
                let sfa = self.arena.region("qwen36_proj_sfa", sfa_n.max(4), 16)?;
                let sfb = self.arena.region("qwen36_proj_sfb", sfb_n.max(4), 16)?;
                let ws = self
                    .arena
                    .region("qwen36_proj_cutlass_ws", ws_n.max(16), 256)?;
                unsafe {
                    lib.launch_prep_sfa(in_amax.device_ptr(), sfa.device_ptr(), m as i32, k as i32, stream)?;
                    lib.launch_prep_sfb(b_blockscale, sfb.device_ptr(), n as i32, k as i32, stream)?;
                    lib.launch_fp8_gemm_blockscale(
                        out_f16,
                        in_fp8.device_ptr(),
                        weight_fp8,
                        sfa.device_ptr(),
                        sfb.device_ptr(),
                        m as i32,
                        n as i32,
                        k as i32,
                        ws.device_ptr(),
                        ws_n,
                        stream,
                    )?;
                }
                return Ok(());
            }
        }

        // Phase 3b: m≥2 path through cuBLASLt blockwise fp8_gemm.
        //   1) Per-(token, 128-K-block) f16→fp8 amax-quantise the
        //      input → scratch `[M, K] fp8` + `[M, ceil(K/128)] f32`
        //      activation scales (mode VEC128_32F = 4 in cuBLASLt).
        //   2) `cublaslt.fp8_gemm_blockwise(...)` with the activation
        //      scales above and the existing `[N/128, K/128]` weight
        //      blockscale (mode BLK128x128_32F = 5). Both modes set
        //      explicitly on the matmul descriptor.
        //   3) f16 output is written directly to `out_f16`.
        //
        // The per-K-block input scale + 128×128 weight scale are the
        // cuBLASLt-native modes that match Qwen's checkpoint layout —
        // no scale transpose, no lossy reduction to per-channel.
        let fp8_bytes = (m as usize) * (k as usize); // E4M3 = 1 byte/elem
        let k_blocks = (k as usize + 127) / 128;
        let scale_bytes = (m as usize) * k_blocks * 4; // f32 per (row, K-block)
        let in_fp8 = self
            .arena
            .region("qwen36_proj_in_fp8", fp8_bytes, 16)?;
        let in_scale = self
            .arena
            .region("qwen36_proj_in_scale", scale_bytes, 16)?;
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            // Quantise launch: grid=(K_blocks, M, 1), block=(128, 1, 1).
            let mut out_fp8_ptr = in_fp8.device_ptr();
            let mut out_scale_ptr = in_scale.device_ptr();
            let mut in_ptr = input_f16;
            let mut k_i: i32 = k as i32;
            let args = [
                (&mut out_fp8_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut out_scale_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut in_ptr) as *mut u64 as *mut core::ffi::c_void,
                (&mut k_i) as *mut i32 as *mut core::ffi::c_void,
            ];
            let rc = cuLaunchKernel(
                self.outside_kernels.fn_fp8_quantize_per_token_f16.raw() as CUfunction,
                /*grid*/ k_blocks as u32, m, 1,
                /*block*/ 128, 1, 1,
                /*shared*/ 0,
                stream as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if rc != CUresult::CUDA_SUCCESS {
                return Err(rvllm_core::RvllmError::cuda(
                    "qwen36 fp8_proj_dispatch: fp8_quantize_per_token_f16 launch",
                    rvllm_core::CudaErrorKind::LaunchFailed,
                    rvllm_core::CudaCtx::setup(),
                ));
            }
        }
        // First try the cuBLASLt blockwise FP8 path. On
        // sm_100/sm_120 (Blackwell-server / RTX 5090) this dispatches
        // to a tensor-core kernel for ≥128×N×K. On sm_121 (GB10
        // consumer Blackwell) cuBLASLt does NOT ship a blockwise FP8
        // kernel today — `Algo­GetHeuristic` returns "no algo".
        //
        // The fall-back is a looped-m=1 GEMV reference: correct, but
        // not faster than the existing per-token loop. It lets Phase
        // 4 wire the dispatcher into the prefill loop without
        // breaking sm_121 — when CUDA 13.x eventually adds a blockwise
        // sm_121 kernel, the same call site immediately picks it up.
        // Phase 3c will plug in the CUTLASS SM120 blockwise GEMM
        // (already built for Gemma) as the sm_121 fast path.
        #[cfg(feature = "cuda")]
        let blockwise_result = self.cublaslt.fp8_gemm_blockwise(
            in_fp8.device_ptr(),
            weight_fp8,
            out_f16,
            m as i32,
            n as i32,
            k as i32,
            in_scale.device_ptr(),
            b_blockscale,
            stream,
        );
        #[cfg(feature = "cuda")]
        match blockwise_result {
            Ok(()) => return Ok(()),
            Err(_) => {
                // Fall back: N×Fp8GemvF16InLaunch at m=1, one row at a
                // time. We log the no-algo path once per process so
                // operators see "blockwise no-algo, falling back" rather
                // than silently paying the per-token cost on every batch.
                use std::sync::atomic::{AtomicBool, Ordering};
                static LOGGED: AtomicBool = AtomicBool::new(false);
                if !LOGGED.swap(true, Ordering::Relaxed) {
                    tracing::warn!(
                        "qwen36 fp8_proj_dispatch: cuBLASLt has no blockwise FP8 \
                         algo for this arch (likely sm_121). Falling back to \
                         looped m=1 GEMV — correct but not accelerated. \
                         Phase 3c will plug in CUTLASS SM120."
                    );
                }
                let row_bytes_in = (k as u64) * 2;
                let row_bytes_out = (n as u64) * 2;
                for row in 0..(m as u64) {
                    let row_in = input_f16 + row * row_bytes_in;
                    let row_out = out_f16 + row * row_bytes_out;
                    rvllm_fused::gemma4_launcher::Fp8GemvF16InLaunch {
                        m: 1,
                        n,
                        k,
                    }
                    .launch(
                        kernel_gemv,
                        row_out,
                        weight_fp8,
                        b_blockscale,
                        row_in,
                        stream,
                    )?;
                }
                Ok(())
            }
        }
        // Non-cuda build: the entire m≥2 path was cfg-gated out; we
        // can't actually run anything, so signal a loud error rather
        // than silently returning Ok().
        #[cfg(not(feature = "cuda"))]
        Err(rvllm_core::RvllmError::cuda(
            "qwen36 fp8_proj_dispatch: m≥2 path requires `cuda` feature",
            rvllm_core::CudaErrorKind::Other,
            rvllm_core::CudaCtx::setup(),
        ))
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
