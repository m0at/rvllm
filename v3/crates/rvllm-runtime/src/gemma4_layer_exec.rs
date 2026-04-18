//! Gemma 4 layer forward -- 14 kernel launches per layer.
//!
//! Differs from the Llama/Qwen path (layer_exec.rs) in:
//!   - 4 norms per layer (input, post_attn, pre_ff, post_ff)
//!   - QK-norm (RMSNorm on Q and K heads before RoPE)
//!   - v_norm (parameter-free RMS norm on V after projection)
//!   - GELU(tanh) activation instead of SiLU
//!   - Partial RoPE (only rotate first rotary_dim dims per head)
//!   - Per-layer KV head count (sliding vs global)
//!   - head_dim = 256 (requires FA3 .so compiled for 256)
//!   - Per-layer learnable scalar (applied ONCE after both sub-blocks)
//!
//! Launch sequence:
//!   1.  fused_rmsnorm_fp8_quant          input_layernorm
//!   2.  fp8_gemm (cuBLASLt)             Q||K||V projection
//!  2b.  vnorm_f16                       parameter-free RMS norm on V
//!   3.  fused_qk_rmsnorm                QK-norm on Q and K heads
//!   4.  fused_rope_partial_fp8kv        partial RoPE + FP8 Q + paged KV
//!   5.  paged_decode / paged_prefill    FA3 attention (head_dim=256)
//!   6.  quantize_fp8_per_token          attn_out -> fp8
//!   7.  fp8_gemm_residual (cuBLASLt)    O proj += residual
//!   8.  fused_rmsnorm                   post_attention_layernorm (norm only)
//!   9.  fused_rmsnorm_fp8_quant         pre_feedforward_layernorm
//!  10.  fp8_gemm (cuBLASLt)             gate||up projection
//!  11.  fused_gelu_mul_fp8_quant        GELU(tanh)(gate) * up -> FP8
//!  12.  fp8_gemm_residual (cuBLASLt)    down proj += residual
//!  13.  fused_rmsnorm                   post_feedforward_layernorm (norm only)
//!  14.  residual_scale_f16              residual *= layer_scalar (once)

use rvllm_core::Result;
use rvllm_cutlass::{CublasLt, CutlassLib, Fp8GemmPlan};
use rvllm_fused::FusedRmsnormFp8QuantLaunch;
use rvllm_fused::gemma4_launcher;
use rvllm_kernels::KernelFn;

use rvllm_attention::{
    Fa3Kernels, PagedDecodeFp8Launcher, PagedDecodeParams,
};

use rvllm_loader::gemma4_arch::Gemma4LayerType;

#[derive(Copy, Clone, Debug)]
pub struct Gemma4LayerDims {
    pub num_tokens: u32,
    pub hidden: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub rotary_dim: u32,
    pub intermediate: u32,
    pub block_size: u32,
    pub max_blocks_per_seq: u32,
    pub num_blocks_total: u32,
    pub attn_scale: f32,
    pub rms_eps: f32,
    pub layer_type: Gemma4LayerType,
    pub sliding_window: u32,
}

#[derive(Copy, Clone, Debug)]
pub struct Gemma4LayerWeightPtrs {
    pub attn_norm_gamma: u64,
    pub post_attn_norm_gamma: u64,
    pub pre_ff_norm_gamma: u64,
    pub post_ff_norm_gamma: u64,
    pub q_norm_gamma: u64,
    pub k_norm_gamma: u64,
    pub qkv_fp8: u64,
    pub qkv_scale: u64,
    pub o_fp8: u64,
    pub o_scale: u64,
    pub gate_up_fp8: u64,
    pub gate_up_scale: u64,
    pub down_fp8: u64,
    pub down_scale: u64,
    pub layer_scalar_ptr: u64, // [1] f16, per-layer residual multiplier
}

#[derive(Copy, Clone, Debug)]
pub struct Gemma4LayerScratch {
    pub hidden_fp8: u64,
    pub hidden_scale: u64,
    pub q_out: u64,
    pub k_out: u64,
    pub v_out: u64,
    pub q_normed: u64,
    pub k_normed: u64,
    pub q_fp8: u64,
    pub k_cache: u64,
    pub v_cache: u64,
    pub q_scale_ptr: u64,
    pub kv_scale_ptr: u64,
    pub attn_out: u64,
    pub attn_out_fp8: u64,
    pub attn_out_scale: u64,
    pub delta_f16: u64,
    pub gate_up_out: u64,
    pub gate_up_fp8: u64,
    pub gate_up_scale: u64,
    pub mlp_out_fp8: u64,
    pub mlp_out_scale: u64,
    pub cutlass_workspace: u64,
    pub cutlass_workspace_bytes: usize,
    pub fa3_workspace: u64,
}

#[derive(Clone, Debug)]
pub struct Gemma4GemmPlans {
    pub qkv: Fp8GemmPlan,
    pub o: Fp8GemmPlan,
    pub gate_up: Fp8GemmPlan,
    pub down: Fp8GemmPlan,
}

#[derive(Copy, Clone, Debug)]
pub struct Gemma4MetadataPtrs {
    pub positions: u64,
    pub slot_mapping: u64,
    pub cos: u64,
    pub sin: u64,
    pub block_tables: u64,
    pub context_lens: u64,
}

#[derive(Copy, Clone, Debug)]
pub struct Gemma4LayerKernels {
    pub fused_rmsnorm: KernelFn,
    pub fused_rmsnorm_fp8_quant: KernelFn,
    pub fused_qk_rmsnorm: KernelFn,
    pub fused_rope_partial_fp8kv: KernelFn,
    pub fused_gelu_mul: KernelFn,
    pub quantize_fp8_per_token: KernelFn,
    pub residual_scale_f16: KernelFn,
    pub vnorm_f16: KernelFn,
    pub vector_add_f16: KernelFn,
    pub bf16_to_f16_sat: KernelFn,
    pub rmsnorm_inplace_bf16: KernelFn,
    pub vector_add_bf16_to_f16: KernelFn,
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn gemma4_forward(
    dims: Gemma4LayerDims,
    kernels: &Gemma4LayerKernels,
    weights: &Gemma4LayerWeightPtrs,
    scratch: &Gemma4LayerScratch,
    meta: &Gemma4MetadataPtrs,
    cublaslt: &CublasLt,
    sliding_attention: &Fa3Kernels,
    global_attention: &Fa3Kernels,
    residual: u64,
    stream: u64,
) -> Result<()> {
    let q_dim = dims.num_heads * dims.head_dim;
    let _kv_dim = dims.num_kv_heads * dims.head_dim;
    let qkv_rows = (dims.num_heads + 2 * dims.num_kv_heads) * dims.head_dim;

    #[cfg(feature = "cuda")]
    let dbg_layer: i32 = {
        use std::sync::atomic::{AtomicU32, Ordering};
        static DBG_CTR: AtomicU32 = AtomicU32::new(0);
        let cnt = DBG_CTR.fetch_add(1, Ordering::Relaxed);
        if cnt < 2 && std::env::var("RVLLM_DBG_LAYER").is_ok() {
            cnt as i32
        } else {
            -1
        }
    };
    #[cfg(feature = "cuda")]
    macro_rules! probe {
        ($label:expr, $ptr:expr, $n:expr) => {
            if dbg_layer >= 0 {
                cudarc::driver::sys::cuStreamSynchronize(stream as _);
                let mut s = [0u16; 4];
                cudarc::driver::sys::cuMemcpyDtoH_v2(s.as_mut_ptr() as *mut _, $ptr, 8);
                let v: Vec<f32> = s.iter().map(|&x| crate::bring_up::f16_to_f32(x)).collect();
                eprintln!("    [L{} {}] first4={:.4?}", dbg_layer, $label, v);
            }
        };
    }
    #[cfg(feature = "cuda")]
    macro_rules! probe_f32 {
        ($label:expr, $ptr:expr) => {
            if dbg_layer >= 0 {
                cudarc::driver::sys::cuStreamSynchronize(stream as _);
                let mut v = [0.0f32; 1];
                cudarc::driver::sys::cuMemcpyDtoH_v2(v.as_mut_ptr() as *mut _, $ptr, 4);
                eprintln!("    [L{} {}] = {:.6e}", dbg_layer, $label, v[0]);
            }
        };
    }
    #[cfg(feature = "cuda")]
    macro_rules! probe_amax {
        ($label:expr, $ptr:expr, $n:expr) => {
            if dbg_layer >= 0 {
                cudarc::driver::sys::cuStreamSynchronize(stream as _);
                let count = $n as usize;
                let mut buf = vec![0u16; count];
                cudarc::driver::sys::cuMemcpyDtoH_v2(buf.as_mut_ptr() as *mut _, $ptr, count * 2);
                let mut amax: f32 = 0.0;
                for &x in &buf {
                    let v = crate::bring_up::f16_to_f32(x).abs();
                    if v > amax { amax = v; }
                }
                eprintln!("    [L{} {}] amax={:.4}", dbg_layer, $label, amax);
            }
        };
    }

    // 1. input_layernorm -> FP8 quant
    FusedRmsnormFp8QuantLaunch {
        num_tokens: dims.num_tokens,
        hidden: dims.hidden,
        eps: dims.rms_eps,
    }
    .launch(
        kernels.fused_rmsnorm_fp8_quant,
        scratch.hidden_fp8,
        scratch.hidden_scale,
        residual,
        weights.attn_norm_gamma,
        stream,
    )?;

    #[cfg(feature = "cuda")]
    probe!("after_step1_residual", residual, dims.hidden);

    // 2. Q||K||V projection (cuBLASLt FP8 GEMM)
    #[cfg(feature = "cuda")]
    cublaslt.fp8_gemm(
        scratch.hidden_fp8,
        weights.qkv_fp8,
        scratch.q_out,
        dims.num_tokens as i32,
        qkv_rows as i32,
        dims.hidden as i32,
        scratch.hidden_scale,
        weights.qkv_scale,
        stream,
    )?;

    // 2b. V-norm: parameter-free RMS normalization on V heads.
    gemma4_launcher::VnormF16Launch {
        num_tokens: dims.num_tokens,
        num_kv_heads: dims.num_kv_heads,
        head_dim: dims.head_dim,
        eps: dims.rms_eps,
    }
    .launch(kernels.vnorm_f16, scratch.v_out, stream)?;

    // 3. QK-norm: RMSNorm on each Q head and each K head independently.
    // Input: q_out [num_tokens, num_heads, head_dim], k_out [num_tokens, num_kv_heads, head_dim]
    // q_norm_gamma, k_norm_gamma are [head_dim] vectors.
    // Output: q_normed, k_normed (f16, same shape)
    gemma4_launcher::FusedQkRmsnormLaunch {
        num_tokens: dims.num_tokens,
        num_heads: dims.num_heads,
        num_kv_heads: dims.num_kv_heads,
        head_dim: dims.head_dim,
        eps: dims.rms_eps,
    }
    .launch(
        kernels.fused_qk_rmsnorm,
        scratch.q_out,
        scratch.k_out,
        scratch.q_normed,
        scratch.k_normed,
        weights.q_norm_gamma,
        weights.k_norm_gamma,
        stream,
    )?;

    // 4. Partial RoPE + FP8 quantize Q + paged KV cache write
    // Only the first `rotary_dim` elements of each head get rotation;
    // the rest pass through unchanged.
    gemma4_launcher::FusedRopePartialFp8KvLaunch {
        num_tokens: dims.num_tokens,
        num_heads: dims.num_heads,
        num_kv_heads: dims.num_kv_heads,
        head_dim: dims.head_dim,
        rotary_dim: dims.rotary_dim,
    }
    .launch(
        kernels.fused_rope_partial_fp8kv,
        scratch.q_normed,
        scratch.k_normed,
        scratch.v_out,
        scratch.q_fp8,
        scratch.k_cache,
        scratch.v_cache,
        meta.cos,
        meta.sin,
        meta.positions,
        meta.slot_mapping,
        scratch.q_scale_ptr,
        scratch.kv_scale_ptr,
        stream,
    )?;

    // 5. Attention backend selection:
    //    - sliding layers: FA3 SM90 at head_dim=256
    //    - global layers: fallback paged attention at head_dim=512
    // For sliding layers: context_lens should be clamped to
    // min(real_ctx, sliding_window) by the scheduler before this call.
    let attention = match dims.layer_type {
        Gemma4LayerType::SlidingAttention => sliding_attention,
        Gemma4LayerType::GlobalAttention => global_attention,
    };
    let decode = PagedDecodeFp8Launcher::new(attention);
    let window_size_left: i32 = match dims.layer_type {
        Gemma4LayerType::SlidingAttention => (dims.sliding_window as i32) - 1,
        Gemma4LayerType::GlobalAttention => -1,
    };
    let decode_params = PagedDecodeParams {
        num_seqs: dims.num_tokens,
        num_heads: dims.num_heads,
        num_kv_heads: dims.num_kv_heads,
        head_dim: dims.head_dim,
        block_size: dims.block_size,
        max_blocks_per_seq: dims.max_blocks_per_seq,
        num_blocks_total: dims.num_blocks_total,
        scale: dims.attn_scale,
        window_size_left,
    };
    decode.launch(
        decode_params,
        scratch.attn_out,
        scratch.q_fp8,
        scratch.k_cache,
        scratch.v_cache,
        meta.block_tables,
        meta.context_lens,
        scratch.fa3_workspace,
        scratch.q_scale_ptr,
        scratch.kv_scale_ptr,
        scratch.kv_scale_ptr,
        stream,
    )?;

    // 6. quantize attn_out -> fp8 per-token
    rvllm_fused::QuantizeFp8PerTokenLaunch {
        num_tokens: dims.num_tokens,
        dim: q_dim,
    }
    .launch(
        kernels.quantize_fp8_per_token,
        scratch.attn_out_fp8,
        scratch.attn_out_scale,
        scratch.attn_out,
        stream,
    )?;

    // 7. O proj -> delta buffer (bf16 output to avoid f16 overflow)
    #[cfg(feature = "cuda")]
    cublaslt.fp8_gemm_bf16(
        scratch.attn_out_fp8,
        weights.o_fp8,
        scratch.delta_f16,
        dims.num_tokens as i32,
        dims.hidden as i32,
        q_dim as i32,
        scratch.attn_out_scale,
        weights.o_scale,
        stream,
    )?;

    // 8. post_attention_layernorm on the DELTA (bf16 in-place)
    gemma4_launcher::RmsnormInplaceLaunch {
        num_tokens: dims.num_tokens,
        hidden: dims.hidden,
        eps: dims.rms_eps,
    }
    .launch(
        kernels.rmsnorm_inplace_bf16,
        scratch.delta_f16,
        weights.post_attn_norm_gamma,
        stream,
    )?;

    // 8b. residual(f16) += normed delta(bf16)
    gemma4_launcher::VectorAddF16Launch {
        n: dims.num_tokens * dims.hidden,
    }
    .launch(kernels.vector_add_bf16_to_f16, residual, scratch.delta_f16, stream)?;

    #[cfg(feature = "cuda")]
    probe!("after_step8_residual", residual, dims.hidden);

    // 9. pre_feedforward_layernorm -> FP8 quant
    FusedRmsnormFp8QuantLaunch {
        num_tokens: dims.num_tokens,
        hidden: dims.hidden,
        eps: dims.rms_eps,
    }
    .launch(
        kernels.fused_rmsnorm_fp8_quant,
        scratch.hidden_fp8,
        scratch.hidden_scale,
        residual,
        weights.pre_ff_norm_gamma,
        stream,
    )?;

    #[cfg(feature = "cuda")]
    probe_f32!("step9_hidden_scale", scratch.hidden_scale);
    #[cfg(feature = "cuda")]
    probe_f32!("step9_gate_up_wscale", weights.gate_up_scale);

    // 10. gate||up projection (cuBLASLt)
    #[cfg(feature = "cuda")]
    cublaslt.fp8_gemm(
        scratch.hidden_fp8,
        weights.gate_up_fp8,
        scratch.gate_up_out,
        dims.num_tokens as i32,
        (2 * dims.intermediate) as i32,
        dims.hidden as i32,
        scratch.hidden_scale,
        weights.gate_up_scale,
        stream,
    )?;

    #[cfg(feature = "cuda")]
    probe!("step10_gate_up_out", scratch.gate_up_out, dims.intermediate);

    // 11. GELU(tanh)(gate) * up -> FP8
    gemma4_launcher::FusedGeluMulFp8QuantLaunch {
        num_tokens: dims.num_tokens,
        intermediate: dims.intermediate,
    }
    .launch(
        kernels.fused_gelu_mul,
        scratch.mlp_out_fp8,
        scratch.mlp_out_scale,
        scratch.gate_up_out,
        stream,
    )?;

    #[cfg(feature = "cuda")]
    probe_f32!("step11_mlp_out_scale", scratch.mlp_out_scale);

    // 12. Down proj -> delta buffer (bf16 output, then convert to f16 with saturation)
    #[cfg(feature = "cuda")]
    cublaslt.fp8_gemm_bf16(
        scratch.mlp_out_fp8,
        weights.down_fp8,
        scratch.delta_f16,
        dims.num_tokens as i32,
        dims.hidden as i32,
        dims.intermediate as i32,
        scratch.mlp_out_scale,
        weights.down_scale,
        stream,
    )?;

    #[cfg(feature = "cuda")]
    probe!("step12_delta_bf16", scratch.delta_f16, dims.hidden);

    // 13. post_feedforward_layernorm on the DELTA (bf16 in-place)
    gemma4_launcher::RmsnormInplaceLaunch {
        num_tokens: dims.num_tokens,
        hidden: dims.hidden,
        eps: dims.rms_eps,
    }
    .launch(
        kernels.rmsnorm_inplace_bf16,
        scratch.delta_f16,
        weights.post_ff_norm_gamma,
        stream,
    )?;

    // 13b. residual(f16) += normed delta(bf16)
    gemma4_launcher::VectorAddF16Launch {
        n: dims.num_tokens * dims.hidden,
    }
    .launch(kernels.vector_add_bf16_to_f16, residual, scratch.delta_f16, stream)?;

    #[cfg(feature = "cuda")]
    probe!("after_step13_residual", residual, dims.hidden);

    // 14. residual *= layer_scalar (once per layer, after both sub-blocks)
    gemma4_launcher::ResidualScaleF16Launch {
        num_tokens: dims.num_tokens,
        hidden: dims.hidden,
    }
    .launch(
        kernels.residual_scale_f16,
        residual,
        weights.layer_scalar_ptr,
        stream,
    )?;

    #[cfg(feature = "cuda")]
    probe!("after_step14_residual", residual, dims.hidden);

    #[cfg(not(feature = "cuda"))]
    {
        let _ = (cublaslt, qkv_rows, _kv_dim);
    }
    Ok(())
}

pub unsafe fn logit_softcap(
    kernel: KernelFn,
    logits_ptr: u64,
    num_tokens: u32,
    vocab: u32,
    cap: f32,
    stream: u64,
) -> Result<()> {
    gemma4_launcher::LogitSoftcapLaunch {
        num_tokens,
        vocab,
        cap,
    }
    .launch(kernel, logits_ptr, stream)
}
