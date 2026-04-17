//! Layer forward as a pure function per spec 09.
//!
//! The kernel sequence for one Llama-style decoder layer (GQA + FP8 W +
//! fused residual add + rmsnorm + silu-mul), using v2's kernel APIs:
//!   1. fused_add_rmsnorm_fp8_quant   (attn_norm)
//!   2..4. fp8_gemm x3                (Q, K, V projections separately)
//!   5. fused_rope_cache_f16          (RoPE q/k + write KV pages)
//!   6. paged_decode                  (FA3 SM90)
//!   7. quantize_fp8_per_token        (attn_out -> fp8)
//!   8. fp8_gemm_residual             (O proj, epilogue += residual)
//!   9. fused_add_rmsnorm_fp8_quant   (mlp_norm)
//!  10. fp8_gemm                      (gate||up fused proj)
//!  11. fused_silu_mul_fp8_quant      (SiLU(gate)*up, quantize)
//!  12. fp8_gemm_residual             (Down proj, epilogue += residual)
//!
//! 12 launches per layer.

use rvllm_core::Result;
use rvllm_cutlass::{CublasLt, CutlassLib, Fp8GemmPlan};
use rvllm_fused::{
    ArgmaxLaunch, FusedAddRmsnormFp8QuantLaunch, FusedRopeKvWriteLaunch,
    FusedSiluMulFp8QuantLaunch,
};
use rvllm_kernels::KernelFn;

use rvllm_attention::{Fa3Kernels, PagedDecodeLauncher, PagedDecodeParams};

#[derive(Copy, Clone, Debug)]
pub struct LayerDims {
    pub num_tokens: u32,
    pub hidden: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub intermediate: u32,
    pub block_size: u32,
    pub max_blocks_per_seq: u32,
    pub num_blocks_total: u32,
    pub attn_scale: f32,
    pub rms_eps: f32,
}

/// Per-layer device pointers. `qkv_fp8` is a packed (q_rows + 2*kv_rows)
/// x hidden weight matrix; we issue 3 sub-GEMMs over it.
#[derive(Copy, Clone, Debug)]
pub struct LayerWeights {
    pub attn_norm_gamma: u64,
    pub qkv_fp8: u64,
    pub qkv_scale: u64,
    pub qkv_bias: u64, // [q_dim + 2*kv_dim] f16
    pub o_fp8: u64,
    pub o_scale: u64,
    pub mlp_norm_gamma: u64,
    pub gate_up_fp8: u64,
    pub gate_up_scale: u64,
    pub down_fp8: u64,
    pub down_scale: u64,
}

#[derive(Copy, Clone, Debug)]
pub struct LayerScratch {
    pub hidden_fp8: u64,
    pub hidden_scale: u64,
    pub q_out: u64,
    pub k_out: u64,
    pub v_out: u64,
    pub k_cache: u64,
    pub v_cache: u64,
    pub attn_out: u64,
    pub attn_out_fp8: u64,
    pub attn_out_scale: u64,
    pub gate_up_out: u64,
    pub gate_up_fp8: u64,
    pub gate_up_scale: u64,
    pub mlp_out_fp8: u64,
    pub mlp_out_scale: u64,
    pub cutlass_workspace: u64,
    pub cutlass_workspace_bytes: usize,
    pub fa3_workspace: u64,
}

#[derive(Copy, Clone, Debug)]
pub struct MetadataPtrs {
    pub positions: u64,
    pub slot_mapping: u64,
    pub cos: u64,
    pub sin: u64,
    pub block_tables: u64,
    pub context_lens: u64,
}

#[derive(Copy, Clone, Debug)]
pub struct LayerKernels {
    pub fused_add_rmsnorm: KernelFn,
    pub fused_rope_cache: KernelFn,
    pub fused_silu_mul: KernelFn,
    pub quantize_fp8_per_token: KernelFn,
    pub add_bias_f16: KernelFn,
}

#[derive(Clone, Debug)]
pub struct LayerGemmPlans {
    /// Fused Q||K||V projection: N = (num_heads + 2*num_kv_heads) * head_dim.
    pub qkv: Fp8GemmPlan,
    pub o: Fp8GemmPlan,        // residual-fused
    pub gate_up: Fp8GemmPlan,
    pub down: Fp8GemmPlan,     // residual-fused
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn forward(
    dims: LayerDims,
    kernels: &LayerKernels,
    weights: &LayerWeights,
    scratch: &LayerScratch,
    meta: &MetadataPtrs,
    plans: &LayerGemmPlans,
    cutlass: &CutlassLib,
    cublaslt: &CublasLt,
    fa3: &Fa3Kernels,
    residual: u64,
    stream: u64,
) -> Result<()> {
    let q_dim = dims.num_heads * dims.head_dim;
    let kv_dim = dims.num_kv_heads * dims.head_dim;

    // 1. rmsnorm(residual) + fp8 quant, residual updated in-place.
    FusedAddRmsnormFp8QuantLaunch {
        num_tokens: dims.num_tokens,
        hidden: dims.hidden,
        eps: dims.rms_eps,
    }
    .launch(
        kernels.fused_add_rmsnorm,
        scratch.hidden_fp8,
        scratch.hidden_scale,
        residual,
        residual,
        residual,
        weights.attn_norm_gamma,
        stream,
    )?;

    // 2. Fused Q||K||V projection + f16 bias via cuBLASLt (one launch
    //    replaces cutlass_fp8_gemm + add_bias_f16). Output is packed
    //    [num_tokens, q_dim + 2*kv_dim] f16. q_out/k_out/v_out are byte
    //    offsets into the same buffer (set by bring_up).
    let qkv_n = dims.num_heads * dims.head_dim + 2 * dims.num_kv_heads * dims.head_dim;
    #[cfg(feature = "cuda")]
    cublaslt.fp8_gemm_bias(
        scratch.hidden_fp8,
        weights.qkv_fp8,
        weights.qkv_bias,
        scratch.q_out,
        dims.num_tokens as i32,
        qkv_n as i32,
        dims.hidden as i32,
        scratch.hidden_scale,
        weights.qkv_scale,
        stream,
    )?;
    // Suppress unused warnings when cuda feature is off.
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (cublaslt, kernels.add_bias_f16, plans, qkv_n);
    }

    // 5. RoPE q/k in place, write k/v into paged cache.
    FusedRopeKvWriteLaunch {
        num_tokens: dims.num_tokens,
        num_heads: dims.num_heads,
        num_kv_heads: dims.num_kv_heads,
        head_dim: dims.head_dim,
    }
    .launch(
        kernels.fused_rope_cache,
        scratch.q_out,
        scratch.k_out,
        scratch.v_out,
        scratch.k_cache,
        scratch.v_cache,
        meta.cos,
        meta.sin,
        meta.positions,
        meta.slot_mapping,
        stream,
    )?;

    // 6. FA3 paged decode.
    let decode = PagedDecodeLauncher::new(fa3);
    let decode_params = PagedDecodeParams {
        num_seqs: dims.num_tokens,
        num_heads: dims.num_heads,
        num_kv_heads: dims.num_kv_heads,
        head_dim: dims.head_dim,
        block_size: dims.block_size,
        max_blocks_per_seq: dims.max_blocks_per_seq,
        num_blocks_total: dims.num_blocks_total,
        scale: dims.attn_scale,
    };
    decode.launch(
        decode_params,
        scratch.attn_out,
        scratch.q_out,
        scratch.k_cache,
        scratch.v_cache,
        meta.block_tables,
        meta.context_lens,
        scratch.fa3_workspace,
        stream,
    )?;

    // 7. quantize attn_out -> fp8 (per-token).
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

    // 8. O proj residual epilogue.
    #[cfg(feature = "cuda")]
    cutlass.launch_fp8_gemm_residual(
        &plans.o,
        residual,
        scratch.attn_out_fp8,
        weights.o_fp8,
        scratch.attn_out_scale,
        weights.o_scale,
        residual,
        scratch.cutlass_workspace,
        scratch.cutlass_workspace_bytes,
        stream,
    )?;

    // 9. pre-MLP norm + fp8 quant.
    FusedAddRmsnormFp8QuantLaunch {
        num_tokens: dims.num_tokens,
        hidden: dims.hidden,
        eps: dims.rms_eps,
    }
    .launch(
        kernels.fused_add_rmsnorm,
        scratch.hidden_fp8,
        scratch.hidden_scale,
        residual,
        residual,
        residual,
        weights.mlp_norm_gamma,
        stream,
    )?;

    // 10. gate||up proj.
    #[cfg(feature = "cuda")]
    cutlass.launch_fp8_gemm(
        &plans.gate_up,
        scratch.gate_up_out,
        scratch.hidden_fp8,
        weights.gate_up_fp8,
        scratch.hidden_scale,
        weights.gate_up_scale,
        scratch.cutlass_workspace,
        scratch.cutlass_workspace_bytes,
        stream,
    )?;

    // 11. silu*mul -> fp8.
    FusedSiluMulFp8QuantLaunch {
        num_tokens: dims.num_tokens,
        intermediate: dims.intermediate,
    }
    .launch(
        kernels.fused_silu_mul,
        scratch.mlp_out_fp8,
        scratch.mlp_out_scale,
        scratch.gate_up_out,
        stream,
    )?;

    // 12. Down proj residual epilogue.
    #[cfg(feature = "cuda")]
    cutlass.launch_fp8_gemm_residual(
        &plans.down,
        residual,
        scratch.mlp_out_fp8,
        weights.down_fp8,
        scratch.mlp_out_scale,
        weights.down_scale,
        residual,
        scratch.cutlass_workspace,
        scratch.cutlass_workspace_bytes,
        stream,
    )?;

    #[cfg(not(feature = "cuda"))]
    {
        let _ = (cutlass, plans, stream, kv_dim);
    }
    Ok(())
}

pub unsafe fn argmax_sample(
    num_tokens: u32,
    vocab: u32,
    kernel: KernelFn,
    logits_ptr: u64,
    out_ptr: u64,
    stream: u64,
) -> Result<()> {
    ArgmaxLaunch { num_tokens, vocab }.launch(kernel, logits_ptr, out_ptr, stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_dims_are_plausible() {
        let d = LayerDims {
            num_tokens: 128,
            hidden: 3584,
            num_heads: 28,
            num_kv_heads: 4,
            head_dim: 128,
            intermediate: 18944,
            block_size: 64,
            max_blocks_per_seq: 33,
            num_blocks_total: 1024,
            attn_scale: 1.0 / 11.313708,
            rms_eps: 1e-6,
        };
        assert_eq!(d.num_heads * d.head_dim, 3584);
        assert_eq!(d.num_kv_heads * d.head_dim, 512);
    }
}
