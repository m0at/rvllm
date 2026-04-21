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
//!  11. fused_act_mul_fp8_quant        (SiLU or GELU(tanh), then *up, quantize)
//!  12. fp8_gemm_residual             (Down proj, epilogue += residual)
//!
//! 12 launches per layer.

use rvllm_core::Result;
use rvllm_cutlass::{CublasLt, CutlassBackend, Fp8GemmPlan};
use rvllm_fused::{ArgmaxLaunch, FusedSiluMulFp8QuantLaunch};
use rvllm_kernels::KernelFn;
pub use rvllm_loader::{LayerAttnType, MlpActivation};

use rvllm_attention::{
    AttentionBackend, PagedDecodeFp8Launcher, PagedDecodeParams, PagedPrefillFp8Launcher,
    PagedPrefillParams,
};

/// Which phase is this `forward()` call executing? Decode = 1 Q token
/// per seq (FA3 paged_decode). Prefill = multi-Q per seq with causal
/// mask (FA3 paged_prefill). GEMMs scale with dims.num_tokens in both.
#[derive(Copy, Clone, Debug)]
pub enum LayerPhase {
    Decode,
    Prefill {
        cu_seqlens_q: u64, // [batch+1] i32 prefix sum on device
        max_seqlen_q: u32, // longest Q seq length
    },
}

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
    pub qkv_bias: u64, // 0 = no bias, else [q_dim + 2*kv_dim] f16
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
    pub q_out: u64,        // f16, QKV GEMM output (Q half)
    pub k_out: u64,        // f16, QKV GEMM output (K half)
    pub v_out: u64,        // f16, QKV GEMM output (V half)
    pub q_fp8: u64,        // fp8, post-rope Q consumed by FA3 (FP8 KV path)
    pub k_cache: u64,      // fp8 (1 byte/elem) paged K cache, this layer's base
    pub v_cache: u64,      // fp8 (1 byte/elem) paged V cache, this layer's base
    pub q_scale_ptr: u64,  // f32 per-tensor scale for Q (used by rope + FA3)
    pub kv_scale_ptr: u64, // f32 per-tensor scale for K and V (shared)
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
    pub fused_rmsnorm: KernelFn,
    pub fused_add_rmsnorm: KernelFn,
    pub fused_rope_cache_fp8kv: KernelFn,
    pub fused_silu_mul: KernelFn,
    pub fused_gelu_mul: Option<KernelFn>,
    pub mlp_activation: MlpActivation,
    pub quantize_fp8_per_token: KernelFn,
    pub add_bias_f16: KernelFn,
}

#[derive(Clone, Debug)]
pub struct LayerGemmPlans {
    /// Fused Q||K||V projection: N = (num_heads + 2*num_kv_heads) * head_dim.
    pub qkv: Fp8GemmPlan,
    pub o: Fp8GemmPlan, // residual-fused
    pub gate_up: Fp8GemmPlan,
    pub down: Fp8GemmPlan, // residual-fused
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn forward(
    dims: LayerDims,
    kernels: &LayerKernels,
    weights: &LayerWeights,
    scratch: &LayerScratch,
    meta: &MetadataPtrs,
    plans: &LayerGemmPlans,
    cutlass: &CutlassBackend,
    cublaslt: &CublasLt,
    fa3: &AttentionBackend,
    residual: u64,
    stream: u64,
    attn_type: LayerAttnType,
) -> Result<()> {
    forward_phase(
        dims,
        kernels,
        weights,
        scratch,
        meta,
        plans,
        cutlass,
        cublaslt,
        fa3,
        residual,
        stream,
        LayerPhase::Decode,
        attn_type,
    )
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn forward_phase(
    dims: LayerDims,
    kernels: &LayerKernels,
    weights: &LayerWeights,
    scratch: &LayerScratch,
    meta: &MetadataPtrs,
    _plans: &LayerGemmPlans,
    _cutlass: &CutlassBackend,
    cublaslt: &CublasLt,
    fa3: &AttentionBackend,
    residual: u64,
    stream: u64,
    phase: LayerPhase,
    attn_type: LayerAttnType,
) -> Result<()> {
    if attn_type == LayerAttnType::Linear {
        return Err(rvllm_core::RvllmError::Loader {
            err: rvllm_core::LoaderError::Corrupt {
                detail: "GDN linear attention not yet implemented -- this model uses Qwen3.5/3.6 hybrid attention".into(),
            },
            ctx: rvllm_core::LoaderCtx { path: std::path::PathBuf::new(), tensor: None },
            bt: std::backtrace::Backtrace::capture(),
        });
    }
    let q_dim = dims.num_heads * dims.head_dim;
    let _kv_dim = dims.num_kv_heads * dims.head_dim;

    // 1. rmsnorm(residual) + fp8 quant. The residual add was already
    //    done by the prior layer's down-proj cuBLASLt epilogue (beta=1).
    //    Using norm-only (not fused_add) avoids a double-add that caused
    //    residual to 2× per layer → f16 overflow → NaN by layer 16.
    rvllm_fused::FusedRmsnormFp8QuantLaunch {
        num_tokens: dims.num_tokens,
        hidden: dims.hidden,
        eps: dims.rms_eps,
    }
    .launch(
        kernels.fused_rmsnorm,
        scratch.hidden_fp8,
        scratch.hidden_scale,
        residual,
        weights.attn_norm_gamma,
        stream,
    )?;

    #[cfg(feature = "cuda")]
    let dbg_l0 = {
        static DONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        std::env::var("RVLLM_DBG_LAYER0").ok().as_deref() == Some("1")
            && !DONE.swap(true, std::sync::atomic::Ordering::Relaxed)
    };
    #[cfg(not(feature = "cuda"))]
    let dbg_l0 = false;

    #[cfg(feature = "cuda")]
    if dbg_l0 {
        unsafe {
            cudarc::driver::sys::cuStreamSynchronize(stream as _);
            let mut s = [0u16; 4];
            cudarc::driver::sys::cuMemcpyDtoH_v2(s.as_mut_ptr() as *mut _, residual, 8);
            let v: Vec<f32> = s.iter().map(|&x| crate::bring_up::f16_to_f32(x)).collect();
            eprintln!(
                "  [L0] input_residual: [{:.6}, {:.6}, {:.6}, {:.6}]",
                v[0], v[1], v[2], v[3]
            );
            let mut sc = [0f32; 1];
            cudarc::driver::sys::cuMemcpyDtoH_v2(
                sc.as_mut_ptr() as *mut _,
                scratch.hidden_scale,
                4,
            );
        }
    }

    // 2. Fused Q||K||V projection + f16 bias via cuBLASLt (one launch
    //    replaces cutlass_fp8_gemm + add_bias_f16). Output is packed
    //    [num_tokens, q_dim + 2*kv_dim] f16. q_out/k_out/v_out are byte
    //    offsets into the same buffer (set by bring_up).
    let qkv_n = dims.num_heads * dims.head_dim + 2 * dims.num_kv_heads * dims.head_dim;
    #[cfg(feature = "cuda")]
    if weights.qkv_bias != 0 {
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
    } else {
        cublaslt.fp8_gemm(
            scratch.hidden_fp8,
            weights.qkv_fp8,
            scratch.q_out,
            dims.num_tokens as i32,
            qkv_n as i32,
            dims.hidden as i32,
            scratch.hidden_scale,
            weights.qkv_scale,
            stream,
        )?;
    }
    // Suppress unused warnings when cuda feature is off.
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (cublaslt, kernels.add_bias_f16, _plans, qkv_n);
    }

    #[cfg(feature = "cuda")]
    if dbg_l0 {
        unsafe {
            cudarc::driver::sys::cuStreamSynchronize(stream as _);
            let mut s = [0u16; 4];
            cudarc::driver::sys::cuMemcpyDtoH_v2(s.as_mut_ptr() as *mut _, scratch.q_out, 8);
            let v: Vec<f32> = s.iter().map(|&x| crate::bring_up::f16_to_f32(x)).collect();
            eprintln!(
                "  [L0] after_qkv_gemm(Q[:4]): [{:.6}, {:.6}, {:.6}, {:.6}]",
                v[0], v[1], v[2], v[3]
            );
            let mut sc = [0f32; 1];
            cudarc::driver::sys::cuMemcpyDtoH_v2(
                sc.as_mut_ptr() as *mut _,
                scratch.hidden_scale,
                4,
            );
            eprintln!("  [L0] hidden_scale[0] = {:.6e}", sc[0]);
        }
    }

    // 5. RoPE q/k + FP8-quantize Q + write FP8 K/V into paged cache.
    rvllm_fused::FusedRopeCacheFp8KvLaunch {
        num_tokens: dims.num_tokens,
        num_heads: dims.num_heads,
        num_kv_heads: dims.num_kv_heads,
        head_dim: dims.head_dim,
    }
    .launch(
        kernels.fused_rope_cache_fp8kv,
        scratch.q_out,
        scratch.k_out,
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

    // 6. FA3 attention. Decode (1 Q/seq) vs Prefill (multi-Q/seq causal).
    match phase {
        LayerPhase::Decode => {
            let decode = PagedDecodeFp8Launcher::new(fa3);
            let decode_params = PagedDecodeParams {
                num_seqs: dims.num_tokens,
                num_heads: dims.num_heads,
                num_kv_heads: dims.num_kv_heads,
                head_dim: dims.head_dim,
                block_size: dims.block_size,
                max_blocks_per_seq: dims.max_blocks_per_seq,
                num_blocks_total: dims.num_blocks_total,
                scale: dims.attn_scale,
                window_size_left: -1,
            };
            // Llama/Qwen path: per-slot KV scales are not wired up for
            // this path yet — the Gemma 4 KV-scale precision refactor
            // only landed the scale buffers in `Gemma4LayerScratch`.
            // Pass 0 / 0 here so the build stays green; attempting to
            // run decode through this launcher will read from a null
            // per-slot scale pointer and crash. Safe until someone
            // brings a Llama/Qwen prefill path online.
            decode.launch(
                decode_params,
                scratch.attn_out,
                scratch.q_fp8,
                scratch.k_cache,
                scratch.v_cache,
                0, // k_scale_cache (per-slot, unused on Llama/Qwen)
                0, // v_scale_cache (per-slot, unused on Llama/Qwen)
                meta.block_tables,
                meta.context_lens,
                scratch.fa3_workspace,
                scratch.q_scale_ptr,
                stream,
            )?;
        }
        LayerPhase::Prefill {
            cu_seqlens_q,
            max_seqlen_q,
        } => {
            let prefill = PagedPrefillFp8Launcher::new(fa3);
            // num_tokens for prefill is total_q across the batch.
            let prefill_params = PagedPrefillParams {
                num_tokens: dims.num_tokens,
                // batch size: total_q / max_seqlen_q assuming uniform length
                num_seqs: if max_seqlen_q == 0 {
                    dims.num_tokens
                } else {
                    dims.num_tokens / max_seqlen_q
                },
                num_heads: dims.num_heads,
                num_kv_heads: dims.num_kv_heads,
                head_dim: dims.head_dim,
                block_size: dims.block_size,
                max_blocks_per_seq: dims.max_blocks_per_seq,
                num_blocks_total: dims.num_blocks_total,
                scale: dims.attn_scale,
                window_size_left: -1,
            };
            prefill.launch(
                prefill_params,
                scratch.attn_out,
                scratch.q_fp8,
                scratch.k_cache,
                scratch.v_cache,
                meta.block_tables,
                meta.context_lens,
                cu_seqlens_q,
                scratch.fa3_workspace,
                scratch.q_scale_ptr,
                scratch.kv_scale_ptr,
                scratch.kv_scale_ptr,
                max_seqlen_q,
                stream,
            )?;
        }
    }

    #[cfg(feature = "cuda")]
    if dbg_l0 {
        unsafe {
            cudarc::driver::sys::cuStreamSynchronize(stream as _);
            let mut s = [0u16; 4];
            cudarc::driver::sys::cuMemcpyDtoH_v2(s.as_mut_ptr() as *mut _, scratch.attn_out, 8);
            let v: Vec<f32> = s.iter().map(|&x| crate::bring_up::f16_to_f32(x)).collect();
            eprintln!(
                "  [L0] after_fa3(attn_out[:4]): [{:.6}, {:.6}, {:.6}, {:.6}]",
                v[0], v[1], v[2], v[3]
            );
        }
    }

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

    // 8. O proj with residual epilogue via cuBLASLt.
    #[cfg(feature = "cuda")]
    cublaslt.fp8_gemm_residual(
        scratch.attn_out_fp8,
        weights.o_fp8,
        residual,
        residual,
        dims.num_tokens as i32,
        dims.hidden as i32,
        q_dim as i32,
        scratch.attn_out_scale,
        weights.o_scale,
        stream,
    )?;

    #[cfg(feature = "cuda")]
    if dbg_l0 {
        unsafe {
            cudarc::driver::sys::cuStreamSynchronize(stream as _);
            let mut s = [0u16; 4];
            cudarc::driver::sys::cuMemcpyDtoH_v2(s.as_mut_ptr() as *mut _, residual, 8);
            let v: Vec<f32> = s.iter().map(|&x| crate::bring_up::f16_to_f32(x)).collect();
            eprintln!(
                "  [L0] after_oproj_residual[:4]: [{:.6}, {:.6}, {:.6}, {:.6}]",
                v[0], v[1], v[2], v[3]
            );
        }
    }

    // 9. pre-MLP norm + fp8 quant (norm-only, O-proj epilogue already
    //    added to residual).
    rvllm_fused::FusedRmsnormFp8QuantLaunch {
        num_tokens: dims.num_tokens,
        hidden: dims.hidden,
        eps: dims.rms_eps,
    }
    .launch(
        kernels.fused_rmsnorm,
        scratch.hidden_fp8,
        scratch.hidden_scale,
        residual,
        weights.mlp_norm_gamma,
        stream,
    )?;

    // 10. gate||up proj via cuBLASLt (no bias).
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

    // 11. activation(gate)*up -> fp8. SiLU for Llama/Qwen, GELU(tanh) for Gemma 4.
    let act_kernel = match kernels.mlp_activation {
        MlpActivation::SiLU => kernels.fused_silu_mul,
        MlpActivation::GELUTanh => kernels.fused_gelu_mul.expect(
            "model requires GELU(tanh) activation but fused_gelu_mul_fp8_quant.ptx not loaded",
        ),
    };
    FusedSiluMulFp8QuantLaunch {
        num_tokens: dims.num_tokens,
        intermediate: dims.intermediate,
    }
    .launch(
        act_kernel,
        scratch.mlp_out_fp8,
        scratch.mlp_out_scale,
        scratch.gate_up_out,
        stream,
    )?;

    // 12. Down proj with residual epilogue via cuBLASLt.
    #[cfg(feature = "cuda")]
    cublaslt.fp8_gemm_residual(
        scratch.mlp_out_fp8,
        weights.down_fp8,
        residual,
        residual,
        dims.num_tokens as i32,
        dims.hidden as i32,
        dims.intermediate as i32,
        scratch.mlp_out_scale,
        weights.down_scale,
        stream,
    )?;

    #[cfg(not(feature = "cuda"))]
    {
        let _ = (_cutlass, _plans, stream, _kv_dim);
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
