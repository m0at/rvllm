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
use rvllm_fused::gemma4_launcher;
use rvllm_fused::FusedRmsnormFp8QuantLaunch;
use rvllm_kernels::KernelFn;

use rvllm_attention::{
    Fa3Kernels, PagedDecodeFp8Launcher, PagedDecodeParams,
    PagedPrefillFp8Launcher, PagedPrefillParams,
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
    pub f16_kv: bool,
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
    pub qkv_f16: u64,          // 0 = use FP8, nonzero = use F16 GEMM
    pub o_f16: u64,
    pub gate_up_f16: u64,
    pub down_f16: u64,
    pub qkv_chscale: u64, // 0 = scalar scale, nonzero = per-channel f32 vec
    pub o_chscale: u64,
    pub gate_up_chscale: u64,
    pub down_chscale: u64,
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
    pub gemm_f32_tmp: u64,
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
    pub f32_to_bf16: KernelFn,
    pub f32_to_f16_sat: KernelFn,
    pub scale_cols_f32: KernelFn,
    pub compute_qkv_scales: KernelFn,
    pub fused_gelu_mul_f16: KernelFn,
    pub fused_rope_partial_f16kv: KernelFn,
    pub fused_norm_add_residual: KernelFn,
    pub fused_norm_add_residual_f16: KernelFn,
    pub fused_qkv_rmsnorm: KernelFn,
    pub scale_cols_f16: KernelFn,
}

#[derive(Copy, Clone, Debug)]
pub enum Gemma4Phase {
    Decode,
    Prefill {
        cu_seqlens_q: u64,
        max_seqlen_q: u32,
        num_seqs: u32,
    },
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn gemma4_forward(
    dims: Gemma4LayerDims,
    kernels: &Gemma4LayerKernels,
    weights: &Gemma4LayerWeightPtrs,
    scratch: &Gemma4LayerScratch,
    meta: &Gemma4MetadataPtrs,
    cublaslt: &CublasLt,
    cutlass: &CutlassLib,
    sliding_attention: &Fa3Kernels,
    global_attention: &Fa3Kernels,
    residual: u64,
    stream: u64,
) -> Result<()> {
    gemma4_forward_phase(dims, kernels, weights, scratch, meta, cublaslt, cutlass,
        sliding_attention, global_attention, residual, stream, Gemma4Phase::Decode)
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn gemma4_forward_phase(
    dims: Gemma4LayerDims,
    kernels: &Gemma4LayerKernels,
    weights: &Gemma4LayerWeightPtrs,
    scratch: &Gemma4LayerScratch,
    meta: &Gemma4MetadataPtrs,
    cublaslt: &CublasLt,
    cutlass: &CutlassLib,
    sliding_attention: &Fa3Kernels,
    global_attention: &Fa3Kernels,
    residual: u64,
    stream: u64,
    phase: Gemma4Phase,
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
                    if v > amax {
                        amax = v;
                    }
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
    #[cfg(feature = "cuda")]
    probe_f32!("step1_hidden_scale", scratch.hidden_scale);
    #[cfg(feature = "cuda")]
    probe_f32!("step1_qkv_wscale", weights.qkv_scale);
    #[cfg(feature = "cuda")]
    {
        if dbg_layer >= 0 {
            cudarc::driver::sys::cuStreamSynchronize(stream as _);
            let mut hs = [0.0f32; 1];
            let mut ws = [0.0f32; 1];
            cudarc::driver::sys::cuMemcpyDtoH_v2(
                hs.as_mut_ptr() as *mut _,
                scratch.hidden_scale,
                4,
            );
            cudarc::driver::sys::cuMemcpyDtoH_v2(ws.as_mut_ptr() as *mut _, weights.qkv_scale, 4);
            eprintln!(
                "    [L{} step1_scale_product] hidden*qkv = {:.6e} * {:.6e} = {:.6e}",
                dbg_layer,
                hs[0],
                ws[0],
                hs[0] * ws[0]
            );
        }
    }

    #[cfg(feature = "cuda")]
    {
        if dbg_layer >= 0 {
            cudarc::driver::sys::cuStreamSynchronize(stream as _);
            let mut wb = [0u8; 8];
            cudarc::driver::sys::cuMemcpyDtoH_v2(wb.as_mut_ptr() as *mut _, weights.qkv_fp8, 8);
            eprintln!("    [L{} step2_qkv_fp8_bytes] first8={:?}", dbg_layer, wb);
            let mut hb = [0u8; 8];
            cudarc::driver::sys::cuMemcpyDtoH_v2(hb.as_mut_ptr() as *mut _, scratch.hidden_fp8, 8);
            eprintln!(
                "    [L{} step2_hidden_fp8_bytes] first8={:?}",
                dbg_layer, hb
            );
        }
    }

    // 2. Q||K||V projection
    #[cfg(feature = "cuda")]
    if weights.qkv_f16 != 0 {
        // F16 path: copy residual to delta_f16 scratch, apply rmsnorm in-place, use as GEMM input
        cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
            scratch.delta_f16,
            residual,
            (dims.num_tokens * dims.hidden * 2) as _,
            stream as _,
        );
        gemma4_launcher::RmsnormInplaceLaunch {
            num_tokens: dims.num_tokens,
            hidden: dims.hidden,
            eps: dims.rms_eps,
        }
        .launch(
            kernels.fused_rmsnorm,
            scratch.delta_f16,
            weights.attn_norm_gamma,
            stream,
        )?;
        cublaslt.f16_gemm_f32(
            scratch.delta_f16,
            weights.qkv_f16,
            scratch.gemm_f32_tmp,
            dims.num_tokens as i32,
            qkv_rows as i32,
            dims.hidden as i32,
            stream,
        )?;
        gemma4_launcher::Bf16ToF16SatLaunch {
            n: dims.num_tokens * qkv_rows,
        }
        .launch(
            kernels.f32_to_f16_sat,
            scratch.q_out,
            scratch.gemm_f32_tmp,
            stream,
        )?;
    } else if weights.qkv_chscale != 0 {
        if cutlass.fp8_gemm_channelscale.is_some() {
            cutlass.launch_fp8_gemm_channelscale(
                scratch.q_out, scratch.hidden_fp8, weights.qkv_fp8,
                scratch.hidden_scale, weights.qkv_chscale,
                dims.num_tokens as i32, qkv_rows as i32, dims.hidden as i32,
                scratch.cutlass_workspace, scratch.cutlass_workspace_bytes,
                stream,
            )?;
        } else {
            cublaslt.fp8_gemm_f32(
                scratch.hidden_fp8, weights.qkv_fp8, scratch.gemm_f32_tmp,
                dims.num_tokens as i32, qkv_rows as i32, dims.hidden as i32,
                scratch.hidden_scale, weights.qkv_scale,
                stream,
            )?;
            launch_scale_cols_f32(
                kernels.scale_cols_f32,
                scratch.gemm_f32_tmp,
                weights.qkv_chscale,
                dims.num_tokens,
                qkv_rows,
                stream,
            )?;
            gemma4_launcher::Bf16ToF16SatLaunch {
                n: dims.num_tokens * qkv_rows,
            }
            .launch(
                kernels.f32_to_f16_sat,
                scratch.q_out,
                scratch.gemm_f32_tmp,
                stream,
            )?;
        }
    } else {
        cublaslt.fp8_gemm(
            scratch.hidden_fp8, weights.qkv_fp8, scratch.q_out,
            dims.num_tokens as i32, qkv_rows as i32, dims.hidden as i32,
            scratch.hidden_scale, weights.qkv_scale, stream,
        )?;
    }

    #[cfg(feature = "cuda")]
    probe!("step2_q_proj", scratch.q_out, dims.hidden);
    #[cfg(feature = "cuda")]
    {
        if dbg_layer >= 0 {
            let k_offset = q_dim as u64 * 2;
            let v_offset = (q_dim + dims.num_kv_heads * dims.head_dim) as u64 * 2;
            cudarc::driver::sys::cuStreamSynchronize(stream as _);
            let mut sk = [0u16; 4];
            cudarc::driver::sys::cuMemcpyDtoH_v2(
                sk.as_mut_ptr() as *mut _,
                scratch.q_out + k_offset,
                8,
            );
            let kv: Vec<f32> = sk.iter().map(|&x| crate::bring_up::f16_to_f32(x)).collect();
            eprintln!("    [L{} step2_k_proj] first4={:.4?}", dbg_layer, kv);
            let mut sv = [0u16; 4];
            cudarc::driver::sys::cuMemcpyDtoH_v2(
                sv.as_mut_ptr() as *mut _,
                scratch.q_out + v_offset,
                8,
            );
            let vv: Vec<f32> = sv.iter().map(|&x| crate::bring_up::f16_to_f32(x)).collect();
            eprintln!("    [L{} step2_v_proj] first4={:.4?}", dbg_layer, vv);
        }
    }

    // 2b+3. Fused QKV-norm: Q/K with learned gamma, V parameter-free
    gemma4_launcher::FusedQkvRmsnormLaunch {
        num_tokens: dims.num_tokens,
        num_heads: dims.num_heads,
        num_kv_heads: dims.num_kv_heads,
        head_dim: dims.head_dim,
        eps: dims.rms_eps,
    }
    .launch(
        kernels.fused_qkv_rmsnorm,
        scratch.q_out,
        scratch.k_out,
        scratch.v_out,
        scratch.q_normed,
        scratch.k_normed,
        weights.q_norm_gamma,
        weights.k_norm_gamma,
        stream,
    )?;

    #[cfg(feature = "cuda")]
    probe!("step3_q_norm", scratch.q_normed, dims.hidden);
    #[cfg(feature = "cuda")]
    probe!("step3_k_norm", scratch.k_normed, dims.hidden);

    // 4-5. RoPE + attention (F16 or FP8 KV cache, decode or prefill)
    let attention = match dims.layer_type {
        Gemma4LayerType::SlidingAttention => sliding_attention,
        Gemma4LayerType::GlobalAttention => global_attention,
    };
    let window_size_left: i32 = match dims.layer_type {
        Gemma4LayerType::SlidingAttention => (dims.sliding_window as i32) - 1,
        Gemma4LayerType::GlobalAttention => -1,
    };

    #[cfg(feature = "cuda")]
    match phase {
        Gemma4Phase::Decode => {
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
            if dims.f16_kv {
                // F16 KV cache path: RoPE outputs F16 Q and F16 KV cache
                rope_f16kv(dims, kernels, scratch, meta, stream)?;
                let decode = rvllm_attention::PagedDecodeLauncher::new(attention);
                decode.launch(
                    decode_params,
                    scratch.attn_out,
                    scratch.q_normed,
                    scratch.k_cache,
                    scratch.v_cache,
                    meta.block_tables,
                    meta.context_lens,
                    scratch.fa3_workspace,
                    stream,
                )?;
            } else {
                rope_fp8kv(dims, kernels, scratch, meta, stream)?;
                let decode = PagedDecodeFp8Launcher::new(attention);
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
            }
        }
        Gemma4Phase::Prefill { cu_seqlens_q, max_seqlen_q, num_seqs } => {
            // Prefill always uses FP8 KV path (no F16 prefill kernel available)
            rope_fp8kv(dims, kernels, scratch, meta, stream)?;
            let prefill_params = PagedPrefillParams {
                num_tokens: dims.num_tokens,
                num_seqs,
                num_heads: dims.num_heads,
                num_kv_heads: dims.num_kv_heads,
                head_dim: dims.head_dim,
                block_size: dims.block_size,
                max_blocks_per_seq: dims.max_blocks_per_seq,
                num_blocks_total: dims.num_blocks_total,
                scale: dims.attn_scale,
                window_size_left,
            };
            // SM90 decode + SM89 prefill: the SM90 Varlen kernel exceeds H100's 232KB
            // smem limit at hdim>=256. SM89 prefill works at all head dims.
            // Prefill runs once per request; decode (SM90, CUDA graphed) is the hot path.
            let prefill = PagedPrefillFp8Launcher::new(global_attention);
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
    #[cfg(not(feature = "cuda"))]
    let _ = phase;

    #[cfg(feature = "cuda")]
    probe!("step5_attn_out", scratch.attn_out, q_dim);

    // 6. quantize attn_out -> fp8 per-token (skip when F16 KV + F16 O-proj)
    if !dims.f16_kv || weights.o_f16 == 0 {
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
    }

    // 7-8. O proj + channelscale + post_attn norm + residual add
    #[cfg(feature = "cuda")]
    if weights.o_f16 != 0 {
        cublaslt.f16_gemm_f32(
            scratch.attn_out, weights.o_f16, scratch.gemm_f32_tmp,
            dims.num_tokens as i32, dims.hidden as i32, q_dim as i32, stream,
        )?;
        gemma4_launcher::FusedNormAddResidualLaunch {
            num_tokens: dims.num_tokens, hidden: dims.hidden, eps: dims.rms_eps,
        }.launch(kernels.fused_norm_add_residual, scratch.gemm_f32_tmp,
            weights.post_attn_norm_gamma, residual, 0, stream)?;
    } else if weights.o_chscale != 0 {
        cublaslt.fp8_gemm_f32(
            scratch.attn_out_fp8, weights.o_fp8, scratch.gemm_f32_tmp,
            dims.num_tokens as i32, dims.hidden as i32, q_dim as i32,
            scratch.attn_out_scale, weights.o_scale, stream,
        )?;
        gemma4_launcher::FusedNormAddResidualF16Launch {
            num_tokens: dims.num_tokens, hidden: dims.hidden, eps: dims.rms_eps,
        }.launch(kernels.fused_norm_add_residual_f16, scratch.gemm_f32_tmp,
            weights.o_chscale, weights.post_attn_norm_gamma, residual, 0, stream)?;
    } else {
        cublaslt.fp8_gemm_f32(
            scratch.attn_out_fp8, weights.o_fp8, scratch.gemm_f32_tmp,
            dims.num_tokens as i32, dims.hidden as i32, q_dim as i32,
            scratch.attn_out_scale, weights.o_scale, stream,
        )?;
        gemma4_launcher::FusedNormAddResidualLaunch {
            num_tokens: dims.num_tokens, hidden: dims.hidden, eps: dims.rms_eps,
        }.launch(kernels.fused_norm_add_residual, scratch.gemm_f32_tmp,
            weights.post_attn_norm_gamma, residual, 0, stream)?;
    }

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

    // 10. gate||up projection
    #[cfg(feature = "cuda")]
    if weights.gate_up_f16 != 0 {
        // F16 path: norm residual into gate_up_out scratch, then F16 GEMM
        cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
            scratch.gate_up_out,
            residual,
            (dims.num_tokens * dims.hidden * 2) as _,
            stream as _,
        );
        gemma4_launcher::RmsnormInplaceLaunch {
            num_tokens: dims.num_tokens,
            hidden: dims.hidden,
            eps: dims.rms_eps,
        }
        .launch(
            kernels.fused_rmsnorm,
            scratch.gate_up_out,
            weights.pre_ff_norm_gamma,
            stream,
        )?;
        cublaslt.f16_gemm_f32(
            scratch.gate_up_out,
            weights.gate_up_f16,
            scratch.gemm_f32_tmp,
            dims.num_tokens as i32,
            (2 * dims.intermediate) as i32,
            dims.hidden as i32,
            stream,
        )?;
        gemma4_launcher::Bf16ToF16SatLaunch {
            n: dims.num_tokens * 2 * dims.intermediate,
        }
        .launch(
            kernels.f32_to_f16_sat,
            scratch.gate_up_out,
            scratch.gemm_f32_tmp,
            stream,
        )?;
    } else if weights.gate_up_chscale != 0 {
        if cutlass.fp8_gemm_channelscale.is_some() {
            cutlass.launch_fp8_gemm_channelscale(
                scratch.gate_up_out, scratch.hidden_fp8, weights.gate_up_fp8,
                scratch.hidden_scale, weights.gate_up_chscale,
                dims.num_tokens as i32, (2 * dims.intermediate) as i32, dims.hidden as i32,
                scratch.cutlass_workspace, scratch.cutlass_workspace_bytes,
                stream,
            )?;
        } else {
            cublaslt.fp8_gemm_f32(
                scratch.hidden_fp8, weights.gate_up_fp8, scratch.gemm_f32_tmp,
                dims.num_tokens as i32, (2 * dims.intermediate) as i32, dims.hidden as i32,
                scratch.hidden_scale, weights.gate_up_scale,
                stream,
            )?;
            launch_scale_cols_f32(
                kernels.scale_cols_f32,
                scratch.gemm_f32_tmp,
                weights.gate_up_chscale,
                dims.num_tokens,
                2 * dims.intermediate,
                stream,
            )?;
            gemma4_launcher::Bf16ToF16SatLaunch {
                n: dims.num_tokens * 2 * dims.intermediate,
            }
            .launch(
                kernels.f32_to_f16_sat,
                scratch.gate_up_out,
                scratch.gemm_f32_tmp,
                stream,
            )?;
        }
    } else {
        cublaslt.fp8_gemm(
            scratch.hidden_fp8, weights.gate_up_fp8, scratch.gate_up_out,
            dims.num_tokens as i32, (2 * dims.intermediate) as i32, dims.hidden as i32,
            scratch.hidden_scale, weights.gate_up_scale, stream,
        )?;
    }

    #[cfg(feature = "cuda")]
    probe!("step10_gate_up_out", scratch.gate_up_out, dims.intermediate);

    // 11-12. GELU*up + down_proj
    #[cfg(feature = "cuda")]
    if weights.down_f16 != 0 {
        // F16 path: GELU output to separate buffer (can't alias gate_up_out)
        {
            let mut out = scratch.gate_up_fp8; // use gate_up_fp8 scratch as f16 gelu output
            let mut inp = scratch.gate_up_out;
            let mut inter = dims.intermediate as i32;
            let args = [
                (&mut out) as *mut u64 as *mut core::ffi::c_void,
                (&mut inp) as *mut u64 as *mut core::ffi::c_void,
                (&mut inter) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block = (dims.intermediate.min(1024), 1, 1);
            let grid = (dims.num_tokens, 1, 1);
            rvllm_fused::launch_raw(kernels.fused_gelu_mul_f16, grid, block, 0, stream, &args)?;
        }
        // F16 GEMM for down_proj (reads from gate_up_fp8 where GELU output was stored)
        cublaslt.f16_gemm_f32(
            scratch.gate_up_fp8,
            weights.down_f16,
            scratch.gemm_f32_tmp,
            dims.num_tokens as i32,
            dims.hidden as i32,
            dims.intermediate as i32,
            stream,
        )?;
    } else {
        // FP8 path
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
        if weights.down_chscale != 0 {
            cublaslt.fp8_gemm_f32(
                scratch.mlp_out_fp8, weights.down_fp8, scratch.gemm_f32_tmp,
                dims.num_tokens as i32, dims.hidden as i32, dims.intermediate as i32,
                scratch.mlp_out_scale, weights.down_scale, stream,
            )?;
            gemma4_launcher::FusedNormAddResidualF16Launch {
                num_tokens: dims.num_tokens, hidden: dims.hidden, eps: dims.rms_eps,
            }.launch(kernels.fused_norm_add_residual_f16, scratch.gemm_f32_tmp,
                weights.down_chscale, weights.post_ff_norm_gamma, residual,
                weights.layer_scalar_ptr, stream)?;
        } else {
            cublaslt.fp8_gemm_f32(
                scratch.mlp_out_fp8, weights.down_fp8, scratch.gemm_f32_tmp,
                dims.num_tokens as i32, dims.hidden as i32, dims.intermediate as i32,
                scratch.mlp_out_scale, weights.down_scale, stream,
            )?;
            gemma4_launcher::FusedNormAddResidualLaunch {
                num_tokens: dims.num_tokens, hidden: dims.hidden, eps: dims.rms_eps,
            }.launch(kernels.fused_norm_add_residual, scratch.gemm_f32_tmp,
                weights.post_ff_norm_gamma, residual,
                weights.layer_scalar_ptr, stream)?;
        }
    }

    #[cfg(feature = "cuda")]
    probe!("after_step14_residual", residual, dims.hidden);

    #[cfg(not(feature = "cuda"))]
    {
        let _ = (cublaslt, qkv_rows, _kv_dim);
    }
    Ok(())
}

#[cfg(feature = "cuda")]
unsafe fn launch_scale_cols_f32(
    kernel: KernelFn,
    data: u64,
    scale: u64,
    m: u32,
    n: u32,
    stream: u64,
) -> Result<()> {
    let total = m * n;
    let mut data = data;
    let mut scale = scale;
    let mut m_i = m as i32;
    let mut n_i = n as i32;
    let args = [
        (&mut data) as *mut u64 as *mut core::ffi::c_void,
        (&mut scale) as *mut u64 as *mut core::ffi::c_void,
        (&mut m_i) as *mut i32 as *mut core::ffi::c_void,
        (&mut n_i) as *mut i32 as *mut core::ffi::c_void,
    ];
    let block = (256u32, 1, 1);
    let grid = ((total + 255) / 256, 1, 1);
    rvllm_fused::launch_raw(kernel, grid, block, 0, stream, &args)
}

#[cfg(feature = "cuda")]
unsafe fn launch_scale_cols_f16(
    kernel: KernelFn,
    data: u64,
    scale: u64,
    m: u32,
    n: u32,
    stream: u64,
) -> Result<()> {
    let mut data = data;
    let mut scale = scale;
    let mut m_i = m as i32;
    let mut n_i = n as i32;
    let args = [
        (&mut data) as *mut u64 as *mut core::ffi::c_void,
        (&mut scale) as *mut u64 as *mut core::ffi::c_void,
        (&mut m_i) as *mut i32 as *mut core::ffi::c_void,
        (&mut n_i) as *mut i32 as *mut core::ffi::c_void,
    ];
    let block = (256u32, 1, 1);
    let grid = (((n + 255) / 256), m, 1);
    rvllm_fused::launch_raw(kernel, grid, block, 0, stream, &args)
}

#[cfg(feature = "cuda")]
unsafe fn fp8_gemm_f32_with_channelscale_fallback(
    cublaslt: &CublasLt,
    scale_cols_kernel: KernelFn,
    a_fp8: u64,
    b_fp8: u64,
    d_f32: u64,
    m: i32,
    n: i32,
    k: i32,
    a_scale: u64,
    b_scale: u64,
    b_channelscale: u64,
    stream: u64,
) -> Result<()> {
    match cublaslt.fp8_gemm_f32_channelscale(
        a_fp8,
        b_fp8,
        d_f32,
        m,
        n,
        k,
        a_scale,
        b_channelscale,
        stream,
    ) {
        Ok(()) => Ok(()),
        Err(_) => {
            cublaslt.fp8_gemm_f32(a_fp8, b_fp8, d_f32, m, n, k, a_scale, b_scale, stream)?;
            launch_scale_cols_f32(
                scale_cols_kernel,
                d_f32,
                b_channelscale,
                m as u32,
                n as u32,
                stream,
            )
        }
    }
}

#[cfg(feature = "cuda")]
unsafe fn rope_f16kv(
    dims: Gemma4LayerDims,
    kernels: &Gemma4LayerKernels,
    scratch: &Gemma4LayerScratch,
    meta: &Gemma4MetadataPtrs,
    stream: u64,
) -> Result<()> {
    let mut q_in = scratch.q_normed;
    let mut k_in = scratch.k_normed;
    let mut v_in = scratch.v_out;
    let mut q_out = scratch.q_normed;
    let mut k_cache = scratch.k_cache;
    let mut v_cache = scratch.v_cache;
    let mut cos = meta.cos;
    let mut sin = meta.sin;
    let mut positions = meta.positions;
    let mut slot_mapping = meta.slot_mapping;
    let mut nt = dims.num_tokens as i32;
    let mut nh = dims.num_heads as i32;
    let mut nkvh = dims.num_kv_heads as i32;
    let mut hd = dims.head_dim as i32;
    let mut rd = dims.rotary_dim as i32;
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
    let max_heads = dims.num_heads.max(dims.num_kv_heads);
    let grid = (dims.num_tokens, max_heads, 1);
    let block = ((dims.head_dim / 2).max(32), 1, 1);
    rvllm_fused::launch_raw(kernels.fused_rope_partial_f16kv, grid, block, 0, stream, &args)
}

#[cfg(feature = "cuda")]
unsafe fn rope_fp8kv(
    dims: Gemma4LayerDims,
    kernels: &Gemma4LayerKernels,
    scratch: &Gemma4LayerScratch,
    meta: &Gemma4MetadataPtrs,
    stream: u64,
) -> Result<()> {
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
    )
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
