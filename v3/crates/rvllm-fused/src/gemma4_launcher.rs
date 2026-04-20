//! Gemma 4 fused kernel launchers.
//!
//! New kernels not in the Llama/Qwen baseline:
//!   - FusedGeluMulFp8Quant:  GELU(tanh)(gate) * up -> FP8
//!   - FusedQkRmsnorm:        per-head RMSNorm on Q and K
//!   - FusedRopePartialFp8Kv: partial RoPE (rotary_dim < head_dim)
//!   - RmsnormInplace:        RMSNorm applied in-place (no FP8 output)
//!   - LogitSoftcap:          30 * tanh(logits / 30)

use rvllm_core::Result;
use rvllm_kernels::KernelFn;

use crate::launch_raw::launch_raw;
use crate::launcher::require_multiple;

fn invalid(field: &'static str, reason: &'static str) -> rvllm_core::RvllmError {
    rvllm_core::RvllmError::Sampling {
        err: rvllm_core::SamplingError::InvalidParams {
            reason: format!("{field}: {reason}"),
        },
        ctx: rvllm_core::SampleCtx {
            op: "validate",
            stream: 0,
        },
    }
}

// ---------------------------------------------------------------------------
// fused_gelu_mul_fp8_quant
// ---------------------------------------------------------------------------

pub struct FusedGeluMulFp8QuantLaunch {
    pub num_tokens: u32,
    pub intermediate: u32,
}

impl FusedGeluMulFp8QuantLaunch {
    pub fn validate(&self) -> Result<()> {
        require_multiple(self.intermediate as usize, 8, "intermediate")?;
        if self.num_tokens == 0 {
            return Err(invalid("num_tokens", "must be > 0"));
        }
        Ok(())
    }

    /// Kernel sig: `(out_fp8, scale, gate_up_f16, intermediate)`.
    /// Same layout as fused_silu_mul but uses GELU(tanh) instead of SiLU.
    ///
    /// # Safety
    /// Caller owns pointers for the call's duration.
    pub unsafe fn launch(
        &self,
        kernel: KernelFn,
        out_fp8: u64,
        scale: u64,
        gate_up: u64,
        stream: u64,
    ) -> Result<()> {
        self.validate()?;
        let mut out_fp8 = out_fp8;
        let mut scale = scale;
        let mut gate_up = gate_up;
        let mut intermediate = self.intermediate as i32;
        let args = [
            (&mut out_fp8) as *mut u64 as *mut core::ffi::c_void,
            (&mut scale) as *mut u64 as *mut core::ffi::c_void,
            (&mut gate_up) as *mut u64 as *mut core::ffi::c_void,
            (&mut intermediate) as *mut i32 as *mut core::ffi::c_void,
        ];
        const SMEM: u32 = 32 * 4;
        let block = (self.intermediate.min(1024), 1, 1);
        let grid = (self.num_tokens, 1, 1);
        launch_raw(kernel, grid, block, SMEM, stream, &args)
    }
}

// ---------------------------------------------------------------------------
// fused_qk_rmsnorm
// ---------------------------------------------------------------------------

pub struct FusedQkRmsnormLaunch {
    pub num_tokens: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub eps: f32,
}

impl FusedQkRmsnormLaunch {
    pub fn validate(&self) -> Result<()> {
        if self.head_dim == 0 || self.num_heads == 0 {
            return Err(invalid("qk_rmsnorm", "zero dim"));
        }
        Ok(())
    }

    /// Kernel sig: `(q_in, k_in, q_out, k_out, q_gamma, k_gamma,
    ///   num_tokens, num_heads, num_kv_heads, head_dim, eps)`.
    ///
    /// Applies RMSNorm independently to each (token, head) vector.
    /// q_gamma and k_gamma are [head_dim] scale vectors.
    ///
    /// # Safety
    /// Caller owns pointers.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        kernel: KernelFn,
        q_in: u64,
        k_in: u64,
        q_out: u64,
        k_out: u64,
        q_gamma: u64,
        k_gamma: u64,
        stream: u64,
    ) -> Result<()> {
        self.validate()?;
        let mut q_in = q_in;
        let mut k_in = k_in;
        let mut q_out = q_out;
        let mut k_out = k_out;
        let mut q_gamma = q_gamma;
        let mut k_gamma = k_gamma;
        let mut num_tokens = self.num_tokens as i32;
        let mut num_heads = self.num_heads as i32;
        let mut num_kv_heads = self.num_kv_heads as i32;
        let mut head_dim = self.head_dim as i32;
        let mut eps = self.eps;
        let args = [
            (&mut q_in) as *mut u64 as *mut core::ffi::c_void,
            (&mut k_in) as *mut u64 as *mut core::ffi::c_void,
            (&mut q_out) as *mut u64 as *mut core::ffi::c_void,
            (&mut k_out) as *mut u64 as *mut core::ffi::c_void,
            (&mut q_gamma) as *mut u64 as *mut core::ffi::c_void,
            (&mut k_gamma) as *mut u64 as *mut core::ffi::c_void,
            (&mut num_tokens) as *mut i32 as *mut core::ffi::c_void,
            (&mut num_heads) as *mut i32 as *mut core::ffi::c_void,
            (&mut num_kv_heads) as *mut i32 as *mut core::ffi::c_void,
            (&mut head_dim) as *mut i32 as *mut core::ffi::c_void,
            (&mut eps) as *mut f32 as *mut core::ffi::c_void,
        ];
        let total_heads = self.num_heads + self.num_kv_heads;
        let grid = (self.num_tokens, total_heads, 1);
        let block = (self.head_dim.min(1024), 1, 1);
        const SMEM: u32 = 32 * 4;
        launch_raw(kernel, grid, block, SMEM, stream, &args)
    }
}

// ---------------------------------------------------------------------------
// fused_qkv_rmsnorm: QK-norm (with gamma) + V-norm (parameter-free) in one launch
// ---------------------------------------------------------------------------

pub struct FusedQkvRmsnormLaunch {
    pub num_tokens: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub eps: f32,
    /// Row stride of the upstream QKV GEMM output (in f16 elements).
    /// Callers pass `q_dim + 2*kv_dim` so token-stride reads span the
    /// full interleaved row; the old code's implicit component-stride
    /// only worked at `num_tokens == 1`.
    pub src_row_stride: u32,
}

impl FusedQkvRmsnormLaunch {
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        kernel: KernelFn,
        q_in: u64,
        k_in: u64,
        v_in: u64,
        q_out: u64,
        k_out: u64,
        v_out: u64,
        q_gamma: u64,
        k_gamma: u64,
        stream: u64,
    ) -> Result<()> {
        let mut q_in = q_in;
        let mut k_in = k_in;
        let mut v_in = v_in;
        let mut q_out = q_out;
        let mut k_out = k_out;
        let mut v_out = v_out;
        let mut q_gamma = q_gamma;
        let mut k_gamma = k_gamma;
        let mut num_tokens = self.num_tokens as i32;
        let mut num_heads = self.num_heads as i32;
        let mut num_kv_heads = self.num_kv_heads as i32;
        let mut head_dim = self.head_dim as i32;
        let mut eps = self.eps;
        let mut src_row_stride = self.src_row_stride as i32;
        let args = [
            (&mut q_in) as *mut u64 as *mut core::ffi::c_void,
            (&mut k_in) as *mut u64 as *mut core::ffi::c_void,
            (&mut v_in) as *mut u64 as *mut core::ffi::c_void,
            (&mut q_out) as *mut u64 as *mut core::ffi::c_void,
            (&mut k_out) as *mut u64 as *mut core::ffi::c_void,
            (&mut v_out) as *mut u64 as *mut core::ffi::c_void,
            (&mut q_gamma) as *mut u64 as *mut core::ffi::c_void,
            (&mut k_gamma) as *mut u64 as *mut core::ffi::c_void,
            (&mut num_tokens) as *mut i32 as *mut core::ffi::c_void,
            (&mut num_heads) as *mut i32 as *mut core::ffi::c_void,
            (&mut num_kv_heads) as *mut i32 as *mut core::ffi::c_void,
            (&mut head_dim) as *mut i32 as *mut core::ffi::c_void,
            (&mut eps) as *mut f32 as *mut core::ffi::c_void,
            (&mut src_row_stride) as *mut i32 as *mut core::ffi::c_void,
        ];
        let total_heads = self.num_heads + 2 * self.num_kv_heads;
        let grid = (self.num_tokens, total_heads, 1);
        let block = (self.head_dim.min(1024), 1, 1);
        const SMEM: u32 = 32 * 4;
        launch_raw(kernel, grid, block, SMEM, stream, &args)
    }
}

// ---------------------------------------------------------------------------
// fused_rope_partial_fp8kv
// ---------------------------------------------------------------------------

pub struct FusedRopePartialFp8KvLaunch {
    pub num_tokens: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub rotary_dim: u32,
}

impl FusedRopePartialFp8KvLaunch {
    pub fn validate(&self) -> Result<()> {
        if self.rotary_dim > self.head_dim {
            return Err(invalid("rotary_dim", "must be <= head_dim"));
        }
        if self.rotary_dim % 2 != 0 {
            return Err(invalid("rotary_dim", "must be even"));
        }
        if self.num_kv_heads == 0 || self.num_heads % self.num_kv_heads != 0 {
            return Err(invalid(
                "num_heads/num_kv_heads",
                "num_heads must be a multiple of num_kv_heads",
            ));
        }
        Ok(())
    }

    /// Kernel sig: `(q, k, v, q_fp8, key_cache, value_cache,
    ///   k_scale_cache, v_scale_cache, q_scale_cache, cos, sin,
    ///   positions, slot_mapping, q_scale, num_tokens, num_heads,
    ///   num_kv_heads, head_dim, rotary_dim)`.
    ///
    /// `q_scale_cache`: optional `[num_tokens * num_heads]` f32
    /// scratch. When non-null the rope kernel computes per-(token,
    /// head) Q amax and writes a dynamic scale; when null the kernel
    /// falls back to the scalar `q_scale_ptr`.
    ///
    /// # Safety
    /// Caller owns all device pointers.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        kernel: KernelFn,
        q_in: u64,
        k_in: u64,
        v_in: u64,
        q_fp8_out: u64,
        k_cache_fp8: u64,
        v_cache_fp8: u64,
        k_scale_cache: u64,
        v_scale_cache: u64,
        q_scale_cache: u64,
        cos: u64,
        sin: u64,
        positions: u64,
        slot_mapping: u64,
        q_scale_ptr: u64,
        stream: u64,
    ) -> Result<()> {
        self.validate()?;
        let mut q_in = q_in;
        let mut k_in = k_in;
        let mut v_in = v_in;
        let mut q_fp8_out = q_fp8_out;
        let mut k_cache_fp8 = k_cache_fp8;
        let mut v_cache_fp8 = v_cache_fp8;
        let mut k_scale_cache = k_scale_cache;
        let mut v_scale_cache = v_scale_cache;
        let mut q_scale_cache = q_scale_cache;
        let mut cos = cos;
        let mut sin = sin;
        let mut positions = positions;
        let mut slot_mapping = slot_mapping;
        let mut q_scale_ptr = q_scale_ptr;
        let mut num_tokens = self.num_tokens as i32;
        let mut num_heads = self.num_heads as i32;
        let mut num_kv_heads = self.num_kv_heads as i32;
        let mut head_dim = self.head_dim as i32;
        let mut rotary_dim = self.rotary_dim as i32;
        let args = [
            (&mut q_in) as *mut u64 as *mut core::ffi::c_void,
            (&mut k_in) as *mut u64 as *mut core::ffi::c_void,
            (&mut v_in) as *mut u64 as *mut core::ffi::c_void,
            (&mut q_fp8_out) as *mut u64 as *mut core::ffi::c_void,
            (&mut k_cache_fp8) as *mut u64 as *mut core::ffi::c_void,
            (&mut v_cache_fp8) as *mut u64 as *mut core::ffi::c_void,
            (&mut k_scale_cache) as *mut u64 as *mut core::ffi::c_void,
            (&mut v_scale_cache) as *mut u64 as *mut core::ffi::c_void,
            (&mut q_scale_cache) as *mut u64 as *mut core::ffi::c_void,
            (&mut cos) as *mut u64 as *mut core::ffi::c_void,
            (&mut sin) as *mut u64 as *mut core::ffi::c_void,
            (&mut positions) as *mut u64 as *mut core::ffi::c_void,
            (&mut slot_mapping) as *mut u64 as *mut core::ffi::c_void,
            (&mut q_scale_ptr) as *mut u64 as *mut core::ffi::c_void,
            (&mut num_tokens) as *mut i32 as *mut core::ffi::c_void,
            (&mut num_heads) as *mut i32 as *mut core::ffi::c_void,
            (&mut num_kv_heads) as *mut i32 as *mut core::ffi::c_void,
            (&mut head_dim) as *mut i32 as *mut core::ffi::c_void,
            (&mut rotary_dim) as *mut i32 as *mut core::ffi::c_void,
        ];
        let max_heads = self.num_heads.max(self.num_kv_heads);
        let grid = (self.num_tokens, max_heads, 1);
        let block = ((self.head_dim / 2).max(32), 1, 1);
        launch_raw(kernel, grid, block, 0, stream, &args)
    }
}

// ---------------------------------------------------------------------------
// rmsnorm_inplace (no FP8 output, norm-only for post_attn / post_ff)
// ---------------------------------------------------------------------------

pub struct RmsnormInplaceLaunch {
    pub num_tokens: u32,
    pub hidden: u32,
    pub eps: f32,
}

impl RmsnormInplaceLaunch {
    pub fn validate(&self) -> Result<()> {
        require_multiple(self.hidden as usize, 8, "hidden")?;
        if self.num_tokens == 0 {
            return Err(invalid("num_tokens", "must be > 0"));
        }
        Ok(())
    }

    /// Applies RMSNorm in-place: x[i] = gamma[i] * x[i] / rms(x).
    /// Uses rmsnorm_inplace_f16_kernel (4 args: x, gamma, eps, hidden).
    pub unsafe fn launch(
        &self,
        kernel: KernelFn,
        x_inout: u64,
        gamma: u64,
        stream: u64,
    ) -> Result<()> {
        self.validate()?;
        let mut x = x_inout;
        let mut gamma = gamma;
        let mut eps = self.eps;
        let mut hidden = self.hidden as i32;
        let args = [
            (&mut x) as *mut u64 as *mut core::ffi::c_void,
            (&mut gamma) as *mut u64 as *mut core::ffi::c_void,
            (&mut eps) as *mut f32 as *mut core::ffi::c_void,
            (&mut hidden) as *mut i32 as *mut core::ffi::c_void,
        ];
        const SMEM: u32 = 32 * 4;
        let block = (self.hidden.min(1024), 1, 1);
        let grid = (self.num_tokens, 1, 1);
        launch_raw(kernel, grid, block, SMEM, stream, &args)
    }
}

// ---------------------------------------------------------------------------
// residual_scale_f16 (multiply residual by per-layer scalar)
// ---------------------------------------------------------------------------

pub struct ResidualScaleF16Launch {
    pub num_tokens: u32,
    pub hidden: u32,
}

impl ResidualScaleF16Launch {
    pub fn validate(&self) -> Result<()> {
        if self.num_tokens == 0 {
            return Err(invalid("num_tokens", "must be > 0"));
        }
        if self.hidden == 0 {
            return Err(invalid("hidden", "must be > 0"));
        }
        Ok(())
    }

    /// Multiplies every element of the residual buffer by a single f16
    /// scalar loaded from `scalar_ptr`. Applied in-place.
    ///
    /// Kernel sig: `(residual_f16_inout, scalar_ptr, hidden)`.
    /// Grid: (num_tokens, 1, 1), Block: (min(hidden, 1024), 1, 1).
    ///
    /// # Safety
    /// Caller owns device pointers for the call's duration.
    pub unsafe fn launch(
        &self,
        kernel: KernelFn,
        residual: u64,
        scalar_ptr: u64,
        stream: u64,
    ) -> Result<()> {
        self.validate()?;
        let mut residual = residual;
        let mut scalar_ptr = scalar_ptr;
        let mut hidden = self.hidden as i32;
        let args = [
            (&mut residual) as *mut u64 as *mut core::ffi::c_void,
            (&mut scalar_ptr) as *mut u64 as *mut core::ffi::c_void,
            (&mut hidden) as *mut i32 as *mut core::ffi::c_void,
        ];
        let block = (self.hidden.min(1024), 1, 1);
        let grid = (self.num_tokens, 1, 1);
        launch_raw(kernel, grid, block, 0, stream, &args)
    }
}

// ---------------------------------------------------------------------------
// vnorm_f16 (parameter-free RMS norm on V)
// ---------------------------------------------------------------------------

pub struct VnormF16Launch {
    pub num_tokens: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub eps: f32,
}

impl VnormF16Launch {
    pub fn validate(&self) -> Result<()> {
        if self.num_tokens == 0 {
            return Err(invalid("num_tokens", "must be > 0"));
        }
        if self.num_kv_heads == 0 || self.head_dim == 0 {
            return Err(invalid("vnorm", "zero dim"));
        }
        Ok(())
    }

    /// Kernel sig: `(v_f16_inout, eps, head_dim)`.
    /// Grid: (num_tokens * num_kv_heads), Block: (min(head_dim, 1024)).
    ///
    /// # Safety
    /// Caller owns pointers.
    pub unsafe fn launch(
        &self,
        kernel: KernelFn,
        v_inout: u64,
        stream: u64,
    ) -> Result<()> {
        self.validate()?;
        let mut v = v_inout;
        let mut eps = self.eps;
        let mut head_dim = self.head_dim as i32;
        let args = [
            (&mut v) as *mut u64 as *mut core::ffi::c_void,
            (&mut eps) as *mut f32 as *mut core::ffi::c_void,
            (&mut head_dim) as *mut i32 as *mut core::ffi::c_void,
        ];
        let grid = (self.num_tokens * self.num_kv_heads, 1, 1);
        let block = (self.head_dim.min(1024), 1, 1);
        const SMEM: u32 = 32 * 4;
        launch_raw(kernel, grid, block, SMEM, stream, &args)
    }
}

// ---------------------------------------------------------------------------
// vector_add_f16 (dst += src)
// ---------------------------------------------------------------------------

pub struct VectorAddF16Launch {
    pub n: u32,
}

impl VectorAddF16Launch {
    pub unsafe fn launch(
        &self,
        kernel: KernelFn,
        dst: u64,
        src: u64,
        stream: u64,
    ) -> Result<()> {
        let mut dst = dst;
        let mut src = src;
        let mut n = self.n as i32;
        let args = [
            (&mut dst) as *mut u64 as *mut core::ffi::c_void,
            (&mut src) as *mut u64 as *mut core::ffi::c_void,
            (&mut n) as *mut i32 as *mut core::ffi::c_void,
        ];
        let block = (256u32, 1, 1);
        let grid = ((self.n + 255) / 256, 1, 1);
        launch_raw(kernel, grid, block, 0, stream, &args)
    }
}

// ---------------------------------------------------------------------------
// fused_norm_add_residual: f32->bf16 + rmsnorm + add-to-residual(f16)
// ---------------------------------------------------------------------------

pub struct FusedNormAddResidualLaunch {
    pub num_tokens: u32,
    pub hidden: u32,
    pub eps: f32,
}

impl FusedNormAddResidualLaunch {
    pub unsafe fn launch(
        &self,
        kernel: KernelFn,
        gemm_out: u64,
        gamma: u64,
        residual: u64,
        layer_scalar: u64,
        stream: u64,
    ) -> Result<()> {
        let mut gemm_out = gemm_out;
        let mut gamma = gamma;
        let mut residual = residual;
        let mut layer_scalar = layer_scalar;
        let mut hidden = self.hidden as i32;
        let mut eps = self.eps;
        let args = [
            (&mut gemm_out) as *mut u64 as *mut core::ffi::c_void,
            (&mut gamma) as *mut u64 as *mut core::ffi::c_void,
            (&mut residual) as *mut u64 as *mut core::ffi::c_void,
            (&mut layer_scalar) as *mut u64 as *mut core::ffi::c_void,
            (&mut hidden) as *mut i32 as *mut core::ffi::c_void,
            (&mut eps) as *mut f32 as *mut core::ffi::c_void,
        ];
        let block = (self.hidden.min(1024), 1, 1);
        let grid = (self.num_tokens, 1, 1);
        let smem = self.hidden * 4;
        launch_raw(kernel, grid, block, smem, stream, &args)
    }
}

// ---------------------------------------------------------------------------
// fused_norm_add_residual_f16: scale_cols + rmsnorm + add-to-residual
// Takes F16 GEMM output + per-channel scale, fuses norm + residual add.
// ---------------------------------------------------------------------------

pub struct FusedNormAddResidualF16Launch {
    pub num_tokens: u32,
    pub hidden: u32,
    pub eps: f32,
}

impl FusedNormAddResidualF16Launch {
    pub unsafe fn launch(
        &self,
        kernel: KernelFn,
        gemm_out_f16: u64,
        channelscale: u64,
        gamma: u64,
        residual: u64,
        layer_scalar: u64,
        stream: u64,
    ) -> Result<()> {
        let mut gemm_out_f16 = gemm_out_f16;
        let mut channelscale = channelscale;
        let mut gamma = gamma;
        let mut residual = residual;
        let mut layer_scalar = layer_scalar;
        let mut hidden = self.hidden as i32;
        let mut eps = self.eps;
        let args = [
            (&mut gemm_out_f16) as *mut u64 as *mut core::ffi::c_void,
            (&mut channelscale) as *mut u64 as *mut core::ffi::c_void,
            (&mut gamma) as *mut u64 as *mut core::ffi::c_void,
            (&mut residual) as *mut u64 as *mut core::ffi::c_void,
            (&mut layer_scalar) as *mut u64 as *mut core::ffi::c_void,
            (&mut hidden) as *mut i32 as *mut core::ffi::c_void,
            (&mut eps) as *mut f32 as *mut core::ffi::c_void,
        ];
        let block = (self.hidden.min(1024), 1, 1);
        let grid = (self.num_tokens, 1, 1);
        let smem = self.hidden * 4;
        launch_raw(kernel, grid, block, smem, stream, &args)
    }
}

// ---------------------------------------------------------------------------
// fused_norm_add_residual_f16in: f16-input variant for the Sm121 decode
// fast path. Reads f16 gemm output directly, no channelscale broadcast
// (the preceding `fp8_gemv_wpr_native_f16in` already baked the per-
// channel scale into its output).
// ---------------------------------------------------------------------------

pub struct FusedNormAddResidualF16InLaunch {
    pub num_tokens: u32,
    pub hidden: u32,
    pub eps: f32,
}

impl FusedNormAddResidualF16InLaunch {
    pub unsafe fn launch(
        &self,
        kernel: KernelFn,
        gemm_out_f16: u64,
        gamma: u64,
        residual: u64,
        layer_scalar: u64,
        stream: u64,
    ) -> Result<()> {
        let mut gemm_out_f16 = gemm_out_f16;
        let mut gamma = gamma;
        let mut residual = residual;
        let mut layer_scalar = layer_scalar;
        let mut hidden = self.hidden as i32;
        let mut eps = self.eps;
        let args = [
            (&mut gemm_out_f16) as *mut u64 as *mut core::ffi::c_void,
            (&mut gamma) as *mut u64 as *mut core::ffi::c_void,
            (&mut residual) as *mut u64 as *mut core::ffi::c_void,
            (&mut layer_scalar) as *mut u64 as *mut core::ffi::c_void,
            (&mut hidden) as *mut i32 as *mut core::ffi::c_void,
            (&mut eps) as *mut f32 as *mut core::ffi::c_void,
        ];
        let block = (self.hidden.min(1024), 1, 1);
        let grid = (self.num_tokens, 1, 1);
        let smem = self.hidden * 4;
        launch_raw(kernel, grid, block, smem, stream, &args)
    }
}

// ---------------------------------------------------------------------------
// bf16_to_f16_sat (bf16 -> f16 with saturation clamp)
// ---------------------------------------------------------------------------

pub struct Bf16ToF16SatLaunch {
    pub n: u32,
}

impl Bf16ToF16SatLaunch {
    pub unsafe fn launch(
        &self,
        kernel: KernelFn,
        dst: u64,
        src: u64,
        stream: u64,
    ) -> Result<()> {
        let mut dst = dst;
        let mut src = src;
        let mut n = self.n as i32;
        let args = [
            (&mut dst) as *mut u64 as *mut core::ffi::c_void,
            (&mut src) as *mut u64 as *mut core::ffi::c_void,
            (&mut n) as *mut i32 as *mut core::ffi::c_void,
        ];
        let block = (256u32, 1, 1);
        let grid = ((self.n + 255) / 256, 1, 1);
        launch_raw(kernel, grid, block, 0, stream, &args)
    }
}

// ---------------------------------------------------------------------------
// logit_softcap
// ---------------------------------------------------------------------------

pub struct LogitSoftcapLaunch {
    pub num_tokens: u32,
    pub vocab: u32,
    pub cap: f32,
}

impl LogitSoftcapLaunch {
    pub fn validate(&self) -> Result<()> {
        if self.vocab == 0 || self.num_tokens == 0 {
            return Err(invalid("logit_softcap", "zero dim"));
        }
        if self.cap <= 0.0 {
            return Err(invalid("cap", "must be > 0"));
        }
        Ok(())
    }

    /// Kernel sig: `(logits_f16_inout, vocab, cap)`.
    /// Applies: logits[i] = cap * tanh(logits[i] / cap)
    ///
    /// # Safety
    /// Caller owns pointers.
    pub unsafe fn launch(
        &self,
        kernel: KernelFn,
        logits: u64,
        stream: u64,
    ) -> Result<()> {
        self.validate()?;
        let mut logits = logits;
        let mut vocab = self.vocab as i32;
        let mut cap = self.cap;
        let args = [
            (&mut logits) as *mut u64 as *mut core::ffi::c_void,
            (&mut vocab) as *mut i32 as *mut core::ffi::c_void,
            (&mut cap) as *mut f32 as *mut core::ffi::c_void,
        ];
        let block = (self.vocab.min(1024), 1, 1);
        let grid = (self.num_tokens, 1, 1);
        launch_raw(kernel, grid, block, 0, stream, &args)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gelu_rejects_non_multiple_of_8() {
        let l = FusedGeluMulFp8QuantLaunch {
            num_tokens: 1,
            intermediate: 13,
        };
        assert!(l.validate().is_err());
    }

    #[test]
    fn gelu_accepts_valid() {
        let l = FusedGeluMulFp8QuantLaunch {
            num_tokens: 32,
            intermediate: 21504,
        };
        assert!(l.validate().is_ok());
    }

    #[test]
    fn partial_rope_rejects_rotary_gt_head() {
        let l = FusedRopePartialFp8KvLaunch {
            num_tokens: 1,
            num_heads: 16,
            num_kv_heads: 4,
            head_dim: 256,
            rotary_dim: 512,
        };
        assert!(l.validate().is_err());
    }

    #[test]
    fn partial_rope_accepts_valid() {
        let l = FusedRopePartialFp8KvLaunch {
            num_tokens: 1,
            num_heads: 16,
            num_kv_heads: 16,
            head_dim: 256,
            rotary_dim: 128,
        };
        assert!(l.validate().is_ok());
    }

    #[test]
    fn softcap_rejects_zero_cap() {
        let l = LogitSoftcapLaunch {
            num_tokens: 1,
            vocab: 262144,
            cap: 0.0,
        };
        assert!(l.validate().is_err());
    }

    #[test]
    fn residual_scale_rejects_zero_tokens() {
        let l = ResidualScaleF16Launch {
            num_tokens: 0,
            hidden: 5376,
        };
        assert!(l.validate().is_err());
    }

    #[test]
    fn residual_scale_accepts_valid() {
        let l = ResidualScaleF16Launch {
            num_tokens: 32,
            hidden: 5376,
        };
        assert!(l.validate().is_ok());
    }

    #[test]
    fn qk_rmsnorm_rejects_zero() {
        let l = FusedQkRmsnormLaunch {
            num_tokens: 1,
            num_heads: 0,
            num_kv_heads: 4,
            head_dim: 256,
            eps: 1e-6,
        };
        assert!(l.validate().is_err());
    }
}
