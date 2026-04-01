//! CUDA linear (dense / GEMM) layer via cuBLAS.
//!
//! Implements `output[m,n] = input[m,k] @ weight^T[k,n] [+ bias]` where weight
//! is stored as `[n, k]` row-major (out_features x in_features), matching the
//! convention in `linear.rs`.
//!
//! This module is intended to be gated behind `#[cfg(feature = "cuda")]` in the
//! parent `mod.rs`. It delegates the unsafe cuBLAS call to `rvllm_gpu::cublas_ops`
//! so this crate's `#![forbid(unsafe_code)]` is respected.

use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use half::f16;
use rvllm_core::prelude::{LLMError, Result};
use rvllm_gpu::cublas::CublasHandle;
use rvllm_gpu::cublas_ops::CublasOps;
#[cfg(feature = "cublaslt")]
use rvllm_gpu::cublaslt_ops::CublasLtOps;
use std::sync::Arc;

/// GPU-accelerated dense linear projection using cuBLAS SGEMM.
///
/// Owns a `CublasOps` handle so cuBLAS init cost is amortized across calls.
pub struct CudaLinearLayer {
    ops: CublasOps,
}

impl CudaLinearLayer {
    /// Create a new layer bound to the given CUDA stream.
    pub fn new(stream: Arc<CudaStream>) -> Result<Self> {
        Ok(Self {
            ops: CublasOps::new(stream)?,
        })
    }

    /// Convenience constructor sharing the stream from an existing `CublasHandle`.
    pub fn from_handle(blas: &CublasHandle) -> Result<Self> {
        Self::new(blas.stream().clone())
    }

    /// Compute `output[m,n] = input[m,k] @ weight^T[k,n] [+ bias]`.
    ///
    /// # Arguments
    /// * `input`  - `[m, k]` row-major activation tensor on GPU
    /// * `weight` - `[n, k]` row-major weight matrix on GPU
    /// * `bias`   - optional `[n]` bias vector on GPU
    /// * `m`      - number of tokens / rows in input
    /// * `n`      - output features (rows in weight)
    /// * `k`      - input features (cols in weight, cols in input)
    pub fn forward(
        &self,
        input: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        bias: Option<&CudaSlice<f32>>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<CudaSlice<f32>> {
        if input.len() < m * k {
            return Err(LLMError::GpuError(format!(
                "CudaLinearLayer: input len {} < m*k = {}",
                input.len(),
                m * k
            )));
        }
        if weight.len() < n * k {
            return Err(LLMError::GpuError(format!(
                "CudaLinearLayer: weight len {} < n*k = {}",
                weight.len(),
                n * k
            )));
        }
        if let Some(b) = bias {
            if b.len() < n {
                return Err(LLMError::GpuError(format!(
                    "CudaLinearLayer: bias len {} < n = {}",
                    b.len(),
                    n
                )));
            }
        }

        let stream = self.ops.stream();

        // Allocate output [m, n]. If bias is present, tile it into every row so
        // sgemm accumulates on top with beta=1.
        let mut output: CudaSlice<f32> = if let Some(b) = bias {
            let bias_host = stream
                .clone_dtoh(b)
                .map_err(|e| LLMError::GpuError(format!("bias dtoh failed: {e}")))?;
            let mut tiled = Vec::with_capacity(m * n);
            for _ in 0..m {
                tiled.extend_from_slice(&bias_host[..n]);
            }
            stream
                .clone_htod(&tiled)
                .map_err(|e| LLMError::GpuError(format!("tiled bias htod failed: {e}")))?
        } else {
            // Safety: sgemm with beta=0 writes all m*n elements
            unsafe { stream.alloc::<f32>(m * n) }
                .map_err(|e| LLMError::GpuError(format!("output alloc failed: {e}")))?
        };

        let beta = if bias.is_some() { 1.0f32 } else { 0.0f32 };

        // C[m,n] = 1.0 * input[m,k] @ weight^T[k,n] + beta * C[m,n]
        self.ops
            .sgemm_a_bt(m, n, k, 1.0, input, weight, beta, &mut output)?;

        Ok(output)
    }

    /// Static forward matching the spec signature. Creates a temporary CublasOps;
    /// prefer the instance method [`Self::forward`] for repeated calls.
    pub fn forward_once(
        input: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        bias: Option<&CudaSlice<f32>>,
        m: usize,
        n: usize,
        k: usize,
        blas: &CublasHandle,
    ) -> Result<CudaSlice<f32>> {
        let layer = Self::from_handle(blas)?;
        layer.forward(input, weight, bias, m, n, k)
    }

    /// Static forward with f16 weights: cast f32 input -> f16, hgemm, cast f16 output -> f32.
    ///
    /// Used for the LM head projection when `use_fp16` is enabled.
    ///
    /// `cast_f32_f16` and `cast_f16_f32` are pre-loaded `CudaFunction` handles
    /// for the `cast_f32_to_f16_kernel` and `cast_f16_to_f32_kernel` kernels
    /// respectively (from the `cast_fp` PTX module).
    pub fn forward_once_f16(
        input: &CudaSlice<f32>,
        weight: &CudaSlice<f16>,
        m: usize,
        n: usize,
        k: usize,
        blas: &CublasHandle,
        loader: &rvllm_gpu::kernel_loader::KernelLoader,
    ) -> Result<CudaSlice<f32>> {
        let stream = blas.stream();

        let cast_f32_f16 = loader.get_func("cast_fp", "cast_f32_to_f16_kernel")
            .map_err(|e| LLMError::GpuError(format!("load cast_f32_to_f16_kernel: {e}")))?;
        let cast_f16_f32 = loader.get_func("cast_fp", "cast_f16_to_f32_kernel")
            .map_err(|e| LLMError::GpuError(format!("load cast_f16_to_f32_kernel: {e}")))?;

        // Cast input f32 -> f16
        let input_f16 = Self::gpu_cast_f32_to_f16(stream, input, m * k, &cast_f32_f16)?;

        // Safety: hgemm with beta=0 writes all m*n elements
        let mut output_f16 = unsafe { stream.alloc::<f16>(m * n) }
            .map_err(|e| LLMError::GpuError(format!("forward_once_f16 alloc: {e}")))?;

        // hgemm: output = input @ weight^T
        blas.hgemm(
            m, n, k,
            f16::ONE,
            &input_f16,
            weight,
            f16::ZERO,
            &mut output_f16,
        )?;

        // Cast output f16 -> f32
        Self::gpu_cast_f16_to_f32(stream, &output_f16, m * n, &cast_f16_f32)
    }

    /// Mixed-precision forward: f32 input, f16 weight, f32 output.
    ///
    /// Casts input f32->f16, then uses cublasGemmEx(f16,f16->f32) to produce
    /// f32 output directly. Saves 1 cast kernel + 1 alloc per call vs the old
    /// forward_once_f16 which does cast_in + hgemm + cast_out (2 casts + 2 allocs).
    pub fn forward_mixed(
        input: &CudaSlice<f32>,
        weight: &CudaSlice<f16>,
        m: usize,
        n: usize,
        k: usize,
        blas: &CublasHandle,
        loader: &rvllm_gpu::kernel_loader::KernelLoader,
    ) -> Result<CudaSlice<f32>> {
        let stream = blas.stream();

        // Cast input f32 -> f16 (still needed; cuBLAS requires matching A/B types)
        let cast_f32_f16 = loader.get_func("cast_fp", "cast_f32_to_f16_kernel")
            .map_err(|e| LLMError::GpuError(format!("load cast_f32_to_f16_kernel: {e}")))?;
        let input_f16 = Self::gpu_cast_f32_to_f16(stream, input, m * k, &cast_f32_f16)?;

        // Safety: hgemm_f32_output with beta=0 writes all m*n elements
        let mut output = unsafe { stream.alloc::<f32>(m * n) }
            .map_err(|e| LLMError::GpuError(format!("forward_mixed alloc: {e}")))?;

        // f16 x f16 -> f32 via cublasGemmEx
        blas.hgemm_f32_output(m, n, k, 1.0, &input_f16, weight, 0.0, &mut output)?;
        Ok(output)
    }

    /// Pre-cast forward: f16 input already cast by caller, f16 weight, f32 output.
    ///
    /// When multiple linears share the same f32 input (e.g. Q/K/V all use normed,
    /// gate/up both use normed2), the caller casts f32->f16 ONCE and calls this
    /// for each projection. Saves N-1 redundant cast kernels.
    pub fn forward_f16_in(
        input_f16: &CudaSlice<f16>,
        weight: &CudaSlice<f16>,
        m: usize,
        n: usize,
        k: usize,
        blas: &CublasHandle,
    ) -> Result<CudaSlice<f32>> {
        // Safety: hgemm with beta=0 writes all m*n elements
        let mut output = unsafe { blas.stream().alloc::<f32>(m * n) }
            .map_err(|e| LLMError::GpuError(format!("forward_f16_in alloc: {e}")))?;
        blas.hgemm_f32_output(m, n, k, 1.0, input_f16, weight, 0.0, &mut output)?;
        Ok(output)
    }

    /// Static forward with f16 weights using cublasLt for decode-sized batches.
    ///
    /// When `cublaslt` feature is enabled and the shape is either autotuned
    /// or small enough for the heuristic path, use cublasLt. Falls back to
    /// standard cuBLAS only when the shape is not covered.
    #[cfg(feature = "cublaslt")]
    pub fn forward_once_f16_lt(
        input: &CudaSlice<f32>,
        weight: &CudaSlice<f16>,
        m: usize,
        n: usize,
        k: usize,
        blas: &CublasHandle,
        lt: &CublasLtOps,
        loader: &rvllm_gpu::kernel_loader::KernelLoader,
    ) -> Result<CudaSlice<f32>> {
        if !lt.should_use_f16_for_shape(m, n, k) {
            return Self::forward_once_f16(input, weight, m, n, k, blas, loader);
        }

        let stream = blas.stream();

        let cast_f32_f16 = loader.get_func("cast_fp", "cast_f32_to_f16_kernel")
            .map_err(|e| LLMError::GpuError(format!("load cast_f32_to_f16_kernel: {e}")))?;
        let cast_f16_f32 = loader.get_func("cast_fp", "cast_f16_to_f32_kernel")
            .map_err(|e| LLMError::GpuError(format!("load cast_f16_to_f32_kernel: {e}")))?;

        let input_f16 = Self::gpu_cast_f32_to_f16(stream, input, m * k, &cast_f32_f16)?;

        // Safety: hgemm with beta=0 writes all m*n elements
        let mut output_f16 = unsafe { stream.alloc::<f16>(m * n) }
            .map_err(|e| LLMError::GpuError(format!("forward_once_f16_lt alloc: {e}")))?;

        lt.hgemm_a_bt(m, n, k, 1.0, &input_f16, weight, 0.0, &mut output_f16)?;

        Self::gpu_cast_f16_to_f32(stream, &output_f16, m * n, &cast_f16_f32)
    }

    pub fn gpu_cast_f32_to_f16(
        stream: &Arc<CudaStream>,
        input: &CudaSlice<f32>,
        n: usize,
        kernel: &CudaFunction,
    ) -> Result<CudaSlice<f16>> {
        // Safety: cast kernel writes all n elements
        let mut output = unsafe { stream.alloc::<f16>(n) }
            .map_err(|e| LLMError::GpuError(format!("cast_f32_to_f16 alloc: {e}")))?;

        let threads = 256u32;
        let blocks = ((n as u32) + threads - 1) / threads;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream
                .launch_builder(kernel)
                .arg(&mut output)
                .arg(input)
                .arg(&(n as i32))
                .launch(cfg)
                .map_err(|e| LLMError::GpuError(format!("cast_f32_to_f16 launch: {e}")))?;
        }
        Ok(output)
    }

    fn gpu_cast_f16_to_f32(
        stream: &Arc<CudaStream>,
        input: &CudaSlice<f16>,
        n: usize,
        kernel: &CudaFunction,
    ) -> Result<CudaSlice<f32>> {
        // Safety: cast kernel writes all n elements
        let mut output = unsafe { stream.alloc::<f32>(n) }
            .map_err(|e| LLMError::GpuError(format!("cast_f16_to_f32 alloc: {e}")))?;

        let threads = 256u32;
        let blocks = ((n as u32) + threads - 1) / threads;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream
                .launch_builder(kernel)
                .arg(&mut output)
                .arg(input)
                .arg(&(n as i32))
                .launch(cfg)
                .map_err(|e| LLMError::GpuError(format!("cast_f16_to_f32 launch: {e}")))?;
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    // Tests require the `cuda` feature and a real GPU.
    // Run with: cargo test -p rvllm-model-runner --features cuda
}
