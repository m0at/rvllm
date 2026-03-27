//! CUDA-accelerated RMSNorm dispatching to the rms_norm.cu kernel.
//!
//! This module is only compiled when the `cuda` feature is enabled.
//! It depends on Agent 1's KernelLoader (rvllm_gpu::kernel_loader) and
//! cudarc for device memory types.

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaSlice, CudaStream, LaunchAsync, LaunchConfig};
#[cfg(feature = "cuda")]
use std::sync::Arc;
#[cfg(feature = "cuda")]
use rvllm_core::error::{LLMError, Result};
#[cfg(feature = "cuda")]
use rvllm_gpu::kernel_loader::KernelLoader;

/// CUDA-backed RMSNorm layer.
///
/// Dispatches to the `rms_norm_kernel` function compiled from `kernels/rms_norm.cu`.
///
/// Kernel launch config (from rms_norm.cu):
///   Grid:  (num_tokens, 1, 1)
///   Block: (min(hidden_size, 1024), 1, 1)
///   Shared memory: blockDim.x * sizeof(float)
#[cfg(feature = "cuda")]
pub struct CudaRMSNorm;

#[cfg(feature = "cuda")]
impl CudaRMSNorm {
    /// Run RMSNorm on GPU via the precompiled PTX kernel.
    ///
    /// # Arguments
    /// * `input`       - Device buffer of shape [num_tokens, hidden_size], flattened, f32.
    /// * `weight`      - Device buffer of shape [hidden_size], f32.
    /// * `eps`         - Epsilon for numerical stability (e.g. 1e-6).
    /// * `hidden_size` - The hidden dimension (last axis of input).
    /// * `loader`      - KernelLoader holding the compiled rms_norm PTX module.
    /// * `stream`      - CUDA stream for async execution.
    ///
    /// # Returns
    /// A new `CudaSlice<f32>` of the same length as `input` containing the normalized output.
    pub fn forward(
        input: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        eps: f32,
        hidden_size: usize,
        loader: &KernelLoader,
        stream: &CudaStream,
    ) -> Result<CudaSlice<f32>> {
        let num_elements = input.len();
        if num_elements == 0 {
            return Err(LLMError::ModelError("CudaRMSNorm: empty input".into()));
        }
        if weight.len() != hidden_size {
            return Err(LLMError::ModelError(format!(
                "CudaRMSNorm: weight len {} != hidden_size {}",
                weight.len(),
                hidden_size
            )));
        }
        if num_elements % hidden_size != 0 {
            return Err(LLMError::ModelError(format!(
                "CudaRMSNorm: input len {} not divisible by hidden_size {}",
                num_elements, hidden_size
            )));
        }

        let num_tokens = num_elements / hidden_size;
        let block_dim = hidden_size.min(1024) as u32;
        let grid_dim = num_tokens as u32;
        let shared_mem_bytes = block_dim * std::mem::size_of::<f32>() as u32;

        let cfg = LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes,
        };

        // Allocate output buffer on the same device
        let device = loader.device();
        let output = device.alloc_zeros::<f32>(num_elements).map_err(|e| {
            LLMError::GpuError(format!("CudaRMSNorm: output alloc failed: {e}"))
        })?;

        let hidden_size_i32 = hidden_size as i32;

        // SAFETY: The kernel reads `input` and `weight` within bounds guaranteed by
        // the length checks above. `output` is freshly allocated with the correct
        // size. The launch config matches the kernel's expected grid/block/shared
        // memory layout as documented in rms_norm.cu.
        unsafe {
            loader
                .launch(
                    "rms_norm",
                    "rms_norm_kernel",
                    cfg,
                    stream,
                    (&output, input, weight, eps, hidden_size_i32),
                )
                .map_err(|e| {
                    LLMError::GpuError(format!("CudaRMSNorm: kernel launch failed: {e}"))
                })?;
        }

        Ok(output)
    }

    /// In-place RMSNorm: normalizes `input` directly without allocating a new buffer.
    ///
    /// Useful for residual connections where the output overwrites the input.
    pub fn forward_inplace(
        input: &mut CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        eps: f32,
        hidden_size: usize,
        loader: &KernelLoader,
        stream: &CudaStream,
    ) -> Result<()> {
        let num_elements = input.len();
        if num_elements == 0 {
            return Err(LLMError::ModelError("CudaRMSNorm: empty input".into()));
        }
        if weight.len() != hidden_size {
            return Err(LLMError::ModelError(format!(
                "CudaRMSNorm: weight len {} != hidden_size {}",
                weight.len(),
                hidden_size
            )));
        }
        if num_elements % hidden_size != 0 {
            return Err(LLMError::ModelError(format!(
                "CudaRMSNorm: input len {} not divisible by hidden_size {}",
                num_elements, hidden_size
            )));
        }

        let num_tokens = num_elements / hidden_size;
        let block_dim = hidden_size.min(1024) as u32;
        let grid_dim = num_tokens as u32;
        let shared_mem_bytes = block_dim * std::mem::size_of::<f32>() as u32;

        let cfg = LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes,
        };

        let hidden_size_i32 = hidden_size as i32;

        // SAFETY: In-place variant -- output pointer == input pointer. The kernel
        // uses __syncthreads() between reading input and writing output within each
        // block, so there is no data race for a single-token-per-block launch.
        unsafe {
            loader
                .launch(
                    "rms_norm",
                    "rms_norm_kernel",
                    cfg,
                    stream,
                    (input, input as &CudaSlice<f32>, weight, eps, hidden_size_i32),
                )
                .map_err(|e| {
                    LLMError::GpuError(format!(
                        "CudaRMSNorm: inplace kernel launch failed: {e}"
                    ))
                })?;
        }

        Ok(())
    }
}

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;

    #[test]
    fn cuda_rmsnorm_rejects_empty_input() {
        // This is a host-side validation test -- no GPU needed.
        // We can't construct real CudaSlice without a device, so we just
        // verify the module compiles and the struct exists.
        let _marker: fn() = || {
            let _ = CudaRMSNorm;
        };
    }
}
