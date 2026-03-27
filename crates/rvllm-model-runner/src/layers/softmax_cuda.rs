//! CUDA softmax dispatch via the softmax.cu kernel.
//!
//! Provides `CudaSoftmax` which launches the numerically-stable softmax kernel
//! (online warp-reduction algorithm) on GPU. Used for both attention score
//! normalization and sampling logit conversion.
//!
//! The kernel expects row-major `[num_rows, vocab_size]` f32 input and produces
//! row-major `[num_rows, vocab_size]` f32 output where each row sums to 1.0.
//!
//! Launch config (from softmax.cu):
//!   Grid:  (num_rows, 1, 1)
//!   Block: (min(vocab_size, 1024), 1, 1)
//!   Shared memory: static (warp_max[32] + warp_sum[32] + s_max + s_sum)
//!
//! NOTE: This module requires `unsafe` for cudarc kernel launches. The crate
//! root has `#![forbid(unsafe_code)]` which must be relaxed to `#![deny(...)]`
//! under the `cuda` feature by Agent 20 (Build System + Feature Flags).
#![allow(unsafe_code)]

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, CudaStream, LaunchAsync, LaunchConfig};
use tracing::trace;

use rvllm_core::prelude::{LLMError, Result};

/// Maximum threads per block for the softmax kernel.
const MAX_BLOCK_SIZE: u32 = 1024;

/// Preloaded softmax kernel handle.
///
/// Holds a reference to the compiled PTX module so repeated calls avoid
/// re-loading overhead. Constructed once during model init, then reused
/// for every forward pass.
pub struct CudaSoftmax {
    device: Arc<CudaDevice>,
    func: CudaFunction,
}

impl CudaSoftmax {
    /// Load the softmax PTX and extract the `softmax_kernel` function.
    ///
    /// `ptx_bytes` should be the compiled PTX of `kernels/softmax.cu`.
    /// The caller can embed this at compile time or load from disk via
    /// `RVLLM_KERNEL_DIR`.
    pub fn new(device: Arc<CudaDevice>, ptx_bytes: &[u8]) -> Result<Self> {
        let ptx = std::str::from_utf8(ptx_bytes)
            .map_err(|e| LLMError::GpuError(format!("softmax PTX is not valid UTF-8: {e}")))?;

        device
            .load_ptx(
                cudarc::nvrtc::Ptx::from_src(ptx),
                "softmax",
                &["softmax_kernel"],
            )
            .map_err(|e| LLMError::GpuError(format!("failed to load softmax PTX: {e}")))?;

        let func = device
            .get_func("softmax", "softmax_kernel")
            .ok_or_else(|| {
                LLMError::GpuError("softmax_kernel function not found in PTX module".into())
            })?;

        trace!("CudaSoftmax: loaded softmax_kernel");
        Ok(Self { device, func })
    }

    /// Run softmax over `input` rows, returning a new output buffer.
    ///
    /// * `input`      - device buffer of shape `[num_rows, vocab_size]`, row-major f32
    /// * `num_rows`   - number of rows (batch dimension)
    /// * `vocab_size` - number of columns per row
    /// * `stream`     - CUDA stream for async execution
    ///
    /// Returns a newly-allocated `CudaSlice<f32>` of the same shape containing
    /// the softmax probabilities.
    pub fn forward(
        &self,
        input: &CudaSlice<f32>,
        num_rows: usize,
        vocab_size: usize,
        stream: &CudaStream,
    ) -> Result<CudaSlice<f32>> {
        let total = num_rows * vocab_size;
        if input.len() < total {
            return Err(LLMError::GpuError(format!(
                "softmax input too small: {} elements but need {} ({}x{})",
                input.len(),
                total,
                num_rows,
                vocab_size,
            )));
        }

        let output: CudaSlice<f32> = self
            .device
            .alloc_zeros(total)
            .map_err(|e| LLMError::GpuError(format!("softmax output alloc failed: {e}")))?;

        self.launch(&output, input, num_rows, vocab_size, stream)?;
        Ok(output)
    }

    /// In-place softmax: overwrites `data` with its softmax probabilities.
    ///
    /// Internally allocates a temporary buffer, runs the kernel writing to it,
    /// then does a device-to-device copy back. This avoids aliasing the same
    /// `CudaSlice` as both kernel input and output through Rust references.
    ///
    /// For the hot path where an extra allocation is unacceptable, prefer
    /// `forward()` and swap the buffer at the call site.
    pub fn forward_inplace(
        &self,
        data: &mut CudaSlice<f32>,
        num_rows: usize,
        vocab_size: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let total = num_rows * vocab_size;
        if data.len() < total {
            return Err(LLMError::GpuError(format!(
                "softmax inplace buffer too small: {} elements but need {} ({}x{})",
                data.len(),
                total,
                num_rows,
                vocab_size,
            )));
        }

        // Allocate temporary output, run kernel, then swap.
        let tmp = self.forward(data, num_rows, vocab_size, stream)?;

        // Device-to-device copy: tmp -> data.
        self.device
            .dtod_copy(&tmp, data)
            .map_err(|e| LLMError::GpuError(format!("softmax d2d copy failed: {e}")))?;

        trace!(num_rows, vocab_size, "softmax inplace completed");
        Ok(())
    }

    /// Launch the softmax kernel with the given output and input buffers.
    fn launch(
        &self,
        output: &CudaSlice<f32>,
        input: &CudaSlice<f32>,
        num_rows: usize,
        vocab_size: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let block_x = std::cmp::min(vocab_size as u32, MAX_BLOCK_SIZE);
        let cfg = LaunchConfig {
            grid_dim: (num_rows as u32, 1, 1),
            block_dim: (block_x, 1, 1),
            shared_mem_bytes: 0, // kernel uses static __shared__ declarations
        };

        let vocab_size_i32 = vocab_size as i32;

        // SAFETY: The kernel signature is:
        //   softmax_kernel(float* output, const float* input, int vocab_size)
        //
        // - `output` has at least `num_rows * vocab_size` allocated elements.
        // - `input` has at least `num_rows * vocab_size` elements (validated
        //    by the caller).
        // - `vocab_size_i32` matches the kernel's `int` parameter.
        // - Grid dim (num_rows, 1, 1) launches one block per row.
        // - Block dim (min(vocab_size, 1024), 1, 1) matches the kernel's
        //   stride loop expectations.
        // - `output` and `input` do not alias (separate CudaSlice allocations).
        unsafe {
            self.func
                .clone()
                .launch_on_stream(stream, cfg, (output, input, vocab_size_i32))
                .map_err(|e| LLMError::GpuError(format!("softmax kernel launch failed: {e}")))?;
        }

        trace!(num_rows, vocab_size, block_x, "softmax kernel launched");
        Ok(())
    }

    /// Returns a reference to the underlying CUDA device.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    // CUDA tests require a real GPU and are only runnable when the `cuda`
    // feature is enabled on a machine with NVIDIA drivers + CUDA toolkit.
    //
    // Run with: cargo test -p rvllm-model-runner --features cuda -- softmax_cuda
    //
    // Test outline (requires GPU):
    // 1. Create CudaDevice
    // 2. Load softmax.cu PTX (pre-compiled with nvcc --ptx)
    // 3. Upload known logits [e.g., [1,2,3], [0,0,0]] to GPU
    // 4. Call CudaSoftmax::forward
    // 5. Download result, verify each row sums to ~1.0
    // 6. Verify relative ordering (higher logit -> higher probability)
    // 7. Test forward_inplace produces identical results
}
