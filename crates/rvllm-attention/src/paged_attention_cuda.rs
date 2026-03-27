//! CUDA-backed PagedAttention V2 backend.
//!
//! Dispatches to the `paged_attention_v2_kernel` in `paged_attention.cu` via cudarc.
//! The kernel operates on f32; this module handles f16 <-> f32 conversion at the
//! boundary since the `AttentionBackend` trait uses `GpuBuffer<f16>`.
//!
//! This entire module is gated behind `#[cfg(feature = "cuda")]`.

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, CudaStream, LaunchAsync, LaunchConfig};
use half::f16;
use tracing::{debug, trace};

use rvllm_core::prelude::{LLMError, Result};

use crate::backend::AttentionBackend;
use crate::buffer::GpuBuffer;

/// Name of the PTX module loaded into cudarc.
const MODULE_NAME: &str = "paged_attention";

/// Name of the kernel function within the PTX module.
const KERNEL_NAME: &str = "paged_attention_v2_kernel";

/// CUDA-backed PagedAttention V2 backend.
///
/// Loads the `paged_attention_v2_kernel` PTX and dispatches attention forward
/// passes to the GPU. Each call uploads the f16 host data as f32, launches the
/// kernel, and converts the f32 output back to f16.
pub struct CudaPagedAttention {
    device: Arc<CudaDevice>,
    stream: CudaStream,
}

impl CudaPagedAttention {
    /// Create a new CUDA paged attention backend.
    ///
    /// `ptx_bytes` must contain the compiled PTX for `paged_attention.cu`.
    /// The module is loaded once and reused for every `forward` call.
    pub fn new(device: Arc<CudaDevice>, ptx_bytes: &[u8]) -> Result<Self> {
        let ptx_str = std::str::from_utf8(ptx_bytes)
            .map_err(|e| LLMError::GpuError(format!("PTX is not valid UTF-8: {e}")))?;

        device
            .load_ptx(
                cudarc::nvrtc::Ptx::from_src(ptx_str),
                MODULE_NAME,
                &[KERNEL_NAME],
            )
            .map_err(|e| LLMError::GpuError(format!("failed to load paged_attention PTX: {e}")))?;

        let stream = device
            .fork_default_stream()
            .map_err(|e| LLMError::GpuError(format!("failed to create CUDA stream: {e}")))?;

        debug!("CudaPagedAttention initialized");
        Ok(Self { device, stream })
    }

    /// Convenience constructor that creates a CudaDevice for the given ordinal.
    pub fn with_device(device_id: usize, ptx_bytes: &[u8]) -> Result<Self> {
        let device = CudaDevice::new(device_id)
            .map_err(|e| LLMError::GpuError(format!("CUDA device {device_id} init: {e}")))?;
        Self::new(Arc::new(device), ptx_bytes)
    }

    /// Upload a host f16 slice to the GPU as f32.
    fn upload_f16_as_f32(&self, host: &[f16]) -> Result<CudaSlice<f32>> {
        let f32_data: Vec<f32> = host.iter().map(|v| v.to_f32()).collect();
        self.device
            .htod_sync_copy(&f32_data)
            .map_err(|e| LLMError::GpuError(format!("htod f32 upload failed: {e}")))
    }

    /// Upload a host i32 slice to the GPU.
    fn upload_i32(&self, host: &[i32]) -> Result<CudaSlice<i32>> {
        self.device
            .htod_sync_copy(host)
            .map_err(|e| LLMError::GpuError(format!("htod i32 upload failed: {e}")))
    }
}

impl AttentionBackend for CudaPagedAttention {
    fn forward(
        &self,
        query: &GpuBuffer<f16>,
        key_cache: &GpuBuffer<f16>,
        value_cache: &GpuBuffer<f16>,
        block_tables: &GpuBuffer<i32>,
        context_lens: &GpuBuffer<i32>,
        max_context_len: usize,
        scale: f32,
    ) -> Result<GpuBuffer<f16>> {
        // -- validate shapes --------------------------------------------------
        if query.shape.len() != 3 {
            return Err(LLMError::GpuError(format!(
                "query must be 3-D [num_tokens, num_heads, head_dim], got {} dims",
                query.shape.len()
            )));
        }
        if key_cache.shape.len() != 4 {
            return Err(LLMError::GpuError(format!(
                "key_cache must be 4-D [num_blocks, block_size, num_heads, head_dim], got {} dims",
                key_cache.shape.len()
            )));
        }

        let num_seqs = context_lens.data.len();
        let num_heads = query.shape[1];
        let head_dim = query.shape[2];
        let block_size = key_cache.shape[1];
        let max_blocks = block_tables.shape.get(1).copied().unwrap_or(0);

        if num_seqs == 0 {
            return Ok(GpuBuffer {
                data: Vec::new(),
                shape: vec![0, num_heads, head_dim],
            });
        }

        trace!(
            num_seqs,
            num_heads,
            head_dim,
            block_size,
            max_blocks,
            max_context_len,
            "CudaPagedAttention::forward"
        );

        // -- upload host buffers to GPU as f32 --------------------------------
        let d_query = self.upload_f16_as_f32(&query.data)?;
        let d_key_cache = self.upload_f16_as_f32(&key_cache.data)?;
        let d_value_cache = self.upload_f16_as_f32(&value_cache.data)?;
        let d_block_tables = self.upload_i32(&block_tables.data)?;
        let d_context_lens = self.upload_i32(&context_lens.data)?;

        // -- allocate output on GPU -------------------------------------------
        let output_len = num_seqs * num_heads * head_dim;
        let d_output: CudaSlice<f32> = self
            .device
            .alloc_zeros(output_len)
            .map_err(|e| LLMError::GpuError(format!("output alloc failed: {e}")))?;

        // -- launch configuration ---------------------------------------------
        // Grid:  (num_seqs, num_heads, 1)
        // Block: (head_dim, 1, 1)
        // Shared: block_size * sizeof(f32) + head_dim * sizeof(f32)
        let grid = (num_seqs as u32, num_heads as u32, 1u32);
        let block = (head_dim as u32, 1u32, 1u32);
        let shared_mem_bytes = ((block_size + head_dim) * std::mem::size_of::<f32>()) as u32;

        let cfg = LaunchConfig {
            grid_dim: grid,
            block_dim: block,
            shared_mem_bytes,
        };

        let func = self
            .device
            .get_func(MODULE_NAME, KERNEL_NAME)
            .ok_or_else(|| {
                LLMError::GpuError(format!(
                    "kernel function '{KERNEL_NAME}' not found in module '{MODULE_NAME}'"
                ))
            })?;

        // SAFETY: All device pointers are valid CudaSlice allocations from this
        // device. The grid/block dimensions match the kernel's expectations
        // (one thread per head_dim element, one block per seq*head). Scalar
        // arguments are passed by value. Shared memory is sized to hold
        // block_size + head_dim floats as required by the kernel.
        unsafe {
            func.launch_on_stream(
                &self.stream,
                cfg,
                (
                    &d_output,
                    &d_query,
                    &d_key_cache,
                    &d_value_cache,
                    &d_block_tables,
                    &d_context_lens,
                    scale,
                    num_heads as i32,
                    head_dim as i32,
                    block_size as i32,
                    max_context_len as i32,
                    max_blocks as i32,
                ),
            )
            .map_err(|e| LLMError::GpuError(format!("paged_attention kernel launch failed: {e}")))?;
        }

        // -- synchronize and download -----------------------------------------
        self.device
            .wait_for(&self.stream)
            .map_err(|e| LLMError::GpuError(format!("stream sync failed: {e}")))?;

        let host_output: Vec<f32> = self
            .device
            .dtoh_sync_copy(&d_output)
            .map_err(|e| LLMError::GpuError(format!("dtoh output copy failed: {e}")))?;

        // Convert f32 output back to f16
        let f16_output: Vec<f16> = host_output.iter().map(|v| f16::from_f32(*v)).collect();

        Ok(GpuBuffer {
            data: f16_output,
            shape: vec![num_seqs, num_heads, head_dim],
        })
    }

    fn name(&self) -> &str {
        "CudaPagedAttention"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_paged_attention_name_and_trait() {
        // Verify the type implements AttentionBackend (compile-time check).
        fn _assert_backend<T: AttentionBackend>() {}
        _assert_backend::<CudaPagedAttention>();
    }
}
