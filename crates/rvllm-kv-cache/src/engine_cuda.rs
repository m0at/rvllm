//! CudaCacheEngine: allocates real GPU memory for KV cache blocks via cudarc.
//!
//! This is the CUDA-backed counterpart to [`CacheEngine`](super::CacheEngine),
//! which uses the abstract `GpuAllocator` trait. `CudaCacheEngine` works
//! directly with `cudarc::driver::CudaSlice<f32>` for zero-copy kernel
//! interop and avoids the overhead of the mock-gpu abstraction.

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice};
use tracing::{debug, info};

use rvllm_core::prelude::{BlockId, LLMError, Result};

/// Per-layer paged KV cache engine backed by real CUDA device memory.
///
/// Each transformer layer owns a `(key, value)` pair of `CudaSlice<f32>`
/// buffers. The buffers are logically divided into fixed-size blocks:
///
///   `[num_blocks, block_size, num_heads, head_dim]`  (flattened)
///
/// Block-level operations (copy, swap-in, swap-out) transfer data between
/// GPU and host without going through the `GpuBuffer` abstraction.
pub struct CudaCacheEngine {
    /// Per-layer (key_cache, value_cache) on GPU.
    gpu_cache: Vec<(CudaSlice<f32>, CudaSlice<f32>)>,
    /// CPU-side staging buffers for swap operations, per layer (key, value).
    cpu_cache: Vec<(Vec<f32>, Vec<f32>)>,
    num_layers: usize,
    num_heads: usize,
    head_dim: usize,
    block_size: usize,
    num_gpu_blocks: usize,
    num_cpu_blocks: usize,
    device: Arc<CudaDevice>,
}

impl CudaCacheEngine {
    /// Allocate GPU and CPU KV cache buffers for all layers.
    ///
    /// # Errors
    /// Returns `LLMError::GpuError` if any CUDA allocation fails.
    pub fn new(
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
        block_size: usize,
        num_gpu_blocks: usize,
        num_cpu_blocks: usize,
        device: Arc<CudaDevice>,
    ) -> Result<Self> {
        let elements_per_block = block_size * num_heads * head_dim;
        let gpu_total = num_gpu_blocks * elements_per_block;
        let cpu_total = num_cpu_blocks * elements_per_block;

        info!(
            num_layers,
            num_heads,
            head_dim,
            block_size,
            num_gpu_blocks,
            num_cpu_blocks,
            gpu_bytes = gpu_total * std::mem::size_of::<f32>() * 2,
            "CudaCacheEngine: allocating KV cache"
        );

        let mut gpu_cache = Vec::with_capacity(num_layers);
        let mut cpu_cache = Vec::with_capacity(num_layers);

        for layer in 0..num_layers {
            debug!(layer, gpu_total, "allocating CUDA KV cache");

            // SAFETY: cudarc alloc_zeros returns zero-initialized device memory.
            // No unsafe needed -- cudarc's safe API handles allocation.
            let key_gpu: CudaSlice<f32> = device.alloc_zeros(gpu_total).map_err(|e| {
                LLMError::GpuError(format!(
                    "CUDA key cache alloc failed layer {layer}: {e}"
                ))
            })?;
            let val_gpu: CudaSlice<f32> = device.alloc_zeros(gpu_total).map_err(|e| {
                LLMError::GpuError(format!(
                    "CUDA value cache alloc failed layer {layer}: {e}"
                ))
            })?;
            gpu_cache.push((key_gpu, val_gpu));

            debug!(layer, cpu_total, "allocating CPU staging cache");
            let key_cpu = vec![0.0f32; cpu_total];
            let val_cpu = vec![0.0f32; cpu_total];
            cpu_cache.push((key_cpu, val_cpu));
        }

        Ok(Self {
            gpu_cache,
            cpu_cache,
            num_layers,
            num_heads,
            head_dim,
            block_size,
            num_gpu_blocks,
            num_cpu_blocks,
            device,
        })
    }

    /// Number of f32 elements per cache block.
    pub fn elements_per_block(&self) -> usize {
        self.block_size * self.num_heads * self.head_dim
    }

    /// Reference to the CUDA device.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Number of transformer layers.
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Number of attention heads.
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Dimension of each attention head.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Tokens per cache block.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Number of allocated GPU blocks.
    pub fn num_gpu_blocks(&self) -> usize {
        self.num_gpu_blocks
    }

    /// Number of allocated CPU staging blocks.
    pub fn num_cpu_blocks(&self) -> usize {
        self.num_cpu_blocks
    }

    /// Access the per-layer GPU cache slices.
    pub fn gpu_cache(&self) -> &[(CudaSlice<f32>, CudaSlice<f32>)] {
        &self.gpu_cache
    }

    /// Mutable access to the per-layer GPU cache slices.
    pub fn gpu_cache_mut(&mut self) -> &mut [(CudaSlice<f32>, CudaSlice<f32>)] {
        &mut self.gpu_cache
    }

    /// Zero all GPU KV cache buffers. Call between requests to prevent stale data.
    pub fn zero_cache(&mut self) -> Result<()> {
        for (key, val) in &mut self.gpu_cache {
            let k_len = cudarc::driver::DeviceSlice::len(&*key);
            let v_len = cudarc::driver::DeviceSlice::len(&*val);
            unsafe {
                cudarc::driver::result::memset_d8_sync(
                    *cudarc::driver::DevicePtrMut::device_ptr_mut(key),
                    0,
                    k_len * std::mem::size_of::<f32>(),
                ).map_err(|e| LLMError::GpuError(format!("zero key cache: {e}")))?;
                cudarc::driver::result::memset_d8_sync(
                    *cudarc::driver::DevicePtrMut::device_ptr_mut(val),
                    0,
                    v_len * std::mem::size_of::<f32>(),
                ).map_err(|e| LLMError::GpuError(format!("zero val cache: {e}")))?;
            }
        }
        Ok(())
    }

    /// Copy blocks within GPU cache. Each `(src, dst)` pair copies a full
    /// block across all layers by round-tripping through the host.
    ///
    /// A kernel-based copy_blocks (Agent 9) can replace this with a direct
    /// device-to-device copy for better performance.
    pub fn copy_blocks(&mut self, mapping: &[(BlockId, BlockId)]) -> Result<()> {
        let epb = self.elements_per_block();

        for &(src_id, dst_id) in mapping {
            let src = src_id.0 as usize;
            let dst = dst_id.0 as usize;

            if src >= self.num_gpu_blocks || dst >= self.num_gpu_blocks {
                return Err(LLMError::GpuError(format!(
                    "copy_blocks: block index out of range (src={src}, dst={dst}, max={})",
                    self.num_gpu_blocks
                )));
            }

            let src_off = src * epb;
            let dst_off = dst * epb;

            for (key_buf, val_buf) in &mut self.gpu_cache {
                // Round-trip through host: dtoh full buffer, copy block, htod back.
                // This is correct but slow; Agent 9 provides the kernel-based path.
                let mut key_host = self.device.dtoh_sync_copy(key_buf).map_err(|e| {
                    LLMError::GpuError(format!("copy_blocks dtoh key: {e}"))
                })?;
                let src_slice: Vec<f32> = key_host[src_off..src_off + epb].to_vec();
                key_host[dst_off..dst_off + epb].copy_from_slice(&src_slice);
                self.device.htod_sync_copy_into(&key_host, key_buf).map_err(|e| {
                    LLMError::GpuError(format!("copy_blocks htod key: {e}"))
                })?;

                let mut val_host = self.device.dtoh_sync_copy(val_buf).map_err(|e| {
                    LLMError::GpuError(format!("copy_blocks dtoh val: {e}"))
                })?;
                let src_slice: Vec<f32> = val_host[src_off..src_off + epb].to_vec();
                val_host[dst_off..dst_off + epb].copy_from_slice(&src_slice);
                self.device.htod_sync_copy_into(&val_host, val_buf).map_err(|e| {
                    LLMError::GpuError(format!("copy_blocks htod val: {e}"))
                })?;
            }
        }

        debug!(pairs = mapping.len(), "CudaCacheEngine copy_blocks complete");
        Ok(())
    }

    /// Swap blocks from CPU staging cache into GPU cache.
    /// Each `(cpu_block, gpu_block)` copies CPU -> GPU across all layers.
    pub fn swap_in(&mut self, mapping: &[(BlockId, BlockId)]) -> Result<()> {
        let epb = self.elements_per_block();

        for &(cpu_id, gpu_id) in mapping {
            let cpu_idx = cpu_id.0 as usize;
            let gpu_idx = gpu_id.0 as usize;

            if cpu_idx >= self.num_cpu_blocks {
                return Err(LLMError::GpuError(format!(
                    "swap_in: CPU block {cpu_idx} out of range (max={})",
                    self.num_cpu_blocks
                )));
            }
            if gpu_idx >= self.num_gpu_blocks {
                return Err(LLMError::GpuError(format!(
                    "swap_in: GPU block {gpu_idx} out of range (max={})",
                    self.num_gpu_blocks
                )));
            }

            let cpu_off = cpu_idx * epb;
            let gpu_off = gpu_idx * epb;

            for (layer_idx, ((key_gpu, val_gpu), (key_cpu, val_cpu))) in self
                .gpu_cache
                .iter_mut()
                .zip(self.cpu_cache.iter())
                .enumerate()
            {
                // Read full GPU buffer, overwrite the target block from CPU, write back.
                let mut key_host = self.device.dtoh_sync_copy(key_gpu).map_err(|e| {
                    LLMError::GpuError(format!("swap_in dtoh key layer {layer_idx}: {e}"))
                })?;
                key_host[gpu_off..gpu_off + epb]
                    .copy_from_slice(&key_cpu[cpu_off..cpu_off + epb]);
                self.device.htod_sync_copy_into(&key_host, key_gpu).map_err(|e| {
                    LLMError::GpuError(format!("swap_in htod key layer {layer_idx}: {e}"))
                })?;

                let mut val_host = self.device.dtoh_sync_copy(val_gpu).map_err(|e| {
                    LLMError::GpuError(format!("swap_in dtoh val layer {layer_idx}: {e}"))
                })?;
                val_host[gpu_off..gpu_off + epb]
                    .copy_from_slice(&val_cpu[cpu_off..cpu_off + epb]);
                self.device.htod_sync_copy_into(&val_host, val_gpu).map_err(|e| {
                    LLMError::GpuError(format!("swap_in htod val layer {layer_idx}: {e}"))
                })?;
            }
        }

        debug!(pairs = mapping.len(), "CudaCacheEngine swap_in complete");
        Ok(())
    }

    /// Swap blocks from GPU cache out to CPU staging cache.
    /// Each `(gpu_block, cpu_block)` copies GPU -> CPU across all layers.
    pub fn swap_out(&mut self, mapping: &[(BlockId, BlockId)]) -> Result<()> {
        let epb = self.elements_per_block();

        for &(gpu_id, cpu_id) in mapping {
            let gpu_idx = gpu_id.0 as usize;
            let cpu_idx = cpu_id.0 as usize;

            if gpu_idx >= self.num_gpu_blocks {
                return Err(LLMError::GpuError(format!(
                    "swap_out: GPU block {gpu_idx} out of range (max={})",
                    self.num_gpu_blocks
                )));
            }
            if cpu_idx >= self.num_cpu_blocks {
                return Err(LLMError::GpuError(format!(
                    "swap_out: CPU block {cpu_idx} out of range (max={})",
                    self.num_cpu_blocks
                )));
            }

            let gpu_off = gpu_idx * epb;
            let cpu_off = cpu_idx * epb;

            for (layer_idx, ((key_gpu, val_gpu), (key_cpu, val_cpu))) in self
                .gpu_cache
                .iter()
                .zip(self.cpu_cache.iter_mut())
                .enumerate()
            {
                let key_host = self.device.dtoh_sync_copy(key_gpu).map_err(|e| {
                    LLMError::GpuError(format!("swap_out dtoh key layer {layer_idx}: {e}"))
                })?;
                key_cpu[cpu_off..cpu_off + epb]
                    .copy_from_slice(&key_host[gpu_off..gpu_off + epb]);

                let val_host = self.device.dtoh_sync_copy(val_gpu).map_err(|e| {
                    LLMError::GpuError(format!("swap_out dtoh val layer {layer_idx}: {e}"))
                })?;
                val_cpu[cpu_off..cpu_off + epb]
                    .copy_from_slice(&val_host[gpu_off..gpu_off + epb]);

                let _ = layer_idx;
            }
        }

        debug!(pairs = mapping.len(), "CudaCacheEngine swap_out complete");
        Ok(())
    }

    /// Compute the maximum number of GPU blocks that fit in `available_bytes`
    /// given the current cache configuration.
    pub fn max_blocks_for_memory(
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
        block_size: usize,
        available_bytes: usize,
    ) -> usize {
        let elements_per_block = block_size * num_heads * head_dim;
        // key + value, each f32, across all layers
        let bytes_per_block = 2 * num_layers * elements_per_block * std::mem::size_of::<f32>();
        if bytes_per_block == 0 {
            return 0;
        }
        available_bytes / bytes_per_block
    }
}
