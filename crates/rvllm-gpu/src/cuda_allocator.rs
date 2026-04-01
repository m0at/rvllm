//! CUDA GPU allocator backed by cudarc.

use std::sync::Arc;

use bytemuck::Pod;
use cudarc::driver::CudaDevice;
use tracing::{debug, trace};

use crate::allocator::GpuAllocator;
use crate::buffer::{GpuBuffer, GpuBufferInner};
use crate::device::MemoryInfo;
use crate::Result;

pub struct CudaGpuAllocator {
    device: Arc<CudaDevice>,
}

impl CudaGpuAllocator {
    pub fn new(device_id: usize) -> Result<Self> {
        let device = CudaDevice::new(device_id).map_err(|e| {
            crate::LLMError::MemoryError(format!("CUDA device {device_id} init failed: {e}"))
        })?;
        debug!(device_id, "CudaGpuAllocator created");
        Ok(Self { device })
    }

    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }
}

impl GpuAllocator for CudaGpuAllocator {
    fn alloc<T: Pod + Send>(&self, count: usize) -> Result<GpuBuffer<T>> {
        let bytes = count * std::mem::size_of::<T>();
        trace!(bytes, count, "CUDA alloc");

        let slice = unsafe {
            self.device
                .bind_to_thread()
                .map_err(|e| crate::LLMError::MemoryError(format!("CUDA bind failed: {e}")))?;
            let cu_ptr = cudarc::driver::result::malloc_sync(bytes).map_err(|e| {
                crate::LLMError::MemoryError(format!("CUDA alloc failed ({bytes} bytes): {e}"))
            })?;
            cudarc::driver::result::memset_d8_sync(cu_ptr, 0, bytes).map_err(|e| {
                crate::LLMError::MemoryError(format!("CUDA memset failed ({bytes} bytes): {e}"))
            })?;
            self.device.upgrade_device_ptr::<T>(cu_ptr, count)
        };

        Ok(GpuBuffer {
            inner: GpuBufferInner::Cuda {
                slice,
                device: Arc::clone(&self.device),
            },
        })
    }

    fn free<T: Pod + Send>(&self, buf: GpuBuffer<T>) {
        // cudarc CudaSlice handles deallocation on drop
        drop(buf);
    }

    fn device_memory_info(&self) -> Result<MemoryInfo> {
        let (free, total) = cudarc::driver::result::mem_get_info()
            .map_err(|e| crate::LLMError::MemoryError(format!("CUDA mem_get_info failed: {e}")))?;
        Ok(MemoryInfo {
            total,
            free,
            used: total.saturating_sub(free),
        })
    }
}
