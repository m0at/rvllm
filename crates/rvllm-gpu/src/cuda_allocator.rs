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
        let device = Arc::new(device);
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

        let slice = self.device.alloc_zeros::<T>(count).map_err(|e| {
            crate::LLMError::MemoryError(format!("CUDA alloc failed ({bytes} bytes): {e}"))
        })?;

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
        let (free, total) = cudarc::driver::result::mem_get_info().map_err(|e| {
            crate::LLMError::MemoryError(format!("CUDA mem_get_info failed: {e}"))
        })?;
        Ok(MemoryInfo {
            total,
            free,
            used: total.saturating_sub(free),
        })
    }
}
