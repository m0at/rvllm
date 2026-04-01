//! Async GPU stream wrapper.

#[cfg(feature = "cuda")]
use cudarc::driver::CudaDevice;
#[cfg(feature = "cuda")]
use std::sync::Arc;

use crate::Result;

/// Handle to a GPU execution stream.
///
/// Under `mock-gpu` this is a no-op marker that records the device id.
/// Under `cuda` this wraps a `CudaStream` from cudarc.
pub struct GpuStream {
    device_id: usize,
    #[cfg(feature = "cuda")]
    device: Arc<CudaDevice>,
    #[cfg(feature = "cuda")]
    stream: cudarc::driver::CudaStream,
}

impl GpuStream {
    #[cfg(not(feature = "cuda"))]
    pub fn new(device_id: usize) -> Result<Self> {
        tracing::debug!(device_id, "creating GPU stream (mock)");
        Ok(Self { device_id })
    }

    #[cfg(feature = "cuda")]
    pub fn new(device_id: usize) -> Result<Self> {
        tracing::debug!(device_id, "creating CUDA GPU stream");
        let device = CudaDevice::new(device_id)
            .map_err(|e| crate::LLMError::MemoryError(format!("CUDA device init failed: {e}")))?;
        let stream = device.fork_default_stream().map_err(|e| {
            crate::LLMError::MemoryError(format!("CUDA stream creation failed: {e}"))
        })?;
        Ok(Self {
            device_id,
            device,
            stream,
        })
    }

    pub fn synchronize(&self) -> Result<()> {
        tracing::trace!(device_id = self.device_id, "synchronizing stream");
        #[cfg(feature = "cuda")]
        {
            self.device.wait_for(&self.stream).map_err(|e| {
                crate::LLMError::MemoryError(format!("CUDA stream sync failed: {e}"))
            })?;
        }
        Ok(())
    }

    pub fn device_id(&self) -> usize {
        self.device_id
    }

    #[cfg(feature = "cuda")]
    pub fn cuda_device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    #[cfg(feature = "cuda")]
    pub fn cuda_stream(&self) -> &cudarc::driver::CudaStream {
        &self.stream
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_and_sync() {
        let stream = GpuStream::new(0).unwrap();
        assert_eq!(stream.device_id(), 0);
        stream.synchronize().unwrap();
    }
}
