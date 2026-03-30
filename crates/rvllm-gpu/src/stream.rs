//! Async GPU stream wrapper.

#[cfg(feature = "cuda")]
use cudarc::driver::CudaContext;
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
    context: Arc<CudaContext>,
    #[cfg(feature = "cuda")]
    stream: Arc<cudarc::driver::CudaStream>,
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
        let context = CudaContext::new(device_id)
            .map_err(|e| crate::LLMError::MemoryError(format!("CUDA device init failed: {e}")))?;
        let stream = context.new_stream().map_err(|e| {
            crate::LLMError::MemoryError(format!("CUDA stream creation failed: {e}"))
        })?;
        Ok(Self {
            device_id,
            context,
            stream,
        })
    }

    /// Wrap an existing `Arc<CudaContext>` + `Arc<CudaStream>` without creating new resources.
    /// Used so the GPU worker can replay graphs on the model runner's stream.
    #[cfg(feature = "cuda")]
    pub fn from_arc(
        device_id: usize,
        context: Arc<CudaContext>,
        stream: Arc<cudarc::driver::CudaStream>,
    ) -> Self {
        Self { device_id, context, stream }
    }

    pub fn synchronize(&self) -> Result<()> {
        tracing::trace!(device_id = self.device_id, "synchronizing stream");
        #[cfg(feature = "cuda")]
        {
            self.stream.synchronize().map_err(|e| {
                crate::LLMError::MemoryError(format!("CUDA stream sync failed: {e}"))
            })?;
        }
        Ok(())
    }

    pub fn device_id(&self) -> usize {
        self.device_id
    }

    #[cfg(feature = "cuda")]
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    #[cfg(feature = "cuda")]
    pub fn cuda_stream(&self) -> &Arc<cudarc::driver::CudaStream> {
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
