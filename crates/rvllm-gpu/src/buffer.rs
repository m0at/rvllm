//! GPU memory buffer handle.

#[cfg(not(any(feature = "mock-gpu", feature = "cuda")))]
use std::marker::PhantomData;

use bytemuck::Pod;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut, DeviceSlice as _};
#[cfg(feature = "cuda")]
use std::sync::Arc;

use crate::Result;

/// Owning handle for a contiguous region of GPU memory.
///
/// Under `mock-gpu` this wraps a heap `Vec<T>`.
/// Under `cuda` this wraps a `CudaSlice<T>` from cudarc.
pub struct GpuBuffer<T: Pod + Send> {
    pub(crate) inner: GpuBufferInner<T>,
}

pub(crate) enum GpuBufferInner<T: Pod + Send> {
    #[cfg(all(feature = "mock-gpu", not(feature = "cuda")))]
    Mock {
        data: Vec<T>,
        on_drop: Option<Box<dyn FnOnce(usize) + Send>>,
    },
    #[cfg(feature = "cuda")]
    Cuda {
        slice: CudaSlice<T>,
        device: Arc<cudarc::driver::CudaDevice>,
    },
    #[cfg(not(any(feature = "mock-gpu", feature = "cuda")))]
    #[allow(dead_code)]
    Placeholder {
        ptr: *mut T,
        len: usize,
        _marker: PhantomData<T>,
    },
}

// SAFETY: CudaSlice is device memory managed by cudarc. For mock-gpu everything is a plain Vec.
unsafe impl<T: Pod + Send> Send for GpuBuffer<T> {}
unsafe impl<T: Pod + Send> Sync for GpuBuffer<T> {}

impl<T: Pod + Send> GpuBuffer<T> {
    pub fn len(&self) -> usize {
        match &self.inner {
            #[cfg(all(feature = "mock-gpu", not(feature = "cuda")))]
            GpuBufferInner::Mock { data, .. } => data.len(),
            #[cfg(feature = "cuda")]
            GpuBufferInner::Cuda { slice, .. } => slice.len(),
            #[cfg(not(any(feature = "mock-gpu", feature = "cuda")))]
            GpuBufferInner::Placeholder { len, .. } => *len,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn size_bytes(&self) -> usize {
        self.len() * std::mem::size_of::<T>()
    }

    pub fn as_ptr(&self) -> *const T {
        match &self.inner {
            #[cfg(all(feature = "mock-gpu", not(feature = "cuda")))]
            GpuBufferInner::Mock { data, .. } => data.as_ptr(),
            #[cfg(feature = "cuda")]
            GpuBufferInner::Cuda { slice, .. } => {
                // SAFETY: returns device pointer as raw ptr for kernel launches
                *slice.device_ptr() as *const T
            }
            #[cfg(not(any(feature = "mock-gpu", feature = "cuda")))]
            GpuBufferInner::Placeholder { ptr, .. } => *ptr as *const T,
        }
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        match &mut self.inner {
            #[cfg(all(feature = "mock-gpu", not(feature = "cuda")))]
            GpuBufferInner::Mock { data, .. } => data.as_mut_ptr(),
            #[cfg(feature = "cuda")]
            GpuBufferInner::Cuda { slice, .. } => {
                // SAFETY: returns device pointer as raw mut ptr for kernel launches
                *slice.device_ptr_mut() as *mut T
            }
            #[cfg(not(any(feature = "mock-gpu", feature = "cuda")))]
            GpuBufferInner::Placeholder { ptr, .. } => *ptr,
        }
    }

    pub fn copy_from_host(&mut self, src: &[T]) -> Result<()> {
        if src.len() != self.len() {
            return Err(crate::LLMError::MemoryError(format!(
                "copy_from_host: source len {} != buffer len {}",
                src.len(),
                self.len()
            )));
        }
        match &mut self.inner {
            #[cfg(all(feature = "mock-gpu", not(feature = "cuda")))]
            GpuBufferInner::Mock { data, .. } => {
                data.copy_from_slice(src);
                Ok(())
            }
            #[cfg(feature = "cuda")]
            GpuBufferInner::Cuda { slice, device } => {
                device
                    .bind_to_thread()
                    .map_err(|e| crate::LLMError::MemoryError(format!("CUDA bind failed: {e}")))?;
                unsafe {
                    cudarc::driver::result::memcpy_htod_sync(
                        *DevicePtrMut::device_ptr_mut(slice),
                        src,
                    )
                }
                .map_err(|e| crate::LLMError::MemoryError(format!("CUDA htod copy failed: {e}")))?;
                Ok(())
            }
            #[cfg(not(any(feature = "mock-gpu", feature = "cuda")))]
            GpuBufferInner::Placeholder { .. } => {
                unimplemented!("no GPU backend enabled");
            }
        }
    }

    pub fn copy_to_host(&self) -> Result<Vec<T>> {
        match &self.inner {
            #[cfg(all(feature = "mock-gpu", not(feature = "cuda")))]
            GpuBufferInner::Mock { data, .. } => Ok(data.clone()),
            #[cfg(feature = "cuda")]
            GpuBufferInner::Cuda { slice, device } => {
                device
                    .bind_to_thread()
                    .map_err(|e| crate::LLMError::MemoryError(format!("CUDA bind failed: {e}")))?;
                let len = cudarc::driver::DeviceSlice::len(slice);
                let mut host = vec![T::zeroed(); len];
                unsafe {
                    cudarc::driver::result::memcpy_dtoh_sync(
                        &mut host,
                        *DevicePtr::device_ptr(slice),
                    )
                }
                .map_err(|e| crate::LLMError::MemoryError(format!("CUDA dtoh copy failed: {e}")))?;
                Ok(host)
            }
            #[cfg(not(any(feature = "mock-gpu", feature = "cuda")))]
            GpuBufferInner::Placeholder { .. } => {
                unimplemented!("no GPU backend enabled");
            }
        }
    }
}

impl<T: Pod + Send> Drop for GpuBuffer<T> {
    fn drop(&mut self) {
        match &mut self.inner {
            #[cfg(all(feature = "mock-gpu", not(feature = "cuda")))]
            GpuBufferInner::Mock { data, on_drop } => {
                let bytes = data.len() * std::mem::size_of::<T>();
                if let Some(cb) = on_drop.take() {
                    cb(bytes);
                }
            }
            #[cfg(feature = "cuda")]
            GpuBufferInner::Cuda { .. } => {
                // cudarc CudaSlice handles deallocation on drop automatically
            }
            #[cfg(not(any(feature = "mock-gpu", feature = "cuda")))]
            GpuBufferInner::Placeholder { .. } => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<GpuBuffer<f32>>();
    }
}
