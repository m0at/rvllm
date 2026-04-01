//! Pinned (page-locked) host memory for faster HtoD/DtoH transfers.
//!
//! CUDA pinned memory enables DMA transfers that bypass the CPU page tables,
//! achieving ~2x higher bandwidth compared to pageable memory. This is critical
//! for the embedding lookup upload, logits download, and any swap operations.
//!
//! Under `mock-gpu`, this falls back to normal heap allocation.
//! Under `cuda`, uses `cuMemAllocHost` for true pinned memory.

use bytemuck::Pod;

use crate::Result;

/// A buffer backed by page-locked (pinned) host memory.
///
/// On CUDA systems, this memory is registered with the GPU driver for DMA,
/// providing ~2x faster HtoD/DtoH transfer rates. On mock-gpu, this is a
/// regular heap allocation.
pub struct PinnedBuffer<T: Pod + Send> {
    #[cfg(feature = "cuda")]
    ptr: *mut T,
    #[cfg(feature = "cuda")]
    len: usize,
    #[cfg(all(feature = "mock-gpu", not(feature = "cuda")))]
    data: Vec<T>,
    _marker: std::marker::PhantomData<T>,
}

// SAFETY: Pinned memory is a host allocation; it is safe to send/sync
// just like Vec<T>. The CUDA driver handles the DMA registration.
unsafe impl<T: Pod + Send> Send for PinnedBuffer<T> {}
unsafe impl<T: Pod + Send> Sync for PinnedBuffer<T> {}

impl<T: Pod + Send> PinnedBuffer<T> {
    /// Allocate a pinned buffer of `count` elements, zeroed.
    pub fn new(count: usize) -> Result<Self> {
        if count == 0 {
            #[cfg(feature = "cuda")]
            return Ok(Self {
                ptr: std::ptr::null_mut(),
                len: 0,
                _marker: std::marker::PhantomData,
            });
            #[cfg(all(feature = "mock-gpu", not(feature = "cuda")))]
            return Ok(Self {
                data: Vec::new(),
                _marker: std::marker::PhantomData,
            });
        }

        #[cfg(feature = "cuda")]
        {
            let bytes = count * std::mem::size_of::<T>();
            let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();

            // SAFETY: cuMemAllocHost allocates page-locked memory on the host.
            // The pointer is valid until cuMemFreeHost is called (in Drop).
            let result = unsafe { cudarc::driver::sys::lib().cuMemAllocHost_v2(&mut ptr, bytes) };

            if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                return Err(crate::LLMError::MemoryError(format!(
                    "cuMemAllocHost failed for {} bytes: {:?}",
                    bytes, result
                )));
            }

            // Zero the memory
            // SAFETY: ptr is a valid allocation of `bytes` size.
            unsafe {
                std::ptr::write_bytes(ptr as *mut u8, 0, bytes);
            }

            Ok(Self {
                ptr: ptr as *mut T,
                len: count,
                _marker: std::marker::PhantomData,
            })
        }

        #[cfg(all(feature = "mock-gpu", not(feature = "cuda")))]
        {
            Ok(Self {
                data: vec![T::zeroed(); count],
                _marker: std::marker::PhantomData,
            })
        }
    }

    pub fn len(&self) -> usize {
        #[cfg(feature = "cuda")]
        {
            self.len
        }
        #[cfg(all(feature = "mock-gpu", not(feature = "cuda")))]
        {
            self.data.len()
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn size_bytes(&self) -> usize {
        self.len() * std::mem::size_of::<T>()
    }

    pub fn as_slice(&self) -> &[T] {
        #[cfg(feature = "cuda")]
        {
            if self.ptr.is_null() || self.len == 0 {
                return &[];
            }
            // SAFETY: ptr was allocated with cuMemAllocHost for self.len elements.
            unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
        }
        #[cfg(all(feature = "mock-gpu", not(feature = "cuda")))]
        {
            &self.data
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        #[cfg(feature = "cuda")]
        {
            if self.ptr.is_null() || self.len == 0 {
                return &mut [];
            }
            // SAFETY: ptr was allocated with cuMemAllocHost for self.len elements.
            unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
        }
        #[cfg(all(feature = "mock-gpu", not(feature = "cuda")))]
        {
            &mut self.data
        }
    }

    pub fn as_ptr(&self) -> *const T {
        #[cfg(feature = "cuda")]
        {
            self.ptr as *const T
        }
        #[cfg(all(feature = "mock-gpu", not(feature = "cuda")))]
        {
            self.data.as_ptr()
        }
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        #[cfg(feature = "cuda")]
        {
            self.ptr
        }
        #[cfg(all(feature = "mock-gpu", not(feature = "cuda")))]
        {
            self.data.as_mut_ptr()
        }
    }

    /// Copy from a host slice into this pinned buffer.
    pub fn copy_from_slice(&mut self, src: &[T]) -> Result<()> {
        if src.len() != self.len() {
            return Err(crate::LLMError::MemoryError(format!(
                "PinnedBuffer copy_from_slice: src len {} != buf len {}",
                src.len(),
                self.len()
            )));
        }
        self.as_mut_slice().copy_from_slice(src);
        Ok(())
    }

    /// Copy this pinned buffer to a new Vec.
    pub fn to_vec(&self) -> Vec<T> {
        self.as_slice().to_vec()
    }
}

impl<T: Pod + Send> Drop for PinnedBuffer<T> {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        {
            if !self.ptr.is_null() && self.len > 0 {
                // SAFETY: ptr was allocated with cuMemAllocHost.
                unsafe {
                    let _ =
                        cudarc::driver::sys::lib().cuMemFreeHost(self.ptr as *mut std::ffi::c_void);
                }
            }
        }
        // mock-gpu: Vec drops automatically
    }
}

/// A reusable pool of pinned buffers for amortizing allocation cost.
///
/// Keeps a stack of previously-used buffers. When a buffer of the right
/// size is requested, it pops from the pool instead of allocating.
/// When returned, buffers go back onto the pool.
pub struct PinnedPool<T: Pod + Send> {
    buffers: parking_lot::Mutex<Vec<PinnedBuffer<T>>>,
    elem_count: usize,
}

impl<T: Pod + Send> PinnedPool<T> {
    /// Create a pool that manages buffers of `elem_count` elements each.
    pub fn new(elem_count: usize) -> Self {
        Self {
            buffers: parking_lot::Mutex::new(Vec::new()),
            elem_count,
        }
    }

    /// Pre-allocate `n` buffers into the pool.
    pub fn warm(&self, n: usize) -> Result<()> {
        let mut pool = self.buffers.lock();
        for _ in 0..n {
            pool.push(PinnedBuffer::new(self.elem_count)?);
        }
        Ok(())
    }

    /// Get a buffer from the pool, or allocate a new one.
    pub fn acquire(&self) -> Result<PinnedBuffer<T>> {
        let mut pool = self.buffers.lock();
        match pool.pop() {
            Some(buf) => Ok(buf),
            None => PinnedBuffer::new(self.elem_count),
        }
    }

    /// Return a buffer to the pool for reuse.
    pub fn release(&self, buf: PinnedBuffer<T>) {
        if buf.len() == self.elem_count {
            let mut pool = self.buffers.lock();
            pool.push(buf);
        }
        // Wrong-sized buffers are dropped (freed).
    }

    /// Number of buffers currently in the pool (not in use).
    pub fn available(&self) -> usize {
        self.buffers.lock().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_buffer_basic() {
        let mut buf = PinnedBuffer::<f32>::new(16).unwrap();
        assert_eq!(buf.len(), 16);
        assert_eq!(buf.size_bytes(), 64);

        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        buf.copy_from_slice(&data).unwrap();
        assert_eq!(buf.to_vec(), data);
    }

    #[test]
    fn pinned_buffer_empty() {
        let buf = PinnedBuffer::<u8>::new(0).unwrap();
        assert!(buf.is_empty());
        assert_eq!(buf.size_bytes(), 0);
    }

    #[test]
    fn pinned_pool_reuse() {
        let pool = PinnedPool::<f32>::new(64);
        pool.warm(2).unwrap();
        assert_eq!(pool.available(), 2);

        let buf1 = pool.acquire().unwrap();
        assert_eq!(pool.available(), 1);
        assert_eq!(buf1.len(), 64);

        pool.release(buf1);
        assert_eq!(pool.available(), 2);
    }

    #[test]
    fn pinned_buffer_size_mismatch() {
        let mut buf = PinnedBuffer::<f32>::new(4).unwrap();
        assert!(buf.copy_from_slice(&[1.0, 2.0]).is_err());
    }
}
