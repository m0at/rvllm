//! Pinned (page-locked) host buffer + double-buffer DtoH pool.

use core::marker::PhantomData;

use rvllm_core::{CudaCtx, CudaErrorKind, Result, RvllmError};

pub struct PinnedBuf<T> {
    ptr: *mut T,
    len: usize,
    owns_cuda: bool,
    _not_send_sync: PhantomData<*const ()>,
}

unsafe impl<T: Send> Send for PinnedBuf<T> {}

impl<T: Default + Clone> PinnedBuf<T> {
    /// Allocate via `cuMemAllocHost_v2` when cuda feature is on;
    /// otherwise a heap Box<[T]>.
    pub fn new(len: usize) -> Result<Self> {
        if len == 0 {
            return Ok(Self {
                ptr: core::ptr::null_mut(),
                len: 0,
                owns_cuda: false,
                _not_send_sync: PhantomData,
            });
        }

        #[cfg(feature = "cuda")]
        {
            use cudarc::driver::sys::*;
            // `cuMemAllocHost_v2` requires a current CUDA context on the
            // calling thread. Callers that use this type via the engine's
            // normal bring-up path already have one; direct consumers
            // (including the unit tests in this crate, which is how the
            // workspace `cargo test` tripped here — rvllm-mem tests run
            // after rvllm-loader leaves no CUDA state behind) don't. Do
            // a one-shot init + primary-context retain so the allocation
            // succeeds deterministically. Both calls are idempotent and
            // safe to run many times per process.
            unsafe {
                let _ = cuInit(0);
                let mut dev: CUdevice = 0;
                if cuDeviceGet(&mut dev, 0) == CUresult::CUDA_SUCCESS {
                    let mut ctx: CUcontext = core::ptr::null_mut();
                    if cuDevicePrimaryCtxRetain(&mut ctx, dev) == CUresult::CUDA_SUCCESS {
                        let _ = cuCtxSetCurrent(ctx);
                    }
                }
            }
            let bytes = len * core::mem::size_of::<T>();
            let mut p: *mut core::ffi::c_void = core::ptr::null_mut();
            let r = unsafe { cuMemAllocHost_v2(&mut p, bytes) };
            if r != CUresult::CUDA_SUCCESS {
                return Err(RvllmError::cuda(
                    "cuMemAllocHost_v2",
                    CudaErrorKind::AllocFailed,
                    CudaCtx {
                        stream: 0,
                        kernel: "cuMemAllocHost_v2",
                        launch: None,
                        device: -1,
                    },
                ));
            }
            // Zero + default-init.
            unsafe {
                core::ptr::write_bytes(p as *mut u8, 0, bytes);
            }
            Ok(Self {
                ptr: p as *mut T,
                len,
                owns_cuda: true,
                _not_send_sync: PhantomData,
            })
        }

        #[cfg(not(feature = "cuda"))]
        {
            let data: Box<[T]> = vec![T::default(); len].into_boxed_slice();
            let ptr = Box::into_raw(data) as *mut T;
            Ok(Self {
                ptr,
                len,
                owns_cuda: false,
                _not_send_sync: PhantomData,
            })
        }
    }
}

impl<T> PinnedBuf<T> {
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn as_slice(&self) -> &[T] {
        if self.ptr.is_null() {
            &[]
        } else {
            unsafe { core::slice::from_raw_parts(self.ptr, self.len) }
        }
    }
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        if self.ptr.is_null() {
            &mut []
        } else {
            unsafe { core::slice::from_raw_parts_mut(self.ptr, self.len) }
        }
    }
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
}

impl<T> Drop for PinnedBuf<T> {
    fn drop(&mut self) {
        if self.ptr.is_null() || self.len == 0 {
            return;
        }
        #[cfg(feature = "cuda")]
        unsafe {
            if self.owns_cuda {
                let _ = cudarc::driver::sys::cuMemFreeHost(self.ptr as *mut core::ffi::c_void);
                return;
            }
        }
        #[cfg(not(feature = "cuda"))]
        unsafe {
            let _ = Box::<[T]>::from_raw(core::slice::from_raw_parts_mut(self.ptr, self.len));
        }
    }
}

pub struct PinnedPool<T> {
    buffers: [PinnedBuf<T>; 2],
    write_idx: u8,
}

impl<T: Default + Clone> PinnedPool<T> {
    pub fn new(len_per_buf: usize) -> Result<Self> {
        Ok(Self {
            buffers: [PinnedBuf::new(len_per_buf)?, PinnedBuf::new(len_per_buf)?],
            write_idx: 0,
        })
    }
}

impl<T> PinnedPool<T> {
    pub fn write_idx(&self) -> usize {
        self.write_idx as usize
    }
    pub fn read_idx(&self) -> usize {
        1 - self.write_idx as usize
    }
    pub fn write_buf_mut(&mut self) -> &mut PinnedBuf<T> {
        &mut self.buffers[self.write_idx as usize]
    }
    pub fn read_buf(&self) -> &PinnedBuf<T> {
        &self.buffers[1 - self.write_idx as usize]
    }
    pub fn flip(&mut self) {
        self.write_idx = 1 - self.write_idx;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_flips_between_buffers() {
        let mut p: PinnedPool<i32> = PinnedPool::new(128).unwrap();
        assert_eq!(p.write_idx(), 0);
        assert_eq!(p.read_idx(), 1);
        p.flip();
        assert_eq!(p.write_idx(), 1);
        assert_eq!(p.read_idx(), 0);
    }

    #[test]
    fn buf_is_zero_initialized() {
        let b: PinnedBuf<i32> = PinnedBuf::new(16).unwrap();
        assert_eq!(b.len(), 16);
        assert!(b.as_slice().iter().all(|x| *x == 0));
    }
}
