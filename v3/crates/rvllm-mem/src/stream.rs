//! CUDA compute stream wrapper.

use core::marker::PhantomData;

use rvllm_core::{CudaCtx, CudaErrorKind, Result, RvllmError};

use crate::context::CudaContextHandle;
use crate::cuda_owned::CudaOwned;

pub struct Stream {
    raw: u64,
    _not_send_sync: PhantomData<*const ()>,
}

impl Stream {
    pub fn host_stub() -> Self {
        Self {
            raw: 0,
            _not_send_sync: PhantomData,
        }
    }

    #[cfg(feature = "cuda")]
    pub fn new(ctx: &CudaContextHandle) -> Result<Self> {
        use cudarc::driver::sys::*;
        let mut s: CUstream = std::ptr::null_mut();
        let r = unsafe { cuStreamCreate(&mut s, CUstream_flags::CU_STREAM_NON_BLOCKING as u32) };
        if r != CUresult::CUDA_SUCCESS {
            return Err(RvllmError::cuda(
                "cuStreamCreate",
                CudaErrorKind::StreamFailed,
                CudaCtx {
                    stream: 0,
                    kernel: "cuStreamCreate",
                    launch: None,
                    device: ctx.device(),
                },
            ));
        }
        Ok(Self {
            raw: s as u64,
            _not_send_sync: PhantomData,
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub fn new(_ctx: &CudaContextHandle) -> Result<Self> {
        Ok(Self::host_stub())
    }

    pub fn raw(&self) -> u64 {
        self.raw
    }

    pub fn fence(&self) -> Result<()> {
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            if self.raw != 0 {
                let r = cuStreamSynchronize(self.raw as CUstream);
                if r != CUresult::CUDA_SUCCESS {
                    return Err(RvllmError::cuda(
                        "cuStreamSynchronize",
                        CudaErrorKind::StreamFailed,
                        CudaCtx {
                            stream: self.raw,
                            kernel: "fence",
                            launch: None,
                            device: -1,
                        },
                    ));
                }
            }
        }
        Ok(())
    }
}

impl CudaOwned for Stream {
    fn stream_for_fence(&self) -> &Stream {
        self
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        unsafe {
            if self.raw != 0 {
                let _ = cudarc::driver::sys::cuStreamDestroy_v2(
                    self.raw as cudarc::driver::sys::CUstream,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_stub_fences() {
        let s = Stream::host_stub();
        assert!(s.fence().is_ok());
        assert_eq!(s.raw(), 0);
    }
}
