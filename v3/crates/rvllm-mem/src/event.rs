//! CUDA event wrapper for DtoH coordination.

use core::marker::PhantomData;

use rvllm_core::{CudaCtx, CudaErrorKind, Result, RvllmError};

use crate::cuda_owned::CudaOwned;
use crate::stream::Stream;

pub struct Event<'s> {
    raw: u64,
    stream: &'s Stream,
    _not_send_sync: PhantomData<*const ()>,
}

impl<'s> Event<'s> {
    pub fn host_stub(stream: &'s Stream) -> Self {
        Self {
            raw: 0,
            stream,
            _not_send_sync: PhantomData,
        }
    }

    #[cfg(feature = "cuda")]
    pub fn new(stream: &'s Stream) -> Result<Self> {
        use cudarc::driver::sys::*;
        let mut ev: CUevent = std::ptr::null_mut();
        let r = unsafe { cuEventCreate(&mut ev, CUevent_flags::CU_EVENT_DISABLE_TIMING as u32) };
        if r != CUresult::CUDA_SUCCESS {
            return Err(RvllmError::cuda(
                "cuEventCreate",
                CudaErrorKind::EventFailed,
                CudaCtx {
                    stream: stream.raw(),
                    kernel: "cuEventCreate",
                    launch: None,
                    device: -1,
                },
            ));
        }
        Ok(Self {
            raw: ev as u64,
            stream,
            _not_send_sync: PhantomData,
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub fn new(stream: &'s Stream) -> Result<Self> {
        Ok(Self::host_stub(stream))
    }

    pub fn raw(&self) -> u64 {
        self.raw
    }

    pub fn record(&mut self) -> Result<()> {
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            if self.raw != 0 {
                let r = cuEventRecord(
                    self.raw as CUevent,
                    self.stream.raw() as CUstream,
                );
                if r != CUresult::CUDA_SUCCESS {
                    return Err(RvllmError::cuda(
                        "cuEventRecord",
                        CudaErrorKind::EventFailed,
                        CudaCtx {
                            stream: self.stream.raw(),
                            kernel: "cuEventRecord",
                            launch: None,
                            device: -1,
                        },
                    ));
                }
            }
        }
        Ok(())
    }

    pub fn synchronize(&self) -> Result<()> {
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            if self.raw != 0 {
                let r = cuEventSynchronize(self.raw as CUevent);
                if r != CUresult::CUDA_SUCCESS {
                    return Err(RvllmError::cuda(
                        "cuEventSynchronize",
                        CudaErrorKind::EventFailed,
                        CudaCtx {
                            stream: self.stream.raw(),
                            kernel: "cuEventSynchronize",
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

impl<'s> CudaOwned for Event<'s> {
    fn stream_for_fence(&self) -> &Stream {
        self.stream
    }
}

impl<'s> Drop for Event<'s> {
    fn drop(&mut self) {
        self.fence_then_destroy();
        #[cfg(feature = "cuda")]
        unsafe {
            if self.raw != 0 {
                let _ = cudarc::driver::sys::cuEventDestroy_v2(
                    self.raw as cudarc::driver::sys::CUevent,
                );
            }
        }
    }
}
