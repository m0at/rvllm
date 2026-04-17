//! CUDA context initialization.
//!
//! Called once at engine init. Under `feature = "cuda"`, drives
//! `cuInit(0)` + `cuDeviceGet` + `cuCtxCreate_v2`. Under no-cuda it's
//! a trivial host value so the types compile.

use rvllm_core::{CudaCtx, CudaErrorKind, Result, RvllmError};

#[derive(Debug)]
pub struct CudaContextHandle {
    pub(crate) device: i32,
    #[cfg(feature = "cuda")]
    pub(crate) _ctx: cudarc::driver::sys::CUcontext,
    // Pin to creating thread — context is not Send/Sync.
    _not_send_sync: core::marker::PhantomData<*const ()>,
}

impl CudaContextHandle {
    #[cfg(feature = "cuda")]
    pub fn init(device: i32) -> Result<Self> {
        use cudarc::driver::sys::*;
        let init_res = unsafe { cuInit(0) };
        if init_res != CUresult::CUDA_SUCCESS {
            return Err(RvllmError::cuda(
                "cuInit",
                CudaErrorKind::Other,
                CudaCtx {
                    stream: 0,
                    kernel: "cuInit",
                    launch: None,
                    device,
                },
            ));
        }
        let mut dev: CUdevice = 0;
        let r = unsafe { cuDeviceGet(&mut dev, device) };
        if r != CUresult::CUDA_SUCCESS {
            return Err(RvllmError::cuda(
                "cuDeviceGet",
                CudaErrorKind::Other,
                CudaCtx {
                    stream: 0,
                    kernel: "cuDeviceGet",
                    launch: None,
                    device,
                },
            ));
        }
        let mut ctx: CUcontext = std::ptr::null_mut();
        let r = unsafe { cuCtxCreate_v2(&mut ctx, 0, dev) };
        if r != CUresult::CUDA_SUCCESS {
            return Err(RvllmError::cuda(
                "cuCtxCreate_v2",
                CudaErrorKind::Other,
                CudaCtx {
                    stream: 0,
                    kernel: "cuCtxCreate_v2",
                    launch: None,
                    device,
                },
            ));
        }
        Ok(Self {
            device,
            _ctx: ctx,
            _not_send_sync: core::marker::PhantomData,
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub fn init(device: i32) -> Result<Self> {
        Ok(Self {
            device,
            _not_send_sync: core::marker::PhantomData,
        })
    }

    pub fn host_stub() -> Self {
        Self {
            device: -1,
            #[cfg(feature = "cuda")]
            _ctx: std::ptr::null_mut(),
            _not_send_sync: core::marker::PhantomData,
        }
    }

    pub fn device(&self) -> i32 {
        self.device
    }
}

#[cfg(feature = "cuda")]
impl Drop for CudaContextHandle {
    fn drop(&mut self) {
        if !self._ctx.is_null() {
            unsafe {
                let _ = cudarc::driver::sys::cuCtxDestroy_v2(self._ctx);
            }
        }
    }
}
