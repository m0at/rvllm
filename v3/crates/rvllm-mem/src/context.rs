//! CUDA context initialization.
//!
//! Called once at engine init. Under `feature = "cuda"`, drives
//! `cuInit(0)` + `cuDeviceGet` + primary-context retain. Under no-cuda
//! it's a trivial host value so the types compile.
//!
//! Uses `cuDevicePrimaryCtxRetain` + `cuCtxSetCurrent` instead of
//! legacy `cuCtxCreate_v2`. Rationale: cudarc 0.19 only cfg-wraps
//! `cuCtxCreate_v2` for CUDA toolkits 11.07..12.09, so building with
//! `feature = "cuda"` on a CUDA 13 host fails to resolve that symbol.
//! Primary-context retain is the modern API (what the CUDA runtime
//! itself uses), has no cudarc cfg gate, and is ABI-stable across
//! CUDA 11 / 12 / 13. No behavioural change for the engine — rvllm
//! uses exactly one context for the lifetime of the process anyway.

use rvllm_core::{CudaCtx, CudaErrorKind, Result, RvllmError};

#[derive(Debug)]
pub struct CudaContextHandle {
    pub(crate) device: i32,
    #[cfg(feature = "cuda")]
    pub(crate) cu_device: cudarc::driver::sys::CUdevice,
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
        // Retain the primary context (ref-counted; Release in Drop) and
        // make it current on this thread. Modern replacement for
        // `cuCtxCreate_v2` — see module docs.
        let mut ctx: CUcontext = std::ptr::null_mut();
        let r = unsafe { cuDevicePrimaryCtxRetain(&mut ctx, dev) };
        if r != CUresult::CUDA_SUCCESS {
            return Err(RvllmError::cuda(
                "cuDevicePrimaryCtxRetain",
                CudaErrorKind::Other,
                CudaCtx {
                    stream: 0,
                    kernel: "cuDevicePrimaryCtxRetain",
                    launch: None,
                    device,
                },
            ));
        }
        let r = unsafe { cuCtxSetCurrent(ctx) };
        if r != CUresult::CUDA_SUCCESS {
            // Release the ref we just took so we don't leak on the
            // error path.
            unsafe {
                let _ = cuDevicePrimaryCtxRelease_v2(dev);
            }
            return Err(RvllmError::cuda(
                "cuCtxSetCurrent",
                CudaErrorKind::Other,
                CudaCtx {
                    stream: 0,
                    kernel: "cuCtxSetCurrent",
                    launch: None,
                    device,
                },
            ));
        }
        Ok(Self {
            device,
            cu_device: dev,
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
            cu_device: 0,
            #[cfg(feature = "cuda")]
            _ctx: std::ptr::null_mut(),
            _not_send_sync: core::marker::PhantomData,
        }
    }

    pub fn device(&self) -> i32 {
        self.device
    }

    /// Query the device's compute capability as `(major, minor)`.
    ///
    /// Drives `cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_{MAJOR,MINOR})`.
    /// Callers use this to pick the matching `kernels/<sm_*>/` subdirectory; a
    /// device whose compute capability has no PTX build should be rejected at
    /// bring-up (no silent fallback).
    #[cfg(feature = "cuda")]
    pub fn compute_capability(&self) -> Result<(i32, i32)> {
        use cudarc::driver::sys::*;
        let ordinal = self.device;
        let mut dev: CUdevice = 0;
        let r = unsafe { cuDeviceGet(&mut dev, ordinal) };
        if r != CUresult::CUDA_SUCCESS {
            return Err(RvllmError::cuda(
                "cuDeviceGet",
                CudaErrorKind::Other,
                CudaCtx {
                    stream: 0,
                    kernel: "cuDeviceGet",
                    launch: None,
                    device: ordinal,
                },
            ));
        }
        let mut major: i32 = 0;
        let r = unsafe {
            cuDeviceGetAttribute(
                &mut major,
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                dev,
            )
        };
        if r != CUresult::CUDA_SUCCESS {
            return Err(RvllmError::cuda(
                "cuDeviceGetAttribute(CC_MAJOR)",
                CudaErrorKind::Other,
                CudaCtx {
                    stream: 0,
                    kernel: "cuDeviceGetAttribute",
                    launch: None,
                    device: ordinal,
                },
            ));
        }
        let mut minor: i32 = 0;
        let r = unsafe {
            cuDeviceGetAttribute(
                &mut minor,
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                dev,
            )
        };
        if r != CUresult::CUDA_SUCCESS {
            return Err(RvllmError::cuda(
                "cuDeviceGetAttribute(CC_MINOR)",
                CudaErrorKind::Other,
                CudaCtx {
                    stream: 0,
                    kernel: "cuDeviceGetAttribute",
                    launch: None,
                    device: ordinal,
                },
            ));
        }
        Ok((major, minor))
    }

    /// Host-stub compute capability. Returns an error: callers that need
    /// a real CC must run under `feature = "cuda"`.
    #[cfg(not(feature = "cuda"))]
    pub fn compute_capability(&self) -> Result<(i32, i32)> {
        Err(RvllmError::cuda(
            "compute_capability",
            CudaErrorKind::Other,
            CudaCtx {
                stream: 0,
                kernel: "compute_capability",
                launch: None,
                device: self.device,
            },
        ))
    }
}

#[cfg(feature = "cuda")]
impl Drop for CudaContextHandle {
    fn drop(&mut self) {
        if !self._ctx.is_null() {
            // Release our ref on the primary context (matches the
            // Retain in `init`). The host_stub path leaves cu_device=0
            // and _ctx=null, so this branch never runs for stubs.
            unsafe {
                let _ = cudarc::driver::sys::cuDevicePrimaryCtxRelease_v2(self.cu_device);
            }
        }
    }
}
