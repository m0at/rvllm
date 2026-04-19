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
    /// Compute capability `(major, minor)`. Queried once in `init` and
    /// cached — it can't change over the handle's lifetime, and callers
    /// (manifest resolver, kernel dispatcher, bench harness) ask
    /// repeatedly.
    #[cfg(feature = "cuda")]
    pub(crate) compute_cap: (i32, i32),
    // Pin to creating thread — context is not Send/Sync.
    _not_send_sync: core::marker::PhantomData<*const ()>,
}

/// Build a typed CUDA error. Every call site in this module passes
/// `stream: 0` + `launch: None` — the module is context-setup only,
/// not a launch path. Factored out so the error plumbing stays at
/// one line per failure branch.
fn cuda_err(op: &'static str, device: i32) -> RvllmError {
    RvllmError::cuda(
        op,
        CudaErrorKind::Other,
        CudaCtx {
            stream: 0,
            kernel: op,
            launch: None,
            device,
        },
    )
}

impl CudaContextHandle {
    #[cfg(feature = "cuda")]
    pub fn init(device: i32) -> Result<Self> {
        use cudarc::driver::sys::*;
        if unsafe { cuInit(0) } != CUresult::CUDA_SUCCESS {
            return Err(cuda_err("cuInit", device));
        }
        let mut dev: CUdevice = 0;
        if unsafe { cuDeviceGet(&mut dev, device) } != CUresult::CUDA_SUCCESS {
            return Err(cuda_err("cuDeviceGet", device));
        }
        // Retain the primary context (ref-counted; Release in Drop) and
        // make it current on this thread. Modern replacement for
        // `cuCtxCreate_v2` — see module docs.
        let mut ctx: CUcontext = std::ptr::null_mut();
        if unsafe { cuDevicePrimaryCtxRetain(&mut ctx, dev) } != CUresult::CUDA_SUCCESS {
            return Err(cuda_err("cuDevicePrimaryCtxRetain", device));
        }
        if unsafe { cuCtxSetCurrent(ctx) } != CUresult::CUDA_SUCCESS {
            // Release the ref we just took so we don't leak on error.
            unsafe {
                let _ = cuDevicePrimaryCtxRelease_v2(dev);
            }
            return Err(cuda_err("cuCtxSetCurrent", device));
        }
        Ok(Self {
            device,
            cu_device: dev,
            _ctx: ctx,
            compute_cap: (cc_major, cc_minor),
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
            #[cfg(feature = "cuda")]
            compute_cap: (0, 0),
            _not_send_sync: core::marker::PhantomData,
        }
    }

    #[inline]
    #[must_use]
    pub fn device(&self) -> i32 {
        self.device
    }

    /// Query the device's compute capability as `(major, minor)`.
    ///
    /// Drives `cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_{MAJOR,MINOR})`
    /// against the `CUdevice` handle we already resolved in `init` (so
    /// there's no redundant `cuDeviceGet` per call). Callers use this
    /// to pick the matching `kernels/<sm_*>/` subdirectory; a device
    /// whose compute capability has no PTX build should be rejected at
    /// bring-up (no silent fallback).
    #[cfg(feature = "cuda")]
    pub fn compute_capability(&self) -> Result<(i32, i32)> {
        use cudarc::driver::sys::*;
        let mut major: i32 = 0;
        if unsafe {
            cuDeviceGetAttribute(
                &mut major,
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                self.cu_device,
            )
        } != CUresult::CUDA_SUCCESS
        {
            return Err(cuda_err("cuDeviceGetAttribute(CC_MAJOR)", self.device));
        }
        let mut minor: i32 = 0;
        if unsafe {
            cuDeviceGetAttribute(
                &mut minor,
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                self.cu_device,
            )
        } != CUresult::CUDA_SUCCESS
        {
            return Err(cuda_err("cuDeviceGetAttribute(CC_MINOR)", self.device));
        }
        Ok((major, minor))
    }

    /// Host-stub compute capability. Returns an error: callers that need
    /// a real CC must run under `feature = "cuda"`.
    #[cfg(not(feature = "cuda"))]
    pub fn compute_capability(&self) -> Result<(i32, i32)> {
        Err(cuda_err("compute_capability", self.device))
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
