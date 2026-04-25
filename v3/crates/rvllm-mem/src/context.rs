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

/// Build a typed CUDA error for a failing driver call (context init,
/// device attribute read, primary-context release, …). All call sites
/// in this module share `stream: 0` + `launch: None` because none of
/// them are on the kernel-launch path. `driver_call` is the CUDA
/// function name that failed — it flows into both `op` (rvllm's
/// operation label) and `CudaCtx.kernel` because for a driver call
/// there is no separate "kernel" identity.
fn cuda_err(driver_call: &'static str, device: i32) -> RvllmError {
    RvllmError::cuda(
        driver_call,
        CudaErrorKind::Other,
        CudaCtx {
            stream: 0,
            kernel: driver_call,
            launch: None,
            device,
        },
    )
}

/// One-shot driver call: read a device attribute into an `i32`.
/// Factored out because we query two CC attributes + could grow more
/// later. Returns the typed `RvllmError` that `init` propagates.
#[cfg(feature = "cuda")]
fn device_attr(
    cu_device: cudarc::driver::sys::CUdevice,
    attr: cudarc::driver::sys::CUdevice_attribute,
    device_ordinal: i32,
    op: &'static str,
) -> Result<i32> {
    use cudarc::driver::sys::*;
    let mut value: i32 = 0;
    if unsafe { cuDeviceGetAttribute(&mut value, attr, cu_device) } != CUresult::CUDA_SUCCESS {
        return Err(cuda_err(op, device_ordinal));
    }
    Ok(value)
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

        // Read compute capability now — it's immutable for the
        // device, and downstream code (arch resolver, kernel picker)
        // asks repeatedly. One FFI round trip at init beats one per
        // call site forever.
        let cc_major = device_attr(
            dev,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            device,
            "cuDeviceGetAttribute(CC_MAJOR)",
        )?;
        let cc_minor = device_attr(
            dev,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            device,
            "cuDeviceGetAttribute(CC_MINOR)",
        )?;

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

    /// Rebind the retained primary context as current for the
    /// *calling* thread. `init` already does this on the thread it
    /// runs on; callers that dequeue requests on a long-lived worker
    /// thread should call this once per request as a belt-and-
    /// suspenders guard — empirically on GB10 a context binding that
    /// goes idle around `mpsc::Receiver::blocking_recv` can drop,
    /// producing `cuLaunchKernel` failures on the next launch. One
    /// driver call per request is negligible next to decode cost.
    ///
    /// `Ok(())` under `not(cuda)` — host stub.
    pub fn bind_to_current_thread(&self) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            use cudarc::driver::sys::*;
            if self._ctx.is_null() {
                return Ok(()); // host stub case
            }
            if unsafe { cuCtxSetCurrent(self._ctx) } != CUresult::CUDA_SUCCESS {
                return Err(cuda_err("cuCtxSetCurrent(rebind)", self.device));
            }
        }
        Ok(())
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

    /// Device compute capability `(major, minor)`, read once at init and
    /// cached. Cheap field access — no FFI on the call path. Callers
    /// pass this pair through `CompileTarget::from_compute_capability`
    /// to pick the matching `kernels/<sm_*>/` subdirectory; a device
    /// whose compute capability has no PTX build should be rejected at
    /// bring-up (no silent fallback).
    ///
    /// Only defined under `feature = "cuda"` — every call site is
    /// already cuda-gated (no-cuda builds don't have a real device to
    /// query). A `host_stub()` under cuda returns `(0, 0)`, which
    /// `CompileTarget::from_compute_capability` maps to `None` so the
    /// bring-up path fails closed.
    #[cfg(feature = "cuda")]
    #[inline]
    #[must_use]
    pub fn compute_capability(&self) -> (i32, i32) {
        self.compute_cap
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
