//! Generic `cuLaunchKernel` wrapper.
//!
//! Every fused-kernel launcher goes through `launch_raw` under feature
//! `cuda`. Under no-cuda it's a no-op so validation logic tests without
//! a GPU.

use rvllm_core::{CudaCtx, CudaErrorKind, Launch, Result, RvllmError};
use rvllm_kernels::KernelFn;

/// Low-level cuLaunchKernel wrapper. Args are opaque pointers to the
/// scalars/device-ptrs the kernel reads. Caller is responsible that
/// each entry in `args` points at memory that outlives this call.
///
/// # Safety
/// Caller must ensure `args` elements point at valid storage with
/// types matching the kernel's extern "C" signature.
pub unsafe fn launch_raw(
    kernel: KernelFn,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    shared_mem_bytes: u32,
    stream: u64,
    args: &[*mut core::ffi::c_void],
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        use cudarc::driver::sys::*;
        let r = cuLaunchKernel(
            kernel.raw() as CUfunction,
            grid.0,
            grid.1,
            grid.2,
            block.0,
            block.1,
            block.2,
            shared_mem_bytes,
            stream as CUstream,
            args.as_ptr() as *mut *mut core::ffi::c_void,
            core::ptr::null_mut(),
        );
        if r != CUresult::CUDA_SUCCESS {
            return Err(RvllmError::cuda(
                "cuLaunchKernel",
                CudaErrorKind::LaunchFailed,
                CudaCtx {
                    stream,
                    kernel: kernel.name(),
                    launch: Some(Launch {
                        grid,
                        block,
                        smem: shared_mem_bytes,
                    }),
                    device: -1,
                },
            ));
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (kernel, grid, block, shared_mem_bytes, stream, args);
    }
    Ok(())
}
