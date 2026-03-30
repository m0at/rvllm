//! Cooperative kernel launch support for persistent kernels.
//!
//! Wraps cuLaunchCooperativeKernel (all blocks must be co-resident)
//! and occupancy queries needed to determine the grid size.

use std::ffi::c_void;

use cudarc::driver::result::DriverError;
use cudarc::driver::sys::{self, CUfunction, CUstream};
use cudarc::driver::CudaContext;

/// Launch a kernel cooperatively (all blocks must be resident simultaneously).
///
/// # Safety
/// Caller must ensure kernel args match the kernel signature exactly.
pub unsafe fn launch_cooperative(
    func: CUfunction,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    shared_mem: u32,
    stream: CUstream,
    args: &mut [*mut c_void],
) -> Result<(), DriverError> {
    sys::cuLaunchCooperativeKernel(
        func,
        grid.0,
        grid.1,
        grid.2,
        block.0,
        block.1,
        block.2,
        shared_mem,
        stream,
        args.as_mut_ptr(),
    )
    .result()
}

/// Query the maximum number of active blocks per SM for a given kernel config.
pub unsafe fn max_active_blocks_per_sm(
    func: CUfunction,
    block_size: u32,
    shared_mem: u32,
) -> Result<u32, DriverError> {
    let mut num_blocks: i32 = 0;
    sys::cuOccupancyMaxActiveBlocksPerMultiprocessor(
        &mut num_blocks,
        func,
        block_size as i32,
        shared_mem as usize,
    )
    .result()?;
    Ok(num_blocks as u32)
}

/// Get the number of SMs on the device backing `context`.
pub fn sm_count(context: &CudaContext) -> Result<u32, DriverError> {
    let dev = cudarc::driver::result::device::get(context.ordinal() as i32)?;
    let mut count: i32 = 0;
    unsafe {
        sys::cuDeviceGetAttribute(
            &mut count,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            dev,
        )
        .result()?;
    }
    Ok(count as u32)
}

/// Compute the maximum cooperative grid size (total blocks) for a kernel.
///
/// Returns `blocks_per_sm * num_sms`, which is the max grid.x for
/// a cooperative launch with grid.y=1, grid.z=1.
pub unsafe fn max_cooperative_grid(
    func: CUfunction,
    block_size: u32,
    shared_mem: u32,
    context: &CudaContext,
) -> Result<u32, DriverError> {
    let bpsm = max_active_blocks_per_sm(func, block_size, shared_mem)?;
    let sms = sm_count(context)?;
    Ok(bpsm * sms)
}
