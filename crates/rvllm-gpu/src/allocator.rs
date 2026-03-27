//! GPU allocator trait.

use bytemuck::Pod;

use crate::buffer::GpuBuffer;
use crate::device::MemoryInfo;
use crate::Result;

/// Trait abstracting GPU memory allocation.
pub trait GpuAllocator: Send + Sync {
    fn alloc<T: Pod + Send>(&self, count: usize) -> Result<GpuBuffer<T>>;
    fn free<T: Pod + Send>(&self, buf: GpuBuffer<T>);
    fn device_memory_info(&self) -> Result<MemoryInfo>;
}
