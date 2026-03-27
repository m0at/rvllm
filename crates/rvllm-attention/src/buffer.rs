//! CPU-backed buffer shim for attention kernels.
//!
//! This provides the data+shape GpuBuffer that attention code needs for
//! CPU-mock execution. Previously imported from rvllm_kv_cache::gpu which
//! has been removed.
//!
//! TODO: Replace with rvllm_gpu::prelude::GpuBuffer once it exposes
//! data/shape accessors (or once attention kernels use the opaque buffer
//! API with copy_to_host/copy_from_host).

/// CPU-backed buffer with public data and shape fields for mock execution.
pub struct GpuBuffer<T> {
    pub data: Vec<T>,
    pub shape: Vec<usize>,
}

impl<T: Clone + Default> GpuBuffer<T> {
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn copy_from_slice(&mut self, src: &[T], dst_offset: usize) {
        self.data[dst_offset..dst_offset + src.len()].clone_from_slice(src);
    }

    pub fn copy_to_slice(&self, src_offset: usize, dst: &mut [T], count: usize) {
        dst[..count].clone_from_slice(&self.data[src_offset..src_offset + count]);
    }

    pub fn copy_range(
        &mut self,
        src: &GpuBuffer<T>,
        src_offset: usize,
        dst_offset: usize,
        count: usize,
    ) {
        self.data[dst_offset..dst_offset + count]
            .clone_from_slice(&src.data[src_offset..src_offset + count]);
    }
}
