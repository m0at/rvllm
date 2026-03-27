//! Mock GPU allocator -- heap-backed, zero unsafe.
//! Tracks outstanding allocations so tests can detect leaks.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use bytemuck::Pod;
use tracing::{debug, trace};

use crate::allocator::GpuAllocator;
use crate::buffer::{GpuBuffer, GpuBufferInner};
use crate::device::MemoryInfo;
use crate::Result;

pub struct MockGpuAllocator {
    total: usize,
    allocated: Arc<AtomicUsize>,
    alloc_count: Arc<AtomicUsize>,
}

impl MockGpuAllocator {
    pub fn new(total_bytes: usize) -> Self {
        debug!(total_bytes, "MockGpuAllocator created");
        Self {
            total: total_bytes,
            allocated: Arc::new(AtomicUsize::new(0)),
            alloc_count: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub fn bytes_allocated(&self) -> usize {
        self.allocated.load(Ordering::Relaxed)
    }

    pub fn live_allocations(&self) -> usize {
        self.alloc_count.load(Ordering::Relaxed)
    }
}

impl GpuAllocator for MockGpuAllocator {
    fn alloc<T: Pod + Send>(&self, count: usize) -> Result<GpuBuffer<T>> {
        let bytes = count * std::mem::size_of::<T>();
        let current = self.allocated.load(Ordering::Relaxed);
        if current + bytes > self.total {
            return Err(crate::LLMError::MemoryError(format!(
                "MockGpuAllocator OOM: requested {} bytes, {} of {} in use",
                bytes, current, self.total
            )));
        }
        self.allocated.fetch_add(bytes, Ordering::Relaxed);
        self.alloc_count.fetch_add(1, Ordering::Relaxed);
        trace!(bytes, count, "mock alloc");

        let allocated = Arc::clone(&self.allocated);
        let alloc_count = Arc::clone(&self.alloc_count);
        let on_drop = Box::new(move |freed_bytes: usize| {
            allocated.fetch_sub(freed_bytes, Ordering::Relaxed);
            alloc_count.fetch_sub(1, Ordering::Relaxed);
            trace!(freed_bytes, "mock free (drop)");
        });

        Ok(GpuBuffer {
            inner: GpuBufferInner::Mock {
                data: vec![T::zeroed(); count],
                on_drop: Some(on_drop),
            },
        })
    }

    fn free<T: Pod + Send>(&self, buf: GpuBuffer<T>) {
        drop(buf);
    }

    fn device_memory_info(&self) -> Result<MemoryInfo> {
        let used = self.allocated.load(Ordering::Relaxed);
        Ok(MemoryInfo {
            total: self.total,
            free: self.total.saturating_sub(used),
            used,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MB: usize = 1024 * 1024;

    #[test]
    fn alloc_and_free() {
        let alloc = MockGpuAllocator::new(64 * MB);
        let buf = alloc.alloc::<f32>(1024).unwrap();
        assert_eq!(buf.len(), 1024);
        assert_eq!(buf.size_bytes(), 4096);
        assert_eq!(alloc.bytes_allocated(), 4096);
        assert_eq!(alloc.live_allocations(), 1);

        alloc.free(buf);
        assert_eq!(alloc.bytes_allocated(), 0);
        assert_eq!(alloc.live_allocations(), 0);
    }

    #[test]
    fn oom() {
        let alloc = MockGpuAllocator::new(100);
        assert!(alloc.alloc::<u8>(200).is_err());
    }

    #[test]
    fn drop_frees_memory() {
        let alloc = MockGpuAllocator::new(MB);
        {
            let _buf = alloc.alloc::<u8>(512).unwrap();
            assert_eq!(alloc.bytes_allocated(), 512);
        }
        assert_eq!(alloc.bytes_allocated(), 0);
        assert_eq!(alloc.live_allocations(), 0);
    }

    #[test]
    fn multiple_allocations() {
        let alloc = MockGpuAllocator::new(MB);
        let a = alloc.alloc::<f32>(100).unwrap();
        let b = alloc.alloc::<f64>(50).unwrap();
        assert_eq!(alloc.live_allocations(), 2);
        assert_eq!(alloc.bytes_allocated(), 100 * 4 + 50 * 8);

        drop(a);
        assert_eq!(alloc.live_allocations(), 1);

        drop(b);
        assert_eq!(alloc.live_allocations(), 0);
    }

    #[test]
    fn memory_info() {
        let alloc = MockGpuAllocator::new(1000);
        let _buf = alloc.alloc::<u8>(300).unwrap();
        let info = alloc.device_memory_info().unwrap();
        assert_eq!(info.total, 1000);
        assert_eq!(info.used, 300);
        assert_eq!(info.free, 700);
    }

    #[test]
    fn copy_roundtrip() {
        let alloc = MockGpuAllocator::new(MB);
        let mut buf = alloc.alloc::<f32>(4).unwrap();
        buf.copy_from_host(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let out = buf.copy_to_host().unwrap();
        assert_eq!(out, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn copy_size_mismatch() {
        let alloc = MockGpuAllocator::new(MB);
        let mut buf = alloc.alloc::<u32>(2).unwrap();
        assert!(buf.copy_from_host(&[1, 2, 3]).is_err());
    }

    #[test]
    fn leak_detection() {
        let alloc = MockGpuAllocator::new(MB);
        let buf = alloc.alloc::<u8>(256).unwrap();
        std::mem::forget(buf);
        assert_eq!(alloc.live_allocations(), 1);
        assert_eq!(alloc.bytes_allocated(), 256);
    }
}
