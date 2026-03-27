//! Physical block types and device classification.

use std::sync::atomic::{AtomicUsize, Ordering};

use rvllm_core::prelude::BlockId;

/// Which device a block resides on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceType {
    Gpu,
    Cpu,
}

/// A physical memory block on GPU.
#[derive(Debug)]
pub struct PhysicalBlock {
    block_id: BlockId,
    size_bytes: usize,
    ref_count: AtomicUsize,
    device: DeviceType,
}

impl PhysicalBlock {
    pub fn new(block_id: BlockId, size_bytes: usize) -> Self {
        Self {
            block_id,
            size_bytes,
            ref_count: AtomicUsize::new(1),
            device: DeviceType::Gpu,
        }
    }

    pub fn block_id(&self) -> BlockId {
        self.block_id
    }

    pub fn size_bytes(&self) -> usize {
        self.size_bytes
    }

    pub fn ref_count(&self) -> usize {
        self.ref_count.load(Ordering::Acquire)
    }

    pub fn inc_ref(&self) -> usize {
        self.ref_count.fetch_add(1, Ordering::AcqRel) + 1
    }

    pub fn dec_ref(&self) -> usize {
        let prev = self.ref_count.fetch_sub(1, Ordering::AcqRel);
        assert!(prev > 0, "ref_count underflow on block {}", self.block_id);
        prev - 1
    }

    pub fn device(&self) -> DeviceType {
        self.device
    }
}

impl Clone for PhysicalBlock {
    fn clone(&self) -> Self {
        Self {
            block_id: self.block_id,
            size_bytes: self.size_bytes,
            ref_count: AtomicUsize::new(self.ref_count.load(Ordering::Acquire)),
            device: self.device,
        }
    }
}

/// A physical memory block on CPU (pinned host memory).
#[derive(Debug)]
pub struct CpuBlock {
    block_id: BlockId,
    size_bytes: usize,
    ref_count: AtomicUsize,
    device: DeviceType,
}

impl CpuBlock {
    pub fn new(block_id: BlockId, size_bytes: usize) -> Self {
        Self {
            block_id,
            size_bytes,
            ref_count: AtomicUsize::new(1),
            device: DeviceType::Cpu,
        }
    }

    pub fn block_id(&self) -> BlockId {
        self.block_id
    }

    pub fn size_bytes(&self) -> usize {
        self.size_bytes
    }

    pub fn ref_count(&self) -> usize {
        self.ref_count.load(Ordering::Acquire)
    }

    pub fn inc_ref(&self) -> usize {
        self.ref_count.fetch_add(1, Ordering::AcqRel) + 1
    }

    pub fn dec_ref(&self) -> usize {
        let prev = self.ref_count.fetch_sub(1, Ordering::AcqRel);
        assert!(prev > 0, "ref_count underflow on cpu block {}", self.block_id);
        prev - 1
    }

    pub fn device(&self) -> DeviceType {
        self.device
    }
}

impl Clone for CpuBlock {
    fn clone(&self) -> Self {
        Self {
            block_id: self.block_id,
            size_bytes: self.size_bytes,
            ref_count: AtomicUsize::new(self.ref_count.load(Ordering::Acquire)),
            device: self.device,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physical_block_basics() {
        let b = PhysicalBlock::new(BlockId(0), 4096);
        assert_eq!(b.block_id(), BlockId(0));
        assert_eq!(b.size_bytes(), 4096);
        assert_eq!(b.device(), DeviceType::Gpu);
        assert_eq!(b.ref_count(), 1);
    }

    #[test]
    fn physical_block_ref_counting() {
        let b = PhysicalBlock::new(BlockId(1), 2048);
        assert_eq!(b.inc_ref(), 2);
        assert_eq!(b.inc_ref(), 3);
        assert_eq!(b.ref_count(), 3);
        assert_eq!(b.dec_ref(), 2);
        assert_eq!(b.dec_ref(), 1);
        assert_eq!(b.dec_ref(), 0);
    }

    #[test]
    #[should_panic(expected = "ref_count underflow")]
    fn physical_block_ref_underflow() {
        let b = PhysicalBlock::new(BlockId(2), 1024);
        b.dec_ref(); // 1 -> 0
        b.dec_ref(); // panic
    }

    #[test]
    fn physical_block_clone() {
        let b = PhysicalBlock::new(BlockId(3), 512);
        b.inc_ref();
        let b2 = b.clone();
        assert_eq!(b2.block_id(), BlockId(3));
        assert_eq!(b2.ref_count(), 2);
    }

    #[test]
    fn cpu_block_basics() {
        let b = CpuBlock::new(BlockId(10), 4096);
        assert_eq!(b.block_id(), BlockId(10));
        assert_eq!(b.size_bytes(), 4096);
        assert_eq!(b.device(), DeviceType::Cpu);
        assert_eq!(b.ref_count(), 1);
    }

    #[test]
    fn cpu_block_ref_counting() {
        let b = CpuBlock::new(BlockId(11), 2048);
        assert_eq!(b.inc_ref(), 2);
        assert_eq!(b.dec_ref(), 1);
        assert_eq!(b.dec_ref(), 0);
    }

    #[test]
    fn device_type_eq() {
        assert_eq!(DeviceType::Gpu, DeviceType::Gpu);
        assert_eq!(DeviceType::Cpu, DeviceType::Cpu);
        assert_ne!(DeviceType::Gpu, DeviceType::Cpu);
    }

    #[test]
    fn blocks_are_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<PhysicalBlock>();
        assert_send_sync::<CpuBlock>();
    }
}
