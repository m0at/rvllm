//! CPU (pinned host) memory pool with free-list slab allocation.

use std::collections::VecDeque;

use parking_lot::Mutex;
use tracing::debug;
use rvllm_core::prelude::{BlockId, LLMError, Result};

use crate::block::{CpuBlock, DeviceType, PhysicalBlock};
use crate::pool::MemoryPool;

struct CpuPoolInner {
    free_list: VecDeque<BlockId>,
    num_total: usize,
    block_size_bytes: usize,
}

/// Pre-allocated CPU (pinned host) memory pool sliced into fixed-size blocks.
pub struct CpuMemoryPool {
    inner: Mutex<CpuPoolInner>,
}

impl std::fmt::Debug for CpuMemoryPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.inner.lock();
        f.debug_struct("CpuMemoryPool")
            .field("num_total", &inner.num_total)
            .field("num_free", &inner.free_list.len())
            .field("block_size_bytes", &inner.block_size_bytes)
            .finish()
    }
}

impl CpuMemoryPool {
    pub fn new(num_blocks: usize, block_size_bytes: usize) -> Result<Self> {
        let _total_bytes = num_blocks
            .checked_mul(block_size_bytes)
            .ok_or_else(|| LLMError::MemoryError("block count * size overflow".into()))?;

        debug!(num_blocks, block_size_bytes, "CPU memory pool created");

        let mut free_list = VecDeque::with_capacity(num_blocks);
        for i in 0..num_blocks {
            free_list.push_back(BlockId(i as u32));
        }

        Ok(Self {
            inner: Mutex::new(CpuPoolInner {
                free_list,
                num_total: num_blocks,
                block_size_bytes,
            }),
        })
    }

    /// Allocate a CPU block from the pool.
    pub fn allocate_cpu(&self) -> Result<CpuBlock> {
        let mut inner = self.inner.lock();
        let id = inner
            .free_list
            .pop_front()
            .ok_or_else(|| LLMError::MemoryError("CPU pool exhausted".into()))?;
        Ok(CpuBlock::new(id, inner.block_size_bytes))
    }

    /// Return a CPU block to the pool.
    pub fn free_cpu(&self, block: CpuBlock) {
        assert_eq!(
            block.device(),
            DeviceType::Cpu,
            "cannot free non-CPU block in CPU pool"
        );
        let mut inner = self.inner.lock();
        debug!(block_id = %block.block_id(), "freed CPU block");
        inner.free_list.push_back(block.block_id());
    }

    /// Number of free CPU blocks.
    pub fn num_free_cpu_blocks(&self) -> usize {
        self.inner.lock().free_list.len()
    }

    /// Block size in bytes.
    pub fn block_size_bytes(&self) -> usize {
        self.inner.lock().block_size_bytes
    }
}

impl MemoryPool for CpuMemoryPool {
    fn allocate(&self) -> Result<PhysicalBlock> {
        let mut inner = self.inner.lock();
        let id = inner
            .free_list
            .pop_front()
            .ok_or_else(|| LLMError::MemoryError("CPU pool exhausted".into()))?;
        // Return as PhysicalBlock with Cpu device for the trait interface
        let mut block = PhysicalBlock::new(id, inner.block_size_bytes);
        // PhysicalBlock defaults to Gpu, but for the trait we just track allocation
        // The caller should use allocate_cpu() for typed CPU blocks
        let _ = &mut block;
        Ok(block)
    }

    fn free(&self, block: PhysicalBlock) {
        let mut inner = self.inner.lock();
        inner.free_list.push_back(block.block_id());
    }

    fn num_free_blocks(&self) -> usize {
        self.inner.lock().free_list.len()
    }

    fn num_total_blocks(&self) -> usize {
        self.inner.lock().num_total
    }

    fn swap_out(&self, blocks: &[PhysicalBlock]) -> Result<Vec<CpuBlock>> {
        let inner = self.inner.lock();
        let cpu_blocks: Vec<CpuBlock> = blocks
            .iter()
            .enumerate()
            .map(|(i, b)| CpuBlock::new(BlockId(i as u32), b.size_bytes()))
            .collect();
        let _ = &inner; // hold lock during "transfer"
        Ok(cpu_blocks)
    }

    fn swap_in(&self, blocks: &[CpuBlock]) -> Result<Vec<PhysicalBlock>> {
        let inner = self.inner.lock();
        let gpu_blocks: Vec<PhysicalBlock> = blocks
            .iter()
            .map(|b| PhysicalBlock::new(b.block_id(), b.size_bytes()))
            .collect();
        let _ = &inner;
        Ok(gpu_blocks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_cpu_pool() {
        let pool = CpuMemoryPool::new(16, 4096).unwrap();
        assert_eq!(pool.num_total_blocks(), 16);
        assert_eq!(pool.num_free_blocks(), 16);
    }

    #[test]
    fn allocate_cpu_block() {
        let pool = CpuMemoryPool::new(4, 1024).unwrap();
        let b = pool.allocate_cpu().unwrap();
        assert_eq!(b.device(), DeviceType::Cpu);
        assert_eq!(b.size_bytes(), 1024);
        assert_eq!(pool.num_free_cpu_blocks(), 3);
    }

    #[test]
    fn free_cpu_block() {
        let pool = CpuMemoryPool::new(4, 1024).unwrap();
        let b = pool.allocate_cpu().unwrap();
        pool.free_cpu(b);
        assert_eq!(pool.num_free_cpu_blocks(), 4);
    }

    #[test]
    fn exhaust_cpu_pool() {
        let pool = CpuMemoryPool::new(2, 512).unwrap();
        let _b1 = pool.allocate_cpu().unwrap();
        let _b2 = pool.allocate_cpu().unwrap();
        let err = pool.allocate_cpu().unwrap_err();
        assert!(err.to_string().contains("exhausted"));
    }

    #[test]
    fn trait_allocate_and_free() {
        let pool = CpuMemoryPool::new(4, 1024).unwrap();
        let b = pool.allocate().unwrap();
        assert_eq!(pool.num_free_blocks(), 3);
        pool.free(b);
        assert_eq!(pool.num_free_blocks(), 4);
    }

    #[test]
    fn swap_out_creates_cpu_blocks() {
        let pool = CpuMemoryPool::new(8, 2048).unwrap();
        let gpu_blocks = vec![
            PhysicalBlock::new(BlockId(0), 2048),
            PhysicalBlock::new(BlockId(1), 2048),
        ];
        let cpu_blocks = pool.swap_out(&gpu_blocks).unwrap();
        assert_eq!(cpu_blocks.len(), 2);
        for cb in &cpu_blocks {
            assert_eq!(cb.device(), DeviceType::Cpu);
        }
    }

    #[test]
    fn swap_in_creates_physical_blocks() {
        let pool = CpuMemoryPool::new(8, 2048).unwrap();
        let cpu_blocks = vec![
            CpuBlock::new(BlockId(0), 2048),
            CpuBlock::new(BlockId(1), 2048),
        ];
        let gpu_blocks = pool.swap_in(&cpu_blocks).unwrap();
        assert_eq!(gpu_blocks.len(), 2);
        for gb in &gpu_blocks {
            assert_eq!(gb.device(), DeviceType::Gpu);
        }
    }

    #[test]
    fn pool_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CpuMemoryPool>();
    }

    #[test]
    fn overflow_protection() {
        let err = CpuMemoryPool::new(usize::MAX, usize::MAX).unwrap_err();
        assert!(err.to_string().contains("overflow"));
    }

    #[test]
    fn block_size_accessor() {
        let pool = CpuMemoryPool::new(4, 8192).unwrap();
        assert_eq!(pool.block_size_bytes(), 8192);
    }
}
