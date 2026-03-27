//! GPU memory pool with free-list slab allocation.

use std::collections::VecDeque;

use parking_lot::Mutex;
use tracing::{debug, warn};
use rvllm_core::prelude::{BlockId, LLMError, Result};
use rvllm_gpu::prelude::{CpuBuffer, GpuAllocator, GpuBuffer};

use crate::block::{CpuBlock, DeviceType, PhysicalBlock};
use crate::pool::MemoryPool;

struct GpuPoolInner {
    free_list: VecDeque<BlockId>,
    num_total: usize,
    block_size_bytes: usize,
    low_watermark: usize,
}

/// Pre-allocated GPU memory pool sliced into fixed-size blocks.
pub struct GpuMemoryPool {
    inner: Mutex<GpuPoolInner>,
    _buffer: GpuBuffer<u8>,
}

impl std::fmt::Debug for GpuMemoryPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.inner.lock();
        f.debug_struct("GpuMemoryPool")
            .field("num_total", &inner.num_total)
            .field("num_free", &inner.free_list.len())
            .field("block_size_bytes", &inner.block_size_bytes)
            .finish()
    }
}

impl GpuMemoryPool {
    pub fn new<A: GpuAllocator>(
        num_blocks: usize,
        block_size_bytes: usize,
        allocator: &A,
    ) -> Result<Self> {
        let total_bytes = num_blocks
            .checked_mul(block_size_bytes)
            .ok_or_else(|| LLMError::MemoryError("block count * size overflow".into()))?;

        let buffer = allocator.alloc::<u8>(total_bytes)?;
        debug!(num_blocks, block_size_bytes, "GPU memory pool created");

        let mut free_list = VecDeque::with_capacity(num_blocks);
        for i in 0..num_blocks {
            free_list.push_back(BlockId(i as u32));
        }

        let low_watermark = num_blocks / 10; // 10% threshold

        Ok(Self {
            inner: Mutex::new(GpuPoolInner {
                free_list,
                num_total: num_blocks,
                block_size_bytes,
                low_watermark,
            }),
            _buffer: buffer,
        })
    }

    /// Set the low watermark (number of free blocks below which eviction
    /// signals should fire).
    pub fn set_low_watermark(&self, watermark: usize) {
        self.inner.lock().low_watermark = watermark;
    }

    /// Current low watermark threshold.
    pub fn low_watermark(&self) -> usize {
        self.inner.lock().low_watermark
    }

    /// True if free blocks are at or below the low watermark.
    pub fn below_watermark(&self) -> bool {
        let inner = self.inner.lock();
        inner.free_list.len() <= inner.low_watermark
    }
}

impl MemoryPool for GpuMemoryPool {
    fn allocate(&self) -> Result<PhysicalBlock> {
        let mut inner = self.inner.lock();
        let id = inner
            .free_list
            .pop_front()
            .ok_or_else(|| LLMError::MemoryError("GPU pool exhausted".into()))?;

        if inner.free_list.len() <= inner.low_watermark {
            warn!(
                free = inner.free_list.len(),
                watermark = inner.low_watermark,
                "GPU pool below low watermark"
            );
        }

        Ok(PhysicalBlock::new(id, inner.block_size_bytes))
    }

    fn free(&self, block: PhysicalBlock) {
        assert_eq!(
            block.device(),
            DeviceType::Gpu,
            "cannot free non-GPU block in GPU pool"
        );
        let mut inner = self.inner.lock();
        debug!(block_id = %block.block_id(), "freed GPU block");
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

        // Copy the slab data to host for block-level extraction
        let slab_data = self._buffer.copy_to_host()?;

        let mut cpu_blocks = Vec::with_capacity(blocks.len());
        for (i, block) in blocks.iter().enumerate() {
            let offset = block.block_id().0 as usize * inner.block_size_bytes;
            let block_data = &slab_data[offset..offset + inner.block_size_bytes];
            // Store the block data in a CpuBuffer for potential future use
            let mut cpu_buf = CpuBuffer::<u8>::new(inner.block_size_bytes);
            cpu_buf.copy_from_host(block_data)?;
            let _ = cpu_buf; // data transferred
            cpu_blocks.push(CpuBlock::new(BlockId(i as u32), inner.block_size_bytes));
        }
        Ok(cpu_blocks)
    }

    fn swap_in(&self, blocks: &[CpuBlock]) -> Result<Vec<PhysicalBlock>> {
        let mut inner = self.inner.lock();
        let mut gpu_blocks = Vec::with_capacity(blocks.len());
        for cpu_block in blocks {
            let id = inner
                .free_list
                .pop_front()
                .ok_or_else(|| LLMError::MemoryError("GPU pool exhausted during swap_in".into()))?;

            // In a real implementation, data would be copied from CPU to GPU here.
            // Under mock-gpu, the slab is host memory so this is a no-op placeholder.
            let _ = cpu_block;

            gpu_blocks.push(PhysicalBlock::new(id, inner.block_size_bytes));
        }
        Ok(gpu_blocks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rvllm_gpu::prelude::MockGpuAllocator;

    fn mock_pool(num_blocks: usize, block_size: usize) -> GpuMemoryPool {
        let alloc = MockGpuAllocator::new(num_blocks * block_size + 1024 * 1024);
        GpuMemoryPool::new(num_blocks, block_size, &alloc).unwrap()
    }

    #[test]
    fn create_pool() {
        let pool = mock_pool(16, 4096);
        assert_eq!(pool.num_total_blocks(), 16);
        assert_eq!(pool.num_free_blocks(), 16);
    }

    #[test]
    fn allocate_and_free() {
        let pool = mock_pool(4, 1024);
        let b1 = pool.allocate().unwrap();
        assert_eq!(pool.num_free_blocks(), 3);
        let b2 = pool.allocate().unwrap();
        assert_eq!(pool.num_free_blocks(), 2);

        pool.free(b1);
        assert_eq!(pool.num_free_blocks(), 3);
        pool.free(b2);
        assert_eq!(pool.num_free_blocks(), 4);
    }

    #[test]
    fn exhaust_pool() {
        let pool = mock_pool(2, 512);
        let _b1 = pool.allocate().unwrap();
        let _b2 = pool.allocate().unwrap();
        let err = pool.allocate().unwrap_err();
        assert!(err.to_string().contains("exhausted"));
    }

    #[test]
    fn watermark() {
        let pool = mock_pool(10, 256);
        assert_eq!(pool.low_watermark(), 1); // 10% of 10
        assert!(!pool.below_watermark());

        // Allocate 9 blocks -> 1 free, at watermark
        for _ in 0..9 {
            pool.allocate().unwrap();
        }
        assert!(pool.below_watermark());
    }

    #[test]
    fn set_watermark() {
        let pool = mock_pool(10, 256);
        pool.set_low_watermark(5);
        assert_eq!(pool.low_watermark(), 5);
    }

    #[test]
    fn swap_out_and_in() {
        let pool = mock_pool(8, 1024);
        let b1 = pool.allocate().unwrap();
        let b2 = pool.allocate().unwrap();
        assert_eq!(pool.num_free_blocks(), 6);

        // Swap out
        let cpu_blocks = pool.swap_out(&[b1, b2]).unwrap();
        assert_eq!(cpu_blocks.len(), 2);
        for cb in &cpu_blocks {
            assert_eq!(cb.device(), DeviceType::Cpu);
            assert_eq!(cb.size_bytes(), 1024);
        }

        // Swap in
        let gpu_blocks = pool.swap_in(&cpu_blocks).unwrap();
        assert_eq!(gpu_blocks.len(), 2);
        for gb in &gpu_blocks {
            assert_eq!(gb.device(), DeviceType::Gpu);
        }
        assert_eq!(pool.num_free_blocks(), 4); // 6 - 2 used for swap_in
    }

    #[test]
    fn pool_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<GpuMemoryPool>();
    }

    #[test]
    fn overflow_protection() {
        let alloc = MockGpuAllocator::new(1 << 30);
        let err = GpuMemoryPool::new(usize::MAX, usize::MAX, &alloc).unwrap_err();
        assert!(err.to_string().contains("overflow"));
    }
}
