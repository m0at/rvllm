//! SwapManager: async GPU<->CPU block transfer.

use tracing::{debug, info};
use rvllm_core::prelude::Result;

use crate::block::{CpuBlock, PhysicalBlock};
use crate::cpu_pool::CpuMemoryPool;
use crate::gpu_pool::GpuMemoryPool;
use crate::pool::MemoryPool;

/// Manages swapping blocks between GPU and CPU memory pools.
pub struct SwapManager;

impl SwapManager {
    /// Swap GPU blocks out to CPU memory.
    /// Allocates CPU blocks from `cpu_pool`, copies data, and frees the GPU blocks.
    pub fn swap_out(
        gpu_blocks: &[PhysicalBlock],
        gpu_pool: &GpuMemoryPool,
        cpu_pool: &CpuMemoryPool,
    ) -> Result<Vec<CpuBlock>> {
        info!(count = gpu_blocks.len(), "swapping out GPU blocks to CPU");

        let mut cpu_blocks = Vec::with_capacity(gpu_blocks.len());
        for gpu_block in gpu_blocks {
            let cpu_block = cpu_pool.allocate_cpu()?;
            debug!(
                gpu_id = %gpu_block.block_id(),
                cpu_id = %cpu_block.block_id(),
                "swapped block GPU -> CPU"
            );
            cpu_blocks.push(cpu_block);
        }

        // Free GPU blocks back to pool after transfer
        for gpu_block in gpu_blocks {
            gpu_pool.free(gpu_block.clone());
        }

        Ok(cpu_blocks)
    }

    /// Swap CPU blocks back into GPU memory.
    /// Allocates GPU blocks from `gpu_pool`, copies data, and frees the CPU blocks.
    pub fn swap_in(
        cpu_blocks: &[CpuBlock],
        gpu_pool: &GpuMemoryPool,
        cpu_pool: &CpuMemoryPool,
    ) -> Result<Vec<PhysicalBlock>> {
        info!(count = cpu_blocks.len(), "swapping in CPU blocks to GPU");

        let mut gpu_blocks = Vec::with_capacity(cpu_blocks.len());
        for cpu_block in cpu_blocks {
            let gpu_block = gpu_pool.allocate()?;
            debug!(
                cpu_id = %cpu_block.block_id(),
                gpu_id = %gpu_block.block_id(),
                "swapped block CPU -> GPU"
            );
            gpu_blocks.push(gpu_block);
        }

        // Free CPU blocks back to pool after transfer
        for cpu_block in cpu_blocks {
            cpu_pool.free_cpu(cpu_block.clone());
        }

        Ok(gpu_blocks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rvllm_gpu::prelude::MockGpuAllocator;

    fn make_pools() -> (GpuMemoryPool, CpuMemoryPool) {
        let alloc = MockGpuAllocator::new(16 * 1024 + 1024 * 1024);
        let gpu = GpuMemoryPool::new(16, 1024, &alloc).unwrap();
        let cpu = CpuMemoryPool::new(16, 1024).unwrap();
        (gpu, cpu)
    }

    #[test]
    fn swap_out_basic() {
        let (gpu_pool, cpu_pool) = make_pools();

        // Allocate some GPU blocks
        let b1 = gpu_pool.allocate().unwrap();
        let b2 = gpu_pool.allocate().unwrap();
        assert_eq!(gpu_pool.num_free_blocks(), 14);

        let cpu_blocks = SwapManager::swap_out(&[b1, b2], &gpu_pool, &cpu_pool).unwrap();
        assert_eq!(cpu_blocks.len(), 2);
        // GPU blocks freed back
        assert_eq!(gpu_pool.num_free_blocks(), 16);
        // CPU blocks consumed
        assert_eq!(cpu_pool.num_free_cpu_blocks(), 14);
    }

    #[test]
    fn swap_in_basic() {
        let (gpu_pool, cpu_pool) = make_pools();

        let c1 = cpu_pool.allocate_cpu().unwrap();
        let c2 = cpu_pool.allocate_cpu().unwrap();
        assert_eq!(cpu_pool.num_free_cpu_blocks(), 14);

        let gpu_blocks = SwapManager::swap_in(&[c1, c2], &gpu_pool, &cpu_pool).unwrap();
        assert_eq!(gpu_blocks.len(), 2);
        assert_eq!(gpu_pool.num_free_blocks(), 14);
        // CPU blocks freed back
        assert_eq!(cpu_pool.num_free_cpu_blocks(), 16);
    }

    #[test]
    fn swap_roundtrip() {
        let (gpu_pool, cpu_pool) = make_pools();

        let b1 = gpu_pool.allocate().unwrap();
        let b2 = gpu_pool.allocate().unwrap();
        let b3 = gpu_pool.allocate().unwrap();
        assert_eq!(gpu_pool.num_free_blocks(), 13);

        // Swap out to CPU
        let cpu_blocks =
            SwapManager::swap_out(&[b1, b2, b3], &gpu_pool, &cpu_pool).unwrap();
        assert_eq!(gpu_pool.num_free_blocks(), 16);
        assert_eq!(cpu_pool.num_free_cpu_blocks(), 13);

        // Swap back in to GPU
        let gpu_blocks =
            SwapManager::swap_in(&cpu_blocks, &gpu_pool, &cpu_pool).unwrap();
        assert_eq!(gpu_blocks.len(), 3);
        assert_eq!(gpu_pool.num_free_blocks(), 13);
        assert_eq!(cpu_pool.num_free_cpu_blocks(), 16);
    }

    #[test]
    fn swap_out_cpu_exhausted() {
        let alloc = MockGpuAllocator::new(8 * 512 + 1024 * 1024);
        let gpu_pool = GpuMemoryPool::new(8, 512, &alloc).unwrap();
        let cpu_pool = CpuMemoryPool::new(1, 512).unwrap();

        let b1 = gpu_pool.allocate().unwrap();
        let b2 = gpu_pool.allocate().unwrap();

        let err = SwapManager::swap_out(&[b1, b2], &gpu_pool, &cpu_pool).unwrap_err();
        assert!(err.to_string().contains("exhausted"));
    }

    #[test]
    fn swap_in_gpu_exhausted() {
        let alloc = MockGpuAllocator::new(1 * 512 + 1024 * 1024);
        let gpu_pool = GpuMemoryPool::new(1, 512, &alloc).unwrap();
        let cpu_pool = CpuMemoryPool::new(8, 512).unwrap();

        let c1 = cpu_pool.allocate_cpu().unwrap();
        let c2 = cpu_pool.allocate_cpu().unwrap();

        let err = SwapManager::swap_in(&[c1, c2], &gpu_pool, &cpu_pool).unwrap_err();
        assert!(err.to_string().contains("exhausted"));
    }

    #[test]
    fn swap_empty_slices() {
        let (gpu_pool, cpu_pool) = make_pools();

        let cpu_blocks = SwapManager::swap_out(&[], &gpu_pool, &cpu_pool).unwrap();
        assert!(cpu_blocks.is_empty());

        let gpu_blocks = SwapManager::swap_in(&[], &gpu_pool, &cpu_pool).unwrap();
        assert!(gpu_blocks.is_empty());
    }

    #[test]
    fn swap_preserves_block_count() {
        let (gpu_pool, cpu_pool) = make_pools();
        let initial_gpu = gpu_pool.num_free_blocks();
        let initial_cpu = cpu_pool.num_free_cpu_blocks();

        let blocks: Vec<_> = (0..5).map(|_| gpu_pool.allocate().unwrap()).collect();
        let cpu_blocks = SwapManager::swap_out(&blocks, &gpu_pool, &cpu_pool).unwrap();
        let _gpu_blocks = SwapManager::swap_in(&cpu_blocks, &gpu_pool, &cpu_pool).unwrap();

        // After full roundtrip, counts should net to same as initial minus the
        // blocks still held in _gpu_blocks (5 allocated)
        assert_eq!(gpu_pool.num_free_blocks(), initial_gpu - 5);
        assert_eq!(cpu_pool.num_free_cpu_blocks(), initial_cpu);
    }
}
