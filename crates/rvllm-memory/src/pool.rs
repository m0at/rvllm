//! MemoryPool trait definition.

use rvllm_core::prelude::Result;

use crate::block::{CpuBlock, PhysicalBlock};

/// Trait for GPU and CPU memory pools.
pub trait MemoryPool: Send + Sync {
    /// Allocate a single block from the pool.
    fn allocate(&self) -> Result<PhysicalBlock>;

    /// Return a block to the pool.
    fn free(&self, block: PhysicalBlock);

    /// Number of blocks available for allocation.
    fn num_free_blocks(&self) -> usize;

    /// Total number of blocks managed by this pool.
    fn num_total_blocks(&self) -> usize;

    /// Swap blocks from GPU to CPU, returning CPU-side copies.
    fn swap_out(&self, blocks: &[PhysicalBlock]) -> Result<Vec<CpuBlock>>;

    /// Swap blocks from CPU back to GPU, returning GPU-side blocks.
    fn swap_in(&self, blocks: &[CpuBlock]) -> Result<Vec<PhysicalBlock>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trait_is_object_safe() {
        fn _accept(_pool: &dyn MemoryPool) {}
    }
}
