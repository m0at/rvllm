#![forbid(unsafe_code)]
//! Logical-to-physical block mapping for vllm-rs.
//!
//! Provides `BlockTable` (per-sequence logical-to-physical mapping),
//! `BlockManager` (allocation, free, fork/CoW, swap, prefix sharing),
//! and reference counting on physical blocks.

pub mod block_table;
pub mod manager;
pub mod prefix_cache;

pub use block_table::BlockTable;
pub use manager::{BlockManager, SharedBlockManager};
pub use prefix_cache::PrefixCache;

use rvllm_core::prelude::BlockId;

// Re-export real types from dependency crates.
pub use rvllm_memory::DeviceType as Device;
pub use rvllm_sequence::SequenceStatus;

// ---------------------------------------------------------------------------
// PhysicalBlock: kept local because rvllm_memory::PhysicalBlock uses atomic
// ref counting and size_bytes, while the block manager needs a lightweight
// struct with block_size and device for its own ref-count tracking.
// TODO: unify with rvllm_memory::PhysicalBlock once APIs converge
// ---------------------------------------------------------------------------

/// A physical memory block backing KV-cache data.
#[derive(Debug, Clone)]
pub struct PhysicalBlock {
    pub block_id: BlockId,
    pub block_size: usize,
    pub device: Device,
}

impl PhysicalBlock {
    pub fn new(block_id: BlockId, block_size: usize, device: Device) -> Self {
        Self {
            block_id,
            block_size,
            device,
        }
    }
}

// ---------------------------------------------------------------------------
// Sequence: uses the real rvllm_sequence::Sequence. The block manager accesses
// seq_id and total length via the real Sequence API.
// ---------------------------------------------------------------------------
pub use rvllm_sequence::Sequence;

// ---------------------------------------------------------------------------
// MemoryPool: kept local because rvllm_memory::MemoryPool returns
// Result<PhysicalBlock> while this trait returns Option<BlockId>.
// TODO: unify with rvllm_memory::MemoryPool once APIs converge
// ---------------------------------------------------------------------------

/// Trait for a pool of allocatable physical blocks.
pub trait MemoryPool: Send + Sync {
    /// Allocate a single block, returning its id. None if exhausted.
    fn allocate(&self) -> Option<BlockId>;
    /// Return a block to the free list.
    fn free(&self, block_id: BlockId);
    /// Number of currently free blocks.
    fn free_blocks(&self) -> usize;
    /// Total number of blocks managed by this pool.
    fn total_blocks(&self) -> usize;
}
