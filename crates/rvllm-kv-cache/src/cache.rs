//! KVCache and CacheConfig types.

use half::f16;
use std::mem;

use rvllm_gpu::prelude::GpuBuffer;

/// Holds (key_cache, value_cache) for a single transformer layer.
/// Each buffer is shaped [num_blocks, block_size, num_heads, head_dim].
pub struct KVCache {
    pub key_cache: GpuBuffer<f16>,
    pub value_cache: GpuBuffer<f16>,
    pub num_blocks: usize,
    pub block_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl KVCache {
    /// Number of f16 elements per block (block_size * num_heads * head_dim).
    pub fn elements_per_block(&self) -> usize {
        self.block_size * self.num_heads * self.head_dim
    }

    /// Byte offset for the start of a given block index.
    pub fn block_offset(&self, block_idx: usize) -> usize {
        block_idx * self.elements_per_block()
    }
}

/// Helper for computing block counts from available GPU memory.
pub struct CacheConfig {
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub block_size: usize,
}

impl CacheConfig {
    pub fn new(num_layers: usize, num_heads: usize, head_dim: usize, block_size: usize) -> Self {
        Self {
            num_layers,
            num_heads,
            head_dim,
            block_size,
        }
    }

    /// Bytes consumed by a single KV block (one key + one value) for one layer.
    pub fn block_bytes(&self) -> usize {
        let elements = self.block_size * self.num_heads * self.head_dim;
        // key + value, each f16
        2 * elements * mem::size_of::<f16>()
    }

    /// Total bytes consumed by a single block across all layers.
    pub fn total_block_bytes(&self) -> usize {
        self.num_layers * self.block_bytes()
    }

    /// Compute maximum number of GPU blocks that fit in `available_bytes`.
    pub fn num_blocks_from_memory(&self, available_bytes: usize) -> usize {
        let per_block = self.total_block_bytes();
        if per_block == 0 {
            return 0;
        }
        available_bytes / per_block
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_config_block_bytes() {
        let cfg = CacheConfig::new(32, 32, 128, 16);
        // per layer: 2 * 16 * 32 * 128 * 2 bytes = 262144
        assert_eq!(cfg.block_bytes(), 2 * 16 * 32 * 128 * 2);
        assert_eq!(cfg.total_block_bytes(), 32 * cfg.block_bytes());
    }

    #[test]
    fn num_blocks_calculation() {
        let cfg = CacheConfig::new(2, 4, 64, 8);
        let per_block = cfg.total_block_bytes();
        assert_eq!(cfg.num_blocks_from_memory(per_block * 10), 10);
        assert_eq!(cfg.num_blocks_from_memory(per_block * 10 + 1), 10);
        assert_eq!(cfg.num_blocks_from_memory(0), 0);
    }

    #[test]
    fn zero_config_returns_zero_blocks() {
        let cfg = CacheConfig::new(0, 0, 0, 0);
        assert_eq!(cfg.num_blocks_from_memory(1_000_000), 0);
    }
}
