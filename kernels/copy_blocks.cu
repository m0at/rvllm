// Block copy kernel for KV cache management.
// Copies key and value cache blocks according to a mapping table.
// Used during beam search, prefix caching, and copy-on-write operations.
//
// Launch config:
//   Grid:  (num_pairs, 1, 1)
//   Block: (min(block_size * num_heads * head_dim, 1024), 1, 1)
//   Shared memory: none
//
// Each block copies one (src, dst) pair from the mapping table.

extern "C"
__global__ void copy_blocks_kernel(
    float* __restrict__ key_cache,         // [num_blocks, block_size, num_heads, head_dim]
    float* __restrict__ value_cache,       // [num_blocks, block_size, num_heads, head_dim]
    const long* __restrict__ block_mapping, // [num_pairs, 2]  -- (src_block, dst_block) pairs
    int num_pairs,
    int block_size,
    int num_heads,
    int head_dim
) {
    const int pair_idx = blockIdx.x;
    if (pair_idx >= num_pairs) return;

    const long src_block = block_mapping[pair_idx * 2];
    const long dst_block = block_mapping[pair_idx * 2 + 1];

    const int elems_per_block = block_size * num_heads * head_dim;
    const int src_offset = src_block * elems_per_block;
    const int dst_offset = dst_block * elems_per_block;

    // Copy key cache block
    for (int i = threadIdx.x; i < elems_per_block; i += blockDim.x) {
        key_cache[dst_offset + i] = key_cache[src_offset + i];
    }

    // Copy value cache block
    for (int i = threadIdx.x; i < elems_per_block; i += blockDim.x) {
        value_cache[dst_offset + i] = value_cache[src_offset + i];
    }
}
