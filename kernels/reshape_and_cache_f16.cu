// Half-precision reshape and cache kernel: scatter per-token K/V into paged cache.
// All data is f16 -- pure copy, no type conversion needed.
// Vectorized: uses float2 loads/stores (= 4 halves = 8 bytes per transaction).
//
// Cache layout: [num_blocks, block_size, num_kv_heads, head_dim]
// Input layout: [num_tokens, num_kv_heads, head_dim]
// slot_mapping:  [num_tokens] -- each entry is (block_idx * block_size + block_offset)
//
// Launch config:
//   Grid:  (num_tokens, 1, 1)
//   Block: (min(num_kv_heads * head_dim / 4, 1024), 1, 1)  -- vectorized
//   Shared memory: none

#include <cuda_fp16.h>

extern "C"
__global__ void reshape_and_cache_f16_kernel(
    __half* __restrict__ key_cache,        // [num_blocks, block_size, num_kv_heads, head_dim]
    __half* __restrict__ value_cache,      // [num_blocks, block_size, num_kv_heads, head_dim]
    const __half* __restrict__ key,        // [num_tokens, num_kv_heads, head_dim]
    const __half* __restrict__ value,      // [num_tokens, num_kv_heads, head_dim]
    const int* __restrict__ slot_mapping,  // [num_tokens]
    int num_tokens,
    int num_kv_heads,
    int head_dim
) {
    const int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    const int tid = threadIdx.x;
    const int kv_dim = num_kv_heads * head_dim;
    const int slot = slot_mapping[token_idx];

    const int cache_offset = slot * kv_dim;
    const int src_offset = token_idx * kv_dim;

    // Vectorized: reinterpret __half* as float2* to load/store 4 halves (8 bytes) at once.
    // float2 = 8 bytes = 4 x __half, giving 4x bandwidth vs scalar half stores.
    const int kv_dim_vec = kv_dim >> 2;  // groups of 4 halves
    const float2* __restrict__ key_src_v = reinterpret_cast<const float2*>(key + src_offset);
    const float2* __restrict__ val_src_v = reinterpret_cast<const float2*>(value + src_offset);
    float2* __restrict__ key_dst_v = reinterpret_cast<float2*>(key_cache + cache_offset);
    float2* __restrict__ val_dst_v = reinterpret_cast<float2*>(value_cache + cache_offset);

    for (int i = tid; i < kv_dim_vec; i += blockDim.x) {
        key_dst_v[i] = key_src_v[i];
        val_dst_v[i] = val_src_v[i];
    }

    // Scalar tail for kv_dim not divisible by 4
    const int tail_start = kv_dim_vec << 2;
    for (int i = tail_start + tid; i < kv_dim; i += blockDim.x) {
        key_cache[cache_offset + i] = key[src_offset + i];
        value_cache[cache_offset + i] = value[src_offset + i];
    }
}
