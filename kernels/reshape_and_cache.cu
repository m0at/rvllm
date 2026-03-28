// Reshape and cache kernel: scatter per-token K/V into paged cache.
//
// For each token, copies its KV vector into the paged cache at the position
// given by slot_mapping[token_idx].
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
__global__ void reshape_and_cache_kernel(
    float* __restrict__ key_cache,        // [num_blocks, block_size, num_kv_heads, head_dim]
    float* __restrict__ value_cache,      // [num_blocks, block_size, num_kv_heads, head_dim]
    const float* __restrict__ key,        // [num_tokens, num_kv_heads, head_dim]
    const float* __restrict__ value,      // [num_tokens, num_kv_heads, head_dim]
    const int* __restrict__ slot_mapping, // [num_tokens]
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

    // Vectorized path: float4 stores (16 bytes per transaction, 4x bandwidth)
    const int kv_dim_vec = kv_dim >> 2;  // kv_dim / 4
    const float4* __restrict__ key_src_v = reinterpret_cast<const float4*>(key + src_offset);
    const float4* __restrict__ val_src_v = reinterpret_cast<const float4*>(value + src_offset);
    float4* __restrict__ key_dst_v = reinterpret_cast<float4*>(key_cache + cache_offset);
    float4* __restrict__ val_dst_v = reinterpret_cast<float4*>(value_cache + cache_offset);

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

// ============================================================================
// FP16 variant: writes f32 input K/V into f16 paged cache.
// Input K/V are f32 (from QKV projection), cache is f16 for 2x VRAM savings.
// Vectorized: load float4 (4 f32), convert to 4 f16, store as 2x half2.
// ============================================================================

extern "C"
__global__ void reshape_and_cache_f16_kernel(
    __half* __restrict__ key_cache,       // [num_blocks, block_size, num_kv_heads, head_dim] in f16
    __half* __restrict__ value_cache,     // [num_blocks, block_size, num_kv_heads, head_dim] in f16
    const float* __restrict__ key,        // [num_tokens, num_kv_heads, head_dim] in f32
    const float* __restrict__ value,      // [num_tokens, num_kv_heads, head_dim] in f32
    const int* __restrict__ slot_mapping, // [num_tokens]
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

    // Vectorized: load 4 f32 via float4, convert to 4 f16, store as half2 pair (8 bytes)
    const int kv_dim_vec = kv_dim >> 2;
    const float4* __restrict__ key_src_v = reinterpret_cast<const float4*>(key + src_offset);
    const float4* __restrict__ val_src_v = reinterpret_cast<const float4*>(value + src_offset);
    // Output: treat as ushort4 (4 x 16-bit = 8 bytes, matching 4 halves)
    ushort4* __restrict__ key_dst_v = reinterpret_cast<ushort4*>(key_cache + cache_offset);
    ushort4* __restrict__ val_dst_v = reinterpret_cast<ushort4*>(value_cache + cache_offset);

    for (int i = tid; i < kv_dim_vec; i += blockDim.x) {
        float4 kv = key_src_v[i];
        __half h0 = __float2half(kv.x);
        __half h1 = __float2half(kv.y);
        __half h2 = __float2half(kv.z);
        __half h3 = __float2half(kv.w);
        ushort4 packed;
        packed.x = *reinterpret_cast<const unsigned short*>(&h0);
        packed.y = *reinterpret_cast<const unsigned short*>(&h1);
        packed.z = *reinterpret_cast<const unsigned short*>(&h2);
        packed.w = *reinterpret_cast<const unsigned short*>(&h3);
        key_dst_v[i] = packed;

        float4 vv = val_src_v[i];
        h0 = __float2half(vv.x);
        h1 = __float2half(vv.y);
        h2 = __float2half(vv.z);
        h3 = __float2half(vv.w);
        packed.x = *reinterpret_cast<const unsigned short*>(&h0);
        packed.y = *reinterpret_cast<const unsigned short*>(&h1);
        packed.z = *reinterpret_cast<const unsigned short*>(&h2);
        packed.w = *reinterpret_cast<const unsigned short*>(&h3);
        val_dst_v[i] = packed;
    }

    // Scalar tail for kv_dim not divisible by 4
    const int tail_start = kv_dim_vec << 2;
    for (int i = tail_start + tid; i < kv_dim; i += blockDim.x) {
        key_cache[cache_offset + i] = __float2half(key[src_offset + i]);
        value_cache[cache_offset + i] = __float2half(value[src_offset + i]);
    }
}
