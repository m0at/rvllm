// Fused RoPE + KV Cache Write kernel.
// Applies rotary position embeddings to Q and K in-place,
// then writes K and V to the paged KV cache in a single kernel.
// Saves 1 kernel launch vs separate RoPE + cache_write.
//
// Grid: (num_tokens, max(num_heads, num_kv_heads), 1)
// Block: (head_dim, 1, 1). Threads 0..rotary_pairs rotate the active
// half-split pairs; all threads copy one K/V dimension to cache.

#include <cuda_fp16.h>

extern "C"
__global__ void fused_rope_cache_f16_kernel(
    __half* __restrict__ q,           // [num_tokens, num_heads * head_dim] -- RoPE applied in-place
    __half* __restrict__ k,           // [num_tokens, num_kv_heads * head_dim] -- RoPE applied in-place
    const __half* __restrict__ v,     // [num_tokens, num_kv_heads * head_dim] -- read-only
    __half* __restrict__ key_cache,   // [num_blocks, block_size, num_kv_heads, head_dim]
    __half* __restrict__ value_cache, // [num_blocks, block_size, num_kv_heads, head_dim]
    const float* __restrict__ cos_table, // [max_pos, half_dim]
    const float* __restrict__ sin_table, // [max_pos, half_dim]
    const int* __restrict__ positions,   // [num_tokens]
    const int* __restrict__ slot_mapping, // [num_tokens]
    int num_tokens,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int rope_stride,
    int rotary_dim,
    int cache_stride
) {
    const int token_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int half_dim = head_dim / 2;
    const int rotary_pairs = rotary_dim / 2;
    const int tid = threadIdx.x;
    if (tid >= head_dim) return;

    const int pos = positions[token_idx];
    const bool do_rope = tid < rotary_pairs;
    const float cos_val = do_rope ? cos_table[pos * rope_stride + tid] : 1.0f;
    const float sin_val = do_rope ? sin_table[pos * rope_stride + tid] : 0.0f;

    // Apply half-split RoPE to Q (all heads)
    if (do_rope && head_idx < num_heads) {
        int q_base = (token_idx * num_heads + head_idx) * head_dim;
        int d0 = tid;
        int d1 = half_dim + tid;
        float q0 = __half2float(q[q_base + d0]);
        float q1 = __half2float(q[q_base + d1]);
        q[q_base + d0] = __float2half(q0 * cos_val - q1 * sin_val);
        q[q_base + d1] = __float2half(q0 * sin_val + q1 * cos_val);
    }

    // Apply RoPE to K (kv_heads only) + write to KV cache
    if (head_idx < num_kv_heads) {
        int k_base = (token_idx * num_kv_heads + head_idx) * head_dim;
        if (do_rope) {
            int d0 = tid;
            int d1 = half_dim + tid;
            float k0 = __half2float(k[k_base + d0]);
            float k1 = __half2float(k[k_base + d1]);
            k[k_base + d0] = __float2half(k0 * cos_val - k1 * sin_val);
            k[k_base + d1] = __float2half(k0 * sin_val + k1 * cos_val);
        }
    }
    __syncthreads();

    if (head_idx < num_kv_heads) {
        int slot = slot_mapping[token_idx];
        if (slot >= 0) {
            int k_base = (token_idx * num_kv_heads + head_idx) * head_dim;
            int v_base = (token_idx * num_kv_heads + head_idx) * head_dim;
            int cache_offset = slot * cache_stride + head_idx * head_dim;
            key_cache[cache_offset + tid] = k[k_base + tid];
            value_cache[cache_offset + tid] = v[v_base + tid];
        }
    }
}
