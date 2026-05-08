// Mistral 3.5 KV cache write — store post-RoPE K and post-projection V
// for the current token at slot `position` of the per-layer cache.
//
// Cache layout (BF16, contiguous): [max_pos, n_kv_heads, head_dim]
// Source layout (BF16, contiguous): [n_kv_heads, head_dim]
//
// Launch:
//   Grid:  (n_kv_heads, 1, 1)
//   Block: (head_dim, 1, 1)   — head_dim ≤ 1024

#include <cuda_bf16.h>

extern "C" __global__ void mistral35_kv_cache_write_bf16_kernel(
    const __nv_bfloat16* __restrict__ k_in,    // [n_kv_heads, head_dim]
    const __nv_bfloat16* __restrict__ v_in,
    __nv_bfloat16* __restrict__ k_cache,        // [max_pos, n_kv_heads, head_dim]
    __nv_bfloat16* __restrict__ v_cache,
    int n_kv_heads,
    int head_dim,
    int position
) {
    const int kvh = blockIdx.x;
    const int d   = threadIdx.x;
    if (d >= head_dim) return;
    const int src_off = kvh * head_dim + d;
    const int dst_off = position * n_kv_heads * head_dim + kvh * head_dim + d;
    k_cache[dst_off] = k_in[src_off];
    v_cache[dst_off] = v_in[src_off];
}
