// Per-head transpose for ViT attention's scores @ V step.
//
// Input:  V_strided in [N, num_heads * head_dim] f16 — i.e. V is
//         interleaved across heads, head h's row n at offset
//         `n * num_heads*head_dim + h * head_dim`.
// Output: V_T_all in [num_heads, head_dim, N] f16, contiguous per
//         head — head h's [head_dim, N] slab at offset
//         `h * head_dim * N`, with element [d, n] at
//         `h*head_dim*N + d*N + n`.
//
// Effectively: for each (h, n, d), copy V[n, h*head_dim + d] →
// V_T_all[h, d, n].
//
// Launch:
//   Grid:  (n_tokens, num_heads, 1)
//   Block: (head_dim, 1, 1)            — head_dim ≤ 1024

#include <cuda_fp16.h>

extern "C" __global__ void transpose_heads_v_f16_kernel(
    __half* __restrict__ out,                // [num_heads, head_dim, N]
    const __half* __restrict__ in_strided,   // [N, num_heads * head_dim]
    int num_heads,
    int head_dim,
    int n_tokens
) {
    const int n = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;
    if (d >= head_dim) return;

    const long long src = (long long)n * num_heads * head_dim
                        + (long long)h * head_dim + d;
    const long long dst = (long long)h * head_dim * n_tokens
                        + (long long)d * n_tokens + n;
    out[dst] = in_strided[src];
}
