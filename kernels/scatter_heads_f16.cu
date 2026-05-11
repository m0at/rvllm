// Scatter all heads of a [num_heads, N, head_dim] head-major buffer
// back into a [N, num_heads * head_dim] interleaved attn_out buffer.
//
// out[n, h*head_dim + d] = in[h, n, d]
//
// One launch (vs one per head) for the per-block attention scatter.
//
// Launch:
//   Grid:  (n_tokens, num_heads, 1)
//   Block: (head_dim, 1, 1)

#include <cuda_fp16.h>

extern "C" __global__ void scatter_heads_f16_kernel(
    __half* __restrict__ out,             // [N, num_heads * head_dim]
    const __half* __restrict__ in_hmajor, // [num_heads, N, head_dim]
    int num_heads,
    int head_dim,
    int n_tokens
) {
    const int n = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;
    if (d >= head_dim) return;

    const long long src = (long long)h * n_tokens * head_dim
                        + (long long)n * head_dim + d;
    const long long dst = (long long)n * num_heads * head_dim
                        + (long long)h * head_dim + d;
    out[dst] = in_hmajor[src];
}
