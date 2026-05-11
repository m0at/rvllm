// Scatter all heads from [num_heads, N, head_dim] head-major into
// [N, num_heads * head_dim] interleaved attn_out (bf16 sibling).

#include <cuda_bf16.h>

extern "C" __global__ void scatter_heads_bf16_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ in_hmajor,
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
