// Per-head transpose for ViT batched attention (bf16 sibling).
// [N, num_heads*head_dim] interleaved → [num_heads, head_dim, N]
// head-major. Identical layout semantics to transpose_heads_v_f16.

#include <cuda_bf16.h>

extern "C" __global__ void transpose_heads_v_bf16_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ in_strided,
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
