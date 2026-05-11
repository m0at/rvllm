// Per-head gather/scatter for the bf16 vision attention path.
// bf16 sibling of extract_head_f16 / scatter_head_f16. Layout
// semantics identical (caller's [N, num_heads*head_dim]
// interleaved <-> [N, head_dim] per-head contiguous).

#include <cuda_bf16.h>

extern "C" __global__ void extract_head_bf16_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ in,
    int head_idx,
    int num_heads,
    int head_dim
) {
    const int t = blockIdx.x;
    const int d = threadIdx.x;
    if (d >= head_dim) return;
    const long long src_stride = (long long)num_heads * head_dim;
    const long long src = (long long)t * src_stride + (long long)head_idx * head_dim + d;
    const long long dst = (long long)t * head_dim + d;
    out[dst] = in[src];
}

extern "C" __global__ void scatter_head_bf16_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ in,
    int head_idx,
    int num_heads,
    int head_dim
) {
    const int t = blockIdx.x;
    const int d = threadIdx.x;
    if (d >= head_dim) return;
    const long long dst_stride = (long long)num_heads * head_dim;
    const long long dst = (long long)t * dst_stride + (long long)head_idx * head_dim + d;
    const long long src = (long long)t * head_dim + d;
    out[dst] = in[src];
}
