// Extract one head from a [N, num_heads * head_dim] f16 tensor into a
// contiguous [N, head_dim] buffer. One launch instead of N DtoD calls.
//
// in:   [N, num_heads * head_dim] row-major
// out:  [N, head_dim] row-major
// Per element: out[t, d] = in[t, head_idx * head_dim + d]
//
// Launch:
//   Grid:  (N, 1, 1)
//   Block: (head_dim, 1, 1)

#include <cuda_fp16.h>

extern "C" __global__ void extract_head_f16_kernel(
    __half* __restrict__ out,
    const __half* __restrict__ in,
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

// Scatter one head from contiguous [N, head_dim] into [N, num_heads * head_dim].
extern "C" __global__ void scatter_head_f16_kernel(
    __half* __restrict__ out,
    const __half* __restrict__ in,
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
