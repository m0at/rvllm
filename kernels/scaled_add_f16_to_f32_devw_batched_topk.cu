// Row-batched scaled_add for the routed-MoE k-round step in
// Qwen 3.6 MoE. For each (token m, hidden n):
//
//   acc[m*hidden + n] += top_w[m * top_k + k_round] * f16_to_f32(in[m*hidden + n])
//
// Replaces N per-token launches of `scaled_add_f16_to_f32_devw` per
// k-round with one launch per k-round. Grid = (ceil(hidden/block),
// num_tokens). Each thread computes one output element.
//
// Phase 6b / Round-27.

#include <cuda_fp16.h>

extern "C" __global__ void scaled_add_f16_to_f32_devw_batched_topk_kernel(
    float*        __restrict__ acc,        // [num_tokens, hidden] f32
    const __half* __restrict__ in,         // [num_tokens, hidden] f16
    const float*  __restrict__ top_w,      // [num_tokens, top_k] f32
    int hidden,
    int top_k,
    int k_round,
    int num_tokens
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;
    if (n >= hidden || m >= num_tokens) return;
    float w = top_w[(long long)m * top_k + k_round];
    long long off = (long long)m * hidden + n;
    acc[off] = acc[off] + w * __half2float(in[off]);
}
