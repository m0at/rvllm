// Per-token batched scaled_add for the shared-MoE step. For each
// (token m, hidden n):
//
//   acc[m*hidden + n] += devw[m] * f16_to_f32(in[m*hidden + n])
//
// Replaces N per-token launches of `scaled_add_f16_to_f32_devw`
// (each reading devw[0]) with one launch reading devw[m] per row.
// Different from the topk variant because devw is a flat [N] array
// (one weight per token) rather than [N, top_k] indexed by k_round.
//
// Phase 6c / Round-27.

#include <cuda_fp16.h>

extern "C" __global__ void scaled_add_f16_to_f32_devw_batched_kernel(
    float*        __restrict__ acc,        // [num_tokens, hidden] f32
    const __half* __restrict__ in,         // [num_tokens, hidden] f16
    const float*  __restrict__ devw,       // [num_tokens] f32
    int hidden,
    int num_tokens
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;
    if (n >= hidden || m >= num_tokens) return;
    float w = devw[m];
    long long off = (long long)m * hidden + n;
    acc[off] = acc[off] + w * __half2float(in[off]);
}
