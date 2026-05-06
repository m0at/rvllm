// scaled_add variant that reads the scalar weight from a device
// pointer instead of receiving it by-value. Used to chain a
// shared_gate_dot_sigmoid output directly into the routed_sum
// accumulator with zero host round-trip (Phase 4b-prep iter21).
//
// acc[i] += devw[0] * f16_to_f32(in[i])     for i in 0..n
//
// Launch:
//   Grid:  (ceil(n / BLOCK), 1, 1)
//   Block: (BLOCK, 1, 1)

#include <cuda_fp16.h>

extern "C" __global__ void scaled_add_f16_to_f32_devw_kernel(
    float*        __restrict__ acc,
    const __half* __restrict__ in,
    const float*  __restrict__ devw,
    int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float w = devw[0];
    acc[i] = acc[i] + w * __half2float(in[i]);
}
