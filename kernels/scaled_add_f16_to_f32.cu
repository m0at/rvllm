// Per-expert weighted accumulator for Qwen 3.6 MoE.
//
// acc_f32[i] += weight * f16_to_f32(in_f16[i])    for i in 0..n
//
// Replaces the per-expert host pipeline (DtoH down_region + CPU loop
// f16→f32+scaled-add into routed_sum) with a stream-ordered launch
// (Phase 4b-prep iter13). One launch per expert (top_k=8) per MoE
// layer (30) per token kills 240 small DtoH per token plus the same
// number of CPU passes over n_down=2048 elements.
//
// Inputs all on device:
//   acc_f32 : [n] f32, in/out (must be zeroed before the first
//             expert of a layer)
//   in_f16  : [n] f16
//   weight  : f32 scalar (host-side router-softmax weight or
//             shared-expert sigmoid gate)
//
// Launch:
//   Grid:  (ceil(n / BLOCK), 1, 1)
//   Block: (BLOCK, 1, 1)

#include <cuda_fp16.h>

extern "C" __global__ void scaled_add_f16_to_f32_kernel(
    float*        __restrict__ acc,
    const __half* __restrict__ in,
    float weight,
    int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    acc[i] = acc[i] + weight * __half2float(in[i]);
}
