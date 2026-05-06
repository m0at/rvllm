// In-place residual add: inout_f16[i] = f16(f16_to_f32(inout_f16[i])
//                                           + add_f32[i])
//
// Replaces the last per-MoE-layer host round-trip in
// `apply_layer_moe`: DtoH last_hidden f16 + CPU f16+f32 add +
// HtoD residual (Phase 4b-prep iter19). Combined with iter18's
// GPU MoE accumulator, the entire MoE forward is now device-side.
//
// Layout: both `[n]` rows, pointwise. The f32 source is the
// `routed_sum_region` accumulated by iter18's GPU MoE.
//
// Launch:
//   Grid:  (ceil(n / BLOCK), 1, 1)
//   Block: (BLOCK, 1, 1)

#include <cuda_fp16.h>

extern "C" __global__ void f16_plus_f32_inplace_f16_kernel(
    __half*      __restrict__ inout,
    const float* __restrict__ add,
    int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float h = __half2float(inout[i]);
    inout[i] = __float2half(h + add[i]);
}
