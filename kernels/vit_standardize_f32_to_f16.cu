// Per-channel standardize with f32 input, f16 output:
//   out_f16[t, c] = (x_f32[t, c] - bias_f16[c]) * scale_f16[c]
//
// f32-in / f16-out sibling of vit_standardize_f16. Used by the
// f32-pooler path so the pooler→sqrt(hidden)→standardize chain runs
// in f32 (no f16-overflow on peak activations) and only narrows to
// f16 at the very end, where the std_scale multiplication has
// brought magnitudes back into f16-safe range.
//
// Launch:
//   Grid:  (N, 1, 1)
//   Block: (BLOCK, 1, 1)

#include <cuda_fp16.h>

extern "C" __global__ void vit_standardize_f32_to_f16_kernel(
    __half* __restrict__ out,           // [N, hidden] f16
    const float* __restrict__ x,        // [N, hidden] f32
    const __half* __restrict__ bias,    // [hidden] f16
    const __half* __restrict__ scale,   // [hidden] f16
    int hidden
) {
    const int t = blockIdx.x;
    __half* dst = out + (long long)t * hidden;
    const float* row = x + (long long)t * hidden;
    for (int c = threadIdx.x; c < hidden; c += blockDim.x) {
        float v = (row[c] - __half2float(bias[c])) * __half2float(scale[c]);
        dst[c] = __float2half(v);
    }
}
