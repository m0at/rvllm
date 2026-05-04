// Per-channel standardize: x[t, c] = (x[t, c] - bias[c]) * scale[c]
//
// Used post-pooler in Gemma 4 vision (`config.standardize=True`).
//
// Launch:
//   Grid:  (N, 1, 1)
//   Block: (BLOCK, 1, 1)

#include <cuda_fp16.h>

extern "C" __global__ void vit_standardize_f16_kernel(
    __half* __restrict__ x,             // [N, hidden] in-place
    const __half* __restrict__ bias,    // [hidden]
    const __half* __restrict__ scale,   // [hidden]
    int hidden
) {
    const int t = blockIdx.x;
    __half* row = x + (long long)t * hidden;
    for (int c = threadIdx.x; c < hidden; c += blockDim.x) {
        float v = (__half2float(row[c]) - __half2float(bias[c])) * __half2float(scale[c]);
        row[c] = __float2half(v);
    }
}
