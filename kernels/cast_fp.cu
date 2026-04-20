// f32 <-> f16 cast kernels for mixed-precision inference.
//
// Launch config:
//   Grid:  (ceil(n / 256), 1, 1)
//   Block: (256, 1, 1)
//   Shared memory: none

#include <cuda_fp16.h>

extern "C"
__global__ void cast_f32_to_f16_kernel(
    __half* __restrict__ output,
    const float* __restrict__ input,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __float2half(input[idx]);
    }
}

extern "C"
__global__ void cast_f16_to_f32_kernel(
    float* __restrict__ output,
    const __half* __restrict__ input,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __half2float(input[idx]);
    }
}
