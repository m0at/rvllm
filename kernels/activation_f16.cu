// Half-precision activation kernels: SiLU, fused SiLU*mul, GELU.
// Reads/writes f16, computes in f32 for precision.
//
// Launch config (all kernels):
//   Grid:  (ceil(n / 256), 1, 1)
//   Block: (256, 1, 1)
//   Shared memory: none

#include <cuda_fp16.h>

extern "C"
__global__ void silu_f16_kernel(
    __half* __restrict__ output,
    const __half* __restrict__ input,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __half2float(input[idx]);
        output[idx] = __float2half(x / (1.0f + expf(-x)));
    }
}

extern "C"
__global__ void fused_silu_mul_f16_kernel(
    __half* __restrict__ output,
    const __half* __restrict__ gate,
    const __half* __restrict__ up,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __half2float(gate[idx]);
        output[idx] = __float2half((x / (1.0f + expf(-x))) * __half2float(up[idx]));
    }
}

extern "C"
__global__ void gelu_f16_kernel(
    __half* __restrict__ output,
    const __half* __restrict__ input,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __half2float(input[idx]);
        output[idx] = __float2half(0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x))));
    }
}
