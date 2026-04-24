// out[row, col] = GELU(gate[row, col]) * up[row, col] for f16 matrices.
#include <cuda_fp16.h>

__device__ __forceinline__ float gelu_tanh_approx(float x) {
    const float k = 0.7978845608028654f; // sqrt(2/pi)
    const float c = 0.044715f;
    float inner = k * (x + c * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

extern "C" __global__ void fused_gelu_mul_two_f16_kernel(
    __half* __restrict__ out,
    const __half* __restrict__ gate,
    const __half* __restrict__ up,
    int intermediate
) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= intermediate) return;

    int idx = row * intermediate + col;
    float g = __half2float(gate[idx]);
    float u = __half2float(up[idx]);
    out[idx] = __float2half(gelu_tanh_approx(g) * u);
}
