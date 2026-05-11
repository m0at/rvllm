// out = gelu_pytorch_tanh(gate) * up   pointwise, bf16 sibling.

#include <cuda_bf16.h>

extern "C" __global__ void gelu_tanh_mul_bf16_kernel(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = __bfloat162float(gate[i]);
    float u = __bfloat162float(up[i]);
    const float kSqrt2OverPi = 0.7978845608028654f;
    const float kCoeff       = 0.044715f;
    float inner = kSqrt2OverPi * (g + kCoeff * g * g * g);
    float gel = 0.5f * g * (1.0f + tanhf(inner));
    output[i] = __float2bfloat16(gel * u);
}
