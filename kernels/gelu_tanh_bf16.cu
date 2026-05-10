// gelu_pytorch_tanh activation, pointwise on bf16 (in-place).
//
// y = 0.5·x·(1 + tanh( sqrt(2/pi) · (x + 0.044715·x³) ))
//
// bf16 sibling of gelu_tanh_f16. Used by Pixtral's projector
// (Mistral 3.5 vision):
//   linear_1 → GELU → linear_2
// where activations stay BF16 throughout.
//
// Launch:
//   Grid:  (ceil(n / BLOCK), 1, 1)
//   Block: (BLOCK, 1, 1)
//
// Round-12 (Pixtral vision phase 3d).

#include <cuda_bf16.h>

extern "C" __global__ void gelu_tanh_bf16_kernel(
    __nv_bfloat16* __restrict__ x,    // [n] in-place
    int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = __bfloat162float(x[i]);
    const float kSqrt2OverPi = 0.7978845608028654f;
    const float kCoeff       = 0.044715f;
    float inner = kSqrt2OverPi * (v + kCoeff * v * v * v);
    float g = 0.5f * v * (1.0f + tanhf(inner));
    x[i] = __float2bfloat16(g);
}
