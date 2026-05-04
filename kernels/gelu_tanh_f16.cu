// gelu_pytorch_tanh activation, pointwise on f16.
//
// y = 0.5·x·(1 + tanh( sqrt(2/pi) · (x + 0.044715·x³) ))
//
// Used in:
//   - Qwen ViT MLP (linear_fc1 → GELU → linear_fc2)
//   - Qwen PatchMerger (linear_fc1 → GELU → linear_fc2)
//   - Gemma 4 ViT MLP (gate_proj → GELU → up_proj path? — verify)
//
// Launch:
//   Grid:  (ceil(n / BLOCK), 1, 1)
//   Block: (BLOCK, 1, 1)

#include <cuda_fp16.h>

extern "C" __global__ void gelu_tanh_f16_kernel(
    __half* __restrict__ x,    // [n] in-place
    int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = __half2float(x[i]);
    const float kSqrt2OverPi = 0.7978845608028654f;
    const float kCoeff       = 0.044715f;
    float inner = kSqrt2OverPi * (v + kCoeff * v * v * v);
    float g = 0.5f * v * (1.0f + tanhf(inner));
    x[i] = __float2half(g);
}
