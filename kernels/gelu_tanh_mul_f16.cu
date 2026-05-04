// out = gelu_pytorch_tanh(gate) * up   (pointwise on f16)
//
// Used by Gemma 4 vision MLP whose forward is
//   down(act_fn(gate_proj(x)) * up_proj(x))
// with `act_fn = gelu_pytorch_tanh` (vision_config.hidden_activation).
//
// gelu_pytorch_tanh: y = 0.5·x·(1 + tanh( sqrt(2/pi) · (x + 0.044715·x³) ))

#include <cuda_fp16.h>

extern "C" __global__ void gelu_tanh_mul_f16_kernel(
    __half* __restrict__ output,
    const __half* __restrict__ gate,
    const __half* __restrict__ up,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = __half2float(gate[i]);
    float u = __half2float(up[i]);
    const float kSqrt2OverPi = 0.7978845608028654f;
    const float kCoeff       = 0.044715f;
    float inner = kSqrt2OverPi * (g + kCoeff * g * g * g);
    float gel = 0.5f * g * (1.0f + tanhf(inner));
    output[i] = __float2half(gel * u);
}
