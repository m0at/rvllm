// SwiGLU activation: out = silu(gate) * up = (gate * sigmoid(gate)) * up
// BF16 sibling of silu_mul_f16, used by Mistral 3.5 MLP
// (gate_proj/up_proj/down_proj). Inputs and output are BF16 row-major.
//
// Launch:
//   Grid:  (ceil(n / BLOCK), 1, 1)
//   Block: (BLOCK, 1, 1)

#include <cuda_bf16.h>

extern "C" __global__ void silu_mul_bf16_kernel(
    __nv_bfloat16* __restrict__ output,        // [n] bf16
    const __nv_bfloat16* __restrict__ gate,    // [n] bf16
    const __nv_bfloat16* __restrict__ up,      // [n] bf16
    int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = __bfloat162float(gate[i]);
    float u = __bfloat162float(up[i]);
    float silu = g / (1.0f + expf(-g));
    output[i] = __float2bfloat16(silu * u);
}
