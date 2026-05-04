// SwiGLU activation: out = silu(gate) * up = (gate * sigmoid(gate)) * up
// Used by Gemma 4 vision tower MLP (gate_proj/up_proj/down_proj triplet).
//
// Layout: gate, up, output all [n] f16 row-major. Pointwise.
//
// Launch:
//   Grid:  (ceil(n / BLOCK), 1, 1)
//   Block: (BLOCK, 1, 1)

#include <cuda_fp16.h>

extern "C" __global__ void silu_mul_f16_kernel(
    __half* __restrict__ output,        // [n] f16
    const __half* __restrict__ gate,    // [n] f16
    const __half* __restrict__ up,      // [n] f16
    int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = __half2float(gate[i]);
    float u = __half2float(up[i]);
    float silu = g / (1.0f + expf(-g));   // numerically-stable for moderate |g|
    output[i] = __float2half(silu * u);
}
