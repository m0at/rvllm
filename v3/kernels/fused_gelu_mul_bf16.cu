// GELU(tanh)(gate) * up -> BF16 output (no FP8 quantization).
//
// Cycle 55 step 6 (Phase B): bf16 sibling of fused_gelu_mul_f16. Same
// math (f32 internal computation, GELU(tanh) approximation, gate × up);
// I/O dtypes flip __half → __nv_bfloat16. Used under bf16-native MLP
// chain dispatch.
//
// Input:  gate_up [num_tokens, 2 * intermediate] bf16 (gate || up interleaved)
// Output: out_bf16 [num_tokens, intermediate] bf16
#include <cuda_bf16.h>
#include <math.h>

extern "C" __global__ void __launch_bounds__(1024)
fused_gelu_mul_bf16_kernel(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ gate_up,
    int intermediate
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int gate_offset = row * 2 * intermediate;
    const int up_offset = gate_offset + intermediate;
    const int out_offset = row * intermediate;

    for (int i = tid; i < intermediate; i += stride) {
        float g = __bfloat162float(gate_up[gate_offset + i]);
        float u = __bfloat162float(gate_up[up_offset + i]);
        // GELU(tanh) approximation
        float g3 = g * g * g;
        float inner = 0.7978845608f * (g + 0.044715f * g3);
        float gelu = 0.5f * g * (1.0f + tanhf(inner));
        output[out_offset + i] = __float2bfloat16(gelu * u);
    }
}
