// sigmoid_mul_f16: element-wise `sigmoid(gate) * values` for the
// Qwen 3.6 `attn_output_gate=true` path. The Qwen full-attention
// q_proj output is `[Q half, gate half]`; after RoPE + paged-attn
// produces an attention output of shape `[num_heads, head_dim]`,
// the gate-half of size `[num_heads, head_dim]` is fed through this
// kernel to produce the gated attention output that o_proj consumes.
//
// Math: out[i] = values[i] * (1 / (1 + exp(-gate_logits[i])))
//
// Layout: 1-D over `n = num_heads * head_dim`. Caller fixes the
// per-head split (gate_logits stride matches values stride).

#include <cuda_fp16.h>

extern "C" __global__ void sigmoid_mul_f16_kernel(
    __half* __restrict__ output,           // [n] f16
    const __half* __restrict__ values,     // [n] f16  — attn output
    const __half* __restrict__ gate_logits,// [n] f16  — pre-sigmoid gate
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = __half2float(gate_logits[i]);
    float v = __half2float(values[i]);
    // Numerically-stable sigmoid via 1 / (1 + exp(-x)). For large
    // negative x, expf overflows but the result clamps cleanly to 0
    // through the reciprocal — no NaN or inf surfaces in practice.
    float s = 1.0f / (1.0f + expf(-g));
    output[i] = __float2half(s * v);
}
