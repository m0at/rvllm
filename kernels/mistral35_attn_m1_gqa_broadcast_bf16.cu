// Mistral 3.5 first-token (m=1, pos=0) GQA attention — degenerate case.
//
// At pos=0 with no past KV cache, attention reduces to:
//   for q_head h, dim d:  out[h, d] = softmax(Q_h * K_kv(h)) * V_kv(h)
// With a single key (length-1 sequence), softmax over 1 element is 1.0,
// so out[h, d] = V_kv(h)[d]. RoPE at pos=0 is identity (cos=1, sin=0)
// so Q/K rotation can be skipped on the smoke path.
//
// This kernel performs the GQA broadcast: V is stored compactly as
// [n_kv_heads, head_dim] BF16 row-major; output is [n_q_heads, head_dim]
// BF16 with each q_head copying from kv_head = q_head / gqa_ratio.
//
// Launch:
//   Grid:  (n_q_heads, 1, 1)
//   Block: (head_dim, 1, 1)   — head_dim ≤ 1024 (Mistral 3.5: 128)

#include <cuda_bf16.h>

extern "C" __global__ void mistral35_attn_m1_gqa_broadcast_bf16_kernel(
    __nv_bfloat16* __restrict__ output,      // [n_q_heads * head_dim]
    const __nv_bfloat16* __restrict__ v,     // [n_kv_heads * head_dim]
    int head_dim,
    int gqa_ratio                            // n_q_heads / n_kv_heads
) {
    const int q_head  = blockIdx.x;
    const int d       = threadIdx.x;
    if (d >= head_dim) return;
    const int kv_head = q_head / gqa_ratio;
    output[q_head * head_dim + d] = v[kv_head * head_dim + d];
}
