// Mistral 3.5 Q·Kᵀ scoring for the m=1 decode attention path.
// Computes scores[q_head, t] = (Q[q_head] · K[t, kv_head_for(q_head)]) / sqrt(d)
// for all (q_head, t in 0..past_len). Output is F32.
//
// Memory layouts (BF16):
//   Q:        [n_q_heads,  head_dim]
//   K_cache:  [max_pos,    n_kv_heads, head_dim]
//
// Output layout (F32):
//   scores:   [n_q_heads,  past_len]
//
// Launch:
//   Grid:  (n_q_heads, past_len, 1)
//   Block: (head_dim, 1, 1)   — head_dim ≤ 1024
//   Shared memory: head_dim * sizeof(float)

#include <cuda_bf16.h>

extern "C" __global__ void mistral35_qk_dot_bf16_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    float* __restrict__ scores,
    int head_dim,
    int n_kv_heads,
    int gqa_ratio,
    int past_len,
    float inv_sqrt_d
) {
    const int q_head = blockIdx.x;
    const int t      = blockIdx.y;
    const int d      = threadIdx.x;
    if (t >= past_len || d >= head_dim) return;

    const int kv_head = q_head / gqa_ratio;
    const float qv = __bfloat162float(q[q_head * head_dim + d]);
    const float kv = __bfloat162float(
        k_cache[t * n_kv_heads * head_dim + kv_head * head_dim + d]);

    extern __shared__ float smem[];
    smem[d] = qv * kv;
    __syncthreads();

    // Tree reduction over head_dim threads. head_dim is even (128 in
    // Mistral 3.5); each step halves the active range.
    for (int s = head_dim >> 1; s > 0; s >>= 1) {
        if (d < s) smem[d] += smem[d + s];
        __syncthreads();
    }
    if (d == 0) {
        scores[q_head * past_len + t] = smem[0] * inv_sqrt_d;
    }
}
