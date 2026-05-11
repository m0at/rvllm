// Mistral 3.5 m=1 decode attention finisher — softmax over scores +
// weighted V sum, per q_head:
//
//   weights[t] = softmax(scores[q_head, :])[t]
//   out[q_head, d] = Σ_t weights[t] * V[t, kv_head_for(q_head), d]
//
// Memory layouts:
//   scores:   F32  [n_q_heads, past_len]
//   v_cache:  BF16 [max_pos, n_kv_heads, head_dim]
//   out:      BF16 [n_q_heads, head_dim]
//
// Launch:
//   Grid:  (n_q_heads, 1, 1)
//   Block: (head_dim, 1, 1)
//   Shared memory: past_len * sizeof(float)
//
// Numerically stable softmax (subtract max, exp, divide by sum).

#include <cuda_bf16.h>
#include <float.h>

extern "C" __global__ void mistral35_softmax_v_bf16_kernel(
    const float* __restrict__ scores,
    const __nv_bfloat16* __restrict__ v_cache,
    __nv_bfloat16* __restrict__ out,
    int head_dim,
    int n_kv_heads,
    int gqa_ratio,
    int past_len
) {
    const int q_head  = blockIdx.x;
    const int kv_head = q_head / gqa_ratio;
    const int d       = threadIdx.x;
    if (d >= head_dim) return;

    extern __shared__ float weights[];

    // Stage 1: thread 0 computes softmax(scores[q_head, :]) → weights[t].
    if (d == 0) {
        float m = -FLT_MAX;
        for (int t = 0; t < past_len; t++) {
            const float s = scores[q_head * past_len + t];
            if (s > m) m = s;
        }
        float sum_exp = 0.0f;
        for (int t = 0; t < past_len; t++) {
            const float e = expf(scores[q_head * past_len + t] - m);
            weights[t] = e;
            sum_exp += e;
        }
        const float inv_sum = (sum_exp > 0.0f) ? 1.0f / sum_exp : 0.0f;
        for (int t = 0; t < past_len; t++) {
            weights[t] *= inv_sum;
        }
    }
    __syncthreads();

    // Stage 2: each thread computes one output dim.
    float acc = 0.0f;
    for (int t = 0; t < past_len; t++) {
        const float v = __bfloat162float(
            v_cache[t * n_kv_heads * head_dim + kv_head * head_dim + d]);
        acc += weights[t] * v;
    }
    out[q_head * head_dim + d] = __float2bfloat16(acc);
}
