// =============================================================
//  mistral35_qk_dot_gqa_t_bf16.cu
// =============================================================
//
// Codex #2 new review — batched chunk-prefill QK·dot for a strip
// of T queries. Identical per-(t, qh, k_pos) numerics to the m=1
// `mistral35_qk_dot_gqa_bf16_kernel`; the only structural change
// is a third grid dim covering `t_count` queries instead of one.
// Numerical reductions are bit-for-bit the same (same tree-reduce
// over head_dim threads, same inv_sqrt_d ordering), so the per-T
// loop callers can swap in this kernel for byte-identical scores.
//
// Memory layouts:
//   q_t:        BF16  [T, n_q_heads,  head_dim]
//   k_cache:    BF16  [max_pos,       n_kv_heads, head_dim]
//   scores_t:   F32   [T, n_q_heads,  max_past_len]
//
// Causal mask: each row t's softmax-input slice covers positions
// [0, pos_start + t]. Positions beyond that range are written
// `-INFINITY` so the downstream softmax kernel masks them out
// naturally (exp(-INF) → 0, so they neither contribute to max
// nor to sum).
//
// Launch:
//   Grid:  (n_kv_heads, max_past_len, T)
//   Block: (head_dim, 1, 1)
//   Smem:  head_dim * sizeof(float)
//
// Per CTA (kv, k_pos, t):
//   1. Cooperative load of K[k_pos, kv, :] into smem.
//   2. For each gi in [0, gqa_ratio):
//        qh = kv * gqa_ratio + gi
//        product[d] = Q_t[t, qh, d] * K[k_pos, kv, d]
//        reduce → score
//        if k_pos > pos_start + t: score = -INF
//        scores_t[t, qh, k_pos] = score * inv_sqrt_d

#include <cuda_bf16.h>
#include <float.h>

extern "C" __global__ void mistral35_qk_dot_gqa_t_bf16_kernel(
    const __nv_bfloat16* __restrict__ q_t,           // [T, n_q,  head_dim]
    const __nv_bfloat16* __restrict__ k_cache,       // [max_pos, n_kv, head_dim]
    float*               __restrict__ scores_t,      // [T, n_q,  max_past_len]
    int head_dim,
    int n_q_heads,
    int n_kv_heads,
    int gqa_ratio,
    int max_past_len,
    int pos_start,
    int t_count,
    float inv_sqrt_d)
{
    const int kv_head = blockIdx.x;
    const int k_pos   = blockIdx.y;
    const int t       = blockIdx.z;
    const int d       = threadIdx.x;
    if (kv_head >= n_kv_heads || k_pos >= max_past_len || t >= t_count) return;
    if (d >= head_dim) return;

    extern __shared__ float smem[];

    // Stage 1: load K[k_pos, kv, :] into smem once.
    const float k_val = __bfloat162float(
        k_cache[(long long)k_pos * n_kv_heads * head_dim
                + (long long)kv_head * head_dim + d]);
    smem[d] = k_val;
    __syncthreads();

    const long long q_row_stride = (long long)n_q_heads * head_dim;
    const long long sc_row_stride = (long long)n_q_heads * (long long)max_past_len;
    const int causal_limit = pos_start + t;
    const bool out_of_range = (k_pos > causal_limit);

    const int qh_base = kv_head * gqa_ratio;
    #pragma unroll 1
    for (int gi = 0; gi < gqa_ratio; ++gi) {
        const int qh = qh_base + gi;
        // Q[t, qh, d]
        const float qv = __bfloat162float(
            q_t[(long long)t * q_row_stride
                + (long long)qh * head_dim + d]);
        smem[d] = qv * k_val;
        __syncthreads();
        for (int s = head_dim >> 1; s > 0; s >>= 1) {
            if (d < s) smem[d] += smem[d + s];
            __syncthreads();
        }
        if (d == 0) {
            const float score = out_of_range
                ? -FLT_MAX
                : smem[0] * inv_sqrt_d;
            scores_t[(long long)t * sc_row_stride
                     + (long long)qh * max_past_len + k_pos] = score;
        }
        __syncthreads();
    }
}
