// =============================================================
//  mistral35_v_dot_gqa_t_bf16.cu
// =============================================================
//
// Codex #2 new review — batched chunk-prefill V·probs finisher
// for a strip of T queries. Sibling to
// `mistral35_v_dot_gqa_bf16_kernel`; per-(kv, t) numerics
// identical, with a third grid dim covering T.
//
// Causal mask: probs_t entries beyond pos_start + t are 0 (the
// softmax_inplace_f32_t producer normalises only the masked-in
// prefix; positions > pos_start + t were -INF before exp, so
// they end up exactly 0). Iterating the full max_past_len range
// adds zero contribution — byte-for-byte equivalent to a
// length-clamped V·probs on the [0, pos_start + t] prefix.
//
// Memory layouts:
//   probs_t:  F32   [T, n_q,  max_past_len]
//   v_cache:  BF16  [max_pos, n_kv, head_dim]
//   out_t:    BF16  [T, n_q,  head_dim]    (= [T, hidden])
//
// Launch:
//   Grid:  (n_kv_heads, T, 1)
//   Block: (head_dim, 1, 1)
//   Smem:  TILE_T * head_dim * 2 (V tile, BF16)
//        + G * TILE_T * 4        (probs tile, F32)

#include <cuda_bf16.h>

#ifndef MISTRAL35_V_DOT_GQA_T_TILE_T
#define MISTRAL35_V_DOT_GQA_T_TILE_T 32
#endif
#ifndef MISTRAL35_V_DOT_GQA_T_RATIO
#define MISTRAL35_V_DOT_GQA_T_RATIO 12
#endif

extern "C" __global__ void mistral35_v_dot_gqa_t_bf16_kernel(
    const float*         __restrict__ probs_t,   // [T, n_q, max_past_len]
    const __nv_bfloat16* __restrict__ v_cache,   // [max_pos, n_kv, head_dim]
    __nv_bfloat16*       __restrict__ out_t,     // [T, n_q, head_dim]
    int head_dim,
    int n_q_heads,
    int n_kv_heads,
    int gqa_ratio,
    int max_past_len,
    int t_count)
{
    constexpr int TILE_T = MISTRAL35_V_DOT_GQA_T_TILE_T;
    constexpr int G      = MISTRAL35_V_DOT_GQA_T_RATIO;

    const int kv = blockIdx.x;
    const int t  = blockIdx.y;
    const int d  = threadIdx.x;
    if (kv >= n_kv_heads || t >= t_count || d >= head_dim) return;
    if (gqa_ratio != G) return;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* v_tile = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    float*         p_tile = reinterpret_cast<float*>(v_tile + TILE_T * head_dim);

    float acc[G];
    #pragma unroll
    for (int g = 0; g < G; ++g) acc[g] = 0.0f;

    const int qh_base = kv * G;
    const int tpb     = blockDim.x;
    const long long probs_t_row =
        (long long)t * (long long)n_q_heads * max_past_len;

    for (int t_base = 0; t_base < max_past_len; t_base += TILE_T) {
        const int tile = (max_past_len - t_base < TILE_T)
                         ? (max_past_len - t_base) : TILE_T;

        // Cooperative load of V[t_base .. t_base+tile, kv, :].
        for (int slot = d; slot < tile * head_dim; slot += tpb) {
            const int tt = slot / head_dim;
            const int dd = slot % head_dim;
            v_tile[tt * head_dim + dd] =
                v_cache[(long long)(t_base + tt) * n_kv_heads * head_dim
                        + (long long)kv * head_dim + dd];
        }
        // Cooperative load of probs_t[t, qh_base..qh_base+G, t_base..t_base+tile].
        for (int slot = d; slot < G * tile; slot += tpb) {
            const int g  = slot / tile;
            const int tt = slot % tile;
            p_tile[g * TILE_T + tt] =
                probs_t[probs_t_row
                        + (long long)(qh_base + g) * max_past_len
                        + (long long)(t_base + tt)];
        }
        __syncthreads();

        #pragma unroll
        for (int g = 0; g < G; ++g) {
            float a = acc[g];
            for (int tt = 0; tt < tile; ++tt) {
                const float v = __bfloat162float(v_tile[tt * head_dim + d]);
                a += p_tile[g * TILE_T + tt] * v;
            }
            acc[g] = a;
        }
        __syncthreads();
    }

    const long long out_row_stride =
        (long long)n_q_heads * head_dim;
    #pragma unroll
    for (int g = 0; g < G; ++g) {
        const int qh = qh_base + g;
        out_t[(long long)t * out_row_stride
              + (long long)qh * head_dim + d] = __float2bfloat16(acc[g]);
    }
}
