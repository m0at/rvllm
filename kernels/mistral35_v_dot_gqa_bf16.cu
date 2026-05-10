// Mistral 3.5 GQA-aware V-dot for the m=1 decode attention finisher.
//
//   out[qh, d] = Σ_t probs[qh, t] · V[t, kv_head_for(qh), d]
//
// Pairs with `mistral35_softmax_inplace_f32` (which produces the
// normalised probs in `scores`) and replaces the V-load half of
// the legacy `mistral35_softmax_v_bf16` kernel. Old layout:
// grid (n_q_heads), each block re-fetches V[t, kv_head, :] from
// global → 12× redundant V reads per kv_head on Mistral 3.5.
//
// New layout: grid (n_kv_heads), each CTA streams V tiles for one
// kv_head and folds the dot product against all `gqa_ratio`
// Q-heads sharing it in registers. V bandwidth drops by `gqa_ratio`
// (12× for Mistral 3.5).
//
// Memory layouts:
//   probs:    F32  [n_q_heads, past_len]                (in-place softmax output)
//   v_cache:  BF16 [max_pos,  n_kv_heads, head_dim]
//   out:      BF16 [n_q_heads, head_dim]
//
// Hardcoded constants (Mistral 3.5):
//   gqa_ratio = 12  (96 Q-heads / 8 KV-heads). The kernel is
//   templated on it via a compile-time constant; mismatching
//   `gqa_ratio` at launch will be detected by the Rust launcher.
//   head_dim = 128.
//
// Launch:
//   Grid:  (n_kv_heads, 1, 1)
//   Block: (head_dim, 1, 1)  — head_dim threads
//   Smem:  TILE_T * head_dim * 2  (V tile, BF16)
//        + GQA_RATIO * TILE_T * 4 (probs tile, F32)

#include <cuda_bf16.h>

#ifndef MISTRAL35_V_DOT_GQA_TILE_T
#define MISTRAL35_V_DOT_GQA_TILE_T 32
#endif
#ifndef MISTRAL35_V_DOT_GQA_RATIO
#define MISTRAL35_V_DOT_GQA_RATIO 12
#endif

extern "C" __global__ void mistral35_v_dot_gqa_bf16_kernel(
    const float* __restrict__ probs,
    const __nv_bfloat16* __restrict__ v_cache,
    __nv_bfloat16* __restrict__ out,
    int head_dim,
    int n_kv_heads,
    int gqa_ratio,    // must equal MISTRAL35_V_DOT_GQA_RATIO; checked by launcher
    int past_len
) {
    constexpr int TILE_T = MISTRAL35_V_DOT_GQA_TILE_T;
    constexpr int G      = MISTRAL35_V_DOT_GQA_RATIO;

    const int kv = blockIdx.x;
    const int d  = threadIdx.x;
    if (kv >= n_kv_heads || d >= head_dim) return;
    if (gqa_ratio != G) return;  // launcher invariant

    extern __shared__ char smem_raw[];
    __nv_bfloat16* v_tile = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    float*         p_tile = reinterpret_cast<float*>(v_tile + TILE_T * head_dim);

    // Per-thread accumulators: one float per Q-head sharing this kv_head.
    float acc[G];
    #pragma unroll
    for (int g = 0; g < G; ++g) acc[g] = 0.0f;

    const int qh_base = kv * G;
    const int tpb     = blockDim.x;

    for (int t_base = 0; t_base < past_len; t_base += TILE_T) {
        const int tile = (past_len - t_base < TILE_T) ? (past_len - t_base) : TILE_T;

        // Cooperative load of V_tile[tile][head_dim] from global.
        for (int slot = d; slot < tile * head_dim; slot += tpb) {
            const int t  = slot / head_dim;
            const int dd = slot % head_dim;
            v_tile[t * head_dim + dd] =
                v_cache[(long long)(t_base + t) * n_kv_heads * head_dim
                        + (long long)kv * head_dim + dd];
        }
        // Cooperative load of p_tile[G][tile].
        for (int slot = d; slot < G * tile; slot += tpb) {
            const int g = slot / tile;
            const int t = slot % tile;
            p_tile[g * TILE_T + t] =
                probs[(long long)(qh_base + g) * past_len + (t_base + t)];
        }
        __syncthreads();

        // Accumulate for this thread's `d` across all G Q-heads.
        // Tile is small (≤TILE_T); the inner loop unrolls manageably.
        #pragma unroll
        for (int g = 0; g < G; ++g) {
            float a = acc[g];
            for (int t = 0; t < tile; ++t) {
                const float v = __bfloat162float(v_tile[t * head_dim + d]);
                a += p_tile[g * TILE_T + t] * v;
            }
            acc[g] = a;
        }
        __syncthreads();
    }

    // Emit one output row per Q-head in this kv group.
    #pragma unroll
    for (int g = 0; g < G; ++g) {
        const int qh = qh_base + g;
        out[(long long)qh * head_dim + d] = __float2bfloat16(acc[g]);
    }
}
