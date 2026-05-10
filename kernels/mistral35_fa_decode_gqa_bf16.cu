// Mistral 3.5 fused FlashAttention-2 decode kernel for the m=1
// generation path. Single kernel replaces the existing two-kernel
// pipeline (mistral35_qk_dot_gqa_bf16 + mistral35_softmax_v_bf16
// OR mistral35_softmax_inplace_f32 + mistral35_v_dot_gqa_bf16):
//
//   out[qh, d] = Σ_t softmax(Q[qh] · K[t]^T / √d)[t] * V[t, d]
//
// Online softmax keeps (m, l) running max + sum per Q-head in
// registers — scores and probs never materialize in DRAM. Q is
// loaded once into shared memory; K and V are streamed in tiles
// per kv_head (one CTA per kv_head shares each tile across the
// `gqa_ratio = 12` Q-heads that map to it).
//
// Memory layouts:
//   q:        BF16  [n_q_heads,  head_dim]
//   k_cache:  BF16  [max_pos,   n_kv_heads, head_dim]
//   v_cache:  BF16  [max_pos,   n_kv_heads, head_dim]
//   out:      BF16  [n_q_heads,  head_dim]
//
// Hardcoded constants (Mistral 3.5):
//   gqa_ratio = 12   (96 Q-heads / 8 KV-heads)
//   head_dim  = 128
// The kernel exits cleanly if a launcher passes a different
// gqa_ratio; the launcher must enforce this.
//
// Launch:
//   Grid:  (n_kv_heads, 1, 1)
//   Block: (head_dim, 1, 1)   — head_dim threads
//   Smem:  G*head_dim*2 (Q)
//        + TILE_T*head_dim*2 (K tile)
//        + TILE_T*head_dim*2 (V tile)
//        + reduction scratch
//
// FlashAttention-2 online softmax:
//   For each tile s of K[s..s+TILE_T]:
//     compute partial scores S[g][t] = Q[g] · K[s+t] * inv_sqrt_d
//     new_max = max(running_max, max_t S[g][t])
//     correction = exp(running_max - new_max)
//     running_sum *= correction
//     running_out *= correction
//     for t in tile:
//       p = exp(S[g][t] - new_max)
//       running_sum += p
//       running_out[d] += p * V[s+t, d]
//     running_max = new_max
//   final: out[d] = running_out[d] / running_sum

#include <cuda_bf16.h>
#include <float.h>

#ifndef MISTRAL35_FA_DEC_TILE_T
#define MISTRAL35_FA_DEC_TILE_T 32
#endif
#ifndef MISTRAL35_FA_DEC_G
#define MISTRAL35_FA_DEC_G 12
#endif

extern "C" __global__ void mistral35_fa_decode_gqa_bf16_kernel(
    const __nv_bfloat16* __restrict__ q,         // [n_q_heads, head_dim]
    const __nv_bfloat16* __restrict__ k_cache,   // [max_pos, n_kv, head_dim]
    const __nv_bfloat16* __restrict__ v_cache,   // [max_pos, n_kv, head_dim]
    __nv_bfloat16* __restrict__ out,             // [n_q_heads, head_dim]
    int head_dim,
    int n_kv_heads,
    int gqa_ratio,
    int past_len,
    float inv_sqrt_d
) {
    constexpr int TILE_T = MISTRAL35_FA_DEC_TILE_T;
    constexpr int G      = MISTRAL35_FA_DEC_G;

    const int kv = blockIdx.x;
    const int d  = threadIdx.x;
    if (kv >= n_kv_heads || d >= head_dim) return;
    if (gqa_ratio != G) return;  // launcher invariant

    const int tpb = blockDim.x;
    const int qh_base = kv * G;

    // Shared layout (byte offsets):
    //   q_sh    : [G][head_dim] BF16
    //   k_sh    : [TILE_T][head_dim] BF16
    //   v_sh    : [TILE_T][head_dim] BF16
    //   smem_red: [tpb] f32 (reduction scratch)
    extern __shared__ char smem_raw[];
    __nv_bfloat16* q_sh = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* k_sh = q_sh + G * head_dim;
    __nv_bfloat16* v_sh = k_sh + TILE_T * head_dim;
    float*         red  = reinterpret_cast<float*>(v_sh + TILE_T * head_dim);

    // Load Q rows for all G Q-heads sharing this kv_head into smem.
    for (int slot = d; slot < G * head_dim; slot += tpb) {
        const int g = slot / head_dim;
        const int dd = slot % head_dim;
        const int qh = qh_base + g;
        q_sh[g * head_dim + dd] = q[qh * head_dim + dd];
    }
    __syncthreads();

    // Per-(Q-head, head_dim) running state. Each thread owns one
    // `d` and holds the running output accumulator + (max, sum)
    // for every Q-head in this kv group.
    float m_state[G];
    float l_state[G];
    float o_state[G];   // running output for this thread's `d`, per q-head
    #pragma unroll
    for (int g = 0; g < G; ++g) {
        m_state[g] = -FLT_MAX;
        l_state[g] = 0.0f;
        o_state[g] = 0.0f;
    }

    // Per-tile scratch: each thread also needs the tile's S[g][t]
    // values to compute exp(s - new_max) and the new running_sum.
    // Stash them in registers via stack array; TILE_T is small.
    float S_tile[G][TILE_T];

    for (int t_base = 0; t_base < past_len; t_base += TILE_T) {
        const int tile = (past_len - t_base < TILE_T)
                         ? (past_len - t_base) : TILE_T;

        // (a) Cooperative load K tile [tile][head_dim].
        for (int slot = d; slot < tile * head_dim; slot += tpb) {
            const int t = slot / head_dim;
            const int dd = slot % head_dim;
            k_sh[t * head_dim + dd] =
                k_cache[(long long)(t_base + t) * n_kv_heads * head_dim
                        + (long long)kv * head_dim + dd];
        }
        // (b) Cooperative load V tile [tile][head_dim].
        for (int slot = d; slot < tile * head_dim; slot += tpb) {
            const int t = slot / head_dim;
            const int dd = slot % head_dim;
            v_sh[t * head_dim + dd] =
                v_cache[(long long)(t_base + t) * n_kv_heads * head_dim
                        + (long long)kv * head_dim + dd];
        }
        __syncthreads();

        // (c) Compute S[g][t] = Q[g, :] · K[t, :] * inv_sqrt_d for
        //     all (g in 0..G, t in 0..tile). Use one parallel
        //     reduction per (g, t) over head_dim threads.
        #pragma unroll
        for (int g = 0; g < G; ++g) {
            for (int t = 0; t < TILE_T; ++t) {
                if (t >= tile) { S_tile[g][t] = -FLT_MAX; continue; }
                const float qv = __bfloat162float(q_sh[g * head_dim + d]);
                const float kv_val = __bfloat162float(k_sh[t * head_dim + d]);
                float prod = qv * kv_val;
                // Block reduce over `d` threads via shared `red`.
                red[d] = prod;
                __syncthreads();
                for (int s = tpb >> 1; s > 0; s >>= 1) {
                    if (d < s) red[d] += red[d + s];
                    __syncthreads();
                }
                S_tile[g][t] = red[0] * inv_sqrt_d;
                __syncthreads();  // reuse `red` next iteration
            }
        }

        // (d) Online softmax update + output accumulation.
        //     Per-q-head: find tile max, correct running state,
        //     accumulate p*V for this thread's `d`.
        #pragma unroll
        for (int g = 0; g < G; ++g) {
            // Tile max for this q-head.
            float tile_max = -FLT_MAX;
            for (int t = 0; t < tile; ++t) {
                if (S_tile[g][t] > tile_max) tile_max = S_tile[g][t];
            }
            const float new_max = (tile_max > m_state[g]) ? tile_max : m_state[g];
            const float corr = (m_state[g] == -FLT_MAX)
                                ? 0.0f
                                : __expf(m_state[g] - new_max);
            // Correct the running sum.
            l_state[g] = l_state[g] * corr;
            // Correct the running output (this thread's d slot only).
            o_state[g] = o_state[g] * corr;
            // Add tile contributions.
            for (int t = 0; t < tile; ++t) {
                const float p = __expf(S_tile[g][t] - new_max);
                l_state[g] += p;
                const float v_val =
                    __bfloat162float(v_sh[t * head_dim + d]);
                o_state[g] += p * v_val;
            }
            m_state[g] = new_max;
        }
        __syncthreads();
    }

    // (e) Finalize: divide by running sum and write back.
    #pragma unroll
    for (int g = 0; g < G; ++g) {
        const int qh = qh_base + g;
        const float inv_l = (l_state[g] > 0.0f) ? 1.0f / l_state[g] : 0.0f;
        const float final_v = o_state[g] * inv_l;
        out[(long long)qh * head_dim + d] = __float2bfloat16(final_v);
    }
}
