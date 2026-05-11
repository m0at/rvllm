// Mistral 3.5 batched causal FA-2 prefill kernel — codex review #2.
//
// Drop-in batched form of `mistral35_fa_decode_gqa_bf16_kernel`. The
// decode kernel processes one query row (m=1) against the full KV
// cache. The chunked-prefill path previously called it T times per
// layer, one launch per query row inside the chunk. At T=383 / 88
// layers that's 33K kernel launches per request for attention alone,
// and the per-launch overhead bottlenecks the path.
//
// This kernel processes all T queries in a single launch:
//   Grid:  (n_kv_heads, T, 1)
//   Block: (head_dim, 1, 1)
//
// Per (kv, t) pair the FA-2 math is *unchanged*: the same online
// softmax, the same K/V tile streaming, the same warp-shuffle
// score reduction. The only differences from the decode kernel are:
//
//   * Q is indexed at `[t, qh, d]` rather than `[qh, d]`.
//   * Output is indexed at `[t, qh, d]` rather than `[qh, d]`.
//   * The causal mask: query at row t attends to KV slots
//     `[0, pos_start + t]` inclusive. The kernel derives this from
//     `pos_start + blockIdx.y + 1` and treats it exactly like the
//     decode kernel's `past_len`.
//
// Memory layouts:
//   q_t:      BF16  [T,         n_q_heads,  head_dim]
//   k_cache:  BF16  [max_pos,   n_kv_heads, head_dim]
//   v_cache:  BF16  [max_pos,   n_kv_heads, head_dim]
//   out_t:    BF16  [T,         n_q_heads,  head_dim]
//
// Hardcoded constants (Mistral 3.5): gqa_ratio = 12, head_dim = 128.

#include <cuda_bf16.h>
#include <float.h>

#ifndef MISTRAL35_FA_PREF_TILE_T
#define MISTRAL35_FA_PREF_TILE_T 32
#endif
#ifndef MISTRAL35_FA_PREF_G
#define MISTRAL35_FA_PREF_G 12
#endif

extern "C" __global__ void mistral35_fa_prefill_gqa_bf16_kernel(
    const __nv_bfloat16* __restrict__ q_t,      // [T, n_q,  head_dim]
    const __nv_bfloat16* __restrict__ k_cache,  // [max_pos, n_kv, head_dim]
    const __nv_bfloat16* __restrict__ v_cache,  // [max_pos, n_kv, head_dim]
    __nv_bfloat16*       __restrict__ out_t,    // [T, n_q,  head_dim]
    int head_dim,
    int n_kv_heads,
    int gqa_ratio,
    int pos_start,
    int t_count,
    float inv_sqrt_d
) {
    constexpr int TILE_T = MISTRAL35_FA_PREF_TILE_T;
    constexpr int G      = MISTRAL35_FA_PREF_G;

    const int kv = blockIdx.x;
    const int t  = blockIdx.y;
    const int d  = threadIdx.x;
    if (kv >= n_kv_heads || t >= t_count || d >= head_dim) return;
    if (gqa_ratio != G) return;  // launcher invariant

    const int tpb = blockDim.x;
    const int qh_base = kv * G;
    const int n_q_heads = n_kv_heads * G;
    const int past_len = pos_start + t + 1;  // causal: inclusive

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

    // Load Q rows for all G Q-heads (this kv-group, this t) into smem.
    const long long q_row_stride = (long long)n_q_heads * head_dim;
    for (int slot = d; slot < G * head_dim; slot += tpb) {
        const int g  = slot / head_dim;
        const int dd = slot % head_dim;
        const int qh = qh_base + g;
        q_sh[g * head_dim + dd] =
            q_t[(long long)t * q_row_stride + (long long)qh * head_dim + dd];
    }
    __syncthreads();

    float m_state[G];
    float l_state[G];
    float o_state[G];
    #pragma unroll
    for (int g = 0; g < G; ++g) {
        m_state[g] = -FLT_MAX;
        l_state[g] = 0.0f;
        o_state[g] = 0.0f;
    }

    float S_tile[G][TILE_T];

    for (int t_base = 0; t_base < past_len; t_base += TILE_T) {
        const int tile = (past_len - t_base < TILE_T)
                         ? (past_len - t_base) : TILE_T;

        for (int slot = d; slot < tile * head_dim; slot += tpb) {
            const int tt = slot / head_dim;
            const int dd = slot % head_dim;
            k_sh[tt * head_dim + dd] =
                k_cache[(long long)(t_base + tt) * n_kv_heads * head_dim
                        + (long long)kv * head_dim + dd];
        }
        for (int slot = d; slot < tile * head_dim; slot += tpb) {
            const int tt = slot / head_dim;
            const int dd = slot % head_dim;
            v_sh[tt * head_dim + dd] =
                v_cache[(long long)(t_base + tt) * n_kv_heads * head_dim
                        + (long long)kv * head_dim + dd];
        }
        __syncthreads();

        const int warp_id = d >> 5;
        const int lane_id = d & 31;
        #pragma unroll
        for (int g = 0; g < G; ++g) {
            for (int tt = 0; tt < TILE_T; ++tt) {
                if (tt >= tile) { S_tile[g][tt] = -FLT_MAX; continue; }
                const float qv = __bfloat162float(q_sh[g * head_dim + d]);
                const float kv_val = __bfloat162float(k_sh[tt * head_dim + d]);
                float prod = qv * kv_val;
                prod += __shfl_xor_sync(0xFFFFFFFFu, prod, 16);
                prod += __shfl_xor_sync(0xFFFFFFFFu, prod, 8);
                prod += __shfl_xor_sync(0xFFFFFFFFu, prod, 4);
                prod += __shfl_xor_sync(0xFFFFFFFFu, prod, 2);
                prod += __shfl_xor_sync(0xFFFFFFFFu, prod, 1);
                if (lane_id == 0) red[warp_id] = prod;
                __syncthreads();
                if (warp_id == 0) {
                    float reduced = (lane_id < 4) ? red[lane_id] : 0.0f;
                    reduced += __shfl_xor_sync(0xFFFFFFFFu, reduced, 2);
                    reduced += __shfl_xor_sync(0xFFFFFFFFu, reduced, 1);
                    if (lane_id == 0) red[0] = reduced;
                }
                __syncthreads();
                S_tile[g][tt] = red[0] * inv_sqrt_d;
            }
        }

        #pragma unroll
        for (int g = 0; g < G; ++g) {
            float tile_max = -FLT_MAX;
            for (int tt = 0; tt < tile; ++tt) {
                if (S_tile[g][tt] > tile_max) tile_max = S_tile[g][tt];
            }
            const float new_max = (tile_max > m_state[g]) ? tile_max : m_state[g];
            const float corr = (m_state[g] == -FLT_MAX)
                                ? 0.0f
                                : __expf(m_state[g] - new_max);
            l_state[g] = l_state[g] * corr;
            o_state[g] = o_state[g] * corr;
            for (int tt = 0; tt < tile; ++tt) {
                const float p = __expf(S_tile[g][tt] - new_max);
                l_state[g] += p;
                const float v_val = __bfloat162float(v_sh[tt * head_dim + d]);
                o_state[g] += p * v_val;
            }
            m_state[g] = new_max;
        }
        __syncthreads();
    }

    #pragma unroll
    for (int g = 0; g < G; ++g) {
        const int qh = qh_base + g;
        const float inv_l = (l_state[g] > 0.0f) ? 1.0f / l_state[g] : 0.0f;
        const float final_v = o_state[g] * inv_l;
        out_t[(long long)t * q_row_stride + (long long)qh * head_dim + d] =
            __float2bfloat16(final_v);
    }
}
