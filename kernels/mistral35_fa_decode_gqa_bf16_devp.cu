// Device-pointer past_len variant of mistral35_fa_decode_gqa_bf16
// — codex review #2 graph-capture follow-up. Math, fragment layout,
// online-softmax accumulator, and warp-shuffle reduction are 1:1
// with `mistral35_fa_decode_gqa_bf16_kernel`. The only API change:
// `past_len` is passed by device pointer so the captured graph
// can be replayed across decode steps without rewriting kernel
// node parameters.

#include <cuda_bf16.h>
#include <float.h>

#ifndef MISTRAL35_FA_DEC_TILE_T
#define MISTRAL35_FA_DEC_TILE_T 32
#endif
#ifndef MISTRAL35_FA_DEC_G
#define MISTRAL35_FA_DEC_G 12
#endif

extern "C" __global__ void mistral35_fa_decode_gqa_bf16_devp_kernel(
    const __nv_bfloat16* __restrict__ q,         // [n_q_heads, head_dim]
    const __nv_bfloat16* __restrict__ k_cache,   // [max_pos, n_kv, head_dim]
    const __nv_bfloat16* __restrict__ v_cache,   // [max_pos, n_kv, head_dim]
    __nv_bfloat16* __restrict__ out,             // [n_q_heads, head_dim]
    int head_dim,
    int n_kv_heads,
    int gqa_ratio,
    const int* __restrict__ past_len_ptr,
    float inv_sqrt_d
) {
    constexpr int TILE_T = MISTRAL35_FA_DEC_TILE_T;
    constexpr int G      = MISTRAL35_FA_DEC_G;

    const int kv = blockIdx.x;
    const int d  = threadIdx.x;
    if (kv >= n_kv_heads || d >= head_dim) return;
    if (gqa_ratio != G) return;
    const int past_len = *past_len_ptr;

    const int tpb = blockDim.x;
    const int qh_base = kv * G;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* q_sh = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* k_sh = q_sh + G * head_dim;
    __nv_bfloat16* v_sh = k_sh + TILE_T * head_dim;
    float*         red  = reinterpret_cast<float*>(v_sh + TILE_T * head_dim);

    for (int slot = d; slot < G * head_dim; slot += tpb) {
        const int g = slot / head_dim;
        const int dd = slot % head_dim;
        const int qh = qh_base + g;
        q_sh[g * head_dim + dd] = q[qh * head_dim + dd];
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
            const int t = slot / head_dim;
            const int dd = slot % head_dim;
            k_sh[t * head_dim + dd] =
                k_cache[(long long)(t_base + t) * n_kv_heads * head_dim
                        + (long long)kv * head_dim + dd];
        }
        for (int slot = d; slot < tile * head_dim; slot += tpb) {
            const int t = slot / head_dim;
            const int dd = slot % head_dim;
            v_sh[t * head_dim + dd] =
                v_cache[(long long)(t_base + t) * n_kv_heads * head_dim
                        + (long long)kv * head_dim + dd];
        }
        __syncthreads();

        const int warp_id = d >> 5;
        const int lane_id = d & 31;
        #pragma unroll
        for (int g = 0; g < G; ++g) {
            for (int t = 0; t < TILE_T; ++t) {
                if (t >= tile) { S_tile[g][t] = -FLT_MAX; continue; }
                const float qv = __bfloat162float(q_sh[g * head_dim + d]);
                const float kv_val = __bfloat162float(k_sh[t * head_dim + d]);
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
                S_tile[g][t] = red[0] * inv_sqrt_d;
            }
        }

        #pragma unroll
        for (int g = 0; g < G; ++g) {
            float tile_max = -FLT_MAX;
            for (int t = 0; t < tile; ++t) {
                if (S_tile[g][t] > tile_max) tile_max = S_tile[g][t];
            }
            const float new_max = (tile_max > m_state[g]) ? tile_max : m_state[g];
            const float corr = (m_state[g] == -FLT_MAX)
                                ? 0.0f
                                : __expf(m_state[g] - new_max);
            l_state[g] = l_state[g] * corr;
            o_state[g] = o_state[g] * corr;
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

    #pragma unroll
    for (int g = 0; g < G; ++g) {
        const int qh = qh_base + g;
        const float inv_l = (l_state[g] > 0.0f) ? 1.0f / l_state[g] : 0.0f;
        const float final_v = o_state[g] * inv_l;
        out[(long long)qh * head_dim + d] = __float2bfloat16(final_v);
    }
}
