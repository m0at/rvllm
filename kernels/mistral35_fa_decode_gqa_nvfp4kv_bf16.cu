// Mistral 3.5 fused FlashAttention-2 decode kernel with NVFP4 KV
// cache. Mirrors `mistral35_fa_decode_gqa_bf16_kernel` exactly,
// except the K and V tile loads dequantise from NVFP4 (packed
// nibbles + per-16-element E4M3 scale) into the same BF16 shared
// memory layout the rest of the kernel already consumes.
//
// Storage layout the kernel reads (matches
// mistral35_kv_cache_write_nvfp4_bf16 output):
//   k_packed: u8     [max_pos, n_kv, head_dim/2]
//                    low nibble  = elem 2i
//                    high nibble = elem 2i+1
//   k_scale:  E4M3   [max_pos, n_kv, head_dim/16]
//                    per 16-element block
//   v_packed / v_scale identical.
//
// Online softmax + per-q-head register accumulators are unchanged
// from the BF16-KV sibling. The win is reading 0.5625 bytes per
// KV element instead of 2 bytes — at past_len ≈ 4000 across 88
// layers that turns the dominant KV-bandwidth pressure
// (~700 MB/token BF16) into ~200 MB/token NVFP4. Net: ≈ 40 %
// decode tok/s gain at long context where KV bandwidth exceeds
// weight bandwidth.
//
// Launch:
//   Grid:  (n_kv_heads, 1, 1)
//   Block: (head_dim, 1, 1)
//   Smem:  same as bf16 sibling
//          (Q + K_tile + V_tile + reduction scratch)

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <float.h>

#include "nvfp4_utils.cuh"

#ifndef MISTRAL35_FA_DEC_NVFP4_TILE_T
#define MISTRAL35_FA_DEC_NVFP4_TILE_T 32
#endif
#ifndef MISTRAL35_FA_DEC_NVFP4_G
#define MISTRAL35_FA_DEC_NVFP4_G 12
#endif

extern "C" __global__ void mistral35_fa_decode_gqa_nvfp4kv_bf16_kernel(
    const __nv_bfloat16* __restrict__ q,      // [n_q_heads, head_dim]
    const uint8_t*       __restrict__ k_packed,  // [max_pos, n_kv, head_dim/2]
    const uint8_t*       __restrict__ k_scale,   // [max_pos, n_kv, head_dim/16]
    const uint8_t*       __restrict__ v_packed,  // [max_pos, n_kv, head_dim/2]
    const uint8_t*       __restrict__ v_scale,   // [max_pos, n_kv, head_dim/16]
    __nv_bfloat16*       __restrict__ out,    // [n_q_heads, head_dim]
    int head_dim,
    int n_kv_heads,
    int gqa_ratio,
    int past_len,
    float inv_sqrt_d
) {
    constexpr int TILE_T = MISTRAL35_FA_DEC_NVFP4_TILE_T;
    constexpr int G      = MISTRAL35_FA_DEC_NVFP4_G;

    const int kv = blockIdx.x;
    const int d  = threadIdx.x;
    if (kv >= n_kv_heads || d >= head_dim) return;
    if (gqa_ratio != G) return;

    const int tpb = blockDim.x;
    const int qh_base = kv * G;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* q_sh = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* k_sh = q_sh + G * head_dim;
    __nv_bfloat16* v_sh = k_sh + TILE_T * head_dim;
    float*         red  = reinterpret_cast<float*>(v_sh + TILE_T * head_dim);

    // Load Q rows for all G Q-heads.
    for (int slot = d; slot < G * head_dim; slot += tpb) {
        const int g = slot / head_dim;
        const int dd = slot % head_dim;
        const int qh = qh_base + g;
        q_sh[g * head_dim + dd] = q[qh * head_dim + dd];
    }
    __syncthreads();

    // Per-q-head running state in registers.
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

    const int blocks_per_head = head_dim / 16;  // 8 for head_dim=128

    for (int t_base = 0; t_base < past_len; t_base += TILE_T) {
        const int tile = (past_len - t_base < TILE_T)
                         ? (past_len - t_base) : TILE_T;

        // (a) Cooperative dequant-load K tile from NVFP4 → BF16
        //     shared. Each thread handles one (t, dd) cell; reads
        //     its own scale byte + packed byte, decodes the
        //     correct nibble, writes BF16 into k_sh.
        for (int slot = d; slot < tile * head_dim; slot += tpb) {
            const int t = slot / head_dim;
            const int dd = slot % head_dim;
            const long long t_abs = t_base + t;
            const long long packed_off =
                t_abs * (long long)n_kv_heads * (head_dim / 2)
                + (long long)kv * (head_dim / 2)
                + (dd >> 1);
            const long long scale_off =
                t_abs * (long long)n_kv_heads * (head_dim / 16)
                + (long long)kv * (head_dim / 16)
                + (dd >> 4);
            const uint8_t byte = k_packed[packed_off];
            const __nv_fp8_e4m3 e4m3 =
                *reinterpret_cast<const __nv_fp8_e4m3*>(k_scale + scale_off);
            const float scale = float(e4m3);
            const uint32_t nib = (dd & 1) ? ((byte >> 4) & 0xFu)
                                          : (byte & 0xFu);
            const float val = rvllm_nvfp4::fp4_decode(nib) * scale;
            k_sh[t * head_dim + dd] = __float2bfloat16(val);
        }
        // (b) Same dequant-load for V tile.
        for (int slot = d; slot < tile * head_dim; slot += tpb) {
            const int t = slot / head_dim;
            const int dd = slot % head_dim;
            const long long t_abs = t_base + t;
            const long long packed_off =
                t_abs * (long long)n_kv_heads * (head_dim / 2)
                + (long long)kv * (head_dim / 2)
                + (dd >> 1);
            const long long scale_off =
                t_abs * (long long)n_kv_heads * (head_dim / 16)
                + (long long)kv * (head_dim / 16)
                + (dd >> 4);
            const uint8_t byte = v_packed[packed_off];
            const __nv_fp8_e4m3 e4m3 =
                *reinterpret_cast<const __nv_fp8_e4m3*>(v_scale + scale_off);
            const float scale = float(e4m3);
            const uint32_t nib = (dd & 1) ? ((byte >> 4) & 0xFu)
                                          : (byte & 0xFu);
            const float val = rvllm_nvfp4::fp4_decode(nib) * scale;
            v_sh[t * head_dim + dd] = __float2bfloat16(val);
        }
        __syncthreads();
        (void)blocks_per_head;

        // (c) Compute S[g][t]. Warp-shuffle reduction, same as the
        //     BF16-KV sibling (mistral35_fa_decode_gqa_bf16).
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

        // (d) Online softmax + output accumulation.
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

    // (e) Finalize.
    #pragma unroll
    for (int g = 0; g < G; ++g) {
        const int qh = qh_base + g;
        const float inv_l = (l_state[g] > 0.0f) ? 1.0f / l_state[g] : 0.0f;
        const float final_v = o_state[g] * inv_l;
        out[(long long)qh * head_dim + d] = __float2bfloat16(final_v);
    }
}
