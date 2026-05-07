// Row-batched indirect-expert variant of dual_silu_indirect for
// the gate+up FFN-up step in Qwen 3.6 MoE. Per token row m, reads
// its expert index from top_idx[m * top_k + k_round]. Equivalent
// to launching the per-token kernel with M=1 once per (token,
// k_round) pair, but in ONE launch with grid.y = num_tokens.
//
// Phase 6b / Round-27.

#include <cuda_fp16.h>

__device__ __forceinline__ float fp8e4m3_to_float_dsi_rt(unsigned char val) {
    unsigned int s = (val >> 7) & 1u;
    unsigned int e = (val >> 3) & 0xFu;
    unsigned int m = val & 0x7u;
    unsigned int f32_bits = (s << 31) | ((e + 120u) << 23) | (m << 20);
    unsigned int is_normal = (e != 0u) & ((e != 0xFu) | (m != 0x7u));
    f32_bits &= (unsigned int)(-(int)is_normal);
    return __uint_as_float(f32_bits);
}
__device__ __forceinline__ void fp8x2_to_f32_dsi_rt(unsigned short packed_fp8x2,
                                                     float& f0, float& f1) {
    unsigned int f16x2;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(f16x2) : "h"(packed_fp8x2));
    unsigned short lo = (unsigned short)(f16x2);
    unsigned short hi = (unsigned short)(f16x2 >> 16);
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
}

extern "C"
__global__ void fp8_gemv_blockwise_wpr_native_f16in_dual_silu_indirect_batched_topk_kernel(
    __half* __restrict__       out_silu,        // [num_tokens, N] f16
    const unsigned char* __restrict__ base_w_g, // [num_experts, N, K] fp8
    const unsigned char* __restrict__ base_w_u,
    const float* __restrict__  base_s_g,        // [num_experts, N/128, K/128] f32
    const float* __restrict__  base_s_u,
    const __half* __restrict__ input,           // [num_tokens, K] f16
    const int* __restrict__    top_idx,         // [num_tokens, top_k] i32
    long long w_stride,
    long long s_stride,
    int N, int K,
    int num_col_blocks,
    int top_k,
    int k_round,
    int num_tokens
) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int n = blockIdx.x * 8 + warp;
    int m = blockIdx.y;
    if (n >= N || m >= num_tokens) return;

    int e = top_idx[(long long)m * top_k + k_round];

    const unsigned char* w_g = base_w_g + (long long)e * w_stride;
    const unsigned char* w_u = base_w_u + (long long)e * w_stride;
    const float*         s_g = base_s_g + (long long)e * s_stride;
    const float*         s_u = base_s_u + (long long)e * s_stride;

    int scale_row = n >> 7;
    const unsigned char* wg_row = w_g + (long long)n * K;
    const unsigned char* wu_row = w_u + (long long)n * K;
    const __half*        x_row  = input + (long long)m * K;

    float acc_g0 = 0.0f, acc_g1 = 0.0f;
    float acc_u0 = 0.0f, acc_u1 = 0.0f;

    for (int k = lane * 8; k + 7 < K; k += 256) {
        unsigned long long wg8 = __ldg(reinterpret_cast<const unsigned long long*>(wg_row + k));
        unsigned long long wu8 = __ldg(reinterpret_cast<const unsigned long long*>(wu_row + k));

        unsigned long long x_lo = __ldg(reinterpret_cast<const unsigned long long*>(x_row + k));
        unsigned long long x_hi = __ldg(reinterpret_cast<const unsigned long long*>(x_row + k + 4));

        int sc0 = k >> 7;
        float sg0 = __ldg(&s_g[scale_row * num_col_blocks + sc0]);
        float su0 = __ldg(&s_u[scale_row * num_col_blocks + sc0]);
        int sc4 = (k + 4) >> 7;
        float sg4 = (sc4 != sc0) ? __ldg(&s_g[scale_row * num_col_blocks + sc4]) : sg0;
        float su4 = (sc4 != sc0) ? __ldg(&s_u[scale_row * num_col_blocks + sc4]) : su0;

        float gw0, gw1, gw2, gw3, gw4, gw5, gw6, gw7;
        fp8x2_to_f32_dsi_rt((unsigned short)(wg8),       gw0, gw1);
        fp8x2_to_f32_dsi_rt((unsigned short)(wg8 >> 16), gw2, gw3);
        fp8x2_to_f32_dsi_rt((unsigned short)(wg8 >> 32), gw4, gw5);
        fp8x2_to_f32_dsi_rt((unsigned short)(wg8 >> 48), gw6, gw7);

        float uw0, uw1, uw2, uw3, uw4, uw5, uw6, uw7;
        fp8x2_to_f32_dsi_rt((unsigned short)(wu8),       uw0, uw1);
        fp8x2_to_f32_dsi_rt((unsigned short)(wu8 >> 16), uw2, uw3);
        fp8x2_to_f32_dsi_rt((unsigned short)(wu8 >> 32), uw4, uw5);
        fp8x2_to_f32_dsi_rt((unsigned short)(wu8 >> 48), uw6, uw7);

        float x0, x1, x2, x3, x4, x5, x6, x7;
        asm("cvt.f32.f16 %0, %1;" : "=f"(x0) : "h"((unsigned short)(x_lo)));
        asm("cvt.f32.f16 %0, %1;" : "=f"(x1) : "h"((unsigned short)(x_lo >> 16)));
        asm("cvt.f32.f16 %0, %1;" : "=f"(x2) : "h"((unsigned short)(x_lo >> 32)));
        asm("cvt.f32.f16 %0, %1;" : "=f"(x3) : "h"((unsigned short)(x_lo >> 48)));
        asm("cvt.f32.f16 %0, %1;" : "=f"(x4) : "h"((unsigned short)(x_hi)));
        asm("cvt.f32.f16 %0, %1;" : "=f"(x5) : "h"((unsigned short)(x_hi >> 16)));
        asm("cvt.f32.f16 %0, %1;" : "=f"(x6) : "h"((unsigned short)(x_hi >> 32)));
        asm("cvt.f32.f16 %0, %1;" : "=f"(x7) : "h"((unsigned short)(x_hi >> 48)));

        acc_g0 += gw0 * sg0 * x0;
        acc_g0 += gw1 * sg0 * x1;
        acc_g0 += gw2 * sg0 * x2;
        acc_g0 += gw3 * sg0 * x3;
        acc_g1 += gw4 * sg4 * x4;
        acc_g1 += gw5 * sg4 * x5;
        acc_g1 += gw6 * sg4 * x6;
        acc_g1 += gw7 * sg4 * x7;

        acc_u0 += uw0 * su0 * x0;
        acc_u0 += uw1 * su0 * x1;
        acc_u0 += uw2 * su0 * x2;
        acc_u0 += uw3 * su0 * x3;
        acc_u1 += uw4 * su4 * x4;
        acc_u1 += uw5 * su4 * x5;
        acc_u1 += uw6 * su4 * x6;
        acc_u1 += uw7 * su4 * x7;
    }

    float acc_g = acc_g0 + acc_g1;
    float acc_u = acc_u0 + acc_u1;

    {
        int aligned_k = (K / 8) * 8;
        for (int kr = aligned_k + lane; kr < K; kr += 32) {
            int sc = kr >> 7;
            float sg = __ldg(&s_g[scale_row * num_col_blocks + sc]);
            float su = __ldg(&s_u[scale_row * num_col_blocks + sc]);
            float xv = __half2float(__ldg(x_row + kr));
            acc_g += fp8e4m3_to_float_dsi_rt(__ldg(wg_row + kr)) * sg * xv;
            acc_u += fp8e4m3_to_float_dsi_rt(__ldg(wu_row + kr)) * su * xv;
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc_g += __shfl_down_sync(0xffffffff, acc_g, offset);
        acc_u += __shfl_down_sync(0xffffffff, acc_u, offset);
    }

    if (lane == 0) {
        float silu = acc_g / (1.0f + expf(-acc_g));
        out_silu[(long long)m * N + n] = __float2half(silu * acc_u);
    }
}
