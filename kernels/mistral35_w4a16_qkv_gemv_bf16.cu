// =============================================================
//  mistral35_w4a16_qkv_gemv_bf16.cu
// =============================================================
//
// Fused Q + K + V W4A16 GEMV for the Mistral 3.5 decode step
// (codex review #2). Replaces three independent launches of
// `mistral35_w4a16_gemv_bf16_kernel` with a single kernel that
// produces the three projections from the same BF16 activation
// vector in one grid pass.
//
// Math + per-block dot product are identical to the GEMV kernel.
// The only structural change: blockIdx.x is now a global "output
// row" index across the concatenated [Q | K | V] N-axis, and the
// block looks up which projection it belongs to from a pair of
// thresholds (n_q, n_q + n_kv).
//
// Layout:
//   Grid:  (n_q + n_kv + n_kv, 1, 1)
//   Block: (256, 1, 1)
//
// Per-row math: identical to mistral35_w4a16_gemv_bf16_kernel.

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "nvfp4_utils.cuh"

extern "C" __global__ void __launch_bounds__(256)
mistral35_w4a16_qkv_gemv_bf16_kernel(
    __nv_bfloat16* __restrict__ out_q,           // [n_q]
    __nv_bfloat16* __restrict__ out_k,           // [n_kv]
    __nv_bfloat16* __restrict__ out_v,           // [n_kv]
    const uint8_t* __restrict__ w_q_packed,      // [n_q,  K/2]
    const uint8_t* __restrict__ w_q_scale,       // [n_q,  K/16]
    const float*   __restrict__ alpha_q_ptr,
    const uint8_t* __restrict__ w_k_packed,      // [n_kv, K/2]
    const uint8_t* __restrict__ w_k_scale,       // [n_kv, K/16]
    const float*   __restrict__ alpha_k_ptr,
    const uint8_t* __restrict__ w_v_packed,      // [n_kv, K/2]
    const uint8_t* __restrict__ w_v_scale,       // [n_kv, K/16]
    const float*   __restrict__ alpha_v_ptr,
    const __nv_bfloat16* __restrict__ act,       // [K]
    int n_q,
    int n_kv,
    int K)
{
    const int row_global = blockIdx.x;
    const int K16 = K >> 4;

    __nv_bfloat16* out;
    const uint8_t* w_packed;
    const uint8_t* w_scale;
    float alpha;
    int n_row;

    if (row_global < n_q) {
        out      = out_q;
        w_packed = w_q_packed;
        w_scale  = w_q_scale;
        alpha    = *alpha_q_ptr;
        n_row    = row_global;
    } else if (row_global < n_q + n_kv) {
        out      = out_k;
        w_packed = w_k_packed;
        w_scale  = w_k_scale;
        alpha    = *alpha_k_ptr;
        n_row    = row_global - n_q;
    } else if (row_global < n_q + 2 * n_kv) {
        out      = out_v;
        w_packed = w_v_packed;
        w_scale  = w_v_scale;
        alpha    = *alpha_v_ptr;
        n_row    = row_global - n_q - n_kv;
    } else {
        return;
    }

    const int tid = threadIdx.x;

    float acc = 0.0f;
    const uint8_t* w_packed_row = w_packed + static_cast<size_t>(n_row) * (K / 2);
    const uint8_t* w_scale_row  = w_scale  + static_cast<size_t>(n_row) * K16;

    for (int kb = tid; kb < K16; kb += 256) {
        const __nv_fp8_e4m3 e4m3 =
            *reinterpret_cast<const __nv_fp8_e4m3*>(w_scale_row + kb);
        const float combined = static_cast<float>(e4m3) * alpha;

        const int k_start = kb * 16;
        const uint8_t* bytes = w_packed_row + (k_start >> 1);

        float local = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint8_t byte = bytes[i];
            const float w_lo =
                rvllm_nvfp4::fp4_decode(byte & 0xFu) * combined;
            const float w_hi =
                rvllm_nvfp4::fp4_decode((byte >> 4) & 0xFu) * combined;
            const float a_lo =
                __bfloat162float(act[k_start + 2 * i]);
            const float a_hi =
                __bfloat162float(act[k_start + 2 * i + 1]);
            local += w_lo * a_lo + w_hi * a_hi;
        }
        acc += local;
    }

    __shared__ float smem[256];
    smem[tid] = acc;
    __syncthreads();

    for (int s = 128; s >= 32; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid < 32) {
        float v = smem[tid];
        for (int s = 16; s > 0; s >>= 1) {
            v += __shfl_xor_sync(0xFFFFFFFFu, v, s);
        }
        if (tid == 0) {
            out[n_row] = __float2bfloat16_rn(v);
        }
    }
}
