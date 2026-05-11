// =============================================================
//  mistral35_w4a16_gate_up_gemv_bf16.cu
// =============================================================
//
// Fused gate + up W4A16 GEMV for the Mistral 3.5 decode-step MLP
// (codex review #2). Replaces two independent launches of
// `mistral35_w4a16_gemv_bf16_kernel` with a single kernel that
// produces gate[i_size] and up[i_size] from the same BF16
// activation in one grid pass.
//
// Grid:  (2 * i_size, 1, 1)
// Block: (256, 1, 1)
// Per-row math identical to the GEMV kernel.

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "nvfp4_utils.cuh"

extern "C" __global__ void __launch_bounds__(256)
mistral35_w4a16_gate_up_gemv_bf16_kernel(
    __nv_bfloat16* __restrict__ out_gate,        // [i_size]
    __nv_bfloat16* __restrict__ out_up,          // [i_size]
    const uint8_t* __restrict__ w_gate_packed,
    const uint8_t* __restrict__ w_gate_scale,
    const float*   __restrict__ alpha_gate_ptr,
    const uint8_t* __restrict__ w_up_packed,
    const uint8_t* __restrict__ w_up_scale,
    const float*   __restrict__ alpha_up_ptr,
    const __nv_bfloat16* __restrict__ act,       // [K]
    int i_size,
    int K)
{
    const int row_global = blockIdx.x;
    const int K16 = K >> 4;

    __nv_bfloat16* out;
    const uint8_t* w_packed;
    const uint8_t* w_scale;
    float alpha;
    int n_row;

    if (row_global < i_size) {
        out      = out_gate;
        w_packed = w_gate_packed;
        w_scale  = w_gate_scale;
        alpha    = *alpha_gate_ptr;
        n_row    = row_global;
    } else if (row_global < 2 * i_size) {
        out      = out_up;
        w_packed = w_up_packed;
        w_scale  = w_up_scale;
        alpha    = *alpha_up_ptr;
        n_row    = row_global - i_size;
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
