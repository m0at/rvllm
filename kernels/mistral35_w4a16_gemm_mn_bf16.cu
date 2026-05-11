// =============================================================
//  mistral35_w4a16_gemm_mn_bf16.cu
// =============================================================
//
// Fused M>1 W4A16 GEMM for Mistral 3.5 NVFP4 weights — codex
// review #1. Removes the prefill path's
//   dequant W → BF16 scratch (~244 GB cumulative per chunk)
//   → cublasLt bf16_gemm_f32 → f32→bf16 cast
// pipeline by reading the packed weight + per-16-elem E4M3 scale
// + scalar global-scale-inverse directly inside the GEMM and
// streaming the dequant through registers. No multi-hundred-MB
// BF16 weight scratch is allocated at all on this path.
//
// Math (matches mistral35_w4a16_gemv_bf16_kernel exactly):
//   out[m, n]  =  Σ_k act[m, k] * w[n, k]
//   w[n, k]    =  fp4_decode(nibble_k) * scale_e4m3[n, k/16] * alpha
//   alpha      =  *alpha_ptr  (= 1 / weight_global_scale_disk
//                              = amax_tensor / 2688)
//
// Layout:
//   out:        [M, N]              BF16  row-major
//   w_packed:   [N, K/2]            u8     low nibble = elem 2i,
//                                          high nibble = elem 2i+1
//   w_scale:    [N, K/16]           E4M3 (u8)
//   alpha_ptr:  scalar              f32
//   act:        [M, K]              BF16  row-major
//
// Launch:
//   Grid:  (N, ceil(M / M_TILE), 1)
//   Block: (256, 1, 1)
//
// Each CTA computes one output column `n` for a tile of up to
// M_TILE consecutive activation rows. Threads stride across the
// K-block axis (one K-block = 16 elements packed in 8 bytes), do
// the dequant inline, and accumulate per-row partial dot products.
// Final block-reduction folds the 256 thread partials per row.
//
// Compared to the dequant+cuBLAS path:
//   * No N*K BF16 weight scratch (per-shape: 302 MB for 12288×12288,
//     706 MB for 28672×12288, 706 MB for 12288×28672, plus output
//     f32 scratch).
//   * One launch instead of three (dequant + GEMM + cast).
//   * Activations are still re-read per CTA — every CTA reads its
//     own [M_TILE, K] slice, no smem activation sharing across N.
//     That's an obvious next-iteration win (tensor-core mma.sync
//     with BM/BN tiling that reuses both A and B in smem).
//
// Hardcoded constants: M_TILE = 8. Smem per CTA = M_TILE * 256 * 4
// = 8 KiB. Register accumulator footprint per thread = M_TILE f32
// (32 bytes) + 16-element dequant scratch (64 bytes).

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "nvfp4_utils.cuh"

#ifndef MISTRAL35_W4A16_GEMM_M_TILE
#define MISTRAL35_W4A16_GEMM_M_TILE 8
#endif

extern "C" __global__ void __launch_bounds__(256)
mistral35_w4a16_gemm_mn_bf16_kernel(
    __nv_bfloat16* __restrict__ out,             // [M, N]
    const uint8_t* __restrict__ w_packed,        // [N, K/2]
    const uint8_t* __restrict__ w_scale,         // [N, K/16] E4M3
    const float* __restrict__ alpha_ptr,         // scalar (= 1/gs_disk)
    const __nv_bfloat16* __restrict__ act,       // [M, K]
    int M,
    int N,
    int K)
{
    constexpr int M_TILE = MISTRAL35_W4A16_GEMM_M_TILE;

    const int n = blockIdx.x;
    if (n >= N) return;
    const int m_base = static_cast<int>(blockIdx.y) * M_TILE;
    if (m_base >= M) return;

    const int tid = threadIdx.x;
    const int K16 = K >> 4;
    const float alpha = *alpha_ptr;

    const uint8_t* w_packed_row = w_packed + static_cast<size_t>(n) * (K / 2);
    const uint8_t* w_scale_row  = w_scale  + static_cast<size_t>(n) * K16;

    float acc[M_TILE];
    #pragma unroll
    for (int mi = 0; mi < M_TILE; ++mi) acc[mi] = 0.0f;

    const int m_count = (M - m_base < M_TILE) ? (M - m_base) : M_TILE;

    for (int kb = tid; kb < K16; kb += 256) {
        const __nv_fp8_e4m3 e4m3 =
            *reinterpret_cast<const __nv_fp8_e4m3*>(w_scale_row + kb);
        const float combined = static_cast<float>(e4m3) * alpha;

        const int k_start = kb * 16;
        const uint8_t* bytes = w_packed_row + (k_start >> 1);

        float w_dec[16];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint8_t byte = bytes[i];
            w_dec[2 * i]     =
                rvllm_nvfp4::fp4_decode(byte & 0xFu) * combined;
            w_dec[2 * i + 1] =
                rvllm_nvfp4::fp4_decode((byte >> 4) & 0xFu) * combined;
        }

        #pragma unroll
        for (int mi = 0; mi < M_TILE; ++mi) {
            if (mi >= m_count) break;
            const int m = m_base + mi;
            const __nv_bfloat16* act_row =
                act + static_cast<size_t>(m) * K + k_start;
            float local = 0.0f;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                local += w_dec[i] * __bfloat162float(act_row[i]);
            }
            acc[mi] += local;
        }
    }

    __shared__ float smem[M_TILE][256];
    #pragma unroll
    for (int mi = 0; mi < M_TILE; ++mi) {
        smem[mi][tid] = acc[mi];
    }
    __syncthreads();

    #pragma unroll
    for (int s = 128; s >= 32; s >>= 1) {
        if (tid < s) {
            #pragma unroll
            for (int mi = 0; mi < M_TILE; ++mi) {
                smem[mi][tid] += smem[mi][tid + s];
            }
        }
        __syncthreads();
    }

    if (tid < 32) {
        #pragma unroll
        for (int mi = 0; mi < M_TILE; ++mi) {
            if (mi >= m_count) continue;
            float v = smem[mi][tid];
            #pragma unroll
            for (int s = 16; s > 0; s >>= 1) {
                v += __shfl_xor_sync(0xFFFFFFFFu, v, s);
            }
            if (tid == 0) {
                const int m = m_base + mi;
                out[static_cast<size_t>(m) * N + n] =
                    __float2bfloat16_rn(v);
            }
        }
    }
}
