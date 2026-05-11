// =============================================================
//  mistral35_w4a16_gemm_mma_bf16.cu
// =============================================================
//
// Tensor-core W4A16 M>1 GEMM for Mistral 3.5 NVFP4 weights —
// codex review #1 follow-up. Same on-disk layout as the
// CUDA-core kernel (`mistral35_w4a16_gemm_mn_bf16.cu`) but the
// dot product runs on `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`
// tensor cores. Dequant from packed NVFP4 nibbles + E4M3 scale +
// alpha is streamed through registers directly into the B fragment.
//
// First iteration intentionally minimal: one warp per CTA, BM=16,
// BN=8, BK=16. Grid: (N/8, ceil(M/16)). Per-K iteration each warp:
//   1. Loads its 8-bf16 A fragment from `act` (row-major BF16).
//   2. Dequants its 4 packed nibbles → 4 BF16 values forming the
//      4-bf16 B fragment (col-major view of [N][K]).
//   3. Issues one `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`.
// After K loop, writes the 4-f32 D fragment back to `out[M][N]`
// row-major as BF16.
//
// What this DOES win over the CUDA-core path:
//   * Tensor-core throughput vs scalar muladds.
//   * Same memory-bandwidth fix as the CUDA-core kernel — no BF16
//     weight scratch is materialised.
//
// What this does NOT win yet (future iterations):
//   * No smem A-reuse across multiple BN tiles — each CTA re-reads
//     its A rows from DRAM. The dequant+cuBLAS baseline reads A
//     once via the cuBLAS tile schedule. A multi-warp BM×BN tile
//     with shared activation smem closes that gap.
//   * No async copy / cp.async / multi-stage pipeline.
//   * Only the single-warp shape is exercised; the BN=8 grid
//     creates many small CTAs (96 CTAs per M-row at N=12288).
//
// Math (matches gemv_bf16 + gemm_mn_bf16 exactly):
//   out[m, n]  =  Σ_k act[m, k] * w[n, k]
//   w[n, k]    =  fp4_decode(nibble_k) * scale_e4m3[n, k/16] * alpha

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "nvfp4_utils.cuh"

// Pack two bf16 (lo, hi) into a single u32. `lo` becomes the low
// 16 bits, `hi` the high 16 bits — matches the per-lane fragment
// layout expected by mma.sync m16n8k16 for both A (a[i] = {col c0,
// col c0+1}) and B (b[i] = {row r0, row r0+1}).
__device__ __forceinline__ uint32_t pack_two_bf16(
    const __nv_bfloat16 lo, const __nv_bfloat16 hi)
{
    uint32_t r;
    const uint16_t lo16 = *reinterpret_cast<const uint16_t*>(&lo);
    const uint16_t hi16 = *reinterpret_cast<const uint16_t*>(&hi);
    r = (static_cast<uint32_t>(hi16) << 16) | static_cast<uint32_t>(lo16);
    return r;
}

extern "C" __global__ void __launch_bounds__(32)
mistral35_w4a16_gemm_mma_bf16_kernel(
    __nv_bfloat16* __restrict__ out,             // [M, N]
    const uint8_t* __restrict__ w_packed,        // [N, K/2]
    const uint8_t* __restrict__ w_scale,         // [N, K/16] E4M3
    const float* __restrict__ alpha_ptr,         // scalar (= 1/gs_disk)
    const __nv_bfloat16* __restrict__ act,       // [M, K]
    int M,
    int N,
    int K)
{
    // Single warp per CTA, BM=16, BN=8, BK=16.
    const int n_outer = static_cast<int>(blockIdx.x) * 8;
    const int m_outer = static_cast<int>(blockIdx.y) * 16;
    if (n_outer >= N || m_outer >= M) return;

    const int lane = static_cast<int>(threadIdx.x);
    const int gid  = lane >> 2;   // 0..7
    const int tg   = lane & 3;    // 0..3

    const int row_lo = m_outer + gid;
    const int row_hi = row_lo + 8;
    const int n_col  = n_outer + gid;
    const int col_lo = n_outer + tg * 2;
    const int col_hi = col_lo + 1;

    const float alpha = *alpha_ptr;
    const __nv_bfloat16 zero_bf = __float2bfloat16_rn(0.0f);

    float d[4];
    d[0] = 0.0f; d[1] = 0.0f; d[2] = 0.0f; d[3] = 0.0f;

    const size_t scale_row_stride = static_cast<size_t>(K >> 4);
    const size_t packed_row_stride = static_cast<size_t>(K >> 1);

    for (int k_outer = 0; k_outer < K; k_outer += 16) {
        const int c_lo = tg * 2;
        const int c_hi = c_lo + 8;
        const int k0 = k_outer + c_lo;
        const int k8 = k_outer + c_hi;

        // ── A fragment (row-major, [16 × 16]) ────────────────
        // 8 bf16 per lane → 4 u32. PTX layout (row.col):
        //   a[0] = {A[r_lo, c_lo+0], A[r_lo, c_lo+1]}
        //   a[1] = {A[r_hi, c_lo+0], A[r_hi, c_lo+1]}
        //   a[2] = {A[r_lo, c_hi+0], A[r_lo, c_hi+1]}
        //   a[3] = {A[r_hi, c_hi+0], A[r_hi, c_hi+1]}
        uint32_t a[4];
        {
            __nv_bfloat16 v0, v1, v2, v3, v4, v5, v6, v7;
            v0 = (row_lo < M) ? act[static_cast<size_t>(row_lo) * K + k0    ] : zero_bf;
            v1 = (row_lo < M) ? act[static_cast<size_t>(row_lo) * K + k0 + 1] : zero_bf;
            v2 = (row_hi < M) ? act[static_cast<size_t>(row_hi) * K + k0    ] : zero_bf;
            v3 = (row_hi < M) ? act[static_cast<size_t>(row_hi) * K + k0 + 1] : zero_bf;
            v4 = (row_lo < M) ? act[static_cast<size_t>(row_lo) * K + k8    ] : zero_bf;
            v5 = (row_lo < M) ? act[static_cast<size_t>(row_lo) * K + k8 + 1] : zero_bf;
            v6 = (row_hi < M) ? act[static_cast<size_t>(row_hi) * K + k8    ] : zero_bf;
            v7 = (row_hi < M) ? act[static_cast<size_t>(row_hi) * K + k8 + 1] : zero_bf;
            a[0] = pack_two_bf16(v0, v1);
            a[1] = pack_two_bf16(v2, v3);
            a[2] = pack_two_bf16(v4, v5);
            a[3] = pack_two_bf16(v6, v7);
        }

        // ── B fragment (col-major, [16 × 8]) ────────────────
        // 4 bf16 per lane → 2 u32. PTX layout (row.col):
        //   b[0] = {B[c_lo+0, n], B[c_lo+1, n]}
        //   b[1] = {B[c_hi+0, n], B[c_hi+1, n]}
        // For us B[k, n] = w[n, k], so the lane reads four nibbles
        // from row `n` at columns {c_lo, c_lo+1, c_hi, c_hi+1}.
        uint32_t b[2];
        if (n_col < N) {
            const __nv_fp8_e4m3 e4m3 =
                *reinterpret_cast<const __nv_fp8_e4m3*>(
                    w_scale + static_cast<size_t>(n_col) * scale_row_stride
                            + static_cast<size_t>(k_outer >> 4));
            const float scale = static_cast<float>(e4m3) * alpha;

            const size_t row_pack_off =
                static_cast<size_t>(n_col) * packed_row_stride
                + static_cast<size_t>(k_outer >> 1);
            const uint8_t byte_lo = w_packed[row_pack_off + tg];
            const uint8_t byte_hi = w_packed[row_pack_off + tg + 4];

            const float w0 = rvllm_nvfp4::fp4_decode(byte_lo & 0xFu) * scale;
            const float w1 = rvllm_nvfp4::fp4_decode((byte_lo >> 4) & 0xFu) * scale;
            const float w2 = rvllm_nvfp4::fp4_decode(byte_hi & 0xFu) * scale;
            const float w3 = rvllm_nvfp4::fp4_decode((byte_hi >> 4) & 0xFu) * scale;

            const __nv_bfloat16 bf0 = __float2bfloat16_rn(w0);
            const __nv_bfloat16 bf1 = __float2bfloat16_rn(w1);
            const __nv_bfloat16 bf2 = __float2bfloat16_rn(w2);
            const __nv_bfloat16 bf3 = __float2bfloat16_rn(w3);

            b[0] = pack_two_bf16(bf0, bf1);
            b[1] = pack_two_bf16(bf2, bf3);
        } else {
            b[0] = 0u;
            b[1] = 0u;
        }

        // ── MMA ─────────────────────────────────────────────
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%0, %1, %2, %3};\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
              "r"(b[0]), "r"(b[1])
        );
    }

    // ── Store D fragment ────────────────────────────────────
    // d[0] = D[row_lo, col_lo + 0]
    // d[1] = D[row_lo, col_lo + 1]
    // d[2] = D[row_lo+8, col_lo + 0]
    // d[3] = D[row_lo+8, col_lo + 1]
    if (row_lo < M && col_lo < N) {
        out[static_cast<size_t>(row_lo) * N + col_lo] =
            __float2bfloat16_rn(d[0]);
    }
    if (row_lo < M && col_hi < N) {
        out[static_cast<size_t>(row_lo) * N + col_hi] =
            __float2bfloat16_rn(d[1]);
    }
    if (row_hi < M && col_lo < N) {
        out[static_cast<size_t>(row_hi) * N + col_lo] =
            __float2bfloat16_rn(d[2]);
    }
    if (row_hi < M && col_hi < N) {
        out[static_cast<size_t>(row_hi) * N + col_hi] =
            __float2bfloat16_rn(d[3]);
    }
}
