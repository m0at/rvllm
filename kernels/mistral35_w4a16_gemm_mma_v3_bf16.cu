// =============================================================
//  mistral35_w4a16_gemm_mma_v3_bf16.cu
// =============================================================
//
// Codex #1 step 4 — BM=64 variant. Same 4-warp × BN=32 layout as
// v2 but each warp now processes 4 stacked M-tiles of 16 rows each
// (16 × 4 = 64 rows per CTA). A read from DRAM amortises over 4×
// more output rows than v2; B (packed weight) read amortises
// over the same 4 output rows × 32 N cols.
//
// Per K-iteration each warp:
//   (1) Cooperative A-tile load [BM=64, BK=16] BF16 → smem (1 KB
//       per CTA, 128 threads, each loads a u32 = 2 bf16 from one
//       of the 64 rows × 8 u32-cols).
//   (2) Pack 4 per-warp A fragments (one per m-tile) from smem.
//   (3) Dequant 1 per-warp B fragment (4 nibbles → 4 BF16).
//   (4) 4 mma.sync per warp, accumulating into 4 D fragments.
//
// Grid: (N/32, ceil(M/64), 1)  Block: (128, 1, 1).
//
// Math, on-disk layout, lane / fragment mapping all identical to
// v1/v2 — only the M tile size + register-resident D fragment
// count change.

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "nvfp4_utils.cuh"

__device__ __forceinline__ uint32_t pack_two_bf16(
    const __nv_bfloat16 lo, const __nv_bfloat16 hi)
{
    const uint16_t lo16 = *reinterpret_cast<const uint16_t*>(&lo);
    const uint16_t hi16 = *reinterpret_cast<const uint16_t*>(&hi);
    return (static_cast<uint32_t>(hi16) << 16) |
            static_cast<uint32_t>(lo16);
}

extern "C" __global__ void __launch_bounds__(128)
mistral35_w4a16_gemm_mma_v3_bf16_kernel(
    __nv_bfloat16* __restrict__ out,             // [M, N]
    const uint8_t* __restrict__ w_packed,        // [N, K/2]
    const uint8_t* __restrict__ w_scale,         // [N, K/16] E4M3
    const float* __restrict__ alpha_ptr,         // scalar (= 1/gs_disk)
    const __nv_bfloat16* __restrict__ act,       // [M, K]
    int M,
    int N,
    int K)
{
    constexpr int M_PER_TILE = 16;
    constexpr int M_TILES    = 4;       // 4 × 16 = BM
    constexpr int BM         = M_PER_TILE * M_TILES;  // 64
    constexpr int BN         = 32;      // 4 warps × 8 cols
    constexpr int BK         = 16;

    const int n_outer = static_cast<int>(blockIdx.x) * BN;
    const int m_outer = static_cast<int>(blockIdx.y) * BM;
    if (n_outer >= N || m_outer >= M) return;

    const int tid     = static_cast<int>(threadIdx.x);
    const int warp_id = tid >> 5;          // 0..3
    const int lane    = tid & 31;          // 0..31
    const int gid     = lane >> 2;         // 0..7
    const int tg      = lane & 3;          // 0..3

    const int n_warp_base = n_outer + warp_id * 8;
    const int n_col       = n_warp_base + gid;
    const int col_lo      = n_warp_base + tg * 2;
    const int col_hi      = col_lo + 1;

    const float alpha = *alpha_ptr;
    const __nv_bfloat16 zero_bf = __float2bfloat16_rn(0.0f);

    // smem A tile: [BM=64, BK=16] BF16 = 2048 bytes per CTA.
    __shared__ __nv_bfloat16 smem_a[BM][BK];

    // 4 D fragments per warp (one per M-tile), 4 f32 each = 16 f32
    // per lane.
    float d0[4]; d0[0]=0.0f; d0[1]=0.0f; d0[2]=0.0f; d0[3]=0.0f;
    float d1[4]; d1[0]=0.0f; d1[1]=0.0f; d1[2]=0.0f; d1[3]=0.0f;
    float d2[4]; d2[0]=0.0f; d2[1]=0.0f; d2[2]=0.0f; d2[3]=0.0f;
    float d3[4]; d3[0]=0.0f; d3[1]=0.0f; d3[2]=0.0f; d3[3]=0.0f;

    const size_t scale_row_stride  = static_cast<size_t>(K >> 4);
    const size_t packed_row_stride = static_cast<size_t>(K >> 1);

    for (int k_outer = 0; k_outer < K; k_outer += BK) {
        // ── (1) Cooperative A-tile load [64, 16] ─────────────
        // 128 threads × 4 u32 each = 512 u32 = 1024 bf16. Layout
        // as [BM=64, BK/2=8] u32 grid → 64 × 8 = 512 cells.
        // Each thread loads `M_TILES = 4` rows: tid → (row_base,
        // col_u32) where row_base = (tid >> 3) ∈ [0, 16) and
        // col_u32 = tid & 7 ∈ [0, 8). For each m_local in
        // {0,1,2,3} the absolute row is row_base + m_local * 16.
        {
            const int row_base = tid >> 3;          // 0..15
            const int col_u32  = tid & 7;            // 0..7
            const int k_pos    = col_u32 * 2;
            #pragma unroll
            for (int m_local = 0; m_local < M_TILES; ++m_local) {
                const int row = row_base + m_local * M_PER_TILE;
                const int m_global = m_outer + row;
                uint32_t v32;
                if (m_global < M && (k_outer + k_pos + 1) < K) {
                    v32 = *reinterpret_cast<const uint32_t*>(
                        act + static_cast<size_t>(m_global) * K
                            + k_outer + k_pos);
                } else if (m_global < M && (k_outer + k_pos) < K) {
                    const __nv_bfloat16 lo16 = act[
                        static_cast<size_t>(m_global) * K + k_outer + k_pos];
                    v32 = pack_two_bf16(lo16, zero_bf);
                } else {
                    v32 = 0u;
                }
                *reinterpret_cast<uint32_t*>(&smem_a[row][k_pos]) = v32;
            }
        }
        __syncthreads();

        // ── (3) Dequant per-warp B fragment ──────────────────
        // Same B fragment is shared across all M tiles in this
        // warp — compute once per K iter.
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

            b[0] = pack_two_bf16(
                __float2bfloat16_rn(w0), __float2bfloat16_rn(w1));
            b[1] = pack_two_bf16(
                __float2bfloat16_rn(w2), __float2bfloat16_rn(w3));
        } else {
            b[0] = 0u;
            b[1] = 0u;
        }

        // ── (2)+(4) Per M-tile: pack A frag, then MMA. ──────
        const int c_lo = tg * 2;
        const int c_hi = c_lo + 8;
        #pragma unroll
        for (int m_local = 0; m_local < M_TILES; ++m_local) {
            const int row_lo_smem = gid     + m_local * M_PER_TILE;
            const int row_hi_smem = gid + 8 + m_local * M_PER_TILE;
            uint32_t a[4];
            {
                const __nv_bfloat16 v0 = smem_a[row_lo_smem][c_lo    ];
                const __nv_bfloat16 v1 = smem_a[row_lo_smem][c_lo + 1];
                const __nv_bfloat16 v2 = smem_a[row_hi_smem][c_lo    ];
                const __nv_bfloat16 v3 = smem_a[row_hi_smem][c_lo + 1];
                const __nv_bfloat16 v4 = smem_a[row_lo_smem][c_hi    ];
                const __nv_bfloat16 v5 = smem_a[row_lo_smem][c_hi + 1];
                const __nv_bfloat16 v6 = smem_a[row_hi_smem][c_hi    ];
                const __nv_bfloat16 v7 = smem_a[row_hi_smem][c_hi + 1];
                a[0] = pack_two_bf16(v0, v1);
                a[1] = pack_two_bf16(v2, v3);
                a[2] = pack_two_bf16(v4, v5);
                a[3] = pack_two_bf16(v6, v7);
            }
            float* d = (m_local == 0) ? d0
                     : (m_local == 1) ? d1
                     : (m_local == 2) ? d2
                                       : d3;
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

        __syncthreads();
    }

    // ── Store D fragments ──────────────────────────────────
    #pragma unroll
    for (int m_local = 0; m_local < M_TILES; ++m_local) {
        const int row_lo = m_outer + gid     + m_local * M_PER_TILE;
        const int row_hi = m_outer + gid + 8 + m_local * M_PER_TILE;
        const float* d = (m_local == 0) ? d0
                       : (m_local == 1) ? d1
                       : (m_local == 2) ? d2
                                         : d3;
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
}
