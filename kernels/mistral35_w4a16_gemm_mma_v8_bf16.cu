// =============================================================
//  mistral35_w4a16_gemm_mma_v8_bf16.cu
// =============================================================
//
// Codex #1 step 9 — persistent-CTA variant. The breakthrough idea
// vs v4–v7: one CTA owns a single BN-wide N slice and sweeps over
// ALL M tiles inside the CTA. Previously each (n_tile, m_chunk)
// pair was its own CTA, which made the same N slice of weight
// get re-read from DRAM (M / BM_chunk) times — at M=2048,
// BM_chunk=128 that's 16× redundant weight reads.
//
// Persistent CTA collapses those weight reads to one pass per N
// slice, dropping weight DRAM traffic by (M / BM_chunk)× for the
// prefill chunk path.
//
// Layout:
//   Grid:  (N/32, 1, 1)          — one CTA per N=32 col slice
//   Block: (128, 1, 1)            — 4 warps
//   Smem:  8192 B  (2-stage 128 × 16 BF16 A buffer, BM_chunk=128)
//
// Math, on-disk layout, lane / fragment mapping identical to
// v1–v7. The B fragment is still dequanted on-fly per K-iter per
// warp (it's tiny — 4 nibbles per lane); the A side uses the
// 2-stage cp.async pipeline from v6.
//
// Per CTA:
//   for m_outer in 0..M, step BM_chunk:
//     zero D fragments
//     preload stage 0 + 1 of A
//     for k_outer in 0..K, step BK:
//       prefetch next A stage
//       wait + sync
//       dequant B
//       mma over M_TILES
//     store D
//
// IMPORTANT: ALL cp.async groups must be drained before starting
// the next m_outer's preload, or stale prefetches from the prior
// m_outer would corrupt the new tile.

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "nvfp4_utils.cuh"

__device__ __forceinline__ uint32_t pack_two_bf16_v8(
    const __nv_bfloat16 lo, const __nv_bfloat16 hi)
{
    const uint16_t lo16 = *reinterpret_cast<const uint16_t*>(&lo);
    const uint16_t hi16 = *reinterpret_cast<const uint16_t*>(&hi);
    return (static_cast<uint32_t>(hi16) << 16) |
            static_cast<uint32_t>(lo16);
}

__device__ __forceinline__ void cp_async_16_v8(
    uint32_t smem_addr_arg,
    const void* __restrict__ gmem)
{
#if __CUDA_ARCH__ >= 800
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;\n"
        :
        : "r"(smem_addr_arg), "l"(gmem));
#else
    (void)smem_addr_arg; (void)gmem;
#endif
}

__device__ __forceinline__ void cp_async_commit_v8() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template <int N>
__device__ __forceinline__ void cp_async_wait_group_v8() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}

__device__ __forceinline__ uint32_t smem_addr_v8(const void* ptr) {
    uint32_t addr;
    asm("{ .reg .u64 u64addr;\n"
        " cvta.to.shared.u64 u64addr, %1;\n"
        " cvt.u32.u64 %0, u64addr; }\n"
        : "=r"(addr) : "l"(ptr));
    return addr;
}

extern "C" __global__ void __launch_bounds__(128)
mistral35_w4a16_gemm_mma_v8_bf16_kernel(
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
    constexpr int M_TILES    = 8;
    constexpr int BM_CHUNK   = M_PER_TILE * M_TILES;  // 128
    constexpr int BN         = 32;
    constexpr int BK         = 16;
    constexpr int STAGES     = 2;

    const int n_outer = static_cast<int>(blockIdx.x) * BN;
    if (n_outer >= N) return;

    const int tid     = static_cast<int>(threadIdx.x);
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    const int gid     = lane >> 2;
    const int tg      = lane & 3;

    const int n_warp_base = n_outer + warp_id * 8;
    const int n_col       = n_warp_base + gid;
    const int col_lo      = n_warp_base + tg * 2;
    const int col_hi      = col_lo + 1;

    const float alpha = *alpha_ptr;

    __shared__ __nv_bfloat16 smem_a[STAGES][BM_CHUNK][BK];

    const int load_row_lo = tid >> 1;       // 0..63
    const int load_row_hi = load_row_lo + 64;  // 64..127
    const int load_half   = tid & 1;
    const int load_k_off  = load_half * 8;

    const size_t scale_row_stride  = static_cast<size_t>(K >> 4);
    const size_t packed_row_stride = static_cast<size_t>(K >> 1);
    const int K_iters = K / BK;

    auto issue_stage_load = [&](int stage_idx, int m_outer_arg, int k_outer_arg) {
        #pragma unroll
        for (int rsel = 0; rsel < 2; ++rsel) {
            const int local_row = (rsel == 0) ? load_row_lo : load_row_hi;
            const int m_global  = m_outer_arg + local_row;
            __nv_bfloat16* smem_dst = &smem_a[stage_idx][local_row][load_k_off];
            const uint32_t sm_off   = smem_addr_v8(smem_dst);
            const __nv_bfloat16* src =
                act + static_cast<size_t>(m_global) * K
                    + k_outer_arg + load_k_off;
            if (m_global < M && (k_outer_arg + load_k_off + 8) <= K) {
                cp_async_16_v8(sm_off, reinterpret_cast<const void*>(src));
            } else {
                uint4 z = make_uint4(0, 0, 0, 0);
                *reinterpret_cast<uint4*>(smem_dst) = z;
            }
        }
    };

    // ── Outer M-chunk loop: persistent over the M axis ──────
    for (int m_outer = 0; m_outer < M; m_outer += BM_CHUNK) {
        float d[M_TILES][4];
        #pragma unroll
        for (int m = 0; m < M_TILES; ++m) {
            d[m][0] = 0.0f; d[m][1] = 0.0f; d[m][2] = 0.0f; d[m][3] = 0.0f;
        }

        // Preload stage 0.
        issue_stage_load(0, m_outer, 0);
        cp_async_commit_v8();

        for (int k_iter = 0; k_iter < K_iters; ++k_iter) {
            const int k_outer = k_iter * BK;
            const int stage = k_iter & 1;
            const int next_stage = (k_iter + 1) & 1;

            if (k_iter + 1 < K_iters) {
                issue_stage_load(next_stage, m_outer, k_outer + BK);
                cp_async_commit_v8();
            }

            if (k_iter + 1 < K_iters) {
                cp_async_wait_group_v8<1>();
            } else {
                cp_async_wait_group_v8<0>();
            }
            __syncthreads();

            // ── B fragment dequant ───────────────────────────
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

                b[0] = pack_two_bf16_v8(
                    __float2bfloat16_rn(w0), __float2bfloat16_rn(w1));
                b[1] = pack_two_bf16_v8(
                    __float2bfloat16_rn(w2), __float2bfloat16_rn(w3));
            } else {
                b[0] = 0u;
                b[1] = 0u;
            }

            // ── Per M-tile MMA ──────────────────────────────
            const int c_lo = tg * 2;
            const int c_hi = c_lo + 8;
            #pragma unroll
            for (int m_local = 0; m_local < M_TILES; ++m_local) {
                const int row_lo_smem = gid     + m_local * M_PER_TILE;
                const int row_hi_smem = gid + 8 + m_local * M_PER_TILE;
                uint32_t a[4];
                {
                    const __nv_bfloat16 v0 = smem_a[stage][row_lo_smem][c_lo    ];
                    const __nv_bfloat16 v1 = smem_a[stage][row_lo_smem][c_lo + 1];
                    const __nv_bfloat16 v2 = smem_a[stage][row_hi_smem][c_lo    ];
                    const __nv_bfloat16 v3 = smem_a[stage][row_hi_smem][c_lo + 1];
                    const __nv_bfloat16 v4 = smem_a[stage][row_lo_smem][c_hi    ];
                    const __nv_bfloat16 v5 = smem_a[stage][row_lo_smem][c_hi + 1];
                    const __nv_bfloat16 v6 = smem_a[stage][row_hi_smem][c_hi    ];
                    const __nv_bfloat16 v7 = smem_a[stage][row_hi_smem][c_hi + 1];
                    a[0] = pack_two_bf16_v8(v0, v1);
                    a[1] = pack_two_bf16_v8(v2, v3);
                    a[2] = pack_two_bf16_v8(v4, v5);
                    a[3] = pack_two_bf16_v8(v6, v7);
                }
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0, %1, %2, %3}, "
                    "{%4, %5, %6, %7}, "
                    "{%8, %9}, "
                    "{%0, %1, %2, %3};\n"
                    : "+f"(d[m_local][0]), "+f"(d[m_local][1]),
                      "+f"(d[m_local][2]), "+f"(d[m_local][3])
                    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                      "r"(b[0]), "r"(b[1])
                );
            }
            __syncthreads();
        }

        // ── Drain any leftover async groups before next m_outer
        cp_async_wait_group_v8<0>();
        __syncthreads();

        // ── Store D fragments for this m_chunk ──────────────
        #pragma unroll
        for (int m_local = 0; m_local < M_TILES; ++m_local) {
            const int row_lo = m_outer + gid     + m_local * M_PER_TILE;
            const int row_hi = m_outer + gid + 8 + m_local * M_PER_TILE;
            if (row_lo < M && col_lo < N) {
                out[static_cast<size_t>(row_lo) * N + col_lo] =
                    __float2bfloat16_rn(d[m_local][0]);
            }
            if (row_lo < M && col_hi < N) {
                out[static_cast<size_t>(row_lo) * N + col_hi] =
                    __float2bfloat16_rn(d[m_local][1]);
            }
            if (row_hi < M && col_lo < N) {
                out[static_cast<size_t>(row_hi) * N + col_lo] =
                    __float2bfloat16_rn(d[m_local][2]);
            }
            if (row_hi < M && col_hi < N) {
                out[static_cast<size_t>(row_hi) * N + col_hi] =
                    __float2bfloat16_rn(d[m_local][3]);
            }
        }
    }
}
