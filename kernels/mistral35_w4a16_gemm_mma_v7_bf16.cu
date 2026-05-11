// =============================================================
//  mistral35_w4a16_gemm_mma_v7_bf16.cu
// =============================================================
//
// Codex #1 step 7 — cp.async 2-stage K-pipeline on top of v4
// (BM=128, BN=32, 4 warps, smem A-tile reuse). Adds:
//
//   * Two-stage smem A buffer (smem_a[2][BM][BK]). One stage is
//     loaded asynchronously while the other feeds mma.
//   * 16-byte `cp.async.ca.shared.global` loads — two per thread
//     covering one row's BK=16 BF16 elements (32 bytes total).
//     Lane → row mapping: row = tid / 2, half = tid % 2.
//   * Pipeline shape: preload stage 0 → for each K-iter, prefetch
//     stage (k+1)%2 then wait_group + sync + compute stage k%2.
//     Drain handles the final compute.
//
// The win comes from hiding A's DRAM latency under the 8 mma per
// warp per K-iter (V4-style M_TILES=8). Memory: 4 KiB / iter
// stays the same; compute: ~32 mma per CTA per iter.
//
// Math, on-disk layout, lane / fragment mapping identical to
// v1–v4.

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "nvfp4_utils.cuh"

__device__ __forceinline__ uint32_t pack_two_bf16_v7(
    const __nv_bfloat16 lo, const __nv_bfloat16 hi)
{
    const uint16_t lo16 = *reinterpret_cast<const uint16_t*>(&lo);
    const uint16_t hi16 = *reinterpret_cast<const uint16_t*>(&hi);
    return (static_cast<uint32_t>(hi16) << 16) |
            static_cast<uint32_t>(lo16);
}

// Issue a single 16-byte cp.async.ca.shared.global copy. `smem`
// is the destination smem offset (bytes from smem base via the
// per-CTA static segment). `gmem` is a generic device pointer.
__device__ __forceinline__ void cp_async_16(
    uint32_t smem_addr,
    const void* __restrict__ gmem)
{
#if __CUDA_ARCH__ >= 800
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;\n"
        :
        : "r"(smem_addr), "l"(gmem));
#else
    (void)smem_addr; (void)gmem;
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}

__device__ __forceinline__ uint32_t smem_addr(const void* ptr) {
    uint32_t addr;
    asm("{ .reg .u64 u64addr;\n"
        " cvta.to.shared.u64 u64addr, %1;\n"
        " cvt.u32.u64 %0, u64addr; }\n"
        : "=r"(addr) : "l"(ptr));
    return addr;
}

extern "C" __global__ void __launch_bounds__(128)
mistral35_w4a16_gemm_mma_v7_bf16_kernel(
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
    constexpr int BM         = M_PER_TILE * M_TILES;  // 128
    constexpr int BN         = 32;
    constexpr int BK         = 16;
    constexpr int STAGES     = 3;

    const int n_outer = static_cast<int>(blockIdx.x) * BN;
    const int m_outer = static_cast<int>(blockIdx.y) * BM;
    if (n_outer >= N || m_outer >= M) return;

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

    // Two-stage smem buffer. Layout: [stage][BM][BK] BF16.
    __shared__ __nv_bfloat16 smem_a[STAGES][BM][BK];

    // Per-thread load pattern: 128 threads × 2 issues each, 16
    // bytes per issue = 4096 bytes / stage. Each thread owns one
    // (row, half-of-row) cell at row = tid >> 1, half = tid & 1
    // (half ∈ {0, 1} → K offset {0, 8}).
    //
    // But BM=128 needs 128 rows touched, and we only have 128
    // threads × 1 row each at this layout = 128 rows.
    // Actually: tid / 2 ∈ [0, 64) covers 64 rows, and each thread
    // does 2 halves of one row only with the natural layout. We
    // need to cover 128 rows. So each thread owns TWO rows: one
    // half of row (tid >> 1) and one half of row (tid >> 1) + 64.
    // We issue 4 cp.async per thread (2 rows × 2 halves) per
    // stage to fill 128 rows × 32 bytes = 4096 bytes.
    const int load_row_lo = tid >> 1;       // 0..63
    const int load_row_hi = load_row_lo + 64;  // 64..127
    const int load_half   = tid & 1;        // 0 or 1
    const int load_k_off  = load_half * 8;

    auto issue_stage_load = [&](int stage_idx, int k_outer_arg) {
        // Issue 4 cp.async per thread for this stage:
        //   row=load_row_lo, half=load_half     ←  16B
        //   row=load_row_hi, half=load_half     ←  16B
        // For BK=16 BF16 (32B / row), each thread covers half of
        // 2 rows; whole CTA covers 128 rows × 32B = 4 KiB.
        #pragma unroll
        for (int rsel = 0; rsel < 2; ++rsel) {
            const int local_row = (rsel == 0) ? load_row_lo : load_row_hi;
            const int m_global  = m_outer + local_row;
            __nv_bfloat16* smem_dst = &smem_a[stage_idx][local_row][load_k_off];
            const uint32_t sm_off   = smem_addr(smem_dst);
            const __nv_bfloat16* src =
                act + static_cast<size_t>(m_global) * K
                    + k_outer_arg + load_k_off;
            if (m_global < M && (k_outer_arg + load_k_off + 8) <= K) {
                cp_async_16(sm_off, reinterpret_cast<const void*>(src));
            } else {
                // OOB: zero the 8 BF16 lane via a sync 16B store
                // (cp.async with OOB pointer is undefined; the
                // straightforward fix is a __stwt on the smem
                // pointer for the 16B segment).
                uint4 z = make_uint4(0, 0, 0, 0);
                *reinterpret_cast<uint4*>(smem_dst) = z;
            }
        }
    };

    const int K_iters = K / BK;

    // Preload stages 0 and 1 (if K_iters >= 2). The main loop
    // then maintains STAGES-1 in-flight cp.async groups.
    issue_stage_load(0, 0);
    cp_async_commit();
    if (K_iters >= 2) {
        issue_stage_load(1, BK);
        cp_async_commit();
    }

    float d[M_TILES][4];
    #pragma unroll
    for (int m = 0; m < M_TILES; ++m) {
        d[m][0] = 0.0f; d[m][1] = 0.0f; d[m][2] = 0.0f; d[m][3] = 0.0f;
    }

    const size_t scale_row_stride  = static_cast<size_t>(K >> 4);
    const size_t packed_row_stride = static_cast<size_t>(K >> 1);

    for (int k_iter = 0; k_iter < K_iters; ++k_iter) {
        const int k_outer = k_iter * BK;
        const int stage = k_iter % STAGES;

        // Issue prefetch for stage (k_iter + 2) if it exists.
        // The two earliest stages (0 and 1) were preloaded above.
        if (k_iter + 2 < K_iters) {
            const int prefetch_stage = (k_iter + 2) % STAGES;
            issue_stage_load(prefetch_stage, k_outer + 2 * BK);
            cp_async_commit();
        }

        // Drain bookkeeping. After the issues above the pending
        // group count for the NEXT wait is:
        //   * 2 when both the current-stage and the next-stage
        //     prefetches are still in flight (i.e. there's a
        //     k_iter + 2 prefetch outstanding).
        //   * 1 when only the next-stage prefetch is outstanding
        //     (k_iter + 1 < K_iters but k_iter + 2 >= K_iters).
        //   * 0 on the final iteration.
        if (k_iter + 2 < K_iters) {
            cp_async_wait_group<2>();
        } else if (k_iter + 1 < K_iters) {
            cp_async_wait_group<1>();
        } else {
            cp_async_wait_group<0>();
        }
        __syncthreads();

        // ── B fragment dequant (shared across all M-tiles) ───
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

            b[0] = pack_two_bf16_v7(
                __float2bfloat16_rn(w0), __float2bfloat16_rn(w1));
            b[1] = pack_two_bf16_v7(
                __float2bfloat16_rn(w2), __float2bfloat16_rn(w3));
        } else {
            b[0] = 0u;
            b[1] = 0u;
        }

        // ── Per M-tile: pack A frag from smem[stage], then MMA.
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
                a[0] = pack_two_bf16_v7(v0, v1);
                a[1] = pack_two_bf16_v7(v2, v3);
                a[2] = pack_two_bf16_v7(v4, v5);
                a[3] = pack_two_bf16_v7(v6, v7);
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

    // ── Store D fragments ──────────────────────────────────
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
