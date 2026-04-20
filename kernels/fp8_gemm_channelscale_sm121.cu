// Native FP8 E4M3 GEMM for sm_121 with row × col-block scale epilogue.
//
// Replaces the `fp8_gemm_channelscale_or_fallback` scalar-scale
// fallback on Blackwell consumer (sm_121). The existing fallback
// drops the per-channel weight block-scale (cuBLASLt's channelscale
// heuristic LaunchFailed-s on GB10), which costs PPL on the
// prefill / M>1 path. This kernel does the full row×col-scale math
// natively via `mma.sync.m16n8k32.f32.e4m3.e4m3.f32` (Blackwell FP8
// tensor-core MMA).
//
// Scale layout (matches `rvllm-loader::gemma4_load` + cuBLASLt
// channelscale convention):
//   A is FP8 [M, K], a_scale is f32 [M]           (per-token)
//   B is FP8 [N, K], b_blockscale is f32 [ceil(N/128), ceil(K/128)]
//   (128×128 block-scale; Gemma 4 fp8-block weight format)
//
// Output: f16 [M, N].
//
// Tile shape: M_TILE=16, N_TILE=128, K_TILE=128.
//   * K_TILE=128 matches the block-scale K granularity — exactly one
//     b_blockscale value covers each K_TILE iteration, applied per
//     K_TILE to the per-tile partial accumulator (K axis must scale
//     per block, not once at the end).
//   * N_TILE=128 matches one row of b_blockscale → one scalar scale
//     per block for the whole N_TILE.
//   * M_TILE=16 is the mma M-dimension (one m-fragment). Skinny M
//     (prefill M≤16) fills one M-block exactly.
//
// One thread-block = 4 warps, 128 threads. Grid (N/N_TILE, M/M_TILE).
// Shared memory: 16 * 128 (A tile) + 128 * 128 (B tile) = 18 KiB.
//
// Inline-asm MMA syntax: PTX ISA 8.3+ (CUDA 12.8+). nvcc 13.x
// required. Emitted from the `__CUDA_ARCH__ >= 1000` branch only;
// pre-Blackwell archs get an empty TU so the multi-arch `build.sh`
// doesn't explode.

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000

namespace {

// Tile shape and mma shape constants.
constexpr int M_TILE = 16;
constexpr int N_TILE = 128;
constexpr int K_TILE = 128; // == block-scale K granularity (128×128 blocks)

constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 32;

constexpr int NUM_N_MMA = N_TILE / MMA_N;   // 16
constexpr int NUM_K_MMA = K_TILE / MMA_K;   // 4

constexpr int WARPS_PER_BLOCK = 4;
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;
constexpr int N_MMA_PER_WARP = NUM_N_MMA / WARPS_PER_BLOCK; // 4

// mma.sync.m16n8k32.row.col.f32.e4m3.e4m3.f32
__device__ __forceinline__ void mma_m16n8k32_e4m3(
    float &d0, float &d1, float &d2, float &d3,
    const uint32_t &a0, const uint32_t &a1, const uint32_t &a2, const uint32_t &a3,
    const uint32_t &b0, const uint32_t &b1
) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1)
    );
}

// Read the A fragment for mma.m16n8k32 from shared memory.
// A tile layout in smem: row-major, [M_TILE=16, K_TILE=128] FP8.
// For mma.m16n8k32.row with M=16, K=32 (per MMA), per-thread layout:
//   row_group = lane >> 2           (rows 0..7)
//   col_base  = (lane & 3) * 4      (cols 0,4,8,12 within the 32-col k-tile)
// A-reg[0] = A[row_group][col_base..+3]        (4 bytes = 4 FP8)
// A-reg[1] = A[row_group + 8][col_base..+3]
// A-reg[2] = A[row_group][col_base + 16..+19]
// A-reg[3] = A[row_group + 8][col_base + 16..+19]
//
// `k_off` is the byte offset into the K axis where this mma's 32-col
// slice starts (0, 32, 64, 96 inside the 128-col K_TILE).
__device__ __forceinline__ void load_a_frag_from_smem(
    const uint8_t *smem_a,   // [M_TILE, K_TILE]
    int k_off,
    int lane,
    uint32_t &a0, uint32_t &a1, uint32_t &a2, uint32_t &a3
) {
    int row_group = lane >> 2;
    int col_base  = (lane & 3) * 4;
    int stride = K_TILE;

    const uint8_t *base = smem_a + row_group * stride + k_off + col_base;
    a0 = *reinterpret_cast<const uint32_t *>(base);
    a2 = *reinterpret_cast<const uint32_t *>(base + 16);

    const uint8_t *base8 = smem_a + (row_group + 8) * stride + k_off + col_base;
    a1 = *reinterpret_cast<const uint32_t *>(base8);
    a3 = *reinterpret_cast<const uint32_t *>(base8 + 16);
}

// Read the B fragment for mma.m16n8k32 from shared memory.
// B tile layout in smem: row-major [N_TILE=128, K_TILE=128] FP8 —
// where `row` = N index, `col` = K index. For mma.m16n8k32.col with
// N=8, K=32 per MMA, the B-matrix (from MMA's perspective) is in
// col-major layout relative to K. Since our smem stores B row-major
// with N outer and K inner, each "row" of B-smem *is* one K-col for
// the mma's N-dim element. Per-thread layout for col-major B:
//   row_group = lane >> 2              (which of the 8 n-rows)
//   col_base  = (lane & 3) * 4         (0,4,8,12 within 32-col K)
// B-reg[0] = B[n_off + row_group][k_off + col_base..+3]
// B-reg[1] = B[n_off + row_group][k_off + col_base + 16..+19]
//
// `n_off` is the byte offset of the 8-col n-slice within N_TILE.
__device__ __forceinline__ void load_b_frag_from_smem(
    const uint8_t *smem_b,   // [N_TILE, K_TILE]
    int n_off,
    int k_off,
    int lane,
    uint32_t &b0, uint32_t &b1
) {
    int row_group = lane >> 2;
    int col_base  = (lane & 3) * 4;
    int stride = K_TILE;

    const uint8_t *base = smem_b + (n_off + row_group) * stride + k_off + col_base;
    b0 = *reinterpret_cast<const uint32_t *>(base);
    b1 = *reinterpret_cast<const uint32_t *>(base + 16);
}

// Cooperatively load A and B tiles into shared memory. Threads process
// 16 FP8 bytes each via u128 (int4) vectorised loads where alignment
// allows; falls back to 4-byte loads for the tail.
template <int TILE_ROWS, int TILE_COLS>
__device__ __forceinline__ void cooperative_load_tile(
    uint8_t *smem,
    const uint8_t *global_base,
    int global_stride,
    int row_start,
    int col_start,
    int rows_max,
    int cols_max,
    int tid,
    int nthreads
) {
    const int total_bytes = TILE_ROWS * TILE_COLS;
    for (int idx = tid * 16; idx < total_bytes; idx += nthreads * 16) {
        int row = idx / TILE_COLS;
        int col = idx % TILE_COLS;
        int g_row = row_start + row;
        int g_col = col_start + col;
        uint8_t *dst = smem + idx;

        if (g_row < rows_max) {
            const uint8_t *src = global_base + (size_t)g_row * global_stride + g_col;
            // Bounds-checked 16-byte vector load. The K axis we load along
            // is always a multiple of 16 on the shapes we target (Gemma 4
            // hidden/intermediate are 5376/21504 — both 128-aligned), so
            // the col-end bounds check reduces to a single compare.
            if (g_col + 16 <= cols_max) {
                *reinterpret_cast<int4 *>(dst) = *reinterpret_cast<const int4 *>(src);
            } else {
                #pragma unroll
                for (int b = 0; b < 16; b++) {
                    dst[b] = (g_col + b < cols_max) ? src[b] : 0;
                }
            }
        } else {
            // Row out of bounds — zero pad.
            *reinterpret_cast<int4 *>(dst) = make_int4(0, 0, 0, 0);
        }
    }
}

} // namespace

extern "C" __global__ void fp8_gemm_channelscale_sm121_kernel(
    __half *__restrict__ C,                      // [M, N]
    const uint8_t *__restrict__ A,               // [M, K] FP8 E4M3
    const uint8_t *__restrict__ B,               // [N, K] FP8 E4M3
    const float *__restrict__ a_scale,           // [M] per-token
    const float *__restrict__ b_blockscale,      // [ceil(N/128), ceil(K/128)]
    int M, int N, int K,
    int num_k_blocks                             // = ceil(K / 128)
) {
    const int mb = blockIdx.y;
    const int nb = blockIdx.x;
    const int m_start = mb * M_TILE;
    const int n_start = nb * N_TILE;

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    __shared__ __align__(16) uint8_t smem_a[M_TILE * K_TILE];
    __shared__ __align__(16) uint8_t smem_b[N_TILE * K_TILE];

    // This warp is responsible for n-mma indices
    //   [warp * N_MMA_PER_WARP, (warp + 1) * N_MMA_PER_WARP).
    const int warp_n_mma_begin = warp * N_MMA_PER_WARP;

    // Final scaled accumulator: M_TILE × (N_MMA_PER_WARP × MMA_N) f32
    // values, 4 per thread per n-mma as per m16n8 output layout.
    float final_acc[N_MMA_PER_WARP][4] = {{0.0f}};

    // Walk the K axis in K_TILE=128 chunks (one k-block of weight
    // scale per iteration).
    const int k_block_count = (K + K_TILE - 1) / K_TILE;
    for (int kb = 0; kb < k_block_count; kb++) {
        const int k_start = kb * K_TILE;

        // Load A + B tiles cooperatively.
        cooperative_load_tile<M_TILE, K_TILE>(
            smem_a, A, K, m_start, k_start, M, K, tid, THREADS_PER_BLOCK
        );
        cooperative_load_tile<N_TILE, K_TILE>(
            smem_b, B, K, n_start, k_start, N, K, tid, THREADS_PER_BLOCK
        );
        __syncthreads();

        // Block-scale for this (n-block, k-block). N_TILE == 128 == one
        // full n-block, so there's exactly one scale per iteration.
        const float b_s = b_blockscale[(n_start / 128) * num_k_blocks + kb];

        // Per-block partial accumulator, cleared every k-block so the
        // b_blockscale applies to this k-block only.
        float part_acc[N_MMA_PER_WARP][4] = {{0.0f}};

        // K_TILE=128 → 4 mma-K iterations; each consumes 32 FP8 cols.
        #pragma unroll
        for (int km = 0; km < NUM_K_MMA; km++) {
            const int k_off = km * MMA_K;

            uint32_t a0, a1, a2, a3;
            load_a_frag_from_smem(smem_a, k_off, lane, a0, a1, a2, a3);

            #pragma unroll
            for (int nm = 0; nm < N_MMA_PER_WARP; nm++) {
                const int n_mma_idx = warp_n_mma_begin + nm;
                const int n_off = n_mma_idx * MMA_N;

                uint32_t b0, b1;
                load_b_frag_from_smem(smem_b, n_off, k_off, lane, b0, b1);

                mma_m16n8k32_e4m3(
                    part_acc[nm][0], part_acc[nm][1],
                    part_acc[nm][2], part_acc[nm][3],
                    a0, a1, a2, a3, b0, b1
                );
            }
        }

        // Fold this k-block's contribution into the final accumulator,
        // scaled by b_blockscale[nb, kb]. a_scale is folded in below at
        // the epilogue (it does not depend on k).
        #pragma unroll
        for (int nm = 0; nm < N_MMA_PER_WARP; nm++) {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                final_acc[nm][i] += part_acc[nm][i] * b_s;
            }
        }

        __syncthreads();
    }

    // Epilogue.
    // mma.m16n8 D-matrix layout per thread (PTX ISA §9.7.14):
    //   group      = lane >> 2        (0..7)
    //   col_base   = (lane & 3) * 2   (0, 2, 4, 6 within the 8-col n-tile)
    //   value[0] = D[group    ][col_base    ]
    //   value[1] = D[group    ][col_base + 1]
    //   value[2] = D[group + 8][col_base    ]
    //   value[3] = D[group + 8][col_base + 1]
    const int group    = lane >> 2;
    const int col_base = (lane & 3) * 2;

    #pragma unroll
    for (int nm = 0; nm < N_MMA_PER_WARP; nm++) {
        const int n_mma_idx = warp_n_mma_begin + nm;
        const int n_off = n_mma_idx * MMA_N;

        #pragma unroll
        for (int vi = 0; vi < 4; vi++) {
            const int mi = group + ((vi >> 1) ? 8 : 0);
            const int ci = col_base + (vi & 1);
            const int m_global = m_start + mi;
            const int n_global = n_start + n_off + ci;

            if (m_global < M && n_global < N) {
                const float a_s = a_scale[m_global];
                const float val = final_acc[nm][vi] * a_s;
                C[(size_t)m_global * N + n_global] = __float2half(val);
            }
        }
    }
}

#else

// Pre-Blackwell: emit an empty TU — the kernel symbol simply isn't
// present in PTX built for sm_80/sm_89/sm_90, and the Rust side gates
// resolution on `CompileTarget::Sm121` the same way it does for
// `Fp8GemvVariant::WprNative`.

#endif // __CUDA_ARCH__ >= 1000
