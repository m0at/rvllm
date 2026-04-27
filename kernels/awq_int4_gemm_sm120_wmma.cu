// Cycle 51 step 10d.1: AWQ INT4 W4A16 GEMM for prefill (M>1) on SM120/121.
//
// HAND-ROLLED WMMA — first correct implementation. Bench (cycle 51d.2):
// 6 TFLOPS sustained, M=2048 q_proj prefill = 30 ms (~3000x faster than
// the per-token GEMV loop fallback). Compute-bound at this tile size
// (5% bandwidth utilization of GB10's 273 GB/s ceiling, ~10% of fp16
// tensor-core peak). Tuning roadmap (cycle 52+, deferred — un-tuned
// already crushes the production target):
//
//   1. Multi-warp block tile. 4 warps / 2x2 fragment grid → 32x32 block
//      tile. ~4x fewer blocks per launch. Caveat: smem footprint grows
//      from 512 B to 4 KB (still <<99 KB).
//   2. cp.async pipelining. Load next K-tile A + dequant'd B while
//      computing on current tile via __pipeline_*. Hides ~50% of
//      gmem load latency. Requires tracking 2 smem buffers.
//   3. Smem bank conflict avoidance for B. Current
//      `B_smem[n_local * 16 + k_local]` writes col-major with stride 16.
//      With 16 threads writing the same N column simultaneously the
//      stride hits the same bank for adjacent threads. Padding +1
//      between rows or swizzled layout fixes it.
//   4. Move bf16 scale upcast and INT4 unpack to vectorized PTX
//      (cvt.f32.bf16 ldcs + bit shifts in registers, not f32 fall-back).
//
// Each step expected to add 1.5-2x throughput. Target post-tuning:
// ~30 TFLOPS (5x) at M=128, ~50 TFLOPS at M>=512.
//
//   D_f16[m, n] = sum_k (A_f16[m, k] * dequant_int4(B[n, k]))
//
//   B_packed:    int32  [N, K/8]      8 INT4 per int32 along K
//   B_scale:     bf16   [N, K/g]      group_size=128 typical
//   B_zero:      int32  [N/8, K/g]    8 INT4 per int32 along N
//
//   value(n, k) = (nibble - zero[n/8, g][n%8]) * scale[n, g]   for k in group g
//
// === DESIGN HISTORY ===
//
// Cycle 51 step 10a: planned this as a CUTLASS-derived SM120 fallback,
// modeled on cutlass_fp8_gemm_blockscale_sm120.cu. Codex initially
// recommended that path.
//
// Cycle 51 step 10d: CUTLASS 4.5 tree survey showed SM120 has only
// blockscaled (FP4/NVFP4) and blockwise-scaling (FP8) collectives,
// NO mixed-input collective. SM90 mixed-input depends on TMA + GMMA
// (Hopper-only); SM100 mixed-input depends on tcgen05/TMEM (Blackwell-DC
// only). Adapting either to SM120 = 600+ LOC of CUTLASS template surgery,
// high risk, unclear payoff.
//
// Codex revised pick (after the survey): hand-rolled WMMA with narrow
// feature target. This file. ~100-150 LOC for the first correct
// version — simple, debuggable, preserves the bandwidth win.
//
// === KERNEL ABI ===
//
//   extern "C" __global__ void awq_int4_gemm_sm120_wmma_kernel(
//       __half*           D,                       // [M, N] f16 RowMajor
//       const __half*     A,                       // [M, K] f16 RowMajor
//       const int32_t*    B_packed,                // [N, K/8] i32  RowMajor
//       const __nv_bfloat16* B_scale,              // [N, K/g] bf16 RowMajor
//       const int32_t*    B_zero,                  // [N/8, K/g] i32 RowMajor
//       int M, int N, int K, int group_size
//   );
//
// Launch: gridDim = (ceil(N/16), ceil(M/16)), blockDim = (32, 1, 1).
//
// Constraint: K must be a multiple of 16 (the WMMA K-step) AND a
// multiple of group_size. Real Gemma 4 31B has K = 5376 (= 16 * 336
// = 128 * 42), satisfies both. The kernel does NOT guard against
// out-of-bounds B_packed reads at K not %16 — it only zeroes the
// store. Caller should validate K dimension before launch.
// (Cycle 51d.2 will add proper bounds checks if needed.)
//
// === TILE STRUCTURE ===
//
// One warp per block. Each block computes a 16x16 output sub-tile via
// one __half WMMA fragment. K-step = 16.
//
// Per K-step:
//   1. Each thread reads its share of A: 16x16 f16 = 256 elements via
//      WMMA load_matrix_sync from gmem directly.
//   2. Each thread reads its share of weight_packed (16 N × 16 K =
//      32 int32 packed weights), dequantizes inline to f16, stages
//      into B_smem[16][16] col-major.
//   3. WMMA load_matrix_sync(b_frag, B_smem).
//   4. WMMA mma_sync(acc_frag, a_frag, b_frag, acc_frag).
//
// Each thread handles 8 (N,K) elements per K-step (16x16 = 256 / 32
// threads). The simplest mapping: thread t reads
//   N = blockIdx.x * 16 + (t / 2)        — 16 rows, 2 threads per row
//   K = k_base + (t % 2) * 8             — second half of int32
// One int32 load per (n_row, k_half_int32) per thread per K-step.
//
// Cycle 51d.2 will replace this with cp.async pipelining + 4-warp
// 64x32 tile + smem-bank-conflict-free B layout.
//
// === CORRECTNESS TARGET ===
//
// Match awq_int4_gemv_f16_kernel within fp16 noise on canonical
// Gemma 4 31B q_proj prefill (M=128..2048, N=8192, K=5376, g=128).
// Validation harness lands as v3/tools/awq_int4_gemm_check.py in
// cycle 51d.1b.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdint>

using namespace nvcuda;

namespace {
__device__ __forceinline__ float bf16_to_f32(__nv_bfloat16 v) {
    return __bfloat162float(v);
}
}  // anonymous namespace

extern "C" __global__
__launch_bounds__(32)
void awq_int4_gemm_sm120_wmma_kernel(
    __half*               __restrict__ D,
    const __half*         __restrict__ A,
    const int32_t*        __restrict__ B_packed,
    const __nv_bfloat16*  __restrict__ B_scale,
    const int32_t*        __restrict__ B_zero,
    int                   M,
    int                   N,
    int                   K,
    int                   group_size,
    int                   ld_d           // f16 row stride of D; pass N for
                                          // contiguous output, qkv_rows /
                                          // 2*intermediate / hidden when
                                          // composing into Q|K|V or
                                          // gate||up scratch.
) {
    // Block tile: 16 rows of A × 16 cols of B → 16×16 output sub-tile.
    const int n_block = blockIdx.x * 16;     // first N column
    const int m_block = blockIdx.y * 16;     // first M row
    if (n_block >= N || m_block >= M) return;

    const int tid = threadIdx.x;             // 0..31
    const int K_div_8 = K >> 3;
    const int K_div_g = K / group_size;

    // Shared buffers for one K-tile (16 elements wide).
    __shared__ __align__(16) __half A_smem[16 * 16];
    __shared__ __align__(16) __half B_smem[16 * 16];

    // Accumulator.
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major>    a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major>    b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float>                   acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    // Per-thread mapping for the 16×16 dequant tile:
    //   row = tid >> 1                     (0..15)  — N row in this block
    //   half = tid & 1                     (0/1)    — which 8-K-lane subtile
    // Each thread dequants 8 K-elements per K-step into B_smem.
    const int n_local   = tid >> 1;          // 0..15 (within block)
    const int k_half    = tid & 1;           // 0 or 1
    const int n_global  = n_block + n_local; // global N row (clamped use below)
    const int n_in_zero = n_global & 7;      // bit-position inside zero int32
    const int n_zero_g  = n_global >> 3;     // zero row index

    for (int k_base = 0; k_base < K; k_base += 16) {
        // ---- 1. Stage A_smem from gmem (RowMajor [M, K]) ----
        // 16 rows × 16 cols = 256 f16 elements; 32 threads → 8 each.
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int idx = i * 32 + tid;     // 0..255 covers full tile
            const int r   = idx >> 4;          // 0..15 rows
            const int c   = idx & 0xF;         // 0..15 cols
            const int gm  = m_block + r;
            const int gk  = k_base  + c;
            __half v = __float2half(0.0f);
            if (gm < M && gk < K) {
                v = A[(size_t)gm * K + gk];
            }
            A_smem[r * 16 + c] = v;
        }

        // ---- 2. Dequantize 16 N rows × 16 K cols of B into B_smem ----
        // Each thread owns one N row + one 8-K-lane subtile.
        // weight_packed at [n_global, (k_base + k_half * 8) / 8]
        //                = [n_global, (k_base / 8) + k_half]
        const int g       = k_base / group_size;        // K-group (all 16 share)
        float scale_f     = 0.0f;
        int   zero_pack   = 0;
        int32_t w_pack    = 0;
        if (n_global < N) {
            scale_f   = bf16_to_f32(B_scale[(size_t)n_global * K_div_g + g]);
            zero_pack = B_zero  [(size_t)n_zero_g * K_div_g + g];
            w_pack    = B_packed[(size_t)n_global * K_div_8 + (k_base >> 3) + k_half];
        }
        const int   z_int = (zero_pack >> (4 * n_in_zero)) & 0xF;
        const float z_f   = (float)z_int;

        // 8 lanes, write into B_smem col-major [16][16] (col-major for
        // wmma::matrix_b col_major load): smem[k * 16 + n].
        #pragma unroll
        for (int lane = 0; lane < 8; ++lane) {
            const int   w_int = (w_pack >> (4 * lane)) & 0xF;
            const float dq    = ((float)w_int - z_f) * scale_f;
            const int   k_local = k_half * 8 + lane;       // 0..15
            const int   gk      = k_base + k_local;
            __half v = __float2half(0.0f);
            if (n_global < N && gk < K) {
                v = __float2half(dq);
            }
            // WMMA matrix_b is logically [K, N]. col_major with ldb=16
            // means: element at (k_local, n_local) lives at
            // n_local * ldb + k_local — N is the column stride, K is
            // contiguous within a column. Codex review of cycle 51d.1
            // caught the reversed indexing in the first version.
            B_smem[n_local * 16 + k_local] = v;
        }

        __syncthreads();

        // ---- 3. WMMA load + accumulate ----
        wmma::load_matrix_sync(a_frag, A_smem, 16);
        wmma::load_matrix_sync(b_frag, B_smem, 16);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

        __syncthreads();
    }

    // ---- 4. Store 16×16 output sub-tile ----
    // Convert f32 accumulator → f16 in shared first (wmma stores to f32
    // requires float* / int* targets; round-trip through smem).
    __shared__ __align__(16) float acc_smem[16 * 16];
    wmma::store_matrix_sync(acc_smem, acc_frag, 16, wmma::mem_row_major);
    __syncthreads();

    // 32 threads cover 256 elements: 8 each.
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int idx = i * 32 + tid;
        const int r   = idx >> 4;
        const int c   = idx & 0xF;
        const int gm  = m_block + r;
        const int gn  = n_block + c;
        if (gm < M && gn < N) {
            D[(size_t)gm * ld_d + gn] = __float2half(acc_smem[r * 16 + c]);
        }
    }
}
