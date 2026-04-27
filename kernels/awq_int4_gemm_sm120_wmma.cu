// Cycle 51 step 10d: AWQ INT4 W4A16 GEMM for prefill (M>1) on SM120/121.
//
// HAND-ROLLED WMMA path. SKELETON — kernel body lands across cycles
// 51d.1 / 51d.2 / 51d.3 (tile structure, INT4 unpack, validation).
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
// Cycle 51 step 10a (first commit): planned this as a CUTLASS-derived
// SM120 fallback, modeled on cutlass_fp8_gemm_blockscale_sm120.cu.
// Codex initially recommended that path.
//
// Cycle 51 step 10d (this commit): survey of CUTLASS 4.5 tree at
// cutlass/include/cutlass/gemm/collective/ shows SM120 has only
// blockscaled (FP4/NVFP4) and blockwise-scaling (FP8) collectives.
// **NO SM120 mixed-input collective exists.** The mixed-input lives
// at SM90 (sm90_mma_tma_gmma_rs_warpspecialized_mixed_input.hpp,
// Hopper GMMA + TMA) and SM100 (Blackwell-DC, tcgen05/TMEM).
// Adapting either to SM120 is template surgery (600+ LOC) with high
// risk and unclear payoff because the architectural assumptions
// (TMA, cluster multicast, tcgen05) do not line up.
//
// Codex revised recommendation (cycle 51d consult): **hand-rolled
// WMMA SM120 INT4xF16 kernel with narrow feature target**. Bounded
// scope (~400-500 LOC), preserves the bandwidth win, debuggable,
// avoids both pre-dequant (defeats the win) and the GEMV-loop
// (structurally slow prefill). File renamed accordingly:
// `cutlass_*` -> `awq_int4_gemm_sm120_wmma.cu`.
//
// === KERNEL DESIGN ===
//
// Tile structure (initial, fixed):
//   Block: 256 threads (8 warps).
//   Each warp owns a 16x16 output sub-tile via __half WMMA fragments.
//   Block tile: 64x64 (4 warp-rows x 4 warp-cols of 16x16 each).
//   K-step: 16 elements/iteration (one WMMA K dimension).
//
//   Block-level shared memory layout:
//     A_smem: 64 x 64 f16  =  8 KB
//     B_smem: 64 x 16 f16  =  2 KB  (dequantized weights, double-buffered)
//
//   At each K-step:
//     1. Cooperative cp.async load A_smem from gmem.
//     2. Cooperative cp.async load weight_packed (int32 [N, K/8]) +
//        scale (bf16) + zero_point (int32) for the current K-tile.
//     3. Per-warp dequant in registers: unpack 8 nibbles / int32, apply
//        (nibble - zero) * scale to get f16 weight.
//     4. Stage dequant'd f16 weights into B_smem (16 elements/k-step).
//     5. wmma::mma_sync into accumulator fragments.
//
//   N-stride along the 8-INT4-packed-int32 axis: 8 N-rows share one
//   int32 zero-point. Block tile of 64 N-rows reads 8 int32s per
//   K-group; each warp owns 16 rows = 2 int32s.
//
// Caller ABI:
//
//   extern "C" int awq_int4_gemm_sm120_wmma(
//       __half*           D,                       // [M, N] f16 out (RowMajor)
//       const __half*     A,                       // [M, K] f16 act (RowMajor)
//       const int32_t*    B_packed,                // [N, K/8] i32  (RowMajor)
//       const __nv_bfloat16* B_scale,              // [N, K/g] bf16 (RowMajor)
//       const int32_t*    B_zero,                  // [N/8, K/g] i32 (RowMajor)
//       int M, int N, int K, int group_size,
//       cudaStream_t stream
//   );
//
// First impl target (cycle 51d.1): correctness on canonical Gemma 4
// 31B q_proj prefill (M=128..2048, N=8192, K=5376, g=128). Match
// awq_int4_gemv_f16_kernel within fp16 noise (~0.01 max abs).
//
// Cycle 51d.2: bandwidth tuning - cp.async pipelining, smem bank
// conflict avoidance on the bf16 scale loads.
//
// Cycle 51d.3: build_awq_int4_gemm_sm120_wmma.sh + Rust wrapper crate
// + replace AwqInt4GemvF16PrefillLoop calls in the 4 exec_layer
// dispatch sites.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdint>

// IMPLEMENTATION DEFERRED. Compiles as a TU with no symbol so the
// build script can place it in the kernels/ tree without breaking.
// Cycle 51d.1 replaces this with the actual kernel body.
extern "C" __global__ void awq_int4_gemm_sm120_wmma_placeholder() {
    // intentional no-op
}
