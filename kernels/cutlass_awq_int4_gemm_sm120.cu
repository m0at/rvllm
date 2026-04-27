// Cycle 51 step 10a: AWQ INT4 W4A16 GEMM for prefill (M>1) on SM120/121.
//
// SKELETON — design documented; kernel body lands across cycles 51b-d.
//
//   D_f16[m, n] = sum_k (A_f16[m, k] * dequant_int4(B[n, k]))
//
//   B_packed:    int32  [N, K/8]      8 INT4 per int32 along K
//   B_scale:     bf16   [N, K/g]      group_size=128 typical
//   B_zero:      int32  [N/8, K/g]    8 INT4 per int32 along N
//
//   value(n, k) = (nibble - zero[n/8, g][n%8]) * scale[n, g]   for k in group g
//
// Codex recommendation (cycle 51 design consult, 2026-04-27): pragmatic
// CUTLASS-derived SM120 fallback. Template comes from
// cutlass_fp8_gemm_blockscale_sm120.cu (an SM120 blockwise FP8 GEMM
// already in the tree, derived from CUTLASS example 87a) — strip the
// FP8 input handling, replace with INT4 dequant operator on the B
// operand, retain f16 epilogue.
//
// SM120 vs SM90 deltas:
//   * No tcgen05 / TMEM (no Blackwell-DC). Use conventional WMMA-tiled
//     SIMT path. cluster-aware features in CUTLASS 4.5 mixed_input
//     example 55 / 69 stripped.
//   * Shared memory: 99 KB per SM. 128×128 tile + INT4 dequant scratch
//     fits comfortably (~96 KB SMEM budget).
//   * Bandwidth: 273 GB/s LPDDR5x — INT4 weight reads halve the FP8
//     bandwidth bill, matching the GEMV decode kernel's win pattern.
//
// Caller contract (matches the SM120 FP8 GEMM ABI for drop-in
// replacement at the launcher level):
//
//   extern "C" int cutlass_awq_int4_gemm_sm120(
//       __half*           D,                    // [M, N] f16 out
//       const __half*     A,                    // [M, K] f16 act
//       const int32_t*    B_packed,             // [N, K/8] i32
//       const __nv_bfloat16* B_scale,           // [N, K/g] bf16
//       const int32_t*    B_zero,               // [N/8, K/g] i32
//       int M, int N, int K, int group_size,
//       cudaStream_t stream
//   );
//
// First implementation target (cycle 51b): correctness on canonical
// Gemma 4 31B prefill shapes (M=2048 typical max, N=8192 q_proj /
// 5376 down_proj, K=5376 / 21504, g=128). Numerical match to the
// decode GEMV kernel within fp16 noise (~0.01 max abs, ~0.001 mean
// rel). Bench target: > 50 TFLOPS at M=128 (vs CUTLASS SM120 FP8
// blockscale's 102 TFLOPS at N=4608, K=5376 — INT4 weight half-
// bandwidth should keep within 2x).
//
// Cycle 51c: integration into the layer-exec dispatch sites — replace
// the four AwqInt4GemvF16Launch FeatureNotAvailable rejects for M>1
// with this GEMM, keep the GEMV for M=1.
//
// Cycle 51d: validate end-to-end against ebircak/gemma-4-31B-it-4bit-
// W4A16-AWQ. Target TTFT for 2k-token prompt: < 5s (vs 60+s for the
// per-token GEMV loop fallback).

// IMPLEMENTATION DEFERRED. Compiles as a TU with no symbol so the
// build script can place it in the kernels/ tree without breaking.
// Cycle 51b replaces this with the actual kernel.

#include <cuda_runtime.h>
extern "C" __global__ void cutlass_awq_int4_gemm_sm120_placeholder() {
    // intentional no-op
}
