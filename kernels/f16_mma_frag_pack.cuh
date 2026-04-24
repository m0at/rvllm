// Fragment-packing helpers for the SM80-era f16 tensor-core MMA
//     mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
//
// This is the MMA the FlashInfer xqa NVFP4 path uses on SM120 after
// on-the-fly dequant via `cvt.rn.f16x2.e2m1x2` (see
// `flashinfer/data/csrc/xqa/mma.cuh:36` + `utils.cuh:757`). Task
// `aa01001nvf4f16mma` ports that pattern into our FA2 NVFP4 decode
// + prefill kernels. This header is header-only; callers populate
// f16 smem tiles with Q / K / V fragments (possibly dequanted from
// NVFP4 via the `cvt.rn` helper), then pack per-lane fragments
// through the helpers below.
//
// Tile shape: m = 16, n = 8, k = 16. Per-lane fragments:
//   A: 4 × u32 = 8 × f16 covering a 16×16 slice of the A tile
//   B: 2 × u32 = 4 × f16 covering an  8×16 slice of the B tile
//   D: 4 × f32 covering a 16×8 slice of the accumulator tile
//
// Per-lane layout (PTX ISA 8.8 §9.7.13.4, `row.col`):
//   lane `i`, row_lo = i/4, row_hi = row_lo + 8, c_lo = (i%4)*2
//   A: a[0]={A[row_lo, c_lo], A[row_lo, c_lo+1]}  // low-high in u32
//      a[1]={A[row_hi, c_lo], A[row_hi, c_lo+1]}
//      a[2]={A[row_lo, c_lo+8], A[row_lo, c_lo+9]}
//      a[3]={A[row_hi, c_lo+8], A[row_hi, c_lo+9]}
//   B (col-major [N][K] where lane i's n=i/4, k_lo=(i%4)*2):
//      b[0]={B[n, k_lo], B[n, k_lo+1]}
//      b[1]={B[n, k_lo+8], B[n, k_lo+9]}
//   D: d[0]=D[row_lo,  c_lo+0]  d[1]=D[row_lo,  c_lo+1]
//      d[2]=D[row_hi, c_lo+0]  d[3]=D[row_hi, c_lo+1]
//
// The D layout is the SAME as FP8 m16n8k32 (shape m16n8 is identical;
// only the k dim differs). Reuse `rvllm::unpack_d_frag_to_smem_m16n8`
// from `fp8_mma_frag_pack.cuh`.

#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include "fp8_mma_frag_pack.cuh"  // reuse the m16n8 D unpacker + zero_mma_d_frag

namespace rvllm_f16mma {

// Pack per-lane A fragment from a row-major [16 × K] f16 smem tile
// (K ≥ 16). `stride_bytes` is the byte stride between consecutive
// rows — usually `K * sizeof(half)` but allow slack for interleaved
// layouts.
__device__ __forceinline__ void pack_a_frag_row_major_m16k16_f16(
    const __half* smem_a,
    int           stride_bytes,
    uint32_t      a[4],
    int           lane)
{
    const int r_lo = lane >> 2;            // lane / 4
    const int r_hi = r_lo + 8;
    const int c_lo = (lane & 3) << 1;       // (lane % 4) * 2
    const int c_hi = c_lo + 8;
    const unsigned char* base = reinterpret_cast<const unsigned char*>(smem_a);
    a[0] = *reinterpret_cast<const uint32_t*>(base + r_lo * stride_bytes + c_lo * (int)sizeof(__half));
    a[1] = *reinterpret_cast<const uint32_t*>(base + r_hi * stride_bytes + c_lo * (int)sizeof(__half));
    a[2] = *reinterpret_cast<const uint32_t*>(base + r_lo * stride_bytes + c_hi * (int)sizeof(__half));
    a[3] = *reinterpret_cast<const uint32_t*>(base + r_hi * stride_bytes + c_hi * (int)sizeof(__half));
}

// Pack per-lane B fragment from a col-major [8 × K] f16 smem tile,
// i.e. an [N][K] storage where N is the outer axis and `stride_bytes`
// is the byte stride between consecutive N rows.
__device__ __forceinline__ void pack_b_frag_col_major_n8k16_f16(
    const __half* smem_b,
    int           stride_bytes,
    uint32_t      b[2],
    int           lane)
{
    const int n    = lane >> 2;
    const int k_lo = (lane & 3) << 1;
    const int k_hi = k_lo + 8;
    const unsigned char* base = reinterpret_cast<const unsigned char*>(smem_b);
    b[0] = *reinterpret_cast<const uint32_t*>(base + n * stride_bytes + k_lo * (int)sizeof(__half));
    b[1] = *reinterpret_cast<const uint32_t*>(base + n * stride_bytes + k_hi * (int)sizeof(__half));
}

// D unpack (16×8 f32) — identical mapping to FP8 m16n8k32 D, reuse.
__device__ __forceinline__ void unpack_d_frag_to_smem_m16n8(
    float*       smem_d,
    int          stride_bytes,
    const float  d[4],
    int          lane)
{
    rvllm::unpack_d_frag_to_smem_m16n8(smem_d, stride_bytes, d, lane);
}

// Explicit per-register zero via PTX `mov.f32`, reusing the FP8
// helper (same four-register dedup bug class as documented at
// `rvllm::zero_mma_d_frag` — naked `float d[4] = {0}` compiles to a
// single mov that only initialises d[0]).
__device__ __forceinline__ void zero_mma_d_frag(float d[4]) {
    rvllm::zero_mma_d_frag(d);
}

// Thin wrapper. `"+f"` (read-write) constraint is load-bearing here
// for the same reason as the FP8 variant — with separate `"=f"`
// output and `"f"` input constraints, the compiler deduplicates the
// four C-side registers when d is zero-initialised and the MMA reads
// garbage for three of the four C positions.
__device__ __forceinline__ void mma_m16n8k16_f16_f16_f32(
    float           d[4],
    const uint32_t  a[4],
    const uint32_t  b[2])
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1])
    );
#else
    (void)a; (void)b;
    d[0] = d[1] = d[2] = d[3] = 0.0f;
#endif
}

} // namespace rvllm_f16mma
