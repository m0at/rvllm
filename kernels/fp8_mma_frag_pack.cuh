// Fragment-packing helpers for the FP8 E4M3 tensor-core MMA
//     mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32
//
// The per-lane fragment layout is fixed by the PTX spec — see the
// comment at the top of `kernels/fp8_e4m3_mma_probe.cu`. This header
// encapsulates the lane → (row, col, k) arithmetic so the full
// attention kernel can load Q / K / V / P tiles out of shared memory
// without open-coded byte shuffling.
//
// Each A fragment is 4 × u32 (= 16 FP8 bytes) covering a 16 × 32
// slice of a row-major source. Each B fragment is 2 × u32 (= 8 FP8
// bytes) covering an 8 × 32 slice of a col-major source. A "row" of
// a u32 load reads 4 contiguous FP8 bytes along the k axis — the
// source-tile stride controls how rows are addressed.
//
// These helpers are intentionally `__forceinline__`: the compiler
// should collapse them into simple `ld.shared.u32` + 4-byte offset
// arithmetic at the call site. Upgrading to `ldmatrix` is a later
// optimisation if smem bandwidth becomes a bottleneck (Phase F3+).
//
// Assumptions about the source tile:
//  * `smem_a` / `smem_b` point at the start of a contiguous tile in
//    shared memory, aligned to 4 bytes.
//  * `stride_bytes` is the byte stride between consecutive rows
//    (A: row stride = k_stride) or columns (B: col stride = k_stride
//    for a col-major [N][K] storage where n is outer).
//  * The tile is wide enough: A is at least (m=16) × (k=32) bytes,
//    B is at least (n=8) × (k=32) bytes.

#pragma once

#include <cstdint>

namespace rvllm {

// Pack the per-lane A fragment for `mma.sync m16n8k32` from a
// row-major [16 × K] FP8 tile starting at `smem_a`, where rows are
// spaced `stride_bytes` apart. `lane` is `threadIdx.x % 32`.
//
// Layout reminder (`i = lane`):
//   a[0]: row (i/4),    k = (i%4)*8 + [0..3]
//   a[1]: row (i/4+8),  k = (i%4)*8 + [0..3]
//   a[2]: row (i/4),    k = (i%4)*8 + [4..7]
//   a[3]: row (i/4+8),  k = (i%4)*8 + [4..7]
__device__ __forceinline__ void pack_a_frag_row_major_m16k32(
    const unsigned char* smem_a,
    int                  stride_bytes,
    uint32_t             a[4],
    int                  lane)
{
    const int r_lo  = lane >> 2;           // lane / 4
    const int r_hi  = r_lo + 8;
    const int k_lo  = (lane & 3) << 3;     // (lane % 4) * 8
    const int k_hi  = k_lo + 4;
    const unsigned char* row_lo = smem_a + r_lo * stride_bytes;
    const unsigned char* row_hi = smem_a + r_hi * stride_bytes;
    a[0] = *reinterpret_cast<const uint32_t*>(row_lo + k_lo);
    a[1] = *reinterpret_cast<const uint32_t*>(row_hi + k_lo);
    a[2] = *reinterpret_cast<const uint32_t*>(row_lo + k_hi);
    a[3] = *reinterpret_cast<const uint32_t*>(row_hi + k_hi);
}

// Pack the per-lane B fragment for `mma.sync m16n8k32` from a
// col-major [8 × K] FP8 tile starting at `smem_b`, where consecutive
// columns (n-rows of the [N][K] storage) are spaced `stride_bytes`
// apart. `lane` is `threadIdx.x % 32`.
//
// Layout reminder (`i = lane`):
//   b[0]: col (i/4), k = (i%4)*8 + [0..3]
//   b[1]: col (i/4), k = (i%4)*8 + [4..7]
__device__ __forceinline__ void pack_b_frag_col_major_n8k32(
    const unsigned char* smem_b,
    int                  stride_bytes,
    uint32_t             b[2],
    int                  lane)
{
    const int n     = lane >> 2;
    const int k_lo  = (lane & 3) << 3;
    const int k_hi  = k_lo + 4;
    const unsigned char* col = smem_b + n * stride_bytes;
    b[0] = *reinterpret_cast<const uint32_t*>(col + k_lo);
    b[1] = *reinterpret_cast<const uint32_t*>(col + k_hi);
}

// Unpack the per-lane D fragment from `mma.sync m16n8k32` into a
// [16 × 8] f32 tile in shared memory. `smem_d` has row stride
// `stride_bytes` (usually 8 × sizeof(float) = 32, but allow slack
// for interleaving with other state).
//
// Layout reminder (`i = lane`):
//   d[0]: row (i/4),    col (i%4)*2 + 0
//   d[1]: row (i/4),    col (i%4)*2 + 1
//   d[2]: row (i/4+8),  col (i%4)*2 + 0
//   d[3]: row (i/4+8),  col (i%4)*2 + 1
__device__ __forceinline__ void unpack_d_frag_to_smem_m16n8(
    float*       smem_d,
    int          stride_bytes,
    const float  d[4],
    int          lane)
{
    const int r_lo = lane >> 2;
    const int r_hi = r_lo + 8;
    const int c    = (lane & 3) << 1;
    float* row_lo = reinterpret_cast<float*>(
        reinterpret_cast<unsigned char*>(smem_d) + r_lo * stride_bytes);
    float* row_hi = reinterpret_cast<float*>(
        reinterpret_cast<unsigned char*>(smem_d) + r_hi * stride_bytes);
    row_lo[c + 0] = d[0];
    row_lo[c + 1] = d[1];
    row_hi[c + 0] = d[2];
    row_hi[c + 1] = d[3];
}

// Explicitly zero all four D-fragment registers via PTX `mov.f32`.
// Callers that want a "fresh" (non-accumulating) MMA output must run
// this BEFORE the MMA — a naked `float d[4] = {0}` compiles to a
// single `mov.f32 %0, 0` that only initialises the first slot and
// leaves d[1..3] carrying stale values from an earlier compiler-
// hoisted reg lifetime. That was the F4 P·V breakage in Phase F6.
__device__ __forceinline__ void zero_mma_d_frag(float d[4]) {
    asm volatile(
        "mov.f32 %0, 0f00000000;\n\t"
        "mov.f32 %1, 0f00000000;\n\t"
        "mov.f32 %2, 0f00000000;\n\t"
        "mov.f32 %3, 0f00000000;"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
    );
}

// Thin wrapper around the PTX. `d` is in-out accumulator — value at
// entry is the `C` operand, value at exit is `D`. Using `"+f"`
// (read-write) instead of separate `"=f"` output and `"f"` input is
// load-bearing: with separate constraints, when d is initialised to
// `{0, 0, 0, 0}` the compiler deduplicates all four C-side registers
// to a single register (observed as `{%f409, %f409, %f409, %f409}`
// in PTX). The tensor core then reads garbage for three of the four
// C positions and returns NaN / inf — the F4 P·V breakage that
// Phase F6 tracked down.
__device__ __forceinline__ void mma_m16n8k32_e4m3_e4m3_f32(
    float           d[4],
    const uint32_t  a[4],
    const uint32_t  b[2])
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    asm volatile(
        "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
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

} // namespace rvllm
