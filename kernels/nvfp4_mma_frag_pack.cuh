// Fragment-packing helpers for the Blackwell native NVFP4 (e2m1)
// tensor-core MMA. Production instruction:
//
//     mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X
//         .m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
//
// Validated end-to-end by `nvfp4_mma_probe_check.py` +
// `nvfp4_mma_layout_probe.py` (abs_err = 0.0 vs fp64 reference for
// small integer inputs; K_eff = 64 confirmed).
//
// Layout (derived empirically via the one-hot sweep in
// `nvfp4_mma_layout_probe.py`):
//   * A: 4 × u32 per lane = 16 bytes = 32 e2m1/lane. M-row per lane:
//       reg_idx ∈ {0, 2} → m = lane / 4       (rows 0-7)
//       reg_idx ∈ {1, 3} → m = lane / 4 + 8   (rows 8-15)
//     4 lanes share each m-row pair. 8 nibbles per reg cover K.
//   * B: 2 × u32 per lane =  8 bytes = 16 e2m1/lane.
//       reg ∈ {0, 1} → n = lane / 4    (cols 0-7)
//     4 lanes share each n-col. 8 nibbles per reg cover K.
//   * D: 4 × f32 per lane, standard m16n8 layout (same as fp8/fp16).
//
// Byte-level layout is IDENTICAL to the e4m3 fp8 packer — each byte
// just now holds two e2m1 values (low nibble = even K, high nibble =
// odd K within that byte's K-stripe). Consequence: the existing
// `pack_a_frag_row_major_m16k32` / `pack_b_frag_col_major_n8k32`
// helpers from `fp8_mma_frag_pack.cuh` are REUSED verbatim — the only
// semantic change is K advances by 32 bytes = 64 e2m1 per MMA.
//
// Scale handling: the MMA takes per-operand `sfa`, `sfb` registers
// (u32 = 4 × ue4m3 bytes, one scale per K=16 block). These map
// directly onto NVFP4's per-16-elem E4M3 microscale stored alongside
// the cache — the FA2 integration can feed them in without a
// dequantize step. `scale_vec::4X` means 4 scales per MMA per
// operand, covering K=64 in 4 sub-blocks of K=16.
//
// E4M3 encoding reminders (for `ue4m3` scales):
//   Value 1.0 = bits 0x38 (sign=0, exp=0111, mant=000, bias=7).
//   Value 128 = bits 0x70 — a factor of 128, NOT 1.0.

#pragma once

#include <cstdint>
#include "fp8_mma_frag_pack.cuh"

namespace rvllm_nvfp4 {

// Re-export the byte-level packers under the NVFP4 namespace. The
// byte-level semantics are identical; the only difference is the
// element interpretation at the MMA site.
__device__ __forceinline__ void pack_a_frag_e2m1_m16k64(
    const unsigned char* smem_a,
    int                  stride_bytes,   // row stride in BYTES
    uint32_t             a[4],
    int                  lane)
{
    // Byte layout matches e4m3 — reuse the packer.
    rvllm::pack_a_frag_row_major_m16k32(smem_a, stride_bytes, a, lane);
}

__device__ __forceinline__ void pack_b_frag_e2m1_n8k64(
    const unsigned char* smem_b,
    int                  stride_bytes,   // col stride in BYTES
    uint32_t             b[2],
    int                  lane)
{
    rvllm::pack_b_frag_col_major_n8k32(smem_b, stride_bytes, b, lane);
}

__device__ __forceinline__ void unpack_d_frag_to_smem_m16n8(
    float*       smem_d,
    int          stride_bytes,
    const float  d[4],
    int          lane)
{
    rvllm::unpack_d_frag_to_smem_m16n8(smem_d, stride_bytes, d, lane);
}

__device__ __forceinline__ void zero_mma_d_frag(float d[4]) {
    rvllm::zero_mma_d_frag(d);
}

// DEPRECATED — kept only for the existing `nvfp4_mma_probe_kernel`
// smoke test that confirms the assembler accepts the opcode. The
// `kind::f8f6f4.e2m1.e2m1` variant does NOT produce correct MMA
// semantics on sm_120/sm_121 — the all-ones × all-ones probe returns
// D=2 per cell (should be K=32) and single-nibble one-hot produces
// negative values. The production NVFP4 MMA is
// `kind::mxf4nvf4.block_scale` — see `mma_m16n8k64_nvfp4_e2m1_e2m1`
// below. Do not use this wrapper for FA2 integration.
__device__ __forceinline__ void mma_m16n8k32_e2m1_e2m1_f32(
    float           d[4],
    const uint32_t  a[4],
    const uint32_t  b[2])
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    asm volatile(
        "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e2m1.f32 "
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

// Production NVFP4 MMA. Matches the CUTLASS SM120_16x8x64_TN_VS
// blockscaled variant (cute/arch/mma_sm120.hpp:3216 family):
//     mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X
//         .m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
//
// Contract:
//   * K = 64 e2m1 values per MMA (vs 32 for `.kind::f8f6f4`).
//   * A: 4×u32 per lane = 16 bytes = 32 e2m1/lane. 32 lanes × 32 = 1024
//       = 16 rows × 64 K ✓. Byte layout matches CUTLASS's fp8 packer at
//       byte granularity (each byte holds 2 e2m1 values).
//   * B: 2×u32 per lane =  8 bytes = 16 e2m1/lane. 32 × 16 = 512
//       = 8 rows × 64 K ✓.
//   * D: 4×f32 per lane, standard 16×8 m16n8 D layout (unchanged).
//   * `sfa` / `sfb`: 4× `ue4m3` scales packed into a u32 per operand
//     per lane — one scale per K=16 block (matches NVFP4's per-16-elem
//     microscale exactly). Hardware applies `scale_A[k/16] *
//     scale_B[k/16]` to each K=16 partial product before adding to the
//     accumulator, so FA2 integration no longer needs a separate
//     post-MMA rescale step — the MMA consumes the cache's
//     `key_cache_scale` / `value_cache_scale` bytes directly.
//   * `bidA = bidB = tidA = tidB = 0` — uniform scale-table indexing.
//
// Accumulator pattern: `+f` (read-write D, same as the old wrapper) to
// keep the FA2 inner loop's chained-MMA pattern compact. `C` is
// implicitly D here.
__device__ __forceinline__ void mma_m16n8k64_nvfp4_e2m1_e2m1_f32_ue4m3(
    float           d[4],
    const uint32_t  a[4],
    const uint32_t  b[2],
    uint32_t        sfa,
    uint32_t        sfb)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X"
        ".m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3}, "                     // d[0..3]
        "{%4, %5, %6, %7}, "                     // a[0..3]
        "{%8, %9}, "                             // b[0..1]
        "{%0, %1, %2, %3}, "                     // c = d (accumulator)
        "{%10}, "                                // sfa (1 × u32 = 4 × ue4m3)
        "{%11, %11}, "                           // bidA, tidA — both 0
        "{%12}, "                                // sfb
        "{%11, %11};\n"                          // bidB, tidB — both 0
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "r"(sfa),
          "h"(uint16_t(0)),
          "r"(sfb)
    );
#else
    (void)a; (void)b; (void)sfa; (void)sfb;
    d[0] = d[1] = d[2] = d[3] = 0.0f;
#endif
}

// Path A MMA wrapper — plain `kind::f8f6f4` mixed E4M3 × E2M1, unscaled.
//
//     mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e2m1.f32
//
// For our flash-attention integration: Q is FP8 E4M3, K is NVFP4 (stored
// packed 2-nibble-per-byte; the caller's packer MUST unpack to
// one-e2m1-per-byte with the CUTLASS `<<2` shift, i.e. 0b0000ABCD →
// 0b00ABCD00 — see `mma_traits_sm120.hpp:211`).
//
// Unscaled: per-K=16 NVFP4 block scales are applied post-MMA by the
// caller (issue two MMAs per K=32 tile, one per K=16 sub-block, each
// with the "other half" of K zeroed out in B; multiply each accumulator
// by its per-K=16 scale; sum).
//
// Register shapes: A = 4×u32 (m=16, k=32 fp8 bytes), B = 2×u32 (n=8,
// k=32 e2m1 bytes in `<<2`-shifted form), D = 4×f32. Same as the plain
// E4M3×E4M3 variant.
__device__ __forceinline__ void mma_m16n8k32_f8f6f4_e4m3_e2m1_f32(
    float           d[4],
    const uint32_t  a[4],
    const uint32_t  b[2])
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    asm volatile(
        "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e2m1.f32 "
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

// Path A B-fragment packer. Reads NVFP4 bytes (2 e2m1 per byte, low
// nibble = even K, high nibble = odd K) from a [8 rows × k_nibbles_per_row/2
// bytes] smem tile starting at `smem_b_nvfp4`. Unpacks into one-per-byte
// `<<2`-shifted form in the lane's b[2] fragment.
//
// `kblk_idx` selects which K=16 sub-block is "active":
//   * kblk_idx == 0 → K[0..15] live, K[16..31] zeroed.
//   * kblk_idx == 1 → K[16..31] live, K[0..15] zeroed.
// This is the per-MMA masking that lets the FA2 caller apply per-K=16
// NVFP4 scales post-MMA via two MMA issues per K=32 tile.
//
// Per the standard fp8 B-packer layout, each lane owns:
//   n = lane / 4
//   k_byte_lo = (lane % 4) * 8 + [0..3]  (in the unpacked f8f6f4 K-byte
//                                         space, so these are K-indices
//                                         0..31 of the e2m1 row)
//   k_byte_hi = (lane % 4) * 8 + [4..7]
// Each lane therefore lives entirely in ONE K=16 sub-block:
//   lane % 4 ∈ {0, 1} → K-block 0   (K[0..15])
//   lane % 4 ∈ {2, 3} → K-block 1   (K[16..31])
//
// So the mask is a single per-lane check against `kblk_idx`.
//
// `stride_bytes_nvfp4` is the row stride of `smem_b_nvfp4` in packed
// bytes (= k_nibbles_per_row / 2, typically 16 for a K=32 tile).
__device__ __forceinline__ void pack_b_frag_e4m3_x_e2m1_m16k32_path_a(
    const unsigned char* smem_b_nvfp4,
    int                  stride_bytes_nvfp4,
    int                  kblk_idx,       // 0 or 1
    uint32_t             b[2],
    int                  lane)
{
    const int n         = lane >> 2;
    const int lane_kblk = (lane & 3) >> 1;   // 0 for lane%4 ∈ {0,1}, 1 for {2,3}
    if (lane_kblk != kblk_idx) {
        b[0] = 0u;
        b[1] = 0u;
        return;
    }
    // Active lane — unpack 4 NVFP4 bytes per register (= 8 e2m1 values)
    // into a u32 holding 4 packed-shifted f8f6f4 bytes. Wait — b[0] must
    // hold the 4 K-values as 4 bytes for the MMA, one e2m1 per byte in
    // `<<2`-shifted form. So 4 e2m1 → 4 bytes → 1 u32 of output b[0]
    // requires 4 *input* e2m1 values = 2 packed NVFP4 bytes.
    //
    // Lane L (active): reads K-positions (L%4)*8 + [0..3] for b[0] and
    // (L%4)*8 + [4..7] for b[1]. In packed NVFP4 storage these are 4
    // contiguous e2m1 values = 2 NVFP4 bytes each.
    const int k_lo   = (lane & 3) << 3;          // e2m1-index
    const int k_hi   = k_lo + 4;
    const int kl_byte0 = k_lo >> 1;              // packed-byte index
    const int kh_byte0 = k_hi >> 1;
    const unsigned char* row = smem_b_nvfp4 + n * stride_bytes_nvfp4;

    // Pull 2 packed bytes for b[0] (covering K k_lo..k_lo+3).
    uint32_t pb0 = row[kl_byte0 + 0];           // k_lo+0 (low) + k_lo+1 (high)
    uint32_t pb1 = row[kl_byte0 + 1];           // k_lo+2 (low) + k_lo+3 (high)
    // Same for b[1].
    uint32_t pb2 = row[kh_byte0 + 0];
    uint32_t pb3 = row[kh_byte0 + 1];

    // Unpack 2 e2m1 per byte to 1 e2m1 per byte with `<<2` shift.
    // Low nibble → byte [0], high nibble → byte [1] within each input pair.
    #define EXPAND_NVFP4_PAIR(p0, p1) (                                    \
          (((p0)     ) & 0xFu) << 2                                        \
        | ((((p0) >> 4) & 0xFu) << 2) << 8                                 \
        | ((((p1)     ) & 0xFu) << 2) << 16                                \
        | ((((p1) >> 4) & 0xFu) << 2) << 24 )
    b[0] = EXPAND_NVFP4_PAIR(pb0, pb1);
    b[1] = EXPAND_NVFP4_PAIR(pb2, pb3);
    #undef EXPAND_NVFP4_PAIR
}

} // namespace rvllm_nvfp4
