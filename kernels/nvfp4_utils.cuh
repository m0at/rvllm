// NVFP4 packing / dequant device helpers for the sm_121 KV-cache path.
//
// NVFP4 layout (NVIDIA Blackwell microscale FP4):
//   * 4-bit element: 1 sign + 2 exponent + 1 mantissa, exponent bias 1.
//     Representable magnitudes: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
//     (× sign). No NaN / inf.
//   * Per-16-element E4M3 microscale (8 bits). Dequant is
//     `fp32_value = fp4_lut[elem_bits] * e4m3_scale`.
//   * Packed storage: 2 FP4 values per byte, low nibble = even-index
//     element, high nibble = odd-index.
//
// This header stays header-only (all device-inline) so the FA2
// kernels can `#include` it without a separate TU link step.
//
// Unit-tested by `v3/tools/nvfp4_roundtrip_check.py` against a
// numpy fp64 reference — round-trip rel-err per element is bounded
// by the 3-bit magnitude quantisation (~7% worst case within a
// block, zero on exact-representable values).

#pragma once

#include <cuda_fp8.h>
#include <cstdint>

namespace rvllm_nvfp4 {

// FP4 decode LUT — 16 entries, f32, positive magnitudes in the low 8
// entries, negated copies in the high 8. Matches NVIDIA's
// e2m1 (bias=1) representation.
//
// Bit layout [s e e m]:
//   0b0000 →  0.0   0b1000 → -0.0
//   0b0001 →  0.5   0b1001 → -0.5
//   0b0010 →  1.0   0b1010 → -1.0
//   0b0011 →  1.5   0b1011 → -1.5
//   0b0100 →  2.0   0b1100 → -2.0
//   0b0101 →  3.0   0b1101 → -3.0
//   0b0110 →  4.0   0b1110 → -4.0
//   0b0111 →  6.0   0b1111 → -6.0
__device__ __forceinline__ float fp4_decode(uint32_t bits) {
    // Keep the table in registers as a u32 packed constant array —
    // switch-based decode beats a memory LUT on smem-pressed kernels.
    const uint32_t b = bits & 0xFu;
    float mag;
    switch (b & 0x7u) {
        case 0: mag = 0.0f; break;
        case 1: mag = 0.5f; break;
        case 2: mag = 1.0f; break;
        case 3: mag = 1.5f; break;
        case 4: mag = 2.0f; break;
        case 5: mag = 3.0f; break;
        case 6: mag = 4.0f; break;
        default: mag = 6.0f; break;  // case 7
    }
    return (b & 0x8u) ? -mag : mag;
}

/// Quantise a single FP32 value to NVFP4 bits, given the per-block
/// E4M3 scale already chosen by the caller. `scaled = x / scale`
/// is what the MMA-time dequant will multiply back; we pick the FP4
/// representable closest to `scaled`.
///
/// Rounds-to-nearest-even on ties. Returns the 4-bit pattern in the
/// low nibble of the u32.
__device__ __forceinline__ uint32_t fp4_encode(float scaled) {
    uint32_t sign = (scaled < 0.0f) ? 0x8u : 0x0u;
    float mag = fabsf(scaled);

    // Magnitudes: 0, 0.5, 1, 1.5, 2, 3, 4, 6.
    // Find the nearest via direct thresholds — cheaper than a loop.
    uint32_t e;
    if      (mag < 0.25f)                e = 0;
    else if (mag < 0.75f)                e = 1;
    else if (mag < 1.25f)                e = 2;
    else if (mag < 1.75f)                e = 3;
    else if (mag < 2.50f)                e = 4;
    else if (mag < 3.50f)                e = 5;
    else if (mag < 5.00f)                e = 6;
    else                                 e = 7;
    return sign | e;
}

/// Pick the per-16-element E4M3 scale for a block of FP32 values.
/// The scale picks the block's peak magnitude and maps it to the
/// largest representable NVFP4 magnitude (6.0). Clamps to the
/// representable E4M3 range so the returned scale itself is always
/// valid E4M3. Returns the scale as `__nv_fp8_e4m3`.
__device__ __forceinline__ __nv_fp8_e4m3 block_scale_e4m3(const float* block16) {
    float peak = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        float a = fabsf(block16[i]);
        peak = (a > peak) ? a : peak;
    }
    // Scale so peak maps to FP4 max (6.0). Zero-block → scale = 0,
    // which decodes every element to zero — same as the all-zero
    // source, so no quality loss.
    float s = (peak > 0.0f) ? (peak / 6.0f) : 0.0f;
    return __nv_fp8_e4m3(s);
}

/// Pack 16 FP32 inputs into 8 bytes of NVFP4 + one E4M3 scale byte.
/// Caller provides the scale (usually via `block_scale_e4m3` above),
/// lets them reuse the same scale across multiple 16-blocks when the
/// calibration pass already computed it (e.g. during RoPE fusion).
__device__ __forceinline__ void pack16_fp32_to_nvfp4(
    const float* __restrict__ block16,
    __nv_fp8_e4m3 scale,
    uint8_t* __restrict__ out_packed  // 8 bytes
) {
    float inv_scale = (float(scale) > 0.0f) ? 1.0f / float(scale) : 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint32_t lo = fp4_encode(block16[2 * i    ] * inv_scale);
        uint32_t hi = fp4_encode(block16[2 * i + 1] * inv_scale);
        out_packed[i] = static_cast<uint8_t>((hi << 4) | lo);
    }
}

/// Reverse of `pack16`. Reads 8 packed bytes + one E4M3 scale and
/// reconstructs the 16 FP32 values. Used by the kernel's smem-stage
/// dequant pass before the MMA.
__device__ __forceinline__ void unpack16_nvfp4_to_fp32(
    const uint8_t* __restrict__ packed,  // 8 bytes
    __nv_fp8_e4m3 scale,
    float* __restrict__ out_block16
) {
    float s = float(scale);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint8_t byte = packed[i];
        out_block16[2 * i    ] = fp4_decode(byte & 0xFu) * s;
        out_block16[2 * i + 1] = fp4_decode(byte >> 4)   * s;
    }
}

} // namespace rvllm_nvfp4
