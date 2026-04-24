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
///
/// Scalar switch-decode path — keeps working on pre-Blackwell archs
/// where `cvt.rn.f16x2.e2m1x2` is not available. The fast path below
/// (`unpack16_nvfp4_to_fp32_fast`) replaces the switch with four
/// hardware `cvt` instructions and is ~6× faster on sm_120+, used
/// from the Fa2 decode/prefill kernels.
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

/// Vectorised E2M1 → FP32 dequant for one 16-element block, using
/// the Blackwell `cvt.rn.f16x2.e2m1x2` PTX instruction (single-cycle
/// hardware conversion from two packed E2M1 nibbles to two FP16
/// values). 4 `cvt` ops cover the full 16-element block.
///
/// Prefer this variant inside the Fa2 hot path — measured ~6× faster
/// than the scalar switch-decode on sm_120/121. Falls back to the
/// switch-decode on pre-Blackwell archs where the cvt isn't encoded.
///
/// Inputs:
///   `packed_u64` — 8 packed bytes interpreted as one u64, little
///       endian (byte 0 = nibbles for elements 0,1 ; byte 7 = 14,15).
///       Callers loading from smem / gmem should use
///       `reinterpret_cast<const uint64_t*>(p)[0]` to get this shape.
///   `scale_f32` — pre-converted E4M3 scale as float, broadcast
///       across all 16 outputs. Loaded once by the caller per block.
__device__ __forceinline__ void unpack16_nvfp4_to_fp32_fast(
    uint64_t packed_u64,
    float    scale_f32,
    float*   __restrict__ out_block16
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    // Split the u64 into two u32 halves; each halves carries 4 bytes
    // = 8 packed E2M1 elements.
    uint32_t lo32 = static_cast<uint32_t>(packed_u64 & 0xFFFFFFFFull);
    uint32_t hi32 = static_cast<uint32_t>(packed_u64 >> 32);

    uint32_t h01, h23, h45, h67;   // 4 × f16x2, cover the low 8 elements
    uint32_t h89, hAB, hCD, hEF;   // 4 × f16x2, cover the high 8 elements

    asm volatile(
        "{\n"
        ".reg .b8 byte0, byte1, byte2, byte3;\n"
        "mov.b32 {byte0, byte1, byte2, byte3}, %4;\n"
        "cvt.rn.f16x2.e2m1x2 %0, byte0;\n"
        "cvt.rn.f16x2.e2m1x2 %1, byte1;\n"
        "cvt.rn.f16x2.e2m1x2 %2, byte2;\n"
        "cvt.rn.f16x2.e2m1x2 %3, byte3;\n"
        "}"
        : "=r"(h01), "=r"(h23), "=r"(h45), "=r"(h67)
        : "r"(lo32)
    );
    asm volatile(
        "{\n"
        ".reg .b8 byte0, byte1, byte2, byte3;\n"
        "mov.b32 {byte0, byte1, byte2, byte3}, %4;\n"
        "cvt.rn.f16x2.e2m1x2 %0, byte0;\n"
        "cvt.rn.f16x2.e2m1x2 %1, byte1;\n"
        "cvt.rn.f16x2.e2m1x2 %2, byte2;\n"
        "cvt.rn.f16x2.e2m1x2 %3, byte3;\n"
        "}"
        : "=r"(h89), "=r"(hAB), "=r"(hCD), "=r"(hEF)
        : "r"(hi32)
    );

    // Extend f16x2 → 2 × f32, multiply by broadcast scale.
    // `__half22float2` is a 2-cycle hw instruction on Blackwell.
    auto emit = [&] (uint32_t h2, int i) {
        float2 f = __half22float2(*reinterpret_cast<__half2 const*>(&h2));
        out_block16[i    ] = f.x * scale_f32;
        out_block16[i + 1] = f.y * scale_f32;
    };
    emit(h01,  0); emit(h23,  2); emit(h45,  4); emit(h67,  6);
    emit(h89,  8); emit(hAB, 10); emit(hCD, 12); emit(hEF, 14);
#else
    // Fallback: scalar switch-decode.
    uint8_t bytes[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        bytes[i] = static_cast<uint8_t>(packed_u64 >> (8 * i));
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        out_block16[2 * i    ] = fp4_decode(bytes[i] & 0xFu)      * scale_f32;
        out_block16[2 * i + 1] = fp4_decode((bytes[i] >> 4) & 0xFu) * scale_f32;
    }
#endif
}

/// f16-output variant of `unpack16_nvfp4_to_fp32_fast`. Produces the
/// 16 scaled values directly as f16 in shared memory — avoids the
/// f16 → f32 expansion the _fp32 variant does at the end, so smem
/// traffic halves and the output is ready for f16 MMA operands.
///
/// Scale multiplication stays in f32 for precision, then casts back
/// to f16 per element. On Blackwell the `hmul2` + `cvt` path is cheap
/// enough that keeping scale as f32 is the right tradeoff.
///
/// Used by the Fa2 NVFP4 decode / prefill kernels after the f16-MMA
/// port (task aa01001nvf4f16mma). Output `out_block16` is a f16
/// array with 16 contiguous elements.
__device__ __forceinline__ void unpack16_nvfp4_to_f16_fast(
    uint64_t packed_u64,
    float    scale_f32,
    __half*  __restrict__ out_block16
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    uint32_t lo32 = static_cast<uint32_t>(packed_u64 & 0xFFFFFFFFull);
    uint32_t hi32 = static_cast<uint32_t>(packed_u64 >> 32);

    uint32_t h01, h23, h45, h67;   // low 8 elems as 4 × f16x2
    uint32_t h89, hAB, hCD, hEF;

    asm volatile(
        "{\n"
        ".reg .b8 byte0, byte1, byte2, byte3;\n"
        "mov.b32 {byte0, byte1, byte2, byte3}, %4;\n"
        "cvt.rn.f16x2.e2m1x2 %0, byte0;\n"
        "cvt.rn.f16x2.e2m1x2 %1, byte1;\n"
        "cvt.rn.f16x2.e2m1x2 %2, byte2;\n"
        "cvt.rn.f16x2.e2m1x2 %3, byte3;\n"
        "}"
        : "=r"(h01), "=r"(h23), "=r"(h45), "=r"(h67)
        : "r"(lo32)
    );
    asm volatile(
        "{\n"
        ".reg .b8 byte0, byte1, byte2, byte3;\n"
        "mov.b32 {byte0, byte1, byte2, byte3}, %4;\n"
        "cvt.rn.f16x2.e2m1x2 %0, byte0;\n"
        "cvt.rn.f16x2.e2m1x2 %1, byte1;\n"
        "cvt.rn.f16x2.e2m1x2 %2, byte2;\n"
        "cvt.rn.f16x2.e2m1x2 %3, byte3;\n"
        "}"
        : "=r"(h89), "=r"(hAB), "=r"(hCD), "=r"(hEF)
        : "r"(hi32)
    );

    // Each f16x2 packs (val0, val1). Expand to f32, scale, cast to
    // f16 pair, store as one u32.
    auto emit = [&] (uint32_t h2, int i) {
        float2 f = __half22float2(*reinterpret_cast<__half2 const*>(&h2));
        __half2 out = __floats2half2_rn(f.x * scale_f32, f.y * scale_f32);
        *reinterpret_cast<__half2*>(out_block16 + i) = out;
    };
    emit(h01,  0); emit(h23,  2); emit(h45,  4); emit(h67,  6);
    emit(h89,  8); emit(hAB, 10); emit(hCD, 12); emit(hEF, 14);
#else
    // Scalar fallback for pre-Blackwell archs.
    uint8_t bytes[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        bytes[i] = static_cast<uint8_t>(packed_u64 >> (8 * i));
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        out_block16[2 * i    ] = __float2half(fp4_decode(bytes[i] & 0xFu)      * scale_f32);
        out_block16[2 * i + 1] = __float2half(fp4_decode((bytes[i] >> 4) & 0xFu) * scale_f32);
    }
#endif
}

} // namespace rvllm_nvfp4
