//! NVFP4 (FP4 E2M1 packed 2/byte + FP8 E4M3 group scales) -> int8 per-row
//! SIMD dequant. Used by modelopt-quantized MoE weights.
//!
//! Layout (per HuggingFace modelopt producer):
//!   - Packed weights: little-endian nibbles, low nibble = even index,
//!     high nibble = odd index. Each byte holds 2 FP4 values.
//!   - One FP8 E4M3 scale per group of 16 consecutive FP4 values.
//!   - Effective f32 value = fp4_decode(nibble) * fp8_e4m3_decode(scale).
//!
//! Row scale = max(abs(row)) / 127.0, floored at 1e-8.

const std = @import("std");

pub const GROUP_SIZE: usize = 16;

pub const Nvfp4Block = extern struct {
    packed_vals: [8]u8,
    scale_e4m3: u8,
};

// -- FP4 E2M1 decode table (exact values per spec) ---------------------------

const FP4_TABLE: [16]f32 = .{
    0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
};

inline fn fp4Decode(nibble: u8) f32 {
    return FP4_TABLE[nibble & 0x0F];
}

// -- FP8 E4M3 decode ---------------------------------------------------------
// sign(1) + exp(4, bias=7) + mantissa(3)
// 0x00 = +0.0, 0x80 = -0.0
// exp==0xF AND mantissa==0x7 -> NaN (modelopt / E4M3 "finite" variant uses
// 0x7F / 0xFF as NaN). We treat any exp==0xF, mantissa==0x7 as NaN.
fn fp8E4m3Decode(bits: u8) f32 {
    const sign: u32 = @as(u32, bits >> 7);
    const exp: u8 = (bits >> 3) & 0x0F;
    const mant: u8 = bits & 0x07;

    if (exp == 0 and mant == 0) {
        // +/-0
        return if (sign == 1) -0.0 else 0.0;
    }

    if (exp == 0x0F and mant == 0x07) {
        return std.math.nan(f32);
    }

    var f: f32 = undefined;
    if (exp == 0) {
        // subnormal: value = (-1)^s * 2^(1-7) * (mant/8)
        const m: f32 = @as(f32, @floatFromInt(mant)) / 8.0;
        f = std.math.ldexp(m, -6);
    } else {
        // normal: value = (-1)^s * 2^(exp-7) * (1 + mant/8)
        const m: f32 = 1.0 + @as(f32, @floatFromInt(mant)) / 8.0;
        const e: i32 = @as(i32, @intCast(exp)) - 7;
        f = std.math.ldexp(m, e);
    }
    if (sign == 1) f = -f;
    return f;
}

// -- Row dequant -------------------------------------------------------------

const V: usize = 16; // one FP4 group per vector

/// Dequant one row of NVFP4 weights to int8.
/// packed.len must equal n_cols / 2.
/// scales_e4m3.len must equal n_cols / GROUP_SIZE.
/// out_i8.len must equal n_cols.
/// n_cols must be a multiple of GROUP_SIZE (= 16).
pub fn nvfp4ToInt8Row(
    packed_bytes: []const u8,
    scales_e4m3: []const u8,
    out_i8: []i8,
    out_row_scale: *f32,
) void {
    const n = out_i8.len;
    std.debug.assert(n % GROUP_SIZE == 0);
    std.debug.assert(packed_bytes.len * 2 == n);
    std.debug.assert(scales_e4m3.len * GROUP_SIZE == n);

    // Scratch for the full row in f32. For typical hidden_size (1536, 3072, etc.)
    // this is a few KB on the stack via a bounded buffer, else we fall back to a
    // two-pass design. Use a fixed stack buffer sized for common widths; larger
    // rows are handled in GROUP_SIZE-wise chunks with running max.
    var max_abs: f32 = 0.0;
    const F32Vec = @Vector(V, f32);

    // First pass: decode + track max(abs) per group.
    // We decode each group-of-16 into a local vector, take max(abs), and stash
    // the decoded f32 block in a fixed-size staging area sized per group so we
    // can do the quantize pass with the final scale.
    //
    // We need to re-decode in the second pass OR buffer all values. Buffering
    // n f32 values is the simplest correct path. Cap allocation via a bounded
    // on-stack buffer for small rows; bigger rows use a two-pass decode.
    //
    // Strategy: always two-pass. Pass 1 computes max; pass 2 decodes and
    // quantizes. FP4 decode is cheap; avoids any allocation.

    // Pass 1: scan groups, compute max(abs)
    var g: usize = 0;
    while (g < n / GROUP_SIZE) : (g += 1) {
        const scale = fp8E4m3Decode(scales_e4m3[g]);
        var vec: F32Vec = @splat(0.0);
        const byte_off = g * (GROUP_SIZE / 2);
        inline for (0..GROUP_SIZE / 2) |j| {
            const b = packed_bytes[byte_off + j];
            const lo = fp4Decode(b & 0x0F);
            const hi = fp4Decode(b >> 4);
            vec[2 * j] = lo;
            vec[2 * j + 1] = hi;
        }
        const scaled = vec * @as(F32Vec, @splat(scale));
        const absv = @abs(scaled);
        const gmax = @reduce(.Max, absv);
        if (gmax > max_abs) max_abs = gmax;
    }

    var row_scale = max_abs / 127.0;
    if (row_scale < 1e-8) row_scale = 1e-8;
    out_row_scale.* = row_scale;
    const inv_scale: f32 = 1.0 / row_scale;
    const inv_vec: F32Vec = @splat(inv_scale);

    // Pass 2: decode + quantize
    g = 0;
    while (g < n / GROUP_SIZE) : (g += 1) {
        const scale = fp8E4m3Decode(scales_e4m3[g]);
        var vec: F32Vec = @splat(0.0);
        const byte_off = g * (GROUP_SIZE / 2);
        inline for (0..GROUP_SIZE / 2) |j| {
            const b = packed_bytes[byte_off + j];
            const lo = fp4Decode(b & 0x0F);
            const hi = fp4Decode(b >> 4);
            vec[2 * j] = lo;
            vec[2 * j + 1] = hi;
        }
        const scaled = vec * @as(F32Vec, @splat(scale));
        const qf = scaled * inv_vec;
        // Round to nearest, clamp to i8 range.
        const rounded = @round(qf);
        const clamped_hi = @min(rounded, @as(F32Vec, @splat(127.0)));
        const clamped = @max(clamped_hi, @as(F32Vec, @splat(-128.0)));
        const qi: @Vector(V, i32) = @intFromFloat(clamped);
        const out_off = g * GROUP_SIZE;
        inline for (0..GROUP_SIZE) |k| {
            out_i8[out_off + k] = @intCast(qi[k]);
        }
    }
}

// -- Matrix dequant (row-parallel) -------------------------------------------

/// Dequant a row-major NVFP4 matrix to int8 with one rescale per row.
/// packed is rows * (cols/2) bytes, row-major.
/// scales_e4m3 is rows * (cols/GROUP_SIZE) bytes, row-major.
/// out_i8 is rows * cols bytes, row-major.
/// out_row_scales has length == rows.
pub fn nvfp4ToInt8Matrix(
    packed_bytes: []const u8,
    scales_e4m3: []const u8,
    rows: usize,
    cols: usize,
    out_i8: []i8,
    out_row_scales: []f32,
) void {
    std.debug.assert(cols % GROUP_SIZE == 0);
    std.debug.assert(packed_bytes.len == rows * (cols / 2));
    std.debug.assert(scales_e4m3.len == rows * (cols / GROUP_SIZE));
    std.debug.assert(out_i8.len == rows * cols);
    std.debug.assert(out_row_scales.len == rows);

    const packed_stride = cols / 2;
    const scale_stride = cols / GROUP_SIZE;

    var r: usize = 0;
    while (r < rows) : (r += 1) {
        const p_off = r * packed_stride;
        const s_off = r * scale_stride;
        const o_off = r * cols;
        nvfp4ToInt8Row(
            packed_bytes[p_off .. p_off + packed_stride],
            scales_e4m3[s_off .. s_off + scale_stride],
            out_i8[o_off .. o_off + cols],
            &out_row_scales[r],
        );
    }
}

// -- Tests -------------------------------------------------------------------

test "fp4 decode table correctness" {
    const expected = [_]f32{
        0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    };
    for (0..16) |i| {
        const got = fp4Decode(@intCast(i));
        try std.testing.expectEqual(expected[i], got);
    }
}

test "fp8 e4m3 decode correctness" {
    // +0.0
    try std.testing.expectEqual(@as(f32, 0.0), fp8E4m3Decode(0x00));
    // -0.0: bits equal but sign bit negative
    const neg_zero = fp8E4m3Decode(0x80);
    try std.testing.expect(neg_zero == 0.0);
    try std.testing.expect(std.math.signbit(neg_zero));

    // 1.0 = sign=0, exp=7(bias), mant=0 -> bits = 0 0111 000 = 0x38
    try std.testing.expectEqual(@as(f32, 1.0), fp8E4m3Decode(0x38));
    // 2.0 = exp=8, mant=0 -> 0 1000 000 = 0x40
    try std.testing.expectEqual(@as(f32, 2.0), fp8E4m3Decode(0x40));
    // 0.5 = exp=6, mant=0 -> 0 0110 000 = 0x30
    try std.testing.expectEqual(@as(f32, 0.5), fp8E4m3Decode(0x30));
    // -1.0 = 1 0111 000 = 0xB8
    try std.testing.expectEqual(@as(f32, -1.0), fp8E4m3Decode(0xB8));
    // 1.5 = exp=7, mant=4 -> 1 + 4/8 = 1.5, 0 0111 100 = 0x3C
    try std.testing.expectEqual(@as(f32, 1.5), fp8E4m3Decode(0x3C));
    // NaN: 0x7F
    try std.testing.expect(std.math.isNan(fp8E4m3Decode(0x7F)));
    try std.testing.expect(std.math.isNan(fp8E4m3Decode(0xFF)));
}

test "roundtrip known blocks" {
    // Build 2 groups of 16 (cols = 32), one row.
    // Group 0: nibbles 0..15 (values {0, .5, 1, 1.5, 2, 3, 4, 6, -0, -.5, -1, -1.5, -2, -3, -4, -6})
    // scale = 1.0 (FP8 E4M3 bits 0x38)
    // Group 1: all nibble 2 (=1.0), scale = 0.5 (FP8 E4M3 bits 0x30)
    const cols: usize = 32;
    var packed_bytes: [16]u8 = undefined;
    // Group 0: pair (lo, hi) = (0,1), (2,3), ... -> bytes: 0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE
    packed_bytes[0] = 0x10;
    packed_bytes[1] = 0x32;
    packed_bytes[2] = 0x54;
    packed_bytes[3] = 0x76;
    packed_bytes[4] = 0x98;
    packed_bytes[5] = 0xBA;
    packed_bytes[6] = 0xDC;
    packed_bytes[7] = 0xFE;
    // Group 1: all nibble 2 -> byte 0x22 repeated
    for (8..16) |i| packed_bytes[i] = 0x22;

    const scales = [_]u8{ 0x38, 0x30 }; // 1.0, 0.5

    var out_i8: [cols]i8 = undefined;
    var row_scale: f32 = 0.0;
    nvfp4ToInt8Row(packed_bytes[0..], scales[0..], out_i8[0..], &row_scale);

    // Group 0 values after scale=1.0: {0,.5,1,1.5,2,3,4,6,-0,-.5,-1,-1.5,-2,-3,-4,-6}
    // Group 1 values after scale=0.5: all 0.5
    // max(abs) = 6.0 -> row_scale = 6/127
    const expected_rs: f32 = 6.0 / 127.0;
    try std.testing.expectApproxEqAbs(expected_rs, row_scale, 1e-6);

    const inv = 1.0 / expected_rs;
    // Check a few key quantized values.
    // value 6.0 -> round(6 * 127 / 6) = 127
    try std.testing.expectEqual(@as(i8, 127), out_i8[7]);
    // value -6.0 -> -127 (round) -- not -128 since we use symmetric scale
    try std.testing.expectEqual(@as(i8, -127), out_i8[15]);
    // value 0.0 -> 0
    try std.testing.expectEqual(@as(i8, 0), out_i8[0]);
    // value 0.5 in group 1 -> round(0.5 * inv) where inv ~= 21.1667
    const expected_q: i32 = @intFromFloat(@round(0.5 * inv));
    try std.testing.expectEqual(@as(i8, @intCast(expected_q)), out_i8[16]);

    // Matrix wrapper: 2 identical rows.
    const rows: usize = 2;
    var mpacked: [32]u8 = undefined;
    @memcpy(mpacked[0..16], packed_bytes[0..]);
    @memcpy(mpacked[16..32], packed_bytes[0..]);
    var mscales: [4]u8 = undefined;
    @memcpy(mscales[0..2], scales[0..]);
    @memcpy(mscales[2..4], scales[0..]);
    var m_out: [64]i8 = undefined;
    var m_rs: [2]f32 = undefined;
    nvfp4ToInt8Matrix(mpacked[0..], mscales[0..], rows, cols, m_out[0..], m_rs[0..]);
    try std.testing.expectApproxEqAbs(expected_rs, m_rs[0], 1e-6);
    try std.testing.expectApproxEqAbs(expected_rs, m_rs[1], 1e-6);
    for (0..cols) |i| {
        try std.testing.expectEqual(out_i8[i], m_out[i]);
        try std.testing.expectEqual(out_i8[i], m_out[cols + i]);
    }
}

test "row scale floor at 1e-8 for all-zero input" {
    // All nibbles zero, any scale -> values all 0 -> row_scale should clamp to 1e-8.
    const cols: usize = 16;
    const packed_bytes = [_]u8{0} ** 8;
    const scales = [_]u8{0x38}; // scale = 1.0
    var out_i8: [cols]i8 = undefined;
    var rs: f32 = 0.0;
    nvfp4ToInt8Row(packed_bytes[0..], scales[0..], out_i8[0..], &rs);
    try std.testing.expectEqual(@as(f32, 1e-8), rs);
    for (out_i8) |v| try std.testing.expectEqual(@as(i8, 0), v);
}
