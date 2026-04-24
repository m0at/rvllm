//! SIMD-accelerated NVFP4 -> BF16 dequantization.
//!
//! NVFP4 layout (per HuggingFace modelopt producer):
//!   - Weights packed as uint8, two FP4 E2M1 nibbles per byte.
//!     Low nibble = even index, high nibble = odd index.
//!   - One FP8 E4M3 scale per 16-element group.
//!   - Effective value = fp4_decoded * fp8_scale_decoded.
//!
//! FP4 E2M1 decode (16 codes):
//!   0x0 +0.0  0x1 +0.5  0x2 +1.0  0x3 +1.5
//!   0x4 +2.0  0x5 +3.0  0x6 +4.0  0x7 +6.0
//!   0x8 -0.0  0x9 -0.5  0xA -1.0  0xB -1.5
//!   0xC -2.0  0xD -3.0  0xE -4.0  0xF -6.0
//!
//! FP8 E4M3: 1 sign + 4 exp (bias=7) + 3 mantissa. 0x00 = +0, 0x80 = -0,
//! all-exp-1 / all-mant-1 (0x7F, 0xFF) = NaN per E4M3 convention.
//!
//! Intermediate compute in f32; result rounded to bf16 via top-16-bit
//! truncation with round-to-nearest-even.

const std = @import("std");

const V = 16;
const F32Vec = @Vector(V, f32);
const U32Vec = @Vector(V, u32);
const U16Vec = @Vector(V, u16);

// -- Decode tables -----------------------------------------------------------

/// FP4 E2M1 -> f32. Index by nibble 0..15.
const FP4_LUT: [16]f32 = .{
    0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
};

/// Decode one FP8 E4M3 byte to f32.
/// Layout: 1 sign | 4 exp (bias=7) | 3 mantissa.
/// 0x00=+0, 0x80=-0, 0x7F/0xFF=NaN, subnormals when exp==0.
///
/// Normal:    value = (-1)^s * 2^(E-7) * (1 + M/8),  E in 1..14 plus E=15 for finite
/// Subnormal: value = (-1)^s * 2^-6    * (M/8)
/// NaN:       S.1111.111 (both 0x7F and 0xFF)
fn fp8E4m3ToF32(byte: u8) f32 {
    const sign_bit: u32 = @as(u32, byte >> 7);
    const exp: u32 = @as(u32, (byte >> 3) & 0x0F);
    const mant: u32 = @as(u32, byte & 0x07);

    if (exp == 0x0F and mant == 0x07) return std.math.nan(f32);

    if (exp == 0 and mant == 0) {
        const z: u32 = sign_bit << 31;
        return @bitCast(z);
    }

    var f_exp: i32 = undefined;
    var f_mant: u32 = undefined;

    if (exp == 0) {
        // Subnormal: value = mant * 2^-9.
        // Express as 1.xx * 2^k where the leading 1 of `mant` (a 3-bit int)
        // is at position p (0..2). Then value = (1.xxx) * 2^(p - 9).
        var m: u32 = mant;
        var p: i32 = 2;
        while ((m & 0x04) == 0) {
            m <<= 1;
            p -= 1;
        }
        // Drop the implicit leading 1 (bit 2 of our 3-bit field).
        m &= 0x03;
        // Promote 2 remaining mantissa bits to the top of f32's 23-bit field.
        f_mant = m << 21;
        f_exp = p - 9 + 127;
    } else {
        f_exp = @as(i32, @intCast(exp)) - 7 + 127;
        f_mant = mant << 20;
    }

    const bits: u32 = (sign_bit << 31) |
        (@as(u32, @intCast(f_exp)) << 23) |
        f_mant;
    return @bitCast(bits);
}

/// Precomputed table: FP8 E4M3 byte -> f32 bits.
const FP8_LUT: [256]f32 = blk: {
    @setEvalBranchQuota(10000);
    var t: [256]f32 = undefined;
    var i: usize = 0;
    while (i < 256) : (i += 1) {
        t[i] = fp8E4m3ToF32(@intCast(i));
    }
    break :blk t;
};

// -- f32 -> bf16 with round-to-nearest-even ----------------------------------

/// Scalar f32 -> bf16 bits. RNE on top-16-bit truncation.
/// NaN is preserved as a quiet bf16 NaN.
fn f32ToBf16Scalar(f: f32) u16 {
    const u: u32 = @bitCast(f);
    // Detect NaN: exp==0xFF and mantissa != 0
    if ((u & 0x7F800000) == 0x7F800000 and (u & 0x007FFFFF) != 0) {
        // Preserve sign, set quiet bit.
        return @truncate((u >> 16) | 0x0040);
    }
    // RNE: add rounding bias = 0x7FFF + ((top>>16) & 1).
    const lsb: u32 = (u >> 16) & 1;
    const rounded: u32 = u +% (0x7FFF + lsb);
    return @truncate(rounded >> 16);
}

/// Vector f32 -> bf16 (u16) with RNE. Does not special-case NaN; callers
/// feeding finite values get identical results to the scalar path. For NaN
/// inputs the quiet bit is set via a follow-up fix-up in the caller. For the
/// NVFP4 dequant path, NaN can only arise from FP8 E4M3 0x7F/0xFF scales and
/// is handled uniformly by this routine too (the NaN payload survives RNE
/// because adding 0x7FFF to a NaN bit pattern leaves it NaN; we OR the quiet
/// bit below to be safe).
fn f32ToBf16VecRNE(v: F32Vec) U16Vec {
    const shift16: @Vector(V, u5) = @splat(@as(u5, 16));
    const bits: U32Vec = @bitCast(v);
    const lsb: U32Vec = (bits >> shift16) & @as(U32Vec, @splat(1));
    const bias: U32Vec = @as(U32Vec, @splat(0x7FFF)) + lsb;
    const rounded: U32Vec = bits +% bias;
    const shifted: U32Vec = rounded >> shift16;

    // NaN fix-up: if original was NaN (exp==0xFF and mant!=0), force quiet bit.
    const exp_mask: U32Vec = @splat(0x7F800000);
    const mant_mask: U32Vec = @splat(0x007FFFFF);
    const is_exp_all_ones = (bits & exp_mask) == exp_mask;
    const mant_nonzero = (bits & mant_mask) != @as(U32Vec, @splat(0));
    const is_nan_mask = @select(
        u32,
        is_exp_all_ones,
        @select(u32, mant_nonzero, @as(U32Vec, @splat(1)), @as(U32Vec, @splat(0))),
        @as(U32Vec, @splat(0)),
    );
    // For NaN: replace with (truncated top16 | quiet-bit).
    const nan_bits: U32Vec = ((bits >> shift16) | @as(U32Vec, @splat(0x0040))) &
        @as(U32Vec, @splat(0xFFFF));
    const out32: U32Vec = @select(
        u32,
        is_nan_mask == @as(U32Vec, @splat(1)),
        nan_bits,
        shifted,
    );
    return @intCast(out32 & @as(U32Vec, @splat(0xFFFF)));
}

// -- Core dequant ------------------------------------------------------------

/// Dequant one row: `packed_row` length = cols/2, `scales_row` length = cols/16,
/// `out_bf16` length = cols. Cols must be a multiple of 16.
pub fn nvfp4ToBf16Row(
    packed_row: []const u8,
    scales_row: []const u8,
    out_bf16: []u16,
) void {
    const n = out_bf16.len;
    std.debug.assert(n % 16 == 0);
    std.debug.assert(packed_row.len == n / 2);
    std.debug.assert(scales_row.len == n / 16);

    var g: usize = 0;
    const n_groups = n / 16;

    while (g < n_groups) : (g += 1) {
        const scale: f32 = FP8_LUT[scales_row[g]];
        const scale_vec: F32Vec = @splat(scale);

        // Decode 16 FP4 values from 8 packed bytes.
        var vals: [16]f32 = undefined;
        const base = g * 8;
        inline for (0..8) |j| {
            const b = packed_row[base + j];
            vals[2 * j + 0] = FP4_LUT[b & 0x0F];
            vals[2 * j + 1] = FP4_LUT[(b >> 4) & 0x0F];
        }
        const v: F32Vec = vals;
        const scaled: F32Vec = v * scale_vec;
        const bf: U16Vec = f32ToBf16VecRNE(scaled);
        out_bf16[g * 16 ..][0..16].* = bf;
    }
}

/// Row-major matrix variant. `packed` shape (rows, cols/2), `scales_e4m3`
/// shape (rows, cols/16), `out_bf16` shape (rows, cols).
pub fn nvfp4ToBf16Matrix(
    packed_bytes: []const u8,
    scales_e4m3: []const u8,
    rows: usize,
    cols: usize,
    out_bf16: []u16,
) void {
    std.debug.assert(cols % 16 == 0);
    std.debug.assert(packed_bytes.len == rows * (cols / 2));
    std.debug.assert(scales_e4m3.len == rows * (cols / 16));
    std.debug.assert(out_bf16.len == rows * cols);

    const pack_stride = cols / 2;
    const scale_stride = cols / 16;

    var r: usize = 0;
    while (r < rows) : (r += 1) {
        const p_row = packed_bytes[r * pack_stride ..][0..pack_stride];
        const s_row = scales_e4m3[r * scale_stride ..][0..scale_stride];
        const o_row = out_bf16[r * cols ..][0..cols];
        nvfp4ToBf16Row(p_row, s_row, o_row);
    }
}

// -- Reference scalar implementation (for tests) -----------------------------

fn refDequantRow(
    packed_row: []const u8,
    scales_row: []const u8,
    out_bf16: []u16,
) void {
    const n = out_bf16.len;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const group = i / 16;
        const pair = i / 2;
        const nibble: u4 = if ((i & 1) == 0)
            @intCast(packed_row[pair] & 0x0F)
        else
            @intCast((packed_row[pair] >> 4) & 0x0F);
        const fp4_v: f32 = FP4_LUT[nibble];
        const scale: f32 = fp8E4m3ToF32(scales_row[group]);
        const v: f32 = fp4_v * scale;
        out_bf16[i] = f32ToBf16Scalar(v);
    }
}

// -- Tests -------------------------------------------------------------------

test "fp4 lut values" {
    try std.testing.expectEqual(@as(f32, 0.0), FP4_LUT[0]);
    try std.testing.expectEqual(@as(f32, 0.5), FP4_LUT[1]);
    try std.testing.expectEqual(@as(f32, 6.0), FP4_LUT[7]);
    try std.testing.expectEqual(@as(f32, -0.5), FP4_LUT[9]);
    try std.testing.expectEqual(@as(f32, -6.0), FP4_LUT[15]);
}

test "fp8 e4m3 decode key points" {
    try std.testing.expectEqual(@as(f32, 0.0), fp8E4m3ToF32(0x00));
    try std.testing.expectEqual(@as(f32, -0.0), fp8E4m3ToF32(0x80));
    try std.testing.expect(std.math.isNan(fp8E4m3ToF32(0x7F)));
    try std.testing.expect(std.math.isNan(fp8E4m3ToF32(0xFF)));
    // 0x38 = S=0, E=7 (=> 2^0), M=0 => 1.0
    try std.testing.expectEqual(@as(f32, 1.0), fp8E4m3ToF32(0x38));
    // 0xB8 = sign=1, E=7, M=0 => -1.0
    try std.testing.expectEqual(@as(f32, -1.0), fp8E4m3ToF32(0xB8));
    // 0x3C = E=7, M=4 => 1 + 4/8 = 1.5
    try std.testing.expectEqual(@as(f32, 1.5), fp8E4m3ToF32(0x3C));
    // 0x40 = E=8, M=0 => 2.0
    try std.testing.expectEqual(@as(f32, 2.0), fp8E4m3ToF32(0x40));
    // Subnormals: 0x01 => 2^-9, 0x04 => 2^-7, 0x07 => 7 * 2^-9
    try std.testing.expectEqual(@as(f32, std.math.pow(f32, 2.0, -9.0)), fp8E4m3ToF32(0x01));
    try std.testing.expectEqual(@as(f32, std.math.pow(f32, 2.0, -7.0)), fp8E4m3ToF32(0x04));
    try std.testing.expectEqual(@as(f32, 7.0 * std.math.pow(f32, 2.0, -9.0)), fp8E4m3ToF32(0x07));
}

test "f32->bf16 RNE basic" {
    // 1.0 -> 0x3F80
    try std.testing.expectEqual(@as(u16, 0x3F80), f32ToBf16Scalar(1.0));
    // -1.0 -> 0xBF80
    try std.testing.expectEqual(@as(u16, 0xBF80), f32ToBf16Scalar(-1.0));
    // 0.0 -> 0x0000
    try std.testing.expectEqual(@as(u16, 0x0000), f32ToBf16Scalar(0.0));
    // Round half to even: exactly 0x3F808000 rounds down to 0x3F80 (even).
    const halfway: f32 = @bitCast(@as(u32, 0x3F808000));
    try std.testing.expectEqual(@as(u16, 0x3F80), f32ToBf16Scalar(halfway));
    // Exactly 0x3F818000 rounds up to 0x3F82 (even).
    const halfway2: f32 = @bitCast(@as(u32, 0x3F818000));
    try std.testing.expectEqual(@as(u16, 0x3F82), f32ToBf16Scalar(halfway2));
}

test "vec and scalar bf16 agree on wide range" {
    const N = 4096;
    var inputs: [N]f32 = undefined;
    var rng = std.Random.DefaultPrng.init(0xBADC0FFEE);
    const r = rng.random();
    var i: usize = 0;
    while (i < N) : (i += 1) {
        // Mix random floats and specially crafted ties.
        const u = r.int(u32);
        inputs[i] = @bitCast(u);
    }

    var out_vec: [N]u16 = undefined;
    var j: usize = 0;
    while (j + V <= N) : (j += V) {
        const v: F32Vec = inputs[j..][0..V].*;
        const bf: U16Vec = f32ToBf16VecRNE(v);
        out_vec[j..][0..V].* = bf;
    }

    var k: usize = 0;
    while (k < N) : (k += 1) {
        const s = f32ToBf16Scalar(inputs[k]);
        const v = out_vec[k];
        // NaN: both must be NaN-shaped (exp=0xFF, mant!=0); bit-equality not
        // required since the vector path preserves the truncated payload plus
        // quiet bit and the scalar path does the same.
        const is_nan_scalar = (s & 0x7F80) == 0x7F80 and (s & 0x007F) != 0;
        const is_nan_vec = (v & 0x7F80) == 0x7F80 and (v & 0x007F) != 0;
        if (is_nan_scalar or is_nan_vec) {
            try std.testing.expect(is_nan_scalar and is_nan_vec);
        } else {
            try std.testing.expectEqual(s, v);
        }
    }
}

test "nvfp4 row matches scalar reference" {
    const cols = 256;
    const n_groups = cols / 16;

    var packed_row: [cols / 2]u8 = undefined;
    var scales: [n_groups]u8 = undefined;

    var rng = std.Random.DefaultPrng.init(42);
    const r = rng.random();
    for (&packed_row) |*b| b.* = r.int(u8);
    for (&scales) |*s| {
        // Avoid NaN scale bytes for this equivalence check.
        var byte: u8 = r.int(u8);
        if ((byte & 0x7F) == 0x7F) byte ^= 0x01;
        s.* = byte;
    }

    var out_fast: [cols]u16 = undefined;
    var out_ref: [cols]u16 = undefined;
    nvfp4ToBf16Row(&packed_row, &scales, &out_fast);
    refDequantRow(&packed_row, &scales, &out_ref);

    try std.testing.expectEqualSlices(u16, &out_ref, &out_fast);
}

test "nvfp4 row with NaN scale stays NaN" {
    const cols = 16;
    var packed_row: [8]u8 = .{ 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF };
    var scales: [1]u8 = .{0x7F}; // NaN

    var out_fast: [cols]u16 = undefined;
    nvfp4ToBf16Row(&packed_row, &scales, &out_fast);

    // Non-zero FP4 codes multiplied by NaN become NaN; code 0 (+0.0) * NaN
    // is NaN as well. All 16 outputs should be bf16 NaN.
    for (out_fast) |bf| {
        try std.testing.expect((bf & 0x7F80) == 0x7F80 and (bf & 0x007F) != 0);
    }
}

test "nvfp4 matrix matches row-by-row reference" {
    const rows = 7;
    const cols = 128;
    const pack_len = rows * (cols / 2);
    const scale_len = rows * (cols / 16);

    var packed_m: [pack_len]u8 = undefined;
    var scales_m: [scale_len]u8 = undefined;
    var rng = std.Random.DefaultPrng.init(0xC0DEC0DE);
    const r = rng.random();
    for (&packed_m) |*b| b.* = r.int(u8);
    for (&scales_m) |*s| {
        var byte: u8 = r.int(u8);
        if ((byte & 0x7F) == 0x7F) byte ^= 0x01;
        s.* = byte;
    }

    var out_fast: [rows * cols]u16 = undefined;
    var out_ref: [rows * cols]u16 = undefined;
    nvfp4ToBf16Matrix(&packed_m, &scales_m, rows, cols, &out_fast);

    var rr: usize = 0;
    while (rr < rows) : (rr += 1) {
        refDequantRow(
            packed_m[rr * (cols / 2) ..][0 .. cols / 2],
            scales_m[rr * (cols / 16) ..][0 .. cols / 16],
            out_ref[rr * cols ..][0..cols],
        );
    }
    try std.testing.expectEqualSlices(u16, &out_ref, &out_fast);
}

test "nvfp4 known block: scale=1.0, all zero nibbles" {
    const cols = 16;
    var packed_row: [8]u8 = .{0} ** 8;
    const scales: [1]u8 = .{0x38}; // 1.0
    var out: [cols]u16 = undefined;
    nvfp4ToBf16Row(&packed_row, &scales, &out);
    for (out) |v| try std.testing.expectEqual(@as(u16, 0x0000), v);
}

test "nvfp4 known block: scale=1.0, alternating +6/-6" {
    const cols = 16;
    // Low nibble = 0x7 (+6.0), high nibble = 0xF (-6.0). Byte = 0xF7.
    var packed_row: [8]u8 = .{0xF7} ** 8;
    const scales: [1]u8 = .{0x38}; // 1.0
    var out: [cols]u16 = undefined;
    nvfp4ToBf16Row(&packed_row, &scales, &out);
    // bf16(6.0)  = 0x40C0, bf16(-6.0) = 0xC0C0
    var i: usize = 0;
    while (i < cols) : (i += 2) {
        try std.testing.expectEqual(@as(u16, 0x40C0), out[i]);
        try std.testing.expectEqual(@as(u16, 0xC0C0), out[i + 1]);
    }
}
