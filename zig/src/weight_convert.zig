//! SIMD-accelerated weight format conversion for model loading.
//! BF16 -> F16 and F32 -> F16 at full memory bandwidth.

const std = @import("std");

const V = 16;
const F32Vec = @Vector(V, f32);
const U16Vec = @Vector(V, u16);
const U32Vec = @Vector(V, u32);

// -- BF16 -> F16 --------------------------------------------------------------

/// Convert BF16 (as raw u16 bits) to F16 (as raw u16 bits).
/// BF16: 1 sign + 8 exp + 7 mantissa (same exp range as f32)
/// F16:  1 sign + 5 exp + 10 mantissa
/// Route: bf16 bits -> f32 (shift left 16) -> f16
pub fn bf16ToF16(src: []const u16, dst: []u16) void {
    const n = src.len;
    var i: usize = 0;

    while (i + V <= n) : (i += V) {
        const bits: U16Vec = src[i..][0..V].*;
        // BF16 -> F32: zero-extend to u32, shift left 16
        const wide: U32Vec = @intCast(bits);
        const shifted: U32Vec = wide << @splat(@as(u5, 16));
        const f32_vals: F32Vec = @bitCast(shifted);
        // F32 -> F16 via truncation
        const f16_vals: @Vector(V, f16) = @floatCast(f32_vals);
        dst[i..][0..V].* = @bitCast(f16_vals);
    }

    // Scalar tail
    while (i < n) : (i += 1) {
        const wide: u32 = @as(u32, src[i]) << 16;
        const f: f32 = @bitCast(wide);
        const h: f16 = @floatCast(f);
        dst[i] = @bitCast(h);
    }
}

// -- F32 -> F16 ---------------------------------------------------------------

/// Convert F32 to F16 (as raw u16 bits).
pub fn f32ToF16(src: []const f32, dst: []u16) void {
    const n = src.len;
    var i: usize = 0;

    while (i + V <= n) : (i += V) {
        const f32_vals: F32Vec = src[i..][0..V].*;
        const f16_vals: @Vector(V, f16) = @floatCast(f32_vals);
        dst[i..][0..V].* = @bitCast(f16_vals);
    }

    // Scalar tail
    while (i < n) : (i += 1) {
        const h: f16 = @floatCast(src[i]);
        dst[i] = @bitCast(h);
    }
}
