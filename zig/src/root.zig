pub const simd_math = @import("simd_math.zig");
pub const weight_convert = @import("weight_convert.zig");

// ---- C ABI: SIMD math -------------------------------------------------------

export fn rvz_softmax(logits: [*]const f32, out: [*]f32, n: usize) void {
    simd_math.softmax(logits[0..n], out[0..n]);
}

export fn rvz_log_softmax(logits: [*]const f32, out: [*]f32, n: usize) void {
    simd_math.logSoftmax(logits[0..n], out[0..n]);
}

export fn rvz_argmax_f32(data: [*]const f32, n: usize) u32 {
    return simd_math.argmax(data[0..n]);
}

export fn rvz_max_f32(data: [*]const f32, n: usize) f32 {
    return simd_math.maxVal(data[0..n]);
}

export fn rvz_temperature_scale(logits: [*]f32, n: usize, inv_temp: f32) void {
    simd_math.scale(logits[0..n], inv_temp);
}

// ---- C ABI: Weight conversion ------------------------------------------------

export fn rvz_bf16_to_f16(src: [*]const u16, dst: [*]u16, n: usize) void {
    weight_convert.bf16ToF16(src[0..n], dst[0..n]);
}

export fn rvz_f32_to_f16(src: [*]const f32, dst: [*]u16, n: usize) void {
    weight_convert.f32ToF16(src[0..n], dst[0..n]);
}

// ---- Tests -------------------------------------------------------------------

const std = @import("std");
const testing = std.testing;

test "softmax sums to 1" {
    const logits = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var out: [4]f32 = undefined;
    simd_math.softmax(&logits, &out);
    var sum: f32 = 0;
    for (out) |v| sum += v;
    try testing.expectApproxEqAbs(sum, 1.0, 1e-5);
}

test "softmax large values stable" {
    const logits = [_]f32{ 1000.0, 1001.0, 1002.0 };
    var out: [3]f32 = undefined;
    simd_math.softmax(&logits, &out);
    var sum: f32 = 0;
    for (out) |v| sum += v;
    try testing.expectApproxEqAbs(sum, 1.0, 1e-5);
    try testing.expect(out[2] > out[1]);
    try testing.expect(out[1] > out[0]);
}

test "argmax finds max" {
    var logits = [_]f32{ 0.0, 0.1, 0.9, 0.3 };
    try testing.expectEqual(simd_math.argmax(&logits), 2);
}

test "argmax large" {
    var logits: [1024]f32 = [_]f32{0.0} ** 1024;
    logits[777] = 99.0;
    try testing.expectEqual(simd_math.argmax(&logits), 777);
}

test "temperature scale" {
    var logits = [_]f32{ 1.0, 2.0, 3.0 };
    simd_math.scale(&logits, 2.0); // inv_temp=2.0 means temp=0.5
    try testing.expectApproxEqAbs(logits[0], 2.0, 1e-6);
    try testing.expectApproxEqAbs(logits[1], 4.0, 1e-6);
    try testing.expectApproxEqAbs(logits[2], 6.0, 1e-6);
}

test "bf16 to f16 round-trip" {
    // BF16 1.0 = 0x3F80
    const src = [_]u16{ 0x3F80, 0x4000 }; // 1.0, 2.0 in bf16
    var dst: [2]u16 = undefined;
    weight_convert.bf16ToF16(&src, &dst);
    // F16 1.0 = 0x3C00, F16 2.0 = 0x4000
    try testing.expectEqual(dst[0], 0x3C00);
    try testing.expectEqual(dst[1], 0x4000);
}

test "f32 to f16" {
    const src = [_]f32{ 1.0, 2.0 };
    var dst: [2]u16 = undefined;
    weight_convert.f32ToF16(&src, &dst);
    try testing.expectEqual(dst[0], 0x3C00);
    try testing.expectEqual(dst[1], 0x4000);
}

test "log_softmax matches softmax" {
    const logits = [_]f32{ 1.0, 2.0, 3.0 };
    var sm: [3]f32 = undefined;
    var lsm: [3]f32 = undefined;
    simd_math.softmax(&logits, &sm);
    simd_math.logSoftmax(&logits, &lsm);
    for (0..3) |i| {
        try testing.expectApproxEqAbs(@exp(lsm[i]), sm[i], 1e-5);
    }
}
