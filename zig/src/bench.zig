const std = @import("std");
const simd_math = @import("simd_math.zig");
const weight_convert = @import("weight_convert.zig");

const VOCAB: usize = 128_000;
const ITERS: usize = 10_000;
const WEIGHT_N: usize = 4096 * 4096;

const print = std.debug.print;

pub fn main() !void {
    print("rvllm-zig benchmark (vocab={d}, iters={d})\n\n", .{ VOCAB, ITERS });

    const alloc = std.heap.page_allocator;
    const logits = try alloc.alloc(f32, VOCAB);
    defer alloc.free(logits);
    const out = try alloc.alloc(f32, VOCAB);
    defer alloc.free(out);
    const bf16_src = try alloc.alloc(u16, WEIGHT_N);
    defer alloc.free(bf16_src);
    const f16_dst = try alloc.alloc(u16, WEIGHT_N);
    defer alloc.free(f16_dst);
    const f32_src = try alloc.alloc(f32, WEIGHT_N);
    defer alloc.free(f32_src);

    // Fill with data
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();
    for (logits) |*v| v.* = rng.float(f32) * 20.0 - 10.0;
    for (bf16_src) |*v| v.* = rng.int(u16);
    for (f32_src) |*v| v.* = rng.float(f32) * 2.0 - 1.0;

    // softmax
    {
        var timer = try std.time.Timer.start();
        for (0..ITERS) |_| {
            simd_math.softmax(logits, out);
            std.mem.doNotOptimizeAway(&out[0]);
        }
        const ns = timer.read();
        const us_per = @as(f64, @floatFromInt(ns)) / @as(f64, ITERS) / 1000.0;
        print("softmax({d}):     {d:.1} us/call\n", .{ VOCAB, us_per });
    }

    // log_softmax
    {
        var timer = try std.time.Timer.start();
        for (0..ITERS) |_| {
            simd_math.logSoftmax(logits, out);
            std.mem.doNotOptimizeAway(&out[0]);
        }
        const ns = timer.read();
        const us_per = @as(f64, @floatFromInt(ns)) / @as(f64, ITERS) / 1000.0;
        print("log_softmax({d}): {d:.1} us/call\n", .{ VOCAB, us_per });
    }

    // argmax
    {
        var timer = try std.time.Timer.start();
        for (0..ITERS) |_| {
            const idx = simd_math.argmax(logits);
            std.mem.doNotOptimizeAway(&idx);
        }
        const ns = timer.read();
        const us_per = @as(f64, @floatFromInt(ns)) / @as(f64, ITERS) / 1000.0;
        print("argmax({d}):      {d:.1} us/call\n", .{ VOCAB, us_per });
    }

    // max
    {
        var timer = try std.time.Timer.start();
        for (0..ITERS) |_| {
            const mx = simd_math.maxVal(logits);
            std.mem.doNotOptimizeAway(&mx);
        }
        const ns = timer.read();
        const us_per = @as(f64, @floatFromInt(ns)) / @as(f64, ITERS) / 1000.0;
        print("max({d}):         {d:.1} us/call\n", .{ VOCAB, us_per });
    }

    // scale
    {
        var timer = try std.time.Timer.start();
        for (0..ITERS) |_| {
            simd_math.scale(logits, 0.5);
            std.mem.doNotOptimizeAway(&logits[0]);
        }
        const ns = timer.read();
        const us_per = @as(f64, @floatFromInt(ns)) / @as(f64, ITERS) / 1000.0;
        print("scale({d}):       {d:.1} us/call\n", .{ VOCAB, us_per });
    }

    print("\n", .{});

    // bf16->f16
    {
        const w_iters: usize = 100;
        var timer = try std.time.Timer.start();
        for (0..w_iters) |_| {
            weight_convert.bf16ToF16(bf16_src, f16_dst);
            std.mem.doNotOptimizeAway(&f16_dst[0]);
        }
        const ns = timer.read();
        const ms_per = @as(f64, @floatFromInt(ns)) / @as(f64, w_iters) / 1e6;
        const bytes_per_sec = @as(f64, @floatFromInt(WEIGHT_N * 2)) / (@as(f64, @floatFromInt(ns)) / @as(f64, w_iters));
        const gb_per_s = bytes_per_sec * 1e9 / (1024.0 * 1024.0 * 1024.0);
        print("bf16->f16({d}M): {d:.2} ms ({d:.1} GB/s)\n", .{ WEIGHT_N / 1_000_000, ms_per, gb_per_s });
    }

    // f32->f16
    {
        const w_iters: usize = 100;
        var timer = try std.time.Timer.start();
        for (0..w_iters) |_| {
            weight_convert.f32ToF16(f32_src, f16_dst);
            std.mem.doNotOptimizeAway(&f16_dst[0]);
        }
        const ns = timer.read();
        const ms_per = @as(f64, @floatFromInt(ns)) / @as(f64, w_iters) / 1e6;
        const bytes_per_sec = @as(f64, @floatFromInt(WEIGHT_N * 4)) / (@as(f64, @floatFromInt(ns)) / @as(f64, w_iters));
        const gb_per_s = bytes_per_sec * 1e9 / (1024.0 * 1024.0 * 1024.0);
        print("f32->f16({d}M):  {d:.2} ms ({d:.1} GB/s)\n", .{ WEIGHT_N / 1_000_000, ms_per, gb_per_s });
    }
}
