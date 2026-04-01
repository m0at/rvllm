//! SIMD-accelerated math for LLM sampling.
//! Uses @Vector for portable SIMD: NEON on aarch64, AVX-512 on x86_64.

const std = @import("std");
const builtin = @import("builtin");

// Vector width: 16 floats (512-bit on AVX-512, split to 4x128 on NEON).
const V = 16;
const Vec = @Vector(V, f32);
const IdxVec = @Vector(V, u32);

const neg_inf: f32 = -std.math.inf(f32);
const neg_inf_vec: Vec = @splat(neg_inf);

// -- softmax ------------------------------------------------------------------

/// 3-pass numerically stable softmax. Reads `src`, writes `dst`.
pub fn softmax(src: []const f32, dst: []f32) void {
    const n = src.len;
    if (n == 0) return;

    // Pass 1: max
    const mx = maxVal(src);
    const mx_vec: Vec = @splat(mx);

    // Pass 2: exp(x - max) + accumulate sum
    var sum_vec: Vec = @splat(@as(f32, 0));
    var i: usize = 0;
    while (i + V <= n) : (i += V) {
        const v: Vec = src[i..][0..V].*;
        const e = @exp(v - mx_vec);
        dst[i..][0..V].* = e;
        sum_vec += e;
    }
    // Scalar tail
    var sum_tail: f32 = 0;
    while (i < n) : (i += 1) {
        const e = @exp(src[i] - mx);
        dst[i] = e;
        sum_tail += e;
    }
    const total = @reduce(.Add, sum_vec) + sum_tail;

    if (total == 0.0) {
        @memset(dst[0..n], 0.0);
        return;
    }

    // Pass 3: normalize
    const inv: Vec = @splat(1.0 / total);
    i = 0;
    while (i + V <= n) : (i += V) {
        const v: Vec = dst[i..][0..V].*;
        dst[i..][0..V].* = v * inv;
    }
    const inv_s = 1.0 / total;
    while (i < n) : (i += 1) {
        dst[i] *= inv_s;
    }
}

// -- log_softmax --------------------------------------------------------------

/// Log-softmax: log(softmax(x)) = x - max - log(sum(exp(x - max)))
pub fn logSoftmax(src: []const f32, dst: []f32) void {
    const n = src.len;
    if (n == 0) return;

    const mx = maxVal(src);
    const mx_vec: Vec = @splat(mx);

    // Accumulate sum(exp(x - max))
    var sum_vec: Vec = @splat(@as(f32, 0));
    var i: usize = 0;
    while (i + V <= n) : (i += V) {
        const v: Vec = src[i..][0..V].*;
        sum_vec += @exp(v - mx_vec);
    }
    var sum_tail: f32 = 0;
    while (i < n) : (i += 1) {
        sum_tail += @exp(src[i] - mx);
    }
    const lse = mx + @log(@reduce(.Add, sum_vec) + sum_tail);
    const lse_vec: Vec = @splat(lse);

    // x - lse
    i = 0;
    while (i + V <= n) : (i += V) {
        const v: Vec = src[i..][0..V].*;
        dst[i..][0..V].* = v - lse_vec;
    }
    while (i < n) : (i += 1) {
        dst[i] = src[i] - lse;
    }
}

// -- argmax -------------------------------------------------------------------

/// Returns index of maximum element. Returns 0 for empty slice.
pub fn argmax(data: []const f32) u32 {
    const n = data.len;
    if (n == 0) return 0;

    // SIMD path: track max values and their indices per lane
    var best_vals: Vec = @splat(neg_inf);
    var best_idxs: IdxVec = @splat(0);

    // Base index vector: [0, 1, 2, ..., V-1]
    var base_idx: IdxVec = undefined;
    inline for (0..V) |j| {
        base_idx[j] = @intCast(j);
    }
    const stride: IdxVec = @splat(V);

    var i: usize = 0;
    while (i + V <= n) : (i += V) {
        const v: Vec = data[i..][0..V].*;
        const mask = v > best_vals;
        best_vals = @select(f32, mask, v, best_vals);
        best_idxs = @select(u32, mask, base_idx, best_idxs);
        base_idx += stride;
    }

    // Reduce lanes
    var best_val: f32 = neg_inf;
    var best_idx: u32 = 0;
    inline for (0..V) |j| {
        if (best_vals[j] > best_val) {
            best_val = best_vals[j];
            best_idx = best_idxs[j];
        }
    }

    // Scalar tail
    while (i < n) : (i += 1) {
        if (data[i] > best_val) {
            best_val = data[i];
            best_idx = @intCast(i);
        }
    }
    return best_idx;
}

// -- max ----------------------------------------------------------------------

/// Maximum value in slice. Returns -inf for empty.
pub fn maxVal(data: []const f32) f32 {
    const n = data.len;
    if (n == 0) return neg_inf;

    var acc: Vec = @splat(neg_inf);
    var i: usize = 0;
    while (i + V <= n) : (i += V) {
        const v: Vec = data[i..][0..V].*;
        acc = @max(acc, v);
    }
    var best = @reduce(.Max, acc);
    while (i < n) : (i += 1) {
        if (data[i] > best) best = data[i];
    }
    return best;
}

// -- scale --------------------------------------------------------------------

/// In-place multiply by scalar (temperature scaling: inv_temp = 1/T).
pub fn scale(data: []f32, factor: f32) void {
    const n = data.len;
    const fv: Vec = @splat(factor);
    var i: usize = 0;
    while (i + V <= n) : (i += V) {
        const v: Vec = data[i..][0..V].*;
        data[i..][0..V].* = v * fv;
    }
    while (i < n) : (i += 1) {
        data[i] *= factor;
    }
}
