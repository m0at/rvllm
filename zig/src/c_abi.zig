//! C ABI surface for the rvllm Zig shared library.
//! Consumed by `tpu/harness/nvfp4_loader.py` via Python ctypes.
//!
//! The symbols exported here are stable and versioned by build; Python side
//! binds via `RVLLM_ZIG_LIB` env var or the default shared-lib install path.

const std = @import("std");

const nvfp4_i8 = @import("nvfp4_to_int8.zig");
const nvfp4_bf = @import("nvfp4_to_bf16.zig");
const bpe = @import("tiktoken_bpe.zig");

// --- error codes -------------------------------------------------------------

const E_OK: i32 = 0;
const E_SHAPE: i32 = 1;
const E_OOM: i32 = 2;
const E_RUNTIME: i32 = 3;
const E_CAPACITY: i32 = 4;

// --- NVFP4 -> int8 -----------------------------------------------------------

/// Dequantize an NVFP4 matrix to row-major int8 with per-row scales.
///
/// packed_ptr     : rows * cols / 2 bytes
/// scales_ptr     : rows * cols / 16 bytes (FP8 E4M3 bits)
/// out_i8_ptr     : rows * cols bytes
/// out_row_scales : rows f32 values
/// n_threads      : 0 = library default (std.Thread.getCpuCount)
export fn rvllm_nvfp4_to_int8(
    packed_ptr: [*]const u8,
    packed_len: usize,
    scales_ptr: [*]const u8,
    scales_len: usize,
    rows: usize,
    cols: usize,
    out_i8_ptr: [*]i8,
    out_row_scales_ptr: [*]f32,
    n_threads: usize,
) callconv(.c) i32 {
    _ = n_threads; // thread pool is internal to the kernel module.
    if (rows == 0 or cols == 0) return E_SHAPE;
    if (cols % 16 != 0) return E_SHAPE;
    if (packed_len != rows * cols / 2) return E_SHAPE;
    if (scales_len != rows * cols / 16) return E_SHAPE;

    const packed_slice = packed_ptr[0..packed_len];
    const scales_slice = scales_ptr[0..scales_len];
    const out_slice = out_i8_ptr[0 .. rows * cols];
    const row_scales = out_row_scales_ptr[0..rows];

    nvfp4_i8.nvfp4ToInt8Matrix(
        packed_slice,
        scales_slice,
        rows,
        cols,
        out_slice,
        row_scales,
    );
    return E_OK;
}

// --- NVFP4 -> bf16 -----------------------------------------------------------

/// Dequantize an NVFP4 matrix to row-major bf16 (raw u16 bits).
///
/// out_bf16_ptr : rows * cols u16 values (bf16 bits)
export fn rvllm_nvfp4_to_bf16(
    packed_ptr: [*]const u8,
    packed_len: usize,
    scales_ptr: [*]const u8,
    scales_len: usize,
    rows: usize,
    cols: usize,
    out_bf16_ptr: [*]u16,
    n_threads: usize,
) callconv(.c) i32 {
    _ = n_threads;
    if (rows == 0 or cols == 0) return E_SHAPE;
    if (cols % 16 != 0) return E_SHAPE;
    if (packed_len != rows * cols / 2) return E_SHAPE;
    if (scales_len != rows * cols / 16) return E_SHAPE;

    const packed_slice = packed_ptr[0..packed_len];
    const scales_slice = scales_ptr[0..scales_len];
    const out_slice = out_bf16_ptr[0 .. rows * cols];

    nvfp4_bf.nvfp4ToBf16Matrix(
        packed_slice,
        scales_slice,
        rows,
        cols,
        out_slice,
    );
    return E_OK;
}

// --- BPE tokenizer -----------------------------------------------------------

const BpeHandle = extern struct {
    // Opaque on the Python side. Internally a pointer to the Zig Bpe struct.
    inner: *bpe.Bpe,
};

/// Construct a BPE tokenizer from a `tokenizer.json` path.
/// Returns pointer via `out_handle`. On failure returns non-zero and *out_handle is null.
export fn rvllm_bpe_init(
    path_ptr: [*]const u8,
    path_len: usize,
    out_handle: **anyopaque,
) callconv(.c) i32 {
    const allocator = std.heap.c_allocator;
    const path = path_ptr[0..path_len];

    const inner = allocator.create(bpe.Bpe) catch return E_OOM;
    inner.* = bpe.Bpe.init(allocator, path) catch {
        allocator.destroy(inner);
        return E_RUNTIME;
    };
    out_handle.* = @ptrCast(inner);
    return E_OK;
}

export fn rvllm_bpe_close(handle: *anyopaque) callconv(.c) void {
    const allocator = std.heap.c_allocator;
    const inner: *bpe.Bpe = @ptrCast(@alignCast(handle));
    inner.deinit();
    allocator.destroy(inner);
}

/// Encode `text` into ids; writes up to `out_ids_cap` ids and sets *out_ids_len.
/// Returns E_CAPACITY if the result exceeds the provided buffer.
export fn rvllm_bpe_encode(
    bpe_handle: *anyopaque,
    text_ptr: [*]const u8,
    text_len: usize,
    out_ids_ptr: [*]u32,
    out_ids_cap: usize,
    out_ids_len: *usize,
) callconv(.c) i32 {
    const allocator = std.heap.c_allocator;
    const inner: *bpe.Bpe = @ptrCast(@alignCast(bpe_handle));

    var list = std.ArrayList(u32).init(allocator);
    defer list.deinit();

    inner.encode(text_ptr[0..text_len], &list) catch return E_RUNTIME;

    out_ids_len.* = list.items.len;
    if (list.items.len > out_ids_cap) return E_CAPACITY;

    const dst = out_ids_ptr[0..list.items.len];
    @memcpy(dst, list.items);
    return E_OK;
}

/// Decode ids back into UTF-8 bytes.
export fn rvllm_bpe_decode(
    bpe_handle: *anyopaque,
    ids_ptr: [*]const u32,
    ids_len: usize,
    out_bytes_ptr: [*]u8,
    out_bytes_cap: usize,
    out_bytes_len: *usize,
) callconv(.c) i32 {
    const allocator = std.heap.c_allocator;
    const inner: *bpe.Bpe = @ptrCast(@alignCast(bpe_handle));

    var list = std.ArrayList(u8).init(allocator);
    defer list.deinit();

    inner.decode(ids_ptr[0..ids_len], &list) catch return E_RUNTIME;

    out_bytes_len.* = list.items.len;
    if (list.items.len > out_bytes_cap) return E_CAPACITY;

    const dst = out_bytes_ptr[0..list.items.len];
    @memcpy(dst, list.items);
    return E_OK;
}
