//! rvllm_dequant — end-to-end NVFP4 -> int8|bf16 dequantizer.
//!
//! Usage:
//!   rvllm_dequant --in  <nvfp4_model_dir>
//!                 --out <dequantized_model_dir>
//!                 --dtype {int8|bf16}
//!                 --threads <N>
//!
//! Reads every `*.safetensors` shard in --in, dequantizes NVFP4 tensors
//! (paired `.weight` packed + `.weight_scale` FP8 E4M3), copies non-NVFP4
//! tensors through unchanged, writes out shards with updated headers.

const std = @import("std");

const nvfp4_i8 = @import("nvfp4_to_int8.zig");
const nvfp4_bf = @import("nvfp4_to_bf16.zig");

const OutDtype = enum { int8, bf16 };

const Args = struct {
    in_dir: []const u8,
    out_dir: []const u8,
    dtype: OutDtype,
    threads: usize,
};

const USAGE =
    \\rvllm_dequant — NVFP4 safetensors dequantizer
    \\
    \\  --in PATH        input model directory (containing *.safetensors)
    \\  --out PATH       output directory (will be created)
    \\  --dtype DTYPE    one of: int8, bf16
    \\  --threads N      worker threads (default = logical CPU count)
    \\  -h, --help       show this message
    \\
;

fn stderrPrint(comptime fmt: []const u8, args: anytype) void {
    var buf: [4096]u8 = undefined;
    const s = std.fmt.bufPrint(&buf, fmt, args) catch return;
    _ = std.fs.File.stderr().writeAll(s) catch {};
}

fn parseArgs(allocator: std.mem.Allocator) !Args {
    var it = try std.process.argsWithAllocator(allocator);
    defer it.deinit();

    _ = it.next(); // program name

    var in_dir: ?[]const u8 = null;
    var out_dir: ?[]const u8 = null;
    var dtype: ?OutDtype = null;
    var threads: usize = 0;

    while (it.next()) |raw| {
        if (std.mem.eql(u8, raw, "-h") or std.mem.eql(u8, raw, "--help")) {
            _ = std.fs.File.stdout().writeAll(USAGE) catch {};
            std.process.exit(0);
        } else if (std.mem.eql(u8, raw, "--in")) {
            const v = it.next() orelse return error.MissingValue;
            in_dir = try allocator.dupe(u8, v);
        } else if (std.mem.eql(u8, raw, "--out")) {
            const v = it.next() orelse return error.MissingValue;
            out_dir = try allocator.dupe(u8, v);
        } else if (std.mem.eql(u8, raw, "--dtype")) {
            const v = it.next() orelse return error.MissingValue;
            if (std.mem.eql(u8, v, "int8")) {
                dtype = .int8;
            } else if (std.mem.eql(u8, v, "bf16")) {
                dtype = .bf16;
            } else return error.InvalidDtype;
        } else if (std.mem.eql(u8, raw, "--threads")) {
            const v = it.next() orelse return error.MissingValue;
            threads = try std.fmt.parseInt(usize, v, 10);
        } else {
            return error.UnknownArg;
        }
    }

    if (in_dir == null or out_dir == null or dtype == null) {
        return error.MissingRequired;
    }
    if (threads == 0) threads = std.Thread.getCpuCount() catch 1;

    return Args{
        .in_dir = in_dir.?,
        .out_dir = out_dir.?,
        .dtype = dtype.?,
        .threads = threads,
    };
}

// --- safetensors header parser ----------------------------------------------

pub const TensorEntry = struct {
    name: []const u8,
    dtype: []const u8, // e.g. "BF16", "F32", "U8", "F8_E4M3", "I8"
    shape: []const u64,
    data_offsets: [2]u64,
};

pub const SafetensorsHeader = struct {
    arena: std.heap.ArenaAllocator,
    tensors: []TensorEntry,
    header_bytes: u64,
    file_len: u64,

    pub fn deinit(self: *SafetensorsHeader) void {
        self.arena.deinit();
    }
};

pub fn parseSafetensorsHeader(
    allocator: std.mem.Allocator,
    path: []const u8,
) !SafetensorsHeader {
    var file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    const stat = try file.stat();
    const file_len = stat.size;
    if (file_len < 8) return error.TruncatedFile;

    var len_buf: [8]u8 = undefined;
    _ = try file.readAll(&len_buf);
    const json_len = std.mem.readInt(u64, &len_buf, .little);
    if (json_len > 1 << 30) return error.HeaderTooLarge;
    if (8 + json_len > file_len) return error.TruncatedHeader;

    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const a = arena.allocator();

    const json_buf = try a.alloc(u8, json_len);
    _ = try file.readAll(json_buf);

    const parsed = try std.json.parseFromSliceLeaky(
        std.json.Value,
        a,
        json_buf,
        .{ .ignore_unknown_fields = true },
    );
    if (parsed != .object) return error.BadHeader;
    const obj = parsed.object;

    var entries: std.ArrayList(TensorEntry) = .empty;

    var it = obj.iterator();
    while (it.next()) |kv| {
        const name = kv.key_ptr.*;
        if (std.mem.eql(u8, name, "__metadata__")) continue;

        if (kv.value_ptr.* != .object) continue;
        const body = kv.value_ptr.object;

        const dtype_v = body.get("dtype") orelse return error.BadHeader;
        const shape_v = body.get("shape") orelse return error.BadHeader;
        const offs_v = body.get("data_offsets") orelse return error.BadHeader;

        if (dtype_v != .string) return error.BadHeader;
        const dtype_str = try a.dupe(u8, dtype_v.string);

        if (shape_v != .array) return error.BadHeader;
        var shape_out = try a.alloc(u64, shape_v.array.items.len);
        for (shape_v.array.items, 0..) |v, i| {
            if (v != .integer) return error.BadHeader;
            shape_out[i] = @intCast(v.integer);
        }

        if (offs_v != .array or offs_v.array.items.len != 2) return error.BadHeader;
        if (offs_v.array.items[0] != .integer or offs_v.array.items[1] != .integer) return error.BadHeader;
        const off_a: u64 = @intCast(offs_v.array.items[0].integer);
        const off_b: u64 = @intCast(offs_v.array.items[1].integer);

        try entries.append(a, .{
            .name = try a.dupe(u8, name),
            .dtype = dtype_str,
            .shape = shape_out,
            .data_offsets = .{ off_a, off_b },
        });
    }

    return .{
        .arena = arena,
        .tensors = try entries.toOwnedSlice(a),
        .header_bytes = 8 + json_len,
        .file_len = file_len,
    };
}

// --- shard processing --------------------------------------------------------

fn isNvfp4Packed(entry: TensorEntry) bool {
    if (!std.mem.eql(u8, entry.dtype, "U8")) return false;
    return std.mem.endsWith(u8, entry.name, ".weight") or
        std.mem.endsWith(u8, entry.name, ".weight_packed");
}

fn findScaleTensor(
    entries: []const TensorEntry,
    base: []const u8,
) ?*const TensorEntry {
    var buf: [512]u8 = undefined;
    const key = std.fmt.bufPrint(&buf, "{s}_scale", .{base}) catch return null;
    for (entries) |*e| {
        if (std.mem.eql(u8, e.name, key)) return e;
    }
    return null;
}

const ProgressCounter = struct {
    bytes_done: std.atomic.Value(u64),
    last_logged: std.atomic.Value(u64),

    pub fn init() ProgressCounter {
        return .{
            .bytes_done = std.atomic.Value(u64).init(0),
            .last_logged = std.atomic.Value(u64).init(0),
        };
    }

    pub fn add(self: *ProgressCounter, delta: u64) void {
        const new_total = self.bytes_done.fetchAdd(delta, .monotonic) + delta;
        const prev = self.last_logged.load(.monotonic);
        if (new_total >= prev + (1 << 30)) {
            if (self.last_logged.cmpxchgStrong(prev, new_total, .monotonic, .monotonic) == null) {
                stderrPrint("[rvllm_dequant] processed {d} GiB\n", .{new_total >> 30});
            }
        }
    }
};

const OutBuild = struct {
    allocator: std.mem.Allocator,
    entries: std.ArrayList(TensorEntry) = .empty,
    data: std.ArrayList(u8) = .empty,
};

fn processShard(
    allocator: std.mem.Allocator,
    in_path: []const u8,
    out_path: []const u8,
    dtype: OutDtype,
    progress: *ProgressCounter,
) !void {
    var hdr = try parseSafetensorsHeader(allocator, in_path);
    defer hdr.deinit();

    var in_file = try std.fs.cwd().openFile(in_path, .{});
    defer in_file.close();
    const data_origin = hdr.header_bytes;

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var ob: OutBuild = .{ .allocator = a };

    // Names of scale tensors that will be consumed by their paired packed tensor.
    var consumed = std.StringHashMap(void).init(a);
    for (hdr.tensors) |entry| {
        if (!isNvfp4Packed(entry)) continue;
        const scale = findScaleTensor(hdr.tensors, entry.name) orelse continue;
        try consumed.put(scale.name, {});
    }

    for (hdr.tensors) |entry| {
        if (consumed.contains(entry.name)) continue;

        if (isNvfp4Packed(entry)) {
            const scale_opt = findScaleTensor(hdr.tensors, entry.name);
            if (scale_opt == null) {
                try copyThrough(&in_file, data_origin, entry, &ob);
                continue;
            }
            try dequantTensor(
                &in_file,
                data_origin,
                entry,
                scale_opt.?.*,
                dtype,
                &ob,
                progress,
            );
        } else {
            try copyThrough(&in_file, data_origin, entry, &ob);
        }
    }

    try writeShard(out_path, ob.entries.items, ob.data.items);
}

fn copyThrough(
    in_file: *std.fs.File,
    data_origin: u64,
    entry: TensorEntry,
    ob: *OutBuild,
) !void {
    const len = entry.data_offsets[1] - entry.data_offsets[0];
    const start = ob.data.items.len;
    try ob.data.resize(ob.allocator, start + len);
    try in_file.seekTo(data_origin + entry.data_offsets[0]);
    _ = try in_file.readAll(ob.data.items[start .. start + len]);

    try ob.entries.append(ob.allocator, .{
        .name = entry.name,
        .dtype = entry.dtype,
        .shape = entry.shape,
        .data_offsets = .{ start, start + len },
    });
}

fn dequantTensor(
    in_file: *std.fs.File,
    data_origin: u64,
    packed_entry: TensorEntry,
    scale_entry: TensorEntry,
    dtype: OutDtype,
    ob: *OutBuild,
    progress: *ProgressCounter,
) !void {
    if (packed_entry.shape.len < 2) return error.BadShape;
    const rows: usize = @intCast(packed_entry.shape[0]);
    const packed_cols: usize = @intCast(packed_entry.shape[packed_entry.shape.len - 1]);
    const logical_cols = packed_cols * 2;
    if (logical_cols % 16 != 0) return error.BadShape;

    const packed_len = packed_entry.data_offsets[1] - packed_entry.data_offsets[0];
    const scale_len = scale_entry.data_offsets[1] - scale_entry.data_offsets[0];
    if (packed_len != rows * packed_cols) return error.BadShape;
    if (scale_len != rows * (logical_cols / 16)) return error.BadShape;

    const packed_buf = try ob.allocator.alloc(u8, packed_len);
    try in_file.seekTo(data_origin + packed_entry.data_offsets[0]);
    _ = try in_file.readAll(packed_buf);

    const scale_buf = try ob.allocator.alloc(u8, scale_len);
    try in_file.seekTo(data_origin + scale_entry.data_offsets[0]);
    _ = try in_file.readAll(scale_buf);

    const n_elems: usize = rows * logical_cols;

    const start = ob.data.items.len;
    switch (dtype) {
        .int8 => {
            try ob.data.resize(ob.allocator, start + n_elems);
            const row_scales = try ob.allocator.alloc(f32, rows);
            nvfp4_i8.nvfp4ToInt8Matrix(
                packed_buf,
                scale_buf,
                rows,
                logical_cols,
                @ptrCast(ob.data.items[start .. start + n_elems]),
                row_scales,
            );
            const scale_off = ob.data.items.len;
            const scale_bytes = row_scales.len * @sizeOf(f32);
            try ob.data.resize(ob.allocator, scale_off + scale_bytes);
            @memcpy(ob.data.items[scale_off .. scale_off + scale_bytes], std.mem.sliceAsBytes(row_scales));

            try ob.entries.append(ob.allocator, .{
                .name = packed_entry.name,
                .dtype = "I8",
                .shape = packed_entry.shape,
                .data_offsets = .{ start, start + n_elems },
            });
            var buf: [512]u8 = undefined;
            const rowscale_name = try std.fmt.bufPrint(&buf, "{s}_rowscale", .{packed_entry.name});
            const rowscale_owned = try ob.allocator.dupe(u8, rowscale_name);
            const shape_owned = try ob.allocator.dupe(u64, &[_]u64{@as(u64, rows)});
            try ob.entries.append(ob.allocator, .{
                .name = rowscale_owned,
                .dtype = "F32",
                .shape = shape_owned,
                .data_offsets = .{ scale_off, scale_off + scale_bytes },
            });
        },
        .bf16 => {
            const bytes = n_elems * 2;
            try ob.data.resize(ob.allocator, start + bytes);
            const out_u16: [*]u16 = @ptrCast(@alignCast(ob.data.items[start..].ptr));
            nvfp4_bf.nvfp4ToBf16Matrix(
                packed_buf,
                scale_buf,
                rows,
                logical_cols,
                out_u16[0..n_elems],
            );
            const shape_owned = try ob.allocator.alloc(u64, packed_entry.shape.len);
            @memcpy(shape_owned, packed_entry.shape);
            shape_owned[shape_owned.len - 1] = @as(u64, logical_cols);
            try ob.entries.append(ob.allocator, .{
                .name = packed_entry.name,
                .dtype = "BF16",
                .shape = shape_owned,
                .data_offsets = .{ start, start + bytes },
            });
        },
    }

    progress.add(packed_len + scale_len);
}

fn writeShard(path: []const u8, entries: []const TensorEntry, data: []const u8) !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var json_buf: std.ArrayList(u8) = .empty;

    try json_buf.append(a, '{');
    for (entries, 0..) |entry, i| {
        if (i > 0) try json_buf.append(a, ',');
        try json_buf.append(a, '"');
        try appendJsonEscaped(&json_buf, a, entry.name);
        try json_buf.appendSlice(a, "\":{\"dtype\":\"");
        try json_buf.appendSlice(a, entry.dtype);
        try json_buf.appendSlice(a, "\",\"shape\":[");
        for (entry.shape, 0..) |dim, j| {
            if (j > 0) try json_buf.append(a, ',');
            var buf: [32]u8 = undefined;
            const s = try std.fmt.bufPrint(&buf, "{d}", .{dim});
            try json_buf.appendSlice(a, s);
        }
        try json_buf.appendSlice(a, "],\"data_offsets\":[");
        var nbuf: [64]u8 = undefined;
        const s1 = try std.fmt.bufPrint(&nbuf, "{d}", .{entry.data_offsets[0]});
        try json_buf.appendSlice(a, s1);
        try json_buf.append(a, ',');
        const s2 = try std.fmt.bufPrint(&nbuf, "{d}", .{entry.data_offsets[1]});
        try json_buf.appendSlice(a, s2);
        try json_buf.appendSlice(a, "]}");
    }
    try json_buf.append(a, '}');

    // Pad header to 8-byte alignment for nicer downstream mmap.
    const pad = (8 - (json_buf.items.len % 8)) % 8;
    var k: usize = 0;
    while (k < pad) : (k += 1) try json_buf.append(a, ' ');

    var out_file = try std.fs.cwd().createFile(path, .{ .truncate = true });
    defer out_file.close();

    var len_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_buf, @as(u64, @intCast(json_buf.items.len)), .little);
    try out_file.writeAll(&len_buf);
    try out_file.writeAll(json_buf.items);
    try out_file.writeAll(data);
}

fn appendJsonEscaped(list: *std.ArrayList(u8), a: std.mem.Allocator, s: []const u8) !void {
    for (s) |c| {
        switch (c) {
            '"' => try list.appendSlice(a, "\\\""),
            '\\' => try list.appendSlice(a, "\\\\"),
            '\n' => try list.appendSlice(a, "\\n"),
            '\r' => try list.appendSlice(a, "\\r"),
            '\t' => try list.appendSlice(a, "\\t"),
            else => try list.append(a, c),
        }
    }
}

// --- main --------------------------------------------------------------------

pub fn main() !void {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = parseArgs(allocator) catch |err| {
        stderrPrint("rvllm_dequant: arg error: {s}\n", .{@errorName(err)});
        _ = std.fs.File.stderr().writeAll(USAGE) catch {};
        std.process.exit(2);
    };

    try std.fs.cwd().makePath(args.out_dir);

    var dir = try std.fs.cwd().openDir(args.in_dir, .{ .iterate = true });
    defer dir.close();

    var progress = ProgressCounter.init();

    var walker = try dir.walk(allocator);
    defer walker.deinit();

    var shards: std.ArrayList([]const u8) = .empty;
    defer {
        for (shards.items) |s| allocator.free(s);
        shards.deinit(allocator);
    }

    while (try walker.next()) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.basename, ".safetensors")) continue;
        try shards.append(allocator, try allocator.dupe(u8, entry.basename));
    }

    stderrPrint("[rvllm_dequant] {d} shard(s), dtype={s}, threads={d}\n", .{
        shards.items.len,
        @tagName(args.dtype),
        args.threads,
    });

    for (shards.items) |name| {
        const in_path = try std.fs.path.join(allocator, &.{ args.in_dir, name });
        defer allocator.free(in_path);
        const out_path = try std.fs.path.join(allocator, &.{ args.out_dir, name });
        defer allocator.free(out_path);

        stderrPrint("[rvllm_dequant] processing {s}\n", .{name});
        processShard(allocator, in_path, out_path, args.dtype, &progress) catch |err| {
            stderrPrint("[rvllm_dequant] {s}: {s}\n", .{ name, @errorName(err) });
            std.process.exit(1);
        };
    }

    stderrPrint("[rvllm_dequant] done\n", .{});
}
