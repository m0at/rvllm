//! tiktoken-style BPE tokenizer loader + encoder for HuggingFace tokenizer.json.
//!
//! Target: MiniMax-M2.7 (200064-entry vocab). The model uses a GPT-4 style
//! tiktoken pre-tokenize regex. We implement that regex as a hand-rolled
//! byte-level state machine. ASCII is handled precisely; multi-byte UTF-8
//! runs are grouped as single alphabetic chunks (treated as \p{L}+).
//!
//! Pre-tokenize pattern we emulate:
//!   (?i:'s|'t|'re|'ve|'m|'ll|'d)
//! | [^\r\n\p{L}\p{N}]? \p{L}+
//! | \p{N}{1,3}
//! |  ?[^\s\p{L}\p{N}]+ [\r\n]*
//! | \s*[\r\n]+
//! | \s+(?!\S)
//! | \s+
//!
//! Scope: Thread-safe encode/decode (const receiver; no internal mutation).
//! Batch encode via std.Thread.Pool.

const std = @import("std");

// -- character classes --------------------------------------------------------

inline fn isAsciiLetter(b: u8) bool {
    return (b >= 'a' and b <= 'z') or (b >= 'A' and b <= 'Z');
}

inline fn isAsciiDigit(b: u8) bool {
    return b >= '0' and b <= '9';
}

inline fn isAsciiWs(b: u8) bool {
    return b == ' ' or b == '\t' or b == '\r' or b == '\n' or b == 0x0B or b == 0x0C;
}

inline fn isNewline(b: u8) bool {
    return b == '\r' or b == '\n';
}

inline fn isUtf8Lead(b: u8) bool {
    return b >= 0x80;
}

/// Length of UTF-8 sequence starting at lead byte. 1 for ASCII, 2..4 for multi.
/// For invalid leads we return 1 to make progress.
inline fn utf8Len(b: u8) u3 {
    if (b < 0x80) return 1;
    if ((b & 0xE0) == 0xC0) return 2;
    if ((b & 0xF0) == 0xE0) return 3;
    if ((b & 0xF8) == 0xF0) return 4;
    return 1;
}

/// A "letter" class for the regex: ASCII letter OR any UTF-8 multibyte start.
/// Minimum-viable treatment per spec: all multi-byte UTF-8 runs are treated
/// as alphabetic (\p{L}+).
inline fn isLetterLike(b: u8) bool {
    return isAsciiLetter(b) or isUtf8Lead(b);
}

/// \p{N} minimal: ASCII digit. (Multi-byte digits fall under letter-like.)
inline fn isNumeric(b: u8) bool {
    return isAsciiDigit(b);
}

/// [^\s\p{L}\p{N}] minimal: ASCII punctuation / symbol, i.e. not whitespace,
/// not letter-like, not digit.
inline fn isPunctLike(b: u8) bool {
    return !(isAsciiWs(b) or isLetterLike(b) or isNumeric(b));
}

// -- pre-tokenize state machine ----------------------------------------------

const Span = struct { start: usize, end: usize };

/// Case-insensitive match for one of the contractions at position i.
/// Returns length (including the leading apostrophe) if match, else 0.
fn matchContraction(text: []const u8, i: usize) usize {
    if (i >= text.len or text[i] != '\'') return 0;
    const rest = text[i + 1 ..];
    const Pair = struct { pat: []const u8, len: usize };
    // Order matters: longest first for overlapping prefixes.
    const pairs = [_]Pair{
        .{ .pat = "ll", .len = 3 },
        .{ .pat = "re", .len = 3 },
        .{ .pat = "ve", .len = 3 },
        .{ .pat = "s", .len = 2 },
        .{ .pat = "t", .len = 2 },
        .{ .pat = "m", .len = 2 },
        .{ .pat = "d", .len = 2 },
    };
    for (pairs) |p| {
        if (rest.len >= p.pat.len) {
            var ok = true;
            for (p.pat, 0..) |c, k| {
                const r = rest[k];
                const rl = if (r >= 'A' and r <= 'Z') r + 32 else r;
                if (rl != c) {
                    ok = false;
                    break;
                }
            }
            if (ok) return p.len;
        }
    }
    return 0;
}

/// Emit one pre-token span starting at i. Returns the new index.
/// Caller appends the span to the output list unless start == end.
fn nextPreToken(text: []const u8, i_in: usize, out: *Span) usize {
    const i = i_in;
    const n = text.len;
    if (i >= n) {
        out.* = .{ .start = i, .end = i };
        return i;
    }

    // 1) contractions: 's 't 're 've 'm 'll 'd (case-insensitive ASCII)
    {
        const cl = matchContraction(text, i);
        if (cl != 0) {
            out.* = .{ .start = i, .end = i + cl };
            return i + cl;
        }
    }

    // 2) [^\r\n\p{L}\p{N}]? \p{L}+
    //    optional single leading non-\r\n, non-letter, non-digit byte,
    //    followed by one or more letter-like runs (UTF-8 multibyte counted here).
    {
        var j = i;
        if (!isNewline(text[j]) and !isLetterLike(text[j]) and !isNumeric(text[j])) {
            // peek next byte to see if a letter follows; only then consume prefix
            const nxt = j + 1;
            if (nxt < n and isLetterLike(text[nxt])) {
                j = nxt;
            }
        }
        if (j < n and isLetterLike(text[j])) {
            // consume one or more letter-like code points
            while (j < n and isLetterLike(text[j])) {
                j += utf8Len(text[j]);
            }
            out.* = .{ .start = i, .end = j };
            return j;
        }
    }

    // 3) \p{N}{1,3}  — ASCII digits, 1-3 at a time
    if (isNumeric(text[i])) {
        var j = i;
        var cnt: usize = 0;
        while (j < n and isNumeric(text[j]) and cnt < 3) : (cnt += 1) j += 1;
        out.* = .{ .start = i, .end = j };
        return j;
    }

    // 4)  ?[^\s\p{L}\p{N}]+ [\r\n]*   — optional leading space then punct run
    //    then any trailing \r\n's.
    if (text[i] == ' ' and i + 1 < n and isPunctLike(text[i + 1])) {
        var j = i + 1;
        while (j < n and isPunctLike(text[j])) j += 1;
        while (j < n and isNewline(text[j])) j += 1;
        out.* = .{ .start = i, .end = j };
        return j;
    }
    if (isPunctLike(text[i])) {
        var j = i;
        while (j < n and isPunctLike(text[j])) j += 1;
        while (j < n and isNewline(text[j])) j += 1;
        out.* = .{ .start = i, .end = j };
        return j;
    }

    // 5) \s*[\r\n]+   — whitespace run ending with newline(s)
    if (isAsciiWs(text[i])) {
        // scan ahead: does this whitespace run contain a newline?
        var j = i;
        while (j < n and isAsciiWs(text[j]) and !isNewline(text[j])) j += 1;
        if (j < n and isNewline(text[j])) {
            // alt 5: \s* then \r\n+
            while (j < n and isNewline(text[j])) j += 1;
            out.* = .{ .start = i, .end = j };
            return j;
        }
        // 6) \s+(?!\S)  — trailing whitespace before end-of-string counts as
        //    "not followed by non-space". Also: \s+ when at EOF.
        // 7) \s+
        j = i;
        while (j < n and isAsciiWs(text[j])) j += 1;
        // If not at EOF and next is non-space, tiktoken emits all-but-last
        // whitespace byte so a following word can take its leading space.
        if (j < n and !isAsciiWs(text[j]) and j - i > 1) {
            j -= 1;
        }
        out.* = .{ .start = i, .end = j };
        return j;
    }

    // Fallback: single byte (shouldn't reach here for well-formed UTF-8 text).
    out.* = .{ .start = i, .end = i + 1 };
    return i + 1;
}

// -- BPE core -----------------------------------------------------------------

pub const TokenId = u32;

/// Map from byte-sequence (the token string as stored in tokenizer.json vocab,
/// still using GPT-2 style byte2unicode — we just use the raw UTF-8 bytes as
/// key since the JSON already contains them in byte2unicode form).
const VocabMap = std.StringHashMap(TokenId);

/// Merges are stored as a map from concatenated "left<space>right" byte string
/// to a rank (lower rank = merge first).
const MergeMap = std.StringHashMap(u32);

/// Reverse: id -> bytes. Owned by `Bpe`.
const IdToBytes = std.ArrayList([]const u8);

pub const Bpe = struct {
    allocator: std.mem.Allocator,
    /// Raw backing store for all vocab keys (owned).
    vocab_arena: std.heap.ArenaAllocator,
    vocab: VocabMap,
    merges: MergeMap,
    id_to_bytes: std.ArrayList([]const u8),

    pub fn deinit(self: *Bpe) void {
        self.vocab.deinit();
        self.merges.deinit();
        self.id_to_bytes.deinit();
        self.vocab_arena.deinit();
    }

    /// Load a tokenizer.json (HuggingFace format). Supported schema subset:
    ///   { "model": { "type": "BPE", "vocab": {..}, "merges": [".."] } }
    pub fn init(
        allocator: std.mem.Allocator,
        tokenizer_json_path: []const u8,
    ) !Bpe {
        const file = try std.fs.cwd().openFile(tokenizer_json_path, .{});
        defer file.close();
        const stat = try file.stat();
        const buf = try allocator.alloc(u8, stat.size);
        defer allocator.free(buf);
        const read = try file.readAll(buf);
        if (read != stat.size) return error.ShortRead;
        return initFromBytes(allocator, buf);
    }

    /// Construct from an in-memory tokenizer.json.
    pub fn initFromBytes(
        allocator: std.mem.Allocator,
        json_bytes: []const u8,
    ) !Bpe {
        var arena = std.heap.ArenaAllocator.init(allocator);
        errdefer arena.deinit();
        const arena_alloc = arena.allocator();

        var parsed = try std.json.parseFromSlice(
            std.json.Value,
            allocator,
            json_bytes,
            .{ .ignore_unknown_fields = true },
        );
        defer parsed.deinit();

        const root = parsed.value;
        if (root != .object) return error.BadTokenizerJson;
        const model_v = root.object.get("model") orelse return error.BadTokenizerJson;
        if (model_v != .object) return error.BadTokenizerJson;

        const vocab_v = model_v.object.get("vocab") orelse return error.BadTokenizerJson;
        if (vocab_v != .object) return error.BadTokenizerJson;

        const merges_v = model_v.object.get("merges") orelse return error.BadTokenizerJson;
        if (merges_v != .array) return error.BadTokenizerJson;

        var vocab = VocabMap.init(allocator);
        errdefer vocab.deinit();
        var merges = MergeMap.init(allocator);
        errdefer merges.deinit();

        // First pass: compute max id so we can size the id->bytes vector.
        var max_id: u32 = 0;
        var it1 = vocab_v.object.iterator();
        while (it1.next()) |e| {
            if (e.value_ptr.* != .integer) return error.BadTokenizerJson;
            const id_i = e.value_ptr.integer;
            if (id_i < 0) return error.BadTokenizerJson;
            const id: u32 = @intCast(id_i);
            if (id > max_id) max_id = id;
        }

        var id_to_bytes = std.ArrayList([]const u8).init(allocator);
        errdefer id_to_bytes.deinit();
        try id_to_bytes.resize(@as(usize, max_id) + 1);
        for (id_to_bytes.items) |*slot| slot.* = "";

        // Second pass: populate vocab map and id->bytes, copying keys into arena.
        var it2 = vocab_v.object.iterator();
        while (it2.next()) |e| {
            const key_copy = try arena_alloc.dupe(u8, e.key_ptr.*);
            const id: u32 = @intCast(e.value_ptr.integer);
            try vocab.put(key_copy, id);
            id_to_bytes.items[id] = key_copy;
        }

        // Merges: each element is either a string "left right" or a 2-array [left, right].
        // We store as concatenated "left\x00right" in arena to avoid split ambiguity
        // when tokens contain spaces. (Rank = array index.)
        var rank: u32 = 0;
        for (merges_v.array.items) |m| {
            switch (m) {
                .string => |s| {
                    const sp_idx = std.mem.indexOfScalar(u8, s, ' ') orelse return error.BadTokenizerJson;
                    const left = s[0..sp_idx];
                    const right = s[sp_idx + 1 ..];
                    const key = try std.fmt.allocPrint(arena_alloc, "{s}\x00{s}", .{ left, right });
                    try merges.put(key, rank);
                },
                .array => |arr| {
                    if (arr.items.len != 2) return error.BadTokenizerJson;
                    if (arr.items[0] != .string or arr.items[1] != .string) return error.BadTokenizerJson;
                    const key = try std.fmt.allocPrint(arena_alloc, "{s}\x00{s}", .{
                        arr.items[0].string,
                        arr.items[1].string,
                    });
                    try merges.put(key, rank);
                },
                else => return error.BadTokenizerJson,
            }
            rank += 1;
        }

        return .{
            .allocator = allocator,
            .vocab_arena = arena,
            .vocab = vocab,
            .merges = merges,
            .id_to_bytes = id_to_bytes,
        };
    }

    /// Internal BPE over one pre-token chunk. Writes its final IDs into `out`.
    fn bpeChunk(self: *const Bpe, chunk: []const u8, out: *std.ArrayList(TokenId)) !void {
        if (chunk.len == 0) return;

        // If the whole chunk exists as a single vocab entry, take it.
        if (self.vocab.get(chunk)) |id| {
            try out.append(id);
            return;
        }

        // Build initial parts: each byte as its own piece (by byte value).
        // For GPT-2/tiktoken-style vocabs, single-byte tokens are guaranteed to
        // exist. We use the raw bytes as keys and let the merges table drive.
        var parts = std.ArrayList([]const u8).init(self.allocator);
        defer parts.deinit();
        try parts.ensureTotalCapacity(chunk.len);
        for (chunk) |*b| {
            // Slice of length 1 viewing into the chunk.
            try parts.append(@as(*const [1]u8, @ptrCast(b))[0..]);
        }

        // Repeatedly find the lowest-rank adjacent merge and apply it.
        var buf = std.ArrayList(u8).init(self.allocator);
        defer buf.deinit();

        while (parts.items.len >= 2) {
            var best_rank: u32 = std.math.maxInt(u32);
            var best_idx: usize = 0;
            var found = false;
            var i: usize = 0;
            while (i + 1 < parts.items.len) : (i += 1) {
                buf.clearRetainingCapacity();
                try buf.appendSlice(parts.items[i]);
                try buf.append(0);
                try buf.appendSlice(parts.items[i + 1]);
                if (self.merges.get(buf.items)) |r| {
                    if (r < best_rank) {
                        best_rank = r;
                        best_idx = i;
                        found = true;
                    }
                }
            }
            if (!found) break;

            // Merge parts[best_idx] + parts[best_idx+1] into one piece. We need
            // a contiguous slice; since the original parts came from `chunk`,
            // and we only ever merge adjacent contiguous slices, the merged
            // piece is also a contiguous slice of `chunk`.
            const left = parts.items[best_idx];
            const right = parts.items[best_idx + 1];
            const left_start = @intFromPtr(left.ptr) - @intFromPtr(chunk.ptr);
            const merged_len = left.len + right.len;
            const merged = chunk[left_start .. left_start + merged_len];
            parts.items[best_idx] = merged;
            _ = parts.orderedRemove(best_idx + 1);
        }

        for (parts.items) |p| {
            const id = self.vocab.get(p) orelse return error.UnknownPiece;
            try out.append(id);
        }
    }

    /// Thread-safe: encode `text` to token IDs, appending to `out`.
    pub fn encode(
        self: *const Bpe,
        text: []const u8,
        out: *std.ArrayList(TokenId),
    ) !void {
        var i: usize = 0;
        var span: Span = .{ .start = 0, .end = 0 };
        while (i < text.len) {
            i = nextPreToken(text, i, &span);
            if (span.end > span.start) {
                try self.bpeChunk(text[span.start..span.end], out);
            } else {
                // Shouldn't happen; guard against infinite loop.
                i += 1;
            }
        }
    }

    /// Thread-safe: decode token IDs into raw bytes, appending to `out`.
    pub fn decode(
        self: *const Bpe,
        ids: []const TokenId,
        out: *std.ArrayList(u8),
    ) !void {
        for (ids) |id| {
            if (@as(usize, id) >= self.id_to_bytes.items.len) return error.IdOutOfRange;
            try out.appendSlice(self.id_to_bytes.items[id]);
        }
    }
};

// -- parallel batch encode ----------------------------------------------------

const BatchCtx = struct {
    bpe: *const Bpe,
    prompts: []const []const u8,
    outs: []std.ArrayList(TokenId),
    next: std.atomic.Value(usize),
    err: std.atomic.Value(usize), // 0 = ok, 1 = err
};

fn batchWorker(ctx: *BatchCtx) void {
    while (true) {
        const i = ctx.next.fetchAdd(1, .monotonic);
        if (i >= ctx.prompts.len) return;
        ctx.bpe.encode(ctx.prompts[i], &ctx.outs[i]) catch {
            _ = ctx.err.store(1, .monotonic);
            return;
        };
    }
}

/// Batch-encode `prompts` into `out_arrays` (len must match) using `n_threads`.
/// If `n_threads` == 0, runs serially on the caller thread.
pub fn encodeBatchParallel(
    bpe: *const Bpe,
    prompts: []const []const u8,
    out_arrays: []std.ArrayList(TokenId),
    n_threads: usize,
) !void {
    if (prompts.len != out_arrays.len) return error.LengthMismatch;

    if (n_threads == 0 or prompts.len <= 1) {
        for (prompts, 0..) |p, i| try bpe.encode(p, &out_arrays[i]);
        return;
    }

    var ctx = BatchCtx{
        .bpe = bpe,
        .prompts = prompts,
        .outs = out_arrays,
        .next = std.atomic.Value(usize).init(0),
        .err = std.atomic.Value(usize).init(0),
    };

    const threads = try bpe.allocator.alloc(std.Thread, n_threads);
    defer bpe.allocator.free(threads);

    var started: usize = 0;
    while (started < n_threads) : (started += 1) {
        threads[started] = std.Thread.spawn(.{}, batchWorker, .{&ctx}) catch break;
    }
    // Also run on the current thread to use it.
    batchWorker(&ctx);
    for (threads[0..started]) |t| t.join();
    if (ctx.err.load(.monotonic) != 0) return error.EncodeFailed;
}

// -- tests --------------------------------------------------------------------

const testing = std.testing;

test "deterministic encode of ASCII string" {
    const allocator = testing.allocator;

    // Minimal JSON: vocab of single-byte entries for lowercase letters, space,
    // and '!' — enough to encode "hello world!".
    const json =
        \\{"model":{"type":"BPE","vocab":{
        \\"h":0,"e":1,"l":2,"o":3," ":4,"w":5,"r":6,"d":7,"!":8
        \\},"merges":[]}}
    ;

    var bpe = try Bpe.initFromBytes(allocator, json);
    defer bpe.deinit();

    var out1 = std.ArrayList(TokenId).init(allocator);
    defer out1.deinit();
    var out2 = std.ArrayList(TokenId).init(allocator);
    defer out2.deinit();

    try bpe.encode("hello", &out1);
    try bpe.encode("hello", &out2);
    try testing.expectEqualSlices(TokenId, out1.items, out2.items);
    try testing.expect(out1.items.len > 0);
}

test "decode(encode(x)) roundtrip for ASCII" {
    const allocator = testing.allocator;
    const json =
        \\{"model":{"type":"BPE","vocab":{
        \\"h":0,"e":1,"l":2,"o":3," ":4,"w":5,"r":6,"d":7,"!":8,",":9
        \\},"merges":[]}}
    ;

    var bpe = try Bpe.initFromBytes(allocator, json);
    defer bpe.deinit();

    var ids = std.ArrayList(TokenId).init(allocator);
    defer ids.deinit();
    var back = std.ArrayList(u8).init(allocator);
    defer back.deinit();

    const input = "hello, world!";
    try bpe.encode(input, &ids);
    try bpe.decode(ids.items, &back);
    try testing.expectEqualStrings(input, back.items);
}

test "encode uses merges" {
    const allocator = testing.allocator;
    // Vocab has 'h','i', and 'hi'. Merge ['h','i'] rank 0.
    const json =
        \\{"model":{"type":"BPE","vocab":{
        \\"h":0,"i":1,"hi":2
        \\},"merges":["h i"]}}
    ;
    var bpe = try Bpe.initFromBytes(allocator, json);
    defer bpe.deinit();

    var ids = std.ArrayList(TokenId).init(allocator);
    defer ids.deinit();
    try bpe.encode("hi", &ids);
    try testing.expectEqual(@as(usize, 1), ids.items.len);
    try testing.expectEqual(@as(TokenId, 2), ids.items[0]);
}

test "pre-tokenize contractions and punctuation" {
    const allocator = testing.allocator;
    // Vocab with enough bytes to round-trip; merges empty so output is bytes.
    const json =
        \\{"model":{"type":"BPE","vocab":{
        \\"I":0,"'":1,"m":2," ":3,"h":4,"e":5,"r":6,"!":7
        \\},"merges":[]}}
    ;
    var bpe = try Bpe.initFromBytes(allocator, json);
    defer bpe.deinit();

    var ids = std.ArrayList(TokenId).init(allocator);
    defer ids.deinit();
    var back = std.ArrayList(u8).init(allocator);
    defer back.deinit();

    const input = "I'm her!";
    try bpe.encode(input, &ids);
    try bpe.decode(ids.items, &back);
    try testing.expectEqualStrings(input, back.items);
}

test "batch parallel encode matches serial" {
    const allocator = testing.allocator;
    const json =
        \\{"model":{"type":"BPE","vocab":{
        \\"h":0,"e":1,"l":2,"o":3," ":4,"w":5,"r":6,"d":7,"!":8,",":9,"t":10,"i":11,"s":12
        \\},"merges":[]}}
    ;
    var bpe = try Bpe.initFromBytes(allocator, json);
    defer bpe.deinit();

    const prompts = [_][]const u8{
        "hello world",
        "this is",
        "hello, world!",
    };

    var serial: [3]std.ArrayList(TokenId) = undefined;
    var parallel: [3]std.ArrayList(TokenId) = undefined;
    for (&serial) |*a| a.* = std.ArrayList(TokenId).init(allocator);
    defer for (&serial) |*a| a.deinit();
    for (&parallel) |*a| a.* = std.ArrayList(TokenId).init(allocator);
    defer for (&parallel) |*a| a.deinit();

    for (prompts, 0..) |p, i| try bpe.encode(p, &serial[i]);
    try encodeBatchParallel(&bpe, &prompts, &parallel, 2);

    for (0..prompts.len) |i| {
        try testing.expectEqualSlices(TokenId, serial[i].items, parallel[i].items);
    }
}

// TODO(agent-16): add hard-coded (text, expected_ids) parity tests against
// HuggingFace `tokenizers` for MiniMax-M2.7 once the actual tokenizer.json is
// available on-box. Suggested vectors:
//   "Hello, world!"          -> <ids-from-hf>
//   "the quick brown fox"    -> <ids-from-hf>
//   "éèê"     -> <ids-from-hf>  (multi-byte UTF-8)
