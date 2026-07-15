const std = @import("std");

pub const TokenId = u32;
pub const byte_vocabulary_size: usize = 256;

pub const SpecialToken = enum(u2) {
    padding,
    beginning_of_sequence,
    end_of_sequence,
    unknown,
};

pub const Merge = struct {
    left: TokenId,
    right: TokenId,
};

pub const TrainOptions = struct {
    maximum_merges: usize = 128,
    minimum_frequency: usize = 2,

    fn validate(self: TrainOptions) !void {
        if (self.minimum_frequency < 2) return error.InvalidMinimumFrequency;
        if (self.maximum_merges > std.math.maxInt(TokenId) - byte_vocabulary_size - 4) {
            return error.TooManyMerges;
        }
    }
};

/// Deterministic byte-level byte-pair encoding.
///
/// Every input byte is part of the base vocabulary, so encoding cannot produce
/// an unknown token. Four control-token IDs are reserved after the learned
/// merge vocabulary and are skipped by `decode`.
pub const BpeTokenizer = struct {
    allocator: std.mem.Allocator,
    merges: []Merge,

    const magic = "ZBPE";
    const format_version: u32 = 1;

    pub fn train(
        allocator: std.mem.Allocator,
        corpus: []const u8,
        options: TrainOptions,
    ) !BpeTokenizer {
        try options.validate();
        if (corpus.len == 0) return error.EmptyCorpus;

        var working: std.ArrayList(TokenId) = .empty;
        defer working.deinit(allocator);
        try working.ensureTotalCapacity(allocator, corpus.len);
        for (corpus) |byte| try working.append(allocator, byte);

        var learned: std.ArrayList(Merge) = .empty;
        defer learned.deinit(allocator);
        try learned.ensureTotalCapacity(allocator, options.maximum_merges);

        while (learned.items.len < options.maximum_merges and working.items.len > 1) {
            var counts = std.AutoHashMap(u64, usize).init(allocator);
            defer counts.deinit();
            for (working.items[0 .. working.items.len - 1], working.items[1..]) |left, right| {
                const result = try counts.getOrPut(pairKey(left, right));
                if (!result.found_existing) result.value_ptr.* = 0;
                result.value_ptr.* += 1;
            }

            var best_key: ?u64 = null;
            var best_count: usize = 0;
            var iterator = counts.iterator();
            while (iterator.next()) |entry| {
                const count = entry.value_ptr.*;
                const key = entry.key_ptr.*;
                if (count > best_count or
                    (count == best_count and (best_key == null or key < best_key.?)))
                {
                    best_key = key;
                    best_count = count;
                }
            }
            if (best_key == null or best_count < options.minimum_frequency) break;

            const pair = mergeFromKey(best_key.?);
            const merged_token: TokenId = @intCast(byte_vocabulary_size + learned.items.len);
            try learned.append(allocator, pair);
            var next: std.ArrayList(TokenId) = .empty;
            errdefer next.deinit(allocator);
            try applyMerge(allocator, working.items, pair, merged_token, &next);
            working.deinit(allocator);
            working = next;
        }

        const merges = try allocator.alloc(Merge, learned.items.len);
        @memcpy(merges, learned.items);
        return .{ .allocator = allocator, .merges = merges };
    }

    pub fn deinit(self: *BpeTokenizer) void {
        self.allocator.free(self.merges);
        self.* = undefined;
    }

    pub fn vocabularySize(self: BpeTokenizer) usize {
        return byte_vocabulary_size + self.merges.len +
            @typeInfo(SpecialToken).@"enum".fields.len;
    }

    pub fn mergeCount(self: BpeTokenizer) usize {
        return self.merges.len;
    }

    pub fn mergeAt(self: BpeTokenizer, index: usize) Merge {
        return self.merges[index];
    }

    pub fn specialTokenId(self: BpeTokenizer, special: SpecialToken) TokenId {
        return @intCast(byte_vocabulary_size + self.merges.len + @intFromEnum(special));
    }

    pub fn isSpecialToken(self: BpeTokenizer, token: TokenId) bool {
        const first = byte_vocabulary_size + self.merges.len;
        return token >= first and token < self.vocabularySize();
    }

    pub fn encode(self: BpeTokenizer, allocator: std.mem.Allocator, text: []const u8) ![]TokenId {
        var tokens: std.ArrayList(TokenId) = .empty;
        defer tokens.deinit(allocator);
        try tokens.ensureTotalCapacity(allocator, text.len);
        for (text) |byte| try tokens.append(allocator, byte);

        for (self.merges, 0..) |pair, index| {
            if (tokens.items.len < 2) break;
            var next: std.ArrayList(TokenId) = .empty;
            errdefer next.deinit(allocator);
            const merged_token: TokenId = @intCast(byte_vocabulary_size + index);
            try applyMerge(allocator, tokens.items, pair, merged_token, &next);
            tokens.deinit(allocator);
            tokens = next;
        }

        const result = try allocator.alloc(TokenId, tokens.items.len);
        @memcpy(result, tokens.items);
        return result;
    }

    /// Decodes learned and byte tokens. Reserved control tokens are omitted.
    pub fn decode(self: BpeTokenizer, allocator: std.mem.Allocator, tokens: []const TokenId) ![]u8 {
        var bytes: std.ArrayList(u8) = .empty;
        defer bytes.deinit(allocator);
        for (tokens) |token| {
            if (self.isSpecialToken(token)) continue;
            try self.expandToken(allocator, token, &bytes);
        }
        const result = try allocator.alloc(u8, bytes.items.len);
        @memcpy(result, bytes.items);
        return result;
    }

    /// Returns the byte spelling of one learned or base token.
    pub fn tokenBytes(
        self: BpeTokenizer,
        allocator: std.mem.Allocator,
        token: TokenId,
    ) ![]u8 {
        if (self.isSpecialToken(token)) return error.SpecialTokenHasNoBytes;
        var bytes: std.ArrayList(u8) = .empty;
        defer bytes.deinit(allocator);
        try self.expandToken(allocator, token, &bytes);
        const result = try allocator.alloc(u8, bytes.items.len);
        @memcpy(result, bytes.items);
        return result;
    }

    pub fn write(self: BpeTokenizer, writer: anytype) !void {
        try writer.writeAll(magic);
        try writer.writeInt(u32, format_version, .little);
        try writer.writeInt(u32, @intCast(self.merges.len), .little);
        for (self.merges) |pair| {
            try writer.writeInt(TokenId, pair.left, .little);
            try writer.writeInt(TokenId, pair.right, .little);
        }
    }

    pub fn read(allocator: std.mem.Allocator, reader: anytype) !BpeTokenizer {
        var file_magic: [magic.len]u8 = undefined;
        @memcpy(&file_magic, try reader.take(magic.len));
        if (!std.mem.eql(u8, &file_magic, magic)) return error.InvalidFileFormat;
        if (try reader.takeInt(u32, .little) != format_version) {
            return error.UnsupportedVersion;
        }
        const merge_count: usize = @intCast(try reader.takeInt(u32, .little));
        if (merge_count > std.math.maxInt(TokenId) - byte_vocabulary_size - 4) {
            return error.TooManyMerges;
        }
        const merges = try allocator.alloc(Merge, merge_count);
        errdefer allocator.free(merges);
        for (merges, 0..) |*pair, index| {
            pair.* = .{
                .left = try reader.takeInt(TokenId, .little),
                .right = try reader.takeInt(TokenId, .little),
            };
            const token_limit = byte_vocabulary_size + index;
            if (pair.left >= token_limit or pair.right >= token_limit) {
                return error.InvalidMerge;
            }
            for (merges[0..index]) |previous| {
                if (previous.left == pair.left and previous.right == pair.right) {
                    return error.DuplicateMerge;
                }
            }
        }
        return .{ .allocator = allocator, .merges = merges };
    }

    fn expandToken(
        self: BpeTokenizer,
        allocator: std.mem.Allocator,
        token: TokenId,
        bytes: *std.ArrayList(u8),
    ) !void {
        if (token < byte_vocabulary_size) {
            try bytes.append(allocator, @intCast(token));
            return;
        }
        const merge_index = token - byte_vocabulary_size;
        if (merge_index >= self.merges.len) return error.InvalidToken;
        const value = self.merges[merge_index];
        try self.expandToken(allocator, value.left, bytes);
        try self.expandToken(allocator, value.right, bytes);
    }
};

fn pairKey(left: TokenId, right: TokenId) u64 {
    return (@as(u64, left) << 32) | right;
}

fn mergeFromKey(key: u64) Merge {
    return .{
        .left = @truncate(key >> 32),
        .right = @truncate(key),
    };
}

fn applyMerge(
    allocator: std.mem.Allocator,
    input: []const TokenId,
    merge: Merge,
    merged_token: TokenId,
    output: *std.ArrayList(TokenId),
) !void {
    try output.ensureTotalCapacity(allocator, input.len);
    var index: usize = 0;
    while (index < input.len) {
        if (index + 1 < input.len and input[index] == merge.left and
            input[index + 1] == merge.right)
        {
            try output.append(allocator, merged_token);
            index += 2;
        } else {
            try output.append(allocator, input[index]);
            index += 1;
        }
    }
}

test "byte BPE training is deterministic and compresses repeated text" {
    const testing = std.testing;
    const corpus = "low lower lowest low lower lowest low lower lowest";
    var first = try BpeTokenizer.train(testing.allocator, corpus, .{
        .maximum_merges = 24,
        .minimum_frequency = 2,
    });
    defer first.deinit();
    var second = try BpeTokenizer.train(testing.allocator, corpus, .{
        .maximum_merges = 24,
        .minimum_frequency = 2,
    });
    defer second.deinit();
    try testing.expectEqualSlices(Merge, first.merges, second.merges);
    const tokens = try first.encode(testing.allocator, corpus);
    defer testing.allocator.free(tokens);
    try testing.expect(tokens.len < corpus.len / 2);
    const decoded = try first.decode(testing.allocator, tokens);
    defer testing.allocator.free(decoded);
    try testing.expectEqualStrings(corpus, decoded);
}

test "byte BPE round trips arbitrary UTF-8 and skips controls" {
    const testing = std.testing;
    var tokenizer = try BpeTokenizer.train(testing.allocator, "hei hei hello hello", .{});
    defer tokenizer.deinit();
    const text = "Hei, maailma! 👋";
    const encoded = try tokenizer.encode(testing.allocator, text);
    defer testing.allocator.free(encoded);
    const with_controls = try testing.allocator.alloc(TokenId, encoded.len + 2);
    defer testing.allocator.free(with_controls);
    with_controls[0] = tokenizer.specialTokenId(.beginning_of_sequence);
    @memcpy(with_controls[1 .. 1 + encoded.len], encoded);
    with_controls[with_controls.len - 1] = tokenizer.specialTokenId(.end_of_sequence);
    const decoded = try tokenizer.decode(testing.allocator, with_controls);
    defer testing.allocator.free(decoded);
    try testing.expectEqualStrings(text, decoded);
    try testing.expect(tokenizer.isSpecialToken(tokenizer.specialTokenId(.padding)));
}

test "byte BPE vocabulary serializes without changing token IDs" {
    const testing = std.testing;
    var tokenizer = try BpeTokenizer.train(
        testing.allocator,
        "banana bandana banana bandana",
        .{ .maximum_merges = 16 },
    );
    defer tokenizer.deinit();
    var storage: [4096]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&storage);
    try tokenizer.write(&writer);
    var reader: std.Io.Reader = .fixed(writer.buffered());
    var restored = try BpeTokenizer.read(testing.allocator, &reader);
    defer restored.deinit();
    try testing.expectEqualSlices(Merge, tokenizer.merges, restored.merges);
    const original = try tokenizer.encode(testing.allocator, "banana");
    defer testing.allocator.free(original);
    const loaded = try restored.encode(testing.allocator, "banana");
    defer testing.allocator.free(loaded);
    try testing.expectEqualSlices(TokenId, original, loaded);
}
