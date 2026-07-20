const std = @import("std");

pub const Tokenizer = struct {
    pub const vocab = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~\n";
    const fallback_id = std.mem.indexOfScalar(u8, vocab, ' ') orelse 0;

    pub fn vocabSize() usize {
        return vocab.len;
    }

    /// Stable FNV-1a identity stored in checkpoints. Vocabulary size alone is
    /// insufficient because reordered tokens silently change model meaning.
    pub fn vocabularyFingerprint() u64 {
        var hash: u64 = 0xcbf29ce484222325;
        for (vocab) |byte| {
            hash ^= byte;
            hash *%= 0x100000001b3;
        }
        return hash;
    }

    pub fn encodeChar(ch: u8) usize {
        return std.mem.indexOfScalar(u8, vocab, ch) orelse fallback_id;
    }

    pub fn decodeToken(token: usize) u8 {
        if (token >= vocab.len) return '?';
        return vocab[token];
    }

    pub fn encode(allocator: std.mem.Allocator, text: []const u8) ![]usize {
        const tokens = try allocator.alloc(usize, text.len);
        for (text, 0..) |ch, i| {
            tokens[i] = encodeChar(ch);
        }
        return tokens;
    }

    pub fn decode(allocator: std.mem.Allocator, tokens: []const usize) ![]u8 {
        const text = try allocator.alloc(u8, tokens.len);
        for (tokens, 0..) |token, i| {
            text[i] = decodeToken(token);
        }
        return text;
    }
};

pub const Config = struct {
    block_size: usize = 16,
    n_layer: usize = 2,
    n_head: usize = 2,
    n_embd: usize = 32,
    dropout: f64 = 0.0,
    vocab_size: usize = Tokenizer.vocabSize(),

    pub fn mlpHidden(self: Config) usize {
        return 4 * self.n_embd;
    }
};

pub const OptimizerKind = enum {
    sgd,
    adamw,
};

pub const LearningRateSchedule = enum {
    constant,
    linear,
    cosine,
};

fn fuzzTokenizer(_: void, smith: *std.testing.Smith) !void {
    @disableInstrumentation();
    var buffer: [256]u8 = undefined;
    const len = smith.sliceWeightedBytes(&buffer, &.{
        .rangeAtMost(u8, 0, 255, 1),
        .rangeAtMost(u8, ' ', '~', 4),
        .value(u8, '\n', 1),
    });
    const input = buffer[0..len];
    const tokens = try Tokenizer.encode(std.testing.allocator, input);
    defer std.testing.allocator.free(tokens);
    const decoded = try Tokenizer.decode(std.testing.allocator, tokens);
    defer std.testing.allocator.free(decoded);
    try std.testing.expectEqual(input.len, decoded.len);
    for (input, decoded) |source, result| {
        const expected = if (std.mem.indexOfScalar(u8, Tokenizer.vocab, source) != null) source else ' ';
        try std.testing.expectEqual(expected, result);
    }
}

test "tokenizer accepts fuzzed byte input" {
    try std.testing.fuzz({}, fuzzTokenizer, .{ .corpus = &.{ "", "hello", "\x00\xff", Tokenizer.vocab } });
}
