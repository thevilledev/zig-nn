const std = @import("std");

pub const Tokenizer = struct {
    pub const vocab = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~\n";
    const fallback_id = std.mem.indexOfScalar(u8, vocab, ' ') orelse 0;

    pub fn vocabSize() usize {
        return vocab.len;
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
