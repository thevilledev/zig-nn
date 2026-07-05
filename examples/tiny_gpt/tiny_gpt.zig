const std = @import("std");
const nn = @import("nn");

const Matrix = nn.Matrix;
const testing = std.testing;
const math = std.math;

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

const CliOptions = struct {
    prompt: []const u8 = "to be",
    tokens: usize = 80,
    temperature: f64 = 1.0,
    top_k: usize = 8,
    seed: u64 = 42,
    model_config: Config = .{},
    config_overridden: bool = false,
    corpus: CorpusPreset = .auto,
    corpus_path: ?[]const u8 = null,
    checkpoint_in: ?[]const u8 = null,
    checkpoint_out: ?[]const u8 = null,
    resume_checkpoint: ?[]const u8 = null,
    train_demo_head: bool = true,
    train_full: bool = false,
    train_epochs: usize = 350,
    learning_rate: f64 = 0.5,
    full_train_steps: usize = 120,
    full_learning_rate: f64 = 0.03,
    min_learning_rate: f64 = 0.0,
    lr_schedule: LearningRateSchedule = .constant,
    warmup_steps: usize = 0,
    full_train_stride: usize = 1,
    full_batch_size: usize = 1,
    full_optimizer: OptimizerKind = .sgd,
    full_weight_decay: f64 = 0.0,
    adam_beta1: f64 = 0.9,
    adam_beta2: f64 = 0.999,
    adam_epsilon: f64 = 1e-8,
    train_chars: usize = 512,
    validation_chars: usize = 0,
    eval_split: f64 = 0.0,
    eval_windows: usize = 12,
    prior_chars: usize = 32768,
    corpus_prior: bool = true,
    summary_path: ?[]const u8 = null,
};

const CorpusPreset = enum {
    auto,
    toy,
    shakespeare,
    tinystories,
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

const Corpus = struct {
    text: []const u8,
    label: []const u8,
    path: ?[]const u8,
    owned: bool,
    allocator: std.mem.Allocator,

    fn deinit(self: *Corpus) void {
        if (self.owned) {
            self.allocator.free(self.text);
        }
    }
};

const fallback_toy_corpus =
    \\to be, or not to be, that is the question.
    \\the tiny model learns from a little play.
    \\to be a small machine is still to speak.
    \\we train the head on clear short lines.
    \\the model says: to be, to learn, to make.
    \\a small transformer can follow a prompt.
    \\to be clear, the demo repeats simple words.
    \\
;

const toy_corpus_path = "examples/tiny_gpt/data/toy.txt";
const shakespeare_corpus_path = "examples/tiny_gpt/data/shakespeare/input.txt";
const tinystories_corpus_path = "examples/tiny_gpt/data/tinystories/tinystories_1mb.txt";
const max_corpus_bytes = 8 * 1024 * 1024;
const checkpoint_format_version: u32 = 2;

const Linear = struct {
    weights: Matrix,
    bias: Matrix,

    fn init(
        allocator: std.mem.Allocator,
        in_features: usize,
        out_features: usize,
        rand: anytype,
        stddev: f64,
    ) !Linear {
        var weights = try Matrix.init(allocator, in_features, out_features);
        errdefer weights.deinit();
        var bias = try Matrix.init(allocator, 1, out_features);
        errdefer bias.deinit();

        try fillNormal(&weights, rand, stddev);
        bias.fill(0.0);

        return .{ .weights = weights, .bias = bias };
    }

    fn deinit(self: *Linear) void {
        self.weights.deinit();
        self.bias.deinit();
    }

    fn forward(self: *const Linear, input: Matrix, allocator: std.mem.Allocator) !Matrix {
        var output = try input.dotProduct(self.weights, allocator);
        errdefer output.deinit();

        for (0..output.rows) |row| {
            for (0..output.cols) |col| {
                try output.set(row, col, (try output.get(row, col)) + (try self.bias.get(0, col)));
            }
        }

        return output;
    }

    fn parameterCount(self: *const Linear) usize {
        return self.weights.rows * self.weights.cols + self.bias.cols;
    }
};

const LayerNorm = struct {
    weight: Matrix,
    bias: Matrix,
    eps: f64 = 1e-5,

    fn init(allocator: std.mem.Allocator, size: usize) !LayerNorm {
        var weight = try Matrix.init(allocator, 1, size);
        errdefer weight.deinit();
        var bias = try Matrix.init(allocator, 1, size);
        errdefer bias.deinit();

        weight.fill(1.0);
        bias.fill(0.0);

        return .{ .weight = weight, .bias = bias };
    }

    fn deinit(self: *LayerNorm) void {
        self.weight.deinit();
        self.bias.deinit();
    }

    fn forward(self: *const LayerNorm, input: Matrix, allocator: std.mem.Allocator) !Matrix {
        var output = try Matrix.init(allocator, input.rows, input.cols);
        errdefer output.deinit();

        for (0..input.rows) |row| {
            var mean: f64 = 0.0;
            for (0..input.cols) |col| {
                mean += try input.get(row, col);
            }
            mean /= @as(f64, @floatFromInt(input.cols));

            var variance: f64 = 0.0;
            for (0..input.cols) |col| {
                const centered = (try input.get(row, col)) - mean;
                variance += centered * centered;
            }
            variance /= @as(f64, @floatFromInt(input.cols));

            const inv_std = 1.0 / @sqrt(variance + self.eps);
            for (0..input.cols) |col| {
                const normalized = ((try input.get(row, col)) - mean) * inv_std;
                const scaled = normalized * (try self.weight.get(0, col)) + (try self.bias.get(0, col));
                try output.set(row, col, scaled);
            }
        }

        return output;
    }

    fn parameterCount(self: *const LayerNorm) usize {
        return self.weight.cols + self.bias.cols;
    }
};

const CausalSelfAttention = struct {
    c_attn: Linear,
    c_proj: Linear,
    n_head: usize,
    n_embd: usize,

    fn init(
        allocator: std.mem.Allocator,
        n_embd: usize,
        n_head: usize,
        rand: anytype,
        residual_stddev: f64,
    ) !CausalSelfAttention {
        var c_attn = try Linear.init(allocator, n_embd, 3 * n_embd, rand, 0.02);
        errdefer c_attn.deinit();
        var c_proj = try Linear.init(allocator, n_embd, n_embd, rand, residual_stddev);
        errdefer c_proj.deinit();

        return .{
            .c_attn = c_attn,
            .c_proj = c_proj,
            .n_head = n_head,
            .n_embd = n_embd,
        };
    }

    fn deinit(self: *CausalSelfAttention) void {
        self.c_attn.deinit();
        self.c_proj.deinit();
    }

    fn forward(self: *CausalSelfAttention, input: Matrix, allocator: std.mem.Allocator) !Matrix {
        const seq_len = input.rows;
        const head_dim = self.n_embd / self.n_head;
        const scale = 1.0 / @sqrt(@as(f64, @floatFromInt(head_dim)));

        var qkv = try self.c_attn.forward(input, allocator);
        defer qkv.deinit();

        var context = try Matrix.init(allocator, seq_len, self.n_embd);
        errdefer context.deinit();
        context.fill(0.0);

        var scores = try allocator.alloc(f64, seq_len);
        defer allocator.free(scores);
        const probs = try allocator.alloc(f64, seq_len);
        defer allocator.free(probs);

        for (0..self.n_head) |head| {
            const head_offset = head * head_dim;
            for (0..seq_len) |query_pos| {
                for (0..seq_len) |key_pos| {
                    var score: f64 = 0.0;
                    if (key_pos <= query_pos) {
                        for (0..head_dim) |d| {
                            const q = try qkv.get(query_pos, head_offset + d);
                            const k = try qkv.get(key_pos, self.n_embd + head_offset + d);
                            score += q * k;
                        }
                        score *= scale;
                    }
                    scores[key_pos] = score;
                }

                causalSoftmax(scores, query_pos, probs);

                for (0..head_dim) |d| {
                    var value: f64 = 0.0;
                    for (0..query_pos + 1) |key_pos| {
                        const v = try qkv.get(key_pos, 2 * self.n_embd + head_offset + d);
                        value += probs[key_pos] * v;
                    }
                    try context.set(query_pos, head_offset + d, value);
                }
            }
        }

        const output = try self.c_proj.forward(context, allocator);
        context.deinit();
        return output;
    }

    fn parameterCount(self: *const CausalSelfAttention) usize {
        return self.c_attn.parameterCount() + self.c_proj.parameterCount();
    }
};

const MLP = struct {
    c_fc: Linear,
    c_proj: Linear,

    fn init(
        allocator: std.mem.Allocator,
        config: Config,
        rand: anytype,
        residual_stddev: f64,
    ) !MLP {
        var c_fc = try Linear.init(allocator, config.n_embd, config.mlpHidden(), rand, 0.02);
        errdefer c_fc.deinit();
        var c_proj = try Linear.init(allocator, config.mlpHidden(), config.n_embd, rand, residual_stddev);
        errdefer c_proj.deinit();

        return .{ .c_fc = c_fc, .c_proj = c_proj };
    }

    fn deinit(self: *MLP) void {
        self.c_fc.deinit();
        self.c_proj.deinit();
    }

    fn forward(self: *MLP, input: Matrix, allocator: std.mem.Allocator) !Matrix {
        var hidden = try self.c_fc.forward(input, allocator);
        defer hidden.deinit();

        for (0..hidden.rows) |row| {
            for (0..hidden.cols) |col| {
                try hidden.set(row, col, gelu(try hidden.get(row, col)));
            }
        }

        return self.c_proj.forward(hidden, allocator);
    }

    fn parameterCount(self: *const MLP) usize {
        return self.c_fc.parameterCount() + self.c_proj.parameterCount();
    }
};

const Block = struct {
    ln_1: LayerNorm,
    attn: CausalSelfAttention,
    ln_2: LayerNorm,
    mlp: MLP,

    fn init(allocator: std.mem.Allocator, config: Config, rand: anytype) !Block {
        const residual_stddev = 0.02 / @sqrt(2.0 * @as(f64, @floatFromInt(config.n_layer)));

        var ln_1 = try LayerNorm.init(allocator, config.n_embd);
        errdefer ln_1.deinit();
        var attn = try CausalSelfAttention.init(allocator, config.n_embd, config.n_head, rand, residual_stddev);
        errdefer attn.deinit();
        var ln_2 = try LayerNorm.init(allocator, config.n_embd);
        errdefer ln_2.deinit();
        var mlp = try MLP.init(allocator, config, rand, residual_stddev);
        errdefer mlp.deinit();

        return .{ .ln_1 = ln_1, .attn = attn, .ln_2 = ln_2, .mlp = mlp };
    }

    fn deinit(self: *Block) void {
        self.ln_1.deinit();
        self.attn.deinit();
        self.ln_2.deinit();
        self.mlp.deinit();
    }

    fn forward(self: *Block, input: Matrix, allocator: std.mem.Allocator) !Matrix {
        var normed_for_attention = try self.ln_1.forward(input, allocator);
        defer normed_for_attention.deinit();

        var attention_output = try self.attn.forward(normed_for_attention, allocator);
        defer attention_output.deinit();

        var with_attention = try input.add(attention_output, allocator);
        defer with_attention.deinit();

        var normed_for_mlp = try self.ln_2.forward(with_attention, allocator);
        defer normed_for_mlp.deinit();

        var mlp_output = try self.mlp.forward(normed_for_mlp, allocator);
        defer mlp_output.deinit();

        return with_attention.add(mlp_output, allocator);
    }

    fn parameterCount(self: *const Block) usize {
        return self.ln_1.parameterCount() +
            self.attn.parameterCount() +
            self.ln_2.parameterCount() +
            self.mlp.parameterCount();
    }
};

pub const TinyGPT = struct {
    allocator: std.mem.Allocator,
    config: Config,
    token_embedding: Matrix,
    position_embedding: Matrix,
    blocks: []Block,
    ln_f: LayerNorm,
    lm_head: Linear,
    prng: std.Random.DefaultPrng,

    pub fn init(allocator: std.mem.Allocator, config: Config, seed: u64) !TinyGPT {
        if (config.n_embd == 0 or config.n_head == 0 or config.n_layer == 0 or config.block_size == 0) {
            return error.InvalidConfig;
        }
        if (config.n_embd % config.n_head != 0) {
            return error.InvalidHeadCount;
        }

        var prng = std.Random.DefaultPrng.init(seed);
        const rand = prng.random();

        var token_embedding = try Matrix.init(allocator, config.vocab_size, config.n_embd);
        errdefer token_embedding.deinit();
        try fillNormal(&token_embedding, rand, 0.02);

        var position_embedding = try Matrix.init(allocator, config.block_size, config.n_embd);
        errdefer position_embedding.deinit();
        try fillNormal(&position_embedding, rand, 0.02);

        const blocks = try allocator.alloc(Block, config.n_layer);
        var initialized_blocks: usize = 0;
        errdefer {
            for (blocks[0..initialized_blocks]) |*block| {
                block.deinit();
            }
            allocator.free(blocks);
        }

        for (blocks) |*block| {
            block.* = try Block.init(allocator, config, rand);
            initialized_blocks += 1;
        }

        var ln_f = try LayerNorm.init(allocator, config.n_embd);
        errdefer ln_f.deinit();

        var lm_head = try Linear.init(allocator, config.n_embd, config.vocab_size, rand, 0.02);
        errdefer lm_head.deinit();

        return .{
            .allocator = allocator,
            .config = config,
            .token_embedding = token_embedding,
            .position_embedding = position_embedding,
            .blocks = blocks,
            .ln_f = ln_f,
            .lm_head = lm_head,
            .prng = prng,
        };
    }

    pub fn deinit(self: *TinyGPT) void {
        self.token_embedding.deinit();
        self.position_embedding.deinit();
        for (self.blocks) |*block| {
            block.deinit();
        }
        self.allocator.free(self.blocks);
        self.ln_f.deinit();
        self.lm_head.deinit();
    }

    pub fn forward(self: *TinyGPT, tokens: []const usize) !Matrix {
        var activations = try self.hidden(tokens);
        defer activations.deinit();

        return self.logitsFromHidden(activations);
    }

    fn hidden(self: *TinyGPT, tokens: []const usize) !Matrix {
        if (tokens.len == 0) return error.EmptySequence;
        if (tokens.len > self.config.block_size) return error.ContextTooLong;

        var current = try self.embed(tokens);

        for (self.blocks) |*block| {
            const next = block.forward(current, self.allocator) catch |err| {
                current.deinit();
                return err;
            };
            current.deinit();
            current = next;
        }

        const normed = self.ln_f.forward(current, self.allocator) catch |err| {
            current.deinit();
            return err;
        };
        current.deinit();

        return normed;
    }

    pub fn generateTokens(
        self: *TinyGPT,
        allocator: std.mem.Allocator,
        prompt: []const u8,
        max_new_tokens: usize,
        temperature: f64,
        top_k: usize,
    ) ![]usize {
        const prompt_len = if (prompt.len == 0) 1 else prompt.len;
        var tokens = try allocator.alloc(usize, prompt_len + max_new_tokens);
        errdefer allocator.free(tokens);

        if (prompt.len == 0) {
            tokens[0] = Tokenizer.encodeChar(' ');
        } else {
            for (prompt, 0..) |ch, i| {
                tokens[i] = Tokenizer.encodeChar(ch);
            }
        }

        var token_count = prompt_len;
        for (0..max_new_tokens) |_| {
            const context_start = if (token_count > self.config.block_size)
                token_count - self.config.block_size
            else
                0;
            var logits = try self.forward(tokens[context_start..token_count]);
            defer logits.deinit();

            tokens[token_count] = try self.sampleNextToken(logits, temperature, top_k);
            token_count += 1;
        }

        return tokens;
    }

    pub fn generateTokensWithCorpusPrior(
        self: *TinyGPT,
        allocator: std.mem.Allocator,
        prompt: []const u8,
        max_new_tokens: usize,
        temperature: f64,
        top_k: usize,
        corpus: []const u8,
    ) ![]usize {
        const corpus_tokens = try Tokenizer.encode(allocator, corpus);
        defer allocator.free(corpus_tokens);

        const prompt_len = if (prompt.len == 0) 1 else prompt.len;
        var tokens = try allocator.alloc(usize, prompt_len + max_new_tokens);
        errdefer allocator.free(tokens);

        if (prompt.len == 0) {
            tokens[0] = Tokenizer.encodeChar(' ');
        } else {
            for (prompt, 0..) |ch, i| {
                tokens[i] = Tokenizer.encodeChar(ch);
            }
        }

        var token_count = prompt_len;
        for (0..max_new_tokens) |_| {
            const context_start = if (token_count > self.config.block_size)
                token_count - self.config.block_size
            else
                0;
            var logits = try self.forward(tokens[context_start..token_count]);
            defer logits.deinit();

            tokens[token_count] = try self.sampleNextTokenWithCorpusPrior(
                logits,
                tokens[0..token_count],
                corpus_tokens,
                temperature,
                top_k,
            );
            token_count += 1;
        }

        return tokens;
    }

    pub fn parameterCount(self: *const TinyGPT) usize {
        var count = self.token_embedding.rows * self.token_embedding.cols +
            self.position_embedding.rows * self.position_embedding.cols +
            self.ln_f.parameterCount() +
            self.lm_head.parameterCount();

        for (self.blocks) |*block| {
            count += block.parameterCount();
        }

        return count;
    }

    pub fn saveToFile(self: *const TinyGPT, path: []const u8) !void {
        try self.saveToFileWithMetadata(path, "");
    }

    pub fn saveToFileWithMetadata(self: *const TinyGPT, path: []const u8, metadata_json: []const u8) !void {
        const io = std.Options.debug_io;
        const file = try std.Io.Dir.cwd().createFile(io, path, .{});
        defer file.close(io);

        var buffer: [4096]u8 = undefined;
        var file_writer = file.writerStreaming(io, &buffer);
        const writer = &file_writer.interface;

        try writer.writeAll("TGPT");
        try writer.writeInt(u32, checkpoint_format_version, .little);
        try writer.writeInt(u32, @intCast(self.config.block_size), .little);
        try writer.writeInt(u32, @intCast(self.config.n_layer), .little);
        try writer.writeInt(u32, @intCast(self.config.n_head), .little);
        try writer.writeInt(u32, @intCast(self.config.n_embd), .little);
        try writer.writeInt(u32, @intCast(self.config.vocab_size), .little);

        try writeMatrix(writer, self.token_embedding);
        try writeMatrix(writer, self.position_embedding);
        for (self.blocks) |*block| {
            try writeLayerNorm(writer, block.ln_1);
            try writeLinear(writer, block.attn.c_attn);
            try writeLinear(writer, block.attn.c_proj);
            try writeLayerNorm(writer, block.ln_2);
            try writeLinear(writer, block.mlp.c_fc);
            try writeLinear(writer, block.mlp.c_proj);
        }
        try writeLayerNorm(writer, self.ln_f);
        try writeLinear(writer, self.lm_head);
        try writer.writeInt(u32, @intCast(metadata_json.len), .little);
        try writer.writeAll(metadata_json);

        try writer.flush();
    }

    pub fn loadFromFile(allocator: std.mem.Allocator, path: []const u8) !TinyGPT {
        const io = std.Options.debug_io;
        const file = try std.Io.Dir.cwd().openFile(io, path, .{});
        defer file.close(io);

        var buffer: [4096]u8 = undefined;
        var file_reader = file.readerStreaming(io, &buffer);
        const reader = &file_reader.interface;

        var magic: [4]u8 = undefined;
        @memcpy(&magic, try reader.take(4));
        if (!std.mem.eql(u8, &magic, "TGPT")) return error.InvalidFileFormat;

        const version = try reader.takeInt(u32, .little);
        if (version != 1 and version != checkpoint_format_version) return error.UnsupportedVersion;

        const config: Config = .{
            .block_size = @intCast(try reader.takeInt(u32, .little)),
            .n_layer = @intCast(try reader.takeInt(u32, .little)),
            .n_head = @intCast(try reader.takeInt(u32, .little)),
            .n_embd = @intCast(try reader.takeInt(u32, .little)),
            .vocab_size = @intCast(try reader.takeInt(u32, .little)),
        };
        if (config.vocab_size != Tokenizer.vocabSize()) return error.UnsupportedVocabulary;

        var model = try TinyGPT.init(allocator, config, 0);
        errdefer model.deinit();

        try readMatrixInto(reader, &model.token_embedding);
        try readMatrixInto(reader, &model.position_embedding);
        for (model.blocks) |*block| {
            try readLayerNormInto(reader, &block.ln_1);
            try readLinearInto(reader, &block.attn.c_attn);
            try readLinearInto(reader, &block.attn.c_proj);
            try readLayerNormInto(reader, &block.ln_2);
            try readLinearInto(reader, &block.mlp.c_fc);
            try readLinearInto(reader, &block.mlp.c_proj);
        }
        try readLayerNormInto(reader, &model.ln_f);
        try readLinearInto(reader, &model.lm_head);
        if (version >= 2) {
            const metadata_len = try reader.takeInt(u32, .little);
            if (metadata_len > 0) _ = try reader.take(metadata_len);
        }

        return model;
    }

    fn embed(self: *TinyGPT, tokens: []const usize) !Matrix {
        var output = try Matrix.init(self.allocator, tokens.len, self.config.n_embd);
        errdefer output.deinit();

        for (tokens, 0..) |token, pos| {
            if (token >= self.config.vocab_size) return error.TokenOutOfVocabulary;
            for (0..self.config.n_embd) |dim| {
                const token_value = try self.token_embedding.get(token, dim);
                const position_value = try self.position_embedding.get(pos, dim);
                try output.set(pos, dim, token_value + position_value);
            }
        }

        return output;
    }

    fn logitsFromHidden(self: *TinyGPT, activations: Matrix) !Matrix {
        return self.lm_head.forward(activations, self.allocator);
    }

    fn sampleNextToken(self: *TinyGPT, logits: Matrix, temperature: f64, top_k: usize) !usize {
        const row = logits.rows - 1;
        const vocab_size = logits.cols;
        const safe_temperature = if (temperature <= 0.0) 1.0 else temperature;

        var scores = try self.allocator.alloc(f64, vocab_size);
        defer self.allocator.free(scores);
        for (0..vocab_size) |token| {
            scores[token] = (try logits.get(row, token)) / safe_temperature;
        }

        return self.sampleFromScores(scores, top_k);
    }

    fn sampleNextTokenWithCorpusPrior(
        self: *TinyGPT,
        logits: Matrix,
        context: []const usize,
        corpus_tokens: []const usize,
        temperature: f64,
        top_k: usize,
    ) !usize {
        const row = logits.rows - 1;
        const vocab_size = logits.cols;
        const safe_temperature = if (temperature <= 0.0) 1.0 else temperature;

        var scores = try self.allocator.alloc(f64, vocab_size);
        defer self.allocator.free(scores);
        for (0..vocab_size) |token| {
            scores[token] = 0.05 * (try logits.get(row, token)) / safe_temperature;
        }

        const matched = try addCorpusPrior(self.allocator, scores, context, corpus_tokens, 3.5);
        if (!matched) {
            for (0..vocab_size) |token| {
                scores[token] = (try logits.get(row, token)) / safe_temperature;
            }
        }

        return self.sampleFromScores(scores, top_k);
    }

    fn sampleFromScores(self: *TinyGPT, scores: []const f64, top_k: usize) !usize {
        const vocab_size = scores.len;
        const allowed = try self.allocator.alloc(bool, vocab_size);
        defer self.allocator.free(allowed);
        try chooseTopK(scores, top_k, allowed);

        var max_score = -math.inf(f64);
        for (scores, 0..) |score, token| {
            if (allowed[token]) max_score = @max(max_score, score);
        }

        var sum: f64 = 0.0;
        for (scores, 0..) |score, token| {
            if (allowed[token]) {
                sum += math.exp(score - max_score);
            }
        }

        if (sum == 0.0) return error.InvalidProbabilityDistribution;

        const draw = self.prng.random().float(f64) * sum;
        var cumulative: f64 = 0.0;
        for (scores, 0..) |score, token| {
            if (!allowed[token]) continue;
            cumulative += math.exp(score - max_score);
            if (draw <= cumulative) return token;
        }

        return vocab_size - 1;
    }
};

pub fn prepareNextTokenTargets(allocator: std.mem.Allocator, tokens: []const usize) ![]usize {
    if (tokens.len < 2) return error.SequenceTooShort;
    const targets = try allocator.alloc(usize, tokens.len - 1);
    @memcpy(targets, tokens[1..]);
    return targets;
}

pub fn crossEntropyLoss(logits: Matrix, targets: []const usize) !f64 {
    if (logits.rows != targets.len) return error.DimensionMismatch;
    if (targets.len == 0) return error.EmptySequence;

    var total_loss: f64 = 0.0;
    for (targets, 0..) |target, row| {
        if (target >= logits.cols) return error.TokenOutOfVocabulary;

        var max_logit = -math.inf(f64);
        for (0..logits.cols) |col| {
            max_logit = @max(max_logit, try logits.get(row, col));
        }

        var exp_sum: f64 = 0.0;
        for (0..logits.cols) |col| {
            exp_sum += math.exp((try logits.get(row, col)) - max_logit);
        }

        const target_logit = try logits.get(row, target);
        total_loss += -target_logit + max_logit + @log(exp_sum);
    }

    return total_loss / @as(f64, @floatFromInt(targets.len));
}

fn writeMatrix(writer: anytype, matrix: Matrix) !void {
    try writer.writeInt(u32, @intCast(matrix.rows), .little);
    try writer.writeInt(u32, @intCast(matrix.cols), .little);
    for (matrix.data) |value| {
        try writer.writeInt(i64, @as(i64, @bitCast(value)), .little);
    }
}

fn readMatrixInto(reader: anytype, matrix: *Matrix) !void {
    const rows: usize = @intCast(try reader.takeInt(u32, .little));
    const cols: usize = @intCast(try reader.takeInt(u32, .little));
    if (rows != matrix.rows or cols != matrix.cols) return error.DimensionMismatch;
    for (matrix.data) |*value| {
        value.* = @as(f64, @bitCast(try reader.takeInt(i64, .little)));
    }
}

fn writeLinear(writer: anytype, linear: Linear) !void {
    try writeMatrix(writer, linear.weights);
    try writeMatrix(writer, linear.bias);
}

fn readLinearInto(reader: anytype, linear: *Linear) !void {
    try readMatrixInto(reader, &linear.weights);
    try readMatrixInto(reader, &linear.bias);
}

fn writeLayerNorm(writer: anytype, layer_norm: LayerNorm) !void {
    try writeMatrix(writer, layer_norm.weight);
    try writeMatrix(writer, layer_norm.bias);
}

fn readLayerNormInto(reader: anytype, layer_norm: *LayerNorm) !void {
    try readMatrixInto(reader, &layer_norm.weight);
    try readMatrixInto(reader, &layer_norm.bias);
}

fn loadCorpus(allocator: std.mem.Allocator, options: CliOptions) !Corpus {
    if (options.corpus_path) |path| {
        return (try readCorpusFile(allocator, path, "custom corpus")) orelse error.MissingCorpusFile;
    }

    switch (options.corpus) {
        .auto => {
            if (try readCorpusFile(allocator, tinystories_corpus_path, "TinyStories 1MB")) |corpus| return corpus;
            if (try readCorpusFile(allocator, shakespeare_corpus_path, "Tiny Shakespeare")) |corpus| return corpus;
            if (try readCorpusFile(allocator, toy_corpus_path, "toy corpus")) |corpus| return corpus;
            return fallbackToyCorpus(allocator);
        },
        .toy => {
            if (try readCorpusFile(allocator, toy_corpus_path, "toy corpus")) |corpus| return corpus;
            return fallbackToyCorpus(allocator);
        },
        .shakespeare => {
            return (try readCorpusFile(allocator, shakespeare_corpus_path, "Tiny Shakespeare")) orelse error.MissingCorpusFile;
        },
        .tinystories => {
            return (try readCorpusFile(allocator, tinystories_corpus_path, "TinyStories 1MB")) orelse error.MissingCorpusFile;
        },
    }
}

fn readCorpusFile(
    allocator: std.mem.Allocator,
    path: []const u8,
    label: []const u8,
) !?Corpus {
    const text = std.Io.Dir.cwd().readFileAlloc(
        std.Options.debug_io,
        path,
        allocator,
        .limited(max_corpus_bytes),
    ) catch |err| switch (err) {
        error.FileNotFound => return null,
        else => return err,
    };

    return .{
        .text = text,
        .label = label,
        .path = path,
        .owned = true,
        .allocator = allocator,
    };
}

fn fallbackToyCorpus(allocator: std.mem.Allocator) Corpus {
    return .{
        .text = fallback_toy_corpus,
        .label = "built-in toy corpus",
        .path = null,
        .owned = false,
        .allocator = allocator,
    };
}

fn limitedCorpus(text: []const u8, limit: usize) []const u8 {
    if (limit == 0 or limit >= text.len) return text;
    return text[0..limit];
}

fn resolvedEvalChars(text_len: usize, train_limit: usize, validation_chars: usize, eval_split: f64) usize {
    if (validation_chars > 0) return validation_chars;
    if (eval_split <= 0.0 or text_len < 3) return 0;

    const max_eval = if (train_limit == 0 or train_limit >= text_len)
        text_len - 1
    else
        text_len - train_limit;
    if (max_eval < 2) return 0;

    const requested = @max(
        @as(usize, 1),
        @as(usize, @intFromFloat(@ceil(@as(f64, @floatFromInt(text_len)) * eval_split))),
    );
    return @min(requested, max_eval);
}

fn trainingCorpus(text: []const u8, train_limit: usize, validation_chars: usize) []const u8 {
    var end = if (train_limit == 0 or train_limit >= text.len) text.len else train_limit;
    if ((train_limit == 0 or train_limit >= text.len) and validation_chars > 0 and text.len > validation_chars + 1) {
        end = text.len - validation_chars;
    }
    return text[0..end];
}

fn validationCorpus(text: []const u8, train_limit: usize, validation_chars: usize) ?[]const u8 {
    if (validation_chars == 0 or text.len < 2) return null;

    const start = if (train_limit == 0 or train_limit >= text.len) blk: {
        if (text.len <= validation_chars + 1) return null;
        break :blk text.len - validation_chars;
    } else train_limit;

    if (start >= text.len - 1) return null;
    const end = @min(start + validation_chars, text.len);
    if (end - start < 2) return null;
    return text[start..end];
}

const TrainingStats = struct {
    samples: usize,
    initial_loss: f64,
    final_loss: f64,
};

fn trainOutputHeadOnCorpus(
    model: *TinyGPT,
    corpus: []const u8,
    epochs: usize,
    learning_rate: f64,
) !TrainingStats {
    const allocator = model.allocator;
    const tokens = try Tokenizer.encode(allocator, corpus);
    defer allocator.free(tokens);
    if (tokens.len < 2) return error.SequenceTooShort;

    var features = try Matrix.init(allocator, tokens.len - 1, model.config.n_embd);
    defer features.deinit();

    const targets = try allocator.alloc(usize, tokens.len - 1);
    defer allocator.free(targets);

    for (targets, 0..) |*target, sample| {
        const next_pos = sample + 1;
        const context_start = if (next_pos > model.config.block_size)
            next_pos - model.config.block_size
        else
            0;

        var hidden = try model.hidden(tokens[context_start..next_pos]);
        defer hidden.deinit();

        target.* = tokens[next_pos];
        const last_row = hidden.rows - 1;
        for (0..model.config.n_embd) |dim| {
            try features.set(sample, dim, try hidden.get(last_row, dim));
        }
    }

    const initial_loss = try outputHeadLoss(model, features, targets);
    for (0..epochs) |_| {
        var logits = try model.lm_head.forward(features, allocator);
        defer logits.deinit();
        try trainOutputHeadStep(model, features, logits, targets, learning_rate);
    }
    const final_loss = try outputHeadLoss(model, features, targets);

    return .{
        .samples = targets.len,
        .initial_loss = initial_loss,
        .final_loss = final_loss,
    };
}

fn outputHeadLoss(model: *TinyGPT, features: Matrix, targets: []const usize) !f64 {
    var logits = try model.lm_head.forward(features, model.allocator);
    defer logits.deinit();
    return crossEntropyLoss(logits, targets);
}

fn trainOutputHeadStep(
    model: *TinyGPT,
    features: Matrix,
    logits: Matrix,
    targets: []const usize,
    learning_rate: f64,
) !void {
    const allocator = model.allocator;
    var gradient = try Matrix.init(allocator, logits.rows, logits.cols);
    defer gradient.deinit();

    for (targets, 0..) |target, row| {
        var max_logit = -math.inf(f64);
        for (0..logits.cols) |col| {
            max_logit = @max(max_logit, try logits.get(row, col));
        }

        var exp_sum: f64 = 0.0;
        for (0..logits.cols) |col| {
            exp_sum += math.exp((try logits.get(row, col)) - max_logit);
        }

        for (0..logits.cols) |col| {
            var value = math.exp((try logits.get(row, col)) - max_logit) / exp_sum;
            if (col == target) value -= 1.0;
            try gradient.set(row, col, value);
        }
    }

    const scale = learning_rate / @as(f64, @floatFromInt(targets.len));
    for (0..model.lm_head.weights.rows) |feature_col| {
        for (0..model.lm_head.weights.cols) |token_col| {
            var grad_sum: f64 = 0.0;
            for (0..features.rows) |row| {
                grad_sum += (try features.get(row, feature_col)) * (try gradient.get(row, token_col));
            }

            const updated = (try model.lm_head.weights.get(feature_col, token_col)) - scale * grad_sum;
            try model.lm_head.weights.set(feature_col, token_col, updated);
        }
    }

    for (0..model.lm_head.bias.cols) |token_col| {
        var grad_sum: f64 = 0.0;
        for (0..gradient.rows) |row| {
            grad_sum += try gradient.get(row, token_col);
        }

        const updated = (try model.lm_head.bias.get(0, token_col)) - scale * grad_sum;
        try model.lm_head.bias.set(0, token_col, updated);
    }
}

pub const FullTrainingOptions = struct {
    steps: usize = 120,
    learning_rate: f64 = 0.03,
    min_learning_rate: f64 = 0.0,
    lr_schedule: LearningRateSchedule = .constant,
    warmup_steps: usize = 0,
    context_len: usize = 16,
    stride: usize = 1,
    batch_size: usize = 1,
    optimizer: OptimizerKind = .sgd,
    weight_decay: f64 = 0.0,
    adam_beta1: f64 = 0.9,
    adam_beta2: f64 = 0.999,
    adam_epsilon: f64 = 1e-8,
    eval_windows: usize = 12,
};

pub const FullTrainingStats = struct {
    steps: usize,
    updates: usize,
    batch_size: usize,
    tokens_seen: usize,
    initial_loss: f64,
    final_loss: f64,
    validation_initial_loss: ?f64 = null,
    validation_final_loss: ?f64 = null,
    learning_rate_initial: f64,
    learning_rate_final: f64,
    learning_rate_min: f64,
    eval_windows: usize,
};

const ParamState = struct {
    ptr: [*]f64,
    len: usize,
    m: []f64,
    v: []f64,
};

const TrainingOptimizer = struct {
    allocator: std.mem.Allocator,
    kind: OptimizerKind,
    learning_rate: f64,
    weight_decay: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    step: usize = 1,
    states: std.ArrayList(ParamState) = .empty,

    fn init(allocator: std.mem.Allocator, options: FullTrainingOptions) TrainingOptimizer {
        return .{
            .allocator = allocator,
            .kind = options.optimizer,
            .learning_rate = options.learning_rate,
            .weight_decay = options.weight_decay,
            .beta1 = options.adam_beta1,
            .beta2 = options.adam_beta2,
            .epsilon = options.adam_epsilon,
        };
    }

    fn deinit(self: *TrainingOptimizer) void {
        for (self.states.items) |state| {
            self.allocator.free(state.m);
            self.allocator.free(state.v);
        }
        self.states.deinit(self.allocator);
    }

    fn beginStep(self: *TrainingOptimizer, step: usize) void {
        self.step = @max(step, @as(usize, 1));
    }

    fn update(self: *TrainingOptimizer, matrix: *Matrix, index: usize, gradient: f64) !void {
        if (index >= matrix.data.len) return error.DimensionMismatch;
        switch (self.kind) {
            .sgd => {
                const decay_gradient = if (self.weight_decay == 0.0)
                    0.0
                else
                    self.weight_decay * matrix.data[index];
                matrix.data[index] -= self.learning_rate * (gradient + decay_gradient);
            },
            .adamw => {
                const state = try self.stateFor(matrix.data);
                const beta1_correction = 1.0 - math.pow(f64, self.beta1, @as(f64, @floatFromInt(self.step)));
                const beta2_correction = 1.0 - math.pow(f64, self.beta2, @as(f64, @floatFromInt(self.step)));

                state.m[index] = self.beta1 * state.m[index] + (1.0 - self.beta1) * gradient;
                state.v[index] = self.beta2 * state.v[index] + (1.0 - self.beta2) * gradient * gradient;

                const m_hat = state.m[index] / beta1_correction;
                const v_hat = state.v[index] / beta2_correction;
                if (self.weight_decay != 0.0) {
                    matrix.data[index] *= 1.0 - self.learning_rate * self.weight_decay;
                }
                matrix.data[index] -= self.learning_rate * m_hat / (@sqrt(v_hat) + self.epsilon);
            },
        }
    }

    fn stateFor(self: *TrainingOptimizer, data: []f64) !*ParamState {
        for (self.states.items) |*state| {
            if (state.ptr == data.ptr and state.len == data.len) {
                return state;
            }
        }

        const m = try self.allocator.alloc(f64, data.len);
        errdefer self.allocator.free(m);
        const v = try self.allocator.alloc(f64, data.len);
        errdefer self.allocator.free(v);
        @memset(m, 0.0);
        @memset(v, 0.0);

        try self.states.append(self.allocator, .{
            .ptr = data.ptr,
            .len = data.len,
            .m = m,
            .v = v,
        });
        return &self.states.items[self.states.items.len - 1];
    }
};

const AttentionForwardCache = struct {
    qkv: Matrix,
    context: Matrix,
    probs: []f64,
    output: Matrix,
    allocator: std.mem.Allocator,

    fn deinit(self: *AttentionForwardCache) void {
        self.qkv.deinit();
        self.context.deinit();
        self.allocator.free(self.probs);
        self.output.deinit();
    }
};

const BlockTrainCache = struct {
    input: Matrix,
    ln1: Matrix,
    qkv: Matrix,
    attn_context: Matrix,
    attn_probs: []f64,
    attn_output: Matrix,
    with_attention: Matrix,
    ln2: Matrix,
    mlp_preact: Matrix,
    mlp_activated: Matrix,
    mlp_output: Matrix,
    output: Matrix,
    allocator: std.mem.Allocator,

    fn deinit(self: *BlockTrainCache) void {
        self.ln1.deinit();
        self.qkv.deinit();
        self.attn_context.deinit();
        self.allocator.free(self.attn_probs);
        self.attn_output.deinit();
        self.with_attention.deinit();
        self.ln2.deinit();
        self.mlp_preact.deinit();
        self.mlp_activated.deinit();
        self.mlp_output.deinit();
        self.output.deinit();
    }
};

const SequenceTrainCache = struct {
    embedding: Matrix,
    blocks: []BlockTrainCache,
    final_input: Matrix,
    final_norm: Matrix,
    logits: Matrix,
    allocator: std.mem.Allocator,

    fn deinit(self: *SequenceTrainCache) void {
        self.logits.deinit();
        self.final_norm.deinit();
        var i = self.blocks.len;
        while (i > 0) {
            i -= 1;
            self.blocks[i].deinit();
        }
        self.allocator.free(self.blocks);
        self.embedding.deinit();
    }
};

fn scheduledLearningRate(options: FullTrainingOptions, step: usize) f64 {
    if (options.steps == 0) return options.learning_rate;

    const current_step = step + 1;
    if (options.warmup_steps > 0 and current_step <= options.warmup_steps) {
        return options.learning_rate * @as(f64, @floatFromInt(current_step)) /
            @as(f64, @floatFromInt(options.warmup_steps));
    }

    const decay_steps = if (options.steps > options.warmup_steps)
        options.steps - options.warmup_steps
    else
        1;
    const schedule_step = if (step >= options.warmup_steps)
        step - options.warmup_steps
    else
        0;
    const progress = if (decay_steps <= 1)
        0.0
    else
        @min(
            @as(f64, @floatFromInt(schedule_step)) / @as(f64, @floatFromInt(decay_steps - 1)),
            1.0,
        );
    const floor = @min(options.min_learning_rate, options.learning_rate);

    return switch (options.lr_schedule) {
        .constant => options.learning_rate,
        .linear => options.learning_rate - (options.learning_rate - floor) * progress,
        .cosine => floor + 0.5 * (options.learning_rate - floor) *
            (1.0 + math.cos(math.pi * progress)),
    };
}

pub fn trainFullOnCorpus(
    model: *TinyGPT,
    corpus: []const u8,
    validation_corpus: ?[]const u8,
    options: FullTrainingOptions,
) !FullTrainingStats {
    const allocator = model.allocator;
    const tokens = try Tokenizer.encode(allocator, corpus);
    defer allocator.free(tokens);
    if (tokens.len < 2) return error.SequenceTooShort;

    const context_len = @min(@max(options.context_len, @as(usize, 1)), @min(model.config.block_size, tokens.len - 1));
    const stride = @max(options.stride, @as(usize, 1));
    const window_count = tokens.len - context_len;
    if (window_count == 0) return error.SequenceTooShort;
    const batch_size = @max(options.batch_size, @as(usize, 1));
    const eval_windows = @max(options.eval_windows, @as(usize, 1));
    const update_count = options.steps * batch_size;

    var validation_tokens: ?[]usize = null;
    defer if (validation_tokens) |items| allocator.free(items);
    var validation_initial_loss: ?f64 = null;
    if (validation_corpus) |validation_text| {
        const encoded = try Tokenizer.encode(allocator, validation_text);
        if (encoded.len > context_len) {
            validation_tokens = encoded;
            validation_initial_loss = try averageCorpusLoss(model, encoded, context_len, eval_windows);
        } else {
            allocator.free(encoded);
        }
    }

    const initial_loss = try averageCorpusLoss(model, tokens, context_len, eval_windows);
    var optimizer_options = options;
    optimizer_options.learning_rate = options.learning_rate / @as(f64, @floatFromInt(batch_size));
    var optimizer = TrainingOptimizer.init(allocator, optimizer_options);
    defer optimizer.deinit();

    var update_index: usize = 0;
    for (0..options.steps) |step| {
        optimizer.learning_rate = scheduledLearningRate(options, step) /
            @as(f64, @floatFromInt(batch_size));
        for (0..batch_size) |batch_item| {
            const start = ((step * batch_size + batch_item) * stride) % window_count;
            optimizer.beginStep(update_index + 1);
            update_index += 1;
            const inputs = tokens[start .. start + context_len];
            const targets = tokens[start + 1 .. start + context_len + 1];
            _ = try trainFullStep(model, inputs, targets, &optimizer);
        }
    }
    const final_loss = try averageCorpusLoss(model, tokens, context_len, eval_windows);
    const validation_final_loss: ?f64 = if (validation_tokens) |items|
        try averageCorpusLoss(model, items, context_len, eval_windows)
    else
        null;

    const final_learning_rate = if (options.steps == 0)
        options.learning_rate
    else
        scheduledLearningRate(options, options.steps - 1);

    return .{
        .steps = options.steps,
        .updates = update_count,
        .batch_size = batch_size,
        .tokens_seen = update_count * context_len,
        .initial_loss = initial_loss,
        .final_loss = final_loss,
        .validation_initial_loss = validation_initial_loss,
        .validation_final_loss = validation_final_loss,
        .learning_rate_initial = if (options.steps == 0) options.learning_rate else scheduledLearningRate(options, 0),
        .learning_rate_final = final_learning_rate,
        .learning_rate_min = @min(options.min_learning_rate, options.learning_rate),
        .eval_windows = eval_windows,
    };
}

fn averageCorpusLoss(
    model: *TinyGPT,
    tokens: []const usize,
    context_len: usize,
    max_windows: usize,
) !f64 {
    if (tokens.len <= context_len) return error.SequenceTooShort;

    const window_count = tokens.len - context_len;
    const samples = @max(@min(max_windows, window_count), @as(usize, 1));
    const stride = @max(window_count / samples, @as(usize, 1));

    var total: f64 = 0.0;
    var count: usize = 0;
    var start: usize = 0;
    while (start < window_count and count < samples) : ({
        start += stride;
        count += 1;
    }) {
        var logits = try model.forward(tokens[start .. start + context_len]);
        defer logits.deinit();
        total += try crossEntropyLoss(logits, tokens[start + 1 .. start + context_len + 1]);
    }

    return total / @as(f64, @floatFromInt(count));
}

fn trainFullStep(
    model: *TinyGPT,
    inputs: []const usize,
    targets: []const usize,
    optimizer: *TrainingOptimizer,
) !f64 {
    if (inputs.len != targets.len) return error.DimensionMismatch;

    var cache = try forwardForTraining(model, inputs);
    defer cache.deinit();

    const loss = try crossEntropyLoss(cache.logits, targets);
    var grad_logits = try crossEntropyGradient(model.allocator, cache.logits, targets);
    defer grad_logits.deinit();

    var grad_final_norm = try linearBackward(
        &model.lm_head,
        cache.final_norm,
        grad_logits,
        model.allocator,
        optimizer,
    );
    defer grad_final_norm.deinit();

    var grad_current = try layerNormBackward(
        &model.ln_f,
        cache.final_input,
        grad_final_norm,
        model.allocator,
        optimizer,
    );
    errdefer grad_current.deinit();

    var i = model.blocks.len;
    while (i > 0) {
        i -= 1;
        const next_grad = try blockBackward(
            &model.blocks[i],
            &cache.blocks[i],
            grad_current,
            model.allocator,
            optimizer,
        );
        grad_current.deinit();
        grad_current = next_grad;
    }

    try applyEmbeddingGradient(model, inputs, grad_current, optimizer);
    grad_current.deinit();

    return loss;
}

fn forwardForTraining(model: *TinyGPT, tokens: []const usize) !SequenceTrainCache {
    var embedding = try model.embed(tokens);
    errdefer embedding.deinit();

    const blocks = try model.allocator.alloc(BlockTrainCache, model.blocks.len);
    var initialized_blocks: usize = 0;
    errdefer {
        var i = initialized_blocks;
        while (i > 0) {
            i -= 1;
            blocks[i].deinit();
        }
        model.allocator.free(blocks);
    }

    var current = embedding;
    for (model.blocks, 0..) |*block, i| {
        blocks[i] = try blockForwardForTraining(block, current, model.allocator);
        initialized_blocks += 1;
        current = blocks[i].output;
    }

    var final_norm = try model.ln_f.forward(current, model.allocator);
    errdefer final_norm.deinit();
    var logits = try model.lm_head.forward(final_norm, model.allocator);
    errdefer logits.deinit();

    return .{
        .embedding = embedding,
        .blocks = blocks,
        .final_input = current,
        .final_norm = final_norm,
        .logits = logits,
        .allocator = model.allocator,
    };
}

fn blockForwardForTraining(
    block: *Block,
    input: Matrix,
    allocator: std.mem.Allocator,
) !BlockTrainCache {
    var ln1 = try block.ln_1.forward(input, allocator);
    errdefer ln1.deinit();

    var attn = try attentionForwardForTraining(&block.attn, ln1, allocator);
    errdefer attn.deinit();

    var with_attention = try input.add(attn.output, allocator);
    errdefer with_attention.deinit();

    var ln2 = try block.ln_2.forward(with_attention, allocator);
    errdefer ln2.deinit();

    var mlp_preact = try block.mlp.c_fc.forward(ln2, allocator);
    errdefer mlp_preact.deinit();

    var mlp_activated = try Matrix.init(allocator, mlp_preact.rows, mlp_preact.cols);
    errdefer mlp_activated.deinit();
    for (mlp_preact.data, 0..) |value, idx| {
        mlp_activated.data[idx] = gelu(value);
    }

    var mlp_output = try block.mlp.c_proj.forward(mlp_activated, allocator);
    errdefer mlp_output.deinit();

    var output = try with_attention.add(mlp_output, allocator);
    errdefer output.deinit();

    return .{
        .input = input,
        .ln1 = ln1,
        .qkv = attn.qkv,
        .attn_context = attn.context,
        .attn_probs = attn.probs,
        .attn_output = attn.output,
        .with_attention = with_attention,
        .ln2 = ln2,
        .mlp_preact = mlp_preact,
        .mlp_activated = mlp_activated,
        .mlp_output = mlp_output,
        .output = output,
        .allocator = allocator,
    };
}

fn attentionForwardForTraining(
    attn: *CausalSelfAttention,
    input: Matrix,
    allocator: std.mem.Allocator,
) !AttentionForwardCache {
    const seq_len = input.rows;
    const head_dim = attn.n_embd / attn.n_head;
    const scale = 1.0 / @sqrt(@as(f64, @floatFromInt(head_dim)));

    var qkv = try attn.c_attn.forward(input, allocator);
    errdefer qkv.deinit();

    var context = try Matrix.init(allocator, seq_len, attn.n_embd);
    errdefer context.deinit();
    context.fill(0.0);

    const probs = try allocator.alloc(f64, attn.n_head * seq_len * seq_len);
    errdefer allocator.free(probs);
    @memset(probs, 0.0);

    var scores = try allocator.alloc(f64, seq_len);
    defer allocator.free(scores);

    for (0..attn.n_head) |head| {
        const head_offset = head * head_dim;
        for (0..seq_len) |query_pos| {
            var max_score = -math.inf(f64);
            for (0..query_pos + 1) |key_pos| {
                var score: f64 = 0.0;
                for (0..head_dim) |d| {
                    const q = qkv.data[query_pos * qkv.cols + head_offset + d];
                    const k = qkv.data[key_pos * qkv.cols + attn.n_embd + head_offset + d];
                    score += q * k;
                }
                score *= scale;
                scores[key_pos] = score;
                max_score = @max(max_score, score);
            }

            var exp_sum: f64 = 0.0;
            for (0..query_pos + 1) |key_pos| {
                const value = math.exp(scores[key_pos] - max_score);
                probs[attentionProbIndex(seq_len, head, query_pos, key_pos)] = value;
                exp_sum += value;
            }

            for (0..query_pos + 1) |key_pos| {
                probs[attentionProbIndex(seq_len, head, query_pos, key_pos)] /= exp_sum;
            }

            for (0..head_dim) |d| {
                var value: f64 = 0.0;
                for (0..query_pos + 1) |key_pos| {
                    const prob = probs[attentionProbIndex(seq_len, head, query_pos, key_pos)];
                    const v = qkv.data[key_pos * qkv.cols + 2 * attn.n_embd + head_offset + d];
                    value += prob * v;
                }
                context.data[query_pos * context.cols + head_offset + d] = value;
            }
        }
    }

    var output = try attn.c_proj.forward(context, allocator);
    errdefer output.deinit();

    return .{
        .qkv = qkv,
        .context = context,
        .probs = probs,
        .output = output,
        .allocator = allocator,
    };
}

fn blockBackward(
    block: *Block,
    cache: *const BlockTrainCache,
    grad_output: Matrix,
    allocator: std.mem.Allocator,
    optimizer: *TrainingOptimizer,
) !Matrix {
    var grad_ln2_out = try mlpBackward(
        &block.mlp,
        cache.ln2,
        cache.mlp_preact,
        cache.mlp_activated,
        grad_output,
        allocator,
        optimizer,
    );
    defer grad_ln2_out.deinit();

    var grad_with_from_ln2 = try layerNormBackward(
        &block.ln_2,
        cache.with_attention,
        grad_ln2_out,
        allocator,
        optimizer,
    );
    defer grad_with_from_ln2.deinit();

    var grad_with_attention = try grad_output.add(grad_with_from_ln2, allocator);
    defer grad_with_attention.deinit();

    var grad_ln1_out = try attentionBackward(
        &block.attn,
        cache.ln1,
        cache.qkv,
        cache.attn_context,
        cache.attn_probs,
        grad_with_attention,
        allocator,
        optimizer,
    );
    defer grad_ln1_out.deinit();

    var grad_input_from_ln1 = try layerNormBackward(
        &block.ln_1,
        cache.input,
        grad_ln1_out,
        allocator,
        optimizer,
    );
    defer grad_input_from_ln1.deinit();

    return grad_with_attention.add(grad_input_from_ln1, allocator);
}

fn mlpBackward(
    mlp: *MLP,
    input: Matrix,
    preact: Matrix,
    activated: Matrix,
    grad_output: Matrix,
    allocator: std.mem.Allocator,
    optimizer: *TrainingOptimizer,
) !Matrix {
    var grad_activated = try linearBackward(&mlp.c_proj, activated, grad_output, allocator, optimizer);
    defer grad_activated.deinit();

    var grad_preact = try Matrix.init(allocator, preact.rows, preact.cols);
    defer grad_preact.deinit();
    for (preact.data, 0..) |value, idx| {
        grad_preact.data[idx] = grad_activated.data[idx] * geluDerivative(value);
    }

    return linearBackward(&mlp.c_fc, input, grad_preact, allocator, optimizer);
}

fn attentionBackward(
    attn: *CausalSelfAttention,
    input: Matrix,
    qkv: Matrix,
    context: Matrix,
    probs: []const f64,
    grad_output: Matrix,
    allocator: std.mem.Allocator,
    optimizer: *TrainingOptimizer,
) !Matrix {
    const seq_len = input.rows;
    const head_dim = attn.n_embd / attn.n_head;
    const scale = 1.0 / @sqrt(@as(f64, @floatFromInt(head_dim)));

    var grad_context = try linearBackward(&attn.c_proj, context, grad_output, allocator, optimizer);
    defer grad_context.deinit();

    var grad_qkv = try Matrix.init(allocator, qkv.rows, qkv.cols);
    defer grad_qkv.deinit();
    grad_qkv.fill(0.0);

    var grad_probs = try allocator.alloc(f64, seq_len);
    defer allocator.free(grad_probs);

    for (0..attn.n_head) |head| {
        const head_offset = head * head_dim;
        for (0..seq_len) |query_pos| {
            var prob_grad_dot: f64 = 0.0;
            for (0..query_pos + 1) |key_pos| {
                var grad_prob: f64 = 0.0;
                for (0..head_dim) |d| {
                    const grad = grad_context.data[query_pos * grad_context.cols + head_offset + d];
                    const value = qkv.data[key_pos * qkv.cols + 2 * attn.n_embd + head_offset + d];
                    grad_prob += grad * value;
                }
                grad_probs[key_pos] = grad_prob;
                prob_grad_dot += grad_prob * probs[attentionProbIndex(seq_len, head, query_pos, key_pos)];
            }

            for (0..query_pos + 1) |key_pos| {
                const prob = probs[attentionProbIndex(seq_len, head, query_pos, key_pos)];
                const grad_score = prob * (grad_probs[key_pos] - prob_grad_dot);
                for (0..head_dim) |d| {
                    const q_index = query_pos * qkv.cols + head_offset + d;
                    const k_index = key_pos * qkv.cols + attn.n_embd + head_offset + d;
                    const v_index = key_pos * qkv.cols + 2 * attn.n_embd + head_offset + d;
                    const grad_context_value = grad_context.data[query_pos * grad_context.cols + head_offset + d];

                    grad_qkv.data[q_index] += grad_score * qkv.data[k_index] * scale;
                    grad_qkv.data[k_index] += grad_score * qkv.data[q_index] * scale;
                    grad_qkv.data[v_index] += grad_context_value * prob;
                }
            }
        }
    }

    return linearBackward(&attn.c_attn, input, grad_qkv, allocator, optimizer);
}

fn linearBackward(
    linear: *Linear,
    input: Matrix,
    grad_output: Matrix,
    allocator: std.mem.Allocator,
    optimizer: *TrainingOptimizer,
) !Matrix {
    if (input.rows != grad_output.rows or input.cols != linear.weights.rows or grad_output.cols != linear.weights.cols) {
        return error.DimensionMismatch;
    }

    var grad_input = try Matrix.init(allocator, input.rows, input.cols);
    errdefer grad_input.deinit();

    for (0..input.rows) |row| {
        for (0..input.cols) |in_col| {
            var sum: f64 = 0.0;
            for (0..grad_output.cols) |out_col| {
                sum += grad_output.data[row * grad_output.cols + out_col] *
                    linear.weights.data[in_col * linear.weights.cols + out_col];
            }
            grad_input.data[row * grad_input.cols + in_col] = sum;
        }
    }

    for (0..linear.weights.rows) |in_col| {
        for (0..linear.weights.cols) |out_col| {
            var grad_sum: f64 = 0.0;
            for (0..input.rows) |row| {
                grad_sum += input.data[row * input.cols + in_col] *
                    grad_output.data[row * grad_output.cols + out_col];
            }
            try optimizer.update(&linear.weights, in_col * linear.weights.cols + out_col, grad_sum);
        }
    }

    for (0..linear.bias.cols) |out_col| {
        var grad_sum: f64 = 0.0;
        for (0..grad_output.rows) |row| {
            grad_sum += grad_output.data[row * grad_output.cols + out_col];
        }
        try optimizer.update(&linear.bias, out_col, grad_sum);
    }

    return grad_input;
}

fn layerNormBackward(
    layer_norm: *LayerNorm,
    input: Matrix,
    grad_output: Matrix,
    allocator: std.mem.Allocator,
    optimizer: *TrainingOptimizer,
) !Matrix {
    if (input.rows != grad_output.rows or input.cols != grad_output.cols or input.cols != layer_norm.weight.cols) {
        return error.DimensionMismatch;
    }

    var grad_input = try Matrix.init(allocator, input.rows, input.cols);
    errdefer grad_input.deinit();

    const grad_weight = try allocator.alloc(f64, input.cols);
    defer allocator.free(grad_weight);
    const grad_bias = try allocator.alloc(f64, input.cols);
    defer allocator.free(grad_bias);
    const xhat = try allocator.alloc(f64, input.cols);
    defer allocator.free(xhat);
    const dy_gamma = try allocator.alloc(f64, input.cols);
    defer allocator.free(dy_gamma);

    @memset(grad_weight, 0.0);
    @memset(grad_bias, 0.0);

    const width = @as(f64, @floatFromInt(input.cols));
    for (0..input.rows) |row| {
        var mean: f64 = 0.0;
        for (0..input.cols) |col| {
            mean += input.data[row * input.cols + col];
        }
        mean /= width;

        var variance: f64 = 0.0;
        for (0..input.cols) |col| {
            const centered = input.data[row * input.cols + col] - mean;
            variance += centered * centered;
        }
        variance /= width;

        const inv_std = 1.0 / @sqrt(variance + layer_norm.eps);
        var sum_dy_gamma: f64 = 0.0;
        var sum_dy_gamma_xhat: f64 = 0.0;

        for (0..input.cols) |col| {
            const idx = row * input.cols + col;
            xhat[col] = (input.data[idx] - mean) * inv_std;
            const grad = grad_output.data[idx];
            grad_weight[col] += grad * xhat[col];
            grad_bias[col] += grad;
            dy_gamma[col] = grad * layer_norm.weight.data[col];
            sum_dy_gamma += dy_gamma[col];
            sum_dy_gamma_xhat += dy_gamma[col] * xhat[col];
        }

        for (0..input.cols) |col| {
            const value = (inv_std / width) *
                (width * dy_gamma[col] - sum_dy_gamma - xhat[col] * sum_dy_gamma_xhat);
            grad_input.data[row * grad_input.cols + col] = value;
        }
    }

    for (0..input.cols) |col| {
        try optimizer.update(&layer_norm.weight, col, grad_weight[col]);
        try optimizer.update(&layer_norm.bias, col, grad_bias[col]);
    }

    return grad_input;
}

fn crossEntropyGradient(
    allocator: std.mem.Allocator,
    logits: Matrix,
    targets: []const usize,
) !Matrix {
    if (logits.rows != targets.len) return error.DimensionMismatch;
    var gradient = try Matrix.init(allocator, logits.rows, logits.cols);
    errdefer gradient.deinit();

    const scale = 1.0 / @as(f64, @floatFromInt(targets.len));
    for (targets, 0..) |target, row| {
        if (target >= logits.cols) return error.TokenOutOfVocabulary;

        var max_logit = -math.inf(f64);
        for (0..logits.cols) |col| {
            max_logit = @max(max_logit, logits.data[row * logits.cols + col]);
        }

        var exp_sum: f64 = 0.0;
        for (0..logits.cols) |col| {
            exp_sum += math.exp(logits.data[row * logits.cols + col] - max_logit);
        }

        for (0..logits.cols) |col| {
            var value = math.exp(logits.data[row * logits.cols + col] - max_logit) / exp_sum;
            if (col == target) value -= 1.0;
            gradient.data[row * gradient.cols + col] = value * scale;
        }
    }

    return gradient;
}

fn applyEmbeddingGradient(
    model: *TinyGPT,
    tokens: []const usize,
    grad_embedding: Matrix,
    optimizer: *TrainingOptimizer,
) !void {
    if (tokens.len != grad_embedding.rows or grad_embedding.cols != model.config.n_embd) {
        return error.DimensionMismatch;
    }

    for (tokens, 0..) |token, pos| {
        if (token >= model.config.vocab_size) return error.TokenOutOfVocabulary;
        for (0..model.config.n_embd) |dim| {
            const grad = grad_embedding.data[pos * grad_embedding.cols + dim];
            try optimizer.update(&model.token_embedding, token * model.token_embedding.cols + dim, grad);
            try optimizer.update(&model.position_embedding, pos * model.position_embedding.cols + dim, grad);
        }
    }
}

fn attentionProbIndex(seq_len: usize, head: usize, query_pos: usize, key_pos: usize) usize {
    return (head * seq_len + query_pos) * seq_len + key_pos;
}

fn parseArgs(args: []const [:0]const u8) !CliOptions {
    var options: CliOptions = .{};
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--prompt")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.prompt = args[i];
        } else if (std.mem.eql(u8, arg, "--tokens")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.tokens = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--temperature")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.temperature = try std.fmt.parseFloat(f64, args[i]);
        } else if (std.mem.eql(u8, arg, "--top-k")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.top_k = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--seed")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.seed = try std.fmt.parseInt(u64, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--block-size")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.model_config.block_size = try std.fmt.parseInt(usize, args[i], 10);
            options.config_overridden = true;
        } else if (std.mem.eql(u8, arg, "--layers") or std.mem.eql(u8, arg, "--n-layer")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.model_config.n_layer = try std.fmt.parseInt(usize, args[i], 10);
            options.config_overridden = true;
        } else if (std.mem.eql(u8, arg, "--heads") or std.mem.eql(u8, arg, "--n-head")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.model_config.n_head = try std.fmt.parseInt(usize, args[i], 10);
            options.config_overridden = true;
        } else if (std.mem.eql(u8, arg, "--embd") or std.mem.eql(u8, arg, "--n-embd") or std.mem.eql(u8, arg, "--embedding-size")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.model_config.n_embd = try std.fmt.parseInt(usize, args[i], 10);
            options.config_overridden = true;
        } else if (std.mem.eql(u8, arg, "--corpus")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.corpus = try parseCorpusPreset(args[i]);
            options.corpus_path = null;
        } else if (std.mem.eql(u8, arg, "--corpus-path")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.corpus_path = args[i];
        } else if (std.mem.eql(u8, arg, "--load-checkpoint")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.checkpoint_in = args[i];
        } else if (std.mem.eql(u8, arg, "--resume-checkpoint")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.checkpoint_in = args[i];
            options.resume_checkpoint = args[i];
        } else if (std.mem.eql(u8, arg, "--save-checkpoint")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.checkpoint_out = args[i];
        } else if (std.mem.eql(u8, arg, "--no-train")) {
            options.train_demo_head = false;
            options.train_full = false;
        } else if (std.mem.eql(u8, arg, "--train-full")) {
            options.train_full = true;
            options.train_demo_head = false;
        } else if (std.mem.eql(u8, arg, "--train-epochs")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.train_epochs = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--learning-rate")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.learning_rate = try std.fmt.parseFloat(f64, args[i]);
        } else if (std.mem.eql(u8, arg, "--full-train-steps")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.full_train_steps = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--full-learning-rate")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.full_learning_rate = try std.fmt.parseFloat(f64, args[i]);
        } else if (std.mem.eql(u8, arg, "--min-learning-rate")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.min_learning_rate = try std.fmt.parseFloat(f64, args[i]);
        } else if (std.mem.eql(u8, arg, "--lr-schedule")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.lr_schedule = try parseLearningRateSchedule(args[i]);
        } else if (std.mem.eql(u8, arg, "--warmup-steps")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.warmup_steps = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--full-train-stride")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.full_train_stride = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--full-batch-size") or std.mem.eql(u8, arg, "--batch-size")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.full_batch_size = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--optimizer")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.full_optimizer = try parseOptimizerKind(args[i]);
        } else if (std.mem.eql(u8, arg, "--weight-decay")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.full_weight_decay = try std.fmt.parseFloat(f64, args[i]);
        } else if (std.mem.eql(u8, arg, "--adam-beta1")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.adam_beta1 = try std.fmt.parseFloat(f64, args[i]);
        } else if (std.mem.eql(u8, arg, "--adam-beta2")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.adam_beta2 = try std.fmt.parseFloat(f64, args[i]);
        } else if (std.mem.eql(u8, arg, "--adam-epsilon")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.adam_epsilon = try std.fmt.parseFloat(f64, args[i]);
        } else if (std.mem.eql(u8, arg, "--train-chars")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.train_chars = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--validation-chars")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.validation_chars = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--eval-chars")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.validation_chars = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--eval-split")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.eval_split = try std.fmt.parseFloat(f64, args[i]);
        } else if (std.mem.eql(u8, arg, "--eval-windows")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.eval_windows = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--prior-chars")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.prior_chars = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--no-corpus-prior")) {
            options.corpus_prior = false;
        } else if (std.mem.eql(u8, arg, "--summary-path")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.summary_path = args[i];
        } else if (std.mem.eql(u8, arg, "--help")) {
            return error.HelpRequested;
        } else {
            return error.UnknownArgument;
        }
    }
    return options;
}

fn parseCorpusPreset(value: []const u8) !CorpusPreset {
    if (std.mem.eql(u8, value, "auto")) return .auto;
    if (std.mem.eql(u8, value, "toy")) return .toy;
    if (std.mem.eql(u8, value, "shakespeare")) return .shakespeare;
    if (std.mem.eql(u8, value, "tinystories")) return .tinystories;
    return error.UnknownCorpusPreset;
}

fn parseOptimizerKind(value: []const u8) !OptimizerKind {
    if (std.mem.eql(u8, value, "sgd")) return .sgd;
    if (std.mem.eql(u8, value, "adamw")) return .adamw;
    return error.UnknownOptimizer;
}

fn parseLearningRateSchedule(value: []const u8) !LearningRateSchedule {
    if (std.mem.eql(u8, value, "constant")) return .constant;
    if (std.mem.eql(u8, value, "linear")) return .linear;
    if (std.mem.eql(u8, value, "cosine")) return .cosine;
    return error.UnknownLearningRateSchedule;
}

fn printUsage(writer: anytype) !void {
    try writer.writeAll(
        \\Usage: zig build run_tiny_gpt -- [options]
        \\
        \\Options:
        \\  --prompt <text>          Prompt to continue (default: "to be")
        \\  --tokens <n>             Number of new tokens to sample (default: 80)
        \\  --temperature <float>    Sampling temperature (default: 1.0)
        \\  --top-k <n>              Restrict sampling to top k tokens (default: 8)
        \\  --seed <n>               Deterministic seed (default: 42)
        \\  --block-size <n>         Context length for new models (default: 16)
        \\  --layers <n>             Transformer block count for new models (default: 2)
        \\  --heads <n>              Attention head count for new models (default: 2)
        \\  --embd <n>               Embedding/channel size for new models (default: 32)
        \\  --corpus <name>          auto, toy, shakespeare, or tinystories
        \\  --corpus-path <path>     Use a custom UTF-8 text corpus
        \\  --load-checkpoint <path> Load a TinyGPT checkpoint before generation/training
        \\  --resume-checkpoint <p>  Load a checkpoint and save back to it unless overridden
        \\  --save-checkpoint <path> Save the model checkpoint after optional training
        \\  --no-train               Skip the tiny demo-corpus output-head training
        \\  --train-full             Train all Transformer weights instead of only the head
        \\  --train-epochs <n>       Output-head training epochs (default: 350)
        \\  --learning-rate <float>  Output-head learning rate (default: 0.5)
        \\  --full-train-steps <n>   Full-model context-window updates (default: 120)
        \\  --full-learning-rate <f> Full-model learning rate (default: 0.03)
        \\  --min-learning-rate <f>  Schedule floor for full training (default: 0)
        \\  --lr-schedule <name>     constant, linear, or cosine (default: constant)
        \\  --warmup-steps <n>       Linear warmup steps before decay (default: 0)
        \\  --full-train-stride <n>  Full-model corpus window stride (default: 1)
        \\  --full-batch-size <n>    Full-model corpus windows per step (default: 1)
        \\  --optimizer <name>       sgd or adamw for full training (default: sgd)
        \\  --weight-decay <f>       Full-training weight decay (default: 0)
        \\  --adam-beta1 <f>         AdamW beta1 (default: 0.9)
        \\  --adam-beta2 <f>         AdamW beta2 (default: 0.999)
        \\  --adam-epsilon <f>       AdamW epsilon (default: 1e-8)
        \\  --train-chars <n>        Corpus prefix for output-head training (default: 512)
        \\  --validation-chars <n>   Corpus holdout chars after the train slice (default: 0)
        \\  --eval-chars <n>         Alias for --validation-chars
        \\  --eval-split <float>     Eval fraction used when validation chars are 0
        \\  --eval-windows <n>       Windows sampled for train/eval loss (default: 12)
        \\  --prior-chars <n>        Corpus prefix for readable sampling prior (default: 32768)
        \\  --no-corpus-prior        Skip the tiny backoff corpus prior used for readable sampling
        \\  --summary-path <path>    Write a deterministic JSON run summary
        \\
        \\Data:
        \\  `--corpus auto` uses prepared TinyStories, then Tiny Shakespeare, then toy.
        \\  Run `nnctl data tiny-gpt` to download the sourced corpora.
        \\
    );
}

fn writeJsonString(writer: anytype, text: []const u8) !void {
    const hex = "0123456789abcdef";
    try writer.writeByte('"');
    for (text) |ch| {
        switch (ch) {
            '"' => try writer.writeAll("\\\""),
            '\\' => try writer.writeAll("\\\\"),
            '\n' => try writer.writeAll("\\n"),
            '\r' => try writer.writeAll("\\r"),
            '\t' => try writer.writeAll("\\t"),
            else => {
                if (ch < 32) {
                    try writer.writeAll("\\u00");
                    try writer.writeByte(hex[ch >> 4]);
                    try writer.writeByte(hex[ch & 0x0f]);
                } else {
                    try writer.writeByte(ch);
                }
            },
        }
    }
    try writer.writeByte('"');
}

fn writeOptionalJsonString(writer: anytype, text: ?[]const u8) !void {
    if (text) |value| {
        try writeJsonString(writer, value);
    } else {
        try writer.writeAll("null");
    }
}

fn writeOptionalJsonFloat(writer: anytype, value: ?f64) !void {
    if (value) |number| {
        try writer.print("{d:.6}", .{number});
    } else {
        try writer.writeAll("null");
    }
}

fn runSummaryJson(
    allocator: std.mem.Allocator,
    model: *const TinyGPT,
    options: CliOptions,
    corpus: Corpus,
    training_corpus: []const u8,
    validation_corpus: ?[]const u8,
    training_stats: ?TrainingStats,
    full_training_stats: ?FullTrainingStats,
    checkpoint_out: ?[]const u8,
    generated_text: []const u8,
) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    errdefer out.deinit();
    const writer = &out.writer;
    const config = model.config;
    const training_mode = if (full_training_stats != null)
        "full"
    else if (training_stats != null)
        "head"
    else
        "none";

    try writer.writeAll("{\n");
    try writer.writeAll("  \"schema\": \"zig-nn.tiny-gpt.run-summary.v1\",\n");
    try writer.print("  \"seed\": {},\n", .{options.seed});
    try writer.writeAll("  \"corpus\": {\n");
    try writer.writeAll("    \"label\": ");
    try writeJsonString(writer, corpus.label);
    try writer.writeAll(",\n    \"path\": ");
    try writeOptionalJsonString(writer, corpus.path);
    try writer.print(",\n    \"chars\": {}\n", .{corpus.text.len});
    try writer.writeAll("  },\n");

    try writer.writeAll("  \"split\": {\n");
    try writer.print("    \"train_chars\": {},\n", .{training_corpus.len});
    try writer.print("    \"eval_chars\": {},\n", .{if (validation_corpus) |text| text.len else 0});
    try writer.print("    \"eval_split\": {d:.6},\n", .{options.eval_split});
    try writer.print("    \"eval_windows\": {}\n", .{options.eval_windows});
    try writer.writeAll("  },\n");

    try writer.writeAll("  \"model\": {\n");
    try writer.print("    \"block_size\": {},\n", .{config.block_size});
    try writer.print("    \"layers\": {},\n", .{config.n_layer});
    try writer.print("    \"heads\": {},\n", .{config.n_head});
    try writer.print("    \"embd\": {},\n", .{config.n_embd});
    try writer.print("    \"vocab_size\": {},\n", .{config.vocab_size});
    try writer.print("    \"parameters\": {}\n", .{model.parameterCount()});
    try writer.writeAll("  },\n");

    try writer.writeAll("  \"training\": {\n");
    try writer.writeAll("    \"mode\": ");
    try writeJsonString(writer, training_mode);
    try writer.print(",\n    \"steps\": {},\n", .{if (full_training_stats) |stats| stats.steps else 0});
    try writer.print("    \"updates\": {},\n", .{if (full_training_stats) |stats| stats.updates else 0});
    try writer.print("    \"batch_size\": {},\n", .{if (full_training_stats) |stats| stats.batch_size else options.full_batch_size});
    try writer.print("    \"tokens_seen\": {},\n", .{if (full_training_stats) |stats| stats.tokens_seen else 0});
    try writer.writeAll("    \"optimizer\": ");
    try writeJsonString(writer, @tagName(options.full_optimizer));
    try writer.writeAll(",\n    \"lr_schedule\": ");
    try writeJsonString(writer, @tagName(options.lr_schedule));
    try writer.print(",\n    \"learning_rate\": {d:.6},\n", .{options.full_learning_rate});
    try writer.print("    \"min_learning_rate\": {d:.6},\n", .{options.min_learning_rate});
    try writer.writeAll("    \"learning_rate_initial\": ");
    try writeOptionalJsonFloat(writer, if (full_training_stats) |stats| stats.learning_rate_initial else null);
    try writer.writeAll(",\n    \"learning_rate_final\": ");
    try writeOptionalJsonFloat(writer, if (full_training_stats) |stats| stats.learning_rate_final else null);
    try writer.print(",\n    \"warmup_steps\": {},\n", .{options.warmup_steps});
    try writer.print("    \"weight_decay\": {d:.6}\n", .{options.full_weight_decay});
    try writer.writeAll("  },\n");

    try writer.writeAll("  \"loss\": {\n");
    if (full_training_stats) |stats| {
        try writer.print("    \"train_initial\": {d:.6},\n", .{stats.initial_loss});
        try writer.print("    \"train_final\": {d:.6},\n", .{stats.final_loss});
        try writer.writeAll("    \"eval_initial\": ");
        try writeOptionalJsonFloat(writer, stats.validation_initial_loss);
        try writer.writeAll(",\n    \"eval_final\": ");
        try writeOptionalJsonFloat(writer, stats.validation_final_loss);
        try writer.writeByte('\n');
    } else if (training_stats) |stats| {
        try writer.print("    \"train_initial\": {d:.6},\n", .{stats.initial_loss});
        try writer.print("    \"train_final\": {d:.6},\n", .{stats.final_loss});
        try writer.writeAll("    \"eval_initial\": null,\n");
        try writer.writeAll("    \"eval_final\": null\n");
    } else {
        try writer.writeAll("    \"train_initial\": null,\n");
        try writer.writeAll("    \"train_final\": null,\n");
        try writer.writeAll("    \"eval_initial\": null,\n");
        try writer.writeAll("    \"eval_final\": null\n");
    }
    try writer.writeAll("  },\n");

    try writer.writeAll("  \"checkpoint\": {\n");
    try writer.writeAll("    \"loaded\": ");
    try writeOptionalJsonString(writer, options.checkpoint_in);
    try writer.writeAll(",\n    \"saved\": ");
    try writeOptionalJsonString(writer, checkpoint_out);
    try writer.writeAll("\n  },\n");

    try writer.writeAll("  \"sample\": {\n");
    try writer.writeAll("    \"prompt\": ");
    try writeJsonString(writer, options.prompt);
    try writer.print(",\n    \"tokens\": {},\n", .{options.tokens});
    try writer.print("    \"temperature\": {d:.6},\n", .{options.temperature});
    try writer.print("    \"top_k\": {},\n", .{options.top_k});
    try writer.writeAll("    \"text\": ");
    try writeJsonString(writer, generated_text);
    try writer.writeAll("\n  }\n");
    try writer.writeAll("}\n");

    return out.toOwnedSlice();
}

fn writeTextFile(path: []const u8, text: []const u8) !void {
    const io = std.Options.debug_io;
    const file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);

    var buffer: [4096]u8 = undefined;
    var file_writer = file.writerStreaming(io, &buffer);
    const writer = &file_writer.interface;
    try writer.writeAll(text);
    try writer.flush();
}

pub fn main(init: std.process.Init) !void {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writerStreaming(std.Options.debug_io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};

    const args = try init.minimal.args.toSlice(init.arena.allocator());

    const options = parseArgs(args[1..]) catch |err| {
        try printUsage(stdout);
        if (err == error.HelpRequested) return;
        return err;
    };
    if (options.eval_split < 0.0 or options.eval_split >= 1.0) return error.InvalidEvalSplit;
    if (options.eval_windows == 0) return error.InvalidEvalWindows;
    if (options.full_learning_rate < 0.0 or options.min_learning_rate < 0.0) return error.InvalidLearningRate;

    if (options.checkpoint_in != null and options.config_overridden) {
        return error.ConfigOverridesCheckpoint;
    }
    const checkpoint_out = if (options.checkpoint_out) |path| path else options.resume_checkpoint;

    var model = if (options.checkpoint_in) |path|
        try TinyGPT.loadFromFile(allocator, path)
    else
        try TinyGPT.init(allocator, options.model_config, options.seed);
    defer model.deinit();
    const config = model.config;

    var corpus = loadCorpus(allocator, options) catch |err| {
        if (err == error.MissingCorpusFile) {
            try stdout.print(
                "Missing corpus file. Run `nnctl data tiny-gpt`, use `--corpus toy`, or pass `--corpus-path <path>`.\n",
                .{},
            );
        }
        return err;
    };
    defer corpus.deinit();

    const eval_chars = resolvedEvalChars(corpus.text.len, options.train_chars, options.validation_chars, options.eval_split);
    const training_corpus = trainingCorpus(corpus.text, options.train_chars, eval_chars);
    const validation_corpus = validationCorpus(corpus.text, options.train_chars, eval_chars);
    const prior_corpus = limitedCorpus(corpus.text, options.prior_chars);

    const training_stats: ?TrainingStats = if (options.train_demo_head and !options.train_full)
        try trainOutputHeadOnCorpus(&model, training_corpus, options.train_epochs, options.learning_rate)
    else
        null;
    const full_training_stats: ?FullTrainingStats = if (options.train_full)
        try trainFullOnCorpus(&model, training_corpus, validation_corpus, .{
            .steps = options.full_train_steps,
            .learning_rate = options.full_learning_rate,
            .min_learning_rate = options.min_learning_rate,
            .lr_schedule = options.lr_schedule,
            .warmup_steps = options.warmup_steps,
            .context_len = config.block_size,
            .stride = options.full_train_stride,
            .batch_size = options.full_batch_size,
            .optimizer = options.full_optimizer,
            .weight_decay = options.full_weight_decay,
            .adam_beta1 = options.adam_beta1,
            .adam_beta2 = options.adam_beta2,
            .adam_epsilon = options.adam_epsilon,
            .eval_windows = options.eval_windows,
        })
    else
        null;

    const generated_tokens = if (options.corpus_prior)
        try model.generateTokensWithCorpusPrior(
            allocator,
            options.prompt,
            options.tokens,
            options.temperature,
            options.top_k,
            prior_corpus,
        )
    else
        try model.generateTokens(
            allocator,
            options.prompt,
            options.tokens,
            options.temperature,
            options.top_k,
        );
    defer allocator.free(generated_tokens);

    const generated_text = try Tokenizer.decode(allocator, generated_tokens);
    defer allocator.free(generated_text);
    const summary_json = try runSummaryJson(
        allocator,
        &model,
        options,
        corpus,
        training_corpus,
        validation_corpus,
        training_stats,
        full_training_stats,
        checkpoint_out,
        generated_text,
    );
    defer allocator.free(summary_json);

    if (checkpoint_out) |path| {
        try model.saveToFileWithMetadata(path, summary_json);
    }
    if (options.summary_path) |path| {
        try writeTextFile(path, summary_json);
    }

    try stdout.print("\n=== Tiny GPT Example ===\n\n", .{});
    try stdout.print("Architecture: {} layers, {} heads, {} channels, block size {}, vocab {}\n", .{
        config.n_layer,
        config.n_head,
        config.n_embd,
        config.block_size,
        config.vocab_size,
    });
    try stdout.print("Parameters: {}\n", .{model.parameterCount()});
    try stdout.print("Seed: {}\n", .{options.seed});
    if (options.checkpoint_in) |path| {
        try stdout.print("Checkpoint loaded: {s}\n", .{path});
    }
    try stdout.print("Corpus: {s}", .{corpus.label});
    if (corpus.path) |path| {
        try stdout.print(" ({s})", .{path});
    }
    try stdout.print(", {} chars loaded\n", .{corpus.text.len});
    try stdout.print("Split: {} train chars, {} eval chars", .{
        training_corpus.len,
        if (validation_corpus) |text| text.len else 0,
    });
    if (options.eval_split > 0.0 and options.validation_chars == 0) {
        try stdout.print(" ({d:.2}% eval split)", .{options.eval_split * 100.0});
    }
    try stdout.print("\n", .{});
    if (full_training_stats) |stats| {
        try stdout.print("Full training: {} steps, {} updates, batch {}, {} tokens seen, {} train chars\n", .{
            stats.steps,
            stats.updates,
            stats.batch_size,
            stats.tokens_seen,
            training_corpus.len,
        });
        try stdout.print("Full optimizer: {s}, learning rate {d:.6}, weight decay {d:.6}\n", .{
            @tagName(options.full_optimizer),
            options.full_learning_rate,
            options.full_weight_decay,
        });
        try stdout.print("LR schedule: {s}, warmup {}, min {d:.6}, final {d:.6}\n", .{
            @tagName(options.lr_schedule),
            options.warmup_steps,
            stats.learning_rate_min,
            stats.learning_rate_final,
        });
        try stdout.print("Full training loss: {d:.4} -> {d:.4}\n", .{ stats.initial_loss, stats.final_loss });
        if (stats.validation_initial_loss) |initial_validation| {
            if (stats.validation_final_loss) |final_validation| {
                try stdout.print("Validation loss: {d:.4} -> {d:.4} ({} sampled windows)\n", .{
                    initial_validation,
                    final_validation,
                    stats.eval_windows,
                });
            }
        } else {
            try stdout.print("Validation loss: unavailable (no eval slice long enough for block size {})\n", .{config.block_size});
        }
    } else if (training_stats) |stats| {
        try stdout.print("Demo training: {} frozen-context samples, {} output-head epochs, {} chars\n", .{
            stats.samples,
            options.train_epochs,
            training_corpus.len,
        });
        try stdout.print("Demo training loss: {d:.4} -> {d:.4}\n", .{ stats.initial_loss, stats.final_loss });
    } else {
        try stdout.print("Demo training: skipped, output head remains random\n", .{});
    }
    if (checkpoint_out) |path| {
        try stdout.print("Checkpoint saved: {s} (metadata embedded)\n", .{path});
    }
    if (options.summary_path) |path| {
        try stdout.print("Run summary saved: {s}\n", .{path});
    }
    try stdout.print("Readable sampling: {s}", .{if (options.corpus_prior) "corpus prior enabled" else "corpus prior disabled"});
    if (options.corpus_prior) {
        try stdout.print(", {} chars\n", .{prior_corpus.len});
    } else {
        try stdout.print("\n", .{});
    }
    try stdout.print("\nPrompt: \"{s}\"\n\n{s}\n", .{ options.prompt, generated_text });

    const prompt_tokens = try Tokenizer.encode(allocator, options.prompt);
    defer allocator.free(prompt_tokens);
    if (prompt_tokens.len > 1 and prompt_tokens.len - 1 <= config.block_size) {
        const targets = try prepareNextTokenTargets(allocator, prompt_tokens);
        defer allocator.free(targets);

        var logits = try model.forward(prompt_tokens[0 .. prompt_tokens.len - 1]);
        defer logits.deinit();

        const loss = try crossEntropyLoss(logits, targets);
        try stdout.print("\nPrompt next-token loss: {d:.4}\n", .{loss});
    }

    if (full_training_stats != null) {
        try stdout.print("\nThe Transformer body, embeddings, layer norms, and output head were updated with manual next-token backpropagation.\n", .{});
    } else if (training_stats != null and options.corpus_prior) {
        try stdout.print("\nThe Transformer body is frozen; the output head and tiny corpus prior make this a readable miniature language demo.\n", .{});
    } else if (training_stats != null) {
        try stdout.print("\nThe Transformer body is frozen; the output head is trained on the tiny built-in corpus for a readable demo.\n", .{});
    } else {
        try stdout.print("\nThis is an untrained decoder-only Transformer. It demonstrates the architecture and sampling path.\n", .{});
    }
}

fn fillNormal(matrix: *Matrix, rand: anytype, stddev: f64) !void {
    for (0..matrix.rows) |row| {
        for (0..matrix.cols) |col| {
            try matrix.set(row, col, randomNormal(rand) * stddev);
        }
    }
}

fn randomNormal(rand: anytype) f64 {
    const sample_a = @max(rand.float(f64), 1e-12);
    const sample_b = rand.float(f64);
    return @sqrt(-2.0 * @log(sample_a)) * @cos(2.0 * math.pi * sample_b);
}

fn gelu(x: f64) f64 {
    const inner = @sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + math.tanh(inner));
}

fn geluDerivative(x: f64) f64 {
    const sqrt_2_over_pi = @sqrt(2.0 / math.pi);
    const inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
    const tanh_inner = math.tanh(inner);
    const inner_derivative = sqrt_2_over_pi * (1.0 + 3.0 * 0.044715 * x * x);
    return 0.5 * (1.0 + tanh_inner) +
        0.5 * x * (1.0 - tanh_inner * tanh_inner) * inner_derivative;
}

fn causalSoftmax(scores: []const f64, query_pos: usize, output: []f64) void {
    var max_score = -math.inf(f64);
    for (0..query_pos + 1) |i| {
        max_score = @max(max_score, scores[i]);
    }

    var sum: f64 = 0.0;
    for (scores, 0..) |score, i| {
        if (i <= query_pos) {
            const value = math.exp(score - max_score);
            output[i] = value;
            sum += value;
        } else {
            output[i] = 0.0;
        }
    }

    for (0..query_pos + 1) |i| {
        output[i] /= sum;
    }
}

fn addCorpusPrior(
    allocator: std.mem.Allocator,
    scores: []f64,
    context: []const usize,
    corpus_tokens: []const usize,
    weight: f64,
) !bool {
    if (context.len == 0 or corpus_tokens.len < 2) return false;

    const max_order = @min(@min(context.len, 12), corpus_tokens.len - 1);
    var counts = try allocator.alloc(usize, scores.len);
    defer allocator.free(counts);

    var order = max_order;
    while (order > 0) : (order -= 1) {
        @memset(counts, 0);
        var total: usize = 0;
        const suffix = context[context.len - order ..];

        for (order..corpus_tokens.len) |pos| {
            if (std.mem.eql(usize, corpus_tokens[pos - order .. pos], suffix)) {
                counts[corpus_tokens[pos]] += 1;
                total += 1;
            }
        }

        if (total == 0) continue;

        const smoothing = 0.001;
        const denom = @as(f64, @floatFromInt(total)) + smoothing * @as(f64, @floatFromInt(scores.len));
        for (scores, 0..) |*score, token| {
            const count = @as(f64, @floatFromInt(counts[token]));
            score.* += weight * @log((count + smoothing) / denom);
        }
        return true;
    }

    return false;
}

fn chooseTopK(scores: []const f64, top_k: usize, allowed: []bool) !void {
    if (scores.len != allowed.len) return error.DimensionMismatch;

    if (top_k == 0 or top_k >= scores.len) {
        @memset(allowed, true);
        return;
    }

    @memset(allowed, false);
    for (0..top_k) |_| {
        var best_index: ?usize = null;
        var best_score = -math.inf(f64);
        for (scores, 0..) |score, i| {
            if (!allowed[i] and score > best_score) {
                best_score = score;
                best_index = i;
            }
        }

        if (best_index) |i| {
            allowed[i] = true;
        }
    }
}

test "tokenizer encode decode round trip" {
    const allocator = testing.allocator;
    const text = "to be, or not!";

    const tokens = try Tokenizer.encode(allocator, text);
    defer allocator.free(tokens);
    const decoded = try Tokenizer.decode(allocator, tokens);
    defer allocator.free(decoded);

    try testing.expectEqualStrings(text, decoded);
}

test "causal mask removes future positions" {
    const scores = [_]f64{ 1.0, 2.0, 100.0, 200.0 };
    var probabilities: [scores.len]f64 = undefined;

    causalSoftmax(&scores, 1, &probabilities);

    try testing.expect(probabilities[0] > 0.0);
    try testing.expect(probabilities[1] > 0.0);
    try testing.expectEqual(@as(f64, 0.0), probabilities[2]);
    try testing.expectEqual(@as(f64, 0.0), probabilities[3]);
    try testing.expectApproxEqAbs(@as(f64, 1.0), probabilities[0] + probabilities[1], 1e-12);
}

test "forward returns sequence by vocab logits" {
    const allocator = testing.allocator;
    const config: Config = .{};
    var model = try TinyGPT.init(allocator, config, 123);
    defer model.deinit();

    const tokens = [_]usize{
        Tokenizer.encodeChar('t'),
        Tokenizer.encodeChar('o'),
        Tokenizer.encodeChar(' '),
        Tokenizer.encodeChar('b'),
        Tokenizer.encodeChar('e'),
    };

    var logits = try model.forward(&tokens);
    defer logits.deinit();

    try testing.expectEqual(tokens.len, logits.rows);
    try testing.expectEqual(Tokenizer.vocabSize(), logits.cols);
}

test "generation stays in vocabulary and appends requested tokens" {
    const allocator = testing.allocator;
    const config: Config = .{};
    var model = try TinyGPT.init(allocator, config, 42);
    defer model.deinit();

    const prompt = "to";
    const new_tokens = 12;
    const generated = try model.generateTokens(allocator, prompt, new_tokens, 1.0, 4);
    defer allocator.free(generated);

    try testing.expectEqual(prompt.len + new_tokens, generated.len);
    for (generated) |token| {
        try testing.expect(token < Tokenizer.vocabSize());
    }
}

test "cross entropy returns finite positive loss" {
    const allocator = testing.allocator;
    const config: Config = .{};
    var model = try TinyGPT.init(allocator, config, 7);
    defer model.deinit();

    const text = "to be";
    const tokens = try Tokenizer.encode(allocator, text);
    defer allocator.free(tokens);
    const targets = try prepareNextTokenTargets(allocator, tokens);
    defer allocator.free(targets);

    var logits = try model.forward(tokens[0 .. tokens.len - 1]);
    defer logits.deinit();

    const loss = try crossEntropyLoss(logits, targets);
    try testing.expect(loss > 0.0);
    try testing.expect(!math.isNan(loss));
    try testing.expect(loss != math.inf(f64));
}

test "demo output head training lowers corpus loss" {
    const allocator = testing.allocator;
    const config: Config = .{
        .block_size = 8,
        .n_layer = 1,
        .n_head = 2,
        .n_embd = 16,
    };
    var model = try TinyGPT.init(allocator, config, 99);
    defer model.deinit();

    const stats = try trainOutputHeadOnCorpus(&model, "to be or to learn.\n", 20, 0.5);

    try testing.expect(stats.samples > 0);
    try testing.expect(stats.final_loss < stats.initial_loss);
}

test "tiny gpt checkpoint round trip preserves logits" {
    const allocator = testing.allocator;
    const config: Config = .{
        .block_size = 8,
        .n_layer = 1,
        .n_head = 2,
        .n_embd = 16,
    };
    var model = try TinyGPT.init(allocator, config, 11);
    defer model.deinit();

    const path = "test_tiny_gpt_checkpoint.bin";
    try model.saveToFile(path);
    defer std.Io.Dir.cwd().deleteFile(std.Options.debug_io, path) catch {};

    var loaded = try TinyGPT.loadFromFile(allocator, path);
    defer loaded.deinit();

    const tokens = [_]usize{
        Tokenizer.encodeChar('t'),
        Tokenizer.encodeChar('o'),
        Tokenizer.encodeChar(' '),
    };

    var original_logits = try model.forward(&tokens);
    defer original_logits.deinit();
    var loaded_logits = try loaded.forward(&tokens);
    defer loaded_logits.deinit();

    try testing.expectEqual(original_logits.rows, loaded_logits.rows);
    try testing.expectEqual(original_logits.cols, loaded_logits.cols);
    for (original_logits.data, 0..) |value, idx| {
        try testing.expectApproxEqAbs(value, loaded_logits.data[idx], 1e-12);
    }
}

test "checkpoint metadata footer is embedded and remains loadable" {
    const allocator = testing.allocator;
    const config: Config = .{
        .block_size = 8,
        .n_layer = 1,
        .n_head = 2,
        .n_embd = 16,
    };
    var model = try TinyGPT.init(allocator, config, 12);
    defer model.deinit();

    const path = "test_tiny_gpt_checkpoint_metadata.bin";
    const metadata = "{\"schema\":\"test-metadata\"}\n";
    try model.saveToFileWithMetadata(path, metadata);
    defer std.Io.Dir.cwd().deleteFile(std.Options.debug_io, path) catch {};

    const bytes = try std.Io.Dir.cwd().readFileAlloc(std.Options.debug_io, path, allocator, .limited(1024 * 1024));
    defer allocator.free(bytes);
    try testing.expect(std.mem.indexOf(u8, bytes, metadata) != null);

    var loaded = try TinyGPT.loadFromFile(allocator, path);
    defer loaded.deinit();
    try testing.expectEqual(model.config.block_size, loaded.config.block_size);
}

test "eval split derives a deterministic holdout slice" {
    const text = "abcdefghijklmnopqrstuvwxyz";
    const eval_chars = resolvedEvalChars(text.len, 10, 0, 0.25);
    const train = trainingCorpus(text, 10, eval_chars);
    const eval = validationCorpus(text, 10, eval_chars).?;

    try testing.expectEqual(@as(usize, 7), eval_chars);
    try testing.expectEqualStrings("abcdefghij", train);
    try testing.expectEqualStrings("klmnopq", eval);
}

test "eval split remains disjoint when train limit covers corpus" {
    const text = "abcdefghijklmnopqrstuvwxyz";
    const eval_chars = resolvedEvalChars(text.len, 999, 0, 0.25);
    const train = trainingCorpus(text, 999, eval_chars);
    const eval = validationCorpus(text, 999, eval_chars).?;

    try testing.expectEqualStrings("abcdefghijklmnopqrs", train);
    try testing.expectEqualStrings("tuvwxyz", eval);
}

test "learning rate schedule reaches warmup peak and final floor" {
    const options: FullTrainingOptions = .{
        .steps = 6,
        .learning_rate = 0.1,
        .min_learning_rate = 0.01,
        .lr_schedule = .linear,
        .warmup_steps = 2,
    };

    try testing.expectApproxEqAbs(0.05, scheduledLearningRate(options, 0), 1e-12);
    try testing.expectApproxEqAbs(0.1, scheduledLearningRate(options, 1), 1e-12);
    try testing.expectApproxEqAbs(0.01, scheduledLearningRate(options, 5), 1e-12);
}

test "full transformer training lowers corpus loss" {
    const allocator = testing.allocator;
    const config: Config = .{
        .block_size = 4,
        .n_layer = 1,
        .n_head = 2,
        .n_embd = 8,
    };
    var model = try TinyGPT.init(allocator, config, 21);
    defer model.deinit();

    const stats = try trainFullOnCorpus(&model, "abababababababab\n", "babababababababa\n", .{
        .steps = 40,
        .learning_rate = 0.05,
        .context_len = config.block_size,
        .stride = 1,
        .batch_size = 2,
        .optimizer = .adamw,
        .weight_decay = 0.01,
        .lr_schedule = .cosine,
        .min_learning_rate = 0.01,
        .eval_windows = 4,
    });

    try testing.expect(stats.tokens_seen > 0);
    try testing.expect(stats.updates == stats.steps * stats.batch_size);
    try testing.expect(stats.final_loss < stats.initial_loss);
    try testing.expect(stats.validation_initial_loss != null);
    try testing.expectEqual(@as(usize, 4), stats.eval_windows);
    try testing.expectApproxEqAbs(0.01, stats.learning_rate_final, 1e-12);
}
