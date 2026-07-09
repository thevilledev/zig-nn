const std = @import("std");
const nn = @import("nn");
const config_mod = @import("config.zig");

const Matrix = nn.Matrix;
const math = std.math;

pub const Config = config_mod.Config;
pub const Tokenizer = config_mod.Tokenizer;
pub const LayerNorm = nn.LayerNorm;

const checkpoint_format_version: u32 = 2;

pub const Linear = struct {
    weights: Matrix,
    bias: Matrix,

    pub fn init(
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

    pub fn deinit(self: *Linear) void {
        self.weights.deinit();
        self.bias.deinit();
    }

    pub fn forward(self: *const Linear, input: Matrix, allocator: std.mem.Allocator) !Matrix {
        var output = try input.dotProduct(self.weights, allocator);
        errdefer output.deinit();

        for (0..output.rows) |row| {
            for (0..output.cols) |col| {
                try output.set(row, col, (try output.get(row, col)) + (try self.bias.get(0, col)));
            }
        }

        return output;
    }

    pub fn parameterCount(self: *const Linear) usize {
        return self.weights.rows * self.weights.cols + self.bias.cols;
    }
};

pub const CausalSelfAttention = struct {
    c_attn: Linear,
    c_proj: Linear,
    n_head: usize,
    n_embd: usize,

    pub fn init(
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

    pub fn deinit(self: *CausalSelfAttention) void {
        self.c_attn.deinit();
        self.c_proj.deinit();
    }

    pub fn forward(self: *CausalSelfAttention, input: Matrix, allocator: std.mem.Allocator) !Matrix {
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

    pub fn parameterCount(self: *const CausalSelfAttention) usize {
        return self.c_attn.parameterCount() + self.c_proj.parameterCount();
    }
};

pub const MLP = struct {
    c_fc: Linear,
    c_proj: Linear,

    pub fn init(
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

    pub fn deinit(self: *MLP) void {
        self.c_fc.deinit();
        self.c_proj.deinit();
    }

    pub fn forward(self: *MLP, input: Matrix, allocator: std.mem.Allocator) !Matrix {
        var hidden = try self.c_fc.forward(input, allocator);
        defer hidden.deinit();

        for (0..hidden.rows) |row| {
            for (0..hidden.cols) |col| {
                try hidden.set(row, col, gelu(try hidden.get(row, col)));
            }
        }

        return self.c_proj.forward(hidden, allocator);
    }

    pub fn parameterCount(self: *const MLP) usize {
        return self.c_fc.parameterCount() + self.c_proj.parameterCount();
    }
};

pub const Block = struct {
    ln_1: LayerNorm,
    attn: CausalSelfAttention,
    ln_2: LayerNorm,
    mlp: MLP,

    pub fn init(allocator: std.mem.Allocator, config: Config, rand: anytype) !Block {
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

    pub fn deinit(self: *Block) void {
        self.ln_1.deinit();
        self.attn.deinit();
        self.ln_2.deinit();
        self.mlp.deinit();
    }

    pub fn forward(self: *Block, input: Matrix, allocator: std.mem.Allocator) !Matrix {
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

    pub fn parameterCount(self: *const Block) usize {
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

    pub fn hidden(self: *TinyGPT, tokens: []const usize) !Matrix {
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

    pub fn toDeviceDecoder(self: *const TinyGPT, context: *nn.ExecutionContext) !nn.Transformer.Decoder {
        var random_source = std.Random.DefaultPrng.init(0);
        var decoder = try nn.Transformer.Decoder.init(context, .{
            .vocabulary_size = self.config.vocab_size,
            .maximum_sequence_length = self.config.block_size,
            .layers = self.config.n_layer,
            .heads = self.config.n_head,
            .channels = self.config.n_embd,
            .hidden_features = self.config.mlpHidden(),
        }, random_source.random());
        errdefer decoder.deinit();

        try matrixToTensor(self.allocator, self.token_embedding, &decoder.token_embedding.weights);
        try matrixToTensor(self.allocator, self.position_embedding, &decoder.position_embedding.weights);
        for (self.blocks, decoder.blocks) |source, *target| {
            try matrixToTensor(self.allocator, source.ln_1.weight, &target.attention_norm.gamma);
            try matrixToTensor(self.allocator, source.ln_1.bias, &target.attention_norm.beta);
            try combinedQkvToDevice(self.allocator, source.attn.c_attn, &target.attention);
            try linearToDevice(self.allocator, source.attn.c_proj, &target.attention.output);
            try matrixToTensor(self.allocator, source.ln_2.weight, &target.feed_forward_norm.gamma);
            try matrixToTensor(self.allocator, source.ln_2.bias, &target.feed_forward_norm.beta);
            try linearToDevice(self.allocator, source.mlp.c_fc, &target.feed_forward.expand);
            try linearToDevice(self.allocator, source.mlp.c_proj, &target.feed_forward.project);
        }
        try matrixToTensor(self.allocator, self.ln_f.weight, &decoder.final_norm.gamma);
        try matrixToTensor(self.allocator, self.ln_f.bias, &decoder.final_norm.beta);
        try linearToDevice(self.allocator, self.lm_head, &decoder.language_model_head);
        return decoder;
    }

    pub fn syncFromDeviceDecoder(self: *TinyGPT, decoder: *const nn.Transformer.Decoder) !void {
        if (decoder.config.vocabulary_size != self.config.vocab_size or
            decoder.config.maximum_sequence_length != self.config.block_size or
            decoder.config.layers != self.config.n_layer or
            decoder.config.heads != self.config.n_head or
            decoder.config.channels != self.config.n_embd)
        {
            return error.DimensionMismatch;
        }
        try tensorToMatrix(self.allocator, decoder.token_embedding.weights, &self.token_embedding);
        try tensorToMatrix(self.allocator, decoder.position_embedding.weights, &self.position_embedding);
        for (self.blocks, decoder.blocks) |*target, source| {
            try tensorToMatrix(self.allocator, source.attention_norm.gamma, &target.ln_1.weight);
            try tensorToMatrix(self.allocator, source.attention_norm.beta, &target.ln_1.bias);
            try combinedQkvFromDevice(self.allocator, source.attention, &target.attn.c_attn);
            try linearFromDevice(self.allocator, source.attention.output, &target.attn.c_proj);
            try tensorToMatrix(self.allocator, source.feed_forward_norm.gamma, &target.ln_2.weight);
            try tensorToMatrix(self.allocator, source.feed_forward_norm.beta, &target.ln_2.bias);
            try linearFromDevice(self.allocator, source.feed_forward.expand, &target.mlp.c_fc);
            try linearFromDevice(self.allocator, source.feed_forward.project, &target.mlp.c_proj);
        }
        try tensorToMatrix(self.allocator, decoder.final_norm.gamma, &self.ln_f.weight);
        try tensorToMatrix(self.allocator, decoder.final_norm.beta, &self.ln_f.bias);
        try linearFromDevice(self.allocator, decoder.language_model_head, &self.lm_head);
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

    pub fn embed(self: *TinyGPT, tokens: []const usize) !Matrix {
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

fn matrixToTensor(allocator: std.mem.Allocator, source: Matrix, target: *nn.Tensor) !void {
    if (source.rows != target.shape.dims[0] or source.cols != target.shape.dims[1]) return error.DimensionMismatch;
    const values = try allocator.alloc(f32, source.data.len);
    defer allocator.free(values);
    for (source.data, values) |source_value, *target_value| target_value.* = @floatCast(source_value);
    try target.writeF32(values);
}

fn tensorToMatrix(allocator: std.mem.Allocator, source: nn.Tensor, target: *Matrix) !void {
    if (target.rows != source.shape.dims[0] or target.cols != source.shape.dims[1]) return error.DimensionMismatch;
    const values = try allocator.alloc(f32, source.elementCount());
    defer allocator.free(values);
    try source.readF32(values);
    for (values, target.data) |source_value, *target_value| target_value.* = source_value;
}

fn linearToDevice(allocator: std.mem.Allocator, source: Linear, target: *nn.Transformer.Linear) !void {
    try matrixToTensor(allocator, source.weights, &target.weights);
    try matrixToTensor(allocator, source.bias, &target.bias);
}

fn linearFromDevice(allocator: std.mem.Allocator, source: nn.Transformer.Linear, target: *Linear) !void {
    try tensorToMatrix(allocator, source.weights, &target.weights);
    try tensorToMatrix(allocator, source.bias, &target.bias);
}

fn combinedQkvToDevice(allocator: std.mem.Allocator, source: Linear, target: *nn.Transformer.CausalSelfAttention) !void {
    const channels = target.channels;
    if (source.weights.rows != channels or source.weights.cols != 3 * channels or source.bias.cols != 3 * channels) {
        return error.DimensionMismatch;
    }
    const matrix_values = try allocator.alloc(f32, channels * channels);
    defer allocator.free(matrix_values);
    const bias_values = try allocator.alloc(f32, channels);
    defer allocator.free(bias_values);
    const projections = [_]*nn.Transformer.Linear{ &target.query, &target.key, &target.value };
    for (projections, 0..) |projection, part| {
        for (0..channels) |row| {
            for (0..channels) |col| {
                matrix_values[row * channels + col] = @floatCast(source.weights.data[row * source.weights.cols + part * channels + col]);
            }
        }
        for (0..channels) |col| bias_values[col] = @floatCast(source.bias.data[part * channels + col]);
        try projection.weights.writeF32(matrix_values);
        try projection.bias.writeF32(bias_values);
    }
}

fn combinedQkvFromDevice(allocator: std.mem.Allocator, source: nn.Transformer.CausalSelfAttention, target: *Linear) !void {
    const channels = source.channels;
    if (target.weights.rows != channels or target.weights.cols != 3 * channels or target.bias.cols != 3 * channels) {
        return error.DimensionMismatch;
    }
    const matrix_values = try allocator.alloc(f32, channels * channels);
    defer allocator.free(matrix_values);
    const bias_values = try allocator.alloc(f32, channels);
    defer allocator.free(bias_values);
    const projections = [_]nn.Transformer.Linear{ source.query, source.key, source.value };
    for (projections, 0..) |projection, part| {
        try projection.weights.readF32(matrix_values);
        try projection.bias.readF32(bias_values);
        for (0..channels) |row| {
            for (0..channels) |col| {
                target.weights.data[row * target.weights.cols + part * channels + col] = matrix_values[row * channels + col];
            }
        }
        for (0..channels) |col| target.bias.data[part * channels + col] = bias_values[col];
    }
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

pub fn gelu(x: f64) f64 {
    const inner = @sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + math.tanh(inner));
}

pub fn geluDerivative(x: f64) f64 {
    const sqrt_2_over_pi = @sqrt(2.0 / math.pi);
    const inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
    const tanh_inner = math.tanh(inner);
    const inner_derivative = sqrt_2_over_pi * (1.0 + 3.0 * 0.044715 * x * x);
    return 0.5 * (1.0 + tanh_inner) +
        0.5 * x * (1.0 - tanh_inner * tanh_inner) * inner_derivative;
}

pub fn causalSoftmax(scores: []const f64, query_pos: usize, output: []f64) void {
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
