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
    pub const vocab = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:!?'-\n";
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
    train_demo_head: bool = true,
    train_epochs: usize = 350,
    learning_rate: f64 = 0.5,
    corpus_prior: bool = true,
};

const demo_corpus =
    \\to be, or not to be, that is the question.
    \\the tiny model learns from a little play.
    \\to be a small machine is still to speak.
    \\we train the head on clear short lines.
    \\the model says: to be, to learn, to make.
    \\a small transformer can follow a prompt.
    \\to be clear, the demo repeats simple words.
    \\
;

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
        } else if (std.mem.eql(u8, arg, "--no-train")) {
            options.train_demo_head = false;
        } else if (std.mem.eql(u8, arg, "--train-epochs")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.train_epochs = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--learning-rate")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.learning_rate = try std.fmt.parseFloat(f64, args[i]);
        } else if (std.mem.eql(u8, arg, "--no-corpus-prior")) {
            options.corpus_prior = false;
        } else if (std.mem.eql(u8, arg, "--help")) {
            return error.HelpRequested;
        } else {
            return error.UnknownArgument;
        }
    }
    return options;
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
        \\  --no-train               Skip the tiny demo-corpus output-head training
        \\  --train-epochs <n>       Output-head training epochs (default: 350)
        \\  --learning-rate <float>  Output-head learning rate (default: 0.5)
        \\  --no-corpus-prior        Skip the tiny backoff corpus prior used for readable sampling
        \\
    );
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

    const config: Config = .{};
    var model = try TinyGPT.init(allocator, config, options.seed);
    defer model.deinit();

    const training_stats: ?TrainingStats = if (options.train_demo_head)
        try trainOutputHeadOnCorpus(&model, demo_corpus, options.train_epochs, options.learning_rate)
    else
        null;

    const generated_tokens = if (options.corpus_prior)
        try model.generateTokensWithCorpusPrior(
            allocator,
            options.prompt,
            options.tokens,
            options.temperature,
            options.top_k,
            demo_corpus,
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
    if (training_stats) |stats| {
        try stdout.print("Demo training: {} frozen-context samples, {} output-head epochs\n", .{ stats.samples, options.train_epochs });
        try stdout.print("Demo training loss: {d:.4} -> {d:.4}\n", .{ stats.initial_loss, stats.final_loss });
    } else {
        try stdout.print("Demo training: skipped, output head remains random\n", .{});
    }
    try stdout.print("Readable sampling: {s}\n", .{if (options.corpus_prior) "tiny corpus prior enabled" else "corpus prior disabled"});
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

    if (training_stats != null and options.corpus_prior) {
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
