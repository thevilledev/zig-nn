const std = @import("std");
const nn = @import("nn");
const config_mod = @import("config.zig");
const model_mod = @import("model.zig");
const loss_mod = @import("loss.zig");

const Matrix = nn.Matrix;
const math = std.math;

pub const OptimizerKind = config_mod.OptimizerKind;
pub const LearningRateSchedule = config_mod.LearningRateSchedule;
const Tokenizer = config_mod.Tokenizer;
const TinyGPT = model_mod.TinyGPT;
const Linear = model_mod.Linear;
const LayerNorm = model_mod.LayerNorm;
const CausalSelfAttention = model_mod.CausalSelfAttention;
const MLP = model_mod.MLP;
const Block = model_mod.Block;
const gelu = model_mod.gelu;
const geluDerivative = model_mod.geluDerivative;
const crossEntropyLoss = loss_mod.crossEntropyLoss;

pub const TrainingStats = struct {
    samples: usize,
    initial_loss: f64,
    final_loss: f64,
};

pub fn trainOutputHeadOnCorpus(
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

pub fn scheduledLearningRate(options: FullTrainingOptions, step: usize) f64 {
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
