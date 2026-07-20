const std = @import("std");
const dimension_utils = @import("dimensions.zig");
const modules = @import("modules.zig");
const tensor = @import("tensor.zig");

const Allocator = std.mem.Allocator;
const ExecutionContext = tensor.ExecutionContext;
const Tensor = tensor.Tensor;

/// Compatibility aliases for callers that used the Transformer namespace.
pub const Linear = modules.Linear;
pub const LinearGradients = modules.LinearGradients;

pub const Sgd = struct {
    learning_rate: f32,

    pub fn init(learning_rate: f32) !Sgd {
        if (learning_rate <= 0 or !std.math.isFinite(learning_rate)) return error.InvalidLearningRate;
        return .{ .learning_rate = learning_rate };
    }

    pub fn stepLinear(self: Sgd, context: *ExecutionContext, linear: *Linear, gradients: LinearGradients) !void {
        if (!linear.weights.shape.eql(gradients.weights.shape) or !linear.bias.shape.eql(gradients.bias.shape)) {
            return error.DimensionMismatch;
        }

        var scaled_weights = try context.scale(gradients.weights, self.learning_rate);
        defer scaled_weights.deinit();
        var scaled_bias = try context.scale(gradients.bias, self.learning_rate);
        defer scaled_bias.deinit();
        var updated_weights = try context.subtract(linear.weights, scaled_weights);
        errdefer updated_weights.deinit();
        const updated_bias = try context.subtract(linear.bias, scaled_bias);

        linear.weights.deinit();
        linear.bias.deinit();
        linear.weights = updated_weights;
        linear.bias = updated_bias;
    }

    pub fn stepTensor(self: Sgd, context: *ExecutionContext, parameter: *Tensor, gradient: Tensor) !void {
        if (!parameter.shape.eql(gradient.shape)) return error.DimensionMismatch;
        var scaled_gradient = try context.scale(gradient, self.learning_rate);
        defer scaled_gradient.deinit();
        const updated = try context.subtract(parameter.*, scaled_gradient);
        parameter.deinit();
        parameter.* = updated;
    }

    pub fn stepLayerNorm(self: Sgd, context: *ExecutionContext, layer_norm: *LayerNorm, gradients: tensor.LayerNormGradients) !void {
        if (!layer_norm.gamma.shape.eql(gradients.gamma.shape) or !layer_norm.beta.shape.eql(gradients.beta.shape)) {
            return error.DimensionMismatch;
        }
        var scaled_gamma = try context.scale(gradients.gamma, self.learning_rate);
        defer scaled_gamma.deinit();
        var scaled_beta = try context.scale(gradients.beta, self.learning_rate);
        defer scaled_beta.deinit();
        var updated_gamma = try context.subtract(layer_norm.gamma, scaled_gamma);
        errdefer updated_gamma.deinit();
        const updated_beta = try context.subtract(layer_norm.beta, scaled_beta);
        layer_norm.gamma.deinit();
        layer_norm.beta.deinit();
        layer_norm.gamma = updated_gamma;
        layer_norm.beta = updated_beta;
    }

    pub fn stepFeedForward(self: Sgd, context: *ExecutionContext, feed_forward: *FeedForward, gradients: FeedForwardGradients) !void {
        try self.stepLinear(context, &feed_forward.expand, .{
            .input = gradients.input,
            .weights = gradients.expand_weights,
            .bias = gradients.expand_bias,
        });
        try self.stepLinear(context, &feed_forward.project, .{
            .input = gradients.input,
            .weights = gradients.project_weights,
            .bias = gradients.project_bias,
        });
    }

    pub fn stepAttention(self: Sgd, context: *ExecutionContext, attention: *CausalSelfAttention, gradients: CausalSelfAttentionGradients) !void {
        try self.stepLinear(context, &attention.query, .{ .input = gradients.input, .weights = gradients.query_weights, .bias = gradients.query_bias });
        try self.stepLinear(context, &attention.key, .{ .input = gradients.input, .weights = gradients.key_weights, .bias = gradients.key_bias });
        try self.stepLinear(context, &attention.value, .{ .input = gradients.input, .weights = gradients.value_weights, .bias = gradients.value_bias });
        try self.stepLinear(context, &attention.output, .{ .input = gradients.input, .weights = gradients.output_weights, .bias = gradients.output_bias });
    }

    pub fn stepCrossAttention(self: Sgd, context: *ExecutionContext, attention: *CrossAttention, gradients: CrossAttentionGradients) !void {
        try self.stepLinear(context, &attention.query, .{ .input = gradients.query_input, .weights = gradients.query_weights, .bias = gradients.query_bias });
        try self.stepLinear(context, &attention.key, .{ .input = gradients.memory_input, .weights = gradients.key_weights, .bias = gradients.key_bias });
        try self.stepLinear(context, &attention.value, .{ .input = gradients.memory_input, .weights = gradients.value_weights, .bias = gradients.value_bias });
        try self.stepLinear(context, &attention.output, .{ .input = gradients.query_input, .weights = gradients.output_weights, .bias = gradients.output_bias });
    }

    pub fn stepBlock(self: Sgd, context: *ExecutionContext, block: *Block, gradients: BlockGradients) !void {
        try self.stepLayerNorm(context, &block.attention_norm, gradients.attention_norm);
        try self.stepAttention(context, &block.attention, gradients.attention);
        try self.stepLayerNorm(context, &block.feed_forward_norm, gradients.feed_forward_norm);
        try self.stepFeedForward(context, &block.feed_forward, gradients.feed_forward);
    }
};

pub const Embedding = struct {
    weights: Tensor,

    pub fn init(context: *ExecutionContext, entries: usize, features: usize, random: std.Random) !Embedding {
        if (entries == 0 or features == 0) return error.InvalidDimension;
        const allocator = context.device.allocator;
        const values = try allocator.alloc(f32, try std.math.mul(usize, entries, features));
        defer allocator.free(values);
        const bound = @sqrt(3.0 / @as(f32, @floatFromInt(features)));
        for (values) |*value| value.* = (random.float(f32) * 2.0 - 1.0) * bound;
        return .{ .weights = try context.upload(&.{ entries, features }, values) };
    }

    pub fn deinit(self: *Embedding) void {
        self.weights.deinit();
        self.* = undefined;
    }

    pub fn forward(self: Embedding, context: *ExecutionContext, indices: Tensor) !Tensor {
        return context.embedding(self.weights, indices);
    }

    pub fn backward(self: Embedding, context: *ExecutionContext, indices: Tensor, output_gradient: Tensor) !Tensor {
        return context.embeddingBackward(indices, output_gradient, self.weights.shape.dims[0]);
    }

    pub fn parameterCount(self: Embedding) usize {
        return self.weights.elementCount();
    }
};

pub const SoftmaxCrossEntropy = struct {
    probabilities: Tensor,
    gradient: Tensor,

    pub fn init(context: *ExecutionContext, logits: Tensor, targets: Tensor) !SoftmaxCrossEntropy {
        if (logits.shape.rank != 2 or !logits.shape.eql(targets.shape)) return error.DimensionMismatch;
        var probabilities = try context.softmax(logits);
        errdefer probabilities.deinit();
        var difference = try context.subtract(probabilities, targets);
        defer difference.deinit();
        const row_count = @as(f32, @floatFromInt(logits.shape.dims[0]));
        const gradient = try context.scale(difference, 1.0 / row_count);
        return .{ .probabilities = probabilities, .gradient = gradient };
    }

    pub fn deinit(self: *SoftmaxCrossEntropy) void {
        self.probabilities.deinit();
        self.gradient.deinit();
        self.* = undefined;
    }

    /// Reads probabilities and one-hot targets after the execution batch has
    /// completed and returns the mean cross-entropy loss.
    pub fn loss(self: SoftmaxCrossEntropy, context: *ExecutionContext, targets: Tensor) !f32 {
        if (!self.probabilities.shape.eql(targets.shape)) return error.DimensionMismatch;
        const allocator = context.device.allocator;
        const probability_values = try allocator.alloc(f32, self.probabilities.elementCount());
        defer allocator.free(probability_values);
        const target_values = try allocator.alloc(f32, targets.elementCount());
        defer allocator.free(target_values);
        try context.readback(self.probabilities, probability_values);
        try context.readback(targets, target_values);

        var total: f32 = 0;
        for (probability_values, target_values) |probability, target| {
            if (target != 0) total -= target * @log(@max(probability, 1e-7));
        }
        return total / @as(f32, @floatFromInt(self.probabilities.shape.dims[0]));
    }
};

/// Trainable affine layer normalization over the final tensor dimension.
pub const LayerNorm = struct {
    gamma: Tensor,
    beta: Tensor,
    epsilon: f32,

    pub fn init(context: *ExecutionContext, features: usize, epsilon: f32) !LayerNorm {
        if (features == 0 or epsilon <= 0) return error.InvalidDimension;
        const allocator = context.device.allocator;
        const values = try allocator.alloc(f32, features);
        defer allocator.free(values);

        @memset(values, 1);
        var gamma = try context.upload(&.{ 1, features }, values);
        errdefer gamma.deinit();
        @memset(values, 0);
        const beta = try context.upload(&.{ 1, features }, values);
        return .{ .gamma = gamma, .beta = beta, .epsilon = epsilon };
    }

    pub fn deinit(self: *LayerNorm) void {
        self.gamma.deinit();
        self.beta.deinit();
        self.* = undefined;
    }

    pub fn forward(self: *const LayerNorm, context: *ExecutionContext, input: Tensor) !Tensor {
        return context.layerNorm(input, self.gamma, self.beta, self.epsilon);
    }

    pub fn backward(self: *const LayerNorm, context: *ExecutionContext, input: Tensor, output_gradient: Tensor) !tensor.LayerNormGradients {
        return context.layerNormBackward(input, self.gamma, output_gradient, self.epsilon);
    }

    pub fn parameterCount(self: LayerNorm) usize {
        return self.gamma.elementCount() + self.beta.elementCount();
    }
};

/// Multi-head causal self-attention with independent Q, K, and V projections.
pub const CausalSelfAttention = struct {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    heads: usize,
    channels: usize,

    pub fn init(context: *ExecutionContext, channels: usize, heads: usize, random: std.Random) !CausalSelfAttention {
        if (heads == 0 or channels == 0 or channels % heads != 0) return error.InvalidHeadCount;
        var query = try Linear.init(context, channels, channels, random);
        errdefer query.deinit();
        var key = try Linear.init(context, channels, channels, random);
        errdefer key.deinit();
        var value = try Linear.init(context, channels, channels, random);
        errdefer value.deinit();
        const output = try Linear.init(context, channels, channels, random);
        return .{
            .query = query,
            .key = key,
            .value = value,
            .output = output,
            .heads = heads,
            .channels = channels,
        };
    }

    pub fn deinit(self: *CausalSelfAttention) void {
        self.query.deinit();
        self.key.deinit();
        self.value.deinit();
        self.output.deinit();
        self.* = undefined;
    }

    pub fn forward(self: *const CausalSelfAttention, context: *ExecutionContext, input: Tensor) !Tensor {
        if (input.shape.rank != 2 or input.shape.dims[1] != self.channels) return error.DimensionMismatch;
        var query = try self.query.forward(context, input);
        defer query.deinit();
        var key = try self.key.forward(context, input);
        defer key.deinit();
        var value = try self.value.forward(context, input);
        defer value.deinit();
        var attended = try context.causalSelfAttention(query, key, value, self.heads);
        defer attended.deinit();
        return self.output.forward(context, attended);
    }

    pub fn backward(self: *const CausalSelfAttention, context: *ExecutionContext, input: Tensor, output_gradient: Tensor) !CausalSelfAttentionGradients {
        if (input.shape.rank != 2 or input.shape.dims[1] != self.channels or !input.shape.eql(output_gradient.shape)) {
            return error.DimensionMismatch;
        }
        var query = try self.query.forward(context, input);
        defer query.deinit();
        var key = try self.key.forward(context, input);
        defer key.deinit();
        var value = try self.value.forward(context, input);
        defer value.deinit();
        var attended = try context.causalSelfAttention(query, key, value, self.heads);
        defer attended.deinit();
        var output_gradients = try self.output.backward(context, attended, output_gradient);
        errdefer output_gradients.deinit();
        var attention_gradients = try context.causalSelfAttentionBackward(query, key, value, output_gradients.input, self.heads);
        defer attention_gradients.deinit();
        var query_gradients = try self.query.backward(context, input, attention_gradients.query);
        errdefer query_gradients.deinit();
        var key_gradients = try self.key.backward(context, input, attention_gradients.key);
        errdefer key_gradients.deinit();
        var value_gradients = try self.value.backward(context, input, attention_gradients.value);
        errdefer value_gradients.deinit();
        var query_key_gradient = try context.add(query_gradients.input, key_gradients.input);
        defer query_key_gradient.deinit();
        const input_gradient = try context.add(query_key_gradient, value_gradients.input);
        output_gradients.input.deinit();
        query_gradients.input.deinit();
        key_gradients.input.deinit();
        value_gradients.input.deinit();
        return .{
            .input = input_gradient,
            .query_weights = query_gradients.weights,
            .query_bias = query_gradients.bias,
            .key_weights = key_gradients.weights,
            .key_bias = key_gradients.bias,
            .value_weights = value_gradients.weights,
            .value_bias = value_gradients.bias,
            .output_weights = output_gradients.weights,
            .output_bias = output_gradients.bias,
        };
    }

    pub fn forwardCached(self: *const CausalSelfAttention, context: *ExecutionContext, input: Tensor, key_cache: *Tensor, value_cache: *Tensor, position: usize) !Tensor {
        if (input.shape.rank != 2 or input.shape.dims[0] != 1 or input.shape.dims[1] != self.channels) return error.DimensionMismatch;
        var query = try self.query.forward(context, input);
        defer query.deinit();
        var key = try self.key.forward(context, input);
        defer key.deinit();
        var value = try self.value.forward(context, input);
        defer value.deinit();
        var attended = try context.cachedSelfAttention(query, key, value, key_cache, value_cache, position, self.heads);
        defer attended.deinit();
        return self.output.forward(context, attended);
    }

    pub fn parameterCount(self: CausalSelfAttention) usize {
        return self.query.parameterCount() + self.key.parameterCount() +
            self.value.parameterCount() + self.output.parameterCount();
    }
};

pub const CausalSelfAttentionGradients = struct {
    input: Tensor,
    query_weights: Tensor,
    query_bias: Tensor,
    key_weights: Tensor,
    key_bias: Tensor,
    value_weights: Tensor,
    value_bias: Tensor,
    output_weights: Tensor,
    output_bias: Tensor,

    pub fn deinit(self: *CausalSelfAttentionGradients) void {
        self.input.deinit();
        self.query_weights.deinit();
        self.query_bias.deinit();
        self.key_weights.deinit();
        self.key_bias.deinit();
        self.value_weights.deinit();
        self.value_bias.deinit();
        self.output_weights.deinit();
        self.output_bias.deinit();
        self.* = undefined;
    }
};

/// Parameter and input gradients produced by unmasked self-attention.
pub const FullSelfAttentionGradients = CausalSelfAttentionGradients;

/// Single-head, unmasked self-attention for bidirectional encoder blocks.
/// The implementation is expressed through Tensor matmuls and softmax so it
/// stays on every supported device without a separate attention kernel.
pub const FullSelfAttention = struct {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    ones: Tensor,
    tokens: usize,
    channels: usize,

    pub fn init(
        context: *ExecutionContext,
        tokens: usize,
        channels: usize,
        random: std.Random,
    ) !FullSelfAttention {
        if (tokens == 0 or channels == 0) return error.InvalidDimension;
        var query = try Linear.init(context, channels, channels, random);
        errdefer query.deinit();
        var key = try Linear.init(context, channels, channels, random);
        errdefer key.deinit();
        var value = try Linear.init(context, channels, channels, random);
        errdefer value.deinit();
        var output = try Linear.init(context, channels, channels, random);
        errdefer output.deinit();
        var ones = try context.createTensor(&.{ 1, tokens });
        ones.fill(1);
        return .{
            .query = query,
            .key = key,
            .value = value,
            .output = output,
            .ones = ones,
            .tokens = tokens,
            .channels = channels,
        };
    }

    pub fn deinit(self: *FullSelfAttention) void {
        self.query.deinit();
        self.key.deinit();
        self.value.deinit();
        self.output.deinit();
        self.ones.deinit();
        self.* = undefined;
    }

    pub fn forward(
        self: *const FullSelfAttention,
        context: *ExecutionContext,
        input: Tensor,
    ) !Tensor {
        try self.validateInput(input);
        var query = try self.query.forward(context, input);
        defer query.deinit();
        var key = try self.key.forward(context, input);
        defer key.deinit();
        var value = try self.value.forward(context, input);
        defer value.deinit();
        var key_transpose = try context.transpose(key);
        defer key_transpose.deinit();
        var scores = try context.matmul(query, key_transpose);
        defer scores.deinit();
        var scaled_scores = try context.scale(
            scores,
            1.0 / @sqrt(@as(f32, @floatFromInt(self.channels))),
        );
        defer scaled_scores.deinit();
        var probabilities = try context.softmax(scaled_scores);
        defer probabilities.deinit();
        var attended = try context.matmul(probabilities, value);
        defer attended.deinit();
        return self.output.forward(context, attended);
    }

    pub fn backward(
        self: *const FullSelfAttention,
        context: *ExecutionContext,
        input: Tensor,
        output_gradient: Tensor,
    ) !FullSelfAttentionGradients {
        try self.validateInput(input);
        if (!input.shape.eql(output_gradient.shape)) return error.DimensionMismatch;
        var query = try self.query.forward(context, input);
        defer query.deinit();
        var key = try self.key.forward(context, input);
        defer key.deinit();
        var value = try self.value.forward(context, input);
        defer value.deinit();
        var key_transpose = try context.transpose(key);
        defer key_transpose.deinit();
        var scores = try context.matmul(query, key_transpose);
        defer scores.deinit();
        const attention_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(self.channels)));
        var scaled_scores = try context.scale(scores, attention_scale);
        defer scaled_scores.deinit();
        var probabilities = try context.softmax(scaled_scores);
        defer probabilities.deinit();
        var attended = try context.matmul(probabilities, value);
        defer attended.deinit();

        var output_gradients = try self.output.backward(context, attended, output_gradient);
        errdefer output_gradients.deinit();
        var value_transpose = try context.transpose(value);
        defer value_transpose.deinit();
        var probability_gradient = try context.matmul(
            output_gradients.input,
            value_transpose,
        );
        defer probability_gradient.deinit();
        var weighted_probability_gradient = try context.multiply(
            probabilities,
            probability_gradient,
        );
        defer weighted_probability_gradient.deinit();
        var weighted_transpose = try context.transpose(weighted_probability_gradient);
        defer weighted_transpose.deinit();
        var row_sums_transpose = try context.sumRows(weighted_transpose);
        defer row_sums_transpose.deinit();
        var row_sums = try context.transpose(row_sums_transpose);
        defer row_sums.deinit();
        var broadcast_row_sums = try context.matmul(row_sums, self.ones);
        defer broadcast_row_sums.deinit();
        var centered_probability_gradient = try context.subtract(
            probability_gradient,
            broadcast_row_sums,
        );
        defer centered_probability_gradient.deinit();
        var unscaled_score_gradient = try context.multiply(
            probabilities,
            centered_probability_gradient,
        );
        defer unscaled_score_gradient.deinit();
        var score_gradient = try context.scale(
            unscaled_score_gradient,
            attention_scale,
        );
        defer score_gradient.deinit();

        var query_gradient = try context.matmul(score_gradient, key);
        defer query_gradient.deinit();
        var score_gradient_transpose = try context.transpose(score_gradient);
        defer score_gradient_transpose.deinit();
        var key_gradient = try context.matmul(score_gradient_transpose, query);
        defer key_gradient.deinit();
        var probabilities_transpose = try context.transpose(probabilities);
        defer probabilities_transpose.deinit();
        var value_gradient = try context.matmul(
            probabilities_transpose,
            output_gradients.input,
        );
        defer value_gradient.deinit();

        var query_gradients = try self.query.backward(context, input, query_gradient);
        errdefer query_gradients.deinit();
        var key_gradients = try self.key.backward(context, input, key_gradient);
        errdefer key_gradients.deinit();
        var value_gradients = try self.value.backward(context, input, value_gradient);
        errdefer value_gradients.deinit();
        var query_key_gradient = try context.add(
            query_gradients.input,
            key_gradients.input,
        );
        defer query_key_gradient.deinit();
        const input_gradient = try context.add(query_key_gradient, value_gradients.input);

        output_gradients.input.deinit();
        query_gradients.input.deinit();
        key_gradients.input.deinit();
        value_gradients.input.deinit();
        return .{
            .input = input_gradient,
            .query_weights = query_gradients.weights,
            .query_bias = query_gradients.bias,
            .key_weights = key_gradients.weights,
            .key_bias = key_gradients.bias,
            .value_weights = value_gradients.weights,
            .value_bias = value_gradients.bias,
            .output_weights = output_gradients.weights,
            .output_bias = output_gradients.bias,
        };
    }

    pub fn parameterCount(self: FullSelfAttention) usize {
        return self.query.parameterCount() + self.key.parameterCount() +
            self.value.parameterCount() + self.output.parameterCount();
    }

    fn validateInput(self: FullSelfAttention, input: Tensor) !void {
        if (input.shape.rank != 2 or input.shape.dims[0] != self.tokens or
            input.shape.dims[1] != self.channels)
        {
            return error.DimensionMismatch;
        }
    }
};

/// Encoder-decoder attention: decoder queries attend to a separate encoder
/// memory. The returned attention matrix has shape
/// `[decoder_tokens, encoder_tokens]`, which makes learned alignments directly
/// inspectable in teaching experiments.
pub const CrossAttention = struct {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    ones: Tensor,
    memory_tokens: usize,
    channels: usize,

    pub fn init(
        context: *ExecutionContext,
        memory_tokens: usize,
        channels: usize,
        random: std.Random,
    ) !CrossAttention {
        if (memory_tokens == 0 or channels == 0) return error.InvalidDimension;
        var query = try Linear.init(context, channels, channels, random);
        errdefer query.deinit();
        var key = try Linear.init(context, channels, channels, random);
        errdefer key.deinit();
        var value = try Linear.init(context, channels, channels, random);
        errdefer value.deinit();
        var output = try Linear.init(context, channels, channels, random);
        errdefer output.deinit();
        var ones = try context.createTensor(&.{ 1, memory_tokens });
        ones.fill(1);
        return .{
            .query = query,
            .key = key,
            .value = value,
            .output = output,
            .ones = ones,
            .memory_tokens = memory_tokens,
            .channels = channels,
        };
    }

    pub fn deinit(self: *CrossAttention) void {
        self.query.deinit();
        self.key.deinit();
        self.value.deinit();
        self.output.deinit();
        self.ones.deinit();
        self.* = undefined;
    }

    pub fn attentionWeights(
        self: *const CrossAttention,
        context: *ExecutionContext,
        decoder_queries: Tensor,
        encoder_memory: Tensor,
    ) !Tensor {
        try self.validateInputs(decoder_queries, encoder_memory);
        var query = try self.query.forward(context, decoder_queries);
        defer query.deinit();
        var key = try self.key.forward(context, encoder_memory);
        defer key.deinit();
        var key_transpose = try context.transpose(key);
        defer key_transpose.deinit();
        var scores = try context.matmul(query, key_transpose);
        defer scores.deinit();
        var scaled_scores = try context.scale(
            scores,
            1.0 / @sqrt(@as(f32, @floatFromInt(self.channels))),
        );
        defer scaled_scores.deinit();
        return context.softmax(scaled_scores);
    }

    pub fn forward(
        self: *const CrossAttention,
        context: *ExecutionContext,
        decoder_queries: Tensor,
        encoder_memory: Tensor,
    ) !Tensor {
        try self.validateInputs(decoder_queries, encoder_memory);
        var probabilities = try self.attentionWeights(context, decoder_queries, encoder_memory);
        defer probabilities.deinit();
        var value = try self.value.forward(context, encoder_memory);
        defer value.deinit();
        var attended = try context.matmul(probabilities, value);
        defer attended.deinit();
        return self.output.forward(context, attended);
    }

    pub fn backward(
        self: *const CrossAttention,
        context: *ExecutionContext,
        decoder_queries: Tensor,
        encoder_memory: Tensor,
        output_gradient: Tensor,
    ) !CrossAttentionGradients {
        try self.validateInputs(decoder_queries, encoder_memory);
        if (!decoder_queries.shape.eql(output_gradient.shape)) return error.DimensionMismatch;
        var query = try self.query.forward(context, decoder_queries);
        defer query.deinit();
        var key = try self.key.forward(context, encoder_memory);
        defer key.deinit();
        var value = try self.value.forward(context, encoder_memory);
        defer value.deinit();
        var key_transpose = try context.transpose(key);
        defer key_transpose.deinit();
        var scores = try context.matmul(query, key_transpose);
        defer scores.deinit();
        const attention_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(self.channels)));
        var scaled_scores = try context.scale(scores, attention_scale);
        defer scaled_scores.deinit();
        var probabilities = try context.softmax(scaled_scores);
        defer probabilities.deinit();
        var attended = try context.matmul(probabilities, value);
        defer attended.deinit();

        var output_gradients = try self.output.backward(context, attended, output_gradient);
        errdefer output_gradients.deinit();
        var value_transpose = try context.transpose(value);
        defer value_transpose.deinit();
        var probability_gradient = try context.matmul(output_gradients.input, value_transpose);
        defer probability_gradient.deinit();
        var weighted_probability_gradient = try context.multiply(probabilities, probability_gradient);
        defer weighted_probability_gradient.deinit();
        var weighted_transpose = try context.transpose(weighted_probability_gradient);
        defer weighted_transpose.deinit();
        var row_sums_transpose = try context.sumRows(weighted_transpose);
        defer row_sums_transpose.deinit();
        var row_sums = try context.transpose(row_sums_transpose);
        defer row_sums.deinit();
        var broadcast_row_sums = try context.matmul(row_sums, self.ones);
        defer broadcast_row_sums.deinit();
        var centered_probability_gradient = try context.subtract(probability_gradient, broadcast_row_sums);
        defer centered_probability_gradient.deinit();
        var unscaled_score_gradient = try context.multiply(probabilities, centered_probability_gradient);
        defer unscaled_score_gradient.deinit();
        var score_gradient = try context.scale(unscaled_score_gradient, attention_scale);
        defer score_gradient.deinit();

        var query_gradient = try context.matmul(score_gradient, key);
        defer query_gradient.deinit();
        var score_gradient_transpose = try context.transpose(score_gradient);
        defer score_gradient_transpose.deinit();
        var key_gradient = try context.matmul(score_gradient_transpose, query);
        defer key_gradient.deinit();
        var probabilities_transpose = try context.transpose(probabilities);
        defer probabilities_transpose.deinit();
        var value_gradient = try context.matmul(probabilities_transpose, output_gradients.input);
        defer value_gradient.deinit();

        var query_gradients = try self.query.backward(context, decoder_queries, query_gradient);
        errdefer query_gradients.deinit();
        var key_gradients = try self.key.backward(context, encoder_memory, key_gradient);
        errdefer key_gradients.deinit();
        var value_gradients = try self.value.backward(context, encoder_memory, value_gradient);
        errdefer value_gradients.deinit();
        const memory_input = try context.add(key_gradients.input, value_gradients.input);

        output_gradients.input.deinit();
        key_gradients.input.deinit();
        value_gradients.input.deinit();
        return .{
            .query_input = query_gradients.input,
            .memory_input = memory_input,
            .query_weights = query_gradients.weights,
            .query_bias = query_gradients.bias,
            .key_weights = key_gradients.weights,
            .key_bias = key_gradients.bias,
            .value_weights = value_gradients.weights,
            .value_bias = value_gradients.bias,
            .output_weights = output_gradients.weights,
            .output_bias = output_gradients.bias,
        };
    }

    pub fn parameterTensorCount(_: CrossAttention) usize {
        return 8;
    }

    pub fn parameterCount(self: CrossAttention) usize {
        return self.query.parameterCount() + self.key.parameterCount() +
            self.value.parameterCount() + self.output.parameterCount();
    }

    pub fn parameters(self: *CrossAttention, output: []*Tensor) ![]*Tensor {
        if (output.len < self.parameterTensorCount()) return error.InsufficientBuffer;
        output[0] = &self.query.weights;
        output[1] = &self.query.bias;
        output[2] = &self.key.weights;
        output[3] = &self.key.bias;
        output[4] = &self.value.weights;
        output[5] = &self.value.bias;
        output[6] = &self.output.weights;
        output[7] = &self.output.bias;
        return output[0..8];
    }

    fn validateInputs(self: CrossAttention, decoder_queries: Tensor, encoder_memory: Tensor) !void {
        if (decoder_queries.shape.rank != 2 or decoder_queries.shape.dims[0] == 0 or decoder_queries.shape.dims[1] != self.channels or
            encoder_memory.shape.rank != 2 or encoder_memory.shape.dims[0] != self.memory_tokens or encoder_memory.shape.dims[1] != self.channels)
        {
            return error.DimensionMismatch;
        }
    }
};

pub const CrossAttentionGradients = struct {
    query_input: Tensor,
    memory_input: Tensor,
    query_weights: Tensor,
    query_bias: Tensor,
    key_weights: Tensor,
    key_bias: Tensor,
    value_weights: Tensor,
    value_bias: Tensor,
    output_weights: Tensor,
    output_bias: Tensor,

    pub fn deinit(self: *CrossAttentionGradients) void {
        self.query_input.deinit();
        self.memory_input.deinit();
        self.query_weights.deinit();
        self.query_bias.deinit();
        self.key_weights.deinit();
        self.key_bias.deinit();
        self.value_weights.deinit();
        self.value_bias.deinit();
        self.output_weights.deinit();
        self.output_bias.deinit();
        self.* = undefined;
    }

    pub fn parameterTensorCount(_: CrossAttentionGradients) usize {
        return 8;
    }

    pub fn parameterGradients(self: CrossAttentionGradients, output: []Tensor) ![]Tensor {
        if (output.len < self.parameterTensorCount()) return error.InsufficientBuffer;
        output[0] = self.query_weights;
        output[1] = self.query_bias;
        output[2] = self.key_weights;
        output[3] = self.key_bias;
        output[4] = self.value_weights;
        output[5] = self.value_bias;
        output[6] = self.output_weights;
        output[7] = self.output_bias;
        return output[0..8];
    }
};

/// Bidirectional multi-head self-attention over padded batches. Callers pass a
/// `[batch * heads, tokens, tokens]` attention mask and a
/// `[batch, tokens, channels]` token mask, keeping mask application explicit
/// and inspectable.
pub const MaskedMultiHeadSelfAttention = struct {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    channels: usize,
    heads: usize,
    head_width: usize,

    pub fn init(context: *ExecutionContext, channels: usize, heads: usize, random: std.Random) !MaskedMultiHeadSelfAttention {
        if (channels == 0 or heads == 0 or channels % heads != 0) return error.InvalidDimension;
        var query = try Linear.init(context, channels, channels, random);
        errdefer query.deinit();
        var key = try Linear.init(context, channels, channels, random);
        errdefer key.deinit();
        var value = try Linear.init(context, channels, channels, random);
        errdefer value.deinit();
        const output = try Linear.init(context, channels, channels, random);
        return .{
            .query = query,
            .key = key,
            .value = value,
            .output = output,
            .channels = channels,
            .heads = heads,
            .head_width = channels / heads,
        };
    }

    pub fn deinit(self: *MaskedMultiHeadSelfAttention) void {
        self.query.deinit();
        self.key.deinit();
        self.value.deinit();
        self.output.deinit();
        self.* = undefined;
    }

    pub fn forward(self: *const MaskedMultiHeadSelfAttention, context: *ExecutionContext, input: Tensor, attention_mask: Tensor, token_mask: Tensor) !Tensor {
        const dimensions = try self.validate(input, attention_mask, token_mask);
        var input_flat = input;
        input_flat.shape = try tensor.Shape.init(&.{ dimensions.flat_tokens, self.channels });
        var query = try self.projectHeads(context, &self.query, input_flat, dimensions.batch, dimensions.tokens);
        defer query.deinit();
        var key = try self.projectHeads(context, &self.key, input_flat, dimensions.batch, dimensions.tokens);
        defer key.deinit();
        var value = try self.projectHeads(context, &self.value, input_flat, dimensions.batch, dimensions.tokens);
        defer value.deinit();
        var scores = try context.batchedMatmul(query, key, false, true);
        defer scores.deinit();
        var scaled_scores = try context.scale(scores, 1.0 / @sqrt(@as(f32, @floatFromInt(self.head_width))));
        defer scaled_scores.deinit();
        var probabilities = try context.maskedSoftmax(scaled_scores, attention_mask);
        defer probabilities.deinit();
        var attended = try context.batchedMatmul(probabilities, value, false, false);
        defer attended.deinit();
        var merged = try context.mergeHeads(attended, dimensions.batch, self.heads);
        defer merged.deinit();
        try merged.reshape(&.{ dimensions.flat_tokens, self.channels });
        var projected = try self.output.forward(context, merged);
        defer projected.deinit();
        try projected.reshape(input.shape.slice());
        return context.multiply(projected, token_mask);
    }

    pub fn backward(self: *const MaskedMultiHeadSelfAttention, context: *ExecutionContext, input: Tensor, attention_mask: Tensor, token_mask: Tensor, output_gradient: Tensor) !FullSelfAttentionGradients {
        const dimensions = try self.validate(input, attention_mask, token_mask);
        if (!input.shape.eql(output_gradient.shape)) return error.DimensionMismatch;
        var input_flat = input;
        input_flat.shape = try tensor.Shape.init(&.{ dimensions.flat_tokens, self.channels });
        var query = try self.projectHeads(context, &self.query, input_flat, dimensions.batch, dimensions.tokens);
        defer query.deinit();
        var key = try self.projectHeads(context, &self.key, input_flat, dimensions.batch, dimensions.tokens);
        defer key.deinit();
        var value = try self.projectHeads(context, &self.value, input_flat, dimensions.batch, dimensions.tokens);
        defer value.deinit();
        var scores = try context.batchedMatmul(query, key, false, true);
        defer scores.deinit();
        const attention_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(self.head_width)));
        var scaled_scores = try context.scale(scores, attention_scale);
        defer scaled_scores.deinit();
        var probabilities = try context.maskedSoftmax(scaled_scores, attention_mask);
        defer probabilities.deinit();
        var attended = try context.batchedMatmul(probabilities, value, false, false);
        defer attended.deinit();
        var merged = try context.mergeHeads(attended, dimensions.batch, self.heads);
        defer merged.deinit();
        try merged.reshape(&.{ dimensions.flat_tokens, self.channels });

        var masked_output_gradient = try context.multiply(output_gradient, token_mask);
        defer masked_output_gradient.deinit();
        try masked_output_gradient.reshape(&.{ dimensions.flat_tokens, self.channels });
        var output_gradients = try self.output.backward(context, merged, masked_output_gradient);
        errdefer output_gradients.deinit();
        try output_gradients.input.reshape(&.{ dimensions.batch, dimensions.tokens, self.heads, self.head_width });
        var attended_gradient = try context.splitHeads(output_gradients.input);
        defer attended_gradient.deinit();
        var probability_gradient = try context.batchedMatmul(attended_gradient, value, false, true);
        defer probability_gradient.deinit();
        var scaled_score_gradient = try context.maskedSoftmaxBackward(probabilities, probability_gradient, attention_mask);
        defer scaled_score_gradient.deinit();
        var score_gradient = try context.scale(scaled_score_gradient, attention_scale);
        defer score_gradient.deinit();
        var query_gradient = try context.batchedMatmul(score_gradient, key, false, false);
        defer query_gradient.deinit();
        var key_gradient = try context.batchedMatmul(score_gradient, query, true, false);
        defer key_gradient.deinit();
        var value_gradient = try context.batchedMatmul(probabilities, attended_gradient, true, false);
        defer value_gradient.deinit();

        var query_merged = try context.mergeHeads(query_gradient, dimensions.batch, self.heads);
        defer query_merged.deinit();
        try query_merged.reshape(&.{ dimensions.flat_tokens, self.channels });
        var key_merged = try context.mergeHeads(key_gradient, dimensions.batch, self.heads);
        defer key_merged.deinit();
        try key_merged.reshape(&.{ dimensions.flat_tokens, self.channels });
        var value_merged = try context.mergeHeads(value_gradient, dimensions.batch, self.heads);
        defer value_merged.deinit();
        try value_merged.reshape(&.{ dimensions.flat_tokens, self.channels });
        var query_gradients = try self.query.backward(context, input_flat, query_merged);
        errdefer query_gradients.deinit();
        var key_gradients = try self.key.backward(context, input_flat, key_merged);
        errdefer key_gradients.deinit();
        var value_gradients = try self.value.backward(context, input_flat, value_merged);
        errdefer value_gradients.deinit();
        var query_key_gradient = try context.add(query_gradients.input, key_gradients.input);
        defer query_key_gradient.deinit();
        var input_gradient_flat = try context.add(query_key_gradient, value_gradients.input);
        defer input_gradient_flat.deinit();
        try input_gradient_flat.reshape(input.shape.slice());
        const input_gradient = try context.multiply(input_gradient_flat, token_mask);

        output_gradients.input.deinit();
        query_gradients.input.deinit();
        key_gradients.input.deinit();
        value_gradients.input.deinit();
        return .{
            .input = input_gradient,
            .query_weights = query_gradients.weights,
            .query_bias = query_gradients.bias,
            .key_weights = key_gradients.weights,
            .key_bias = key_gradients.bias,
            .value_weights = value_gradients.weights,
            .value_bias = value_gradients.bias,
            .output_weights = output_gradients.weights,
            .output_bias = output_gradients.bias,
        };
    }

    pub fn parameterCount(self: MaskedMultiHeadSelfAttention) usize {
        return self.query.parameterCount() + self.key.parameterCount() + self.value.parameterCount() + self.output.parameterCount();
    }

    fn projectHeads(self: MaskedMultiHeadSelfAttention, context: *ExecutionContext, projection: *const Linear, input_flat: Tensor, batch: usize, tokens: usize) !Tensor {
        var projected = try projection.forward(context, input_flat);
        defer projected.deinit();
        try projected.reshape(&.{ batch, tokens, self.heads, self.head_width });
        return context.splitHeads(projected);
    }

    fn validate(self: MaskedMultiHeadSelfAttention, input: Tensor, attention_mask: Tensor, token_mask: Tensor) !struct { batch: usize, tokens: usize, flat_tokens: usize } {
        if (input.shape.rank != 3 or input.shape.dims[2] != self.channels or !input.shape.eql(token_mask.shape)) return error.DimensionMismatch;
        const batch = input.shape.dims[0];
        const tokens = input.shape.dims[1];
        if (attention_mask.shape.rank != 3 or
            !dimension_utils.matches(attention_mask.shape.dims[0], &.{ batch, self.heads }) or
            attention_mask.shape.dims[1] != tokens or
            attention_mask.shape.dims[2] != tokens)
        {
            return error.DimensionMismatch;
        }
        return .{
            .batch = batch,
            .tokens = tokens,
            .flat_tokens = try dimension_utils.product(&.{ batch, tokens }),
        };
    }
};

/// Host-side masks derived from sequence lengths. They are uploaded once and
/// then reused by attention, residual masking, and mean pooling.
pub const PaddingMasks = struct {
    allocator: Allocator,
    attention: []f32,
    tokens: []f32,
    mean_pool: []f32,
    batch: usize,
    maximum_tokens: usize,
    heads: usize,
    channels: usize,

    pub fn init(allocator: Allocator, lengths: []const usize, maximum_tokens: usize, heads: usize, channels: usize) !PaddingMasks {
        if (lengths.len == 0 or maximum_tokens == 0 or heads == 0 or channels == 0) return error.InvalidDimension;
        const attention_count = try dimension_utils.product(&.{ lengths.len, heads, maximum_tokens, maximum_tokens });
        const token_count = try dimension_utils.product(&.{ lengths.len, maximum_tokens, channels });
        const mean_pool_count = try dimension_utils.product(&.{ lengths.len, maximum_tokens });
        const attention = try allocator.alloc(f32, attention_count);
        errdefer allocator.free(attention);
        const tokens = try allocator.alloc(f32, token_count);
        errdefer allocator.free(tokens);
        const mean_pool = try allocator.alloc(f32, mean_pool_count);
        errdefer allocator.free(mean_pool);
        @memset(attention, 0);
        @memset(tokens, 0);
        @memset(mean_pool, 0);
        for (lengths, 0..) |length, batch_index| {
            if (length == 0 or length > maximum_tokens) return error.InvalidSequenceLength;
            const inverse_length = 1.0 / @as(f32, @floatFromInt(length));
            for (0..maximum_tokens) |token| {
                if (token >= length) continue;
                @memset(tokens[(batch_index * maximum_tokens + token) * channels .. (batch_index * maximum_tokens + token + 1) * channels], 1);
                mean_pool[batch_index * maximum_tokens + token] = inverse_length;
            }
            for (0..heads) |head| {
                const head_offset = (batch_index * heads + head) * maximum_tokens * maximum_tokens;
                for (0..length) |query| {
                    @memset(attention[head_offset + query * maximum_tokens .. head_offset + query * maximum_tokens + length], 1);
                }
            }
        }
        return .{
            .allocator = allocator,
            .attention = attention,
            .tokens = tokens,
            .mean_pool = mean_pool,
            .batch = lengths.len,
            .maximum_tokens = maximum_tokens,
            .heads = heads,
            .channels = channels,
        };
    }

    pub fn deinit(self: *PaddingMasks) void {
        self.allocator.free(self.attention);
        self.allocator.free(self.tokens);
        self.allocator.free(self.mean_pool);
        self.* = undefined;
    }
};

test "padding masks reject overflowing dimensions" {
    try std.testing.expectError(
        error.DimensionOverflow,
        PaddingMasks.init(std.testing.allocator, &.{1}, std.math.maxInt(usize), 2, 1),
    );
}

/// Position-wise GELU feed-forward network.
pub const FeedForward = struct {
    expand: Linear,
    project: Linear,
    channels: usize,

    pub fn init(context: *ExecutionContext, channels: usize, hidden_features: usize, random: std.Random) !FeedForward {
        if (channels == 0 or hidden_features == 0) return error.InvalidDimension;
        var expand = try Linear.init(context, channels, hidden_features, random);
        errdefer expand.deinit();
        const project = try Linear.init(context, hidden_features, channels, random);
        return .{ .expand = expand, .project = project, .channels = channels };
    }

    pub fn deinit(self: *FeedForward) void {
        self.expand.deinit();
        self.project.deinit();
        self.* = undefined;
    }

    pub fn forward(self: *const FeedForward, context: *ExecutionContext, input: Tensor) !Tensor {
        if (input.shape.rank != 2 or input.shape.dims[1] != self.channels) return error.DimensionMismatch;
        var activated = try context.linearGelu(input, self.expand.weights, self.expand.bias);
        defer activated.deinit();
        return self.project.forward(context, activated);
    }

    pub fn backward(self: *const FeedForward, context: *ExecutionContext, input: Tensor, output_gradient: Tensor) !FeedForwardGradients {
        if (input.shape.rank != 2 or input.shape.dims[1] != self.channels) return error.DimensionMismatch;
        var expanded = try self.expand.forward(context, input);
        defer expanded.deinit();
        var activated = try context.gelu(expanded);
        defer activated.deinit();
        var project_gradients = try self.project.backward(context, activated, output_gradient);
        errdefer project_gradients.deinit();
        var derivative = try context.geluDerivative(expanded);
        defer derivative.deinit();
        var expanded_gradient = try context.multiply(project_gradients.input, derivative);
        defer expanded_gradient.deinit();
        const expand_gradients = try self.expand.backward(context, input, expanded_gradient);
        project_gradients.input.deinit();
        return .{
            .input = expand_gradients.input,
            .expand_weights = expand_gradients.weights,
            .expand_bias = expand_gradients.bias,
            .project_weights = project_gradients.weights,
            .project_bias = project_gradients.bias,
        };
    }

    pub fn parameterCount(self: FeedForward) usize {
        return self.expand.parameterCount() + self.project.parameterCount();
    }
};

pub const FeedForwardGradients = struct {
    input: Tensor,
    expand_weights: Tensor,
    expand_bias: Tensor,
    project_weights: Tensor,
    project_bias: Tensor,

    pub fn deinit(self: *FeedForwardGradients) void {
        self.input.deinit();
        self.expand_weights.deinit();
        self.expand_bias.deinit();
        self.project_weights.deinit();
        self.project_bias.deinit();
        self.* = undefined;
    }
};

/// Pre-normalization Transformer encoder block with unmasked attention.
pub const EncoderBlock = struct {
    attention_norm: LayerNorm,
    attention: FullSelfAttention,
    feed_forward_norm: LayerNorm,
    feed_forward: FeedForward,
    tokens: usize,
    channels: usize,

    pub fn init(
        context: *ExecutionContext,
        tokens: usize,
        channels: usize,
        hidden_features: usize,
        random: std.Random,
    ) !EncoderBlock {
        var attention_norm = try LayerNorm.init(context, channels, 1e-5);
        errdefer attention_norm.deinit();
        var attention = try FullSelfAttention.init(context, tokens, channels, random);
        errdefer attention.deinit();
        var feed_forward_norm = try LayerNorm.init(context, channels, 1e-5);
        errdefer feed_forward_norm.deinit();
        const feed_forward = try FeedForward.init(
            context,
            channels,
            hidden_features,
            random,
        );
        return .{
            .attention_norm = attention_norm,
            .attention = attention,
            .feed_forward_norm = feed_forward_norm,
            .feed_forward = feed_forward,
            .tokens = tokens,
            .channels = channels,
        };
    }

    pub fn deinit(self: *EncoderBlock) void {
        self.attention_norm.deinit();
        self.attention.deinit();
        self.feed_forward_norm.deinit();
        self.feed_forward.deinit();
        self.* = undefined;
    }

    pub fn forward(
        self: *const EncoderBlock,
        context: *ExecutionContext,
        input: Tensor,
    ) !Tensor {
        try self.validateInput(input);
        var normalized_attention = try self.attention_norm.forward(context, input);
        defer normalized_attention.deinit();
        var attention_output = try self.attention.forward(context, normalized_attention);
        defer attention_output.deinit();
        var attention_residual = try context.add(input, attention_output);
        defer attention_residual.deinit();
        var normalized_feed_forward = try self.feed_forward_norm.forward(
            context,
            attention_residual,
        );
        defer normalized_feed_forward.deinit();
        var feed_forward_output = try self.feed_forward.forward(
            context,
            normalized_feed_forward,
        );
        defer feed_forward_output.deinit();
        return context.add(attention_residual, feed_forward_output);
    }

    pub fn backward(
        self: *const EncoderBlock,
        context: *ExecutionContext,
        input: Tensor,
        output_gradient: Tensor,
    ) !EncoderBlockGradients {
        try self.validateInput(input);
        if (!input.shape.eql(output_gradient.shape)) return error.DimensionMismatch;
        var normalized_attention = try self.attention_norm.forward(context, input);
        defer normalized_attention.deinit();
        var attention_output = try self.attention.forward(context, normalized_attention);
        defer attention_output.deinit();
        var attention_residual = try context.add(input, attention_output);
        defer attention_residual.deinit();
        var normalized_feed_forward = try self.feed_forward_norm.forward(
            context,
            attention_residual,
        );
        defer normalized_feed_forward.deinit();

        const feed_forward_gradients = try self.feed_forward.backward(
            context,
            normalized_feed_forward,
            output_gradient,
        );
        errdefer {
            var owned = feed_forward_gradients;
            owned.deinit();
        }
        const feed_forward_norm_gradients = try self.feed_forward_norm.backward(
            context,
            attention_residual,
            feed_forward_gradients.input,
        );
        errdefer {
            var owned = feed_forward_norm_gradients;
            owned.deinit();
        }
        var attention_residual_gradient = try context.add(
            output_gradient,
            feed_forward_norm_gradients.input,
        );
        defer attention_residual_gradient.deinit();
        const attention_gradients = try self.attention.backward(
            context,
            normalized_attention,
            attention_residual_gradient,
        );
        errdefer {
            var owned = attention_gradients;
            owned.deinit();
        }
        const attention_norm_gradients = try self.attention_norm.backward(
            context,
            input,
            attention_gradients.input,
        );
        errdefer {
            var owned = attention_norm_gradients;
            owned.deinit();
        }
        const input_gradient = try context.add(
            attention_residual_gradient,
            attention_norm_gradients.input,
        );
        return .{
            .input = input_gradient,
            .attention_norm = attention_norm_gradients,
            .attention = attention_gradients,
            .feed_forward_norm = feed_forward_norm_gradients,
            .feed_forward = feed_forward_gradients,
        };
    }

    pub fn parameterTensorCount(_: EncoderBlock) usize {
        return 16;
    }

    pub fn parameterCount(self: EncoderBlock) usize {
        return self.attention_norm.parameterCount() + self.attention.parameterCount() +
            self.feed_forward_norm.parameterCount() + self.feed_forward.parameterCount();
    }

    pub fn parameters(self: *EncoderBlock, output: []*Tensor) ![]*Tensor {
        if (output.len < self.parameterTensorCount()) return error.InsufficientBuffer;
        output[0] = &self.attention_norm.gamma;
        output[1] = &self.attention_norm.beta;
        output[2] = &self.attention.query.weights;
        output[3] = &self.attention.query.bias;
        output[4] = &self.attention.key.weights;
        output[5] = &self.attention.key.bias;
        output[6] = &self.attention.value.weights;
        output[7] = &self.attention.value.bias;
        output[8] = &self.attention.output.weights;
        output[9] = &self.attention.output.bias;
        output[10] = &self.feed_forward_norm.gamma;
        output[11] = &self.feed_forward_norm.beta;
        output[12] = &self.feed_forward.expand.weights;
        output[13] = &self.feed_forward.expand.bias;
        output[14] = &self.feed_forward.project.weights;
        output[15] = &self.feed_forward.project.bias;
        return output[0..16];
    }

    fn validateInput(self: EncoderBlock, input: Tensor) !void {
        if (input.shape.rank != 2 or input.shape.dims[0] != self.tokens or
            input.shape.dims[1] != self.channels)
        {
            return error.DimensionMismatch;
        }
    }
};

/// Dynamic-batch encoder block with padding-aware multi-head attention.
pub const MaskedEncoderBlock = struct {
    attention_norm: LayerNorm,
    attention: MaskedMultiHeadSelfAttention,
    feed_forward_norm: LayerNorm,
    feed_forward: FeedForward,
    channels: usize,

    pub fn init(context: *ExecutionContext, channels: usize, heads: usize, hidden_features: usize, random: std.Random) !MaskedEncoderBlock {
        var attention_norm = try LayerNorm.init(context, channels, 1e-5);
        errdefer attention_norm.deinit();
        var attention = try MaskedMultiHeadSelfAttention.init(context, channels, heads, random);
        errdefer attention.deinit();
        var feed_forward_norm = try LayerNorm.init(context, channels, 1e-5);
        errdefer feed_forward_norm.deinit();
        const feed_forward = try FeedForward.init(context, channels, hidden_features, random);
        return .{
            .attention_norm = attention_norm,
            .attention = attention,
            .feed_forward_norm = feed_forward_norm,
            .feed_forward = feed_forward,
            .channels = channels,
        };
    }

    pub fn deinit(self: *MaskedEncoderBlock) void {
        self.attention_norm.deinit();
        self.attention.deinit();
        self.feed_forward_norm.deinit();
        self.feed_forward.deinit();
        self.* = undefined;
    }

    pub fn forward(self: *const MaskedEncoderBlock, context: *ExecutionContext, input: Tensor, attention_mask: Tensor, token_mask: Tensor) !Tensor {
        const dimensions = try self.validate(input, token_mask);
        var input_flat = input;
        input_flat.shape = try tensor.Shape.init(&.{ dimensions.flat_tokens, self.channels });
        var normalized_attention = try self.attention_norm.forward(context, input_flat);
        defer normalized_attention.deinit();
        try normalized_attention.reshape(input.shape.slice());
        var attention_output = try self.attention.forward(context, normalized_attention, attention_mask, token_mask);
        defer attention_output.deinit();
        var attention_residual_unmasked = try context.add(input, attention_output);
        defer attention_residual_unmasked.deinit();
        var attention_residual = try context.multiply(attention_residual_unmasked, token_mask);
        defer attention_residual.deinit();
        var residual_flat = attention_residual;
        residual_flat.shape = input_flat.shape;
        var normalized_feed_forward = try self.feed_forward_norm.forward(context, residual_flat);
        defer normalized_feed_forward.deinit();
        var feed_forward_output = try self.feed_forward.forward(context, normalized_feed_forward);
        defer feed_forward_output.deinit();
        try feed_forward_output.reshape(input.shape.slice());
        var output_unmasked = try context.add(attention_residual, feed_forward_output);
        defer output_unmasked.deinit();
        return context.multiply(output_unmasked, token_mask);
    }

    pub fn backward(self: *const MaskedEncoderBlock, context: *ExecutionContext, input: Tensor, attention_mask: Tensor, token_mask: Tensor, output_gradient: Tensor) !EncoderBlockGradients {
        const dimensions = try self.validate(input, token_mask);
        if (!input.shape.eql(output_gradient.shape)) return error.DimensionMismatch;
        var input_flat = input;
        input_flat.shape = try tensor.Shape.init(&.{ dimensions.flat_tokens, self.channels });
        var normalized_attention = try self.attention_norm.forward(context, input_flat);
        defer normalized_attention.deinit();
        try normalized_attention.reshape(input.shape.slice());
        var attention_output = try self.attention.forward(context, normalized_attention, attention_mask, token_mask);
        defer attention_output.deinit();
        var attention_residual_unmasked = try context.add(input, attention_output);
        defer attention_residual_unmasked.deinit();
        var attention_residual = try context.multiply(attention_residual_unmasked, token_mask);
        defer attention_residual.deinit();
        var residual_flat = attention_residual;
        residual_flat.shape = input_flat.shape;
        var normalized_feed_forward = try self.feed_forward_norm.forward(context, residual_flat);
        defer normalized_feed_forward.deinit();

        var masked_output_gradient = try context.multiply(output_gradient, token_mask);
        defer masked_output_gradient.deinit();
        var output_gradient_flat = masked_output_gradient;
        output_gradient_flat.shape = input_flat.shape;
        const feed_forward_gradients = try self.feed_forward.backward(context, normalized_feed_forward, output_gradient_flat);
        errdefer {
            var owned = feed_forward_gradients;
            owned.deinit();
        }
        const feed_forward_norm_gradients = try self.feed_forward_norm.backward(context, residual_flat, feed_forward_gradients.input);
        errdefer {
            var owned = feed_forward_norm_gradients;
            owned.deinit();
        }
        var feed_norm_input_view = feed_forward_norm_gradients.input;
        feed_norm_input_view.shape = input.shape;
        var attention_residual_gradient_unmasked = try context.add(masked_output_gradient, feed_norm_input_view);
        defer attention_residual_gradient_unmasked.deinit();
        var attention_residual_gradient = try context.multiply(attention_residual_gradient_unmasked, token_mask);
        defer attention_residual_gradient.deinit();
        const attention_gradients = try self.attention.backward(context, normalized_attention, attention_mask, token_mask, attention_residual_gradient);
        errdefer {
            var owned = attention_gradients;
            owned.deinit();
        }
        var attention_input_view = attention_gradients.input;
        attention_input_view.shape = input_flat.shape;
        const attention_norm_gradients = try self.attention_norm.backward(context, input_flat, attention_input_view);
        errdefer {
            var owned = attention_norm_gradients;
            owned.deinit();
        }
        var norm_input_view = attention_norm_gradients.input;
        norm_input_view.shape = input.shape;
        var input_gradient_unmasked = try context.add(attention_residual_gradient, norm_input_view);
        defer input_gradient_unmasked.deinit();
        const input_gradient = try context.multiply(input_gradient_unmasked, token_mask);
        return .{
            .input = input_gradient,
            .attention_norm = attention_norm_gradients,
            .attention = attention_gradients,
            .feed_forward_norm = feed_forward_norm_gradients,
            .feed_forward = feed_forward_gradients,
        };
    }

    pub fn parameterTensorCount(_: MaskedEncoderBlock) usize {
        return 16;
    }

    pub fn parameterCount(self: MaskedEncoderBlock) usize {
        return self.attention_norm.parameterCount() + self.attention.parameterCount() + self.feed_forward_norm.parameterCount() + self.feed_forward.parameterCount();
    }

    pub fn parameters(self: *MaskedEncoderBlock, output: []*Tensor) ![]*Tensor {
        if (output.len < self.parameterTensorCount()) return error.InsufficientBuffer;
        output[0] = &self.attention_norm.gamma;
        output[1] = &self.attention_norm.beta;
        output[2] = &self.attention.query.weights;
        output[3] = &self.attention.query.bias;
        output[4] = &self.attention.key.weights;
        output[5] = &self.attention.key.bias;
        output[6] = &self.attention.value.weights;
        output[7] = &self.attention.value.bias;
        output[8] = &self.attention.output.weights;
        output[9] = &self.attention.output.bias;
        output[10] = &self.feed_forward_norm.gamma;
        output[11] = &self.feed_forward_norm.beta;
        output[12] = &self.feed_forward.expand.weights;
        output[13] = &self.feed_forward.expand.bias;
        output[14] = &self.feed_forward.project.weights;
        output[15] = &self.feed_forward.project.bias;
        return output[0..16];
    }

    fn validate(self: MaskedEncoderBlock, input: Tensor, token_mask: Tensor) !struct { batch: usize, tokens: usize, flat_tokens: usize } {
        if (input.shape.rank != 3 or input.shape.dims[2] != self.channels or !input.shape.eql(token_mask.shape)) return error.DimensionMismatch;
        return .{
            .batch = input.shape.dims[0],
            .tokens = input.shape.dims[1],
            .flat_tokens = try dimension_utils.product(&.{ input.shape.dims[0], input.shape.dims[1] }),
        };
    }
};

pub const EncoderBlockGradients = struct {
    input: Tensor,
    attention_norm: tensor.LayerNormGradients,
    attention: FullSelfAttentionGradients,
    feed_forward_norm: tensor.LayerNormGradients,
    feed_forward: FeedForwardGradients,

    pub fn deinit(self: *EncoderBlockGradients) void {
        self.input.deinit();
        self.attention_norm.deinit();
        self.attention.deinit();
        self.feed_forward_norm.deinit();
        self.feed_forward.deinit();
        self.* = undefined;
    }

    pub fn parameterTensorCount(_: EncoderBlockGradients) usize {
        return 16;
    }

    pub fn parameterGradients(
        self: EncoderBlockGradients,
        output: []Tensor,
    ) ![]Tensor {
        if (output.len < self.parameterTensorCount()) return error.InsufficientBuffer;
        output[0] = self.attention_norm.gamma;
        output[1] = self.attention_norm.beta;
        output[2] = self.attention.query_weights;
        output[3] = self.attention.query_bias;
        output[4] = self.attention.key_weights;
        output[5] = self.attention.key_bias;
        output[6] = self.attention.value_weights;
        output[7] = self.attention.value_bias;
        output[8] = self.attention.output_weights;
        output[9] = self.attention.output_bias;
        output[10] = self.feed_forward_norm.gamma;
        output[11] = self.feed_forward_norm.beta;
        output[12] = self.feed_forward.expand_weights;
        output[13] = self.feed_forward.expand_bias;
        output[14] = self.feed_forward.project_weights;
        output[15] = self.feed_forward.project_bias;
        return output[0..16];
    }
};

/// Pre-normalization decoder block with causal attention and an MLP.
pub const Block = struct {
    attention_norm: LayerNorm,
    attention: CausalSelfAttention,
    feed_forward_norm: LayerNorm,
    feed_forward: FeedForward,
    channels: usize,

    pub fn init(
        context: *ExecutionContext,
        channels: usize,
        heads: usize,
        hidden_features: usize,
        random: std.Random,
    ) !Block {
        var attention_norm = try LayerNorm.init(context, channels, 1e-5);
        errdefer attention_norm.deinit();
        var attention = try CausalSelfAttention.init(context, channels, heads, random);
        errdefer attention.deinit();
        var feed_forward_norm = try LayerNorm.init(context, channels, 1e-5);
        errdefer feed_forward_norm.deinit();
        const feed_forward = try FeedForward.init(context, channels, hidden_features, random);
        return .{
            .attention_norm = attention_norm,
            .attention = attention,
            .feed_forward_norm = feed_forward_norm,
            .feed_forward = feed_forward,
            .channels = channels,
        };
    }

    pub fn deinit(self: *Block) void {
        self.attention_norm.deinit();
        self.attention.deinit();
        self.feed_forward_norm.deinit();
        self.feed_forward.deinit();
        self.* = undefined;
    }

    pub fn forward(self: *const Block, context: *ExecutionContext, input: Tensor) !Tensor {
        if (input.shape.rank != 2 or input.shape.dims[1] != self.channels) return error.DimensionMismatch;
        var normalized_attention = try self.attention_norm.forward(context, input);
        defer normalized_attention.deinit();
        var attention_output = try self.attention.forward(context, normalized_attention);
        defer attention_output.deinit();
        var attention_residual = try context.add(input, attention_output);
        defer attention_residual.deinit();

        var normalized_feed_forward = try self.feed_forward_norm.forward(context, attention_residual);
        defer normalized_feed_forward.deinit();
        var feed_forward_output = try self.feed_forward.forward(context, normalized_feed_forward);
        defer feed_forward_output.deinit();
        return context.add(attention_residual, feed_forward_output);
    }

    pub fn backward(self: *const Block, context: *ExecutionContext, input: Tensor, output_gradient: Tensor) !BlockGradients {
        if (input.shape.rank != 2 or input.shape.dims[1] != self.channels or !input.shape.eql(output_gradient.shape)) {
            return error.DimensionMismatch;
        }
        var normalized_attention = try self.attention_norm.forward(context, input);
        defer normalized_attention.deinit();
        var attention_output = try self.attention.forward(context, normalized_attention);
        defer attention_output.deinit();
        var attention_residual = try context.add(input, attention_output);
        defer attention_residual.deinit();
        var normalized_feed_forward = try self.feed_forward_norm.forward(context, attention_residual);
        defer normalized_feed_forward.deinit();

        const feed_forward_gradients = try self.feed_forward.backward(context, normalized_feed_forward, output_gradient);
        errdefer {
            var owned = feed_forward_gradients;
            owned.deinit();
        }
        const feed_forward_norm_gradients = try self.feed_forward_norm.backward(context, attention_residual, feed_forward_gradients.input);
        errdefer {
            var owned = feed_forward_norm_gradients;
            owned.deinit();
        }
        var attention_residual_gradient = try context.add(output_gradient, feed_forward_norm_gradients.input);
        defer attention_residual_gradient.deinit();
        const attention_gradients = try self.attention.backward(context, normalized_attention, attention_residual_gradient);
        errdefer {
            var owned = attention_gradients;
            owned.deinit();
        }
        const attention_norm_gradients = try self.attention_norm.backward(context, input, attention_gradients.input);
        errdefer {
            var owned = attention_norm_gradients;
            owned.deinit();
        }
        const input_gradient = try context.add(attention_residual_gradient, attention_norm_gradients.input);

        return .{
            .input = input_gradient,
            .attention_norm = attention_norm_gradients,
            .attention = attention_gradients,
            .feed_forward_norm = feed_forward_norm_gradients,
            .feed_forward = feed_forward_gradients,
        };
    }

    pub fn forwardCached(self: *const Block, context: *ExecutionContext, input: Tensor, key_cache: *Tensor, value_cache: *Tensor, position: usize) !Tensor {
        var normalized_attention = try self.attention_norm.forward(context, input);
        defer normalized_attention.deinit();
        var attention_output = try self.attention.forwardCached(context, normalized_attention, key_cache, value_cache, position);
        defer attention_output.deinit();
        var attention_residual = try context.add(input, attention_output);
        defer attention_residual.deinit();
        var normalized_feed_forward = try self.feed_forward_norm.forward(context, attention_residual);
        defer normalized_feed_forward.deinit();
        var feed_forward_output = try self.feed_forward.forward(context, normalized_feed_forward);
        defer feed_forward_output.deinit();
        return context.add(attention_residual, feed_forward_output);
    }

    pub fn parameterCount(self: Block) usize {
        return self.attention_norm.parameterCount() + self.attention.parameterCount() +
            self.feed_forward_norm.parameterCount() + self.feed_forward.parameterCount();
    }
};

pub const BlockGradients = struct {
    input: Tensor,
    attention_norm: tensor.LayerNormGradients,
    attention: CausalSelfAttentionGradients,
    feed_forward_norm: tensor.LayerNormGradients,
    feed_forward: FeedForwardGradients,

    pub fn deinit(self: *BlockGradients) void {
        self.input.deinit();
        self.attention_norm.deinit();
        self.attention.deinit();
        self.feed_forward_norm.deinit();
        self.feed_forward.deinit();
        self.* = undefined;
    }
};

pub const DecoderConfig = struct {
    vocabulary_size: usize,
    maximum_sequence_length: usize,
    layers: usize,
    heads: usize,
    channels: usize,
    hidden_features: usize,

    pub fn validate(self: DecoderConfig) !void {
        if (self.vocabulary_size == 0 or self.maximum_sequence_length == 0 or self.layers == 0 or
            self.heads == 0 or self.channels == 0 or self.hidden_features == 0)
        {
            return error.InvalidConfig;
        }
        if (self.channels % self.heads != 0) return error.InvalidHeadCount;
    }
};

/// Trainable decoder-only Transformer whose parameters remain on one device.
pub const Decoder = struct {
    allocator: Allocator,
    config: DecoderConfig,
    token_embedding: Embedding,
    position_embedding: Embedding,
    blocks: []Block,
    final_norm: LayerNorm,
    language_model_head: Linear,

    pub fn init(context: *ExecutionContext, config: DecoderConfig, random: std.Random) !Decoder {
        try config.validate();
        const allocator = context.device.allocator;
        var token_embedding = try Embedding.init(context, config.vocabulary_size, config.channels, random);
        errdefer token_embedding.deinit();
        var position_embedding = try Embedding.init(context, config.maximum_sequence_length, config.channels, random);
        errdefer position_embedding.deinit();
        const blocks = try allocator.alloc(Block, config.layers);
        var initialized_blocks: usize = 0;
        errdefer {
            for (blocks[0..initialized_blocks]) |*block| block.deinit();
            allocator.free(blocks);
        }
        for (blocks) |*block| {
            block.* = try Block.init(context, config.channels, config.heads, config.hidden_features, random);
            initialized_blocks += 1;
        }
        var final_norm = try LayerNorm.init(context, config.channels, 1e-5);
        errdefer final_norm.deinit();
        const language_model_head = try Linear.init(context, config.channels, config.vocabulary_size, random);
        return .{
            .allocator = allocator,
            .config = config,
            .token_embedding = token_embedding,
            .position_embedding = position_embedding,
            .blocks = blocks,
            .final_norm = final_norm,
            .language_model_head = language_model_head,
        };
    }

    pub fn deinit(self: *Decoder) void {
        self.token_embedding.deinit();
        self.position_embedding.deinit();
        for (self.blocks) |*block| block.deinit();
        self.allocator.free(self.blocks);
        self.final_norm.deinit();
        self.language_model_head.deinit();
        self.* = undefined;
    }

    pub fn forward(self: *const Decoder, context: *ExecutionContext, tokens: Tensor, positions: Tensor) !Tensor {
        try self.validateIndices(tokens, positions);
        var token_values = try self.token_embedding.forward(context, tokens);
        defer token_values.deinit();
        var position_values = try self.position_embedding.forward(context, positions);
        defer position_values.deinit();
        var current = try context.add(token_values, position_values);
        defer current.deinit();
        for (self.blocks) |*block| {
            const next = try block.forward(context, current);
            current.deinit();
            current = next;
        }
        var normalized = try self.final_norm.forward(context, current);
        defer normalized.deinit();
        return self.language_model_head.forward(context, normalized);
    }

    pub fn trainStep(self: *Decoder, context: *ExecutionContext, tokens: Tensor, positions: Tensor, targets: Tensor, optimizer: Sgd) !f32 {
        try self.validateIndices(tokens, positions);
        if (targets.shape.rank != 2 or targets.shape.dims[0] != tokens.shape.dims[0] or
            targets.shape.dims[1] != self.config.vocabulary_size)
        {
            return error.DimensionMismatch;
        }

        const block_inputs = try self.allocator.alloc(Tensor, self.blocks.len);
        var initialized_inputs: usize = 0;
        defer {
            for (block_inputs[0..initialized_inputs]) |*block_input| block_input.deinit();
            self.allocator.free(block_inputs);
        }
        const block_gradients = try self.allocator.alloc(BlockGradients, self.blocks.len);
        var gradient_start = self.blocks.len;
        defer {
            for (block_gradients[gradient_start..]) |*gradient| gradient.deinit();
            self.allocator.free(block_gradients);
        }

        try context.beginBatch();
        var batch_active = true;
        errdefer if (batch_active) context.endBatch() catch {};
        var token_values = try self.token_embedding.forward(context, tokens);
        defer token_values.deinit();
        var position_values = try self.position_embedding.forward(context, positions);
        defer position_values.deinit();
        var current = try context.add(token_values, position_values);
        var current_owned = true;
        defer if (current_owned) current.deinit();
        for (self.blocks, 0..) |*block, index| {
            block_inputs[index] = current;
            initialized_inputs += 1;
            current_owned = false;
            current = try block.forward(context, block_inputs[index]);
            current_owned = true;
        }
        var normalized = try self.final_norm.forward(context, current);
        defer normalized.deinit();
        var logits = try self.language_model_head.forward(context, normalized);
        defer logits.deinit();
        var objective = try SoftmaxCrossEntropy.init(context, logits, targets);
        defer objective.deinit();
        var head_gradients = try self.language_model_head.backward(context, normalized, objective.gradient);
        defer head_gradients.deinit();
        try optimizer.stepLinear(context, &self.language_model_head, head_gradients);
        var norm_gradients = try self.final_norm.backward(context, current, head_gradients.input);
        defer norm_gradients.deinit();
        try optimizer.stepLayerNorm(context, &self.final_norm, norm_gradients);

        var current_gradient = norm_gradients.input;
        var index = self.blocks.len;
        while (index > 0) {
            index -= 1;
            block_gradients[index] = try self.blocks[index].backward(context, block_inputs[index], current_gradient);
            gradient_start = index;
            try optimizer.stepBlock(context, &self.blocks[index], block_gradients[index]);
            current_gradient = block_gradients[index].input;
        }
        var token_gradient = try self.token_embedding.backward(context, tokens, current_gradient);
        defer token_gradient.deinit();
        var position_gradient = try self.position_embedding.backward(context, positions, current_gradient);
        defer position_gradient.deinit();
        try optimizer.stepTensor(context, &self.token_embedding.weights, token_gradient);
        try optimizer.stepTensor(context, &self.position_embedding.weights, position_gradient);

        try context.endBatch();
        batch_active = false;
        return objective.loss(context, targets);
    }

    pub fn initCache(self: Decoder, context: *ExecutionContext) !DecoderCache {
        const keys = try self.allocator.alloc(Tensor, self.blocks.len);
        var initialized_keys: usize = 0;
        errdefer {
            for (keys[0..initialized_keys]) |*key| key.deinit();
            self.allocator.free(keys);
        }
        const values = try self.allocator.alloc(Tensor, self.blocks.len);
        var initialized_values: usize = 0;
        errdefer {
            for (values[0..initialized_values]) |*value| value.deinit();
            self.allocator.free(values);
        }
        for (keys, values) |*key, *value| {
            key.* = try context.createTensor(&.{ self.config.maximum_sequence_length, self.config.channels });
            initialized_keys += 1;
            key.fill(0);
            value.* = try context.createTensor(&.{ self.config.maximum_sequence_length, self.config.channels });
            initialized_values += 1;
            value.fill(0);
        }
        return .{ .allocator = self.allocator, .keys = keys, .values = values, .position = 0 };
    }

    pub fn forwardToken(self: *const Decoder, context: *ExecutionContext, token: Tensor, cache: *DecoderCache) !Tensor {
        if (token.shape.rank != 2 or token.shape.dims[0] != 1 or token.shape.dims[1] != 1 or
            cache.keys.len != self.blocks.len or cache.values.len != self.blocks.len or
            cache.position >= self.config.maximum_sequence_length)
        {
            return error.DimensionMismatch;
        }
        const position_value = [_]f32{@floatFromInt(cache.position)};
        var position = try context.upload(&.{ 1, 1 }, &position_value);
        defer position.deinit();
        var token_value = try self.token_embedding.forward(context, token);
        defer token_value.deinit();
        var position_embedding = try self.position_embedding.forward(context, position);
        defer position_embedding.deinit();
        var current = try context.add(token_value, position_embedding);
        defer current.deinit();
        for (self.blocks, 0..) |*block, index| {
            const next = try block.forwardCached(context, current, &cache.keys[index], &cache.values[index], cache.position);
            current.deinit();
            current = next;
        }
        var normalized = try self.final_norm.forward(context, current);
        defer normalized.deinit();
        const logits = try self.language_model_head.forward(context, normalized);
        cache.position += 1;
        return logits;
    }

    pub fn parameterCount(self: Decoder) usize {
        var count = self.token_embedding.parameterCount() + self.position_embedding.parameterCount() +
            self.final_norm.parameterCount() + self.language_model_head.parameterCount();
        for (self.blocks) |block| count += block.parameterCount();
        return count;
    }

    fn validateIndices(self: Decoder, tokens: Tensor, positions: Tensor) !void {
        if (tokens.shape.rank != 2 or positions.shape.rank != 2 or tokens.shape.dims[1] != 1 or positions.shape.dims[1] != 1 or
            tokens.shape.dims[0] != positions.shape.dims[0] or tokens.shape.dims[0] > self.config.maximum_sequence_length)
        {
            return error.DimensionMismatch;
        }
    }
};

pub const DecoderCache = struct {
    allocator: Allocator,
    keys: []Tensor,
    values: []Tensor,
    position: usize,

    pub fn deinit(self: *DecoderCache) void {
        for (self.keys) |*key| key.deinit();
        for (self.values) |*value| value.deinit();
        self.allocator.free(self.keys);
        self.allocator.free(self.values);
        self.* = undefined;
    }

    pub fn reset(self: *DecoderCache) void {
        for (self.keys) |*key| key.fill(0);
        for (self.values) |*value| value.fill(0);
        self.position = 0;
    }
};

fn setIdentityLinear2(linear: *Linear) !void {
    try linear.weights.writeF32(&.{
        1, 0,
        0, 1,
    });
    linear.bias.fill(0);
}

test "unmasked attention can use future tokens" {
    const testing = std.testing;
    var device = try tensor.Device.init(testing.allocator, .cpu);
    defer device.deinit();
    var context = ExecutionContext.init(&device);
    var full_prng = std.Random.DefaultPrng.init(1);
    var full = try FullSelfAttention.init(&context, 3, 2, full_prng.random());
    defer full.deinit();
    var causal_prng = std.Random.DefaultPrng.init(1);
    var causal = try CausalSelfAttention.init(&context, 2, 1, causal_prng.random());
    defer causal.deinit();

    try setIdentityLinear2(&full.query);
    try setIdentityLinear2(&full.key);
    try setIdentityLinear2(&full.value);
    try setIdentityLinear2(&full.output);
    try setIdentityLinear2(&causal.query);
    try setIdentityLinear2(&causal.key);
    try setIdentityLinear2(&causal.value);
    try setIdentityLinear2(&causal.output);

    var input_a = try context.upload(&.{ 3, 2 }, &.{
        1, 0,
        0, 1,
        4, 0,
    });
    defer input_a.deinit();
    var input_b = try context.upload(&.{ 3, 2 }, &.{
        1,  0,
        0,  1,
        -4, 0,
    });
    defer input_b.deinit();
    var full_a = try full.forward(&context, input_a);
    defer full_a.deinit();
    var full_b = try full.forward(&context, input_b);
    defer full_b.deinit();
    var causal_a = try causal.forward(&context, input_a);
    defer causal_a.deinit();
    var causal_b = try causal.forward(&context, input_b);
    defer causal_b.deinit();

    var full_a_values: [6]f32 = undefined;
    var full_b_values: [6]f32 = undefined;
    var causal_a_values: [6]f32 = undefined;
    var causal_b_values: [6]f32 = undefined;
    try context.readback(full_a, &full_a_values);
    try context.readback(full_b, &full_b_values);
    try context.readback(causal_a, &causal_a_values);
    try context.readback(causal_b, &causal_b_values);

    try testing.expect(@abs(full_a_values[0] - full_b_values[0]) > 1);
    try testing.expectApproxEqAbs(causal_a_values[0], causal_b_values[0], 1e-6);
    try testing.expectApproxEqAbs(causal_a_values[1], causal_b_values[1], 1e-6);
}

test "unmasked attention input gradients match finite differences" {
    const testing = std.testing;
    var device = try tensor.Device.init(testing.allocator, .cpu);
    defer device.deinit();
    var context = ExecutionContext.init(&device);
    var prng = std.Random.DefaultPrng.init(23);
    var attention = try FullSelfAttention.init(&context, 2, 2, prng.random());
    defer attention.deinit();
    const input_values = [_]f32{ 0.2, -0.35, 0.5, 0.1 };
    var input = try context.upload(&.{ 2, 2 }, &input_values);
    defer input.deinit();
    var output_gradient = try context.upload(&.{ 2, 2 }, &.{ 1, 1, 1, 1 });
    defer output_gradient.deinit();
    var gradients = try attention.backward(&context, input, output_gradient);
    defer gradients.deinit();
    var analytical: [4]f32 = undefined;
    try context.readback(gradients.input, &analytical);

    const epsilon: f32 = 1e-3;
    for (0..input_values.len) |input_index| {
        var plus_values = input_values;
        plus_values[input_index] += epsilon;
        var plus_input = try context.upload(&.{ 2, 2 }, &plus_values);
        defer plus_input.deinit();
        var plus_output = try attention.forward(&context, plus_input);
        defer plus_output.deinit();
        var plus_result: [4]f32 = undefined;
        try context.readback(plus_output, &plus_result);

        var minus_values = input_values;
        minus_values[input_index] -= epsilon;
        var minus_input = try context.upload(&.{ 2, 2 }, &minus_values);
        defer minus_input.deinit();
        var minus_output = try attention.forward(&context, minus_input);
        defer minus_output.deinit();
        var minus_result: [4]f32 = undefined;
        try context.readback(minus_output, &minus_result);

        const plus_sum = plus_result[0] + plus_result[1] + plus_result[2] + plus_result[3];
        const minus_sum = minus_result[0] + minus_result[1] + minus_result[2] + minus_result[3];
        const numerical = (plus_sum - minus_sum) / (2 * epsilon);
        try testing.expectApproxEqAbs(numerical, analytical[input_index], 3e-3);
    }
}

test "cross attention exposes decoder-to-encoder alignment" {
    const testing = std.testing;
    const preferences = [_]tensor.DevicePreference{ .cpu, .metal, .cuda, .rocm };
    for (preferences) |preference| {
        var device = tensor.Device.init(testing.allocator, preference) catch |err| switch (err) {
            error.BackendUnavailable => continue,
            else => return err,
        };
        defer device.deinit();
        var context = ExecutionContext.init(&device);
        var prng = std.Random.DefaultPrng.init(31);
        var attention = try CrossAttention.init(&context, 3, 2, prng.random());
        defer attention.deinit();
        try setIdentityLinear2(&attention.query);
        try setIdentityLinear2(&attention.key);
        try setIdentityLinear2(&attention.value);
        try setIdentityLinear2(&attention.output);

        var queries = try context.upload(&.{ 2, 2 }, &.{ 6, 0, 0, 6 });
        defer queries.deinit();
        var memory = try context.upload(&.{ 3, 2 }, &.{ 1, 0, 0, 1, -1, 0 });
        defer memory.deinit();
        context.resetStats();
        var weights = try attention.attentionWeights(&context, queries, memory);
        defer weights.deinit();
        var output = try attention.forward(&context, queries, memory);
        defer output.deinit();
        try testing.expectEqual(@as(usize, 0), context.stats.readbacks);

        var weight_values: [6]f32 = undefined;
        var output_values: [4]f32 = undefined;
        try context.readback(weights, &weight_values);
        try context.readback(output, &output_values);
        try testing.expectEqualSlices(usize, &.{ 2, 3 }, weights.shape.slice());
        try testing.expect(weight_values[0] > 0.95);
        try testing.expect(weight_values[4] > 0.95);
        try testing.expect(output_values[0] > 0.9);
        try testing.expect(output_values[3] > 0.9);
    }
}

test "cross attention query and memory gradients match finite differences" {
    const testing = std.testing;
    var device = try tensor.Device.init(testing.allocator, .cpu);
    defer device.deinit();
    var context = ExecutionContext.init(&device);
    var prng = std.Random.DefaultPrng.init(37);
    var attention = try CrossAttention.init(&context, 2, 2, prng.random());
    defer attention.deinit();
    const query_values = [_]f32{ 0.2, -0.35, 0.5, 0.1 };
    const memory_values = [_]f32{ -0.4, 0.3, 0.6, -0.2 };
    var queries = try context.upload(&.{ 2, 2 }, &query_values);
    defer queries.deinit();
    var memory = try context.upload(&.{ 2, 2 }, &memory_values);
    defer memory.deinit();
    var output_gradient = try context.upload(&.{ 2, 2 }, &.{ 1, 1, 1, 1 });
    defer output_gradient.deinit();
    var gradients = try attention.backward(&context, queries, memory, output_gradient);
    defer gradients.deinit();
    var query_analytical: [4]f32 = undefined;
    var memory_analytical: [4]f32 = undefined;
    try context.readback(gradients.query_input, &query_analytical);
    try context.readback(gradients.memory_input, &memory_analytical);

    const epsilon: f32 = 1e-3;
    for (0..query_values.len) |input_index| {
        var plus = query_values;
        plus[input_index] += epsilon;
        var plus_input = try context.upload(&.{ 2, 2 }, &plus);
        defer plus_input.deinit();
        var plus_output = try attention.forward(&context, plus_input, memory);
        defer plus_output.deinit();
        var plus_result: [4]f32 = undefined;
        try context.readback(plus_output, &plus_result);
        var minus = query_values;
        minus[input_index] -= epsilon;
        var minus_input = try context.upload(&.{ 2, 2 }, &minus);
        defer minus_input.deinit();
        var minus_output = try attention.forward(&context, minus_input, memory);
        defer minus_output.deinit();
        var minus_result: [4]f32 = undefined;
        try context.readback(minus_output, &minus_result);
        var plus_sum: f32 = 0;
        var minus_sum: f32 = 0;
        for (plus_result) |value| plus_sum += value;
        for (minus_result) |value| minus_sum += value;
        try testing.expectApproxEqAbs((plus_sum - minus_sum) / (2 * epsilon), query_analytical[input_index], 5e-3);
    }
    for (0..memory_values.len) |input_index| {
        var plus = memory_values;
        plus[input_index] += epsilon;
        var plus_input = try context.upload(&.{ 2, 2 }, &plus);
        defer plus_input.deinit();
        var plus_output = try attention.forward(&context, queries, plus_input);
        defer plus_output.deinit();
        var plus_result: [4]f32 = undefined;
        try context.readback(plus_output, &plus_result);
        var minus = memory_values;
        minus[input_index] -= epsilon;
        var minus_input = try context.upload(&.{ 2, 2 }, &minus);
        defer minus_input.deinit();
        var minus_output = try attention.forward(&context, queries, minus_input);
        defer minus_output.deinit();
        var minus_result: [4]f32 = undefined;
        try context.readback(minus_output, &minus_result);
        var plus_sum: f32 = 0;
        var minus_sum: f32 = 0;
        for (plus_result) |value| plus_sum += value;
        for (minus_result) |value| minus_sum += value;
        try testing.expectApproxEqAbs((plus_sum - minus_sum) / (2 * epsilon), memory_analytical[input_index], 5e-3);
    }
}

test "masked multi-head attention ignores padding on every available device" {
    const testing = std.testing;
    const preferences = [_]tensor.DevicePreference{ .cpu, .metal, .cuda, .rocm };
    for (preferences) |preference| {
        var device = tensor.Device.init(testing.allocator, preference) catch |err| switch (err) {
            error.BackendUnavailable => continue,
            else => return err,
        };
        defer device.deinit();
        var context = ExecutionContext.init(&device);
        var prng = std.Random.DefaultPrng.init(91);
        var attention = try MaskedMultiHeadSelfAttention.init(&context, 4, 2, prng.random());
        defer attention.deinit();
        var input_a = try context.upload(&.{ 1, 3, 4 }, &.{ 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0 });
        defer input_a.deinit();
        var input_b = try context.upload(&.{ 1, 3, 4 }, &.{ 1, 0, 0, 1, 0, 1, 1, 0, 90, -80, 70, -60 });
        defer input_b.deinit();
        var token_mask = try context.upload(&.{ 1, 3, 4 }, &.{ 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0 });
        defer token_mask.deinit();
        var attention_mask = try context.upload(&.{ 2, 3, 3 }, &.{
            1, 1, 0, 1, 1, 0, 0, 0, 0,
            1, 1, 0, 1, 1, 0, 0, 0, 0,
        });
        defer attention_mask.deinit();
        context.resetStats();
        var output_a = try attention.forward(&context, input_a, attention_mask, token_mask);
        defer output_a.deinit();
        var output_b = try attention.forward(&context, input_b, attention_mask, token_mask);
        defer output_b.deinit();
        var output_gradient = try context.createTensor(&.{ 1, 3, 4 });
        defer output_gradient.deinit();
        output_gradient.fill(1);
        var gradients = try attention.backward(&context, input_a, attention_mask, token_mask, output_gradient);
        defer gradients.deinit();
        try testing.expectEqual(@as(usize, 0), context.stats.readbacks);
        var values_a: [12]f32 = undefined;
        var values_b: [12]f32 = undefined;
        try context.readback(output_a, &values_a);
        try context.readback(output_b, &values_b);
        for (values_a[0..8], values_b[0..8]) |a, b| try testing.expectApproxEqAbs(a, b, 1e-5);
        try testing.expectEqualSlices(f32, &.{ 0, 0, 0, 0 }, values_a[8..12]);
        try testing.expectEqualSlices(f32, &.{ 0, 0, 0, 0 }, values_b[8..12]);
    }
}

test "masked multi-head attention input gradient matches finite differences" {
    const testing = std.testing;
    var device = try tensor.Device.init(testing.allocator, .cpu);
    defer device.deinit();
    var context = ExecutionContext.init(&device);
    var prng = std.Random.DefaultPrng.init(17);
    var attention = try MaskedMultiHeadSelfAttention.init(&context, 2, 1, prng.random());
    defer attention.deinit();
    const input_values = [_]f32{ 0.2, -0.3, 0.5, 0.1 };
    var input = try context.upload(&.{ 1, 2, 2 }, &input_values);
    defer input.deinit();
    var attention_mask = try context.upload(&.{ 1, 2, 2 }, &.{ 1, 1, 1, 1 });
    defer attention_mask.deinit();
    var token_mask = try context.upload(&.{ 1, 2, 2 }, &.{ 1, 1, 1, 1 });
    defer token_mask.deinit();
    var output_gradient = try context.upload(&.{ 1, 2, 2 }, &.{ 1, 1, 1, 1 });
    defer output_gradient.deinit();
    var gradients = try attention.backward(&context, input, attention_mask, token_mask, output_gradient);
    defer gradients.deinit();
    var analytical: [4]f32 = undefined;
    try context.readback(gradients.input, &analytical);
    const epsilon: f32 = 1e-3;
    for (0..input_values.len) |input_index| {
        var plus = input_values;
        plus[input_index] += epsilon;
        var plus_input = try context.upload(&.{ 1, 2, 2 }, &plus);
        defer plus_input.deinit();
        var plus_output = try attention.forward(&context, plus_input, attention_mask, token_mask);
        defer plus_output.deinit();
        var plus_values: [4]f32 = undefined;
        try context.readback(plus_output, &plus_values);
        var minus = input_values;
        minus[input_index] -= epsilon;
        var minus_input = try context.upload(&.{ 1, 2, 2 }, &minus);
        defer minus_input.deinit();
        var minus_output = try attention.forward(&context, minus_input, attention_mask, token_mask);
        defer minus_output.deinit();
        var minus_values: [4]f32 = undefined;
        try context.readback(minus_output, &minus_values);
        var plus_sum: f32 = 0;
        var minus_sum: f32 = 0;
        for (plus_values) |value| plus_sum += value;
        for (minus_values) |value| minus_sum += value;
        try testing.expectApproxEqAbs((plus_sum - minus_sum) / (2 * epsilon), analytical[input_index], 5e-3);
    }
}

test "encoder block forward and backward stay on each available device" {
    const testing = std.testing;
    const preferences = [_]tensor.DevicePreference{ .cpu, .metal, .cuda, .rocm };
    for (preferences) |preference| {
        var device = tensor.Device.init(testing.allocator, preference) catch |err| switch (err) {
            error.BackendUnavailable => continue,
            else => return err,
        };
        defer device.deinit();
        var context = ExecutionContext.init(&device);
        var prng = std.Random.DefaultPrng.init(29);
        var block = try EncoderBlock.init(&context, 3, 2, 4, prng.random());
        defer block.deinit();
        try testing.expectEqual(@as(usize, 54), block.parameterCount());
        try testing.expectEqual(@as(usize, 16), block.parameterTensorCount());
        var input = try context.upload(&.{ 3, 2 }, &.{
            0.2,  -0.1,
            0.5,  0.3,
            -0.4, 0.7,
        });
        defer input.deinit();
        var output_gradient = try context.upload(&.{ 3, 2 }, &.{
            1, 1,
            1, 1,
            1, 1,
        });
        defer output_gradient.deinit();

        var output = try block.forward(&context, input);
        defer output.deinit();
        var gradients = try block.backward(&context, input, output_gradient);
        defer gradients.deinit();
        var parameter_storage: [16]*Tensor = undefined;
        const parameters = try block.parameters(&parameter_storage);
        var gradient_storage: [16]Tensor = undefined;
        const parameter_gradients = try gradients.parameterGradients(&gradient_storage);
        for (parameters, parameter_gradients) |parameter, gradient| {
            try testing.expect(parameter.shape.eql(gradient.shape));
        }
        try testing.expect(output.shape.eql(input.shape));
        try testing.expect(gradients.input.shape.eql(input.shape));
        try testing.expectEqual(@as(usize, 0), context.stats.readbacks);
    }
}

test "decoder block runs a finite device-resident forward pass" {
    const testing = std.testing;
    const allocator: Allocator = testing.allocator;

    var device = try tensor.Device.init(allocator, .auto);
    defer device.deinit();
    var context = ExecutionContext.init(&device);
    var prng = std.Random.DefaultPrng.init(42);
    var block = try Block.init(&context, 4, 2, 8, prng.random());
    defer block.deinit();
    try testing.expectEqual(@as(usize, 172), block.parameterCount());

    context.resetStats();
    var input = try context.upload(&.{ 3, 4 }, &.{
        0.1,  0.2,  -0.3, 0.4,
        0.5,  -0.6, 0.7,  0.8,
        -0.9, 1.0,  1.1,  -1.2,
    });
    defer input.deinit();

    try context.beginBatch();
    var output = try block.forward(&context, input);
    try context.endBatch();
    defer output.deinit();

    var values: [12]f32 = undefined;
    try context.readback(output, &values);
    for (values) |value| try testing.expect(std.math.isFinite(value));
    try testing.expectEqualSlices(usize, &.{ 3, 4 }, output.shape.slice());
    try testing.expectEqual(@as(usize, 16), context.stats.kernels);

    const runtime_stats = context.backendStats();
    if (device.backendType() == .CPU) {
        try testing.expectEqual(@as(usize, 0), runtime_stats.kernel_launches);
    } else {
        try testing.expectEqual(@as(usize, 18), runtime_stats.kernel_launches);
        try testing.expectEqual(@as(usize, 1), runtime_stats.device_to_host_transfers);
        try testing.expectEqual(@as(usize, 1), runtime_stats.synchronizations);
    }
}

test "linear classifier trains with device-resident gradients" {
    const testing = std.testing;
    const allocator: Allocator = testing.allocator;

    var device = try tensor.Device.init(allocator, .auto);
    defer device.deinit();
    var context = ExecutionContext.init(&device);
    var prng = std.Random.DefaultPrng.init(7);
    var classifier = try Linear.init(&context, 2, 2, prng.random());
    defer classifier.deinit();
    const optimizer = try Sgd.init(0.25);

    var inputs = try context.upload(&.{ 4, 2 }, &.{
        1.0, 0.0,
        0.0, 1.0,
        1.2, 0.1,
        0.1, 1.2,
    });
    defer inputs.deinit();
    var targets = try context.upload(&.{ 4, 2 }, &.{
        1, 0,
        0, 1,
        1, 0,
        0, 1,
    });
    defer targets.deinit();

    var initial_loss: f32 = 0;
    var final_loss: f32 = 0;
    for (0..40) |step| {
        try context.beginBatch();
        var logits = try classifier.forward(&context, inputs);
        var objective = try SoftmaxCrossEntropy.init(&context, logits, targets);
        var gradients = try classifier.backward(&context, inputs, objective.gradient);
        try optimizer.stepLinear(&context, &classifier, gradients);
        gradients.deinit();
        logits.deinit();
        try context.endBatch();

        const loss = try objective.loss(&context, targets);
        objective.deinit();
        if (step == 0) initial_loss = loss;
        final_loss = loss;
    }

    try testing.expect(final_loss < initial_loss * 0.35);
    try testing.expect(final_loss < 0.2);
}

test "decoder block backward updates every parameter group" {
    const testing = std.testing;
    const allocator: Allocator = testing.allocator;
    var device = try tensor.Device.init(allocator, .cpu);
    defer device.deinit();
    var context = ExecutionContext.init(&device);
    var prng = std.Random.DefaultPrng.init(99);
    var block = try Block.init(&context, 4, 2, 8, prng.random());
    defer block.deinit();
    const optimizer = try Sgd.init(0.01);
    var input = try context.upload(&.{ 3, 4 }, &.{
        0.1,  0.2,  -0.3, 0.4,
        0.5,  -0.6, 0.7,  0.8,
        -0.9, 1.0,  1.1,  -1.2,
    });
    defer input.deinit();
    var output_gradient = try context.upload(&.{ 3, 4 }, &.{
        0.2,  -0.1, 0.3,  -0.4,
        -0.5, 0.6,  -0.7, 0.8,
        0.9,  -1.0, 0.2,  0.1,
    });
    defer output_gradient.deinit();

    try context.beginBatch();
    var gradients = try block.backward(&context, input, output_gradient);
    try optimizer.stepBlock(&context, &block, gradients);
    try context.endBatch();
    defer gradients.deinit();

    var input_gradient: [12]f32 = undefined;
    try context.readback(gradients.input, &input_gradient);
    for (input_gradient) |value| try testing.expect(std.math.isFinite(value));
}

test "decoder trains token embeddings and transformer body end to end" {
    const testing = std.testing;
    const allocator: Allocator = testing.allocator;
    var device = try tensor.Device.init(allocator, .cpu);
    defer device.deinit();
    var context = ExecutionContext.init(&device);
    var prng = std.Random.DefaultPrng.init(123);
    var decoder = try Decoder.init(&context, .{
        .vocabulary_size = 3,
        .maximum_sequence_length = 3,
        .layers = 1,
        .heads = 1,
        .channels = 4,
        .hidden_features = 8,
    }, prng.random());
    defer decoder.deinit();
    try testing.expectEqual(@as(usize, 219), decoder.parameterCount());
    var tokens = try context.upload(&.{ 3, 1 }, &.{ 0, 1, 2 });
    defer tokens.deinit();
    var positions = try context.upload(&.{ 3, 1 }, &.{ 0, 1, 2 });
    defer positions.deinit();
    var targets = try context.upload(&.{ 3, 3 }, &.{
        0, 1, 0,
        0, 0, 1,
        1, 0, 0,
    });
    defer targets.deinit();
    const optimizer = try Sgd.init(0.03);

    var initial_loss: f32 = 0;
    var final_loss: f32 = 0;
    for (0..50) |step| {
        const loss = try decoder.trainStep(&context, tokens, positions, targets, optimizer);
        if (step == 0) initial_loss = loss;
        final_loss = loss;
    }
    try testing.expect(final_loss < initial_loss * 0.5);
    try testing.expect(final_loss < 0.4);

    try context.beginBatch();
    var logits = try decoder.forward(&context, tokens, positions);
    try context.endBatch();
    defer logits.deinit();
    try testing.expectEqualSlices(usize, &.{ 3, 3 }, logits.shape.slice());
}

test "decoder KV cache matches full causal logits" {
    const testing = std.testing;
    const allocator: Allocator = testing.allocator;
    var device = try tensor.Device.init(allocator, .cpu);
    defer device.deinit();
    var context = ExecutionContext.init(&device);
    var prng = std.Random.DefaultPrng.init(321);
    var decoder = try Decoder.init(&context, .{
        .vocabulary_size = 5,
        .maximum_sequence_length = 4,
        .layers = 1,
        .heads = 2,
        .channels = 4,
        .hidden_features = 8,
    }, prng.random());
    defer decoder.deinit();
    var cache = try decoder.initCache(&context);
    defer cache.deinit();
    const token_values = [_]f32{ 1, 2, 3 };
    const position_values = [_]f32{ 0, 1, 2 };

    for (0..token_values.len) |position| {
        var token = try context.upload(&.{ 1, 1 }, token_values[position .. position + 1]);
        defer token.deinit();
        var prefix = try context.upload(&.{ position + 1, 1 }, token_values[0 .. position + 1]);
        defer prefix.deinit();
        var positions = try context.upload(&.{ position + 1, 1 }, position_values[0 .. position + 1]);
        defer positions.deinit();
        try context.beginBatch();
        var cached_logits = try decoder.forwardToken(&context, token, &cache);
        var full_logits = try decoder.forward(&context, prefix, positions);
        try context.endBatch();
        defer cached_logits.deinit();
        defer full_logits.deinit();
        var cached_values: [5]f32 = undefined;
        try context.readback(cached_logits, &cached_values);
        const full_values = try allocator.alloc(f32, (position + 1) * 5);
        defer allocator.free(full_values);
        try context.readback(full_logits, full_values);
        const last_row = full_values[position * 5 ..][0..5];
        for (cached_values, last_row) |cached_value, full_value| {
            try testing.expectApproxEqAbs(full_value, cached_value, 2e-4);
        }
    }
    try testing.expectEqual(@as(usize, 3), cache.position);
    cache.reset();
    try testing.expectEqual(@as(usize, 0), cache.position);
}
