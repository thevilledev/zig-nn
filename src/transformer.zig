const std = @import("std");
const tensor = @import("tensor.zig");

const Allocator = std.mem.Allocator;
const ExecutionContext = tensor.ExecutionContext;
const Tensor = tensor.Tensor;

/// Device-resident affine projection used by Transformer components.
pub const Linear = struct {
    weights: Tensor,
    bias: Tensor,

    pub fn init(
        context: *ExecutionContext,
        in_features: usize,
        out_features: usize,
        random: std.Random,
    ) !Linear {
        if (in_features == 0 or out_features == 0) return error.InvalidDimension;

        const allocator = context.device.allocator;
        const weight_count = try std.math.mul(usize, in_features, out_features);
        const values = try allocator.alloc(f32, weight_count);
        defer allocator.free(values);
        const denominator = @as(f32, @floatFromInt(in_features + out_features));
        const bound = @sqrt(6.0 / denominator);
        for (values) |*value| value.* = (random.float(f32) * 2.0 - 1.0) * bound;

        var weights = try context.upload(&.{ in_features, out_features }, values);
        errdefer weights.deinit();
        const zero_bias = try allocator.alloc(f32, out_features);
        defer allocator.free(zero_bias);
        @memset(zero_bias, 0);
        const bias = try context.upload(&.{ 1, out_features }, zero_bias);
        return .{ .weights = weights, .bias = bias };
    }

    pub fn deinit(self: *Linear) void {
        self.weights.deinit();
        self.bias.deinit();
        self.* = undefined;
    }

    pub fn forward(self: *const Linear, context: *ExecutionContext, input: Tensor) !Tensor {
        return context.linear(input, self.weights, self.bias);
    }

    pub fn backward(self: *const Linear, context: *ExecutionContext, input: Tensor, output_gradient: Tensor) !LinearGradients {
        if (input.shape.rank != 2 or output_gradient.shape.rank != 2 or
            input.shape.dims[0] != output_gradient.shape.dims[0] or
            input.shape.dims[1] != self.weights.shape.dims[0] or
            output_gradient.shape.dims[1] != self.weights.shape.dims[1])
        {
            return error.DimensionMismatch;
        }

        var transposed_weights = try context.transpose(self.weights);
        defer transposed_weights.deinit();
        var input_gradient = try context.matmul(output_gradient, transposed_weights);
        errdefer input_gradient.deinit();
        var transposed_input = try context.transpose(input);
        defer transposed_input.deinit();
        var weights_gradient = try context.matmul(transposed_input, output_gradient);
        errdefer weights_gradient.deinit();
        const bias_gradient = try context.sumRows(output_gradient);
        return .{
            .input = input_gradient,
            .weights = weights_gradient,
            .bias = bias_gradient,
        };
    }

    pub fn parameterCount(self: Linear) usize {
        return self.weights.elementCount() + self.bias.elementCount();
    }
};

pub const LinearGradients = struct {
    input: Tensor,
    weights: Tensor,
    bias: Tensor,

    pub fn deinit(self: *LinearGradients) void {
        self.input.deinit();
        self.weights.deinit();
        self.bias.deinit();
        self.* = undefined;
    }
};

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

    pub fn parameterCount(self: CausalSelfAttention) usize {
        return self.query.parameterCount() + self.key.parameterCount() +
            self.value.parameterCount() + self.output.parameterCount();
    }
};

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
        var expanded = try self.expand.forward(context, input);
        defer expanded.deinit();
        var activated = try context.gelu(expanded);
        defer activated.deinit();
        return self.project.forward(context, activated);
    }

    pub fn parameterCount(self: FeedForward) usize {
        return self.expand.parameterCount() + self.project.parameterCount();
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

    pub fn parameterCount(self: Block) usize {
        return self.attention_norm.parameterCount() + self.attention.parameterCount() +
            self.feed_forward_norm.parameterCount() + self.feed_forward.parameterCount();
    }
};

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
    try testing.expectEqual(@as(usize, 18), context.stats.kernels);

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
