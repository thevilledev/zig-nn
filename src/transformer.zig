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

    pub fn parameterCount(self: Linear) usize {
        return self.weights.elementCount() + self.bias.elementCount();
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
