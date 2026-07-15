const std = @import("std");
const tensor = @import("tensor.zig");

const ExecutionContext = tensor.ExecutionContext;
const Tensor = tensor.Tensor;

/// Device-resident affine projection shared by trainable Tensor models.
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

test "linear module exposes forward and backward shapes" {
    const testing = std.testing;

    var device = try tensor.Device.init(testing.allocator, .cpu);
    defer device.deinit();
    var context = ExecutionContext.init(&device);

    var prng = std.Random.DefaultPrng.init(42);
    var linear = try Linear.init(&context, 3, 2, prng.random());
    defer linear.deinit();
    var input = try context.upload(&.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer input.deinit();
    var output = try linear.forward(&context, input);
    defer output.deinit();
    try testing.expect(output.shape.eql(try tensor.Shape.init(&.{ 2, 2 })));

    var output_gradient = try context.upload(&.{ 2, 2 }, &.{ 1, 1, 1, 1 });
    defer output_gradient.deinit();
    var gradients = try linear.backward(&context, input, output_gradient);
    defer gradients.deinit();

    try testing.expect(gradients.input.shape.eql(input.shape));
    try testing.expect(gradients.weights.shape.eql(linear.weights.shape));
    try testing.expect(gradients.bias.shape.eql(linear.bias.shape));
}
