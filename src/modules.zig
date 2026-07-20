const std = @import("std");
const tensor = @import("tensor.zig");

const ExecutionContext = tensor.ExecutionContext;
const Tensor = tensor.Tensor;

pub const ActivationKind = tensor.ActivationKind;

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

    pub fn prepareInference(self: *Linear) !void {
        try self.weights.prepareInferenceWeight();
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

pub const MlpConfig = struct {
    widths: []const usize,
    hidden_activation: ActivationKind = .relu,
    output_activation: ActivationKind = .linear,
};

/// A device-resident multilayer perceptron with explicit forward caches and
/// gradients. The caller owns each returned forward or gradient bundle.
pub const Mlp = struct {
    allocator: std.mem.Allocator,
    layers: []Linear,
    hidden_activation: ActivationKind,
    output_activation: ActivationKind,

    pub fn init(context: *ExecutionContext, config: MlpConfig, random: std.Random) !Mlp {
        if (config.widths.len < 2) return error.InvalidLayerCount;
        for (config.widths) |width| {
            if (width == 0) return error.InvalidDimension;
        }

        const allocator = context.device.allocator;
        const layers = try allocator.alloc(Linear, config.widths.len - 1);
        errdefer allocator.free(layers);

        var initialized: usize = 0;
        errdefer for (layers[0..initialized]) |*layer| layer.deinit();
        for (layers, 0..) |*layer, index| {
            layer.* = try Linear.init(context, config.widths[index], config.widths[index + 1], random);
            initialized += 1;
        }

        return .{
            .allocator = allocator,
            .layers = layers,
            .hidden_activation = config.hidden_activation,
            .output_activation = config.output_activation,
        };
    }

    pub fn deinit(self: *Mlp) void {
        for (self.layers) |*layer| layer.deinit();
        self.allocator.free(self.layers);
        self.* = undefined;
    }

    pub fn forward(self: *const Mlp, context: *ExecutionContext, input: Tensor) !MlpForward {
        if (input.shape.rank != 2 or input.shape.dims[1] != self.layers[0].weights.shape.dims[0]) {
            return error.DimensionMismatch;
        }

        const pre_activations = try self.allocator.alloc(Tensor, self.layers.len);
        errdefer self.allocator.free(pre_activations);
        const hidden_activations = try self.allocator.alloc(Tensor, self.layers.len - 1);
        errdefer self.allocator.free(hidden_activations);

        var pre_count: usize = 0;
        errdefer for (pre_activations[0..pre_count]) |*value| value.deinit();
        var hidden_count: usize = 0;
        errdefer for (hidden_activations[0..hidden_count]) |*value| value.deinit();

        var current = input;
        for (self.layers, 0..) |*layer, index| {
            pre_activations[index] = try layer.forward(context, current);
            pre_count += 1;
            const kind = self.activationForLayer(index);
            const activated = try context.activate(pre_activations[index], kind);
            if (index + 1 == self.layers.len) {
                return .{
                    .output = activated,
                    .cache = .{
                        .allocator = self.allocator,
                        .input = input,
                        .pre_activations = pre_activations,
                        .hidden_activations = hidden_activations,
                    },
                };
            }
            hidden_activations[index] = activated;
            hidden_count += 1;
            current = activated;
        }
        unreachable;
    }

    pub fn backward(
        self: *const Mlp,
        context: *ExecutionContext,
        cache: *const MlpCache,
        output_gradient: Tensor,
    ) !MlpGradients {
        if (cache.pre_activations.len != self.layers.len or
            cache.hidden_activations.len + 1 != self.layers.len or
            !output_gradient.shape.eql(cache.pre_activations[self.layers.len - 1].shape))
        {
            return error.DimensionMismatch;
        }

        const layers = try self.allocator.alloc(LinearGradients, self.layers.len);
        errdefer self.allocator.free(layers);
        var initialized_from = self.layers.len;
        errdefer for (layers[initialized_from..]) |*gradient| gradient.deinit();

        var current_gradient = output_gradient;
        var index = self.layers.len;
        while (index > 0) {
            index -= 1;
            var derivative = try context.activationDerivative(cache.pre_activations[index], self.activationForLayer(index));
            defer derivative.deinit();
            var activation_gradient = try context.multiply(current_gradient, derivative);
            defer activation_gradient.deinit();

            const layer_input = if (index == 0) cache.input else cache.hidden_activations[index - 1];
            layers[index] = try self.layers[index].backward(context, layer_input, activation_gradient);
            initialized_from = index;
            current_gradient = layers[index].input;
        }

        return .{ .allocator = self.allocator, .layers = layers };
    }

    pub fn parameterCount(self: Mlp) usize {
        var total: usize = 0;
        for (self.layers) |layer| total += layer.parameterCount();
        return total;
    }

    pub fn parameterTensorCount(self: Mlp) usize {
        return self.layers.len * 2;
    }

    pub fn parameters(self: *Mlp, output: []*Tensor) ![]*Tensor {
        if (output.len < self.parameterTensorCount()) return error.InsufficientBuffer;
        var index: usize = 0;
        for (self.layers) |*layer| {
            output[index] = &layer.weights;
            output[index + 1] = &layer.bias;
            index += 2;
        }
        return output[0..index];
    }

    fn activationForLayer(self: Mlp, index: usize) ActivationKind {
        return if (index + 1 == self.layers.len) self.output_activation else self.hidden_activation;
    }
};

pub const MlpForward = struct {
    output: Tensor,
    cache: MlpCache,

    pub fn deinit(self: *MlpForward) void {
        self.output.deinit();
        self.cache.deinit();
        self.* = undefined;
    }
};

pub const MlpCache = struct {
    allocator: std.mem.Allocator,
    input: Tensor,
    pre_activations: []Tensor,
    hidden_activations: []Tensor,

    pub fn deinit(self: *MlpCache) void {
        for (self.pre_activations) |*value| value.deinit();
        for (self.hidden_activations) |*value| value.deinit();
        self.allocator.free(self.pre_activations);
        self.allocator.free(self.hidden_activations);
        self.* = undefined;
    }
};

pub const MlpGradients = struct {
    allocator: std.mem.Allocator,
    layers: []LinearGradients,

    pub fn deinit(self: *MlpGradients) void {
        for (self.layers) |*gradient| gradient.deinit();
        self.allocator.free(self.layers);
        self.* = undefined;
    }

    pub fn input(self: MlpGradients) Tensor {
        return self.layers[0].input;
    }

    pub fn parameterTensorCount(self: MlpGradients) usize {
        return self.layers.len * 2;
    }

    pub fn parameterGradients(self: MlpGradients, output: []Tensor) ![]Tensor {
        if (output.len < self.parameterTensorCount()) return error.InsufficientBuffer;
        var index: usize = 0;
        for (self.layers) |gradient| {
            output[index] = gradient.weights;
            output[index + 1] = gradient.bias;
            index += 2;
        }
        return output[0..index];
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

test "mlp backward input gradient matches finite differences" {
    const testing = std.testing;

    var device = try tensor.Device.init(testing.allocator, .cpu);
    defer device.deinit();
    var context = ExecutionContext.init(&device);
    var prng = std.Random.DefaultPrng.init(7);
    var mlp = try Mlp.init(&context, .{
        .widths = &.{ 2, 3, 1 },
        .hidden_activation = .tanh,
        .output_activation = .linear,
    }, prng.random());
    defer mlp.deinit();

    const input_values = [_]f32{ 0.25, -0.4 };
    var input = try context.upload(&.{ 1, 2 }, &input_values);
    defer input.deinit();
    var forward = try mlp.forward(&context, input);
    defer forward.deinit();
    var output_gradient = try context.upload(&.{ 1, 1 }, &.{1});
    defer output_gradient.deinit();
    var gradients = try mlp.backward(&context, &forward.cache, output_gradient);
    defer gradients.deinit();
    var analytical: [2]f32 = undefined;
    try context.readback(gradients.input(), &analytical);

    const epsilon: f32 = 1e-3;
    for (0..2) |input_index| {
        var plus_values = input_values;
        plus_values[input_index] += epsilon;
        var plus_input = try context.upload(&.{ 1, 2 }, &plus_values);
        defer plus_input.deinit();
        var plus_forward = try mlp.forward(&context, plus_input);
        defer plus_forward.deinit();
        var plus_output: [1]f32 = undefined;
        try context.readback(plus_forward.output, &plus_output);

        var minus_values = input_values;
        minus_values[input_index] -= epsilon;
        var minus_input = try context.upload(&.{ 1, 2 }, &minus_values);
        defer minus_input.deinit();
        var minus_forward = try mlp.forward(&context, minus_input);
        defer minus_forward.deinit();
        var minus_output: [1]f32 = undefined;
        try context.readback(minus_forward.output, &minus_output);

        const numerical = (plus_output[0] - minus_output[0]) / (2 * epsilon);
        try testing.expectApproxEqAbs(numerical, analytical[input_index], 2e-3);
    }
}
