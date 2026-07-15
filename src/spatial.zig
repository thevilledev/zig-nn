const std = @import("std");
const Matrix = @import("matrix.zig").Matrix;

pub const ImageShape = struct {
    height: usize,
    width: usize,
    channels: usize,

    pub fn validate(self: ImageShape) !void {
        if (self.height == 0 or self.width == 0 or self.channels == 0) {
            return error.InvalidDimension;
        }
        _ = try std.math.mul(usize, self.height, self.width);
        _ = try self.elementCount();
    }

    pub fn elementCount(self: ImageShape) !usize {
        return std.math.mul(
            usize,
            try std.math.mul(usize, self.height, self.width),
            self.channels,
        );
    }
};

pub const Conv2dConfig = struct {
    input: ImageShape,
    out_channels: usize,
    kernel_height: usize,
    kernel_width: usize,
    stride: usize = 1,
    padding: usize = 0,

    pub fn outputShape(self: Conv2dConfig) !ImageShape {
        try self.input.validate();
        if (self.out_channels == 0 or self.kernel_height == 0 or
            self.kernel_width == 0 or self.stride == 0)
        {
            return error.InvalidDimension;
        }
        const doubled_padding = try std.math.mul(usize, self.padding, 2);
        const padded_height = try std.math.add(usize, self.input.height, doubled_padding);
        const padded_width = try std.math.add(usize, self.input.width, doubled_padding);
        if (self.kernel_height > padded_height or self.kernel_width > padded_width) {
            return error.KernelTooLarge;
        }
        return .{
            .height = (padded_height - self.kernel_height) / self.stride + 1,
            .width = (padded_width - self.kernel_width) / self.stride + 1,
            .channels = self.out_channels,
        };
    }

    fn kernelElements(self: Conv2dConfig) !usize {
        return std.math.mul(
            usize,
            try std.math.mul(usize, self.kernel_height, self.kernel_width),
            self.input.channels,
        );
    }
};

/// CPU-first educational 2D convolution over flattened NHWC image batches.
/// Each input matrix row is one image; weights use
/// `[kernel_height * kernel_width * in_channels, out_channels]` layout.
pub const Conv2d = struct {
    allocator: std.mem.Allocator,
    config: Conv2dConfig,
    output_shape: ImageShape,
    weights: Matrix,
    bias: Matrix,

    pub fn init(
        allocator: std.mem.Allocator,
        config: Conv2dConfig,
        random: std.Random,
    ) !Conv2d {
        const output_shape = try config.outputShape();
        const kernel_elements = try config.kernelElements();
        var weights = try Matrix.init(allocator, kernel_elements, config.out_channels);
        errdefer weights.deinit();
        var bias = try Matrix.init(allocator, 1, config.out_channels);
        errdefer bias.deinit();

        const fan_in = @as(f64, @floatFromInt(kernel_elements));
        const bound = @sqrt(6.0 / (fan_in + @as(f64, @floatFromInt(config.out_channels))));
        for (weights.data) |*value| {
            value.* = (random.float(f64) * 2.0 - 1.0) * bound;
        }
        return .{
            .allocator = allocator,
            .config = config,
            .output_shape = output_shape,
            .weights = weights,
            .bias = bias,
        };
    }

    pub fn deinit(self: *Conv2d) void {
        self.weights.deinit();
        self.bias.deinit();
        self.* = undefined;
    }

    pub fn forward(self: Conv2d, input: Matrix) !Matrix {
        try self.validateInput(input);
        const output_elements = try self.output_shape.elementCount();
        var output = try Matrix.init(self.allocator, input.rows, output_elements);
        errdefer output.deinit();

        for (0..input.rows) |batch| {
            for (0..self.output_shape.height) |output_y| {
                for (0..self.output_shape.width) |output_x| {
                    for (0..self.output_shape.channels) |output_channel| {
                        var sum = self.bias.data[output_channel];
                        for (0..self.config.kernel_height) |kernel_y| {
                            const source_y = self.sourceCoordinate(output_y, kernel_y) orelse continue;
                            for (0..self.config.kernel_width) |kernel_x| {
                                const source_x = self.sourceCoordinateX(output_x, kernel_x) orelse continue;
                                for (0..self.config.input.channels) |input_channel| {
                                    const input_index = self.inputIndex(
                                        source_y,
                                        source_x,
                                        input_channel,
                                    );
                                    const weight_index = self.weightIndex(
                                        kernel_y,
                                        kernel_x,
                                        input_channel,
                                        output_channel,
                                    );
                                    sum += input.data[batch * input.cols + input_index] *
                                        self.weights.data[weight_index];
                                }
                            }
                        }
                        const output_index = self.outputIndex(
                            output_y,
                            output_x,
                            output_channel,
                        );
                        output.data[batch * output.cols + output_index] = sum;
                    }
                }
            }
        }
        return output;
    }

    pub fn backward(
        self: Conv2d,
        input: Matrix,
        output_gradient: Matrix,
    ) !Conv2dGradients {
        try self.validateInput(input);
        const output_elements = try self.output_shape.elementCount();
        if (output_gradient.rows != input.rows or output_gradient.cols != output_elements) {
            return error.DimensionMismatch;
        }

        var input_gradient = try Matrix.init(self.allocator, input.rows, input.cols);
        errdefer input_gradient.deinit();
        var weights_gradient = try Matrix.init(
            self.allocator,
            self.weights.rows,
            self.weights.cols,
        );
        errdefer weights_gradient.deinit();
        var bias_gradient = try Matrix.init(self.allocator, 1, self.bias.cols);
        errdefer bias_gradient.deinit();

        for (0..input.rows) |batch| {
            for (0..self.output_shape.height) |output_y| {
                for (0..self.output_shape.width) |output_x| {
                    for (0..self.output_shape.channels) |output_channel| {
                        const output_index = self.outputIndex(
                            output_y,
                            output_x,
                            output_channel,
                        );
                        const gradient = output_gradient.data[
                            batch * output_gradient.cols + output_index
                        ];
                        bias_gradient.data[output_channel] += gradient;
                        for (0..self.config.kernel_height) |kernel_y| {
                            const source_y = self.sourceCoordinate(output_y, kernel_y) orelse continue;
                            for (0..self.config.kernel_width) |kernel_x| {
                                const source_x = self.sourceCoordinateX(output_x, kernel_x) orelse continue;
                                for (0..self.config.input.channels) |input_channel| {
                                    const input_index = self.inputIndex(
                                        source_y,
                                        source_x,
                                        input_channel,
                                    );
                                    const weight_index = self.weightIndex(
                                        kernel_y,
                                        kernel_x,
                                        input_channel,
                                        output_channel,
                                    );
                                    const flat_input = batch * input.cols + input_index;
                                    weights_gradient.data[weight_index] +=
                                        input.data[flat_input] * gradient;
                                    input_gradient.data[flat_input] +=
                                        self.weights.data[weight_index] * gradient;
                                }
                            }
                        }
                    }
                }
            }
        }
        return .{
            .input = input_gradient,
            .weights = weights_gradient,
            .bias = bias_gradient,
        };
    }

    pub fn applyGradients(
        self: *Conv2d,
        gradients: Conv2dGradients,
        learning_rate: f64,
    ) !void {
        if (!std.math.isFinite(learning_rate) or learning_rate <= 0) {
            return error.InvalidLearningRate;
        }
        if (gradients.weights.rows != self.weights.rows or
            gradients.weights.cols != self.weights.cols or
            gradients.bias.rows != self.bias.rows or
            gradients.bias.cols != self.bias.cols)
        {
            return error.DimensionMismatch;
        }
        for (self.weights.data, gradients.weights.data) |*parameter, gradient| {
            parameter.* -= learning_rate * gradient;
        }
        for (self.bias.data, gradients.bias.data) |*parameter, gradient| {
            parameter.* -= learning_rate * gradient;
        }
    }

    pub fn parameterCount(self: Conv2d) usize {
        return self.weights.data.len + self.bias.data.len;
    }

    fn validateInput(self: Conv2d, input: Matrix) !void {
        if (input.rows == 0 or input.cols != try self.config.input.elementCount()) {
            return error.DimensionMismatch;
        }
    }

    fn sourceCoordinate(self: Conv2d, output: usize, kernel: usize) ?usize {
        const padded = output * self.config.stride + kernel;
        if (padded < self.config.padding) return null;
        const source = padded - self.config.padding;
        return if (source < self.config.input.height) source else null;
    }

    fn sourceCoordinateX(self: Conv2d, output: usize, kernel: usize) ?usize {
        const padded = output * self.config.stride + kernel;
        if (padded < self.config.padding) return null;
        const source = padded - self.config.padding;
        return if (source < self.config.input.width) source else null;
    }

    fn inputIndex(self: Conv2d, y: usize, x: usize, channel: usize) usize {
        return (y * self.config.input.width + x) * self.config.input.channels + channel;
    }

    fn outputIndex(self: Conv2d, y: usize, x: usize, channel: usize) usize {
        return (y * self.output_shape.width + x) * self.output_shape.channels + channel;
    }

    fn weightIndex(
        self: Conv2d,
        y: usize,
        x: usize,
        input_channel: usize,
        output_channel: usize,
    ) usize {
        const row = (y * self.config.kernel_width + x) *
            self.config.input.channels + input_channel;
        return row * self.config.out_channels + output_channel;
    }
};

pub const Conv2dGradients = struct {
    input: Matrix,
    weights: Matrix,
    bias: Matrix,

    pub fn deinit(self: *Conv2dGradients) void {
        self.input.deinit();
        self.weights.deinit();
        self.bias.deinit();
        self.* = undefined;
    }
};

pub const MaxPool2dConfig = struct {
    input: ImageShape,
    window_height: usize,
    window_width: usize,
    stride: usize,

    pub fn outputShape(self: MaxPool2dConfig) !ImageShape {
        try self.input.validate();
        if (self.window_height == 0 or self.window_width == 0 or self.stride == 0) {
            return error.InvalidDimension;
        }
        if (self.window_height > self.input.height or self.window_width > self.input.width) {
            return error.WindowTooLarge;
        }
        return .{
            .height = (self.input.height - self.window_height) / self.stride + 1,
            .width = (self.input.width - self.window_width) / self.stride + 1,
            .channels = self.input.channels,
        };
    }
};

/// Max pooling over flattened NHWC batches. Backward recomputes each winning
/// input index and sends the upstream gradient to the first matching maximum.
pub const MaxPool2d = struct {
    allocator: std.mem.Allocator,
    config: MaxPool2dConfig,
    output_shape: ImageShape,

    pub fn init(allocator: std.mem.Allocator, config: MaxPool2dConfig) !MaxPool2d {
        return .{
            .allocator = allocator,
            .config = config,
            .output_shape = try config.outputShape(),
        };
    }

    pub fn forward(self: MaxPool2d, input: Matrix) !Matrix {
        try self.validateInput(input);
        var output = try Matrix.init(
            self.allocator,
            input.rows,
            try self.output_shape.elementCount(),
        );
        errdefer output.deinit();
        for (0..input.rows) |batch| {
            for (0..self.output_shape.height) |output_y| {
                for (0..self.output_shape.width) |output_x| {
                    for (0..self.output_shape.channels) |channel| {
                        const winner = self.winnerIndex(input, batch, output_y, output_x, channel);
                        const output_index = self.outputIndex(output_y, output_x, channel);
                        output.data[batch * output.cols + output_index] =
                            input.data[batch * input.cols + winner];
                    }
                }
            }
        }
        return output;
    }

    pub fn backward(
        self: MaxPool2d,
        input: Matrix,
        output_gradient: Matrix,
    ) !Matrix {
        try self.validateInput(input);
        if (output_gradient.rows != input.rows or
            output_gradient.cols != try self.output_shape.elementCount())
        {
            return error.DimensionMismatch;
        }
        var input_gradient = try Matrix.init(self.allocator, input.rows, input.cols);
        errdefer input_gradient.deinit();
        for (0..input.rows) |batch| {
            for (0..self.output_shape.height) |output_y| {
                for (0..self.output_shape.width) |output_x| {
                    for (0..self.output_shape.channels) |channel| {
                        const winner = self.winnerIndex(input, batch, output_y, output_x, channel);
                        const output_index = self.outputIndex(output_y, output_x, channel);
                        input_gradient.data[batch * input.cols + winner] +=
                            output_gradient.data[batch * output_gradient.cols + output_index];
                    }
                }
            }
        }
        return input_gradient;
    }

    fn validateInput(self: MaxPool2d, input: Matrix) !void {
        if (input.rows == 0 or input.cols != try self.config.input.elementCount()) {
            return error.DimensionMismatch;
        }
    }

    fn winnerIndex(
        self: MaxPool2d,
        input: Matrix,
        batch: usize,
        output_y: usize,
        output_x: usize,
        channel: usize,
    ) usize {
        var winner = self.inputIndex(
            output_y * self.config.stride,
            output_x * self.config.stride,
            channel,
        );
        var maximum = input.data[batch * input.cols + winner];
        for (0..self.config.window_height) |window_y| {
            for (0..self.config.window_width) |window_x| {
                const index = self.inputIndex(
                    output_y * self.config.stride + window_y,
                    output_x * self.config.stride + window_x,
                    channel,
                );
                const value = input.data[batch * input.cols + index];
                if (value > maximum) {
                    maximum = value;
                    winner = index;
                }
            }
        }
        return winner;
    }

    fn inputIndex(self: MaxPool2d, y: usize, x: usize, channel: usize) usize {
        return (y * self.config.input.width + x) * self.config.input.channels + channel;
    }

    fn outputIndex(self: MaxPool2d, y: usize, x: usize, channel: usize) usize {
        return (y * self.output_shape.width + x) * self.output_shape.channels + channel;
    }
};

test "conv2d forward and backward expose spatial arithmetic" {
    const testing = std.testing;
    var prng = std.Random.DefaultPrng.init(1);
    var convolution = try Conv2d.init(testing.allocator, .{
        .input = .{ .height = 3, .width = 3, .channels = 1 },
        .out_channels = 1,
        .kernel_height = 2,
        .kernel_width = 2,
    }, prng.random());
    defer convolution.deinit();
    @memset(convolution.weights.data, 1);
    @memset(convolution.bias.data, 0);

    var input = try Matrix.init(testing.allocator, 1, 9);
    defer input.deinit();
    for (input.data, 0..) |*value, index| value.* = @floatFromInt(index + 1);
    var output = try convolution.forward(input);
    defer output.deinit();
    try testing.expectEqualSlices(f64, &.{ 12, 16, 24, 28 }, output.data);

    var output_gradient = try Matrix.init(testing.allocator, 1, 4);
    defer output_gradient.deinit();
    @memset(output_gradient.data, 1);
    var gradients = try convolution.backward(input, output_gradient);
    defer gradients.deinit();
    try testing.expectEqualSlices(f64, &.{ 1, 2, 1, 2, 4, 2, 1, 2, 1 }, gradients.input.data);
    try testing.expectEqualSlices(f64, &.{ 12, 16, 24, 28 }, gradients.weights.data);
    try testing.expectEqualSlices(f64, &.{4}, gradients.bias.data);
}

test "conv2d handles padding and validates gradient updates" {
    const testing = std.testing;
    var prng = std.Random.DefaultPrng.init(2);
    var convolution = try Conv2d.init(testing.allocator, .{
        .input = .{ .height = 2, .width = 2, .channels = 1 },
        .out_channels = 2,
        .kernel_height = 3,
        .kernel_width = 3,
        .padding = 1,
    }, prng.random());
    defer convolution.deinit();
    try testing.expectEqual(@as(usize, 2), convolution.output_shape.height);
    try testing.expectEqual(@as(usize, 2), convolution.output_shape.width);
    try testing.expectEqual(@as(usize, 20), convolution.parameterCount());
    const borrowed_gradients: Conv2dGradients = .{
        .input = convolution.weights,
        .weights = convolution.weights,
        .bias = convolution.bias,
    };
    try testing.expectError(
        error.InvalidLearningRate,
        convolution.applyGradients(borrowed_gradients, 0),
    );
}

test "max pool routes gradients to window maxima" {
    const testing = std.testing;
    const pool = try MaxPool2d.init(testing.allocator, .{
        .input = .{ .height = 4, .width = 4, .channels = 1 },
        .window_height = 2,
        .window_width = 2,
        .stride = 2,
    });
    var input = try Matrix.init(testing.allocator, 1, 16);
    defer input.deinit();
    for (input.data, 0..) |*value, index| value.* = @floatFromInt(index + 1);
    var output = try pool.forward(input);
    defer output.deinit();
    try testing.expectEqualSlices(f64, &.{ 6, 8, 14, 16 }, output.data);

    var output_gradient = try Matrix.init(testing.allocator, 1, 4);
    defer output_gradient.deinit();
    @memset(output_gradient.data, 1);
    var input_gradient = try pool.backward(input, output_gradient);
    defer input_gradient.deinit();
    try testing.expectEqualSlices(
        f64,
        &.{ 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1 },
        input_gradient.data,
    );
}
