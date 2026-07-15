const std = @import("std");
const nn = @import("nn");

const image_shape: nn.Spatial.ImageShape = .{
    .height = 6,
    .width = 6,
    .channels = 1,
};
const class_count: usize = 2;
const train_samples: usize = 96;
const test_samples: usize = 48;
const default_epochs: usize = 80;
const default_learning_rate: f64 = 0.08;

const Dataset = struct {
    allocator: std.mem.Allocator,
    inputs: nn.Matrix,
    labels: []usize,

    fn deinit(self: *Dataset) void {
        self.inputs.deinit();
        self.allocator.free(self.labels);
        self.* = undefined;
    }
};

const Metrics = struct {
    loss: f64,
    accuracy: f64,
};

const Forward = struct {
    convolution: nn.Matrix,
    activation: nn.Matrix,
    pooled: nn.Matrix,
    logits: nn.Matrix,
    probabilities: nn.Matrix,

    fn deinit(self: *Forward) void {
        self.convolution.deinit();
        self.activation.deinit();
        self.pooled.deinit();
        self.logits.deinit();
        self.probabilities.deinit();
        self.* = undefined;
    }
};

const Model = struct {
    allocator: std.mem.Allocator,
    convolution: nn.Spatial.Conv2d,
    pool: nn.Spatial.MaxPool2d,
    classifier_weights: nn.Matrix,
    classifier_bias: nn.Matrix,

    fn init(allocator: std.mem.Allocator, seed: u64) !Model {
        var prng = std.Random.DefaultPrng.init(seed);
        var random = prng.random();
        var convolution = try nn.Spatial.Conv2d.init(allocator, .{
            .input = image_shape,
            .out_channels = 4,
            .kernel_height = 3,
            .kernel_width = 3,
            .padding = 1,
        }, random);
        errdefer convolution.deinit();
        const pool = try nn.Spatial.MaxPool2d.init(allocator, .{
            .input = convolution.output_shape,
            .window_height = 2,
            .window_width = 2,
            .stride = 2,
        });
        const pooled_features = try pool.output_shape.elementCount();
        var classifier_weights = try nn.Matrix.init(
            allocator,
            pooled_features,
            class_count,
        );
        errdefer classifier_weights.deinit();
        var classifier_bias = try nn.Matrix.init(allocator, 1, class_count);
        errdefer classifier_bias.deinit();
        const bound = @sqrt(6.0 / @as(f64, @floatFromInt(pooled_features + class_count)));
        for (classifier_weights.data) |*weight| {
            weight.* = (random.float(f64) * 2 - 1) * bound;
        }
        return .{
            .allocator = allocator,
            .convolution = convolution,
            .pool = pool,
            .classifier_weights = classifier_weights,
            .classifier_bias = classifier_bias,
        };
    }

    fn deinit(self: *Model) void {
        self.convolution.deinit();
        self.classifier_weights.deinit();
        self.classifier_bias.deinit();
        self.* = undefined;
    }

    fn forward(self: Model, input: nn.Matrix) !Forward {
        var convolution = try self.convolution.forward(input);
        errdefer convolution.deinit();
        var activation = try relu(self.allocator, convolution);
        errdefer activation.deinit();
        var pooled = try self.pool.forward(activation);
        errdefer pooled.deinit();
        var logits = try pooled.dotProduct(self.classifier_weights, self.allocator);
        errdefer logits.deinit();
        addRowBias(&logits, self.classifier_bias);
        const probabilities = try softmax(self.allocator, logits);
        return .{
            .convolution = convolution,
            .activation = activation,
            .pooled = pooled,
            .logits = logits,
            .probabilities = probabilities,
        };
    }

    fn trainBatch(
        self: *Model,
        input: nn.Matrix,
        labels: []const usize,
        learning_rate: f64,
    ) !Metrics {
        var forward_pass = try self.forward(input);
        defer forward_pass.deinit();
        const metrics = measure(forward_pass.probabilities, labels);

        var logits_gradient = try nn.Matrix.copy(forward_pass.probabilities, self.allocator);
        defer logits_gradient.deinit();
        const batch_scale = 1.0 / @as(f64, @floatFromInt(input.rows));
        for (labels, 0..) |label, row| {
            logits_gradient.data[row * class_count + label] -= 1;
        }
        for (logits_gradient.data) |*gradient| gradient.* *= batch_scale;

        var pooled_transpose = try forward_pass.pooled.transpose(self.allocator);
        defer pooled_transpose.deinit();
        var classifier_weights_gradient = try pooled_transpose.dotProduct(
            logits_gradient,
            self.allocator,
        );
        defer classifier_weights_gradient.deinit();
        var classifier_bias_gradient = try logits_gradient.sumRows(self.allocator);
        defer classifier_bias_gradient.deinit();
        var classifier_weights_transpose = try self.classifier_weights.transpose(self.allocator);
        defer classifier_weights_transpose.deinit();
        var pooled_gradient = try logits_gradient.dotProduct(
            classifier_weights_transpose,
            self.allocator,
        );
        defer pooled_gradient.deinit();
        var activation_gradient = try self.pool.backward(
            forward_pass.activation,
            pooled_gradient,
        );
        defer activation_gradient.deinit();
        applyReluDerivative(&activation_gradient, forward_pass.convolution);
        var convolution_gradients = try self.convolution.backward(
            input,
            activation_gradient,
        );
        defer convolution_gradients.deinit();

        try self.convolution.applyGradients(convolution_gradients, learning_rate);
        subtractScaled(
            &self.classifier_weights,
            classifier_weights_gradient,
            learning_rate,
        );
        subtractScaled(
            &self.classifier_bias,
            classifier_bias_gradient,
            learning_rate,
        );
        return metrics;
    }

    fn evaluate(self: Model, input: nn.Matrix, labels: []const usize) !Metrics {
        var forward_pass = try self.forward(input);
        defer forward_pass.deinit();
        return measure(forward_pass.probabilities, labels);
    }

    fn parameterCount(self: Model) usize {
        return self.convolution.parameterCount() +
            self.classifier_weights.data.len + self.classifier_bias.data.len;
    }
};

pub fn main() !void {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var training_data = try generatePatterns(allocator, train_samples, 41);
    defer training_data.deinit();
    var testing_data = try generatePatterns(allocator, test_samples, 99);
    defer testing_data.deinit();
    var model = try Model.init(allocator, 7);
    defer model.deinit();

    const initial = try model.evaluate(training_data.inputs, training_data.labels);
    std.debug.print("CNN pattern classifier\n", .{});
    std.debug.print(
        "6x6 image -> 3x3 conv (4 filters) -> ReLU -> 2x2 max-pool -> softmax\n",
        .{},
    );
    std.debug.print("parameters: {d}\n", .{model.parameterCount()});
    std.debug.print(
        "epoch {d:>3}: loss {d:.5}, train accuracy {d:.2}%\n",
        .{ 0, initial.loss, initial.accuracy * 100 },
    );
    for (0..default_epochs) |epoch| {
        const metrics = try model.trainBatch(
            training_data.inputs,
            training_data.labels,
            default_learning_rate,
        );
        if ((epoch + 1) % 20 == 0) {
            std.debug.print(
                "epoch {d:>3}: loss {d:.5}, train accuracy {d:.2}%\n",
                .{ epoch + 1, metrics.loss, metrics.accuracy * 100 },
            );
        }
    }
    const final_train = try model.evaluate(training_data.inputs, training_data.labels);
    const final_test = try model.evaluate(testing_data.inputs, testing_data.labels);
    std.debug.print(
        "final: loss {d:.5}, train {d:.2}%, held-out {d:.2}%\n",
        .{ final_train.loss, final_train.accuracy * 100, final_test.accuracy * 100 },
    );
    printFilter(model.convolution, 0);
}

fn generatePatterns(allocator: std.mem.Allocator, count: usize, seed: u64) !Dataset {
    if (count == 0) return error.EmptyDataset;
    var inputs = try nn.Matrix.init(allocator, count, try image_shape.elementCount());
    errdefer inputs.deinit();
    const labels = try allocator.alloc(usize, count);
    errdefer allocator.free(labels);
    var prng = std.Random.DefaultPrng.init(seed);
    var random = prng.random();

    for (0..count) |sample| {
        const label = sample % class_count;
        labels[sample] = label;
        const line = 2 + random.uintLessThan(usize, 2);
        for (0..image_shape.height) |y| {
            for (0..image_shape.width) |x| {
                const on_line = if (label == 0) x == line else y == line;
                const noise = random.float(f64) * 0.12;
                const value = if (on_line) 0.88 + noise else noise;
                inputs.data[sample * inputs.cols + y * image_shape.width + x] = value;
            }
        }
    }
    return .{ .allocator = allocator, .inputs = inputs, .labels = labels };
}

fn relu(allocator: std.mem.Allocator, input: nn.Matrix) !nn.Matrix {
    const result = try nn.Matrix.init(allocator, input.rows, input.cols);
    for (input.data, result.data) |value, *output| output.* = @max(value, 0);
    return result;
}

fn applyReluDerivative(gradient: *nn.Matrix, pre_activation: nn.Matrix) void {
    for (gradient.data, pre_activation.data) |*value, activation| {
        if (activation <= 0) value.* = 0;
    }
}

fn addRowBias(matrix: *nn.Matrix, bias: nn.Matrix) void {
    std.debug.assert(bias.rows == 1 and bias.cols == matrix.cols);
    for (0..matrix.rows) |row| {
        for (0..matrix.cols) |column| {
            matrix.data[row * matrix.cols + column] += bias.data[column];
        }
    }
}

fn softmax(allocator: std.mem.Allocator, logits: nn.Matrix) !nn.Matrix {
    var probabilities = try nn.Matrix.init(allocator, logits.rows, logits.cols);
    for (0..logits.rows) |row| {
        const offset = row * logits.cols;
        var maximum = logits.data[offset];
        for (logits.data[offset + 1 .. offset + logits.cols]) |value| {
            maximum = @max(maximum, value);
        }
        var total: f64 = 0;
        for (0..logits.cols) |column| {
            const value = @exp(logits.data[offset + column] - maximum);
            probabilities.data[offset + column] = value;
            total += value;
        }
        for (probabilities.data[offset .. offset + logits.cols]) |*value| value.* /= total;
    }
    return probabilities;
}

fn measure(probabilities: nn.Matrix, labels: []const usize) Metrics {
    std.debug.assert(probabilities.rows == labels.len);
    var loss: f64 = 0;
    var correct: usize = 0;
    for (labels, 0..) |label, row| {
        const offset = row * probabilities.cols;
        const probability = @max(probabilities.data[offset + label], 1e-12);
        loss -= @log(probability);
        var prediction: usize = 0;
        for (1..probabilities.cols) |column| {
            if (probabilities.data[offset + column] > probabilities.data[offset + prediction]) {
                prediction = column;
            }
        }
        if (prediction == label) correct += 1;
    }
    const count = @as(f64, @floatFromInt(labels.len));
    return .{
        .loss = loss / count,
        .accuracy = @as(f64, @floatFromInt(correct)) / count,
    };
}

fn subtractScaled(parameter: *nn.Matrix, gradient: nn.Matrix, scale: f64) void {
    std.debug.assert(parameter.rows == gradient.rows and parameter.cols == gradient.cols);
    for (parameter.data, gradient.data) |*value, delta| value.* -= scale * delta;
}

fn printFilter(convolution: nn.Spatial.Conv2d, output_channel: usize) void {
    std.debug.print("first learned 3x3 filter:\n", .{});
    for (0..convolution.config.kernel_height) |y| {
        for (0..convolution.config.kernel_width) |x| {
            const row = y * convolution.config.kernel_width + x;
            const index = row * convolution.config.out_channels + output_channel;
            std.debug.print("{d:>8.4}", .{convolution.weights.data[index]});
        }
        std.debug.print("\n", .{});
    }
}

test "CNN example learns orientation from synthetic images" {
    const testing = std.testing;
    var training_data = try generatePatterns(testing.allocator, train_samples, 41);
    defer training_data.deinit();
    var testing_data = try generatePatterns(testing.allocator, test_samples, 99);
    defer testing_data.deinit();
    var model = try Model.init(testing.allocator, 7);
    defer model.deinit();

    const initial = try model.evaluate(training_data.inputs, training_data.labels);
    for (0..60) |_| {
        _ = try model.trainBatch(
            training_data.inputs,
            training_data.labels,
            default_learning_rate,
        );
    }
    const final = try model.evaluate(training_data.inputs, training_data.labels);
    const held_out = try model.evaluate(testing_data.inputs, testing_data.labels);
    try testing.expect(final.loss < initial.loss * 0.35);
    try testing.expect(final.accuracy >= 0.95);
    try testing.expect(held_out.accuracy >= 0.9);
}

test "CNN pattern data is deterministic and balanced" {
    const testing = std.testing;
    var first = try generatePatterns(testing.allocator, 8, 12);
    defer first.deinit();
    var second = try generatePatterns(testing.allocator, 8, 12);
    defer second.deinit();
    try testing.expectEqualSlices(f64, first.inputs.data, second.inputs.data);
    try testing.expectEqualSlices(usize, first.labels, second.labels);
    try testing.expectEqualSlices(usize, &.{ 0, 1, 0, 1, 0, 1, 0, 1 }, first.labels);
}
