const std = @import("std");
const nn = @import("nn");

const window: usize = 12;
const train_windows: usize = 80;
const test_windows: usize = 32;
const training_steps: usize = 220;

const Windows = struct {
    inputs: nn.Matrix,
    targets: nn.Matrix,
    mask: []f64,

    fn deinit(self: *Windows, allocator: std.mem.Allocator) void {
        self.inputs.deinit();
        self.targets.deinit();
        allocator.free(self.mask);
        self.* = undefined;
    }
};

const Metrics = struct {
    model_mse: f64,
    persistence_mse: f64,
};

const Result = struct {
    initial_mse: f64,
    final: Metrics,
    learned_weights: [2]f64,
    targets: [4]f64,
    predictions: [4]f64,
};

pub fn main(init: std.process.Init) !void {
    const result = try runExperiment(init.gpa);
    std.debug.print("Causal Conv1d rolling forecast\n", .{});
    std.debug.print(
        "masked held-out MSE: {d:.6} -> {d:.6}\n" ++
            "persistence baseline MSE: {d:.6}\n" ++
            "learned causal kernel: [{d:.3}, {d:.3}]\n",
        .{
            result.initial_mse,
            result.final.model_mse,
            result.final.persistence_mse,
            result.learned_weights[0],
            result.learned_weights[1],
        },
    );
    std.debug.print("rolling predictions:\n", .{});
    for (result.targets, result.predictions) |target, prediction| {
        std.debug.print("  target {d: >7.3}, predicted {d: >7.3}\n", .{ target, prediction });
    }
    std.debug.print(
        "lesson: a shared causal kernel learns the repeating dynamics without reading future values\n",
        .{},
    );
}

fn runExperiment(allocator: std.mem.Allocator) !Result {
    var series: [train_windows + test_windows + window + 2]f64 = undefined;
    generateSeries(&series);
    var training = try makeWindows(allocator, &series, 0, train_windows);
    defer training.deinit(allocator);
    var testing = try makeWindows(
        allocator,
        &series,
        train_windows,
        test_windows,
    );
    defer testing.deinit(allocator);
    var prng = std.Random.DefaultPrng.init(17);
    var convolution = try nn.Spatial.Conv1d.init(allocator, .{
        .input = .{ .steps = window, .channels = 1 },
        .out_channels = 1,
        .kernel_size = 2,
    }, prng.random());
    defer convolution.deinit();
    const initial = try evaluate(allocator, convolution, testing, null);
    for (0..training_steps) |_| {
        var predictions = try convolution.forward(training.inputs);
        defer predictions.deinit();
        var objective = try nn.Sequence.maskedMeanSquaredError(
            allocator,
            predictions.data,
            training.targets.data,
            training.mask,
        );
        defer objective.deinit();
        var output_gradient = try nn.Matrix.init(
            allocator,
            predictions.rows,
            predictions.cols,
        );
        defer output_gradient.deinit();
        @memcpy(output_gradient.data, objective.gradient);
        var gradients = try convolution.backward(training.inputs, output_gradient);
        defer gradients.deinit();
        try convolution.applyGradients(gradients, 0.25);
    }
    var sample_predictions: [4]f64 = undefined;
    const final = try evaluate(allocator, convolution, testing, &sample_predictions);
    var sample_targets: [4]f64 = undefined;
    const sample_row = test_windows - 1;
    @memcpy(
        &sample_targets,
        testing.targets.data[sample_row * window + window - 4 ..][0..4],
    );
    return .{
        .initial_mse = initial.model_mse,
        .final = final,
        .learned_weights = .{ convolution.weights.data[0], convolution.weights.data[1] },
        .targets = sample_targets,
        .predictions = sample_predictions,
    };
}

fn evaluate(
    allocator: std.mem.Allocator,
    convolution: nn.Spatial.Conv1d,
    windows: Windows,
    sample: ?*[4]f64,
) !Metrics {
    var predictions = try convolution.forward(windows.inputs);
    defer predictions.deinit();
    var objective = try nn.Sequence.maskedMeanSquaredError(
        allocator,
        predictions.data,
        windows.targets.data,
        windows.mask,
    );
    defer objective.deinit();
    var persistence_loss: f64 = 0;
    var observed: usize = 0;
    for (windows.inputs.data, windows.targets.data, windows.mask) |input, target, mask| {
        if (mask == 0) continue;
        const difference = input - target;
        persistence_loss += difference * difference;
        observed += 1;
    }
    if (sample) |values| {
        const sample_row = windows.inputs.rows - 1;
        @memcpy(
            values,
            predictions.data[sample_row * window + window - 4 ..][0..4],
        );
    }
    return .{
        .model_mse = objective.loss,
        .persistence_mse = persistence_loss / @as(f64, @floatFromInt(observed)),
    };
}

fn makeWindows(
    allocator: std.mem.Allocator,
    series: []const f64,
    first: usize,
    count: usize,
) !Windows {
    if (first + count + window >= series.len) return error.InsufficientSeries;
    var inputs = try nn.Matrix.init(allocator, count, window);
    errdefer inputs.deinit();
    var targets = try nn.Matrix.init(allocator, count, window);
    errdefer targets.deinit();
    const mask = try allocator.alloc(f64, count * window);
    errdefer allocator.free(mask);
    for (0..count) |row| {
        const start = first + row;
        @memcpy(inputs.data[row * window ..][0..window], series[start..][0..window]);
        @memcpy(targets.data[row * window ..][0..window], series[start + 1 ..][0..window]);
        mask[row * window] = 0;
        @memset(mask[row * window + 1 ..][0 .. window - 1], 1);
    }
    return .{ .inputs = inputs, .targets = targets, .mask = mask };
}

fn generateSeries(series: []f64) void {
    series[0] = 0;
    series[1] = 0.3894183423;
    for (series[2..], 2..) |*value, index| {
        value.* = 1.842121988 * series[index - 1] - series[index - 2];
    }
}

test "causal convolution beats a persistence forecast" {
    const result = try runExperiment(std.testing.allocator);
    try std.testing.expect(result.final.model_mse < result.initial_mse * 0.05);
    try std.testing.expect(result.final.model_mse < result.final.persistence_mse * 0.1);
    try std.testing.expectApproxEqAbs(@as(f64, -1), result.learned_weights[0], 0.05);
    try std.testing.expectApproxEqAbs(@as(f64, 1.8421), result.learned_weights[1], 0.05);
}
