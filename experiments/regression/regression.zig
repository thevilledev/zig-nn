const std = @import("std");
const builtin = @import("builtin");
const nn = @import("nn");

const Network = nn.Network;
const Matrix = nn.Matrix;
const Activation = nn.Activation;
const events = @import("experiment_events");

const sample_count: usize = 1000;
const batch_size: usize = 100;
const curve_points: usize = 65;
const default_epochs: usize = 500;
const default_learning_rate: f64 = 0.001;
const default_seed: u64 = 42;

const Options = struct {
    format: events.Format = .human,
    epochs: usize = default_epochs,
    learning_rate: f64 = default_learning_rate,
    seed: u64 = default_seed,
};

const Point = struct {
    x: f64,
    y: f64,
};

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const options = try parseArgs(args[1..]);

    var gpa: std.heap.DebugAllocator(.{}) = .init;
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    var x_data = try Matrix.init(allocator, sample_count, 1);
    var y_data = try Matrix.init(allocator, sample_count, 1);
    defer x_data.deinit();
    defer y_data.deinit();
    try generateDataset(&x_data, &y_data, options.seed);

    var network = Network.init(allocator, options.learning_rate, .MeanSquaredError);
    defer network.deinit();
    var model_prng = std.Random.DefaultPrng.init(options.seed ^ 0x9e3779b97f4a7c15);
    const model_random = model_prng.random();
    try network.addLayerWithRandom(1, 32, Activation.relu, Activation.relu_derivative, model_random);
    try network.addLayerWithRandom(32, 16, Activation.relu, Activation.relu_derivative, model_random);
    try network.addLayerWithRandom(16, 1, Activation.linear, Activation.linear_derivative, model_random);

    var stdout_buffer: [16 * 1024]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writerStreaming(std.Options.debug_io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};

    if (options.format == .ndjson) {
        const target_curve = makeTargetCurve();
        const training_samples = try sampleTrainingData(x_data, y_data);
        try events.emit(stdout, .{
            .v = 1,
            .type = "run_started",
            .experiment = "regression",
            .data = .{
                .config = .{
                    .epochs = options.epochs,
                    .learning_rate = options.learning_rate,
                    .seed = options.seed,
                },
                .topology = &[_]usize{ 1, 32, 16, 1 },
                .activations = &[_][]const u8{ "relu", "relu", "linear" },
                .target_curve = target_curve,
                .training_samples = training_samples,
                .execution = .{
                    .requested_backend = "cpu",
                    .selected_backend = "cpu",
                    .optimize = @tagName(builtin.mode),
                },
            },
        });
        try emitSnapshot(allocator, stdout, &network, 0, options.epochs);
    }

    var final_loss: f64 = 0;
    for (0..options.epochs) |epoch| {
        var total_loss: f64 = 0;
        var batch: usize = 0;
        while (batch < sample_count / batch_size) : (batch += 1) {
            const start = batch * batch_size;
            const end = @min(start + batch_size, sample_count);
            var batch_x = try x_data.extractBatch(start, end, allocator);
            defer batch_x.deinit();
            var batch_y = try y_data.extractBatch(start, end, allocator);
            defer batch_y.deinit();
            total_loss += try network.trainBatch(batch_x, batch_y);
        }
        final_loss = total_loss / @as(f64, @floatFromInt(sample_count / batch_size));

        if (options.format == .ndjson) {
            if (events.shouldEmitMetric(epoch, options.epochs)) {
                try events.emit(stdout, .{
                    .v = 1,
                    .type = "metric",
                    .experiment = "regression",
                    .step = epoch + 1,
                    .total_steps = options.epochs,
                    .data = .{ .name = "loss", .value = final_loss },
                });
            }
            if (events.shouldEmitSnapshot(epoch, options.epochs)) {
                try emitSnapshot(allocator, stdout, &network, epoch + 1, options.epochs);
            }
        } else {
            const interval = @max(@as(usize, 1), options.epochs / 10);
            if (epoch % interval == 0) {
                try stdout.print("Epoch {d}: Average Loss = {d:.6}\n", .{ epoch, final_loss });
            }
        }
    }

    if (options.format == .ndjson) {
        const prediction_curve = try predictCurve(allocator, &network);
        try events.emit(stdout, .{
            .v = 1,
            .type = "run_completed",
            .experiment = "regression",
            .step = options.epochs,
            .total_steps = options.epochs,
            .data = .{ .final_loss = final_loss, .prediction_curve = prediction_curve },
        });
    } else {
        try printHumanEvaluation(allocator, stdout, &network);
    }
}

fn parseArgs(args: []const []const u8) !Options {
    var options: Options = .{};
    var index: usize = 0;
    while (index < args.len) : (index += 1) {
        const arg = args[index];
        if (std.mem.eql(u8, arg, "--format")) {
            index += 1;
            if (index >= args.len) return error.MissingArgument;
            options.format = try events.parseFormat(args[index]);
        } else if (std.mem.eql(u8, arg, "--epochs")) {
            index += 1;
            if (index >= args.len) return error.MissingArgument;
            options.epochs = try std.fmt.parseInt(usize, args[index], 10);
            if (options.epochs == 0) return error.InvalidEpochCount;
        } else if (std.mem.eql(u8, arg, "--learning-rate")) {
            index += 1;
            if (index >= args.len) return error.MissingArgument;
            options.learning_rate = try std.fmt.parseFloat(f64, args[index]);
            if (!std.math.isFinite(options.learning_rate) or options.learning_rate <= 0) return error.InvalidLearningRate;
        } else if (std.mem.eql(u8, arg, "--seed")) {
            index += 1;
            if (index >= args.len) return error.MissingArgument;
            options.seed = try std.fmt.parseInt(u64, args[index], 10);
        } else {
            return error.UnknownArgument;
        }
    }
    return options;
}

fn generateDataset(x_data: *Matrix, y_data: *Matrix, seed: u64) !void {
    var rng = std.Random.DefaultPrng.init(seed);
    const random = rng.random();
    for (0..sample_count) |index| {
        const u = random.float(f64);
        const x = if (u < 0.4)
            if (random.boolean()) -4.0 + random.float(f64) * 1.5 else 2.5 + random.float(f64) * 1.5
        else
            -2.5 + random.float(f64) * 5.0;
        try x_data.set(index, 0, x);
        try y_data.set(index, 0, targetFunction(x));
    }
}

fn targetFunction(x: f64) f64 {
    return x * x * @sin(x);
}

fn makeTargetCurve() [curve_points]Point {
    var points: [curve_points]Point = undefined;
    for (&points, 0..) |*point, index| {
        const x = -4.0 + 8.0 * @as(f64, @floatFromInt(index)) / @as(f64, @floatFromInt(curve_points - 1));
        point.* = .{ .x = x, .y = targetFunction(x) };
    }
    return points;
}

fn sampleTrainingData(x_data: Matrix, y_data: Matrix) ![80]Point {
    var points: [80]Point = undefined;
    for (&points, 0..) |*point, index| {
        const row = index * (sample_count / points.len);
        point.* = .{ .x = try x_data.get(row, 0), .y = try y_data.get(row, 0) };
    }
    return points;
}

fn predictCurve(allocator: std.mem.Allocator, network: *Network) ![curve_points]Point {
    var input = try Matrix.init(allocator, curve_points, 1);
    defer input.deinit();
    for (0..curve_points) |index| {
        const x = -4.0 + 8.0 * @as(f64, @floatFromInt(index)) / @as(f64, @floatFromInt(curve_points - 1));
        try input.set(index, 0, x);
    }
    var output = try network.forward(input);
    defer output.deinit();
    var points: [curve_points]Point = undefined;
    for (&points, 0..) |*point, index| {
        point.* = .{ .x = try input.get(index, 0), .y = try output.get(index, 0) };
    }
    return points;
}

fn emitSnapshot(allocator: std.mem.Allocator, writer: *std.Io.Writer, network: *Network, step: usize, total: usize) !void {
    const prediction_curve = try predictCurve(allocator, network);
    try events.emit(writer, .{
        .v = 1,
        .type = "snapshot",
        .experiment = "regression",
        .step = step,
        .total_steps = total,
        .data = .{ .kind = "regression_curve", .predictions = prediction_curve },
    });
}

fn printHumanEvaluation(allocator: std.mem.Allocator, writer: *std.Io.Writer, network: *Network) !void {
    try writer.print("\nTesting the network:\n", .{});
    const test_points = [_]f64{ -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0 };
    for (test_points) |x| {
        var input = try Matrix.init(allocator, 1, 1);
        defer input.deinit();
        try input.set(0, 0, x);
        var prediction = try network.forward(input);
        defer prediction.deinit();
        const actual = targetFunction(x);
        const predicted = try prediction.get(0, 0);
        try writer.print("x = {d:6.3}: Predicted = {d:8.3}, Actual = {d:8.3}, Error = {d:8.3}\n", .{
            x,
            predicted,
            actual,
            @abs(predicted - actual),
        });
    }
}

test "regression learning options parse and validate" {
    const options = try parseArgs(&.{ "--format", "ndjson", "--epochs", "75", "--learning-rate", "0.002", "--seed", "9" });
    try std.testing.expectEqual(events.Format.ndjson, options.format);
    try std.testing.expectEqual(@as(usize, 75), options.epochs);
    try std.testing.expectEqual(@as(u64, 9), options.seed);
    try std.testing.expectApproxEqAbs(@as(f64, 0.002), options.learning_rate, 1e-12);
    try std.testing.expectError(error.InvalidEpochCount, parseArgs(&.{ "--epochs", "0" }));
}

test "regression dataset is stable for a supplied seed" {
    const allocator = std.testing.allocator;
    var first_x = try Matrix.init(allocator, sample_count, 1);
    defer first_x.deinit();
    var first_y = try Matrix.init(allocator, sample_count, 1);
    defer first_y.deinit();
    var second_x = try Matrix.init(allocator, sample_count, 1);
    defer second_x.deinit();
    var second_y = try Matrix.init(allocator, sample_count, 1);
    defer second_y.deinit();

    try generateDataset(&first_x, &first_y, 42);
    try generateDataset(&second_x, &second_y, 42);
    try std.testing.expectEqualSlices(f64, first_x.data, second_x.data);
    try std.testing.expectEqualSlices(f64, first_y.data, second_y.data);
}
