const std = @import("std");
const builtin = @import("builtin");
const nn = @import("nn");

const Matrix = nn.Matrix;
const Network = nn.Network;
const Activation = nn.Activation;
const events = @import("experiment_events");

const point_count: usize = 1000;
const batch_size: usize = 32;
const display_point_count: usize = 300;
const probability_grid_size: usize = 32;
const default_epochs: usize = 1000;
const default_learning_rate: f64 = 0.001;
const default_seed: u64 = 42;

const Options = struct {
    format: events.Format = .human,
    epochs: usize = default_epochs,
    learning_rate: f64 = default_learning_rate,
    seed: u64 = default_seed,
};

const Sample = struct {
    x: f64,
    y: f64,
    label: u8,
};

const Probability = struct {
    x: f64,
    y: f64,
    value: f64,
};

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const options = try parseArgs(args[1..]);

    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var network = Network.init(allocator, options.learning_rate, .BinaryCrossEntropy);
    defer network.deinit();
    var model_prng = std.Random.DefaultPrng.init(options.seed ^ 0x9e3779b97f4a7c15);
    const model_random = model_prng.random();
    try network.addLayerWithRandom(2, 16, Activation.relu, Activation.relu_derivative, model_random);
    try network.addLayerWithRandom(16, 16, Activation.relu, Activation.relu_derivative, model_random);
    try network.addLayerWithRandom(16, 1, Activation.sigmoid, Activation.sigmoid_derivative, model_random);

    var training_data = try Matrix.init(allocator, point_count, 2);
    var labels = try Matrix.init(allocator, point_count, 1);
    defer training_data.deinit();
    defer labels.deinit();
    try generateDataset(&training_data, &labels, options.seed);

    var stdout_buffer: [32 * 1024]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writerStreaming(std.Options.debug_io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};

    if (options.format == .ndjson) {
        const samples = try displaySamples(training_data, labels);
        try events.emit(stdout, .{
            .v = 1,
            .type = "run_started",
            .experiment = "binary-classification",
            .data = .{
                .config = .{
                    .epochs = options.epochs,
                    .learning_rate = options.learning_rate,
                    .seed = options.seed,
                },
                .topology = &[_]usize{ 2, 16, 16, 1 },
                .activations = &[_][]const u8{ "relu", "relu", "sigmoid" },
                .samples = samples,
                .boundary_radius = 0.5,
                .grid_size = probability_grid_size,
                .execution = .{
                    .requested_backend = "cpu",
                    .selected_backend = "cpu",
                    .optimize = @tagName(builtin.mode),
                },
            },
        });
        try emitSnapshot(allocator, stdout, &network, 0, options.epochs);
    } else {
        std.debug.print("Training network...\n", .{});
    }

    var final_loss: f64 = 0;
    for (0..options.epochs) |epoch| {
        var total_loss: f64 = 0;
        var batches: usize = 0;
        var batch_start: usize = 0;
        while (batch_start < point_count) : (batch_start += batch_size) {
            const batch_end = @min(batch_start + batch_size, point_count);
            var batch_inputs = try Matrix.extractBatch(training_data, batch_start, batch_end, allocator);
            var batch_labels = try Matrix.extractBatch(labels, batch_start, batch_end, allocator);
            defer batch_inputs.deinit();
            defer batch_labels.deinit();
            total_loss += try network.trainBatch(batch_inputs, batch_labels);
            batches += 1;
        }
        final_loss = total_loss / @as(f64, @floatFromInt(batches));

        if (options.format == .ndjson) {
            if (events.shouldEmitMetric(epoch, options.epochs)) {
                try events.emit(stdout, .{
                    .v = 1,
                    .type = "metric",
                    .experiment = "binary-classification",
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
                std.debug.print("Epoch {d}: Loss = {d:.6}\n", .{ epoch, final_loss });
            }
        }
    }

    const accuracy = try evaluateAccuracy(allocator, &network, training_data, labels);
    if (options.format == .ndjson) {
        const probabilities = try probabilityGrid(allocator, &network);
        try events.emit(stdout, .{
            .v = 1,
            .type = "run_completed",
            .experiment = "binary-classification",
            .step = options.epochs,
            .total_steps = options.epochs,
            .data = .{
                .final_loss = final_loss,
                .training_accuracy = accuracy,
                .probabilities = probabilities,
            },
        });
    } else {
        try printHumanBoundary(allocator, &network);
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

fn generateDataset(training_data: *Matrix, labels: *Matrix, seed: u64) !void {
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    for (0..point_count) |index| {
        const x = random.float(f64) * 2.0 - 1.0;
        const y = random.float(f64) * 2.0 - 1.0;
        try training_data.set(index, 0, x);
        try training_data.set(index, 1, y);
        try labels.set(index, 0, if (@sqrt(x * x + y * y) <= 0.5) 1 else 0);
    }
}

fn displaySamples(training_data: Matrix, labels: Matrix) ![display_point_count]Sample {
    var samples: [display_point_count]Sample = undefined;
    for (&samples, 0..) |*sample, index| {
        const row = index * (point_count / display_point_count);
        sample.* = .{
            .x = try training_data.get(row, 0),
            .y = try training_data.get(row, 1),
            .label = if (try labels.get(row, 0) >= 0.5) 1 else 0,
        };
    }
    return samples;
}

fn probabilityGrid(allocator: std.mem.Allocator, network: *Network) ![probability_grid_size * probability_grid_size]Probability {
    const count = probability_grid_size * probability_grid_size;
    var inputs = try Matrix.init(allocator, count, 2);
    defer inputs.deinit();
    for (0..probability_grid_size) |row| {
        for (0..probability_grid_size) |column| {
            const index = row * probability_grid_size + column;
            const x = -1.0 + 2.0 * @as(f64, @floatFromInt(column)) / @as(f64, @floatFromInt(probability_grid_size - 1));
            const y = -1.0 + 2.0 * @as(f64, @floatFromInt(row)) / @as(f64, @floatFromInt(probability_grid_size - 1));
            try inputs.set(index, 0, x);
            try inputs.set(index, 1, y);
        }
    }
    var output = try network.forward(inputs);
    defer output.deinit();
    var probabilities: [count]Probability = undefined;
    for (&probabilities, 0..) |*probability, index| {
        probability.* = .{
            .x = try inputs.get(index, 0),
            .y = try inputs.get(index, 1),
            .value = try output.get(index, 0),
        };
    }
    return probabilities;
}

fn emitSnapshot(allocator: std.mem.Allocator, writer: *std.Io.Writer, network: *Network, step: usize, total: usize) !void {
    const probabilities = try probabilityGrid(allocator, network);
    try events.emit(writer, .{
        .v = 1,
        .type = "snapshot",
        .experiment = "binary-classification",
        .step = step,
        .total_steps = total,
        .data = .{ .kind = "decision_boundary", .probabilities = probabilities },
    });
}

fn evaluateAccuracy(allocator: std.mem.Allocator, network: *Network, training_data: Matrix, labels: Matrix) !f64 {
    _ = allocator;
    var predictions = try network.forward(training_data);
    defer predictions.deinit();
    var correct: usize = 0;
    for (0..point_count) |index| {
        if ((try predictions.get(index, 0) >= 0.5) == (try labels.get(index, 0) >= 0.5)) correct += 1;
    }
    return @as(f64, @floatFromInt(correct)) / @as(f64, @floatFromInt(point_count));
}

fn printHumanBoundary(allocator: std.mem.Allocator, network: *Network) !void {
    std.debug.print("\nDecision boundary:\n", .{});
    const grid_size: usize = 20;
    const step = 2.0 / @as(f64, @floatFromInt(grid_size));
    var y: f64 = -1.0;
    for (0..grid_size) |_| {
        var x: f64 = -1.0;
        for (0..grid_size) |_| {
            var input = try Matrix.init(allocator, 1, 2);
            defer input.deinit();
            try input.set(0, 0, x);
            try input.set(0, 1, y);
            var output = try network.forward(input);
            defer output.deinit();
            std.debug.print("{s} ", .{if (try output.get(0, 0) > 0.5) "●" else "○"});
            x += step;
        }
        std.debug.print("\n", .{});
        y += step;
    }
}

test "binary classification learning options parse and validate" {
    const options = try parseArgs(&.{ "--format", "ndjson", "--epochs", "120", "--learning-rate", "0.004", "--seed", "11" });
    try std.testing.expectEqual(events.Format.ndjson, options.format);
    try std.testing.expectEqual(@as(usize, 120), options.epochs);
    try std.testing.expectEqual(@as(u64, 11), options.seed);
    try std.testing.expectApproxEqAbs(@as(f64, 0.004), options.learning_rate, 1e-12);
    try std.testing.expectError(error.InvalidLearningRate, parseArgs(&.{ "--learning-rate", "nan" }));
}

test "classification dataset is stable for a supplied seed" {
    const allocator = std.testing.allocator;
    var first_data = try Matrix.init(allocator, point_count, 2);
    defer first_data.deinit();
    var first_labels = try Matrix.init(allocator, point_count, 1);
    defer first_labels.deinit();
    var second_data = try Matrix.init(allocator, point_count, 2);
    defer second_data.deinit();
    var second_labels = try Matrix.init(allocator, point_count, 1);
    defer second_labels.deinit();

    try generateDataset(&first_data, &first_labels, 42);
    try generateDataset(&second_data, &second_labels, 42);
    try std.testing.expectEqualSlices(f64, first_data.data, second_data.data);
    try std.testing.expectEqualSlices(f64, first_labels.data, second_labels.data);
}
