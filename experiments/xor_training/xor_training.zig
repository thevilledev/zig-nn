const std = @import("std");
const nn = @import("nn");

const Network = nn.Network;
const Matrix = nn.Matrix;
const Activation = nn.Activation;
const events = @import("experiment_events");

const default_epochs: usize = 10_000;
const default_learning_rate: f64 = 0.3;
const default_seed: u64 = 42;
const batch_size: usize = 4;

const Options = struct {
    output_file: ?[]const u8 = null,
    format: events.Format = .human,
    epochs: usize = default_epochs,
    learning_rate: f64 = default_learning_rate,
    seed: u64 = default_seed,
};

const Prediction = struct {
    x1: f64,
    x2: f64,
    predicted: f64,
    expected: f64,
};

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const options = try parseArgs(args[1..]);

    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writerStreaming(std.Options.debug_io, &stdout_buffer);
    const writer = &stdout_writer.interface;
    defer writer.flush() catch {};

    if (options.format == .human) {
        try writer.print("\n=== XOR Neural Network Training Experiment ===\n\n", .{});
    }

    var network = Network.init(allocator, options.learning_rate, .MeanSquaredError);
    defer network.deinit();

    var model_prng = std.Random.DefaultPrng.init(options.seed ^ 0x9e3779b97f4a7c15);
    const model_random = model_prng.random();
    try network.addLayerWithRandom(2, 6, Activation.tanh, Activation.tanh_derivative, model_random);
    try network.addLayerWithRandom(6, 4, Activation.tanh, Activation.tanh_derivative, model_random);
    try network.addLayerWithRandom(4, 1, Activation.sigmoid, Activation.sigmoid_derivative, model_random);

    if (options.format == .human) {
        try printArchitecture(writer, &network);
    }

    var inputs = try makeInputs(allocator);
    defer inputs.deinit();
    var targets = try makeTargets(allocator);
    defer targets.deinit();

    if (options.format == .ndjson) {
        try events.emit(writer, .{
            .v = 1,
            .type = "run_started",
            .experiment = "xor-training",
            .data = .{
                .config = .{
                    .epochs = options.epochs,
                    .learning_rate = options.learning_rate,
                    .seed = options.seed,
                },
                .topology = &[_]usize{ 2, 6, 4, 1 },
                .activations = &[_][]const u8{ "tanh", "tanh", "sigmoid" },
            },
        });
        try emitSnapshot(writer, &network, inputs, targets, 0, options.epochs);
    } else {
        try printPredictions(writer, "Initial predictions (before training):", &network, inputs, targets, false);
        try writer.print("\nTraining Progress:\n", .{});
        try writer.print("Each '=' represents 2% progress. Loss shown every 5%.\n", .{});
        try writer.print("[", .{});
        try writer.flush();
    }

    var shuffle_prng = std.Random.DefaultPrng.init(options.seed ^ 0xd1b54a32d192ed03);
    const shuffle_random = shuffle_prng.random();
    var last_progress: usize = 0;
    var final_loss: f64 = 0;

    for (0..options.epochs) |epoch| {
        var indices = [_]usize{ 0, 1, 2, 3 };
        shuffle_random.shuffle(usize, &indices);

        var batch_inputs = try Matrix.init(allocator, batch_size, inputs.cols);
        defer batch_inputs.deinit();
        var batch_targets = try Matrix.init(allocator, batch_size, targets.cols);
        defer batch_targets.deinit();

        for (indices, 0..) |sample_index, row| {
            for (0..inputs.cols) |column| {
                try batch_inputs.set(row, column, try inputs.get(sample_index, column));
            }
            try batch_targets.set(row, 0, try targets.get(sample_index, 0));
        }

        final_loss = try network.trainBatch(batch_inputs, batch_targets);

        if (options.format == .ndjson) {
            if (events.shouldEmitMetric(epoch, options.epochs)) {
                try events.emit(writer, .{
                    .v = 1,
                    .type = "metric",
                    .experiment = "xor-training",
                    .step = epoch + 1,
                    .total_steps = options.epochs,
                    .data = .{ .name = "loss", .value = final_loss },
                });
            }
            if (events.shouldEmitSnapshot(epoch, options.epochs)) {
                try emitSnapshot(writer, &network, inputs, targets, epoch + 1, options.epochs);
            }
        } else {
            const progress = ((epoch + 1) * 50) / options.epochs;
            while (last_progress < progress) {
                try writer.print("=", .{});
                last_progress += 1;
            }
            const loss_interval = @max(@as(usize, 1), options.epochs / 20);
            if (epoch % loss_interval == 0 or epoch + 1 == options.epochs) {
                try writer.print(" {d:.4}", .{final_loss});
            }
            try writer.flush();
        }
    }

    const final_predictions = try collectPredictions(&network, inputs, targets);
    const correct = predictionAccuracy(final_predictions);

    if (options.format == .ndjson) {
        try events.emit(writer, .{
            .v = 1,
            .type = "run_completed",
            .experiment = "xor-training",
            .step = options.epochs,
            .total_steps = options.epochs,
            .data = .{
                .final_loss = final_loss,
                .correct = correct,
                .total = final_predictions.len,
                .predictions = final_predictions,
            },
        });
    } else {
        try writer.print("]\n\n", .{});
        try printPredictionRows(writer, "Final predictions (after training):", final_predictions, true);
        try writer.print("\nAccuracy: {d}/{d} correct predictions\n", .{ correct, final_predictions.len });
    }

    if (options.output_file) |file| {
        try network.saveToFile(file);
        if (options.format == .human) {
            try writer.print("\nModel saved to: {s}\n", .{file});
        }
    }
}

fn parseArgs(args: []const []const u8) !Options {
    var options: Options = .{};
    var index: usize = 0;
    while (index < args.len) : (index += 1) {
        const arg = args[index];
        if (std.mem.eql(u8, arg, "--output")) {
            index += 1;
            if (index >= args.len) return error.MissingArgument;
            options.output_file = args[index];
        } else if (std.mem.startsWith(u8, arg, "--output=")) {
            options.output_file = arg["--output=".len..];
        } else if (std.mem.eql(u8, arg, "--format")) {
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

fn makeInputs(allocator: std.mem.Allocator) !Matrix {
    var inputs = try Matrix.init(allocator, 4, 2);
    errdefer inputs.deinit();
    const values = [_][2]f64{ .{ 0, 0 }, .{ 0, 1 }, .{ 1, 0 }, .{ 1, 1 } };
    for (values, 0..) |value, row| {
        try inputs.set(row, 0, value[0]);
        try inputs.set(row, 1, value[1]);
    }
    return inputs;
}

fn makeTargets(allocator: std.mem.Allocator) !Matrix {
    var targets = try Matrix.init(allocator, 4, 1);
    errdefer targets.deinit();
    const values = [_]f64{ 0, 1, 1, 0 };
    for (values, 0..) |value, row| try targets.set(row, 0, value);
    return targets;
}

fn collectPredictions(network: *Network, inputs: Matrix, targets: Matrix) ![4]Prediction {
    var output = try network.predict(inputs);
    defer output.deinit();
    var result: [4]Prediction = undefined;
    for (&result, 0..) |*prediction, row| {
        prediction.* = .{
            .x1 = try inputs.get(row, 0),
            .x2 = try inputs.get(row, 1),
            .predicted = try output.get(row, 0),
            .expected = try targets.get(row, 0),
        };
    }
    return result;
}

fn predictionAccuracy(predictions: [4]Prediction) usize {
    var correct: usize = 0;
    for (predictions) |prediction| {
        if (@abs(prediction.predicted - prediction.expected) < 0.5) correct += 1;
    }
    return correct;
}

fn emitSnapshot(writer: *std.Io.Writer, network: *Network, inputs: Matrix, targets: Matrix, step: usize, total: usize) !void {
    const predictions = try collectPredictions(network, inputs, targets);
    try events.emit(writer, .{
        .v = 1,
        .type = "snapshot",
        .experiment = "xor-training",
        .step = step,
        .total_steps = total,
        .data = .{ .kind = "xor_predictions", .predictions = predictions },
    });
}

fn printArchitecture(writer: *std.Io.Writer, network: *const Network) !void {
    try writer.print(
        "Network Architecture:\n" ++
            "- Input layer:     2 neurons\n" ++
            "- Hidden layer 1:  6 neurons (tanh)\n" ++
            "- Hidden layer 2:  4 neurons (tanh)\n" ++
            "- Output layer:    1 neuron  (sigmoid)\n" ++
            "\nTraining Parameters:\n" ++
            "- Learning rate:   {d:.3}\n" ++
            "- Loss function:   {s}\n",
        .{ network.learning_rate, @tagName(network.loss_function) },
    );
}

fn printPredictions(writer: *std.Io.Writer, label: []const u8, network: *Network, inputs: Matrix, targets: Matrix, include_correct: bool) !void {
    const predictions = try collectPredictions(network, inputs, targets);
    try printPredictionRows(writer, label, predictions, include_correct);
}

fn printPredictionRows(writer: *std.Io.Writer, label: []const u8, predictions: [4]Prediction, include_correct: bool) !void {
    try writer.print("\n{s}\n", .{label});
    if (include_correct) {
        try writer.print("┌─────────┬──────────┬──────────┬──────────┐\n", .{});
        try writer.print("│ Input   │ Output   │ Expected │ Correct? │\n", .{});
        try writer.print("├─────────┼──────────┼──────────┼──────────┤\n", .{});
        for (predictions) |prediction| {
            const is_correct = @abs(prediction.predicted - prediction.expected) < 0.5;
            try writer.print("│ [{d}, {d}]  │ {d:.4}   │ {d}        │ {s}        │\n", .{
                prediction.x1,
                prediction.x2,
                prediction.predicted,
                prediction.expected,
                if (is_correct) "✓" else "✗",
            });
        }
        try writer.print("└─────────┴──────────┴──────────┴──────────┘\n", .{});
    } else {
        try writer.print("┌─────────┬──────────┬──────────┐\n", .{});
        try writer.print("│ Input   │ Output   │ Expected │\n", .{});
        try writer.print("├─────────┼──────────┼──────────┤\n", .{});
        for (predictions) |prediction| {
            try writer.print("│ [{d}, {d}]  │ {d:.4}   │ {d}        │\n", .{
                prediction.x1,
                prediction.x2,
                prediction.predicted,
                prediction.expected,
            });
        }
        try writer.print("└─────────┴──────────┴──────────┘\n", .{});
    }
}

test "xor learning options parse and validate" {
    const options = try parseArgs(&.{ "--format", "ndjson", "--epochs", "250", "--learning-rate", "0.2", "--seed", "7" });
    try std.testing.expectEqual(events.Format.ndjson, options.format);
    try std.testing.expectEqual(@as(usize, 250), options.epochs);
    try std.testing.expectEqual(@as(u64, 7), options.seed);
    try std.testing.expectApproxEqAbs(@as(f64, 0.2), options.learning_rate, 1e-12);
    try std.testing.expectError(error.InvalidEpochCount, parseArgs(&.{ "--epochs", "0" }));
    try std.testing.expectError(error.InvalidLearningRate, parseArgs(&.{ "--learning-rate", "0" }));
}

test "xor shuffle stream is stable for a supplied seed" {
    var first = [_]usize{ 0, 1, 2, 3 };
    var second = [_]usize{ 0, 1, 2, 3 };
    var first_prng = std.Random.DefaultPrng.init(42 ^ 0xd1b54a32d192ed03);
    var second_prng = std.Random.DefaultPrng.init(42 ^ 0xd1b54a32d192ed03);
    first_prng.random().shuffle(usize, &first);
    second_prng.random().shuffle(usize, &second);
    try std.testing.expectEqualSlices(usize, &first, &second);
}
