const std = @import("std");
const builtin = @import("builtin");
const nn = @import("nn");
const events = @import("experiment_events");

const sample_count: usize = 128;
const spectrum_bins: usize = sample_count / 2 + 1;
const hidden_width: usize = 32;
const default_steps: usize = 1_000;
const default_learning_rate: f64 = 0.01;
const default_fourier_bands: usize = 9;
const default_seed: u64 = 42;
const highest_target_frequency: usize = 9;

const Options = struct {
    format: events.Format = .human,
    steps: usize = default_steps,
    learning_rate: f64 = default_learning_rate,
    fourier_bands: usize = default_fourier_bands,
    seed: u64 = default_seed,
};

const Point = struct {
    x: f64,
    y: f64,
};

const Evaluation = struct {
    loss: f64,
    curve: [sample_count]Point,
    amplitudes: [spectrum_bins]f64,

    fn harmonicError(self: Evaluation, target: [spectrum_bins]f64, frequency: usize) f64 {
        return @abs(self.amplitudes[frequency] - target[frequency]);
    }
};

const ModelResult = struct {
    name: []const u8,
    parameter_count: usize,
    initial_loss: f64,
    final_loss: f64,
    frequency_1_error: f64,
    frequency_9_error: f64,
};

const ExperimentResult = struct {
    models: [2]ModelResult,
    early_step: usize,
    early_raw_frequency_1_error: f64,
    early_raw_frequency_9_error: f64,
};

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const options = parseArgs(args[1..]) catch |err| {
        if (err == error.HelpRequested) {
            printUsage();
            return;
        }
        printUsage();
        return err;
    };

    var stdout_buffer: [64 * 1024]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writerStreaming(std.Options.debug_io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};

    const result = try runExperimentStreaming(
        init.gpa,
        options,
        if (options.format == .ndjson) stdout else null,
    );
    if (options.format == .human) try printHuman(stdout, options, result);
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
        } else if (std.mem.eql(u8, arg, "--steps")) {
            index += 1;
            if (index >= args.len) return error.MissingArgument;
            options.steps = try std.fmt.parseInt(usize, args[index], 10);
            if (options.steps == 0) return error.InvalidStepCount;
        } else if (std.mem.eql(u8, arg, "--learning-rate")) {
            index += 1;
            if (index >= args.len) return error.MissingArgument;
            options.learning_rate = try std.fmt.parseFloat(f64, args[index]);
            if (!std.math.isFinite(options.learning_rate) or options.learning_rate <= 0) {
                return error.InvalidLearningRate;
            }
        } else if (std.mem.eql(u8, arg, "--fourier-bands")) {
            index += 1;
            if (index >= args.len) return error.MissingArgument;
            options.fourier_bands = try std.fmt.parseInt(usize, args[index], 10);
            if (options.fourier_bands == 0 or options.fourier_bands > 16) {
                return error.InvalidFourierBands;
            }
        } else if (std.mem.eql(u8, arg, "--seed")) {
            index += 1;
            if (index >= args.len) return error.MissingArgument;
            options.seed = try std.fmt.parseInt(u64, args[index], 10);
        } else if (std.mem.eql(u8, arg, "--help")) {
            return error.HelpRequested;
        } else {
            return error.UnknownArgument;
        }
    }
    return options;
}

fn printUsage() void {
    std.debug.print(
        \\Usage: spectral_learning [options]
        \\
        \\Options:
        \\  --format <name>          human or ndjson (default: human)
        \\  --steps <count>          full-batch updates per model (default: 1000)
        \\  --learning-rate <value>  SGD step size (default: 0.01)
        \\  --fourier-bands <count>  sine/cosine input bands, 1 through 16 (default: 9)
        \\  --seed <number>          model initialization seed (default: 42)
        \\  --help                   show this help
        \\
    , .{});
}

fn runExperiment(allocator: std.mem.Allocator, options: Options) !ExperimentResult {
    return runExperimentStreaming(allocator, options, null);
}

fn runExperimentStreaming(
    allocator: std.mem.Allocator,
    options: Options,
    writer: ?*std.Io.Writer,
) !ExperimentResult {
    var coordinates = try makeCoordinates(allocator);
    defer coordinates.deinit();
    var targets = try makeTargets(allocator, coordinates);
    defer targets.deinit();
    var encoded = try makeFourierInput(allocator, coordinates, options.fourier_bands);
    defer encoded.deinit();

    var raw = nn.Network.init(allocator, options.learning_rate, .MeanSquaredError);
    defer raw.deinit();
    var raw_prng = std.Random.DefaultPrng.init(options.seed ^ 0x9e3779b97f4a7c15);
    try addRegressionLayers(&raw, 1, raw_prng.random());

    var fourier = nn.Network.init(allocator, options.learning_rate, .MeanSquaredError);
    defer fourier.deinit();
    var fourier_prng = std.Random.DefaultPrng.init(options.seed ^ 0xd1b54a32d192ed03);
    try addRegressionLayers(&fourier, encoded.cols, fourier_prng.random());

    const raw_parameters = parameterCount(raw);
    const fourier_parameters = parameterCount(fourier);
    const target_evaluation = try evaluateTarget(allocator, coordinates, targets);
    const initial_raw = try evaluateNetwork(&raw, coordinates, coordinates, targets);
    const initial_fourier = try evaluateNetwork(&fourier, encoded, coordinates, targets);
    const spectrum_limit = @max(highest_target_frequency, options.fourier_bands);

    if (writer) |output| {
        const raw_topology = [_]usize{ 1, hidden_width, hidden_width, 1 };
        const fourier_topology = [_]usize{ encoded.cols, hidden_width, hidden_width, 1 };
        const model_metadata = [_]struct {
            name: []const u8,
            input_representation: []const u8,
            topology: []const usize,
            parameter_count: usize,
        }{
            .{
                .name = "raw",
                .input_representation = "coordinate",
                .topology = &raw_topology,
                .parameter_count = raw_parameters,
            },
            .{
                .name = "fourier",
                .input_representation = "coordinate plus harmonic pairs",
                .topology = &fourier_topology,
                .parameter_count = fourier_parameters,
            },
        };
        try events.emit(output, .{
            .v = 1,
            .type = "run_started",
            .experiment = "spectral-learning",
            .data = .{
                .config = .{
                    .steps = options.steps,
                    .learning_rate = options.learning_rate,
                    .fourier_bands = options.fourier_bands,
                    .seed = options.seed,
                },
                .topology = &raw_topology,
                .activations = &[_][]const u8{ "relu", "relu", "linear" },
                .models = &model_metadata,
                .target_curve = &target_evaluation.curve,
                .target_spectrum = target_evaluation.amplitudes[0 .. spectrum_limit + 1],
                .spectrum_limit = spectrum_limit,
                .execution = .{
                    .requested_backend = "cpu",
                    .selected_backend = "cpu",
                    .optimize = @tagName(builtin.mode),
                },
            },
        });
        try emitLossMetrics(output, 0, options.steps, initial_raw.loss, initial_fourier.loss);
        try emitSnapshot(
            output,
            0,
            options.steps,
            spectrum_limit,
            target_evaluation.amplitudes,
            initial_raw,
            initial_fourier,
        );
    }

    const early_step = @min(@as(usize, 200), options.steps);
    var early_raw_frequency_1_error: f64 = initial_raw.harmonicError(target_evaluation.amplitudes, 1);
    var early_raw_frequency_9_error: f64 = initial_raw.harmonicError(target_evaluation.amplitudes, 9);

    for (0..options.steps) |step_index| {
        const raw_loss = try raw.trainBatch(coordinates, targets);
        const fourier_loss = try fourier.trainBatch(encoded, targets);
        const step = step_index + 1;

        if (step == early_step) {
            const early = try evaluateNetwork(&raw, coordinates, coordinates, targets);
            early_raw_frequency_1_error = early.harmonicError(target_evaluation.amplitudes, 1);
            early_raw_frequency_9_error = early.harmonicError(target_evaluation.amplitudes, 9);
        }

        if (writer) |output| {
            if (events.shouldEmitMetric(step_index, options.steps)) {
                try emitLossMetrics(output, step, options.steps, raw_loss, fourier_loss);
            }
            if (events.shouldEmitSnapshot(step_index, options.steps)) {
                const raw_evaluation = try evaluateNetwork(&raw, coordinates, coordinates, targets);
                const fourier_evaluation = try evaluateNetwork(&fourier, encoded, coordinates, targets);
                try emitSnapshot(
                    output,
                    step,
                    options.steps,
                    spectrum_limit,
                    target_evaluation.amplitudes,
                    raw_evaluation,
                    fourier_evaluation,
                );
            }
        }
    }

    const final_raw = try evaluateNetwork(&raw, coordinates, coordinates, targets);
    const final_fourier = try evaluateNetwork(&fourier, encoded, coordinates, targets);
    const result = ExperimentResult{
        .models = .{
            .{
                .name = "raw",
                .parameter_count = raw_parameters,
                .initial_loss = initial_raw.loss,
                .final_loss = final_raw.loss,
                .frequency_1_error = final_raw.harmonicError(target_evaluation.amplitudes, 1),
                .frequency_9_error = final_raw.harmonicError(target_evaluation.amplitudes, 9),
            },
            .{
                .name = "fourier",
                .parameter_count = fourier_parameters,
                .initial_loss = initial_fourier.loss,
                .final_loss = final_fourier.loss,
                .frequency_1_error = final_fourier.harmonicError(target_evaluation.amplitudes, 1),
                .frequency_9_error = final_fourier.harmonicError(target_evaluation.amplitudes, 9),
            },
        },
        .early_step = early_step,
        .early_raw_frequency_1_error = early_raw_frequency_1_error,
        .early_raw_frequency_9_error = early_raw_frequency_9_error,
    };

    if (writer) |output| {
        try events.emit(output, .{
            .v = 1,
            .type = "run_completed",
            .experiment = "spectral-learning",
            .step = options.steps,
            .total_steps = options.steps,
            .data = result,
        });
    }
    return result;
}

fn makeCoordinates(allocator: std.mem.Allocator) !nn.Matrix {
    var coordinates = try nn.Matrix.init(allocator, sample_count, 1);
    errdefer coordinates.deinit();
    for (coordinates.data, 0..) |*coordinate, index| {
        coordinate.* = @as(f64, @floatFromInt(index)) / @as(f64, @floatFromInt(sample_count));
    }
    return coordinates;
}

fn makeTargets(allocator: std.mem.Allocator, coordinates: nn.Matrix) !nn.Matrix {
    var targets = try nn.Matrix.init(allocator, coordinates.rows, 1);
    errdefer targets.deinit();
    for (coordinates.data, targets.data) |coordinate, *target| target.* = targetFunction(coordinate);
    return targets;
}

fn makeFourierInput(allocator: std.mem.Allocator, coordinates: nn.Matrix, bands: usize) !nn.Matrix {
    const features = try nn.Spectral.fourierFeatureCount(bands);
    var encoded = try nn.Matrix.init(allocator, coordinates.rows, features);
    errdefer encoded.deinit();
    try nn.Spectral.encodeFourierFeatures(f64, coordinates.data, bands, encoded.data);
    return encoded;
}

fn targetFunction(x: f64) f64 {
    return (@sin(2 * std.math.pi * x) +
        @sin(2 * std.math.pi * 3 * x) +
        @sin(2 * std.math.pi * 9 * x)) / 3;
}

fn addRegressionLayers(network: *nn.Network, input_width: usize, random: std.Random) !void {
    try network.addLayerWithRandom(input_width, hidden_width, nn.Activation.relu, nn.Activation.relu_derivative, random);
    try network.addLayerWithRandom(hidden_width, hidden_width, nn.Activation.relu, nn.Activation.relu_derivative, random);
    try network.addLayerWithRandom(hidden_width, 1, nn.Activation.linear, nn.Activation.linear_derivative, random);
}

fn parameterCount(network: nn.Network) usize {
    var total: usize = 0;
    for (network.layers.items) |layer| switch (layer) {
        .Standard => |standard| total += standard.weights.data.len + standard.bias.data.len,
        .Gated => |gated| total += gated.linear_weights.data.len + gated.linear_bias.data.len +
            gated.gate_weights.data.len + gated.gate_bias.data.len,
    };
    return total;
}

fn evaluateTarget(allocator: std.mem.Allocator, coordinates: nn.Matrix, targets: nn.Matrix) !Evaluation {
    var curve: [sample_count]Point = undefined;
    for (&curve, 0..) |*point, index| point.* = .{
        .x = coordinates.data[index],
        .y = targets.data[index],
    };
    var amplitudes: [spectrum_bins]f64 = undefined;
    try nn.Spectral.realAmplitudeSpectrum(allocator, f64, targets.data, &amplitudes);
    return .{ .loss = 0, .curve = curve, .amplitudes = amplitudes };
}

fn evaluateNetwork(
    network: *nn.Network,
    inputs: nn.Matrix,
    coordinates: nn.Matrix,
    targets: nn.Matrix,
) !Evaluation {
    var predictions = try network.predict(inputs);
    defer predictions.deinit();
    const loss = try network.calculateLoss(predictions, targets);
    var curve: [sample_count]Point = undefined;
    for (&curve, 0..) |*point, index| point.* = .{
        .x = coordinates.data[index],
        .y = predictions.data[index],
    };
    var amplitudes: [spectrum_bins]f64 = undefined;
    try nn.Spectral.realAmplitudeSpectrum(network.allocator, f64, predictions.data, &amplitudes);
    return .{ .loss = loss, .curve = curve, .amplitudes = amplitudes };
}

fn emitLossMetrics(
    writer: *std.Io.Writer,
    step: usize,
    total: usize,
    raw_loss: f64,
    fourier_loss: f64,
) !void {
    try events.emit(writer, .{
        .v = 1,
        .type = "metric",
        .experiment = "spectral-learning",
        .step = step,
        .total_steps = total,
        .data = .{ .name = "loss", .series = "raw", .value = raw_loss },
    });
    try events.emit(writer, .{
        .v = 1,
        .type = "metric",
        .experiment = "spectral-learning",
        .step = step,
        .total_steps = total,
        .data = .{ .name = "loss", .series = "fourier", .value = fourier_loss },
    });
}

fn emitSnapshot(
    writer: *std.Io.Writer,
    step: usize,
    total: usize,
    spectrum_limit: usize,
    target_amplitudes: [spectrum_bins]f64,
    raw: Evaluation,
    fourier: Evaluation,
) !void {
    inline for (.{
        .{ .name = "raw f1", .value = raw.harmonicError(target_amplitudes, 1) },
        .{ .name = "raw f9", .value = raw.harmonicError(target_amplitudes, 9) },
        .{ .name = "fourier f1", .value = fourier.harmonicError(target_amplitudes, 1) },
        .{ .name = "fourier f9", .value = fourier.harmonicError(target_amplitudes, 9) },
    }) |metric| try events.emit(writer, .{
        .v = 1,
        .type = "metric",
        .experiment = "spectral-learning",
        .step = step,
        .total_steps = total,
        .data = .{ .name = "harmonic_error", .series = metric.name, .value = metric.value },
    });

    const series = [_]struct {
        name: []const u8,
        curve: []const Point,
        amplitudes: []const f64,
    }{
        .{
            .name = "raw",
            .curve = &raw.curve,
            .amplitudes = raw.amplitudes[0 .. spectrum_limit + 1],
        },
        .{
            .name = "fourier",
            .curve = &fourier.curve,
            .amplitudes = fourier.amplitudes[0 .. spectrum_limit + 1],
        },
    };
    try events.emit(writer, .{
        .v = 1,
        .type = "snapshot",
        .experiment = "spectral-learning",
        .step = step,
        .total_steps = total,
        .data = .{ .kind = "spectral_learning", .series = &series },
    });
}

fn printHuman(writer: *std.Io.Writer, options: Options, result: ExperimentResult) !void {
    try writer.print(
        "Spectral learning with raw coordinates and Fourier features\n" ++
            "target frequencies: 1, 3, and 9 cycles; steps: {d}; learning rate: {d:.4}; Fourier bands: {d}\n\n",
        .{ options.steps, options.learning_rate, options.fourier_bands },
    );
    for (result.models) |model| {
        try writer.print(
            "{s}: parameters={d}, loss {d:.6} -> {d:.6}, |amplitude error| f1={d:.6}, f9={d:.6}\n",
            .{
                model.name,
                model.parameter_count,
                model.initial_loss,
                model.final_loss,
                model.frequency_1_error,
                model.frequency_9_error,
            },
        );
    }
    try writer.print(
        "\nAt step {d}, raw-coordinate error was f1={d:.6} and f9={d:.6}.\n" ++
            "The comparison changes the input representation and first-layer parameter count; it is not a parameter-matched benchmark.\n",
        .{ result.early_step, result.early_raw_frequency_1_error, result.early_raw_frequency_9_error },
    );
}

test "spectral learning options parse and validate" {
    const options = try parseArgs(&.{
        "--format",
        "ndjson",
        "--steps",
        "200",
        "--learning-rate",
        "0.02",
        "--fourier-bands",
        "12",
        "--seed",
        "7",
    });
    try std.testing.expectEqual(events.Format.ndjson, options.format);
    try std.testing.expectEqual(@as(usize, 200), options.steps);
    try std.testing.expectApproxEqAbs(@as(f64, 0.02), options.learning_rate, 1e-12);
    try std.testing.expectEqual(@as(usize, 12), options.fourier_bands);
    try std.testing.expectEqual(@as(u64, 7), options.seed);
    try std.testing.expectError(error.InvalidStepCount, parseArgs(&.{ "--steps", "0" }));
    try std.testing.expectError(error.InvalidFourierBands, parseArgs(&.{ "--fourier-bands", "17" }));
}

test "target has equal amplitude at the three learning frequencies" {
    var coordinates = try makeCoordinates(std.testing.allocator);
    defer coordinates.deinit();
    var targets = try makeTargets(std.testing.allocator, coordinates);
    defer targets.deinit();
    const evaluation = try evaluateTarget(std.testing.allocator, coordinates, targets);
    inline for (.{ 1, 3, 9 }) |frequency| {
        try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), evaluation.amplitudes[frequency], 1e-10);
    }
}

test "Fourier features improve the high-frequency fit while raw learning is spectrally biased" {
    const result = try runExperiment(std.testing.allocator, .{});
    for (result.models) |model| {
        try std.testing.expect(std.math.isFinite(model.final_loss));
        try std.testing.expect(model.final_loss < model.initial_loss);
    }
    try std.testing.expect(result.early_raw_frequency_1_error < result.early_raw_frequency_9_error);
    try std.testing.expect(result.models[1].frequency_9_error < result.models[0].frequency_9_error);
}
