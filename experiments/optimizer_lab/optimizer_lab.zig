const std = @import("std");
const builtin = @import("builtin");
const nn = @import("nn");
const events = @import("experiment_events");

const default_seed: u64 = 42;
const default_steps: usize = 200;
const train_samples: usize = 256;
const test_samples: usize = 128;
const grid_size: usize = 24;
const grid_count: usize = grid_size * grid_size;
const layer_widths = [_]usize{ 2, 16, 1 };

const Options = struct {
    backend: nn.DevicePreference = .auto,
    format: events.Format = .human,
    steps: usize = default_steps,
    seed: u64 = default_seed,
};

const Dataset = struct {
    allocator: std.mem.Allocator,
    features: []f32,
    labels: []f32,

    fn deinit(self: *Dataset) void {
        self.allocator.free(self.features);
        self.allocator.free(self.labels);
        self.* = undefined;
    }
};

const Evaluation = struct {
    loss: f32,
    accuracy: f32,
};

const Sample = struct {
    x: f32,
    y: f32,
    label: f32,
};

const Probability = struct {
    x: f32,
    y: f32,
    value: f32,
};

const OptimizerSpec = struct {
    name: []const u8,
    config: nn.Training.OptimizerConfig,
};

const TrainingResult = struct {
    optimizer: []const u8,
    learning_rate: f32,
    initial_loss: f32,
    final_loss: f32,
    test_accuracy: f32,
    update_steps: usize,
    training_kernels: usize,
    training_readbacks: usize,
};

const ExperimentResult = struct {
    backend: nn.BackendType,
    results: [3]TrainingResult,
};

const Boundary = struct {
    name: []const u8,
    predictions: [grid_count]Probability,
};

const Session = struct {
    name: []const u8,
    learning_rate: f32,
    model: nn.Modules.Mlp,
    optimizer: nn.Training.Optimizer,
    parameter_buffer: []*nn.Tensor,
    gradient_buffer: []nn.Tensor,
    allocator: std.mem.Allocator,
    initial_loss: f32 = 0,
    training_kernels: usize = 0,
    training_readbacks: usize = 0,

    fn init(
        self: *Session,
        allocator: std.mem.Allocator,
        context: *nn.ExecutionContext,
        seed: u64,
        spec: OptimizerSpec,
    ) !void {
        var prng = std.Random.DefaultPrng.init(seed);
        const model = try nn.Modules.Mlp.init(context, .{
            .widths = &layer_widths,
            .hidden_activation = .tanh,
            .output_activation = .sigmoid,
        }, prng.random());
        self.* = .{
            .name = spec.name,
            .learning_rate = spec.config.learning_rate,
            .model = model,
            .optimizer = undefined,
            .parameter_buffer = undefined,
            .gradient_buffer = undefined,
            .allocator = allocator,
        };
        errdefer self.model.deinit();

        self.parameter_buffer = try allocator.alloc(*nn.Tensor, self.model.parameterTensorCount());
        errdefer allocator.free(self.parameter_buffer);
        self.gradient_buffer = try allocator.alloc(nn.Tensor, self.model.parameterTensorCount());
        errdefer allocator.free(self.gradient_buffer);
        const parameters = try self.model.parameters(self.parameter_buffer);
        self.optimizer = try nn.Training.Optimizer.init(context, spec.config, parameters);
    }

    fn deinit(self: *Session) void {
        self.optimizer.deinit();
        self.allocator.free(self.gradient_buffer);
        self.allocator.free(self.parameter_buffer);
        self.model.deinit();
        self.* = undefined;
    }

    fn trainStep(
        self: *Session,
        context: *nn.ExecutionContext,
        features: nn.Tensor,
        labels: nn.Tensor,
        label_count: usize,
    ) !void {
        const before = context.stats;
        try context.beginBatch();
        var batch_active = true;
        errdefer if (batch_active) context.endBatch() catch {};

        var forward = try self.model.forward(context, features);
        defer forward.deinit();
        var difference = try context.subtract(forward.output, labels);
        defer difference.deinit();
        var output_gradient = try context.scale(
            difference,
            2.0 / @as(f32, @floatFromInt(label_count)),
        );
        defer output_gradient.deinit();
        var gradients = try self.model.backward(context, &forward.cache, output_gradient);
        defer gradients.deinit();
        const parameters = try self.model.parameters(self.parameter_buffer);
        const parameter_gradients = try gradients.parameterGradients(self.gradient_buffer);
        const update = try self.optimizer.step(context, parameters, parameter_gradients);
        std.debug.assert(update.updated);

        try context.endBatch();
        batch_active = false;
        self.training_kernels += context.stats.kernels - before.kernels;
        self.training_readbacks += context.stats.readbacks - before.readbacks;
    }
};

const optimizer_specs = [_]OptimizerSpec{
    .{ .name = "sgd", .config = .{
        .kind = .sgd,
        .learning_rate = 0.3,
        .max_gradient_norm = 5,
    } },
    .{ .name = "momentum", .config = .{
        .kind = .momentum,
        .learning_rate = 0.08,
        .momentum = 0.9,
        .max_gradient_norm = 5,
    } },
    .{ .name = "adamw", .config = .{
        .kind = .adamw,
        .learning_rate = 0.03,
        .weight_decay = 0.001,
        .max_gradient_norm = 5,
    } },
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
        if (std.mem.eql(u8, arg, "--backend")) {
            index += 1;
            if (index >= args.len) return error.MissingArgument;
            options.backend = try parseBackend(args[index]);
        } else if (std.mem.eql(u8, arg, "--format")) {
            index += 1;
            if (index >= args.len) return error.MissingArgument;
            options.format = try events.parseFormat(args[index]);
        } else if (std.mem.eql(u8, arg, "--steps")) {
            index += 1;
            if (index >= args.len) return error.MissingArgument;
            options.steps = try std.fmt.parseInt(usize, args[index], 10);
            if (options.steps == 0) return error.InvalidStepCount;
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

fn parseBackend(value: []const u8) !nn.DevicePreference {
    if (std.mem.eql(u8, value, "cpu")) return .cpu;
    if (std.mem.eql(u8, value, "auto")) return .auto;
    if (std.mem.eql(u8, value, "metal")) return .metal;
    if (std.mem.eql(u8, value, "cuda")) return .cuda;
    if (std.mem.eql(u8, value, "rocm")) return .rocm;
    return error.UnknownBackend;
}

fn printUsage() void {
    std.debug.print(
        \\Usage: optimizer_lab [options]
        \\
        \\Options:
        \\  --backend <name>  cpu, auto, metal, cuda, or rocm (default: auto)
        \\  --format <name>   human or ndjson (default: human)
        \\  --steps <count>   full-batch updates per optimizer (default: 200)
        \\  --seed <number>   model and dataset seed (default: 42)
        \\  --help            show this help
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
    var training_data = try generateTwoMoons(allocator, train_samples, options.seed);
    defer training_data.deinit();
    var testing_data = try generateTwoMoons(
        allocator,
        test_samples,
        options.seed ^ 0x9e3779b97f4a7c15,
    );
    defer testing_data.deinit();

    var device = try nn.Device.init(allocator, options.backend);
    defer device.deinit();
    var context = nn.ExecutionContext.init(&device);
    var training_features = try context.upload(&.{ train_samples, 2 }, training_data.features);
    defer training_features.deinit();
    var training_labels = try context.upload(&.{ train_samples, 1 }, training_data.labels);
    defer training_labels.deinit();
    var testing_features = try context.upload(&.{ test_samples, 2 }, testing_data.features);
    defer testing_features.deinit();

    const grid_values = makeGridValues();
    var grid_tensor = try context.upload(&.{ grid_count, 2 }, &grid_values);
    defer grid_tensor.deinit();
    const samples = collectSamples(training_data);

    var sessions: [optimizer_specs.len]Session = undefined;
    var initialized: usize = 0;
    defer for (sessions[0..initialized]) |*session| session.deinit();
    for (optimizer_specs, 0..) |spec, index| {
        try sessions[index].init(allocator, &context, options.seed, spec);
        initialized += 1;
        sessions[index].initial_loss = (try evaluate(
            allocator,
            &context,
            &sessions[index].model,
            training_features,
            training_data.labels,
        )).loss;
    }

    if (writer) |output| {
        try events.emit(output, .{
            .v = 1,
            .type = "run_started",
            .experiment = "optimizer-lab",
            .data = .{
                .config = .{ .steps = options.steps, .seed = options.seed },
                .topology = &layer_widths,
                .activations = &[_][]const u8{ "tanh", "sigmoid" },
                .samples = samples,
                .grid_size = grid_size,
                .execution = .{
                    .requested_backend = backendPreferenceName(options.backend),
                    .selected_backend = backendName(device.backendType()),
                    .optimize = @tagName(builtin.mode),
                },
            },
        });
        try emitReport(
            allocator,
            output,
            &context,
            &sessions,
            training_features,
            training_data.labels,
            testing_features,
            testing_data.labels,
            grid_tensor,
            grid_values,
            0,
            options.steps,
            .{},
            .{},
        );
    }

    context.resetStats();
    for (0..options.steps) |step_index| {
        for (&sessions) |*session| {
            try session.trainStep(
                &context,
                training_features,
                training_labels,
                training_data.labels.len,
            );
        }
        if (writer) |output| {
            if (events.shouldEmitSnapshot(step_index, options.steps)) {
                const execution_stats = context.stats;
                const backend_stats = context.backendStats();
                try emitReport(
                    allocator,
                    output,
                    &context,
                    &sessions,
                    training_features,
                    training_data.labels,
                    testing_features,
                    testing_data.labels,
                    grid_tensor,
                    grid_values,
                    step_index + 1,
                    options.steps,
                    execution_stats,
                    backend_stats,
                );
                context.resetStats();
            }
        }
    }

    var results: [sessions.len]TrainingResult = undefined;
    for (&sessions, 0..) |*session, index| {
        const final = try evaluate(
            allocator,
            &context,
            &session.model,
            training_features,
            training_data.labels,
        );
        const testing = try evaluate(
            allocator,
            &context,
            &session.model,
            testing_features,
            testing_data.labels,
        );
        results[index] = .{
            .optimizer = session.name,
            .learning_rate = session.learning_rate,
            .initial_loss = session.initial_loss,
            .final_loss = final.loss,
            .test_accuracy = testing.accuracy,
            .update_steps = session.optimizer.update_steps,
            .training_kernels = session.training_kernels,
            .training_readbacks = session.training_readbacks,
        };
    }

    if (writer) |output| {
        try events.emit(output, .{
            .v = 1,
            .type = "run_completed",
            .experiment = "optimizer-lab",
            .step = options.steps,
            .total_steps = options.steps,
            .data = .{ .results = results },
        });
    }
    return .{ .backend = device.backendType(), .results = results };
}

fn emitReport(
    allocator: std.mem.Allocator,
    writer: *std.Io.Writer,
    context: *nn.ExecutionContext,
    sessions: *[optimizer_specs.len]Session,
    training_features: nn.Tensor,
    training_labels: []const f32,
    testing_features: nn.Tensor,
    testing_labels: []const f32,
    grid_tensor: nn.Tensor,
    grid_values: [grid_count * 2]f32,
    step: usize,
    total: usize,
    execution_stats: nn.ExecutionStats,
    backend_stats: nn.BackendRuntimeStats,
) !void {
    for (sessions) |*session| {
        const training = try evaluate(allocator, context, &session.model, training_features, training_labels);
        const testing = try evaluate(allocator, context, &session.model, testing_features, testing_labels);
        try events.emit(writer, .{
            .v = 1,
            .type = "metric",
            .experiment = "optimizer-lab",
            .step = step,
            .total_steps = total,
            .data = .{ .name = "loss", .series = session.name, .value = training.loss },
        });
        try events.emit(writer, .{
            .v = 1,
            .type = "metric",
            .experiment = "optimizer-lab",
            .step = step,
            .total_steps = total,
            .data = .{ .name = "accuracy", .series = session.name, .value = testing.accuracy },
        });
    }

    const boundaries = try collectBoundaries(context, sessions, grid_tensor, grid_values);
    try events.emit(writer, .{
        .v = 1,
        .type = "snapshot",
        .experiment = "optimizer-lab",
        .step = step,
        .total_steps = total,
        .data = .{
            .kind = "optimizer_comparison",
            .optimizers = boundaries,
            .telemetry = .{ .execution = execution_stats, .backend = backend_stats },
        },
    });
}

fn collectBoundaries(
    context: *nn.ExecutionContext,
    sessions: *[optimizer_specs.len]Session,
    grid_tensor: nn.Tensor,
    grid_values: [grid_count * 2]f32,
) ![optimizer_specs.len]Boundary {
    var boundaries: [optimizer_specs.len]Boundary = undefined;
    for (sessions, 0..) |*session, index| {
        var forward = try session.model.forward(context, grid_tensor);
        defer forward.deinit();
        var values: [grid_count]f32 = undefined;
        try context.readback(forward.output, &values);
        var predictions: [grid_count]Probability = undefined;
        for (&predictions, 0..) |*prediction, point| {
            prediction.* = .{
                .x = grid_values[point * 2],
                .y = grid_values[point * 2 + 1],
                .value = values[point],
            };
        }
        boundaries[index] = .{ .name = session.name, .predictions = predictions };
    }
    return boundaries;
}

fn evaluate(
    allocator: std.mem.Allocator,
    context: *nn.ExecutionContext,
    model: *const nn.Modules.Mlp,
    features: nn.Tensor,
    labels: []const f32,
) !Evaluation {
    var forward = try model.forward(context, features);
    defer forward.deinit();
    const predictions = try allocator.alloc(f32, labels.len);
    defer allocator.free(predictions);
    try context.readback(forward.output, predictions);

    var squared_error: f32 = 0;
    var correct: usize = 0;
    for (predictions, labels) |prediction, label| {
        const difference = prediction - label;
        squared_error += difference * difference;
        if ((prediction >= 0.5) == (label >= 0.5)) correct += 1;
    }
    const count = @as(f32, @floatFromInt(labels.len));
    return .{
        .loss = squared_error / count,
        .accuracy = @as(f32, @floatFromInt(correct)) / count,
    };
}

fn makeGridValues() [grid_count * 2]f32 {
    var values: [grid_count * 2]f32 = undefined;
    for (0..grid_size) |row| {
        for (0..grid_size) |column| {
            const index = row * grid_size + column;
            values[index * 2] = -1.6 + 3.2 * @as(f32, @floatFromInt(column)) / @as(f32, @floatFromInt(grid_size - 1));
            values[index * 2 + 1] = -1.2 + 2.4 * @as(f32, @floatFromInt(row)) / @as(f32, @floatFromInt(grid_size - 1));
        }
    }
    return values;
}

fn collectSamples(dataset: Dataset) [train_samples]Sample {
    var samples: [train_samples]Sample = undefined;
    for (&samples, 0..) |*sample, index| {
        sample.* = .{
            .x = dataset.features[index * 2],
            .y = dataset.features[index * 2 + 1],
            .label = dataset.labels[index],
        };
    }
    return samples;
}

fn generateTwoMoons(allocator: std.mem.Allocator, count: usize, seed: u64) !Dataset {
    if (count == 0) return error.EmptyDataset;
    const features = try allocator.alloc(f32, count * 2);
    errdefer allocator.free(features);
    const labels = try allocator.alloc(f32, count);
    errdefer allocator.free(labels);

    var prng = std.Random.DefaultPrng.init(seed);
    var random = prng.random();
    for (0..count) |index| {
        const class = index % 2;
        const angle = random.float(f32) * std.math.pi;
        const noise_x = (random.float(f32) * 2 - 1) * 0.08;
        const noise_y = (random.float(f32) * 2 - 1) * 0.08;
        const moon_x = if (class == 0) @cos(angle) else 1 - @cos(angle);
        const moon_y = if (class == 0) @sin(angle) else 0.5 - @sin(angle);
        features[index * 2] = moon_x - 0.5 + noise_x;
        features[index * 2 + 1] = moon_y - 0.25 + noise_y;
        labels[index] = @floatFromInt(class);
    }
    return .{ .allocator = allocator, .features = features, .labels = labels };
}

fn printHuman(writer: *std.Io.Writer, options: Options, experiment: ExperimentResult) !void {
    try writer.print(
        "Optimizer lab: two-moons classification on {s}\n" ++
            "same initialization, data, and {d} full-batch updates\n\n",
        .{ backendName(experiment.backend), options.steps },
    );
    try writer.print(
        "optimizer   lr       loss before -> after   accuracy   updates   kernels   readbacks\n",
        .{},
    );
    for (experiment.results) |result| {
        try writer.print(
            "{s: <10} {d:.4}   {d:.5} -> {d:.5}    {d:>6.2}%   {d:>7}   {d:>7}   {d:>9}\n",
            .{
                result.optimizer,
                result.learning_rate,
                result.initial_loss,
                result.final_loss,
                result.test_accuracy * 100,
                result.update_steps,
                result.training_kernels,
                result.training_readbacks,
            },
        );
    }
    try writer.print(
        "\nThe learning rates are intentionally optimizer-specific; compare the\n" ++
            "loss reduction and held-out accuracy, not only the raw step size.\n",
        .{},
    );
}

fn backendPreferenceName(preference: nn.DevicePreference) []const u8 {
    return @tagName(preference);
}

fn backendName(backend: nn.BackendType) []const u8 {
    return switch (backend) {
        .CPU => "cpu",
        .Metal => "metal",
        .CUDA => "cuda",
        .ROCm => "rocm",
    };
}

test "optimizer lab parses reproducible options" {
    const options = try parseArgs(&.{ "--backend", "cpu", "--format", "ndjson", "--steps", "12", "--seed", "7" });
    try std.testing.expectEqual(nn.DevicePreference.cpu, options.backend);
    try std.testing.expectEqual(events.Format.ndjson, options.format);
    try std.testing.expectEqual(@as(usize, 12), options.steps);
    try std.testing.expectEqual(@as(u64, 7), options.seed);
    try std.testing.expectError(error.UnknownBackend, parseBackend("quantum"));
    try std.testing.expectError(error.InvalidStepCount, parseArgs(&.{ "--steps", "0" }));
}

test "optimizer lab data is deterministic and balanced" {
    var first = try generateTwoMoons(std.testing.allocator, 8, 17);
    defer first.deinit();
    var second = try generateTwoMoons(std.testing.allocator, 8, 17);
    defer second.deinit();

    try std.testing.expectEqualSlices(f32, first.features, second.features);
    try std.testing.expectEqualSlices(f32, first.labels, second.labels);
    try std.testing.expectEqualSlices(f32, &.{ 0, 1, 0, 1, 0, 1, 0, 1 }, first.labels);
}

test "optimizer lab improves all three models without training readbacks" {
    const experiment = try runExperiment(std.testing.allocator, .{
        .backend = .cpu,
        .steps = 120,
        .seed = default_seed,
    });
    for (experiment.results) |result| {
        try std.testing.expect(result.final_loss < result.initial_loss * 0.8);
        try std.testing.expect(result.test_accuracy >= 0.8);
        try std.testing.expectEqual(@as(usize, 120), result.update_steps);
        try std.testing.expectEqual(@as(usize, 0), result.training_readbacks);
    }
}
