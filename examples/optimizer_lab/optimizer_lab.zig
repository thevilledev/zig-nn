const std = @import("std");
const nn = @import("nn");

const default_seed: u64 = 42;
const default_steps: usize = 200;
const train_samples: usize = 256;
const test_samples: usize = 128;
const layer_widths = [_]usize{ 2, 16, 1 };

const Options = struct {
    backend: nn.DevicePreference = .auto,
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
    const experiment = try runExperiment(init.gpa, options);

    std.debug.print(
        "Optimizer lab: two-moons classification on {s}\n" ++
            "same initialization, data, and {d} full-batch updates\n\n",
        .{ backendName(experiment.backend), options.steps },
    );
    std.debug.print(
        "optimizer   lr       loss before -> after   accuracy   updates   kernels   readbacks\n",
        .{},
    );
    for (experiment.results) |result| {
        std.debug.print(
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
    std.debug.print(
        "\nThe learning rates are intentionally optimizer-specific; compare the\n" ++
            "loss reduction and held-out accuracy, not only the raw step size.\n",
        .{},
    );
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
        \\  --steps <count>   full-batch updates per optimizer (default: 200)
        \\  --seed <number>   model and dataset seed (default: 42)
        \\  --help            show this help
        \\
    , .{});
}

fn runExperiment(allocator: std.mem.Allocator, options: Options) !ExperimentResult {
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
    var training_features = try context.upload(
        &.{ train_samples, 2 },
        training_data.features,
    );
    defer training_features.deinit();
    var training_labels = try context.upload(
        &.{ train_samples, 1 },
        training_data.labels,
    );
    defer training_labels.deinit();
    var testing_features = try context.upload(
        &.{ test_samples, 2 },
        testing_data.features,
    );
    defer testing_features.deinit();

    const specs = [_]OptimizerSpec{
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
    var results: [specs.len]TrainingResult = undefined;
    for (specs, 0..) |spec, index| {
        results[index] = try trainOne(
            allocator,
            &context,
            training_features,
            training_labels,
            training_data.labels,
            testing_features,
            testing_data.labels,
            options.seed,
            options.steps,
            spec,
        );
    }

    return .{ .backend = device.backendType(), .results = results };
}

fn trainOne(
    allocator: std.mem.Allocator,
    context: *nn.ExecutionContext,
    training_features: nn.Tensor,
    training_labels: nn.Tensor,
    training_label_values: []const f32,
    testing_features: nn.Tensor,
    testing_label_values: []const f32,
    seed: u64,
    steps: usize,
    spec: OptimizerSpec,
) !TrainingResult {
    var prng = std.Random.DefaultPrng.init(seed);
    var model = try nn.Modules.Mlp.init(context, .{
        .widths = &layer_widths,
        .hidden_activation = .tanh,
        .output_activation = .sigmoid,
    }, prng.random());
    defer model.deinit();

    const parameter_buffer = try allocator.alloc(*nn.Tensor, model.parameterTensorCount());
    defer allocator.free(parameter_buffer);
    const parameters = try model.parameters(parameter_buffer);
    var optimizer = try nn.Training.Optimizer.init(context, spec.config, parameters);
    defer optimizer.deinit();
    const gradient_buffer = try allocator.alloc(nn.Tensor, model.parameterTensorCount());
    defer allocator.free(gradient_buffer);

    const initial = try evaluate(
        allocator,
        context,
        &model,
        training_features,
        training_label_values,
    );
    context.resetStats();

    for (0..steps) |_| {
        try context.beginBatch();
        var batch_active = true;
        errdefer if (batch_active) context.endBatch() catch {};

        var forward = try model.forward(context, training_features);
        defer forward.deinit();
        var difference = try context.subtract(forward.output, training_labels);
        defer difference.deinit();
        var output_gradient = try context.scale(
            difference,
            2.0 / @as(f32, @floatFromInt(training_label_values.len)),
        );
        defer output_gradient.deinit();
        var gradients = try model.backward(context, &forward.cache, output_gradient);
        defer gradients.deinit();
        const parameter_gradients = try gradients.parameterGradients(gradient_buffer);
        const update = try optimizer.step(context, parameters, parameter_gradients);
        std.debug.assert(update.updated);

        try context.endBatch();
        batch_active = false;
    }

    const training_stats = context.stats;
    const final = try evaluate(
        allocator,
        context,
        &model,
        training_features,
        training_label_values,
    );
    const testing = try evaluate(
        allocator,
        context,
        &model,
        testing_features,
        testing_label_values,
    );
    return .{
        .optimizer = spec.name,
        .learning_rate = spec.config.learning_rate,
        .initial_loss = initial.loss,
        .final_loss = final.loss,
        .test_accuracy = testing.accuracy,
        .update_steps = optimizer.update_steps,
        .training_kernels = training_stats.kernels,
        .training_readbacks = training_stats.readbacks,
    };
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

fn backendName(backend: nn.BackendType) []const u8 {
    return switch (backend) {
        .CPU => "cpu",
        .Metal => "metal",
        .CUDA => "cuda",
        .ROCm => "rocm",
    };
}

test "optimizer lab parses reproducible options" {
    const options = try parseArgs(&.{ "--backend", "cpu", "--steps", "12", "--seed", "7" });
    try std.testing.expectEqual(nn.DevicePreference.cpu, options.backend);
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
