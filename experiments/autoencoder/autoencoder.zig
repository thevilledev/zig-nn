const std = @import("std");
const nn = @import("nn");

const image_side: usize = 4;
const feature_count: usize = image_side * image_side;
const prototype_count: usize = 4;
const train_samples: usize = 128;
const test_samples: usize = 32;
const default_steps: usize = 300;
const default_seed: u64 = 42;
const widths = [_]usize{ feature_count, 8, 2, 8, feature_count };

const Options = struct {
    backend: nn.DevicePreference = .auto,
    steps: usize = default_steps,
    seed: u64 = default_seed,
};

const Dataset = struct {
    allocator: std.mem.Allocator,
    noisy: []f32,
    clean: []f32,
    labels: []usize,

    fn deinit(self: *Dataset) void {
        self.allocator.free(self.noisy);
        self.allocator.free(self.clean);
        self.allocator.free(self.labels);
        self.* = undefined;
    }
};

const Metrics = struct {
    mse: f32,
    pixel_accuracy: f32,
};

const ExperimentResult = struct {
    backend: nn.BackendType,
    initial: Metrics,
    final: Metrics,
    update_steps: usize,
    training_kernels: usize,
    training_readbacks: usize,
    latent_centers: [prototype_count][2]f32,
    noisy_sample: [feature_count]f32,
    clean_sample: [feature_count]f32,
    reconstruction: [feature_count]f32,
};

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const options = parseArgs(args[1..]) catch |err| {
        printUsage();
        if (err == error.HelpRequested) return;
        return err;
    };
    const result = try runExperiment(init.gpa, options);

    std.debug.print("Denoising autoencoder on {s}\n", .{backendName(result.backend)});
    std.debug.print("architecture: 16 -> 8 -> 2 -> 8 -> 16\n", .{});
    std.debug.print(
        "reconstruction MSE: {d:.6} -> {d:.6}\n" ++
            "held-out pixel accuracy: {d:.2}%\n" ++
            "updates: {d}, kernels: {d}, training readbacks: {d}\n",
        .{
            result.initial.mse,
            result.final.mse,
            result.final.pixel_accuracy * 100,
            result.update_steps,
            result.training_kernels,
            result.training_readbacks,
        },
    );
    std.debug.print("\nlatent centers by pattern:\n", .{});
    for (result.latent_centers, 0..) |center, prototype| {
        std.debug.print(
            "  {s: <14} ({d:>7.3}, {d:>7.3})\n",
            .{ prototypeName(prototype), center[0], center[1] },
        );
    }
    printImage("noisy input", &result.noisy_sample);
    printImage("clean target", &result.clean_sample);
    printImage("reconstruction", &result.reconstruction);
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
        \\Usage: autoencoder [options]
        \\
        \\Options:
        \\  --backend <name>  cpu, auto, metal, cuda, or rocm (default: auto)
        \\  --steps <count>   full-batch training steps (default: 300)
        \\  --seed <number>   model and dataset seed (default: 42)
        \\  --help            show this help
        \\
    , .{});
}

fn runExperiment(allocator: std.mem.Allocator, options: Options) !ExperimentResult {
    var training_data = try generateDataset(allocator, train_samples, options.seed);
    defer training_data.deinit();
    var testing_data = try generateDataset(
        allocator,
        test_samples,
        options.seed ^ 0x9e3779b97f4a7c15,
    );
    defer testing_data.deinit();
    var device = try nn.Device.init(allocator, options.backend);
    defer device.deinit();
    var context = nn.ExecutionContext.init(&device);
    var training_input = try context.upload(
        &.{ train_samples, feature_count },
        training_data.noisy,
    );
    defer training_input.deinit();
    var training_target = try context.upload(
        &.{ train_samples, feature_count },
        training_data.clean,
    );
    defer training_target.deinit();
    var testing_input = try context.upload(
        &.{ test_samples, feature_count },
        testing_data.noisy,
    );
    defer testing_input.deinit();

    var prng = std.Random.DefaultPrng.init(options.seed);
    var model = try nn.Modules.Mlp.init(&context, .{
        .widths = &widths,
        .hidden_activation = .tanh,
        .output_activation = .sigmoid,
    }, prng.random());
    defer model.deinit();
    const parameter_buffer = try allocator.alloc(*nn.Tensor, model.parameterTensorCount());
    defer allocator.free(parameter_buffer);
    const parameters = try model.parameters(parameter_buffer);
    var optimizer = try nn.Training.Optimizer.init(&context, .{
        .kind = .adamw,
        .learning_rate = 0.025,
        .weight_decay = 0.0005,
        .max_gradient_norm = 5,
    }, parameters);
    defer optimizer.deinit();
    const gradient_buffer = try allocator.alloc(nn.Tensor, model.parameterTensorCount());
    defer allocator.free(gradient_buffer);

    const initial = try evaluate(
        allocator,
        &context,
        &model,
        testing_input,
        testing_data.clean,
    );
    context.resetStats();
    for (0..options.steps) |_| {
        try context.beginBatch();
        var batch_active = true;
        errdefer if (batch_active) context.endBatch() catch {};

        var forward = try model.forward(&context, training_input);
        defer forward.deinit();
        var difference = try context.subtract(forward.output, training_target);
        defer difference.deinit();
        const element_count = train_samples * feature_count;
        var output_gradient = try context.scale(
            difference,
            2.0 / @as(f32, @floatFromInt(element_count)),
        );
        defer output_gradient.deinit();
        var gradients = try model.backward(&context, &forward.cache, output_gradient);
        defer gradients.deinit();
        const parameter_gradients = try gradients.parameterGradients(gradient_buffer);
        const update = try optimizer.step(&context, parameters, parameter_gradients);
        std.debug.assert(update.updated);

        try context.endBatch();
        batch_active = false;
    }
    const training_stats = context.stats;
    const final = try evaluate(
        allocator,
        &context,
        &model,
        testing_input,
        testing_data.clean,
    );

    const sample_index = mostCorruptedSample(testing_data);
    const summary = try summarize(
        allocator,
        &context,
        &model,
        testing_input,
        testing_data.labels,
        sample_index,
    );
    var noisy_sample: [feature_count]f32 = undefined;
    var clean_sample: [feature_count]f32 = undefined;
    const sample_offset = sample_index * feature_count;
    @memcpy(&noisy_sample, testing_data.noisy[sample_offset..][0..feature_count]);
    @memcpy(&clean_sample, testing_data.clean[sample_offset..][0..feature_count]);
    return .{
        .backend = device.backendType(),
        .initial = initial,
        .final = final,
        .update_steps = optimizer.update_steps,
        .training_kernels = training_stats.kernels,
        .training_readbacks = training_stats.readbacks,
        .latent_centers = summary.latent_centers,
        .noisy_sample = noisy_sample,
        .clean_sample = clean_sample,
        .reconstruction = summary.reconstruction,
    };
}

const Summary = struct {
    latent_centers: [prototype_count][2]f32,
    reconstruction: [feature_count]f32,
};

fn summarize(
    allocator: std.mem.Allocator,
    context: *nn.ExecutionContext,
    model: *const nn.Modules.Mlp,
    input: nn.Tensor,
    labels: []const usize,
    sample_index: usize,
) !Summary {
    var forward = try model.forward(context, input);
    defer forward.deinit();
    const output_values = try allocator.alloc(f32, labels.len * feature_count);
    defer allocator.free(output_values);
    try context.readback(forward.output, output_values);
    const bottleneck = forward.cache.hidden_activations[1];
    const latent_values = try allocator.alloc(f32, labels.len * 2);
    defer allocator.free(latent_values);
    try context.readback(bottleneck, latent_values);

    var centers = [_][2]f32{.{ 0, 0 }} ** prototype_count;
    var counts = [_]usize{0} ** prototype_count;
    for (labels, 0..) |label, sample| {
        centers[label][0] += latent_values[sample * 2];
        centers[label][1] += latent_values[sample * 2 + 1];
        counts[label] += 1;
    }
    for (&centers, counts) |*center, count| {
        const divisor = @as(f32, @floatFromInt(count));
        center[0] /= divisor;
        center[1] /= divisor;
    }
    var reconstruction: [feature_count]f32 = undefined;
    const sample_offset = sample_index * feature_count;
    @memcpy(&reconstruction, output_values[sample_offset..][0..feature_count]);
    return .{ .latent_centers = centers, .reconstruction = reconstruction };
}

fn mostCorruptedSample(dataset: Dataset) usize {
    var selected: usize = 0;
    var selected_error: f32 = -1;
    for (0..dataset.labels.len) |sample| {
        const offset = sample * feature_count;
        var sample_error: f32 = 0;
        for (
            dataset.noisy[offset..][0..feature_count],
            dataset.clean[offset..][0..feature_count],
        ) |noisy, clean| {
            sample_error += @abs(noisy - clean);
        }
        if (sample_error > selected_error) {
            selected = sample;
            selected_error = sample_error;
        }
    }
    return selected;
}

fn evaluate(
    allocator: std.mem.Allocator,
    context: *nn.ExecutionContext,
    model: *const nn.Modules.Mlp,
    input: nn.Tensor,
    targets: []const f32,
) !Metrics {
    var forward = try model.forward(context, input);
    defer forward.deinit();
    const output_values = try allocator.alloc(f32, targets.len);
    defer allocator.free(output_values);
    try context.readback(forward.output, output_values);
    var squared_error: f32 = 0;
    var correct: usize = 0;
    for (output_values, targets) |output, target| {
        const difference = output - target;
        squared_error += difference * difference;
        if ((output >= 0.5) == (target >= 0.5)) correct += 1;
    }
    const count = @as(f32, @floatFromInt(targets.len));
    return .{
        .mse = squared_error / count,
        .pixel_accuracy = @as(f32, @floatFromInt(correct)) / count,
    };
}

fn generateDataset(allocator: std.mem.Allocator, count: usize, seed: u64) !Dataset {
    if (count == 0) return error.EmptyDataset;
    const noisy = try allocator.alloc(f32, count * feature_count);
    errdefer allocator.free(noisy);
    const clean = try allocator.alloc(f32, count * feature_count);
    errdefer allocator.free(clean);
    const labels = try allocator.alloc(usize, count);
    errdefer allocator.free(labels);
    var prng = std.Random.DefaultPrng.init(seed);
    var random = prng.random();

    for (0..count) |sample| {
        const prototype = sample % prototype_count;
        labels[sample] = prototype;
        for (0..image_side) |y| {
            for (0..image_side) |x| {
                const index = sample * feature_count + y * image_side + x;
                const target: f32 = if (prototypePixel(prototype, y, x)) 1 else 0;
                clean[index] = target;
                const jitter = (random.float(f32) * 2 - 1) * 0.22;
                const flipped = random.float(f32) < 0.04;
                const base = if (flipped) 1 - target else target;
                noisy[index] = std.math.clamp(base + jitter, 0, 1);
            }
        }
    }
    return .{ .allocator = allocator, .noisy = noisy, .clean = clean, .labels = labels };
}

fn prototypePixel(prototype: usize, y: usize, x: usize) bool {
    return switch (prototype) {
        0 => x == 1,
        1 => y == 2,
        2 => x == y,
        3 => x + y == image_side - 1,
        else => unreachable,
    };
}

fn prototypeName(prototype: usize) []const u8 {
    return switch (prototype) {
        0 => "vertical",
        1 => "horizontal",
        2 => "diagonal",
        3 => "anti-diagonal",
        else => unreachable,
    };
}

fn printImage(label: []const u8, values: []const f32) void {
    std.debug.print("\n{s}:\n", .{label});
    for (0..image_side) |y| {
        for (0..image_side) |x| {
            const value = values[y * image_side + x];
            const symbol: []const u8 = if (value >= 0.65)
                "# "
            else if (value >= 0.35)
                "+ "
            else
                ". ";
            std.debug.print("{s}", .{symbol});
        }
        std.debug.print("\n", .{});
    }
}

fn backendName(backend: nn.BackendType) []const u8 {
    return switch (backend) {
        .CPU => "cpu",
        .Metal => "metal",
        .CUDA => "cuda",
        .ROCm => "rocm",
    };
}

test "autoencoder options validate backend and steps" {
    const options = try parseArgs(&.{ "--backend", "cpu", "--steps", "12", "--seed", "7" });
    try std.testing.expectEqual(nn.DevicePreference.cpu, options.backend);
    try std.testing.expectEqual(@as(usize, 12), options.steps);
    try std.testing.expectEqual(@as(u64, 7), options.seed);
    try std.testing.expectError(error.UnknownBackend, parseBackend("quantum"));
    try std.testing.expectError(error.InvalidStepCount, parseArgs(&.{ "--steps", "0" }));
}

test "autoencoder dataset is deterministic and balanced" {
    var first = try generateDataset(std.testing.allocator, 8, 3);
    defer first.deinit();
    var second = try generateDataset(std.testing.allocator, 8, 3);
    defer second.deinit();
    try std.testing.expectEqualSlices(f32, first.noisy, second.noisy);
    try std.testing.expectEqualSlices(f32, first.clean, second.clean);
    try std.testing.expectEqualSlices(usize, first.labels, second.labels);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 2, 3, 0, 1, 2, 3 }, first.labels);
}

test "autoencoder learns denoising through a two-value bottleneck" {
    const result = try runExperiment(std.testing.allocator, .{
        .backend = .cpu,
        .steps = 220,
        .seed = default_seed,
    });
    try std.testing.expect(result.final.mse < result.initial.mse * 0.25);
    try std.testing.expect(result.final.pixel_accuracy >= 0.95);
    try std.testing.expectEqual(@as(usize, 220), result.update_steps);
    try std.testing.expectEqual(@as(usize, 0), result.training_readbacks);
}
