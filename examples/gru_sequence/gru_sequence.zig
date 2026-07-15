const std = @import("std");
const nn = @import("nn");

const sequence_length: usize = 6;
const input_size: usize = 2;
const hidden_size: usize = 6;
const gru_parameter_tensors: usize = 12;
const total_parameter_tensors: usize = 14;
const train_samples: usize = 64;
const test_samples: usize = 32;
const default_steps: usize = 80;
const default_seed: u64 = 42;

const Options = struct {
    backend: nn.DevicePreference = .auto,
    steps: usize = default_steps,
    seed: u64 = default_seed,
};

const Dataset = struct {
    allocator: std.mem.Allocator,
    sample_count: usize,
    inputs: []f32,
    labels: []f32,

    fn deinit(self: *Dataset) void {
        self.allocator.free(self.inputs);
        self.allocator.free(self.labels);
        self.* = undefined;
    }

    fn timestep(self: Dataset, index: usize) []const f32 {
        const offset = index * self.sample_count * input_size;
        return self.inputs[offset..][0 .. self.sample_count * input_size];
    }
};

const Metrics = struct {
    mse: f32,
    accuracy: f32,
};

const ExperimentResult = struct {
    backend: nn.BackendType,
    initial: Metrics,
    final: Metrics,
    update_steps: usize,
    training_kernels: usize,
    training_readbacks: usize,
    sample_predictions: [4]f32,
    sample_sequences: [4][sequence_length]f32,
    sample_labels: [4]f32,
};

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const options = parseArgs(args[1..]) catch |err| {
        printUsage();
        if (err == error.HelpRequested) return;
        return err;
    };
    const result = try runExperiment(init.gpa, options);
    std.debug.print("GRU selective-memory lesson on {s}\n", .{backendName(result.backend)});
    std.debug.print(
        "task: remember the marked first bit across {d} distractor steps\n" ++
            "model: GRU({d} inputs, {d} hidden) -> sigmoid\n",
        .{ sequence_length - 1, input_size, hidden_size },
    );
    std.debug.print(
        "MSE: {d:.6} -> {d:.6}, held-out accuracy: {d:.2}%\n" ++
            "updates: {d}, kernels: {d}, training readbacks: {d}\n\n",
        .{
            result.initial.mse,
            result.final.mse,
            result.final.accuracy * 100,
            result.update_steps,
            result.training_kernels,
            result.training_readbacks,
        },
    );
    for (result.sample_sequences, result.sample_labels, result.sample_predictions) |sequence, label, prediction| {
        std.debug.print("marked [{d:.0}] then ", .{sequence[0]});
        for (sequence[1..]) |distractor| std.debug.print("{d:.0} ", .{distractor});
        std.debug.print(
            "=> target {d:.0}, prediction {d:.3}\n",
            .{ label, prediction },
        );
    }
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
        \\Usage: gru_sequence [options]
        \\
        \\Options:
        \\  --backend <name>  cpu, auto, metal, cuda, or rocm (default: auto)
        \\  --steps <count>   full-batch BPTT updates (default: 80)
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
    const training_inputs = try uploadSequence(allocator, &context, training_data);
    defer deinitSequence(allocator, training_inputs);
    const testing_inputs = try uploadSequence(allocator, &context, testing_data);
    defer deinitSequence(allocator, testing_inputs);
    var training_labels = try context.upload(&.{ train_samples, 1 }, training_data.labels);
    defer training_labels.deinit();

    var prng = std.Random.DefaultPrng.init(options.seed);
    var gru = try nn.Recurrent.Gru.init(&context, input_size, hidden_size, prng.random());
    defer gru.deinit();
    var output_head = try nn.Modules.Linear.init(&context, hidden_size, 1, prng.random());
    defer output_head.deinit();
    var parameter_buffer: [total_parameter_tensors]*nn.Tensor = undefined;
    const gru_parameters = try gru.parameters(parameter_buffer[0..gru_parameter_tensors]);
    std.debug.assert(gru_parameters.len == gru_parameter_tensors);
    parameter_buffer[12] = &output_head.weights;
    parameter_buffer[13] = &output_head.bias;
    const parameters = parameter_buffer[0..];
    var optimizer = try nn.Training.Optimizer.init(&context, .{
        .kind = .adamw,
        .learning_rate = 0.02,
        .weight_decay = 0.0005,
        .max_gradient_norm = 5,
    }, parameters);
    defer optimizer.deinit();

    const initial = try evaluate(
        allocator,
        &context,
        &gru,
        &output_head,
        testing_inputs,
        testing_data.labels,
        null,
    );
    context.resetStats();
    for (0..options.steps) |_| {
        try trainStep(
            &context,
            &gru,
            &output_head,
            &optimizer,
            parameters,
            training_inputs,
            training_labels,
        );
    }
    const training_stats = context.stats;
    var predictions: [4]f32 = undefined;
    const final = try evaluate(
        allocator,
        &context,
        &gru,
        &output_head,
        testing_inputs,
        testing_data.labels,
        &predictions,
    );
    var sample_sequences: [4][sequence_length]f32 = undefined;
    var sample_labels: [4]f32 = undefined;
    for (0..4) |sample| {
        sample_labels[sample] = testing_data.labels[sample];
        for (0..sequence_length) |timestep| {
            sample_sequences[sample][timestep] =
                testing_data.timestep(timestep)[sample * input_size];
        }
    }
    return .{
        .backend = device.backendType(),
        .initial = initial,
        .final = final,
        .update_steps = optimizer.update_steps,
        .training_kernels = training_stats.kernels,
        .training_readbacks = training_stats.readbacks,
        .sample_predictions = predictions,
        .sample_sequences = sample_sequences,
        .sample_labels = sample_labels,
    };
}

fn trainStep(
    context: *nn.ExecutionContext,
    gru: *const nn.Recurrent.Gru,
    output_head: *const nn.Modules.Linear,
    optimizer: *nn.Training.Optimizer,
    parameters: []const *nn.Tensor,
    inputs: []const nn.Tensor,
    labels: nn.Tensor,
) !void {
    try context.beginBatch();
    var batch_active = true;
    errdefer if (batch_active) context.endBatch() catch {};

    var initial_hidden = try context.createTensor(&.{ train_samples, hidden_size });
    defer initial_hidden.deinit();
    var forwards: [sequence_length]nn.Recurrent.GruForward = undefined;
    var forward_count: usize = 0;
    defer for (forwards[0..forward_count]) |*forward| forward.deinit();
    var hidden = initial_hidden;
    for (inputs, 0..) |input, timestep| {
        forwards[timestep] = try gru.forward(context, input, hidden);
        forward_count += 1;
        hidden = forwards[timestep].output;
    }

    var logits = try output_head.forward(context, forwards[sequence_length - 1].output);
    defer logits.deinit();
    var predictions = try context.activate(logits, .sigmoid);
    defer predictions.deinit();
    var difference = try context.subtract(predictions, labels);
    defer difference.deinit();
    var prediction_gradient = try context.scale(
        difference,
        2.0 / @as(f32, @floatFromInt(train_samples)),
    );
    defer prediction_gradient.deinit();
    var sigmoid_derivative = try context.activationDerivative(logits, .sigmoid);
    defer sigmoid_derivative.deinit();
    var logits_gradient = try context.multiply(prediction_gradient, sigmoid_derivative);
    defer logits_gradient.deinit();
    var head_gradients = try output_head.backward(
        context,
        forwards[sequence_length - 1].output,
        logits_gradient,
    );
    defer head_gradients.deinit();

    var step_gradients: [sequence_length]nn.Recurrent.GruGradients = undefined;
    var gradient_start: usize = sequence_length;
    defer for (step_gradients[gradient_start..]) |*gradient| gradient.deinit();
    var accumulated: [gru_parameter_tensors]nn.Tensor = undefined;
    var accumulated_count: usize = 0;
    defer for (accumulated[0..accumulated_count]) |*gradient| gradient.deinit();
    var hidden_gradient = head_gradients.input;
    var reverse = sequence_length;
    while (reverse > 0) {
        reverse -= 1;
        step_gradients[reverse] = try gru.backward(
            context,
            &forwards[reverse].cache,
            hidden_gradient,
        );
        gradient_start = reverse;
        hidden_gradient = step_gradients[reverse].previous_hidden;
        var current: [gru_parameter_tensors]nn.Tensor = undefined;
        const current_slice = try step_gradients[reverse].parameterGradients(&current);
        if (accumulated_count == 0) {
            for (current_slice, 0..) |gradient, index| {
                accumulated[index] = try context.scale(gradient, 1);
                accumulated_count += 1;
            }
        } else {
            for (current_slice, 0..) |gradient, index| {
                const combined = try context.add(accumulated[index], gradient);
                accumulated[index].deinit();
                accumulated[index] = combined;
            }
        }
    }

    var all_gradients: [total_parameter_tensors]nn.Tensor = undefined;
    @memcpy(all_gradients[0..gru_parameter_tensors], accumulated[0..]);
    all_gradients[12] = head_gradients.weights;
    all_gradients[13] = head_gradients.bias;
    const update = try optimizer.step(context, parameters, &all_gradients);
    std.debug.assert(update.updated);

    try context.endBatch();
    batch_active = false;
}

fn evaluate(
    allocator: std.mem.Allocator,
    context: *nn.ExecutionContext,
    gru: *const nn.Recurrent.Gru,
    output_head: *const nn.Modules.Linear,
    inputs: []const nn.Tensor,
    labels: []const f32,
    sample_predictions: ?*[4]f32,
) !Metrics {
    var initial_hidden = try context.createTensor(&.{ labels.len, hidden_size });
    defer initial_hidden.deinit();
    var forwards: [sequence_length]nn.Recurrent.GruForward = undefined;
    var forward_count: usize = 0;
    defer for (forwards[0..forward_count]) |*forward| forward.deinit();
    var hidden = initial_hidden;
    for (inputs, 0..) |input, timestep| {
        forwards[timestep] = try gru.forward(context, input, hidden);
        forward_count += 1;
        hidden = forwards[timestep].output;
    }
    var logits = try output_head.forward(context, forwards[sequence_length - 1].output);
    defer logits.deinit();
    var predictions = try context.activate(logits, .sigmoid);
    defer predictions.deinit();
    const values = try allocator.alloc(f32, labels.len);
    defer allocator.free(values);
    try context.readback(predictions, values);
    if (sample_predictions) |samples| @memcpy(samples, values[0..4]);

    var squared_error: f32 = 0;
    var correct: usize = 0;
    for (values, labels) |prediction, label| {
        const difference = prediction - label;
        squared_error += difference * difference;
        if ((prediction >= 0.5) == (label >= 0.5)) correct += 1;
    }
    const count = @as(f32, @floatFromInt(labels.len));
    return .{
        .mse = squared_error / count,
        .accuracy = @as(f32, @floatFromInt(correct)) / count,
    };
}

fn uploadSequence(
    allocator: std.mem.Allocator,
    context: *nn.ExecutionContext,
    dataset: Dataset,
) ![]nn.Tensor {
    const tensors = try allocator.alloc(nn.Tensor, sequence_length);
    errdefer allocator.free(tensors);
    var initialized: usize = 0;
    errdefer for (tensors[0..initialized]) |*tensor| tensor.deinit();
    for (tensors, 0..) |*tensor, timestep| {
        tensor.* = try context.upload(
            &.{ dataset.sample_count, input_size },
            dataset.timestep(timestep),
        );
        initialized += 1;
    }
    return tensors;
}

fn deinitSequence(allocator: std.mem.Allocator, tensors: []nn.Tensor) void {
    for (tensors) |*tensor| tensor.deinit();
    allocator.free(tensors);
}

fn generateDataset(allocator: std.mem.Allocator, count: usize, seed: u64) !Dataset {
    if (count == 0) return error.EmptyDataset;
    const inputs = try allocator.alloc(f32, sequence_length * count * input_size);
    errdefer allocator.free(inputs);
    const labels = try allocator.alloc(f32, count);
    errdefer allocator.free(labels);
    var prng = std.Random.DefaultPrng.init(seed);
    var random = prng.random();
    for (0..count) |sample| {
        const label: f32 = @floatFromInt(sample % 2);
        labels[sample] = label;
        for (0..sequence_length) |timestep| {
            const offset = (timestep * count + sample) * input_size;
            inputs[offset] = if (timestep == 0)
                label
            else if (random.boolean())
                1
            else
                0;
            inputs[offset + 1] = if (timestep == 0) 1 else 0;
        }
    }
    return .{
        .allocator = allocator,
        .sample_count = count,
        .inputs = inputs,
        .labels = labels,
    };
}

fn backendName(backend: nn.BackendType) []const u8 {
    return switch (backend) {
        .CPU => "cpu",
        .Metal => "metal",
        .CUDA => "cuda",
        .ROCm => "rocm",
    };
}

test "GRU sequence options validate backend and steps" {
    const options = try parseArgs(&.{ "--backend", "cpu", "--steps", "12", "--seed", "7" });
    try std.testing.expectEqual(nn.DevicePreference.cpu, options.backend);
    try std.testing.expectEqual(@as(usize, 12), options.steps);
    try std.testing.expectEqual(@as(u64, 7), options.seed);
    try std.testing.expectError(error.UnknownBackend, parseBackend("quantum"));
    try std.testing.expectError(error.InvalidStepCount, parseArgs(&.{ "--steps", "0" }));
}

test "GRU sequence dataset is deterministic and balanced" {
    var first = try generateDataset(std.testing.allocator, 8, 3);
    defer first.deinit();
    var second = try generateDataset(std.testing.allocator, 8, 3);
    defer second.deinit();
    try std.testing.expectEqualSlices(f32, first.inputs, second.inputs);
    try std.testing.expectEqualSlices(f32, first.labels, second.labels);
    try std.testing.expectEqualSlices(f32, &.{ 0, 1, 0, 1, 0, 1, 0, 1 }, first.labels);
}

test "GRU learns to remember the marked bit through distractors" {
    const result = try runExperiment(std.testing.allocator, .{
        .backend = .cpu,
        .steps = 60,
        .seed = default_seed,
    });
    try std.testing.expect(result.final.mse < result.initial.mse * 0.25);
    try std.testing.expect(result.final.accuracy >= 0.95);
    try std.testing.expectEqual(@as(usize, 60), result.update_steps);
    try std.testing.expectEqual(@as(usize, 0), result.training_readbacks);
}
