const std = @import("std");
const nn = @import("nn");

const sequence_length: usize = 4;
const channels: usize = 4;
const hidden_features: usize = 8;
const encoder_parameter_tensors: usize = 16;
const total_parameter_tensors: usize = 18;
const train_samples: usize = 64;
const test_samples: usize = 32;
const default_steps: usize = 200;
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

    fn sample(self: Dataset, index: usize) []const f32 {
        const offset = index * sequence_length * channels;
        return self.inputs[offset..][0 .. sequence_length * channels];
    }

    fn bit(self: Dataset, sample_index: usize, token_index: usize) f32 {
        return self.sample(sample_index)[token_index * channels];
    }
};

const Metrics = struct {
    mse: f32,
    first_token_accuracy: f32,
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
    std.debug.print("Transformer encoder context lesson on {s}\n", .{backendName(result.backend)});
    std.debug.print(
        "task: predict the final bit from the first token\n" ++
            "model: unmasked attention + residual MLP -> sigmoid\n",
        .{},
    );
    std.debug.print(
        "first-token MSE: {d:.6} -> {d:.6}, held-out accuracy: {d:.2}%\n" ++
            "updates: {d}, kernels: {d}, training readbacks: {d}\n\n",
        .{
            result.initial.mse,
            result.final.mse,
            result.final.first_token_accuracy * 100,
            result.update_steps,
            result.training_kernels,
            result.training_readbacks,
        },
    );
    for (result.sample_sequences, result.sample_labels, result.sample_predictions) |sequence, label, prediction| {
        std.debug.print("tokens ", .{});
        for (sequence) |bit| std.debug.print("{d:.0} ", .{bit});
        std.debug.print(
            "=> final {d:.0}, first-token prediction {d:.3}\n",
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
        \\Usage: transformer_encoder [options]
        \\
        \\Options:
        \\  --backend <name>  cpu, auto, metal, cuda, or rocm (default: auto)
        \\  --steps <count>   single-sequence AdamW updates (default: 200)
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
    const training_inputs = try uploadDataset(allocator, &context, training_data);
    defer deinitTensors(allocator, training_inputs);
    const testing_inputs = try uploadDataset(allocator, &context, testing_data);
    defer deinitTensors(allocator, testing_inputs);
    var target_zero = try context.upload(&.{ sequence_length, 1 }, &.{ 0, 0, 0, 0 });
    defer target_zero.deinit();
    var target_one = try context.upload(&.{ sequence_length, 1 }, &.{ 1, 1, 1, 1 });
    defer target_one.deinit();
    var first_token_mask = try context.upload(&.{ sequence_length, 1 }, &.{ 1, 0, 0, 0 });
    defer first_token_mask.deinit();

    var prng = std.Random.DefaultPrng.init(options.seed);
    var encoder = try nn.Transformer.EncoderBlock.init(
        &context,
        sequence_length,
        channels,
        hidden_features,
        prng.random(),
    );
    defer encoder.deinit();
    var output_head = try nn.Modules.Linear.init(&context, channels, 1, prng.random());
    defer output_head.deinit();
    var parameter_buffer: [total_parameter_tensors]*nn.Tensor = undefined;
    const encoder_parameters = try encoder.parameters(parameter_buffer[0..encoder_parameter_tensors]);
    std.debug.assert(encoder_parameters.len == encoder_parameter_tensors);
    parameter_buffer[16] = &output_head.weights;
    parameter_buffer[17] = &output_head.bias;
    const parameters = parameter_buffer[0..];
    var optimizer = try nn.Training.Optimizer.init(&context, .{
        .kind = .adamw,
        .learning_rate = 0.008,
        .weight_decay = 0.0005,
        .max_gradient_norm = 3,
    }, parameters);
    defer optimizer.deinit();

    const initial = try evaluate(
        allocator,
        &context,
        &encoder,
        &output_head,
        testing_inputs,
        testing_data.labels,
        null,
    );
    context.resetStats();
    for (0..options.steps) |step| {
        const sample_index = step % training_inputs.len;
        const target = if (training_data.labels[sample_index] == 0)
            target_zero
        else
            target_one;
        try trainStep(
            &context,
            &encoder,
            &output_head,
            &optimizer,
            parameters,
            training_inputs[sample_index],
            target,
            first_token_mask,
        );
    }
    const training_stats = context.stats;
    var predictions: [4]f32 = undefined;
    const final = try evaluate(
        allocator,
        &context,
        &encoder,
        &output_head,
        testing_inputs,
        testing_data.labels,
        &predictions,
    );
    var sample_sequences: [4][sequence_length]f32 = undefined;
    var sample_labels: [4]f32 = undefined;
    for (0..4) |sample_index| {
        sample_labels[sample_index] = testing_data.labels[sample_index];
        for (0..sequence_length) |token_index| {
            sample_sequences[sample_index][token_index] = if (testing_data.bit(sample_index, token_index) > 0) 1 else 0;
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
    encoder: *const nn.Transformer.EncoderBlock,
    output_head: *const nn.Modules.Linear,
    optimizer: *nn.Training.Optimizer,
    parameters: []const *nn.Tensor,
    input: nn.Tensor,
    target: nn.Tensor,
    first_token_mask: nn.Tensor,
) !void {
    try context.beginBatch();
    var batch_active = true;
    errdefer if (batch_active) context.endBatch() catch {};

    var encoded = try encoder.forward(context, input);
    defer encoded.deinit();
    var logits = try output_head.forward(context, encoded);
    defer logits.deinit();
    var predictions = try context.activate(logits, .sigmoid);
    defer predictions.deinit();
    var difference = try context.subtract(predictions, target);
    defer difference.deinit();
    var masked_difference = try context.multiply(difference, first_token_mask);
    defer masked_difference.deinit();
    var prediction_gradient = try context.scale(masked_difference, 2);
    defer prediction_gradient.deinit();
    var sigmoid_derivative = try context.activationDerivative(logits, .sigmoid);
    defer sigmoid_derivative.deinit();
    var logits_gradient = try context.multiply(prediction_gradient, sigmoid_derivative);
    defer logits_gradient.deinit();
    var head_gradients = try output_head.backward(context, encoded, logits_gradient);
    defer head_gradients.deinit();
    var encoder_gradients = try encoder.backward(context, input, head_gradients.input);
    defer encoder_gradients.deinit();

    var encoder_gradient_buffer: [encoder_parameter_tensors]nn.Tensor = undefined;
    const encoder_parameter_gradients = try encoder_gradients.parameterGradients(
        &encoder_gradient_buffer,
    );
    var all_gradients: [total_parameter_tensors]nn.Tensor = undefined;
    @memcpy(all_gradients[0..encoder_parameter_tensors], encoder_parameter_gradients);
    all_gradients[16] = head_gradients.weights;
    all_gradients[17] = head_gradients.bias;
    const update = try optimizer.step(context, parameters, &all_gradients);
    std.debug.assert(update.updated);

    try context.endBatch();
    batch_active = false;
}

fn evaluate(
    allocator: std.mem.Allocator,
    context: *nn.ExecutionContext,
    encoder: *const nn.Transformer.EncoderBlock,
    output_head: *const nn.Modules.Linear,
    inputs: []const nn.Tensor,
    labels: []const f32,
    sample_predictions: ?*[4]f32,
) !Metrics {
    var squared_error: f32 = 0;
    var correct: usize = 0;
    for (inputs, labels, 0..) |input, label, sample_index| {
        try context.beginBatch();
        var batch_active = true;
        errdefer if (batch_active) context.endBatch() catch {};
        var encoded = try encoder.forward(context, input);
        defer encoded.deinit();
        var logits = try output_head.forward(context, encoded);
        defer logits.deinit();
        var predictions = try context.activate(logits, .sigmoid);
        defer predictions.deinit();
        try context.endBatch();
        batch_active = false;

        const values = try allocator.alloc(f32, sequence_length);
        defer allocator.free(values);
        try context.readback(predictions, values);
        const prediction = values[0];
        if (sample_predictions) |samples| {
            if (sample_index < samples.len) samples[sample_index] = prediction;
        }
        const difference = prediction - label;
        squared_error += difference * difference;
        if ((prediction >= 0.5) == (label >= 0.5)) correct += 1;
    }
    const count = @as(f32, @floatFromInt(labels.len));
    return .{
        .mse = squared_error / count,
        .first_token_accuracy = @as(f32, @floatFromInt(correct)) / count,
    };
}

fn uploadDataset(
    allocator: std.mem.Allocator,
    context: *nn.ExecutionContext,
    dataset: Dataset,
) ![]nn.Tensor {
    const tensors = try allocator.alloc(nn.Tensor, dataset.sample_count);
    errdefer allocator.free(tensors);
    var initialized: usize = 0;
    errdefer for (tensors[0..initialized]) |*tensor| tensor.deinit();
    for (tensors, 0..) |*tensor, sample_index| {
        tensor.* = try context.upload(
            &.{ sequence_length, channels },
            dataset.sample(sample_index),
        );
        initialized += 1;
    }
    return tensors;
}

fn deinitTensors(allocator: std.mem.Allocator, tensors: []nn.Tensor) void {
    for (tensors) |*tensor| tensor.deinit();
    allocator.free(tensors);
}

fn generateDataset(allocator: std.mem.Allocator, count: usize, seed: u64) !Dataset {
    if (count == 0) return error.EmptyDataset;
    const inputs = try allocator.alloc(f32, count * sequence_length * channels);
    errdefer allocator.free(inputs);
    const labels = try allocator.alloc(f32, count);
    errdefer allocator.free(labels);
    var prng = std.Random.DefaultPrng.init(seed);
    var random = prng.random();
    for (0..count) |sample_index| {
        const label: f32 = @floatFromInt(sample_index % 2);
        labels[sample_index] = label;
        for (0..sequence_length) |token_index| {
            const offset = (sample_index * sequence_length + token_index) * channels;
            const bit: f32 = if (token_index == sequence_length - 1)
                label
            else if (random.boolean())
                1
            else
                0;
            inputs[offset] = bit * 2 - 1;
            inputs[offset + 1] = if (token_index == 0) 1 else 0;
            inputs[offset + 2] = if (token_index == sequence_length - 1) 1 else 0;
            inputs[offset + 3] = @as(f32, @floatFromInt(token_index)) /
                @as(f32, @floatFromInt(sequence_length - 1));
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

test "Transformer encoder options validate backend and steps" {
    const options = try parseArgs(&.{ "--backend", "cpu", "--steps", "12", "--seed", "7" });
    try std.testing.expectEqual(nn.DevicePreference.cpu, options.backend);
    try std.testing.expectEqual(@as(usize, 12), options.steps);
    try std.testing.expectEqual(@as(u64, 7), options.seed);
    try std.testing.expectError(error.UnknownBackend, parseBackend("quantum"));
    try std.testing.expectError(error.InvalidStepCount, parseArgs(&.{ "--steps", "0" }));
}

test "Transformer encoder dataset marks the query and final positions" {
    var dataset = try generateDataset(std.testing.allocator, 4, 3);
    defer dataset.deinit();
    for (0..dataset.sample_count) |sample_index| {
        const sample = dataset.sample(sample_index);
        try std.testing.expectEqual(@as(f32, 1), sample[1]);
        try std.testing.expectEqual(@as(f32, 0), sample[2]);
        const last = (sequence_length - 1) * channels;
        try std.testing.expectEqual(@as(f32, 0), sample[last + 1]);
        try std.testing.expectEqual(@as(f32, 1), sample[last + 2]);
        const final_bit: f32 = if (sample[last] > 0) 1 else 0;
        try std.testing.expectEqual(dataset.labels[sample_index], final_bit);
    }
}

test "Transformer encoder learns to retrieve the final token" {
    const result = try runExperiment(std.testing.allocator, .{
        .backend = .cpu,
        .steps = 160,
        .seed = default_seed,
    });
    try std.testing.expect(result.final.mse < result.initial.mse * 0.3);
    try std.testing.expect(result.final.first_token_accuracy >= 0.95);
    try std.testing.expectEqual(@as(usize, 160), result.update_steps);
    try std.testing.expectEqual(@as(usize, 0), result.training_readbacks);
}
