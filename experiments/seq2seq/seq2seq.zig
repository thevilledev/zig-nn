const std = @import("std");
const nn = @import("nn");

const sequence_length: usize = 3;
const vocabulary_size: usize = 4;
const channels: usize = 8;
const train_samples: usize = 48;
const test_samples: usize = 16;
const default_steps: usize = 300;

const source_words = [_][]const u8{ "red", "blue", "green", "gold" };
const target_words = [_][]const u8{ "rouge", "bleu", "vert", "or" };

const Options = struct {
    backend: nn.DevicePreference = .auto,
    steps: usize = default_steps,
    seed: u64 = 42,
};

const Sample = struct {
    memory: nn.Tensor,
    targets: nn.Tensor,
    source: [sequence_length]usize,

    fn deinit(self: *Sample) void {
        self.memory.deinit();
        self.targets.deinit();
        self.* = undefined;
    }
};

const Dataset = struct {
    allocator: std.mem.Allocator,
    samples: []Sample,

    fn deinit(self: *Dataset) void {
        for (self.samples) |*sample| sample.deinit();
        self.allocator.free(self.samples);
        self.* = undefined;
    }
};

const Metrics = struct {
    loss: f32,
    token_accuracy: f32,
    sequence_accuracy: f32,
};

const Result = struct {
    backend: nn.BackendType,
    initial: Metrics,
    final: Metrics,
    source: [sequence_length]usize,
    prediction: [sequence_length]usize,
    alignment: [sequence_length][sequence_length]f32,
    training_kernels: usize,
    training_readbacks: usize,
    fixed_padding_efficiency: f32,
    bucketed_padding_efficiency: f32,
    relative_throughput: f32,
    accumulated_microbatches: usize,
};

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const options = parseArgs(args[1..]) catch |err| {
        printUsage();
        if (err == error.HelpRequested) return;
        return err;
    };
    const result = try runExperiment(init.gpa, options);
    std.debug.print("Cross-attention seq2seq lesson on {s}\n", .{backendName(result.backend)});
    std.debug.print("task: translate color words while reversing their order\n", .{});
    std.debug.print(
        "held-out loss: {d:.4} -> {d:.4}; token accuracy: {d:.1}% -> {d:.1}%; exact sequences: {d:.1}%\n" ++
            "training kernels: {d}, training readbacks: {d}\n" ++
            "dynamic batches: padding efficiency {d:.1}% -> {d:.1}%, estimated throughput {d:.2}x\n" ++
            "gradient accumulation: {d} microbatches per optimizer update\n\n",
        .{
            result.initial.loss,
            result.final.loss,
            result.initial.token_accuracy * 100,
            result.final.token_accuracy * 100,
            result.final.sequence_accuracy * 100,
            result.training_kernels,
            result.training_readbacks,
            result.fixed_padding_efficiency * 100,
            result.bucketed_padding_efficiency * 100,
            result.relative_throughput,
            result.accumulated_microbatches,
        },
    );
    std.debug.print("source:    ", .{});
    for (result.source) |token| std.debug.print("{s} ", .{source_words[token]});
    std.debug.print("\npredicted: ", .{});
    for (result.prediction) |token| std.debug.print("{s} ", .{target_words[token]});
    std.debug.print("\n\nalignment rows (target) x columns (source):\n", .{});
    for (result.alignment, 0..) |row, target_position| {
        std.debug.print("  target {d}: ", .{target_position});
        for (row) |weight| std.debug.print("{d:.3} ", .{weight});
        std.debug.print("\n", .{});
    }
    std.debug.print("lesson: decoder queries learn where to read from encoder memory\n", .{});
}

fn parseArgs(args: []const []const u8) !Options {
    var options: Options = .{};
    var index: usize = 0;
    while (index < args.len) : (index += 1) {
        if (std.mem.eql(u8, args[index], "--backend")) {
            index += 1;
            if (index >= args.len) return error.MissingArgument;
            options.backend = try parseBackend(args[index]);
        } else if (std.mem.eql(u8, args[index], "--steps")) {
            index += 1;
            if (index >= args.len) return error.MissingArgument;
            options.steps = try std.fmt.parseInt(usize, args[index], 10);
            if (options.steps == 0) return error.InvalidStepCount;
        } else if (std.mem.eql(u8, args[index], "--seed")) {
            index += 1;
            if (index >= args.len) return error.MissingArgument;
            options.seed = try std.fmt.parseInt(u64, args[index], 10);
        } else if (std.mem.eql(u8, args[index], "--help")) {
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
        \\Usage: seq2seq [options]
        \\
        \\Options:
        \\  --backend <name>  cpu, auto, metal, cuda, or rocm (default: auto)
        \\  --steps <count>   cross-attention SGD updates (default: 300)
        \\  --seed <number>   model and data split seed (default: 42)
        \\  --help            show this help
        \\
    , .{});
}

fn runExperiment(allocator: std.mem.Allocator, options: Options) !Result {
    var device = try nn.Device.init(allocator, options.backend);
    defer device.deinit();
    var context = nn.ExecutionContext.init(&device);
    var split_prng = std.Random.DefaultPrng.init(options.seed ^ 0x9e3779b97f4a7c15);
    var indices: [train_samples + test_samples]usize = undefined;
    for (&indices, 0..) |*index, value| index.* = value;
    split_prng.random().shuffle(usize, &indices);
    var training = try makeDataset(allocator, &context, indices[0..train_samples]);
    defer training.deinit();
    var testing = try makeDataset(allocator, &context, indices[train_samples..]);
    defer testing.deinit();
    var queries = try makeQueries(&context);
    defer queries.deinit();

    var model_prng = std.Random.DefaultPrng.init(options.seed);
    var attention = try nn.Transformer.CrossAttention.init(&context, sequence_length, channels, model_prng.random());
    defer attention.deinit();
    var output = try nn.Modules.Linear.init(&context, channels, vocabulary_size, model_prng.random());
    defer output.deinit();
    const optimizer = try nn.Transformer.Sgd.init(0.07);

    const initial = try evaluate(allocator, &context, &attention, &output, queries, testing.samples);
    context.resetStats();
    for (0..options.steps) |step| {
        const sample = training.samples[step % training.samples.len];
        try trainStep(&context, &attention, &output, optimizer, queries, sample);
    }
    const training_stats = context.stats;
    const final = try evaluate(allocator, &context, &attention, &output, queries, testing.samples);
    var prediction: [sequence_length]usize = undefined;
    var alignment: [sequence_length][sequence_length]f32 = undefined;
    try inspectSample(allocator, &context, &attention, &output, queries, testing.samples[0], &prediction, &alignment);
    const batching = try batchingLesson(allocator);
    return .{
        .backend = device.backendType(),
        .initial = initial,
        .final = final,
        .source = testing.samples[0].source,
        .prediction = prediction,
        .alignment = alignment,
        .training_kernels = training_stats.kernels,
        .training_readbacks = training_stats.readbacks,
        .fixed_padding_efficiency = batching.fixed_efficiency,
        .bucketed_padding_efficiency = batching.bucketed_efficiency,
        .relative_throughput = batching.bucketed_efficiency / batching.fixed_efficiency,
        .accumulated_microbatches = batching.microbatches,
    };
}

fn batchingLesson(allocator: std.mem.Allocator) !struct {
    fixed_efficiency: f32,
    bucketed_efficiency: f32,
    microbatches: usize,
} {
    const lengths = [_]nn.Sequence.ExampleLengths{
        .{ .source = 2, .target = 3 },
        .{ .source = 9, .target = 8 },
        .{ .source = 3, .target = 2 },
        .{ .source = 8, .target = 9 },
        .{ .source = 4, .target = 4 },
        .{ .source = 10, .target = 9 },
    };
    var plan = try nn.Sequence.BatchPlan.init(allocator, &lengths, 2);
    defer plan.deinit();
    var actual_tokens: usize = 0;
    var maximum_source: usize = 0;
    var maximum_target: usize = 0;
    for (lengths) |length| {
        actual_tokens += length.source + length.target;
        maximum_source = @max(maximum_source, length.source);
        maximum_target = @max(maximum_target, length.target);
    }
    const fixed_slots = lengths.len * (maximum_source + maximum_target);
    var accumulator = try nn.Sequence.GradientAccumulator.init(allocator, 2);
    defer accumulator.deinit();
    for (plan.batches) |batch| {
        try accumulator.add(&.{ 0.25, -0.5 }, batch.len);
    }
    var accumulated: [2]f32 = undefined;
    try accumulator.finish(&accumulated);
    if (accumulated[0] != 0.25 or accumulated[1] != -0.5) {
        return error.GradientAccumulationFailed;
    }
    return .{
        .fixed_efficiency = @as(f32, @floatFromInt(actual_tokens)) /
            @as(f32, @floatFromInt(fixed_slots)),
        .bucketed_efficiency = plan.efficiency(),
        .microbatches = accumulator.microbatches,
    };
}

fn trainStep(
    context: *nn.ExecutionContext,
    attention: *nn.Transformer.CrossAttention,
    output: *nn.Modules.Linear,
    optimizer: nn.Transformer.Sgd,
    queries: nn.Tensor,
    sample: Sample,
) !void {
    try context.beginBatch();
    var batch_active = true;
    errdefer if (batch_active) context.endBatch() catch {};
    var attended = try attention.forward(context, queries, sample.memory);
    defer attended.deinit();
    var logits = try output.forward(context, attended);
    defer logits.deinit();
    var objective = try nn.Transformer.SoftmaxCrossEntropy.init(context, logits, sample.targets);
    defer objective.deinit();
    var output_gradients = try output.backward(context, attended, objective.gradient);
    defer output_gradients.deinit();
    var attention_gradients = try attention.backward(context, queries, sample.memory, output_gradients.input);
    defer attention_gradients.deinit();
    try optimizer.stepLinear(context, output, output_gradients);
    try optimizer.stepCrossAttention(context, attention, attention_gradients);
    try context.endBatch();
    batch_active = false;
}

fn evaluate(
    allocator: std.mem.Allocator,
    context: *nn.ExecutionContext,
    attention: *const nn.Transformer.CrossAttention,
    output: *const nn.Modules.Linear,
    queries: nn.Tensor,
    samples: []const Sample,
) !Metrics {
    var total_loss: f32 = 0;
    var correct_tokens: usize = 0;
    var correct_sequences: usize = 0;
    const probabilities = try allocator.alloc(f32, sequence_length * vocabulary_size);
    defer allocator.free(probabilities);
    for (samples) |sample| {
        try context.beginBatch();
        var attended = try attention.forward(context, queries, sample.memory);
        defer attended.deinit();
        var logits = try output.forward(context, attended);
        defer logits.deinit();
        var distribution = try context.softmax(logits);
        defer distribution.deinit();
        try context.endBatch();
        try context.readback(distribution, probabilities);
        var sequence_correct = true;
        for (0..sequence_length) |target_position| {
            const expected = sample.source[sequence_length - 1 - target_position];
            const row = probabilities[target_position * vocabulary_size ..][0..vocabulary_size];
            total_loss -= @log(@max(row[expected], 1e-7));
            const predicted = argmax(row);
            if (predicted == expected) {
                correct_tokens += 1;
            } else {
                sequence_correct = false;
            }
        }
        if (sequence_correct) correct_sequences += 1;
    }
    const token_count = samples.len * sequence_length;
    return .{
        .loss = total_loss / @as(f32, @floatFromInt(token_count)),
        .token_accuracy = @as(f32, @floatFromInt(correct_tokens)) / @as(f32, @floatFromInt(token_count)),
        .sequence_accuracy = @as(f32, @floatFromInt(correct_sequences)) / @as(f32, @floatFromInt(samples.len)),
    };
}

fn inspectSample(
    allocator: std.mem.Allocator,
    context: *nn.ExecutionContext,
    attention: *const nn.Transformer.CrossAttention,
    output: *const nn.Modules.Linear,
    queries: nn.Tensor,
    sample: Sample,
    prediction: *[sequence_length]usize,
    alignment: *[sequence_length][sequence_length]f32,
) !void {
    try context.beginBatch();
    var weights = try attention.attentionWeights(context, queries, sample.memory);
    defer weights.deinit();
    var attended = try attention.forward(context, queries, sample.memory);
    defer attended.deinit();
    var logits = try output.forward(context, attended);
    defer logits.deinit();
    var probabilities = try context.softmax(logits);
    defer probabilities.deinit();
    try context.endBatch();
    const probability_values = try allocator.alloc(f32, sequence_length * vocabulary_size);
    defer allocator.free(probability_values);
    var weight_values: [sequence_length * sequence_length]f32 = undefined;
    try context.readback(probabilities, probability_values);
    try context.readback(weights, &weight_values);
    for (0..sequence_length) |position| {
        prediction[position] = argmax(probability_values[position * vocabulary_size ..][0..vocabulary_size]);
        @memcpy(&alignment[position], weight_values[position * sequence_length ..][0..sequence_length]);
    }
}

fn makeDataset(allocator: std.mem.Allocator, context: *nn.ExecutionContext, indices: []const usize) !Dataset {
    const samples = try allocator.alloc(Sample, indices.len);
    errdefer allocator.free(samples);
    var initialized: usize = 0;
    errdefer for (samples[0..initialized]) |*sample| sample.deinit();
    for (indices, samples) |index, *sample| {
        sample.* = try makeSample(context, index);
        initialized += 1;
    }
    return .{ .allocator = allocator, .samples = samples };
}

fn makeSample(context: *nn.ExecutionContext, index: usize) !Sample {
    var source: [sequence_length]usize = undefined;
    var value = index;
    for (&source) |*token| {
        token.* = value % vocabulary_size;
        value /= vocabulary_size;
    }
    var memory_values: [sequence_length * channels]f32 = @splat(0);
    var target_values: [sequence_length * vocabulary_size]f32 = @splat(0);
    for (source, 0..) |token, position| {
        memory_values[position * channels + token] = 1;
        memory_values[position * channels + vocabulary_size + position] = 1;
        memory_values[position * channels + channels - 1] = 1;
        const target_position = sequence_length - 1 - position;
        target_values[target_position * vocabulary_size + token] = 1;
    }
    var memory = try context.upload(&.{ sequence_length, channels }, &memory_values);
    errdefer memory.deinit();
    const targets = try context.upload(&.{ sequence_length, vocabulary_size }, &target_values);
    return .{ .memory = memory, .targets = targets, .source = source };
}

fn makeQueries(context: *nn.ExecutionContext) !nn.Tensor {
    var values: [sequence_length * channels]f32 = @splat(0);
    for (0..sequence_length) |position| {
        values[position * channels + vocabulary_size + position] = 1;
        values[position * channels + channels - 1] = 1;
    }
    return context.upload(&.{ sequence_length, channels }, &values);
}

fn argmax(values: []const f32) usize {
    var best: usize = 0;
    for (values[1..], 1..) |value, index| {
        if (value > values[best]) best = index;
    }
    return best;
}

fn backendName(backend: nn.BackendType) []const u8 {
    return switch (backend) {
        .CPU => "CPU",
        .Metal => "Metal",
        .CUDA => "CUDA",
        .ROCm => "ROCm",
    };
}

test "cross attention learns translation and reversal on held-out sequences" {
    const result = try runExperiment(std.testing.allocator, .{ .backend = .cpu, .steps = default_steps, .seed = 42 });
    try std.testing.expectEqual(@as(usize, 0), result.training_readbacks);
    try std.testing.expect(result.training_kernels > 0);
    try std.testing.expect(result.final.loss < result.initial.loss * 0.35);
    try std.testing.expect(result.final.token_accuracy > 0.9);
    try std.testing.expect(result.final.sequence_accuracy > 0.75);
    try std.testing.expect(result.bucketed_padding_efficiency > result.fixed_padding_efficiency);
    try std.testing.expect(result.relative_throughput > 1.2);
    try std.testing.expectEqual(@as(usize, 3), result.accumulated_microbatches);
    for (0..sequence_length) |target_position| {
        try std.testing.expectEqual(result.source[sequence_length - 1 - target_position], result.prediction[target_position]);
        try std.testing.expect(argmax(&result.alignment[target_position]) == sequence_length - 1 - target_position);
    }
}
