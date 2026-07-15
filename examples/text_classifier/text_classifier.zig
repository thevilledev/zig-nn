const std = @import("std");
const nn = @import("nn");

const batch_size: usize = 10;
const maximum_tokens: usize = 4;
const channels: usize = 8;
const heads: usize = 2;
const hidden_features: usize = 16;
const vocabulary_size: usize = 11;
const default_steps: usize = 60;
const parameter_count: usize = 19;

const Token = enum(usize) {
    padding,
    good,
    great,
    love,
    enjoy,
    bad,
    awful,
    hate,
    boring,
    movie,
    not,
};

const token_values = [_]f32{
    token(.good),   token(.movie), 0,             0,
    token(.great),  token(.movie), 0,             0,
    token(.love),   token(.movie), 0,             0,
    token(.enjoy),  token(.good),  token(.movie), 0,
    token(.not),    token(.bad),   token(.movie), 0,
    token(.bad),    token(.movie), 0,             0,
    token(.awful),  token(.movie), 0,             0,
    token(.hate),   token(.movie), 0,             0,
    token(.boring), token(.bad),   token(.movie), 0,
    token(.not),    token(.good),  token(.movie), 0,
};

const lengths = [_]usize{ 2, 2, 2, 3, 3, 2, 2, 2, 3, 3 };
const labels = [_]usize{ 1, 1, 1, 1, 1, 0, 0, 0, 0, 0 };
const label_mask = [_]f32{1} ** batch_size;

const Options = struct {
    backend: nn.DevicePreference = .auto,
    steps: usize = default_steps,
    seed: u64 = 42,
};

const Metrics = struct {
    loss: f32,
    accuracy: f32,
    probabilities: [batch_size * 2]f32,
};

const Result = struct {
    backend: nn.BackendType,
    initial: Metrics,
    final: Metrics,
    padding_difference: f32,
    training_readbacks: usize,
    kernels: usize,
};

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const options = parseArgs(args[1..]) catch |err| {
        printUsage();
        if (err == error.HelpRequested) return;
        return err;
    };
    const result = try runExperiment(init.gpa, options);
    std.debug.print("Masked Transformer text classifier on {s}\n", .{backendName(result.backend)});
    std.debug.print(
        "cross-entropy: {d:.4} -> {d:.4}\n" ++
            "accuracy: {d:.0}% -> {d:.0}%\n" ++
            "largest probability change after replacing padding: {d:.8}\n" ++
            "kernels: {d}, training readbacks: {d}\n",
        .{
            result.initial.loss,
            result.final.loss,
            result.initial.accuracy * 100,
            result.final.accuracy * 100,
            result.padding_difference,
            result.kernels,
            result.training_readbacks,
        },
    );
    std.debug.print("context examples: 'not bad movie' -> positive, 'not good movie' -> negative\n", .{});
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
        \\Usage: text_classifier [options]
        \\
        \\Options:
        \\  --backend <name>  cpu, auto, metal, cuda, or rocm
        \\  --steps <count>   full-batch AdamW updates (default: 60)
        \\  --seed <number>   initialization seed
        \\  --help            show this help
        \\
    , .{});
}

fn runExperiment(allocator: std.mem.Allocator, options: Options) !Result {
    var device = try nn.Device.init(allocator, options.backend);
    defer device.deinit();
    var context = nn.ExecutionContext.init(&device);
    var prng = std.Random.DefaultPrng.init(options.seed);
    var embedding = try nn.Transformer.Embedding.init(&context, vocabulary_size, channels, prng.random());
    defer embedding.deinit();
    var encoder = try nn.Transformer.MaskedEncoderBlock.init(&context, channels, heads, hidden_features, prng.random());
    defer encoder.deinit();
    var classifier = try nn.Modules.Linear.init(&context, channels, 2, prng.random());
    defer classifier.deinit();

    var host_masks = try nn.Transformer.PaddingMasks.init(allocator, &lengths, maximum_tokens, heads, channels);
    defer host_masks.deinit();
    var indices = try context.upload(&.{ batch_size * maximum_tokens, 1 }, &token_values);
    defer indices.deinit();
    var changed_values = token_values;
    for (lengths, 0..) |length, row| {
        for (length..maximum_tokens) |position| changed_values[row * maximum_tokens + position] = if ((row + position) % 2 == 0) token(.great) else token(.awful);
    }
    var changed_indices = try context.upload(&.{ batch_size * maximum_tokens, 1 }, &changed_values);
    defer changed_indices.deinit();
    var attention_mask = try context.upload(&.{ batch_size * heads, maximum_tokens, maximum_tokens }, host_masks.attention);
    defer attention_mask.deinit();
    var token_mask = try context.upload(&.{ batch_size, maximum_tokens, channels }, host_masks.tokens);
    defer token_mask.deinit();
    var mean_pool = try context.upload(&.{ batch_size, 1, maximum_tokens }, host_masks.mean_pool);
    defer mean_pool.deinit();

    var parameter_storage: [parameter_count]*nn.Tensor = undefined;
    parameter_storage[0] = &embedding.weights;
    _ = try encoder.parameters(parameter_storage[1..17]);
    parameter_storage[17] = &classifier.weights;
    parameter_storage[18] = &classifier.bias;
    const parameters = parameter_storage[0..];
    var optimizer = try nn.Training.Optimizer.init(&context, .{
        .kind = .adamw,
        .learning_rate = 0.018,
        .weight_decay = 0.0005,
        .max_gradient_norm = 3,
    }, parameters);
    defer optimizer.deinit();

    const initial = try evaluate(allocator, &context, &embedding, &encoder, &classifier, indices, attention_mask, token_mask, mean_pool);
    context.resetStats();
    for (0..options.steps) |_| {
        try trainStep(&context, &embedding, &encoder, &classifier, &optimizer, parameters, indices, attention_mask, token_mask, mean_pool);
    }
    const training_stats = context.stats;
    const final = try evaluate(allocator, &context, &embedding, &encoder, &classifier, indices, attention_mask, token_mask, mean_pool);
    const changed = try evaluate(allocator, &context, &embedding, &encoder, &classifier, changed_indices, attention_mask, token_mask, mean_pool);
    var padding_difference: f32 = 0;
    for (final.probabilities, changed.probabilities) |expected, actual| padding_difference = @max(padding_difference, @abs(expected - actual));
    return .{
        .backend = device.backendType(),
        .initial = initial,
        .final = final,
        .padding_difference = padding_difference,
        .training_readbacks = training_stats.readbacks,
        .kernels = training_stats.kernels,
    };
}

fn forward(
    context: *nn.ExecutionContext,
    embedding: *const nn.Transformer.Embedding,
    encoder: *const nn.Transformer.MaskedEncoderBlock,
    classifier: *const nn.Modules.Linear,
    indices: nn.Tensor,
    attention_mask: nn.Tensor,
    token_mask: nn.Tensor,
    mean_pool: nn.Tensor,
) !nn.Tensor {
    var embedded = try embedding.forward(context, indices);
    defer embedded.deinit();
    try embedded.reshape(&.{ batch_size, maximum_tokens, channels });
    var masked_embeddings = try context.multiply(embedded, token_mask);
    defer masked_embeddings.deinit();
    var encoded = try encoder.forward(context, masked_embeddings, attention_mask, token_mask);
    defer encoded.deinit();
    var pooled = try context.batchedMatmul(mean_pool, encoded, false, false);
    defer pooled.deinit();
    try pooled.reshape(&.{ batch_size, channels });
    return classifier.forward(context, pooled);
}

fn evaluate(
    allocator: std.mem.Allocator,
    context: *nn.ExecutionContext,
    embedding: *const nn.Transformer.Embedding,
    encoder: *const nn.Transformer.MaskedEncoderBlock,
    classifier: *const nn.Modules.Linear,
    indices: nn.Tensor,
    attention_mask: nn.Tensor,
    token_mask: nn.Tensor,
    mean_pool: nn.Tensor,
) !Metrics {
    try context.beginBatch();
    var batch_active = true;
    errdefer if (batch_active) context.endBatch() catch {};
    var logits = try forward(context, embedding, encoder, classifier, indices, attention_mask, token_mask, mean_pool);
    defer logits.deinit();
    var probabilities = try context.softmax(logits);
    defer probabilities.deinit();
    try context.endBatch();
    batch_active = false;
    var values: [batch_size * 2]f32 = undefined;
    try context.readback(probabilities, &values);
    _ = allocator;
    var correct: usize = 0;
    var loss: f32 = 0;
    for (labels, 0..) |label, row| {
        const predicted: usize = if (values[row * 2 + 1] > values[row * 2]) 1 else 0;
        if (predicted == label) correct += 1;
        loss -= @log(@max(values[row * 2 + label], 1e-12));
    }
    return .{
        .loss = loss / @as(f32, @floatFromInt(batch_size)),
        .accuracy = @as(f32, @floatFromInt(correct)) / @as(f32, @floatFromInt(batch_size)),
        .probabilities = values,
    };
}

fn trainStep(
    context: *nn.ExecutionContext,
    embedding: *nn.Transformer.Embedding,
    encoder: *nn.Transformer.MaskedEncoderBlock,
    classifier: *nn.Modules.Linear,
    optimizer: *nn.Training.Optimizer,
    parameters: []const *nn.Tensor,
    indices: nn.Tensor,
    attention_mask: nn.Tensor,
    token_mask: nn.Tensor,
    mean_pool: nn.Tensor,
) !void {
    try context.beginBatch();
    var batch_active = true;
    errdefer if (batch_active) context.endBatch() catch {};
    var embedded = try embedding.forward(context, indices);
    defer embedded.deinit();
    try embedded.reshape(&.{ batch_size, maximum_tokens, channels });
    var masked_embeddings = try context.multiply(embedded, token_mask);
    defer masked_embeddings.deinit();
    var encoded = try encoder.forward(context, masked_embeddings, attention_mask, token_mask);
    defer encoded.deinit();
    var pooled = try context.batchedMatmul(mean_pool, encoded, false, false);
    defer pooled.deinit();
    try pooled.reshape(&.{ batch_size, channels });
    var logits = try classifier.forward(context, pooled);
    defer logits.deinit();
    var objective = try context.maskedSparseCrossEntropy(logits, &labels, &label_mask);
    defer objective.deinit();
    var head_gradients = try classifier.backward(context, pooled, objective.gradient);
    defer head_gradients.deinit();
    try head_gradients.input.reshape(&.{ batch_size, 1, channels });
    var pooled_gradient = try context.batchedMatmul(mean_pool, head_gradients.input, true, false);
    defer pooled_gradient.deinit();
    var encoder_gradients = try encoder.backward(context, masked_embeddings, attention_mask, token_mask, pooled_gradient);
    defer encoder_gradients.deinit();
    var embedding_output_gradient = encoder_gradients.input;
    embedding_output_gradient.shape = try nn.TensorShape.init(&.{ batch_size * maximum_tokens, channels });
    var embedding_gradient = try embedding.backward(context, indices, embedding_output_gradient);
    defer embedding_gradient.deinit();
    var encoder_gradient_storage: [16]nn.Tensor = undefined;
    const encoder_parameter_gradients = try encoder_gradients.parameterGradients(&encoder_gradient_storage);
    var gradient_storage: [parameter_count]nn.Tensor = undefined;
    gradient_storage[0] = embedding_gradient;
    @memcpy(gradient_storage[1..17], encoder_parameter_gradients);
    gradient_storage[17] = head_gradients.weights;
    gradient_storage[18] = head_gradients.bias;
    _ = try optimizer.step(context, parameters, &gradient_storage);
    try context.endBatch();
    batch_active = false;
}

fn token(value: Token) f32 {
    return @floatFromInt(@intFromEnum(value));
}

fn backendName(backend: nn.BackendType) []const u8 {
    return switch (backend) {
        .CPU => "CPU",
        .Metal => "Metal",
        .CUDA => "CUDA",
        .ROCm => "ROCm",
    };
}

test "masked Transformer classifies context and ignores padding tokens" {
    const result = try runExperiment(std.testing.allocator, .{ .backend = .cpu, .steps = 50, .seed = 42 });
    try std.testing.expectEqual(@as(usize, 0), result.training_readbacks);
    try std.testing.expect(result.final.loss < result.initial.loss * 0.4);
    try std.testing.expect(result.final.accuracy >= 0.9);
    try std.testing.expect(result.padding_difference < 1e-6);
}
