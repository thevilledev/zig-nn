const std = @import("std");
const builtin = @import("builtin");
const nn = @import("nn");
const events = @import("experiment_events");

const pair_count: usize = 6;
const embedding_features: usize = 6;
const default_steps: usize = 40;

const query_vocabulary = [_][]const u8{ "feline", "puppy", "railway", "flight", "noodles", "loaf" };
const document_vocabulary = [_][]const u8{
    "cat",      "sleeps", "sofa",  "dog",    "plays", "park",  "train", "station",
    "airplane", "sky",    "pasta", "tomato", "sauce", "bread", "oven",
};
const training_queries = [_][]const u8{ "feline", "puppy", "railway", "flight", "noodles", "loaf" };
const held_out_queries = [_][]const u8{
    "feline furniture", "puppy outdoors", "railway timetable",
    "flight clouds",    "noodles dinner", "loaf breakfast",
};
const documents = [_][]const u8{
    "cat sleeps sofa",    "dog plays park", "train station", "airplane sky",
    "pasta tomato sauce", "bread oven",
};

const Options = struct {
    backend: nn.DevicePreference = .auto,
    format: events.Format = .human,
    steps: usize = default_steps,
    seed: u64 = 42,
};

const Metrics = struct {
    loss: f32,
    recall_at_one: f32,
    mean_reciprocal_rank: f32,
};

const Result = struct {
    backend: nn.BackendType,
    initial: Metrics,
    final: Metrics,
    query: []const u8,
    neighbors: [3]nn.Retrieval.Neighbor,
    training_kernels: usize,
    training_readbacks: usize,
};

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const options = parseArgs(args[1..]) catch |err| {
        printUsage();
        if (err == error.HelpRequested) return;
        return err;
    };

    var stdout_buffer: [16 * 1024]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writerStreaming(std.Options.debug_io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};

    const result = try runExperimentStreaming(
        init.gpa,
        options,
        if (options.format == .ndjson) stdout else null,
    );
    if (options.format == .human) try printHuman(stdout, result);
}

fn parseArgs(args: []const []const u8) !Options {
    var options: Options = .{};
    var index: usize = 0;
    while (index < args.len) : (index += 1) {
        if (std.mem.eql(u8, args[index], "--backend")) {
            index += 1;
            if (index >= args.len) return error.MissingArgument;
            options.backend = try parseBackend(args[index]);
        } else if (std.mem.eql(u8, args[index], "--format")) {
            index += 1;
            if (index >= args.len) return error.MissingArgument;
            options.format = try events.parseFormat(args[index]);
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
        \\Usage: semantic_search [options]
        \\
        \\Options:
        \\  --backend <name>  cpu, auto, metal, cuda, or rocm (default: auto)
        \\  --format <name>   human or ndjson (default: human)
        \\  --steps <count>   full-batch contrastive updates (default: 40)
        \\  --seed <number>   model seed (default: 42)
        \\  --help            show this help
        \\
    , .{});
}

fn runExperiment(allocator: std.mem.Allocator, options: Options) !Result {
    return runExperimentStreaming(allocator, options, null);
}

fn runExperimentStreaming(
    allocator: std.mem.Allocator,
    options: Options,
    writer: ?*std.Io.Writer,
) !Result {
    var device = try nn.Device.init(allocator, options.backend);
    defer device.deinit();
    var context = nn.ExecutionContext.init(&device);
    const training_query_values = try bagOfWords(allocator, &training_queries, &query_vocabulary);
    defer allocator.free(training_query_values);
    const held_out_query_values = try bagOfWords(allocator, &held_out_queries, &query_vocabulary);
    defer allocator.free(held_out_query_values);
    const document_values = try bagOfWords(allocator, &documents, &document_vocabulary);
    defer allocator.free(document_values);
    var training_query_tensor = try context.upload(&.{ pair_count, query_vocabulary.len }, training_query_values);
    defer training_query_tensor.deinit();
    var held_out_query_tensor = try context.upload(&.{ pair_count, query_vocabulary.len }, held_out_query_values);
    defer held_out_query_tensor.deinit();
    var document_tensor = try context.upload(&.{ pair_count, document_vocabulary.len }, document_values);
    defer document_tensor.deinit();

    var prng = std.Random.DefaultPrng.init(options.seed);
    var query_encoder = try nn.Modules.Linear.init(&context, query_vocabulary.len, embedding_features, prng.random());
    defer query_encoder.deinit();
    var document_encoder = try nn.Modules.Linear.init(&context, document_vocabulary.len, embedding_features, prng.random());
    defer document_encoder.deinit();
    const optimizer = try nn.Transformer.Sgd.init(0.12);

    const initial = try evaluate(
        allocator,
        &context,
        &query_encoder,
        &document_encoder,
        training_query_tensor,
        held_out_query_tensor,
        document_tensor,
    );
    if (writer) |output| {
        try events.emit(output, .{
            .v = 1,
            .type = "run_started",
            .experiment = "semantic-search",
            .data = .{
                .config = .{ .steps = options.steps, .seed = options.seed },
                .topology = &[_]usize{ query_vocabulary.len, embedding_features, document_vocabulary.len },
                .activations = &[_][]const u8{ "linear", "cosine" },
                .queries = &held_out_queries,
                .documents = &documents,
                .execution = .{
                    .requested_backend = @tagName(options.backend),
                    .selected_backend = backendName(device.backendType()),
                    .optimize = @tagName(builtin.mode),
                },
            },
        });
        try emitReport(
            allocator,
            output,
            &context,
            &query_encoder,
            &document_encoder,
            training_query_tensor,
            held_out_query_tensor,
            document_tensor,
            0,
            options.steps,
            .{},
            .{},
        );
    }

    context.resetStats();
    var training_kernels: usize = 0;
    var training_readbacks: usize = 0;
    for (0..options.steps) |step_index| {
        try trainStep(&context, &query_encoder, &document_encoder, optimizer, training_query_tensor, document_tensor);
        if (writer) |output| {
            if (events.shouldEmitSnapshot(step_index, options.steps)) {
                const execution_stats = context.stats;
                const backend_stats = context.backendStats();
                training_kernels += execution_stats.kernels;
                training_readbacks += execution_stats.readbacks;
                try emitReport(
                    allocator,
                    output,
                    &context,
                    &query_encoder,
                    &document_encoder,
                    training_query_tensor,
                    held_out_query_tensor,
                    document_tensor,
                    step_index + 1,
                    options.steps,
                    execution_stats,
                    backend_stats,
                );
                context.resetStats();
            }
        }
    }
    if (writer == null) {
        training_kernels = context.stats.kernels;
        training_readbacks = context.stats.readbacks;
    }

    const final = try evaluate(
        allocator,
        &context,
        &query_encoder,
        &document_encoder,
        training_query_tensor,
        held_out_query_tensor,
        document_tensor,
    );
    const neighbors = try sampleRanking(
        allocator,
        &context,
        &query_encoder,
        &document_encoder,
        held_out_query_tensor,
        document_tensor,
        2,
    );
    if (writer) |output| {
        try events.emit(output, .{
            .v = 1,
            .type = "run_completed",
            .experiment = "semantic-search",
            .step = options.steps,
            .total_steps = options.steps,
            .data = .{
                .initial = initial,
                .final = final,
                .training_kernels = training_kernels,
                .training_readbacks = training_readbacks,
            },
        });
    }
    return .{
        .backend = device.backendType(),
        .initial = initial,
        .final = final,
        .query = held_out_queries[2],
        .neighbors = neighbors,
        .training_kernels = training_kernels,
        .training_readbacks = training_readbacks,
    };
}

fn emitReport(
    allocator: std.mem.Allocator,
    writer: *std.Io.Writer,
    context: *nn.ExecutionContext,
    query_encoder: *const nn.Modules.Linear,
    document_encoder: *const nn.Modules.Linear,
    training_queries_tensor: nn.Tensor,
    held_out_queries_tensor: nn.Tensor,
    documents_tensor: nn.Tensor,
    step: usize,
    total: usize,
    execution_stats: nn.ExecutionStats,
    backend_stats: nn.BackendRuntimeStats,
) !void {
    const metrics = try evaluate(
        allocator,
        context,
        query_encoder,
        document_encoder,
        training_queries_tensor,
        held_out_queries_tensor,
        documents_tensor,
    );
    inline for (.{
        .{ "loss", metrics.loss },
        .{ "recall_at_one", metrics.recall_at_one },
        .{ "mean_reciprocal_rank", metrics.mean_reciprocal_rank },
    }) |metric| {
        try events.emit(writer, .{
            .v = 1,
            .type = "metric",
            .experiment = "semantic-search",
            .step = step,
            .total_steps = total,
            .data = .{ .name = metric[0], .value = metric[1] },
        });
    }
    const similarities = try similarityMatrix(
        allocator,
        context,
        query_encoder,
        document_encoder,
        held_out_queries_tensor,
        documents_tensor,
    );
    try events.emit(writer, .{
        .v = 1,
        .type = "snapshot",
        .experiment = "semantic-search",
        .step = step,
        .total_steps = total,
        .data = .{
            .kind = "semantic_similarity",
            .similarities = similarities,
            .rows = pair_count,
            .columns = pair_count,
            .telemetry = .{ .execution = execution_stats, .backend = backend_stats },
        },
    });
}

fn trainStep(
    context: *nn.ExecutionContext,
    query_encoder: *nn.Modules.Linear,
    document_encoder: *nn.Modules.Linear,
    optimizer: nn.Transformer.Sgd,
    query_inputs: nn.Tensor,
    document_inputs: nn.Tensor,
) !void {
    try context.beginBatch();
    var batch_active = true;
    errdefer if (batch_active) context.endBatch() catch {};
    var query_embeddings = try query_encoder.forward(context, query_inputs);
    defer query_embeddings.deinit();
    var document_embeddings = try document_encoder.forward(context, document_inputs);
    defer document_embeddings.deinit();
    var objective = try nn.Retrieval.SymmetricInfoNCE.init(context, query_embeddings, document_embeddings, 0.2);
    defer objective.deinit();
    var query_gradients = try query_encoder.backward(context, query_inputs, objective.left_gradient);
    defer query_gradients.deinit();
    var document_gradients = try document_encoder.backward(context, document_inputs, objective.right_gradient);
    defer document_gradients.deinit();
    try optimizer.stepLinear(context, query_encoder, query_gradients);
    try optimizer.stepLinear(context, document_encoder, document_gradients);
    try context.endBatch();
    batch_active = false;
}

fn evaluate(
    allocator: std.mem.Allocator,
    context: *nn.ExecutionContext,
    query_encoder: *const nn.Modules.Linear,
    document_encoder: *const nn.Modules.Linear,
    training_queries_tensor: nn.Tensor,
    held_out_queries_tensor: nn.Tensor,
    documents_tensor: nn.Tensor,
) !Metrics {
    try context.beginBatch();
    var training_query_embeddings = try query_encoder.forward(context, training_queries_tensor);
    defer training_query_embeddings.deinit();
    var held_out_query_embeddings = try query_encoder.forward(context, held_out_queries_tensor);
    defer held_out_query_embeddings.deinit();
    var document_embeddings = try document_encoder.forward(context, documents_tensor);
    defer document_embeddings.deinit();
    var objective = try nn.Retrieval.SymmetricInfoNCE.init(context, training_query_embeddings, document_embeddings, 0.2);
    defer objective.deinit();
    try context.endBatch();
    const loss = try objective.loss(context);
    const query_values = try allocator.alloc(f32, pair_count * embedding_features);
    defer allocator.free(query_values);
    const document_values = try allocator.alloc(f32, pair_count * embedding_features);
    defer allocator.free(document_values);
    try context.readback(held_out_query_embeddings, query_values);
    try context.readback(document_embeddings, document_values);
    var correct: usize = 0;
    var reciprocal_rank: f32 = 0;
    for (0..pair_count) |query_index| {
        const neighbors = try nn.Retrieval.topKCosine(
            allocator,
            query_values[query_index * embedding_features ..][0..embedding_features],
            document_values,
            embedding_features,
            pair_count,
        );
        defer allocator.free(neighbors);
        if (neighbors[0].index == query_index) correct += 1;
        for (neighbors, 1..) |neighbor, rank| {
            if (neighbor.index == query_index) {
                reciprocal_rank += 1.0 / @as(f32, @floatFromInt(rank));
                break;
            }
        }
    }
    return .{
        .loss = loss,
        .recall_at_one = @as(f32, @floatFromInt(correct)) / pair_count,
        .mean_reciprocal_rank = reciprocal_rank / pair_count,
    };
}

fn similarityMatrix(
    allocator: std.mem.Allocator,
    context: *nn.ExecutionContext,
    query_encoder: *const nn.Modules.Linear,
    document_encoder: *const nn.Modules.Linear,
    queries: nn.Tensor,
    document_inputs: nn.Tensor,
) ![pair_count * pair_count]f32 {
    try context.beginBatch();
    var query_embeddings = try query_encoder.forward(context, queries);
    defer query_embeddings.deinit();
    var document_embeddings = try document_encoder.forward(context, document_inputs);
    defer document_embeddings.deinit();
    try context.endBatch();
    const query_values = try allocator.alloc(f32, pair_count * embedding_features);
    defer allocator.free(query_values);
    const document_values = try allocator.alloc(f32, pair_count * embedding_features);
    defer allocator.free(document_values);
    try context.readback(query_embeddings, query_values);
    try context.readback(document_embeddings, document_values);

    var similarities: [pair_count * pair_count]f32 = undefined;
    for (0..pair_count) |row| {
        for (0..pair_count) |column| {
            similarities[row * pair_count + column] = cosine(
                query_values[row * embedding_features ..][0..embedding_features],
                document_values[column * embedding_features ..][0..embedding_features],
            );
        }
    }
    return similarities;
}

fn cosine(left: []const f32, right: []const f32) f32 {
    var dot: f32 = 0;
    var left_squared: f32 = 0;
    var right_squared: f32 = 0;
    for (left, right) |left_value, right_value| {
        dot += left_value * right_value;
        left_squared += left_value * left_value;
        right_squared += right_value * right_value;
    }
    return dot / (@sqrt(left_squared) * @sqrt(right_squared));
}

fn sampleRanking(
    allocator: std.mem.Allocator,
    context: *nn.ExecutionContext,
    query_encoder: *const nn.Modules.Linear,
    document_encoder: *const nn.Modules.Linear,
    queries: nn.Tensor,
    document_inputs: nn.Tensor,
    query_index: usize,
) ![3]nn.Retrieval.Neighbor {
    try context.beginBatch();
    var query_embeddings = try query_encoder.forward(context, queries);
    defer query_embeddings.deinit();
    var document_embeddings = try document_encoder.forward(context, document_inputs);
    defer document_embeddings.deinit();
    try context.endBatch();
    const query_values = try allocator.alloc(f32, pair_count * embedding_features);
    defer allocator.free(query_values);
    const document_values = try allocator.alloc(f32, pair_count * embedding_features);
    defer allocator.free(document_values);
    try context.readback(query_embeddings, query_values);
    try context.readback(document_embeddings, document_values);
    const ranked = try nn.Retrieval.topKCosine(
        allocator,
        query_values[query_index * embedding_features ..][0..embedding_features],
        document_values,
        embedding_features,
        3,
    );
    defer allocator.free(ranked);
    return .{ ranked[0], ranked[1], ranked[2] };
}

fn bagOfWords(
    allocator: std.mem.Allocator,
    phrases: []const []const u8,
    vocabulary: []const []const u8,
) ![]f32 {
    const values = try allocator.alloc(f32, phrases.len * vocabulary.len);
    @memset(values, 0);
    for (phrases, 0..) |phrase, row| {
        var words = std.mem.tokenizeScalar(u8, phrase, ' ');
        while (words.next()) |word| {
            for (vocabulary, 0..) |entry, column| {
                if (std.mem.eql(u8, word, entry)) values[row * vocabulary.len + column] += 1;
            }
        }
    }
    return values;
}

fn printHuman(writer: *std.Io.Writer, result: Result) !void {
    try writer.print("Contrastive semantic-search lesson on {s}\n", .{backendName(result.backend)});
    try writer.print("dual encoders map different query/document vocabularies into one space\n", .{});
    try writer.print(
        "InfoNCE loss: {d:.4} -> {d:.4}; recall@1: {d:.1}% -> {d:.1}%; final MRR: {d:.3}\n" ++
            "training kernels: {d}, training readbacks: {d}\n\n",
        .{
            result.initial.loss,
            result.final.loss,
            result.initial.recall_at_one * 100,
            result.final.recall_at_one * 100,
            result.final.mean_reciprocal_rank,
            result.training_kernels,
            result.training_readbacks,
        },
    );
    try writer.print("query: \"{s}\"\n", .{result.query});
    for (result.neighbors, 1..) |neighbor, rank| {
        try writer.print("  {d}. {s: <20} cosine {d:.3}\n", .{ rank, documents[neighbor.index], neighbor.score });
    }
    try writer.print("lesson: in-batch negatives teach paired texts to outrank every mismatch\n", .{});
}

fn backendName(backend: nn.BackendType) []const u8 {
    return switch (backend) {
        .CPU => "cpu",
        .Metal => "metal",
        .CUDA => "cuda",
        .ROCm => "rocm",
    };
}

test "semantic search parses structured output" {
    const options = try parseArgs(&.{ "--backend", "cpu", "--format", "ndjson", "--steps", "12", "--seed", "7" });
    try std.testing.expectEqual(nn.DevicePreference.cpu, options.backend);
    try std.testing.expectEqual(events.Format.ndjson, options.format);
    try std.testing.expectEqual(@as(usize, 12), options.steps);
}

test "contrastive dual encoders retrieve held-out query wording" {
    const result = try runExperiment(std.testing.allocator, .{ .backend = .cpu, .steps = default_steps, .seed = 42 });
    try std.testing.expectEqual(@as(usize, 0), result.training_readbacks);
    try std.testing.expect(result.training_kernels > 0);
    try std.testing.expect(result.final.loss < result.initial.loss * 0.1);
    try std.testing.expectEqual(@as(f32, 1), result.final.recall_at_one);
    try std.testing.expectEqual(@as(f32, 1), result.final.mean_reciprocal_rank);
    try std.testing.expectEqual(@as(usize, 2), result.neighbors[0].index);
}
