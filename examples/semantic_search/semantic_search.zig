const std = @import("std");
const nn = @import("nn");

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
    const result = try runExperiment(init.gpa, options);
    std.debug.print("Contrastive semantic-search lesson on {s}\n", .{backendName(result.backend)});
    std.debug.print("dual encoders map different query/document vocabularies into one space\n", .{});
    std.debug.print(
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
    std.debug.print("query: \"{s}\"\n", .{result.query});
    for (result.neighbors, 1..) |neighbor, rank| {
        std.debug.print("  {d}. {s: <20} cosine {d:.3}\n", .{ rank, documents[neighbor.index], neighbor.score });
    }
    std.debug.print("lesson: in-batch negatives teach paired texts to outrank every mismatch\n", .{});
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
        \\Usage: semantic_search [options]
        \\
        \\Options:
        \\  --backend <name>  cpu, auto, metal, cuda, or rocm (default: auto)
        \\  --steps <count>   full-batch contrastive updates (default: 40)
        \\  --seed <number>   model seed (default: 42)
        \\  --help            show this help
        \\
    , .{});
}

fn runExperiment(allocator: std.mem.Allocator, options: Options) !Result {
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
    context.resetStats();
    for (0..options.steps) |_| {
        try trainStep(&context, &query_encoder, &document_encoder, optimizer, training_query_tensor, document_tensor);
    }
    const training_stats = context.stats;
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
    return .{
        .backend = device.backendType(),
        .initial = initial,
        .final = final,
        .query = held_out_queries[2],
        .neighbors = neighbors,
        .training_kernels = training_stats.kernels,
        .training_readbacks = training_stats.readbacks,
    };
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

fn backendName(backend: nn.BackendType) []const u8 {
    return switch (backend) {
        .CPU => "CPU",
        .Metal => "Metal",
        .CUDA => "CUDA",
        .ROCm => "ROCm",
    };
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
