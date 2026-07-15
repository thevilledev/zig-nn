const std = @import("std");
const nn = @import("nn");

const dimensions: usize = 10;
const default_steps: usize = 180;
const negative_batch_count: usize = 4;

const Word = enum(usize) {
    king,
    queen,
    man,
    woman,
    prince,
    princess,
    boy,
    girl,
    royal,
    crown,
    paris,
    france,
    berlin,
    germany,
    rome,
    italy,
    capital,
    europe,
    cat,
    kitten,
    dog,
    puppy,
    pet,
    animal,
};

const vocabulary = [_][]const u8{
    "king",   "queen",   "man",   "woman", "prince",  "princess",
    "boy",    "girl",    "royal", "crown", "paris",   "france",
    "berlin", "germany", "rome",  "italy", "capital", "europe",
    "cat",    "kitten",  "dog",   "puppy", "pet",     "animal",
};

const sentences = [_][]const usize{
    &.{ word(.king), word(.man), word(.royal), word(.crown) },
    &.{ word(.queen), word(.woman), word(.royal), word(.crown) },
    &.{ word(.prince), word(.boy), word(.royal), word(.crown) },
    &.{ word(.princess), word(.girl), word(.royal), word(.crown) },
    &.{ word(.king), word(.queen), word(.royal), word(.crown) },
    &.{ word(.prince), word(.princess), word(.royal), word(.crown) },
    &.{ word(.paris), word(.france), word(.capital), word(.europe) },
    &.{ word(.berlin), word(.germany), word(.capital), word(.europe) },
    &.{ word(.rome), word(.italy), word(.capital), word(.europe) },
    &.{ word(.cat), word(.kitten), word(.pet), word(.animal) },
    &.{ word(.dog), word(.puppy), word(.pet), word(.animal) },
    &.{ word(.cat), word(.dog), word(.pet), word(.animal) },
    &.{ word(.kitten), word(.puppy), word(.pet), word(.animal) },
};

const Options = struct {
    backend: nn.DevicePreference = .auto,
    steps: usize = default_steps,
    seed: u64 = 42,
};

const Result = struct {
    backend: nn.BackendType,
    initial_related: f32,
    initial_unrelated: f32,
    final_related: f32,
    final_unrelated: f32,
    nearest_to_king: usize,
    nearest_similarity: f32,
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
    std.debug.print("Word2Vec skip-gram lesson on {s}\n", .{backendName(result.backend)});
    std.debug.print("objective: distinguish real context words from unigram negatives\n", .{});
    std.debug.print(
        "mean related cosine:   {d:.3} -> {d:.3}\n" ++
            "mean unrelated cosine: {d:.3} -> {d:.3}\n" ++
            "nearest to king: {s} ({d:.3})\n" ++
            "kernels: {d}, training readbacks: {d}\n",
        .{
            result.initial_related,
            result.final_related,
            result.initial_unrelated,
            result.final_unrelated,
            vocabulary[result.nearest_to_king],
            result.nearest_similarity,
            result.kernels,
            result.training_readbacks,
        },
    );
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
        \\Usage: word2vec [options]
        \\
        \\Options:
        \\  --backend <name>  cpu, auto, metal, cuda, or rocm
        \\  --steps <count>   full-batch updates (default: 180)
        \\  --seed <number>   initialization and sampling seed
        \\  --help            show this help
        \\
    , .{});
}

fn runExperiment(allocator: std.mem.Allocator, options: Options) !Result {
    var device = try nn.Device.init(allocator, options.backend);
    defer device.deinit();
    var context = nn.ExecutionContext.init(&device);
    var prng = std.Random.DefaultPrng.init(options.seed);
    const random = prng.random();
    var model = try nn.Embeddings.SkipGram.init(&context, vocabulary.len, dimensions, random);
    defer model.deinit();
    var parameters = model.parameters();
    var optimizer = try nn.Training.Optimizer.init(&context, .{
        .kind = .adamw,
        .learning_rate = 0.035,
        .weight_decay = 0.0002,
        .max_gradient_norm = 4,
    }, &parameters);
    defer optimizer.deinit();

    var counts = [_]usize{1} ** vocabulary.len;
    var pair_list: std.ArrayList(nn.Embeddings.SkipGramPair) = .empty;
    defer pair_list.deinit(allocator);
    for (sentences) |sentence| {
        for (sentence) |token| counts[token] += 1;
        const pairs = try nn.Embeddings.generateSkipGramPairs(allocator, sentence, 2);
        defer allocator.free(pairs);
        try pair_list.appendSlice(allocator, pairs);
    }
    var sampler = try nn.Embeddings.NegativeSampler.init(allocator, &counts, 0.75);
    defer sampler.deinit();
    var batches: [negative_batch_count]nn.Embeddings.NegativeSamplingBatch = undefined;
    var initialized_batches: usize = 0;
    defer for (batches[0..initialized_batches]) |*batch| batch.deinit();
    for (&batches) |*batch| {
        batch.* = try nn.Embeddings.NegativeSamplingBatch.init(allocator, pair_list.items, sampler, 3, random);
        initialized_batches += 1;
    }

    const initial_embeddings = try allocator.alloc(f32, vocabulary.len * dimensions);
    defer allocator.free(initial_embeddings);
    try context.readback(model.input.weights, initial_embeddings);
    const initial_related = try meanRelated(initial_embeddings);
    const initial_unrelated = try meanUnrelated(initial_embeddings);

    context.resetStats();
    for (0..options.steps) |step| {
        _ = try model.trainBatch(&context, &optimizer, batches[step % batches.len]);
    }
    const training_stats = context.stats;
    const final_embeddings = try allocator.alloc(f32, vocabulary.len * dimensions);
    defer allocator.free(final_embeddings);
    try context.readback(model.input.weights, final_embeddings);
    const nearest = try nearestNeighbor(final_embeddings, word(.king));
    return .{
        .backend = device.backendType(),
        .initial_related = initial_related,
        .initial_unrelated = initial_unrelated,
        .final_related = try meanRelated(final_embeddings),
        .final_unrelated = try meanUnrelated(final_embeddings),
        .nearest_to_king = nearest.index,
        .nearest_similarity = nearest.similarity,
        .training_readbacks = training_stats.readbacks,
        .kernels = training_stats.kernels,
    };
}

fn meanRelated(embeddings: []const f32) !f32 {
    return meanPairs(embeddings, &.{
        .{ word(.king), word(.queen) },
        .{ word(.prince), word(.princess) },
        .{ word(.paris), word(.berlin) },
        .{ word(.cat), word(.dog) },
    });
}

fn meanUnrelated(embeddings: []const f32) !f32 {
    return meanPairs(embeddings, &.{
        .{ word(.king), word(.berlin) },
        .{ word(.queen), word(.puppy) },
        .{ word(.paris), word(.cat) },
        .{ word(.crown), word(.animal) },
    });
}

fn meanPairs(embeddings: []const f32, pairs: []const [2]usize) !f32 {
    var total: f32 = 0;
    for (pairs) |pair| total += try nn.Embeddings.cosineSimilarity(vector(embeddings, pair[0]), vector(embeddings, pair[1]));
    return total / @as(f32, @floatFromInt(pairs.len));
}

fn nearestNeighbor(embeddings: []const f32, query: usize) !struct { index: usize, similarity: f32 } {
    var best_index: usize = 0;
    var best_similarity = -std.math.inf(f32);
    for (0..vocabulary.len) |candidate| {
        if (candidate == query) continue;
        const similarity = try nn.Embeddings.cosineSimilarity(vector(embeddings, query), vector(embeddings, candidate));
        if (similarity > best_similarity) {
            best_index = candidate;
            best_similarity = similarity;
        }
    }
    return .{ .index = best_index, .similarity = best_similarity };
}

fn vector(embeddings: []const f32, token: usize) []const f32 {
    return embeddings[token * dimensions .. (token + 1) * dimensions];
}

fn word(value: Word) usize {
    return @intFromEnum(value);
}

fn backendName(backend: nn.BackendType) []const u8 {
    return switch (backend) {
        .CPU => "CPU",
        .Metal => "Metal",
        .CUDA => "CUDA",
        .ROCm => "ROCm",
    };
}

test "skip-gram learns corpus neighborhoods without training readbacks" {
    const result = try runExperiment(std.testing.allocator, .{ .backend = .cpu, .steps = 140, .seed = 42 });
    try std.testing.expectEqual(@as(usize, 0), result.training_readbacks);
    try std.testing.expect(result.kernels > 0);
    try std.testing.expect(result.final_related > result.final_unrelated + 0.15);
    try std.testing.expect(result.final_related > result.initial_related);
}
