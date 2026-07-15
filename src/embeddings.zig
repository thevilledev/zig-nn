const std = @import("std");
const tensor = @import("tensor.zig");
const training = @import("training.zig");
const transformer = @import("transformer.zig");

const ExecutionContext = tensor.ExecutionContext;
const Tensor = tensor.Tensor;

pub const SkipGramPair = struct {
    center: usize,
    context: usize,
};

/// Emits both directions of every token pair inside a symmetric context
/// window. Sentence boundaries should be handled by calling this once per
/// sentence.
pub fn generateSkipGramPairs(allocator: std.mem.Allocator, tokens: []const usize, window: usize) ![]SkipGramPair {
    if (window == 0) return error.InvalidWindow;
    var pairs: std.ArrayList(SkipGramPair) = .empty;
    defer pairs.deinit(allocator);
    for (tokens, 0..) |center, center_index| {
        const start = center_index -| window;
        const end = @min(tokens.len, center_index +| window +| 1);
        for (tokens[start..end], start..) |context, context_index| {
            if (context_index == center_index) continue;
            try pairs.append(allocator, .{ .center = center, .context = context });
        }
    }
    return pairs.toOwnedSlice(allocator);
}

/// Samples vocabulary entries from the Word2Vec `count^power` distribution.
pub const NegativeSampler = struct {
    allocator: std.mem.Allocator,
    cumulative: []f64,

    pub fn init(allocator: std.mem.Allocator, counts: []const usize, power: f64) !NegativeSampler {
        if (counts.len < 2 or !std.math.isFinite(power) or power <= 0) return error.InvalidDistribution;
        const cumulative = try allocator.alloc(f64, counts.len);
        errdefer allocator.free(cumulative);
        var total: f64 = 0;
        for (counts, cumulative) |count, *entry| {
            if (count == 0) return error.InvalidDistribution;
            const weight = std.math.pow(f64, @floatFromInt(count), power);
            if (!std.math.isFinite(weight) or weight <= 0) return error.InvalidDistribution;
            total += weight;
            if (!std.math.isFinite(total)) return error.InvalidDistribution;
            entry.* = total;
        }
        var previous: f64 = 0;
        for (cumulative) |*entry| {
            entry.* /= total;
            if (!std.math.isFinite(entry.*) or entry.* <= previous) return error.InvalidDistribution;
            previous = entry.*;
        }
        cumulative[cumulative.len - 1] = 1;
        return .{ .allocator = allocator, .cumulative = cumulative };
    }

    pub fn deinit(self: *NegativeSampler) void {
        self.allocator.free(self.cumulative);
        self.* = undefined;
    }

    pub fn sample(self: NegativeSampler, random: std.Random, excluded: usize) !usize {
        if (excluded >= self.cumulative.len) return error.InvalidToken;
        const excluded_start = if (excluded == 0) 0 else self.cumulative[excluded - 1];
        const excluded_end = self.cumulative[excluded];
        const remaining_mass = 1 - (excluded_end - excluded_start);
        if (remaining_mass <= 0) return error.InvalidDistribution;

        var draw = random.float(f64) * remaining_mass;
        if (draw >= excluded_start) draw = excluded_end + (draw - excluded_start);
        draw = @min(draw, std.math.nextAfter(f64, 1, 0));
        const candidate = std.sort.upperBound(f64, self.cumulative, draw, struct {
            fn compare(value: f64, entry: f64) std.math.Order {
                return std.math.order(value, entry);
            }
        }.compare);
        if (candidate >= self.cumulative.len or candidate == excluded) return error.InvalidDistribution;
        return candidate;
    }
};

/// Host-side sparse batch description. Uploading it is an explicit boundary;
/// all forward, backward, and optimizer work after that stays on the device.
pub const NegativeSamplingBatch = struct {
    allocator: std.mem.Allocator,
    centers: []f32,
    contexts: []f32,
    labels: []f32,

    pub fn init(
        allocator: std.mem.Allocator,
        pairs: []const SkipGramPair,
        sampler: NegativeSampler,
        negatives_per_positive: usize,
        random: std.Random,
    ) !NegativeSamplingBatch {
        if (pairs.len == 0 or negatives_per_positive == 0) return error.EmptyBatch;
        const rows_per_pair = std.math.add(usize, negatives_per_positive, 1) catch return error.DimensionOverflow;
        const row_count = std.math.mul(usize, pairs.len, rows_per_pair) catch return error.DimensionOverflow;
        const centers = try allocator.alloc(f32, row_count);
        errdefer allocator.free(centers);
        const contexts = try allocator.alloc(f32, row_count);
        errdefer allocator.free(contexts);
        const labels = try allocator.alloc(f32, row_count);
        errdefer allocator.free(labels);
        var row: usize = 0;
        for (pairs) |pair| {
            if (pair.center >= sampler.cumulative.len or pair.context >= sampler.cumulative.len) return error.InvalidToken;
            centers[row] = @floatFromInt(pair.center);
            contexts[row] = @floatFromInt(pair.context);
            labels[row] = 1;
            row += 1;
            for (0..negatives_per_positive) |_| {
                centers[row] = @floatFromInt(pair.center);
                contexts[row] = @floatFromInt(try sampler.sample(random, pair.context));
                labels[row] = 0;
                row += 1;
            }
        }
        return .{ .allocator = allocator, .centers = centers, .contexts = contexts, .labels = labels };
    }

    pub fn deinit(self: *NegativeSamplingBatch) void {
        self.allocator.free(self.centers);
        self.allocator.free(self.contexts);
        self.allocator.free(self.labels);
        self.* = undefined;
    }

    pub fn rowCount(self: NegativeSamplingBatch) usize {
        return self.labels.len;
    }
};

/// Two-table skip-gram model trained with binary negative sampling.
pub const SkipGram = struct {
    input: transformer.Embedding,
    output: transformer.Embedding,

    pub fn init(context: *ExecutionContext, vocabulary_size: usize, dimensions: usize, random: std.Random) !SkipGram {
        var input = try transformer.Embedding.init(context, vocabulary_size, dimensions, random);
        errdefer input.deinit();
        const output = try transformer.Embedding.init(context, vocabulary_size, dimensions, random);
        return .{ .input = input, .output = output };
    }

    pub fn deinit(self: *SkipGram) void {
        self.input.deinit();
        self.output.deinit();
        self.* = undefined;
    }

    pub fn parameters(self: *SkipGram) [2]*Tensor {
        return .{ &self.input.weights, &self.output.weights };
    }

    /// Runs one full-batch update. The scalar loss is intentionally not read
    /// back; callers can inspect embedding geometry between training phases.
    pub fn trainBatch(
        self: *SkipGram,
        context: *ExecutionContext,
        optimizer: *training.Optimizer,
        batch: NegativeSamplingBatch,
    ) !training.StepResult {
        if (batch.rowCount() == 0 or batch.centers.len != batch.rowCount() or batch.contexts.len != batch.rowCount()) return error.DimensionMismatch;
        const rows = batch.rowCount();
        var center_indices = try context.upload(&.{ rows, 1 }, batch.centers);
        defer center_indices.deinit();
        var context_indices = try context.upload(&.{ rows, 1 }, batch.contexts);
        defer context_indices.deinit();
        var labels = try context.upload(&.{ rows, 1 }, batch.labels);
        defer labels.deinit();

        try context.beginBatch();
        var batch_active = true;
        errdefer if (batch_active) context.endBatch() catch {};
        var center_vectors = try self.input.forward(context, center_indices);
        defer center_vectors.deinit();
        var context_vectors = try self.output.forward(context, context_indices);
        defer context_vectors.deinit();
        var products = try context.multiply(center_vectors, context_vectors);
        defer products.deinit();
        var products_t = try context.transpose(products);
        defer products_t.deinit();
        var scores_t = try context.sumRows(products_t);
        defer scores_t.deinit();
        var scores = try context.transpose(scores_t);
        defer scores.deinit();
        var probabilities = try context.activate(scores, .sigmoid);
        defer probabilities.deinit();
        var difference = try context.subtract(probabilities, labels);
        defer difference.deinit();
        var score_gradient = try context.scale(difference, 1.0 / @as(f32, @floatFromInt(rows)));
        defer score_gradient.deinit();
        var ones = try context.createTensor(&.{ 1, center_vectors.shape.dims[1] });
        defer ones.deinit();
        ones.fill(1);
        var expanded_gradient = try context.matmul(score_gradient, ones);
        defer expanded_gradient.deinit();
        var center_vector_gradient = try context.multiply(context_vectors, expanded_gradient);
        defer center_vector_gradient.deinit();
        var context_vector_gradient = try context.multiply(center_vectors, expanded_gradient);
        defer context_vector_gradient.deinit();
        var input_gradient = try self.input.backward(context, center_indices, center_vector_gradient);
        defer input_gradient.deinit();
        var output_gradient = try self.output.backward(context, context_indices, context_vector_gradient);
        defer output_gradient.deinit();
        const model_parameters = self.parameters();
        const result = try optimizer.step(context, &model_parameters, &.{ input_gradient, output_gradient });
        try context.endBatch();
        batch_active = false;
        return result;
    }
};

pub fn cosineSimilarity(left: []const f32, right: []const f32) !f32 {
    if (left.len == 0 or left.len != right.len) return error.DimensionMismatch;
    var dot: f64 = 0;
    var left_square: f64 = 0;
    var right_square: f64 = 0;
    for (left, right) |a, b| {
        if (!std.math.isFinite(a) or !std.math.isFinite(b)) return error.NonFiniteInput;
        const left_value: f64 = a;
        const right_value: f64 = b;
        dot += left_value * right_value;
        left_square += left_value * left_value;
        right_square += right_value * right_value;
    }
    if (left_square == 0 or right_square == 0) return 0;
    const similarity = dot / @sqrt(left_square * right_square);
    return @floatCast(std.math.clamp(similarity, -1, 1));
}

test "skip-gram pairs respect sentence edges and window" {
    const pairs = try generateSkipGramPairs(std.testing.allocator, &.{ 1, 2, 3 }, 1);
    defer std.testing.allocator.free(pairs);
    try std.testing.expectEqualSlices(SkipGramPair, &.{
        .{ .center = 1, .context = 2 },
        .{ .center = 2, .context = 1 },
        .{ .center = 2, .context = 3 },
        .{ .center = 3, .context = 2 },
    }, pairs);
}

test "skip-gram pairs tolerate a saturating context window" {
    const pairs = try generateSkipGramPairs(std.testing.allocator, &.{ 1, 2, 3 }, std.math.maxInt(usize));
    defer std.testing.allocator.free(pairs);

    try std.testing.expectEqual(@as(usize, 6), pairs.len);
}

test "negative sampling batch labels positives and excludes their targets" {
    var sampler = try NegativeSampler.init(std.testing.allocator, &.{ 8, 4, 2 }, 0.75);
    defer sampler.deinit();
    var prng = std.Random.DefaultPrng.init(7);
    var batch = try NegativeSamplingBatch.init(std.testing.allocator, &.{.{ .center = 0, .context = 1 }}, sampler, 4, prng.random());
    defer batch.deinit();
    try std.testing.expectEqual(@as(usize, 5), batch.rowCount());
    try std.testing.expectEqual(@as(f32, 1), batch.labels[0]);
    try std.testing.expectEqual(@as(f32, 1), batch.contexts[0]);
    for (batch.labels[1..], batch.contexts[1..]) |label, context| {
        try std.testing.expectEqual(@as(f32, 0), label);
        try std.testing.expect(context != 1);
    }
}

test "negative sampling rejects non-finite distributions and overflowing batches" {
    try std.testing.expectError(
        error.InvalidDistribution,
        NegativeSampler.init(std.testing.allocator, &.{ 2, 3 }, std.math.floatMax(f64)),
    );
    try std.testing.expectError(
        error.InvalidDistribution,
        NegativeSampler.init(std.testing.allocator, &.{ std.math.maxInt(usize), 1 }, 1),
    );

    var sampler = try NegativeSampler.init(std.testing.allocator, &.{ 1, 1 }, 1);
    defer sampler.deinit();
    var prng = std.Random.DefaultPrng.init(1);
    try std.testing.expectError(
        error.DimensionOverflow,
        NegativeSamplingBatch.init(
            std.testing.allocator,
            &.{.{ .center = 0, .context = 1 }},
            sampler,
            std.math.maxInt(usize),
            prng.random(),
        ),
    );
}

test "cosine similarity stays finite for large vectors" {
    const maximum = std.math.floatMax(f32);
    try std.testing.expectApproxEqAbs(
        @as(f32, 1),
        try cosineSimilarity(&.{ maximum, maximum }, &.{ maximum, maximum }),
        1e-6,
    );
    try std.testing.expectError(
        error.NonFiniteInput,
        cosineSimilarity(&.{std.math.nan(f32)}, &.{1}),
    );
}

test "skip-gram update keeps the training path device resident" {
    var device = try tensor.Device.init(std.testing.allocator, .cpu);
    defer device.deinit();
    var context = ExecutionContext.init(&device);
    var prng = std.Random.DefaultPrng.init(42);
    var model = try SkipGram.init(&context, 4, 3, prng.random());
    defer model.deinit();
    var parameters = model.parameters();
    var optimizer = try training.Optimizer.init(&context, .{ .kind = .adamw, .learning_rate = 0.03 }, &parameters);
    defer optimizer.deinit();
    var sampler = try NegativeSampler.init(std.testing.allocator, &.{ 3, 3, 3, 3 }, 0.75);
    defer sampler.deinit();
    var batch = try NegativeSamplingBatch.init(std.testing.allocator, &.{
        .{ .center = 0, .context = 1 },
        .{ .center = 2, .context = 3 },
    }, sampler, 2, prng.random());
    defer batch.deinit();
    context.resetStats();
    const result = try model.trainBatch(&context, &optimizer, batch);
    try std.testing.expect(result.updated);
    try std.testing.expectEqual(@as(usize, 0), context.stats.readbacks);
}
