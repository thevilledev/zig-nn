const std = @import("std");
const tensor = @import("tensor.zig");

const Allocator = std.mem.Allocator;
const ExecutionContext = tensor.ExecutionContext;
const Tensor = tensor.Tensor;

/// Symmetric in-batch InfoNCE. Row `i` on each side treats row `i` on the
/// other side as its positive and every other row as a negative. Inputs can be
/// raw learned projections or pre-normalized embeddings.
pub const SymmetricInfoNCE = struct {
    left_to_right: tensor.MaskedSparseCrossEntropy,
    right_to_left: tensor.MaskedSparseCrossEntropy,
    left_gradient: Tensor,
    right_gradient: Tensor,

    pub fn init(
        context: *ExecutionContext,
        left: Tensor,
        right: Tensor,
        temperature: f32,
    ) !SymmetricInfoNCE {
        try validatePair(left, right);
        if (!std.math.isFinite(temperature) or temperature <= 0) return error.InvalidTemperature;
        const batch = left.shape.dims[0];
        const allocator = context.device.allocator;
        const targets = try allocator.alloc(usize, batch);
        defer allocator.free(targets);
        const mask = try allocator.alloc(f32, batch);
        defer allocator.free(mask);
        for (targets, mask, 0..) |*target, *weight, index| {
            target.* = index;
            weight.* = 1;
        }

        var logits = try scaledDotSimilarities(context, left, right, temperature);
        defer logits.deinit();
        var left_to_right = try context.maskedSparseCrossEntropy(logits, targets, mask);
        errdefer left_to_right.deinit();
        var transposed_logits = try context.transpose(logits);
        defer transposed_logits.deinit();
        var right_to_left = try context.maskedSparseCrossEntropy(transposed_logits, targets, mask);
        errdefer right_to_left.deinit();
        var right_gradient_transpose = try context.transpose(right_to_left.gradient);
        defer right_gradient_transpose.deinit();
        var bidirectional_gradient = try context.add(left_to_right.gradient, right_gradient_transpose);
        defer bidirectional_gradient.deinit();
        var score_gradient = try context.scale(bidirectional_gradient, 0.5 / temperature);
        defer score_gradient.deinit();
        var left_gradient = try context.matmul(score_gradient, right);
        errdefer left_gradient.deinit();
        var score_gradient_transpose = try context.transpose(score_gradient);
        defer score_gradient_transpose.deinit();
        const right_gradient = try context.matmul(score_gradient_transpose, left);
        return .{
            .left_to_right = left_to_right,
            .right_to_left = right_to_left,
            .left_gradient = left_gradient,
            .right_gradient = right_gradient,
        };
    }

    pub fn deinit(self: *SymmetricInfoNCE) void {
        self.left_to_right.deinit();
        self.right_to_left.deinit();
        self.left_gradient.deinit();
        self.right_gradient.deinit();
        self.* = undefined;
    }

    /// Explicit reporting boundary. Training gradients remain device-resident
    /// until this method is called.
    pub fn loss(self: SymmetricInfoNCE, context: *ExecutionContext) !f32 {
        const left_loss = try self.left_to_right.meanLoss(context);
        const right_loss = try self.right_to_left.meanLoss(context);
        return 0.5 * (left_loss + right_loss);
    }
};

/// Computes `left @ right^T / temperature` on the selected tensor backend.
pub fn scaledDotSimilarities(
    context: *ExecutionContext,
    left: Tensor,
    right: Tensor,
    temperature: f32,
) !Tensor {
    try validatePair(left, right);
    if (!std.math.isFinite(temperature) or temperature <= 0) return error.InvalidTemperature;
    var right_transpose = try context.transpose(right);
    defer right_transpose.deinit();
    var similarities = try context.matmul(left, right_transpose);
    defer similarities.deinit();
    return context.scale(similarities, 1.0 / temperature);
}

pub const Neighbor = struct {
    index: usize,
    score: f32,
};

pub const IvfSearch = struct {
    allocator: Allocator,
    neighbors: []Neighbor,
    candidates_scored: usize,

    pub fn deinit(self: *IvfSearch) void {
        self.allocator.free(self.neighbors);
        self.* = undefined;
    }
};

/// Inspectable inverted-file cosine index.
///
/// Documents are normalized once, assigned to deterministic spherical-k-means
/// centroids, and searched by probing the nearest lists. The index exposes the
/// candidate count so experiments can show the recall/latency tradeoff rather
/// than treating approximate search as a black box.
pub const IvfIndex = struct {
    allocator: Allocator,
    features: usize,
    document_count: usize,
    list_count: usize,
    documents: []f32,
    centroids: []f32,
    assignments: []usize,

    pub fn init(
        allocator: Allocator,
        documents: []const f32,
        features: usize,
        list_count: usize,
        iterations: usize,
    ) !IvfIndex {
        if (features == 0 or documents.len == 0 or documents.len % features != 0 or
            list_count == 0 or iterations == 0)
        {
            return error.InvalidIndexConfiguration;
        }
        const document_count = documents.len / features;
        if (list_count > document_count) return error.InvalidIndexConfiguration;
        const normalized_documents = try allocator.alloc(f32, documents.len);
        errdefer allocator.free(normalized_documents);
        for (0..document_count) |document_index| {
            try normalizeInto(
                documents[document_index * features ..][0..features],
                normalized_documents[document_index * features ..][0..features],
            );
        }
        const centroids = try allocator.alloc(f32, list_count * features);
        errdefer allocator.free(centroids);
        for (0..list_count) |list_index| {
            const seed_index = list_index * document_count / list_count;
            @memcpy(
                centroids[list_index * features ..][0..features],
                normalized_documents[seed_index * features ..][0..features],
            );
        }
        const assignments = try allocator.alloc(usize, document_count);
        errdefer allocator.free(assignments);
        @memset(assignments, 0);
        const sums = try allocator.alloc(f32, list_count * features);
        defer allocator.free(sums);
        const counts = try allocator.alloc(usize, list_count);
        defer allocator.free(counts);
        for (0..iterations) |_| {
            assignToCentroids(
                normalized_documents,
                features,
                centroids,
                assignments,
            );
            @memset(sums, 0);
            @memset(counts, 0);
            for (0..document_count) |document_index| {
                const list_index = assignments[document_index];
                counts[list_index] += 1;
                for (0..features) |feature| {
                    sums[list_index * features + feature] +=
                        normalized_documents[document_index * features + feature];
                }
            }
            for (0..list_count) |list_index| {
                if (counts[list_index] == 0) continue;
                try normalizeInto(
                    sums[list_index * features ..][0..features],
                    centroids[list_index * features ..][0..features],
                );
            }
        }
        assignToCentroids(
            normalized_documents,
            features,
            centroids,
            assignments,
        );
        return .{
            .allocator = allocator,
            .features = features,
            .document_count = document_count,
            .list_count = list_count,
            .documents = normalized_documents,
            .centroids = centroids,
            .assignments = assignments,
        };
    }

    pub fn deinit(self: *IvfIndex) void {
        self.allocator.free(self.documents);
        self.allocator.free(self.centroids);
        self.allocator.free(self.assignments);
        self.* = undefined;
    }

    pub fn search(
        self: IvfIndex,
        query: []const f32,
        k: usize,
        probes: usize,
    ) !IvfSearch {
        if (query.len != self.features or k == 0 or k > self.document_count or
            probes == 0 or probes > self.list_count)
        {
            return error.InvalidSearch;
        }
        const normalized_query = try self.allocator.alloc(f32, self.features);
        defer self.allocator.free(normalized_query);
        try normalizeInto(query, normalized_query);
        const nearest_lists = try topKCosine(
            self.allocator,
            normalized_query,
            self.centroids,
            self.features,
            probes,
        );
        defer self.allocator.free(nearest_lists);
        const selected = try self.allocator.alloc(bool, self.list_count);
        defer self.allocator.free(selected);
        @memset(selected, false);
        for (nearest_lists) |list| selected[list.index] = true;

        const candidates = try self.allocator.alloc(Neighbor, self.document_count);
        errdefer self.allocator.free(candidates);
        var candidate_count: usize = 0;
        for (self.assignments, 0..) |list_index, document_index| {
            if (!selected[list_index]) continue;
            const document = self.documents[document_index * self.features ..][0..self.features];
            var score: f32 = 0;
            for (normalized_query, document) |query_value, document_value| {
                score += query_value * document_value;
            }
            candidates[candidate_count] = .{ .index = document_index, .score = score };
            candidate_count += 1;
        }
        if (candidate_count < k) return error.InsufficientCandidates;
        std.mem.sort(Neighbor, candidates[0..candidate_count], {}, neighborBefore);
        const neighbors = try self.allocator.realloc(candidates, k);
        return .{
            .allocator = self.allocator,
            .neighbors = neighbors,
            .candidates_scored = candidate_count,
        };
    }
};

/// Ranks row-major document embeddings by cosine similarity. Equal scores use
/// document index as a deterministic tie-breaker.
pub fn topKCosine(
    allocator: Allocator,
    query: []const f32,
    documents: []const f32,
    features: usize,
    k: usize,
) ![]Neighbor {
    if (features == 0 or query.len != features or documents.len == 0 or documents.len % features != 0) {
        return error.DimensionMismatch;
    }
    const document_count = documents.len / features;
    if (k == 0 or k > document_count) return error.InvalidK;
    const query_norm = vectorNorm(query);
    if (query_norm == 0) return error.ZeroNorm;
    const candidates = try allocator.alloc(Neighbor, document_count);
    errdefer allocator.free(candidates);
    for (candidates, 0..) |*candidate, index| {
        const document = documents[index * features ..][0..features];
        const document_norm = vectorNorm(document);
        if (document_norm == 0) return error.ZeroNorm;
        var dot: f32 = 0;
        for (query, document) |query_value, document_value| dot += query_value * document_value;
        candidate.* = .{ .index = index, .score = dot / (query_norm * document_norm) };
    }
    std.mem.sort(Neighbor, candidates, {}, neighborBefore);
    return allocator.realloc(candidates, k);
}

pub fn recallAtK(exact: []const Neighbor, approximate: []const Neighbor) !f32 {
    if (exact.len == 0 or exact.len != approximate.len) return error.DimensionMismatch;
    var matches: usize = 0;
    for (exact) |expected| {
        for (approximate) |actual| {
            if (expected.index == actual.index) {
                matches += 1;
                break;
            }
        }
    }
    return @as(f32, @floatFromInt(matches)) /
        @as(f32, @floatFromInt(exact.len));
}

fn assignToCentroids(
    documents: []const f32,
    features: usize,
    centroids: []const f32,
    assignments: []usize,
) void {
    const list_count = centroids.len / features;
    for (assignments, 0..) |*assignment, document_index| {
        const document = documents[document_index * features ..][0..features];
        var best_list: usize = 0;
        var best_score = -std.math.inf(f32);
        for (0..list_count) |list_index| {
            const centroid = centroids[list_index * features ..][0..features];
            var score: f32 = 0;
            for (document, centroid) |document_value, centroid_value| {
                score += document_value * centroid_value;
            }
            if (score > best_score) {
                best_score = score;
                best_list = list_index;
            }
        }
        assignment.* = best_list;
    }
}

fn normalizeInto(input: []const f32, output: []f32) !void {
    if (input.len == 0 or input.len != output.len) return error.DimensionMismatch;
    var squared_norm: f32 = 0;
    for (input) |value| {
        if (!std.math.isFinite(value)) return error.InvalidVector;
        squared_norm += value * value;
    }
    if (!std.math.isFinite(squared_norm) or squared_norm == 0) return error.ZeroNorm;
    const inverse_norm = 1.0 / @sqrt(squared_norm);
    for (input, output) |value, *normalized| normalized.* = value * inverse_norm;
}

fn neighborBefore(_: void, left: Neighbor, right: Neighbor) bool {
    return left.score > right.score or
        (left.score == right.score and left.index < right.index);
}

fn validatePair(left: Tensor, right: Tensor) !void {
    if (left.shape.rank != 2 or right.shape.rank != 2 or left.shape.dims[0] == 0 or
        left.shape.dims[0] != right.shape.dims[0] or left.shape.dims[1] != right.shape.dims[1])
    {
        return error.DimensionMismatch;
    }
}

fn vectorNorm(values: []const f32) f32 {
    var squared: f32 = 0;
    for (values) |value| squared += value * value;
    return @sqrt(squared);
}

fn objectiveLoss(context: *ExecutionContext, left: Tensor, right: Tensor, temperature: f32) !f32 {
    var objective = try SymmetricInfoNCE.init(context, left, right, temperature);
    defer objective.deinit();
    return objective.loss(context);
}

test "symmetric InfoNCE aligns paired rows on every available device" {
    const testing = std.testing;
    const preferences = [_]tensor.DevicePreference{ .cpu, .metal, .cuda, .rocm };
    for (preferences) |preference| {
        var device = tensor.Device.init(testing.allocator, preference) catch |err| switch (err) {
            error.BackendUnavailable => continue,
            else => return err,
        };
        defer device.deinit();
        var context = ExecutionContext.init(&device);
        var left = try context.upload(&.{ 3, 3 }, &.{ 3, 0, 0, 0, 3, 0, 0, 0, 3 });
        defer left.deinit();
        var right = try context.upload(&.{ 3, 3 }, &.{ 3, 0, 0, 0, 3, 0, 0, 0, 3 });
        defer right.deinit();
        context.resetStats();
        var objective = try SymmetricInfoNCE.init(&context, left, right, 0.5);
        defer objective.deinit();
        try testing.expectEqual(@as(usize, 0), context.stats.readbacks);
        try testing.expectEqualSlices(usize, left.shape.slice(), objective.left_gradient.shape.slice());
        try testing.expectEqualSlices(usize, right.shape.slice(), objective.right_gradient.shape.slice());
        try testing.expect(try objective.loss(&context) < 1e-5);
    }
}

test "symmetric InfoNCE embedding gradients match finite differences" {
    const testing = std.testing;
    var device = try tensor.Device.init(testing.allocator, .cpu);
    defer device.deinit();
    var context = ExecutionContext.init(&device);
    const left_values = [_]f32{ 0.2, -0.3, 0.5, 0.1 };
    const right_values = [_]f32{ -0.4, 0.3, 0.6, -0.2 };
    const temperature: f32 = 0.7;
    var left = try context.upload(&.{ 2, 2 }, &left_values);
    defer left.deinit();
    var right = try context.upload(&.{ 2, 2 }, &right_values);
    defer right.deinit();
    var objective = try SymmetricInfoNCE.init(&context, left, right, temperature);
    defer objective.deinit();
    var left_analytical: [4]f32 = undefined;
    var right_analytical: [4]f32 = undefined;
    try context.readback(objective.left_gradient, &left_analytical);
    try context.readback(objective.right_gradient, &right_analytical);

    const epsilon: f32 = 1e-3;
    for (0..left_values.len) |index| {
        var plus = left_values;
        plus[index] += epsilon;
        var plus_tensor = try context.upload(&.{ 2, 2 }, &plus);
        defer plus_tensor.deinit();
        const plus_loss = try objectiveLoss(&context, plus_tensor, right, temperature);
        var minus = left_values;
        minus[index] -= epsilon;
        var minus_tensor = try context.upload(&.{ 2, 2 }, &minus);
        defer minus_tensor.deinit();
        const minus_loss = try objectiveLoss(&context, minus_tensor, right, temperature);
        try testing.expectApproxEqAbs((plus_loss - minus_loss) / (2 * epsilon), left_analytical[index], 4e-3);
    }
    for (0..right_values.len) |index| {
        var plus = right_values;
        plus[index] += epsilon;
        var plus_tensor = try context.upload(&.{ 2, 2 }, &plus);
        defer plus_tensor.deinit();
        const plus_loss = try objectiveLoss(&context, left, plus_tensor, temperature);
        var minus = right_values;
        minus[index] -= epsilon;
        var minus_tensor = try context.upload(&.{ 2, 2 }, &minus);
        defer minus_tensor.deinit();
        const minus_loss = try objectiveLoss(&context, left, minus_tensor, temperature);
        try testing.expectApproxEqAbs((plus_loss - minus_loss) / (2 * epsilon), right_analytical[index], 4e-3);
    }
}

test "top-k cosine ranking is deterministic" {
    const neighbors = try topKCosine(
        std.testing.allocator,
        &.{ 1, 0 },
        &.{ 0, 1, 1, 0, 1, 1, -1, 0 },
        2,
        3,
    );
    defer std.testing.allocator.free(neighbors);
    try std.testing.expectEqual(@as(usize, 1), neighbors[0].index);
    try std.testing.expectEqual(@as(usize, 2), neighbors[1].index);
    try std.testing.expectEqual(@as(usize, 0), neighbors[2].index);
    try std.testing.expect(neighbors[0].score > neighbors[1].score);
}

test "inverted-file search trades candidates for exact recall" {
    const documents = [_]f32{
        1.0,  0.0, 1.0,  0.1, 0.9,  -0.1, 0.8,  0.2,
        0.0,  1.0, 0.1,  1.0, -0.1, 0.9,  0.2,  0.8,
        -1.0, 0.0, -0.9, 0.1, -1.0, -0.1, -0.8, 0.2,
    };
    var index = try IvfIndex.init(std.testing.allocator, &documents, 2, 3, 6);
    defer index.deinit();
    const exact = try topKCosine(
        std.testing.allocator,
        &.{ 1, 0.05 },
        &documents,
        2,
        3,
    );
    defer std.testing.allocator.free(exact);
    var approximate = try index.search(&.{ 1, 0.05 }, 3, 1);
    defer approximate.deinit();
    try std.testing.expectEqual(@as(f32, 1), try recallAtK(exact, approximate.neighbors));
    try std.testing.expect(approximate.candidates_scored < index.document_count);
}
