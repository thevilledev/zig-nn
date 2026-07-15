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
    std.mem.sort(Neighbor, candidates, {}, struct {
        fn before(_: void, left: Neighbor, right: Neighbor) bool {
            return left.score > right.score or (left.score == right.score and left.index < right.index);
        }
    }.before);
    return allocator.realloc(candidates, k);
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
