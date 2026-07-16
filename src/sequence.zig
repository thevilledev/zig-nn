const std = @import("std");

pub const ExampleLengths = struct {
    source: usize,
    target: usize,
};

pub const Batch = struct {
    start: usize,
    len: usize,
    maximum_source: usize,
    maximum_target: usize,
    actual_tokens: usize,
    padded_tokens: usize,

    pub fn efficiency(self: Batch) f32 {
        return @as(f32, @floatFromInt(self.actual_tokens)) /
            @as(f32, @floatFromInt(self.padded_tokens));
    }
};

/// A deterministic length-sorted batch plan. `order` maps each batch position
/// back to the caller's example index, so dataset storage never has to move.
pub const BatchPlan = struct {
    allocator: std.mem.Allocator,
    order: []usize,
    batches: []Batch,

    pub fn init(
        allocator: std.mem.Allocator,
        lengths: []const ExampleLengths,
        batch_size: usize,
    ) !BatchPlan {
        if (lengths.len == 0 or batch_size == 0) return error.InvalidBatch;
        const order = try allocator.alloc(usize, lengths.len);
        errdefer allocator.free(order);
        for (order, 0..) |*index, value| index.* = value;
        std.mem.sort(usize, order, lengths, struct {
            fn before(context: []const ExampleLengths, left: usize, right: usize) bool {
                const left_span = @max(context[left].source, context[left].target);
                const right_span = @max(context[right].source, context[right].target);
                return left_span < right_span or
                    (left_span == right_span and left < right);
            }
        }.before);

        const batch_count = std.math.divCeil(usize, lengths.len, batch_size) catch
            return error.InvalidBatch;
        const batches = try allocator.alloc(Batch, batch_count);
        errdefer allocator.free(batches);
        for (batches, 0..) |*batch, batch_index| {
            const start = batch_index * batch_size;
            const count = @min(batch_size, lengths.len - start);
            var maximum_source: usize = 0;
            var maximum_target: usize = 0;
            var actual_tokens: usize = 0;
            for (order[start .. start + count]) |example_index| {
                const value = lengths[example_index];
                if (value.source == 0 or value.target == 0) return error.InvalidLength;
                maximum_source = @max(maximum_source, value.source);
                maximum_target = @max(maximum_target, value.target);
                actual_tokens = try std.math.add(
                    usize,
                    actual_tokens,
                    try std.math.add(usize, value.source, value.target),
                );
            }
            batch.* = .{
                .start = start,
                .len = count,
                .maximum_source = maximum_source,
                .maximum_target = maximum_target,
                .actual_tokens = actual_tokens,
                .padded_tokens = try std.math.mul(
                    usize,
                    count,
                    try std.math.add(usize, maximum_source, maximum_target),
                ),
            };
        }
        return .{ .allocator = allocator, .order = order, .batches = batches };
    }

    pub fn deinit(self: *BatchPlan) void {
        self.allocator.free(self.order);
        self.allocator.free(self.batches);
        self.* = undefined;
    }

    pub fn indices(self: BatchPlan, batch_index: usize) []const usize {
        const batch = self.batches[batch_index];
        return self.order[batch.start .. batch.start + batch.len];
    }

    pub fn efficiency(self: BatchPlan) f32 {
        var actual: usize = 0;
        var padded: usize = 0;
        for (self.batches) |batch| {
            actual += batch.actual_tokens;
            padded += batch.padded_tokens;
        }
        return @as(f32, @floatFromInt(actual)) /
            @as(f32, @floatFromInt(padded));
    }
};

/// Encoder padding, decoder padding, and causal decoder masks for one batch.
/// Values are stored as zero/one floats so they can be uploaded directly to a
/// tensor backend.
pub const Masks = struct {
    allocator: std.mem.Allocator,
    batch: usize,
    maximum_source: usize,
    maximum_target: usize,
    encoder: []f32,
    decoder: []f32,
    causal: []f32,

    pub fn init(
        allocator: std.mem.Allocator,
        source_lengths: []const usize,
        target_lengths: []const usize,
    ) !Masks {
        if (source_lengths.len == 0 or source_lengths.len != target_lengths.len) {
            return error.InvalidBatch;
        }
        var maximum_source: usize = 0;
        var maximum_target: usize = 0;
        for (source_lengths, target_lengths) |source, target| {
            if (source == 0 or target == 0) return error.InvalidLength;
            maximum_source = @max(maximum_source, source);
            maximum_target = @max(maximum_target, target);
        }
        const encoder = try allocator.alloc(
            f32,
            try std.math.mul(usize, source_lengths.len, maximum_source),
        );
        errdefer allocator.free(encoder);
        const decoder = try allocator.alloc(
            f32,
            try std.math.mul(usize, target_lengths.len, maximum_target),
        );
        errdefer allocator.free(decoder);
        const causal = try allocator.alloc(
            f32,
            try std.math.mul(
                usize,
                target_lengths.len,
                try std.math.mul(usize, maximum_target, maximum_target),
            ),
        );
        @memset(encoder, 0);
        @memset(decoder, 0);
        @memset(causal, 0);
        for (source_lengths, target_lengths, 0..) |source, target, batch_index| {
            @memset(encoder[batch_index * maximum_source ..][0..source], 1);
            @memset(decoder[batch_index * maximum_target ..][0..target], 1);
            const causal_offset = batch_index * maximum_target * maximum_target;
            for (0..target) |query| {
                @memset(
                    causal[causal_offset + query * maximum_target ..][0 .. query + 1],
                    1,
                );
            }
        }
        return .{
            .allocator = allocator,
            .batch = source_lengths.len,
            .maximum_source = maximum_source,
            .maximum_target = maximum_target,
            .encoder = encoder,
            .decoder = decoder,
            .causal = causal,
        };
    }

    pub fn deinit(self: *Masks) void {
        self.allocator.free(self.encoder);
        self.allocator.free(self.decoder);
        self.allocator.free(self.causal);
        self.* = undefined;
    }
};

/// Accumulates mean microbatch gradients without giving small final batches
/// disproportionate weight. Callers add each microbatch mean with its example
/// count, then request the full-batch mean with `finish`.
pub const GradientAccumulator = struct {
    allocator: std.mem.Allocator,
    sums: []f32,
    examples: usize = 0,
    microbatches: usize = 0,

    pub fn init(allocator: std.mem.Allocator, parameter_count: usize) !GradientAccumulator {
        if (parameter_count == 0) return error.InvalidGradient;
        const sums = try allocator.alloc(f32, parameter_count);
        @memset(sums, 0);
        return .{ .allocator = allocator, .sums = sums };
    }

    pub fn deinit(self: *GradientAccumulator) void {
        self.allocator.free(self.sums);
        self.* = undefined;
    }

    pub fn add(self: *GradientAccumulator, mean_gradient: []const f32, examples: usize) !void {
        if (examples == 0 or mean_gradient.len != self.sums.len) {
            return error.InvalidGradient;
        }
        for (self.sums, mean_gradient) |*sum, gradient| {
            if (!std.math.isFinite(gradient)) return error.InvalidGradient;
            sum.* += gradient * @as(f32, @floatFromInt(examples));
        }
        self.examples = try std.math.add(usize, self.examples, examples);
        self.microbatches += 1;
    }

    pub fn finish(self: GradientAccumulator, output: []f32) !void {
        if (self.examples == 0 or output.len != self.sums.len) {
            return error.InvalidGradient;
        }
        const scale = 1.0 / @as(f32, @floatFromInt(self.examples));
        for (output, self.sums) |*value, sum| value.* = sum * scale;
    }
};

pub const MaskedForecastLoss = struct {
    allocator: std.mem.Allocator,
    loss: f64,
    gradient: []f64,

    pub fn deinit(self: *MaskedForecastLoss) void {
        self.allocator.free(self.gradient);
        self.* = undefined;
    }
};

/// Mean squared forecasting loss over observed target positions. A zero mask
/// excludes warm-up or padded timesteps from both the objective and gradient.
pub fn maskedMeanSquaredError(
    allocator: std.mem.Allocator,
    predictions: []const f64,
    targets: []const f64,
    mask: []const f64,
) !MaskedForecastLoss {
    if (predictions.len == 0 or predictions.len != targets.len or
        predictions.len != mask.len)
    {
        return error.DimensionMismatch;
    }
    const gradient = try allocator.alloc(f64, predictions.len);
    errdefer allocator.free(gradient);
    @memset(gradient, 0);
    var loss: f64 = 0;
    var observed: usize = 0;
    for (predictions, targets, mask, gradient) |prediction, target, weight, *value| {
        if (!std.math.isFinite(prediction) or !std.math.isFinite(target) or
            (weight != 0 and weight != 1))
        {
            return error.InvalidForecast;
        }
        if (weight == 0) continue;
        const difference = prediction - target;
        loss += difference * difference;
        value.* = 2 * difference;
        observed += 1;
    }
    if (observed == 0) return error.EmptyMask;
    const scale = 1.0 / @as(f64, @floatFromInt(observed));
    for (gradient) |*value| value.* *= scale;
    return .{ .allocator = allocator, .loss = loss * scale, .gradient = gradient };
}

test "length bucketing improves padding efficiency" {
    const lengths = [_]ExampleLengths{
        .{ .source = 2, .target = 3 },
        .{ .source = 9, .target = 8 },
        .{ .source = 3, .target = 2 },
        .{ .source = 8, .target = 9 },
    };
    var plan = try BatchPlan.init(std.testing.allocator, &lengths, 2);
    defer plan.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 2 }, plan.indices(0));
    try std.testing.expect(plan.efficiency() > 0.85);
    const fixed_efficiency = @as(f32, 44) / @as(f32, 72);
    try std.testing.expect(plan.efficiency() > fixed_efficiency);
}

test "sequence masks exclude padding and future decoder tokens" {
    var masks = try Masks.init(std.testing.allocator, &.{ 2, 3 }, &.{ 3, 1 });
    defer masks.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 1, 0, 1, 1, 1 }, masks.encoder);
    try std.testing.expectEqualSlices(f32, &.{ 1, 1, 1, 1, 0, 0 }, masks.decoder);
    try std.testing.expectEqualSlices(
        f32,
        &.{
            1, 0, 0,
            1, 1, 0,
            1, 1, 1,
            1, 0, 0,
            0, 0, 0,
            0, 0, 0,
        },
        masks.causal,
    );
}

test "gradient accumulation weights uneven microbatches by examples" {
    var accumulator = try GradientAccumulator.init(std.testing.allocator, 2);
    defer accumulator.deinit();
    try accumulator.add(&.{ 1, 3 }, 3);
    try accumulator.add(&.{ 5, 7 }, 1);
    var mean: [2]f32 = undefined;
    try accumulator.finish(&mean);
    try std.testing.expectEqualSlices(f32, &.{ 2, 4 }, &mean);
    try std.testing.expectEqual(@as(usize, 2), accumulator.microbatches);
}

test "masked forecasting loss excludes warm-up positions" {
    var objective = try maskedMeanSquaredError(
        std.testing.allocator,
        &.{ 100, 2, 4 },
        &.{ 0, 1, 5 },
        &.{ 0, 1, 1 },
    );
    defer objective.deinit();
    try std.testing.expectEqual(@as(f64, 1), objective.loss);
    try std.testing.expectEqualSlices(f64, &.{ 0, 1, -1 }, objective.gradient);
}
