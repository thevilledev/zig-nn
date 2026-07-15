const std = @import("std");

/// Exact gradients for a batched linear-chain CRF objective.
pub const CrfGradients = struct {
    allocator: std.mem.Allocator,
    emissions: []f32,
    start: []f32,
    transitions: []f32,
    end: []f32,

    pub fn deinit(self: *CrfGradients) void {
        self.allocator.free(self.emissions);
        self.allocator.free(self.start);
        self.allocator.free(self.transitions);
        self.allocator.free(self.end);
        self.* = undefined;
    }
};

pub const CrfObjective = struct {
    loss: f32,
    gradients: CrfGradients,

    pub fn deinit(self: *CrfObjective) void {
        self.gradients.deinit();
        self.* = undefined;
    }
};

/// A first-order conditional random field over tag sequences.
///
/// Emissions use `[batch, maximum_tokens, tags]` row-major storage. `lengths`
/// exclude padding, so neither the partition function nor gradients include
/// padded timesteps.
pub const LinearChainCrf = struct {
    allocator: std.mem.Allocator,
    tags: usize,
    start: []f32,
    transitions: []f32,
    end: []f32,

    pub fn init(allocator: std.mem.Allocator, tags: usize) !LinearChainCrf {
        if (tags < 2) return error.InvalidTagCount;
        const start = try allocator.alloc(f32, tags);
        errdefer allocator.free(start);
        const transitions = try allocator.alloc(f32, tags * tags);
        errdefer allocator.free(transitions);
        const end = try allocator.alloc(f32, tags);
        @memset(start, 0);
        @memset(transitions, 0);
        @memset(end, 0);
        return .{ .allocator = allocator, .tags = tags, .start = start, .transitions = transitions, .end = end };
    }

    pub fn deinit(self: *LinearChainCrf) void {
        self.allocator.free(self.start);
        self.allocator.free(self.transitions);
        self.allocator.free(self.end);
        self.* = undefined;
    }

    pub fn lossAndGradient(
        self: LinearChainCrf,
        emissions: []const f32,
        targets: []const usize,
        lengths: []const usize,
        maximum_tokens: usize,
    ) !CrfObjective {
        const batch = try self.validateBatch(emissions, targets, lengths, maximum_tokens);
        const emission_gradients = try self.allocator.alloc(f32, emissions.len);
        errdefer self.allocator.free(emission_gradients);
        @memset(emission_gradients, 0);
        const start_gradients = try self.allocator.alloc(f32, self.tags);
        errdefer self.allocator.free(start_gradients);
        @memset(start_gradients, 0);
        const transition_gradients = try self.allocator.alloc(f32, self.tags * self.tags);
        errdefer self.allocator.free(transition_gradients);
        @memset(transition_gradients, 0);
        const end_gradients = try self.allocator.alloc(f32, self.tags);
        errdefer self.allocator.free(end_gradients);
        @memset(end_gradients, 0);

        var total_loss: f32 = 0;
        for (lengths, 0..) |length, batch_index| {
            const alpha = try self.allocator.alloc(f32, length * self.tags);
            defer self.allocator.free(alpha);
            const beta = try self.allocator.alloc(f32, length * self.tags);
            defer self.allocator.free(beta);
            const emission_offset = batch_index * maximum_tokens * self.tags;
            const target_offset = batch_index * maximum_tokens;
            self.forward(emissions[emission_offset..], length, alpha);
            self.backward(emissions[emission_offset..], length, beta);
            const log_partition = self.logPartition(alpha[(length - 1) * self.tags .. length * self.tags]);
            total_loss += log_partition - self.pathScore(
                emissions[emission_offset..],
                targets[target_offset..],
                length,
            );

            for (0..length) |token| {
                for (0..self.tags) |tag| {
                    const marginal = @exp(alpha[token * self.tags + tag] + beta[token * self.tags + tag] - log_partition);
                    emission_gradients[emission_offset + token * self.tags + tag] += marginal;
                }
                emission_gradients[emission_offset + token * self.tags + targets[target_offset + token]] -= 1;
            }
            for (0..self.tags) |tag| {
                const first_marginal = @exp(alpha[tag] + beta[tag] - log_partition);
                start_gradients[tag] += first_marginal;
                const last_index = (length - 1) * self.tags + tag;
                const last_marginal = @exp(alpha[last_index] + self.end[tag] - log_partition);
                end_gradients[tag] += last_marginal;
            }
            start_gradients[targets[target_offset]] -= 1;
            end_gradients[targets[target_offset + length - 1]] -= 1;
            for (1..length) |token| {
                for (0..self.tags) |previous| {
                    for (0..self.tags) |current| {
                        const pair_marginal = @exp(
                            alpha[(token - 1) * self.tags + previous] +
                                self.transitions[previous * self.tags + current] +
                                emissions[emission_offset + token * self.tags + current] +
                                beta[token * self.tags + current] - log_partition,
                        );
                        transition_gradients[previous * self.tags + current] += pair_marginal;
                    }
                }
                const previous_target = targets[target_offset + token - 1];
                const current_target = targets[target_offset + token];
                transition_gradients[previous_target * self.tags + current_target] -= 1;
            }
        }

        const inverse_batch = 1.0 / @as(f32, @floatFromInt(batch));
        for (emission_gradients) |*gradient| gradient.* *= inverse_batch;
        for (start_gradients) |*gradient| gradient.* *= inverse_batch;
        for (transition_gradients) |*gradient| gradient.* *= inverse_batch;
        for (end_gradients) |*gradient| gradient.* *= inverse_batch;
        return .{
            .loss = total_loss * inverse_batch,
            .gradients = .{
                .allocator = self.allocator,
                .emissions = emission_gradients,
                .start = start_gradients,
                .transitions = transition_gradients,
                .end = end_gradients,
            },
        };
    }

    pub fn negativeLogLikelihood(self: LinearChainCrf, emissions: []const f32, targets: []const usize, lengths: []const usize, maximum_tokens: usize) !f32 {
        var objective = try self.lossAndGradient(emissions, targets, lengths, maximum_tokens);
        defer objective.deinit();
        return objective.loss;
    }

    pub fn applyGradient(self: *LinearChainCrf, gradients: CrfGradients, learning_rate: f32) !void {
        if (!std.math.isFinite(learning_rate) or learning_rate <= 0 or
            gradients.start.len != self.tags or gradients.end.len != self.tags or gradients.transitions.len != self.tags * self.tags)
        {
            return error.InvalidGradient;
        }
        for (self.start, gradients.start) |*parameter, gradient| parameter.* -= learning_rate * gradient;
        for (self.transitions, gradients.transitions) |*parameter, gradient| parameter.* -= learning_rate * gradient;
        for (self.end, gradients.end) |*parameter, gradient| parameter.* -= learning_rate * gradient;
    }

    /// Viterbi decoding for one unpadded sequence. Returns its path score.
    pub fn decode(self: LinearChainCrf, emissions: []const f32, length: usize, output: []usize) !f32 {
        if (length == 0 or emissions.len < length * self.tags or output.len < length) return error.DimensionMismatch;
        const scores = try self.allocator.alloc(f32, length * self.tags);
        defer self.allocator.free(scores);
        const backpointers = try self.allocator.alloc(usize, length * self.tags);
        defer self.allocator.free(backpointers);
        for (0..self.tags) |tag| scores[tag] = self.start[tag] + emissions[tag];
        for (1..length) |token| {
            for (0..self.tags) |current| {
                var best_previous: usize = 0;
                var best_score = -std.math.inf(f32);
                for (0..self.tags) |previous| {
                    const candidate = scores[(token - 1) * self.tags + previous] + self.transitions[previous * self.tags + current];
                    if (candidate > best_score) {
                        best_score = candidate;
                        best_previous = previous;
                    }
                }
                scores[token * self.tags + current] = best_score + emissions[token * self.tags + current];
                backpointers[token * self.tags + current] = best_previous;
            }
        }
        var best_tag: usize = 0;
        var best_score = -std.math.inf(f32);
        for (0..self.tags) |tag| {
            const candidate = scores[(length - 1) * self.tags + tag] + self.end[tag];
            if (candidate > best_score) {
                best_score = candidate;
                best_tag = tag;
            }
        }
        output[length - 1] = best_tag;
        var token = length - 1;
        while (token > 0) : (token -= 1) {
            output[token - 1] = backpointers[token * self.tags + output[token]];
        }
        return best_score;
    }

    fn forward(self: LinearChainCrf, emissions: []const f32, length: usize, alpha: []f32) void {
        for (0..self.tags) |tag| alpha[tag] = self.start[tag] + emissions[tag];
        for (1..length) |token| {
            for (0..self.tags) |current| {
                var maximum = -std.math.inf(f32);
                for (0..self.tags) |previous| maximum = @max(maximum, alpha[(token - 1) * self.tags + previous] + self.transitions[previous * self.tags + current]);
                var sum: f32 = 0;
                for (0..self.tags) |previous| sum += @exp(alpha[(token - 1) * self.tags + previous] + self.transitions[previous * self.tags + current] - maximum);
                alpha[token * self.tags + current] = emissions[token * self.tags + current] + maximum + @log(sum);
            }
        }
    }

    fn backward(self: LinearChainCrf, emissions: []const f32, length: usize, beta: []f32) void {
        for (0..self.tags) |tag| beta[(length - 1) * self.tags + tag] = self.end[tag];
        var token = length - 1;
        while (token > 0) {
            token -= 1;
            for (0..self.tags) |previous| {
                var maximum = -std.math.inf(f32);
                for (0..self.tags) |current| maximum = @max(maximum, self.transitions[previous * self.tags + current] + emissions[(token + 1) * self.tags + current] + beta[(token + 1) * self.tags + current]);
                var sum: f32 = 0;
                for (0..self.tags) |current| sum += @exp(self.transitions[previous * self.tags + current] + emissions[(token + 1) * self.tags + current] + beta[(token + 1) * self.tags + current] - maximum);
                beta[token * self.tags + previous] = maximum + @log(sum);
            }
        }
    }

    fn logPartition(self: LinearChainCrf, last_alpha: []const f32) f32 {
        var maximum = -std.math.inf(f32);
        for (last_alpha, self.end) |score, ending| maximum = @max(maximum, score + ending);
        var sum: f32 = 0;
        for (last_alpha, self.end) |score, ending| sum += @exp(score + ending - maximum);
        return maximum + @log(sum);
    }

    fn pathScore(self: LinearChainCrf, emissions: []const f32, targets: []const usize, length: usize) f32 {
        var score = self.start[targets[0]] + emissions[targets[0]];
        for (1..length) |token| {
            score += self.transitions[targets[token - 1] * self.tags + targets[token]] + emissions[token * self.tags + targets[token]];
        }
        return score + self.end[targets[length - 1]];
    }

    fn validateBatch(self: LinearChainCrf, emissions: []const f32, targets: []const usize, lengths: []const usize, maximum_tokens: usize) !usize {
        if (maximum_tokens == 0 or lengths.len == 0 or emissions.len != lengths.len * maximum_tokens * self.tags or targets.len != lengths.len * maximum_tokens) return error.DimensionMismatch;
        for (lengths, 0..) |length, batch_index| {
            if (length == 0 or length > maximum_tokens) return error.InvalidSequenceLength;
            for (targets[batch_index * maximum_tokens .. batch_index * maximum_tokens + length]) |tag| if (tag >= self.tags) return error.InvalidTag;
        }
        return lengths.len;
    }
};

test "CRF transition and emission gradients match finite differences" {
    const testing = std.testing;
    var crf = try LinearChainCrf.init(testing.allocator, 2);
    defer crf.deinit();
    crf.start[0] = 0.1;
    crf.start[1] = -0.2;
    crf.transitions[0] = 0.3;
    crf.transitions[1] = -0.1;
    crf.transitions[2] = 0.2;
    crf.transitions[3] = 0.05;
    crf.end[0] = -0.1;
    crf.end[1] = 0.15;
    var emissions = [_]f32{ 0.4, -0.3, 0.1, 0.5, -0.2, 0.3 };
    var objective = try crf.lossAndGradient(&emissions, &.{ 0, 1, 1 }, &.{3}, 3);
    defer objective.deinit();
    const epsilon: f32 = 1e-3;
    const original_transition = crf.transitions[1];
    crf.transitions[1] = original_transition + epsilon;
    const transition_plus = try crf.negativeLogLikelihood(&emissions, &.{ 0, 1, 1 }, &.{3}, 3);
    crf.transitions[1] = original_transition - epsilon;
    const transition_minus = try crf.negativeLogLikelihood(&emissions, &.{ 0, 1, 1 }, &.{3}, 3);
    crf.transitions[1] = original_transition;
    try testing.expectApproxEqAbs((transition_plus - transition_minus) / (2 * epsilon), objective.gradients.transitions[1], 2e-4);
    const original_emission = emissions[3];
    emissions[3] = original_emission + epsilon;
    const emission_plus = try crf.negativeLogLikelihood(&emissions, &.{ 0, 1, 1 }, &.{3}, 3);
    emissions[3] = original_emission - epsilon;
    const emission_minus = try crf.negativeLogLikelihood(&emissions, &.{ 0, 1, 1 }, &.{3}, 3);
    emissions[3] = original_emission;
    try testing.expectApproxEqAbs((emission_plus - emission_minus) / (2 * epsilon), objective.gradients.emissions[3], 2e-4);
}

test "Viterbi uses transition structure instead of local argmax" {
    const testing = std.testing;
    var crf = try LinearChainCrf.init(testing.allocator, 2);
    defer crf.deinit();
    crf.transitions[0 * 2 + 1] = 2;
    crf.transitions[1 * 2 + 0] = 2;
    var decoded: [3]usize = undefined;
    _ = try crf.decode(&.{ 1, 0, 1, 0, 1, 0 }, 3, &decoded);
    try testing.expectEqualSlices(usize, &.{ 0, 1, 0 }, &decoded);
}
