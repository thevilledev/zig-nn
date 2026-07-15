const std = @import("std");

pub const SamplingConfig = struct {
    /// Zero selects greedy decoding. Positive values scale logits before
    /// truncation and sampling.
    temperature: f64 = 1,
    /// Zero keeps the full vocabulary.
    top_k: usize = 0,
    /// One keeps the full post-top-k distribution.
    top_p: f64 = 1,
    /// Values above one reduce logits for tokens present in `recent_tokens`.
    repetition_penalty: f64 = 1,

    pub fn validate(self: SamplingConfig) !void {
        if (!std.math.isFinite(self.temperature) or self.temperature < 0) return error.InvalidTemperature;
        if (!std.math.isFinite(self.top_p) or self.top_p <= 0 or self.top_p > 1) return error.InvalidTopP;
        if (!std.math.isFinite(self.repetition_penalty) or self.repetition_penalty < 1) return error.InvalidRepetitionPenalty;
    }
};

pub const Candidate = struct {
    token: usize,
    logit: f64,
    probability: f64,
};

/// Returns the retained distribution in descending logit order. Top-k is
/// applied before nucleus top-p, matching common generation APIs.
pub fn distribution(
    allocator: std.mem.Allocator,
    logits: []const f64,
    recent_tokens: []const usize,
    config: SamplingConfig,
) ![]Candidate {
    try config.validate();
    if (logits.len == 0) return error.EmptyVocabulary;
    const candidates = try allocator.alloc(Candidate, logits.len);
    errdefer allocator.free(candidates);
    for (logits, candidates, 0..) |logit, *candidate, token| {
        if (!std.math.isFinite(logit)) return error.InvalidLogit;
        var adjusted = applyRepetitionPenalty(logit, token, recent_tokens, config.repetition_penalty);
        if (config.temperature > 0) adjusted /= config.temperature;
        candidate.* = .{ .token = token, .logit = adjusted, .probability = 0 };
    }
    sortDescending(candidates);
    if (config.temperature == 0) {
        candidates[0].probability = 1;
        const greedy = try allocator.realloc(candidates, 1);
        return greedy;
    }

    var retained = if (config.top_k == 0) candidates.len else @min(config.top_k, candidates.len);
    const maximum = candidates[0].logit;
    var total: f64 = 0;
    for (candidates[0..retained]) |*candidate| {
        candidate.probability = @exp(candidate.logit - maximum);
        total += candidate.probability;
    }
    for (candidates[0..retained]) |*candidate| candidate.probability /= total;
    if (config.top_p < 1) {
        var cumulative: f64 = 0;
        var nucleus_size: usize = 0;
        while (nucleus_size < retained) : (nucleus_size += 1) {
            cumulative += candidates[nucleus_size].probability;
            if (cumulative >= config.top_p) {
                nucleus_size += 1;
                break;
            }
        }
        retained = @max(@as(usize, 1), nucleus_size);
    }
    total = 0;
    for (candidates[0..retained]) |candidate| total += candidate.probability;
    for (candidates[0..retained]) |*candidate| candidate.probability /= total;
    return allocator.realloc(candidates, retained);
}

pub fn sample(
    allocator: std.mem.Allocator,
    random: std.Random,
    logits: []const f64,
    recent_tokens: []const usize,
    config: SamplingConfig,
) !usize {
    const candidates = try distribution(allocator, logits, recent_tokens, config);
    defer allocator.free(candidates);
    const draw = random.float(f64);
    var cumulative: f64 = 0;
    for (candidates) |candidate| {
        cumulative += candidate.probability;
        if (draw <= cumulative) return candidate.token;
    }
    return candidates[candidates.len - 1].token;
}

fn applyRepetitionPenalty(logit: f64, token: usize, recent_tokens: []const usize, penalty: f64) f64 {
    if (penalty == 1 or std.mem.indexOfScalar(usize, recent_tokens, token) == null) return logit;
    return if (logit >= 0) logit / penalty else logit * penalty;
}

fn sortDescending(candidates: []Candidate) void {
    std.mem.sort(Candidate, candidates, {}, struct {
        fn before(_: void, lhs: Candidate, rhs: Candidate) bool {
            return lhs.logit > rhs.logit or (lhs.logit == rhs.logit and lhs.token < rhs.token);
        }
    }.before);
}

test "top-k and top-p retain the smallest probability nucleus" {
    const candidates = try distribution(std.testing.allocator, &.{ 4, 3, 2, 1 }, &.{}, .{ .top_k = 3, .top_p = 0.8 });
    defer std.testing.allocator.free(candidates);
    try std.testing.expectEqual(@as(usize, 2), candidates.len);
    try std.testing.expectEqual(@as(usize, 0), candidates[0].token);
    try std.testing.expectEqual(@as(usize, 1), candidates[1].token);
    try std.testing.expectApproxEqAbs(@as(f64, 1), candidates[0].probability + candidates[1].probability, 1e-12);
}

test "zero temperature is greedy after repetition penalty" {
    const candidates = try distribution(std.testing.allocator, &.{ 4, 3.5, 1 }, &.{0}, .{ .temperature = 0, .repetition_penalty = 2 });
    defer std.testing.allocator.free(candidates);
    try std.testing.expectEqual(@as(usize, 1), candidates.len);
    try std.testing.expectEqual(@as(usize, 1), candidates[0].token);
    try std.testing.expectEqual(@as(f64, 1), candidates[0].probability);
}

test "seeded nucleus samples never escape retained candidates" {
    var prng = std.Random.DefaultPrng.init(42);
    for (0..100) |_| {
        const token = try sample(std.testing.allocator, prng.random(), &.{ 4, 3, 2, 1 }, &.{}, .{ .top_p = 0.7 });
        try std.testing.expect(token == 0 or token == 1);
    }
}
