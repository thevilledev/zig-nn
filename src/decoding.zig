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

pub const BeamSearchConfig = struct {
    width: usize = 4,
    minimum_tokens: usize = 0,
    maximum_tokens: usize = 32,
    length_penalty: f64 = 0.6,

    fn validate(self: BeamSearchConfig) !void {
        if (self.width == 0 or self.maximum_tokens == 0 or
            self.minimum_tokens > self.maximum_tokens or
            !std.math.isFinite(self.length_penalty) or self.length_penalty < 0)
        {
            return error.InvalidBeamSearch;
        }
    }
};

pub const BeamSearchResult = struct {
    allocator: std.mem.Allocator,
    tokens: []usize,
    log_probability: f64,
    normalized_score: f64,

    pub fn deinit(self: *BeamSearchResult) void {
        self.allocator.free(self.tokens);
        self.* = undefined;
    }
};

const Beam = struct {
    tokens: []usize,
    log_probability: f64,
    ended: bool,
};

/// Length-normalized beam search over an autoregressive next-token callback.
/// The callback receives the complete prefix, including `start_token`, and
/// fills one log probability per vocabulary item.
pub fn beamSearch(
    allocator: std.mem.Allocator,
    start_token: usize,
    end_token: usize,
    vocabulary_size: usize,
    config: BeamSearchConfig,
    context: anytype,
    nextLogProbabilities: anytype,
) !BeamSearchResult {
    try config.validate();
    if (vocabulary_size < 2 or start_token >= vocabulary_size or
        end_token >= vocabulary_size)
    {
        return error.InvalidVocabulary;
    }
    var beams = try allocator.alloc(Beam, 1);
    beams[0] = .{
        .tokens = try allocator.dupe(usize, &.{start_token}),
        .log_probability = 0,
        .ended = false,
    };
    errdefer {
        for (beams) |beam| allocator.free(beam.tokens);
        allocator.free(beams);
    }

    for (0..config.maximum_tokens) |_| {
        var candidates: std.ArrayList(Beam) = .empty;
        defer candidates.deinit(allocator);
        errdefer for (candidates.items) |candidate| allocator.free(candidate.tokens);
        for (beams) |beam| {
            if (beam.ended) {
                try candidates.append(allocator, .{
                    .tokens = try allocator.dupe(usize, beam.tokens),
                    .log_probability = beam.log_probability,
                    .ended = true,
                });
                continue;
            }
            const log_probabilities = try allocator.alloc(f64, vocabulary_size);
            defer allocator.free(log_probabilities);
            try nextLogProbabilities(context, beam.tokens, log_probabilities);
            for (log_probabilities, 0..) |log_probability, token| {
                const generated_length = beam.tokens.len;
                if (token == end_token and generated_length < config.minimum_tokens) {
                    continue;
                }
                if (!std.math.isFinite(log_probability)) return error.InvalidLogProbability;
                const sequence = try allocator.alloc(usize, beam.tokens.len + 1);
                @memcpy(sequence[0..beam.tokens.len], beam.tokens);
                sequence[beam.tokens.len] = token;
                try candidates.append(allocator, .{
                    .tokens = sequence,
                    .log_probability = beam.log_probability + log_probability,
                    .ended = token == end_token,
                });
            }
        }
        std.mem.sort(Beam, candidates.items, config, struct {
            fn before(search_config: BeamSearchConfig, left: Beam, right: Beam) bool {
                const left_score = normalizedBeamScore(left, search_config.length_penalty);
                const right_score = normalizedBeamScore(right, search_config.length_penalty);
                if (left_score != right_score) return left_score > right_score;
                return lexicographicBefore(left.tokens, right.tokens);
            }
        }.before);

        const retained = @min(config.width, candidates.items.len);
        const next_beams = try allocator.alloc(Beam, retained);
        for (candidates.items[0..retained], next_beams) |candidate, *next_beam| {
            next_beam.* = candidate;
        }
        for (candidates.items[retained..]) |candidate| allocator.free(candidate.tokens);
        for (beams) |beam| allocator.free(beam.tokens);
        allocator.free(beams);
        beams = next_beams;

        var all_ended = true;
        for (beams) |beam| all_ended = all_ended and beam.ended;
        if (all_ended) break;
    }
    defer {
        for (beams) |beam| allocator.free(beam.tokens);
        allocator.free(beams);
    }
    std.mem.sort(Beam, beams, config, struct {
        fn before(search_config: BeamSearchConfig, left: Beam, right: Beam) bool {
            const left_score = normalizedBeamScore(left, search_config.length_penalty);
            const right_score = normalizedBeamScore(right, search_config.length_penalty);
            if (left_score != right_score) return left_score > right_score;
            return lexicographicBefore(left.tokens, right.tokens);
        }
    }.before);
    const best = beams[0];
    const tokens = try allocator.dupe(usize, best.tokens[1..]);
    return .{
        .allocator = allocator,
        .tokens = tokens,
        .log_probability = best.log_probability,
        .normalized_score = normalizedBeamScore(best, config.length_penalty),
    };
}

fn normalizedBeamScore(beam: Beam, length_penalty: f64) f64 {
    const generated = @max(@as(usize, 1), beam.tokens.len - 1);
    const penalty = std.math.pow(
        f64,
        (@as(f64, @floatFromInt(generated)) + 5.0) / 6.0,
        length_penalty,
    );
    return beam.log_probability / penalty;
}

fn lexicographicBefore(left: []const usize, right: []const usize) bool {
    for (left[0..@min(left.len, right.len)], right[0..@min(left.len, right.len)]) |a, b| {
        if (a != b) return a < b;
    }
    return left.len < right.len;
}

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

test "beam search recovers a sequence hidden by greedy choice" {
    const Scorer = struct {
        fn score(_: void, prefix: []const usize, output: []f64) !void {
            @memset(output, -100);
            if (prefix.len == 1) {
                output[1] = @log(@as(f64, 0.6));
                output[2] = @log(@as(f64, 0.4));
            } else if (prefix[prefix.len - 1] == 1) {
                output[3] = @log(@as(f64, 0.55));
                output[1] = @log(@as(f64, 0.45));
            } else {
                output[3] = @log(@as(f64, 0.99));
                output[2] = @log(@as(f64, 0.01));
            }
        }
    };
    var greedy = try beamSearch(
        std.testing.allocator,
        0,
        3,
        4,
        .{ .width = 1, .maximum_tokens = 2, .length_penalty = 0 },
        {},
        Scorer.score,
    );
    defer greedy.deinit();
    var beam = try beamSearch(
        std.testing.allocator,
        0,
        3,
        4,
        .{ .width = 2, .maximum_tokens = 2, .length_penalty = 0 },
        {},
        Scorer.score,
    );
    defer beam.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 3 }, greedy.tokens);
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, beam.tokens);
    try std.testing.expect(beam.log_probability > greedy.log_probability);
}
