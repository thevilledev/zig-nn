const std = @import("std");

pub const Objective = struct {
    allocator: std.mem.Allocator,
    loss: f32,
    gradient: []f32,

    pub fn deinit(self: *Objective) void {
        self.allocator.free(self.gradient);
        self.* = undefined;
    }
};

/// Blank-aware CTC forward-backward over one sequence. `logits` use
/// `[timesteps, classes]` row-major storage and the returned gradient is with
/// respect to those logits.
pub fn lossAndGradient(
    allocator: std.mem.Allocator,
    logits: []const f32,
    timesteps: usize,
    classes: usize,
    labels: []const usize,
    blank: usize,
) !Objective {
    try validate(logits, timesteps, classes, labels, blank);
    const state_count = 2 * labels.len + 1;
    const states = try allocator.alloc(usize, state_count);
    defer allocator.free(states);
    @memset(states, blank);
    for (labels, 0..) |label, index| states[2 * index + 1] = label;
    const log_probabilities = try allocator.alloc(f32, logits.len);
    defer allocator.free(log_probabilities);
    logitsToLogProbabilities(logits, timesteps, classes, log_probabilities);
    const alpha = try allocator.alloc(f32, timesteps * state_count);
    defer allocator.free(alpha);
    const beta = try allocator.alloc(f32, timesteps * state_count);
    defer allocator.free(beta);
    @memset(alpha, -std.math.inf(f32));
    @memset(beta, -std.math.inf(f32));

    alpha[0] = log_probabilities[blank];
    if (state_count > 1) alpha[1] = log_probabilities[states[1]];
    for (1..timesteps) |timestep| {
        for (0..state_count) |state| {
            var total = alpha[(timestep - 1) * state_count + state];
            if (state > 0) {
                total = logAdd(
                    total,
                    alpha[(timestep - 1) * state_count + state - 1],
                );
            }
            if (state > 1 and states[state] != blank and
                states[state] != states[state - 2])
            {
                total = logAdd(
                    total,
                    alpha[(timestep - 1) * state_count + state - 2],
                );
            }
            alpha[timestep * state_count + state] = total +
                log_probabilities[timestep * classes + states[state]];
        }
    }
    const last_offset = (timesteps - 1) * state_count;
    const log_likelihood = if (state_count == 1)
        alpha[last_offset]
    else
        logAdd(
            alpha[last_offset + state_count - 1],
            alpha[last_offset + state_count - 2],
        );
    if (!std.math.isFinite(log_likelihood)) return error.ImpossibleAlignment;

    beta[last_offset + state_count - 1] = 0;
    if (state_count > 1) beta[last_offset + state_count - 2] = 0;
    var next_timestep = timesteps - 1;
    while (next_timestep > 0) {
        const timestep = next_timestep - 1;
        for (0..state_count) |state| {
            var total = beta[next_timestep * state_count + state] +
                log_probabilities[next_timestep * classes + states[state]];
            if (state + 1 < state_count) {
                total = logAdd(
                    total,
                    beta[next_timestep * state_count + state + 1] +
                        log_probabilities[next_timestep * classes + states[state + 1]],
                );
            }
            if (state + 2 < state_count and states[state + 2] != blank and
                states[state] != states[state + 2])
            {
                total = logAdd(
                    total,
                    beta[next_timestep * state_count + state + 2] +
                        log_probabilities[next_timestep * classes + states[state + 2]],
                );
            }
            beta[timestep * state_count + state] = total;
        }
        next_timestep = timestep;
    }

    const gradient = try allocator.alloc(f32, logits.len);
    errdefer allocator.free(gradient);
    for (0..timesteps) |timestep| {
        for (0..classes) |class| {
            gradient[timestep * classes + class] =
                @exp(log_probabilities[timestep * classes + class]);
        }
        for (states, 0..) |class, state| {
            const posterior = @exp(
                alpha[timestep * state_count + state] +
                    beta[timestep * state_count + state] - log_likelihood,
            );
            gradient[timestep * classes + class] -= posterior;
        }
    }
    return .{
        .allocator = allocator,
        .loss = -log_likelihood,
        .gradient = gradient,
    };
}

pub fn greedyDecode(
    allocator: std.mem.Allocator,
    logits: []const f32,
    timesteps: usize,
    classes: usize,
    blank: usize,
) ![]usize {
    if (timesteps == 0 or classes < 2 or blank >= classes or
        logits.len != timesteps * classes)
    {
        return error.DimensionMismatch;
    }
    var decoded: std.ArrayList(usize) = .empty;
    defer decoded.deinit(allocator);
    var previous: ?usize = null;
    for (0..timesteps) |timestep| {
        const row = logits[timestep * classes ..][0..classes];
        var best: usize = 0;
        for (row[1..], 1..) |value, class| if (value > row[best]) {
            best = class;
        };
        if (best != blank and (previous == null or previous.? != best)) {
            try decoded.append(allocator, best);
        }
        previous = best;
    }
    return decoded.toOwnedSlice(allocator);
}

const Prefix = struct {
    tokens: []usize,
    blank_probability: f64,
    nonblank_probability: f64,

    fn total(self: Prefix) f64 {
        return self.blank_probability + self.nonblank_probability;
    }
};

/// Prefix beam decoding in probability space. This is intentionally a small
/// reference implementation; production decoders normally stay in log space
/// and fuse language-model scores.
pub fn prefixBeamDecode(
    allocator: std.mem.Allocator,
    logits: []const f32,
    timesteps: usize,
    classes: usize,
    blank: usize,
    beam_width: usize,
) ![]usize {
    if (timesteps == 0 or classes < 2 or blank >= classes or beam_width == 0 or
        logits.len != timesteps * classes)
    {
        return error.DimensionMismatch;
    }
    var beams = initial: {
        const tokens = try allocator.alloc(usize, 0);
        errdefer allocator.free(tokens);
        const prefixes = try allocator.alloc(Prefix, 1);
        prefixes[0] = .{
            .tokens = tokens,
            .blank_probability = 1,
            .nonblank_probability = 0,
        };
        break :initial prefixes;
    };
    errdefer freePrefixes(allocator, beams);
    const probabilities = try allocator.alloc(f64, classes);
    defer allocator.free(probabilities);

    for (0..timesteps) |timestep| {
        probabilitiesFromLogits(
            logits[timestep * classes ..][0..classes],
            probabilities,
        );
        var next: std.ArrayList(Prefix) = .empty;
        defer next.deinit(allocator);
        errdefer for (next.items) |prefix| allocator.free(prefix.tokens);
        for (beams) |beam| {
            try updatePrefix(
                allocator,
                &next,
                beam.tokens,
                beam.total() * probabilities[blank],
                0,
            );
            for (0..classes) |class| {
                if (class == blank) continue;
                const last = if (beam.tokens.len == 0)
                    null
                else
                    beam.tokens[beam.tokens.len - 1];
                if (last != null and last.? == class) {
                    try updatePrefix(
                        allocator,
                        &next,
                        beam.tokens,
                        0,
                        beam.nonblank_probability * probabilities[class],
                    );
                    const extended = try appendToken(allocator, beam.tokens, class);
                    defer allocator.free(extended);
                    try updatePrefix(
                        allocator,
                        &next,
                        extended,
                        0,
                        beam.blank_probability * probabilities[class],
                    );
                } else {
                    const extended = try appendToken(allocator, beam.tokens, class);
                    defer allocator.free(extended);
                    try updatePrefix(
                        allocator,
                        &next,
                        extended,
                        0,
                        beam.total() * probabilities[class],
                    );
                }
            }
        }
        std.mem.sort(Prefix, next.items, {}, prefixBefore);
        const retained = @min(beam_width, next.items.len);
        const next_beams = try allocator.alloc(Prefix, retained);
        @memcpy(next_beams, next.items[0..retained]);
        for (next.items[retained..]) |prefix| allocator.free(prefix.tokens);
        freePrefixes(allocator, beams);
        beams = next_beams;
    }
    defer freePrefixes(allocator, beams);
    std.mem.sort(Prefix, beams, {}, prefixBefore);
    return allocator.dupe(usize, beams[0].tokens);
}

fn validate(
    logits: []const f32,
    timesteps: usize,
    classes: usize,
    labels: []const usize,
    blank: usize,
) !void {
    if (timesteps == 0 or classes < 2 or blank >= classes or
        logits.len != timesteps * classes)
    {
        return error.DimensionMismatch;
    }
    var required_timesteps = labels.len;
    for (labels, 0..) |label, index| {
        if (label >= classes or label == blank) return error.InvalidLabel;
        if (index > 0 and labels[index - 1] == label) required_timesteps += 1;
    }
    if (required_timesteps > timesteps) return error.ImpossibleAlignment;
    for (logits) |logit| if (!std.math.isFinite(logit)) return error.InvalidLogit;
}

fn logitsToLogProbabilities(
    logits: []const f32,
    timesteps: usize,
    classes: usize,
    output: []f32,
) void {
    for (0..timesteps) |timestep| {
        const row = logits[timestep * classes ..][0..classes];
        var maximum = row[0];
        for (row[1..]) |value| maximum = @max(maximum, value);
        var total: f32 = 0;
        for (row) |value| total += @exp(value - maximum);
        const normalizer = maximum + @log(total);
        for (row, output[timestep * classes ..][0..classes]) |value, *result| {
            result.* = value - normalizer;
        }
    }
}

fn probabilitiesFromLogits(logits: []const f32, output: []f64) void {
    var maximum = logits[0];
    for (logits[1..]) |value| maximum = @max(maximum, value);
    var total: f64 = 0;
    for (logits, output) |value, *probability| {
        probability.* = @exp(@as(f64, value - maximum));
        total += probability.*;
    }
    for (output) |*probability| probability.* /= total;
}

fn logAdd(left: f32, right: f32) f32 {
    if (left == -std.math.inf(f32)) return right;
    if (right == -std.math.inf(f32)) return left;
    const maximum = @max(left, right);
    return maximum + @log(@exp(left - maximum) + @exp(right - maximum));
}

fn appendToken(allocator: std.mem.Allocator, prefix: []const usize, token: usize) ![]usize {
    const result = try allocator.alloc(usize, prefix.len + 1);
    @memcpy(result[0..prefix.len], prefix);
    result[prefix.len] = token;
    return result;
}

fn updatePrefix(
    allocator: std.mem.Allocator,
    prefixes: *std.ArrayList(Prefix),
    tokens: []const usize,
    blank_addition: f64,
    nonblank_addition: f64,
) !void {
    for (prefixes.items) |*prefix| {
        if (std.mem.eql(usize, prefix.tokens, tokens)) {
            prefix.blank_probability += blank_addition;
            prefix.nonblank_probability += nonblank_addition;
            return;
        }
    }
    const owned_tokens = try allocator.dupe(usize, tokens);
    errdefer allocator.free(owned_tokens);
    try prefixes.append(allocator, .{
        .tokens = owned_tokens,
        .blank_probability = blank_addition,
        .nonblank_probability = nonblank_addition,
    });
}

fn prefixBefore(_: void, left: Prefix, right: Prefix) bool {
    if (left.total() != right.total()) return left.total() > right.total();
    for (left.tokens[0..@min(left.tokens.len, right.tokens.len)], right.tokens[0..@min(left.tokens.len, right.tokens.len)]) |a, b| {
        if (a != b) return a < b;
    }
    return left.tokens.len < right.tokens.len;
}

fn freePrefixes(allocator: std.mem.Allocator, prefixes: []Prefix) void {
    for (prefixes) |prefix| allocator.free(prefix.tokens);
    allocator.free(prefixes);
}

test "CTC gradients match finite differences" {
    var logits = [_]f32{
        0.2, 0.8,  -0.1,
        0.7, 0.1,  0.3,
        0.1, -0.2, 0.9,
    };
    var objective = try lossAndGradient(std.testing.allocator, &logits, 3, 3, &.{ 1, 2 }, 0);
    defer objective.deinit();
    const epsilon: f32 = 1e-3;
    const original = logits[4];
    logits[4] = original + epsilon;
    var plus = try lossAndGradient(std.testing.allocator, &logits, 3, 3, &.{ 1, 2 }, 0);
    defer plus.deinit();
    logits[4] = original - epsilon;
    var minus = try lossAndGradient(std.testing.allocator, &logits, 3, 3, &.{ 1, 2 }, 0);
    defer minus.deinit();
    try std.testing.expectApproxEqAbs(
        (plus.loss - minus.loss) / (2 * epsilon),
        objective.gradient[4],
        2e-3,
    );
}

test "CTC greedy and prefix decoding collapse blanks and repeats" {
    const logits = [_]f32{
        0, 4, 0,
        4, 0, 0,
        0, 0, 4,
        0, 0, 3,
        4, 0, 0,
    };
    const greedy = try greedyDecode(std.testing.allocator, &logits, 5, 3, 0);
    defer std.testing.allocator.free(greedy);
    const prefix = try prefixBeamDecode(std.testing.allocator, &logits, 5, 3, 0, 4);
    defer std.testing.allocator.free(prefix);
    try std.testing.expectEqualSlices(usize, &.{ 1, 2 }, greedy);
    try std.testing.expectEqualSlices(usize, &.{ 1, 2 }, prefix);
}
