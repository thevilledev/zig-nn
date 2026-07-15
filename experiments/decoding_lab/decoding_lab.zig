const std = @import("std");
const nn = @import("nn");

const vocabulary = [_][]const u8{ "the", "cat", "dog", "sat", "ran", "." };
const logits = [_]f64{ 4.2, 3.7, 3.0, 2.0, 1.0, 0.0 };

const Result = struct {
    greedy_token: usize,
    top_k_tokens: [3]usize,
    top_k_probabilities: [3]f64,
    nucleus_tokens: [vocabulary.len]usize,
    nucleus_probabilities: [vocabulary.len]f64,
    nucleus_len: usize,
    repetition_winner: usize,
    sample_counts: [vocabulary.len]usize,
};

pub fn main(init: std.process.Init) !void {
    const result = try runLab(init.gpa);

    std.debug.print("Decoding strategies over one next-token logit vector\n", .{});
    std.debug.print("greedy: {s}\n\n", .{vocabulary[result.greedy_token]});
    std.debug.print("top-k (k=3):\n", .{});
    for (result.top_k_tokens, result.top_k_probabilities) |token, probability| {
        std.debug.print("  {s: <4} {d:.3}\n", .{ vocabulary[token], probability });
    }
    std.debug.print("\nnucleus (p=0.75):\n", .{});
    for (result.nucleus_tokens[0..result.nucleus_len], result.nucleus_probabilities[0..result.nucleus_len]) |token, probability| {
        std.debug.print("  {s: <4} {d:.3}\n", .{ vocabulary[token], probability });
    }
    std.debug.print("\n1,000 seeded nucleus samples:\n", .{});
    for (vocabulary, result.sample_counts) |token, count| {
        std.debug.print("  {s: <4} {d}\n", .{ token, count });
    }
    std.debug.print(
        "\nrepetition penalty: after generating '{s}', greedy changes to '{s}'\n" ++
            "lesson: top-k caps candidate count; top-p adapts it to probability mass\n",
        .{ vocabulary[result.greedy_token], vocabulary[result.repetition_winner] },
    );
}

fn runLab(allocator: std.mem.Allocator) !Result {
    const greedy = try nn.Decoding.distribution(allocator, &logits, &.{}, .{ .temperature = 0 });
    defer allocator.free(greedy);
    const top_k = try nn.Decoding.distribution(allocator, &logits, &.{}, .{ .top_k = 3 });
    defer allocator.free(top_k);
    const nucleus = try nn.Decoding.distribution(allocator, &logits, &.{}, .{ .top_p = 0.75 });
    defer allocator.free(nucleus);
    const penalized = try nn.Decoding.distribution(
        allocator,
        &logits,
        &.{greedy[0].token},
        .{ .temperature = 0, .repetition_penalty = 2 },
    );
    defer allocator.free(penalized);

    var result: Result = .{
        .greedy_token = greedy[0].token,
        .top_k_tokens = undefined,
        .top_k_probabilities = undefined,
        .nucleus_tokens = @splat(0),
        .nucleus_probabilities = @splat(0),
        .nucleus_len = nucleus.len,
        .repetition_winner = penalized[0].token,
        .sample_counts = @splat(0),
    };
    for (top_k, 0..) |candidate, index| {
        result.top_k_tokens[index] = candidate.token;
        result.top_k_probabilities[index] = candidate.probability;
    }
    for (nucleus, 0..) |candidate, index| {
        result.nucleus_tokens[index] = candidate.token;
        result.nucleus_probabilities[index] = candidate.probability;
    }

    var prng = std.Random.DefaultPrng.init(42);
    for (0..1000) |_| {
        const token = try nn.Decoding.sample(allocator, prng.random(), &logits, &.{}, .{ .top_p = 0.75 });
        result.sample_counts[token] += 1;
    }
    return result;
}

test "decoding lab contrasts fixed and adaptive truncation" {
    const result = try runLab(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 0), result.greedy_token);
    try std.testing.expectEqual(@as(usize, 3), result.top_k_tokens.len);
    try std.testing.expectEqual(@as(usize, 2), result.nucleus_len);
    try std.testing.expectEqual(@as(usize, 1), result.repetition_winner);
    try std.testing.expect(result.sample_counts[0] > result.sample_counts[1]);
    try std.testing.expect(result.sample_counts[1] > 0);
    for (result.sample_counts[2..]) |count| try std.testing.expectEqual(@as(usize, 0), count);
}
