const std = @import("std");
const nn = @import("nn");

const tags: usize = 3;
const batch_size: usize = 4;
const maximum_tokens: usize = 5;
const default_steps: usize = 80;

const Tag = enum(usize) { outside, begin_person, inside_person };

const lengths = [_]usize{ 4, 3, 5, 2 };
const targets = [_]usize{
    tag(.outside),      tag(.begin_person),  tag(.inside_person), tag(.outside),       0,
    tag(.begin_person), tag(.inside_person), tag(.outside),       0,                   0,
    tag(.outside),      tag(.outside),       tag(.begin_person),  tag(.inside_person), tag(.outside),
    tag(.begin_person), tag(.outside),       0,                   0,                   0,
};

const emissions = [_]f32{
    3, 0, 0,   0, 3,   0.5, 0, 2.4, 2.0, 3, 0,   0,   0, 0, 0,
    0, 3, 0.5, 0, 2.4, 2.0, 3, 0,   0,   0, 0,   0,   0, 0, 0,
    3, 0, 0,   3, 0,   0,   0, 3,   0.5, 0, 2.4, 2.0, 3, 0, 0,
    0, 3, 0.5, 3, 0,   0,   0, 0,   0,   0, 0,   0,   0, 0, 0,
};

const sentences = [_][]const []const u8{
    &.{ "met", "Alice", "Smith", "today" },
    &.{ "Bob", "Jones", "works" },
    &.{ "we", "called", "Carol", "White", "yesterday" },
    &.{ "Dana", "arrived" },
};

const Result = struct {
    initial_loss: f32,
    final_loss: f32,
    independent_accuracy: f32,
    crf_accuracy: f32,
    padding_loss_difference: f32,
    begin_to_inside: f32,
    decoded: [batch_size][maximum_tokens]usize,
};

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const steps = parseArgs(args[1..]) catch |err| {
        printUsage();
        if (err == error.HelpRequested) return;
        return err;
    };
    const result = try runExperiment(init.gpa, steps);
    std.debug.print("Linear-chain CRF sequence-tagging lesson\n", .{});
    std.debug.print(
        "negative log-likelihood: {d:.4} -> {d:.4}\n" ++
            "local argmax accuracy: {d:.1}%\n" ++
            "Viterbi CRF accuracy: {d:.1}%\n" ++
            "learned B-PER -> I-PER score: {d:.3}\n" ++
            "loss change after corrupting padding: {d:.8}\n\n",
        .{
            result.initial_loss,
            result.final_loss,
            result.independent_accuracy * 100,
            result.crf_accuracy * 100,
            result.begin_to_inside,
            result.padding_loss_difference,
        },
    );
    for (sentences, lengths, result.decoded) |sentence, length, decoded| {
        for (sentence[0..length], decoded[0..length]) |word, predicted| {
            std.debug.print("{s}/{s} ", .{ word, tagName(predicted) });
        }
        std.debug.print("\n", .{});
    }
}

fn parseArgs(args: []const []const u8) !usize {
    var steps: usize = default_steps;
    var index: usize = 0;
    while (index < args.len) : (index += 1) {
        if (std.mem.eql(u8, args[index], "--steps")) {
            index += 1;
            if (index >= args.len) return error.MissingArgument;
            steps = try std.fmt.parseInt(usize, args[index], 10);
            if (steps == 0) return error.InvalidStepCount;
        } else if (std.mem.eql(u8, args[index], "--help")) {
            return error.HelpRequested;
        } else {
            return error.UnknownArgument;
        }
    }
    return steps;
}

fn printUsage() void {
    std.debug.print("Usage: sequence_tagging [--steps count]\n", .{});
}

fn runExperiment(allocator: std.mem.Allocator, steps: usize) !Result {
    var crf = try nn.Structured.LinearChainCrf.init(allocator, tags);
    defer crf.deinit();
    const initial_loss = try crf.negativeLogLikelihood(&emissions, &targets, &lengths, maximum_tokens);
    const independent_accuracy = independentAccuracy();
    for (0..steps) |_| {
        var objective = try crf.lossAndGradient(&emissions, &targets, &lengths, maximum_tokens);
        defer objective.deinit();
        try crf.applyGradient(objective.gradients, 0.22);
    }
    const final_loss = try crf.negativeLogLikelihood(&emissions, &targets, &lengths, maximum_tokens);
    var decoded = [_][maximum_tokens]usize{[_]usize{0} ** maximum_tokens} ** batch_size;
    var correct: usize = 0;
    var token_count: usize = 0;
    for (lengths, 0..) |length, batch_index| {
        const emission_offset = batch_index * maximum_tokens * tags;
        _ = try crf.decode(emissions[emission_offset..], length, &decoded[batch_index]);
        for (decoded[batch_index][0..length], targets[batch_index * maximum_tokens .. batch_index * maximum_tokens + length]) |predicted, expected| {
            if (predicted == expected) correct += 1;
            token_count += 1;
        }
    }
    var corrupted_padding = emissions;
    for (lengths, 0..) |length, batch_index| {
        for (length..maximum_tokens) |token_index| {
            const offset = (batch_index * maximum_tokens + token_index) * tags;
            corrupted_padding[offset] = -1000;
            corrupted_padding[offset + 1] = 1000;
            corrupted_padding[offset + 2] = 500;
        }
    }
    const corrupted_loss = try crf.negativeLogLikelihood(&corrupted_padding, &targets, &lengths, maximum_tokens);
    return .{
        .initial_loss = initial_loss,
        .final_loss = final_loss,
        .independent_accuracy = independent_accuracy,
        .crf_accuracy = @as(f32, @floatFromInt(correct)) / @as(f32, @floatFromInt(token_count)),
        .padding_loss_difference = @abs(corrupted_loss - final_loss),
        .begin_to_inside = crf.transitions[tag(.begin_person) * tags + tag(.inside_person)],
        .decoded = decoded,
    };
}

fn independentAccuracy() f32 {
    var correct: usize = 0;
    var count: usize = 0;
    for (lengths, 0..) |length, batch_index| {
        for (0..length) |token_index| {
            const offset = (batch_index * maximum_tokens + token_index) * tags;
            var predicted: usize = 0;
            for (1..tags) |candidate| if (emissions[offset + candidate] > emissions[offset + predicted]) {
                predicted = candidate;
            };
            if (predicted == targets[batch_index * maximum_tokens + token_index]) correct += 1;
            count += 1;
        }
    }
    return @as(f32, @floatFromInt(correct)) / @as(f32, @floatFromInt(count));
}

fn tag(value: Tag) usize {
    return @intFromEnum(value);
}

fn tagName(value: usize) []const u8 {
    return switch (@as(Tag, @enumFromInt(value))) {
        .outside => "O",
        .begin_person => "B-PER",
        .inside_person => "I-PER",
    };
}

test "CRF learns BIO transitions and ignores padded emissions" {
    const result = try runExperiment(std.testing.allocator, 60);
    try std.testing.expect(result.final_loss < result.initial_loss * 0.5);
    try std.testing.expect(result.independent_accuracy < 0.85);
    try std.testing.expectEqual(@as(f32, 1), result.crf_accuracy);
    try std.testing.expectEqual(@as(f32, 0), result.padding_loss_difference);
    try std.testing.expect(result.begin_to_inside > 0);
}
