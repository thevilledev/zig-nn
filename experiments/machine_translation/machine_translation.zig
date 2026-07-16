const std = @import("std");
const nn = @import("nn");

const vocabulary = [_][]const u8{
    "<pad>", "<bos>",  "<eos>",
    "i",     "like",   "cats",
    "dogs",  "je",     "aime",
    "chats", "chiens",
};
const pad: usize = 0;
const bos: usize = 1;
const eos: usize = 2;
const maximum_source: usize = 3;
const maximum_target: usize = 4;
const default_steps: usize = 180;

const Example = struct {
    source: [maximum_source]usize,
    source_length: usize,
    decoder_input: [maximum_target]usize,
    target: [maximum_target]usize,
    target_length: usize,
};

const examples = [_]Example{
    .{
        .source = .{ 3, 4, 5 },
        .source_length = 3,
        .decoder_input = .{ bos, 7, 8, 9 },
        .target = .{ 7, 8, 9, eos },
        .target_length = 4,
    },
    .{
        .source = .{ 3, 4, 6 },
        .source_length = 3,
        .decoder_input = .{ bos, 7, 8, 10 },
        .target = .{ 7, 8, 10, eos },
        .target_length = 4,
    },
    .{
        .source = .{ 3, 5, pad },
        .source_length = 2,
        .decoder_input = .{ bos, 7, 9, pad },
        .target = .{ 7, 9, eos, pad },
        .target_length = 3,
    },
    .{
        .source = .{ 3, 6, pad },
        .source_length = 2,
        .decoder_input = .{ bos, 7, 10, pad },
        .target = .{ 7, 10, eos, pad },
        .target_length = 3,
    },
};

const Result = struct {
    allocator: std.mem.Allocator,
    initial_loss: f32,
    final_loss: f32,
    greedy: []usize,
    beam: []usize,
    alignment: []f32,
    alignment_rows: usize,

    fn deinit(self: *Result) void {
        self.allocator.free(self.greedy);
        self.allocator.free(self.beam);
        self.allocator.free(self.alignment);
        self.* = undefined;
    }
};

const DecodeContext = struct {
    model: *const nn.Translation.TeacherForcedEncoderDecoder,
    source: []const usize,
    source_length: usize,

    fn score(self: *const DecodeContext, prefix: []const usize, output: []f64) !void {
        try self.model.nextLogProbabilities(
            self.source,
            self.source_length,
            prefix,
            output,
        );
    }
};

pub fn main(init: std.process.Init) !void {
    const result_steps = try parseArgs(
        try init.minimal.args.toSlice(init.arena.allocator()),
    );
    var result = try runExperiment(init.gpa, result_steps);
    defer result.deinit();

    const sample = examples[0];
    std.debug.print("Teacher-forced machine translation\n", .{});
    std.debug.print(
        "stack: positional encoder -> causal decoder -> masked cross-attention\n" ++
            "objective: token cross-entropy {d:.4} -> {d:.4}\n",
        .{ result.initial_loss, result.final_loss },
    );
    std.debug.print("source: ", .{});
    printTokens(sample.source[0..sample.source_length]);
    std.debug.print("greedy: ", .{});
    printTokens(result.greedy);
    std.debug.print("beam:   ", .{});
    printTokens(result.beam);
    std.debug.print("\ncross-attention alignment (target rows, source columns):\n", .{});
    std.debug.print("          ", .{});
    for (sample.source[0..sample.source_length]) |token| {
        std.debug.print("{s: >8}", .{vocabulary[token]});
    }
    std.debug.print("\n", .{});
    for (0..result.alignment_rows) |row| {
        std.debug.print("{s: >8}  ", .{vocabulary[sample.target[row]]});
        for (result.alignment[row * sample.source_length ..][0..sample.source_length]) |weight| {
            std.debug.print("{d: >8.3}", .{weight});
        }
        std.debug.print("\n", .{});
    }
    std.debug.print(
        "\nbeam width 3 uses length-normalized sequence scores; padding is masked before attention\n",
        .{},
    );
}

fn parseArgs(args: []const []const u8) !usize {
    var steps = default_steps;
    var index: usize = 1;
    while (index < args.len) : (index += 1) {
        if (std.mem.eql(u8, args[index], "--steps")) {
            index += 1;
            if (index >= args.len) return error.MissingArgument;
            steps = try std.fmt.parseInt(usize, args[index], 10);
            if (steps == 0) return error.InvalidStepCount;
        } else if (std.mem.eql(u8, args[index], "--help")) {
            std.debug.print("Usage: machine_translation [--steps <count>]\n", .{});
            return error.HelpRequested;
        } else {
            return error.UnknownArgument;
        }
    }
    return steps;
}

fn runExperiment(allocator: std.mem.Allocator, steps: usize) !Result {
    var prng = std.Random.DefaultPrng.init(42);
    var model = try nn.Translation.TeacherForcedEncoderDecoder.init(
        allocator,
        .{
            .vocabulary_size = vocabulary.len,
            .maximum_tokens = maximum_target,
            .padding_token = pad,
            .beginning_token = bos,
            .end_token = eos,
        },
        prng.random(),
    );
    defer model.deinit();
    const initial_loss = try datasetLoss(&model);
    for (0..steps) |_| {
        for (examples) |example| {
            var objective = try model.lossAndGradient(
                &example.source,
                example.source_length,
                example.decoder_input[0..example.target_length],
                example.target[0..example.target_length],
            );
            defer objective.deinit();
            try model.applyGradient(objective.gradients, 0.3);
        }
    }
    const final_loss = try datasetLoss(&model);
    const sample = examples[0];
    const decode_context: DecodeContext = .{
        .model = &model,
        .source = &sample.source,
        .source_length = sample.source_length,
    };
    var greedy_result = try nn.Decoding.beamSearch(
        allocator,
        bos,
        eos,
        vocabulary.len,
        .{ .width = 1, .maximum_tokens = maximum_target },
        &decode_context,
        DecodeContext.score,
    );
    defer greedy_result.deinit();
    var beam_result = try nn.Decoding.beamSearch(
        allocator,
        bos,
        eos,
        vocabulary.len,
        .{
            .width = 3,
            .minimum_tokens = sample.source_length + 1,
            .maximum_tokens = maximum_target,
            .length_penalty = 2.0,
        },
        &decode_context,
        DecodeContext.score,
    );
    defer beam_result.deinit();
    const greedy = try allocator.dupe(usize, greedy_result.tokens);
    errdefer allocator.free(greedy);
    const beam = try allocator.dupe(usize, beam_result.tokens);
    errdefer allocator.free(beam);
    const alignment_rows = @min(sample.target_length, beam.len);
    const alignment = try allocator.alloc(
        f32,
        alignment_rows * sample.source_length,
    );
    errdefer allocator.free(alignment);
    var prefix: [maximum_target]usize = undefined;
    prefix[0] = bos;
    for (0..alignment_rows) |row| {
        const weights = try model.crossAttentionAlignment(
            allocator,
            &sample.source,
            sample.source_length,
            prefix[0 .. row + 1],
            row,
        );
        defer allocator.free(weights);
        @memcpy(
            alignment[row * sample.source_length ..][0..sample.source_length],
            weights,
        );
        if (row + 1 < prefix.len and row < beam.len) prefix[row + 1] = beam[row];
    }
    return .{
        .allocator = allocator,
        .initial_loss = initial_loss,
        .final_loss = final_loss,
        .greedy = greedy,
        .beam = beam,
        .alignment = alignment,
        .alignment_rows = alignment_rows,
    };
}

fn datasetLoss(model: *const nn.Translation.TeacherForcedEncoderDecoder) !f32 {
    var total: f32 = 0;
    for (examples) |example| {
        var objective = try model.lossAndGradient(
            &example.source,
            example.source_length,
            example.decoder_input[0..example.target_length],
            example.target[0..example.target_length],
        );
        defer objective.deinit();
        total += objective.loss;
    }
    return total / @as(f32, @floatFromInt(examples.len));
}

fn printTokens(tokens: []const usize) void {
    for (tokens) |token| {
        std.debug.print("{s} ", .{vocabulary[token]});
        if (token == eos) break;
    }
    std.debug.print("\n", .{});
}

test "teacher forcing produces an exact beam translation" {
    var result = try runExperiment(std.testing.allocator, default_steps);
    defer result.deinit();
    try std.testing.expect(result.final_loss < result.initial_loss * 0.15);
    try std.testing.expectEqualSlices(usize, &.{ 7, 8, 9, eos }, result.beam);
    try std.testing.expectEqualSlices(usize, &.{ 7, 8, 9, eos }, result.greedy);
    for (0..3) |row| {
        const weights = result.alignment[row * 3 ..][0..3];
        try std.testing.expect(weights[row] > 0.9);
    }
}
