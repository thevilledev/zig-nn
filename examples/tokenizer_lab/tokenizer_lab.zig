const std = @import("std");
const nn = @import("nn");

const default_merges: usize = 64;
const default_minimum_frequency: usize = 2;
const default_text = "learning learners learned — oppiminen";

const corpus =
    "learn learning learner learned learners learnable " ++
    "token tokens tokenize tokenized tokenizer tokenization " ++
    "language languages linguistic linguistics " ++
    "learn learning learner learned learners learnable " ++
    "token tokens tokenize tokenized tokenizer tokenization " ++
    "language languages linguistic linguistics " ++
    "neural network neural networks networked networking " ++
    "learn learning learner learned learners learnable " ++
    "token tokens tokenize tokenized tokenizer tokenization";

const Options = struct {
    maximum_merges: usize = default_merges,
    minimum_frequency: usize = default_minimum_frequency,
    text: []const u8 = default_text,
};

const Experiment = struct {
    allocator: std.mem.Allocator,
    tokenizer: nn.Text.BpeTokenizer,
    sample_tokens: []nn.Text.TokenId,
    decoded: []u8,
    corpus_bytes: usize,
    corpus_words: usize,
    corpus_bpe_tokens: usize,
    serialized_bytes: usize,

    fn deinit(self: *Experiment) void {
        self.tokenizer.deinit();
        self.allocator.free(self.sample_tokens);
        self.allocator.free(self.decoded);
        self.* = undefined;
    }

    fn compressionRatio(self: Experiment) f32 {
        return @as(f32, @floatFromInt(self.corpus_bpe_tokens)) /
            @as(f32, @floatFromInt(self.corpus_bytes));
    }
};

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const options = parseArgs(args[1..]) catch |err| {
        printUsage();
        if (err == error.HelpRequested) return;
        return err;
    };
    var experiment = try runExperiment(init.gpa, options);
    defer experiment.deinit();

    std.debug.print("Byte-level BPE tokenizer lab\n", .{});
    std.debug.print(
        "corpus: {d} bytes, {d} whitespace words\n" ++
            "learned merges: {d}, vocabulary: {d} including controls\n" ++
            "byte tokens: {d} -> BPE tokens: {d} ({d:.1}% of byte count)\n" ++
            "serialized vocabulary: {d} bytes\n\n",
        .{
            experiment.corpus_bytes,
            experiment.corpus_words,
            experiment.tokenizer.mergeCount(),
            experiment.tokenizer.vocabularySize(),
            experiment.corpus_bytes,
            experiment.corpus_bpe_tokens,
            experiment.compressionRatio() * 100,
            experiment.serialized_bytes,
        },
    );
    std.debug.print("sample: {s}\n", .{options.text});
    std.debug.print("tokens: ", .{});
    for (experiment.sample_tokens) |token| {
        const spelling = try experiment.tokenizer.tokenBytes(init.gpa, token);
        defer init.gpa.free(spelling);
        std.debug.print("{d}:", .{token});
        printSpelling(spelling);
        std.debug.print(" ", .{});
    }
    std.debug.print("\nround trip: {s}\n\n", .{experiment.decoded});
    std.debug.print("first learned pieces:\n", .{});
    const shown = @min(experiment.tokenizer.mergeCount(), 12);
    for (0..shown) |index| {
        const token: nn.Text.TokenId = @intCast(nn.Text.byte_vocabulary_size + index);
        const spelling = try experiment.tokenizer.tokenBytes(init.gpa, token);
        defer init.gpa.free(spelling);
        std.debug.print("  {d}: ", .{token});
        printSpelling(spelling);
        std.debug.print("\n", .{});
    }
    std.debug.print(
        "control IDs: pad={d}, bos={d}, eos={d}, unknown={d}\n",
        .{
            experiment.tokenizer.specialTokenId(.padding),
            experiment.tokenizer.specialTokenId(.beginning_of_sequence),
            experiment.tokenizer.specialTokenId(.end_of_sequence),
            experiment.tokenizer.specialTokenId(.unknown),
        },
    );
}

fn parseArgs(args: []const []const u8) !Options {
    var options: Options = .{};
    var index: usize = 0;
    while (index < args.len) : (index += 1) {
        const arg = args[index];
        if (std.mem.eql(u8, arg, "--merges")) {
            index += 1;
            if (index >= args.len) return error.MissingArgument;
            options.maximum_merges = try std.fmt.parseInt(usize, args[index], 10);
        } else if (std.mem.eql(u8, arg, "--minimum-frequency")) {
            index += 1;
            if (index >= args.len) return error.MissingArgument;
            options.minimum_frequency = try std.fmt.parseInt(usize, args[index], 10);
            if (options.minimum_frequency < 2) return error.InvalidMinimumFrequency;
        } else if (std.mem.eql(u8, arg, "--text")) {
            index += 1;
            if (index >= args.len) return error.MissingArgument;
            options.text = args[index];
        } else if (std.mem.eql(u8, arg, "--help")) {
            return error.HelpRequested;
        } else {
            return error.UnknownArgument;
        }
    }
    return options;
}

fn printUsage() void {
    std.debug.print(
        \\Usage: tokenizer_lab [options]
        \\
        \\Options:
        \\  --merges <count>             maximum learned BPE merges (default: 64)
        \\  --minimum-frequency <count>  minimum pair frequency (default: 2)
        \\  --text <value>               text to encode and inspect
        \\  --help                       show this help
        \\
    , .{});
}

fn runExperiment(allocator: std.mem.Allocator, options: Options) !Experiment {
    var tokenizer = try nn.Text.BpeTokenizer.train(allocator, corpus, .{
        .maximum_merges = options.maximum_merges,
        .minimum_frequency = options.minimum_frequency,
    });
    errdefer tokenizer.deinit();
    const corpus_tokens = try tokenizer.encode(allocator, corpus);
    defer allocator.free(corpus_tokens);
    const sample_tokens = try tokenizer.encode(allocator, options.text);
    errdefer allocator.free(sample_tokens);
    const decoded = try tokenizer.decode(allocator, sample_tokens);
    errdefer allocator.free(decoded);
    if (!std.mem.eql(u8, options.text, decoded)) return error.RoundTripFailed;

    var storage: [8192]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&storage);
    try tokenizer.write(&writer);
    const serialized = writer.buffered();
    var reader: std.Io.Reader = .fixed(serialized);
    var restored = try nn.Text.BpeTokenizer.read(allocator, &reader);
    defer restored.deinit();
    const restored_tokens = try restored.encode(allocator, options.text);
    defer allocator.free(restored_tokens);
    if (!std.mem.eql(nn.Text.TokenId, sample_tokens, restored_tokens)) {
        return error.SerializationChangedTokens;
    }
    return .{
        .allocator = allocator,
        .tokenizer = tokenizer,
        .sample_tokens = sample_tokens,
        .decoded = decoded,
        .corpus_bytes = corpus.len,
        .corpus_words = countWords(corpus),
        .corpus_bpe_tokens = corpus_tokens.len,
        .serialized_bytes = serialized.len,
    };
}

fn countWords(text: []const u8) usize {
    var words = std.mem.tokenizeAny(u8, text, " \t\r\n");
    var count: usize = 0;
    while (words.next() != null) count += 1;
    return count;
}

fn printSpelling(bytes: []const u8) void {
    std.debug.print("'", .{});
    if (std.unicode.utf8ValidateSlice(bytes)) {
        std.debug.print("{s}", .{bytes});
    } else {
        for (bytes) |byte| std.debug.print("\\x{x}", .{byte});
    }
    std.debug.print("'", .{});
}

test "tokenizer lab options validate merge frequency" {
    const options = try parseArgs(&.{
        "--merges",
        "12",
        "--minimum-frequency",
        "3",
        "--text",
        "hello",
    });
    try std.testing.expectEqual(@as(usize, 12), options.maximum_merges);
    try std.testing.expectEqual(@as(usize, 3), options.minimum_frequency);
    try std.testing.expectEqualStrings("hello", options.text);
    try std.testing.expectError(
        error.InvalidMinimumFrequency,
        parseArgs(&.{ "--minimum-frequency", "1" }),
    );
}

test "tokenizer lab demonstrates compression and serialization" {
    var experiment = try runExperiment(std.testing.allocator, .{});
    defer experiment.deinit();
    try std.testing.expect(experiment.tokenizer.mergeCount() > 20);
    try std.testing.expect(experiment.compressionRatio() < 0.5);
    try std.testing.expect(experiment.serialized_bytes > 12);
    try std.testing.expectEqualStrings(default_text, experiment.decoded);
}
