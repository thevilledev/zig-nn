const std = @import("std");
const nn = @import("nn");
const config_mod = @import("config.zig");
const model_mod = @import("model.zig");
const loss_mod = @import("loss.zig");
const training = @import("training.zig");
const cli = @import("cli.zig");

const testing = std.testing;
const math = std.math;
const Config = config_mod.Config;
const Tokenizer = config_mod.Tokenizer;
const TinyGPT = model_mod.TinyGPT;
const FullTrainingOptions = training.FullTrainingOptions;
const prepareNextTokenTargets = loss_mod.prepareNextTokenTargets;
const crossEntropyLoss = loss_mod.crossEntropyLoss;
const trainOutputHeadOnCorpus = training.trainOutputHeadOnCorpus;
const trainFullOnCorpus = training.trainFullOnCorpus;
const trainFullOnCorpusDevice = training.trainFullOnCorpusDevice;
const scheduledLearningRate = training.scheduledLearningRate;
const causalSoftmax = model_mod.causalSoftmax;
const resolvedEvalChars = cli.resolvedEvalChars;
const trainingCorpus = cli.trainingCorpus;
const validationCorpus = cli.validationCorpus;

test "tokenizer encode decode round trip" {
    const allocator = testing.allocator;
    const text = "to be, or not!";

    const tokens = try Tokenizer.encode(allocator, text);
    defer allocator.free(tokens);
    const decoded = try Tokenizer.decode(allocator, tokens);
    defer allocator.free(decoded);

    try testing.expectEqualStrings(text, decoded);
}

test "causal mask removes future positions" {
    const scores = [_]f64{ 1.0, 2.0, 100.0, 200.0 };
    var probabilities: [scores.len]f64 = undefined;

    causalSoftmax(&scores, 1, &probabilities);

    try testing.expect(probabilities[0] > 0.0);
    try testing.expect(probabilities[1] > 0.0);
    try testing.expectEqual(@as(f64, 0.0), probabilities[2]);
    try testing.expectEqual(@as(f64, 0.0), probabilities[3]);
    try testing.expectApproxEqAbs(@as(f64, 1.0), probabilities[0] + probabilities[1], 1e-12);
}

test "forward returns sequence by vocab logits" {
    const allocator = testing.allocator;
    const config: Config = .{};
    var model = try TinyGPT.init(allocator, config, 123);
    defer model.deinit();

    const tokens = [_]usize{
        Tokenizer.encodeChar('t'),
        Tokenizer.encodeChar('o'),
        Tokenizer.encodeChar(' '),
        Tokenizer.encodeChar('b'),
        Tokenizer.encodeChar('e'),
    };

    var logits = try model.forward(&tokens);
    defer logits.deinit();

    try testing.expectEqual(tokens.len, logits.rows);
    try testing.expectEqual(Tokenizer.vocabSize(), logits.cols);
}

test "generation stays in vocabulary and appends requested tokens" {
    const allocator = testing.allocator;
    const config: Config = .{};
    var model = try TinyGPT.init(allocator, config, 42);
    defer model.deinit();

    const prompt = "to";
    const new_tokens = 12;
    const generated = try model.generateTokens(allocator, prompt, new_tokens, 1.0, 4);
    defer allocator.free(generated);

    try testing.expectEqual(prompt.len + new_tokens, generated.len);
    for (generated) |token| {
        try testing.expect(token < Tokenizer.vocabSize());
    }
}

test "cross entropy returns finite positive loss" {
    const allocator = testing.allocator;
    const config: Config = .{};
    var model = try TinyGPT.init(allocator, config, 7);
    defer model.deinit();

    const text = "to be";
    const tokens = try Tokenizer.encode(allocator, text);
    defer allocator.free(tokens);
    const targets = try prepareNextTokenTargets(allocator, tokens);
    defer allocator.free(targets);

    var logits = try model.forward(tokens[0 .. tokens.len - 1]);
    defer logits.deinit();

    const loss = try crossEntropyLoss(logits, targets);
    try testing.expect(loss > 0.0);
    try testing.expect(!math.isNan(loss));
    try testing.expect(loss != math.inf(f64));
}

test "demo output head training lowers corpus loss" {
    const allocator = testing.allocator;
    const config: Config = .{
        .block_size = 8,
        .n_layer = 1,
        .n_head = 2,
        .n_embd = 16,
    };
    var model = try TinyGPT.init(allocator, config, 99);
    defer model.deinit();

    const stats = try trainOutputHeadOnCorpus(&model, "to be or to learn.\n", 20, 0.5);

    try testing.expect(stats.samples > 0);
    try testing.expect(stats.final_loss < stats.initial_loss);
}

test "tiny gpt checkpoint round trip preserves logits" {
    const allocator = testing.allocator;
    const config: Config = .{
        .block_size = 8,
        .n_layer = 1,
        .n_head = 2,
        .n_embd = 16,
    };
    var model = try TinyGPT.init(allocator, config, 11);
    defer model.deinit();

    const path = "test_tiny_gpt_checkpoint.bin";
    try model.saveToFile(path);
    defer std.Io.Dir.cwd().deleteFile(std.Options.debug_io, path) catch {};

    var loaded = try TinyGPT.loadFromFile(allocator, path);
    defer loaded.deinit();

    const tokens = [_]usize{
        Tokenizer.encodeChar('t'),
        Tokenizer.encodeChar('o'),
        Tokenizer.encodeChar(' '),
    };

    var original_logits = try model.forward(&tokens);
    defer original_logits.deinit();
    var loaded_logits = try loaded.forward(&tokens);
    defer loaded_logits.deinit();

    try testing.expectEqual(original_logits.rows, loaded_logits.rows);
    try testing.expectEqual(original_logits.cols, loaded_logits.cols);
    for (original_logits.data, 0..) |value, idx| {
        try testing.expectApproxEqAbs(value, loaded_logits.data[idx], 1e-12);
    }
}

test "checkpoint metadata footer is embedded and remains loadable" {
    const allocator = testing.allocator;
    const config: Config = .{
        .block_size = 8,
        .n_layer = 1,
        .n_head = 2,
        .n_embd = 16,
    };
    var model = try TinyGPT.init(allocator, config, 12);
    defer model.deinit();

    const path = "test_tiny_gpt_checkpoint_metadata.bin";
    const metadata = "{\"schema\":\"test-metadata\"}\n";
    try model.saveToFileWithMetadata(path, metadata);
    defer std.Io.Dir.cwd().deleteFile(std.Options.debug_io, path) catch {};

    const bytes = try std.Io.Dir.cwd().readFileAlloc(std.Options.debug_io, path, allocator, .limited(1024 * 1024));
    defer allocator.free(bytes);
    try testing.expect(std.mem.indexOf(u8, bytes, metadata) != null);

    var loaded = try TinyGPT.loadFromFile(allocator, path);
    defer loaded.deinit();
    try testing.expectEqual(model.config.block_size, loaded.config.block_size);
}

test "eval split derives a deterministic holdout slice" {
    const text = "abcdefghijklmnopqrstuvwxyz";
    const eval_chars = resolvedEvalChars(text.len, 10, 0, 0.25);
    const train = trainingCorpus(text, 10, eval_chars);
    const eval = validationCorpus(text, 10, eval_chars).?;

    try testing.expectEqual(@as(usize, 7), eval_chars);
    try testing.expectEqualStrings("abcdefghij", train);
    try testing.expectEqualStrings("klmnopq", eval);
}

test "eval split remains disjoint when train limit covers corpus" {
    const text = "abcdefghijklmnopqrstuvwxyz";
    const eval_chars = resolvedEvalChars(text.len, 999, 0, 0.25);
    const train = trainingCorpus(text, 999, eval_chars);
    const eval = validationCorpus(text, 999, eval_chars).?;

    try testing.expectEqualStrings("abcdefghijklmnopqrs", train);
    try testing.expectEqualStrings("tuvwxyz", eval);
}

test "perplexity converts next-token loss" {
    try testing.expectApproxEqAbs(@as(f64, 1.0), cli.perplexityFromLoss(0.0), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 10.0), cli.perplexityFromLoss(@log(@as(f64, 10.0))), 1e-12);
}

test "coherent small preset sets model-only training defaults" {
    const options = try cli.parseArgs(&.{ "--preset", "coherent-small" });

    try testing.expect(options.train_full);
    try testing.expect(!options.train_demo_head);
    try testing.expect(!options.corpus_prior);
    try testing.expectEqual(cli.CorpusPreset.tinystories, options.corpus);
    try testing.expectEqual(@as(usize, 64), options.model_config.block_size);
    try testing.expectEqual(@as(usize, 4), options.model_config.n_layer);
    try testing.expectEqual(@as(usize, 4), options.model_config.n_head);
    try testing.expectEqual(@as(usize, 128), options.model_config.n_embd);
    try testing.expectEqual(@as(usize, 3000), options.full_train_steps);
    try testing.expectApproxEqAbs(@as(f64, 0.001), options.full_learning_rate, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.0001), options.min_learning_rate, 1e-12);
    try testing.expectEqual(config_mod.LearningRateSchedule.cosine, options.lr_schedule);
    try testing.expectEqual(config_mod.OptimizerKind.adamw, options.full_optimizer);
    try testing.expectEqual(@as(usize, 0), options.train_chars);
    try testing.expectEqual(@as(usize, 64), options.eval_windows);
}

test "explicit args override coherent small preset regardless of order" {
    const options = try cli.parseArgs(&.{
        "--corpus",       "toy",
        "--block-size",   "8",
        "--preset",       "coherent-small",
        "--top-k",        "4",
        "--corpus-prior",
    });

    try testing.expectEqual(cli.CorpusPreset.toy, options.corpus);
    try testing.expectEqual(@as(usize, 8), options.model_config.block_size);
    try testing.expectEqual(@as(usize, 4), options.top_k);
    try testing.expect(options.corpus_prior);
    try testing.expect(options.train_full);
}

test "learning rate schedule reaches warmup peak and final floor" {
    const options: FullTrainingOptions = .{
        .steps = 6,
        .learning_rate = 0.1,
        .min_learning_rate = 0.01,
        .lr_schedule = .linear,
        .warmup_steps = 2,
    };

    try testing.expectApproxEqAbs(0.05, scheduledLearningRate(options, 0), 1e-12);
    try testing.expectApproxEqAbs(0.1, scheduledLearningRate(options, 1), 1e-12);
    try testing.expectApproxEqAbs(0.01, scheduledLearningRate(options, 5), 1e-12);
}

test "full transformer training lowers corpus loss" {
    const allocator = testing.allocator;
    const config: Config = .{
        .block_size = 4,
        .n_layer = 1,
        .n_head = 2,
        .n_embd = 8,
    };
    var model = try TinyGPT.init(allocator, config, 21);
    defer model.deinit();

    const stats = try trainFullOnCorpus(&model, "abababababababab\n", "babababababababa\n", .{
        .steps = 40,
        .learning_rate = 0.05,
        .context_len = config.block_size,
        .stride = 1,
        .batch_size = 2,
        .optimizer = .adamw,
        .weight_decay = 0.01,
        .random_seed = 21,
        .lr_schedule = .cosine,
        .min_learning_rate = 0.01,
        .eval_windows = 4,
    });

    try testing.expect(stats.tokens_seen > 0);
    try testing.expectEqual(stats.steps, stats.updates);
    try testing.expect(stats.final_loss < stats.initial_loss);
    try testing.expect(stats.validation_initial_loss != null);
    try testing.expectEqual(@as(usize, 4), stats.eval_windows);
    try testing.expectApproxEqAbs(0.01, stats.learning_rate_final, 1e-12);
}

test "device decoder adapter preserves TinyGPT logits" {
    const allocator = testing.allocator;
    const config: Config = .{
        .block_size = 4,
        .n_layer = 1,
        .n_head = 2,
        .n_embd = 8,
    };
    var model = try TinyGPT.init(allocator, config, 77);
    defer model.deinit();
    var expected = try model.forward(&.{ 0, 1, 2 });
    defer expected.deinit();

    var device = try nn.Device.init(allocator, .cpu);
    defer device.deinit();
    var context = nn.ExecutionContext.init(&device);
    var decoder = try model.toDeviceDecoder(&context);
    defer decoder.deinit();
    var tokens = try context.upload(&.{ 3, 1 }, &.{ 0, 1, 2 });
    defer tokens.deinit();
    var positions = try context.upload(&.{ 3, 1 }, &.{ 0, 1, 2 });
    defer positions.deinit();
    try context.beginBatch();
    var actual = try decoder.forward(&context, tokens, positions);
    try context.endBatch();
    defer actual.deinit();
    const actual_values = try allocator.alloc(f32, actual.elementCount());
    defer allocator.free(actual_values);
    try context.readback(actual, actual_values);

    for (expected.data, actual_values) |expected_value, actual_value| {
        try testing.expectApproxEqAbs(@as(f32, @floatCast(expected_value)), actual_value, 5e-4);
    }

    const original_weight = model.lm_head.weights.data[0];
    decoder.language_model_head.weights.fill(0.25);
    try model.syncFromDeviceDecoder(&decoder);
    try testing.expect(model.lm_head.weights.data[0] != original_weight);
    try testing.expectApproxEqAbs(@as(f64, 0.25), model.lm_head.weights.data[0], 1e-7);
}

test "CLI parses explicit tensor backend" {
    const options = try cli.parseArgs(&.{ "--train-full", "--backend", "cpu", "--optimizer", "sgd" });
    try testing.expectEqual(nn.DevicePreference.cpu, options.device_preference.?);
    try testing.expectError(error.UnknownBackend, cli.parseArgs(&.{ "--backend", "quantum" }));
}

test "device TinyGPT training syncs learned parameters back" {
    const allocator = testing.allocator;
    const config: Config = .{
        .block_size = 3,
        .n_layer = 1,
        .n_head = 1,
        .n_embd = 4,
    };
    var model = try TinyGPT.init(allocator, config, 31);
    defer model.deinit();
    const stats = try trainFullOnCorpusDevice(&model, "abababababababab\n", null, .{
        .steps = 30,
        .learning_rate = 0.03,
        .context_len = config.block_size,
        .optimizer = .sgd,
        .batch_size = 1,
        .random_seed = 31,
        .eval_windows = 4,
    }, .cpu);
    try testing.expect(stats.final_loss < stats.initial_loss);
    try testing.expectEqual(@as(usize, 30), stats.updates);
    try testing.expectEqual(@as(usize, 90), stats.tokens_seen);
}

test "device KV generation matches greedy legacy generation" {
    const allocator = testing.allocator;
    const config: Config = .{
        .block_size = 4,
        .n_layer = 1,
        .n_head = 2,
        .n_embd = 8,
    };
    var legacy = try TinyGPT.init(allocator, config, 44);
    defer legacy.deinit();
    var device_model = try TinyGPT.init(allocator, config, 44);
    defer device_model.deinit();
    const expected = try legacy.generateTokens(allocator, "ab", 6, 1.0, 1);
    defer allocator.free(expected);
    const actual = try device_model.generateTokensDevice(allocator, "ab", 6, 1.0, 1, .cpu);
    defer allocator.free(actual);
    try testing.expectEqualSlices(usize, expected, actual);
}
