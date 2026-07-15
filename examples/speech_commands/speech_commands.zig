const std = @import("std");
const nn = @import("nn");
const data = @import("dataset.zig");
const speech = @import("model.zig");

const default_data_dir = "data/mini_speech_commands";
const default_checkpoint = "speech-commands.bin";

const Options = struct {
    train: bool = false,
    data_dir: []const u8 = default_data_dir,
    output: []const u8 = default_checkpoint,
    model: []const u8 = default_checkpoint,
    inputs: []const []const u8 = &.{},
    epochs: usize = 12,
    batch_size: usize = 64,
    learning_rate: f32 = 0.003,
    seed: u64 = 42,
    backend: nn.DevicePreference = .auto,
    top_k: usize = 3,
};

const Metrics = struct {
    loss: f32,
    accuracy: f32,
    confusion: [speech.class_count * speech.class_count]usize = @splat(0),
};

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const options = parseArgs(init.arena.allocator(), args[1..]) catch |err| {
        printUsage();
        if (err == error.HelpRequested) return;
        return err;
    };
    if (options.train) {
        try train(init.gpa, options);
    } else {
        try infer(init.gpa, options);
    }
}

fn parseArgs(allocator: std.mem.Allocator, args: []const []const u8) !Options {
    var options: Options = .{};
    var inputs: std.ArrayList([]const u8) = .empty;
    defer inputs.deinit(allocator);
    var index: usize = 0;
    while (index < args.len) : (index += 1) {
        const argument = args[index];
        if (std.mem.eql(u8, argument, "--train")) {
            options.train = true;
        } else if (std.mem.eql(u8, argument, "--data-dir")) {
            index += 1;
            if (index >= args.len) return error.MissingArgument;
            options.data_dir = args[index];
        } else if (std.mem.eql(u8, argument, "--output")) {
            index += 1;
            if (index >= args.len) return error.MissingArgument;
            options.output = args[index];
        } else if (std.mem.eql(u8, argument, "--model")) {
            index += 1;
            if (index >= args.len) return error.MissingArgument;
            options.model = args[index];
        } else if (std.mem.eql(u8, argument, "--input")) {
            index += 1;
            if (index >= args.len) return error.MissingArgument;
            try inputs.append(allocator, args[index]);
        } else if (std.mem.eql(u8, argument, "--epochs")) {
            index += 1;
            if (index >= args.len) return error.MissingArgument;
            options.epochs = try std.fmt.parseInt(usize, args[index], 10);
        } else if (std.mem.eql(u8, argument, "--batch-size")) {
            index += 1;
            if (index >= args.len) return error.MissingArgument;
            options.batch_size = try std.fmt.parseInt(usize, args[index], 10);
        } else if (std.mem.eql(u8, argument, "--learning-rate")) {
            index += 1;
            if (index >= args.len) return error.MissingArgument;
            options.learning_rate = try std.fmt.parseFloat(f32, args[index]);
        } else if (std.mem.eql(u8, argument, "--seed")) {
            index += 1;
            if (index >= args.len) return error.MissingArgument;
            options.seed = try std.fmt.parseInt(u64, args[index], 10);
        } else if (std.mem.eql(u8, argument, "--backend")) {
            index += 1;
            if (index >= args.len) return error.MissingArgument;
            options.backend = try parseBackend(args[index]);
        } else if (std.mem.eql(u8, argument, "--top-k")) {
            index += 1;
            if (index >= args.len) return error.MissingArgument;
            options.top_k = try std.fmt.parseInt(usize, args[index], 10);
        } else if (std.mem.eql(u8, argument, "--help")) {
            return error.HelpRequested;
        } else {
            return error.UnknownArgument;
        }
    }
    if (options.epochs == 0 or options.batch_size == 0 or
        !std.math.isFinite(options.learning_rate) or options.learning_rate <= 0 or
        options.top_k == 0 or options.top_k > speech.class_count)
    {
        return error.InvalidOption;
    }
    if (!options.train and inputs.items.len == 0) return error.MissingInput;
    options.inputs = try inputs.toOwnedSlice(allocator);
    return options;
}

fn printUsage() void {
    std.debug.print(
        \\Usage:
        \\  speech_commands --train [training options]
        \\  speech_commands --model <checkpoint> --input <clip.wav> [--input <clip.wav> ...]
        \\
        \\Training options:
        \\  --data-dir <path>       Mini Speech Commands directory
        \\  --output <path>         best-validation checkpoint (default: speech-commands.bin)
        \\  --epochs <count>        training epochs (default: 12)
        \\  --batch-size <count>    mini-batch size (default: 64)
        \\  --learning-rate <value> AdamW learning rate (default: 0.003)
        \\  --seed <number>         split, shuffle, and initialization seed
        \\
        \\Shared options:
        \\  --backend <name>        cpu, auto, metal, cuda, or rocm
        \\  --help                  show this help
        \\
        \\Inference options:
        \\  --model <path>          trained ZNSC checkpoint
        \\  --input <path>          PCM16 WAV clip; repeat for multiple clips
        \\  --top-k <count>         ranked predictions to print (default: 3)
        \\
    , .{});
}

fn train(allocator: std.mem.Allocator, options: Options) !void {
    var frontend = try nn.Audio.LogMelFrontend.init(allocator, .{});
    defer frontend.deinit();
    std.debug.print("Loading and featurizing Mini Speech Commands from {s}...\n", .{options.data_dir});
    var dataset = data.Dataset.load(allocator, options.data_dir, options.seed, frontend) catch |err| {
        if (err == error.FileNotFound) {
            std.debug.print("Dataset not found. Run `nnctl data speech-commands` first.\n", .{});
        }
        return err;
    };
    defer dataset.deinit();
    std.debug.print("Featurized {d} clips into {d} log-mel values each.\n", .{ dataset.entries.len, speech.feature_count });
    const train_indices = try dataset.indices(allocator, .train);
    defer allocator.free(train_indices);
    const validation_indices = try dataset.indices(allocator, .validation);
    defer allocator.free(validation_indices);
    const test_indices = try dataset.indices(allocator, .testing);
    defer allocator.free(test_indices);
    const normalizer = try dataset.computeNormalizer(train_indices);
    dataset.applyNormalizer(normalizer);

    var device = try nn.Device.init(allocator, options.backend);
    defer device.deinit();
    var context = nn.ExecutionContext.init(&device);
    var prng = std.Random.DefaultPrng.init(options.seed);
    var model = speech.SpeechModel.init(&context, prng.random()) catch |err| {
        std.debug.print("Model initialization failed: {s}\n", .{@errorName(err)});
        return err;
    };
    defer model.deinit();
    model.mean = normalizer.mean;
    model.stddev = normalizer.stddev;
    var parameter_storage: [speech.parameter_count]*nn.Tensor = undefined;
    const parameters = try model.parameters(&parameter_storage);
    var optimizer = try nn.Training.Optimizer.init(&context, .{
        .kind = .adamw,
        .learning_rate = options.learning_rate,
        .weight_decay = 0.0001,
        .max_gradient_norm = 5,
    }, parameters);
    defer optimizer.deinit();

    const shuffled = try allocator.dupe(usize, train_indices);
    defer allocator.free(shuffled);
    const batch_features = try allocator.alloc(f32, options.batch_size * speech.feature_count);
    defer allocator.free(batch_features);
    const batch_labels = try allocator.alloc(usize, options.batch_size);
    defer allocator.free(batch_labels);

    std.debug.print(
        "Speech command MLP on {s}: {d} train, {d} validation, {d} test, {d} parameters\n",
        .{ backendName(device.backendType()), train_indices.len, validation_indices.len, test_indices.len, model.network.parameterCount() },
    );
    var best_accuracy: f32 = -1;
    var best_loss: f32 = std.math.inf(f32);
    for (0..options.epochs) |epoch| {
        @memcpy(shuffled, train_indices);
        prng.random().shuffle(usize, shuffled);
        var weighted_loss: f32 = 0;
        var trained: usize = 0;
        var start: usize = 0;
        while (start < shuffled.len) : (start += options.batch_size) {
            const count = @min(options.batch_size, shuffled.len - start);
            gatherBatch(dataset, shuffled[start .. start + count], batch_features[0 .. count * speech.feature_count], batch_labels[0..count]);
            const loss = try model.trainBatch(&context, &optimizer, parameters, batch_features[0 .. count * speech.feature_count], batch_labels[0..count]);
            weighted_loss += loss * @as(f32, @floatFromInt(count));
            trained += count;
        }
        const validation = try evaluate(allocator, &model, &context, dataset, validation_indices, options.batch_size);
        const training_loss = weighted_loss / @as(f32, @floatFromInt(trained));
        std.debug.print(
            "epoch {d:>2}/{d}: train loss {d:.4}, validation loss {d:.4}, accuracy {d:.2}%\n",
            .{ epoch + 1, options.epochs, training_loss, validation.loss, validation.accuracy * 100 },
        );
        if (validation.accuracy > best_accuracy or
            (validation.accuracy == best_accuracy and validation.loss < best_loss))
        {
            best_accuracy = validation.accuracy;
            best_loss = validation.loss;
            try model.save(&context, options.output);
        }
    }

    var best_model = try speech.SpeechModel.load(&context, options.output);
    defer best_model.deinit();
    const testing = try evaluate(allocator, &best_model, &context, dataset, test_indices, options.batch_size);
    std.debug.print(
        "best validation {d:.2}%; held-out test loss {d:.4}, accuracy {d:.2}%\ncheckpoint: {s}\n",
        .{ best_accuracy * 100, testing.loss, testing.accuracy * 100, options.output },
    );
    printConfusion(testing.confusion);
}

fn infer(allocator: std.mem.Allocator, options: Options) !void {
    var device = try nn.Device.init(allocator, options.backend);
    defer device.deinit();
    var context = nn.ExecutionContext.init(&device);
    var model = try speech.SpeechModel.load(&context, options.model);
    defer model.deinit();
    var frontend = try nn.Audio.LogMelFrontend.init(allocator, .{});
    defer frontend.deinit();
    const io = std.Options.debug_io;
    for (options.inputs) |path| {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(16 * 1024 * 1024));
        defer allocator.free(bytes);
        var waveform = try nn.Audio.decodeWav(allocator, bytes);
        defer waveform.deinit();
        const prepared = try nn.Audio.prepareOneSecond(allocator, waveform);
        defer allocator.free(prepared);
        const features = try frontend.extract(prepared);
        defer allocator.free(features);
        try model.normalize(features);
        const probabilities = try model.predict(allocator, &context, features);
        defer allocator.free(probabilities);
        std.debug.print("{s} ({s})\n", .{ path, backendName(device.backendType()) });
        printRanked(probabilities, options.top_k);
    }
}

fn gatherBatch(dataset: data.Dataset, indices: []const usize, features: []f32, targets: []usize) void {
    std.debug.assert(features.len == indices.len * speech.feature_count and targets.len == indices.len);
    for (indices, 0..) |dataset_index, row| {
        @memcpy(features[row * speech.feature_count .. (row + 1) * speech.feature_count], dataset.featureRow(dataset_index));
        targets[row] = dataset.entries[dataset_index].label;
    }
}

fn evaluate(
    allocator: std.mem.Allocator,
    model: *const speech.SpeechModel,
    context: *nn.ExecutionContext,
    dataset: data.Dataset,
    indices: []const usize,
    batch_size: usize,
) !Metrics {
    if (indices.len == 0) return error.EmptyDatasetSplit;
    const features = try allocator.alloc(f32, batch_size * speech.feature_count);
    defer allocator.free(features);
    const targets = try allocator.alloc(usize, batch_size);
    defer allocator.free(targets);
    var result: Metrics = .{ .loss = 0, .accuracy = 0 };
    var correct: usize = 0;
    var start: usize = 0;
    while (start < indices.len) : (start += batch_size) {
        const count = @min(batch_size, indices.len - start);
        gatherBatch(dataset, indices[start .. start + count], features[0 .. count * speech.feature_count], targets[0..count]);
        const probabilities = try model.predict(allocator, context, features[0 .. count * speech.feature_count]);
        defer allocator.free(probabilities);
        for (targets[0..count], 0..) |expected, row| {
            const row_values = probabilities[row * speech.class_count .. (row + 1) * speech.class_count];
            const predicted = maximumIndex(row_values);
            result.loss -= @log(@max(row_values[expected], 1e-12));
            if (predicted == expected) correct += 1;
            result.confusion[expected * speech.class_count + predicted] += 1;
        }
    }
    result.loss /= @as(f32, @floatFromInt(indices.len));
    result.accuracy = @as(f32, @floatFromInt(correct)) / @as(f32, @floatFromInt(indices.len));
    return result;
}

fn printConfusion(confusion: [speech.class_count * speech.class_count]usize) void {
    std.debug.print("confusion matrix (rows=true, columns=predicted):\n          ", .{});
    for (speech.labels) |label| std.debug.print("{s:>7}", .{label});
    std.debug.print("\n", .{});
    for (speech.labels, 0..) |label, row| {
        std.debug.print("{s:>9} ", .{label});
        for (0..speech.class_count) |column| {
            std.debug.print("{d:>7}", .{confusion[row * speech.class_count + column]});
        }
        std.debug.print("\n", .{});
    }
}

fn printRanked(probabilities: []const f32, top_k: usize) void {
    var ranked: [speech.class_count]usize = undefined;
    for (&ranked, 0..) |*index, value| index.* = value;
    for (0..ranked.len) |left| {
        for (left + 1..ranked.len) |right| {
            if (probabilities[ranked[right]] > probabilities[ranked[left]]) {
                std.mem.swap(usize, &ranked[left], &ranked[right]);
            }
        }
    }
    for (ranked[0..top_k], 0..) |label, rank| {
        std.debug.print("  {d}. {s}: {d:.2}%\n", .{ rank + 1, speech.labels[label], probabilities[label] * 100 });
    }
}

fn maximumIndex(values: []const f32) usize {
    var best: usize = 0;
    for (values[1..], 1..) |value, index| if (value > values[best]) {
        best = index;
    };
    return best;
}

fn parseBackend(value: []const u8) !nn.DevicePreference {
    if (std.mem.eql(u8, value, "cpu")) return .cpu;
    if (std.mem.eql(u8, value, "auto")) return .auto;
    if (std.mem.eql(u8, value, "metal")) return .metal;
    if (std.mem.eql(u8, value, "cuda")) return .cuda;
    if (std.mem.eql(u8, value, "rocm")) return .rocm;
    return error.UnknownBackend;
}

fn backendName(backend: nn.BackendType) []const u8 {
    return switch (backend) {
        .CPU => "CPU",
        .Metal => "Metal",
        .CUDA => "CUDA",
        .ROCm => "ROCm",
    };
}

test "speech command MLP learns separable synthetic features" {
    const testing = std.testing;
    var device = try nn.Device.init(testing.allocator, .cpu);
    defer device.deinit();
    var context = nn.ExecutionContext.init(&device);
    var prng = std.Random.DefaultPrng.init(9);
    var model = try speech.SpeechModel.init(&context, prng.random());
    defer model.deinit();
    var parameter_storage: [speech.parameter_count]*nn.Tensor = undefined;
    const parameters = try model.parameters(&parameter_storage);
    var optimizer = try nn.Training.Optimizer.init(&context, .{
        .kind = .adamw,
        .learning_rate = 0.03,
        .max_gradient_norm = 5,
    }, parameters);
    defer optimizer.deinit();
    var features: [speech.class_count * speech.feature_count]f32 = @splat(0);
    var targets: [speech.class_count]usize = undefined;
    for (0..speech.class_count) |label| {
        targets[label] = label;
        features[label * speech.feature_count + label] = 3;
    }
    for (0..80) |_| _ = try model.trainBatch(&context, &optimizer, parameters, &features, &targets);
    const probabilities = try model.predict(testing.allocator, &context, &features);
    defer testing.allocator.free(probabilities);
    for (0..speech.class_count) |label| {
        try testing.expectEqual(label, maximumIndex(probabilities[label * speech.class_count .. (label + 1) * speech.class_count]));
    }
}

test "speech command speaker split is deterministic" {
    const testing = std.testing;
    const first = data.splitForFilename("speaker_nohash_0.wav", 42);
    try testing.expectEqual(first, data.splitForFilename("speaker_nohash_5.wav", 42));
    try testing.expectEqual(first, data.splitForFilename("speaker_nohash_18.wav", 42));
}

test "speech command checkpoint round trip preserves predictions and normalization" {
    const testing = std.testing;
    const path = "test_speech_commands.bin";
    const io = std.Options.debug_io;
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};
    var device = try nn.Device.init(testing.allocator, .cpu);
    defer device.deinit();
    var context = nn.ExecutionContext.init(&device);
    var prng = std.Random.DefaultPrng.init(123);
    var original = try speech.SpeechModel.init(&context, prng.random());
    defer original.deinit();
    original.mean[0] = 2;
    original.stddev[0] = 4;
    var features: [speech.feature_count]f32 = @splat(0.25);
    try original.normalize(&features);
    const expected = try original.predict(testing.allocator, &context, &features);
    defer testing.allocator.free(expected);
    try original.save(&context, path);
    var loaded = try speech.SpeechModel.load(&context, path);
    defer loaded.deinit();
    try testing.expectEqual(original.mean, loaded.mean);
    try testing.expectEqual(original.stddev, loaded.stddev);
    const actual = try loaded.predict(testing.allocator, &context, &features);
    defer testing.allocator.free(actual);
    try testing.expectEqualSlices(f32, expected, actual);
}
