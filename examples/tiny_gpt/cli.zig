const std = @import("std");
const nn = @import("nn");
const config_mod = @import("config.zig");
const model_module = @import("model.zig");
const loss_mod = @import("loss.zig");
const training = @import("training.zig");

const Config = config_mod.Config;
const Tokenizer = config_mod.Tokenizer;
const OptimizerKind = config_mod.OptimizerKind;
const LearningRateSchedule = config_mod.LearningRateSchedule;
const TinyGPT = model_module.TinyGPT;
const prepareNextTokenTargets = loss_mod.prepareNextTokenTargets;
const crossEntropyLoss = loss_mod.crossEntropyLoss;
const TrainingStats = training.TrainingStats;
const FullTrainingOptions = training.FullTrainingOptions;
const FullTrainingStats = training.FullTrainingStats;
const trainOutputHeadOnCorpus = training.trainOutputHeadOnCorpus;
const trainFullOnCorpus = training.trainFullOnCorpus;
const trainFullOnCorpusDevice = training.trainFullOnCorpusDevice;

pub const CliOptions = struct {
    prompt: []const u8 = "to be",
    tokens: usize = 80,
    temperature: f64 = 1.0,
    top_k: usize = 8,
    seed: u64 = 42,
    preset: TrainingPreset = .none,
    model_config: Config = .{},
    config_overridden: bool = false,
    corpus: CorpusPreset = .auto,
    corpus_path: ?[]const u8 = null,
    checkpoint_in: ?[]const u8 = null,
    checkpoint_out: ?[]const u8 = null,
    resume_checkpoint: ?[]const u8 = null,
    train_demo_head: bool = true,
    train_full: bool = false,
    train_epochs: usize = 350,
    learning_rate: f64 = 0.5,
    full_train_steps: usize = 120,
    full_learning_rate: f64 = 0.03,
    min_learning_rate: f64 = 0.0,
    lr_schedule: LearningRateSchedule = .constant,
    warmup_steps: usize = 0,
    full_train_stride: usize = 1,
    full_batch_size: usize = 1,
    full_optimizer: OptimizerKind = .sgd,
    device_preference: ?nn.DevicePreference = null,
    full_weight_decay: f64 = 0.0,
    adam_beta1: f64 = 0.9,
    adam_beta2: f64 = 0.999,
    adam_epsilon: f64 = 1e-8,
    train_chars: usize = 512,
    validation_chars: usize = 0,
    eval_split: f64 = 0.0,
    eval_windows: usize = 12,
    prior_chars: usize = 32768,
    corpus_prior: bool = true,
    summary_path: ?[]const u8 = null,
};

pub const TrainingPreset = enum {
    none,
    coherent_small,
};

pub const CorpusPreset = enum {
    auto,
    toy,
    shakespeare,
    tinystories,
};

const Corpus = struct {
    text: []const u8,
    label: []const u8,
    path: ?[]const u8,
    owned: bool,
    allocator: std.mem.Allocator,

    fn deinit(self: *Corpus) void {
        if (self.owned) {
            self.allocator.free(self.text);
        }
    }
};

const fallback_toy_corpus =
    \\to be, or not to be, that is the question.
    \\the tiny model learns from a little play.
    \\to be a small machine is still to speak.
    \\we train the head on clear short lines.
    \\the model says: to be, to learn, to make.
    \\a small transformer can follow a prompt.
    \\to be clear, the demo repeats simple words.
    \\
;

const toy_corpus_path = "examples/tiny_gpt/data/toy.txt";
const shakespeare_corpus_path = "examples/tiny_gpt/data/shakespeare/input.txt";
const tinystories_corpus_path = "examples/tiny_gpt/data/tinystories/tinystories_1mb.txt";
const max_corpus_bytes = 8 * 1024 * 1024;
fn loadCorpus(allocator: std.mem.Allocator, options: CliOptions) !Corpus {
    if (options.corpus_path) |path| {
        return (try readCorpusFile(allocator, path, "custom corpus")) orelse error.MissingCorpusFile;
    }

    switch (options.corpus) {
        .auto => {
            if (try readCorpusFile(allocator, tinystories_corpus_path, "TinyStories 1MB")) |corpus| return corpus;
            if (try readCorpusFile(allocator, shakespeare_corpus_path, "Tiny Shakespeare")) |corpus| return corpus;
            if (try readCorpusFile(allocator, toy_corpus_path, "toy corpus")) |corpus| return corpus;
            return fallbackToyCorpus(allocator);
        },
        .toy => {
            if (try readCorpusFile(allocator, toy_corpus_path, "toy corpus")) |corpus| return corpus;
            return fallbackToyCorpus(allocator);
        },
        .shakespeare => {
            return (try readCorpusFile(allocator, shakespeare_corpus_path, "Tiny Shakespeare")) orelse error.MissingCorpusFile;
        },
        .tinystories => {
            return (try readCorpusFile(allocator, tinystories_corpus_path, "TinyStories 1MB")) orelse error.MissingCorpusFile;
        },
    }
}

fn readCorpusFile(
    allocator: std.mem.Allocator,
    path: []const u8,
    label: []const u8,
) !?Corpus {
    const text = std.Io.Dir.cwd().readFileAlloc(
        std.Options.debug_io,
        path,
        allocator,
        .limited(max_corpus_bytes),
    ) catch |err| switch (err) {
        error.FileNotFound => return null,
        else => return err,
    };

    return .{
        .text = text,
        .label = label,
        .path = path,
        .owned = true,
        .allocator = allocator,
    };
}

fn fallbackToyCorpus(allocator: std.mem.Allocator) Corpus {
    return .{
        .text = fallback_toy_corpus,
        .label = "built-in toy corpus",
        .path = null,
        .owned = false,
        .allocator = allocator,
    };
}

fn limitedCorpus(text: []const u8, limit: usize) []const u8 {
    if (limit == 0 or limit >= text.len) return text;
    return text[0..limit];
}

pub fn resolvedEvalChars(text_len: usize, train_limit: usize, validation_chars: usize, eval_split: f64) usize {
    if (validation_chars > 0) return validation_chars;
    if (eval_split <= 0.0 or text_len < 3) return 0;

    const max_eval = if (train_limit == 0 or train_limit >= text_len)
        text_len - 1
    else
        text_len - train_limit;
    if (max_eval < 2) return 0;

    const requested = @max(
        @as(usize, 1),
        @as(usize, @intFromFloat(@ceil(@as(f64, @floatFromInt(text_len)) * eval_split))),
    );
    return @min(requested, max_eval);
}

pub fn trainingCorpus(text: []const u8, train_limit: usize, validation_chars: usize) []const u8 {
    var end = if (train_limit == 0 or train_limit >= text.len) text.len else train_limit;
    if ((train_limit == 0 or train_limit >= text.len) and validation_chars > 0 and text.len > validation_chars + 1) {
        end = text.len - validation_chars;
    }
    return text[0..end];
}

pub fn validationCorpus(text: []const u8, train_limit: usize, validation_chars: usize) ?[]const u8 {
    if (validation_chars == 0 or text.len < 2) return null;

    const start = if (train_limit == 0 or train_limit >= text.len) blk: {
        if (text.len <= validation_chars + 1) return null;
        break :blk text.len - validation_chars;
    } else train_limit;

    if (start >= text.len - 1) return null;
    const end = @min(start + validation_chars, text.len);
    if (end - start < 2) return null;
    return text[start..end];
}

pub fn perplexityFromLoss(loss: f64) f64 {
    if (loss >= @log(std.math.floatMax(f64))) return std.math.inf(f64);
    return std.math.exp(loss);
}

fn applyTrainingPreset(options: *CliOptions, preset: TrainingPreset) void {
    switch (preset) {
        .none => {},
        .coherent_small => {
            options.train_full = true;
            options.train_demo_head = false;
            options.model_config = .{
                .block_size = 64,
                .n_layer = 4,
                .n_head = 4,
                .n_embd = 128,
            };
            options.config_overridden = true;
            options.corpus = .tinystories;
            options.corpus_path = null;
            options.full_train_steps = 3000;
            options.full_learning_rate = 0.001;
            options.min_learning_rate = 0.0001;
            options.lr_schedule = .cosine;
            options.warmup_steps = 100;
            options.full_batch_size = 12;
            options.full_optimizer = .adamw;
            options.full_weight_decay = 0.01;
            options.train_chars = 0;
            options.eval_split = 0.05;
            options.eval_windows = 64;
            options.temperature = 0.8;
            options.top_k = 16;
            options.corpus_prior = false;
        },
    }
}

fn parseTrainingPreset(value: []const u8) !TrainingPreset {
    if (std.mem.eql(u8, value, "none")) return .none;
    if (std.mem.eql(u8, value, "coherent-small")) return .coherent_small;
    return error.UnknownTrainingPreset;
}

fn presetFromArgs(args: []const [:0]const u8) !TrainingPreset {
    var preset: TrainingPreset = .none;
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--preset")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            preset = try parseTrainingPreset(args[i]);
        }
    }
    return preset;
}

pub fn parseArgs(args: []const [:0]const u8) !CliOptions {
    var options: CliOptions = .{};
    const preset = try presetFromArgs(args);
    options.preset = preset;
    applyTrainingPreset(&options, preset);

    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--preset")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.preset = try parseTrainingPreset(args[i]);
        } else if (std.mem.eql(u8, arg, "--prompt")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.prompt = args[i];
        } else if (std.mem.eql(u8, arg, "--tokens")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.tokens = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--temperature")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.temperature = try std.fmt.parseFloat(f64, args[i]);
        } else if (std.mem.eql(u8, arg, "--top-k")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.top_k = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--seed")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.seed = try std.fmt.parseInt(u64, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--block-size")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.model_config.block_size = try std.fmt.parseInt(usize, args[i], 10);
            options.config_overridden = true;
        } else if (std.mem.eql(u8, arg, "--layers") or std.mem.eql(u8, arg, "--n-layer")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.model_config.n_layer = try std.fmt.parseInt(usize, args[i], 10);
            options.config_overridden = true;
        } else if (std.mem.eql(u8, arg, "--heads") or std.mem.eql(u8, arg, "--n-head")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.model_config.n_head = try std.fmt.parseInt(usize, args[i], 10);
            options.config_overridden = true;
        } else if (std.mem.eql(u8, arg, "--embd") or std.mem.eql(u8, arg, "--n-embd") or std.mem.eql(u8, arg, "--embedding-size")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.model_config.n_embd = try std.fmt.parseInt(usize, args[i], 10);
            options.config_overridden = true;
        } else if (std.mem.eql(u8, arg, "--corpus")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.corpus = try parseCorpusPreset(args[i]);
            options.corpus_path = null;
        } else if (std.mem.eql(u8, arg, "--corpus-path")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.corpus_path = args[i];
        } else if (std.mem.eql(u8, arg, "--load-checkpoint")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.checkpoint_in = args[i];
        } else if (std.mem.eql(u8, arg, "--resume-checkpoint")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.checkpoint_in = args[i];
            options.resume_checkpoint = args[i];
        } else if (std.mem.eql(u8, arg, "--save-checkpoint")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.checkpoint_out = args[i];
        } else if (std.mem.eql(u8, arg, "--no-train")) {
            options.train_demo_head = false;
            options.train_full = false;
        } else if (std.mem.eql(u8, arg, "--train-full")) {
            options.train_full = true;
            options.train_demo_head = false;
        } else if (std.mem.eql(u8, arg, "--train-epochs")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.train_epochs = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--learning-rate")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.learning_rate = try std.fmt.parseFloat(f64, args[i]);
        } else if (std.mem.eql(u8, arg, "--full-train-steps")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.full_train_steps = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--full-learning-rate")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.full_learning_rate = try std.fmt.parseFloat(f64, args[i]);
        } else if (std.mem.eql(u8, arg, "--min-learning-rate")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.min_learning_rate = try std.fmt.parseFloat(f64, args[i]);
        } else if (std.mem.eql(u8, arg, "--lr-schedule")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.lr_schedule = try parseLearningRateSchedule(args[i]);
        } else if (std.mem.eql(u8, arg, "--warmup-steps")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.warmup_steps = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--full-train-stride")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.full_train_stride = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--full-batch-size") or std.mem.eql(u8, arg, "--batch-size")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.full_batch_size = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--optimizer")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.full_optimizer = try parseOptimizerKind(args[i]);
        } else if (std.mem.eql(u8, arg, "--backend")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.device_preference = try parseDevicePreference(args[i]);
        } else if (std.mem.eql(u8, arg, "--weight-decay")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.full_weight_decay = try std.fmt.parseFloat(f64, args[i]);
        } else if (std.mem.eql(u8, arg, "--adam-beta1")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.adam_beta1 = try std.fmt.parseFloat(f64, args[i]);
        } else if (std.mem.eql(u8, arg, "--adam-beta2")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.adam_beta2 = try std.fmt.parseFloat(f64, args[i]);
        } else if (std.mem.eql(u8, arg, "--adam-epsilon")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.adam_epsilon = try std.fmt.parseFloat(f64, args[i]);
        } else if (std.mem.eql(u8, arg, "--train-chars")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.train_chars = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--validation-chars")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.validation_chars = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--eval-chars")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.validation_chars = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--eval-split")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.eval_split = try std.fmt.parseFloat(f64, args[i]);
        } else if (std.mem.eql(u8, arg, "--eval-windows")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.eval_windows = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--prior-chars")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.prior_chars = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--corpus-prior")) {
            options.corpus_prior = true;
        } else if (std.mem.eql(u8, arg, "--no-corpus-prior")) {
            options.corpus_prior = false;
        } else if (std.mem.eql(u8, arg, "--summary-path")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.summary_path = args[i];
        } else if (std.mem.eql(u8, arg, "--help")) {
            return error.HelpRequested;
        } else {
            return error.UnknownArgument;
        }
    }
    return options;
}

fn parseCorpusPreset(value: []const u8) !CorpusPreset {
    if (std.mem.eql(u8, value, "auto")) return .auto;
    if (std.mem.eql(u8, value, "toy")) return .toy;
    if (std.mem.eql(u8, value, "shakespeare")) return .shakespeare;
    if (std.mem.eql(u8, value, "tinystories")) return .tinystories;
    return error.UnknownCorpusPreset;
}

fn parseOptimizerKind(value: []const u8) !OptimizerKind {
    if (std.mem.eql(u8, value, "sgd")) return .sgd;
    if (std.mem.eql(u8, value, "adamw")) return .adamw;
    return error.UnknownOptimizer;
}

fn parseDevicePreference(value: []const u8) !nn.DevicePreference {
    if (std.mem.eql(u8, value, "cpu")) return .cpu;
    if (std.mem.eql(u8, value, "auto")) return .auto;
    if (std.mem.eql(u8, value, "metal")) return .metal;
    if (std.mem.eql(u8, value, "cuda")) return .cuda;
    return error.UnknownBackend;
}

fn parseLearningRateSchedule(value: []const u8) !LearningRateSchedule {
    if (std.mem.eql(u8, value, "constant")) return .constant;
    if (std.mem.eql(u8, value, "linear")) return .linear;
    if (std.mem.eql(u8, value, "cosine")) return .cosine;
    return error.UnknownLearningRateSchedule;
}

fn printUsage(writer: anytype) !void {
    try writer.writeAll(
        \\Usage: zig build run_tiny_gpt -- [options]
        \\
        \\Options:
        \\  --preset <name>          none or coherent-small
        \\  --prompt <text>          Prompt to continue (default: "to be")
        \\  --tokens <n>             Number of new tokens to sample (default: 80)
        \\  --temperature <float>    Sampling temperature (default: 1.0)
        \\  --top-k <n>              Restrict sampling to top k tokens (default: 8)
        \\  --seed <n>               Deterministic seed (default: 42)
        \\  --block-size <n>         Context length for new models (default: 16)
        \\  --layers <n>             Transformer block count for new models (default: 2)
        \\  --heads <n>              Attention head count for new models (default: 2)
        \\  --embd <n>               Embedding/channel size for new models (default: 32)
        \\  --corpus <name>          auto, toy, shakespeare, or tinystories
        \\  --corpus-path <path>     Use a custom UTF-8 text corpus
        \\  --load-checkpoint <path> Load a TinyGPT checkpoint before generation/training
        \\  --resume-checkpoint <p>  Load a checkpoint and save back to it unless overridden
        \\  --save-checkpoint <path> Save the model checkpoint after optional training
        \\  --no-train               Skip the tiny demo-corpus output-head training
        \\  --train-full             Train all Transformer weights instead of only the head
        \\  --train-epochs <n>       Output-head training epochs (default: 350)
        \\  --learning-rate <float>  Output-head learning rate (default: 0.5)
        \\  --full-train-steps <n>   Full-model context-window updates (default: 120)
        \\  --full-learning-rate <f> Full-model learning rate (default: 0.03)
        \\  --min-learning-rate <f>  Schedule floor for full training (default: 0)
        \\  --lr-schedule <name>     constant, linear, or cosine (default: constant)
        \\  --warmup-steps <n>       Linear warmup steps before decay (default: 0)
        \\  --full-train-stride <n>  Full-model corpus window stride (default: 1)
        \\  --full-batch-size <n>    Full-model corpus windows per step (default: 1)
        \\  --optimizer <name>       sgd or adamw for full training (default: sgd)
        \\  --backend <name>         Train via cpu, auto, metal, or cuda tensor backend
        \\  --weight-decay <f>       Full-training weight decay (default: 0)
        \\  --adam-beta1 <f>         AdamW beta1 (default: 0.9)
        \\  --adam-beta2 <f>         AdamW beta2 (default: 0.999)
        \\  --adam-epsilon <f>       AdamW epsilon (default: 1e-8)
        \\  --train-chars <n>        Corpus prefix for output-head training (default: 512)
        \\  --validation-chars <n>   Corpus holdout chars after the train slice (default: 0)
        \\  --eval-chars <n>         Alias for --validation-chars
        \\  --eval-split <float>     Eval fraction used when validation chars are 0
        \\  --eval-windows <n>       Windows sampled for train/eval loss (default: 12)
        \\  --prior-chars <n>        Corpus prefix for readable sampling prior (default: 32768)
        \\  --corpus-prior           Enable the tiny backoff corpus prior
        \\  --no-corpus-prior        Skip the tiny backoff corpus prior used for readable sampling
        \\  --summary-path <path>    Write a deterministic JSON run summary
        \\
        \\Data:
        \\  `--corpus auto` uses prepared TinyStories, then Tiny Shakespeare, then toy.
        \\  Run `nnctl data tiny-gpt` to download the sourced corpora.
        \\
    );
}

fn writeJsonString(writer: anytype, text: []const u8) !void {
    const hex = "0123456789abcdef";
    try writer.writeByte('"');
    for (text) |ch| {
        switch (ch) {
            '"' => try writer.writeAll("\\\""),
            '\\' => try writer.writeAll("\\\\"),
            '\n' => try writer.writeAll("\\n"),
            '\r' => try writer.writeAll("\\r"),
            '\t' => try writer.writeAll("\\t"),
            else => {
                if (ch < 32) {
                    try writer.writeAll("\\u00");
                    try writer.writeByte(hex[ch >> 4]);
                    try writer.writeByte(hex[ch & 0x0f]);
                } else {
                    try writer.writeByte(ch);
                }
            },
        }
    }
    try writer.writeByte('"');
}

fn writeOptionalJsonString(writer: anytype, text: ?[]const u8) !void {
    if (text) |value| {
        try writeJsonString(writer, value);
    } else {
        try writer.writeAll("null");
    }
}

fn writeOptionalJsonFloat(writer: anytype, value: ?f64) !void {
    if (value) |number| {
        try writer.print("{d:.6}", .{number});
    } else {
        try writer.writeAll("null");
    }
}

fn writeOptionalJsonPerplexity(writer: anytype, loss: ?f64) !void {
    if (loss) |value| {
        try writer.print("{d:.6}", .{perplexityFromLoss(value)});
    } else {
        try writer.writeAll("null");
    }
}

fn runSummaryJson(
    allocator: std.mem.Allocator,
    model: *const TinyGPT,
    options: CliOptions,
    corpus: Corpus,
    training_corpus: []const u8,
    validation_corpus: ?[]const u8,
    training_stats: ?TrainingStats,
    full_training_stats: ?FullTrainingStats,
    checkpoint_out: ?[]const u8,
    generated_text: []const u8,
) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    errdefer out.deinit();
    const writer = &out.writer;
    const config = model.config;
    const training_mode = if (full_training_stats != null)
        "full"
    else if (training_stats != null)
        "head"
    else
        "none";

    try writer.writeAll("{\n");
    try writer.writeAll("  \"schema\": \"zig-nn.tiny-gpt.run-summary.v1\",\n");
    try writer.print("  \"seed\": {},\n", .{options.seed});
    try writer.writeAll("  \"corpus\": {\n");
    try writer.writeAll("    \"label\": ");
    try writeJsonString(writer, corpus.label);
    try writer.writeAll(",\n    \"path\": ");
    try writeOptionalJsonString(writer, corpus.path);
    try writer.print(",\n    \"chars\": {}\n", .{corpus.text.len});
    try writer.writeAll("  },\n");

    try writer.writeAll("  \"split\": {\n");
    try writer.print("    \"train_chars\": {},\n", .{training_corpus.len});
    try writer.print("    \"eval_chars\": {},\n", .{if (validation_corpus) |text| text.len else 0});
    try writer.print("    \"eval_split\": {d:.6},\n", .{options.eval_split});
    try writer.print("    \"eval_windows\": {}\n", .{options.eval_windows});
    try writer.writeAll("  },\n");

    try writer.writeAll("  \"model\": {\n");
    try writer.print("    \"block_size\": {},\n", .{config.block_size});
    try writer.print("    \"layers\": {},\n", .{config.n_layer});
    try writer.print("    \"heads\": {},\n", .{config.n_head});
    try writer.print("    \"embd\": {},\n", .{config.n_embd});
    try writer.print("    \"vocab_size\": {},\n", .{config.vocab_size});
    try writer.print("    \"parameters\": {}\n", .{model.parameterCount()});
    try writer.writeAll("  },\n");

    try writer.writeAll("  \"training\": {\n");
    try writer.writeAll("    \"mode\": ");
    try writeJsonString(writer, training_mode);
    try writer.print(",\n    \"steps\": {},\n", .{if (full_training_stats) |stats| stats.steps else 0});
    try writer.print("    \"updates\": {},\n", .{if (full_training_stats) |stats| stats.updates else 0});
    try writer.print("    \"batch_size\": {},\n", .{if (full_training_stats) |stats| stats.batch_size else options.full_batch_size});
    try writer.print("    \"tokens_seen\": {},\n", .{if (full_training_stats) |stats| stats.tokens_seen else 0});
    try writer.writeAll("    \"optimizer\": ");
    try writeJsonString(writer, @tagName(options.full_optimizer));
    try writer.writeAll(",\n    \"lr_schedule\": ");
    try writeJsonString(writer, @tagName(options.lr_schedule));
    try writer.print(",\n    \"learning_rate\": {d:.6},\n", .{options.full_learning_rate});
    try writer.print("    \"min_learning_rate\": {d:.6},\n", .{options.min_learning_rate});
    try writer.writeAll("    \"learning_rate_initial\": ");
    try writeOptionalJsonFloat(writer, if (full_training_stats) |stats| stats.learning_rate_initial else null);
    try writer.writeAll(",\n    \"learning_rate_final\": ");
    try writeOptionalJsonFloat(writer, if (full_training_stats) |stats| stats.learning_rate_final else null);
    try writer.print(",\n    \"warmup_steps\": {},\n", .{options.warmup_steps});
    try writer.print("    \"weight_decay\": {d:.6}\n", .{options.full_weight_decay});
    try writer.writeAll("  },\n");

    try writer.writeAll("  \"loss\": {\n");
    if (full_training_stats) |stats| {
        try writer.print("    \"train_initial\": {d:.6},\n", .{stats.initial_loss});
        try writer.print("    \"train_final\": {d:.6},\n", .{stats.final_loss});
        try writer.writeAll("    \"eval_initial\": ");
        try writeOptionalJsonFloat(writer, stats.validation_initial_loss);
        try writer.writeAll(",\n    \"eval_final\": ");
        try writeOptionalJsonFloat(writer, stats.validation_final_loss);
        try writer.writeByte('\n');
    } else if (training_stats) |stats| {
        try writer.print("    \"train_initial\": {d:.6},\n", .{stats.initial_loss});
        try writer.print("    \"train_final\": {d:.6},\n", .{stats.final_loss});
        try writer.writeAll("    \"eval_initial\": null,\n");
        try writer.writeAll("    \"eval_final\": null\n");
    } else {
        try writer.writeAll("    \"train_initial\": null,\n");
        try writer.writeAll("    \"train_final\": null,\n");
        try writer.writeAll("    \"eval_initial\": null,\n");
        try writer.writeAll("    \"eval_final\": null\n");
    }
    try writer.writeAll("  },\n");

    try writer.writeAll("  \"perplexity\": {\n");
    if (full_training_stats) |stats| {
        try writer.print("    \"train_initial\": {d:.6},\n", .{perplexityFromLoss(stats.initial_loss)});
        try writer.print("    \"train_final\": {d:.6},\n", .{perplexityFromLoss(stats.final_loss)});
        try writer.writeAll("    \"eval_initial\": ");
        try writeOptionalJsonPerplexity(writer, stats.validation_initial_loss);
        try writer.writeAll(",\n    \"eval_final\": ");
        try writeOptionalJsonPerplexity(writer, stats.validation_final_loss);
        try writer.writeByte('\n');
    } else if (training_stats) |stats| {
        try writer.print("    \"train_initial\": {d:.6},\n", .{perplexityFromLoss(stats.initial_loss)});
        try writer.print("    \"train_final\": {d:.6},\n", .{perplexityFromLoss(stats.final_loss)});
        try writer.writeAll("    \"eval_initial\": null,\n");
        try writer.writeAll("    \"eval_final\": null\n");
    } else {
        try writer.writeAll("    \"train_initial\": null,\n");
        try writer.writeAll("    \"train_final\": null,\n");
        try writer.writeAll("    \"eval_initial\": null,\n");
        try writer.writeAll("    \"eval_final\": null\n");
    }
    try writer.writeAll("  },\n");

    try writer.writeAll("  \"checkpoint\": {\n");
    try writer.writeAll("    \"loaded\": ");
    try writeOptionalJsonString(writer, options.checkpoint_in);
    try writer.writeAll(",\n    \"saved\": ");
    try writeOptionalJsonString(writer, checkpoint_out);
    try writer.writeAll("\n  },\n");

    try writer.writeAll("  \"sample\": {\n");
    try writer.writeAll("    \"prompt\": ");
    try writeJsonString(writer, options.prompt);
    try writer.print(",\n    \"tokens\": {},\n", .{options.tokens});
    try writer.print("    \"temperature\": {d:.6},\n", .{options.temperature});
    try writer.print("    \"top_k\": {},\n", .{options.top_k});
    try writer.writeAll("    \"text\": ");
    try writeJsonString(writer, generated_text);
    try writer.writeAll("\n  }\n");
    try writer.writeAll("}\n");

    return out.toOwnedSlice();
}

fn writeTextFile(path: []const u8, text: []const u8) !void {
    const io = std.Options.debug_io;
    const file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);

    var buffer: [4096]u8 = undefined;
    var file_writer = file.writerStreaming(io, &buffer);
    const writer = &file_writer.interface;
    try writer.writeAll(text);
    try writer.flush();
}

pub fn main(init: std.process.Init) !void {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writerStreaming(std.Options.debug_io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};

    const args = try init.minimal.args.toSlice(init.arena.allocator());

    const options = parseArgs(args[1..]) catch |err| {
        try printUsage(stdout);
        if (err == error.HelpRequested) return;
        return err;
    };
    if (options.eval_split < 0.0 or options.eval_split >= 1.0) return error.InvalidEvalSplit;
    if (options.eval_windows == 0) return error.InvalidEvalWindows;
    if (options.full_learning_rate < 0.0 or options.min_learning_rate < 0.0) return error.InvalidLearningRate;

    if (options.checkpoint_in != null and options.config_overridden) {
        return error.ConfigOverridesCheckpoint;
    }
    const checkpoint_out = if (options.checkpoint_out) |path| path else options.resume_checkpoint;

    var model = if (options.checkpoint_in) |path|
        try TinyGPT.loadFromFile(allocator, path)
    else
        try TinyGPT.init(allocator, options.model_config, options.seed);
    defer model.deinit();
    const config = model.config;

    var corpus = loadCorpus(allocator, options) catch |err| {
        if (err == error.MissingCorpusFile) {
            try stdout.print(
                "Missing corpus file. Run `nnctl data tiny-gpt`, use `--corpus toy`, or pass `--corpus-path <path>`.\n",
                .{},
            );
        }
        return err;
    };
    defer corpus.deinit();

    const eval_chars = resolvedEvalChars(corpus.text.len, options.train_chars, options.validation_chars, options.eval_split);
    const training_corpus = trainingCorpus(corpus.text, options.train_chars, eval_chars);
    const validation_corpus = validationCorpus(corpus.text, options.train_chars, eval_chars);
    const prior_corpus = limitedCorpus(corpus.text, options.prior_chars);

    const training_stats: ?TrainingStats = if (options.train_demo_head and !options.train_full)
        try trainOutputHeadOnCorpus(&model, training_corpus, options.train_epochs, options.learning_rate)
    else
        null;
    const full_training_options: FullTrainingOptions = .{
        .steps = options.full_train_steps,
        .learning_rate = options.full_learning_rate,
        .min_learning_rate = options.min_learning_rate,
        .lr_schedule = options.lr_schedule,
        .warmup_steps = options.warmup_steps,
        .context_len = config.block_size,
        .stride = options.full_train_stride,
        .batch_size = options.full_batch_size,
        .optimizer = options.full_optimizer,
        .weight_decay = options.full_weight_decay,
        .gradient_clip = 1.0,
        .random_seed = options.seed,
        .adam_beta1 = options.adam_beta1,
        .adam_beta2 = options.adam_beta2,
        .adam_epsilon = options.adam_epsilon,
        .eval_windows = options.eval_windows,
    };
    const full_training_stats: ?FullTrainingStats = if (options.train_full)
        if (options.device_preference) |preference|
            try trainFullOnCorpusDevice(&model, training_corpus, validation_corpus, full_training_options, preference)
        else
            try trainFullOnCorpus(&model, training_corpus, validation_corpus, full_training_options)
    else
        null;

    const generated_tokens = if (options.corpus_prior)
        try model.generateTokensWithCorpusPrior(
            allocator,
            options.prompt,
            options.tokens,
            options.temperature,
            options.top_k,
            prior_corpus,
        )
    else
        try model.generateTokens(
            allocator,
            options.prompt,
            options.tokens,
            options.temperature,
            options.top_k,
        );
    defer allocator.free(generated_tokens);

    const generated_text = try Tokenizer.decode(allocator, generated_tokens);
    defer allocator.free(generated_text);
    const summary_json = try runSummaryJson(
        allocator,
        &model,
        options,
        corpus,
        training_corpus,
        validation_corpus,
        training_stats,
        full_training_stats,
        checkpoint_out,
        generated_text,
    );
    defer allocator.free(summary_json);

    if (checkpoint_out) |path| {
        try model.saveToFileWithMetadata(path, summary_json);
    }
    if (options.summary_path) |path| {
        try writeTextFile(path, summary_json);
    }

    try stdout.print("\n=== Tiny GPT Example ===\n\n", .{});
    try stdout.print("Architecture: {} layers, {} heads, {} channels, block size {}, vocab {}\n", .{
        config.n_layer,
        config.n_head,
        config.n_embd,
        config.block_size,
        config.vocab_size,
    });
    try stdout.print("Parameters: {}\n", .{model.parameterCount()});
    try stdout.print("Seed: {}\n", .{options.seed});
    if (options.checkpoint_in) |path| {
        try stdout.print("Checkpoint loaded: {s}\n", .{path});
    }
    try stdout.print("Corpus: {s}", .{corpus.label});
    if (corpus.path) |path| {
        try stdout.print(" ({s})", .{path});
    }
    try stdout.print(", {} chars loaded\n", .{corpus.text.len});
    try stdout.print("Split: {} train chars, {} eval chars", .{
        training_corpus.len,
        if (validation_corpus) |text| text.len else 0,
    });
    if (options.eval_split > 0.0 and options.validation_chars == 0) {
        try stdout.print(" ({d:.2}% eval split)", .{options.eval_split * 100.0});
    }
    try stdout.print("\n", .{});
    if (full_training_stats) |stats| {
        try stdout.print("Full training: {} steps, {} updates, batch {}, {} tokens seen, {} train chars\n", .{
            stats.steps,
            stats.updates,
            stats.batch_size,
            stats.tokens_seen,
            training_corpus.len,
        });
        try stdout.print("Full optimizer: {s}, learning rate {d:.6}, weight decay {d:.6}\n", .{
            @tagName(options.full_optimizer),
            options.full_learning_rate,
            options.full_weight_decay,
        });
        try stdout.print("LR schedule: {s}, warmup {}, min {d:.6}, final {d:.6}\n", .{
            @tagName(options.lr_schedule),
            options.warmup_steps,
            stats.learning_rate_min,
            stats.learning_rate_final,
        });
        try stdout.print("Full training loss: {d:.4} -> {d:.4}\n", .{ stats.initial_loss, stats.final_loss });
        try stdout.print("Full training perplexity: {d:.2} -> {d:.2}\n", .{
            perplexityFromLoss(stats.initial_loss),
            perplexityFromLoss(stats.final_loss),
        });
        if (stats.validation_initial_loss) |initial_validation| {
            if (stats.validation_final_loss) |final_validation| {
                try stdout.print("Validation loss: {d:.4} -> {d:.4} ({} sampled windows)\n", .{
                    initial_validation,
                    final_validation,
                    stats.eval_windows,
                });
                try stdout.print("Validation perplexity: {d:.2} -> {d:.2}\n", .{
                    perplexityFromLoss(initial_validation),
                    perplexityFromLoss(final_validation),
                });
            }
        } else {
            try stdout.print("Validation loss: unavailable (no eval slice long enough for block size {})\n", .{config.block_size});
        }
    } else if (training_stats) |stats| {
        try stdout.print("Demo training: {} frozen-context samples, {} output-head epochs, {} chars\n", .{
            stats.samples,
            options.train_epochs,
            training_corpus.len,
        });
        try stdout.print("Demo training loss: {d:.4} -> {d:.4}\n", .{ stats.initial_loss, stats.final_loss });
        try stdout.print("Demo training perplexity: {d:.2} -> {d:.2}\n", .{
            perplexityFromLoss(stats.initial_loss),
            perplexityFromLoss(stats.final_loss),
        });
    } else {
        try stdout.print("Demo training: skipped, output head remains random\n", .{});
    }
    if (checkpoint_out) |path| {
        try stdout.print("Checkpoint saved: {s} (metadata embedded)\n", .{path});
    }
    if (options.summary_path) |path| {
        try stdout.print("Run summary saved: {s}\n", .{path});
    }
    try stdout.print("Readable sampling: {s}", .{if (options.corpus_prior) "corpus prior enabled" else "corpus prior disabled"});
    if (options.corpus_prior) {
        try stdout.print(", {} chars\n", .{prior_corpus.len});
    } else {
        try stdout.print("\n", .{});
    }
    try stdout.print("\nPrompt: \"{s}\"\n\n{s}\n", .{ options.prompt, generated_text });

    const prompt_tokens = try Tokenizer.encode(allocator, options.prompt);
    defer allocator.free(prompt_tokens);
    if (prompt_tokens.len > 1 and prompt_tokens.len - 1 <= config.block_size) {
        const targets = try prepareNextTokenTargets(allocator, prompt_tokens);
        defer allocator.free(targets);

        var logits = try model.forward(prompt_tokens[0 .. prompt_tokens.len - 1]);
        defer logits.deinit();

        const loss = try crossEntropyLoss(logits, targets);
        try stdout.print("\nPrompt next-token loss: {d:.4}\n", .{loss});
    }

    if (full_training_stats != null) {
        try stdout.print("\nThe Transformer body, embeddings, layer norms, and output head were updated with manual next-token backpropagation.\n", .{});
    } else if (training_stats != null and options.corpus_prior) {
        try stdout.print("\nThe Transformer body is frozen; the output head and tiny corpus prior make this a readable miniature language demo.\n", .{});
    } else if (training_stats != null) {
        try stdout.print("\nThe Transformer body is frozen; the output head is trained on the tiny built-in corpus for a readable demo.\n", .{});
    } else {
        try stdout.print("\nThis is an untrained decoder-only Transformer. It demonstrates the architecture and sampling path.\n", .{});
    }
}
