const std = @import("std");
const config = @import("config.zig");
const model = @import("model.zig");
const loss = @import("loss.zig");
const training = @import("training.zig");
const cli = @import("cli.zig");

pub const Config = config.Config;
pub const Tokenizer = config.Tokenizer;
pub const OptimizerKind = config.OptimizerKind;
pub const LearningRateSchedule = config.LearningRateSchedule;
pub const TinyGPT = model.TinyGPT;
pub const prepareNextTokenTargets = loss.prepareNextTokenTargets;
pub const crossEntropyLoss = loss.crossEntropyLoss;
pub const TrainingStats = training.TrainingStats;
pub const FullTrainingOptions = training.FullTrainingOptions;
pub const FullTrainingStats = training.FullTrainingStats;
pub const trainOutputHeadOnCorpus = training.trainOutputHeadOnCorpus;
pub const trainFullOnCorpus = training.trainFullOnCorpus;
pub const trainFullOnCorpusDevice = training.trainFullOnCorpusDevice;

pub fn main(init: std.process.Init) !void {
    return cli.main(init);
}

test {
    _ = @import("tiny_gpt_tests.zig");
}
