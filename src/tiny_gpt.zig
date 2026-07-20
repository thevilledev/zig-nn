//! Reusable TinyGPT model and checkpoint implementation.
//!
//! The experiment entry points remain in `experiments/tiny_gpt`; this module
//! owns the model, tokenizer, device adapter, and TGPT v1-v3 compatibility.

const config = @import("tiny_gpt/config.zig");
const model = @import("tiny_gpt/model.zig");

pub const Config = config.Config;
pub const Tokenizer = config.Tokenizer;
pub const OptimizerKind = config.OptimizerKind;
pub const LearningRateSchedule = config.LearningRateSchedule;
pub const Model = model.TinyGPT;
pub const TinyGPT = model.TinyGPT;
pub const Linear = model.Linear;
pub const LayerNorm = model.LayerNorm;
pub const CausalSelfAttention = model.CausalSelfAttention;
pub const MLP = model.MLP;
pub const Block = model.Block;
pub const gelu = model.gelu;
pub const geluDerivative = model.geluDerivative;
pub const causalSoftmax = model.causalSoftmax;
pub const forwardDeviceTokens = model.forwardDeviceTokens;

test {
    _ = config;
    _ = model;
}
