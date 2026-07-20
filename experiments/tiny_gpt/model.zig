const tiny = @import("nn").TinyGPT;

pub const Config = tiny.Config;
pub const Tokenizer = tiny.Tokenizer;
pub const TinyGPT = tiny.Model;
pub const Linear = tiny.Linear;
pub const LayerNorm = tiny.LayerNorm;
pub const CausalSelfAttention = tiny.CausalSelfAttention;
pub const MLP = tiny.MLP;
pub const Block = tiny.Block;
pub const gelu = tiny.gelu;
pub const geluDerivative = tiny.geluDerivative;
pub const causalSoftmax = tiny.causalSoftmax;
