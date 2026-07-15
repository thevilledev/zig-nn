const std = @import("std");
const nn = @import("nn");

const Matrix = nn.Matrix;
const math = std.math;

pub fn prepareNextTokenTargets(allocator: std.mem.Allocator, tokens: []const usize) ![]usize {
    if (tokens.len < 2) return error.SequenceTooShort;
    const targets = try allocator.alloc(usize, tokens.len - 1);
    @memcpy(targets, tokens[1..]);
    return targets;
}

pub fn crossEntropyLoss(logits: Matrix, targets: []const usize) !f64 {
    if (logits.rows != targets.len) return error.DimensionMismatch;
    if (targets.len == 0) return error.EmptySequence;

    var total_loss: f64 = 0.0;
    for (targets, 0..) |target, row| {
        if (target >= logits.cols) return error.TokenOutOfVocabulary;

        var max_logit = -math.inf(f64);
        for (0..logits.cols) |col| {
            max_logit = @max(max_logit, try logits.get(row, col));
        }

        var exp_sum: f64 = 0.0;
        for (0..logits.cols) |col| {
            exp_sum += math.exp((try logits.get(row, col)) - max_logit);
        }

        const target_logit = try logits.get(row, target);
        total_loss += -target_logit + max_logit + @log(exp_sum);
    }

    return total_loss / @as(f64, @floatFromInt(targets.len));
}
