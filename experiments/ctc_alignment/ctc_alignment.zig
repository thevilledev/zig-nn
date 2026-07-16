const std = @import("std");
const nn = @import("nn");

const blank: usize = 0;
const classes = [_][]const u8{ "_", "A", "C", "T", "O" };
const target = [_]usize{ 2, 1, 3 };
const timesteps: usize = 7;
const training_steps: usize = 35;

const Result = struct {
    allocator: std.mem.Allocator,
    initial_loss: f32,
    final_loss: f32,
    greedy: []usize,
    prefix: []usize,
    frame_path: [timesteps]usize,

    fn deinit(self: *Result) void {
        self.allocator.free(self.greedy);
        self.allocator.free(self.prefix);
        self.* = undefined;
    }
};

pub fn main(init: std.process.Init) !void {
    var result = try runExperiment(init.gpa);
    defer result.deinit();
    std.debug.print("CTC alignment over noisy character frames\n", .{});
    std.debug.print(
        "blank-aware forward-backward loss: {d:.4} -> {d:.4}\n",
        .{ result.initial_loss, result.final_loss },
    );
    std.debug.print("frame argmax: ", .{});
    printTokens(&result.frame_path);
    std.debug.print("greedy collapse: ", .{});
    printTokens(result.greedy);
    std.debug.print("prefix beam:    ", .{});
    printTokens(result.prefix);
    std.debug.print(
        "lesson: blanks and repeated frame labels permit monotonic alignment without frame targets\n",
        .{},
    );
}

fn runExperiment(allocator: std.mem.Allocator) !Result {
    var logits = [_]f32{
        0.2, 0.1, 1.6, 0.0, -0.2,
        1.4, 0.2, 1.0, 0.0, -0.1,
        0.2, 1.5, 0.1, 0.0, -0.1,
        0.3, 1.2, 0.0, 0.2, -0.2,
        1.5, 0.4, 0.1, 0.5, -0.1,
        0.2, 0.0, 0.1, 1.6, -0.2,
        1.5, 0.0, 0.1, 1.0, -0.2,
    };
    var initial = try nn.Ctc.lossAndGradient(
        allocator,
        &logits,
        timesteps,
        classes.len,
        &target,
        blank,
    );
    defer initial.deinit();
    for (0..training_steps) |_| {
        var objective = try nn.Ctc.lossAndGradient(
            allocator,
            &logits,
            timesteps,
            classes.len,
            &target,
            blank,
        );
        defer objective.deinit();
        for (&logits, objective.gradient) |*logit, gradient| {
            logit.* -= 0.8 * gradient;
        }
    }
    var final = try nn.Ctc.lossAndGradient(
        allocator,
        &logits,
        timesteps,
        classes.len,
        &target,
        blank,
    );
    defer final.deinit();
    const greedy = try nn.Ctc.greedyDecode(
        allocator,
        &logits,
        timesteps,
        classes.len,
        blank,
    );
    errdefer allocator.free(greedy);
    const prefix = try nn.Ctc.prefixBeamDecode(
        allocator,
        &logits,
        timesteps,
        classes.len,
        blank,
        8,
    );
    errdefer allocator.free(prefix);
    var frame_path: [timesteps]usize = undefined;
    for (0..timesteps) |timestep| {
        const row = logits[timestep * classes.len ..][0..classes.len];
        frame_path[timestep] = argmax(row);
    }
    return .{
        .allocator = allocator,
        .initial_loss = initial.loss,
        .final_loss = final.loss,
        .greedy = greedy,
        .prefix = prefix,
        .frame_path = frame_path,
    };
}

fn argmax(values: []const f32) usize {
    var best: usize = 0;
    for (values[1..], 1..) |value, index| if (value > values[best]) {
        best = index;
    };
    return best;
}

fn printTokens(tokens: []const usize) void {
    for (tokens) |token| std.debug.print("{s} ", .{classes[token]});
    std.debug.print("\n", .{});
}

test "CTC optimization recovers the unsegmented target" {
    var result = try runExperiment(std.testing.allocator);
    defer result.deinit();
    try std.testing.expect(result.final_loss < result.initial_loss * 0.2);
    try std.testing.expectEqualSlices(usize, &target, result.greedy);
    try std.testing.expectEqualSlices(usize, &target, result.prefix);
}
