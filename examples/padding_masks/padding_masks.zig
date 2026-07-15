const std = @import("std");
const nn = @import("nn");

const Result = struct {
    backend: nn.BackendType,
    pooled: [4]f32,
    changed_padding: [4]f32,
    loss: f32,
    kernels: usize,
    training_readbacks: usize,
};

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const preference = parseArgs(args[1..]) catch |err| {
        printUsage();
        if (err == error.HelpRequested) return;
        return err;
    };
    const result = try runLab(init.gpa, preference);
    std.debug.print("Padding-mask sequence lesson on {s}\n", .{backendName(result.backend)});
    std.debug.print("two sequences have lengths 2 and 3 inside a length-4 batch\n", .{});
    std.debug.print("masked attention pooling:\n", .{});
    for (0..2) |row| {
        std.debug.print(
            "  sequence {d}: [{d:.4}, {d:.4}] -> changed padding [{d:.4}, {d:.4}]\n",
            .{
                row,
                result.pooled[row * 2],
                result.pooled[row * 2 + 1],
                result.changed_padding[row * 2],
                result.changed_padding[row * 2 + 1],
            },
        );
    }
    std.debug.print(
        "masked sparse cross-entropy: {d:.5}\n" ++
            "kernels: {d}, training readbacks: {d}\n" ++
            "lesson: padding values can change wildly without changing pooled representations\n",
        .{ result.loss, result.kernels, result.training_readbacks },
    );
}

fn parseArgs(args: []const []const u8) !nn.DevicePreference {
    var preference: nn.DevicePreference = .auto;
    var index: usize = 0;
    while (index < args.len) : (index += 1) {
        if (std.mem.eql(u8, args[index], "--backend")) {
            index += 1;
            if (index >= args.len) return error.MissingArgument;
            preference = try parseBackend(args[index]);
        } else if (std.mem.eql(u8, args[index], "--help")) {
            return error.HelpRequested;
        } else {
            return error.UnknownArgument;
        }
    }
    return preference;
}

fn parseBackend(value: []const u8) !nn.DevicePreference {
    if (std.mem.eql(u8, value, "cpu")) return .cpu;
    if (std.mem.eql(u8, value, "auto")) return .auto;
    if (std.mem.eql(u8, value, "metal")) return .metal;
    if (std.mem.eql(u8, value, "cuda")) return .cuda;
    if (std.mem.eql(u8, value, "rocm")) return .rocm;
    return error.UnknownBackend;
}

fn printUsage() void {
    std.debug.print(
        \\Usage: padding_masks [--backend cpu|auto|metal|cuda|rocm]
        \\
    , .{});
}

fn runLab(allocator: std.mem.Allocator, preference: nn.DevicePreference) !Result {
    var device = try nn.Device.init(allocator, preference);
    defer device.deinit();
    var context = nn.ExecutionContext.init(&device);

    const original_values = [_]f32{
        1.0, 0.0, 0.8, 0.2, 9.0, 9.0, -8.0, 7.0,
        0.0, 1.0, 0.2, 0.8, 0.1, 0.9, 6.0,  -7.0,
    };
    const changed_values = [_]f32{
        1.0, 0.0, 0.8, 0.2, 900.0, -900.0, -800.0, 700.0,
        0.0, 1.0, 0.2, 0.8, 0.1,   0.9,    -600.0, 700.0,
    };
    var values = try context.upload(&.{ 2, 4, 2 }, &original_values);
    defer values.deinit();
    var changed = try context.upload(&.{ 2, 4, 2 }, &changed_values);
    defer changed.deinit();
    var scores = try context.upload(&.{ 2, 1, 4 }, &.{ 1.2, 0.4, 100, -100, 0.9, 0.4, 1.1, 80 });
    defer scores.deinit();
    var changed_scores = try context.upload(&.{ 2, 1, 4 }, &.{ 1.2, 0.4, -900, 900, 0.9, 0.4, 1.1, -800 });
    defer changed_scores.deinit();
    var mask = try context.upload(&.{ 2, 1, 4 }, &.{ 1, 1, 0, 0, 1, 1, 1, 0 });
    defer mask.deinit();
    var classifier_weights = try context.upload(&.{ 2, 2 }, &.{ 2, -1, -1, 2 });
    defer classifier_weights.deinit();
    var classifier_bias = try context.upload(&.{ 1, 2 }, &.{ 0, 0 });
    defer classifier_bias.deinit();

    context.resetStats();
    var probabilities = try context.maskedSoftmax(scores, mask);
    defer probabilities.deinit();
    var pooled = try context.batchedMatmul(probabilities, values, false, false);
    defer pooled.deinit();
    var changed_probabilities = try context.maskedSoftmax(changed_scores, mask);
    defer changed_probabilities.deinit();
    var changed_pooled = try context.batchedMatmul(changed_probabilities, changed, false, false);
    defer changed_pooled.deinit();

    var pooled_view = pooled;
    try pooled_view.reshape(&.{ 2, 2 });
    var logits = try context.linear(pooled_view, classifier_weights, classifier_bias);
    defer logits.deinit();
    var cross_entropy = try context.maskedSparseCrossEntropy(logits, &.{ 0, 1 }, &.{ 1, 1 });
    defer cross_entropy.deinit();
    var dropped = try context.dropout(pooled, 0.25, 42, true);
    defer dropped.deinit();
    var unit_gradient = try context.createTensor(&.{ 2, 1, 2 });
    defer unit_gradient.deinit();
    unit_gradient.fill(1);
    var dropout_gradient = try context.dropoutBackward(unit_gradient, dropped.mask);
    defer dropout_gradient.deinit();

    const training_stats = context.stats;
    var pooled_values: [4]f32 = undefined;
    var changed_padding_values: [4]f32 = undefined;
    try context.readback(pooled, &pooled_values);
    try context.readback(changed_pooled, &changed_padding_values);
    const loss = try cross_entropy.meanLoss(&context);
    return .{
        .backend = device.backendType(),
        .pooled = pooled_values,
        .changed_padding = changed_padding_values,
        .loss = loss,
        .kernels = training_stats.kernels,
        .training_readbacks = training_stats.readbacks,
    };
}

fn backendName(backend: nn.BackendType) []const u8 {
    return switch (backend) {
        .CPU => "CPU",
        .Metal => "Metal",
        .CUDA => "CUDA",
        .ROCm => "ROCm",
    };
}

test "padding values do not affect masked sequence representations" {
    const result = try runLab(std.testing.allocator, .cpu);
    try std.testing.expectEqual(@as(usize, 0), result.training_readbacks);
    try std.testing.expect(result.kernels > 0);
    try std.testing.expect(result.loss < 0.3);
    for (result.pooled, result.changed_padding) |expected, actual| {
        try std.testing.expectApproxEqAbs(expected, actual, 1e-6);
    }
    try std.testing.expect(result.pooled[0] > result.pooled[1]);
    try std.testing.expect(result.pooled[3] > result.pooled[2]);
}
