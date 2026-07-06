const std = @import("std");
const nn = @import("nn");

const Matrix = nn.Matrix;
const BackendMatrix = nn.BackendMatrix;

const TrainingResult = struct {
    backend_name: []const u8,
    before_loss: f64,
    after_loss: f64,
    prediction_at_two: f64,
};

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();

    const result = try runBackendTraining(arena.allocator());
    std.debug.print("Backend training demo ({s})\n", .{result.backend_name});
    std.debug.print("loss: {d:.6} -> {d:.6}\n", .{ result.before_loss, result.after_loss });
    std.debug.print("prediction for x=2: {d:.4}\n", .{result.prediction_at_two});
}

fn requestedBackendType() nn.BackendType {
    if (nn.enable_cuda) return .CUDA;
    if (nn.enable_metal) return .Metal;
    return .CPU;
}

fn backendName(backend_type: nn.BackendType) []const u8 {
    return switch (backend_type) {
        .CPU => "cpu",
        .Metal => "metal",
        .CUDA => "cuda",
    };
}

fn runBackendTraining(allocator: std.mem.Allocator) !TrainingResult {
    var network = nn.Network.init(allocator, 0.05, .MeanSquaredError);
    defer network.deinit();
    try network.addLayer(1, 1, nn.Activation.linear, nn.Activation.linear_derivative);

    const layer = network.layers.items[0].Standard;
    layer.weights.fill(0.0);
    layer.bias.fill(0.0);

    var inputs = try Matrix.init(allocator, 4, 1);
    defer inputs.deinit();
    var targets = try Matrix.init(allocator, 4, 1);
    defer targets.deinit();

    for (0..4) |row| {
        const x = @as(f64, @floatFromInt(row));
        try inputs.set(row, 0, x);
        try targets.set(row, 0, x * 2.0);
    }

    var backend = try nn.createBackend(allocator, requestedBackendType());
    defer backend.deinit();

    const backend_inputs = try BackendMatrix.fromMatrix(backend, inputs, allocator);
    defer backend_inputs.deinit();
    const backend_targets = try BackendMatrix.fromMatrix(backend, targets, allocator);
    defer backend_targets.deinit();

    const before_predictions = try network.predictBackend(backend_inputs);
    defer before_predictions.deinit();
    const before_loss = try network.calculateLossBackend(before_predictions, backend_targets);

    for (0..32) |_| {
        _ = try network.trainBatchBackend(backend_inputs, backend_targets);
    }

    const after_predictions = try network.predictBackend(backend_inputs);
    defer after_predictions.deinit();
    const after_loss = try network.calculateLossBackend(after_predictions, backend_targets);

    return .{
        .backend_name = backendName(backend.getBackendType()),
        .before_loss = before_loss,
        .after_loss = after_loss,
        .prediction_at_two = after_predictions.get(2, 0),
    };
}

test "backend training example reduces loss" {
    const result = try runBackendTraining(std.testing.allocator);

    try std.testing.expect(result.after_loss < result.before_loss);
    try std.testing.expect(result.prediction_at_two > 2.0);
}
