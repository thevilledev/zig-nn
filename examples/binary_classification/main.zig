const std = @import("std");
const nn = @import("zig-nn");

const Matrix = nn.Matrix;
const Network = nn.Network;
const Layer = nn.Layer;
const Activation = nn.Activation;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create a simple network for binary classification with a smaller learning rate
    var net = Network.init(allocator, 0.001, .CrossEntropy);
    defer net.deinit();

    // Add layers: 2 inputs -> 16 hidden -> 16 hidden -> 1 output
    // Using larger hidden layers for better capacity
    try net.addLayer(2, 16, Activation.relu, Activation.relu_derivative);
    try net.addLayer(16, 16, Activation.relu, Activation.relu_derivative);
    try net.addLayer(16, 1, Activation.sigmoid, Activation.sigmoid_derivative);

    // Generate training data
    const num_points = 1000;
    var training_data = try Matrix.init(allocator, num_points, 2);
    var labels = try Matrix.init(allocator, num_points, 1);
    defer training_data.deinit();
    defer labels.deinit();

    var prng = std.Random.DefaultPrng.init(42);
    var random = prng.random();

    // Generate points and labels
    var i: usize = 0;
    while (i < num_points) : (i += 1) {
        // Generate random points in [-1, 1] x [-1, 1]
        const x = random.float(f64) * 2.0 - 1.0;
        const y = random.float(f64) * 2.0 - 1.0;
        training_data.set(i, 0, x);
        training_data.set(i, 1, y);

        // Label: 1 if inside unit circle, 0 if outside
        const distance = std.math.sqrt(x * x + y * y);
        labels.set(i, 0, if (distance <= 0.5) 1.0 else 0.0);
    }

    // Training parameters
    const num_epochs = 1000;
    const batch_size = 32;

    std.debug.print("Training network...\n", .{});

    // Training loop with mini-batches
    var epoch: usize = 0;
    while (epoch < num_epochs) : (epoch += 1) {
        var total_loss: f64 = 0.0;
        var batch_start: usize = 0;

        while (batch_start < num_points) : (batch_start += batch_size) {
            const batch_end = @min(batch_start + batch_size, num_points);
            const batch_inputs = try extractBatch(training_data, batch_start, batch_end, allocator);
            const batch_labels = try extractBatch(labels, batch_start, batch_end, allocator);
            defer batch_inputs.deinit();
            defer batch_labels.deinit();

            const loss = try net.trainBatch(batch_inputs, batch_labels);
            total_loss += loss;
        }

        if (epoch % 100 == 0) {
            const avg_loss = total_loss / @as(f64, @floatFromInt((num_points + batch_size - 1) / batch_size));
            std.debug.print("Epoch {d}: Loss = {d:.6}\n", .{ epoch, avg_loss });
        }
    }

    // Test the network on a grid of points
    const grid_size = 20;
    const step = 2.0 / @as(f64, @floatFromInt(grid_size));

    std.debug.print("\nDecision boundary:\n", .{});
    var y: f64 = -1.0;
    var row: usize = 0;
    while (row < grid_size) : (row += 1) {
        var x: f64 = -1.0;
        var col: usize = 0;
        while (col < grid_size) : (col += 1) {
            var test_point = try Matrix.init(allocator, 1, 2);
            defer test_point.deinit();
            test_point.set(0, 0, x);
            test_point.set(0, 1, y);

            const output = try net.forward(test_point);
            defer output.deinit();

            const prediction = output.get(0, 0);
            const symbol = if (prediction > 0.5) "●" else "○";
            std.debug.print("{s} ", .{symbol});

            x += step;
        }
        std.debug.print("\n", .{});
        y += step;
    }
}
