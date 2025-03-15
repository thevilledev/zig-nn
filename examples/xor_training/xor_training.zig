const std = @import("std");
const nn = @import("zig-nn");
const Network = nn.Network;
const Matrix = nn.Matrix;
const Activation = nn.Activation;
const LossFunction = nn.LossFunction;

/// This example demonstrates training a neural network on the XOR problem
/// XOR truth table:
/// 0 XOR 0 = 0
/// 0 XOR 1 = 1
/// 1 XOR 0 = 1
/// 1 XOR 1 = 0
pub fn main() !void {
    // Initialize allocator
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create a console writer
    const stdout = std.io.getStdOut().writer();

    try stdout.print("\n=== XOR Neural Network Training Example ===\n\n", .{});

    // Create a network with two hidden layers and Mean Squared Error loss
    // Use a higher learning rate for faster convergence
    var network = Network.init(allocator, 0.3, .MeanSquaredError);
    defer network.deinit();

    // First hidden layer: 2 inputs -> 6 hidden neurons with tanh activation
    try network.addLayer(2, 6, Activation.tanh, Activation.tanh_derivative);

    // Second hidden layer: 6 inputs -> 4 hidden neurons with tanh activation
    try network.addLayer(6, 4, Activation.tanh, Activation.tanh_derivative);

    // Output layer: 4 inputs -> 1 output with sigmoid activation
    try network.addLayer(4, 1, Activation.sigmoid, Activation.sigmoid_derivative);

    try stdout.print("Network created with structure:\n", .{});
    try stdout.print("Input size: {}\n", .{try network.getInputSize()});
    try stdout.print("Output size: {}\n", .{try network.getOutputSize()});
    try stdout.print("Learning rate: {d}\n", .{network.learning_rate});
    try stdout.print("Loss function: {s}\n\n", .{@tagName(network.loss_function)});

    // Create XOR training data
    var inputs = try Matrix.init(allocator, 4, 2);
    defer inputs.deinit();
    inputs.set(0, 0, 0.0);
    inputs.set(0, 1, 0.0); // [0, 0] -> 0
    inputs.set(1, 0, 0.0);
    inputs.set(1, 1, 1.0); // [0, 1] -> 1
    inputs.set(2, 0, 1.0);
    inputs.set(2, 1, 0.0); // [1, 0] -> 1
    inputs.set(3, 0, 1.0);
    inputs.set(3, 1, 1.0); // [1, 1] -> 0

    var targets = try Matrix.init(allocator, 4, 1);
    defer targets.deinit();
    targets.set(0, 0, 0.0);
    targets.set(1, 0, 1.0);
    targets.set(2, 0, 1.0);
    targets.set(3, 0, 0.0);

    // Initial prediction before training
    {
        var initial_pred = try network.predict(inputs);
        defer initial_pred.deinit();

        try stdout.print("\nInitial predictions before training:\n", .{});
        try stdout.print("Input\t\tOutput\tExpected\n", .{});
        try stdout.print("-----\t\t------\t--------\n", .{});

        for (0..4) |i| {
            const x1 = inputs.get(i, 0);
            const x2 = inputs.get(i, 1);
            const y_pred = initial_pred.get(i, 0);
            const y_true = targets.get(i, 0);

            try stdout.print("[{d:.0}, {d:.0}]\t\t{d:.4}\t{d:.0}\n", .{ x1, x2, y_pred, y_true });
        }
    }

    // Train the network
    try stdout.print("\nTraining network on XOR problem...\n", .{});

    const epochs: usize = 10000;
    const batch_size: usize = 4;

    // Train the network using the built-in train method
    const loss_history = try network.train(inputs, targets, epochs, batch_size);
    defer allocator.free(loss_history);

    // Print training results
    try stdout.print("\nTraining completed after {d} epochs.\n", .{epochs});
    try stdout.print("Initial loss: {d:.6}\n", .{loss_history[0]});
    try stdout.print("Final loss: {d:.6}\n\n", .{loss_history[epochs - 1]});

    // Test the network
    try stdout.print("Testing trained network on XOR inputs:\n", .{});
    try stdout.print("Input\t\tOutput\tExpected\n", .{});
    try stdout.print("-----\t\t------\t--------\n", .{});

    var predictions = try network.predict(inputs);
    defer predictions.deinit();

    for (0..4) |i| {
        const x1 = inputs.get(i, 0);
        const x2 = inputs.get(i, 1);
        const y_pred = predictions.get(i, 0);
        const y_true = targets.get(i, 0);

        try stdout.print("[{d:.0}, {d:.0}]\t\t{d:.4}\t{d:.0}\n", .{ x1, x2, y_pred, y_true });
    }

    // Visualize decision boundary
    try stdout.print("\nDecision Boundary Visualization:\n", .{});
    const grid_size: usize = 20;
    const step: f64 = 1.0 / @as(f64, @floatFromInt(grid_size - 1));

    for (0..grid_size) |i| {
        const y = 1.0 - @as(f64, @floatFromInt(i)) * step;

        for (0..grid_size) |j| {
            const x = @as(f64, @floatFromInt(j)) * step;

            // Create input for a single point
            var point = try Matrix.init(allocator, 1, 2);
            defer point.deinit();
            point.set(0, 0, x);
            point.set(0, 1, y);

            // Predict
            var result = try network.predict(point);
            defer result.deinit();

            const output = result.get(0, 0);

            // Print character based on output value
            if (output < 0.3) {
                try stdout.print("·", .{});
            } else if (output > 0.7) {
                try stdout.print("O", .{});
            } else {
                try stdout.print("▒", .{});
            }
        }
        try stdout.print("\n", .{});
    }

    try stdout.print("\nLegend: · (0.0-0.3), ▒ (0.3-0.7), O (0.7-1.0)\n", .{});
}
