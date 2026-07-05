const std = @import("std");
const nn = @import("nn");
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
pub fn main(init: std.process.Init) !void {
    // Parse command line arguments
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    var output_file: ?[]const u8 = null;
    var arg_index: usize = 1;
    while (arg_index < args.len) : (arg_index += 1) {
        if (std.mem.eql(u8, args[arg_index], "--output")) {
            arg_index += 1;
            if (arg_index < args.len) output_file = args[arg_index];
        }
    }

    // Initialize allocator
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writerStreaming(std.Options.debug_io, &stdout_buffer);
    const writer = &stdout_writer.interface;
    defer writer.flush() catch {};

    try writer.print("\n=== XOR Neural Network Training Example ===\n\n", .{});

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

    try writer.print("Network Architecture:\n", .{});
    try writer.print("- Input layer:     2 neurons\n", .{});
    try writer.print("- Hidden layer 1:  6 neurons (tanh)\n", .{});
    try writer.print("- Hidden layer 2:  4 neurons (tanh)\n", .{});
    try writer.print("- Output layer:    1 neuron  (sigmoid)\n", .{});
    try writer.print("\nTraining Parameters:\n", .{});
    try writer.print("- Learning rate:   {d:.3}\n", .{network.learning_rate});
    try writer.print("- Loss function:   {s}\n", .{@tagName(network.loss_function)});
    try writer.flush();

    // Create XOR training data
    var inputs = try Matrix.init(allocator, 4, 2);
    defer inputs.deinit();
    try inputs.set(0, 0, 0.0);
    try inputs.set(0, 1, 0.0); // [0, 0] -> 0
    try inputs.set(1, 0, 0.0);
    try inputs.set(1, 1, 1.0); // [0, 1] -> 1
    try inputs.set(2, 0, 1.0);
    try inputs.set(2, 1, 0.0); // [1, 0] -> 1
    try inputs.set(3, 0, 1.0);
    try inputs.set(3, 1, 1.0); // [1, 1] -> 0

    var targets = try Matrix.init(allocator, 4, 1);
    defer targets.deinit();
    try targets.set(0, 0, 0.0);
    try targets.set(1, 0, 1.0);
    try targets.set(2, 0, 1.0);
    try targets.set(3, 0, 0.0);

    // Initial prediction before training
    {
        var initial_pred = try network.predict(inputs);
        defer initial_pred.deinit();

        try writer.print("\nInitial predictions (before training):\n", .{});
        try writer.print("┌─────────┬──────────┬──────────┐\n", .{});
        try writer.print("│ Input   │ Output   │ Expected │\n", .{});
        try writer.print("├─────────┼──────────┼──────────┤\n", .{});

        for (0..4) |i| {
            const x1 = try inputs.get(i, 0);
            const x2 = try inputs.get(i, 1);
            const y_pred = try initial_pred.get(i, 0);
            const y_true = try targets.get(i, 0);

            try writer.print("│ [{d}, {d}]  │ {d:.4}   │ {d}        │\n", .{ x1, x2, y_pred, y_true });
        }
        try writer.print("└─────────┴──────────┴──────────┘\n", .{});
        try writer.flush();
    }

    // Train the network
    try writer.print("\nTraining Progress:\n", .{});
    try writer.print("Each '=' represents 2% progress. Loss shown every 5%.\n", .{});
    try writer.flush();

    const epochs: usize = 10000;
    const batch_size: usize = 4;
    var last_progress: usize = 0;

    // Custom training loop for better progress display
    var seed: u64 = undefined;
    std.Options.debug_io.random(std.mem.asBytes(&seed));
    var prng = std.Random.DefaultPrng.init(seed);
    const rand = prng.random();
    var loss_history = try allocator.alloc(f64, epochs);
    defer allocator.free(loss_history);

    try writer.print("[", .{});
    try writer.flush();

    for (0..epochs) |epoch| {
        // Create a random permutation for shuffling
        var indices = try allocator.alloc(usize, inputs.rows);
        defer allocator.free(indices);
        for (0..inputs.rows) |i| {
            indices[i] = i;
        }
        rand.shuffle(usize, indices);

        // Train on shuffled batch
        var batch_inputs = try Matrix.init(allocator, batch_size, inputs.cols);
        defer batch_inputs.deinit();
        var batch_targets = try Matrix.init(allocator, batch_size, targets.cols);
        defer batch_targets.deinit();

        for (0..batch_size) |i| {
            const sample_idx = indices[i];
            for (0..inputs.cols) |j| {
                try batch_inputs.set(i, j, try inputs.get(sample_idx, j));
            }
            for (0..targets.cols) |j| {
                try batch_targets.set(i, j, try targets.get(sample_idx, j));
            }
        }

        const loss = try network.trainBatch(batch_inputs, batch_targets);
        loss_history[epoch] = loss;

        // Update progress bar
        const progress = (epoch * 50) / epochs;
        while (last_progress < progress) {
            try writer.print("=", .{});
            try writer.flush();
            last_progress += 1;
        }

        // Print loss every 5% (500 epochs)
        if (epoch % 500 == 0 or epoch == epochs - 1) {
            try writer.print(" {d:.4}", .{loss});
            try writer.flush();
        }
    }

    try writer.print("]\n\n", .{});
    try writer.flush();

    // Test the trained network
    try writer.print("Final predictions (after training):\n", .{});
    try writer.print("┌─────────┬──────────┬──────────┬──────────┐\n", .{});
    try writer.print("│ Input   │ Output   │ Expected │ Correct? │\n", .{});
    try writer.print("├─────────┼──────────┼──────────┼──────────┤\n", .{});

    var predictions = try network.predict(inputs);
    defer predictions.deinit();

    var correct: usize = 0;
    for (0..4) |i| {
        const x1 = try inputs.get(i, 0);
        const x2 = try inputs.get(i, 1);
        const y_pred = try predictions.get(i, 0);
        const y_true = try targets.get(i, 0);
        const is_correct = @abs(y_pred - y_true) < 0.5;
        if (is_correct) correct += 1;

        try writer.print("│ [{d}, {d}]  │ {d:.4}   │ {d}        │ {s}        │\n", .{ x1, x2, y_pred, y_true, if (is_correct) "✓" else "✗" });
    }
    try writer.print("└─────────┴──────────┴──────────┴──────────┘\n", .{});
    try writer.print("\nAccuracy: {d}/{d} correct predictions\n", .{ correct, 4 });
    try writer.flush();

    // Save the trained model if output file is specified
    if (output_file) |file| {
        try network.saveToFile(file);
        try writer.print("\nModel saved to: {s}\n", .{file});
        try writer.flush();
    }
}
