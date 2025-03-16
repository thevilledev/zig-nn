const std = @import("std");
const nn = @import("nn");
const Network = nn.Network;
const Matrix = nn.Matrix;
const Activation = nn.Activation;

pub fn main() !void {
    // Initialize allocator
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create a console writer
    const stdout = std.io.getStdOut().writer();

    try stdout.print("\n=== Gated Neural Network Example ===\n\n", .{});

    // Create a network with gated layers
    var network = Network.init(allocator, 0.1, .MeanSquaredError);
    defer network.deinit();

    try stdout.print("Creating network structure...\n", .{});

    // Input layer: 2 inputs -> 3 hidden neurons with ReLU activation
    try network.addLayer(2, 3, Activation.relu, Activation.relu_derivative);
    try stdout.print("Added input layer: 2 inputs -> 3 neurons (ReLU)\n", .{});

    // Hidden layer 1: 3 inputs -> 3 outputs with GLU activation
    try network.addGatedLayer(3, 3, false);
    try stdout.print("Added GLU layer: 3 inputs -> 3 outputs\n", .{});

    // Hidden layer 2: 3 inputs -> 2 outputs with SwiGLU activation
    try network.addGatedLayer(3, 2, true);
    try stdout.print("Added SwiGLU layer: 3 inputs -> 2 outputs\n", .{});

    // Output layer: 2 inputs -> 1 output with sigmoid activation
    try network.addLayer(2, 1, Activation.sigmoid, Activation.sigmoid_derivative);
    try stdout.print("Added output layer: 2 inputs -> 1 output (Sigmoid)\n", .{});

    try stdout.print("\nNetwork structure:\n", .{});
    try stdout.print("Input size: {}\n", .{try network.getInputSize()});
    try stdout.print("Output size: {}\n\n", .{try network.getOutputSize()});

    // Test forward propagation with sample inputs
    const samples = [_][2]f64{
        .{ 0.0, 0.0 },
        .{ 1.0, 0.0 },
        .{ 0.0, 1.0 },
        .{ 1.0, 1.0 },
    };

    try stdout.print("Running forward propagation on sample inputs:\n", .{});

    // Process each sample
    for (samples) |sample| {
        var input = try Matrix.init(allocator, 1, 2);
        defer input.deinit();

        input.set(0, 0, sample[0]);
        input.set(0, 1, sample[1]);

        var output = try network.forward(input);
        defer output.deinit();

        try stdout.print("Input [{d:.1}, {d:.1}] -> Output: {d:.6}\n", .{ sample[0], sample[1], output.get(0, 0) });
    }

    try stdout.print("\nNote: Since weights are initialized randomly, outputs will vary between runs.\n", .{});
    try stdout.print("This example demonstrates the structure and forward propagation of a neural network using GLU and SwiGLU layers.\n", .{});
}
