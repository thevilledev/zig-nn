const std = @import("std");
const nn = @import("nn");
const Network = nn.Network;
const Matrix = nn.Matrix;
const Activation = nn.Activation;

/// This example demonstrates a simple neural network structure for the XOR problem
/// XOR truth table:
/// 0 XOR 0 = 0
/// 0 XOR 1 = 1
/// 1 XOR 0 = 1
/// 1 XOR 1 = 0
pub fn main() !void {
    // Initialize allocator
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writerStreaming(std.Options.debug_io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};

    try stdout.print("\n=== Simple Neural Network Example ===\n\n", .{});

    // Create a simple network with one hidden layer
    var network = Network.init(allocator, 0.1, .MeanSquaredError);
    defer network.deinit();

    // Input layer: 2 inputs -> 3 hidden neurons with ReLU activation
    try network.addLayer(2, 3, Activation.relu, Activation.relu_derivative);

    // Output layer: 3 inputs -> 1 output with sigmoid activation
    try network.addLayer(3, 1, Activation.sigmoid, Activation.sigmoid_derivative);

    try stdout.print("Network created with structure:\n", .{});
    try stdout.print("Input size: {}\n", .{try network.getInputSize()});
    try stdout.print("Output size: {}\n\n", .{try network.getOutputSize()});

    // Define XOR input data
    const inputs = [_][2]f64{
        .{ 0.0, 0.0 },
        .{ 0.0, 1.0 },
        .{ 1.0, 0.0 },
        .{ 1.0, 1.0 },
    };

    try stdout.print("Running forward propagation on XOR inputs:\n", .{});

    // Process each input
    for (inputs) |input_values| {
        var input = try Matrix.init(allocator, 1, 2);
        defer input.deinit();

        try input.set(0, 0, input_values[0]);
        try input.set(0, 1, input_values[1]);

        var output = try network.forward(input);
        defer output.deinit();

        try stdout.print("Input [{d:.1}, {d:.1}] -> Output: {d:.6}\n", .{ input_values[0], input_values[1], try output.get(0, 0) });
    }

    try stdout.print("\nNote: Since weights are initialized randomly, outputs will not match XOR truth table.\n", .{});
    try stdout.print("This example demonstrates a forward pass; run `zig build run_xor_training` for training.\n", .{});
}
