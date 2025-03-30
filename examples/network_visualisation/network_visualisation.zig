const std = @import("std");
const nn = @import("nn");
const Network = nn.Network;
const Matrix = nn.Matrix;
const Activation = nn.Activation;
const visualiseNetwork = nn.visualiseNetwork;

/// This example demonstrates visualizing different neural network architectures
/// using the ASCII visualization tool.
pub fn main() !void {
    // Initialize allocator
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create a console writer
    const stdout = std.io.getStdOut().writer();

    try stdout.print("\n=== Neural Network Visualization Examples ===\n\n", .{});

    // Example 1: Simple XOR Network (2-3-1)
    {
        try stdout.print("Example 1: Simple XOR Network (2-3-1)\n", .{});
        try stdout.print("----------------------------------------\n", .{});

        var network = Network.init(allocator, 0.1, .MeanSquaredError);
        defer network.deinit();

        try network.addLayer(2, 3, Activation.relu, Activation.relu_derivative);
        try network.addLayer(3, 1, Activation.sigmoid, Activation.sigmoid_derivative);

        var viz = try visualiseNetwork(&network, allocator);
        defer viz.deinit();

        try stdout.print("{s}\n\n", .{viz.items});
    }

    // Example 2: Multi-layer Network (3-5-4-2)
    {
        try stdout.print("Example 2: Multi-layer Network (3-5-4-2)\n", .{});
        try stdout.print("------------------------------------------\n", .{});

        var network = Network.init(allocator, 0.1, .MeanSquaredError);
        defer network.deinit();

        try network.addLayer(3, 5, Activation.relu, Activation.relu_derivative);
        try network.addLayer(5, 4, Activation.relu, Activation.relu_derivative);
        try network.addLayer(4, 2, Activation.sigmoid, Activation.sigmoid_derivative);

        var viz = try visualiseNetwork(&network, allocator);
        defer viz.deinit();

        try stdout.print("{s}\n\n", .{viz.items});
    }

    // Example 3: Deep Network (2-10-8-6-4-1)
    {
        try stdout.print("Example 3: Deep Network (2-10-8-6-4-1)\n", .{});
        try stdout.print("------------------------------------------\n", .{});

        var network = Network.init(allocator, 0.1, .MeanSquaredError);
        defer network.deinit();

        try network.addLayer(2, 10, Activation.tanh, Activation.tanh_derivative);
        try network.addLayer(10, 8, Activation.relu, Activation.relu_derivative);
        try network.addLayer(8, 6, Activation.relu, Activation.relu_derivative);
        try network.addLayer(6, 4, Activation.relu, Activation.relu_derivative);
        try network.addLayer(4, 1, Activation.sigmoid, Activation.sigmoid_derivative);

        var viz = try visualiseNetwork(&network, allocator);
        defer viz.deinit();

        try stdout.print("{s}\n\n", .{viz.items});
    }

    // Example 4: Wide Network (4-20-4)
    {
        try stdout.print("Example 4: Wide Network (4-20-4)\n", .{});
        try stdout.print("----------------------------------\n", .{});

        var network = Network.init(allocator, 0.1, .MeanSquaredError);
        defer network.deinit();

        try network.addLayer(4, 20, Activation.relu, Activation.relu_derivative);
        try network.addLayer(20, 4, Activation.sigmoid, Activation.sigmoid_derivative);

        var viz = try visualiseNetwork(&network, allocator);
        defer viz.deinit();

        try stdout.print("{s}\n\n", .{viz.items});
    }

    // Example 5: Gated Network with mixed layer types
    {
        try stdout.print("Example 5: Gated Network with mixed layer types\n", .{});
        try stdout.print("----------------------------------------------\n", .{});

        var network = Network.init(allocator, 0.1, .MeanSquaredError);
        defer network.deinit();

        try network.addLayer(2, 4, Activation.relu, Activation.relu_derivative);
        try network.addGatedLayer(4, 3, true); // Use SwiGLU activation
        try network.addLayer(3, 1, Activation.sigmoid, Activation.sigmoid_derivative);

        var viz = try visualiseNetwork(&network, allocator);
        defer viz.deinit();

        try stdout.print("{s}\n", .{viz.items});
    }

    try stdout.print("\nNetwork visualization allows for easier understanding of\n", .{});
    try stdout.print("neural network architecture and can help with debugging.\n", .{});
}
