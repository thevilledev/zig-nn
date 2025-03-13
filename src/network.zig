const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const testing = std.testing;
const Layer = @import("layer.zig").Layer;
const Matrix = @import("matrix.zig").Matrix;
const Activation = @import("activation.zig").Activation;

/// Neural Network structure
pub const Network = struct {
    layers: ArrayList(Layer),
    allocator: Allocator,

    /// Initialize a new neural network
    pub fn init(allocator: Allocator) Network {
        return Network{
            .layers = ArrayList(Layer).init(allocator),
            .allocator = allocator,
        };
    }

    /// Free all resources
    pub fn deinit(self: *Network) void {
        for (self.layers.items) |layer| {
            layer.deinit();
        }
        self.layers.deinit();
    }

    /// Add a new layer to the network
    pub fn addLayer(self: *Network, input_size: usize, output_size: usize, activation_fn: *const fn (f64) f64, activation_derivative_fn: *const fn (f64) f64) !void {
        const layer = try Layer.init(
            self.allocator,
            input_size,
            output_size,
            activation_fn,
            activation_derivative_fn
        );
        try self.layers.append(layer);
    }

    /// Forward propagation through the network
    pub fn forward(self: Network, input: Matrix) !Matrix {
        if (self.layers.items.len == 0) {
            return error.EmptyNetwork;
        }

        var current = input;
        std.debug.print("Input dimensions: {}x{}\n", .{ current.rows, current.cols });

        for (self.layers.items, 0..) |layer, i| {
            std.debug.print("Layer {} - Expected input size: {}, Current input cols: {}\n", 
                .{ i, layer.getInputSize(), current.cols });

            if (current.cols != layer.getInputSize()) {
                return error.InvalidInputDimensions;
            }

            const next = try layer.forward(current);
            std.debug.print("Layer {} output dimensions: {}x{}\n", .{ i, next.rows, next.cols });

            if (current.data.ptr != input.data.ptr) {
                current.deinit();
            }
            current = next;
        }

        return current;
    }

    /// Get the input size required by the network
    pub fn getInputSize(self: Network) !usize {
        if (self.layers.items.len == 0) {
            return error.EmptyNetwork;
        }
        return self.layers.items[0].getInputSize();
    }

    /// Get the output size produced by the network
    pub fn getOutputSize(self: Network) !usize {
        if (self.layers.items.len == 0) {
            return error.EmptyNetwork;
        }
        return self.layers.items[self.layers.items.len - 1].getOutputSize();
    }
};

// Tests
test "network initialization and layer addition" {
    const allocator = testing.allocator;

    var network = Network.init(allocator);
    defer network.deinit();

    try network.addLayer(2, 3, Activation.sigmoid, Activation.sigmoid_derivative);
    try network.addLayer(3, 1, Activation.sigmoid, Activation.sigmoid_derivative);

    try testing.expectEqual(@as(usize, 2), try network.getInputSize());
    try testing.expectEqual(@as(usize, 1), try network.getOutputSize());
}

test "network forward propagation" {
    const allocator = testing.allocator;

    var network = Network.init(allocator);
    defer network.deinit();

    // Create a simple network with 2 inputs, 3 hidden neurons, and 1 output
    try network.addLayer(2, 3, Activation.sigmoid, Activation.sigmoid_derivative);
    try network.addLayer(3, 1, Activation.sigmoid, Activation.sigmoid_derivative);

    // Set up input (1x2 matrix)
    var input = try Matrix.init(allocator, 1, 2);
    defer input.deinit();
    input.set(0, 0, 1.0);
    input.set(0, 1, 0.5);

    // Print dimensions before forward pass
    std.debug.print("\nNetwork test - Input matrix dimensions: {}x{}\n", .{ input.rows, input.cols });
    std.debug.print("Network test - First layer input size: {}\n", .{try network.getInputSize()});

    // Forward propagation
    var output = try network.forward(input);
    defer output.deinit();

    // Check output dimensions
    try testing.expectEqual(@as(usize, 1), output.rows);
    try testing.expectEqual(@as(usize, 1), output.cols);

    // The actual output value will depend on the random weights,
    // but it should be between 0 and 1 (sigmoid output range)
    const value = output.get(0, 0);
    try testing.expect(value >= 0.0 and value <= 1.0);
} 