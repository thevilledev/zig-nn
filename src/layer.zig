const std = @import("std");
const Matrix = @import("matrix.zig").Matrix;
const Activation = @import("activation.zig").Activation;
const testing = std.testing;

/// Layer structure representing a single layer in the neural network
pub const Layer = struct {
    weights: Matrix,
    bias: Matrix,
    activation_fn: *const fn (f64) f64,
    activation_derivative_fn: *const fn (f64) f64,
    allocator: std.mem.Allocator,

    /// Initialize a new layer with random weights and biases
    pub fn init(
        allocator: std.mem.Allocator,
        input_size: usize,
        output_size: usize,
        activation_fn: *const fn (f64) f64,
        activation_derivative_fn: *const fn (f64) f64,
    ) !Layer {
        // For a layer with input_size inputs and output_size outputs,
        // we need a weight matrix of size (input_size x output_size)
        var weights = try Matrix.init(allocator, input_size, output_size);
        var bias = try Matrix.init(allocator, 1, output_size);

        // Initialize weights with small random values
        weights.randomize(-0.5, 0.5);
        bias.randomize(-0.5, 0.5);

        return Layer{
            .weights = weights,
            .bias = bias,
            .activation_fn = activation_fn,
            .activation_derivative_fn = activation_derivative_fn,
            .allocator = allocator,
        };
    }

    /// Free the layer's resources
    pub fn deinit(self: Layer) void {
        self.weights.deinit();
        self.bias.deinit();
    }

    /// Forward propagation through the layer
    pub fn forward(self: Layer, input: Matrix) !Matrix {
        // Input shape: (batch_size x input_size)
        // Weights shape: (input_size x output_size)
        // Result shape: (batch_size x output_size)
        var weighted_sum = try input.multiply(self.weights, self.allocator);
        defer weighted_sum.deinit();

        // Add bias to each row
        var biased = try weighted_sum.add(self.bias, self.allocator);
        defer biased.deinit();

        // Apply activation function
        return Activation.apply(biased, self.activation_fn, self.allocator);
    }

    /// Get the number of inputs this layer accepts
    pub fn getInputSize(self: Layer) usize {
        return self.weights.rows;
    }

    /// Get the number of outputs this layer produces
    pub fn getOutputSize(self: Layer) usize {
        return self.weights.cols;
    }
};

// Tests
test "layer initialization" {
    const allocator = testing.allocator;
    
    var layer = try Layer.init(
        allocator,
        3,  // input size
        2,  // output size
        Activation.sigmoid,
        Activation.sigmoid_derivative
    );
    defer layer.deinit();

    try testing.expectEqual(@as(usize, 3), layer.getInputSize());
    try testing.expectEqual(@as(usize, 2), layer.getOutputSize());
}

test "layer forward propagation" {
    const allocator = testing.allocator;
    
    // Create a layer with known weights and biases for testing
    var layer = try Layer.init(
        allocator,
        2,  // input size
        1,  // output size
        Activation.sigmoid,
        Activation.sigmoid_derivative
    );
    defer layer.deinit();

    // Set specific weights and bias for testing
    layer.weights.set(0, 0, 0.5);
    layer.weights.set(1, 0, 0.5);
    layer.bias.set(0, 0, 0.0);

    // Create input (1x2 matrix)
    var input = try Matrix.init(allocator, 1, 2);
    defer input.deinit();
    input.set(0, 0, 1.0);
    input.set(0, 1, 1.0);

    // Forward propagation
    var output = try layer.forward(input);
    defer output.deinit();

    // Expected output: sigmoid(1.0 * 0.5 + 1.0 * 0.5 + 0.0) = sigmoid(1.0) ≈ 0.7311
    try testing.expectApproxEqAbs(
        @as(f64, 0.7310585786300049),
        output.get(0, 0),
        0.0001
    );
} 