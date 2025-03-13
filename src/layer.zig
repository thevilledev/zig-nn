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

/// Gated Layer structure implementing GLU and SwiGLU
pub const GatedLayer = struct {
    // Linear part weights and bias
    linear_weights: Matrix,
    linear_bias: Matrix,

    // Gating part weights and bias
    gate_weights: Matrix,
    gate_bias: Matrix,

    // Whether to use SwiGLU (true) or GLU (false)
    use_swiglu: bool,

    allocator: std.mem.Allocator,

    /// Initialize a new gated layer with random weights and biases
    pub fn init(
        allocator: std.mem.Allocator,
        input_size: usize,
        output_size: usize,
        use_swiglu: bool,
    ) !GatedLayer {
        // For a gated layer, we need two sets of weights and biases
        var linear_weights = try Matrix.init(allocator, input_size, output_size);
        var linear_bias = try Matrix.init(allocator, 1, output_size);
        var gate_weights = try Matrix.init(allocator, input_size, output_size);
        var gate_bias = try Matrix.init(allocator, 1, output_size);

        // Initialize weights with small random values
        linear_weights.randomize(-0.5, 0.5);
        linear_bias.randomize(-0.5, 0.5);
        gate_weights.randomize(-0.5, 0.5);
        gate_bias.randomize(-0.5, 0.5);

        return GatedLayer{
            .linear_weights = linear_weights,
            .linear_bias = linear_bias,
            .gate_weights = gate_weights,
            .gate_bias = gate_bias,
            .use_swiglu = use_swiglu,
            .allocator = allocator,
        };
    }

    /// Free the layer's resources
    pub fn deinit(self: GatedLayer) void {
        self.linear_weights.deinit();
        self.linear_bias.deinit();
        self.gate_weights.deinit();
        self.gate_bias.deinit();
    }

    /// Forward propagation through the gated layer
    pub fn forward(self: GatedLayer, input: Matrix) !Matrix {
        // Calculate linear part: input * linear_weights + linear_bias
        var linear_weighted_sum = try input.multiply(self.linear_weights, self.allocator);
        defer linear_weighted_sum.deinit();

        var linear_biased = try linear_weighted_sum.add(self.linear_bias, self.allocator);
        defer linear_biased.deinit();

        // Calculate gating part: input * gate_weights + gate_bias
        var gate_weighted_sum = try input.multiply(self.gate_weights, self.allocator);
        defer gate_weighted_sum.deinit();

        var gate_biased = try gate_weighted_sum.add(self.gate_bias, self.allocator);
        defer gate_biased.deinit();

        // Apply GLU or SwiGLU
        if (self.use_swiglu) {
            return Activation.applySwiGLU(linear_biased, gate_biased, self.allocator);
        } else {
            return Activation.applyGLU(linear_biased, gate_biased, self.allocator);
        }
    }

    /// Get the number of inputs this layer accepts
    pub fn getInputSize(self: GatedLayer) usize {
        return self.linear_weights.rows;
    }

    /// Get the number of outputs this layer produces
    pub fn getOutputSize(self: GatedLayer) usize {
        return self.linear_weights.cols;
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

test "gated layer initialization" {
    const allocator = testing.allocator;

    var glu_layer = try GatedLayer.init(
        allocator,
        3,  // input size
        2,  // output size
        false  // use GLU
    );
    defer glu_layer.deinit();

    try testing.expectEqual(@as(usize, 3), glu_layer.getInputSize());
    try testing.expectEqual(@as(usize, 2), glu_layer.getOutputSize());

    var swiglu_layer = try GatedLayer.init(
        allocator,
        4,  // input size
        3,  // output size
        true  // use SwiGLU
    );
    defer swiglu_layer.deinit();

    try testing.expectEqual(@as(usize, 4), swiglu_layer.getInputSize());
    try testing.expectEqual(@as(usize, 3), swiglu_layer.getOutputSize());
}

test "GLU layer forward propagation" {
    const allocator = testing.allocator;

    // Create a GLU layer with known weights and biases for testing
    var layer = try GatedLayer.init(
        allocator,
        2,  // input size
        1,  // output size
        false  // use GLU
    );
    defer layer.deinit();

    // Set specific weights and biases for testing
    layer.linear_weights.set(0, 0, 1.0);
    layer.linear_weights.set(1, 0, 1.0);
    layer.linear_bias.set(0, 0, 0.0);

    layer.gate_weights.set(0, 0, 0.5);
    layer.gate_weights.set(1, 0, 0.5);
    layer.gate_bias.set(0, 0, 0.0);

    // Create input (1x2 matrix)
    var input = try Matrix.init(allocator, 1, 2);
    defer input.deinit();
    input.set(0, 0, 1.0);
    input.set(0, 1, 1.0);

    // Forward propagation
    var output = try layer.forward(input);
    defer output.deinit();

    // Expected linear part: 1.0 * 1.0 + 1.0 * 1.0 + 0.0 = 2.0
    // Expected gate part: sigmoid(1.0 * 0.5 + 1.0 * 0.5 + 0.0) = sigmoid(1.0) ≈ 0.7311
    // Expected output: 2.0 * 0.7311 ≈ 1.4622
    try testing.expectApproxEqAbs(
        @as(f64, 1.4621171572600098),
        output.get(0, 0),
        0.0001
    );
}

test "SwiGLU layer forward propagation" {
    const allocator = testing.allocator;

    // Create a SwiGLU layer with known weights and biases for testing
    var layer = try GatedLayer.init(
        allocator,
        2,  // input size
        1,  // output size
        true  // use SwiGLU
    );
    defer layer.deinit();

    // Set specific weights and biases for testing
    layer.linear_weights.set(0, 0, 1.0);
    layer.linear_weights.set(1, 0, 1.0);
    layer.linear_bias.set(0, 0, 0.0);

    layer.gate_weights.set(0, 0, 0.5);
    layer.gate_weights.set(1, 0, 0.5);
    layer.gate_bias.set(0, 0, 0.0);

    // Create input (1x2 matrix)
    var input = try Matrix.init(allocator, 1, 2);
    defer input.deinit();
    input.set(0, 0, 1.0);
    input.set(0, 1, 1.0);

    // Forward propagation
    var output = try layer.forward(input);
    defer output.deinit();

    // Expected linear part: 1.0 * 1.0 + 1.0 * 1.0 + 0.0 = 2.0
    // Expected gate part: swish(1.0 * 0.5 + 1.0 * 0.5 + 0.0) = swish(1.0) ≈ 0.7311
    // Expected output: 2.0 * 0.7311 ≈ 1.4622
    try testing.expectApproxEqAbs(
        @as(f64, 1.4621171572600098),
        output.get(0, 0),
        0.0001
    );
}