const std = @import("std");
const Matrix = @import("matrix.zig").Matrix;
const Activation = @import("activation.zig").Activation;
const testing = std.testing;

/// Neural network layer implementation supporting both standard and gated architectures
/// This module provides two types of layers:
/// 1. Layer: Standard neural network layer with weights, biases, and activation function
/// 2. GatedLayer: Advanced layer with gating mechanism (GLU/SwiGLU) for controlled information flow
///
/// Mathematical foundations:
/// - Standard layer: output = activation(W·x + b)
/// - Gated layer: output = (W₁·x + b₁) ⊗ activation(W₂·x + b₂)
/// Where:
/// - W, W₁, W₂ are weight matrices
/// - b, b₁, b₂ are bias vectors
/// - x is the input vector
/// - ⊗ is element-wise multiplication
/// Standard neural network layer with full connectivity
/// Implements forward and backward propagation for training
/// Maintains state for backpropagation by storing intermediate values
pub const Layer = struct {
    weights: Matrix, // Weight matrix W
    bias: Matrix, // Bias vector b
    activation_fn: *const fn (f64) f64, // Activation function σ(x)
    activation_derivative_fn: *const fn (f64) f64, // Derivative σ'(x)
    allocator: std.mem.Allocator,
    // Cached values for backpropagation
    last_input: ?Matrix, // Input x
    last_output: ?Matrix, // Output y = σ(W·x + b)
    last_weighted_sum: ?Matrix, // Pre-activation z = W·x + b

    /// Initializes a new fully connected layer
    /// Mathematical initialization:
    /// - W ∈ ℝ^(input_size × output_size), initialized uniformly in [-0.5, 0.5]
    /// - b ∈ ℝ^output_size, initialized uniformly in [-0.5, 0.5]
    ///
    /// Parameters:
    ///   - allocator: Memory allocator for matrices
    ///   - input_size: Number of input features
    ///   - output_size: Number of neurons in this layer
    ///   - activation_fn: Activation function σ(x)
    ///   - activation_derivative_fn: Derivative of activation function σ'(x)
    ///
    /// Time complexity: O(input_size * output_size)
    /// Memory complexity: O(input_size * output_size)
    pub fn init(
        allocator: std.mem.Allocator,
        input_size: usize,
        output_size: usize,
        activation_fn: *const fn (f64) f64,
        activation_derivative_fn: *const fn (f64) f64,
    ) !Layer {
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
            .last_input = null,
            .last_output = null,
            .last_weighted_sum = null,
        };
    }

    /// Frees all allocated memory
    /// Includes weights, biases, and cached matrices for backpropagation
    /// Time complexity: O(1)
    pub fn deinit(self: Layer) void {
        self.weights.deinit();
        self.bias.deinit();

        // Clean up stored matrices if they exist
        if (self.last_input) |input| {
            input.deinit();
        }
        if (self.last_output) |output| {
            output.deinit();
        }
        if (self.last_weighted_sum) |weighted_sum| {
            weighted_sum.deinit();
        }
    }

    /// Performs forward propagation through the layer
    /// Mathematical formula: y = σ(W·x + b)
    /// Where:
    /// - x is the input matrix (batch_size × input_size)
    /// - W is the weight matrix (input_size × output_size)
    /// - b is the bias vector (1 × output_size)
    /// - σ is the activation function
    ///
    /// Parameters:
    ///   - input: Input matrix x with shape (batch_size × input_size)
    /// Returns: Output matrix y with shape (batch_size × output_size)
    ///
    /// Time complexity: O(batch_size * input_size * output_size)
    /// Memory complexity: O(batch_size * output_size)
    pub fn forward(self: *Layer, input: Matrix) !Matrix {
        // Store a copy of the input for backpropagation
        if (self.last_input != null) {
            self.last_input.?.deinit();
            self.last_input = null;
        }
        self.last_input = try Matrix.copy(input, self.allocator);
        errdefer if (self.last_input) |last_input| last_input.deinit();

        // Calculate weighted sum z = W·x
        var weighted_sum = try input.dotProduct(self.weights, self.allocator);
        defer weighted_sum.deinit();

        // Store weighted sum for backpropagation
        if (self.last_weighted_sum != null) {
            self.last_weighted_sum.?.deinit();
            self.last_weighted_sum = null;
        }
        self.last_weighted_sum = try Matrix.copy(weighted_sum, self.allocator);
        errdefer if (self.last_weighted_sum) |last_weighted_sum| last_weighted_sum.deinit();

        // Add bias to each row (broadcast bias to match batch size)
        var biased = try Matrix.init(self.allocator, weighted_sum.rows, weighted_sum.cols);
        errdefer biased.deinit();
        for (0..weighted_sum.rows) |i| {
            for (0..weighted_sum.cols) |j| {
                biased.set(i, j, weighted_sum.get(i, j) + self.bias.get(0, j));
            }
        }

        // Apply activation function y = σ(z)
        const output = try Activation.apply(biased, self.activation_fn, self.allocator);
        biased.deinit();

        // Store output for backpropagation
        if (self.last_output != null) {
            self.last_output.?.deinit();
            self.last_output = null;
        }
        self.last_output = try Matrix.copy(output, self.allocator);
        errdefer if (self.last_output) |last_output| last_output.deinit();

        return output;
    }

    /// Performs backpropagation through the layer
    /// Implements gradient descent update step
    /// Mathematical formulas:
    /// 1. δz = δy ⊗ σ'(z) [element-wise]
    /// 2. δW = xᵀ·δz
    /// 3. δb = sum(δz, axis=0)
    /// 4. δx = δz·Wᵀ
    /// 5. W = W - η·δW
    /// 6. b = b - η·δb
    /// Where:
    /// - δy is the output gradient
    /// - z is the pre-activation (weighted sum)
    /// - σ' is the activation derivative
    /// - η is the learning rate
    ///
    /// Parameters:
    ///   - output_gradient: Gradient δy with respect to layer output
    ///   - learning_rate: Learning rate η for gradient descent
    /// Returns: Input gradient δx for backpropagating to previous layer
    ///
    /// Time complexity: O(batch_size * input_size * output_size)
    /// Memory complexity: O(batch_size * max(input_size, output_size))
    pub fn backward(self: *Layer, output_gradient: Matrix, learning_rate: f64) !Matrix {
        if (self.last_input == null or self.last_output == null or self.last_weighted_sum == null) {
            return error.NoForwardPassPerformed;
        }

        // Calculate activation derivative σ'(z)
        var activation_derivative = try Matrix.init(self.allocator, self.last_weighted_sum.?.rows, self.last_weighted_sum.?.cols);
        defer activation_derivative.deinit();

        for (0..self.last_weighted_sum.?.rows) |i| {
            for (0..self.last_weighted_sum.?.cols) |j| {
                const value = self.last_weighted_sum.?.get(i, j);
                activation_derivative.set(i, j, self.activation_derivative_fn(value));
            }
        }

        // Calculate δz = δy ⊗ σ'(z)
        var weighted_sum_gradient = try output_gradient.elementWiseMultiply(activation_derivative, self.allocator);

        // Calculate δW = xᵀ·δz
        var transposed_input = try self.last_input.?.transpose(self.allocator);
        defer transposed_input.deinit();

        var weights_gradient = try transposed_input.dotProduct(weighted_sum_gradient, self.allocator);
        defer weights_gradient.deinit();

        // Calculate δb = sum(δz, axis=0)
        var bias_gradient = try weighted_sum_gradient.sumRows(self.allocator);
        defer bias_gradient.deinit();

        // Calculate δx = δz·Wᵀ
        var transposed_weights = try self.weights.transpose(self.allocator);
        defer transposed_weights.deinit();

        const input_gradient = try weighted_sum_gradient.dotProduct(transposed_weights, self.allocator);
        defer weighted_sum_gradient.deinit();

        // Update weights: W = W - η·δW
        var scaled_weights_gradient = try weights_gradient.scale(learning_rate, self.allocator);
        defer scaled_weights_gradient.deinit();

        const new_weights = try self.weights.subtract(scaled_weights_gradient, self.allocator);
        self.weights.deinit();
        self.weights = new_weights;

        // Update bias: b = b - η·δb
        var scaled_bias_gradient = try bias_gradient.scale(learning_rate, self.allocator);
        defer scaled_bias_gradient.deinit();

        const new_bias = try self.bias.subtract(scaled_bias_gradient, self.allocator);
        self.bias.deinit();
        self.bias = new_bias;

        return input_gradient;
    }

    /// Returns the number of input features the layer accepts
    pub fn getInputSize(self: Layer) usize {
        return self.weights.rows;
    }

    /// Returns the number of neurons (outputs) in the layer
    pub fn getOutputSize(self: Layer) usize {
        return self.weights.cols;
    }
};

/// Advanced neural network layer with gating mechanism
/// Implements either GLU (Gated Linear Unit) or SwiGLU variant
/// GLU and SwiGLU are particularly effective in transformer architectures
/// Mathematical formula:
/// - GLU: output = (W₁·x + b₁) ⊗ sigmoid(W₂·x + b₂)
/// - SwiGLU: output = (W₁·x + b₁) ⊗ swish(W₂·x + b₂)
pub const GatedLayer = struct {
    // Linear transformation parameters
    linear_weights: Matrix, // W₁
    linear_bias: Matrix, // b₁

    // Gating transformation parameters
    gate_weights: Matrix, // W₂
    gate_bias: Matrix, // b₂

    // Configuration
    use_swiglu: bool, // Whether to use SwiGLU (true) or GLU (false)

    allocator: std.mem.Allocator,

    // Cached values for backpropagation
    last_input: ?Matrix, // Input x
    last_linear_output: ?Matrix, // Linear part: W₁·x + b₁
    last_gate_output: ?Matrix, // Gate part: W₂·x + b₂
    last_output: ?Matrix, // Final output

    /// Initializes a new gated layer (GLU or SwiGLU)
    /// Mathematical initialization:
    /// - W₁, W₂ ∈ ℝ^(input_size × output_size), initialized uniformly in [-0.5, 0.5]
    /// - b₁, b₂ ∈ ℝ^output_size, initialized uniformly in [-0.5, 0.5]
    ///
    /// Parameters:
    ///   - allocator: Memory allocator for matrices
    ///   - input_size: Number of input features
    ///   - output_size: Number of neurons in this layer
    ///   - use_swiglu: Whether to use SwiGLU (true) or GLU (false)
    ///
    /// Time complexity: O(input_size * output_size)
    /// Memory complexity: O(input_size * output_size)
    pub fn init(
        allocator: std.mem.Allocator,
        input_size: usize,
        output_size: usize,
        use_swiglu: bool,
    ) !GatedLayer {
        // Initialize two sets of weights and biases
        var linear_weights = try Matrix.init(allocator, input_size, output_size);
        var linear_bias = try Matrix.init(allocator, 1, output_size);
        var gate_weights = try Matrix.init(allocator, input_size, output_size);
        var gate_bias = try Matrix.init(allocator, 1, output_size);

        // Initialize with small random values
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
            .last_input = null,
            .last_linear_output = null,
            .last_gate_output = null,
            .last_output = null,
        };
    }

    /// Frees all allocated memory
    /// Includes both sets of weights and biases, and cached matrices
    /// Time complexity: O(1)
    pub fn deinit(self: GatedLayer) void {
        self.linear_weights.deinit();
        self.linear_bias.deinit();
        self.gate_weights.deinit();
        self.gate_bias.deinit();

        // Clean up stored matrices
        if (self.last_input) |input| {
            input.deinit();
        }
        if (self.last_linear_output) |output| {
            output.deinit();
        }
        if (self.last_gate_output) |output| {
            output.deinit();
        }
        if (self.last_output) |output| {
            output.deinit();
        }
    }

    /// Forward propagation through the gated layer
    pub fn forward(self: *GatedLayer, input: Matrix) !Matrix {
        // Store a copy of the input for backpropagation
        if (self.last_input != null) {
            self.last_input.?.deinit();
            self.last_input = null;
        }
        self.last_input = try Matrix.copy(input, self.allocator);
        errdefer if (self.last_input) |last_input| last_input.deinit();

        // Calculate linear part: input * linear_weights + linear_bias
        var linear_weighted_sum = try input.dotProduct(self.linear_weights, self.allocator);
        defer linear_weighted_sum.deinit();

        // Add linear bias to each row (broadcast bias to match batch size)
        var linear_biased = try Matrix.init(self.allocator, linear_weighted_sum.rows, linear_weighted_sum.cols);
        errdefer linear_biased.deinit();
        for (0..linear_weighted_sum.rows) |i| {
            for (0..linear_weighted_sum.cols) |j| {
                linear_biased.set(i, j, linear_weighted_sum.get(i, j) + self.linear_bias.get(0, j));
            }
        }

        // Store linear output for backpropagation
        if (self.last_linear_output != null) {
            self.last_linear_output.?.deinit();
            self.last_linear_output = null;
        }
        self.last_linear_output = try Matrix.copy(linear_biased, self.allocator);
        errdefer if (self.last_linear_output) |last_linear_output| last_linear_output.deinit();

        // Calculate gating part: input * gate_weights + gate_bias
        var gate_weighted_sum = try input.dotProduct(self.gate_weights, self.allocator);
        defer gate_weighted_sum.deinit();

        // Add gate bias to each row (broadcast bias to match batch size)
        var gate_biased = try Matrix.init(self.allocator, gate_weighted_sum.rows, gate_weighted_sum.cols);
        errdefer gate_biased.deinit();
        for (0..gate_weighted_sum.rows) |i| {
            for (0..gate_weighted_sum.cols) |j| {
                gate_biased.set(i, j, gate_weighted_sum.get(i, j) + self.gate_bias.get(0, j));
            }
        }

        // Store gate output for backpropagation
        if (self.last_gate_output != null) {
            self.last_gate_output.?.deinit();
            self.last_gate_output = null;
        }
        self.last_gate_output = try Matrix.copy(gate_biased, self.allocator);
        errdefer if (self.last_gate_output) |last_gate_output| last_gate_output.deinit();

        // Apply GLU or SwiGLU
        var output: Matrix = undefined;
        if (self.use_swiglu) {
            output = try Activation.applySwiGLU(linear_biased, gate_biased, self.allocator);
        } else {
            output = try Activation.applyGLU(linear_biased, gate_biased, self.allocator);
        }
        linear_biased.deinit();
        gate_biased.deinit();

        // Store output for backpropagation
        if (self.last_output != null) {
            self.last_output.?.deinit();
            self.last_output = null;
        }
        self.last_output = try Matrix.copy(output, self.allocator);
        errdefer if (self.last_output) |last_output| last_output.deinit();

        return output;
    }

    /// Backpropagation through the gated layer
    /// output_gradient: gradient of the loss with respect to the layer's output
    /// learning_rate: how fast the network learns
    /// Returns: gradient of the loss with respect to the layer's input
    pub fn backward(self: *GatedLayer, output_gradient: Matrix, learning_rate: f64) !Matrix {
        if (self.last_input == null or self.last_linear_output == null or
            self.last_gate_output == null or self.last_output == null)
        {
            return error.NoForwardPassPerformed;
        }

        // Gradients for GLU or SwiGLU
        var linear_gradient: Matrix = undefined;
        var gate_gradient: Matrix = undefined;

        if (self.use_swiglu) {
            // SwiGLU: output = linear * swish(gate)
            // Compute gradients for SwiGLU
            var swish_gate = try Activation.apply(self.last_gate_output.?, Activation.swish, self.allocator);
            defer swish_gate.deinit();

            var swish_derivative = try Activation.apply(self.last_gate_output.?, Activation.swish_derivative, self.allocator);
            defer swish_derivative.deinit();

            // Gradient with respect to linear part
            linear_gradient = try swish_gate.elementWiseMultiply(output_gradient, self.allocator);

            // Gradient with respect to gate part
            var linear_times_gradient = try self.last_linear_output.?.elementWiseMultiply(output_gradient, self.allocator);
            defer linear_times_gradient.deinit();

            gate_gradient = try linear_times_gradient.elementWiseMultiply(swish_derivative, self.allocator);
        } else {
            // GLU: output = linear * sigmoid(gate)
            // Compute gradients for GLU
            var sigmoid_gate = try Activation.apply(self.last_gate_output.?, Activation.sigmoid, self.allocator);
            defer sigmoid_gate.deinit();

            var sigmoid_derivative = try Activation.apply(self.last_gate_output.?, Activation.sigmoid_derivative, self.allocator);
            defer sigmoid_derivative.deinit();

            // Gradient with respect to linear part
            linear_gradient = try sigmoid_gate.elementWiseMultiply(output_gradient, self.allocator);

            // Gradient with respect to gate part
            var linear_times_gradient = try self.last_linear_output.?.elementWiseMultiply(output_gradient, self.allocator);
            defer linear_times_gradient.deinit();

            gate_gradient = try linear_times_gradient.elementWiseMultiply(sigmoid_derivative, self.allocator);
        }
        defer linear_gradient.deinit();
        defer gate_gradient.deinit();

        // Calculate gradients with respect to weights and biases
        var transposed_input = try self.last_input.?.transpose(self.allocator);
        defer transposed_input.deinit();

        // Linear weights gradient
        var linear_weights_gradient = try transposed_input.dotProduct(linear_gradient, self.allocator);
        defer linear_weights_gradient.deinit();

        // Gate weights gradient
        var gate_weights_gradient = try transposed_input.dotProduct(gate_gradient, self.allocator);
        defer gate_weights_gradient.deinit();

        // Linear bias gradient (sum along batch dimension)
        var linear_bias_gradient = try linear_gradient.sumRows(self.allocator);
        defer linear_bias_gradient.deinit();

        // Gate bias gradient (sum along batch dimension)
        var gate_bias_gradient = try gate_gradient.sumRows(self.allocator);
        defer gate_bias_gradient.deinit();

        // Calculate gradient with respect to input
        var transposed_linear_weights = try self.linear_weights.transpose(self.allocator);
        defer transposed_linear_weights.deinit();

        var transposed_gate_weights = try self.gate_weights.transpose(self.allocator);
        defer transposed_gate_weights.deinit();

        var linear_input_gradient = try linear_gradient.dotProduct(transposed_linear_weights, self.allocator);
        defer linear_input_gradient.deinit();

        var gate_input_gradient = try gate_gradient.dotProduct(transposed_gate_weights, self.allocator);
        defer gate_input_gradient.deinit();

        const input_gradient = try linear_input_gradient.add(gate_input_gradient, self.allocator);

        // Update weights and biases
        // Scale gradients by learning rate
        var scaled_linear_weights_gradient = try linear_weights_gradient.scale(learning_rate, self.allocator);
        defer scaled_linear_weights_gradient.deinit();

        var scaled_gate_weights_gradient = try gate_weights_gradient.scale(learning_rate, self.allocator);
        defer scaled_gate_weights_gradient.deinit();

        var scaled_linear_bias_gradient = try linear_bias_gradient.scale(learning_rate, self.allocator);
        defer scaled_linear_bias_gradient.deinit();

        var scaled_gate_bias_gradient = try gate_bias_gradient.scale(learning_rate, self.allocator);
        defer scaled_gate_bias_gradient.deinit();

        // Update linear weights
        const new_linear_weights = try self.linear_weights.subtract(scaled_linear_weights_gradient, self.allocator);
        self.linear_weights.deinit();
        self.linear_weights = new_linear_weights;

        // Update gate weights
        const new_gate_weights = try self.gate_weights.subtract(scaled_gate_weights_gradient, self.allocator);
        self.gate_weights.deinit();
        self.gate_weights = new_gate_weights;

        // Update linear bias
        const new_linear_bias = try self.linear_bias.subtract(scaled_linear_bias_gradient, self.allocator);
        self.linear_bias.deinit();
        self.linear_bias = new_linear_bias;

        // Update gate bias
        const new_gate_bias = try self.gate_bias.subtract(scaled_gate_bias_gradient, self.allocator);
        self.gate_bias.deinit();
        self.gate_bias = new_gate_bias;

        return input_gradient;
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

    var layer = try Layer.init(allocator, 3, // input size
        2, // output size
        Activation.sigmoid, Activation.sigmoid_derivative);
    defer layer.deinit();

    try testing.expectEqual(@as(usize, 3), layer.getInputSize());
    try testing.expectEqual(@as(usize, 2), layer.getOutputSize());
}

test "layer forward propagation" {
    const allocator = testing.allocator;

    // Create a layer with known weights and biases for testing
    var layer = try Layer.init(allocator, 2, // input size
        1, // output size
        Activation.sigmoid, Activation.sigmoid_derivative);
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
    try testing.expectApproxEqAbs(@as(f64, 0.7310585786300049), output.get(0, 0), 0.0001);
}

test "gated layer initialization" {
    const allocator = testing.allocator;

    var glu_layer = try GatedLayer.init(allocator, 3, // input size
        2, // output size
        false // use GLU
    );
    defer glu_layer.deinit();

    try testing.expectEqual(@as(usize, 3), glu_layer.getInputSize());
    try testing.expectEqual(@as(usize, 2), glu_layer.getOutputSize());

    var swiglu_layer = try GatedLayer.init(allocator, 4, // input size
        3, // output size
        true // use SwiGLU
    );
    defer swiglu_layer.deinit();

    try testing.expectEqual(@as(usize, 4), swiglu_layer.getInputSize());
    try testing.expectEqual(@as(usize, 3), swiglu_layer.getOutputSize());
}

test "GLU layer forward propagation" {
    const allocator = testing.allocator;

    // Create a GLU layer with known weights and biases for testing
    var layer = try GatedLayer.init(allocator, 2, // input size
        1, // output size
        false // use GLU
    );
    defer layer.deinit();

    // Set specific weights and biases for testing
    layer.linear_weights.set(0, 0, 1.0);
    layer.linear_weights.set(1, 0, 1.0);
    layer.linear_bias.set(0, 0, 0.0);

    layer.gate_weights.set(0, 0, 0.0);
    layer.gate_weights.set(1, 0, 0.0);
    layer.gate_bias.set(0, 0, 0.0); // sigmoid(0) = 0.5

    // Create input (1x2 matrix)
    var input = try Matrix.init(allocator, 1, 2);
    defer input.deinit();
    input.set(0, 0, 1.0);
    input.set(0, 1, 1.0);

    // Forward propagation
    var output = try layer.forward(input);
    defer output.deinit();

    // Expected output: (1.0 * 1.0 + 1.0 * 1.0 + 0.0) * sigmoid(0.0) = 2.0 * 0.5 = 1.0
    try testing.expectApproxEqAbs(@as(f64, 1.0), output.get(0, 0), 0.0001);
}

test "layer backpropagation" {
    const allocator = testing.allocator;

    // Create a layer with known weights and biases for testing
    var layer = try Layer.init(allocator, 2, // input size
        1, // output size
        Activation.sigmoid, Activation.sigmoid_derivative);
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

    // Create output gradient (1x1 matrix)
    var output_gradient = try Matrix.init(allocator, 1, 1);
    defer output_gradient.deinit();
    output_gradient.set(0, 0, 1.0);

    // Backward propagation
    var input_gradient = try layer.backward(output_gradient, 0.1);
    defer input_gradient.deinit();

    // Check that input gradient has correct dimensions
    try testing.expectEqual(@as(usize, 1), input_gradient.rows);
    try testing.expectEqual(@as(usize, 2), input_gradient.cols);
}

test "gated layer backpropagation" {
    const allocator = testing.allocator;

    // Create a GLU layer with known weights and biases for testing
    var layer = try GatedLayer.init(allocator, 2, // input size
        1, // output size
        false // use GLU
    );
    defer layer.deinit();

    // Set specific weights and biases for testing
    layer.linear_weights.set(0, 0, 1.0);
    layer.linear_weights.set(1, 0, 1.0);
    layer.linear_bias.set(0, 0, 0.0);

    layer.gate_weights.set(0, 0, 0.0);
    layer.gate_weights.set(1, 0, 0.0);
    layer.gate_bias.set(0, 0, 0.0); // sigmoid(0) = 0.5

    // Create input (1x2 matrix)
    var input = try Matrix.init(allocator, 1, 2);
    defer input.deinit();
    input.set(0, 0, 1.0);
    input.set(0, 1, 1.0);

    // Forward propagation
    var output = try layer.forward(input);
    defer output.deinit();

    // Create output gradient (1x1 matrix)
    var output_gradient = try Matrix.init(allocator, 1, 1);
    defer output_gradient.deinit();
    output_gradient.set(0, 0, 1.0);

    // Backward propagation
    var input_gradient = try layer.backward(output_gradient, 0.1);
    defer input_gradient.deinit();

    // Check that input gradient has correct dimensions
    try testing.expectEqual(@as(usize, 1), input_gradient.rows);
    try testing.expectEqual(@as(usize, 2), input_gradient.cols);
}
