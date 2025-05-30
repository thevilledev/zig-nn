const std = @import("std");
const math = std.math;
const testing = std.testing;
const Matrix = @import("matrix.zig").Matrix;

/// Activation functions and their derivatives for neural networks
/// These functions introduce non-linearity into the network, allowing it to learn complex patterns
/// Each activation function has specific properties that make it suitable for different tasks:
/// - Sigmoid: Classic activation, outputs in range (0,1), good for binary classification
/// - ReLU: Most popular modern activation, helps prevent vanishing gradients
/// - Tanh: Similar to sigmoid but outputs in range (-1,1), often better for deep networks
/// - Swish: Self-gated activation function, smooth and non-monotonic
pub const Activation = struct {
    /// Sigmoid activation function
    /// Mathematical form: σ(x) = 1 / (1 + e^(-x))
    /// Properties:
    /// - Output range: (0,1)
    /// - Smooth and differentiable
    /// - Can cause vanishing gradients for deep networks
    /// - Historically popular for classification tasks
    pub fn sigmoid(x: f64) f64 {
        return 1.0 / (1.0 + math.exp(-x));
    }

    /// Linear activation function
    /// Mathematical form: f(x) = x
    /// Properties:
    /// - Output range: (-∞,∞)
    /// - No transformation of input
    /// - Useful for regression tasks
    /// - Preserves both positive and negative values
    pub fn linear(x: f64) f64 {
        return x;
    }

    /// Derivative of linear function
    /// Mathematical form: f'(x) = 1
    /// Properties:
    /// - Constant derivative of 1
    /// - Helps maintain gradient flow in regression tasks
    pub fn linear_derivative(_: f64) f64 {
        return 1.0;
    }

    /// Derivative of sigmoid function
    /// Mathematical form: σ'(x) = σ(x)(1 - σ(x))
    /// Used in backpropagation to compute gradients
    /// Properties:
    /// - Maximum value of 0.25 at x = 0
    /// - Approaches 0 for large |x|
    pub fn sigmoid_derivative(x: f64) f64 {
        const sig = sigmoid(x);
        return sig * (1.0 - sig);
    }

    /// ReLU (Rectified Linear Unit) activation function
    /// Mathematical form: f(x) = max(0, x)
    /// Properties:
    /// - Output range: [0,∞)
    /// - Non-differentiable at x = 0
    /// - Helps prevent vanishing gradients
    /// - Computationally efficient
    /// - Most widely used activation in modern deep learning
    pub fn relu(x: f64) f64 {
        return if (x > 0) x else 0;
    }

    /// Derivative of ReLU function
    /// Mathematical form: f'(x) = 1 if x > 0, else 0
    /// Properties:
    /// - Binary output: {0,1}
    /// - Can cause "dying ReLU" problem when neurons get stuck in negative region
    pub fn relu_derivative(x: f64) f64 {
        return if (x > 0) 1 else 0;
    }

    /// Hyperbolic tangent (tanh) activation function
    /// Mathematical form: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    /// Properties:
    /// - Output range: (-1,1)
    /// - Zero-centered outputs
    /// - Stronger gradients than sigmoid
    /// - Still can have vanishing gradient in deep networks
    pub fn tanh(x: f64) f64 {
        return math.tanh(x);
    }

    /// Derivative of tanh function
    /// Mathematical form: tanh'(x) = 1 - tanh²(x)
    /// Properties:
    /// - Maximum value of 1 at x = 0
    /// - Approaches 0 for large |x|
    pub fn tanh_derivative(x: f64) f64 {
        const t = tanh(x);
        return 1.0 - (t * t);
    }

    /// Swish activation function
    /// Mathematical form: f(x) = x * sigmoid(β*x), where β is a learnable parameter
    /// Here β is fixed to 1.0 for simplicity
    /// Properties:
    /// - Smooth and non-monotonic
    /// - Unbounded above, bounded below
    /// - Self-gated activation
    /// - Sometimes outperforms ReLU in deep networks
    pub fn swish(x: f64) f64 {
        const beta: f64 = 1.0;
        return x * sigmoid(beta * x);
    }

    /// Derivative of swish function
    /// Mathematical form: f'(x) = β*sigmoid(β*x) + x*β*sigmoid(β*x)*(1-sigmoid(β*x))
    /// Properties:
    /// - More complex than other derivatives
    /// - Non-monotonic like the function itself
    pub fn swish_derivative(x: f64) f64 {
        const beta: f64 = 1.0;
        const sig = sigmoid(beta * x);
        return sig + beta * x * sig * (1.0 - sig);
    }

    /// Softmax activation function (element-wise exponential)
    /// Note: This computes the exponential part of softmax.
    /// Properties:
    /// - Applies scaled exponential function to input
    /// - Scaling prevents overflow/underflow
    /// - Used as part of the complete softmax computation
    pub fn softmax(x: f64) f64 {
        // For element-wise application in the network,
        // we want to keep values in a reasonable range
        // but not apply exp() here - that happens in applySoftmax
        return x;
    }

    /// Applies softmax activation to a matrix row-wise
    /// Mathematical form: softmax(x_i) = exp(x_i) / Σ(exp(x_j))
    /// Uses the log-sum-exp trick for numerical stability:
    /// softmax(x_i) = exp(x_i - max) / Σ(exp(x_j - max))
    /// Time complexity: O(rows * cols)
    /// Memory complexity: O(rows * cols)
    pub fn applySoftmax(matrix: Matrix, allocator: std.mem.Allocator) !Matrix {
        var result = try Matrix.init(allocator, matrix.rows, matrix.cols);
        errdefer result.deinit();

        // Process each row independently
        for (0..matrix.rows) |i| {
            // Find max value in row for numerical stability
            var max_val: f64 = -math.inf(f64);
            for (0..matrix.cols) |j| {
                max_val = @max(max_val, matrix.get(i, j));
            }

            // Compute exp(x - max) and sum
            var sum: f64 = 0.0;
            for (0..matrix.cols) |j| {
                const shifted = matrix.get(i, j) - max_val;
                const exp_val = math.exp(shifted);
                result.set(i, j, exp_val);
                sum += exp_val;
            }

            // Normalize by sum
            for (0..matrix.cols) |j| {
                result.set(i, j, result.get(i, j) / sum);
            }
        }

        return result;
    }

    /// Derivative of softmax function with respect to cross-entropy loss
    /// When used with cross-entropy loss, the derivative simplifies to (output - target)
    /// This is because the derivative of softmax combined with cross-entropy loss
    /// has a special form that cancels out the complex terms
    pub fn softmax_derivative(x: f64) f64 {
        return x;
    }

    /// Applies an activation function to each element of a matrix
    /// Essential for forward propagation in neural networks
    /// Time complexity: O(rows * cols)
    /// Memory complexity: O(rows * cols)
    ///
    /// Parameters:
    ///   - matrix: Input matrix to apply activation to
    ///   - func: Pointer to activation function
    ///   - allocator: Memory allocator for result matrix
    /// Returns: New matrix with activation applied element-wise
    pub fn apply(matrix: Matrix, func: *const fn (f64) f64, allocator: std.mem.Allocator) !Matrix {
        var result = try Matrix.init(allocator, matrix.rows, matrix.cols);

        for (0..matrix.rows) |i| {
            for (0..matrix.cols) |j| {
                const value = matrix.get(i, j);
                result.set(i, j, func(value));
            }
        }

        return result;
    }

    /// Applies Gated Linear Unit (GLU) activation
    /// Mathematical form: GLU(x,y) = x ⊗ sigmoid(y)
    /// Where ⊗ is element-wise multiplication
    /// Properties:
    /// - Provides multiplicative interaction
    /// - Helps control information flow
    /// - Used in transformer architectures
    /// Time complexity: O(rows * cols)
    /// Memory complexity: O(rows * cols)
    pub fn applyGLU(linear_part: Matrix, gating_part: Matrix, allocator: std.mem.Allocator) !Matrix {
        if (linear_part.rows != gating_part.rows or linear_part.cols != gating_part.cols) {
            @panic("Matrix dimensions must match for GLU");
        }

        var sigmoid_gate = try apply(gating_part, sigmoid, allocator);
        defer sigmoid_gate.deinit();

        var result = try Matrix.init(allocator, linear_part.rows, linear_part.cols);

        for (0..linear_part.rows) |i| {
            for (0..linear_part.cols) |j| {
                const linear_value = linear_part.get(i, j);
                const gate_value = sigmoid_gate.get(i, j);
                result.set(i, j, linear_value * gate_value);
            }
        }

        return result;
    }

    /// Applies Swish Gated Linear Unit (SwiGLU) activation
    /// Mathematical form: SwiGLU(x,y) = x ⊗ swish(y)
    /// Where ⊗ is element-wise multiplication
    /// Properties:
    /// - Variant of GLU using Swish instead of sigmoid
    /// - Can provide better performance than GLU
    /// - Used in modern transformer architectures
    /// Time complexity: O(rows * cols)
    /// Memory complexity: O(rows * cols)
    pub fn applySwiGLU(linear_part: Matrix, gating_part: Matrix, allocator: std.mem.Allocator) !Matrix {
        if (linear_part.rows != gating_part.rows or linear_part.cols != gating_part.cols) {
            @panic("Matrix dimensions must match for SwiGLU");
        }

        var swish_gate = try apply(gating_part, swish, allocator);
        defer swish_gate.deinit();

        var result = try Matrix.init(allocator, linear_part.rows, linear_part.cols);

        for (0..linear_part.rows) |i| {
            for (0..linear_part.cols) |j| {
                const linear_value = linear_part.get(i, j);
                const gate_value = swish_gate.get(i, j);
                result.set(i, j, linear_value * gate_value);
            }
        }

        return result;
    }
};

// Tests
test "sigmoid function" {
    try testing.expectApproxEqAbs(@as(f64, 0.5), Activation.sigmoid(0.0), 0.0001);
    try testing.expectApproxEqAbs(@as(f64, 0.7310585786300049), Activation.sigmoid(1.0), 0.0001);
    try testing.expectApproxEqAbs(@as(f64, 0.2689414213699951), Activation.sigmoid(-1.0), 0.0001);
}

test "linear function" {
    try testing.expectEqual(@as(f64, 0.0), Activation.linear(0.0));
    try testing.expectEqual(@as(f64, 1.0), Activation.linear(1.0));
    try testing.expectEqual(@as(f64, -1.0), Activation.linear(-1.0));
    try testing.expectEqual(@as(f64, 1.0), Activation.linear_derivative(0.0));
    try testing.expectEqual(@as(f64, 1.0), Activation.linear_derivative(1.0));
    try testing.expectEqual(@as(f64, 1.0), Activation.linear_derivative(-1.0));
}

test "relu function" {
    try testing.expectEqual(@as(f64, 0.0), Activation.relu(-1.0));
    try testing.expectEqual(@as(f64, 0.0), Activation.relu(0.0));
    try testing.expectEqual(@as(f64, 1.0), Activation.relu(1.0));
}

test "tanh function" {
    try testing.expectApproxEqAbs(@as(f64, 0.0), Activation.tanh(0.0), 0.0001);
    try testing.expectApproxEqAbs(@as(f64, 0.7615941559557649), Activation.tanh(1.0), 0.0001);
    try testing.expectApproxEqAbs(@as(f64, -0.7615941559557649), Activation.tanh(-1.0), 0.0001);
}

test "swish function" {
    try testing.expectApproxEqAbs(@as(f64, 0.0), Activation.swish(0.0), 0.0001);
    try testing.expectApproxEqAbs(@as(f64, 0.7310585786300049), Activation.swish(1.0), 0.0001);
    try testing.expectApproxEqAbs(@as(f64, -0.2689414213699951), Activation.swish(-1.0), 0.0001);
}

test "activation function application to matrix" {
    const allocator = testing.allocator;

    var m = try Matrix.init(allocator, 2, 2);
    defer m.deinit();

    m.set(0, 0, -1.0);
    m.set(0, 1, 0.0);
    m.set(1, 0, 1.0);
    m.set(1, 1, 2.0);

    var result = try Activation.apply(m, Activation.relu, allocator);
    defer result.deinit();

    try testing.expectEqual(@as(f64, 0.0), result.get(0, 0));
    try testing.expectEqual(@as(f64, 0.0), result.get(0, 1));
    try testing.expectEqual(@as(f64, 1.0), result.get(1, 0));
    try testing.expectEqual(@as(f64, 2.0), result.get(1, 1));
}

test "GLU function" {
    const allocator = testing.allocator;

    // Create two test matrices
    var linear = try Matrix.init(allocator, 2, 2);
    defer linear.deinit();

    var gating = try Matrix.init(allocator, 2, 2);
    defer gating.deinit();

    // Set test values
    linear.set(0, 0, 1.0);
    linear.set(0, 1, 2.0);
    linear.set(1, 0, 3.0);
    linear.set(1, 1, 4.0);

    gating.set(0, 0, 0.0); // sigmoid(0.0) = 0.5
    gating.set(0, 1, 1.0); // sigmoid(1.0) ≈ 0.731
    gating.set(1, 0, -1.0); // sigmoid(-1.0) ≈ 0.269
    gating.set(1, 1, 2.0); // sigmoid(2.0) ≈ 0.881

    var result = try Activation.applyGLU(linear, gating, allocator);
    defer result.deinit();

    // Expected: linear * sigmoid(gating)
    try testing.expectApproxEqAbs(@as(f64, 0.5), result.get(0, 0), 0.0001);
    try testing.expectApproxEqAbs(@as(f64, 1.462), result.get(0, 1), 0.001);
    try testing.expectApproxEqAbs(@as(f64, 0.807), result.get(1, 0), 0.001);
    try testing.expectApproxEqAbs(@as(f64, 3.524), result.get(1, 1), 0.001);
}

test "SwiGLU function" {
    const allocator = testing.allocator;

    // Create two test matrices
    var linear = try Matrix.init(allocator, 2, 2);
    defer linear.deinit();

    var gating = try Matrix.init(allocator, 2, 2);
    defer gating.deinit();

    // Set test values
    linear.set(0, 0, 1.0);
    linear.set(0, 1, 2.0);
    linear.set(1, 0, 3.0);
    linear.set(1, 1, 4.0);

    gating.set(0, 0, 0.0); // swish(0.0) = 0.0
    gating.set(0, 1, 1.0); // swish(1.0) ≈ 0.731
    gating.set(1, 0, -1.0); // swish(-1.0) ≈ -0.269
    gating.set(1, 1, 2.0); // swish(2.0) ≈ 1.762

    var result = try Activation.applySwiGLU(linear, gating, allocator);
    defer result.deinit();

    // Expected: linear * swish(gating)
    try testing.expectApproxEqAbs(@as(f64, 0.0), result.get(0, 0), 0.0001);
    try testing.expectApproxEqAbs(@as(f64, 1.462), result.get(0, 1), 0.001);
    try testing.expectApproxEqAbs(@as(f64, -0.807), result.get(1, 0), 0.001);
    try testing.expectApproxEqAbs(@as(f64, 7.046), result.get(1, 1), 0.001);
}

test "softmax and applySoftmax" {
    const allocator = testing.allocator;

    // Create test matrix with multiple rows
    var m = try Matrix.init(allocator, 2, 3);
    defer m.deinit();

    // Set test values
    m.set(0, 0, 1.0);
    m.set(0, 1, 2.0);
    m.set(0, 2, 3.0);
    m.set(1, 0, 0.0);
    m.set(1, 1, 1.0);
    m.set(1, 2, 2.0);

    var result = try Activation.applySoftmax(m, allocator);
    defer result.deinit();

    // Test first row [1.0, 2.0, 3.0]
    try testing.expectApproxEqAbs(@as(f64, 0.0900305733), result.get(0, 0), 0.0001);
    try testing.expectApproxEqAbs(@as(f64, 0.2447284714), result.get(0, 1), 0.0001);
    try testing.expectApproxEqAbs(@as(f64, 0.6652409553), result.get(0, 2), 0.0001);

    // Test second row [0.0, 1.0, 2.0]
    try testing.expectApproxEqAbs(@as(f64, 0.0900305733), result.get(1, 0), 0.0001);
    try testing.expectApproxEqAbs(@as(f64, 0.2447284714), result.get(1, 1), 0.0001);
    try testing.expectApproxEqAbs(@as(f64, 0.6652409553), result.get(1, 2), 0.0001);

    // Verify rows sum to 1
    var sum1: f64 = 0.0;
    var sum2: f64 = 0.0;
    for (0..3) |j| {
        sum1 += result.get(0, j);
        sum2 += result.get(1, j);
    }
    try testing.expectApproxEqAbs(@as(f64, 1.0), sum1, 0.0001);
    try testing.expectApproxEqAbs(@as(f64, 1.0), sum2, 0.0001);
}
