const std = @import("std");
const math = std.math;
const testing = std.testing;
const Matrix = @import("matrix.zig").Matrix;

/// Activation functions and their derivatives
pub const Activation = struct {
    /// Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
    pub fn sigmoid(x: f64) f64 {
        return 1.0 / (1.0 + math.exp(-x));
    }

    /// Derivative of sigmoid function: f'(x) = f(x) * (1 - f(x))
    pub fn sigmoid_derivative(x: f64) f64 {
        const sig = sigmoid(x);
        return sig * (1.0 - sig);
    }

    /// ReLU activation function: f(x) = max(0, x)
    pub fn relu(x: f64) f64 {
        return if (x > 0) x else 0;
    }

    /// Derivative of ReLU function: f'(x) = 1 if x > 0, else 0
    pub fn relu_derivative(x: f64) f64 {
        return if (x > 0) 1 else 0;
    }

    /// Tanh activation function: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    pub fn tanh(x: f64) f64 {
        return math.tanh(x);
    }

    /// Derivative of tanh function: f'(x) = 1 - tanh^2(x)
    pub fn tanh_derivative(x: f64) f64 {
        const t = tanh(x);
        return 1.0 - (t * t);
    }

    /// Swish activation function: f(x) = x * sigmoid(beta * x)
    /// Beta is typically set to 1.0
    pub fn swish(x: f64) f64 {
        const beta: f64 = 1.0;
        return x * sigmoid(beta * x);
    }

    /// Derivative of swish function: f'(x) = sigmoid(beta * x) + beta * x * sigmoid(beta * x) * (1 - sigmoid(beta * x))
    pub fn swish_derivative(x: f64) f64 {
        const beta: f64 = 1.0;
        const sig = sigmoid(beta * x);
        return sig + beta * x * sig * (1.0 - sig);
    }

    /// Apply activation function to matrix
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

    /// Apply GLU (Gated Linear Unit) to two matrices
    /// GLU(x, y) = x ⊗ sigmoid(y) where ⊗ is element-wise multiplication
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

    /// Apply SwiGLU (Swish Gated Linear Unit) to two matrices
    /// SwiGLU(x, y) = x ⊗ swish(y) where ⊗ is element-wise multiplication
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

    gating.set(0, 0, 0.0);  // sigmoid(0.0) = 0.5
    gating.set(0, 1, 1.0);  // sigmoid(1.0) ≈ 0.731
    gating.set(1, 0, -1.0); // sigmoid(-1.0) ≈ 0.269
    gating.set(1, 1, 2.0);  // sigmoid(2.0) ≈ 0.881

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

    gating.set(0, 0, 0.0);  // swish(0.0) = 0.0
    gating.set(0, 1, 1.0);  // swish(1.0) ≈ 0.731
    gating.set(1, 0, -1.0); // swish(-1.0) ≈ -0.269
    gating.set(1, 1, 2.0);  // swish(2.0) ≈ 1.762

    var result = try Activation.applySwiGLU(linear, gating, allocator);
    defer result.deinit();

    // Expected: linear * swish(gating)
    try testing.expectApproxEqAbs(@as(f64, 0.0), result.get(0, 0), 0.0001);
    try testing.expectApproxEqAbs(@as(f64, 1.462), result.get(0, 1), 0.001);
    try testing.expectApproxEqAbs(@as(f64, -0.807), result.get(1, 0), 0.001);
    try testing.expectApproxEqAbs(@as(f64, 7.046), result.get(1, 1), 0.001);
} 