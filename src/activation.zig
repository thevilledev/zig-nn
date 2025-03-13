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