const std = @import("std");
const Matrix = @import("matrix.zig").Matrix;
const testing = std.testing;

/// Row-wise layer normalization with learnable affine parameters.
///
/// For each input row, this layer normalizes across columns and then applies:
/// output = ((input - mean) / sqrt(variance + eps)) * weight + bias
pub const LayerNorm = struct {
    weight: Matrix,
    bias: Matrix,
    eps: f64,

    pub fn init(allocator: std.mem.Allocator, size: usize) !LayerNorm {
        return initWithEps(allocator, size, 1e-5);
    }

    pub fn initWithEps(allocator: std.mem.Allocator, size: usize, eps: f64) !LayerNorm {
        if (size == 0) {
            return error.InvalidLayerNormSize;
        }

        var weight = try Matrix.init(allocator, 1, size);
        errdefer weight.deinit();
        var bias = try Matrix.init(allocator, 1, size);
        errdefer bias.deinit();

        weight.fill(1.0);
        bias.fill(0.0);

        return .{ .weight = weight, .bias = bias, .eps = eps };
    }

    pub fn deinit(self: *LayerNorm) void {
        self.weight.deinit();
        self.bias.deinit();
    }

    pub fn forward(self: *const LayerNorm, input: Matrix, allocator: std.mem.Allocator) !Matrix {
        if (input.cols != self.weight.cols or
            self.weight.rows != 1 or
            self.bias.rows != 1 or
            self.bias.cols != self.weight.cols)
        {
            return error.DimensionMismatch;
        }

        var output = try Matrix.init(allocator, input.rows, input.cols);
        errdefer output.deinit();

        const width = @as(f64, @floatFromInt(input.cols));
        for (0..input.rows) |row| {
            var mean: f64 = 0.0;
            for (0..input.cols) |col| {
                mean += try input.get(row, col);
            }
            mean /= width;

            var variance: f64 = 0.0;
            for (0..input.cols) |col| {
                const centered = (try input.get(row, col)) - mean;
                variance += centered * centered;
            }
            variance /= width;

            const inv_std = 1.0 / @sqrt(variance + self.eps);
            for (0..input.cols) |col| {
                const normalized = ((try input.get(row, col)) - mean) * inv_std;
                const scaled = normalized * (try self.weight.get(0, col)) + (try self.bias.get(0, col));
                try output.set(row, col, scaled);
            }
        }

        return output;
    }

    pub fn parameterCount(self: *const LayerNorm) usize {
        return self.weight.cols + self.bias.cols;
    }

    pub fn getInputSize(self: *const LayerNorm) usize {
        return self.weight.cols;
    }

    pub fn getOutputSize(self: *const LayerNorm) usize {
        return self.weight.cols;
    }
};

test "layer norm initializes identity affine parameters" {
    const allocator = testing.allocator;

    var layer_norm = try LayerNorm.init(allocator, 4);
    defer layer_norm.deinit();

    try testing.expectEqual(@as(usize, 4), layer_norm.getInputSize());
    try testing.expectEqual(@as(usize, 4), layer_norm.getOutputSize());
    try testing.expectEqual(@as(usize, 8), layer_norm.parameterCount());

    for (0..4) |col| {
        try testing.expectEqual(@as(f64, 1.0), try layer_norm.weight.get(0, col));
        try testing.expectEqual(@as(f64, 0.0), try layer_norm.bias.get(0, col));
    }
}

test "layer norm rejects zero feature width" {
    const allocator = testing.allocator;

    try testing.expectError(error.InvalidLayerNormSize, LayerNorm.init(allocator, 0));
}

test "layer norm normalizes each row independently" {
    const allocator = testing.allocator;

    var layer_norm = try LayerNorm.initWithEps(allocator, 3, 0.0);
    defer layer_norm.deinit();

    var input = try Matrix.init(allocator, 2, 3);
    defer input.deinit();
    try input.set(0, 0, 1.0);
    try input.set(0, 1, 2.0);
    try input.set(0, 2, 3.0);
    try input.set(1, 0, 2.0);
    try input.set(1, 1, 4.0);
    try input.set(1, 2, 6.0);

    var output = try layer_norm.forward(input, allocator);
    defer output.deinit();

    const expected = @sqrt(1.5);
    try testing.expectApproxEqAbs(-expected, try output.get(0, 0), 1e-12);
    try testing.expectApproxEqAbs(0.0, try output.get(0, 1), 1e-12);
    try testing.expectApproxEqAbs(expected, try output.get(0, 2), 1e-12);
    try testing.expectApproxEqAbs(-expected, try output.get(1, 0), 1e-12);
    try testing.expectApproxEqAbs(0.0, try output.get(1, 1), 1e-12);
    try testing.expectApproxEqAbs(expected, try output.get(1, 2), 1e-12);
}

test "layer norm applies learned affine parameters" {
    const allocator = testing.allocator;

    var layer_norm = try LayerNorm.initWithEps(allocator, 2, 0.0);
    defer layer_norm.deinit();
    try layer_norm.weight.set(0, 0, 2.0);
    try layer_norm.weight.set(0, 1, 0.5);
    try layer_norm.bias.set(0, 0, 10.0);
    try layer_norm.bias.set(0, 1, -10.0);

    var input = try Matrix.init(allocator, 1, 2);
    defer input.deinit();
    try input.set(0, 0, 2.0);
    try input.set(0, 1, 4.0);

    var output = try layer_norm.forward(input, allocator);
    defer output.deinit();

    try testing.expectApproxEqAbs(8.0, try output.get(0, 0), 1e-12);
    try testing.expectApproxEqAbs(-9.5, try output.get(0, 1), 1e-12);
}

test "layer norm rejects mismatched feature width" {
    const allocator = testing.allocator;

    var layer_norm = try LayerNorm.init(allocator, 3);
    defer layer_norm.deinit();

    var input = try Matrix.init(allocator, 1, 2);
    defer input.deinit();

    try testing.expectError(error.DimensionMismatch, layer_norm.forward(input, allocator));
}
