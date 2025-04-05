const std = @import("std");
const Allocator = std.mem.Allocator;
const backend = @import("backend.zig");
const ComputeBackend = backend.ComputeBackend;
const BackendType = backend.BackendType;
const Matrix = backend.Matrix;

/// CPU-specific implementation of a matrix
const CPUMatrix = struct {
    data: []f64,
    allocator: Allocator,
};

/// CPU backend implementation
pub const CPUBackend = struct {
    allocator: Allocator,
    backend_interface: ComputeBackend,

    /// Initialize a new CPU backend
    pub fn init(allocator: Allocator) !*CPUBackend {
        const cpu_backend = try allocator.create(CPUBackend);
        cpu_backend.* = CPUBackend{
            .allocator = allocator,
            .backend_interface = .{
                .ptr = cpu_backend,
                .initMatrixFn = initMatrix,
                .deinitMatrixFn = deinitMatrix,
                .copyMatrixFn = copyMatrix,
                .getMatrixElementFn = getMatrixElement,
                .setMatrixElementFn = setMatrixElement,
                .fillMatrixFn = fillMatrix,
                .dotProductFn = dotProduct,
                .addFn = add,
                .subtractFn = subtract,
                .elementWiseMultiplyFn = elementWiseMultiply,
                .scaleFn = scale,
                .sumRowsFn = sumRows,
                .transposeFn = transpose,
                .extractBatchFn = extractBatch,
                .randomizeFn = randomize,
                .applyActivationFn = applyActivation,
                .applySoftmaxFn = applySoftmax,
                .applyGLUFn = applyGLU,
                .applySwiGLUFn = applySwiGLU,
                .getBackendTypeFn = getBackendType,
            },
        };
        return cpu_backend;
    }

    /// Free all resources used by the CPU backend
    pub fn deinit(self: *CPUBackend) void {
        self.allocator.destroy(self);
    }

    /// Get the backend interface
    pub fn getInterface(self: *CPUBackend) *const ComputeBackend {
        return &self.backend_interface;
    }

    // ComputeBackend interface implementations
    fn initMatrix(_: *anyopaque, allocator: Allocator, rows: usize, cols: usize) error{OutOfMemory}!*Matrix {
        // Allocate the matrix wrapper
        const matrix = try allocator.create(Matrix);
        errdefer allocator.destroy(matrix);

        // Allocate the CPU-specific data
        const cpu_data = try allocator.create(CPUMatrix);
        errdefer allocator.destroy(cpu_data);

        // Allocate the data array
        const data = try allocator.alloc(f64, rows * cols);
        errdefer allocator.free(data);

        // Initialize with zeros
        @memset(data, 0);

        // Set up the CPU matrix data
        cpu_data.* = .{
            .data = data,
            .allocator = allocator,
        };

        // Set up the matrix
        matrix.* = .{
            .rows = rows,
            .cols = cols,
            .backend = undefined, // Will be set by the caller
            .impl_data = cpu_data,
        };

        return matrix;
    }

    fn deinitMatrix(_: *anyopaque, matrix: *Matrix) void {
        // First, cast to get the CPU-specific data
        const cpu_data = @as(*CPUMatrix, @ptrCast(@alignCast(matrix.impl_data)));
        const allocator = cpu_data.allocator;

        // Free the data array
        allocator.free(cpu_data.data);

        // Free the CPU matrix data
        allocator.destroy(cpu_data);

        // Free the matrix wrapper
        allocator.destroy(matrix);
    }

    fn getMatrixElement(_: *anyopaque, matrix: *const Matrix, row: usize, col: usize) f64 {
        if (row >= matrix.rows or col >= matrix.cols) {
            @panic("Index out of bounds");
        }

        const cpu_data = @as(*const CPUMatrix, @ptrCast(@alignCast(matrix.impl_data)));
        return cpu_data.data[row * matrix.cols + col];
    }

    fn setMatrixElement(_: *anyopaque, matrix: *Matrix, row: usize, col: usize, value: f64) void {
        if (row >= matrix.rows or col >= matrix.cols) {
            @panic("Index out of bounds");
        }

        const cpu_data = @as(*CPUMatrix, @ptrCast(@alignCast(matrix.impl_data)));
        cpu_data.data[row * matrix.cols + col] = value;
    }

    fn fillMatrix(_: *anyopaque, matrix: *Matrix, value: f64) void {
        const cpu_data = @as(*CPUMatrix, @ptrCast(@alignCast(matrix.impl_data)));
        for (cpu_data.data) |*element| {
            element.* = value;
        }
    }

    fn copyMatrix(ptr: *anyopaque, source: *const Matrix, allocator: Allocator) error{OutOfMemory}!*Matrix {
        const self = @as(*CPUBackend, @ptrCast(@alignCast(ptr)));

        const result = try initMatrix(undefined, allocator, source.rows, source.cols);
        errdefer deinitMatrix(undefined, result);

        // Set the backend reference
        result.backend = &self.backend_interface;

        const source_cpu_data = @as(*const CPUMatrix, @ptrCast(@alignCast(source.impl_data)));
        const result_cpu_data = @as(*CPUMatrix, @ptrCast(@alignCast(result.impl_data)));

        // Copy all data
        @memcpy(result_cpu_data.data, source_cpu_data.data);

        return result;
    }

    fn dotProduct(ptr: *anyopaque, a: *const Matrix, b: *const Matrix, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!*Matrix {
        if (a.cols != b.rows) {
            return error.DimensionMismatch;
        }

        const self = @as(*CPUBackend, @ptrCast(@alignCast(ptr)));

        const result = try initMatrix(undefined, allocator, a.rows, b.cols);
        errdefer deinitMatrix(undefined, result);

        // Set the backend reference
        result.backend = &self.backend_interface;

        const a_cpu = @as(*const CPUMatrix, @ptrCast(@alignCast(a.impl_data)));
        const b_cpu = @as(*const CPUMatrix, @ptrCast(@alignCast(b.impl_data)));

        for (0..a.rows) |i| {
            for (0..b.cols) |j| {
                var sum: f64 = 0;
                for (0..a.cols) |k| {
                    sum += a_cpu.data[i * a.cols + k] * b_cpu.data[k * b.cols + j];
                }
                setMatrixElement(undefined, result, i, j, sum);
            }
        }

        return result;
    }

    fn add(ptr: *anyopaque, a: *const Matrix, b: *const Matrix, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!*Matrix {
        if (a.rows != b.rows or a.cols != b.cols) {
            return error.DimensionMismatch;
        }

        const self = @as(*CPUBackend, @ptrCast(@alignCast(ptr)));

        const result = try initMatrix(undefined, allocator, a.rows, a.cols);
        errdefer deinitMatrix(undefined, result);

        // Set the backend reference
        result.backend = &self.backend_interface;

        const a_cpu = @as(*const CPUMatrix, @ptrCast(@alignCast(a.impl_data)));
        const b_cpu = @as(*const CPUMatrix, @ptrCast(@alignCast(b.impl_data)));
        const result_cpu = @as(*CPUMatrix, @ptrCast(@alignCast(result.impl_data)));

        for (0..a.rows * a.cols) |i| {
            result_cpu.data[i] = a_cpu.data[i] + b_cpu.data[i];
        }

        return result;
    }

    fn subtract(ptr: *anyopaque, a: *const Matrix, b: *const Matrix, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!*Matrix {
        if (a.rows != b.rows or a.cols != b.cols) {
            return error.DimensionMismatch;
        }

        const self = @as(*CPUBackend, @ptrCast(@alignCast(ptr)));

        const result = try initMatrix(undefined, allocator, a.rows, a.cols);
        errdefer deinitMatrix(undefined, result);

        // Set the backend reference
        result.backend = &self.backend_interface;

        const a_cpu = @as(*const CPUMatrix, @ptrCast(@alignCast(a.impl_data)));
        const b_cpu = @as(*const CPUMatrix, @ptrCast(@alignCast(b.impl_data)));
        const result_cpu = @as(*CPUMatrix, @ptrCast(@alignCast(result.impl_data)));

        for (0..a.rows * a.cols) |i| {
            result_cpu.data[i] = a_cpu.data[i] - b_cpu.data[i];
        }

        return result;
    }

    fn elementWiseMultiply(ptr: *anyopaque, a: *const Matrix, b: *const Matrix, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!*Matrix {
        if (a.rows != b.rows or a.cols != b.cols) {
            return error.DimensionMismatch;
        }

        const self = @as(*CPUBackend, @ptrCast(@alignCast(ptr)));

        const result = try initMatrix(undefined, allocator, a.rows, a.cols);
        errdefer deinitMatrix(undefined, result);

        // Set the backend reference
        result.backend = &self.backend_interface;

        const a_cpu = @as(*const CPUMatrix, @ptrCast(@alignCast(a.impl_data)));
        const b_cpu = @as(*const CPUMatrix, @ptrCast(@alignCast(b.impl_data)));
        const result_cpu = @as(*CPUMatrix, @ptrCast(@alignCast(result.impl_data)));

        for (0..a.rows * a.cols) |i| {
            result_cpu.data[i] = a_cpu.data[i] * b_cpu.data[i];
        }

        return result;
    }

    fn scale(ptr: *anyopaque, matrix: *const Matrix, scalar: f64, allocator: Allocator) error{OutOfMemory}!*Matrix {
        const self = @as(*CPUBackend, @ptrCast(@alignCast(ptr)));

        const result = try initMatrix(undefined, allocator, matrix.rows, matrix.cols);
        errdefer deinitMatrix(undefined, result);

        // Set the backend reference
        result.backend = &self.backend_interface;

        const matrix_cpu = @as(*const CPUMatrix, @ptrCast(@alignCast(matrix.impl_data)));
        const result_cpu = @as(*CPUMatrix, @ptrCast(@alignCast(result.impl_data)));

        for (0..matrix.rows * matrix.cols) |i| {
            result_cpu.data[i] = matrix_cpu.data[i] * scalar;
        }

        return result;
    }

    fn sumRows(ptr: *anyopaque, matrix: *const Matrix, allocator: Allocator) error{OutOfMemory}!*Matrix {
        const self = @as(*CPUBackend, @ptrCast(@alignCast(ptr)));

        const result = try initMatrix(undefined, allocator, 1, matrix.cols);
        errdefer deinitMatrix(undefined, result);

        // Set the backend reference
        result.backend = &self.backend_interface;

        const matrix_cpu = @as(*const CPUMatrix, @ptrCast(@alignCast(matrix.impl_data)));

        for (0..matrix.cols) |j| {
            var sum: f64 = 0;
            for (0..matrix.rows) |i| {
                sum += matrix_cpu.data[i * matrix.cols + j];
            }
            setMatrixElement(undefined, result, 0, j, sum);
        }

        return result;
    }

    fn transpose(ptr: *anyopaque, matrix: *const Matrix, allocator: Allocator) error{OutOfMemory}!*Matrix {
        const self = @as(*CPUBackend, @ptrCast(@alignCast(ptr)));

        const result = try initMatrix(undefined, allocator, matrix.cols, matrix.rows);
        errdefer deinitMatrix(undefined, result);

        // Set the backend reference
        result.backend = &self.backend_interface;

        for (0..matrix.rows) |i| {
            for (0..matrix.cols) |j| {
                const value = getMatrixElement(undefined, matrix, i, j);
                setMatrixElement(undefined, result, j, i, value);
            }
        }

        return result;
    }

    fn extractBatch(ptr: *anyopaque, matrix: *const Matrix, start: usize, end: usize, allocator: Allocator) error{ OutOfMemory, InvalidBatchIndices }!*Matrix {
        if (start >= matrix.rows or end > matrix.rows or start >= end) {
            return error.InvalidBatchIndices;
        }

        const self = @as(*CPUBackend, @ptrCast(@alignCast(ptr)));

        const batch_size = end - start;
        const result = try initMatrix(undefined, allocator, batch_size, matrix.cols);
        errdefer deinitMatrix(undefined, result);

        // Set the backend reference
        result.backend = &self.backend_interface;

        const matrix_cpu = @as(*const CPUMatrix, @ptrCast(@alignCast(matrix.impl_data)));
        const result_cpu = @as(*CPUMatrix, @ptrCast(@alignCast(result.impl_data)));

        for (0..batch_size) |i| {
            const source_offset = (start + i) * matrix.cols;
            const dest_offset = i * matrix.cols;
            @memcpy(result_cpu.data[dest_offset..(dest_offset + matrix.cols)], matrix_cpu.data[source_offset..(source_offset + matrix.cols)]);
        }

        return result;
    }

    fn randomize(_: *anyopaque, matrix: *Matrix, min: f64, max: f64) void {
        var prng = std.Random.DefaultPrng.init(@intCast(std.time.milliTimestamp()));
        const rand = prng.random();

        const matrix_cpu = @as(*CPUMatrix, @ptrCast(@alignCast(matrix.impl_data)));

        for (0..matrix.rows * matrix.cols) |i| {
            const rand_float = rand.float(f64);
            matrix_cpu.data[i] = min + (rand_float * (max - min));
        }
    }

    fn applyActivation(ptr: *anyopaque, matrix: *const Matrix, activation: fn (f64) f64, allocator: Allocator) error{OutOfMemory}!*Matrix {
        const self = @as(*CPUBackend, @ptrCast(@alignCast(ptr)));

        const result = try initMatrix(undefined, allocator, matrix.rows, matrix.cols);
        errdefer deinitMatrix(undefined, result);

        // Set the backend reference
        result.backend = &self.backend_interface;

        const matrix_cpu = @as(*const CPUMatrix, @ptrCast(@alignCast(matrix.impl_data)));
        const result_cpu = @as(*CPUMatrix, @ptrCast(@alignCast(result.impl_data)));

        for (0..matrix.rows * matrix.cols) |i| {
            result_cpu.data[i] = activation(matrix_cpu.data[i]);
        }

        return result;
    }

    fn applySoftmax(ptr: *anyopaque, matrix: *const Matrix, allocator: Allocator) error{OutOfMemory}!*Matrix {
        const self = @as(*CPUBackend, @ptrCast(@alignCast(ptr)));

        const result = try initMatrix(undefined, allocator, matrix.rows, matrix.cols);
        errdefer deinitMatrix(undefined, result);

        // Set the backend reference
        result.backend = &self.backend_interface;

        // Process each row independently
        for (0..matrix.rows) |i| {
            // Find max value in row for numerical stability
            var max_val: f64 = -std.math.inf(f64);
            for (0..matrix.cols) |j| {
                max_val = @max(max_val, getMatrixElement(undefined, matrix, i, j));
            }

            // Compute exp(x - max) and sum
            var sum: f64 = 0.0;
            for (0..matrix.cols) |j| {
                const shifted = getMatrixElement(undefined, matrix, i, j) - max_val;
                const exp_val = std.math.exp(shifted);
                setMatrixElement(undefined, result, i, j, exp_val);
                sum += exp_val;
            }

            // Normalize by sum
            for (0..matrix.cols) |j| {
                const val = getMatrixElement(undefined, result, i, j) / sum;
                setMatrixElement(undefined, result, i, j, val);
            }
        }

        return result;
    }

    fn applyGLU(ptr: *anyopaque, linear_part: *const Matrix, gating_part: *const Matrix, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!*Matrix {
        if (linear_part.rows != gating_part.rows or linear_part.cols != gating_part.cols) {
            return error.DimensionMismatch;
        }

        // Apply sigmoid to the gating part
        const sigmoid_gate = try applyActivation(ptr, gating_part, sigmoid, allocator);
        defer deinitMatrix(undefined, sigmoid_gate);

        // Element-wise multiply with the linear part
        return elementWiseMultiply(ptr, linear_part, sigmoid_gate, allocator);
    }

    fn applySwiGLU(ptr: *anyopaque, linear_part: *const Matrix, gating_part: *const Matrix, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!*Matrix {
        if (linear_part.rows != gating_part.rows or linear_part.cols != gating_part.cols) {
            return error.DimensionMismatch;
        }

        // Apply swish to the gating part
        const swish_gate = try applyActivation(ptr, gating_part, swish, allocator);
        defer deinitMatrix(undefined, swish_gate);

        // Element-wise multiply with the linear part
        return elementWiseMultiply(ptr, linear_part, swish_gate, allocator);
    }

    fn getBackendType(_: *anyopaque) BackendType {
        return .CPU;
    }
};

// Helper activation functions needed for GLU and SwiGLU
fn sigmoid(x: f64) f64 {
    return 1.0 / (1.0 + std.math.exp(-x));
}

fn swish(x: f64) f64 {
    const beta: f64 = 1.0;
    return x * sigmoid(beta * x);
}

/// Creates a CPU backend instance
pub fn createCPUBackend(allocator: Allocator) !*ComputeBackend {
    var cpu_backend = try CPUBackend.init(allocator);
    return &cpu_backend.backend_interface;
}
