const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;

// Build options with safe defaults in case they're not provided
const build_options = struct {
    pub const enable_gpu = false;
    pub const enable_metal = false;
    pub const enable_cuda = false;
};

/// BackendType enum represents the available computation backends
pub const BackendType = enum {
    CPU,
    Metal, // macOS only
    CUDA, // NVIDIA GPUs
};

/// ComputeBackend is an interface for different computation backends (CPU, GPU via Metal, GPU via CUDA)
/// It abstracts away the details of how matrix operations and activation functions are implemented
pub const ComputeBackend = struct {
    // Pointer to the implementation of this backend
    ptr: *anyopaque,

    // Interface functions - these must be implemented by each backend

    // Memory management
    initMatrixFn: *const fn (ptr: *anyopaque, allocator: Allocator, rows: usize, cols: usize) error{OutOfMemory}!*Matrix,
    deinitMatrixFn: *const fn (ptr: *anyopaque, matrix: *Matrix) void,

    // Data transfer
    copyMatrixFn: *const fn (ptr: *anyopaque, source: *const Matrix, allocator: Allocator) error{OutOfMemory}!*Matrix,
    getMatrixElementFn: *const fn (ptr: *anyopaque, matrix: *const Matrix, row: usize, col: usize) f64,
    setMatrixElementFn: *const fn (ptr: *anyopaque, matrix: *Matrix, row: usize, col: usize, value: f64) void,
    fillMatrixFn: *const fn (ptr: *anyopaque, matrix: *Matrix, value: f64) void,

    // Basic matrix operations
    dotProductFn: *const fn (ptr: *anyopaque, a: *const Matrix, b: *const Matrix, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!*Matrix,

    addFn: *const fn (ptr: *anyopaque, a: *const Matrix, b: *const Matrix, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!*Matrix,

    subtractFn: *const fn (ptr: *anyopaque, a: *const Matrix, b: *const Matrix, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!*Matrix,

    elementWiseMultiplyFn: *const fn (ptr: *anyopaque, a: *const Matrix, b: *const Matrix, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!*Matrix,

    scaleFn: *const fn (ptr: *anyopaque, matrix: *const Matrix, scalar: f64, allocator: Allocator) error{OutOfMemory}!*Matrix,

    sumRowsFn: *const fn (ptr: *anyopaque, matrix: *const Matrix, allocator: Allocator) error{OutOfMemory}!*Matrix,

    transposeFn: *const fn (ptr: *anyopaque, matrix: *const Matrix, allocator: Allocator) error{OutOfMemory}!*Matrix,

    extractBatchFn: *const fn (ptr: *anyopaque, matrix: *const Matrix, start: usize, end: usize, allocator: Allocator) error{ OutOfMemory, InvalidBatchIndices }!*Matrix,

    randomizeFn: *const fn (ptr: *anyopaque, matrix: *Matrix, min: f64, max: f64) void,

    // Activation functions
    applyActivationFn: *const fn (ptr: *anyopaque, matrix: *const Matrix, activation: fn (f64) f64, allocator: Allocator) error{OutOfMemory}!*Matrix,

    applySoftmaxFn: *const fn (ptr: *anyopaque, matrix: *const Matrix, allocator: Allocator) error{OutOfMemory}!*Matrix,

    applyGLUFn: *const fn (ptr: *anyopaque, linear_part: *const Matrix, gating_part: *const Matrix, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!*Matrix,

    applySwiGLUFn: *const fn (ptr: *anyopaque, linear_part: *const Matrix, gating_part: *const Matrix, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!*Matrix,

    // Backend type information
    getBackendTypeFn: *const fn (ptr: *anyopaque) BackendType,

    // Helper wrapper functions that call the implementation
    pub fn initMatrix(self: *const ComputeBackend, allocator: Allocator, rows: usize, cols: usize) error{OutOfMemory}!*Matrix {
        return self.initMatrixFn(self.ptr, allocator, rows, cols);
    }

    pub fn deinitMatrix(self: *const ComputeBackend, matrix: *Matrix) void {
        self.deinitMatrixFn(self.ptr, matrix);
    }

    pub fn copyMatrix(self: *const ComputeBackend, source: *const Matrix, allocator: Allocator) error{OutOfMemory}!*Matrix {
        return self.copyMatrixFn(self.ptr, source, allocator);
    }

    pub fn getMatrixElement(self: *const ComputeBackend, matrix: *const Matrix, row: usize, col: usize) f64 {
        return self.getMatrixElementFn(self.ptr, matrix, row, col);
    }

    pub fn setMatrixElement(self: *const ComputeBackend, matrix: *Matrix, row: usize, col: usize, value: f64) void {
        self.setMatrixElementFn(self.ptr, matrix, row, col, value);
    }

    pub fn fillMatrix(self: *const ComputeBackend, matrix: *Matrix, value: f64) void {
        self.fillMatrixFn(self.ptr, matrix, value);
    }

    pub fn dotProduct(self: *const ComputeBackend, a: *const Matrix, b: *const Matrix, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!*Matrix {
        return self.dotProductFn(self.ptr, a, b, allocator);
    }

    pub fn add(self: *const ComputeBackend, a: *const Matrix, b: *const Matrix, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!*Matrix {
        return self.addFn(self.ptr, a, b, allocator);
    }

    pub fn subtract(self: *const ComputeBackend, a: *const Matrix, b: *const Matrix, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!*Matrix {
        return self.subtractFn(self.ptr, a, b, allocator);
    }

    pub fn elementWiseMultiply(self: *const ComputeBackend, a: *const Matrix, b: *const Matrix, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!*Matrix {
        return self.elementWiseMultiplyFn(self.ptr, a, b, allocator);
    }

    pub fn scale(self: *const ComputeBackend, matrix: *const Matrix, scalar: f64, allocator: Allocator) error{OutOfMemory}!*Matrix {
        return self.scaleFn(self.ptr, matrix, scalar, allocator);
    }

    pub fn sumRows(self: *const ComputeBackend, matrix: *const Matrix, allocator: Allocator) error{OutOfMemory}!*Matrix {
        return self.sumRowsFn(self.ptr, matrix, allocator);
    }

    pub fn transpose(self: *const ComputeBackend, matrix: *const Matrix, allocator: Allocator) error{OutOfMemory}!*Matrix {
        return self.transposeFn(self.ptr, matrix, allocator);
    }

    pub fn extractBatch(self: *const ComputeBackend, matrix: *const Matrix, start: usize, end: usize, allocator: Allocator) error{ OutOfMemory, InvalidBatchIndices }!*Matrix {
        return self.extractBatchFn(self.ptr, matrix, start, end, allocator);
    }

    pub fn randomize(self: *const ComputeBackend, matrix: *Matrix, min: f64, max: f64) void {
        self.randomizeFn(self.ptr, matrix, min, max);
    }

    pub fn applyActivation(self: *const ComputeBackend, matrix: *const Matrix, activation: fn (f64) f64, allocator: Allocator) error{OutOfMemory}!*Matrix {
        return self.applyActivationFn(self.ptr, matrix, activation, allocator);
    }

    pub fn applySoftmax(self: *const ComputeBackend, matrix: *const Matrix, allocator: Allocator) error{OutOfMemory}!*Matrix {
        return self.applySoftmaxFn(self.ptr, matrix, allocator);
    }

    pub fn applyGLU(self: *const ComputeBackend, linear_part: *const Matrix, gating_part: *const Matrix, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!*Matrix {
        return self.applyGLUFn(self.ptr, linear_part, gating_part, allocator);
    }

    pub fn applySwiGLU(self: *const ComputeBackend, linear_part: *const Matrix, gating_part: *const Matrix, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!*Matrix {
        return self.applySwiGLUFn(self.ptr, linear_part, gating_part, allocator);
    }

    pub fn getBackendType(self: *const ComputeBackend) BackendType {
        return self.getBackendTypeFn(self.ptr);
    }
};

/// Matrix represents a 2D array of f64 values, now abstracted to support different backends
/// This is a wrapper that contains backend-specific implementation details
pub const Matrix = struct {
    rows: usize,
    cols: usize,
    backend: *const ComputeBackend,
    // Implementation-specific data
    impl_data: *anyopaque,

    /// Creates a new matrix using the specified backend
    pub fn init(backend: *const ComputeBackend, allocator: Allocator, rows: usize, cols: usize) !*Matrix {
        return backend.initMatrix(allocator, rows, cols);
    }

    /// Frees the matrix memory
    pub fn deinit(self: *Matrix) void {
        self.backend.deinitMatrix(self);
    }

    /// Gets value at specified position (i,j)
    pub fn get(self: *const Matrix, row: usize, col: usize) f64 {
        return self.backend.getMatrixElement(self, row, col);
    }

    /// Sets value at specified position (i,j)
    pub fn set(self: *Matrix, row: usize, col: usize, value: f64) void {
        self.backend.setMatrixElement(self, row, col, value);
    }

    /// Fills matrix with a single value
    pub fn fill(self: *Matrix, value: f64) void {
        self.backend.fillMatrix(self, value);
    }

    /// Creates a deep copy of an existing matrix
    pub fn copy(self: *const Matrix, allocator: Allocator) !*Matrix {
        return self.backend.copyMatrix(self, allocator);
    }

    /// Fills matrix with random values in range [min, max]
    pub fn randomize(self: *Matrix, min: f64, max: f64) void {
        self.backend.randomize(self, min, max);
    }

    /// Performs matrix multiplication (dot product): C = A × B
    pub fn dotProduct(self: *const Matrix, other: *const Matrix, allocator: Allocator) !*Matrix {
        return self.backend.dotProduct(self, other, allocator);
    }

    /// Performs element-wise matrix addition: C = A + B
    pub fn add(self: *const Matrix, other: *const Matrix, allocator: Allocator) !*Matrix {
        return self.backend.add(self, other, allocator);
    }

    /// Performs element-wise matrix subtraction: C = A - B
    pub fn subtract(self: *const Matrix, other: *const Matrix, allocator: Allocator) !*Matrix {
        return self.backend.subtract(self, other, allocator);
    }

    /// Performs Hadamard (element-wise) multiplication: C = A ⊙ B
    pub fn elementWiseMultiply(self: *const Matrix, other: *const Matrix, allocator: Allocator) !*Matrix {
        return self.backend.elementWiseMultiply(self, other, allocator);
    }

    /// Scales matrix by a scalar value: B = αA
    pub fn scale(self: *const Matrix, scalar: f64, allocator: Allocator) !*Matrix {
        return self.backend.scale(self, scalar, allocator);
    }

    /// Computes column-wise sum of matrix elements
    pub fn sumRows(self: *const Matrix, allocator: Allocator) !*Matrix {
        return self.backend.sumRows(self, allocator);
    }

    /// Computes matrix transpose: B = Aᵀ
    pub fn transpose(self: *const Matrix, allocator: Allocator) !*Matrix {
        return self.backend.transpose(self, allocator);
    }

    /// Extracts a batch of rows from the matrix
    pub fn extractBatch(self: *const Matrix, start: usize, end: usize, allocator: Allocator) !*Matrix {
        return self.backend.extractBatch(self, start, end, allocator);
    }
};

/// ActivationUtils provides utility functions for working with activation functions across backends
pub const ActivationUtils = struct {
    /// Apply an activation function to each element of a matrix
    pub fn apply(backend: *const ComputeBackend, matrix: *const Matrix, activation_fn: fn (f64) f64, allocator: Allocator) !*Matrix {
        return backend.applyActivation(matrix, activation_fn, allocator);
    }

    /// Apply softmax activation to a matrix row-wise
    pub fn applySoftmax(backend: *const ComputeBackend, matrix: *const Matrix, allocator: Allocator) !*Matrix {
        return backend.applySoftmax(matrix, allocator);
    }

    /// Apply Gated Linear Unit (GLU) activation
    pub fn applyGLU(backend: *const ComputeBackend, linear_part: *const Matrix, gating_part: *const Matrix, allocator: Allocator) !*Matrix {
        return backend.applyGLU(linear_part, gating_part, allocator);
    }

    /// Apply Swish Gated Linear Unit (SwiGLU) activation
    pub fn applySwiGLU(backend: *const ComputeBackend, linear_part: *const Matrix, gating_part: *const Matrix, allocator: Allocator) !*Matrix {
        return backend.applySwiGLU(linear_part, gating_part, allocator);
    }
};

/// This function creates an appropriate backend based on the requested type
/// Falls back to CPU if the requested type is not available
pub fn createBackend(allocator: Allocator, backend_type: BackendType) !*ComputeBackend {
    // Use a simpler implementation that avoids comptime evaluation issues
    if (backend_type == .Metal) {
        if (@import("builtin").os.tag == .macos) {
            if (comptime @hasDecl(@import("root"), "enable_metal") and @import("root").enable_metal) {
                std.debug.print("Metal backend requested. Attempting to create...\n", .{});
                const metal_backend = @import("metal_backend.zig");
                return metal_backend.createMetalBackend(allocator) catch |err| {
                    std.debug.print("Failed to create Metal backend: {}, falling back to CPU\n", .{err});
                    return createCPUBackend(allocator);
                };
            }
            std.debug.print("Metal backend requested but not enabled in build, falling back to CPU\n", .{});
        } else {
            std.debug.print("Metal backend requested but not available on this OS, falling back to CPU\n", .{});
        }
    } else if (backend_type == .CUDA) {
        if (comptime @hasDecl(@import("root"), "enable_cuda") and @import("root").enable_cuda) {
            std.debug.print("CUDA backend requested but not implemented, falling back to CPU\n", .{});
        } else {
            std.debug.print("CUDA backend requested but not enabled in build, falling back to CPU\n", .{});
        }
    }

    // Default to CPU backend
    return createCPUBackend(allocator);
}

/// Signature of the function to create a CPU backend
/// Implementation will be in cpu_backend.zig
pub fn createCPUBackend(allocator: Allocator) !*ComputeBackend {
    const cpu_backend = @import("cpu_backend.zig");
    return cpu_backend.createCPUBackend(allocator);
}
