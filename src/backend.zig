const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;
const cpu_backend_mod = @import("cpu_backend.zig");
const metal_backend_mod = @import("metal_backend.zig"); // Ensure this import exists
const cuda_backend_mod = @import("cuda_backend.zig");
const matrix_mod = @import("matrix.zig");
const root = @import("root.zig");
const BackendInstance = root.BackendInstance;
const CpuMatrix = matrix_mod.Matrix;
const build_options = @import("build_options");

/// BackendType enum represents the available computation backends
pub const BackendType = enum {
    CPU,
    Metal, // macOS only
    CUDA, // NVIDIA GPUs
};

/// Cumulative backend work counters. GPU implementations update these at the
/// point where buffers, transfers, kernels, and synchronization actually occur.
pub const RuntimeStats = struct {
    buffer_allocations: usize = 0,
    host_to_device_transfers: usize = 0,
    host_to_device_bytes: usize = 0,
    device_to_host_transfers: usize = 0,
    device_to_host_bytes: usize = 0,
    kernel_launches: usize = 0,
    vendor_gemm_launches: usize = 0,
    synchronizations: usize = 0,
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
    applyActivationFn: *const fn (ptr: *anyopaque, matrix: *const Matrix, activation: *const fn (f64) f64, allocator: Allocator) error{OutOfMemory}!*Matrix,

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

    pub fn applyActivation(self: *const ComputeBackend, matrix: *const Matrix, activation: *const fn (f64) f64, allocator: Allocator) error{OutOfMemory}!*Matrix {
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

/// Matrix represents a backend-aware 2D array with native f32 storage.
/// Scalar access remains f64-compatible for the legacy public API.
pub const Matrix = struct {
    rows: usize,
    cols: usize,
    backend: BackendInstance, // Store the union instance directly
    // Implementation-specific data (opaque pointer to CPUMatrix or MetalMatrix)
    impl_data: *anyopaque,

    /// Creates a new matrix using the specified backend instance
    pub fn init(backend_instance: BackendInstance, allocator: Allocator, rows: usize, cols: usize) !*Matrix {
        // Call the initMatrix method on the BackendInstance union
        const matrix = try backend_instance.initMatrix(allocator, rows, cols);
        // The backend implementation (cpu/metal initMatrix) should have already created
        // the Matrix struct and set impl_data. We just need to store the backend instance.
        matrix.backend = backend_instance;
        return matrix;
    }

    /// Creates a backend-aware matrix by copying values from the CPU Matrix path.
    pub fn fromMatrix(backend_instance: BackendInstance, source: CpuMatrix, allocator: Allocator) !*Matrix {
        const result = try Matrix.init(backend_instance, allocator, source.rows, source.cols);
        errdefer result.deinit();

        for (0..source.rows) |row| {
            for (0..source.cols) |col| {
                result.set(row, col, try source.get(row, col));
            }
        }

        return result;
    }

    /// Copies this backend-aware matrix into the CPU Matrix representation.
    pub fn toMatrix(self: *const Matrix, allocator: Allocator) !CpuMatrix {
        var result = try CpuMatrix.init(allocator, self.rows, self.cols);
        errdefer result.deinit();

        for (0..self.rows) |row| {
            for (0..self.cols) |col| {
                try result.set(row, col, self.get(row, col));
            }
        }

        return result;
    }

    /// Frees the matrix memory
    pub fn deinit(self: *Matrix) void {
        // Call the deinitMatrix method on the BackendInstance union
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

    /// Replaces all matrix values from a contiguous f32 slice.
    ///
    /// Backends may defer the actual device upload until the matrix is used by
    /// a kernel. Keeping this operation bulk-oriented prevents callers from
    /// coupling tensor construction to per-element backend access.
    pub fn writeF32(self: *Matrix, values: []const f32) !void {
        if (values.len != self.rows * self.cols) {
            return error.DimensionMismatch;
        }
        if (!self.backend.writeMatrixF32(self, values)) return error.DataTransferFailed;
    }

    /// Reads all matrix values into a contiguous f32 slice.
    ///
    /// GPU backends synchronize their host mirror at most once before the
    /// element loop, so this remains one logical readback operation.
    pub fn readF32(self: *const Matrix, values: []f32) !void {
        if (values.len != self.rows * self.cols) {
            return error.DimensionMismatch;
        }

        if (!self.backend.readMatrixF32(self, values)) return error.DataTransferFailed;
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

    // --- Activation Functions directly on Matrix ---
    // These now use the backend instance stored in the matrix

    pub fn applyActivation(self: *const Matrix, activation_fn: *const fn (f64) f64, allocator: Allocator) !*Matrix {
        return self.backend.applyActivation(self, activation_fn, allocator);
    }

    pub fn applySoftmax(self: *const Matrix, allocator: Allocator) !*Matrix {
        return self.backend.applySoftmax(self, allocator);
    }

    pub fn applyGLU(self: *const Matrix, gating_part: *const Matrix, allocator: Allocator) !*Matrix {
        // Assuming GLU is self * sigmoid(gating_part)
        return self.backend.applyGLU(self, gating_part, allocator);
    }

    pub fn applySwiGLU(self: *const Matrix, gating_part: *const Matrix, allocator: Allocator) !*Matrix {
        // Assuming SwiGLU is self * swish(gating_part)
        return self.backend.applySwiGLU(self, gating_part, allocator);
    }
};

/// This function creates an appropriate backend based on the requested type
/// Falls back to CPU if the requested type is not available
pub fn createBackend(allocator: Allocator, backend_type: BackendType) !BackendInstance {
    if (backend_type == .Metal) {
        if (@import("builtin").os.tag == .macos) {
            std.debug.print("Metal backend requested. Attempting to create...\n", .{});
            const metal_ptr = metal_backend_mod.createMetalBackend(allocator) catch |err| {
                std.debug.print("Failed to create Metal backend: {}, falling back to CPU\n", .{err});
                return BackendInstance{ .CPU = try cpu_backend_mod.createCPUBackend(allocator) };
            };
            return BackendInstance{ .Metal = metal_ptr };
        } else {
            std.debug.print("Metal backend requested but not available on this OS, falling back to CPU\n", .{});
        }
    } else if (backend_type == .CUDA) {
        if (build_options.enable_cuda) {
            std.debug.print("CUDA backend requested. Attempting to create...\n", .{});
            const cuda_ptr = cuda_backend_mod.createCUDABackend(allocator) catch |err| {
                std.debug.print("Failed to create CUDA backend: {}, falling back to CPU\n", .{err});
                return BackendInstance{ .CPU = try cpu_backend_mod.createCPUBackend(allocator) };
            };
            return BackendInstance{ .CUDA = cuda_ptr };
        } else {
            std.debug.print("CUDA backend requested but not enabled in build, falling back to CPU\n", .{});
        }
    }

    // Default to CPU backend
    return BackendInstance{ .CPU = try cpu_backend_mod.createCPUBackend(allocator) };
}

/// Signature of the function to create a CPU backend
/// Implementation will be in cpu_backend.zig
pub fn createCPUBackend(allocator: Allocator) !*ComputeBackend {
    const cpu_backend = @import("cpu_backend.zig");
    return cpu_backend.createCPUBackend(allocator);
}
