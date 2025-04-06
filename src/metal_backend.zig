const std = @import("std");
const Allocator = std.mem.Allocator;
const backend = @import("backend.zig");
const ComputeBackend = backend.ComputeBackend;
const BackendType = backend.BackendType;
const Matrix = backend.Matrix;

// Add build options to check if Metal is enabled
const build_options = @import("build_options");
const enable_metal = build_options.enable_metal;

// Define errors
pub const MetalError = error{
    MetalNotSupported,
    MetalDeviceNotFound,
    ShaderCompilationFailed,
    InvalidBatchIndices,
    BufferCreationFailed,
    CommandBufferCreationFailed,
    DataTransferError,
};

// Import Metal wrapper only on macOS and when Metal is enabled
const Metal = if (@import("builtin").os.tag == .macos and enable_metal) struct {
    // Import the Metal C wrapper
    const c = @cImport({
        @cInclude("metal_wrapper.h");
    });

    // Re-export the wrapper types
    pub const Device = c.MTLDeviceRef;
    pub const CommandQueue = c.MTLCommandQueueRef;
    pub const Library = c.MTLLibraryRef;
    pub const Function = c.MTLFunctionRef;
    pub const Buffer = c.MTLBufferRef;
    pub const ComputePipelineState = c.MTLComputePipelineStateRef;

    pub const BufferOptions = struct {
        pub const StorageModeShared: u32 = 0;
    };

    // Metal API functions now call our C wrapper
    pub fn createSystemDefaultDevice() ?*Device {
        const result = c.metal_create_system_default_device();
        return if (result != null) @as(?*Device, @ptrFromInt(@intFromPtr(result))) else null;
    }

    pub fn deviceCreateCommandQueue(device: *Device) ?*CommandQueue {
        const device_ptr = @as(*anyopaque, @ptrCast(device));
        const result = c.metal_device_create_command_queue(device_ptr);
        return if (result != null) @as(?*CommandQueue, @ptrFromInt(@intFromPtr(result))) else null;
    }

    pub fn deviceCreateBuffer(device: *Device, length: usize, options: u32) ?*Buffer {
        const device_ptr = @as(*anyopaque, @ptrCast(device));
        const result = c.metal_device_create_buffer(device_ptr, length, options);
        return if (result != null) @as(?*Buffer, @ptrFromInt(@intFromPtr(result))) else null;
    }
} else struct {
    // Empty struct for non-macOS platforms or when Metal is disabled
    pub const Device = *anyopaque;
    pub const CommandQueue = *anyopaque;
    pub const Library = *anyopaque;
    pub const Function = *anyopaque;
    pub const Buffer = *anyopaque;
    pub const ComputePipelineState = *anyopaque;

    pub const BufferOptions = struct {
        pub const StorageModeShared: u32 = 0;
    };

    pub fn createSystemDefaultDevice() ?*Device {
        return null;
    }

    pub fn deviceCreateCommandQueue(_: *Device) ?*CommandQueue {
        return null;
    }

    pub fn deviceCreateBuffer(_: *Device, _: usize, _: u32) ?*Buffer {
        return null;
    }
};

/// Metal-specific implementation of a matrix
const MetalMatrix = struct {
    rows: usize,
    cols: usize,
    buffer: ?*Metal.Buffer, // GPU-side buffer
    host_data: []f64, // CPU-side data (for transfers)
    allocator: Allocator,
    dirty: bool, // Flag to indicate if CPU data needs to be uploaded to GPU

    pub fn init(allocator: Allocator, rows: usize, cols: usize, device: *Metal.Device) !*MetalMatrix {
        const metal_matrix = try allocator.create(MetalMatrix);

        // Allocate host memory
        const host_data = try allocator.alloc(f64, rows * cols);

        // Allocate GPU buffer (using sizeof(float) since Metal uses 32-bit floats)
        const buffer_size = rows * cols * @sizeOf(f32);
        const buffer = Metal.deviceCreateBuffer(device, buffer_size, Metal.BufferOptions.StorageModeShared) orelse return error.BufferCreationFailed;

        metal_matrix.* = MetalMatrix{
            .rows = rows,
            .cols = cols,
            .buffer = buffer,
            .host_data = host_data,
            .allocator = allocator,
            .dirty = true, // Mark as dirty to ensure first upload
        };

        return metal_matrix;
    }

    pub fn deinit(self: *MetalMatrix) void {
        // Free CPU memory
        self.allocator.free(self.host_data);

        // Buffer would be released automatically by ARC in real implementation
        // but we'd need proper bindings to handle reference counting

        self.allocator.destroy(self);
    }

    // Sync CPU data to GPU if needed
    pub fn syncToGPU(self: *MetalMatrix) void {
        if (!self.dirty) return;

        // Since we don't have actual buffer contents access in our wrapper yet,
        // just mark as clean and keep the CPU-side data as source of truth
        // This prevents the segfault by keeping coherent data on the CPU side

        // In a complete implementation, we would:
        // 1. Get a pointer to the buffer contents
        // 2. Convert f64 to f32 and copy data
        // 3. Mark as clean

        self.dirty = false;
    }

    // Sync GPU data to CPU
    pub fn syncToCPU(_: *MetalMatrix) void {
        // Since we don't have actual GPU computation yet,
        // no need to sync back - CPU data is already the source of truth

        // In a complete implementation, we would:
        // 1. Get a pointer to the buffer contents
        // 2. Convert f32 to f64 and copy data to host_data
    }
};

/// Metal backend implementation
pub const MetalBackend = struct {
    allocator: Allocator,

    // Metal-specific state
    device: ?*Metal.Device,
    command_queue: ?*Metal.CommandQueue,
    library: ?*Metal.Library,

    // Pipeline states for different operations
    matrix_multiply_pipeline: ?*Metal.ComputePipelineState,
    element_wise_add_pipeline: ?*Metal.ComputePipelineState,
    element_wise_subtract_pipeline: ?*Metal.ComputePipelineState,
    element_wise_multiply_pipeline: ?*Metal.ComputePipelineState,
    matrix_scale_pipeline: ?*Metal.ComputePipelineState,
    matrix_transpose_pipeline: ?*Metal.ComputePipelineState,

    // Activation function pipelines
    sigmoid_pipeline: ?*Metal.ComputePipelineState,
    relu_pipeline: ?*Metal.ComputePipelineState,
    tanh_pipeline: ?*Metal.ComputePipelineState,
    swish_pipeline: ?*Metal.ComputePipelineState,

    // Softmax pipelines
    softmax_find_max_pipeline: ?*Metal.ComputePipelineState,
    softmax_exp_sum_pipeline: ?*Metal.ComputePipelineState,
    softmax_normalize_pipeline: ?*Metal.ComputePipelineState,

    // GLU pipelines
    glu_pipeline: ?*Metal.ComputePipelineState,
    swiglu_pipeline: ?*Metal.ComputePipelineState,

    /// Initialize a new Metal backend
    pub fn init(allocator: Allocator) !*MetalBackend {
        if (@import("builtin").os.tag != .macos) {
            return error.MetalNotSupported;
        }

        const device = Metal.createSystemDefaultDevice() orelse
            return error.MetalDeviceNotFound;

        const metal_backend = try allocator.create(MetalBackend);
        metal_backend.* = MetalBackend{
            .allocator = allocator,
            .device = device,
            .command_queue = null, // Initialize all fields to null
            .library = null,
            .matrix_multiply_pipeline = null,
            .element_wise_add_pipeline = null,
            .element_wise_subtract_pipeline = null,
            .element_wise_multiply_pipeline = null,
            .matrix_scale_pipeline = null,
            .matrix_transpose_pipeline = null,
            .sigmoid_pipeline = null,
            .relu_pipeline = null,
            .tanh_pipeline = null,
            .swish_pipeline = null,
            .softmax_find_max_pipeline = null,
            .softmax_exp_sum_pipeline = null,
            .softmax_normalize_pipeline = null,
            .glu_pipeline = null,
            .swiglu_pipeline = null,
        };

        // Create command queue, etc. (handle potential errors)
        metal_backend.command_queue = Metal.deviceCreateCommandQueue(device) orelse {
            // If queue creation fails, destroy the partially initialized backend
            metal_backend.allocator.destroy(metal_backend);
            return error.CommandQueueCreationFailed;
        };

        // TODO: Initialize library, compile shaders, create pipeline states here
        // This requires reading the .metal file, compiling it, getting functions,
        // and creating ComputePipelineState objects. Example:
        // const library_source = try loadShaderSource("shaders.metal");
        // metal_backend.library = try createLibraryFromSource(device, library_source);
        // const matMulFunction = try getFunction(metal_backend.library.?, "matrixMultiply");
        // metal_backend.matrix_multiply_pipeline = try createPipelineState(device, matMulFunction);
        // ... etc for all pipelines ...
        // If any of these fail, destroy the backend and return the error.

        return metal_backend;
    }

    /// Free all resources used by the Metal backend
    pub fn deinit(self: *MetalBackend) void {
        // Release Metal resources (omitted for brevity)
        self.allocator.destroy(self);
    }

    // Helper to get Metal matrix from abstract Matrix
    fn getMetalMatrix(matrix: *const Matrix) *MetalMatrix {
        // Use @alignCast before @ptrCast for safety
        return @as(*MetalMatrix, @ptrCast(@alignCast(matrix.impl_data)));
    }

    // --- Implementation Functions (Make Public) ---

    pub fn initMatrix(ptr: *anyopaque, allocator: Allocator, rows: usize, cols: usize) error{OutOfMemory}!*Matrix {
        // Correctly cast ptr with alignment check
        const self = @as(*MetalBackend, @ptrCast(@alignCast(ptr)));

        // Create Metal matrix
        const metal_matrix = MetalMatrix.init(allocator, rows, cols, self.device.?) catch |err| {
            std.log.warn("Metal matrix initialization failed: {s}", .{@errorName(err)});
            return error.OutOfMemory;
        };

        // Create abstract Matrix
        const matrix = try allocator.create(Matrix);
        matrix.* = Matrix{
            .rows = rows,
            .cols = cols,
            .backend = undefined, // Backend should be set by the caller via BackendInstance
            .impl_data = metal_matrix,
        };

        return matrix;
    }

    pub fn deinitMatrix(_: *anyopaque, matrix: *Matrix) void {
        const metal_matrix = getMetalMatrix(matrix);
        metal_matrix.deinit();

        // Free the abstract Matrix
        const allocator = metal_matrix.allocator;
        allocator.destroy(matrix);
    }

    pub fn copyMatrix(ptr: *anyopaque, source: *const Matrix, allocator: Allocator) error{OutOfMemory}!*Matrix {
        _ = @as(*MetalBackend, @ptrCast(@alignCast(ptr))); // Mark ptr as unused if not needed for init
        const source_metal = getMetalMatrix(source);

        // Create new matrix with same dimensions
        const result = try initMatrix(ptr, allocator, source.rows, source.cols);
        const result_metal = getMetalMatrix(result);

        // Copy data from source to result
        @memcpy(result_metal.host_data, source_metal.host_data);
        result_metal.dirty = true;

        return result;
    }

    pub fn getMatrixElement(_: *anyopaque, matrix: *const Matrix, row: usize, col: usize) f64 {
        if (row >= matrix.rows) return 0.0;
        if (col >= matrix.cols) return 0.0;

        const metal_matrix = getMetalMatrix(matrix);

        // Return 0.0 if host_data is invalid
        if (metal_matrix.host_data.len == 0) return 0.0;

        const index = row * matrix.cols + col;
        if (index >= metal_matrix.host_data.len) return 0.0;

        // Ensure CPU data is up to date
        metal_matrix.syncToCPU();

        // Return element
        return metal_matrix.host_data[index];
    }

    pub fn setMatrixElement(_: *anyopaque, matrix: *Matrix, row: usize, col: usize, value: f64) void {
        const metal_matrix = getMetalMatrix(matrix);

        // Update CPU data
        metal_matrix.host_data[row * matrix.cols + col] = value;
        metal_matrix.dirty = true;
    }

    pub fn fillMatrix(_: *anyopaque, matrix: *Matrix, value: f64) void {
        const metal_matrix = getMetalMatrix(matrix);

        // Fill CPU data
        for (metal_matrix.host_data) |*element| {
            element.* = value;
        }

        metal_matrix.dirty = true;
    }

    pub fn dotProduct(ptr: *anyopaque, a: *const Matrix, b: *const Matrix, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!*Matrix {
        // Check dimensions
        if (a.cols != b.rows) {
            return error.DimensionMismatch;
        }

        // Cast ptr to backend, but mark as unused if not needed in this implementation
        _ = @as(*MetalBackend, @ptrCast(@alignCast(ptr)));

        // Create result matrix
        const result = try initMatrix(ptr, allocator, a.rows, b.cols);

        // Get Metal matrices
        const a_metal = getMetalMatrix(a);
        const b_metal = getMetalMatrix(b);
        const result_metal = getMetalMatrix(result);

        // Ensure data is marked as on GPU
        a_metal.syncToGPU();
        b_metal.syncToGPU();

        // FALLBACK: Perform CPU calculation for now
        // This ensures we have valid data when accessed
        for (0..a.rows) |i| {
            for (0..b.cols) |j| {
                var sum: f64 = 0.0;
                for (0..a.cols) |k| {
                    sum += a_metal.host_data[i * a.cols + k] * b_metal.host_data[k * b.cols + j];
                }
                result_metal.host_data[i * b.cols + j] = sum;
            }
        }

        // In a real implementation, we would use Metal compute kernels

        return result;
    }

    pub fn add(ptr: *anyopaque, a: *const Matrix, b: *const Matrix, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!*Matrix {
        // Check dimensions
        if (a.rows != b.rows or a.cols != b.cols) {
            return error.DimensionMismatch;
        }

        // Cast ptr to backend, but mark as unused if not needed in this implementation
        _ = @as(*MetalBackend, @ptrCast(@alignCast(ptr)));

        // Create result matrix
        const result = try initMatrix(ptr, allocator, a.rows, a.cols);

        // Get Metal matrices
        const a_metal = getMetalMatrix(a);
        const b_metal = getMetalMatrix(b);
        const result_metal = getMetalMatrix(result);

        // Ensure data is marked as on GPU
        a_metal.syncToGPU();
        b_metal.syncToGPU();

        // FALLBACK: Perform CPU calculation for now
        for (0..a.rows * a.cols) |i| {
            result_metal.host_data[i] = a_metal.host_data[i] + b_metal.host_data[i];
        }

        // In a real implementation, we would use Metal compute kernels

        return result;
    }

    pub fn subtract(ptr: *anyopaque, a: *const Matrix, b: *const Matrix, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!*Matrix {
        // Similar to add but using the subtraction kernel
        // Check dimensions
        if (a.rows != b.rows or a.cols != b.cols) {
            return error.DimensionMismatch;
        }

        _ = @as(*MetalBackend, @ptrCast(@alignCast(ptr))); // Mark ptr as unused if not needed for init

        // Create result matrix and perform computation (omitted for brevity)
        const result = try initMatrix(ptr, allocator, a.rows, a.cols);

        return result;
    }

    pub fn elementWiseMultiply(ptr: *anyopaque, a: *const Matrix, b: *const Matrix, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!*Matrix {
        // Check dimensions
        if (a.rows != b.rows or a.cols != b.cols) {
            return error.DimensionMismatch;
        }

        // Cast ptr to backend, but mark as unused if not needed in this implementation
        _ = @as(*MetalBackend, @ptrCast(@alignCast(ptr)));

        // Create result matrix
        const result = try initMatrix(ptr, allocator, a.rows, a.cols);

        // Get Metal matrices
        const a_metal = getMetalMatrix(a);
        const b_metal = getMetalMatrix(b);
        const result_metal = getMetalMatrix(result);

        // Ensure data is marked as on GPU
        a_metal.syncToGPU();
        b_metal.syncToGPU();

        // FALLBACK: Perform CPU calculation for now
        for (0..a.rows * a.cols) |i| {
            result_metal.host_data[i] = a_metal.host_data[i] * b_metal.host_data[i];
        }

        // In a real implementation, we would use Metal compute kernels

        return result;
    }

    pub fn scale(ptr: *anyopaque, matrix: *const Matrix, _: f64, allocator: Allocator) error{OutOfMemory}!*Matrix {
        _ = @as(*MetalBackend, @ptrCast(@alignCast(ptr))); // Mark ptr as unused if not needed for init

        // Create result matrix
        const result = try initMatrix(ptr, allocator, matrix.rows, matrix.cols);

        // Get Metal matrices
        const matrix_metal = getMetalMatrix(matrix);
        const result_metal = getMetalMatrix(result);

        // Ensure data is on GPU
        matrix_metal.syncToGPU();

        // Execute compute kernel (omitted for brevity)

        return result_metal;
    }

    pub fn sumRows(ptr: *anyopaque, matrix: *const Matrix, allocator: Allocator) error{OutOfMemory}!*Matrix {
        _ = @as(*MetalBackend, @ptrCast(@alignCast(ptr))); // Mark ptr as unused if not needed for init

        // Create result matrix (1 x cols)
        const result = try initMatrix(ptr, allocator, 1, matrix.cols);

        // Get Metal matrices
        const matrix_metal = getMetalMatrix(matrix);
        const result_metal = getMetalMatrix(result);

        // Ensure data is on GPU
        matrix_metal.syncToGPU();

        // Execute compute kernel (omitted for brevity)
        // This would likely use a reduction algorithm

        return result_metal;
    }

    pub fn transpose(ptr: *anyopaque, matrix: *const Matrix, allocator: Allocator) error{OutOfMemory}!*Matrix {
        // Cast ptr to backend, but mark as unused if not needed in this implementation
        _ = @as(*MetalBackend, @ptrCast(@alignCast(ptr)));

        // Create result matrix with transposed dimensions
        const result = try initMatrix(ptr, allocator, matrix.cols, matrix.rows);

        // Get Metal matrices
        const matrix_metal = getMetalMatrix(matrix);
        const result_metal = getMetalMatrix(result);

        // Ensure data is marked as on GPU
        matrix_metal.syncToGPU();

        // FALLBACK: Perform CPU calculation for now
        for (0..matrix.rows) |i| {
            for (0..matrix.cols) |j| {
                result_metal.host_data[j * matrix.rows + i] = matrix_metal.host_data[i * matrix.cols + j];
            }
        }

        // In a real implementation, we would use Metal compute kernels

        return result;
    }

    pub fn extractBatch(ptr: *anyopaque, matrix: *const Matrix, start: usize, end: usize, allocator: Allocator) error{ OutOfMemory, InvalidBatchIndices }!*Matrix {
        // Check indices
        if (start >= end or end > matrix.rows) {
            return error.InvalidBatchIndices;
        }

        _ = @as(*MetalBackend, @ptrCast(@alignCast(ptr))); // Mark ptr as unused if not needed for init

        // Create result matrix
        const batch_rows = end - start;
        const result = try initMatrix(ptr, allocator, batch_rows, matrix.cols);

        // Get Metal matrices
        const matrix_metal = getMetalMatrix(matrix);
        const result_metal = getMetalMatrix(result);

        // Copy the batch from CPU data (could be optimized to use GPU)
        const src_offset = start * matrix.cols;
        @memcpy(result_metal.host_data, matrix_metal.host_data[src_offset..][0 .. batch_rows * matrix.cols]);

        result_metal.dirty = true;

        return result;
    }

    pub fn randomize(_: *anyopaque, matrix: *Matrix, min: f64, max: f64) void {
        const metal_matrix = getMetalMatrix(matrix);

        // Initialize random number generator
        var rng = std.Random.DefaultPrng.init(@as(u64, @bitCast(std.time.milliTimestamp())));

        // Fill with random values
        for (metal_matrix.host_data) |*element| {
            const random_float = rng.random().float(f64);
            element.* = min + random_float * (max - min);
        }

        metal_matrix.dirty = true;
    }

    pub fn applyActivation(ptr: *anyopaque, matrix: *const Matrix, activation: fn (f64) f64, allocator: Allocator) error{OutOfMemory}!*Matrix {
        _ = @as(*MetalBackend, @ptrCast(@alignCast(ptr))); // Mark ptr as unused if not needed for init

        // Create result matrix
        const result = try initMatrix(ptr, allocator, matrix.rows, matrix.cols);

        // Get Metal matrices
        const matrix_metal = getMetalMatrix(matrix);
        const result_metal = getMetalMatrix(result);

        // For now, apply on CPU (would select appropriate GPU kernel in full implementation)
        for (0..matrix.rows * matrix.cols) |i| {
            result_metal.host_data[i] = activation(matrix_metal.host_data[i]);
        }

        result_metal.dirty = true;

        return result;
    }

    pub fn applySoftmax(ptr: *anyopaque, matrix: *const Matrix, allocator: Allocator) error{OutOfMemory}!*Matrix {
        _ = @as(*MetalBackend, @ptrCast(@alignCast(ptr))); // Mark ptr as unused if not needed for init

        // Create result matrix
        const result = try initMatrix(ptr, allocator, matrix.rows, matrix.cols);

        // Get Metal matrices
        const matrix_metal = getMetalMatrix(matrix);
        const result_metal = getMetalMatrix(result);

        // Ensure data is on GPU
        matrix_metal.syncToGPU();

        // Execute softmax compute kernels (omitted for brevity)
        // This would use the three-step softmax process in Metal:
        // 1. Find max per row
        // 2. Compute exp(x - max) and sum
        // 3. Normalize by sum

        return result_metal;
    }

    pub fn applyGLU(ptr: *anyopaque, linear_part: *const Matrix, gating_part: *const Matrix, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!*Matrix {
        // Check dimensions
        if (linear_part.rows != gating_part.rows or linear_part.cols != gating_part.cols) {
            return error.DimensionMismatch;
        }

        _ = @as(*MetalBackend, @ptrCast(@alignCast(ptr))); // Mark ptr as unused if not needed for init

        // Create result matrix
        const result = try initMatrix(ptr, allocator, linear_part.rows, linear_part.cols);

        // Get Metal matrices
        const linear_metal = getMetalMatrix(linear_part);
        const gating_metal = getMetalMatrix(gating_part);
        const result_metal = getMetalMatrix(result);

        // Ensure data is on GPU
        linear_metal.syncToGPU();
        gating_metal.syncToGPU();

        // Execute compute kernel (omitted for brevity)
        // GLU: result = linear_part * sigmoid(gating_part)

        return result_metal;
    }

    pub fn applySwiGLU(ptr: *anyopaque, linear_part: *const Matrix, gating_part: *const Matrix, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!*Matrix {
        // Check dimensions
        if (linear_part.rows != gating_part.rows or linear_part.cols != gating_part.cols) {
            return error.DimensionMismatch;
        }

        _ = @as(*MetalBackend, @ptrCast(@alignCast(ptr))); // Mark ptr as unused if not needed for init

        // Create result matrix
        const result = try initMatrix(ptr, allocator, linear_part.rows, linear_part.cols);

        // Get Metal matrices
        const linear_metal = getMetalMatrix(linear_part);
        const gating_metal = getMetalMatrix(gating_part);
        const result_metal = getMetalMatrix(result);

        // Ensure data is on GPU
        linear_metal.syncToGPU();
        gating_metal.syncToGPU();

        // Execute compute kernel (omitted for brevity)
        // SwiGLU: result = linear_part * swish(gating_part)

        return result_metal;
    }

    pub fn getBackendType(_: *anyopaque) BackendType {
        return .Metal;
    }
};

/// Creates a Metal backend instance, returning an opaque pointer
pub fn createMetalBackend(allocator: Allocator) !*anyopaque {
    const metal_backend = try MetalBackend.init(allocator);
    errdefer metal_backend.deinit(); // Ensure cleanup on populate error if any
    return @ptrCast(metal_backend);
}
