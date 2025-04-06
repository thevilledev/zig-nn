//! By convention, root.zig is the root source file when making a library. If
//! you are making an executable, the convention is to delete this file and
//! start with main.zig instead.
const std = @import("std");
const testing = std.testing;

// Import the modules
const matrix_mod = @import("matrix.zig");
const activation_mod = @import("activation.zig");
const layer_mod = @import("layer.zig");
const network_mod = @import("network.zig");
const inference_service_mod = @import("inference_service.zig");
const visualiser_mod = @import("visualiser.zig");
const backend_mod = @import("backend.zig");
const cpu_backend_mod = @import("cpu_backend.zig");
const metal_backend_mod = @import("metal_backend.zig");

/// Library usage example:
///
/// Simple example of creating and using a neural network
///
/// ```zig
/// // Create a neural network
/// var network = Network.init(allocator);
/// defer network.deinit();
///
/// // Add layers (input -> hidden -> output)
/// try network.addLayer(2, 3, Activation.relu, Activation.relu_derivative);
/// try network.addLayer(3, 1, Activation.sigmoid, Activation.sigmoid_derivative);
///
/// // Create input data
/// var input = try Matrix.init(allocator, 1, 2);
/// defer input.deinit();
/// input.set(0, 0, 0.5);
/// input.set(0, 1, 0.8);
///
/// // Forward pass
/// var output = try network.forward(input);
/// defer output.deinit();
///
/// // Get the result
/// const result = output.get(0, 0);
/// ```
pub const Matrix = matrix_mod.Matrix;
pub const Activation = activation_mod.Activation;
pub const Layer = layer_mod.Layer;
pub const GatedLayer = layer_mod.GatedLayer;
pub const Network = network_mod.Network;
pub const LayerType = network_mod.LayerType;
pub const LayerVariant = network_mod.LayerVariant;
pub const InferenceService = inference_service_mod.InferenceService;
pub const visualiseNetwork = visualiser_mod.visualiseNetwork;

// Export backend interface types
pub const BackendMatrix = backend_mod.Matrix;
pub const BackendType = backend_mod.BackendType;

// Export concrete backend types
pub const CPUBackend = cpu_backend_mod.CPUBackend;
pub const MetalBackend = metal_backend_mod.MetalBackend;

// Define the BackendInstance tagged union using *anyopaque
pub const BackendInstance = union(BackendType) {
    CPU: *anyopaque,
    Metal: *anyopaque,
    CUDA: *anyopaque, // Will be a CPU backend that acts as a fallback

    // --- Mirrored Interface Methods ---

    pub fn initMatrix(self: BackendInstance, allocator: std.mem.Allocator, rows: usize, cols: usize) !*BackendMatrix {
        return switch (self) {
            .CPU => |ptr| CPUBackend.initMatrix(ptr, allocator, rows, cols),
            .Metal => |ptr| MetalBackend.initMatrix(ptr, allocator, rows, cols),
            .CUDA => |ptr| CPUBackend.initMatrix(ptr, allocator, rows, cols), // Use CPU as fallback
        };
    }

    pub fn deinitMatrix(self: BackendInstance, matrix: *BackendMatrix) void {
        switch (self) {
            .CPU => |ptr| CPUBackend.deinitMatrix(ptr, matrix),
            .Metal => |ptr| MetalBackend.deinitMatrix(ptr, matrix),
            .CUDA => |ptr| CPUBackend.deinitMatrix(ptr, matrix), // Use CPU as fallback
        }
    }

    pub fn copyMatrix(self: BackendInstance, source: *const BackendMatrix, allocator: std.mem.Allocator) !*BackendMatrix {
        return switch (self) {
            .CPU => |ptr| CPUBackend.copyMatrix(ptr, source, allocator),
            .Metal => |ptr| MetalBackend.copyMatrix(ptr, source, allocator),
            .CUDA => |ptr| CPUBackend.copyMatrix(ptr, source, allocator), // Use CPU as fallback
        };
    }

    pub fn getMatrixElement(self: BackendInstance, matrix: *const BackendMatrix, row: usize, col: usize) f64 {
        return switch (self) {
            .CPU => |ptr| CPUBackend.getMatrixElement(ptr, matrix, row, col),
            .Metal => |ptr| MetalBackend.getMatrixElement(ptr, matrix, row, col),
            .CUDA => |ptr| CPUBackend.getMatrixElement(ptr, matrix, row, col), // Use CPU as fallback
        };
    }

    pub fn setMatrixElement(self: BackendInstance, matrix: *BackendMatrix, row: usize, col: usize, value: f64) void {
        switch (self) {
            .CPU => |ptr| CPUBackend.setMatrixElement(ptr, matrix, row, col, value),
            .Metal => |ptr| MetalBackend.setMatrixElement(ptr, matrix, row, col, value),
            .CUDA => |ptr| CPUBackend.setMatrixElement(ptr, matrix, row, col, value), // Use CPU as fallback
        }
    }

    pub fn fillMatrix(self: BackendInstance, matrix: *BackendMatrix, value: f64) void {
        switch (self) {
            .CPU => |ptr| CPUBackend.fillMatrix(ptr, matrix, value),
            .Metal => |ptr| MetalBackend.fillMatrix(ptr, matrix, value),
            .CUDA => |ptr| CPUBackend.fillMatrix(ptr, matrix, value), // Use CPU as fallback
        }
    }

    pub fn dotProduct(self: BackendInstance, a: *const BackendMatrix, b: *const BackendMatrix, allocator: std.mem.Allocator) !*BackendMatrix {
        return switch (self) {
            .CPU => |ptr| CPUBackend.dotProduct(ptr, a, b, allocator),
            .Metal => |ptr| MetalBackend.dotProduct(ptr, a, b, allocator),
            .CUDA => |ptr| CPUBackend.dotProduct(ptr, a, b, allocator), // Use CPU as fallback
        };
    }

    pub fn add(self: BackendInstance, a: *const BackendMatrix, b: *const BackendMatrix, allocator: std.mem.Allocator) !*BackendMatrix {
        return switch (self) {
            .CPU => |ptr| CPUBackend.add(ptr, a, b, allocator),
            .Metal => |ptr| MetalBackend.add(ptr, a, b, allocator),
            .CUDA => |ptr| CPUBackend.add(ptr, a, b, allocator), // Use CPU as fallback
        };
    }

    pub fn subtract(self: BackendInstance, a: *const BackendMatrix, b: *const BackendMatrix, allocator: std.mem.Allocator) !*BackendMatrix {
        return switch (self) {
            .CPU => |ptr| CPUBackend.subtract(ptr, a, b, allocator),
            .Metal => |ptr| MetalBackend.subtract(ptr, a, b, allocator),
            .CUDA => |ptr| CPUBackend.subtract(ptr, a, b, allocator), // Use CPU as fallback
        };
    }

    pub fn elementWiseMultiply(self: BackendInstance, a: *const BackendMatrix, b: *const BackendMatrix, allocator: std.mem.Allocator) !*BackendMatrix {
        return switch (self) {
            .CPU => |ptr| CPUBackend.elementWiseMultiply(ptr, a, b, allocator),
            .Metal => |ptr| MetalBackend.elementWiseMultiply(ptr, a, b, allocator),
            .CUDA => |ptr| CPUBackend.elementWiseMultiply(ptr, a, b, allocator), // Use CPU as fallback
        };
    }

    pub fn scale(self: BackendInstance, matrix: *const BackendMatrix, scalar: f64, allocator: std.mem.Allocator) !*BackendMatrix {
        return switch (self) {
            .CPU => |ptr| CPUBackend.scale(ptr, matrix, scalar, allocator),
            .Metal => |ptr| MetalBackend.scale(ptr, matrix, scalar, allocator),
            .CUDA => |ptr| CPUBackend.scale(ptr, matrix, scalar, allocator), // Use CPU as fallback
        };
    }

    pub fn sumRows(self: BackendInstance, matrix: *const BackendMatrix, allocator: std.mem.Allocator) !*BackendMatrix {
        return switch (self) {
            .CPU => |ptr| CPUBackend.sumRows(ptr, matrix, allocator),
            .Metal => |ptr| MetalBackend.sumRows(ptr, matrix, allocator),
            .CUDA => |ptr| CPUBackend.sumRows(ptr, matrix, allocator), // Use CPU as fallback
        };
    }

    pub fn transpose(self: BackendInstance, matrix: *const BackendMatrix, allocator: std.mem.Allocator) !*BackendMatrix {
        return switch (self) {
            .CPU => |ptr| CPUBackend.transpose(ptr, matrix, allocator),
            .Metal => |ptr| MetalBackend.transpose(ptr, matrix, allocator),
            .CUDA => |ptr| CPUBackend.transpose(ptr, matrix, allocator), // Use CPU as fallback
        };
    }

    pub fn extractBatch(self: BackendInstance, matrix: *const BackendMatrix, start: usize, end: usize, allocator: std.mem.Allocator) !*BackendMatrix {
        return switch (self) {
            .CPU => |ptr| CPUBackend.extractBatch(ptr, matrix, start, end, allocator),
            .Metal => |ptr| MetalBackend.extractBatch(ptr, matrix, start, end, allocator),
            .CUDA => |ptr| CPUBackend.extractBatch(ptr, matrix, start, end, allocator), // Use CPU as fallback
        };
    }

    pub fn randomize(self: BackendInstance, matrix: *BackendMatrix, min: f64, max: f64) void {
        switch (self) {
            .CPU => |ptr| CPUBackend.randomize(ptr, matrix, min, max),
            .Metal => |ptr| MetalBackend.randomize(ptr, matrix, min, max),
            .CUDA => |ptr| CPUBackend.randomize(ptr, matrix, min, max), // Use CPU as fallback
        }
    }

    pub fn applyActivation(self: BackendInstance, matrix: *const BackendMatrix, activation_fn: fn (f64) f64, allocator: std.mem.Allocator) !*BackendMatrix {
        return switch (self) {
            .CPU => |ptr| CPUBackend.applyActivation(ptr, matrix, activation_fn, allocator),
            .Metal => |ptr| MetalBackend.applyActivation(ptr, matrix, activation_fn, allocator),
            .CUDA => |ptr| CPUBackend.applyActivation(ptr, matrix, activation_fn, allocator), // Use CPU as fallback
        };
    }

    pub fn applySoftmax(self: BackendInstance, matrix: *const BackendMatrix, allocator: std.mem.Allocator) !*BackendMatrix {
        return switch (self) {
            .CPU => |ptr| CPUBackend.applySoftmax(ptr, matrix, allocator),
            .Metal => |ptr| MetalBackend.applySoftmax(ptr, matrix, allocator),
            .CUDA => |ptr| CPUBackend.applySoftmax(ptr, matrix, allocator), // Use CPU as fallback
        };
    }

    pub fn applyGLU(self: BackendInstance, linear_part: *const BackendMatrix, gating_part: *const BackendMatrix, allocator: std.mem.Allocator) !*BackendMatrix {
        return switch (self) {
            .CPU => |ptr| CPUBackend.applyGLU(ptr, linear_part, gating_part, allocator),
            .Metal => |ptr| MetalBackend.applyGLU(ptr, linear_part, gating_part, allocator),
            .CUDA => |ptr| CPUBackend.applyGLU(ptr, linear_part, gating_part, allocator), // Use CPU as fallback
        };
    }

    pub fn applySwiGLU(self: BackendInstance, linear_part: *const BackendMatrix, gating_part: *const BackendMatrix, allocator: std.mem.Allocator) !*BackendMatrix {
        return switch (self) {
            .CPU => |ptr| CPUBackend.applySwiGLU(ptr, linear_part, gating_part, allocator),
            .Metal => |ptr| MetalBackend.applySwiGLU(ptr, linear_part, gating_part, allocator),
            .CUDA => |ptr| CPUBackend.applySwiGLU(ptr, linear_part, gating_part, allocator), // Use CPU as fallback
        };
    }

    // --- End Mirrored Interface Methods ---

    /// Returns the backend type tag
    pub fn getBackendType(self: BackendInstance) BackendType {
        return @as(BackendType, @enumFromInt(@intFromEnum(self)));
    }

    /// Deinitializes the backend instance
    pub fn deinit(self: BackendInstance) void {
        switch (self) {
            .CPU => |ptr| @as(*cpu_backend_mod.CPUBackend, @ptrCast(@alignCast(ptr))).deinit(),
            .Metal => |ptr| @as(*metal_backend_mod.MetalBackend, @ptrCast(@alignCast(ptr))).deinit(),
            .CUDA => |ptr| @as(*cpu_backend_mod.CPUBackend, @ptrCast(@alignCast(ptr))).deinit(), // Use CPU as fallback
        }
    }
};

// Export backend creation functions that return *anyopaque or the union
pub const createCPUBackend = cpu_backend_mod.createCPUBackend;
pub const createMetalBackend = metal_backend_mod.createMetalBackend;
// Keep the generic createBackend that returns the union
pub const createBackend = backend_mod.createBackend;
