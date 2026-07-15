//! By convention, root.zig is the root source file when making a library. If
//! you are making an executable, the convention is to delete this file and
//! start with main.zig instead.
const std = @import("std");
const testing = std.testing;
const build_options = @import("build_options");

// Import the modules
const matrix_mod = @import("matrix.zig");
const activation_mod = @import("activation.zig");
const layer_mod = @import("layer.zig");
const layer_norm_mod = @import("layer_norm.zig");
const network_mod = @import("network.zig");
const inference_service_mod = @import("inference_service.zig");
const visualiser_mod = @import("visualiser.zig");
const backend_mod = @import("backend.zig");
const cpu_backend_mod = @import("cpu_backend.zig");
const metal_backend_mod = @import("metal_backend.zig");
const cuda_backend_mod = @import("cuda_backend.zig");
const rocm_backend_mod = @import("rocm_backend.zig");
const quantization_mod = @import("quantization.zig");
const tensor_mod = @import("tensor.zig");
const text_mod = @import("text.zig");
const modules_mod = @import("modules.zig");
const training_mod = @import("training.zig");
const spatial_mod = @import("spatial.zig");
const recurrent_mod = @import("recurrent.zig");
const reinforcement_mod = @import("reinforcement.zig");
const transformer_mod = @import("transformer.zig");
const embeddings_mod = @import("embeddings.zig");

/// Library usage example:
///
/// Simple example of creating and using a neural network
///
/// ```zig
/// // Create a neural network
/// var network = Network.init(allocator, 0.1, LossFunction.MeanSquaredError);
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
pub const LayerNorm = layer_norm_mod.LayerNorm;
pub const Network = network_mod.Network;
pub const BackendNetwork = network_mod.BackendNetwork;
pub const BackendTrainer = network_mod.BackendTrainer;
pub const BackendOptimizerKind = network_mod.BackendOptimizerKind;
pub const BackendOptimizerConfig = network_mod.BackendOptimizerConfig;
pub const LayerType = network_mod.LayerType;
pub const LayerVariant = network_mod.LayerVariant;
pub const LossFunction = network_mod.LossFunction;
pub const InferenceService = inference_service_mod.InferenceService;
pub const visualiseNetwork = visualiser_mod.visualiseNetwork;
pub const Quantization = quantization_mod;
pub const Tensor = tensor_mod.Tensor;
pub const Text = text_mod;
pub const TensorShape = tensor_mod.Shape;
pub const TensorDType = tensor_mod.DType;
pub const TensorActivation = tensor_mod.ActivationKind;
pub const Device = tensor_mod.Device;
pub const DevicePreference = tensor_mod.DevicePreference;
pub const ExecutionContext = tensor_mod.ExecutionContext;
pub const ExecutionStats = tensor_mod.ExecutionStats;
pub const BackendRuntimeStats = backend_mod.RuntimeStats;
pub const Modules = modules_mod;
pub const Training = training_mod;
pub const Spatial = spatial_mod;
pub const Recurrent = recurrent_mod;
pub const Reinforcement = reinforcement_mod;
pub const Transformer = transformer_mod;
pub const Embeddings = embeddings_mod;

// Export backend interface types
pub const BackendMatrix = backend_mod.Matrix;
pub const BackendType = backend_mod.BackendType;
pub const enable_metal = build_options.enable_metal;
pub const enable_cuda = build_options.enable_cuda;
pub const enable_rocm = build_options.enable_rocm;

// Export concrete backend types
pub const CPUBackend = cpu_backend_mod.CPUBackend;
pub const MetalBackend = metal_backend_mod.MetalBackend;
pub const CUDABackend = cuda_backend_mod.CUDABackend;
pub const ROCmBackend = rocm_backend_mod.ROCmBackend;

// Define the BackendInstance tagged union using *anyopaque
pub const BackendInstance = union(BackendType) {
    CPU: *anyopaque,
    Metal: *anyopaque,
    CUDA: *anyopaque,
    ROCm: *anyopaque,

    // --- Mirrored Interface Methods ---

    fn attachBackend(self: BackendInstance, matrix: *BackendMatrix) *BackendMatrix {
        matrix.backend = self;
        return matrix;
    }

    pub fn initMatrix(self: BackendInstance, allocator: std.mem.Allocator, rows: usize, cols: usize) !*BackendMatrix {
        const matrix = try switch (self) {
            .CPU => |ptr| CPUBackend.initMatrix(ptr, allocator, rows, cols),
            .Metal => |ptr| MetalBackend.initMatrix(ptr, allocator, rows, cols),
            .CUDA => |ptr| CUDABackend.initMatrix(ptr, allocator, rows, cols),
            .ROCm => |ptr| ROCmBackend.initMatrix(ptr, allocator, rows, cols),
        };
        return self.attachBackend(matrix);
    }

    pub fn runtimeStats(self: BackendInstance) BackendRuntimeStats {
        return switch (self) {
            .CPU => |ptr| CPUBackend.runtimeStats(ptr),
            .Metal => |ptr| MetalBackend.runtimeStats(ptr),
            .CUDA => |ptr| CUDABackend.runtimeStats(ptr),
            .ROCm => |ptr| ROCmBackend.runtimeStats(ptr),
        };
    }

    pub fn resetRuntimeStats(self: BackendInstance) void {
        switch (self) {
            .CPU => |ptr| CPUBackend.resetRuntimeStats(ptr),
            .Metal => |ptr| MetalBackend.resetRuntimeStats(ptr),
            .CUDA => |ptr| CUDABackend.resetRuntimeStats(ptr),
            .ROCm => |ptr| ROCmBackend.resetRuntimeStats(ptr),
        }
    }

    pub fn beginBatch(self: BackendInstance) !void {
        switch (self) {
            .CPU => |ptr| try CPUBackend.beginBatch(ptr),
            .Metal => |ptr| try MetalBackend.beginBatch(ptr),
            .CUDA => |ptr| try CUDABackend.beginBatch(ptr),
            .ROCm => |ptr| try ROCmBackend.beginBatch(ptr),
        }
    }

    pub fn endBatch(self: BackendInstance) !void {
        switch (self) {
            .CPU => |ptr| try CPUBackend.endBatch(ptr),
            .Metal => |ptr| try MetalBackend.endBatch(ptr),
            .CUDA => |ptr| try CUDABackend.endBatch(ptr),
            .ROCm => |ptr| try ROCmBackend.endBatch(ptr),
        }
    }

    pub fn synchronize(self: BackendInstance) !void {
        switch (self) {
            .CPU => |ptr| try CPUBackend.synchronize(ptr),
            .Metal => |ptr| try MetalBackend.synchronize(ptr),
            .CUDA => |ptr| try CUDABackend.synchronize(ptr),
            .ROCm => |ptr| try ROCmBackend.synchronize(ptr),
        }
    }

    pub fn deinitMatrix(self: BackendInstance, matrix: *BackendMatrix) void {
        switch (self) {
            .CPU => |ptr| CPUBackend.deinitMatrix(ptr, matrix),
            .Metal => |ptr| MetalBackend.deinitMatrix(ptr, matrix),
            .CUDA => |ptr| CUDABackend.deinitMatrix(ptr, matrix),
            .ROCm => |ptr| ROCmBackend.deinitMatrix(ptr, matrix),
        }
    }

    pub fn copyMatrix(self: BackendInstance, source: *const BackendMatrix, allocator: std.mem.Allocator) !*BackendMatrix {
        const matrix = try switch (self) {
            .CPU => |ptr| CPUBackend.copyMatrix(ptr, source, allocator),
            .Metal => |ptr| MetalBackend.copyMatrix(ptr, source, allocator),
            .CUDA => |ptr| CUDABackend.copyMatrix(ptr, source, allocator),
            .ROCm => |ptr| ROCmBackend.copyMatrix(ptr, source, allocator),
        };
        return self.attachBackend(matrix);
    }

    pub fn getMatrixElement(self: BackendInstance, matrix: *const BackendMatrix, row: usize, col: usize) f64 {
        return switch (self) {
            .CPU => |ptr| CPUBackend.getMatrixElement(ptr, matrix, row, col),
            .Metal => |ptr| MetalBackend.getMatrixElement(ptr, matrix, row, col),
            .CUDA => |ptr| CUDABackend.getMatrixElement(ptr, matrix, row, col),
            .ROCm => |ptr| ROCmBackend.getMatrixElement(ptr, matrix, row, col),
        };
    }

    pub fn writeMatrixF32(self: BackendInstance, matrix: *BackendMatrix, values: []const f32) bool {
        return switch (self) {
            .CPU => |ptr| CPUBackend.writeMatrixF32(ptr, matrix, values),
            .Metal => |ptr| MetalBackend.writeMatrixF32(ptr, matrix, values),
            .CUDA => |ptr| CUDABackend.writeMatrixF32(ptr, matrix, values),
            .ROCm => |ptr| ROCmBackend.writeMatrixF32(ptr, matrix, values),
        };
    }

    pub fn readMatrixF32(self: BackendInstance, matrix: *const BackendMatrix, values: []f32) bool {
        return switch (self) {
            .CPU => |ptr| CPUBackend.readMatrixF32(ptr, matrix, values),
            .Metal => |ptr| MetalBackend.readMatrixF32(ptr, matrix, values),
            .CUDA => |ptr| CUDABackend.readMatrixF32(ptr, matrix, values),
            .ROCm => |ptr| ROCmBackend.readMatrixF32(ptr, matrix, values),
        };
    }

    pub fn setMatrixElement(self: BackendInstance, matrix: *BackendMatrix, row: usize, col: usize, value: f64) void {
        switch (self) {
            .CPU => |ptr| CPUBackend.setMatrixElement(ptr, matrix, row, col, value),
            .Metal => |ptr| MetalBackend.setMatrixElement(ptr, matrix, row, col, value),
            .CUDA => |ptr| CUDABackend.setMatrixElement(ptr, matrix, row, col, value),
            .ROCm => |ptr| ROCmBackend.setMatrixElement(ptr, matrix, row, col, value),
        }
    }

    pub fn fillMatrix(self: BackendInstance, matrix: *BackendMatrix, value: f64) void {
        switch (self) {
            .CPU => |ptr| CPUBackend.fillMatrix(ptr, matrix, value),
            .Metal => |ptr| MetalBackend.fillMatrix(ptr, matrix, value),
            .CUDA => |ptr| CUDABackend.fillMatrix(ptr, matrix, value),
            .ROCm => |ptr| ROCmBackend.fillMatrix(ptr, matrix, value),
        }
    }

    pub fn dotProduct(self: BackendInstance, a: *const BackendMatrix, b: *const BackendMatrix, allocator: std.mem.Allocator) !*BackendMatrix {
        const matrix = try switch (self) {
            .CPU => |ptr| CPUBackend.dotProduct(ptr, a, b, allocator),
            .Metal => |ptr| MetalBackend.dotProduct(ptr, a, b, allocator),
            .CUDA => |ptr| CUDABackend.dotProduct(ptr, a, b, allocator),
            .ROCm => |ptr| ROCmBackend.dotProduct(ptr, a, b, allocator),
        };
        return self.attachBackend(matrix);
    }

    pub fn batchedDotProduct(
        self: BackendInstance,
        a: *const BackendMatrix,
        b: *const BackendMatrix,
        allocator: std.mem.Allocator,
        batch: usize,
        a_rows: usize,
        a_cols: usize,
        b_rows: usize,
        b_cols: usize,
        transpose_a: bool,
        transpose_b: bool,
    ) !*BackendMatrix {
        const matrix = try switch (self) {
            .CPU => |ptr| CPUBackend.batchedDotProduct(ptr, a, b, allocator, batch, a_rows, a_cols, b_rows, b_cols, transpose_a, transpose_b),
            .Metal => |ptr| MetalBackend.batchedDotProduct(ptr, a, b, allocator, batch, a_rows, a_cols, b_rows, b_cols, transpose_a, transpose_b),
            .CUDA => |ptr| CUDABackend.batchedDotProduct(ptr, a, b, allocator, batch, a_rows, a_cols, b_rows, b_cols, transpose_a, transpose_b),
            .ROCm => |ptr| ROCmBackend.batchedDotProduct(ptr, a, b, allocator, batch, a_rows, a_cols, b_rows, b_cols, transpose_a, transpose_b),
        };
        return self.attachBackend(matrix);
    }

    pub fn add(self: BackendInstance, a: *const BackendMatrix, b: *const BackendMatrix, allocator: std.mem.Allocator) !*BackendMatrix {
        const matrix = try switch (self) {
            .CPU => |ptr| CPUBackend.add(ptr, a, b, allocator),
            .Metal => |ptr| MetalBackend.add(ptr, a, b, allocator),
            .CUDA => |ptr| CUDABackend.add(ptr, a, b, allocator),
            .ROCm => |ptr| ROCmBackend.add(ptr, a, b, allocator),
        };
        return self.attachBackend(matrix);
    }

    pub fn addRowBias(self: BackendInstance, matrix: *const BackendMatrix, bias: *const BackendMatrix, allocator: std.mem.Allocator) !*BackendMatrix {
        const result = try switch (self) {
            .CPU => |ptr| CPUBackend.addRowBias(ptr, matrix, bias, allocator),
            .Metal => |ptr| MetalBackend.addRowBias(ptr, matrix, bias, allocator),
            .CUDA => |ptr| CUDABackend.addRowBias(ptr, matrix, bias, allocator),
            .ROCm => |ptr| ROCmBackend.addRowBias(ptr, matrix, bias, allocator),
        };
        return self.attachBackend(result);
    }

    pub fn subtract(self: BackendInstance, a: *const BackendMatrix, b: *const BackendMatrix, allocator: std.mem.Allocator) !*BackendMatrix {
        const matrix = try switch (self) {
            .CPU => |ptr| CPUBackend.subtract(ptr, a, b, allocator),
            .Metal => |ptr| MetalBackend.subtract(ptr, a, b, allocator),
            .CUDA => |ptr| CUDABackend.subtract(ptr, a, b, allocator),
            .ROCm => |ptr| ROCmBackend.subtract(ptr, a, b, allocator),
        };
        return self.attachBackend(matrix);
    }

    pub fn elementWiseMultiply(self: BackendInstance, a: *const BackendMatrix, b: *const BackendMatrix, allocator: std.mem.Allocator) !*BackendMatrix {
        const matrix = try switch (self) {
            .CPU => |ptr| CPUBackend.elementWiseMultiply(ptr, a, b, allocator),
            .Metal => |ptr| MetalBackend.elementWiseMultiply(ptr, a, b, allocator),
            .CUDA => |ptr| CUDABackend.elementWiseMultiply(ptr, a, b, allocator),
            .ROCm => |ptr| ROCmBackend.elementWiseMultiply(ptr, a, b, allocator),
        };
        return self.attachBackend(matrix);
    }

    pub fn scale(self: BackendInstance, matrix: *const BackendMatrix, scalar: f64, allocator: std.mem.Allocator) !*BackendMatrix {
        const result = try switch (self) {
            .CPU => |ptr| CPUBackend.scale(ptr, matrix, scalar, allocator),
            .Metal => |ptr| MetalBackend.scale(ptr, matrix, scalar, allocator),
            .CUDA => |ptr| CUDABackend.scale(ptr, matrix, scalar, allocator),
            .ROCm => |ptr| ROCmBackend.scale(ptr, matrix, scalar, allocator),
        };
        return self.attachBackend(result);
    }

    pub fn optimizerUpdate(
        self: BackendInstance,
        parameter: *BackendMatrix,
        gradient: *const BackendMatrix,
        first_moment: *BackendMatrix,
        second_moment: *BackendMatrix,
        total_squares: *const BackendMatrix,
        config: backend_mod.OptimizerUpdateConfig,
    ) !void {
        return switch (self) {
            .CPU => |ptr| CPUBackend.optimizerUpdate(ptr, parameter, gradient, first_moment, second_moment, total_squares, config),
            .Metal => |ptr| MetalBackend.optimizerUpdate(ptr, parameter, gradient, first_moment, second_moment, total_squares, config),
            .CUDA => |ptr| CUDABackend.optimizerUpdate(ptr, parameter, gradient, first_moment, second_moment, total_squares, config),
            .ROCm => |ptr| ROCmBackend.optimizerUpdate(ptr, parameter, gradient, first_moment, second_moment, total_squares, config),
        };
    }

    pub fn sumRows(self: BackendInstance, matrix: *const BackendMatrix, allocator: std.mem.Allocator) !*BackendMatrix {
        const result = try switch (self) {
            .CPU => |ptr| CPUBackend.sumRows(ptr, matrix, allocator),
            .Metal => |ptr| MetalBackend.sumRows(ptr, matrix, allocator),
            .CUDA => |ptr| CUDABackend.sumRows(ptr, matrix, allocator),
            .ROCm => |ptr| ROCmBackend.sumRows(ptr, matrix, allocator),
        };
        return self.attachBackend(result);
    }

    pub fn transpose(self: BackendInstance, matrix: *const BackendMatrix, allocator: std.mem.Allocator) !*BackendMatrix {
        const result = try switch (self) {
            .CPU => |ptr| CPUBackend.transpose(ptr, matrix, allocator),
            .Metal => |ptr| MetalBackend.transpose(ptr, matrix, allocator),
            .CUDA => |ptr| CUDABackend.transpose(ptr, matrix, allocator),
            .ROCm => |ptr| ROCmBackend.transpose(ptr, matrix, allocator),
        };
        return self.attachBackend(result);
    }

    pub fn extractBatch(self: BackendInstance, matrix: *const BackendMatrix, start: usize, end: usize, allocator: std.mem.Allocator) !*BackendMatrix {
        const result = try switch (self) {
            .CPU => |ptr| CPUBackend.extractBatch(ptr, matrix, start, end, allocator),
            .Metal => |ptr| MetalBackend.extractBatch(ptr, matrix, start, end, allocator),
            .CUDA => |ptr| CUDABackend.extractBatch(ptr, matrix, start, end, allocator),
            .ROCm => |ptr| ROCmBackend.extractBatch(ptr, matrix, start, end, allocator),
        };
        return self.attachBackend(result);
    }

    pub fn randomize(self: BackendInstance, matrix: *BackendMatrix, min: f64, max: f64) void {
        switch (self) {
            .CPU => |ptr| CPUBackend.randomize(ptr, matrix, min, max),
            .Metal => |ptr| MetalBackend.randomize(ptr, matrix, min, max),
            .CUDA => |ptr| CUDABackend.randomize(ptr, matrix, min, max),
            .ROCm => |ptr| ROCmBackend.randomize(ptr, matrix, min, max),
        }
    }

    pub fn applyActivation(self: BackendInstance, matrix: *const BackendMatrix, activation_fn: *const fn (f64) f64, allocator: std.mem.Allocator) !*BackendMatrix {
        const result = try switch (self) {
            .CPU => |ptr| CPUBackend.applyActivation(ptr, matrix, activation_fn, allocator),
            .Metal => |ptr| MetalBackend.applyActivation(ptr, matrix, activation_fn, allocator),
            .CUDA => |ptr| CUDABackend.applyActivation(ptr, matrix, activation_fn, allocator),
            .ROCm => |ptr| ROCmBackend.applyActivation(ptr, matrix, activation_fn, allocator),
        };
        return self.attachBackend(result);
    }

    pub fn applySoftmax(self: BackendInstance, matrix: *const BackendMatrix, allocator: std.mem.Allocator) !*BackendMatrix {
        const result = try switch (self) {
            .CPU => |ptr| CPUBackend.applySoftmax(ptr, matrix, allocator),
            .Metal => |ptr| MetalBackend.applySoftmax(ptr, matrix, allocator),
            .CUDA => |ptr| CUDABackend.applySoftmax(ptr, matrix, allocator),
            .ROCm => |ptr| ROCmBackend.applySoftmax(ptr, matrix, allocator),
        };
        return self.attachBackend(result);
    }

    pub fn layerNorm(self: BackendInstance, matrix: *const BackendMatrix, gamma: *const BackendMatrix, beta: *const BackendMatrix, epsilon: f64, allocator: std.mem.Allocator) !*BackendMatrix {
        const result = try switch (self) {
            .CPU => |ptr| CPUBackend.layerNorm(ptr, matrix, gamma, beta, epsilon, allocator),
            .Metal => |ptr| MetalBackend.layerNorm(ptr, matrix, gamma, beta, epsilon, allocator),
            .CUDA => |ptr| CUDABackend.layerNorm(ptr, matrix, gamma, beta, epsilon, allocator),
            .ROCm => |ptr| ROCmBackend.layerNorm(ptr, matrix, gamma, beta, epsilon, allocator),
        };
        return self.attachBackend(result);
    }

    pub fn layerNormBackward(self: BackendInstance, input: *const BackendMatrix, gamma: *const BackendMatrix, output_gradient: *const BackendMatrix, epsilon: f64, allocator: std.mem.Allocator) !backend_mod.LayerNormGradients {
        var gradients = try switch (self) {
            .CPU => |ptr| CPUBackend.layerNormBackward(ptr, input, gamma, output_gradient, epsilon, allocator),
            .Metal => |ptr| MetalBackend.layerNormBackward(ptr, input, gamma, output_gradient, epsilon, allocator),
            .CUDA => |ptr| CUDABackend.layerNormBackward(ptr, input, gamma, output_gradient, epsilon, allocator),
            .ROCm => |ptr| ROCmBackend.layerNormBackward(ptr, input, gamma, output_gradient, epsilon, allocator),
        };
        gradients.input = self.attachBackend(gradients.input);
        gradients.gamma = self.attachBackend(gradients.gamma);
        gradients.beta = self.attachBackend(gradients.beta);
        return gradients;
    }

    pub fn causalSelfAttention(self: BackendInstance, query: *const BackendMatrix, key: *const BackendMatrix, value: *const BackendMatrix, heads: usize, allocator: std.mem.Allocator) !*BackendMatrix {
        const result = try switch (self) {
            .CPU => |ptr| CPUBackend.causalSelfAttention(ptr, query, key, value, heads, allocator),
            .Metal => |ptr| MetalBackend.causalSelfAttention(ptr, query, key, value, heads, allocator),
            .CUDA => |ptr| CUDABackend.causalSelfAttention(ptr, query, key, value, heads, allocator),
            .ROCm => |ptr| ROCmBackend.causalSelfAttention(ptr, query, key, value, heads, allocator),
        };
        return self.attachBackend(result);
    }

    pub fn causalSelfAttentionBackward(self: BackendInstance, query: *const BackendMatrix, key: *const BackendMatrix, value: *const BackendMatrix, output_gradient: *const BackendMatrix, heads: usize, allocator: std.mem.Allocator) !backend_mod.AttentionGradients {
        var gradients = try switch (self) {
            .CPU => |ptr| CPUBackend.causalSelfAttentionBackward(ptr, query, key, value, output_gradient, heads, allocator),
            .Metal => |ptr| MetalBackend.causalSelfAttentionBackward(ptr, query, key, value, output_gradient, heads, allocator),
            .CUDA => |ptr| CUDABackend.causalSelfAttentionBackward(ptr, query, key, value, output_gradient, heads, allocator),
            .ROCm => |ptr| ROCmBackend.causalSelfAttentionBackward(ptr, query, key, value, output_gradient, heads, allocator),
        };
        gradients.query = self.attachBackend(gradients.query);
        gradients.key = self.attachBackend(gradients.key);
        gradients.value = self.attachBackend(gradients.value);
        return gradients;
    }

    pub fn embeddingLookup(self: BackendInstance, table: *const BackendMatrix, indices: *const BackendMatrix, allocator: std.mem.Allocator) !*BackendMatrix {
        const result = try switch (self) {
            .CPU => |ptr| CPUBackend.embeddingLookup(ptr, table, indices, allocator),
            .Metal => |ptr| MetalBackend.embeddingLookup(ptr, table, indices, allocator),
            .CUDA => |ptr| CUDABackend.embeddingLookup(ptr, table, indices, allocator),
            .ROCm => |ptr| ROCmBackend.embeddingLookup(ptr, table, indices, allocator),
        };
        return self.attachBackend(result);
    }

    pub fn embeddingGradient(self: BackendInstance, indices: *const BackendMatrix, output_gradient: *const BackendMatrix, vocabulary_size: usize, allocator: std.mem.Allocator) !*BackendMatrix {
        const result = try switch (self) {
            .CPU => |ptr| CPUBackend.embeddingGradient(ptr, indices, output_gradient, vocabulary_size, allocator),
            .Metal => |ptr| MetalBackend.embeddingGradient(ptr, indices, output_gradient, vocabulary_size, allocator),
            .CUDA => |ptr| CUDABackend.embeddingGradient(ptr, indices, output_gradient, vocabulary_size, allocator),
            .ROCm => |ptr| ROCmBackend.embeddingGradient(ptr, indices, output_gradient, vocabulary_size, allocator),
        };
        return self.attachBackend(result);
    }

    pub fn cachedSelfAttention(self: BackendInstance, query: *const BackendMatrix, key: *const BackendMatrix, value: *const BackendMatrix, key_cache: *BackendMatrix, value_cache: *BackendMatrix, position: usize, heads: usize, allocator: std.mem.Allocator) !*BackendMatrix {
        const result = try switch (self) {
            .CPU => |ptr| CPUBackend.cachedSelfAttention(ptr, query, key, value, key_cache, value_cache, position, heads, allocator),
            .Metal => |ptr| MetalBackend.cachedSelfAttention(ptr, query, key, value, key_cache, value_cache, position, heads, allocator),
            .CUDA => |ptr| CUDABackend.cachedSelfAttention(ptr, query, key, value, key_cache, value_cache, position, heads, allocator),
            .ROCm => |ptr| ROCmBackend.cachedSelfAttention(ptr, query, key, value, key_cache, value_cache, position, heads, allocator),
        };
        return self.attachBackend(result);
    }

    pub fn applyGLU(self: BackendInstance, linear_part: *const BackendMatrix, gating_part: *const BackendMatrix, allocator: std.mem.Allocator) !*BackendMatrix {
        const result = try switch (self) {
            .CPU => |ptr| CPUBackend.applyGLU(ptr, linear_part, gating_part, allocator),
            .Metal => |ptr| MetalBackend.applyGLU(ptr, linear_part, gating_part, allocator),
            .CUDA => |ptr| CUDABackend.applyGLU(ptr, linear_part, gating_part, allocator),
            .ROCm => |ptr| ROCmBackend.applyGLU(ptr, linear_part, gating_part, allocator),
        };
        return self.attachBackend(result);
    }

    pub fn applySwiGLU(self: BackendInstance, linear_part: *const BackendMatrix, gating_part: *const BackendMatrix, allocator: std.mem.Allocator) !*BackendMatrix {
        const result = try switch (self) {
            .CPU => |ptr| CPUBackend.applySwiGLU(ptr, linear_part, gating_part, allocator),
            .Metal => |ptr| MetalBackend.applySwiGLU(ptr, linear_part, gating_part, allocator),
            .CUDA => |ptr| CUDABackend.applySwiGLU(ptr, linear_part, gating_part, allocator),
            .ROCm => |ptr| ROCmBackend.applySwiGLU(ptr, linear_part, gating_part, allocator),
        };
        return self.attachBackend(result);
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
            .CUDA => |ptr| @as(*cuda_backend_mod.CUDABackend, @ptrCast(@alignCast(ptr))).deinit(),
            .ROCm => |ptr| @as(*rocm_backend_mod.ROCmBackend, @ptrCast(@alignCast(ptr))).deinit(),
        }
    }
};

// Export backend creation functions that return *anyopaque or the union
pub const createCPUBackend = cpu_backend_mod.createCPUBackend;
pub const createMetalBackend = metal_backend_mod.createMetalBackend;
pub const createCUDABackend = cuda_backend_mod.createCUDABackend;
pub const createROCmBackend = rocm_backend_mod.createROCmBackend;
// Keep the generic createBackend that returns the union
pub const createBackend = backend_mod.createBackend;
