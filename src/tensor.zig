const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const backend_mod = @import("backend.zig");
const dimension_utils = @import("dimensions.zig");
const root = @import("root.zig");

const Allocator = std.mem.Allocator;
const BackendInstance = root.BackendInstance;
const BackendMatrix = backend_mod.Matrix;
const BackendType = backend_mod.BackendType;
const Activation = @import("activation.zig").Activation;

pub const max_rank: usize = 4;

/// Compute precision used by the device-oriented tensor runtime.
pub const DType = enum {
    f32,
};

/// Element-wise activations available to device-resident Tensor models.
pub const ActivationKind = enum {
    linear,
    relu,
    sigmoid,
    tanh,
    gelu,
};

pub const OptimizerUpdateKind = backend_mod.OptimizerUpdateKind;
pub const OptimizerUpdateConfig = backend_mod.OptimizerUpdateConfig;

/// User-facing backend selection policy.
pub const DevicePreference = enum {
    cpu,
    auto,
    metal,
    cuda,
    rocm,
};

pub const Shape = struct {
    rank: u3,
    dims: [max_rank]usize,
    len: usize,

    pub fn init(requested: []const usize) !Shape {
        if (requested.len == 0 or requested.len > max_rank) {
            return error.InvalidRank;
        }

        var dims = [_]usize{1} ** max_rank;
        var len: usize = 1;
        for (requested, 0..) |dim, index| {
            if (dim == 0) return error.InvalidDimension;
            len = std.math.mul(usize, len, dim) catch return error.ShapeOverflow;
            dims[index] = dim;
        }

        return .{
            .rank = @intCast(requested.len),
            .dims = dims,
            .len = len,
        };
    }

    pub fn slice(self: *const Shape) []const usize {
        return self.dims[0..self.rank];
    }

    pub fn eql(self: Shape, other: Shape) bool {
        return std.mem.eql(usize, self.dims[0..self.rank], other.dims[0..other.rank]);
    }

    fn matrixDimensions(self: Shape) struct { rows: usize, cols: usize } {
        const cols = self.dims[self.rank - 1];
        return .{ .rows = self.len / cols, .cols = cols };
    }
};

/// Owns one backend instance and enforces exact selection for explicit GPU
/// requests. `auto` is the only preference permitted to fall back to CPU.
pub const Device = struct {
    allocator: Allocator,
    instance: BackendInstance,
    preference: DevicePreference,

    pub fn init(allocator: Allocator, preference: DevicePreference) !Device {
        const requested: BackendType = switch (preference) {
            .cpu => .CPU,
            .metal => .Metal,
            .cuda => .CUDA,
            .rocm => .ROCm,
            .auto => autoBackendType(),
        };

        const instance = try backend_mod.createBackend(allocator, requested);
        if (preference != .auto and instance.getBackendType() != requested) {
            instance.deinit();
            return error.BackendUnavailable;
        }

        return .{
            .allocator = allocator,
            .instance = instance,
            .preference = preference,
        };
    }

    pub fn deinit(self: *Device) void {
        self.instance.deinit();
        self.* = undefined;
    }

    pub fn backendType(self: Device) BackendType {
        return self.instance.getBackendType();
    }

    pub fn runtimeStats(self: Device) backend_mod.RuntimeStats {
        return self.instance.runtimeStats();
    }

    pub fn resetRuntimeStats(self: *Device) void {
        self.instance.resetRuntimeStats();
    }

    pub fn createTensor(self: *Device, shape: []const usize) !Tensor {
        return Tensor.init(self.instance, self.allocator, shape);
    }

    fn autoBackendType() BackendType {
        if (builtin.os.tag == .linux and build_options.enable_cuda) return .CUDA;
        if (builtin.os.tag == .linux and build_options.enable_rocm) return .ROCm;
        if (builtin.os.tag == .macos and build_options.enable_metal) return .Metal;
        return .CPU;
    }
};

/// Rank-aware f32 tensor backed by the existing backend matrix allocation.
/// Rank 1-4 tensors are flattened into a rank-2 physical allocation while the
/// logical shape remains available to neural-network operators.
pub const Tensor = struct {
    matrix: *BackendMatrix,
    shape: Shape,
    dtype: DType = .f32,
    allocator: Allocator,

    pub fn init(
        backend: BackendInstance,
        allocator: Allocator,
        requested_shape: []const usize,
    ) !Tensor {
        const shape = try Shape.init(requested_shape);
        const matrix_dims = shape.matrixDimensions();
        const matrix = try BackendMatrix.init(
            backend,
            allocator,
            matrix_dims.rows,
            matrix_dims.cols,
        );
        return .{
            .matrix = matrix,
            .shape = shape,
            .allocator = allocator,
        };
    }

    pub fn fromF32(
        backend: BackendInstance,
        allocator: Allocator,
        requested_shape: []const usize,
        values: []const f32,
    ) !Tensor {
        var result = try Tensor.init(backend, allocator, requested_shape);
        errdefer result.deinit();
        try result.writeF32(values);
        return result;
    }

    pub fn deinit(self: *Tensor) void {
        self.matrix.deinit();
        self.* = undefined;
    }

    pub fn elementCount(self: Tensor) usize {
        return self.shape.len;
    }

    pub fn backendType(self: Tensor) BackendType {
        return self.matrix.backend.getBackendType();
    }

    pub fn writeF32(self: *Tensor, values: []const f32) !void {
        try self.matrix.writeF32(values);
    }

    pub fn readF32(self: Tensor, values: []f32) !void {
        try self.matrix.readF32(values);
    }

    /// Changes only logical shape metadata; storage and device contents stay
    /// untouched.
    pub fn reshape(self: *Tensor, requested_shape: []const usize) !void {
        const next = try Shape.init(requested_shape);
        if (next.len != self.shape.len) return error.DimensionMismatch;
        self.shape = next;
    }

    pub fn fill(self: *Tensor, value: f32) void {
        self.matrix.fill(@floatCast(value));
    }

    pub fn copy(self: Tensor) !Tensor {
        return .{
            .matrix = try self.matrix.copy(self.allocator),
            .shape = self.shape,
            .dtype = self.dtype,
            .allocator = self.allocator,
        };
    }
};

pub const ExecutionStats = struct {
    uploads: usize = 0,
    upload_bytes: usize = 0,
    readbacks: usize = 0,
    readback_bytes: usize = 0,
    kernels: usize = 0,
    synchronizations: usize = 0,
};

pub const LayerNormGradients = struct {
    input: Tensor,
    gamma: Tensor,
    beta: Tensor,

    pub fn deinit(self: *LayerNormGradients) void {
        self.input.deinit();
        self.gamma.deinit();
        self.beta.deinit();
        self.* = undefined;
    }
};

pub const AttentionGradients = struct {
    query: Tensor,
    key: Tensor,
    value: Tensor,

    pub fn deinit(self: *AttentionGradients) void {
        self.query.deinit();
        self.key.deinit();
        self.value.deinit();
        self.* = undefined;
    }
};

/// Owns both a dropout output and the inverted dropout mask required by the
/// backward pass. The mask is device-resident after one explicit upload.
pub const DropoutResult = struct {
    output: Tensor,
    mask: Tensor,

    pub fn deinit(self: *DropoutResult) void {
        self.output.deinit();
        self.mask.deinit();
        self.* = undefined;
    }
};

/// Sparse class targets with a token-level mask. The gradient is computed on
/// the selected device; `meanLoss` is an explicit reporting readback.
pub const MaskedSparseCrossEntropy = struct {
    probabilities: Tensor,
    gradient: Tensor,
    targets: []usize,
    token_mask: []f32,
    active_weight: f32,
    classes: usize,
    allocator: Allocator,

    pub fn meanLoss(self: MaskedSparseCrossEntropy, context: *ExecutionContext) !f32 {
        const values = try self.allocator.alloc(f32, self.probabilities.shape.len);
        defer self.allocator.free(values);
        try context.readback(self.probabilities, values);
        var total: f32 = 0;
        for (self.targets, self.token_mask, 0..) |target, weight, row| {
            if (weight == 0) continue;
            total -= weight * @log(@max(values[row * self.classes + target], 1.0e-12));
        }
        return total / self.active_weight;
    }

    pub fn deinit(self: *MaskedSparseCrossEntropy) void {
        self.probabilities.deinit();
        self.gradient.deinit();
        self.allocator.free(self.targets);
        self.allocator.free(self.token_mask);
        self.* = undefined;
    }
};

/// Explicit execution boundary for tensor construction, operations, readback,
/// and instrumentation. Backend command batching can be implemented behind
/// this API without another model-code migration.
pub const ExecutionContext = struct {
    device: *Device,
    stats: ExecutionStats = .{},
    batch_active: bool = false,

    pub fn init(device: *Device) ExecutionContext {
        return .{ .device = device };
    }

    fn requireOwned(self: ExecutionContext, tensors: []const Tensor) !void {
        for (tensors) |value| {
            if (!self.device.instance.sameInstance(value.matrix.backend)) {
                return error.BackendMismatch;
            }
        }
    }

    pub fn createTensor(self: *ExecutionContext, shape: []const usize) !Tensor {
        return self.device.createTensor(shape);
    }

    pub fn upload(
        self: *ExecutionContext,
        shape: []const usize,
        values: []const f32,
    ) !Tensor {
        var result = try self.device.createTensor(shape);
        errdefer result.deinit();
        try result.writeF32(values);
        self.stats.uploads += 1;
        self.stats.upload_bytes += values.len * @sizeOf(f32);
        return result;
    }

    pub fn readback(self: *ExecutionContext, tensor: Tensor, values: []f32) !void {
        try self.requireOwned(&.{tensor});
        try tensor.readF32(values);
        self.stats.readbacks += 1;
        self.stats.readback_bytes += values.len * @sizeOf(f32);
    }

    pub fn matmul(self: *ExecutionContext, a: Tensor, b: Tensor) !Tensor {
        try self.requireOwned(&.{ a, b });
        if (a.shape.rank != 2 or b.shape.rank != 2) return error.InvalidRank;
        if (a.shape.dims[1] != b.shape.dims[0]) return error.DimensionMismatch;

        const matrix = try a.matrix.dotProduct(b.matrix, self.device.allocator);
        self.stats.kernels += 1;
        return .{
            .matrix = matrix,
            .shape = try Shape.init(&.{ a.shape.dims[0], b.shape.dims[1] }),
            .allocator = self.device.allocator,
        };
    }

    /// Performs `[batch, rows, inner] x [batch, inner, cols]` matrix
    /// multiplication. Transpose flags affect only each logical matrix, not
    /// the batch dimension.
    pub fn batchedMatmul(self: *ExecutionContext, a: Tensor, b: Tensor, transpose_a: bool, transpose_b: bool) !Tensor {
        try self.requireOwned(&.{ a, b });
        if (a.shape.rank != 3 or b.shape.rank != 3) return error.InvalidRank;
        if (a.shape.dims[0] != b.shape.dims[0]) return error.DimensionMismatch;
        const a_rows = a.shape.dims[1];
        const a_cols = a.shape.dims[2];
        const b_rows = b.shape.dims[1];
        const b_cols = b.shape.dims[2];
        const inner_a = if (transpose_a) a_rows else a_cols;
        const inner_b = if (transpose_b) b_cols else b_rows;
        if (inner_a != inner_b) return error.DimensionMismatch;
        const output_rows = if (transpose_a) a_cols else a_rows;
        const output_cols = if (transpose_b) b_rows else b_cols;

        const matrix = try a.matrix.batchedDotProduct(
            b.matrix,
            self.device.allocator,
            a.shape.dims[0],
            a_rows,
            a_cols,
            b_rows,
            b_cols,
            transpose_a,
            transpose_b,
        );
        self.stats.kernels += 1;
        return .{
            .matrix = matrix,
            .shape = try Shape.init(&.{ a.shape.dims[0], output_rows, output_cols }),
            .allocator = self.device.allocator,
        };
    }

    /// Splits token-major channels into head-major batches:
    /// `[batch, tokens, heads, width] -> [batch * heads, tokens, width]`.
    pub fn splitHeads(self: *ExecutionContext, input: Tensor) !Tensor {
        try self.requireOwned(&.{input});
        if (input.shape.rank != 4) return error.InvalidRank;
        const batch = input.shape.dims[0];
        const tokens = input.shape.dims[1];
        const heads = input.shape.dims[2];
        const width = input.shape.dims[3];
        const matrix = try input.matrix.permuteBatchHeads(self.device.allocator, batch, tokens, heads, width, true);
        self.stats.kernels += 1;
        return .{ .matrix = matrix, .shape = try Shape.init(&.{ batch * heads, tokens, width }), .allocator = self.device.allocator };
    }

    /// Merges head-major batches back into token-major channels:
    /// `[batch * heads, tokens, width] -> [batch, tokens, heads, width]`.
    pub fn mergeHeads(self: *ExecutionContext, input: Tensor, batch: usize, heads: usize) !Tensor {
        try self.requireOwned(&.{input});
        if (input.shape.rank != 3 or batch == 0 or heads == 0 or
            !dimension_utils.matches(input.shape.dims[0], &.{ batch, heads }))
        {
            return error.DimensionMismatch;
        }
        const tokens = input.shape.dims[1];
        const width = input.shape.dims[2];
        const matrix = try input.matrix.permuteBatchHeads(self.device.allocator, batch, tokens, heads, width, false);
        self.stats.kernels += 1;
        return .{ .matrix = matrix, .shape = try Shape.init(&.{ batch, tokens, heads, width }), .allocator = self.device.allocator };
    }

    pub fn add(self: *ExecutionContext, a: Tensor, b: Tensor) !Tensor {
        try self.requireOwned(&.{ a, b });
        if (!a.shape.eql(b.shape)) return error.DimensionMismatch;

        const matrix = try a.matrix.add(b.matrix, self.device.allocator);
        self.stats.kernels += 1;
        return .{
            .matrix = matrix,
            .shape = a.shape,
            .allocator = self.device.allocator,
        };
    }

    pub fn subtract(self: *ExecutionContext, a: Tensor, b: Tensor) !Tensor {
        try self.requireOwned(&.{ a, b });
        if (!a.shape.eql(b.shape)) return error.DimensionMismatch;

        const matrix = try a.matrix.subtract(b.matrix, self.device.allocator);
        self.stats.kernels += 1;
        return .{
            .matrix = matrix,
            .shape = a.shape,
            .allocator = self.device.allocator,
        };
    }

    pub fn multiply(self: *ExecutionContext, a: Tensor, b: Tensor) !Tensor {
        try self.requireOwned(&.{ a, b });
        if (!a.shape.eql(b.shape)) return error.DimensionMismatch;

        const matrix = try a.matrix.elementWiseMultiply(b.matrix, self.device.allocator);
        self.stats.kernels += 1;
        return .{
            .matrix = matrix,
            .shape = a.shape,
            .allocator = self.device.allocator,
        };
    }

    pub fn scale(self: *ExecutionContext, input: Tensor, scalar: f32) !Tensor {
        try self.requireOwned(&.{input});
        const matrix = try input.matrix.scale(scalar, self.device.allocator);
        self.stats.kernels += 1;
        return .{
            .matrix = matrix,
            .shape = input.shape,
            .allocator = self.device.allocator,
        };
    }

    pub fn transpose(self: *ExecutionContext, input: Tensor) !Tensor {
        try self.requireOwned(&.{input});
        if (input.shape.rank != 2) return error.InvalidRank;
        const matrix = try input.matrix.transpose(self.device.allocator);
        self.stats.kernels += 1;
        return .{
            .matrix = matrix,
            .shape = try Shape.init(&.{ input.shape.dims[1], input.shape.dims[0] }),
            .allocator = self.device.allocator,
        };
    }

    pub fn sumRows(self: *ExecutionContext, input: Tensor) !Tensor {
        try self.requireOwned(&.{input});
        if (input.shape.rank != 2) return error.InvalidRank;
        const matrix = try input.matrix.sumRows(self.device.allocator);
        self.stats.kernels += 1;
        return .{
            .matrix = matrix,
            .shape = try Shape.init(&.{ 1, input.shape.dims[1] }),
            .allocator = self.device.allocator,
        };
    }

    /// Returns a device-resident `[1, 1]` Tensor containing the sum of every
    /// squared element, regardless of the input's logical rank.
    pub fn sumSquares(self: *ExecutionContext, input: Tensor) !Tensor {
        var matrix_view = input;
        matrix_view.shape = try Shape.init(&.{ input.matrix.rows, input.matrix.cols });

        var squared = try self.multiply(matrix_view, matrix_view);
        defer squared.deinit();
        var column_sums = try self.sumRows(squared);
        defer column_sums.deinit();
        var column_vector = try self.transpose(column_sums);
        defer column_vector.deinit();
        return self.sumRows(column_vector);
    }

    pub fn optimizerUpdate(
        self: *ExecutionContext,
        parameter: *Tensor,
        gradient: Tensor,
        first_moment: *Tensor,
        second_moment: *Tensor,
        total_squares: Tensor,
        config: OptimizerUpdateConfig,
    ) !void {
        try self.requireOwned(&.{ parameter.*, gradient, first_moment.*, second_moment.*, total_squares });
        if (!parameter.shape.eql(gradient.shape) or
            !parameter.shape.eql(first_moment.shape) or
            !parameter.shape.eql(second_moment.shape) or
            total_squares.shape.rank != 2 or total_squares.shape.dims[0] != 1 or
            total_squares.shape.dims[1] != 1)
        {
            return error.DimensionMismatch;
        }
        try parameter.matrix.optimizerUpdate(
            gradient.matrix,
            first_moment.matrix,
            second_moment.matrix,
            total_squares.matrix,
            config,
        );
        self.stats.kernels += 1;
    }

    pub fn addRowBias(self: *ExecutionContext, input: Tensor, bias: Tensor) !Tensor {
        try self.requireOwned(&.{ input, bias });
        if (input.shape.rank != 2 or bias.shape.rank != 2) return error.InvalidRank;
        if (bias.shape.dims[0] != 1 or bias.shape.dims[1] != input.shape.dims[1]) return error.DimensionMismatch;

        const matrix = try input.matrix.addRowBias(bias.matrix, self.device.allocator);
        self.stats.kernels += 1;
        return .{
            .matrix = matrix,
            .shape = input.shape,
            .allocator = self.device.allocator,
        };
    }

    pub fn linear(self: *ExecutionContext, input: Tensor, weights: Tensor, bias: Tensor) !Tensor {
        var projected = try self.matmul(input, weights);
        defer projected.deinit();
        return self.addRowBias(projected, bias);
    }

    pub fn activate(self: *ExecutionContext, input: Tensor, kind: ActivationKind) !Tensor {
        try self.requireOwned(&.{input});
        const function = activationFunction(kind);
        const matrix = try input.matrix.applyActivation(function, self.device.allocator);
        self.stats.kernels += 1;
        return .{
            .matrix = matrix,
            .shape = input.shape,
            .allocator = self.device.allocator,
        };
    }

    pub fn activationDerivative(self: *ExecutionContext, input: Tensor, kind: ActivationKind) !Tensor {
        try self.requireOwned(&.{input});
        const function = activationDerivativeFunction(kind);
        const matrix = try input.matrix.applyActivation(function, self.device.allocator);
        self.stats.kernels += 1;
        return .{
            .matrix = matrix,
            .shape = input.shape,
            .allocator = self.device.allocator,
        };
    }

    pub fn gelu(self: *ExecutionContext, input: Tensor) !Tensor {
        return self.activate(input, .gelu);
    }

    pub fn geluDerivative(self: *ExecutionContext, input: Tensor) !Tensor {
        return self.activationDerivative(input, .gelu);
    }

    pub fn layerNorm(self: *ExecutionContext, input: Tensor, gamma: Tensor, beta: Tensor, epsilon: f32) !Tensor {
        try self.requireOwned(&.{ input, gamma, beta });
        if (input.shape.rank != 2 or gamma.shape.rank != 2 or beta.shape.rank != 2) return error.InvalidRank;
        if (gamma.shape.dims[0] != 1 or beta.shape.dims[0] != 1 or
            gamma.shape.dims[1] != input.shape.dims[1] or beta.shape.dims[1] != input.shape.dims[1])
        {
            return error.DimensionMismatch;
        }
        const matrix = try input.matrix.layerNorm(gamma.matrix, beta.matrix, epsilon, self.device.allocator);
        self.stats.kernels += 1;
        return .{
            .matrix = matrix,
            .shape = input.shape,
            .allocator = self.device.allocator,
        };
    }

    pub fn layerNormBackward(self: *ExecutionContext, input: Tensor, gamma: Tensor, output_gradient: Tensor, epsilon: f32) !LayerNormGradients {
        try self.requireOwned(&.{ input, gamma, output_gradient });
        if (input.shape.rank != 2 or gamma.shape.rank != 2 or output_gradient.shape.rank != 2) return error.InvalidRank;
        if (!input.shape.eql(output_gradient.shape) or gamma.shape.dims[0] != 1 or gamma.shape.dims[1] != input.shape.dims[1]) {
            return error.DimensionMismatch;
        }
        const gradients = try input.matrix.layerNormBackward(gamma.matrix, output_gradient.matrix, epsilon, self.device.allocator);
        self.stats.kernels += 1;
        return .{
            .input = .{ .matrix = gradients.input, .shape = input.shape, .allocator = self.device.allocator },
            .gamma = .{ .matrix = gradients.gamma, .shape = gamma.shape, .allocator = self.device.allocator },
            .beta = .{ .matrix = gradients.beta, .shape = gamma.shape, .allocator = self.device.allocator },
        };
    }

    pub fn causalSelfAttention(self: *ExecutionContext, query: Tensor, key: Tensor, value: Tensor, heads: usize) !Tensor {
        try self.requireOwned(&.{ query, key, value });
        if (query.shape.rank != 2 or key.shape.rank != 2 or value.shape.rank != 2) return error.InvalidRank;
        if (!query.shape.eql(key.shape) or !query.shape.eql(value.shape) or
            heads == 0 or query.shape.dims[1] % heads != 0)
        {
            return error.DimensionMismatch;
        }
        const matrix = try query.matrix.causalSelfAttention(key.matrix, value.matrix, heads, self.device.allocator);
        self.stats.kernels += 1;
        return .{
            .matrix = matrix,
            .shape = query.shape,
            .allocator = self.device.allocator,
        };
    }

    pub fn causalSelfAttentionBackward(self: *ExecutionContext, query: Tensor, key: Tensor, value: Tensor, output_gradient: Tensor, heads: usize) !AttentionGradients {
        try self.requireOwned(&.{ query, key, value, output_gradient });
        if (query.shape.rank != 2 or key.shape.rank != 2 or value.shape.rank != 2 or output_gradient.shape.rank != 2) return error.InvalidRank;
        if (!query.shape.eql(key.shape) or !query.shape.eql(value.shape) or !query.shape.eql(output_gradient.shape) or
            heads == 0 or query.shape.dims[1] % heads != 0)
        {
            return error.DimensionMismatch;
        }
        const gradients = try query.matrix.causalSelfAttentionBackward(key.matrix, value.matrix, output_gradient.matrix, heads, self.device.allocator);
        self.stats.kernels += 2;
        return .{
            .query = .{ .matrix = gradients.query, .shape = query.shape, .allocator = self.device.allocator },
            .key = .{ .matrix = gradients.key, .shape = key.shape, .allocator = self.device.allocator },
            .value = .{ .matrix = gradients.value, .shape = value.shape, .allocator = self.device.allocator },
        };
    }

    pub fn embedding(self: *ExecutionContext, table: Tensor, indices: Tensor) !Tensor {
        try self.requireOwned(&.{ table, indices });
        if (table.shape.rank != 2 or indices.shape.rank != 2 or indices.shape.dims[1] != 1) return error.InvalidRank;
        const matrix = try table.matrix.embeddingLookup(indices.matrix, self.device.allocator);
        self.stats.kernels += 1;
        return .{
            .matrix = matrix,
            .shape = try Shape.init(&.{ indices.shape.dims[0], table.shape.dims[1] }),
            .allocator = self.device.allocator,
        };
    }

    pub fn embeddingBackward(self: *ExecutionContext, indices: Tensor, output_gradient: Tensor, vocabulary_size: usize) !Tensor {
        try self.requireOwned(&.{ indices, output_gradient });
        if (indices.shape.rank != 2 or output_gradient.shape.rank != 2 or indices.shape.dims[1] != 1 or
            indices.shape.dims[0] != output_gradient.shape.dims[0])
        {
            return error.DimensionMismatch;
        }
        const matrix = try output_gradient.matrix.embeddingGradient(indices.matrix, vocabulary_size, self.device.allocator);
        self.stats.kernels += 1;
        return .{
            .matrix = matrix,
            .shape = try Shape.init(&.{ vocabulary_size, output_gradient.shape.dims[1] }),
            .allocator = self.device.allocator,
        };
    }

    pub fn cachedSelfAttention(self: *ExecutionContext, query: Tensor, key: Tensor, value: Tensor, key_cache: *Tensor, value_cache: *Tensor, position: usize, heads: usize) !Tensor {
        try self.requireOwned(&.{ query, key, value, key_cache.*, value_cache.* });
        if (query.shape.rank != 2 or !query.shape.eql(key.shape) or !query.shape.eql(value.shape) or query.shape.dims[0] != 1 or
            key_cache.shape.rank != 2 or !key_cache.shape.eql(value_cache.shape) or key_cache.shape.dims[1] != query.shape.dims[1] or
            position >= key_cache.shape.dims[0] or heads == 0 or query.shape.dims[1] % heads != 0)
        {
            return error.DimensionMismatch;
        }
        const matrix = try query.matrix.cachedSelfAttention(key.matrix, value.matrix, key_cache.matrix, value_cache.matrix, position, heads, self.device.allocator);
        self.stats.kernels += 1;
        return .{ .matrix = matrix, .shape = query.shape, .allocator = self.device.allocator };
    }

    pub fn softmax(self: *ExecutionContext, input: Tensor) !Tensor {
        try self.requireOwned(&.{input});
        if (input.shape.rank != 2) return error.InvalidRank;
        const matrix = try input.matrix.applySoftmax(self.device.allocator);
        self.stats.kernels += 1;
        return .{
            .matrix = matrix,
            .shape = input.shape,
            .allocator = self.device.allocator,
        };
    }

    /// Row-wise softmax over the final dimension. Zero mask entries receive
    /// exactly zero probability, including rows where every entry is masked.
    pub fn maskedSoftmax(self: *ExecutionContext, input: Tensor, mask: Tensor) !Tensor {
        try self.requireOwned(&.{ input, mask });
        if (input.shape.rank < 2 or input.shape.rank > 3) return error.InvalidRank;
        if (!input.shape.eql(mask.shape)) return error.DimensionMismatch;

        var ones = try self.createTensor(input.shape.slice());
        defer ones.deinit();
        ones.fill(1);
        var mask_offset = try self.subtract(mask, ones);
        defer mask_offset.deinit();
        var penalty = try self.scale(mask_offset, 1.0e9);
        defer penalty.deinit();
        var masked_scores = try self.add(input, penalty);
        defer masked_scores.deinit();
        var matrix_view = masked_scores;
        matrix_view.shape = try Shape.init(&.{ input.shape.len / input.shape.dims[input.shape.rank - 1], input.shape.dims[input.shape.rank - 1] });
        var probabilities = try self.softmax(matrix_view);
        errdefer probabilities.deinit();
        probabilities.shape = input.shape;
        const result = try self.multiply(probabilities, mask);
        probabilities.deinit();
        return result;
    }

    /// Jacobian-vector product for `maskedSoftmax`, evaluated entirely with
    /// device tensor operations.
    pub fn maskedSoftmaxBackward(self: *ExecutionContext, probabilities: Tensor, output_gradient: Tensor, mask: Tensor) !Tensor {
        try self.requireOwned(&.{ probabilities, output_gradient, mask });
        if (probabilities.shape.rank < 2 or probabilities.shape.rank > 3) return error.InvalidRank;
        if (!probabilities.shape.eql(output_gradient.shape) or !probabilities.shape.eql(mask.shape)) return error.DimensionMismatch;
        const cols = probabilities.shape.dims[probabilities.shape.rank - 1];
        const rows = probabilities.shape.len / cols;
        var probabilities_view = probabilities;
        probabilities_view.shape = try Shape.init(&.{ rows, cols });
        var gradient_view = output_gradient;
        gradient_view.shape = probabilities_view.shape;

        var weighted = try self.multiply(probabilities_view, gradient_view);
        defer weighted.deinit();
        var weighted_t = try self.transpose(weighted);
        defer weighted_t.deinit();
        var row_sums_t = try self.sumRows(weighted_t);
        defer row_sums_t.deinit();
        var row_sums = try self.transpose(row_sums_t);
        defer row_sums.deinit();
        var ones = try self.createTensor(&.{ 1, cols });
        defer ones.deinit();
        ones.fill(1);
        var broadcast_sums = try self.matmul(row_sums, ones);
        defer broadcast_sums.deinit();
        var centered = try self.subtract(gradient_view, broadcast_sums);
        defer centered.deinit();
        var jacobian_product = try self.multiply(probabilities_view, centered);
        defer jacobian_product.deinit();
        var mask_view = mask;
        mask_view.shape = probabilities_view.shape;
        var result = try self.multiply(jacobian_product, mask_view);
        result.shape = probabilities.shape;
        return result;
    }

    /// Applies inverted dropout. Supplying a seed makes the mask reproducible,
    /// which keeps learning experiments and gradient tests deterministic.
    pub fn dropout(self: *ExecutionContext, input: Tensor, probability: f32, seed: u64, training: bool) !DropoutResult {
        try self.requireOwned(&.{input});
        if (probability < 0 or probability >= 1) return error.InvalidProbability;
        const values = try self.device.allocator.alloc(f32, input.shape.len);
        defer self.device.allocator.free(values);
        if (!training or probability == 0) {
            @memset(values, 1);
        } else {
            var prng = std.Random.DefaultPrng.init(seed);
            const random = prng.random();
            const inverse_keep = 1.0 / (1.0 - probability);
            for (values) |*value| {
                value.* = if (random.float(f32) < probability) 0 else inverse_keep;
            }
        }
        var mask = try self.upload(input.shape.slice(), values);
        errdefer mask.deinit();
        const output = try self.multiply(input, mask);
        return .{ .output = output, .mask = mask };
    }

    pub fn dropoutBackward(self: *ExecutionContext, output_gradient: Tensor, mask: Tensor) !Tensor {
        return self.multiply(output_gradient, mask);
    }

    /// Computes mean sparse cross-entropy over active tokens. Sparse target
    /// indices are expanded once at the upload boundary; probabilities and the
    /// training gradient remain device-resident.
    pub fn maskedSparseCrossEntropy(self: *ExecutionContext, logits: Tensor, targets: []const usize, token_mask: []const f32) !MaskedSparseCrossEntropy {
        try self.requireOwned(&.{logits});
        if (logits.shape.rank < 2 or logits.shape.rank > 3) return error.InvalidRank;
        const classes = logits.shape.dims[logits.shape.rank - 1];
        const rows = logits.shape.len / classes;
        if (targets.len != rows or token_mask.len != rows) return error.DimensionMismatch;

        var active_weight: f32 = 0;
        const dense_targets = try self.device.allocator.alloc(f32, logits.shape.len);
        defer self.device.allocator.free(dense_targets);
        @memset(dense_targets, 0);
        const dense_mask = try self.device.allocator.alloc(f32, logits.shape.len);
        defer self.device.allocator.free(dense_mask);
        for (targets, token_mask, 0..) |target, weight, row| {
            if (target >= classes or weight < 0) return error.InvalidTarget;
            dense_targets[row * classes + target] = 1;
            @memset(dense_mask[row * classes .. (row + 1) * classes], weight);
            active_weight += weight;
        }
        if (active_weight <= 0) return error.EmptyMask;

        const owned_targets = try self.device.allocator.dupe(usize, targets);
        errdefer self.device.allocator.free(owned_targets);
        const owned_mask = try self.device.allocator.dupe(f32, token_mask);
        errdefer self.device.allocator.free(owned_mask);
        var targets_tensor = try self.upload(logits.shape.slice(), dense_targets);
        defer targets_tensor.deinit();
        var mask_tensor = try self.upload(logits.shape.slice(), dense_mask);
        defer mask_tensor.deinit();
        var logits_view = logits;
        logits_view.shape = try Shape.init(&.{ rows, classes });
        var probabilities = try self.softmax(logits_view);
        errdefer probabilities.deinit();
        probabilities.shape = logits.shape;
        var difference = try self.subtract(probabilities, targets_tensor);
        defer difference.deinit();
        var masked_gradient = try self.multiply(difference, mask_tensor);
        defer masked_gradient.deinit();
        const gradient = try self.scale(masked_gradient, 1.0 / active_weight);
        return .{
            .probabilities = probabilities,
            .gradient = gradient,
            .targets = owned_targets,
            .token_mask = owned_mask,
            .active_weight = active_weight,
            .classes = classes,
            .allocator = self.device.allocator,
        };
    }

    pub fn beginBatch(self: *ExecutionContext) !void {
        if (self.batch_active) return error.BatchAlreadyActive;
        try self.device.instance.beginBatch();
        self.batch_active = true;
    }

    pub fn endBatch(self: *ExecutionContext) !void {
        if (!self.batch_active) return error.NoActiveBatch;
        self.device.instance.endBatch() catch |err| {
            self.batch_active = false;
            return err;
        };
        self.batch_active = false;
        self.stats.synchronizations += 1;
    }

    pub fn synchronize(self: *ExecutionContext) !void {
        try self.device.instance.synchronize();
        self.stats.synchronizations += 1;
    }

    pub fn resetStats(self: *ExecutionContext) void {
        self.stats = .{};
        self.device.resetRuntimeStats();
    }

    pub fn backendStats(self: ExecutionContext) backend_mod.RuntimeStats {
        return self.device.runtimeStats();
    }
};

fn activationFunction(kind: ActivationKind) *const fn (f64) f64 {
    return switch (kind) {
        .linear => Activation.linear,
        .relu => Activation.relu,
        .sigmoid => Activation.sigmoid,
        .tanh => Activation.tanh,
        .gelu => Activation.gelu,
    };
}

fn activationDerivativeFunction(kind: ActivationKind) *const fn (f64) f64 {
    return switch (kind) {
        .linear => Activation.linear_derivative,
        .relu => Activation.relu_derivative,
        .sigmoid => Activation.sigmoid_derivative,
        .tanh => Activation.tanh_derivative,
        .gelu => Activation.gelu_derivative,
    };
}

fn referenceMaskedSoftmaxObjective(scores: []const f32, mask: []const f32, upstream: []const f32) f32 {
    var max_score = -std.math.inf(f32);
    for (scores, mask) |score, active| if (active != 0) {
        max_score = @max(max_score, score);
    };
    var denominator: f32 = 0;
    for (scores, mask) |score, active| if (active != 0) {
        denominator += @exp(score - max_score);
    };
    if (denominator == 0) return 0;
    var objective: f32 = 0;
    for (scores, mask, upstream) |score, active, gradient| if (active != 0) {
        objective += @exp(score - max_score) / denominator * gradient;
    };
    return objective;
}

fn referenceLayerNormObjective(input: []const f32, gamma: []const f32, beta: []const f32, output_gradient: []const f32, rows: usize, cols: usize, epsilon: f32) f32 {
    var objective: f32 = 0;
    for (0..rows) |row| {
        const offset = row * cols;
        var mean: f32 = 0;
        for (0..cols) |col| mean += input[offset + col];
        mean /= @as(f32, @floatFromInt(cols));
        var variance: f32 = 0;
        for (0..cols) |col| {
            const centered = input[offset + col] - mean;
            variance += centered * centered;
        }
        variance /= @as(f32, @floatFromInt(cols));
        const inverse_std = 1.0 / @sqrt(variance + epsilon);
        for (0..cols) |col| {
            const normalized = (input[offset + col] - mean) * inverse_std;
            objective += (normalized * gamma[col] + beta[col]) * output_gradient[offset + col];
        }
    }
    return objective;
}

fn referenceAttentionObjective(query: []const f32, key: []const f32, value: []const f32, output_gradient: []const f32, tokens: usize, channels: usize, heads: usize) f32 {
    std.debug.assert(tokens <= 16);
    var scores: [16]f32 = undefined;
    const head_width = channels / heads;
    const attention_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_width)));
    var objective: f32 = 0;
    for (0..heads) |head| {
        const channel_offset = head * head_width;
        for (0..tokens) |query_position| {
            var max_score = -std.math.inf(f32);
            for (0..query_position + 1) |key_position| {
                var score: f32 = 0;
                for (0..head_width) |channel| {
                    score += query[query_position * channels + channel_offset + channel] *
                        key[key_position * channels + channel_offset + channel];
                }
                scores[key_position] = score * attention_scale;
                max_score = @max(max_score, scores[key_position]);
            }
            var denominator: f32 = 0;
            for (0..query_position + 1) |key_position| {
                scores[key_position] = @exp(scores[key_position] - max_score);
                denominator += scores[key_position];
            }
            for (0..head_width) |channel| {
                var output: f32 = 0;
                for (0..query_position + 1) |key_position| {
                    output += scores[key_position] / denominator * value[key_position * channels + channel_offset + channel];
                }
                objective += output * output_gradient[query_position * channels + channel_offset + channel];
            }
        }
    }
    return objective;
}

test "shape validates rank dimensions and overflow" {
    const testing = std.testing;

    const shape = try Shape.init(&.{ 2, 3, 4 });
    try testing.expectEqual(@as(u3, 3), shape.rank);
    try testing.expectEqual(@as(usize, 24), shape.len);
    try testing.expectEqualSlices(usize, &.{ 2, 3, 4 }, shape.slice());
    try testing.expectError(error.InvalidRank, Shape.init(&.{}));
    try testing.expectError(error.InvalidDimension, Shape.init(&.{ 2, 0 }));
    try testing.expectError(error.ShapeOverflow, Shape.init(&.{ std.math.maxInt(usize), 2 }));
}

test "shape slice borrows the stored dimensions" {
    const testing = std.testing;
    var shape = try Shape.init(&.{ 2, 3, 4 });
    const dimensions = shape.slice();
    try testing.expectEqual(@intFromPtr(&shape.dims[0]), @intFromPtr(dimensions.ptr));
    try testing.expectEqualSlices(usize, &.{ 2, 3, 4 }, dimensions);
}

test "execution context tracks tensor transfers and kernels" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var device = try Device.init(allocator, .cpu);
    defer device.deinit();
    var context = ExecutionContext.init(&device);

    var a = try context.upload(&.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer a.deinit();
    var b = try context.upload(&.{ 3, 2 }, &.{ 7, 8, 9, 10, 11, 12 });
    defer b.deinit();
    var product = try context.matmul(a, b);
    defer product.deinit();

    var actual: [4]f32 = undefined;
    try context.readback(product, &actual);
    try testing.expectEqualSlices(f32, &.{ 58, 64, 139, 154 }, &actual);
    try testing.expectEqual(@as(usize, 2), context.stats.uploads);
    try testing.expectEqual(@as(usize, 1), context.stats.readbacks);
    try testing.expectEqual(@as(usize, 1), context.stats.kernels);
}

test "execution context rejects a separate backend instance" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var first_device = try Device.init(allocator, .cpu);
    defer first_device.deinit();
    var second_device = try Device.init(allocator, .cpu);
    defer second_device.deinit();
    var first_context = ExecutionContext.init(&first_device);
    var second_context = ExecutionContext.init(&second_device);

    try testing.expect(first_device.instance.sameInstance(first_device.instance));
    try testing.expect(!first_device.instance.sameInstance(second_device.instance));

    var local = try first_context.upload(&.{ 1, 2 }, &.{ 1, 2 });
    defer local.deinit();
    var foreign = try second_context.upload(&.{ 1, 2 }, &.{ 3, 4 });
    defer foreign.deinit();

    try testing.expectError(error.BackendMismatch, first_context.add(local, foreign));
    try testing.expectError(error.BackendMismatch, first_context.scale(foreign, 2));
    var values: [2]f32 = undefined;
    try testing.expectError(error.BackendMismatch, first_context.readback(foreign, &values));
    try testing.expectError(
        error.BackendMismatch,
        local.matrix.add(foreign.matrix, allocator),
    );

    var sum = try first_context.add(local, local);
    defer sum.deinit();
    try first_context.readback(sum, &values);
    try testing.expectEqualSlices(f32, &.{ 2, 4 }, &values);
}

test "batched matmul supports logical transposes on every available device" {
    const testing = std.testing;
    const allocator = testing.allocator;
    const preferences = [_]DevicePreference{ .cpu, .metal, .cuda, .rocm };
    for (preferences) |preference| {
        var device = Device.init(allocator, preference) catch |err| switch (err) {
            error.BackendUnavailable => continue,
            else => return err,
        };
        defer device.deinit();
        var context = ExecutionContext.init(&device);
        var a = try context.upload(&.{ 2, 2, 3 }, &.{ 1, 2, 3, 4, 5, 6, 2, 0, 1, 1, 3, 2 });
        defer a.deinit();
        var plain_b = try context.upload(&.{ 2, 3, 2 }, &.{ 1, 2, 3, 4, 5, 6, 1, 0, 0, 1, 1, 1 });
        defer plain_b.deinit();
        var b = try context.upload(&.{ 2, 2, 3 }, &.{ 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1 });
        defer b.deinit();

        context.resetStats();
        var plain_product = try context.batchedMatmul(a, plain_b, false, false);
        defer plain_product.deinit();
        try testing.expectEqualSlices(usize, &.{ 2, 2, 2 }, plain_product.shape.slice());
        try testing.expectEqual(@as(usize, 0), context.stats.readbacks);
        var plain_actual: [8]f32 = undefined;
        try context.readback(plain_product, &plain_actual);
        try testing.expectEqualSlices(f32, &.{ 22, 28, 49, 64, 3, 1, 3, 5 }, &plain_actual);

        context.resetStats();
        var product = try context.batchedMatmul(a, b, false, true);
        defer product.deinit();
        try testing.expectEqualSlices(usize, &.{ 2, 2, 2 }, product.shape.slice());
        try testing.expectEqual(@as(usize, 0), context.stats.readbacks);
        var actual: [8]f32 = undefined;
        try context.readback(product, &actual);
        try testing.expectEqualSlices(f32, &.{ 4, 5, 10, 11, 2, 1, 4, 5 }, &actual);
        if (device.backendType() != .CPU) {
            try testing.expectEqual(@as(usize, 1), context.backendStats().kernel_launches);
        }
    }
}

test "split and merge heads preserve token-major values" {
    const testing = std.testing;
    var device = try Device.init(testing.allocator, .cpu);
    defer device.deinit();
    var context = ExecutionContext.init(&device);
    var input = try context.upload(&.{ 1, 2, 2, 2 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    defer input.deinit();
    var split = try context.splitHeads(input);
    defer split.deinit();
    try testing.expectEqualSlices(usize, &.{ 2, 2, 2 }, split.shape.slice());
    var split_values: [8]f32 = undefined;
    try context.readback(split, &split_values);
    try testing.expectEqualSlices(f32, &.{ 1, 2, 5, 6, 3, 4, 7, 8 }, &split_values);
    var merged = try context.mergeHeads(split, 1, 2);
    defer merged.deinit();
    var merged_values: [8]f32 = undefined;
    try context.readback(merged, &merged_values);
    try testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4, 5, 6, 7, 8 }, &merged_values);
    try testing.expectError(
        error.DimensionMismatch,
        context.mergeHeads(split, std.math.maxInt(usize), 2),
    );
}

test "masked softmax backward matches finite differences" {
    const testing = std.testing;
    const allocator = testing.allocator;
    var device = try Device.init(allocator, .cpu);
    defer device.deinit();
    var context = ExecutionContext.init(&device);
    const scores = [_]f32{ 0.2, -0.7, 1.1 };
    const mask_values = [_]f32{ 1, 0, 1 };
    const upstream_values = [_]f32{ 0.4, -0.2, 1.3 };
    var input = try context.upload(&.{ 1, 3 }, &scores);
    defer input.deinit();
    var mask = try context.upload(&.{ 1, 3 }, &mask_values);
    defer mask.deinit();
    var upstream = try context.upload(&.{ 1, 3 }, &upstream_values);
    defer upstream.deinit();
    var probabilities = try context.maskedSoftmax(input, mask);
    defer probabilities.deinit();
    var gradient = try context.maskedSoftmaxBackward(probabilities, upstream, mask);
    defer gradient.deinit();

    var probability_values: [3]f32 = undefined;
    var gradient_values: [3]f32 = undefined;
    try context.readback(probabilities, &probability_values);
    try context.readback(gradient, &gradient_values);
    try testing.expectEqual(@as(f32, 0), probability_values[1]);
    try testing.expectEqual(@as(f32, 0), gradient_values[1]);
    try testing.expectApproxEqAbs(@as(f32, 1), probability_values[0] + probability_values[2], 1e-5);
    const epsilon: f32 = 1e-3;
    for (0..scores.len) |index| {
        var plus = scores;
        var minus = scores;
        plus[index] += epsilon;
        minus[index] -= epsilon;
        const numerical = (referenceMaskedSoftmaxObjective(&plus, &mask_values, &upstream_values) -
            referenceMaskedSoftmaxObjective(&minus, &mask_values, &upstream_values)) / (2 * epsilon);
        try testing.expectApproxEqAbs(numerical, gradient_values[index], 2e-4);
    }

    var empty_mask = try context.upload(&.{ 1, 3 }, &.{ 0, 0, 0 });
    defer empty_mask.deinit();
    var empty = try context.maskedSoftmax(input, empty_mask);
    defer empty.deinit();
    var empty_values: [3]f32 = undefined;
    try context.readback(empty, &empty_values);
    try testing.expectEqualSlices(f32, &.{ 0, 0, 0 }, &empty_values);
}

test "seeded inverted dropout reuses its mask in backward" {
    const testing = std.testing;
    const allocator = testing.allocator;
    var device = try Device.init(allocator, .cpu);
    defer device.deinit();
    var context = ExecutionContext.init(&device);
    var input = try context.upload(&.{ 2, 4 }, &.{ 2, 2, 2, 2, 2, 2, 2, 2 });
    defer input.deinit();
    var first = try context.dropout(input, 0.5, 42, true);
    defer first.deinit();
    var second = try context.dropout(input, 0.5, 42, true);
    defer second.deinit();
    var output_gradient = try context.upload(&.{ 2, 4 }, &.{ 1, 1, 1, 1, 1, 1, 1, 1 });
    defer output_gradient.deinit();
    var input_gradient = try context.dropoutBackward(output_gradient, first.mask);
    defer input_gradient.deinit();

    var first_mask: [8]f32 = undefined;
    var second_mask: [8]f32 = undefined;
    var output: [8]f32 = undefined;
    var gradient: [8]f32 = undefined;
    try context.readback(first.mask, &first_mask);
    try context.readback(second.mask, &second_mask);
    try context.readback(first.output, &output);
    try context.readback(input_gradient, &gradient);
    try testing.expectEqualSlices(f32, &first_mask, &second_mask);
    for (first_mask, output, gradient) |mask_value, output_value, gradient_value| {
        try testing.expect(mask_value == 0 or mask_value == 2);
        try testing.expectEqual(2 * mask_value, output_value);
        try testing.expectEqual(mask_value, gradient_value);
    }
}

test "masked sparse cross entropy ignores padding rows" {
    const testing = std.testing;
    const allocator = testing.allocator;
    var device = try Device.init(allocator, .cpu);
    defer device.deinit();
    var context = ExecutionContext.init(&device);
    var logits = try context.upload(&.{ 1, 3, 3 }, &.{ 2, 1, 0, 100, -100, 7, 0, 1, 2 });
    defer logits.deinit();
    var loss = try context.maskedSparseCrossEntropy(logits, &.{ 0, 1, 2 }, &.{ 1, 0, 1 });
    defer loss.deinit();
    try testing.expectEqualSlices(usize, logits.shape.slice(), loss.gradient.shape.slice());
    var gradient: [9]f32 = undefined;
    try context.readback(loss.gradient, &gradient);
    try testing.expectEqualSlices(f32, &.{ 0, 0, 0 }, gradient[3..6]);
    try testing.expectApproxEqAbs(@as(f32, 0), gradient[0] + gradient[1] + gradient[2], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0), gradient[6] + gradient[7] + gradient[8], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.407606), try loss.meanLoss(&context), 1e-5);
}

test "tensor reshape preserves storage" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var device = try Device.init(allocator, .cpu);
    defer device.deinit();
    var tensor = try Tensor.fromF32(device.instance, allocator, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer tensor.deinit();

    try tensor.reshape(&.{ 3, 2 });
    try testing.expectEqualSlices(usize, &.{ 3, 2 }, tensor.shape.slice());
    try testing.expectError(error.DimensionMismatch, tensor.reshape(&.{ 4, 2 }));

    var actual: [6]f32 = undefined;
    try tensor.readF32(&actual);
    try testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4, 5, 6 }, &actual);
}

test "explicit device selection never silently falls back" {
    const testing = std.testing;
    const allocator = testing.allocator;

    if (!build_options.enable_metal) {
        try testing.expectError(error.BackendUnavailable, Device.init(allocator, .metal));
    }
    if (!build_options.enable_cuda) {
        try testing.expectError(error.BackendUnavailable, Device.init(allocator, .cuda));
    }
    if (!build_options.enable_rocm) {
        try testing.expectError(error.BackendUnavailable, Device.init(allocator, .rocm));
    }
}

test "auto device supports bulk f32 round trip" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var device = try Device.init(allocator, .auto);
    defer device.deinit();
    var context = ExecutionContext.init(&device);
    context.resetStats();

    const expected = [_]f32{ -1.25, 0.5, 2.75, 4.0 };
    var tensor = try context.upload(&.{ 2, 2 }, &expected);
    defer tensor.deinit();
    var identity = try context.upload(&.{ 2, 2 }, &.{ 1, 0, 0, 1 });
    defer identity.deinit();
    var product = try context.matmul(tensor, identity);
    defer product.deinit();

    var actual: [expected.len]f32 = undefined;
    try context.readback(product, &actual);
    try testing.expectEqualSlices(f32, &expected, &actual);

    const runtime_stats = context.backendStats();
    try testing.expectEqual(@as(usize, 3), runtime_stats.buffer_allocations);

    switch (device.backendType()) {
        .Metal, .CUDA, .ROCm => {
            try testing.expectEqual(@as(usize, 2), runtime_stats.host_to_device_transfers);
            try testing.expectEqual(@as(usize, 1), runtime_stats.device_to_host_transfers);
            try testing.expectEqual(@as(usize, 1), runtime_stats.kernel_launches);
            try testing.expectEqual(@as(usize, 1), runtime_stats.vendor_gemm_launches);
            try testing.expectEqual(@as(usize, 1), runtime_stats.synchronizations);
        },
        .CPU => {
            try testing.expectEqual(@as(usize, 0), runtime_stats.host_to_device_transfers);
            try testing.expectEqual(@as(usize, 0), runtime_stats.device_to_host_transfers);
            try testing.expectEqual(@as(usize, 0), runtime_stats.kernel_launches);
            try testing.expectEqual(@as(usize, 0), runtime_stats.vendor_gemm_launches);
            try testing.expectEqual(@as(usize, 0), runtime_stats.synchronizations);
        },
    }
}

test "execution batch shares one gpu synchronization" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var device = try Device.init(allocator, .auto);
    defer device.deinit();
    var context = ExecutionContext.init(&device);
    context.resetStats();

    var input = try context.upload(&.{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer input.deinit();
    var identity = try context.upload(&.{ 2, 2 }, &.{ 1, 0, 0, 1 });
    defer identity.deinit();

    try context.beginBatch();
    try testing.expectError(error.BatchAlreadyActive, context.beginBatch());
    var product = try context.matmul(input, identity);
    var probabilities = try context.softmax(product);
    defer probabilities.deinit();
    product.deinit();
    try context.endBatch();
    try testing.expectError(error.NoActiveBatch, context.endBatch());

    var actual: [4]f32 = undefined;
    try context.readback(probabilities, &actual);
    try testing.expectApproxEqAbs(@as(f32, 1.0), actual[0] + actual[1], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 1.0), actual[2] + actual[3], 1e-5);

    const runtime_stats = context.backendStats();
    try testing.expectEqual(@as(usize, 4), runtime_stats.buffer_allocations);
    if (device.backendType() == .CPU) {
        try testing.expectEqual(@as(usize, 0), runtime_stats.kernel_launches);
        try testing.expectEqual(@as(usize, 0), runtime_stats.synchronizations);
    } else {
        try testing.expectEqual(@as(usize, 2), runtime_stats.host_to_device_transfers);
        try testing.expectEqual(@as(usize, 1), runtime_stats.device_to_host_transfers);
        try testing.expectEqual(@as(usize, 2), runtime_stats.kernel_launches);
        try testing.expectEqual(@as(usize, 1), runtime_stats.vendor_gemm_launches);
        try testing.expectEqual(@as(usize, 1), runtime_stats.synchronizations);
    }
}

test "transformer tensor primitives match expected values" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var device = try Device.init(allocator, .cpu);
    defer device.deinit();
    var context = ExecutionContext.init(&device);

    var input = try context.upload(&.{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer input.deinit();
    var weights = try context.upload(&.{ 2, 2 }, &.{ 1, 0, 0, 1 });
    defer weights.deinit();
    var bias = try context.upload(&.{ 1, 2 }, &.{ 0.5, -0.5 });
    defer bias.deinit();
    var projected = try context.linear(input, weights, bias);
    defer projected.deinit();

    var projected_values: [4]f32 = undefined;
    try context.readback(projected, &projected_values);
    try testing.expectEqualSlices(f32, &.{ 1.5, 1.5, 3.5, 3.5 }, &projected_values);

    var gamma = try context.upload(&.{ 1, 2 }, &.{ 1, 1 });
    defer gamma.deinit();
    var beta = try context.upload(&.{ 1, 2 }, &.{ 0, 0 });
    defer beta.deinit();
    var normalized = try context.layerNorm(input, gamma, beta, 1e-5);
    defer normalized.deinit();
    var normalized_values: [4]f32 = undefined;
    try context.readback(normalized, &normalized_values);
    try testing.expectApproxEqAbs(@as(f32, -0.99998), normalized_values[0], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 0.99998), normalized_values[1], 1e-4);

    var activated = try context.gelu(input);
    defer activated.deinit();
    var activated_values: [4]f32 = undefined;
    try context.readback(activated, &activated_values);
    try testing.expectApproxEqAbs(@as(f32, @floatCast(Activation.gelu(1.0))), activated_values[0], 1e-5);
}

test "tensor activation kinds match scalar references" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var device = try Device.init(allocator, .cpu);
    defer device.deinit();
    var context = ExecutionContext.init(&device);
    var input = try context.upload(&.{ 1, 3 }, &.{ -1, 0, 1 });
    defer input.deinit();

    const kinds = [_]ActivationKind{ .linear, .relu, .sigmoid, .tanh, .gelu };
    for (kinds) |kind| {
        var output = try context.activate(input, kind);
        defer output.deinit();
        var derivative = try context.activationDerivative(input, kind);
        defer derivative.deinit();

        var output_values: [3]f32 = undefined;
        var derivative_values: [3]f32 = undefined;
        try context.readback(output, &output_values);
        try context.readback(derivative, &derivative_values);
        for ([_]f64{ -1, 0, 1 }, 0..) |value, index| {
            try testing.expectApproxEqAbs(
                @as(f32, @floatCast(activationFunction(kind)(value))),
                output_values[index],
                1e-5,
            );
            try testing.expectApproxEqAbs(
                @as(f32, @floatCast(activationDerivativeFunction(kind)(value))),
                derivative_values[index],
                1e-5,
            );
        }
    }
}

test "sum squares reduces every logical rank on device" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var device = try Device.init(allocator, .cpu);
    defer device.deinit();
    var context = ExecutionContext.init(&device);
    var input = try context.upload(&.{ 1, 2, 2 }, &.{ 1, -2, 3, -4 });
    defer input.deinit();

    context.resetStats();
    var total = try context.sumSquares(input);
    defer total.deinit();
    try testing.expectEqualSlices(usize, &.{ 1, 1 }, total.shape.slice());
    try testing.expectEqual(@as(usize, 0), context.stats.readbacks);
    try testing.expectEqual(@as(usize, 4), context.stats.kernels);

    var actual: [1]f32 = undefined;
    try context.readback(total, &actual);
    try testing.expectApproxEqAbs(@as(f32, 30), actual[0], 1e-5);
}

test "fused optimizer update clips gradients on every available device" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const preferences = [_]DevicePreference{ .cpu, .metal, .cuda, .rocm };
    for (preferences) |preference| {
        var device = Device.init(allocator, preference) catch |err| switch (err) {
            error.BackendUnavailable => continue,
            else => return err,
        };
        defer device.deinit();
        var context = ExecutionContext.init(&device);
        var parameter = try context.upload(&.{ 1, 2 }, &.{ 1, 2 });
        defer parameter.deinit();
        var gradient = try context.upload(&.{ 1, 2 }, &.{ 3, 4 });
        defer gradient.deinit();
        var first_moment = try context.createTensor(&.{ 1, 2 });
        defer first_moment.deinit();
        var second_moment = try context.createTensor(&.{ 1, 2 });
        defer second_moment.deinit();
        var total_squares = try context.upload(&.{ 1, 1 }, &.{25});
        defer total_squares.deinit();

        context.resetStats();
        try context.beginBatch();
        try context.optimizerUpdate(
            &parameter,
            gradient,
            &first_moment,
            &second_moment,
            total_squares,
            .{
                .kind = .sgd,
                .learning_rate = 0.1,
                .beta1 = 0,
                .beta2 = 0,
                .epsilon = 1e-8,
                .weight_decay = 0,
                .bias_correction1 = 1,
                .bias_correction2 = 1,
                .max_gradient_norm = 1,
            },
        );
        try context.endBatch();
        try testing.expectEqual(@as(usize, 0), context.stats.readbacks);

        var actual: [2]f32 = undefined;
        try context.readback(parameter, &actual);
        try testing.expectApproxEqAbs(@as(f32, 0.94), actual[0], 1e-5);
        try testing.expectApproxEqAbs(@as(f32, 1.92), actual[1], 1e-5);
    }
}

test "transformer pipeline stays device resident within a batch" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const input_values = [_]f32{ 1, -2, 0.5, -1, 3, 2 };
    const weight_values = [_]f32{
        0.5, -1,   0,    2,
        1,   0.25, -0.5, 0,
        -2,  1,    0.75, -1,
    };
    const bias_values = [_]f32{ 0.1, -0.2, 0.3, 0.4 };
    const gamma_values = [_]f32{ 1, 0.5, 1.5, -1 };
    const beta_values = [_]f32{ 0, 0.1, -0.2, 0.3 };

    var device = try Device.init(allocator, .auto);
    defer device.deinit();
    var context = ExecutionContext.init(&device);
    context.resetStats();

    var input = try context.upload(&.{ 2, 3 }, &input_values);
    defer input.deinit();
    var weights = try context.upload(&.{ 3, 4 }, &weight_values);
    defer weights.deinit();
    var bias = try context.upload(&.{ 1, 4 }, &bias_values);
    defer bias.deinit();
    var gamma = try context.upload(&.{ 1, 4 }, &gamma_values);
    defer gamma.deinit();
    var beta = try context.upload(&.{ 1, 4 }, &beta_values);
    defer beta.deinit();

    try context.beginBatch();
    var projected = try context.linear(input, weights, bias);
    var activated = try context.gelu(projected);
    projected.deinit();
    var normalized = try context.layerNorm(activated, gamma, beta, 1e-5);
    activated.deinit();
    try context.endBatch();
    defer normalized.deinit();

    var actual: [8]f32 = undefined;
    try context.readback(normalized, &actual);

    var expected: [8]f32 = undefined;
    for (0..2) |row| {
        var activated_row: [4]f32 = undefined;
        for (0..4) |col| {
            var projected_value = bias_values[col];
            for (0..3) |inner| {
                projected_value += input_values[row * 3 + inner] * weight_values[inner * 4 + col];
            }
            activated_row[col] = @floatCast(Activation.gelu(projected_value));
        }

        var mean: f32 = 0;
        for (activated_row) |value| mean += value;
        mean /= 4;
        var variance: f32 = 0;
        for (activated_row) |value| {
            const centered = value - mean;
            variance += centered * centered;
        }
        variance /= 4;
        const inverse_std = 1.0 / @sqrt(variance + 1e-5);
        for (0..4) |col| {
            expected[row * 4 + col] = (activated_row[col] - mean) * inverse_std * gamma_values[col] + beta_values[col];
        }
    }

    for (actual, expected) |actual_value, expected_value| {
        try testing.expectApproxEqAbs(expected_value, actual_value, 2e-4);
    }
    try testing.expectEqual(@as(usize, 4), context.stats.kernels);

    const runtime_stats = context.backendStats();
    try testing.expectEqual(@as(usize, 9), runtime_stats.buffer_allocations);
    if (device.backendType() == .CPU) {
        try testing.expectEqual(@as(usize, 0), runtime_stats.kernel_launches);
        try testing.expectEqual(@as(usize, 0), runtime_stats.synchronizations);
    } else {
        try testing.expectEqual(@as(usize, 5), runtime_stats.host_to_device_transfers);
        try testing.expectEqual(@as(usize, 1), runtime_stats.device_to_host_transfers);
        try testing.expectEqual(@as(usize, 4), runtime_stats.kernel_launches);
        try testing.expectEqual(@as(usize, 1), runtime_stats.vendor_gemm_launches);
        try testing.expectEqual(@as(usize, 1), runtime_stats.synchronizations);
    }
}

test "causal self attention masks future tokens on device" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var device = try Device.init(allocator, .auto);
    defer device.deinit();
    var context = ExecutionContext.init(&device);
    context.resetStats();

    var query = try context.upload(&.{ 3, 4 }, &([_]f32{0} ** 12));
    defer query.deinit();
    var key = try context.upload(&.{ 3, 4 }, &([_]f32{0} ** 12));
    defer key.deinit();
    var value = try context.upload(&.{ 3, 4 }, &.{
        1, 2, 10, 20,
        3, 4, 30, 40,
        5, 6, 50, 60,
    });
    defer value.deinit();

    try testing.expectError(error.DimensionMismatch, context.causalSelfAttention(query, key, value, 3));
    try context.beginBatch();
    var attended = try context.causalSelfAttention(query, key, value, 2);
    try context.endBatch();
    defer attended.deinit();

    var actual: [12]f32 = undefined;
    try context.readback(attended, &actual);
    const expected = [_]f32{
        1, 2, 10, 20,
        2, 3, 20, 30,
        3, 4, 30, 40,
    };
    for (actual, expected) |actual_value, expected_value| {
        try testing.expectApproxEqAbs(expected_value, actual_value, 1e-5);
    }

    const runtime_stats = context.backendStats();
    try testing.expectEqual(@as(usize, 4), runtime_stats.buffer_allocations);
    if (device.backendType() == .CPU) {
        try testing.expectEqual(@as(usize, 0), runtime_stats.kernel_launches);
        try testing.expectEqual(@as(usize, 0), runtime_stats.synchronizations);
    } else {
        try testing.expectEqual(@as(usize, 3), runtime_stats.host_to_device_transfers);
        try testing.expectEqual(@as(usize, 1), runtime_stats.device_to_host_transfers);
        try testing.expectEqual(@as(usize, 1), runtime_stats.kernel_launches);
        try testing.expectEqual(@as(usize, 1), runtime_stats.synchronizations);
    }
}

test "layer norm backward matches finite differences" {
    const testing = std.testing;
    const allocator = testing.allocator;
    var device = try Device.init(allocator, .cpu);
    defer device.deinit();
    var context = ExecutionContext.init(&device);

    const input_values = [_]f32{ 0.2, -0.4, 1.1, 0.7, -1.2, 0.3 };
    const gamma_values = [_]f32{ 1.2, -0.7, 0.5 };
    const beta_values = [_]f32{ 0.1, 0.2, -0.3 };
    const output_gradient_values = [_]f32{ 0.5, -0.2, 0.8, -0.4, 0.7, 0.3 };
    var input = try context.upload(&.{ 2, 3 }, &input_values);
    defer input.deinit();
    var gamma = try context.upload(&.{ 1, 3 }, &gamma_values);
    defer gamma.deinit();
    var output_gradient = try context.upload(&.{ 2, 3 }, &output_gradient_values);
    defer output_gradient.deinit();
    var gradients = try context.layerNormBackward(input, gamma, output_gradient, 1e-5);
    defer gradients.deinit();
    var input_gradient: [6]f32 = undefined;
    var gamma_gradient: [3]f32 = undefined;
    var beta_gradient: [3]f32 = undefined;
    try context.readback(gradients.input, &input_gradient);
    try context.readback(gradients.gamma, &gamma_gradient);
    try context.readback(gradients.beta, &beta_gradient);

    const finite_difference: f32 = 1e-3;
    for (0..input_values.len) |index| {
        var plus = input_values;
        var minus = input_values;
        plus[index] += finite_difference;
        minus[index] -= finite_difference;
        const numerical = (referenceLayerNormObjective(&plus, &gamma_values, &beta_values, &output_gradient_values, 2, 3, 1e-5) -
            referenceLayerNormObjective(&minus, &gamma_values, &beta_values, &output_gradient_values, 2, 3, 1e-5)) / (2 * finite_difference);
        try testing.expectApproxEqAbs(numerical, input_gradient[index], 2e-3);
    }
    for (0..gamma_values.len) |index| {
        var plus = gamma_values;
        var minus = gamma_values;
        plus[index] += finite_difference;
        minus[index] -= finite_difference;
        const numerical = (referenceLayerNormObjective(&input_values, &plus, &beta_values, &output_gradient_values, 2, 3, 1e-5) -
            referenceLayerNormObjective(&input_values, &minus, &beta_values, &output_gradient_values, 2, 3, 1e-5)) / (2 * finite_difference);
        try testing.expectApproxEqAbs(numerical, gamma_gradient[index], 2e-3);
    }
    for (0..beta_values.len) |index| {
        var plus = beta_values;
        var minus = beta_values;
        plus[index] += finite_difference;
        minus[index] -= finite_difference;
        const numerical = (referenceLayerNormObjective(&input_values, &gamma_values, &plus, &output_gradient_values, 2, 3, 1e-5) -
            referenceLayerNormObjective(&input_values, &gamma_values, &minus, &output_gradient_values, 2, 3, 1e-5)) / (2 * finite_difference);
        try testing.expectApproxEqAbs(numerical, beta_gradient[index], 2e-3);
    }
}

test "causal attention backward matches finite differences" {
    const testing = std.testing;
    const allocator = testing.allocator;
    var device = try Device.init(allocator, .cpu);
    defer device.deinit();
    var context = ExecutionContext.init(&device);

    const query_values = [_]f32{ 0.2, -0.4, 0.7, 0.3 };
    const key_values = [_]f32{ -0.5, 0.1, 0.6, -0.2 };
    const value_values = [_]f32{ 0.8, -0.7, 0.4, 1.1 };
    const output_gradient_values = [_]f32{ 0.3, -0.6, 0.9, 0.2 };
    var query = try context.upload(&.{ 2, 2 }, &query_values);
    defer query.deinit();
    var key = try context.upload(&.{ 2, 2 }, &key_values);
    defer key.deinit();
    var value = try context.upload(&.{ 2, 2 }, &value_values);
    defer value.deinit();
    var output_gradient = try context.upload(&.{ 2, 2 }, &output_gradient_values);
    defer output_gradient.deinit();
    var gradients = try context.causalSelfAttentionBackward(query, key, value, output_gradient, 1);
    defer gradients.deinit();
    var query_gradient: [4]f32 = undefined;
    var key_gradient: [4]f32 = undefined;
    var value_gradient: [4]f32 = undefined;
    try context.readback(gradients.query, &query_gradient);
    try context.readback(gradients.key, &key_gradient);
    try context.readback(gradients.value, &value_gradient);

    const finite_difference: f32 = 1e-3;
    for (0..query_values.len) |index| {
        var plus = query_values;
        var minus = query_values;
        plus[index] += finite_difference;
        minus[index] -= finite_difference;
        const numerical = (referenceAttentionObjective(&plus, &key_values, &value_values, &output_gradient_values, 2, 2, 1) -
            referenceAttentionObjective(&minus, &key_values, &value_values, &output_gradient_values, 2, 2, 1)) / (2 * finite_difference);
        try testing.expectApproxEqAbs(numerical, query_gradient[index], 2e-3);
    }
    for (0..key_values.len) |index| {
        var plus = key_values;
        var minus = key_values;
        plus[index] += finite_difference;
        minus[index] -= finite_difference;
        const numerical = (referenceAttentionObjective(&query_values, &plus, &value_values, &output_gradient_values, 2, 2, 1) -
            referenceAttentionObjective(&query_values, &minus, &value_values, &output_gradient_values, 2, 2, 1)) / (2 * finite_difference);
        try testing.expectApproxEqAbs(numerical, key_gradient[index], 2e-3);
    }
    for (0..value_values.len) |index| {
        var plus = value_values;
        var minus = value_values;
        plus[index] += finite_difference;
        minus[index] -= finite_difference;
        const numerical = (referenceAttentionObjective(&query_values, &key_values, &plus, &output_gradient_values, 2, 2, 1) -
            referenceAttentionObjective(&query_values, &key_values, &minus, &output_gradient_values, 2, 2, 1)) / (2 * finite_difference);
        try testing.expectApproxEqAbs(numerical, value_gradient[index], 2e-3);
    }
}

test "transformer backward kernels share one device synchronization" {
    const testing = std.testing;
    const allocator = testing.allocator;
    var device = try Device.init(allocator, .auto);
    defer device.deinit();
    var context = ExecutionContext.init(&device);
    context.resetStats();

    var input = try context.upload(&.{ 2, 2 }, &.{ 0.2, -0.4, 0.7, 0.3 });
    defer input.deinit();
    var gamma = try context.upload(&.{ 1, 2 }, &.{ 1.2, -0.7 });
    defer gamma.deinit();
    var output_gradient = try context.upload(&.{ 2, 2 }, &.{ 0.3, -0.6, 0.9, 0.2 });
    defer output_gradient.deinit();
    var key = try context.upload(&.{ 2, 2 }, &.{ -0.5, 0.1, 0.6, -0.2 });
    defer key.deinit();
    var value = try context.upload(&.{ 2, 2 }, &.{ 0.8, -0.7, 0.4, 1.1 });
    defer value.deinit();

    try context.beginBatch();
    var norm_gradients = try context.layerNormBackward(input, gamma, output_gradient, 1e-5);
    var attention_gradients = try context.causalSelfAttentionBackward(input, key, value, output_gradient, 1);
    try context.endBatch();
    defer norm_gradients.deinit();
    defer attention_gradients.deinit();

    var norm_values: [4]f32 = undefined;
    var attention_values: [4]f32 = undefined;
    try context.readback(norm_gradients.input, &norm_values);
    try context.readback(attention_gradients.query, &attention_values);
    for (norm_values ++ attention_values) |value_item| try testing.expect(std.math.isFinite(value_item));

    const runtime_stats = context.backendStats();
    if (device.backendType() == .CPU) {
        try testing.expectEqual(@as(usize, 0), runtime_stats.kernel_launches);
    } else {
        try testing.expectEqual(@as(usize, 5), runtime_stats.host_to_device_transfers);
        try testing.expectEqual(@as(usize, 2), runtime_stats.device_to_host_transfers);
        try testing.expectEqual(@as(usize, 3), runtime_stats.kernel_launches);
        try testing.expectEqual(@as(usize, 1), runtime_stats.synchronizations);
    }
}

test "embedding lookup and repeated-token gradients stay on device" {
    const testing = std.testing;
    const allocator = testing.allocator;
    var device = try Device.init(allocator, .auto);
    defer device.deinit();
    var context = ExecutionContext.init(&device);
    context.resetStats();

    var table = try context.upload(&.{ 3, 2 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer table.deinit();
    var indices = try context.upload(&.{ 3, 1 }, &.{ 2, 0, 2 });
    defer indices.deinit();
    var output_gradient = try context.upload(&.{ 3, 2 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer output_gradient.deinit();

    try context.beginBatch();
    var embedded = try context.embedding(table, indices);
    var table_gradient = try context.embeddingBackward(indices, output_gradient, 3);
    try context.endBatch();
    defer embedded.deinit();
    defer table_gradient.deinit();
    var embedded_values: [6]f32 = undefined;
    var gradient_values: [6]f32 = undefined;
    try context.readback(embedded, &embedded_values);
    try context.readback(table_gradient, &gradient_values);
    try testing.expectEqualSlices(f32, &.{ 5, 6, 1, 2, 5, 6 }, &embedded_values);
    try testing.expectEqualSlices(f32, &.{ 3, 4, 0, 0, 6, 8 }, &gradient_values);

    const runtime_stats = context.backendStats();
    if (device.backendType() == .CPU) {
        try testing.expectEqual(@as(usize, 0), runtime_stats.kernel_launches);
    } else {
        try testing.expectEqual(@as(usize, 3), runtime_stats.host_to_device_transfers);
        try testing.expectEqual(@as(usize, 2), runtime_stats.device_to_host_transfers);
        try testing.expectEqual(@as(usize, 2), runtime_stats.kernel_launches);
        try testing.expectEqual(@as(usize, 1), runtime_stats.synchronizations);
    }
}
