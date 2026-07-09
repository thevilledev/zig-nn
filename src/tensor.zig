const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const backend_mod = @import("backend.zig");
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

/// User-facing backend selection policy.
pub const DevicePreference = enum {
    cpu,
    auto,
    metal,
    cuda,
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

    pub fn slice(self: Shape) []const usize {
        return self.dims[0..self.rank];
    }

    pub fn eql(self: Shape, other: Shape) bool {
        return std.mem.eql(usize, self.slice(), other.slice());
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
        try tensor.readF32(values);
        self.stats.readbacks += 1;
        self.stats.readback_bytes += values.len * @sizeOf(f32);
    }

    pub fn matmul(self: *ExecutionContext, a: Tensor, b: Tensor) !Tensor {
        if (a.shape.rank != 2 or b.shape.rank != 2) return error.InvalidRank;
        if (a.shape.dims[1] != b.shape.dims[0]) return error.DimensionMismatch;
        if (a.backendType() != b.backendType()) return error.BackendMismatch;

        const matrix = try a.matrix.dotProduct(b.matrix, self.device.allocator);
        self.stats.kernels += 1;
        return .{
            .matrix = matrix,
            .shape = try Shape.init(&.{ a.shape.dims[0], b.shape.dims[1] }),
            .allocator = self.device.allocator,
        };
    }

    pub fn add(self: *ExecutionContext, a: Tensor, b: Tensor) !Tensor {
        if (!a.shape.eql(b.shape)) return error.DimensionMismatch;
        if (a.backendType() != b.backendType()) return error.BackendMismatch;

        const matrix = try a.matrix.add(b.matrix, self.device.allocator);
        self.stats.kernels += 1;
        return .{
            .matrix = matrix,
            .shape = a.shape,
            .allocator = self.device.allocator,
        };
    }

    pub fn subtract(self: *ExecutionContext, a: Tensor, b: Tensor) !Tensor {
        if (!a.shape.eql(b.shape)) return error.DimensionMismatch;
        if (a.backendType() != b.backendType()) return error.BackendMismatch;

        const matrix = try a.matrix.subtract(b.matrix, self.device.allocator);
        self.stats.kernels += 1;
        return .{
            .matrix = matrix,
            .shape = a.shape,
            .allocator = self.device.allocator,
        };
    }

    pub fn multiply(self: *ExecutionContext, a: Tensor, b: Tensor) !Tensor {
        if (!a.shape.eql(b.shape)) return error.DimensionMismatch;
        if (a.backendType() != b.backendType()) return error.BackendMismatch;

        const matrix = try a.matrix.elementWiseMultiply(b.matrix, self.device.allocator);
        self.stats.kernels += 1;
        return .{
            .matrix = matrix,
            .shape = a.shape,
            .allocator = self.device.allocator,
        };
    }

    pub fn scale(self: *ExecutionContext, input: Tensor, scalar: f32) !Tensor {
        const matrix = try input.matrix.scale(scalar, self.device.allocator);
        self.stats.kernels += 1;
        return .{
            .matrix = matrix,
            .shape = input.shape,
            .allocator = self.device.allocator,
        };
    }

    pub fn transpose(self: *ExecutionContext, input: Tensor) !Tensor {
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
        if (input.shape.rank != 2) return error.InvalidRank;
        const matrix = try input.matrix.sumRows(self.device.allocator);
        self.stats.kernels += 1;
        return .{
            .matrix = matrix,
            .shape = try Shape.init(&.{ 1, input.shape.dims[1] }),
            .allocator = self.device.allocator,
        };
    }

    pub fn addRowBias(self: *ExecutionContext, input: Tensor, bias: Tensor) !Tensor {
        if (input.shape.rank != 2 or bias.shape.rank != 2) return error.InvalidRank;
        if (bias.shape.dims[0] != 1 or bias.shape.dims[1] != input.shape.dims[1]) return error.DimensionMismatch;
        if (input.backendType() != bias.backendType()) return error.BackendMismatch;

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

    pub fn gelu(self: *ExecutionContext, input: Tensor) !Tensor {
        const matrix = try input.matrix.applyActivation(Activation.gelu, self.device.allocator);
        self.stats.kernels += 1;
        return .{
            .matrix = matrix,
            .shape = input.shape,
            .allocator = self.device.allocator,
        };
    }

    pub fn geluDerivative(self: *ExecutionContext, input: Tensor) !Tensor {
        const matrix = try input.matrix.applyActivation(Activation.gelu_derivative, self.device.allocator);
        self.stats.kernels += 1;
        return .{
            .matrix = matrix,
            .shape = input.shape,
            .allocator = self.device.allocator,
        };
    }

    pub fn layerNorm(self: *ExecutionContext, input: Tensor, gamma: Tensor, beta: Tensor, epsilon: f32) !Tensor {
        if (input.shape.rank != 2 or gamma.shape.rank != 2 or beta.shape.rank != 2) return error.InvalidRank;
        if (gamma.shape.dims[0] != 1 or beta.shape.dims[0] != 1 or
            gamma.shape.dims[1] != input.shape.dims[1] or beta.shape.dims[1] != input.shape.dims[1])
        {
            return error.DimensionMismatch;
        }
        if (input.backendType() != gamma.backendType() or input.backendType() != beta.backendType()) return error.BackendMismatch;

        const matrix = try input.matrix.layerNorm(gamma.matrix, beta.matrix, epsilon, self.device.allocator);
        self.stats.kernels += 1;
        return .{
            .matrix = matrix,
            .shape = input.shape,
            .allocator = self.device.allocator,
        };
    }

    pub fn layerNormBackward(self: *ExecutionContext, input: Tensor, gamma: Tensor, output_gradient: Tensor, epsilon: f32) !LayerNormGradients {
        if (input.shape.rank != 2 or gamma.shape.rank != 2 or output_gradient.shape.rank != 2) return error.InvalidRank;
        if (!input.shape.eql(output_gradient.shape) or gamma.shape.dims[0] != 1 or gamma.shape.dims[1] != input.shape.dims[1]) {
            return error.DimensionMismatch;
        }
        if (input.backendType() != gamma.backendType() or input.backendType() != output_gradient.backendType()) return error.BackendMismatch;
        const gradients = try input.matrix.layerNormBackward(gamma.matrix, output_gradient.matrix, epsilon, self.device.allocator);
        self.stats.kernels += 1;
        return .{
            .input = .{ .matrix = gradients.input, .shape = input.shape, .allocator = self.device.allocator },
            .gamma = .{ .matrix = gradients.gamma, .shape = gamma.shape, .allocator = self.device.allocator },
            .beta = .{ .matrix = gradients.beta, .shape = gamma.shape, .allocator = self.device.allocator },
        };
    }

    pub fn causalSelfAttention(self: *ExecutionContext, query: Tensor, key: Tensor, value: Tensor, heads: usize) !Tensor {
        if (query.shape.rank != 2 or key.shape.rank != 2 or value.shape.rank != 2) return error.InvalidRank;
        if (!query.shape.eql(key.shape) or !query.shape.eql(value.shape) or
            heads == 0 or query.shape.dims[1] % heads != 0)
        {
            return error.DimensionMismatch;
        }
        if (query.backendType() != key.backendType() or query.backendType() != value.backendType()) return error.BackendMismatch;

        const matrix = try query.matrix.causalSelfAttention(key.matrix, value.matrix, heads, self.device.allocator);
        self.stats.kernels += 1;
        return .{
            .matrix = matrix,
            .shape = query.shape,
            .allocator = self.device.allocator,
        };
    }

    pub fn causalSelfAttentionBackward(self: *ExecutionContext, query: Tensor, key: Tensor, value: Tensor, output_gradient: Tensor, heads: usize) !AttentionGradients {
        if (query.shape.rank != 2 or key.shape.rank != 2 or value.shape.rank != 2 or output_gradient.shape.rank != 2) return error.InvalidRank;
        if (!query.shape.eql(key.shape) or !query.shape.eql(value.shape) or !query.shape.eql(output_gradient.shape) or
            heads == 0 or query.shape.dims[1] % heads != 0)
        {
            return error.DimensionMismatch;
        }
        if (query.backendType() != key.backendType() or query.backendType() != value.backendType() or
            query.backendType() != output_gradient.backendType())
        {
            return error.BackendMismatch;
        }
        const gradients = try query.matrix.causalSelfAttentionBackward(key.matrix, value.matrix, output_gradient.matrix, heads, self.device.allocator);
        self.stats.kernels += 2;
        return .{
            .query = .{ .matrix = gradients.query, .shape = query.shape, .allocator = self.device.allocator },
            .key = .{ .matrix = gradients.key, .shape = key.shape, .allocator = self.device.allocator },
            .value = .{ .matrix = gradients.value, .shape = value.shape, .allocator = self.device.allocator },
        };
    }

    pub fn softmax(self: *ExecutionContext, input: Tensor) !Tensor {
        if (input.shape.rank != 2) return error.InvalidRank;
        const matrix = try input.matrix.applySoftmax(self.device.allocator);
        self.stats.kernels += 1;
        return .{
            .matrix = matrix,
            .shape = input.shape,
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
        .Metal, .CUDA => {
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
