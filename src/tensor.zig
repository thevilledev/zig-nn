const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const backend_mod = @import("backend.zig");
const root = @import("root.zig");

const Allocator = std.mem.Allocator;
const BackendInstance = root.BackendInstance;
const BackendMatrix = backend_mod.Matrix;
const BackendType = backend_mod.BackendType;

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

/// Explicit execution boundary for tensor construction, operations, readback,
/// and instrumentation. Backend command batching can be implemented behind
/// this API without another model-code migration.
pub const ExecutionContext = struct {
    device: *Device,
    stats: ExecutionStats = .{},

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

    pub fn synchronize(self: *ExecutionContext) void {
        self.stats.synchronizations += 1;
    }

    pub fn resetStats(self: *ExecutionContext) void {
        self.stats = .{};
    }
};

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
