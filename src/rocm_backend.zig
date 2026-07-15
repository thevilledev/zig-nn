const std = @import("std");
const Allocator = std.mem.Allocator;
const backend = @import("backend.zig");
const dimensions = @import("dimensions.zig");
const ComputeBackend = backend.ComputeBackend;
const BackendType = backend.BackendType;
const Matrix = backend.Matrix;
const randomSeed = @import("matrix.zig").randomSeed;
const Activation = @import("activation.zig").Activation;

const build_options = @import("build_options");
const enable_rocm = build_options.enable_rocm;
const kernel_source = @embedFile("rocm/kernels.hip");

pub const ROCmError = error{
    ROCmNotEnabled,
    ROCmNotSupported,
    ROCmDeviceNotFound,
    ROCmInitializationFailed,
    KernelCompilationFailed,
    FunctionNotFound,
    BufferCreationFailed,
    DataTransferError,
    CommandExecutionFailed,
    InvalidBatchIndices,
};

const ROCm = if (enable_rocm) struct {
    const c = @cImport({
        @cInclude("rocm_wrapper.h");
    });

    pub const Backend = anyopaque;
    pub const Buffer = anyopaque;

    pub const KernelId = enum(c_int) {
        matrix_add = c.ZIG_NN_ROCM_KERNEL_MATRIX_ADD,
        matrix_subtract = c.ZIG_NN_ROCM_KERNEL_MATRIX_SUBTRACT,
        matrix_element_wise_multiply = c.ZIG_NN_ROCM_KERNEL_MATRIX_ELEMENT_WISE_MULTIPLY,
        apply_sigmoid = c.ZIG_NN_ROCM_KERNEL_APPLY_SIGMOID,
        apply_relu = c.ZIG_NN_ROCM_KERNEL_APPLY_RELU,
        apply_tanh = c.ZIG_NN_ROCM_KERNEL_APPLY_TANH,
        apply_swish = c.ZIG_NN_ROCM_KERNEL_APPLY_SWISH,
        apply_glu = c.ZIG_NN_ROCM_KERNEL_APPLY_GLU,
        apply_swiglu = c.ZIG_NN_ROCM_KERNEL_APPLY_SWIGLU,
        apply_linear = c.ZIG_NN_ROCM_KERNEL_APPLY_LINEAR,
        apply_linear_derivative = c.ZIG_NN_ROCM_KERNEL_APPLY_LINEAR_DERIVATIVE,
        apply_sigmoid_derivative = c.ZIG_NN_ROCM_KERNEL_APPLY_SIGMOID_DERIVATIVE,
        apply_relu_derivative = c.ZIG_NN_ROCM_KERNEL_APPLY_RELU_DERIVATIVE,
        apply_tanh_derivative = c.ZIG_NN_ROCM_KERNEL_APPLY_TANH_DERIVATIVE,
        apply_swish_derivative = c.ZIG_NN_ROCM_KERNEL_APPLY_SWISH_DERIVATIVE,
        matrix_add_row_bias = c.ZIG_NN_ROCM_KERNEL_MATRIX_ADD_ROW_BIAS,
        apply_gelu = c.ZIG_NN_ROCM_KERNEL_APPLY_GELU,
        apply_gelu_derivative = c.ZIG_NN_ROCM_KERNEL_APPLY_GELU_DERIVATIVE,
    };

    pub const MatmulKind = enum {
        failed,
        custom,
        vendor,
    };

    fn fromC(comptime T: type, value: ?*anyopaque) ?*T {
        return if (value) |ptr| @as(*T, @ptrCast(ptr)) else null;
    }

    fn toC(value: anytype) *anyopaque {
        return @as(*anyopaque, @ptrCast(value));
    }

    fn logError(prefix: []const u8, buffer: []const u8) void {
        const message = std.mem.sliceTo(buffer, 0);
        if (message.len > 0) {
            std.log.err("{s}: {s}", .{ prefix, message });
        } else {
            std.log.err("{s}", .{prefix});
        }
    }

    pub fn createBackend() ?*Backend {
        var error_buffer: [4096]u8 = undefined;
        @memset(&error_buffer, 0);
        const result = c.rocm_backend_create(kernel_source, error_buffer[0..].ptr, error_buffer.len);
        if (result == null) {
            logError("ROCm backend creation failed", error_buffer[0..]);
        }
        return fromC(Backend, result);
    }

    pub fn destroyBackend(backend_ref: ?*Backend) void {
        c.rocm_backend_destroy(if (backend_ref) |value| toC(value) else null);
    }

    pub fn deviceName(backend_ref: *Backend, output: []u8) bool {
        return c.rocm_backend_device_name(toC(backend_ref), output.ptr, output.len) != 0;
    }

    pub fn synchronize(backend_ref: *Backend) bool {
        return c.rocm_backend_synchronize(toC(backend_ref)) != 0;
    }

    pub fn createBuffer(backend_ref: *Backend, count: usize) ?*Buffer {
        return fromC(Buffer, c.rocm_backend_create_buffer(toC(backend_ref), count));
    }

    pub fn destroyBuffer(backend_ref: *Backend, buffer: ?*Buffer) void {
        c.rocm_backend_destroy_buffer(toC(backend_ref), if (buffer) |value| toC(value) else null);
    }

    pub fn bufferUploadF32(backend_ref: *Backend, buffer: *Buffer, data: []const f32) bool {
        return c.rocm_buffer_upload_f32(toC(backend_ref), toC(buffer), data.ptr, data.len) != 0;
    }

    pub fn bufferDownloadF32(backend_ref: *Backend, buffer: *Buffer, data: []f32) bool {
        return c.rocm_buffer_download_f32(toC(backend_ref), toC(buffer), data.ptr, data.len) != 0;
    }

    pub fn launchMatrixMultiply(backend_ref: *Backend, a: *Buffer, b: *Buffer, result: *Buffer, a_rows: u32, a_cols: u32, b_cols: u32) MatmulKind {
        return switch (c.rocm_launch_matrix_multiply(toC(backend_ref), toC(a), toC(b), toC(result), a_rows, a_cols, b_cols)) {
            1 => .custom,
            2 => .vendor,
            else => .failed,
        };
    }

    pub fn launchBatchedMatrixMultiply(backend_ref: *Backend, a: *Buffer, b: *Buffer, result: *Buffer, batch: u32, a_rows: u32, a_cols: u32, b_rows: u32, b_cols: u32, transpose_a: bool, transpose_b: bool) bool {
        return c.rocm_launch_batched_matrix_multiply(toC(backend_ref), toC(a), toC(b), toC(result), batch, a_rows, a_cols, b_rows, b_cols, @intFromBool(transpose_a), @intFromBool(transpose_b)) != 0;
    }

    pub fn launchPermuteBatchHeads(backend_ref: *Backend, input: *Buffer, result: *Buffer, batch: u32, tokens: u32, heads: u32, width: u32, split: bool) bool {
        return c.rocm_launch_permute_batch_heads(toC(backend_ref), toC(input), toC(result), batch, tokens, heads, width, @intFromBool(split)) != 0;
    }

    pub fn launchBinaryKernel(backend_ref: *Backend, kernel_id: KernelId, a: *Buffer, b: *Buffer, result: *Buffer, rows: u32, cols: u32) bool {
        return c.rocm_launch_binary_kernel(toC(backend_ref), @intFromEnum(kernel_id), toC(a), toC(b), toC(result), rows, cols) != 0;
    }

    pub fn launchScale(backend_ref: *Backend, input: *Buffer, result: *Buffer, scalar: f32, rows: u32, cols: u32) bool {
        return c.rocm_launch_scale(toC(backend_ref), toC(input), toC(result), scalar, rows, cols) != 0;
    }

    pub fn launchOptimizerUpdate(backend_ref: *Backend, parameter: *Buffer, gradient: *Buffer, first_moment: *Buffer, second_moment: *Buffer, total_squares: *Buffer, config: backend.OptimizerUpdateConfig, size: u32) bool {
        return c.rocm_launch_optimizer_update(toC(backend_ref), toC(parameter), toC(gradient), toC(first_moment), toC(second_moment), toC(total_squares), @intFromEnum(config.kind), size, config.learning_rate, config.beta1, config.beta2, config.epsilon, config.weight_decay, config.bias_correction1, config.bias_correction2, config.max_gradient_norm) != 0;
    }

    pub fn launchTranspose(backend_ref: *Backend, input: *Buffer, result: *Buffer, rows: u32, cols: u32) bool {
        return c.rocm_launch_transpose(toC(backend_ref), toC(input), toC(result), rows, cols) != 0;
    }

    pub fn launchSumRows(backend_ref: *Backend, input: *Buffer, result: *Buffer, rows: u32, cols: u32) bool {
        return c.rocm_launch_sum_rows(toC(backend_ref), toC(input), toC(result), rows, cols) != 0;
    }

    pub fn launchExtractBatch(backend_ref: *Backend, input: *Buffer, result: *Buffer, start: u32, batch_rows: u32, cols: u32) bool {
        return c.rocm_launch_extract_batch(toC(backend_ref), toC(input), toC(result), start, batch_rows, cols) != 0;
    }

    pub fn launchUnaryKernel(backend_ref: *Backend, kernel_id: KernelId, input: *Buffer, result: *Buffer, size: u32) bool {
        return c.rocm_launch_unary_kernel(toC(backend_ref), @intFromEnum(kernel_id), toC(input), toC(result), size) != 0;
    }

    pub fn launchSoftmaxRows(backend_ref: *Backend, input: *Buffer, result: *Buffer, rows: u32, cols: u32) bool {
        return c.rocm_launch_softmax_rows(toC(backend_ref), toC(input), toC(result), rows, cols) != 0;
    }

    pub fn launchGatedKernel(backend_ref: *Backend, kernel_id: KernelId, linear: *Buffer, gating: *Buffer, result: *Buffer, size: u32) bool {
        return c.rocm_launch_gated_kernel(toC(backend_ref), @intFromEnum(kernel_id), toC(linear), toC(gating), toC(result), size) != 0;
    }

    pub fn launchLayerNorm(backend_ref: *Backend, input: *Buffer, gamma: *Buffer, beta: *Buffer, result: *Buffer, rows: u32, cols: u32, epsilon: f32) bool {
        return c.rocm_launch_layer_norm(toC(backend_ref), toC(input), toC(gamma), toC(beta), toC(result), rows, cols, epsilon) != 0;
    }

    pub fn launchLayerNormBackward(backend_ref: *Backend, input: *Buffer, gamma: *Buffer, output_gradient: *Buffer, input_gradient: *Buffer, gamma_gradient: *Buffer, beta_gradient: *Buffer, rows: u32, cols: u32, epsilon: f32) bool {
        return c.rocm_launch_layer_norm_backward(toC(backend_ref), toC(input), toC(gamma), toC(output_gradient), toC(input_gradient), toC(gamma_gradient), toC(beta_gradient), rows, cols, epsilon) != 0;
    }

    pub fn launchCausalSelfAttention(backend_ref: *Backend, query: *Buffer, key: *Buffer, value: *Buffer, result: *Buffer, tokens: u32, channels: u32, heads: u32) bool {
        return c.rocm_launch_causal_self_attention(toC(backend_ref), toC(query), toC(key), toC(value), toC(result), tokens, channels, heads) != 0;
    }

    pub fn launchCausalAttentionProbabilitiesBackward(backend_ref: *Backend, query: *Buffer, key: *Buffer, value: *Buffer, output_gradient: *Buffer, probabilities: *Buffer, score_gradients: *Buffer, tokens: u32, channels: u32, heads: u32) bool {
        return c.rocm_launch_causal_attention_probabilities_backward(toC(backend_ref), toC(query), toC(key), toC(value), toC(output_gradient), toC(probabilities), toC(score_gradients), tokens, channels, heads) != 0;
    }

    pub fn launchCausalAttentionInputGradients(backend_ref: *Backend, query: *Buffer, key: *Buffer, output_gradient: *Buffer, probabilities: *Buffer, score_gradients: *Buffer, query_gradient: *Buffer, key_gradient: *Buffer, value_gradient: *Buffer, tokens: u32, channels: u32, heads: u32) bool {
        return c.rocm_launch_causal_attention_input_gradients(toC(backend_ref), toC(query), toC(key), toC(output_gradient), toC(probabilities), toC(score_gradients), toC(query_gradient), toC(key_gradient), toC(value_gradient), tokens, channels, heads) != 0;
    }

    pub fn launchEmbeddingLookup(backend_ref: *Backend, table: *Buffer, indices: *Buffer, result: *Buffer, vocabulary_size: u32, tokens: u32, channels: u32) bool {
        return c.rocm_launch_embedding_lookup(toC(backend_ref), toC(table), toC(indices), toC(result), vocabulary_size, tokens, channels) != 0;
    }

    pub fn launchEmbeddingGradient(backend_ref: *Backend, indices: *Buffer, output_gradient: *Buffer, result: *Buffer, vocabulary_size: u32, tokens: u32, channels: u32) bool {
        return c.rocm_launch_embedding_gradient(toC(backend_ref), toC(indices), toC(output_gradient), toC(result), vocabulary_size, tokens, channels) != 0;
    }

    pub fn launchCachedSelfAttention(backend_ref: *Backend, query: *Buffer, key: *Buffer, value: *Buffer, key_cache: *Buffer, value_cache: *Buffer, result: *Buffer, position: u32, channels: u32, heads: u32) bool {
        return c.rocm_launch_cached_self_attention(toC(backend_ref), toC(query), toC(key), toC(value), toC(key_cache), toC(value_cache), toC(result), position, channels, heads) != 0;
    }
} else struct {
    pub const Backend = anyopaque;
    pub const Buffer = anyopaque;

    pub const KernelId = enum(c_int) {
        matrix_add,
        matrix_subtract,
        matrix_element_wise_multiply,
        apply_sigmoid,
        apply_relu,
        apply_tanh,
        apply_swish,
        apply_glu,
        apply_swiglu,
        apply_linear,
        apply_linear_derivative,
        apply_sigmoid_derivative,
        apply_relu_derivative,
        apply_tanh_derivative,
        apply_swish_derivative,
        matrix_add_row_bias,
        apply_gelu,
        apply_gelu_derivative,
    };

    pub const MatmulKind = enum {
        failed,
        custom,
        vendor,
    };

    pub fn createBackend() ?*Backend {
        return null;
    }

    pub fn destroyBackend(_: ?*Backend) void {}

    pub fn deviceName(_: *Backend, _: []u8) bool {
        return false;
    }

    pub fn synchronize(_: *Backend) bool {
        return false;
    }

    pub fn createBuffer(_: *Backend, _: usize) ?*Buffer {
        return null;
    }

    pub fn destroyBuffer(_: *Backend, _: ?*Buffer) void {}

    pub fn bufferUploadF32(_: *Backend, _: *Buffer, _: []const f32) bool {
        return false;
    }

    pub fn bufferDownloadF32(_: *Backend, _: *Buffer, _: []f32) bool {
        return false;
    }

    pub fn launchMatrixMultiply(_: *Backend, _: *Buffer, _: *Buffer, _: *Buffer, _: u32, _: u32, _: u32) MatmulKind {
        return .failed;
    }

    pub fn launchBatchedMatrixMultiply(_: *Backend, _: *Buffer, _: *Buffer, _: *Buffer, _: u32, _: u32, _: u32, _: u32, _: u32, _: bool, _: bool) bool {
        return false;
    }

    pub fn launchPermuteBatchHeads(_: *Backend, _: *Buffer, _: *Buffer, _: u32, _: u32, _: u32, _: u32, _: bool) bool {
        return false;
    }

    pub fn launchBinaryKernel(_: *Backend, _: KernelId, _: *Buffer, _: *Buffer, _: *Buffer, _: u32, _: u32) bool {
        return false;
    }

    pub fn launchScale(_: *Backend, _: *Buffer, _: *Buffer, _: f32, _: u32, _: u32) bool {
        return false;
    }

    pub fn launchOptimizerUpdate(_: *Backend, _: *Buffer, _: *Buffer, _: *Buffer, _: *Buffer, _: *Buffer, _: backend.OptimizerUpdateConfig, _: u32) bool {
        return false;
    }

    pub fn launchTranspose(_: *Backend, _: *Buffer, _: *Buffer, _: u32, _: u32) bool {
        return false;
    }

    pub fn launchSumRows(_: *Backend, _: *Buffer, _: *Buffer, _: u32, _: u32) bool {
        return false;
    }

    pub fn launchExtractBatch(_: *Backend, _: *Buffer, _: *Buffer, _: u32, _: u32, _: u32) bool {
        return false;
    }

    pub fn launchUnaryKernel(_: *Backend, _: KernelId, _: *Buffer, _: *Buffer, _: u32) bool {
        return false;
    }

    pub fn launchSoftmaxRows(_: *Backend, _: *Buffer, _: *Buffer, _: u32, _: u32) bool {
        return false;
    }

    pub fn launchGatedKernel(_: *Backend, _: KernelId, _: *Buffer, _: *Buffer, _: *Buffer, _: u32) bool {
        return false;
    }

    pub fn launchLayerNorm(_: *Backend, _: *Buffer, _: *Buffer, _: *Buffer, _: *Buffer, _: u32, _: u32, _: f32) bool {
        return false;
    }

    pub fn launchLayerNormBackward(_: *Backend, _: *Buffer, _: *Buffer, _: *Buffer, _: *Buffer, _: *Buffer, _: *Buffer, _: u32, _: u32, _: f32) bool {
        return false;
    }

    pub fn launchCausalSelfAttention(_: *Backend, _: *Buffer, _: *Buffer, _: *Buffer, _: *Buffer, _: u32, _: u32, _: u32) bool {
        return false;
    }

    pub fn launchCausalAttentionProbabilitiesBackward(_: *Backend, _: *Buffer, _: *Buffer, _: *Buffer, _: *Buffer, _: *Buffer, _: *Buffer, _: u32, _: u32, _: u32) bool {
        return false;
    }

    pub fn launchCausalAttentionInputGradients(_: *Backend, _: *Buffer, _: *Buffer, _: *Buffer, _: *Buffer, _: *Buffer, _: *Buffer, _: *Buffer, _: *Buffer, _: u32, _: u32, _: u32) bool {
        return false;
    }

    pub fn launchEmbeddingLookup(_: *Backend, _: *Buffer, _: *Buffer, _: *Buffer, _: u32, _: u32, _: u32) bool {
        return false;
    }

    pub fn launchEmbeddingGradient(_: *Backend, _: *Buffer, _: *Buffer, _: *Buffer, _: u32, _: u32, _: u32) bool {
        return false;
    }

    pub fn launchCachedSelfAttention(_: *Backend, _: *Buffer, _: *Buffer, _: *Buffer, _: *Buffer, _: *Buffer, _: *Buffer, _: u32, _: u32, _: u32) bool {
        return false;
    }
};

const ROCmMatrix = struct {
    rows: usize,
    cols: usize,
    buffer: ?*ROCm.Buffer,
    host_data: []f32,
    allocator: Allocator,
    backend: *ROCm.Backend,
    owner: *ROCmBackend,
    dirty: bool,
    gpu_dirty: bool,

    pub fn init(allocator: Allocator, rows: usize, cols: usize, owner: *ROCmBackend) !*ROCmMatrix {
        const element_count = dimensions.elementCount(rows, cols) catch return error.OutOfMemory;
        if (element_count == 0) return error.OutOfMemory;
        _ = dimensions.product(&.{ element_count, @sizeOf(f32) }) catch return error.OutOfMemory;

        const rocm_matrix = try allocator.create(ROCmMatrix);
        errdefer allocator.destroy(rocm_matrix);

        const host_data = try allocator.alloc(f32, element_count);
        errdefer allocator.free(host_data);
        @memset(host_data, 0);

        const backend_ref = owner.backend orelse return error.ROCmInitializationFailed;
        const buffer = ROCm.createBuffer(backend_ref, element_count) orelse return error.BufferCreationFailed;
        errdefer ROCm.destroyBuffer(backend_ref, buffer);

        rocm_matrix.* = .{
            .rows = rows,
            .cols = cols,
            .buffer = buffer,
            .host_data = host_data,
            .allocator = allocator,
            .backend = backend_ref,
            .owner = owner,
            .dirty = true,
            .gpu_dirty = false,
        };
        owner.stats.buffer_allocations += 1;
        return rocm_matrix;
    }

    pub fn deinit(self: *ROCmMatrix) void {
        ROCm.destroyBuffer(self.backend, self.buffer);
        self.allocator.free(self.host_data);
        self.allocator.destroy(self);
    }

    pub fn syncToGPU(self: *ROCmMatrix) bool {
        if (!self.dirty) return true;

        const buffer = self.buffer orelse return false;
        if (!ROCm.bufferUploadF32(self.backend, buffer, self.host_data)) {
            return false;
        }
        self.owner.stats.host_to_device_transfers += 1;
        self.owner.stats.host_to_device_bytes += self.host_data.len * @sizeOf(f32);

        self.dirty = false;
        self.gpu_dirty = false;
        return true;
    }

    pub fn syncToCPU(self: *ROCmMatrix) bool {
        if (!self.gpu_dirty) return true;

        if (!self.owner.synchronizeQueued()) return false;

        const buffer = self.buffer orelse return false;
        if (!ROCm.bufferDownloadF32(self.backend, buffer, self.host_data)) {
            return false;
        }
        self.owner.stats.device_to_host_transfers += 1;
        self.owner.stats.device_to_host_bytes += self.host_data.len * @sizeOf(f32);

        self.dirty = false;
        self.gpu_dirty = false;
        return true;
    }

    pub fn markHostModified(self: *ROCmMatrix) void {
        self.dirty = true;
        self.gpu_dirty = false;
    }

    pub fn markGPUModified(self: *ROCmMatrix) void {
        self.dirty = false;
        self.gpu_dirty = true;
    }
};

pub const ROCmBackend = struct {
    allocator: Allocator,
    backend: ?*ROCm.Backend,
    device_name: [256]u8,
    stats: backend.RuntimeStats,
    batch_active: bool,
    synchronized_kernel_count: usize,
    deferred_matrices: std.ArrayList(*Matrix),

    pub fn init(allocator: Allocator) (ROCmError || Allocator.Error)!*ROCmBackend {
        if (!enable_rocm) {
            return error.ROCmNotEnabled;
        }
        if (@import("builtin").os.tag != .linux) {
            return error.ROCmNotSupported;
        }

        const backend_ref = ROCm.createBackend() orelse return error.ROCmInitializationFailed;
        const rocm_backend = allocator.create(ROCmBackend) catch |err| {
            ROCm.destroyBackend(backend_ref);
            return err;
        };

        rocm_backend.* = .{
            .allocator = allocator,
            .backend = backend_ref,
            .device_name = [_]u8{0} ** 256,
            .stats = .{},
            .batch_active = false,
            .synchronized_kernel_count = 0,
            .deferred_matrices = .empty,
        };
        _ = ROCm.deviceName(backend_ref, rocm_backend.device_name[0..]);

        return rocm_backend;
    }

    pub fn deinit(self: *ROCmBackend) void {
        self.batch_active = false;
        _ = self.synchronizeQueued();
        self.releaseDeferredMatrices();
        self.deferred_matrices.deinit(self.allocator);
        ROCm.destroyBackend(self.backend);
        self.allocator.destroy(self);
    }

    pub fn getDeviceName(self: *const ROCmBackend) []const u8 {
        return std.mem.sliceTo(&self.device_name, 0);
    }

    pub fn runtimeStats(ptr: *anyopaque) backend.RuntimeStats {
        const self = @as(*ROCmBackend, @ptrCast(@alignCast(ptr)));
        return self.stats;
    }

    pub fn resetRuntimeStats(ptr: *anyopaque) void {
        const self = @as(*ROCmBackend, @ptrCast(@alignCast(ptr)));
        self.stats = .{};
        self.synchronized_kernel_count = 0;
    }

    pub fn beginBatch(ptr: *anyopaque) !void {
        const self = @as(*ROCmBackend, @ptrCast(@alignCast(ptr)));
        if (self.batch_active) return error.BatchAlreadyActive;
        self.batch_active = true;
    }

    pub fn endBatch(ptr: *anyopaque) !void {
        const self = @as(*ROCmBackend, @ptrCast(@alignCast(ptr)));
        if (!self.batch_active) return error.NoActiveBatch;
        self.batch_active = false;
        if (!self.synchronizeQueued()) return error.CommandExecutionFailed;
    }

    pub fn synchronize(ptr: *anyopaque) !void {
        const self = @as(*ROCmBackend, @ptrCast(@alignCast(ptr)));
        if (!self.synchronizeQueued()) return error.CommandExecutionFailed;
    }

    fn completeSubmittedKernel(self: *ROCmBackend) bool {
        self.stats.kernel_launches += 1;
        if (self.batch_active) return true;
        return self.synchronizeQueued();
    }

    fn synchronizeQueued(self: *ROCmBackend) bool {
        if (self.synchronized_kernel_count != self.stats.kernel_launches) {
            const backend_ref = self.backend orelse return false;
            if (!ROCm.synchronize(backend_ref)) return false;
            self.stats.synchronizations += 1;
            self.synchronized_kernel_count = self.stats.kernel_launches;
        }
        self.releaseDeferredMatrices();
        return true;
    }

    fn releaseDeferredMatrices(self: *ROCmBackend) void {
        for (self.deferred_matrices.items) |matrix| {
            deinitMatrixNow(matrix);
        }
        self.deferred_matrices.clearRetainingCapacity();
    }

    fn deinitMatrixNow(matrix: *Matrix) void {
        const rocm_matrix = getROCmMatrix(matrix);
        const allocator = rocm_matrix.allocator;
        rocm_matrix.deinit();
        allocator.destroy(matrix);
    }

    fn getROCmMatrix(matrix: *const Matrix) *ROCmMatrix {
        return @as(*ROCmMatrix, @ptrCast(@alignCast(matrix.impl_data)));
    }

    fn elementCount(matrix: *const Matrix) usize {
        return matrix.rows * matrix.cols;
    }

    fn toU32(value: usize) u32 {
        return std.math.cast(u32, value) orelse @panic("ROCm backend dimension exceeds u32 range");
    }

    fn ensureHostData(rocm_matrix: *ROCmMatrix) void {
        if (!rocm_matrix.syncToCPU()) {
            @panic("ROCm buffer download failed");
        }
    }

    fn dispatchMatrixMultiply(self: *ROCmBackend, a: *const Matrix, b: *const Matrix, result: *Matrix) bool {
        const backend_ref = self.backend orelse return false;
        const a_rocm = getROCmMatrix(a);
        const b_rocm = getROCmMatrix(b);
        const result_rocm = getROCmMatrix(result);

        if (!a_rocm.syncToGPU() or !b_rocm.syncToGPU()) return false;
        const a_buffer = a_rocm.buffer orelse return false;
        const b_buffer = b_rocm.buffer orelse return false;
        const result_buffer = result_rocm.buffer orelse return false;

        const kind = ROCm.launchMatrixMultiply(backend_ref, a_buffer, b_buffer, result_buffer, toU32(a.rows), toU32(a.cols), toU32(b.cols));
        if (kind == .failed) return false;

        if (kind == .vendor) self.stats.vendor_gemm_launches += 1;
        if (!self.completeSubmittedKernel()) return false;
        result_rocm.markGPUModified();
        return true;
    }

    fn dispatchBatchedMatrixMultiply(self: *ROCmBackend, a: *const Matrix, b: *const Matrix, result: *Matrix, batch: usize, a_rows: usize, a_cols: usize, b_rows: usize, b_cols: usize, transpose_a: bool, transpose_b: bool) bool {
        const backend_ref = self.backend orelse return false;
        const a_rocm = getROCmMatrix(a);
        const b_rocm = getROCmMatrix(b);
        const result_rocm = getROCmMatrix(result);
        if (!a_rocm.syncToGPU() or !b_rocm.syncToGPU()) return false;
        const a_buffer = a_rocm.buffer orelse return false;
        const b_buffer = b_rocm.buffer orelse return false;
        const result_buffer = result_rocm.buffer orelse return false;
        if (!ROCm.launchBatchedMatrixMultiply(backend_ref, a_buffer, b_buffer, result_buffer, toU32(batch), toU32(a_rows), toU32(a_cols), toU32(b_rows), toU32(b_cols), transpose_a, transpose_b)) return false;
        if (!self.completeSubmittedKernel()) return false;
        result_rocm.markGPUModified();
        return true;
    }

    fn dispatchPermuteBatchHeads(self: *ROCmBackend, input: *const Matrix, result: *Matrix, batch: usize, tokens: usize, heads: usize, width: usize, split: bool) bool {
        const backend_ref = self.backend orelse return false;
        const source = getROCmMatrix(input);
        const destination = getROCmMatrix(result);
        if (!source.syncToGPU()) return false;
        const source_buffer = source.buffer orelse return false;
        const destination_buffer = destination.buffer orelse return false;
        if (!ROCm.launchPermuteBatchHeads(backend_ref, source_buffer, destination_buffer, toU32(batch), toU32(tokens), toU32(heads), toU32(width), split)) return false;
        if (!self.completeSubmittedKernel()) return false;
        destination.markGPUModified();
        return true;
    }

    fn dispatchBinaryMatrixKernel(self: *ROCmBackend, kernel_id: ROCm.KernelId, a: *const Matrix, b: *const Matrix, result: *Matrix) bool {
        const backend_ref = self.backend orelse return false;
        const a_rocm = getROCmMatrix(a);
        const b_rocm = getROCmMatrix(b);
        const result_rocm = getROCmMatrix(result);

        if (!a_rocm.syncToGPU() or !b_rocm.syncToGPU()) return false;
        const a_buffer = a_rocm.buffer orelse return false;
        const b_buffer = b_rocm.buffer orelse return false;
        const result_buffer = result_rocm.buffer orelse return false;

        if (!ROCm.launchBinaryKernel(backend_ref, kernel_id, a_buffer, b_buffer, result_buffer, toU32(a.rows), toU32(a.cols))) {
            return false;
        }

        if (!self.completeSubmittedKernel()) return false;
        result_rocm.markGPUModified();
        return true;
    }

    fn dispatchScale(self: *ROCmBackend, matrix: *const Matrix, scalar: f64, result: *Matrix) bool {
        const backend_ref = self.backend orelse return false;
        const matrix_rocm = getROCmMatrix(matrix);
        const result_rocm = getROCmMatrix(result);

        if (!matrix_rocm.syncToGPU()) return false;
        const matrix_buffer = matrix_rocm.buffer orelse return false;
        const result_buffer = result_rocm.buffer orelse return false;

        if (!ROCm.launchScale(backend_ref, matrix_buffer, result_buffer, @floatCast(scalar), toU32(matrix.rows), toU32(matrix.cols))) {
            return false;
        }

        if (!self.completeSubmittedKernel()) return false;
        result_rocm.markGPUModified();
        return true;
    }

    fn dispatchOptimizerUpdate(self: *ROCmBackend, parameter: *Matrix, gradient: *const Matrix, first_moment: *Matrix, second_moment: *Matrix, total_squares: *const Matrix, config: backend.OptimizerUpdateConfig) bool {
        const backend_ref = self.backend orelse return false;
        const parameter_data = getROCmMatrix(parameter);
        const gradient_data = getROCmMatrix(gradient);
        const first_moment_data = getROCmMatrix(first_moment);
        const second_moment_data = getROCmMatrix(second_moment);
        const total_squares_data = getROCmMatrix(total_squares);
        if (!parameter_data.syncToGPU() or !gradient_data.syncToGPU() or
            !first_moment_data.syncToGPU() or !second_moment_data.syncToGPU() or
            !total_squares_data.syncToGPU())
        {
            return false;
        }
        const parameter_buffer = parameter_data.buffer orelse return false;
        const gradient_buffer = gradient_data.buffer orelse return false;
        const first_moment_buffer = first_moment_data.buffer orelse return false;
        const second_moment_buffer = second_moment_data.buffer orelse return false;
        const total_squares_buffer = total_squares_data.buffer orelse return false;
        if (!ROCm.launchOptimizerUpdate(backend_ref, parameter_buffer, gradient_buffer, first_moment_buffer, second_moment_buffer, total_squares_buffer, config, toU32(parameter.rows * parameter.cols))) return false;
        if (!self.completeSubmittedKernel()) return false;
        parameter_data.markGPUModified();
        first_moment_data.markGPUModified();
        second_moment_data.markGPUModified();
        return true;
    }

    fn dispatchTranspose(self: *ROCmBackend, matrix: *const Matrix, result: *Matrix) bool {
        const backend_ref = self.backend orelse return false;
        const matrix_rocm = getROCmMatrix(matrix);
        const result_rocm = getROCmMatrix(result);

        if (!matrix_rocm.syncToGPU()) return false;
        const matrix_buffer = matrix_rocm.buffer orelse return false;
        const result_buffer = result_rocm.buffer orelse return false;

        if (!ROCm.launchTranspose(backend_ref, matrix_buffer, result_buffer, toU32(matrix.rows), toU32(matrix.cols))) {
            return false;
        }

        if (!self.completeSubmittedKernel()) return false;
        result_rocm.markGPUModified();
        return true;
    }

    fn dispatchSumRows(self: *ROCmBackend, matrix: *const Matrix, result: *Matrix) bool {
        const backend_ref = self.backend orelse return false;
        const matrix_rocm = getROCmMatrix(matrix);
        const result_rocm = getROCmMatrix(result);

        if (!matrix_rocm.syncToGPU()) return false;
        const matrix_buffer = matrix_rocm.buffer orelse return false;
        const result_buffer = result_rocm.buffer orelse return false;

        if (!ROCm.launchSumRows(backend_ref, matrix_buffer, result_buffer, toU32(matrix.rows), toU32(matrix.cols))) {
            return false;
        }

        if (!self.completeSubmittedKernel()) return false;
        result_rocm.markGPUModified();
        return true;
    }

    fn dispatchExtractBatch(self: *ROCmBackend, matrix: *const Matrix, start: usize, result: *Matrix) bool {
        const backend_ref = self.backend orelse return false;
        const matrix_rocm = getROCmMatrix(matrix);
        const result_rocm = getROCmMatrix(result);

        if (!matrix_rocm.syncToGPU()) return false;
        const matrix_buffer = matrix_rocm.buffer orelse return false;
        const result_buffer = result_rocm.buffer orelse return false;

        if (!ROCm.launchExtractBatch(backend_ref, matrix_buffer, result_buffer, toU32(start), toU32(result.rows), toU32(result.cols))) {
            return false;
        }

        if (!self.completeSubmittedKernel()) return false;
        result_rocm.markGPUModified();
        return true;
    }

    fn dispatchUnaryKernel(self: *ROCmBackend, kernel_id: ROCm.KernelId, matrix: *const Matrix, result: *Matrix) bool {
        const backend_ref = self.backend orelse return false;
        const matrix_rocm = getROCmMatrix(matrix);
        const result_rocm = getROCmMatrix(result);

        if (!matrix_rocm.syncToGPU()) return false;
        const matrix_buffer = matrix_rocm.buffer orelse return false;
        const result_buffer = result_rocm.buffer orelse return false;

        if (!ROCm.launchUnaryKernel(backend_ref, kernel_id, matrix_buffer, result_buffer, toU32(elementCount(matrix)))) {
            return false;
        }

        if (!self.completeSubmittedKernel()) return false;
        result_rocm.markGPUModified();
        return true;
    }

    fn dispatchSoftmax(self: *ROCmBackend, matrix: *const Matrix, result: *Matrix) bool {
        const backend_ref = self.backend orelse return false;
        const matrix_rocm = getROCmMatrix(matrix);
        const result_rocm = getROCmMatrix(result);

        if (!matrix_rocm.syncToGPU()) return false;
        const matrix_buffer = matrix_rocm.buffer orelse return false;
        const result_buffer = result_rocm.buffer orelse return false;

        if (!ROCm.launchSoftmaxRows(backend_ref, matrix_buffer, result_buffer, toU32(matrix.rows), toU32(matrix.cols))) {
            return false;
        }

        if (!self.completeSubmittedKernel()) return false;
        result_rocm.markGPUModified();
        return true;
    }

    fn dispatchGatedKernel(self: *ROCmBackend, kernel_id: ROCm.KernelId, linear_part: *const Matrix, gating_part: *const Matrix, result: *Matrix) bool {
        const backend_ref = self.backend orelse return false;
        const linear_rocm = getROCmMatrix(linear_part);
        const gating_rocm = getROCmMatrix(gating_part);
        const result_rocm = getROCmMatrix(result);

        if (!linear_rocm.syncToGPU() or !gating_rocm.syncToGPU()) return false;
        const linear_buffer = linear_rocm.buffer orelse return false;
        const gating_buffer = gating_rocm.buffer orelse return false;
        const result_buffer = result_rocm.buffer orelse return false;

        if (!ROCm.launchGatedKernel(backend_ref, kernel_id, linear_buffer, gating_buffer, result_buffer, toU32(elementCount(linear_part)))) {
            return false;
        }

        if (!self.completeSubmittedKernel()) return false;
        result_rocm.markGPUModified();
        return true;
    }

    fn dispatchLayerNorm(self: *ROCmBackend, matrix: *const Matrix, gamma: *const Matrix, beta: *const Matrix, epsilon: f64, result: *Matrix) bool {
        const backend_ref = self.backend orelse return false;
        const matrix_rocm = getROCmMatrix(matrix);
        const gamma_rocm = getROCmMatrix(gamma);
        const beta_rocm = getROCmMatrix(beta);
        const result_rocm = getROCmMatrix(result);

        if (!matrix_rocm.syncToGPU() or !gamma_rocm.syncToGPU() or !beta_rocm.syncToGPU()) return false;
        const matrix_buffer = matrix_rocm.buffer orelse return false;
        const gamma_buffer = gamma_rocm.buffer orelse return false;
        const beta_buffer = beta_rocm.buffer orelse return false;
        const result_buffer = result_rocm.buffer orelse return false;

        if (!ROCm.launchLayerNorm(backend_ref, matrix_buffer, gamma_buffer, beta_buffer, result_buffer, toU32(matrix.rows), toU32(matrix.cols), @floatCast(epsilon))) {
            return false;
        }

        if (!self.completeSubmittedKernel()) return false;
        result_rocm.markGPUModified();
        return true;
    }

    fn dispatchLayerNormBackward(self: *ROCmBackend, input: *const Matrix, gamma: *const Matrix, output_gradient: *const Matrix, epsilon: f64, gradients: backend.LayerNormGradients) bool {
        const backend_ref = self.backend orelse return false;
        const input_data = getROCmMatrix(input);
        const gamma_data = getROCmMatrix(gamma);
        const output_gradient_data = getROCmMatrix(output_gradient);
        const input_gradient_data = getROCmMatrix(gradients.input);
        const gamma_gradient_data = getROCmMatrix(gradients.gamma);
        const beta_gradient_data = getROCmMatrix(gradients.beta);
        if (!input_data.syncToGPU() or !gamma_data.syncToGPU() or !output_gradient_data.syncToGPU()) return false;
        const input_buffer = input_data.buffer orelse return false;
        const gamma_buffer = gamma_data.buffer orelse return false;
        const output_gradient_buffer = output_gradient_data.buffer orelse return false;
        const input_gradient_buffer = input_gradient_data.buffer orelse return false;
        const gamma_gradient_buffer = gamma_gradient_data.buffer orelse return false;
        const beta_gradient_buffer = beta_gradient_data.buffer orelse return false;
        if (!ROCm.launchLayerNormBackward(backend_ref, input_buffer, gamma_buffer, output_gradient_buffer, input_gradient_buffer, gamma_gradient_buffer, beta_gradient_buffer, toU32(input.rows), toU32(input.cols), @floatCast(epsilon))) {
            return false;
        }
        if (!self.completeSubmittedKernel()) return false;
        input_gradient_data.markGPUModified();
        gamma_gradient_data.markGPUModified();
        beta_gradient_data.markGPUModified();
        return true;
    }

    fn dispatchCausalSelfAttention(self: *ROCmBackend, query: *const Matrix, key: *const Matrix, value: *const Matrix, heads: usize, result: *Matrix) bool {
        const backend_ref = self.backend orelse return false;
        const query_rocm = getROCmMatrix(query);
        const key_rocm = getROCmMatrix(key);
        const value_rocm = getROCmMatrix(value);
        const result_rocm = getROCmMatrix(result);
        if (!query_rocm.syncToGPU() or !key_rocm.syncToGPU() or !value_rocm.syncToGPU()) return false;
        const query_buffer = query_rocm.buffer orelse return false;
        const key_buffer = key_rocm.buffer orelse return false;
        const value_buffer = value_rocm.buffer orelse return false;
        const result_buffer = result_rocm.buffer orelse return false;

        if (!ROCm.launchCausalSelfAttention(backend_ref, query_buffer, key_buffer, value_buffer, result_buffer, toU32(query.rows), toU32(query.cols), toU32(heads))) {
            return false;
        }
        if (!self.completeSubmittedKernel()) return false;
        result_rocm.markGPUModified();
        return true;
    }

    fn dispatchCausalSelfAttentionBackward(
        self: *ROCmBackend,
        query: *const Matrix,
        key: *const Matrix,
        value: *const Matrix,
        output_gradient: *const Matrix,
        heads: usize,
        probabilities: *Matrix,
        score_gradients: *Matrix,
        gradients: backend.AttentionGradients,
    ) bool {
        const backend_ref = self.backend orelse return false;
        const query_data = getROCmMatrix(query);
        const key_data = getROCmMatrix(key);
        const value_data = getROCmMatrix(value);
        const output_gradient_data = getROCmMatrix(output_gradient);
        const probabilities_data = getROCmMatrix(probabilities);
        const score_gradients_data = getROCmMatrix(score_gradients);
        const query_gradient_data = getROCmMatrix(gradients.query);
        const key_gradient_data = getROCmMatrix(gradients.key);
        const value_gradient_data = getROCmMatrix(gradients.value);
        if (!query_data.syncToGPU() or !key_data.syncToGPU() or !value_data.syncToGPU() or !output_gradient_data.syncToGPU()) return false;
        const query_buffer = query_data.buffer orelse return false;
        const key_buffer = key_data.buffer orelse return false;
        const value_buffer = value_data.buffer orelse return false;
        const output_gradient_buffer = output_gradient_data.buffer orelse return false;
        const probabilities_buffer = probabilities_data.buffer orelse return false;
        const score_gradients_buffer = score_gradients_data.buffer orelse return false;
        const query_gradient_buffer = query_gradient_data.buffer orelse return false;
        const key_gradient_buffer = key_gradient_data.buffer orelse return false;
        const value_gradient_buffer = value_gradient_data.buffer orelse return false;
        if (!ROCm.launchCausalAttentionProbabilitiesBackward(backend_ref, query_buffer, key_buffer, value_buffer, output_gradient_buffer, probabilities_buffer, score_gradients_buffer, toU32(query.rows), toU32(query.cols), toU32(heads))) {
            return false;
        }
        if (!self.completeSubmittedKernel()) return false;
        probabilities_data.markGPUModified();
        score_gradients_data.markGPUModified();
        if (!ROCm.launchCausalAttentionInputGradients(backend_ref, query_buffer, key_buffer, output_gradient_buffer, probabilities_buffer, score_gradients_buffer, query_gradient_buffer, key_gradient_buffer, value_gradient_buffer, toU32(query.rows), toU32(query.cols), toU32(heads))) {
            return false;
        }
        if (!self.completeSubmittedKernel()) return false;
        query_gradient_data.markGPUModified();
        key_gradient_data.markGPUModified();
        value_gradient_data.markGPUModified();
        return true;
    }

    fn dispatchEmbeddingLookup(self: *ROCmBackend, table: *const Matrix, indices: *const Matrix, result: *Matrix) bool {
        const backend_ref = self.backend orelse return false;
        const table_data = getROCmMatrix(table);
        const indices_data = getROCmMatrix(indices);
        const result_data = getROCmMatrix(result);
        if (!table_data.syncToGPU() or !indices_data.syncToGPU()) return false;
        const table_buffer = table_data.buffer orelse return false;
        const indices_buffer = indices_data.buffer orelse return false;
        const result_buffer = result_data.buffer orelse return false;
        if (!ROCm.launchEmbeddingLookup(backend_ref, table_buffer, indices_buffer, result_buffer, toU32(table.rows), toU32(indices.rows), toU32(table.cols))) return false;
        if (!self.completeSubmittedKernel()) return false;
        result_data.markGPUModified();
        return true;
    }

    fn dispatchEmbeddingGradient(self: *ROCmBackend, indices: *const Matrix, output_gradient: *const Matrix, result: *Matrix) bool {
        const backend_ref = self.backend orelse return false;
        const indices_data = getROCmMatrix(indices);
        const output_gradient_data = getROCmMatrix(output_gradient);
        const result_data = getROCmMatrix(result);
        if (!indices_data.syncToGPU() or !output_gradient_data.syncToGPU()) return false;
        const indices_buffer = indices_data.buffer orelse return false;
        const output_gradient_buffer = output_gradient_data.buffer orelse return false;
        const result_buffer = result_data.buffer orelse return false;
        if (!ROCm.launchEmbeddingGradient(backend_ref, indices_buffer, output_gradient_buffer, result_buffer, toU32(result.rows), toU32(indices.rows), toU32(result.cols))) return false;
        if (!self.completeSubmittedKernel()) return false;
        result_data.markGPUModified();
        return true;
    }

    fn dispatchCachedSelfAttention(self: *ROCmBackend, query: *const Matrix, key: *const Matrix, value: *const Matrix, key_cache: *Matrix, value_cache: *Matrix, position: usize, heads: usize, result: *Matrix) bool {
        const backend_ref = self.backend orelse return false;
        const query_data = getROCmMatrix(query);
        const key_data = getROCmMatrix(key);
        const value_data = getROCmMatrix(value);
        const key_cache_data = getROCmMatrix(key_cache);
        const value_cache_data = getROCmMatrix(value_cache);
        const result_data = getROCmMatrix(result);
        if (!query_data.syncToGPU() or !key_data.syncToGPU() or !value_data.syncToGPU() or !key_cache_data.syncToGPU() or !value_cache_data.syncToGPU()) return false;
        const query_buffer = query_data.buffer orelse return false;
        const key_buffer = key_data.buffer orelse return false;
        const value_buffer = value_data.buffer orelse return false;
        const key_cache_buffer = key_cache_data.buffer orelse return false;
        const value_cache_buffer = value_cache_data.buffer orelse return false;
        const result_buffer = result_data.buffer orelse return false;
        if (!ROCm.launchCachedSelfAttention(backend_ref, query_buffer, key_buffer, value_buffer, key_cache_buffer, value_cache_buffer, result_buffer, toU32(position), toU32(query.cols), toU32(heads))) return false;
        if (!self.completeSubmittedKernel()) return false;
        key_cache_data.markGPUModified();
        value_cache_data.markGPUModified();
        result_data.markGPUModified();
        return true;
    }

    pub fn initMatrix(ptr: *anyopaque, allocator: Allocator, rows: usize, cols: usize) error{OutOfMemory}!*Matrix {
        const self = @as(*ROCmBackend, @ptrCast(@alignCast(ptr)));
        const rocm_matrix = ROCmMatrix.init(allocator, rows, cols, self) catch |err| {
            std.log.warn("ROCm matrix initialization failed: {s}", .{@errorName(err)});
            return error.OutOfMemory;
        };
        errdefer rocm_matrix.deinit();

        const matrix = try allocator.create(Matrix);
        matrix.* = .{
            .rows = rows,
            .cols = cols,
            .backend = undefined,
            .impl_data = rocm_matrix,
        };
        return matrix;
    }

    pub fn deinitMatrix(ptr: *anyopaque, matrix: *Matrix) void {
        const self = @as(*ROCmBackend, @ptrCast(@alignCast(ptr)));
        if (self.batch_active) {
            self.deferred_matrices.append(self.allocator, matrix) catch {
                _ = self.synchronizeQueued();
                deinitMatrixNow(matrix);
            };
            return;
        }
        deinitMatrixNow(matrix);
    }

    pub fn copyMatrix(ptr: *anyopaque, source: *const Matrix, allocator: Allocator) error{OutOfMemory}!*Matrix {
        const source_rocm = getROCmMatrix(source);
        const result = try initMatrix(ptr, allocator, source.rows, source.cols);
        const result_rocm = getROCmMatrix(result);

        ensureHostData(source_rocm);
        @memcpy(result_rocm.host_data, source_rocm.host_data);
        result_rocm.markHostModified();
        return result;
    }

    pub fn getMatrixElement(_: *anyopaque, matrix: *const Matrix, row: usize, col: usize) f64 {
        if (row >= matrix.rows or col >= matrix.cols) {
            @panic("Index out of bounds");
        }

        const rocm_matrix = getROCmMatrix(matrix);
        ensureHostData(rocm_matrix);
        return @floatCast(rocm_matrix.host_data[row * matrix.cols + col]);
    }

    pub fn writeMatrixF32(_: *anyopaque, matrix: *Matrix, values: []const f32) bool {
        const rocm_matrix = getROCmMatrix(matrix);
        if (values.len != rocm_matrix.host_data.len) return false;
        @memcpy(rocm_matrix.host_data, values);
        rocm_matrix.markHostModified();
        return true;
    }

    pub fn readMatrixF32(_: *anyopaque, matrix: *const Matrix, values: []f32) bool {
        const rocm_matrix = getROCmMatrix(matrix);
        if (values.len != rocm_matrix.host_data.len) return false;
        ensureHostData(rocm_matrix);
        @memcpy(values, rocm_matrix.host_data);
        return true;
    }

    pub fn setMatrixElement(_: *anyopaque, matrix: *Matrix, row: usize, col: usize, value: f64) void {
        if (row >= matrix.rows or col >= matrix.cols) {
            @panic("Index out of bounds");
        }

        const rocm_matrix = getROCmMatrix(matrix);
        ensureHostData(rocm_matrix);
        rocm_matrix.host_data[row * matrix.cols + col] = @floatCast(value);
        rocm_matrix.markHostModified();
    }

    pub fn fillMatrix(_: *anyopaque, matrix: *Matrix, value: f64) void {
        const rocm_matrix = getROCmMatrix(matrix);
        for (rocm_matrix.host_data) |*element| {
            element.* = @floatCast(value);
        }
        rocm_matrix.markHostModified();
    }

    pub fn dotProduct(ptr: *anyopaque, a: *const Matrix, b: *const Matrix, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!*Matrix {
        if (a.cols != b.rows) {
            return error.DimensionMismatch;
        }

        const self = @as(*ROCmBackend, @ptrCast(@alignCast(ptr)));
        const result = try initMatrix(ptr, allocator, a.rows, b.cols);
        const a_rocm = getROCmMatrix(a);
        const b_rocm = getROCmMatrix(b);
        const result_rocm = getROCmMatrix(result);

        if (self.dispatchMatrixMultiply(a, b, result)) {
            return result;
        }

        ensureHostData(a_rocm);
        ensureHostData(b_rocm);
        for (0..a.rows) |i| {
            for (0..b.cols) |j| {
                var sum: f32 = 0.0;
                for (0..a.cols) |k| {
                    sum += a_rocm.host_data[i * a.cols + k] * b_rocm.host_data[k * b.cols + j];
                }
                result_rocm.host_data[i * b.cols + j] = sum;
            }
        }
        result_rocm.markHostModified();
        return result;
    }

    pub fn batchedDotProduct(ptr: *anyopaque, a: *const Matrix, b: *const Matrix, allocator: Allocator, batch: usize, a_rows: usize, a_cols: usize, b_rows: usize, b_cols: usize, transpose_a: bool, transpose_b: bool) error{ OutOfMemory, DimensionMismatch }!*Matrix {
        const output_rows = if (transpose_a) a_cols else a_rows;
        const inner_a = if (transpose_a) a_rows else a_cols;
        const inner_b = if (transpose_b) b_cols else b_rows;
        const output_cols = if (transpose_b) b_rows else b_cols;
        const a_elements = dimensions.elementCount(a.rows, a.cols) catch return error.DimensionMismatch;
        const b_elements = dimensions.elementCount(b.rows, b.cols) catch return error.DimensionMismatch;
        if (batch == 0 or inner_a != inner_b or
            !dimensions.matches(a_elements, &.{ batch, a_rows, a_cols }) or
            !dimensions.matches(b_elements, &.{ batch, b_rows, b_cols }))
        {
            return error.DimensionMismatch;
        }

        const self = @as(*ROCmBackend, @ptrCast(@alignCast(ptr)));
        const result_rows = dimensions.product(&.{ batch, output_rows }) catch return error.OutOfMemory;
        const result = try initMatrix(ptr, allocator, result_rows, output_cols);
        const a_rocm = getROCmMatrix(a);
        const b_rocm = getROCmMatrix(b);
        const result_rocm = getROCmMatrix(result);
        if (self.dispatchBatchedMatrixMultiply(a, b, result, batch, a_rows, a_cols, b_rows, b_cols, transpose_a, transpose_b)) return result;

        ensureHostData(a_rocm);
        ensureHostData(b_rocm);
        for (0..batch) |batch_index| {
            const a_offset = batch_index * a_rows * a_cols;
            const b_offset = batch_index * b_rows * b_cols;
            const result_offset = batch_index * output_rows * output_cols;
            for (0..output_rows) |row| {
                for (0..output_cols) |col| {
                    var sum: f32 = 0;
                    for (0..inner_a) |inner| {
                        const a_index = a_offset + if (transpose_a) inner * a_cols + row else row * a_cols + inner;
                        const b_index = b_offset + if (transpose_b) col * b_cols + inner else inner * b_cols + col;
                        sum += a_rocm.host_data[a_index] * b_rocm.host_data[b_index];
                    }
                    result_rocm.host_data[result_offset + row * output_cols + col] = sum;
                }
            }
        }
        result_rocm.markHostModified();
        return result;
    }

    pub fn permuteBatchHeads(ptr: *anyopaque, input: *const Matrix, allocator: Allocator, batch: usize, tokens: usize, heads: usize, width: usize, split: bool) error{ OutOfMemory, DimensionMismatch }!*Matrix {
        const input_elements = dimensions.elementCount(input.rows, input.cols) catch return error.DimensionMismatch;
        if (batch == 0 or tokens == 0 or heads == 0 or width == 0 or
            !dimensions.matches(input_elements, &.{ batch, tokens, heads, width }))
        {
            return error.DimensionMismatch;
        }
        const result_rows = dimensions.product(if (split) &.{ batch, heads, tokens } else &.{ batch, tokens }) catch return error.OutOfMemory;
        const result_cols = if (split)
            width
        else
            dimensions.product(&.{ heads, width }) catch return error.OutOfMemory;
        const self = @as(*ROCmBackend, @ptrCast(@alignCast(ptr)));
        const result = try initMatrix(ptr, allocator, result_rows, result_cols);
        const source = getROCmMatrix(input);
        const destination = getROCmMatrix(result);
        if (self.dispatchPermuteBatchHeads(input, result, batch, tokens, heads, width, split)) return result;
        ensureHostData(source);
        for (0..batch) |batch_index| for (0..heads) |head| for (0..tokens) |token| for (0..width) |channel| {
            const token_major = (((batch_index * tokens + token) * heads + head) * width + channel);
            const head_major = (((batch_index * heads + head) * tokens + token) * width + channel);
            if (split) destination.host_data[head_major] = source.host_data[token_major] else destination.host_data[token_major] = source.host_data[head_major];
        };
        destination.markHostModified();
        return result;
    }

    pub fn add(ptr: *anyopaque, a: *const Matrix, b: *const Matrix, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!*Matrix {
        if (a.rows != b.rows or a.cols != b.cols) {
            return error.DimensionMismatch;
        }

        const self = @as(*ROCmBackend, @ptrCast(@alignCast(ptr)));
        const result = try initMatrix(ptr, allocator, a.rows, a.cols);
        const a_rocm = getROCmMatrix(a);
        const b_rocm = getROCmMatrix(b);
        const result_rocm = getROCmMatrix(result);

        if (self.dispatchBinaryMatrixKernel(.matrix_add, a, b, result)) {
            return result;
        }

        ensureHostData(a_rocm);
        ensureHostData(b_rocm);
        for (0..a.rows * a.cols) |i| {
            result_rocm.host_data[i] = a_rocm.host_data[i] + b_rocm.host_data[i];
        }
        result_rocm.markHostModified();
        return result;
    }

    pub fn addRowBias(ptr: *anyopaque, matrix: *const Matrix, bias: *const Matrix, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!*Matrix {
        if (bias.rows != 1 or bias.cols != matrix.cols) return error.DimensionMismatch;

        const self = @as(*ROCmBackend, @ptrCast(@alignCast(ptr)));
        const result = try initMatrix(ptr, allocator, matrix.rows, matrix.cols);
        const input_data = getROCmMatrix(matrix);
        const bias_data = getROCmMatrix(bias);
        const result_data = getROCmMatrix(result);

        if (self.dispatchBinaryMatrixKernel(.matrix_add_row_bias, matrix, bias, result)) {
            return result;
        }

        ensureHostData(input_data);
        ensureHostData(bias_data);
        for (0..matrix.rows) |row| {
            for (0..matrix.cols) |col| {
                const index = row * matrix.cols + col;
                result_data.host_data[index] = input_data.host_data[index] + bias_data.host_data[col];
            }
        }
        result_data.markHostModified();
        return result;
    }

    pub fn subtract(ptr: *anyopaque, a: *const Matrix, b: *const Matrix, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!*Matrix {
        if (a.rows != b.rows or a.cols != b.cols) {
            return error.DimensionMismatch;
        }

        const self = @as(*ROCmBackend, @ptrCast(@alignCast(ptr)));
        const result = try initMatrix(ptr, allocator, a.rows, a.cols);
        const a_rocm = getROCmMatrix(a);
        const b_rocm = getROCmMatrix(b);
        const result_rocm = getROCmMatrix(result);

        if (self.dispatchBinaryMatrixKernel(.matrix_subtract, a, b, result)) {
            return result;
        }

        ensureHostData(a_rocm);
        ensureHostData(b_rocm);
        for (0..a.rows * a.cols) |i| {
            result_rocm.host_data[i] = a_rocm.host_data[i] - b_rocm.host_data[i];
        }
        result_rocm.markHostModified();
        return result;
    }

    pub fn elementWiseMultiply(ptr: *anyopaque, a: *const Matrix, b: *const Matrix, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!*Matrix {
        if (a.rows != b.rows or a.cols != b.cols) {
            return error.DimensionMismatch;
        }

        const self = @as(*ROCmBackend, @ptrCast(@alignCast(ptr)));
        const result = try initMatrix(ptr, allocator, a.rows, a.cols);
        const a_rocm = getROCmMatrix(a);
        const b_rocm = getROCmMatrix(b);
        const result_rocm = getROCmMatrix(result);

        if (self.dispatchBinaryMatrixKernel(.matrix_element_wise_multiply, a, b, result)) {
            return result;
        }

        ensureHostData(a_rocm);
        ensureHostData(b_rocm);
        for (0..a.rows * a.cols) |i| {
            result_rocm.host_data[i] = a_rocm.host_data[i] * b_rocm.host_data[i];
        }
        result_rocm.markHostModified();
        return result;
    }

    pub fn scale(ptr: *anyopaque, matrix: *const Matrix, scalar: f64, allocator: Allocator) error{OutOfMemory}!*Matrix {
        const self = @as(*ROCmBackend, @ptrCast(@alignCast(ptr)));
        const result = try initMatrix(ptr, allocator, matrix.rows, matrix.cols);
        const matrix_rocm = getROCmMatrix(matrix);
        const result_rocm = getROCmMatrix(result);

        if (self.dispatchScale(matrix, scalar, result)) {
            return result;
        }

        ensureHostData(matrix_rocm);
        for (0..matrix.rows * matrix.cols) |i| {
            result_rocm.host_data[i] = matrix_rocm.host_data[i] * @as(f32, @floatCast(scalar));
        }
        result_rocm.markHostModified();
        return result;
    }

    pub fn optimizerUpdate(ptr: *anyopaque, parameter: *Matrix, gradient: *const Matrix, first_moment: *Matrix, second_moment: *Matrix, total_squares: *const Matrix, config: backend.OptimizerUpdateConfig) error{DimensionMismatch}!void {
        if (parameter.rows != gradient.rows or parameter.cols != gradient.cols or
            parameter.rows != first_moment.rows or parameter.cols != first_moment.cols or
            parameter.rows != second_moment.rows or parameter.cols != second_moment.cols or
            total_squares.rows != 1 or total_squares.cols != 1)
        {
            return error.DimensionMismatch;
        }
        const self = @as(*ROCmBackend, @ptrCast(@alignCast(ptr)));
        if (self.dispatchOptimizerUpdate(parameter, gradient, first_moment, second_moment, total_squares, config)) return;

        const parameter_data = getROCmMatrix(parameter);
        const gradient_data = getROCmMatrix(gradient);
        const first_moment_data = getROCmMatrix(first_moment);
        const second_moment_data = getROCmMatrix(second_moment);
        const total_squares_data = getROCmMatrix(total_squares);
        ensureHostData(parameter_data);
        ensureHostData(gradient_data);
        ensureHostData(first_moment_data);
        ensureHostData(second_moment_data);
        ensureHostData(total_squares_data);
        backend.applyOptimizerUpdate(parameter_data.host_data, gradient_data.host_data, first_moment_data.host_data, second_moment_data.host_data, total_squares_data.host_data[0], config);
        parameter_data.markHostModified();
        first_moment_data.markHostModified();
        second_moment_data.markHostModified();
    }

    pub fn sumRows(ptr: *anyopaque, matrix: *const Matrix, allocator: Allocator) error{OutOfMemory}!*Matrix {
        const self = @as(*ROCmBackend, @ptrCast(@alignCast(ptr)));
        const result = try initMatrix(ptr, allocator, 1, matrix.cols);
        const matrix_rocm = getROCmMatrix(matrix);
        const result_rocm = getROCmMatrix(result);

        if (self.dispatchSumRows(matrix, result)) {
            return result;
        }

        ensureHostData(matrix_rocm);
        for (0..matrix.cols) |j| {
            var sum: f32 = 0.0;
            for (0..matrix.rows) |i| {
                sum += matrix_rocm.host_data[i * matrix.cols + j];
            }
            result_rocm.host_data[j] = sum;
        }
        result_rocm.markHostModified();
        return result;
    }

    pub fn transpose(ptr: *anyopaque, matrix: *const Matrix, allocator: Allocator) error{OutOfMemory}!*Matrix {
        const self = @as(*ROCmBackend, @ptrCast(@alignCast(ptr)));
        const result = try initMatrix(ptr, allocator, matrix.cols, matrix.rows);
        const matrix_rocm = getROCmMatrix(matrix);
        const result_rocm = getROCmMatrix(result);

        if (self.dispatchTranspose(matrix, result)) {
            return result;
        }

        ensureHostData(matrix_rocm);
        for (0..matrix.rows) |i| {
            for (0..matrix.cols) |j| {
                result_rocm.host_data[j * matrix.rows + i] = matrix_rocm.host_data[i * matrix.cols + j];
            }
        }
        result_rocm.markHostModified();
        return result;
    }

    pub fn extractBatch(ptr: *anyopaque, matrix: *const Matrix, start: usize, end: usize, allocator: Allocator) error{ OutOfMemory, InvalidBatchIndices }!*Matrix {
        if (start >= end or end > matrix.rows) {
            return error.InvalidBatchIndices;
        }

        const self = @as(*ROCmBackend, @ptrCast(@alignCast(ptr)));
        const batch_rows = end - start;
        const result = try initMatrix(ptr, allocator, batch_rows, matrix.cols);
        const matrix_rocm = getROCmMatrix(matrix);
        const result_rocm = getROCmMatrix(result);

        if (self.dispatchExtractBatch(matrix, start, result)) {
            return result;
        }

        ensureHostData(matrix_rocm);
        const src_offset = start * matrix.cols;
        @memcpy(result_rocm.host_data, matrix_rocm.host_data[src_offset..][0 .. batch_rows * matrix.cols]);
        result_rocm.markHostModified();
        return result;
    }

    pub fn randomize(_: *anyopaque, matrix: *Matrix, min: f64, max: f64) void {
        const rocm_matrix = getROCmMatrix(matrix);
        var rng = std.Random.DefaultPrng.init(randomSeed());
        for (rocm_matrix.host_data) |*element| {
            const random_float = rng.random().float(f64);
            element.* = @floatCast(min + random_float * (max - min));
        }
        rocm_matrix.markHostModified();
    }

    pub fn applyActivation(ptr: *anyopaque, matrix: *const Matrix, activation: *const fn (f64) f64, allocator: Allocator) error{OutOfMemory}!*Matrix {
        const self = @as(*ROCmBackend, @ptrCast(@alignCast(ptr)));
        const result = try initMatrix(ptr, allocator, matrix.rows, matrix.cols);
        const matrix_rocm = getROCmMatrix(matrix);
        const result_rocm = getROCmMatrix(result);

        const kernel_id: ?ROCm.KernelId = if (activation == Activation.linear)
            .apply_linear
        else if (activation == Activation.linear_derivative)
            .apply_linear_derivative
        else if (activation == Activation.sigmoid)
            .apply_sigmoid
        else if (activation == Activation.sigmoid_derivative)
            .apply_sigmoid_derivative
        else if (activation == Activation.relu)
            .apply_relu
        else if (activation == Activation.relu_derivative)
            .apply_relu_derivative
        else if (activation == Activation.tanh)
            .apply_tanh
        else if (activation == Activation.tanh_derivative)
            .apply_tanh_derivative
        else if (activation == Activation.swish)
            .apply_swish
        else if (activation == Activation.swish_derivative)
            .apply_swish_derivative
        else if (activation == Activation.gelu)
            .apply_gelu
        else if (activation == Activation.gelu_derivative)
            .apply_gelu_derivative
        else
            null;

        if (kernel_id) |id| {
            if (self.dispatchUnaryKernel(id, matrix, result)) {
                return result;
            }
        }

        ensureHostData(matrix_rocm);
        for (0..matrix.rows * matrix.cols) |i| {
            result_rocm.host_data[i] = @floatCast(activation(@floatCast(matrix_rocm.host_data[i])));
        }
        result_rocm.markHostModified();
        return result;
    }

    pub fn applySoftmax(ptr: *anyopaque, matrix: *const Matrix, allocator: Allocator) error{OutOfMemory}!*Matrix {
        const self = @as(*ROCmBackend, @ptrCast(@alignCast(ptr)));
        const result = try initMatrix(ptr, allocator, matrix.rows, matrix.cols);
        const matrix_rocm = getROCmMatrix(matrix);
        const result_rocm = getROCmMatrix(result);

        if (self.dispatchSoftmax(matrix, result)) {
            return result;
        }

        ensureHostData(matrix_rocm);
        for (0..matrix.rows) |i| {
            var max_val: f32 = -std.math.inf(f32);
            for (0..matrix.cols) |j| {
                max_val = @max(max_val, matrix_rocm.host_data[i * matrix.cols + j]);
            }

            var sum: f32 = 0.0;
            for (0..matrix.cols) |j| {
                const idx = i * matrix.cols + j;
                const exp_val = std.math.exp(matrix_rocm.host_data[idx] - max_val);
                result_rocm.host_data[idx] = exp_val;
                sum += exp_val;
            }

            for (0..matrix.cols) |j| {
                const idx = i * matrix.cols + j;
                result_rocm.host_data[idx] /= sum;
            }
        }
        result_rocm.markHostModified();
        return result;
    }

    pub fn layerNorm(ptr: *anyopaque, matrix: *const Matrix, gamma: *const Matrix, beta: *const Matrix, epsilon: f64, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!*Matrix {
        if (gamma.rows != 1 or beta.rows != 1 or gamma.cols != matrix.cols or beta.cols != matrix.cols) {
            return error.DimensionMismatch;
        }

        const self = @as(*ROCmBackend, @ptrCast(@alignCast(ptr)));
        const result = try initMatrix(ptr, allocator, matrix.rows, matrix.cols);
        const input_data = getROCmMatrix(matrix);
        const gamma_data = getROCmMatrix(gamma);
        const beta_data = getROCmMatrix(beta);
        const result_data = getROCmMatrix(result);

        if (self.dispatchLayerNorm(matrix, gamma, beta, epsilon, result)) {
            return result;
        }

        ensureHostData(input_data);
        ensureHostData(gamma_data);
        ensureHostData(beta_data);
        const width = @as(f32, @floatFromInt(matrix.cols));
        const epsilon_f32: f32 = @floatCast(epsilon);
        for (0..matrix.rows) |row| {
            const offset = row * matrix.cols;
            var mean: f32 = 0.0;
            for (0..matrix.cols) |col| mean += input_data.host_data[offset + col];
            mean /= width;
            var variance: f32 = 0.0;
            for (0..matrix.cols) |col| {
                const centered = input_data.host_data[offset + col] - mean;
                variance += centered * centered;
            }
            variance /= width;
            const inverse_std = 1.0 / @sqrt(variance + epsilon_f32);
            for (0..matrix.cols) |col| {
                const normalized = (input_data.host_data[offset + col] - mean) * inverse_std;
                result_data.host_data[offset + col] = normalized * gamma_data.host_data[col] + beta_data.host_data[col];
            }
        }
        result_data.markHostModified();
        return result;
    }

    pub fn layerNormBackward(ptr: *anyopaque, input: *const Matrix, gamma: *const Matrix, output_gradient: *const Matrix, epsilon: f64, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!backend.LayerNormGradients {
        if (gamma.rows != 1 or gamma.cols != input.cols or
            output_gradient.rows != input.rows or output_gradient.cols != input.cols)
        {
            return error.DimensionMismatch;
        }
        const input_gradient = try initMatrix(ptr, allocator, input.rows, input.cols);
        errdefer deinitMatrix(ptr, input_gradient);
        const gamma_gradient = try initMatrix(ptr, allocator, 1, input.cols);
        errdefer deinitMatrix(ptr, gamma_gradient);
        const beta_gradient = try initMatrix(ptr, allocator, 1, input.cols);
        errdefer deinitMatrix(ptr, beta_gradient);
        const gradients: backend.LayerNormGradients = .{
            .input = input_gradient,
            .gamma = gamma_gradient,
            .beta = beta_gradient,
        };
        const self = @as(*ROCmBackend, @ptrCast(@alignCast(ptr)));
        if (self.dispatchLayerNormBackward(input, gamma, output_gradient, epsilon, gradients)) return gradients;

        const input_data = getROCmMatrix(input);
        const gamma_data = getROCmMatrix(gamma);
        const output_gradient_data = getROCmMatrix(output_gradient);
        const input_gradient_data = getROCmMatrix(input_gradient);
        const gamma_gradient_data = getROCmMatrix(gamma_gradient);
        const beta_gradient_data = getROCmMatrix(beta_gradient);
        ensureHostData(input_data);
        ensureHostData(gamma_data);
        ensureHostData(output_gradient_data);
        @memset(gamma_gradient_data.host_data, 0);
        @memset(beta_gradient_data.host_data, 0);
        const width = @as(f32, @floatFromInt(input.cols));
        const epsilon_f32: f32 = @floatCast(epsilon);
        for (0..input.rows) |row| {
            const offset = row * input.cols;
            var mean: f32 = 0;
            for (0..input.cols) |col| mean += input_data.host_data[offset + col];
            mean /= width;
            var variance: f32 = 0;
            for (0..input.cols) |col| {
                const centered = input_data.host_data[offset + col] - mean;
                variance += centered * centered;
            }
            variance /= width;
            const inverse_std = 1.0 / @sqrt(variance + epsilon_f32);
            var sum_gradient: f32 = 0;
            var sum_gradient_normalized: f32 = 0;
            for (0..input.cols) |col| {
                const normalized = (input_data.host_data[offset + col] - mean) * inverse_std;
                const gradient = output_gradient_data.host_data[offset + col];
                const normalized_gradient = gradient * gamma_data.host_data[col];
                sum_gradient += normalized_gradient;
                sum_gradient_normalized += normalized_gradient * normalized;
                gamma_gradient_data.host_data[col] += gradient * normalized;
                beta_gradient_data.host_data[col] += gradient;
            }
            for (0..input.cols) |col| {
                const normalized = (input_data.host_data[offset + col] - mean) * inverse_std;
                const normalized_gradient = output_gradient_data.host_data[offset + col] * gamma_data.host_data[col];
                input_gradient_data.host_data[offset + col] = inverse_std / width *
                    (width * normalized_gradient - sum_gradient - normalized * sum_gradient_normalized);
            }
        }
        input_gradient_data.markHostModified();
        gamma_gradient_data.markHostModified();
        beta_gradient_data.markHostModified();
        return gradients;
    }

    pub fn causalSelfAttention(ptr: *anyopaque, query: *const Matrix, key: *const Matrix, value: *const Matrix, heads: usize, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!*Matrix {
        if (heads == 0 or query.rows != key.rows or query.rows != value.rows or
            query.cols != key.cols or query.cols != value.cols or query.cols % heads != 0)
        {
            return error.DimensionMismatch;
        }

        const self = @as(*ROCmBackend, @ptrCast(@alignCast(ptr)));
        const result = try initMatrix(ptr, allocator, query.rows, query.cols);
        errdefer deinitMatrix(ptr, result);
        const query_data = getROCmMatrix(query);
        const key_data = getROCmMatrix(key);
        const value_data = getROCmMatrix(value);
        const result_data = getROCmMatrix(result);
        if (self.dispatchCausalSelfAttention(query, key, value, heads, result)) return result;

        ensureHostData(query_data);
        ensureHostData(key_data);
        ensureHostData(value_data);
        const probabilities = try allocator.alloc(f32, query.rows);
        defer allocator.free(probabilities);
        const head_width = query.cols / heads;
        const attention_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_width)));
        for (0..query.rows) |query_position| {
            for (0..heads) |head| {
                const channel_offset = head * head_width;
                var max_score = -std.math.inf(f32);
                for (0..query_position + 1) |key_position| {
                    var score: f32 = 0;
                    for (0..head_width) |channel| {
                        score += query_data.host_data[query_position * query.cols + channel_offset + channel] *
                            key_data.host_data[key_position * key.cols + channel_offset + channel];
                    }
                    score *= attention_scale;
                    probabilities[key_position] = score;
                    max_score = @max(max_score, score);
                }
                var denominator: f32 = 0;
                for (0..query_position + 1) |key_position| {
                    const probability = @exp(probabilities[key_position] - max_score);
                    probabilities[key_position] = probability;
                    denominator += probability;
                }
                for (0..head_width) |channel| {
                    var output: f32 = 0;
                    for (0..query_position + 1) |key_position| {
                        output += probabilities[key_position] / denominator *
                            value_data.host_data[key_position * value.cols + channel_offset + channel];
                    }
                    result_data.host_data[query_position * result.cols + channel_offset + channel] = output;
                }
            }
        }
        result_data.markHostModified();
        return result;
    }

    pub fn causalSelfAttentionBackward(ptr: *anyopaque, query: *const Matrix, key: *const Matrix, value: *const Matrix, output_gradient: *const Matrix, heads: usize, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!backend.AttentionGradients {
        if (heads == 0 or query.rows != key.rows or query.rows != value.rows or
            query.cols != key.cols or query.cols != value.cols or query.cols % heads != 0 or
            output_gradient.rows != query.rows or output_gradient.cols != query.cols)
        {
            return error.DimensionMismatch;
        }
        const query_gradient = try initMatrix(ptr, allocator, query.rows, query.cols);
        errdefer deinitMatrix(ptr, query_gradient);
        const key_gradient = try initMatrix(ptr, allocator, key.rows, key.cols);
        errdefer deinitMatrix(ptr, key_gradient);
        const value_gradient = try initMatrix(ptr, allocator, value.rows, value.cols);
        errdefer deinitMatrix(ptr, value_gradient);
        const probabilities = try initMatrix(ptr, allocator, heads * query.rows, query.rows);
        defer deinitMatrix(ptr, probabilities);
        const score_gradients = try initMatrix(ptr, allocator, heads * query.rows, query.rows);
        defer deinitMatrix(ptr, score_gradients);
        const gradients: backend.AttentionGradients = .{
            .query = query_gradient,
            .key = key_gradient,
            .value = value_gradient,
        };
        const self = @as(*ROCmBackend, @ptrCast(@alignCast(ptr)));
        if (self.dispatchCausalSelfAttentionBackward(query, key, value, output_gradient, heads, probabilities, score_gradients, gradients)) return gradients;

        const query_data = getROCmMatrix(query);
        const key_data = getROCmMatrix(key);
        const value_data = getROCmMatrix(value);
        const output_gradient_data = getROCmMatrix(output_gradient);
        const query_gradient_data = getROCmMatrix(query_gradient);
        const key_gradient_data = getROCmMatrix(key_gradient);
        const value_gradient_data = getROCmMatrix(value_gradient);
        ensureHostData(query_data);
        ensureHostData(key_data);
        ensureHostData(value_data);
        ensureHostData(output_gradient_data);
        @memset(query_gradient_data.host_data, 0);
        @memset(key_gradient_data.host_data, 0);
        @memset(value_gradient_data.host_data, 0);
        const host_probabilities = try allocator.alloc(f32, query.rows);
        defer allocator.free(host_probabilities);
        const probability_gradients = try allocator.alloc(f32, query.rows);
        defer allocator.free(probability_gradients);
        const head_width = query.cols / heads;
        const attention_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_width)));
        for (0..heads) |head| {
            const channel_offset = head * head_width;
            for (0..query.rows) |query_position| {
                var max_score = -std.math.inf(f32);
                for (0..query_position + 1) |key_position| {
                    var score: f32 = 0;
                    for (0..head_width) |channel| {
                        score += query_data.host_data[query_position * query.cols + channel_offset + channel] *
                            key_data.host_data[key_position * key.cols + channel_offset + channel];
                    }
                    score *= attention_scale;
                    host_probabilities[key_position] = score;
                    max_score = @max(max_score, score);
                }
                var denominator: f32 = 0;
                for (0..query_position + 1) |key_position| {
                    host_probabilities[key_position] = @exp(host_probabilities[key_position] - max_score);
                    denominator += host_probabilities[key_position];
                }
                var weighted_probability_gradient: f32 = 0;
                for (0..query_position + 1) |key_position| {
                    host_probabilities[key_position] /= denominator;
                    var probability_gradient: f32 = 0;
                    for (0..head_width) |channel| {
                        probability_gradient += output_gradient_data.host_data[query_position * query.cols + channel_offset + channel] *
                            value_data.host_data[key_position * value.cols + channel_offset + channel];
                    }
                    probability_gradients[key_position] = probability_gradient;
                    weighted_probability_gradient += host_probabilities[key_position] * probability_gradient;
                }
                for (0..query_position + 1) |key_position| {
                    const score_gradient = host_probabilities[key_position] *
                        (probability_gradients[key_position] - weighted_probability_gradient);
                    for (0..head_width) |channel| {
                        const query_index = query_position * query.cols + channel_offset + channel;
                        const key_index = key_position * key.cols + channel_offset + channel;
                        query_gradient_data.host_data[query_index] += score_gradient * key_data.host_data[key_index] * attention_scale;
                        key_gradient_data.host_data[key_index] += score_gradient * query_data.host_data[query_index] * attention_scale;
                        value_gradient_data.host_data[key_index] += host_probabilities[key_position] * output_gradient_data.host_data[query_index];
                    }
                }
            }
        }
        query_gradient_data.markHostModified();
        key_gradient_data.markHostModified();
        value_gradient_data.markHostModified();
        return gradients;
    }

    pub fn embeddingLookup(ptr: *anyopaque, table: *const Matrix, indices: *const Matrix, allocator: Allocator) error{ OutOfMemory, DimensionMismatch, InvalidIndex }!*Matrix {
        if (indices.cols != 1) return error.DimensionMismatch;
        const result = try initMatrix(ptr, allocator, indices.rows, table.cols);
        errdefer deinitMatrix(ptr, result);
        const self = @as(*ROCmBackend, @ptrCast(@alignCast(ptr)));
        if (self.dispatchEmbeddingLookup(table, indices, result)) return result;
        const table_data = getROCmMatrix(table);
        const indices_data = getROCmMatrix(indices);
        const result_data = getROCmMatrix(result);
        ensureHostData(table_data);
        ensureHostData(indices_data);
        for (0..indices.rows) |row| {
            const raw_index = indices_data.host_data[row];
            if (!std.math.isFinite(raw_index) or raw_index < 0 or @floor(raw_index) != raw_index or raw_index >= @as(f32, @floatFromInt(table.rows))) return error.InvalidIndex;
            const index: usize = @intFromFloat(raw_index);
            @memcpy(result_data.host_data[row * table.cols ..][0..table.cols], table_data.host_data[index * table.cols ..][0..table.cols]);
        }
        result_data.markHostModified();
        return result;
    }

    pub fn embeddingGradient(ptr: *anyopaque, indices: *const Matrix, output_gradient: *const Matrix, vocabulary_size: usize, allocator: Allocator) error{ OutOfMemory, DimensionMismatch, InvalidIndex }!*Matrix {
        if (indices.cols != 1 or indices.rows != output_gradient.rows or vocabulary_size == 0) return error.DimensionMismatch;
        const result = try initMatrix(ptr, allocator, vocabulary_size, output_gradient.cols);
        errdefer deinitMatrix(ptr, result);
        const self = @as(*ROCmBackend, @ptrCast(@alignCast(ptr)));
        if (self.dispatchEmbeddingGradient(indices, output_gradient, result)) return result;
        const indices_data = getROCmMatrix(indices);
        const output_gradient_data = getROCmMatrix(output_gradient);
        const result_data = getROCmMatrix(result);
        ensureHostData(indices_data);
        ensureHostData(output_gradient_data);
        @memset(result_data.host_data, 0);
        for (0..indices.rows) |row| {
            const raw_index = indices_data.host_data[row];
            if (!std.math.isFinite(raw_index) or raw_index < 0 or @floor(raw_index) != raw_index or raw_index >= @as(f32, @floatFromInt(vocabulary_size))) return error.InvalidIndex;
            const index: usize = @intFromFloat(raw_index);
            for (0..output_gradient.cols) |col| {
                result_data.host_data[index * output_gradient.cols + col] += output_gradient_data.host_data[row * output_gradient.cols + col];
            }
        }
        result_data.markHostModified();
        return result;
    }

    pub fn cachedSelfAttention(ptr: *anyopaque, query: *const Matrix, key: *const Matrix, value: *const Matrix, key_cache: *Matrix, value_cache: *Matrix, position: usize, heads: usize, allocator: Allocator) error{ OutOfMemory, DimensionMismatch, InvalidCachePosition }!*Matrix {
        if (query.rows != 1 or key.rows != 1 or value.rows != 1 or query.cols != key.cols or query.cols != value.cols or
            key_cache.rows != value_cache.rows or key_cache.cols != query.cols or value_cache.cols != query.cols or
            heads == 0 or query.cols % heads != 0 or position >= key_cache.rows)
        {
            return error.DimensionMismatch;
        }
        const result = try initMatrix(ptr, allocator, 1, query.cols);
        errdefer deinitMatrix(ptr, result);
        const self = @as(*ROCmBackend, @ptrCast(@alignCast(ptr)));
        if (self.dispatchCachedSelfAttention(query, key, value, key_cache, value_cache, position, heads, result)) return result;
        const query_data = getROCmMatrix(query);
        const key_data = getROCmMatrix(key);
        const value_data = getROCmMatrix(value);
        const key_cache_data = getROCmMatrix(key_cache);
        const value_cache_data = getROCmMatrix(value_cache);
        const result_data = getROCmMatrix(result);
        ensureHostData(query_data);
        ensureHostData(key_data);
        ensureHostData(value_data);
        ensureHostData(key_cache_data);
        ensureHostData(value_cache_data);
        @memcpy(key_cache_data.host_data[position * query.cols ..][0..query.cols], key_data.host_data[0..query.cols]);
        @memcpy(value_cache_data.host_data[position * query.cols ..][0..query.cols], value_data.host_data[0..query.cols]);
        const probabilities = try allocator.alloc(f32, position + 1);
        defer allocator.free(probabilities);
        const head_width = query.cols / heads;
        const attention_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_width)));
        for (0..heads) |head| {
            const offset = head * head_width;
            var max_score = -std.math.inf(f32);
            for (0..position + 1) |key_position| {
                var score: f32 = 0;
                for (0..head_width) |channel| score += query_data.host_data[offset + channel] * key_cache_data.host_data[key_position * query.cols + offset + channel];
                probabilities[key_position] = score * attention_scale;
                max_score = @max(max_score, probabilities[key_position]);
            }
            var denominator: f32 = 0;
            for (0..position + 1) |key_position| {
                probabilities[key_position] = @exp(probabilities[key_position] - max_score);
                denominator += probabilities[key_position];
            }
            for (0..head_width) |channel| {
                var output: f32 = 0;
                for (0..position + 1) |key_position| output += probabilities[key_position] / denominator * value_cache_data.host_data[key_position * query.cols + offset + channel];
                result_data.host_data[offset + channel] = output;
            }
        }
        key_cache_data.markHostModified();
        value_cache_data.markHostModified();
        result_data.markHostModified();
        return result;
    }

    pub fn applyGLU(ptr: *anyopaque, linear_part: *const Matrix, gating_part: *const Matrix, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!*Matrix {
        if (linear_part.rows != gating_part.rows or linear_part.cols != gating_part.cols) {
            return error.DimensionMismatch;
        }

        const self = @as(*ROCmBackend, @ptrCast(@alignCast(ptr)));
        const result = try initMatrix(ptr, allocator, linear_part.rows, linear_part.cols);
        const linear_rocm = getROCmMatrix(linear_part);
        const gating_rocm = getROCmMatrix(gating_part);
        const result_rocm = getROCmMatrix(result);

        if (self.dispatchGatedKernel(.apply_glu, linear_part, gating_part, result)) {
            return result;
        }

        ensureHostData(linear_rocm);
        ensureHostData(gating_rocm);
        for (0..linear_part.rows * linear_part.cols) |i| {
            const sigmoid = 1.0 / (1.0 + std.math.exp(-gating_rocm.host_data[i]));
            result_rocm.host_data[i] = linear_rocm.host_data[i] * sigmoid;
        }
        result_rocm.markHostModified();
        return result;
    }

    pub fn applySwiGLU(ptr: *anyopaque, linear_part: *const Matrix, gating_part: *const Matrix, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!*Matrix {
        if (linear_part.rows != gating_part.rows or linear_part.cols != gating_part.cols) {
            return error.DimensionMismatch;
        }

        const self = @as(*ROCmBackend, @ptrCast(@alignCast(ptr)));
        const result = try initMatrix(ptr, allocator, linear_part.rows, linear_part.cols);
        const linear_rocm = getROCmMatrix(linear_part);
        const gating_rocm = getROCmMatrix(gating_part);
        const result_rocm = getROCmMatrix(result);

        if (self.dispatchGatedKernel(.apply_swiglu, linear_part, gating_part, result)) {
            return result;
        }

        ensureHostData(linear_rocm);
        ensureHostData(gating_rocm);
        for (0..linear_part.rows * linear_part.cols) |i| {
            const gate = gating_rocm.host_data[i];
            const sigmoid = 1.0 / (1.0 + std.math.exp(-gate));
            result_rocm.host_data[i] = linear_rocm.host_data[i] * gate * sigmoid;
        }
        result_rocm.markHostModified();
        return result;
    }

    pub fn getBackendType(_: *anyopaque) BackendType {
        return .ROCm;
    }
};

pub fn createROCmBackend(allocator: Allocator) !*anyopaque {
    const rocm_backend = try ROCmBackend.init(allocator);
    errdefer rocm_backend.deinit();
    return @ptrCast(rocm_backend);
}

const CPUBackend = @import("cpu_backend.zig").CPUBackend;
const testing = std.testing;

fn initROCmBackendForTest(allocator: Allocator) !*ROCmBackend {
    if (!enable_rocm or @import("builtin").os.tag != .linux) {
        return error.SkipZigTest;
    }

    return ROCmBackend.init(allocator) catch |err| switch (err) {
        error.ROCmNotEnabled, error.ROCmNotSupported, error.ROCmDeviceNotFound, error.ROCmInitializationFailed => error.SkipZigTest,
        else => err,
    };
}

fn fillMatrixPair(
    cpu_ptr: *anyopaque,
    rocm_ptr: *anyopaque,
    cpu_matrix: *Matrix,
    rocm_matrix: *Matrix,
    values: []const f64,
) void {
    for (values, 0..) |value, index| {
        const row = index / cpu_matrix.cols;
        const col = index % cpu_matrix.cols;
        CPUBackend.setMatrixElement(cpu_ptr, cpu_matrix, row, col, value);
        ROCmBackend.setMatrixElement(rocm_ptr, rocm_matrix, row, col, value);
    }
}

fn expectMatricesClose(
    cpu_ptr: *anyopaque,
    rocm_ptr: *anyopaque,
    cpu_matrix: *const Matrix,
    rocm_matrix: *const Matrix,
    tolerance: f64,
) !void {
    try testing.expectEqual(cpu_matrix.rows, rocm_matrix.rows);
    try testing.expectEqual(cpu_matrix.cols, rocm_matrix.cols);

    for (0..cpu_matrix.rows) |row| {
        for (0..cpu_matrix.cols) |col| {
            const expected = CPUBackend.getMatrixElement(cpu_ptr, cpu_matrix, row, col);
            const actual = ROCmBackend.getMatrixElement(rocm_ptr, rocm_matrix, row, col);
            try testing.expectApproxEqAbs(expected, actual, tolerance);
        }
    }
}

test "rocm backend matches cpu backend for matrix operations" {
    const allocator = testing.allocator;

    const cpu_impl = try CPUBackend.init(allocator);
    defer cpu_impl.deinit();
    const rocm_impl = try initROCmBackendForTest(allocator);
    defer rocm_impl.deinit();

    const cpu_ptr: *anyopaque = @ptrCast(cpu_impl);
    const rocm_ptr: *anyopaque = @ptrCast(rocm_impl);

    const a_values = [_]f64{ 1.5, -2.0, 3.25, 0.5, 4.0, -1.25 };
    const b_values = [_]f64{ 2.0, -1.0, 0.25, 3.0, -2.5, 1.75 };
    const c_values = [_]f64{ -0.5, 2.0, 0.25, 3.0, -4.0, 1.25 };

    const cpu_a = try CPUBackend.initMatrix(cpu_ptr, allocator, 2, 3);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_a);
    const rocm_a = try ROCmBackend.initMatrix(rocm_ptr, allocator, 2, 3);
    defer ROCmBackend.deinitMatrix(rocm_ptr, rocm_a);
    fillMatrixPair(cpu_ptr, rocm_ptr, cpu_a, rocm_a, &a_values);

    const cpu_b = try CPUBackend.initMatrix(cpu_ptr, allocator, 3, 2);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_b);
    const rocm_b = try ROCmBackend.initMatrix(rocm_ptr, allocator, 3, 2);
    defer ROCmBackend.deinitMatrix(rocm_ptr, rocm_b);
    fillMatrixPair(cpu_ptr, rocm_ptr, cpu_b, rocm_b, &b_values);

    const cpu_c = try CPUBackend.initMatrix(cpu_ptr, allocator, 2, 3);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_c);
    const rocm_c = try ROCmBackend.initMatrix(rocm_ptr, allocator, 2, 3);
    defer ROCmBackend.deinitMatrix(rocm_ptr, rocm_c);
    fillMatrixPair(cpu_ptr, rocm_ptr, cpu_c, rocm_c, &c_values);

    const tolerance = 1e-4;

    const cpu_dot = try CPUBackend.dotProduct(cpu_ptr, cpu_a, cpu_b, allocator);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_dot);
    const rocm_dot = try ROCmBackend.dotProduct(rocm_ptr, rocm_a, rocm_b, allocator);
    defer ROCmBackend.deinitMatrix(rocm_ptr, rocm_dot);
    try expectMatricesClose(cpu_ptr, rocm_ptr, cpu_dot, rocm_dot, tolerance);

    const cpu_add = try CPUBackend.add(cpu_ptr, cpu_a, cpu_c, allocator);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_add);
    const rocm_add = try ROCmBackend.add(rocm_ptr, rocm_a, rocm_c, allocator);
    defer ROCmBackend.deinitMatrix(rocm_ptr, rocm_add);
    try expectMatricesClose(cpu_ptr, rocm_ptr, cpu_add, rocm_add, tolerance);

    const cpu_subtract = try CPUBackend.subtract(cpu_ptr, cpu_a, cpu_c, allocator);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_subtract);
    const rocm_subtract = try ROCmBackend.subtract(rocm_ptr, rocm_a, rocm_c, allocator);
    defer ROCmBackend.deinitMatrix(rocm_ptr, rocm_subtract);
    try expectMatricesClose(cpu_ptr, rocm_ptr, cpu_subtract, rocm_subtract, tolerance);

    const cpu_multiply = try CPUBackend.elementWiseMultiply(cpu_ptr, cpu_a, cpu_c, allocator);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_multiply);
    const rocm_multiply = try ROCmBackend.elementWiseMultiply(rocm_ptr, rocm_a, rocm_c, allocator);
    defer ROCmBackend.deinitMatrix(rocm_ptr, rocm_multiply);
    try expectMatricesClose(cpu_ptr, rocm_ptr, cpu_multiply, rocm_multiply, tolerance);

    const cpu_scale = try CPUBackend.scale(cpu_ptr, cpu_a, -0.75, allocator);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_scale);
    const rocm_scale = try ROCmBackend.scale(rocm_ptr, rocm_a, -0.75, allocator);
    defer ROCmBackend.deinitMatrix(rocm_ptr, rocm_scale);
    try expectMatricesClose(cpu_ptr, rocm_ptr, cpu_scale, rocm_scale, tolerance);

    const cpu_sum_rows = try CPUBackend.sumRows(cpu_ptr, cpu_a, allocator);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_sum_rows);
    const rocm_sum_rows = try ROCmBackend.sumRows(rocm_ptr, rocm_a, allocator);
    defer ROCmBackend.deinitMatrix(rocm_ptr, rocm_sum_rows);
    try expectMatricesClose(cpu_ptr, rocm_ptr, cpu_sum_rows, rocm_sum_rows, tolerance);

    const cpu_transpose = try CPUBackend.transpose(cpu_ptr, cpu_a, allocator);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_transpose);
    const rocm_transpose = try ROCmBackend.transpose(rocm_ptr, rocm_a, allocator);
    defer ROCmBackend.deinitMatrix(rocm_ptr, rocm_transpose);
    try expectMatricesClose(cpu_ptr, rocm_ptr, cpu_transpose, rocm_transpose, tolerance);

    const cpu_batch = try CPUBackend.extractBatch(cpu_ptr, cpu_a, 1, 2, allocator);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_batch);
    const rocm_batch = try ROCmBackend.extractBatch(rocm_ptr, rocm_a, 1, 2, allocator);
    defer ROCmBackend.deinitMatrix(rocm_ptr, rocm_batch);
    try expectMatricesClose(cpu_ptr, rocm_ptr, cpu_batch, rocm_batch, tolerance);
}

test "rocm backend matches cpu backend for activation operations" {
    const allocator = testing.allocator;

    const cpu_impl = try CPUBackend.init(allocator);
    defer cpu_impl.deinit();
    const rocm_impl = try initROCmBackendForTest(allocator);
    defer rocm_impl.deinit();

    const cpu_ptr: *anyopaque = @ptrCast(cpu_impl);
    const rocm_ptr: *anyopaque = @ptrCast(rocm_impl);

    const a_values = [_]f64{ -1.5, -0.25, 0.0, 0.75, 2.0, 4.0 };
    const gate_values = [_]f64{ 2.0, -1.0, 0.5, 3.0, -2.5, 1.25 };

    const cpu_a = try CPUBackend.initMatrix(cpu_ptr, allocator, 2, 3);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_a);
    const rocm_a = try ROCmBackend.initMatrix(rocm_ptr, allocator, 2, 3);
    defer ROCmBackend.deinitMatrix(rocm_ptr, rocm_a);
    fillMatrixPair(cpu_ptr, rocm_ptr, cpu_a, rocm_a, &a_values);

    const cpu_gate = try CPUBackend.initMatrix(cpu_ptr, allocator, 2, 3);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_gate);
    const rocm_gate = try ROCmBackend.initMatrix(rocm_ptr, allocator, 2, 3);
    defer ROCmBackend.deinitMatrix(rocm_ptr, rocm_gate);
    fillMatrixPair(cpu_ptr, rocm_ptr, cpu_gate, rocm_gate, &gate_values);

    const tolerance = 1e-4;

    const cpu_sigmoid = try CPUBackend.applyActivation(cpu_ptr, cpu_a, Activation.sigmoid, allocator);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_sigmoid);
    const rocm_sigmoid = try ROCmBackend.applyActivation(rocm_ptr, rocm_a, Activation.sigmoid, allocator);
    defer ROCmBackend.deinitMatrix(rocm_ptr, rocm_sigmoid);
    try expectMatricesClose(cpu_ptr, rocm_ptr, cpu_sigmoid, rocm_sigmoid, tolerance);

    const cpu_relu = try CPUBackend.applyActivation(cpu_ptr, cpu_a, Activation.relu, allocator);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_relu);
    const rocm_relu = try ROCmBackend.applyActivation(rocm_ptr, rocm_a, Activation.relu, allocator);
    defer ROCmBackend.deinitMatrix(rocm_ptr, rocm_relu);
    try expectMatricesClose(cpu_ptr, rocm_ptr, cpu_relu, rocm_relu, tolerance);

    const cpu_tanh = try CPUBackend.applyActivation(cpu_ptr, cpu_a, Activation.tanh, allocator);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_tanh);
    const rocm_tanh = try ROCmBackend.applyActivation(rocm_ptr, rocm_a, Activation.tanh, allocator);
    defer ROCmBackend.deinitMatrix(rocm_ptr, rocm_tanh);
    try expectMatricesClose(cpu_ptr, rocm_ptr, cpu_tanh, rocm_tanh, tolerance);

    const cpu_swish = try CPUBackend.applyActivation(cpu_ptr, cpu_a, Activation.swish, allocator);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_swish);
    const rocm_swish = try ROCmBackend.applyActivation(rocm_ptr, rocm_a, Activation.swish, allocator);
    defer ROCmBackend.deinitMatrix(rocm_ptr, rocm_swish);
    try expectMatricesClose(cpu_ptr, rocm_ptr, cpu_swish, rocm_swish, tolerance);

    const cpu_softmax = try CPUBackend.applySoftmax(cpu_ptr, cpu_a, allocator);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_softmax);
    const rocm_softmax = try ROCmBackend.applySoftmax(rocm_ptr, rocm_a, allocator);
    defer ROCmBackend.deinitMatrix(rocm_ptr, rocm_softmax);
    try expectMatricesClose(cpu_ptr, rocm_ptr, cpu_softmax, rocm_softmax, tolerance);

    const cpu_glu = try CPUBackend.applyGLU(cpu_ptr, cpu_a, cpu_gate, allocator);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_glu);
    const rocm_glu = try ROCmBackend.applyGLU(rocm_ptr, rocm_a, rocm_gate, allocator);
    defer ROCmBackend.deinitMatrix(rocm_ptr, rocm_glu);
    try expectMatricesClose(cpu_ptr, rocm_ptr, cpu_glu, rocm_glu, tolerance);

    const cpu_swiglu = try CPUBackend.applySwiGLU(cpu_ptr, cpu_a, cpu_gate, allocator);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_swiglu);
    const rocm_swiglu = try ROCmBackend.applySwiGLU(rocm_ptr, rocm_a, rocm_gate, allocator);
    defer ROCmBackend.deinitMatrix(rocm_ptr, rocm_swiglu);
    try expectMatricesClose(cpu_ptr, rocm_ptr, cpu_swiglu, rocm_swiglu, tolerance);
}
