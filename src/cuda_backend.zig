const std = @import("std");
const Allocator = std.mem.Allocator;
const backend = @import("backend.zig");
const ComputeBackend = backend.ComputeBackend;
const BackendType = backend.BackendType;
const Matrix = backend.Matrix;
const randomSeed = @import("matrix.zig").randomSeed;
const Activation = @import("activation.zig").Activation;

const build_options = @import("build_options");
const enable_cuda = build_options.enable_cuda;
const kernel_source = @embedFile("cuda/kernels.cu");

pub const CUDAError = error{
    CUDANotEnabled,
    CUDANotSupported,
    CUDADeviceNotFound,
    CUDAInitializationFailed,
    KernelCompilationFailed,
    FunctionNotFound,
    BufferCreationFailed,
    DataTransferError,
    CommandExecutionFailed,
    InvalidBatchIndices,
};

const CUDA = if (enable_cuda) struct {
    const c = @cImport({
        @cInclude("cuda_wrapper.h");
    });

    pub const Backend = anyopaque;
    pub const Buffer = anyopaque;

    pub const KernelId = enum(c_int) {
        matrix_add = c.ZIG_NN_CUDA_KERNEL_MATRIX_ADD,
        matrix_subtract = c.ZIG_NN_CUDA_KERNEL_MATRIX_SUBTRACT,
        matrix_element_wise_multiply = c.ZIG_NN_CUDA_KERNEL_MATRIX_ELEMENT_WISE_MULTIPLY,
        apply_sigmoid = c.ZIG_NN_CUDA_KERNEL_APPLY_SIGMOID,
        apply_relu = c.ZIG_NN_CUDA_KERNEL_APPLY_RELU,
        apply_tanh = c.ZIG_NN_CUDA_KERNEL_APPLY_TANH,
        apply_swish = c.ZIG_NN_CUDA_KERNEL_APPLY_SWISH,
        apply_glu = c.ZIG_NN_CUDA_KERNEL_APPLY_GLU,
        apply_swiglu = c.ZIG_NN_CUDA_KERNEL_APPLY_SWIGLU,
        apply_linear = c.ZIG_NN_CUDA_KERNEL_APPLY_LINEAR,
        apply_linear_derivative = c.ZIG_NN_CUDA_KERNEL_APPLY_LINEAR_DERIVATIVE,
        apply_sigmoid_derivative = c.ZIG_NN_CUDA_KERNEL_APPLY_SIGMOID_DERIVATIVE,
        apply_relu_derivative = c.ZIG_NN_CUDA_KERNEL_APPLY_RELU_DERIVATIVE,
        apply_tanh_derivative = c.ZIG_NN_CUDA_KERNEL_APPLY_TANH_DERIVATIVE,
        apply_swish_derivative = c.ZIG_NN_CUDA_KERNEL_APPLY_SWISH_DERIVATIVE,
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
        const result = c.cuda_backend_create(kernel_source, error_buffer[0..].ptr, error_buffer.len);
        if (result == null) {
            logError("CUDA backend creation failed", error_buffer[0..]);
        }
        return fromC(Backend, result);
    }

    pub fn destroyBackend(backend_ref: ?*Backend) void {
        c.cuda_backend_destroy(if (backend_ref) |value| toC(value) else null);
    }

    pub fn deviceName(backend_ref: *Backend, output: []u8) bool {
        return c.cuda_backend_device_name(toC(backend_ref), output.ptr, output.len) != 0;
    }

    pub fn synchronize(backend_ref: *Backend) bool {
        return c.cuda_backend_synchronize(toC(backend_ref)) != 0;
    }

    pub fn createBuffer(backend_ref: *Backend, count: usize) ?*Buffer {
        return fromC(Buffer, c.cuda_backend_create_buffer(toC(backend_ref), count));
    }

    pub fn destroyBuffer(backend_ref: *Backend, buffer: ?*Buffer) void {
        c.cuda_backend_destroy_buffer(toC(backend_ref), if (buffer) |value| toC(value) else null);
    }

    pub fn bufferUploadF32(backend_ref: *Backend, buffer: *Buffer, data: []const f32) bool {
        return c.cuda_buffer_upload_f32(toC(backend_ref), toC(buffer), data.ptr, data.len) != 0;
    }

    pub fn bufferDownloadF32(backend_ref: *Backend, buffer: *Buffer, data: []f32) bool {
        return c.cuda_buffer_download_f32(toC(backend_ref), toC(buffer), data.ptr, data.len) != 0;
    }

    pub fn launchMatrixMultiply(backend_ref: *Backend, a: *Buffer, b: *Buffer, result: *Buffer, a_rows: u32, a_cols: u32, b_cols: u32) MatmulKind {
        return switch (c.cuda_launch_matrix_multiply(toC(backend_ref), toC(a), toC(b), toC(result), a_rows, a_cols, b_cols)) {
            1 => .custom,
            2 => .vendor,
            else => .failed,
        };
    }

    pub fn launchBinaryKernel(backend_ref: *Backend, kernel_id: KernelId, a: *Buffer, b: *Buffer, result: *Buffer, rows: u32, cols: u32) bool {
        return c.cuda_launch_binary_kernel(toC(backend_ref), @intFromEnum(kernel_id), toC(a), toC(b), toC(result), rows, cols) != 0;
    }

    pub fn launchScale(backend_ref: *Backend, input: *Buffer, result: *Buffer, scalar: f32, rows: u32, cols: u32) bool {
        return c.cuda_launch_scale(toC(backend_ref), toC(input), toC(result), scalar, rows, cols) != 0;
    }

    pub fn launchTranspose(backend_ref: *Backend, input: *Buffer, result: *Buffer, rows: u32, cols: u32) bool {
        return c.cuda_launch_transpose(toC(backend_ref), toC(input), toC(result), rows, cols) != 0;
    }

    pub fn launchSumRows(backend_ref: *Backend, input: *Buffer, result: *Buffer, rows: u32, cols: u32) bool {
        return c.cuda_launch_sum_rows(toC(backend_ref), toC(input), toC(result), rows, cols) != 0;
    }

    pub fn launchExtractBatch(backend_ref: *Backend, input: *Buffer, result: *Buffer, start: u32, batch_rows: u32, cols: u32) bool {
        return c.cuda_launch_extract_batch(toC(backend_ref), toC(input), toC(result), start, batch_rows, cols) != 0;
    }

    pub fn launchUnaryKernel(backend_ref: *Backend, kernel_id: KernelId, input: *Buffer, result: *Buffer, size: u32) bool {
        return c.cuda_launch_unary_kernel(toC(backend_ref), @intFromEnum(kernel_id), toC(input), toC(result), size) != 0;
    }

    pub fn launchSoftmaxRows(backend_ref: *Backend, input: *Buffer, result: *Buffer, rows: u32, cols: u32) bool {
        return c.cuda_launch_softmax_rows(toC(backend_ref), toC(input), toC(result), rows, cols) != 0;
    }

    pub fn launchGatedKernel(backend_ref: *Backend, kernel_id: KernelId, linear: *Buffer, gating: *Buffer, result: *Buffer, size: u32) bool {
        return c.cuda_launch_gated_kernel(toC(backend_ref), @intFromEnum(kernel_id), toC(linear), toC(gating), toC(result), size) != 0;
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

    pub fn launchBinaryKernel(_: *Backend, _: KernelId, _: *Buffer, _: *Buffer, _: *Buffer, _: u32, _: u32) bool {
        return false;
    }

    pub fn launchScale(_: *Backend, _: *Buffer, _: *Buffer, _: f32, _: u32, _: u32) bool {
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
};

const CUDAMatrix = struct {
    rows: usize,
    cols: usize,
    buffer: ?*CUDA.Buffer,
    host_data: []f32,
    allocator: Allocator,
    backend: *CUDA.Backend,
    owner: *CUDABackend,
    dirty: bool,
    gpu_dirty: bool,

    pub fn init(allocator: Allocator, rows: usize, cols: usize, owner: *CUDABackend) !*CUDAMatrix {
        const cuda_matrix = try allocator.create(CUDAMatrix);
        errdefer allocator.destroy(cuda_matrix);

        const host_data = try allocator.alloc(f32, rows * cols);
        errdefer allocator.free(host_data);
        @memset(host_data, 0);

        const backend_ref = owner.backend orelse return error.CUDAInitializationFailed;
        const buffer = CUDA.createBuffer(backend_ref, rows * cols) orelse return error.BufferCreationFailed;
        errdefer CUDA.destroyBuffer(backend_ref, buffer);

        cuda_matrix.* = .{
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
        return cuda_matrix;
    }

    pub fn deinit(self: *CUDAMatrix) void {
        CUDA.destroyBuffer(self.backend, self.buffer);
        self.allocator.free(self.host_data);
        self.allocator.destroy(self);
    }

    pub fn syncToGPU(self: *CUDAMatrix) bool {
        if (!self.dirty) return true;

        const buffer = self.buffer orelse return false;
        if (!CUDA.bufferUploadF32(self.backend, buffer, self.host_data)) {
            return false;
        }
        self.owner.stats.host_to_device_transfers += 1;
        self.owner.stats.host_to_device_bytes += self.host_data.len * @sizeOf(f32);

        self.dirty = false;
        self.gpu_dirty = false;
        return true;
    }

    pub fn syncToCPU(self: *CUDAMatrix) bool {
        if (!self.gpu_dirty) return true;

        if (!self.owner.synchronizeQueued()) return false;

        const buffer = self.buffer orelse return false;
        if (!CUDA.bufferDownloadF32(self.backend, buffer, self.host_data)) {
            return false;
        }
        self.owner.stats.device_to_host_transfers += 1;
        self.owner.stats.device_to_host_bytes += self.host_data.len * @sizeOf(f32);

        self.dirty = false;
        self.gpu_dirty = false;
        return true;
    }

    pub fn markHostModified(self: *CUDAMatrix) void {
        self.dirty = true;
        self.gpu_dirty = false;
    }

    pub fn markGPUModified(self: *CUDAMatrix) void {
        self.dirty = false;
        self.gpu_dirty = true;
    }
};

pub const CUDABackend = struct {
    allocator: Allocator,
    backend: ?*CUDA.Backend,
    device_name: [256]u8,
    stats: backend.RuntimeStats,
    batch_active: bool,
    synchronized_kernel_count: usize,
    deferred_matrices: std.ArrayList(*Matrix),

    pub fn init(allocator: Allocator) (CUDAError || Allocator.Error)!*CUDABackend {
        if (!enable_cuda) {
            return error.CUDANotEnabled;
        }
        if (@import("builtin").os.tag != .linux) {
            return error.CUDANotSupported;
        }

        const backend_ref = CUDA.createBackend() orelse return error.CUDAInitializationFailed;
        const cuda_backend = allocator.create(CUDABackend) catch |err| {
            CUDA.destroyBackend(backend_ref);
            return err;
        };

        cuda_backend.* = .{
            .allocator = allocator,
            .backend = backend_ref,
            .device_name = [_]u8{0} ** 256,
            .stats = .{},
            .batch_active = false,
            .synchronized_kernel_count = 0,
            .deferred_matrices = .empty,
        };
        _ = CUDA.deviceName(backend_ref, cuda_backend.device_name[0..]);

        return cuda_backend;
    }

    pub fn deinit(self: *CUDABackend) void {
        self.batch_active = false;
        _ = self.synchronizeQueued();
        self.releaseDeferredMatrices();
        self.deferred_matrices.deinit(self.allocator);
        CUDA.destroyBackend(self.backend);
        self.allocator.destroy(self);
    }

    pub fn getDeviceName(self: *const CUDABackend) []const u8 {
        return std.mem.sliceTo(&self.device_name, 0);
    }

    pub fn runtimeStats(ptr: *anyopaque) backend.RuntimeStats {
        const self = @as(*CUDABackend, @ptrCast(@alignCast(ptr)));
        return self.stats;
    }

    pub fn resetRuntimeStats(ptr: *anyopaque) void {
        const self = @as(*CUDABackend, @ptrCast(@alignCast(ptr)));
        self.stats = .{};
        self.synchronized_kernel_count = 0;
    }

    pub fn beginBatch(ptr: *anyopaque) !void {
        const self = @as(*CUDABackend, @ptrCast(@alignCast(ptr)));
        if (self.batch_active) return error.BatchAlreadyActive;
        self.batch_active = true;
    }

    pub fn endBatch(ptr: *anyopaque) !void {
        const self = @as(*CUDABackend, @ptrCast(@alignCast(ptr)));
        if (!self.batch_active) return error.NoActiveBatch;
        self.batch_active = false;
        if (!self.synchronizeQueued()) return error.CommandExecutionFailed;
    }

    pub fn synchronize(ptr: *anyopaque) !void {
        const self = @as(*CUDABackend, @ptrCast(@alignCast(ptr)));
        if (!self.synchronizeQueued()) return error.CommandExecutionFailed;
    }

    fn completeSubmittedKernel(self: *CUDABackend) bool {
        self.stats.kernel_launches += 1;
        if (self.batch_active) return true;
        return self.synchronizeQueued();
    }

    fn synchronizeQueued(self: *CUDABackend) bool {
        if (self.synchronized_kernel_count != self.stats.kernel_launches) {
            const backend_ref = self.backend orelse return false;
            if (!CUDA.synchronize(backend_ref)) return false;
            self.stats.synchronizations += 1;
            self.synchronized_kernel_count = self.stats.kernel_launches;
        }
        self.releaseDeferredMatrices();
        return true;
    }

    fn releaseDeferredMatrices(self: *CUDABackend) void {
        for (self.deferred_matrices.items) |matrix| {
            deinitMatrixNow(matrix);
        }
        self.deferred_matrices.clearRetainingCapacity();
    }

    fn deinitMatrixNow(matrix: *Matrix) void {
        const cuda_matrix = getCUDAMatrix(matrix);
        const allocator = cuda_matrix.allocator;
        cuda_matrix.deinit();
        allocator.destroy(matrix);
    }

    fn getCUDAMatrix(matrix: *const Matrix) *CUDAMatrix {
        return @as(*CUDAMatrix, @ptrCast(@alignCast(matrix.impl_data)));
    }

    fn elementCount(matrix: *const Matrix) usize {
        return matrix.rows * matrix.cols;
    }

    fn toU32(value: usize) u32 {
        return std.math.cast(u32, value) orelse @panic("CUDA backend dimension exceeds u32 range");
    }

    fn ensureHostData(cuda_matrix: *CUDAMatrix) void {
        if (!cuda_matrix.syncToCPU()) {
            std.log.warn("CUDA buffer download failed; using existing host data", .{});
        }
    }

    fn dispatchMatrixMultiply(self: *CUDABackend, a: *const Matrix, b: *const Matrix, result: *Matrix) bool {
        const backend_ref = self.backend orelse return false;
        const a_cuda = getCUDAMatrix(a);
        const b_cuda = getCUDAMatrix(b);
        const result_cuda = getCUDAMatrix(result);

        if (!a_cuda.syncToGPU() or !b_cuda.syncToGPU()) return false;
        const a_buffer = a_cuda.buffer orelse return false;
        const b_buffer = b_cuda.buffer orelse return false;
        const result_buffer = result_cuda.buffer orelse return false;

        const kind = CUDA.launchMatrixMultiply(backend_ref, a_buffer, b_buffer, result_buffer, toU32(a.rows), toU32(a.cols), toU32(b.cols));
        if (kind == .failed) return false;

        if (kind == .vendor) self.stats.vendor_gemm_launches += 1;
        if (!self.completeSubmittedKernel()) return false;
        result_cuda.markGPUModified();
        return true;
    }

    fn dispatchBinaryMatrixKernel(self: *CUDABackend, kernel_id: CUDA.KernelId, a: *const Matrix, b: *const Matrix, result: *Matrix) bool {
        const backend_ref = self.backend orelse return false;
        const a_cuda = getCUDAMatrix(a);
        const b_cuda = getCUDAMatrix(b);
        const result_cuda = getCUDAMatrix(result);

        if (!a_cuda.syncToGPU() or !b_cuda.syncToGPU()) return false;
        const a_buffer = a_cuda.buffer orelse return false;
        const b_buffer = b_cuda.buffer orelse return false;
        const result_buffer = result_cuda.buffer orelse return false;

        if (!CUDA.launchBinaryKernel(backend_ref, kernel_id, a_buffer, b_buffer, result_buffer, toU32(a.rows), toU32(a.cols))) {
            return false;
        }

        if (!self.completeSubmittedKernel()) return false;
        result_cuda.markGPUModified();
        return true;
    }

    fn dispatchScale(self: *CUDABackend, matrix: *const Matrix, scalar: f64, result: *Matrix) bool {
        const backend_ref = self.backend orelse return false;
        const matrix_cuda = getCUDAMatrix(matrix);
        const result_cuda = getCUDAMatrix(result);

        if (!matrix_cuda.syncToGPU()) return false;
        const matrix_buffer = matrix_cuda.buffer orelse return false;
        const result_buffer = result_cuda.buffer orelse return false;

        if (!CUDA.launchScale(backend_ref, matrix_buffer, result_buffer, @floatCast(scalar), toU32(matrix.rows), toU32(matrix.cols))) {
            return false;
        }

        if (!self.completeSubmittedKernel()) return false;
        result_cuda.markGPUModified();
        return true;
    }

    fn dispatchTranspose(self: *CUDABackend, matrix: *const Matrix, result: *Matrix) bool {
        const backend_ref = self.backend orelse return false;
        const matrix_cuda = getCUDAMatrix(matrix);
        const result_cuda = getCUDAMatrix(result);

        if (!matrix_cuda.syncToGPU()) return false;
        const matrix_buffer = matrix_cuda.buffer orelse return false;
        const result_buffer = result_cuda.buffer orelse return false;

        if (!CUDA.launchTranspose(backend_ref, matrix_buffer, result_buffer, toU32(matrix.rows), toU32(matrix.cols))) {
            return false;
        }

        if (!self.completeSubmittedKernel()) return false;
        result_cuda.markGPUModified();
        return true;
    }

    fn dispatchSumRows(self: *CUDABackend, matrix: *const Matrix, result: *Matrix) bool {
        const backend_ref = self.backend orelse return false;
        const matrix_cuda = getCUDAMatrix(matrix);
        const result_cuda = getCUDAMatrix(result);

        if (!matrix_cuda.syncToGPU()) return false;
        const matrix_buffer = matrix_cuda.buffer orelse return false;
        const result_buffer = result_cuda.buffer orelse return false;

        if (!CUDA.launchSumRows(backend_ref, matrix_buffer, result_buffer, toU32(matrix.rows), toU32(matrix.cols))) {
            return false;
        }

        if (!self.completeSubmittedKernel()) return false;
        result_cuda.markGPUModified();
        return true;
    }

    fn dispatchExtractBatch(self: *CUDABackend, matrix: *const Matrix, start: usize, result: *Matrix) bool {
        const backend_ref = self.backend orelse return false;
        const matrix_cuda = getCUDAMatrix(matrix);
        const result_cuda = getCUDAMatrix(result);

        if (!matrix_cuda.syncToGPU()) return false;
        const matrix_buffer = matrix_cuda.buffer orelse return false;
        const result_buffer = result_cuda.buffer orelse return false;

        if (!CUDA.launchExtractBatch(backend_ref, matrix_buffer, result_buffer, toU32(start), toU32(result.rows), toU32(result.cols))) {
            return false;
        }

        if (!self.completeSubmittedKernel()) return false;
        result_cuda.markGPUModified();
        return true;
    }

    fn dispatchUnaryKernel(self: *CUDABackend, kernel_id: CUDA.KernelId, matrix: *const Matrix, result: *Matrix) bool {
        const backend_ref = self.backend orelse return false;
        const matrix_cuda = getCUDAMatrix(matrix);
        const result_cuda = getCUDAMatrix(result);

        if (!matrix_cuda.syncToGPU()) return false;
        const matrix_buffer = matrix_cuda.buffer orelse return false;
        const result_buffer = result_cuda.buffer orelse return false;

        if (!CUDA.launchUnaryKernel(backend_ref, kernel_id, matrix_buffer, result_buffer, toU32(elementCount(matrix)))) {
            return false;
        }

        if (!self.completeSubmittedKernel()) return false;
        result_cuda.markGPUModified();
        return true;
    }

    fn dispatchSoftmax(self: *CUDABackend, matrix: *const Matrix, result: *Matrix) bool {
        const backend_ref = self.backend orelse return false;
        const matrix_cuda = getCUDAMatrix(matrix);
        const result_cuda = getCUDAMatrix(result);

        if (!matrix_cuda.syncToGPU()) return false;
        const matrix_buffer = matrix_cuda.buffer orelse return false;
        const result_buffer = result_cuda.buffer orelse return false;

        if (!CUDA.launchSoftmaxRows(backend_ref, matrix_buffer, result_buffer, toU32(matrix.rows), toU32(matrix.cols))) {
            return false;
        }

        if (!self.completeSubmittedKernel()) return false;
        result_cuda.markGPUModified();
        return true;
    }

    fn dispatchGatedKernel(self: *CUDABackend, kernel_id: CUDA.KernelId, linear_part: *const Matrix, gating_part: *const Matrix, result: *Matrix) bool {
        const backend_ref = self.backend orelse return false;
        const linear_cuda = getCUDAMatrix(linear_part);
        const gating_cuda = getCUDAMatrix(gating_part);
        const result_cuda = getCUDAMatrix(result);

        if (!linear_cuda.syncToGPU() or !gating_cuda.syncToGPU()) return false;
        const linear_buffer = linear_cuda.buffer orelse return false;
        const gating_buffer = gating_cuda.buffer orelse return false;
        const result_buffer = result_cuda.buffer orelse return false;

        if (!CUDA.launchGatedKernel(backend_ref, kernel_id, linear_buffer, gating_buffer, result_buffer, toU32(elementCount(linear_part)))) {
            return false;
        }

        if (!self.completeSubmittedKernel()) return false;
        result_cuda.markGPUModified();
        return true;
    }

    pub fn initMatrix(ptr: *anyopaque, allocator: Allocator, rows: usize, cols: usize) error{OutOfMemory}!*Matrix {
        const self = @as(*CUDABackend, @ptrCast(@alignCast(ptr)));
        const cuda_matrix = CUDAMatrix.init(allocator, rows, cols, self) catch |err| {
            std.log.warn("CUDA matrix initialization failed: {s}", .{@errorName(err)});
            return error.OutOfMemory;
        };
        errdefer cuda_matrix.deinit();

        const matrix = try allocator.create(Matrix);
        matrix.* = .{
            .rows = rows,
            .cols = cols,
            .backend = undefined,
            .impl_data = cuda_matrix,
        };
        return matrix;
    }

    pub fn deinitMatrix(ptr: *anyopaque, matrix: *Matrix) void {
        const self = @as(*CUDABackend, @ptrCast(@alignCast(ptr)));
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
        const source_cuda = getCUDAMatrix(source);
        const result = try initMatrix(ptr, allocator, source.rows, source.cols);
        const result_cuda = getCUDAMatrix(result);

        ensureHostData(source_cuda);
        @memcpy(result_cuda.host_data, source_cuda.host_data);
        result_cuda.markHostModified();
        return result;
    }

    pub fn getMatrixElement(_: *anyopaque, matrix: *const Matrix, row: usize, col: usize) f64 {
        if (row >= matrix.rows or col >= matrix.cols) {
            @panic("Index out of bounds");
        }

        const cuda_matrix = getCUDAMatrix(matrix);
        if (!cuda_matrix.syncToCPU()) {
            std.log.warn("CUDA buffer download failed while reading matrix element", .{});
        }
        return @floatCast(cuda_matrix.host_data[row * matrix.cols + col]);
    }

    pub fn writeMatrixF32(_: *anyopaque, matrix: *Matrix, values: []const f32) bool {
        const cuda_matrix = getCUDAMatrix(matrix);
        if (values.len != cuda_matrix.host_data.len) return false;
        @memcpy(cuda_matrix.host_data, values);
        cuda_matrix.markHostModified();
        return true;
    }

    pub fn readMatrixF32(_: *anyopaque, matrix: *const Matrix, values: []f32) bool {
        const cuda_matrix = getCUDAMatrix(matrix);
        if (values.len != cuda_matrix.host_data.len) return false;
        ensureHostData(cuda_matrix);
        @memcpy(values, cuda_matrix.host_data);
        return true;
    }

    pub fn setMatrixElement(_: *anyopaque, matrix: *Matrix, row: usize, col: usize, value: f64) void {
        if (row >= matrix.rows or col >= matrix.cols) {
            @panic("Index out of bounds");
        }

        const cuda_matrix = getCUDAMatrix(matrix);
        cuda_matrix.host_data[row * matrix.cols + col] = @floatCast(value);
        cuda_matrix.markHostModified();
    }

    pub fn fillMatrix(_: *anyopaque, matrix: *Matrix, value: f64) void {
        const cuda_matrix = getCUDAMatrix(matrix);
        for (cuda_matrix.host_data) |*element| {
            element.* = @floatCast(value);
        }
        cuda_matrix.markHostModified();
    }

    pub fn dotProduct(ptr: *anyopaque, a: *const Matrix, b: *const Matrix, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!*Matrix {
        if (a.cols != b.rows) {
            return error.DimensionMismatch;
        }

        const self = @as(*CUDABackend, @ptrCast(@alignCast(ptr)));
        const result = try initMatrix(ptr, allocator, a.rows, b.cols);
        const a_cuda = getCUDAMatrix(a);
        const b_cuda = getCUDAMatrix(b);
        const result_cuda = getCUDAMatrix(result);

        if (self.dispatchMatrixMultiply(a, b, result)) {
            return result;
        }

        ensureHostData(a_cuda);
        ensureHostData(b_cuda);
        for (0..a.rows) |i| {
            for (0..b.cols) |j| {
                var sum: f32 = 0.0;
                for (0..a.cols) |k| {
                    sum += a_cuda.host_data[i * a.cols + k] * b_cuda.host_data[k * b.cols + j];
                }
                result_cuda.host_data[i * b.cols + j] = sum;
            }
        }
        result_cuda.markHostModified();
        return result;
    }

    pub fn add(ptr: *anyopaque, a: *const Matrix, b: *const Matrix, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!*Matrix {
        if (a.rows != b.rows or a.cols != b.cols) {
            return error.DimensionMismatch;
        }

        const self = @as(*CUDABackend, @ptrCast(@alignCast(ptr)));
        const result = try initMatrix(ptr, allocator, a.rows, a.cols);
        const a_cuda = getCUDAMatrix(a);
        const b_cuda = getCUDAMatrix(b);
        const result_cuda = getCUDAMatrix(result);

        if (self.dispatchBinaryMatrixKernel(.matrix_add, a, b, result)) {
            return result;
        }

        ensureHostData(a_cuda);
        ensureHostData(b_cuda);
        for (0..a.rows * a.cols) |i| {
            result_cuda.host_data[i] = a_cuda.host_data[i] + b_cuda.host_data[i];
        }
        result_cuda.markHostModified();
        return result;
    }

    pub fn subtract(ptr: *anyopaque, a: *const Matrix, b: *const Matrix, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!*Matrix {
        if (a.rows != b.rows or a.cols != b.cols) {
            return error.DimensionMismatch;
        }

        const self = @as(*CUDABackend, @ptrCast(@alignCast(ptr)));
        const result = try initMatrix(ptr, allocator, a.rows, a.cols);
        const a_cuda = getCUDAMatrix(a);
        const b_cuda = getCUDAMatrix(b);
        const result_cuda = getCUDAMatrix(result);

        if (self.dispatchBinaryMatrixKernel(.matrix_subtract, a, b, result)) {
            return result;
        }

        ensureHostData(a_cuda);
        ensureHostData(b_cuda);
        for (0..a.rows * a.cols) |i| {
            result_cuda.host_data[i] = a_cuda.host_data[i] - b_cuda.host_data[i];
        }
        result_cuda.markHostModified();
        return result;
    }

    pub fn elementWiseMultiply(ptr: *anyopaque, a: *const Matrix, b: *const Matrix, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!*Matrix {
        if (a.rows != b.rows or a.cols != b.cols) {
            return error.DimensionMismatch;
        }

        const self = @as(*CUDABackend, @ptrCast(@alignCast(ptr)));
        const result = try initMatrix(ptr, allocator, a.rows, a.cols);
        const a_cuda = getCUDAMatrix(a);
        const b_cuda = getCUDAMatrix(b);
        const result_cuda = getCUDAMatrix(result);

        if (self.dispatchBinaryMatrixKernel(.matrix_element_wise_multiply, a, b, result)) {
            return result;
        }

        ensureHostData(a_cuda);
        ensureHostData(b_cuda);
        for (0..a.rows * a.cols) |i| {
            result_cuda.host_data[i] = a_cuda.host_data[i] * b_cuda.host_data[i];
        }
        result_cuda.markHostModified();
        return result;
    }

    pub fn scale(ptr: *anyopaque, matrix: *const Matrix, scalar: f64, allocator: Allocator) error{OutOfMemory}!*Matrix {
        const self = @as(*CUDABackend, @ptrCast(@alignCast(ptr)));
        const result = try initMatrix(ptr, allocator, matrix.rows, matrix.cols);
        const matrix_cuda = getCUDAMatrix(matrix);
        const result_cuda = getCUDAMatrix(result);

        if (self.dispatchScale(matrix, scalar, result)) {
            return result;
        }

        ensureHostData(matrix_cuda);
        for (0..matrix.rows * matrix.cols) |i| {
            result_cuda.host_data[i] = matrix_cuda.host_data[i] * @as(f32, @floatCast(scalar));
        }
        result_cuda.markHostModified();
        return result;
    }

    pub fn sumRows(ptr: *anyopaque, matrix: *const Matrix, allocator: Allocator) error{OutOfMemory}!*Matrix {
        const self = @as(*CUDABackend, @ptrCast(@alignCast(ptr)));
        const result = try initMatrix(ptr, allocator, 1, matrix.cols);
        const matrix_cuda = getCUDAMatrix(matrix);
        const result_cuda = getCUDAMatrix(result);

        if (self.dispatchSumRows(matrix, result)) {
            return result;
        }

        ensureHostData(matrix_cuda);
        for (0..matrix.cols) |j| {
            var sum: f32 = 0.0;
            for (0..matrix.rows) |i| {
                sum += matrix_cuda.host_data[i * matrix.cols + j];
            }
            result_cuda.host_data[j] = sum;
        }
        result_cuda.markHostModified();
        return result;
    }

    pub fn transpose(ptr: *anyopaque, matrix: *const Matrix, allocator: Allocator) error{OutOfMemory}!*Matrix {
        const self = @as(*CUDABackend, @ptrCast(@alignCast(ptr)));
        const result = try initMatrix(ptr, allocator, matrix.cols, matrix.rows);
        const matrix_cuda = getCUDAMatrix(matrix);
        const result_cuda = getCUDAMatrix(result);

        if (self.dispatchTranspose(matrix, result)) {
            return result;
        }

        ensureHostData(matrix_cuda);
        for (0..matrix.rows) |i| {
            for (0..matrix.cols) |j| {
                result_cuda.host_data[j * matrix.rows + i] = matrix_cuda.host_data[i * matrix.cols + j];
            }
        }
        result_cuda.markHostModified();
        return result;
    }

    pub fn extractBatch(ptr: *anyopaque, matrix: *const Matrix, start: usize, end: usize, allocator: Allocator) error{ OutOfMemory, InvalidBatchIndices }!*Matrix {
        if (start >= end or end > matrix.rows) {
            return error.InvalidBatchIndices;
        }

        const self = @as(*CUDABackend, @ptrCast(@alignCast(ptr)));
        const batch_rows = end - start;
        const result = try initMatrix(ptr, allocator, batch_rows, matrix.cols);
        const matrix_cuda = getCUDAMatrix(matrix);
        const result_cuda = getCUDAMatrix(result);

        if (self.dispatchExtractBatch(matrix, start, result)) {
            return result;
        }

        ensureHostData(matrix_cuda);
        const src_offset = start * matrix.cols;
        @memcpy(result_cuda.host_data, matrix_cuda.host_data[src_offset..][0 .. batch_rows * matrix.cols]);
        result_cuda.markHostModified();
        return result;
    }

    pub fn randomize(_: *anyopaque, matrix: *Matrix, min: f64, max: f64) void {
        const cuda_matrix = getCUDAMatrix(matrix);
        var rng = std.Random.DefaultPrng.init(randomSeed());
        for (cuda_matrix.host_data) |*element| {
            const random_float = rng.random().float(f64);
            element.* = @floatCast(min + random_float * (max - min));
        }
        cuda_matrix.markHostModified();
    }

    pub fn applyActivation(ptr: *anyopaque, matrix: *const Matrix, activation: *const fn (f64) f64, allocator: Allocator) error{OutOfMemory}!*Matrix {
        const self = @as(*CUDABackend, @ptrCast(@alignCast(ptr)));
        const result = try initMatrix(ptr, allocator, matrix.rows, matrix.cols);
        const matrix_cuda = getCUDAMatrix(matrix);
        const result_cuda = getCUDAMatrix(result);

        const kernel_id: ?CUDA.KernelId = if (activation == Activation.linear)
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
        else
            null;

        if (kernel_id) |id| {
            if (self.dispatchUnaryKernel(id, matrix, result)) {
                return result;
            }
        }

        ensureHostData(matrix_cuda);
        for (0..matrix.rows * matrix.cols) |i| {
            result_cuda.host_data[i] = @floatCast(activation(@floatCast(matrix_cuda.host_data[i])));
        }
        result_cuda.markHostModified();
        return result;
    }

    pub fn applySoftmax(ptr: *anyopaque, matrix: *const Matrix, allocator: Allocator) error{OutOfMemory}!*Matrix {
        const self = @as(*CUDABackend, @ptrCast(@alignCast(ptr)));
        const result = try initMatrix(ptr, allocator, matrix.rows, matrix.cols);
        const matrix_cuda = getCUDAMatrix(matrix);
        const result_cuda = getCUDAMatrix(result);

        if (self.dispatchSoftmax(matrix, result)) {
            return result;
        }

        ensureHostData(matrix_cuda);
        for (0..matrix.rows) |i| {
            var max_val: f32 = -std.math.inf(f32);
            for (0..matrix.cols) |j| {
                max_val = @max(max_val, matrix_cuda.host_data[i * matrix.cols + j]);
            }

            var sum: f32 = 0.0;
            for (0..matrix.cols) |j| {
                const idx = i * matrix.cols + j;
                const exp_val = std.math.exp(matrix_cuda.host_data[idx] - max_val);
                result_cuda.host_data[idx] = exp_val;
                sum += exp_val;
            }

            for (0..matrix.cols) |j| {
                const idx = i * matrix.cols + j;
                result_cuda.host_data[idx] /= sum;
            }
        }
        result_cuda.markHostModified();
        return result;
    }

    pub fn applyGLU(ptr: *anyopaque, linear_part: *const Matrix, gating_part: *const Matrix, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!*Matrix {
        if (linear_part.rows != gating_part.rows or linear_part.cols != gating_part.cols) {
            return error.DimensionMismatch;
        }

        const self = @as(*CUDABackend, @ptrCast(@alignCast(ptr)));
        const result = try initMatrix(ptr, allocator, linear_part.rows, linear_part.cols);
        const linear_cuda = getCUDAMatrix(linear_part);
        const gating_cuda = getCUDAMatrix(gating_part);
        const result_cuda = getCUDAMatrix(result);

        if (self.dispatchGatedKernel(.apply_glu, linear_part, gating_part, result)) {
            return result;
        }

        ensureHostData(linear_cuda);
        ensureHostData(gating_cuda);
        for (0..linear_part.rows * linear_part.cols) |i| {
            const sigmoid = 1.0 / (1.0 + std.math.exp(-gating_cuda.host_data[i]));
            result_cuda.host_data[i] = linear_cuda.host_data[i] * sigmoid;
        }
        result_cuda.markHostModified();
        return result;
    }

    pub fn applySwiGLU(ptr: *anyopaque, linear_part: *const Matrix, gating_part: *const Matrix, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!*Matrix {
        if (linear_part.rows != gating_part.rows or linear_part.cols != gating_part.cols) {
            return error.DimensionMismatch;
        }

        const self = @as(*CUDABackend, @ptrCast(@alignCast(ptr)));
        const result = try initMatrix(ptr, allocator, linear_part.rows, linear_part.cols);
        const linear_cuda = getCUDAMatrix(linear_part);
        const gating_cuda = getCUDAMatrix(gating_part);
        const result_cuda = getCUDAMatrix(result);

        if (self.dispatchGatedKernel(.apply_swiglu, linear_part, gating_part, result)) {
            return result;
        }

        ensureHostData(linear_cuda);
        ensureHostData(gating_cuda);
        for (0..linear_part.rows * linear_part.cols) |i| {
            const gate = gating_cuda.host_data[i];
            const sigmoid = 1.0 / (1.0 + std.math.exp(-gate));
            result_cuda.host_data[i] = linear_cuda.host_data[i] * gate * sigmoid;
        }
        result_cuda.markHostModified();
        return result;
    }

    pub fn getBackendType(_: *anyopaque) BackendType {
        return .CUDA;
    }
};

pub fn createCUDABackend(allocator: Allocator) !*anyopaque {
    const cuda_backend = try CUDABackend.init(allocator);
    errdefer cuda_backend.deinit();
    return @ptrCast(cuda_backend);
}

const CPUBackend = @import("cpu_backend.zig").CPUBackend;
const testing = std.testing;

fn initCUDABackendForTest(allocator: Allocator) !*CUDABackend {
    if (!enable_cuda or @import("builtin").os.tag != .linux) {
        return error.SkipZigTest;
    }

    return CUDABackend.init(allocator) catch |err| switch (err) {
        error.CUDANotEnabled, error.CUDANotSupported, error.CUDADeviceNotFound, error.CUDAInitializationFailed => error.SkipZigTest,
        else => err,
    };
}

fn fillMatrixPair(
    cpu_ptr: *anyopaque,
    cuda_ptr: *anyopaque,
    cpu_matrix: *Matrix,
    cuda_matrix: *Matrix,
    values: []const f64,
) void {
    for (values, 0..) |value, index| {
        const row = index / cpu_matrix.cols;
        const col = index % cpu_matrix.cols;
        CPUBackend.setMatrixElement(cpu_ptr, cpu_matrix, row, col, value);
        CUDABackend.setMatrixElement(cuda_ptr, cuda_matrix, row, col, value);
    }
}

fn expectMatricesClose(
    cpu_ptr: *anyopaque,
    cuda_ptr: *anyopaque,
    cpu_matrix: *const Matrix,
    cuda_matrix: *const Matrix,
    tolerance: f64,
) !void {
    try testing.expectEqual(cpu_matrix.rows, cuda_matrix.rows);
    try testing.expectEqual(cpu_matrix.cols, cuda_matrix.cols);

    for (0..cpu_matrix.rows) |row| {
        for (0..cpu_matrix.cols) |col| {
            const expected = CPUBackend.getMatrixElement(cpu_ptr, cpu_matrix, row, col);
            const actual = CUDABackend.getMatrixElement(cuda_ptr, cuda_matrix, row, col);
            try testing.expectApproxEqAbs(expected, actual, tolerance);
        }
    }
}

test "cuda backend matches cpu backend for matrix operations" {
    const allocator = testing.allocator;

    const cpu_impl = try CPUBackend.init(allocator);
    defer cpu_impl.deinit();
    const cuda_impl = try initCUDABackendForTest(allocator);
    defer cuda_impl.deinit();

    const cpu_ptr: *anyopaque = @ptrCast(cpu_impl);
    const cuda_ptr: *anyopaque = @ptrCast(cuda_impl);

    const a_values = [_]f64{ 1.5, -2.0, 3.25, 0.5, 4.0, -1.25 };
    const b_values = [_]f64{ 2.0, -1.0, 0.25, 3.0, -2.5, 1.75 };
    const c_values = [_]f64{ -0.5, 2.0, 0.25, 3.0, -4.0, 1.25 };

    const cpu_a = try CPUBackend.initMatrix(cpu_ptr, allocator, 2, 3);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_a);
    const cuda_a = try CUDABackend.initMatrix(cuda_ptr, allocator, 2, 3);
    defer CUDABackend.deinitMatrix(cuda_ptr, cuda_a);
    fillMatrixPair(cpu_ptr, cuda_ptr, cpu_a, cuda_a, &a_values);

    const cpu_b = try CPUBackend.initMatrix(cpu_ptr, allocator, 3, 2);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_b);
    const cuda_b = try CUDABackend.initMatrix(cuda_ptr, allocator, 3, 2);
    defer CUDABackend.deinitMatrix(cuda_ptr, cuda_b);
    fillMatrixPair(cpu_ptr, cuda_ptr, cpu_b, cuda_b, &b_values);

    const cpu_c = try CPUBackend.initMatrix(cpu_ptr, allocator, 2, 3);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_c);
    const cuda_c = try CUDABackend.initMatrix(cuda_ptr, allocator, 2, 3);
    defer CUDABackend.deinitMatrix(cuda_ptr, cuda_c);
    fillMatrixPair(cpu_ptr, cuda_ptr, cpu_c, cuda_c, &c_values);

    const tolerance = 1e-4;

    const cpu_dot = try CPUBackend.dotProduct(cpu_ptr, cpu_a, cpu_b, allocator);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_dot);
    const cuda_dot = try CUDABackend.dotProduct(cuda_ptr, cuda_a, cuda_b, allocator);
    defer CUDABackend.deinitMatrix(cuda_ptr, cuda_dot);
    try expectMatricesClose(cpu_ptr, cuda_ptr, cpu_dot, cuda_dot, tolerance);

    const cpu_add = try CPUBackend.add(cpu_ptr, cpu_a, cpu_c, allocator);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_add);
    const cuda_add = try CUDABackend.add(cuda_ptr, cuda_a, cuda_c, allocator);
    defer CUDABackend.deinitMatrix(cuda_ptr, cuda_add);
    try expectMatricesClose(cpu_ptr, cuda_ptr, cpu_add, cuda_add, tolerance);

    const cpu_subtract = try CPUBackend.subtract(cpu_ptr, cpu_a, cpu_c, allocator);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_subtract);
    const cuda_subtract = try CUDABackend.subtract(cuda_ptr, cuda_a, cuda_c, allocator);
    defer CUDABackend.deinitMatrix(cuda_ptr, cuda_subtract);
    try expectMatricesClose(cpu_ptr, cuda_ptr, cpu_subtract, cuda_subtract, tolerance);

    const cpu_multiply = try CPUBackend.elementWiseMultiply(cpu_ptr, cpu_a, cpu_c, allocator);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_multiply);
    const cuda_multiply = try CUDABackend.elementWiseMultiply(cuda_ptr, cuda_a, cuda_c, allocator);
    defer CUDABackend.deinitMatrix(cuda_ptr, cuda_multiply);
    try expectMatricesClose(cpu_ptr, cuda_ptr, cpu_multiply, cuda_multiply, tolerance);

    const cpu_scale = try CPUBackend.scale(cpu_ptr, cpu_a, -0.75, allocator);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_scale);
    const cuda_scale = try CUDABackend.scale(cuda_ptr, cuda_a, -0.75, allocator);
    defer CUDABackend.deinitMatrix(cuda_ptr, cuda_scale);
    try expectMatricesClose(cpu_ptr, cuda_ptr, cpu_scale, cuda_scale, tolerance);

    const cpu_sum_rows = try CPUBackend.sumRows(cpu_ptr, cpu_a, allocator);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_sum_rows);
    const cuda_sum_rows = try CUDABackend.sumRows(cuda_ptr, cuda_a, allocator);
    defer CUDABackend.deinitMatrix(cuda_ptr, cuda_sum_rows);
    try expectMatricesClose(cpu_ptr, cuda_ptr, cpu_sum_rows, cuda_sum_rows, tolerance);

    const cpu_transpose = try CPUBackend.transpose(cpu_ptr, cpu_a, allocator);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_transpose);
    const cuda_transpose = try CUDABackend.transpose(cuda_ptr, cuda_a, allocator);
    defer CUDABackend.deinitMatrix(cuda_ptr, cuda_transpose);
    try expectMatricesClose(cpu_ptr, cuda_ptr, cpu_transpose, cuda_transpose, tolerance);

    const cpu_batch = try CPUBackend.extractBatch(cpu_ptr, cpu_a, 1, 2, allocator);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_batch);
    const cuda_batch = try CUDABackend.extractBatch(cuda_ptr, cuda_a, 1, 2, allocator);
    defer CUDABackend.deinitMatrix(cuda_ptr, cuda_batch);
    try expectMatricesClose(cpu_ptr, cuda_ptr, cpu_batch, cuda_batch, tolerance);
}

test "cuda backend matches cpu backend for activation operations" {
    const allocator = testing.allocator;

    const cpu_impl = try CPUBackend.init(allocator);
    defer cpu_impl.deinit();
    const cuda_impl = try initCUDABackendForTest(allocator);
    defer cuda_impl.deinit();

    const cpu_ptr: *anyopaque = @ptrCast(cpu_impl);
    const cuda_ptr: *anyopaque = @ptrCast(cuda_impl);

    const a_values = [_]f64{ -1.5, -0.25, 0.0, 0.75, 2.0, 4.0 };
    const gate_values = [_]f64{ 2.0, -1.0, 0.5, 3.0, -2.5, 1.25 };

    const cpu_a = try CPUBackend.initMatrix(cpu_ptr, allocator, 2, 3);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_a);
    const cuda_a = try CUDABackend.initMatrix(cuda_ptr, allocator, 2, 3);
    defer CUDABackend.deinitMatrix(cuda_ptr, cuda_a);
    fillMatrixPair(cpu_ptr, cuda_ptr, cpu_a, cuda_a, &a_values);

    const cpu_gate = try CPUBackend.initMatrix(cpu_ptr, allocator, 2, 3);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_gate);
    const cuda_gate = try CUDABackend.initMatrix(cuda_ptr, allocator, 2, 3);
    defer CUDABackend.deinitMatrix(cuda_ptr, cuda_gate);
    fillMatrixPair(cpu_ptr, cuda_ptr, cpu_gate, cuda_gate, &gate_values);

    const tolerance = 1e-4;

    const cpu_sigmoid = try CPUBackend.applyActivation(cpu_ptr, cpu_a, Activation.sigmoid, allocator);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_sigmoid);
    const cuda_sigmoid = try CUDABackend.applyActivation(cuda_ptr, cuda_a, Activation.sigmoid, allocator);
    defer CUDABackend.deinitMatrix(cuda_ptr, cuda_sigmoid);
    try expectMatricesClose(cpu_ptr, cuda_ptr, cpu_sigmoid, cuda_sigmoid, tolerance);

    const cpu_relu = try CPUBackend.applyActivation(cpu_ptr, cpu_a, Activation.relu, allocator);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_relu);
    const cuda_relu = try CUDABackend.applyActivation(cuda_ptr, cuda_a, Activation.relu, allocator);
    defer CUDABackend.deinitMatrix(cuda_ptr, cuda_relu);
    try expectMatricesClose(cpu_ptr, cuda_ptr, cpu_relu, cuda_relu, tolerance);

    const cpu_tanh = try CPUBackend.applyActivation(cpu_ptr, cpu_a, Activation.tanh, allocator);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_tanh);
    const cuda_tanh = try CUDABackend.applyActivation(cuda_ptr, cuda_a, Activation.tanh, allocator);
    defer CUDABackend.deinitMatrix(cuda_ptr, cuda_tanh);
    try expectMatricesClose(cpu_ptr, cuda_ptr, cpu_tanh, cuda_tanh, tolerance);

    const cpu_swish = try CPUBackend.applyActivation(cpu_ptr, cpu_a, Activation.swish, allocator);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_swish);
    const cuda_swish = try CUDABackend.applyActivation(cuda_ptr, cuda_a, Activation.swish, allocator);
    defer CUDABackend.deinitMatrix(cuda_ptr, cuda_swish);
    try expectMatricesClose(cpu_ptr, cuda_ptr, cpu_swish, cuda_swish, tolerance);

    const cpu_softmax = try CPUBackend.applySoftmax(cpu_ptr, cpu_a, allocator);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_softmax);
    const cuda_softmax = try CUDABackend.applySoftmax(cuda_ptr, cuda_a, allocator);
    defer CUDABackend.deinitMatrix(cuda_ptr, cuda_softmax);
    try expectMatricesClose(cpu_ptr, cuda_ptr, cpu_softmax, cuda_softmax, tolerance);

    const cpu_glu = try CPUBackend.applyGLU(cpu_ptr, cpu_a, cpu_gate, allocator);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_glu);
    const cuda_glu = try CUDABackend.applyGLU(cuda_ptr, cuda_a, cuda_gate, allocator);
    defer CUDABackend.deinitMatrix(cuda_ptr, cuda_glu);
    try expectMatricesClose(cpu_ptr, cuda_ptr, cpu_glu, cuda_glu, tolerance);

    const cpu_swiglu = try CPUBackend.applySwiGLU(cpu_ptr, cpu_a, cpu_gate, allocator);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_swiglu);
    const cuda_swiglu = try CUDABackend.applySwiGLU(cuda_ptr, cuda_a, cuda_gate, allocator);
    defer CUDABackend.deinitMatrix(cuda_ptr, cuda_swiglu);
    try expectMatricesClose(cpu_ptr, cuda_ptr, cpu_swiglu, cuda_swiglu, tolerance);
}
