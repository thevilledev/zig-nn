const std = @import("std");
const Allocator = std.mem.Allocator;
const backend = @import("backend.zig");
const ComputeBackend = backend.ComputeBackend;
const BackendType = backend.BackendType;
const Matrix = backend.Matrix;
const randomSeed = @import("matrix.zig").randomSeed;
const Activation = @import("activation.zig").Activation;

// Add build options to check if Metal is enabled
const build_options = @import("build_options");
const enable_metal = build_options.enable_metal;
const shader_source = @embedFile("metal/shaders.metal");

// Define errors
pub const MetalError = error{
    MetalNotSupported,
    MetalDeviceNotFound,
    ShaderCompilationFailed,
    ShaderFunctionNotFound,
    PipelineCreationFailed,
    CommandQueueCreationFailed,
    InvalidBatchIndices,
    BufferCreationFailed,
    CommandBufferCreationFailed,
    CommandEncoderCreationFailed,
    CommandExecutionFailed,
    DataTransferError,
};

// Import Metal wrapper only on macOS and when Metal is enabled
const Metal = if (@import("builtin").os.tag == .macos and enable_metal) struct {
    // Import the Metal C wrapper
    const c = @cImport({
        @cInclude("metal_wrapper.h");
    });

    // Opaque Objective-C Metal object references.
    pub const Device = anyopaque;
    pub const CommandQueue = anyopaque;
    pub const Library = anyopaque;
    pub const Function = anyopaque;
    pub const Buffer = anyopaque;
    pub const ComputePipelineState = anyopaque;
    pub const CommandBuffer = anyopaque;
    pub const ComputeCommandEncoder = anyopaque;

    pub const BufferOptions = struct {
        pub const StorageModeShared: u32 = 0;
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

    // Metal API functions now call our C wrapper
    pub fn createSystemDefaultDevice() ?*Device {
        return fromC(Device, c.metal_create_system_default_device());
    }

    pub fn deviceCreateCommandQueue(device: *Device) ?*CommandQueue {
        return fromC(CommandQueue, c.metal_device_create_command_queue(toC(device)));
    }

    pub fn deviceCreateBuffer(device: *Device, length: usize, options: u32) ?*Buffer {
        return fromC(Buffer, c.metal_device_create_buffer(toC(device), length, options));
    }

    pub fn bufferUploadF32(buffer: *Buffer, data: []const f32) bool {
        return c.metal_buffer_upload_f32(toC(buffer), data.ptr, data.len) != 0;
    }

    pub fn bufferDownloadF32(buffer: *Buffer, data: []f32) bool {
        return c.metal_buffer_download_f32(toC(buffer), data.ptr, data.len) != 0;
    }

    pub fn deviceCreateLibraryFromSource(device: *Device, source: [*:0]const u8) ?*Library {
        var error_buffer: [2048]u8 = undefined;
        @memset(&error_buffer, 0);
        const result = c.metal_device_create_library_from_source(toC(device), source, error_buffer[0..].ptr, error_buffer.len);
        if (result == null) {
            logError("Metal shader compilation failed", error_buffer[0..]);
        }
        return fromC(Library, result);
    }

    pub fn libraryCreateFunction(library: *Library, name: [*:0]const u8) ?*Function {
        return fromC(Function, c.metal_library_create_function(toC(library), name));
    }

    pub fn deviceCreateComputePipelineState(device: *Device, function: *Function) ?*ComputePipelineState {
        var error_buffer: [2048]u8 = undefined;
        @memset(&error_buffer, 0);
        const result = c.metal_device_create_compute_pipeline_state(toC(device), toC(function), error_buffer[0..].ptr, error_buffer.len);
        if (result == null) {
            logError("Metal compute pipeline creation failed", error_buffer[0..]);
        }
        return fromC(ComputePipelineState, result);
    }

    pub fn commandQueueCreateCommandBuffer(command_queue: *CommandQueue) ?*CommandBuffer {
        return fromC(CommandBuffer, c.metal_command_queue_create_command_buffer(toC(command_queue)));
    }

    pub fn commandBufferCreateComputeCommandEncoder(command_buffer: *CommandBuffer) ?*ComputeCommandEncoder {
        return fromC(ComputeCommandEncoder, c.metal_command_buffer_create_compute_command_encoder(toC(command_buffer)));
    }

    pub fn computeEncoderSetPipelineState(encoder: *ComputeCommandEncoder, pipeline: *ComputePipelineState) void {
        c.metal_compute_encoder_set_pipeline_state(toC(encoder), toC(pipeline));
    }

    pub fn computeEncoderSetBuffer(encoder: *ComputeCommandEncoder, buffer: *Buffer, offset: usize, index: u32) void {
        c.metal_compute_encoder_set_buffer(toC(encoder), toC(buffer), offset, index);
    }

    pub fn computeEncoderSetBytes(encoder: *ComputeCommandEncoder, bytes: *const anyopaque, length: usize, index: u32) void {
        c.metal_compute_encoder_set_bytes(toC(encoder), bytes, length, index);
    }

    pub fn computeEncoderDispatchThreads(encoder: *ComputeCommandEncoder, pipeline: *ComputePipelineState, width: usize, height: usize, depth: usize) void {
        c.metal_compute_encoder_dispatch_threads(toC(encoder), toC(pipeline), width, height, depth);
    }

    pub fn computeEncoderEndEncoding(encoder: *ComputeCommandEncoder) void {
        c.metal_compute_encoder_end_encoding(toC(encoder));
    }

    pub fn commandBufferCommit(command_buffer: *CommandBuffer) void {
        c.metal_command_buffer_commit(toC(command_buffer));
    }

    pub fn commandBufferWaitUntilCompleted(command_buffer: *CommandBuffer) bool {
        var error_buffer: [2048]u8 = undefined;
        @memset(&error_buffer, 0);
        const ok = c.metal_command_buffer_wait_until_completed(toC(command_buffer), error_buffer[0..].ptr, error_buffer.len) != 0;
        if (!ok) {
            logError("Metal command buffer failed", error_buffer[0..]);
        }
        return ok;
    }

    pub fn release(resource: ?*anyopaque) void {
        c.metal_release(resource);
    }
} else struct {
    // Empty struct for non-macOS platforms or when Metal is disabled
    pub const Device = anyopaque;
    pub const CommandQueue = anyopaque;
    pub const Library = anyopaque;
    pub const Function = anyopaque;
    pub const Buffer = anyopaque;
    pub const ComputePipelineState = anyopaque;
    pub const CommandBuffer = anyopaque;
    pub const ComputeCommandEncoder = anyopaque;

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

    pub fn bufferUploadF32(_: *Buffer, _: []const f32) bool {
        return false;
    }

    pub fn bufferDownloadF32(_: *Buffer, _: []f32) bool {
        return false;
    }

    pub fn deviceCreateLibraryFromSource(_: *Device, _: [*:0]const u8) ?*Library {
        return null;
    }

    pub fn libraryCreateFunction(_: *Library, _: [*:0]const u8) ?*Function {
        return null;
    }

    pub fn deviceCreateComputePipelineState(_: *Device, _: *Function) ?*ComputePipelineState {
        return null;
    }

    pub fn commandQueueCreateCommandBuffer(_: *CommandQueue) ?*CommandBuffer {
        return null;
    }

    pub fn commandBufferCreateComputeCommandEncoder(_: *CommandBuffer) ?*ComputeCommandEncoder {
        return null;
    }

    pub fn computeEncoderSetPipelineState(_: *ComputeCommandEncoder, _: *ComputePipelineState) void {}
    pub fn computeEncoderSetBuffer(_: *ComputeCommandEncoder, _: *Buffer, _: usize, _: u32) void {}
    pub fn computeEncoderSetBytes(_: *ComputeCommandEncoder, _: *const anyopaque, _: usize, _: u32) void {}
    pub fn computeEncoderDispatchThreads(_: *ComputeCommandEncoder, _: *ComputePipelineState, _: usize, _: usize, _: usize) void {}
    pub fn computeEncoderEndEncoding(_: *ComputeCommandEncoder) void {}
    pub fn commandBufferCommit(_: *CommandBuffer) void {}

    pub fn commandBufferWaitUntilCompleted(_: *CommandBuffer) bool {
        return false;
    }

    pub fn release(_: ?*anyopaque) void {}
};

/// Metal-specific implementation of a matrix
const MetalMatrix = struct {
    rows: usize,
    cols: usize,
    buffer: ?*Metal.Buffer, // GPU-side buffer
    host_data: []f32, // CPU-side data (for transfers)
    allocator: Allocator,
    owner: *MetalBackend,
    dirty: bool, // CPU data needs to be uploaded to GPU
    gpu_dirty: bool, // GPU data needs to be downloaded to CPU

    pub fn init(allocator: Allocator, rows: usize, cols: usize, owner: *MetalBackend) !*MetalMatrix {
        const metal_matrix = try allocator.create(MetalMatrix);
        errdefer allocator.destroy(metal_matrix);

        // Allocate host memory
        const host_data = try allocator.alloc(f32, rows * cols);
        errdefer allocator.free(host_data);
        @memset(host_data, 0);

        // Allocate GPU buffer (using sizeof(float) since Metal uses 32-bit floats)
        const buffer_size = rows * cols * @sizeOf(f32);
        const buffer = Metal.deviceCreateBuffer(owner.device.?, buffer_size, Metal.BufferOptions.StorageModeShared) orelse return error.BufferCreationFailed;
        errdefer Metal.release(buffer);

        metal_matrix.* = MetalMatrix{
            .rows = rows,
            .cols = cols,
            .buffer = buffer,
            .host_data = host_data,
            .allocator = allocator,
            .owner = owner,
            .dirty = true, // Mark as dirty to ensure first upload
            .gpu_dirty = false,
        };
        owner.stats.buffer_allocations += 1;

        return metal_matrix;
    }

    pub fn deinit(self: *MetalMatrix) void {
        Metal.release(self.buffer);

        // Free CPU memory
        self.allocator.free(self.host_data);

        self.allocator.destroy(self);
    }

    // Sync CPU data to GPU if needed
    pub fn syncToGPU(self: *MetalMatrix) bool {
        if (!self.dirty) return true;

        const buffer = self.buffer orelse return false;
        if (!Metal.bufferUploadF32(buffer, self.host_data)) {
            return false;
        }
        self.owner.stats.host_to_device_transfers += 1;
        self.owner.stats.host_to_device_bytes += self.host_data.len * @sizeOf(f32);

        self.dirty = false;
        self.gpu_dirty = false;
        return true;
    }

    // Sync GPU data to CPU
    pub fn syncToCPU(self: *MetalMatrix) bool {
        if (!self.gpu_dirty) return true;

        if (!self.owner.synchronizeQueued()) return false;

        const buffer = self.buffer orelse return false;
        if (!Metal.bufferDownloadF32(buffer, self.host_data)) {
            return false;
        }
        self.owner.stats.device_to_host_transfers += 1;
        self.owner.stats.device_to_host_bytes += self.host_data.len * @sizeOf(f32);

        self.dirty = false;
        self.gpu_dirty = false;
        return true;
    }

    pub fn markHostModified(self: *MetalMatrix) void {
        self.dirty = true;
        self.gpu_dirty = false;
    }

    pub fn markGPUModified(self: *MetalMatrix) void {
        self.dirty = false;
        self.gpu_dirty = true;
    }
};

/// Metal backend implementation
pub const MetalBackend = struct {
    allocator: Allocator,
    stats: backend.RuntimeStats,
    batch_active: bool,
    synchronized_kernel_count: usize,

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
    matrix_sum_rows_pipeline: ?*Metal.ComputePipelineState,
    matrix_extract_batch_pipeline: ?*Metal.ComputePipelineState,

    // Activation function pipelines
    linear_pipeline: ?*Metal.ComputePipelineState,
    linear_derivative_pipeline: ?*Metal.ComputePipelineState,
    sigmoid_pipeline: ?*Metal.ComputePipelineState,
    sigmoid_derivative_pipeline: ?*Metal.ComputePipelineState,
    relu_pipeline: ?*Metal.ComputePipelineState,
    relu_derivative_pipeline: ?*Metal.ComputePipelineState,
    tanh_pipeline: ?*Metal.ComputePipelineState,
    tanh_derivative_pipeline: ?*Metal.ComputePipelineState,
    swish_pipeline: ?*Metal.ComputePipelineState,
    swish_derivative_pipeline: ?*Metal.ComputePipelineState,

    // Softmax pipelines
    softmax_find_max_pipeline: ?*Metal.ComputePipelineState,
    softmax_exp_sum_pipeline: ?*Metal.ComputePipelineState,
    softmax_normalize_pipeline: ?*Metal.ComputePipelineState,
    softmax_rows_pipeline: ?*Metal.ComputePipelineState,

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

        const metal_backend = allocator.create(MetalBackend) catch |err| {
            Metal.release(device);
            return err;
        };
        metal_backend.* = MetalBackend{
            .allocator = allocator,
            .stats = .{},
            .batch_active = false,
            .synchronized_kernel_count = 0,
            .device = device,
            .command_queue = null, // Initialize all fields to null
            .library = null,
            .matrix_multiply_pipeline = null,
            .element_wise_add_pipeline = null,
            .element_wise_subtract_pipeline = null,
            .element_wise_multiply_pipeline = null,
            .matrix_scale_pipeline = null,
            .matrix_transpose_pipeline = null,
            .matrix_sum_rows_pipeline = null,
            .matrix_extract_batch_pipeline = null,
            .linear_pipeline = null,
            .linear_derivative_pipeline = null,
            .sigmoid_pipeline = null,
            .sigmoid_derivative_pipeline = null,
            .relu_pipeline = null,
            .relu_derivative_pipeline = null,
            .tanh_pipeline = null,
            .tanh_derivative_pipeline = null,
            .swish_pipeline = null,
            .swish_derivative_pipeline = null,
            .softmax_find_max_pipeline = null,
            .softmax_exp_sum_pipeline = null,
            .softmax_normalize_pipeline = null,
            .softmax_rows_pipeline = null,
            .glu_pipeline = null,
            .swiglu_pipeline = null,
        };
        errdefer metal_backend.deinit();

        // Create command queue, etc. (handle potential errors)
        metal_backend.command_queue = Metal.deviceCreateCommandQueue(device) orelse return error.CommandQueueCreationFailed;
        metal_backend.library = Metal.deviceCreateLibraryFromSource(device, shader_source) orelse return error.ShaderCompilationFailed;
        try metal_backend.loadPipelines();

        return metal_backend;
    }

    /// Free all resources used by the Metal backend
    pub fn deinit(self: *MetalBackend) void {
        self.batch_active = false;
        _ = self.synchronizeQueued();
        Metal.release(self.matrix_multiply_pipeline);
        Metal.release(self.element_wise_add_pipeline);
        Metal.release(self.element_wise_subtract_pipeline);
        Metal.release(self.element_wise_multiply_pipeline);
        Metal.release(self.matrix_scale_pipeline);
        Metal.release(self.matrix_transpose_pipeline);
        Metal.release(self.matrix_sum_rows_pipeline);
        Metal.release(self.matrix_extract_batch_pipeline);
        Metal.release(self.linear_pipeline);
        Metal.release(self.linear_derivative_pipeline);
        Metal.release(self.sigmoid_pipeline);
        Metal.release(self.sigmoid_derivative_pipeline);
        Metal.release(self.relu_pipeline);
        Metal.release(self.relu_derivative_pipeline);
        Metal.release(self.tanh_pipeline);
        Metal.release(self.tanh_derivative_pipeline);
        Metal.release(self.swish_pipeline);
        Metal.release(self.swish_derivative_pipeline);
        Metal.release(self.softmax_find_max_pipeline);
        Metal.release(self.softmax_exp_sum_pipeline);
        Metal.release(self.softmax_normalize_pipeline);
        Metal.release(self.softmax_rows_pipeline);
        Metal.release(self.glu_pipeline);
        Metal.release(self.swiglu_pipeline);
        Metal.release(self.library);
        Metal.release(self.command_queue);
        Metal.release(self.device);
        self.allocator.destroy(self);
    }

    fn loadPipelines(self: *MetalBackend) !void {
        try self.loadPipeline(&self.matrix_multiply_pipeline, "matrix_multiply");
        try self.loadPipeline(&self.element_wise_add_pipeline, "matrix_add");
        try self.loadPipeline(&self.element_wise_subtract_pipeline, "matrix_subtract");
        try self.loadPipeline(&self.element_wise_multiply_pipeline, "matrix_element_wise_multiply");
        try self.loadPipeline(&self.matrix_scale_pipeline, "matrix_scale");
        try self.loadPipeline(&self.matrix_transpose_pipeline, "matrix_transpose");
        try self.loadPipeline(&self.matrix_sum_rows_pipeline, "matrix_sum_rows");
        try self.loadPipeline(&self.matrix_extract_batch_pipeline, "matrix_extract_batch");
        try self.loadPipeline(&self.linear_pipeline, "apply_linear");
        try self.loadPipeline(&self.linear_derivative_pipeline, "apply_linear_derivative");
        try self.loadPipeline(&self.sigmoid_pipeline, "apply_sigmoid");
        try self.loadPipeline(&self.sigmoid_derivative_pipeline, "apply_sigmoid_derivative");
        try self.loadPipeline(&self.relu_pipeline, "apply_relu");
        try self.loadPipeline(&self.relu_derivative_pipeline, "apply_relu_derivative");
        try self.loadPipeline(&self.tanh_pipeline, "apply_tanh");
        try self.loadPipeline(&self.tanh_derivative_pipeline, "apply_tanh_derivative");
        try self.loadPipeline(&self.swish_pipeline, "apply_swish");
        try self.loadPipeline(&self.swish_derivative_pipeline, "apply_swish_derivative");
        try self.loadPipeline(&self.softmax_find_max_pipeline, "softmax_find_max");
        try self.loadPipeline(&self.softmax_exp_sum_pipeline, "softmax_exp_and_sum");
        try self.loadPipeline(&self.softmax_normalize_pipeline, "softmax_normalize");
        try self.loadPipeline(&self.softmax_rows_pipeline, "softmax_rows");
        try self.loadPipeline(&self.glu_pipeline, "apply_glu");
        try self.loadPipeline(&self.swiglu_pipeline, "apply_swiglu");
    }

    pub fn runtimeStats(ptr: *anyopaque) backend.RuntimeStats {
        const self = @as(*MetalBackend, @ptrCast(@alignCast(ptr)));
        return self.stats;
    }

    pub fn resetRuntimeStats(ptr: *anyopaque) void {
        const self = @as(*MetalBackend, @ptrCast(@alignCast(ptr)));
        self.stats = .{};
        self.synchronized_kernel_count = 0;
    }

    pub fn beginBatch(ptr: *anyopaque) !void {
        const self = @as(*MetalBackend, @ptrCast(@alignCast(ptr)));
        if (self.batch_active) return error.BatchAlreadyActive;
        self.batch_active = true;
    }

    pub fn endBatch(ptr: *anyopaque) !void {
        const self = @as(*MetalBackend, @ptrCast(@alignCast(ptr)));
        if (!self.batch_active) return error.NoActiveBatch;
        self.batch_active = false;
        if (!self.synchronizeQueued()) return error.CommandExecutionFailed;
    }

    pub fn synchronize(ptr: *anyopaque) !void {
        const self = @as(*MetalBackend, @ptrCast(@alignCast(ptr)));
        if (!self.synchronizeQueued()) return error.CommandExecutionFailed;
    }

    fn completeSubmittedKernel(self: *MetalBackend, command_buffer: *Metal.CommandBuffer) bool {
        self.stats.kernel_launches += 1;
        if (self.batch_active) return true;
        if (!Metal.commandBufferWaitUntilCompleted(command_buffer)) return false;
        self.stats.synchronizations += 1;
        self.synchronized_kernel_count = self.stats.kernel_launches;
        return true;
    }

    fn synchronizeQueued(self: *MetalBackend) bool {
        if (self.synchronized_kernel_count == self.stats.kernel_launches) return true;

        const command_queue = self.command_queue orelse return false;
        const command_buffer = Metal.commandQueueCreateCommandBuffer(command_queue) orelse return false;
        defer Metal.release(command_buffer);
        Metal.commandBufferCommit(command_buffer);
        if (!Metal.commandBufferWaitUntilCompleted(command_buffer)) return false;

        self.stats.synchronizations += 1;
        self.synchronized_kernel_count = self.stats.kernel_launches;
        return true;
    }

    fn loadPipeline(self: *MetalBackend, slot: *?*Metal.ComputePipelineState, name: [*:0]const u8) !void {
        const library = self.library orelse return error.ShaderCompilationFailed;
        const device = self.device orelse return error.MetalDeviceNotFound;
        const function = Metal.libraryCreateFunction(library, name) orelse {
            std.log.err("Metal shader function not found: {s}", .{std.mem.span(name)});
            return error.ShaderFunctionNotFound;
        };
        defer Metal.release(function);

        slot.* = Metal.deviceCreateComputePipelineState(device, function) orelse return error.PipelineCreationFailed;
    }

    // Helper to get Metal matrix from abstract Matrix
    fn getMetalMatrix(matrix: *const Matrix) *MetalMatrix {
        // Use @alignCast before @ptrCast for safety
        return @as(*MetalMatrix, @ptrCast(@alignCast(matrix.impl_data)));
    }

    fn elementCount(matrix: *const Matrix) usize {
        return matrix.rows * matrix.cols;
    }

    fn toU32(value: usize) u32 {
        return std.math.cast(u32, value) orelse @panic("Metal backend dimension exceeds u32 range");
    }

    fn setU32(encoder: *Metal.ComputeCommandEncoder, value: u32, index: u32) void {
        var local = value;
        Metal.computeEncoderSetBytes(encoder, @as(*const anyopaque, @ptrCast(&local)), @sizeOf(u32), index);
    }

    fn setF32(encoder: *Metal.ComputeCommandEncoder, value: f32, index: u32) void {
        var local = value;
        Metal.computeEncoderSetBytes(encoder, @as(*const anyopaque, @ptrCast(&local)), @sizeOf(f32), index);
    }

    fn ensureHostData(metal_matrix: *MetalMatrix) void {
        if (!metal_matrix.syncToCPU()) {
            std.log.warn("Metal buffer download failed; using existing host data", .{});
        }
    }

    fn dispatchMatrixMultiply(self: *MetalBackend, a: *const Matrix, b: *const Matrix, result: *Matrix) bool {
        const pipeline = self.matrix_multiply_pipeline orelse return false;
        const command_queue = self.command_queue orelse return false;
        const a_metal = getMetalMatrix(a);
        const b_metal = getMetalMatrix(b);
        const result_metal = getMetalMatrix(result);

        if (!a_metal.syncToGPU() or !b_metal.syncToGPU()) return false;
        const a_buffer = a_metal.buffer orelse return false;
        const b_buffer = b_metal.buffer orelse return false;
        const result_buffer = result_metal.buffer orelse return false;

        const command_buffer = Metal.commandQueueCreateCommandBuffer(command_queue) orelse return false;
        defer Metal.release(command_buffer);
        const encoder = Metal.commandBufferCreateComputeCommandEncoder(command_buffer) orelse return false;
        defer Metal.release(encoder);

        Metal.computeEncoderSetPipelineState(encoder, pipeline);
        Metal.computeEncoderSetBuffer(encoder, a_buffer, 0, 0);
        Metal.computeEncoderSetBuffer(encoder, b_buffer, 0, 1);
        Metal.computeEncoderSetBuffer(encoder, result_buffer, 0, 2);
        setU32(encoder, toU32(a.rows), 3);
        setU32(encoder, toU32(a.cols), 4);
        setU32(encoder, toU32(b.cols), 5);
        Metal.computeEncoderDispatchThreads(encoder, pipeline, b.cols, a.rows, 1);
        Metal.computeEncoderEndEncoding(encoder);
        Metal.commandBufferCommit(command_buffer);
        if (!self.completeSubmittedKernel(command_buffer)) return false;
        result_metal.markGPUModified();
        return true;
    }

    fn dispatchBinaryMatrixKernel(self: *MetalBackend, pipeline: ?*Metal.ComputePipelineState, a: *const Matrix, b: *const Matrix, result: *Matrix) bool {
        const pipeline_state = pipeline orelse return false;
        const command_queue = self.command_queue orelse return false;
        const a_metal = getMetalMatrix(a);
        const b_metal = getMetalMatrix(b);
        const result_metal = getMetalMatrix(result);

        if (!a_metal.syncToGPU() or !b_metal.syncToGPU()) return false;
        const a_buffer = a_metal.buffer orelse return false;
        const b_buffer = b_metal.buffer orelse return false;
        const result_buffer = result_metal.buffer orelse return false;

        const command_buffer = Metal.commandQueueCreateCommandBuffer(command_queue) orelse return false;
        defer Metal.release(command_buffer);
        const encoder = Metal.commandBufferCreateComputeCommandEncoder(command_buffer) orelse return false;
        defer Metal.release(encoder);

        Metal.computeEncoderSetPipelineState(encoder, pipeline_state);
        Metal.computeEncoderSetBuffer(encoder, a_buffer, 0, 0);
        Metal.computeEncoderSetBuffer(encoder, b_buffer, 0, 1);
        Metal.computeEncoderSetBuffer(encoder, result_buffer, 0, 2);
        setU32(encoder, toU32(a.rows), 3);
        setU32(encoder, toU32(a.cols), 4);
        Metal.computeEncoderDispatchThreads(encoder, pipeline_state, a.cols, a.rows, 1);
        Metal.computeEncoderEndEncoding(encoder);
        Metal.commandBufferCommit(command_buffer);
        if (!self.completeSubmittedKernel(command_buffer)) return false;
        result_metal.markGPUModified();
        return true;
    }

    fn dispatchScale(self: *MetalBackend, matrix: *const Matrix, scalar: f64, result: *Matrix) bool {
        const pipeline = self.matrix_scale_pipeline orelse return false;
        const command_queue = self.command_queue orelse return false;
        const matrix_metal = getMetalMatrix(matrix);
        const result_metal = getMetalMatrix(result);

        if (!matrix_metal.syncToGPU()) return false;
        const matrix_buffer = matrix_metal.buffer orelse return false;
        const result_buffer = result_metal.buffer orelse return false;

        const command_buffer = Metal.commandQueueCreateCommandBuffer(command_queue) orelse return false;
        defer Metal.release(command_buffer);
        const encoder = Metal.commandBufferCreateComputeCommandEncoder(command_buffer) orelse return false;
        defer Metal.release(encoder);

        Metal.computeEncoderSetPipelineState(encoder, pipeline);
        Metal.computeEncoderSetBuffer(encoder, matrix_buffer, 0, 0);
        Metal.computeEncoderSetBuffer(encoder, result_buffer, 0, 1);
        setF32(encoder, @floatCast(scalar), 2);
        setU32(encoder, toU32(matrix.rows), 3);
        setU32(encoder, toU32(matrix.cols), 4);
        Metal.computeEncoderDispatchThreads(encoder, pipeline, matrix.cols, matrix.rows, 1);
        Metal.computeEncoderEndEncoding(encoder);
        Metal.commandBufferCommit(command_buffer);
        if (!self.completeSubmittedKernel(command_buffer)) return false;
        result_metal.markGPUModified();
        return true;
    }

    fn dispatchTranspose(self: *MetalBackend, matrix: *const Matrix, result: *Matrix) bool {
        const pipeline = self.matrix_transpose_pipeline orelse return false;
        const command_queue = self.command_queue orelse return false;
        const matrix_metal = getMetalMatrix(matrix);
        const result_metal = getMetalMatrix(result);

        if (!matrix_metal.syncToGPU()) return false;
        const matrix_buffer = matrix_metal.buffer orelse return false;
        const result_buffer = result_metal.buffer orelse return false;

        const command_buffer = Metal.commandQueueCreateCommandBuffer(command_queue) orelse return false;
        defer Metal.release(command_buffer);
        const encoder = Metal.commandBufferCreateComputeCommandEncoder(command_buffer) orelse return false;
        defer Metal.release(encoder);

        Metal.computeEncoderSetPipelineState(encoder, pipeline);
        Metal.computeEncoderSetBuffer(encoder, matrix_buffer, 0, 0);
        Metal.computeEncoderSetBuffer(encoder, result_buffer, 0, 1);
        setU32(encoder, toU32(matrix.rows), 2);
        setU32(encoder, toU32(matrix.cols), 3);
        Metal.computeEncoderDispatchThreads(encoder, pipeline, matrix.rows, matrix.cols, 1);
        Metal.computeEncoderEndEncoding(encoder);
        Metal.commandBufferCommit(command_buffer);
        if (!self.completeSubmittedKernel(command_buffer)) return false;
        result_metal.markGPUModified();
        return true;
    }

    fn dispatchSumRows(self: *MetalBackend, matrix: *const Matrix, result: *Matrix) bool {
        const pipeline = self.matrix_sum_rows_pipeline orelse return false;
        const command_queue = self.command_queue orelse return false;
        const matrix_metal = getMetalMatrix(matrix);
        const result_metal = getMetalMatrix(result);

        if (!matrix_metal.syncToGPU()) return false;
        const matrix_buffer = matrix_metal.buffer orelse return false;
        const result_buffer = result_metal.buffer orelse return false;

        const command_buffer = Metal.commandQueueCreateCommandBuffer(command_queue) orelse return false;
        defer Metal.release(command_buffer);
        const encoder = Metal.commandBufferCreateComputeCommandEncoder(command_buffer) orelse return false;
        defer Metal.release(encoder);

        Metal.computeEncoderSetPipelineState(encoder, pipeline);
        Metal.computeEncoderSetBuffer(encoder, matrix_buffer, 0, 0);
        Metal.computeEncoderSetBuffer(encoder, result_buffer, 0, 1);
        setU32(encoder, toU32(matrix.rows), 2);
        setU32(encoder, toU32(matrix.cols), 3);
        Metal.computeEncoderDispatchThreads(encoder, pipeline, matrix.cols, 1, 1);
        Metal.computeEncoderEndEncoding(encoder);
        Metal.commandBufferCommit(command_buffer);
        if (!self.completeSubmittedKernel(command_buffer)) return false;
        result_metal.markGPUModified();
        return true;
    }

    fn dispatchExtractBatch(self: *MetalBackend, matrix: *const Matrix, start: usize, result: *Matrix) bool {
        const pipeline = self.matrix_extract_batch_pipeline orelse return false;
        const command_queue = self.command_queue orelse return false;
        const matrix_metal = getMetalMatrix(matrix);
        const result_metal = getMetalMatrix(result);

        if (!matrix_metal.syncToGPU()) return false;
        const matrix_buffer = matrix_metal.buffer orelse return false;
        const result_buffer = result_metal.buffer orelse return false;

        const command_buffer = Metal.commandQueueCreateCommandBuffer(command_queue) orelse return false;
        defer Metal.release(command_buffer);
        const encoder = Metal.commandBufferCreateComputeCommandEncoder(command_buffer) orelse return false;
        defer Metal.release(encoder);

        Metal.computeEncoderSetPipelineState(encoder, pipeline);
        Metal.computeEncoderSetBuffer(encoder, matrix_buffer, 0, 0);
        Metal.computeEncoderSetBuffer(encoder, result_buffer, 0, 1);
        setU32(encoder, toU32(start), 2);
        setU32(encoder, toU32(result.rows), 3);
        setU32(encoder, toU32(result.cols), 4);
        Metal.computeEncoderDispatchThreads(encoder, pipeline, result.cols, result.rows, 1);
        Metal.computeEncoderEndEncoding(encoder);
        Metal.commandBufferCommit(command_buffer);
        if (!self.completeSubmittedKernel(command_buffer)) return false;
        result_metal.markGPUModified();
        return true;
    }

    fn dispatchUnaryKernel(self: *MetalBackend, pipeline: ?*Metal.ComputePipelineState, matrix: *const Matrix, result: *Matrix) bool {
        const pipeline_state = pipeline orelse return false;
        const command_queue = self.command_queue orelse return false;
        const matrix_metal = getMetalMatrix(matrix);
        const result_metal = getMetalMatrix(result);

        if (!matrix_metal.syncToGPU()) return false;
        const matrix_buffer = matrix_metal.buffer orelse return false;
        const result_buffer = result_metal.buffer orelse return false;

        const command_buffer = Metal.commandQueueCreateCommandBuffer(command_queue) orelse return false;
        defer Metal.release(command_buffer);
        const encoder = Metal.commandBufferCreateComputeCommandEncoder(command_buffer) orelse return false;
        defer Metal.release(encoder);

        Metal.computeEncoderSetPipelineState(encoder, pipeline_state);
        Metal.computeEncoderSetBuffer(encoder, matrix_buffer, 0, 0);
        Metal.computeEncoderSetBuffer(encoder, result_buffer, 0, 1);
        setU32(encoder, toU32(elementCount(matrix)), 2);
        Metal.computeEncoderDispatchThreads(encoder, pipeline_state, elementCount(matrix), 1, 1);
        Metal.computeEncoderEndEncoding(encoder);
        Metal.commandBufferCommit(command_buffer);
        if (!self.completeSubmittedKernel(command_buffer)) return false;
        result_metal.markGPUModified();
        return true;
    }

    fn dispatchSoftmax(self: *MetalBackend, matrix: *const Matrix, result: *Matrix) bool {
        const pipeline = self.softmax_rows_pipeline orelse return false;
        const command_queue = self.command_queue orelse return false;
        const matrix_metal = getMetalMatrix(matrix);
        const result_metal = getMetalMatrix(result);

        if (!matrix_metal.syncToGPU()) return false;
        const matrix_buffer = matrix_metal.buffer orelse return false;
        const result_buffer = result_metal.buffer orelse return false;

        const command_buffer = Metal.commandQueueCreateCommandBuffer(command_queue) orelse return false;
        defer Metal.release(command_buffer);
        const encoder = Metal.commandBufferCreateComputeCommandEncoder(command_buffer) orelse return false;
        defer Metal.release(encoder);

        Metal.computeEncoderSetPipelineState(encoder, pipeline);
        Metal.computeEncoderSetBuffer(encoder, matrix_buffer, 0, 0);
        Metal.computeEncoderSetBuffer(encoder, result_buffer, 0, 1);
        setU32(encoder, toU32(matrix.rows), 2);
        setU32(encoder, toU32(matrix.cols), 3);
        Metal.computeEncoderDispatchThreads(encoder, pipeline, matrix.rows, 1, 1);
        Metal.computeEncoderEndEncoding(encoder);
        Metal.commandBufferCommit(command_buffer);
        if (!self.completeSubmittedKernel(command_buffer)) return false;
        result_metal.markGPUModified();
        return true;
    }

    fn dispatchGatedKernel(self: *MetalBackend, pipeline: ?*Metal.ComputePipelineState, linear_part: *const Matrix, gating_part: *const Matrix, result: *Matrix) bool {
        const pipeline_state = pipeline orelse return false;
        const command_queue = self.command_queue orelse return false;
        const linear_metal = getMetalMatrix(linear_part);
        const gating_metal = getMetalMatrix(gating_part);
        const result_metal = getMetalMatrix(result);

        if (!linear_metal.syncToGPU() or !gating_metal.syncToGPU()) return false;
        const linear_buffer = linear_metal.buffer orelse return false;
        const gating_buffer = gating_metal.buffer orelse return false;
        const result_buffer = result_metal.buffer orelse return false;

        const command_buffer = Metal.commandQueueCreateCommandBuffer(command_queue) orelse return false;
        defer Metal.release(command_buffer);
        const encoder = Metal.commandBufferCreateComputeCommandEncoder(command_buffer) orelse return false;
        defer Metal.release(encoder);

        Metal.computeEncoderSetPipelineState(encoder, pipeline_state);
        Metal.computeEncoderSetBuffer(encoder, linear_buffer, 0, 0);
        Metal.computeEncoderSetBuffer(encoder, gating_buffer, 0, 1);
        Metal.computeEncoderSetBuffer(encoder, result_buffer, 0, 2);
        setU32(encoder, toU32(elementCount(linear_part)), 3);
        Metal.computeEncoderDispatchThreads(encoder, pipeline_state, elementCount(linear_part), 1, 1);
        Metal.computeEncoderEndEncoding(encoder);
        Metal.commandBufferCommit(command_buffer);
        if (!self.completeSubmittedKernel(command_buffer)) return false;
        result_metal.markGPUModified();
        return true;
    }

    // --- Implementation Functions (Make Public) ---

    pub fn initMatrix(ptr: *anyopaque, allocator: Allocator, rows: usize, cols: usize) error{OutOfMemory}!*Matrix {
        // Correctly cast ptr with alignment check
        const self = @as(*MetalBackend, @ptrCast(@alignCast(ptr)));

        // Create Metal matrix
        const metal_matrix = MetalMatrix.init(allocator, rows, cols, self) catch |err| {
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
        const allocator = metal_matrix.allocator;
        metal_matrix.deinit();

        // Free the abstract Matrix
        allocator.destroy(matrix);
    }

    pub fn copyMatrix(ptr: *anyopaque, source: *const Matrix, allocator: Allocator) error{OutOfMemory}!*Matrix {
        _ = @as(*MetalBackend, @ptrCast(@alignCast(ptr)));
        const source_metal = getMetalMatrix(source);

        // Create new matrix with same dimensions
        const result = try initMatrix(ptr, allocator, source.rows, source.cols);
        const result_metal = getMetalMatrix(result);

        // Copy data from source to result
        ensureHostData(source_metal);
        @memcpy(result_metal.host_data, source_metal.host_data);
        result_metal.markHostModified();

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
        if (!metal_matrix.syncToCPU()) {
            std.log.warn("Metal buffer download failed while reading matrix element", .{});
        }

        // Return element
        return @floatCast(metal_matrix.host_data[index]);
    }

    pub fn writeMatrixF32(_: *anyopaque, matrix: *Matrix, values: []const f32) bool {
        const metal_matrix = getMetalMatrix(matrix);
        if (values.len != metal_matrix.host_data.len) return false;
        @memcpy(metal_matrix.host_data, values);
        metal_matrix.markHostModified();
        return true;
    }

    pub fn readMatrixF32(_: *anyopaque, matrix: *const Matrix, values: []f32) bool {
        const metal_matrix = getMetalMatrix(matrix);
        if (values.len != metal_matrix.host_data.len) return false;
        ensureHostData(metal_matrix);
        @memcpy(values, metal_matrix.host_data);
        return true;
    }

    pub fn setMatrixElement(_: *anyopaque, matrix: *Matrix, row: usize, col: usize, value: f64) void {
        const metal_matrix = getMetalMatrix(matrix);

        // Update CPU data
        metal_matrix.host_data[row * matrix.cols + col] = @floatCast(value);
        metal_matrix.markHostModified();
    }

    pub fn fillMatrix(_: *anyopaque, matrix: *Matrix, value: f64) void {
        const metal_matrix = getMetalMatrix(matrix);

        // Fill CPU data
        for (metal_matrix.host_data) |*element| {
            element.* = @floatCast(value);
        }

        metal_matrix.markHostModified();
    }

    pub fn dotProduct(ptr: *anyopaque, a: *const Matrix, b: *const Matrix, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!*Matrix {
        // Check dimensions
        if (a.cols != b.rows) {
            return error.DimensionMismatch;
        }

        const self = @as(*MetalBackend, @ptrCast(@alignCast(ptr)));

        // Create result matrix
        const result = try initMatrix(ptr, allocator, a.rows, b.cols);

        // Get Metal matrices
        const a_metal = getMetalMatrix(a);
        const b_metal = getMetalMatrix(b);
        const result_metal = getMetalMatrix(result);

        if (self.dispatchMatrixMultiply(a, b, result)) {
            return result;
        }

        ensureHostData(a_metal);
        ensureHostData(b_metal);

        // FALLBACK: Perform CPU calculation for now
        // This ensures we have valid data when accessed
        for (0..a.rows) |i| {
            for (0..b.cols) |j| {
                var sum: f32 = 0.0;
                for (0..a.cols) |k| {
                    sum += a_metal.host_data[i * a.cols + k] * b_metal.host_data[k * b.cols + j];
                }
                result_metal.host_data[i * b.cols + j] = sum;
            }
        }
        result_metal.markHostModified();

        return result;
    }

    pub fn add(ptr: *anyopaque, a: *const Matrix, b: *const Matrix, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!*Matrix {
        // Check dimensions
        if (a.rows != b.rows or a.cols != b.cols) {
            return error.DimensionMismatch;
        }

        const self = @as(*MetalBackend, @ptrCast(@alignCast(ptr)));

        // Create result matrix
        const result = try initMatrix(ptr, allocator, a.rows, a.cols);

        // Get Metal matrices
        const a_metal = getMetalMatrix(a);
        const b_metal = getMetalMatrix(b);
        const result_metal = getMetalMatrix(result);

        if (self.dispatchBinaryMatrixKernel(self.element_wise_add_pipeline, a, b, result)) {
            return result;
        }

        ensureHostData(a_metal);
        ensureHostData(b_metal);

        // FALLBACK: Perform CPU calculation for now
        for (0..a.rows * a.cols) |i| {
            result_metal.host_data[i] = a_metal.host_data[i] + b_metal.host_data[i];
        }
        result_metal.markHostModified();

        return result;
    }

    pub fn subtract(ptr: *anyopaque, a: *const Matrix, b: *const Matrix, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!*Matrix {
        if (a.rows != b.rows or a.cols != b.cols) {
            return error.DimensionMismatch;
        }

        const self = @as(*MetalBackend, @ptrCast(@alignCast(ptr)));

        const result = try initMatrix(ptr, allocator, a.rows, a.cols);
        const a_metal = getMetalMatrix(a);
        const b_metal = getMetalMatrix(b);
        const result_metal = getMetalMatrix(result);

        if (self.dispatchBinaryMatrixKernel(self.element_wise_subtract_pipeline, a, b, result)) {
            return result;
        }

        ensureHostData(a_metal);
        ensureHostData(b_metal);

        for (0..a.rows * a.cols) |i| {
            result_metal.host_data[i] = a_metal.host_data[i] - b_metal.host_data[i];
        }
        result_metal.markHostModified();

        return result;
    }

    pub fn elementWiseMultiply(ptr: *anyopaque, a: *const Matrix, b: *const Matrix, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!*Matrix {
        // Check dimensions
        if (a.rows != b.rows or a.cols != b.cols) {
            return error.DimensionMismatch;
        }

        const self = @as(*MetalBackend, @ptrCast(@alignCast(ptr)));

        // Create result matrix
        const result = try initMatrix(ptr, allocator, a.rows, a.cols);

        // Get Metal matrices
        const a_metal = getMetalMatrix(a);
        const b_metal = getMetalMatrix(b);
        const result_metal = getMetalMatrix(result);

        if (self.dispatchBinaryMatrixKernel(self.element_wise_multiply_pipeline, a, b, result)) {
            return result;
        }

        ensureHostData(a_metal);
        ensureHostData(b_metal);

        // FALLBACK: Perform CPU calculation for now
        for (0..a.rows * a.cols) |i| {
            result_metal.host_data[i] = a_metal.host_data[i] * b_metal.host_data[i];
        }
        result_metal.markHostModified();

        return result;
    }

    pub fn scale(ptr: *anyopaque, matrix: *const Matrix, scalar: f64, allocator: Allocator) error{OutOfMemory}!*Matrix {
        const self = @as(*MetalBackend, @ptrCast(@alignCast(ptr)));

        // Create result matrix
        const result = try initMatrix(ptr, allocator, matrix.rows, matrix.cols);

        // Get Metal matrices
        const matrix_metal = getMetalMatrix(matrix);
        const result_metal = getMetalMatrix(result);

        if (self.dispatchScale(matrix, scalar, result)) {
            return result;
        }

        ensureHostData(matrix_metal);

        for (0..matrix.rows * matrix.cols) |i| {
            result_metal.host_data[i] = matrix_metal.host_data[i] * @as(f32, @floatCast(scalar));
        }
        result_metal.markHostModified();

        return result;
    }

    pub fn sumRows(ptr: *anyopaque, matrix: *const Matrix, allocator: Allocator) error{OutOfMemory}!*Matrix {
        const self = @as(*MetalBackend, @ptrCast(@alignCast(ptr)));

        // Create result matrix (1 x cols)
        const result = try initMatrix(ptr, allocator, 1, matrix.cols);

        // Get Metal matrices
        const matrix_metal = getMetalMatrix(matrix);
        const result_metal = getMetalMatrix(result);

        if (self.dispatchSumRows(matrix, result)) {
            return result;
        }

        ensureHostData(matrix_metal);

        for (0..matrix.cols) |j| {
            var sum: f32 = 0;
            for (0..matrix.rows) |i| {
                sum += matrix_metal.host_data[i * matrix.cols + j];
            }
            result_metal.host_data[j] = sum;
        }
        result_metal.markHostModified();

        return result;
    }

    pub fn transpose(ptr: *anyopaque, matrix: *const Matrix, allocator: Allocator) error{OutOfMemory}!*Matrix {
        const self = @as(*MetalBackend, @ptrCast(@alignCast(ptr)));

        // Create result matrix with transposed dimensions
        const result = try initMatrix(ptr, allocator, matrix.cols, matrix.rows);

        // Get Metal matrices
        const matrix_metal = getMetalMatrix(matrix);
        const result_metal = getMetalMatrix(result);

        if (self.dispatchTranspose(matrix, result)) {
            return result;
        }

        ensureHostData(matrix_metal);

        // FALLBACK: Perform CPU calculation for now
        for (0..matrix.rows) |i| {
            for (0..matrix.cols) |j| {
                result_metal.host_data[j * matrix.rows + i] = matrix_metal.host_data[i * matrix.cols + j];
            }
        }
        result_metal.markHostModified();

        return result;
    }

    pub fn extractBatch(ptr: *anyopaque, matrix: *const Matrix, start: usize, end: usize, allocator: Allocator) error{ OutOfMemory, InvalidBatchIndices }!*Matrix {
        // Check indices
        if (start >= end or end > matrix.rows) {
            return error.InvalidBatchIndices;
        }

        const self = @as(*MetalBackend, @ptrCast(@alignCast(ptr)));

        // Create result matrix
        const batch_rows = end - start;
        const result = try initMatrix(ptr, allocator, batch_rows, matrix.cols);

        // Get Metal matrices
        const matrix_metal = getMetalMatrix(matrix);
        const result_metal = getMetalMatrix(result);

        if (self.dispatchExtractBatch(matrix, start, result)) {
            return result;
        }

        ensureHostData(matrix_metal);

        // Copy the batch from CPU data if GPU dispatch is not available.
        const src_offset = start * matrix.cols;
        @memcpy(result_metal.host_data, matrix_metal.host_data[src_offset..][0 .. batch_rows * matrix.cols]);

        result_metal.markHostModified();

        return result;
    }

    pub fn randomize(_: *anyopaque, matrix: *Matrix, min: f64, max: f64) void {
        const metal_matrix = getMetalMatrix(matrix);

        // Initialize random number generator
        var rng = std.Random.DefaultPrng.init(randomSeed());

        // Fill with random values
        for (metal_matrix.host_data) |*element| {
            const random_float = rng.random().float(f64);
            element.* = @floatCast(min + random_float * (max - min));
        }

        metal_matrix.markHostModified();
    }

    pub fn applyActivation(ptr: *anyopaque, matrix: *const Matrix, activation: *const fn (f64) f64, allocator: Allocator) error{OutOfMemory}!*Matrix {
        const self = @as(*MetalBackend, @ptrCast(@alignCast(ptr)));

        // Create result matrix
        const result = try initMatrix(ptr, allocator, matrix.rows, matrix.cols);

        // Get Metal matrices
        const matrix_metal = getMetalMatrix(matrix);
        const result_metal = getMetalMatrix(result);

        const pipeline: ?*Metal.ComputePipelineState = if (activation == Activation.linear)
            self.linear_pipeline
        else if (activation == Activation.linear_derivative)
            self.linear_derivative_pipeline
        else if (activation == Activation.sigmoid)
            self.sigmoid_pipeline
        else if (activation == Activation.sigmoid_derivative)
            self.sigmoid_derivative_pipeline
        else if (activation == Activation.relu)
            self.relu_pipeline
        else if (activation == Activation.relu_derivative)
            self.relu_derivative_pipeline
        else if (activation == Activation.tanh)
            self.tanh_pipeline
        else if (activation == Activation.tanh_derivative)
            self.tanh_derivative_pipeline
        else if (activation == Activation.swish)
            self.swish_pipeline
        else if (activation == Activation.swish_derivative)
            self.swish_derivative_pipeline
        else
            null;

        if (self.dispatchUnaryKernel(pipeline, matrix, result)) {
            return result;
        }

        ensureHostData(matrix_metal);

        // Unknown custom activations fall back to CPU.
        for (0..matrix.rows * matrix.cols) |i| {
            result_metal.host_data[i] = @floatCast(activation(@floatCast(matrix_metal.host_data[i])));
        }

        result_metal.markHostModified();

        return result;
    }

    pub fn applySoftmax(ptr: *anyopaque, matrix: *const Matrix, allocator: Allocator) error{OutOfMemory}!*Matrix {
        const self = @as(*MetalBackend, @ptrCast(@alignCast(ptr)));

        // Create result matrix
        const result = try initMatrix(ptr, allocator, matrix.rows, matrix.cols);

        // Get Metal matrices
        const matrix_metal = getMetalMatrix(matrix);
        const result_metal = getMetalMatrix(result);

        if (self.dispatchSoftmax(matrix, result)) {
            return result;
        }

        ensureHostData(matrix_metal);

        for (0..matrix.rows) |i| {
            var max_val: f32 = -std.math.inf(f32);
            for (0..matrix.cols) |j| {
                max_val = @max(max_val, matrix_metal.host_data[i * matrix.cols + j]);
            }

            var sum: f32 = 0.0;
            for (0..matrix.cols) |j| {
                const idx = i * matrix.cols + j;
                const exp_val = std.math.exp(matrix_metal.host_data[idx] - max_val);
                result_metal.host_data[idx] = exp_val;
                sum += exp_val;
            }

            for (0..matrix.cols) |j| {
                const idx = i * matrix.cols + j;
                result_metal.host_data[idx] /= sum;
            }
        }
        result_metal.markHostModified();

        return result;
    }

    pub fn applyGLU(ptr: *anyopaque, linear_part: *const Matrix, gating_part: *const Matrix, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!*Matrix {
        // Check dimensions
        if (linear_part.rows != gating_part.rows or linear_part.cols != gating_part.cols) {
            return error.DimensionMismatch;
        }

        const self = @as(*MetalBackend, @ptrCast(@alignCast(ptr)));

        // Create result matrix
        const result = try initMatrix(ptr, allocator, linear_part.rows, linear_part.cols);

        // Get Metal matrices
        const linear_metal = getMetalMatrix(linear_part);
        const gating_metal = getMetalMatrix(gating_part);
        const result_metal = getMetalMatrix(result);

        if (self.dispatchGatedKernel(self.glu_pipeline, linear_part, gating_part, result)) {
            return result;
        }

        ensureHostData(linear_metal);
        ensureHostData(gating_metal);

        for (0..linear_part.rows * linear_part.cols) |i| {
            const sigmoid = 1.0 / (1.0 + std.math.exp(-gating_metal.host_data[i]));
            result_metal.host_data[i] = linear_metal.host_data[i] * sigmoid;
        }
        result_metal.markHostModified();

        return result;
    }

    pub fn applySwiGLU(ptr: *anyopaque, linear_part: *const Matrix, gating_part: *const Matrix, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!*Matrix {
        // Check dimensions
        if (linear_part.rows != gating_part.rows or linear_part.cols != gating_part.cols) {
            return error.DimensionMismatch;
        }

        const self = @as(*MetalBackend, @ptrCast(@alignCast(ptr)));

        // Create result matrix
        const result = try initMatrix(ptr, allocator, linear_part.rows, linear_part.cols);

        // Get Metal matrices
        const linear_metal = getMetalMatrix(linear_part);
        const gating_metal = getMetalMatrix(gating_part);
        const result_metal = getMetalMatrix(result);

        if (self.dispatchGatedKernel(self.swiglu_pipeline, linear_part, gating_part, result)) {
            return result;
        }

        ensureHostData(linear_metal);
        ensureHostData(gating_metal);

        for (0..linear_part.rows * linear_part.cols) |i| {
            const gate = gating_metal.host_data[i];
            const sigmoid = 1.0 / (1.0 + std.math.exp(-gate));
            result_metal.host_data[i] = linear_metal.host_data[i] * gate * sigmoid;
        }
        result_metal.markHostModified();

        return result;
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

const CPUBackend = @import("cpu_backend.zig").CPUBackend;
const testing = std.testing;

fn initMetalBackendForTest(allocator: Allocator) !*MetalBackend {
    if (@import("builtin").os.tag != .macos or !enable_metal) {
        return error.SkipZigTest;
    }

    return MetalBackend.init(allocator) catch |err| switch (err) {
        error.MetalDeviceNotFound => error.SkipZigTest,
        else => err,
    };
}

fn fillMatrixPair(
    cpu_ptr: *anyopaque,
    metal_ptr: *anyopaque,
    cpu_matrix: *Matrix,
    metal_matrix: *Matrix,
    values: []const f64,
) void {
    for (values, 0..) |value, index| {
        const row = index / cpu_matrix.cols;
        const col = index % cpu_matrix.cols;
        CPUBackend.setMatrixElement(cpu_ptr, cpu_matrix, row, col, value);
        MetalBackend.setMatrixElement(metal_ptr, metal_matrix, row, col, value);
    }
}

fn expectMatricesClose(
    cpu_ptr: *anyopaque,
    metal_ptr: *anyopaque,
    cpu_matrix: *const Matrix,
    metal_matrix: *const Matrix,
    tolerance: f64,
) !void {
    try testing.expectEqual(cpu_matrix.rows, metal_matrix.rows);
    try testing.expectEqual(cpu_matrix.cols, metal_matrix.cols);

    for (0..cpu_matrix.rows) |row| {
        for (0..cpu_matrix.cols) |col| {
            const expected = CPUBackend.getMatrixElement(cpu_ptr, cpu_matrix, row, col);
            const actual = MetalBackend.getMatrixElement(metal_ptr, metal_matrix, row, col);
            try testing.expectApproxEqAbs(expected, actual, tolerance);
        }
    }
}

test "metal backend matches cpu backend for matrix operations" {
    const allocator = testing.allocator;

    const cpu_impl = try CPUBackend.init(allocator);
    defer cpu_impl.deinit();
    const metal_impl = try initMetalBackendForTest(allocator);
    defer metal_impl.deinit();

    const cpu_ptr: *anyopaque = @ptrCast(cpu_impl);
    const metal_ptr: *anyopaque = @ptrCast(metal_impl);

    const a_values = [_]f64{ 1.5, -2.0, 3.25, 0.5, 4.0, -1.25 };
    const b_values = [_]f64{ 2.0, -1.0, 0.25, 3.0, -2.5, 1.75 };
    const c_values = [_]f64{ -0.5, 2.0, 0.25, 3.0, -4.0, 1.25 };

    const cpu_a = try CPUBackend.initMatrix(cpu_ptr, allocator, 2, 3);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_a);
    const metal_a = try MetalBackend.initMatrix(metal_ptr, allocator, 2, 3);
    defer MetalBackend.deinitMatrix(metal_ptr, metal_a);
    fillMatrixPair(cpu_ptr, metal_ptr, cpu_a, metal_a, &a_values);

    const cpu_b = try CPUBackend.initMatrix(cpu_ptr, allocator, 3, 2);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_b);
    const metal_b = try MetalBackend.initMatrix(metal_ptr, allocator, 3, 2);
    defer MetalBackend.deinitMatrix(metal_ptr, metal_b);
    fillMatrixPair(cpu_ptr, metal_ptr, cpu_b, metal_b, &b_values);

    const cpu_c = try CPUBackend.initMatrix(cpu_ptr, allocator, 2, 3);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_c);
    const metal_c = try MetalBackend.initMatrix(metal_ptr, allocator, 2, 3);
    defer MetalBackend.deinitMatrix(metal_ptr, metal_c);
    fillMatrixPair(cpu_ptr, metal_ptr, cpu_c, metal_c, &c_values);

    const tolerance = 1e-4;

    const cpu_dot = try CPUBackend.dotProduct(cpu_ptr, cpu_a, cpu_b, allocator);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_dot);
    const metal_dot = try MetalBackend.dotProduct(metal_ptr, metal_a, metal_b, allocator);
    defer MetalBackend.deinitMatrix(metal_ptr, metal_dot);
    try expectMatricesClose(cpu_ptr, metal_ptr, cpu_dot, metal_dot, tolerance);

    const cpu_add = try CPUBackend.add(cpu_ptr, cpu_a, cpu_c, allocator);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_add);
    const metal_add = try MetalBackend.add(metal_ptr, metal_a, metal_c, allocator);
    defer MetalBackend.deinitMatrix(metal_ptr, metal_add);
    try expectMatricesClose(cpu_ptr, metal_ptr, cpu_add, metal_add, tolerance);

    const cpu_subtract = try CPUBackend.subtract(cpu_ptr, cpu_a, cpu_c, allocator);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_subtract);
    const metal_subtract = try MetalBackend.subtract(metal_ptr, metal_a, metal_c, allocator);
    defer MetalBackend.deinitMatrix(metal_ptr, metal_subtract);
    try expectMatricesClose(cpu_ptr, metal_ptr, cpu_subtract, metal_subtract, tolerance);

    const cpu_multiply = try CPUBackend.elementWiseMultiply(cpu_ptr, cpu_a, cpu_c, allocator);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_multiply);
    const metal_multiply = try MetalBackend.elementWiseMultiply(metal_ptr, metal_a, metal_c, allocator);
    defer MetalBackend.deinitMatrix(metal_ptr, metal_multiply);
    try expectMatricesClose(cpu_ptr, metal_ptr, cpu_multiply, metal_multiply, tolerance);

    const cpu_scale = try CPUBackend.scale(cpu_ptr, cpu_a, -0.75, allocator);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_scale);
    const metal_scale = try MetalBackend.scale(metal_ptr, metal_a, -0.75, allocator);
    defer MetalBackend.deinitMatrix(metal_ptr, metal_scale);
    try expectMatricesClose(cpu_ptr, metal_ptr, cpu_scale, metal_scale, tolerance);

    const cpu_sum_rows = try CPUBackend.sumRows(cpu_ptr, cpu_a, allocator);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_sum_rows);
    const metal_sum_rows = try MetalBackend.sumRows(metal_ptr, metal_a, allocator);
    defer MetalBackend.deinitMatrix(metal_ptr, metal_sum_rows);
    try expectMatricesClose(cpu_ptr, metal_ptr, cpu_sum_rows, metal_sum_rows, tolerance);

    const cpu_transpose = try CPUBackend.transpose(cpu_ptr, cpu_a, allocator);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_transpose);
    const metal_transpose = try MetalBackend.transpose(metal_ptr, metal_a, allocator);
    defer MetalBackend.deinitMatrix(metal_ptr, metal_transpose);
    try expectMatricesClose(cpu_ptr, metal_ptr, cpu_transpose, metal_transpose, tolerance);

    const cpu_batch = try CPUBackend.extractBatch(cpu_ptr, cpu_a, 1, 2, allocator);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_batch);
    const metal_batch = try MetalBackend.extractBatch(metal_ptr, metal_a, 1, 2, allocator);
    defer MetalBackend.deinitMatrix(metal_ptr, metal_batch);
    try expectMatricesClose(cpu_ptr, metal_ptr, cpu_batch, metal_batch, tolerance);

    const cpu_chained = try CPUBackend.scale(cpu_ptr, cpu_add, 0.5, allocator);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_chained);
    const metal_chained = try MetalBackend.scale(metal_ptr, metal_add, 0.5, allocator);
    defer MetalBackend.deinitMatrix(metal_ptr, metal_chained);
    try expectMatricesClose(cpu_ptr, metal_ptr, cpu_chained, metal_chained, tolerance);
}

test "metal backend matches cpu backend for activation operations" {
    const allocator = testing.allocator;

    const cpu_impl = try CPUBackend.init(allocator);
    defer cpu_impl.deinit();
    const metal_impl = try initMetalBackendForTest(allocator);
    defer metal_impl.deinit();

    const cpu_ptr: *anyopaque = @ptrCast(cpu_impl);
    const metal_ptr: *anyopaque = @ptrCast(metal_impl);

    const a_values = [_]f64{ -1.5, -0.25, 0.0, 0.75, 2.0, 4.0 };
    const gate_values = [_]f64{ 2.0, -1.0, 0.5, 3.0, -2.5, 1.25 };

    const cpu_a = try CPUBackend.initMatrix(cpu_ptr, allocator, 2, 3);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_a);
    const metal_a = try MetalBackend.initMatrix(metal_ptr, allocator, 2, 3);
    defer MetalBackend.deinitMatrix(metal_ptr, metal_a);
    fillMatrixPair(cpu_ptr, metal_ptr, cpu_a, metal_a, &a_values);

    const cpu_gate = try CPUBackend.initMatrix(cpu_ptr, allocator, 2, 3);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_gate);
    const metal_gate = try MetalBackend.initMatrix(metal_ptr, allocator, 2, 3);
    defer MetalBackend.deinitMatrix(metal_ptr, metal_gate);
    fillMatrixPair(cpu_ptr, metal_ptr, cpu_gate, metal_gate, &gate_values);

    const tolerance = 1e-4;

    const cpu_sigmoid = try CPUBackend.applyActivation(cpu_ptr, cpu_a, Activation.sigmoid, allocator);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_sigmoid);
    const metal_sigmoid = try MetalBackend.applyActivation(metal_ptr, metal_a, Activation.sigmoid, allocator);
    defer MetalBackend.deinitMatrix(metal_ptr, metal_sigmoid);
    try expectMatricesClose(cpu_ptr, metal_ptr, cpu_sigmoid, metal_sigmoid, tolerance);

    const cpu_relu = try CPUBackend.applyActivation(cpu_ptr, cpu_a, Activation.relu, allocator);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_relu);
    const metal_relu = try MetalBackend.applyActivation(metal_ptr, metal_a, Activation.relu, allocator);
    defer MetalBackend.deinitMatrix(metal_ptr, metal_relu);
    try expectMatricesClose(cpu_ptr, metal_ptr, cpu_relu, metal_relu, tolerance);

    const cpu_tanh = try CPUBackend.applyActivation(cpu_ptr, cpu_a, Activation.tanh, allocator);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_tanh);
    const metal_tanh = try MetalBackend.applyActivation(metal_ptr, metal_a, Activation.tanh, allocator);
    defer MetalBackend.deinitMatrix(metal_ptr, metal_tanh);
    try expectMatricesClose(cpu_ptr, metal_ptr, cpu_tanh, metal_tanh, tolerance);

    const cpu_swish = try CPUBackend.applyActivation(cpu_ptr, cpu_a, Activation.swish, allocator);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_swish);
    const metal_swish = try MetalBackend.applyActivation(metal_ptr, metal_a, Activation.swish, allocator);
    defer MetalBackend.deinitMatrix(metal_ptr, metal_swish);
    try expectMatricesClose(cpu_ptr, metal_ptr, cpu_swish, metal_swish, tolerance);

    const cpu_softmax = try CPUBackend.applySoftmax(cpu_ptr, cpu_a, allocator);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_softmax);
    const metal_softmax = try MetalBackend.applySoftmax(metal_ptr, metal_a, allocator);
    defer MetalBackend.deinitMatrix(metal_ptr, metal_softmax);
    try expectMatricesClose(cpu_ptr, metal_ptr, cpu_softmax, metal_softmax, tolerance);

    const cpu_glu = try CPUBackend.applyGLU(cpu_ptr, cpu_a, cpu_gate, allocator);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_glu);
    const metal_glu = try MetalBackend.applyGLU(metal_ptr, metal_a, metal_gate, allocator);
    defer MetalBackend.deinitMatrix(metal_ptr, metal_glu);
    try expectMatricesClose(cpu_ptr, metal_ptr, cpu_glu, metal_glu, tolerance);

    const cpu_swiglu = try CPUBackend.applySwiGLU(cpu_ptr, cpu_a, cpu_gate, allocator);
    defer CPUBackend.deinitMatrix(cpu_ptr, cpu_swiglu);
    const metal_swiglu = try MetalBackend.applySwiGLU(metal_ptr, metal_a, metal_gate, allocator);
    defer MetalBackend.deinitMatrix(metal_ptr, metal_swiglu);
    try expectMatricesClose(cpu_ptr, metal_ptr, cpu_swiglu, metal_swiglu, tolerance);
}
