const std = @import("std");
const Allocator = std.mem.Allocator;
const backend = @import("backend.zig");

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

const compiled_gpu = @import("compiled_gpu_backend.zig");
const implementation = compiled_gpu.Implementation(struct {
    pub const Driver = ROCm;
    pub const Error = ROCmError;
    pub const enabled = enable_rocm;
    pub const not_enabled = error.ROCmNotEnabled;
    pub const not_supported = error.ROCmNotSupported;
    pub const device_not_found = error.ROCmDeviceNotFound;
    pub const initialization_failed = error.ROCmInitializationFailed;
    pub const backend_type = backend.BackendType.ROCm;
});

pub const ROCmBackend = implementation.GpuBackend;

pub fn createROCmBackend(allocator: Allocator) !*anyopaque {
    return implementation.createBackend(allocator);
}

test "rocm backend matches cpu backend for matrix operations" {
    try implementation.testMatrixOperations();
}

test "rocm backend matches cpu backend for activation operations" {
    try implementation.testActivationOperations();
}
