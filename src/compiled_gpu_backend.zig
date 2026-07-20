const std = @import("std");
const Allocator = std.mem.Allocator;
const backend = @import("backend.zig");
const dimensions = @import("dimensions.zig");
const BackendType = backend.BackendType;
const Matrix = backend.Matrix;
const randomSeed = @import("matrix.zig").randomSeed;
const Activation = @import("activation.zig").Activation;

/// Builds the shared Zig orchestration for a compiled GPU driver.
///
/// Vendor adapters provide only the driver ABI and public error identity. All
/// buffer ownership, batching, fallback math, and operation dispatch lives in
/// this implementation.
pub fn Implementation(comptime Config: type) type {
    return struct {
        const Driver = Config.Driver;

        const GpuMatrix = struct {
            rows: usize,
            cols: usize,
            buffer: ?*Driver.Buffer,
            host_data: []f32,
            allocator: Allocator,
            backend: *Driver.Backend,
            owner: *GpuBackend,
            dirty: bool,
            gpu_dirty: bool,

            pub fn init(allocator: Allocator, rows: usize, cols: usize, owner: *GpuBackend) !*GpuMatrix {
                const element_count = dimensions.elementCount(rows, cols) catch return error.OutOfMemory;
                if (element_count == 0) return error.OutOfMemory;
                _ = dimensions.product(&.{ element_count, @sizeOf(f32) }) catch return error.OutOfMemory;

                const gpu_matrix = try allocator.create(GpuMatrix);
                errdefer allocator.destroy(gpu_matrix);

                const host_data = try allocator.alloc(f32, element_count);
                errdefer allocator.free(host_data);
                @memset(host_data, 0);

                const backend_ref = owner.backend orelse return Config.initialization_failed;
                const buffer = Driver.createBuffer(backend_ref, element_count) orelse return error.BufferCreationFailed;
                errdefer Driver.destroyBuffer(backend_ref, buffer);

                gpu_matrix.* = .{
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
                owner.stats.live_buffers += 1;
                return gpu_matrix;
            }

            pub fn deinit(self: *GpuMatrix) void {
                Driver.destroyBuffer(self.backend, self.buffer);
                self.allocator.free(self.host_data);
                const owner = self.owner;
                self.allocator.destroy(self);
                std.debug.assert(owner.stats.live_buffers > 0);
                owner.stats.live_buffers -= 1;
            }

            pub fn syncToGPU(self: *GpuMatrix) bool {
                if (!self.dirty) return true;

                const buffer = self.buffer orelse return false;
                if (!Driver.bufferUploadF32(self.backend, buffer, self.host_data)) {
                    return false;
                }
                self.owner.stats.host_to_device_transfers += 1;
                self.owner.stats.host_to_device_bytes += self.host_data.len * @sizeOf(f32);

                self.dirty = false;
                self.gpu_dirty = false;
                return true;
            }

            pub fn syncToCPU(self: *GpuMatrix) bool {
                if (!self.gpu_dirty) return true;

                if (!self.owner.synchronizeQueued()) return false;

                const buffer = self.buffer orelse return false;
                if (!Driver.bufferDownloadF32(self.backend, buffer, self.host_data)) {
                    return false;
                }
                self.owner.stats.device_to_host_transfers += 1;
                self.owner.stats.device_to_host_bytes += self.host_data.len * @sizeOf(f32);

                self.dirty = false;
                self.gpu_dirty = false;
                return true;
            }

            pub fn markHostModified(self: *GpuMatrix) void {
                self.dirty = true;
                self.gpu_dirty = false;
            }

            pub fn markGPUModified(self: *GpuMatrix) void {
                self.dirty = false;
                self.gpu_dirty = true;
            }
        };

        pub const GpuBackend = struct {
            allocator: Allocator,
            backend: ?*Driver.Backend,
            device_name: [256]u8,
            stats: backend.RuntimeStats,
            batch_active: bool,
            synchronized_kernel_count: usize,
            deferred_matrices: std.ArrayList(*Matrix),

            pub fn init(allocator: Allocator) (Config.Error || Allocator.Error)!*GpuBackend {
                if (!Config.enabled) {
                    return Config.not_enabled;
                }
                if (@import("builtin").os.tag != .linux) {
                    return Config.not_supported;
                }

                const backend_ref = Driver.createBackend() orelse return Config.initialization_failed;
                const gpu_backend = allocator.create(GpuBackend) catch |err| {
                    Driver.destroyBackend(backend_ref);
                    return err;
                };

                gpu_backend.* = .{
                    .allocator = allocator,
                    .backend = backend_ref,
                    .device_name = [_]u8{0} ** 256,
                    .stats = .{},
                    .batch_active = false,
                    .synchronized_kernel_count = 0,
                    .deferred_matrices = .empty,
                };
                _ = Driver.deviceName(backend_ref, gpu_backend.device_name[0..]);

                return gpu_backend;
            }

            pub fn deinit(self: *GpuBackend) void {
                self.batch_active = false;
                _ = self.synchronizeQueued();
                self.releaseDeferredMatrices();
                self.deferred_matrices.deinit(self.allocator);
                Driver.destroyBackend(self.backend);
                self.allocator.destroy(self);
            }

            pub fn getDeviceName(self: *const GpuBackend) []const u8 {
                return std.mem.sliceTo(&self.device_name, 0);
            }

            pub fn runtimeStats(ptr: *anyopaque) backend.RuntimeStats {
                const self = @as(*GpuBackend, @ptrCast(@alignCast(ptr)));
                return self.stats;
            }

            pub fn resetRuntimeStats(ptr: *anyopaque) void {
                const self = @as(*GpuBackend, @ptrCast(@alignCast(ptr)));
                const live_buffers = self.stats.live_buffers;
                self.stats = .{ .live_buffers = live_buffers };
                self.synchronized_kernel_count = 0;
            }

            pub fn beginBatch(ptr: *anyopaque) !void {
                const self = @as(*GpuBackend, @ptrCast(@alignCast(ptr)));
                if (self.batch_active) return error.BatchAlreadyActive;
                self.batch_active = true;
            }

            pub fn endBatch(ptr: *anyopaque) !void {
                const self = @as(*GpuBackend, @ptrCast(@alignCast(ptr)));
                if (!self.batch_active) return error.NoActiveBatch;
                self.batch_active = false;
                if (!self.synchronizeQueued()) return error.CommandExecutionFailed;
            }

            pub fn synchronize(ptr: *anyopaque) !void {
                const self = @as(*GpuBackend, @ptrCast(@alignCast(ptr)));
                if (!self.synchronizeQueued()) return error.CommandExecutionFailed;
            }

            fn completeSubmittedKernel(self: *GpuBackend) bool {
                self.stats.kernel_launches += 1;
                if (self.batch_active) return true;
                return self.synchronizeQueued();
            }

            fn synchronizeQueued(self: *GpuBackend) bool {
                if (self.synchronized_kernel_count != self.stats.kernel_launches) {
                    const backend_ref = self.backend orelse return false;
                    if (!Driver.synchronize(backend_ref)) return false;
                    self.stats.synchronizations += 1;
                    self.synchronized_kernel_count = self.stats.kernel_launches;
                }
                self.releaseDeferredMatrices();
                return true;
            }

            fn releaseDeferredMatrices(self: *GpuBackend) void {
                for (self.deferred_matrices.items) |matrix| {
                    deinitMatrixNow(matrix);
                }
                self.deferred_matrices.clearRetainingCapacity();
            }

            fn deinitMatrixNow(matrix: *Matrix) void {
                const gpu_matrix = getGpuMatrix(matrix);
                const allocator = gpu_matrix.allocator;
                gpu_matrix.deinit();
                allocator.destroy(matrix);
            }

            fn getGpuMatrix(matrix: *const Matrix) *GpuMatrix {
                return @as(*GpuMatrix, @ptrCast(@alignCast(matrix.impl_data)));
            }

            fn elementCount(matrix: *const Matrix) usize {
                return matrix.rows * matrix.cols;
            }

            fn toU32(value: usize) u32 {
                return std.math.cast(u32, value) orelse @panic("GPU backend dimension exceeds u32 range");
            }

            fn ensureHostData(gpu_matrix: *GpuMatrix) void {
                if (!gpu_matrix.syncToCPU()) {
                    @panic("GPU buffer download failed");
                }
            }

            fn dispatchMatrixMultiply(self: *GpuBackend, a: *const Matrix, b: *const Matrix, result: *Matrix) bool {
                const backend_ref = self.backend orelse return false;
                const a_gpu = getGpuMatrix(a);
                const b_gpu = getGpuMatrix(b);
                const result_gpu = getGpuMatrix(result);

                if (!a_gpu.syncToGPU() or !b_gpu.syncToGPU()) return false;
                const a_buffer = a_gpu.buffer orelse return false;
                const b_buffer = b_gpu.buffer orelse return false;
                const result_buffer = result_gpu.buffer orelse return false;

                const kind = Driver.launchMatrixMultiply(backend_ref, a_buffer, b_buffer, result_buffer, toU32(a.rows), toU32(a.cols), toU32(b.cols));
                if (kind == .failed) return false;

                if (kind == .vendor) self.stats.vendor_gemm_launches += 1;
                if (!self.completeSubmittedKernel()) return false;
                result_gpu.markGPUModified();
                return true;
            }

            fn dispatchBatchedMatrixMultiply(self: *GpuBackend, a: *const Matrix, b: *const Matrix, result: *Matrix, batch: usize, a_rows: usize, a_cols: usize, b_rows: usize, b_cols: usize, transpose_a: bool, transpose_b: bool) bool {
                const backend_ref = self.backend orelse return false;
                const a_gpu = getGpuMatrix(a);
                const b_gpu = getGpuMatrix(b);
                const result_gpu = getGpuMatrix(result);
                if (!a_gpu.syncToGPU() or !b_gpu.syncToGPU()) return false;
                const a_buffer = a_gpu.buffer orelse return false;
                const b_buffer = b_gpu.buffer orelse return false;
                const result_buffer = result_gpu.buffer orelse return false;
                if (!Driver.launchBatchedMatrixMultiply(backend_ref, a_buffer, b_buffer, result_buffer, toU32(batch), toU32(a_rows), toU32(a_cols), toU32(b_rows), toU32(b_cols), transpose_a, transpose_b)) return false;
                if (!self.completeSubmittedKernel()) return false;
                result_gpu.markGPUModified();
                return true;
            }

            fn dispatchPermuteBatchHeads(self: *GpuBackend, input: *const Matrix, result: *Matrix, batch: usize, tokens: usize, heads: usize, width: usize, split: bool) bool {
                const backend_ref = self.backend orelse return false;
                const source = getGpuMatrix(input);
                const destination = getGpuMatrix(result);
                if (!source.syncToGPU()) return false;
                const source_buffer = source.buffer orelse return false;
                const destination_buffer = destination.buffer orelse return false;
                if (!Driver.launchPermuteBatchHeads(backend_ref, source_buffer, destination_buffer, toU32(batch), toU32(tokens), toU32(heads), toU32(width), split)) return false;
                if (!self.completeSubmittedKernel()) return false;
                destination.markGPUModified();
                return true;
            }

            fn dispatchBinaryMatrixKernel(self: *GpuBackend, kernel_id: Driver.KernelId, a: *const Matrix, b: *const Matrix, result: *Matrix) bool {
                const backend_ref = self.backend orelse return false;
                const a_gpu = getGpuMatrix(a);
                const b_gpu = getGpuMatrix(b);
                const result_gpu = getGpuMatrix(result);

                if (!a_gpu.syncToGPU() or !b_gpu.syncToGPU()) return false;
                const a_buffer = a_gpu.buffer orelse return false;
                const b_buffer = b_gpu.buffer orelse return false;
                const result_buffer = result_gpu.buffer orelse return false;

                if (!Driver.launchBinaryKernel(backend_ref, kernel_id, a_buffer, b_buffer, result_buffer, toU32(a.rows), toU32(a.cols))) {
                    return false;
                }

                if (!self.completeSubmittedKernel()) return false;
                result_gpu.markGPUModified();
                return true;
            }

            fn dispatchScale(self: *GpuBackend, matrix: *const Matrix, scalar: f64, result: *Matrix) bool {
                const backend_ref = self.backend orelse return false;
                const matrix_gpu = getGpuMatrix(matrix);
                const result_gpu = getGpuMatrix(result);

                if (!matrix_gpu.syncToGPU()) return false;
                const matrix_buffer = matrix_gpu.buffer orelse return false;
                const result_buffer = result_gpu.buffer orelse return false;

                if (!Driver.launchScale(backend_ref, matrix_buffer, result_buffer, @floatCast(scalar), toU32(matrix.rows), toU32(matrix.cols))) {
                    return false;
                }

                if (!self.completeSubmittedKernel()) return false;
                result_gpu.markGPUModified();
                return true;
            }

            fn dispatchOptimizerUpdate(self: *GpuBackend, parameter: *Matrix, gradient: *const Matrix, first_moment: *Matrix, second_moment: *Matrix, total_squares: *const Matrix, config: backend.OptimizerUpdateConfig) bool {
                const backend_ref = self.backend orelse return false;
                const parameter_data = getGpuMatrix(parameter);
                const gradient_data = getGpuMatrix(gradient);
                const first_moment_data = getGpuMatrix(first_moment);
                const second_moment_data = getGpuMatrix(second_moment);
                const total_squares_data = getGpuMatrix(total_squares);
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
                if (!Driver.launchOptimizerUpdate(backend_ref, parameter_buffer, gradient_buffer, first_moment_buffer, second_moment_buffer, total_squares_buffer, config, toU32(parameter.rows * parameter.cols))) return false;
                if (!self.completeSubmittedKernel()) return false;
                parameter_data.markGPUModified();
                first_moment_data.markGPUModified();
                second_moment_data.markGPUModified();
                return true;
            }

            fn dispatchTranspose(self: *GpuBackend, matrix: *const Matrix, result: *Matrix) bool {
                const backend_ref = self.backend orelse return false;
                const matrix_gpu = getGpuMatrix(matrix);
                const result_gpu = getGpuMatrix(result);

                if (!matrix_gpu.syncToGPU()) return false;
                const matrix_buffer = matrix_gpu.buffer orelse return false;
                const result_buffer = result_gpu.buffer orelse return false;

                if (!Driver.launchTranspose(backend_ref, matrix_buffer, result_buffer, toU32(matrix.rows), toU32(matrix.cols))) {
                    return false;
                }

                if (!self.completeSubmittedKernel()) return false;
                result_gpu.markGPUModified();
                return true;
            }

            fn dispatchSumRows(self: *GpuBackend, matrix: *const Matrix, result: *Matrix) bool {
                const backend_ref = self.backend orelse return false;
                const matrix_gpu = getGpuMatrix(matrix);
                const result_gpu = getGpuMatrix(result);

                if (!matrix_gpu.syncToGPU()) return false;
                const matrix_buffer = matrix_gpu.buffer orelse return false;
                const result_buffer = result_gpu.buffer orelse return false;

                if (!Driver.launchSumRows(backend_ref, matrix_buffer, result_buffer, toU32(matrix.rows), toU32(matrix.cols))) {
                    return false;
                }

                if (!self.completeSubmittedKernel()) return false;
                result_gpu.markGPUModified();
                return true;
            }

            fn dispatchExtractBatch(self: *GpuBackend, matrix: *const Matrix, start: usize, result: *Matrix) bool {
                const backend_ref = self.backend orelse return false;
                const matrix_gpu = getGpuMatrix(matrix);
                const result_gpu = getGpuMatrix(result);

                if (!matrix_gpu.syncToGPU()) return false;
                const matrix_buffer = matrix_gpu.buffer orelse return false;
                const result_buffer = result_gpu.buffer orelse return false;

                if (!Driver.launchExtractBatch(backend_ref, matrix_buffer, result_buffer, toU32(start), toU32(result.rows), toU32(result.cols))) {
                    return false;
                }

                if (!self.completeSubmittedKernel()) return false;
                result_gpu.markGPUModified();
                return true;
            }

            fn dispatchUnaryKernel(self: *GpuBackend, kernel_id: Driver.KernelId, matrix: *const Matrix, result: *Matrix) bool {
                const backend_ref = self.backend orelse return false;
                const matrix_gpu = getGpuMatrix(matrix);
                const result_gpu = getGpuMatrix(result);

                if (!matrix_gpu.syncToGPU()) return false;
                const matrix_buffer = matrix_gpu.buffer orelse return false;
                const result_buffer = result_gpu.buffer orelse return false;

                if (!Driver.launchUnaryKernel(backend_ref, kernel_id, matrix_buffer, result_buffer, toU32(elementCount(matrix)))) {
                    return false;
                }

                if (!self.completeSubmittedKernel()) return false;
                result_gpu.markGPUModified();
                return true;
            }

            fn dispatchSoftmax(self: *GpuBackend, matrix: *const Matrix, result: *Matrix) bool {
                const backend_ref = self.backend orelse return false;
                const matrix_gpu = getGpuMatrix(matrix);
                const result_gpu = getGpuMatrix(result);

                if (!matrix_gpu.syncToGPU()) return false;
                const matrix_buffer = matrix_gpu.buffer orelse return false;
                const result_buffer = result_gpu.buffer orelse return false;

                if (!Driver.launchSoftmaxRows(backend_ref, matrix_buffer, result_buffer, toU32(matrix.rows), toU32(matrix.cols))) {
                    return false;
                }

                if (!self.completeSubmittedKernel()) return false;
                result_gpu.markGPUModified();
                return true;
            }

            fn dispatchGatedKernel(self: *GpuBackend, kernel_id: Driver.KernelId, linear_part: *const Matrix, gating_part: *const Matrix, result: *Matrix) bool {
                const backend_ref = self.backend orelse return false;
                const linear_gpu = getGpuMatrix(linear_part);
                const gating_gpu = getGpuMatrix(gating_part);
                const result_gpu = getGpuMatrix(result);

                if (!linear_gpu.syncToGPU() or !gating_gpu.syncToGPU()) return false;
                const linear_buffer = linear_gpu.buffer orelse return false;
                const gating_buffer = gating_gpu.buffer orelse return false;
                const result_buffer = result_gpu.buffer orelse return false;

                if (!Driver.launchGatedKernel(backend_ref, kernel_id, linear_buffer, gating_buffer, result_buffer, toU32(elementCount(linear_part)))) {
                    return false;
                }

                if (!self.completeSubmittedKernel()) return false;
                result_gpu.markGPUModified();
                return true;
            }

            fn dispatchLayerNorm(self: *GpuBackend, matrix: *const Matrix, gamma: *const Matrix, beta: *const Matrix, epsilon: f64, result: *Matrix) bool {
                const backend_ref = self.backend orelse return false;
                const matrix_gpu = getGpuMatrix(matrix);
                const gamma_gpu = getGpuMatrix(gamma);
                const beta_gpu = getGpuMatrix(beta);
                const result_gpu = getGpuMatrix(result);

                if (!matrix_gpu.syncToGPU() or !gamma_gpu.syncToGPU() or !beta_gpu.syncToGPU()) return false;
                const matrix_buffer = matrix_gpu.buffer orelse return false;
                const gamma_buffer = gamma_gpu.buffer orelse return false;
                const beta_buffer = beta_gpu.buffer orelse return false;
                const result_buffer = result_gpu.buffer orelse return false;

                if (!Driver.launchLayerNorm(backend_ref, matrix_buffer, gamma_buffer, beta_buffer, result_buffer, toU32(matrix.rows), toU32(matrix.cols), @floatCast(epsilon))) {
                    return false;
                }

                if (!self.completeSubmittedKernel()) return false;
                result_gpu.markGPUModified();
                return true;
            }

            fn dispatchLayerNormBackward(self: *GpuBackend, input: *const Matrix, gamma: *const Matrix, output_gradient: *const Matrix, epsilon: f64, gradients: backend.LayerNormGradients) bool {
                const backend_ref = self.backend orelse return false;
                const input_data = getGpuMatrix(input);
                const gamma_data = getGpuMatrix(gamma);
                const output_gradient_data = getGpuMatrix(output_gradient);
                const input_gradient_data = getGpuMatrix(gradients.input);
                const gamma_gradient_data = getGpuMatrix(gradients.gamma);
                const beta_gradient_data = getGpuMatrix(gradients.beta);
                if (!input_data.syncToGPU() or !gamma_data.syncToGPU() or !output_gradient_data.syncToGPU()) return false;
                const input_buffer = input_data.buffer orelse return false;
                const gamma_buffer = gamma_data.buffer orelse return false;
                const output_gradient_buffer = output_gradient_data.buffer orelse return false;
                const input_gradient_buffer = input_gradient_data.buffer orelse return false;
                const gamma_gradient_buffer = gamma_gradient_data.buffer orelse return false;
                const beta_gradient_buffer = beta_gradient_data.buffer orelse return false;
                if (!Driver.launchLayerNormBackward(backend_ref, input_buffer, gamma_buffer, output_gradient_buffer, input_gradient_buffer, gamma_gradient_buffer, beta_gradient_buffer, toU32(input.rows), toU32(input.cols), @floatCast(epsilon))) {
                    return false;
                }
                if (!self.completeSubmittedKernel()) return false;
                input_gradient_data.markGPUModified();
                gamma_gradient_data.markGPUModified();
                beta_gradient_data.markGPUModified();
                return true;
            }

            fn dispatchCausalSelfAttention(self: *GpuBackend, query: *const Matrix, key: *const Matrix, value: *const Matrix, heads: usize, result: *Matrix) bool {
                const backend_ref = self.backend orelse return false;
                const query_gpu = getGpuMatrix(query);
                const key_gpu = getGpuMatrix(key);
                const value_gpu = getGpuMatrix(value);
                const result_gpu = getGpuMatrix(result);
                if (!query_gpu.syncToGPU() or !key_gpu.syncToGPU() or !value_gpu.syncToGPU()) return false;
                const query_buffer = query_gpu.buffer orelse return false;
                const key_buffer = key_gpu.buffer orelse return false;
                const value_buffer = value_gpu.buffer orelse return false;
                const result_buffer = result_gpu.buffer orelse return false;

                if (!Driver.launchCausalSelfAttention(backend_ref, query_buffer, key_buffer, value_buffer, result_buffer, toU32(query.rows), toU32(query.cols), toU32(heads))) {
                    return false;
                }
                if (!self.completeSubmittedKernel()) return false;
                result_gpu.markGPUModified();
                return true;
            }

            fn dispatchCausalSelfAttentionBackward(
                self: *GpuBackend,
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
                const query_data = getGpuMatrix(query);
                const key_data = getGpuMatrix(key);
                const value_data = getGpuMatrix(value);
                const output_gradient_data = getGpuMatrix(output_gradient);
                const probabilities_data = getGpuMatrix(probabilities);
                const score_gradients_data = getGpuMatrix(score_gradients);
                const query_gradient_data = getGpuMatrix(gradients.query);
                const key_gradient_data = getGpuMatrix(gradients.key);
                const value_gradient_data = getGpuMatrix(gradients.value);
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
                if (!Driver.launchCausalAttentionProbabilitiesBackward(backend_ref, query_buffer, key_buffer, value_buffer, output_gradient_buffer, probabilities_buffer, score_gradients_buffer, toU32(query.rows), toU32(query.cols), toU32(heads))) {
                    return false;
                }
                if (!self.completeSubmittedKernel()) return false;
                probabilities_data.markGPUModified();
                score_gradients_data.markGPUModified();
                if (!Driver.launchCausalAttentionInputGradients(backend_ref, query_buffer, key_buffer, output_gradient_buffer, probabilities_buffer, score_gradients_buffer, query_gradient_buffer, key_gradient_buffer, value_gradient_buffer, toU32(query.rows), toU32(query.cols), toU32(heads))) {
                    return false;
                }
                if (!self.completeSubmittedKernel()) return false;
                query_gradient_data.markGPUModified();
                key_gradient_data.markGPUModified();
                value_gradient_data.markGPUModified();
                return true;
            }

            fn dispatchEmbeddingLookup(self: *GpuBackend, table: *const Matrix, indices: *const Matrix, result: *Matrix) bool {
                const backend_ref = self.backend orelse return false;
                const table_data = getGpuMatrix(table);
                const indices_data = getGpuMatrix(indices);
                const result_data = getGpuMatrix(result);
                if (!table_data.syncToGPU() or !indices_data.syncToGPU()) return false;
                const table_buffer = table_data.buffer orelse return false;
                const indices_buffer = indices_data.buffer orelse return false;
                const result_buffer = result_data.buffer orelse return false;
                if (!Driver.launchEmbeddingLookup(backend_ref, table_buffer, indices_buffer, result_buffer, toU32(table.rows), toU32(indices.rows), toU32(table.cols))) return false;
                if (!self.completeSubmittedKernel()) return false;
                result_data.markGPUModified();
                return true;
            }

            fn dispatchEmbeddingGradient(self: *GpuBackend, indices: *const Matrix, output_gradient: *const Matrix, result: *Matrix) bool {
                const backend_ref = self.backend orelse return false;
                const indices_data = getGpuMatrix(indices);
                const output_gradient_data = getGpuMatrix(output_gradient);
                const result_data = getGpuMatrix(result);
                if (!indices_data.syncToGPU() or !output_gradient_data.syncToGPU()) return false;
                const indices_buffer = indices_data.buffer orelse return false;
                const output_gradient_buffer = output_gradient_data.buffer orelse return false;
                const result_buffer = result_data.buffer orelse return false;
                if (!Driver.launchEmbeddingGradient(backend_ref, indices_buffer, output_gradient_buffer, result_buffer, toU32(result.rows), toU32(indices.rows), toU32(result.cols))) return false;
                if (!self.completeSubmittedKernel()) return false;
                result_data.markGPUModified();
                return true;
            }

            fn dispatchCachedSelfAttention(self: *GpuBackend, query: *const Matrix, key: *const Matrix, value: *const Matrix, key_cache: *Matrix, value_cache: *Matrix, position: usize, heads: usize, result: *Matrix) bool {
                const backend_ref = self.backend orelse return false;
                const query_data = getGpuMatrix(query);
                const key_data = getGpuMatrix(key);
                const value_data = getGpuMatrix(value);
                const key_cache_data = getGpuMatrix(key_cache);
                const value_cache_data = getGpuMatrix(value_cache);
                const result_data = getGpuMatrix(result);
                if (!query_data.syncToGPU() or !key_data.syncToGPU() or !value_data.syncToGPU() or !key_cache_data.syncToGPU() or !value_cache_data.syncToGPU()) return false;
                const query_buffer = query_data.buffer orelse return false;
                const key_buffer = key_data.buffer orelse return false;
                const value_buffer = value_data.buffer orelse return false;
                const key_cache_buffer = key_cache_data.buffer orelse return false;
                const value_cache_buffer = value_cache_data.buffer orelse return false;
                const result_buffer = result_data.buffer orelse return false;
                if (!Driver.launchCachedSelfAttention(backend_ref, query_buffer, key_buffer, value_buffer, key_cache_buffer, value_cache_buffer, result_buffer, toU32(position), toU32(query.cols), toU32(heads))) return false;
                if (!self.completeSubmittedKernel()) return false;
                key_cache_data.markGPUModified();
                value_cache_data.markGPUModified();
                result_data.markGPUModified();
                return true;
            }

            pub fn initMatrix(ptr: *anyopaque, allocator: Allocator, rows: usize, cols: usize) error{OutOfMemory}!*Matrix {
                const self = @as(*GpuBackend, @ptrCast(@alignCast(ptr)));
                const gpu_matrix = GpuMatrix.init(allocator, rows, cols, self) catch |err| {
                    std.log.warn("GPU matrix initialization failed: {s}", .{@errorName(err)});
                    return error.OutOfMemory;
                };
                errdefer gpu_matrix.deinit();

                const matrix = try allocator.create(Matrix);
                matrix.* = .{
                    .rows = rows,
                    .cols = cols,
                    .backend = undefined,
                    .impl_data = gpu_matrix,
                };
                return matrix;
            }

            pub fn deinitMatrix(ptr: *anyopaque, matrix: *Matrix) void {
                const self = @as(*GpuBackend, @ptrCast(@alignCast(ptr)));
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
                const source_gpu = getGpuMatrix(source);
                const result = try initMatrix(ptr, allocator, source.rows, source.cols);
                const result_gpu = getGpuMatrix(result);

                ensureHostData(source_gpu);
                @memcpy(result_gpu.host_data, source_gpu.host_data);
                result_gpu.markHostModified();
                return result;
            }

            pub fn getMatrixElement(_: *anyopaque, matrix: *const Matrix, row: usize, col: usize) f64 {
                if (row >= matrix.rows or col >= matrix.cols) {
                    @panic("Index out of bounds");
                }

                const gpu_matrix = getGpuMatrix(matrix);
                ensureHostData(gpu_matrix);
                return @floatCast(gpu_matrix.host_data[row * matrix.cols + col]);
            }

            pub fn writeMatrixF32(_: *anyopaque, matrix: *Matrix, values: []const f32) bool {
                const gpu_matrix = getGpuMatrix(matrix);
                if (values.len != gpu_matrix.host_data.len) return false;
                @memcpy(gpu_matrix.host_data, values);
                gpu_matrix.markHostModified();
                return true;
            }

            pub fn readMatrixF32(_: *anyopaque, matrix: *const Matrix, values: []f32) bool {
                const gpu_matrix = getGpuMatrix(matrix);
                if (values.len != gpu_matrix.host_data.len) return false;
                ensureHostData(gpu_matrix);
                @memcpy(values, gpu_matrix.host_data);
                return true;
            }

            pub fn setMatrixElement(_: *anyopaque, matrix: *Matrix, row: usize, col: usize, value: f64) void {
                if (row >= matrix.rows or col >= matrix.cols) {
                    @panic("Index out of bounds");
                }

                const gpu_matrix = getGpuMatrix(matrix);
                ensureHostData(gpu_matrix);
                gpu_matrix.host_data[row * matrix.cols + col] = @floatCast(value);
                gpu_matrix.markHostModified();
            }

            pub fn fillMatrix(_: *anyopaque, matrix: *Matrix, value: f64) void {
                const gpu_matrix = getGpuMatrix(matrix);
                for (gpu_matrix.host_data) |*element| {
                    element.* = @floatCast(value);
                }
                gpu_matrix.markHostModified();
            }

            pub fn dotProduct(ptr: *anyopaque, a: *const Matrix, b: *const Matrix, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!*Matrix {
                if (a.cols != b.rows) {
                    return error.DimensionMismatch;
                }

                const self = @as(*GpuBackend, @ptrCast(@alignCast(ptr)));
                const result = try initMatrix(ptr, allocator, a.rows, b.cols);
                const a_gpu = getGpuMatrix(a);
                const b_gpu = getGpuMatrix(b);
                const result_gpu = getGpuMatrix(result);

                if (self.dispatchMatrixMultiply(a, b, result)) {
                    return result;
                }

                ensureHostData(a_gpu);
                ensureHostData(b_gpu);
                for (0..a.rows) |i| {
                    for (0..b.cols) |j| {
                        var sum: f32 = 0.0;
                        for (0..a.cols) |k| {
                            sum += a_gpu.host_data[i * a.cols + k] * b_gpu.host_data[k * b.cols + j];
                        }
                        result_gpu.host_data[i * b.cols + j] = sum;
                    }
                }
                result_gpu.markHostModified();
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

                const self = @as(*GpuBackend, @ptrCast(@alignCast(ptr)));
                const result_rows = dimensions.product(&.{ batch, output_rows }) catch return error.OutOfMemory;
                const result = try initMatrix(ptr, allocator, result_rows, output_cols);
                const a_gpu = getGpuMatrix(a);
                const b_gpu = getGpuMatrix(b);
                const result_gpu = getGpuMatrix(result);
                if (self.dispatchBatchedMatrixMultiply(a, b, result, batch, a_rows, a_cols, b_rows, b_cols, transpose_a, transpose_b)) return result;

                ensureHostData(a_gpu);
                ensureHostData(b_gpu);
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
                                sum += a_gpu.host_data[a_index] * b_gpu.host_data[b_index];
                            }
                            result_gpu.host_data[result_offset + row * output_cols + col] = sum;
                        }
                    }
                }
                result_gpu.markHostModified();
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
                const self = @as(*GpuBackend, @ptrCast(@alignCast(ptr)));
                const result = try initMatrix(ptr, allocator, result_rows, result_cols);
                const source = getGpuMatrix(input);
                const destination = getGpuMatrix(result);
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

                const self = @as(*GpuBackend, @ptrCast(@alignCast(ptr)));
                const result = try initMatrix(ptr, allocator, a.rows, a.cols);
                const a_gpu = getGpuMatrix(a);
                const b_gpu = getGpuMatrix(b);
                const result_gpu = getGpuMatrix(result);

                if (self.dispatchBinaryMatrixKernel(.matrix_add, a, b, result)) {
                    return result;
                }

                ensureHostData(a_gpu);
                ensureHostData(b_gpu);
                for (0..a.rows * a.cols) |i| {
                    result_gpu.host_data[i] = a_gpu.host_data[i] + b_gpu.host_data[i];
                }
                result_gpu.markHostModified();
                return result;
            }

            pub fn addRowBias(ptr: *anyopaque, matrix: *const Matrix, bias: *const Matrix, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!*Matrix {
                if (bias.rows != 1 or bias.cols != matrix.cols) return error.DimensionMismatch;

                const self = @as(*GpuBackend, @ptrCast(@alignCast(ptr)));
                const result = try initMatrix(ptr, allocator, matrix.rows, matrix.cols);
                const input_data = getGpuMatrix(matrix);
                const bias_data = getGpuMatrix(bias);
                const result_data = getGpuMatrix(result);

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

                const self = @as(*GpuBackend, @ptrCast(@alignCast(ptr)));
                const result = try initMatrix(ptr, allocator, a.rows, a.cols);
                const a_gpu = getGpuMatrix(a);
                const b_gpu = getGpuMatrix(b);
                const result_gpu = getGpuMatrix(result);

                if (self.dispatchBinaryMatrixKernel(.matrix_subtract, a, b, result)) {
                    return result;
                }

                ensureHostData(a_gpu);
                ensureHostData(b_gpu);
                for (0..a.rows * a.cols) |i| {
                    result_gpu.host_data[i] = a_gpu.host_data[i] - b_gpu.host_data[i];
                }
                result_gpu.markHostModified();
                return result;
            }

            pub fn elementWiseMultiply(ptr: *anyopaque, a: *const Matrix, b: *const Matrix, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!*Matrix {
                if (a.rows != b.rows or a.cols != b.cols) {
                    return error.DimensionMismatch;
                }

                const self = @as(*GpuBackend, @ptrCast(@alignCast(ptr)));
                const result = try initMatrix(ptr, allocator, a.rows, a.cols);
                const a_gpu = getGpuMatrix(a);
                const b_gpu = getGpuMatrix(b);
                const result_gpu = getGpuMatrix(result);

                if (self.dispatchBinaryMatrixKernel(.matrix_element_wise_multiply, a, b, result)) {
                    return result;
                }

                ensureHostData(a_gpu);
                ensureHostData(b_gpu);
                for (0..a.rows * a.cols) |i| {
                    result_gpu.host_data[i] = a_gpu.host_data[i] * b_gpu.host_data[i];
                }
                result_gpu.markHostModified();
                return result;
            }

            pub fn scale(ptr: *anyopaque, matrix: *const Matrix, scalar: f64, allocator: Allocator) error{OutOfMemory}!*Matrix {
                const self = @as(*GpuBackend, @ptrCast(@alignCast(ptr)));
                const result = try initMatrix(ptr, allocator, matrix.rows, matrix.cols);
                const matrix_gpu = getGpuMatrix(matrix);
                const result_gpu = getGpuMatrix(result);

                if (self.dispatchScale(matrix, scalar, result)) {
                    return result;
                }

                ensureHostData(matrix_gpu);
                for (0..matrix.rows * matrix.cols) |i| {
                    result_gpu.host_data[i] = matrix_gpu.host_data[i] * @as(f32, @floatCast(scalar));
                }
                result_gpu.markHostModified();
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
                const self = @as(*GpuBackend, @ptrCast(@alignCast(ptr)));
                if (self.dispatchOptimizerUpdate(parameter, gradient, first_moment, second_moment, total_squares, config)) return;

                const parameter_data = getGpuMatrix(parameter);
                const gradient_data = getGpuMatrix(gradient);
                const first_moment_data = getGpuMatrix(first_moment);
                const second_moment_data = getGpuMatrix(second_moment);
                const total_squares_data = getGpuMatrix(total_squares);
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
                const self = @as(*GpuBackend, @ptrCast(@alignCast(ptr)));
                const result = try initMatrix(ptr, allocator, 1, matrix.cols);
                const matrix_gpu = getGpuMatrix(matrix);
                const result_gpu = getGpuMatrix(result);

                if (self.dispatchSumRows(matrix, result)) {
                    return result;
                }

                ensureHostData(matrix_gpu);
                for (0..matrix.cols) |j| {
                    var sum: f32 = 0.0;
                    for (0..matrix.rows) |i| {
                        sum += matrix_gpu.host_data[i * matrix.cols + j];
                    }
                    result_gpu.host_data[j] = sum;
                }
                result_gpu.markHostModified();
                return result;
            }

            pub fn transpose(ptr: *anyopaque, matrix: *const Matrix, allocator: Allocator) error{OutOfMemory}!*Matrix {
                const self = @as(*GpuBackend, @ptrCast(@alignCast(ptr)));
                const result = try initMatrix(ptr, allocator, matrix.cols, matrix.rows);
                const matrix_gpu = getGpuMatrix(matrix);
                const result_gpu = getGpuMatrix(result);

                if (self.dispatchTranspose(matrix, result)) {
                    return result;
                }

                ensureHostData(matrix_gpu);
                for (0..matrix.rows) |i| {
                    for (0..matrix.cols) |j| {
                        result_gpu.host_data[j * matrix.rows + i] = matrix_gpu.host_data[i * matrix.cols + j];
                    }
                }
                result_gpu.markHostModified();
                return result;
            }

            pub fn extractBatch(ptr: *anyopaque, matrix: *const Matrix, start: usize, end: usize, allocator: Allocator) error{ OutOfMemory, InvalidBatchIndices }!*Matrix {
                if (start >= end or end > matrix.rows) {
                    return error.InvalidBatchIndices;
                }

                const self = @as(*GpuBackend, @ptrCast(@alignCast(ptr)));
                const batch_rows = end - start;
                const result = try initMatrix(ptr, allocator, batch_rows, matrix.cols);
                const matrix_gpu = getGpuMatrix(matrix);
                const result_gpu = getGpuMatrix(result);

                if (self.dispatchExtractBatch(matrix, start, result)) {
                    return result;
                }

                ensureHostData(matrix_gpu);
                const src_offset = start * matrix.cols;
                @memcpy(result_gpu.host_data, matrix_gpu.host_data[src_offset..][0 .. batch_rows * matrix.cols]);
                result_gpu.markHostModified();
                return result;
            }

            pub fn randomize(_: *anyopaque, matrix: *Matrix, min: f64, max: f64) void {
                const gpu_matrix = getGpuMatrix(matrix);
                var rng = std.Random.DefaultPrng.init(randomSeed());
                for (gpu_matrix.host_data) |*element| {
                    const random_float = rng.random().float(f64);
                    element.* = @floatCast(min + random_float * (max - min));
                }
                gpu_matrix.markHostModified();
            }

            pub fn applyActivation(ptr: *anyopaque, matrix: *const Matrix, activation: *const fn (f64) f64, allocator: Allocator) error{OutOfMemory}!*Matrix {
                const self = @as(*GpuBackend, @ptrCast(@alignCast(ptr)));
                const result = try initMatrix(ptr, allocator, matrix.rows, matrix.cols);
                const matrix_gpu = getGpuMatrix(matrix);
                const result_gpu = getGpuMatrix(result);

                const kernel_id: ?Driver.KernelId = if (activation == Activation.linear)
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

                ensureHostData(matrix_gpu);
                for (0..matrix.rows * matrix.cols) |i| {
                    result_gpu.host_data[i] = @floatCast(activation(@floatCast(matrix_gpu.host_data[i])));
                }
                result_gpu.markHostModified();
                return result;
            }

            pub fn applySoftmax(ptr: *anyopaque, matrix: *const Matrix, allocator: Allocator) error{OutOfMemory}!*Matrix {
                const self = @as(*GpuBackend, @ptrCast(@alignCast(ptr)));
                const result = try initMatrix(ptr, allocator, matrix.rows, matrix.cols);
                const matrix_gpu = getGpuMatrix(matrix);
                const result_gpu = getGpuMatrix(result);

                if (self.dispatchSoftmax(matrix, result)) {
                    return result;
                }

                ensureHostData(matrix_gpu);
                for (0..matrix.rows) |i| {
                    var max_val: f32 = -std.math.inf(f32);
                    for (0..matrix.cols) |j| {
                        max_val = @max(max_val, matrix_gpu.host_data[i * matrix.cols + j]);
                    }

                    var sum: f32 = 0.0;
                    for (0..matrix.cols) |j| {
                        const idx = i * matrix.cols + j;
                        const exp_val = std.math.exp(matrix_gpu.host_data[idx] - max_val);
                        result_gpu.host_data[idx] = exp_val;
                        sum += exp_val;
                    }

                    for (0..matrix.cols) |j| {
                        const idx = i * matrix.cols + j;
                        result_gpu.host_data[idx] /= sum;
                    }
                }
                result_gpu.markHostModified();
                return result;
            }

            pub fn layerNorm(ptr: *anyopaque, matrix: *const Matrix, gamma: *const Matrix, beta: *const Matrix, epsilon: f64, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!*Matrix {
                if (gamma.rows != 1 or beta.rows != 1 or gamma.cols != matrix.cols or beta.cols != matrix.cols) {
                    return error.DimensionMismatch;
                }

                const self = @as(*GpuBackend, @ptrCast(@alignCast(ptr)));
                const result = try initMatrix(ptr, allocator, matrix.rows, matrix.cols);
                const input_data = getGpuMatrix(matrix);
                const gamma_data = getGpuMatrix(gamma);
                const beta_data = getGpuMatrix(beta);
                const result_data = getGpuMatrix(result);

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
                const self = @as(*GpuBackend, @ptrCast(@alignCast(ptr)));
                if (self.dispatchLayerNormBackward(input, gamma, output_gradient, epsilon, gradients)) return gradients;

                const input_data = getGpuMatrix(input);
                const gamma_data = getGpuMatrix(gamma);
                const output_gradient_data = getGpuMatrix(output_gradient);
                const input_gradient_data = getGpuMatrix(input_gradient);
                const gamma_gradient_data = getGpuMatrix(gamma_gradient);
                const beta_gradient_data = getGpuMatrix(beta_gradient);
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

                const self = @as(*GpuBackend, @ptrCast(@alignCast(ptr)));
                const result = try initMatrix(ptr, allocator, query.rows, query.cols);
                errdefer deinitMatrix(ptr, result);
                const query_data = getGpuMatrix(query);
                const key_data = getGpuMatrix(key);
                const value_data = getGpuMatrix(value);
                const result_data = getGpuMatrix(result);
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
                const self = @as(*GpuBackend, @ptrCast(@alignCast(ptr)));
                if (self.dispatchCausalSelfAttentionBackward(query, key, value, output_gradient, heads, probabilities, score_gradients, gradients)) return gradients;

                const query_data = getGpuMatrix(query);
                const key_data = getGpuMatrix(key);
                const value_data = getGpuMatrix(value);
                const output_gradient_data = getGpuMatrix(output_gradient);
                const query_gradient_data = getGpuMatrix(query_gradient);
                const key_gradient_data = getGpuMatrix(key_gradient);
                const value_gradient_data = getGpuMatrix(value_gradient);
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
                const self = @as(*GpuBackend, @ptrCast(@alignCast(ptr)));
                if (self.dispatchEmbeddingLookup(table, indices, result)) return result;
                const table_data = getGpuMatrix(table);
                const indices_data = getGpuMatrix(indices);
                const result_data = getGpuMatrix(result);
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
                const self = @as(*GpuBackend, @ptrCast(@alignCast(ptr)));
                if (self.dispatchEmbeddingGradient(indices, output_gradient, result)) return result;
                const indices_data = getGpuMatrix(indices);
                const output_gradient_data = getGpuMatrix(output_gradient);
                const result_data = getGpuMatrix(result);
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
                const self = @as(*GpuBackend, @ptrCast(@alignCast(ptr)));
                if (self.dispatchCachedSelfAttention(query, key, value, key_cache, value_cache, position, heads, result)) return result;
                const query_data = getGpuMatrix(query);
                const key_data = getGpuMatrix(key);
                const value_data = getGpuMatrix(value);
                const key_cache_data = getGpuMatrix(key_cache);
                const value_cache_data = getGpuMatrix(value_cache);
                const result_data = getGpuMatrix(result);
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

                const self = @as(*GpuBackend, @ptrCast(@alignCast(ptr)));
                const result = try initMatrix(ptr, allocator, linear_part.rows, linear_part.cols);
                const linear_gpu = getGpuMatrix(linear_part);
                const gating_gpu = getGpuMatrix(gating_part);
                const result_gpu = getGpuMatrix(result);

                if (self.dispatchGatedKernel(.apply_glu, linear_part, gating_part, result)) {
                    return result;
                }

                ensureHostData(linear_gpu);
                ensureHostData(gating_gpu);
                for (0..linear_part.rows * linear_part.cols) |i| {
                    const sigmoid = 1.0 / (1.0 + std.math.exp(-gating_gpu.host_data[i]));
                    result_gpu.host_data[i] = linear_gpu.host_data[i] * sigmoid;
                }
                result_gpu.markHostModified();
                return result;
            }

            pub fn applySwiGLU(ptr: *anyopaque, linear_part: *const Matrix, gating_part: *const Matrix, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!*Matrix {
                if (linear_part.rows != gating_part.rows or linear_part.cols != gating_part.cols) {
                    return error.DimensionMismatch;
                }

                const self = @as(*GpuBackend, @ptrCast(@alignCast(ptr)));
                const result = try initMatrix(ptr, allocator, linear_part.rows, linear_part.cols);
                const linear_gpu = getGpuMatrix(linear_part);
                const gating_gpu = getGpuMatrix(gating_part);
                const result_gpu = getGpuMatrix(result);

                if (self.dispatchGatedKernel(.apply_swiglu, linear_part, gating_part, result)) {
                    return result;
                }

                ensureHostData(linear_gpu);
                ensureHostData(gating_gpu);
                for (0..linear_part.rows * linear_part.cols) |i| {
                    const gate = gating_gpu.host_data[i];
                    const sigmoid = 1.0 / (1.0 + std.math.exp(-gate));
                    result_gpu.host_data[i] = linear_gpu.host_data[i] * gate * sigmoid;
                }
                result_gpu.markHostModified();
                return result;
            }

            pub fn getBackendType(_: *anyopaque) BackendType {
                return Config.backend_type;
            }
        };

        pub fn createBackend(allocator: Allocator) !*anyopaque {
            const gpu_backend = try GpuBackend.init(allocator);
            errdefer gpu_backend.deinit();
            return @ptrCast(gpu_backend);
        }
        const CPUBackend = @import("cpu_backend.zig").CPUBackend;
        const testing = std.testing;

        fn initGpuBackendForTest(allocator: Allocator) !*GpuBackend {
            if (!Config.enabled or @import("builtin").os.tag != .linux) {
                return error.SkipZigTest;
            }

            return GpuBackend.init(allocator) catch |err| {
                if (err == Config.not_enabled or err == Config.not_supported or
                    err == Config.device_not_found or err == Config.initialization_failed)
                {
                    return error.SkipZigTest;
                }
                return err;
            };
        }

        fn fillMatrixPair(
            cpu_ptr: *anyopaque,
            gpu_ptr: *anyopaque,
            cpu_matrix: *Matrix,
            gpu_matrix: *Matrix,
            values: []const f64,
        ) void {
            for (values, 0..) |value, index| {
                const row = index / cpu_matrix.cols;
                const col = index % cpu_matrix.cols;
                CPUBackend.setMatrixElement(cpu_ptr, cpu_matrix, row, col, value);
                GpuBackend.setMatrixElement(gpu_ptr, gpu_matrix, row, col, value);
            }
        }

        fn expectMatricesClose(
            cpu_ptr: *anyopaque,
            gpu_ptr: *anyopaque,
            cpu_matrix: *const Matrix,
            gpu_matrix: *const Matrix,
            tolerance: f64,
        ) !void {
            try testing.expectEqual(cpu_matrix.rows, gpu_matrix.rows);
            try testing.expectEqual(cpu_matrix.cols, gpu_matrix.cols);

            for (0..cpu_matrix.rows) |row| {
                for (0..cpu_matrix.cols) |col| {
                    const expected = CPUBackend.getMatrixElement(cpu_ptr, cpu_matrix, row, col);
                    const actual = GpuBackend.getMatrixElement(gpu_ptr, gpu_matrix, row, col);
                    try testing.expectApproxEqAbs(expected, actual, tolerance);
                }
            }
        }

        pub fn testMatrixOperations() !void {
            const allocator = testing.allocator;

            const cpu_impl = try CPUBackend.init(allocator);
            defer cpu_impl.deinit();
            const gpu_impl = try initGpuBackendForTest(allocator);
            defer gpu_impl.deinit();

            const cpu_ptr: *anyopaque = @ptrCast(cpu_impl);
            const gpu_ptr: *anyopaque = @ptrCast(gpu_impl);

            const a_values = [_]f64{ 1.5, -2.0, 3.25, 0.5, 4.0, -1.25 };
            const b_values = [_]f64{ 2.0, -1.0, 0.25, 3.0, -2.5, 1.75 };
            const c_values = [_]f64{ -0.5, 2.0, 0.25, 3.0, -4.0, 1.25 };

            const cpu_a = try CPUBackend.initMatrix(cpu_ptr, allocator, 2, 3);
            defer CPUBackend.deinitMatrix(cpu_ptr, cpu_a);
            const gpu_a = try GpuBackend.initMatrix(gpu_ptr, allocator, 2, 3);
            defer GpuBackend.deinitMatrix(gpu_ptr, gpu_a);
            fillMatrixPair(cpu_ptr, gpu_ptr, cpu_a, gpu_a, &a_values);

            const cpu_b = try CPUBackend.initMatrix(cpu_ptr, allocator, 3, 2);
            defer CPUBackend.deinitMatrix(cpu_ptr, cpu_b);
            const gpu_b = try GpuBackend.initMatrix(gpu_ptr, allocator, 3, 2);
            defer GpuBackend.deinitMatrix(gpu_ptr, gpu_b);
            fillMatrixPair(cpu_ptr, gpu_ptr, cpu_b, gpu_b, &b_values);

            const cpu_c = try CPUBackend.initMatrix(cpu_ptr, allocator, 2, 3);
            defer CPUBackend.deinitMatrix(cpu_ptr, cpu_c);
            const gpu_c = try GpuBackend.initMatrix(gpu_ptr, allocator, 2, 3);
            defer GpuBackend.deinitMatrix(gpu_ptr, gpu_c);
            fillMatrixPair(cpu_ptr, gpu_ptr, cpu_c, gpu_c, &c_values);

            const tolerance = 1e-4;

            const cpu_dot = try CPUBackend.dotProduct(cpu_ptr, cpu_a, cpu_b, allocator);
            defer CPUBackend.deinitMatrix(cpu_ptr, cpu_dot);
            const gpu_dot = try GpuBackend.dotProduct(gpu_ptr, gpu_a, gpu_b, allocator);
            defer GpuBackend.deinitMatrix(gpu_ptr, gpu_dot);
            try expectMatricesClose(cpu_ptr, gpu_ptr, cpu_dot, gpu_dot, tolerance);

            const cpu_add = try CPUBackend.add(cpu_ptr, cpu_a, cpu_c, allocator);
            defer CPUBackend.deinitMatrix(cpu_ptr, cpu_add);
            const gpu_add = try GpuBackend.add(gpu_ptr, gpu_a, gpu_c, allocator);
            defer GpuBackend.deinitMatrix(gpu_ptr, gpu_add);
            try expectMatricesClose(cpu_ptr, gpu_ptr, cpu_add, gpu_add, tolerance);

            const cpu_subtract = try CPUBackend.subtract(cpu_ptr, cpu_a, cpu_c, allocator);
            defer CPUBackend.deinitMatrix(cpu_ptr, cpu_subtract);
            const gpu_subtract = try GpuBackend.subtract(gpu_ptr, gpu_a, gpu_c, allocator);
            defer GpuBackend.deinitMatrix(gpu_ptr, gpu_subtract);
            try expectMatricesClose(cpu_ptr, gpu_ptr, cpu_subtract, gpu_subtract, tolerance);

            const cpu_multiply = try CPUBackend.elementWiseMultiply(cpu_ptr, cpu_a, cpu_c, allocator);
            defer CPUBackend.deinitMatrix(cpu_ptr, cpu_multiply);
            const gpu_multiply = try GpuBackend.elementWiseMultiply(gpu_ptr, gpu_a, gpu_c, allocator);
            defer GpuBackend.deinitMatrix(gpu_ptr, gpu_multiply);
            try expectMatricesClose(cpu_ptr, gpu_ptr, cpu_multiply, gpu_multiply, tolerance);

            const cpu_scale = try CPUBackend.scale(cpu_ptr, cpu_a, -0.75, allocator);
            defer CPUBackend.deinitMatrix(cpu_ptr, cpu_scale);
            const gpu_scale = try GpuBackend.scale(gpu_ptr, gpu_a, -0.75, allocator);
            defer GpuBackend.deinitMatrix(gpu_ptr, gpu_scale);
            try expectMatricesClose(cpu_ptr, gpu_ptr, cpu_scale, gpu_scale, tolerance);

            const cpu_sum_rows = try CPUBackend.sumRows(cpu_ptr, cpu_a, allocator);
            defer CPUBackend.deinitMatrix(cpu_ptr, cpu_sum_rows);
            const gpu_sum_rows = try GpuBackend.sumRows(gpu_ptr, gpu_a, allocator);
            defer GpuBackend.deinitMatrix(gpu_ptr, gpu_sum_rows);
            try expectMatricesClose(cpu_ptr, gpu_ptr, cpu_sum_rows, gpu_sum_rows, tolerance);

            const cpu_transpose = try CPUBackend.transpose(cpu_ptr, cpu_a, allocator);
            defer CPUBackend.deinitMatrix(cpu_ptr, cpu_transpose);
            const gpu_transpose = try GpuBackend.transpose(gpu_ptr, gpu_a, allocator);
            defer GpuBackend.deinitMatrix(gpu_ptr, gpu_transpose);
            try expectMatricesClose(cpu_ptr, gpu_ptr, cpu_transpose, gpu_transpose, tolerance);

            const cpu_batch = try CPUBackend.extractBatch(cpu_ptr, cpu_a, 1, 2, allocator);
            defer CPUBackend.deinitMatrix(cpu_ptr, cpu_batch);
            const gpu_batch = try GpuBackend.extractBatch(gpu_ptr, gpu_a, 1, 2, allocator);
            defer GpuBackend.deinitMatrix(gpu_ptr, gpu_batch);
            try expectMatricesClose(cpu_ptr, gpu_ptr, cpu_batch, gpu_batch, tolerance);
        }

        pub fn testActivationOperations() !void {
            const allocator = testing.allocator;

            const cpu_impl = try CPUBackend.init(allocator);
            defer cpu_impl.deinit();
            const gpu_impl = try initGpuBackendForTest(allocator);
            defer gpu_impl.deinit();

            const cpu_ptr: *anyopaque = @ptrCast(cpu_impl);
            const gpu_ptr: *anyopaque = @ptrCast(gpu_impl);

            const a_values = [_]f64{ -1.5, -0.25, 0.0, 0.75, 2.0, 4.0 };
            const gate_values = [_]f64{ 2.0, -1.0, 0.5, 3.0, -2.5, 1.25 };

            const cpu_a = try CPUBackend.initMatrix(cpu_ptr, allocator, 2, 3);
            defer CPUBackend.deinitMatrix(cpu_ptr, cpu_a);
            const gpu_a = try GpuBackend.initMatrix(gpu_ptr, allocator, 2, 3);
            defer GpuBackend.deinitMatrix(gpu_ptr, gpu_a);
            fillMatrixPair(cpu_ptr, gpu_ptr, cpu_a, gpu_a, &a_values);

            const cpu_gate = try CPUBackend.initMatrix(cpu_ptr, allocator, 2, 3);
            defer CPUBackend.deinitMatrix(cpu_ptr, cpu_gate);
            const gpu_gate = try GpuBackend.initMatrix(gpu_ptr, allocator, 2, 3);
            defer GpuBackend.deinitMatrix(gpu_ptr, gpu_gate);
            fillMatrixPair(cpu_ptr, gpu_ptr, cpu_gate, gpu_gate, &gate_values);

            const tolerance = 1e-4;

            const cpu_sigmoid = try CPUBackend.applyActivation(cpu_ptr, cpu_a, Activation.sigmoid, allocator);
            defer CPUBackend.deinitMatrix(cpu_ptr, cpu_sigmoid);
            const gpu_sigmoid = try GpuBackend.applyActivation(gpu_ptr, gpu_a, Activation.sigmoid, allocator);
            defer GpuBackend.deinitMatrix(gpu_ptr, gpu_sigmoid);
            try expectMatricesClose(cpu_ptr, gpu_ptr, cpu_sigmoid, gpu_sigmoid, tolerance);

            const cpu_relu = try CPUBackend.applyActivation(cpu_ptr, cpu_a, Activation.relu, allocator);
            defer CPUBackend.deinitMatrix(cpu_ptr, cpu_relu);
            const gpu_relu = try GpuBackend.applyActivation(gpu_ptr, gpu_a, Activation.relu, allocator);
            defer GpuBackend.deinitMatrix(gpu_ptr, gpu_relu);
            try expectMatricesClose(cpu_ptr, gpu_ptr, cpu_relu, gpu_relu, tolerance);

            const cpu_tanh = try CPUBackend.applyActivation(cpu_ptr, cpu_a, Activation.tanh, allocator);
            defer CPUBackend.deinitMatrix(cpu_ptr, cpu_tanh);
            const gpu_tanh = try GpuBackend.applyActivation(gpu_ptr, gpu_a, Activation.tanh, allocator);
            defer GpuBackend.deinitMatrix(gpu_ptr, gpu_tanh);
            try expectMatricesClose(cpu_ptr, gpu_ptr, cpu_tanh, gpu_tanh, tolerance);

            const cpu_swish = try CPUBackend.applyActivation(cpu_ptr, cpu_a, Activation.swish, allocator);
            defer CPUBackend.deinitMatrix(cpu_ptr, cpu_swish);
            const gpu_swish = try GpuBackend.applyActivation(gpu_ptr, gpu_a, Activation.swish, allocator);
            defer GpuBackend.deinitMatrix(gpu_ptr, gpu_swish);
            try expectMatricesClose(cpu_ptr, gpu_ptr, cpu_swish, gpu_swish, tolerance);

            const cpu_softmax = try CPUBackend.applySoftmax(cpu_ptr, cpu_a, allocator);
            defer CPUBackend.deinitMatrix(cpu_ptr, cpu_softmax);
            const gpu_softmax = try GpuBackend.applySoftmax(gpu_ptr, gpu_a, allocator);
            defer GpuBackend.deinitMatrix(gpu_ptr, gpu_softmax);
            try expectMatricesClose(cpu_ptr, gpu_ptr, cpu_softmax, gpu_softmax, tolerance);

            const cpu_glu = try CPUBackend.applyGLU(cpu_ptr, cpu_a, cpu_gate, allocator);
            defer CPUBackend.deinitMatrix(cpu_ptr, cpu_glu);
            const gpu_glu = try GpuBackend.applyGLU(gpu_ptr, gpu_a, gpu_gate, allocator);
            defer GpuBackend.deinitMatrix(gpu_ptr, gpu_glu);
            try expectMatricesClose(cpu_ptr, gpu_ptr, cpu_glu, gpu_glu, tolerance);

            const cpu_swiglu = try CPUBackend.applySwiGLU(cpu_ptr, cpu_a, cpu_gate, allocator);
            defer CPUBackend.deinitMatrix(cpu_ptr, cpu_swiglu);
            const gpu_swiglu = try GpuBackend.applySwiGLU(gpu_ptr, gpu_a, gpu_gate, allocator);
            defer GpuBackend.deinitMatrix(gpu_ptr, gpu_swiglu);
            try expectMatricesClose(cpu_ptr, gpu_ptr, cpu_swiglu, gpu_swiglu, tolerance);
        }
    };
}
