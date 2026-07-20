const std = @import("std");
const Allocator = std.mem.Allocator;
const backend = @import("backend.zig");
const BackendType = backend.BackendType;
const Matrix = backend.Matrix;
const randomSeed = @import("matrix.zig").randomSeed;
const dimensions = @import("dimensions.zig");

const vector_width = 8;
pub const max_output_tiles = 64;
const max_workspace_buffers = 256;

const PackedMatrix = struct {
    data: []f32,
    stride: usize,
};

/// CPU-specific implementation of a matrix
const CPUMatrix = struct {
    data: []f32,
    packed_weight: ?PackedMatrix = null,
    allocator: Allocator,
    owner: *CPUBackend,
};

/// CPU backend implementation
pub const CPUBackend = struct {
    allocator: Allocator,
    stats: backend.RuntimeStats,
    output_tiles: usize,
    workspace_buffers: [max_workspace_buffers][]f32,
    workspace_buffer_count: usize,
    storage_allocations: usize,
    workspace_reuses: usize,

    /// Initialize a new CPU backend
    pub fn init(allocator: Allocator) !*CPUBackend {
        const cpu_backend = try allocator.create(CPUBackend);
        cpu_backend.* = CPUBackend{
            .allocator = allocator,
            .stats = .{},
            .output_tiles = 1,
            .workspace_buffers = undefined,
            .workspace_buffer_count = 0,
            .storage_allocations = 0,
            .workspace_reuses = 0,
        };
        return cpu_backend;
    }

    /// Free all resources used by the CPU backend
    pub fn deinit(self: *CPUBackend) void {
        std.debug.assert(self.stats.live_buffers == 0);
        for (self.workspace_buffers[0..self.workspace_buffer_count]) |buffer| {
            self.allocator.free(buffer);
        }
        self.allocator.destroy(self);
    }

    pub fn initMatrix(ptr: *anyopaque, allocator: Allocator, rows: usize, cols: usize) error{OutOfMemory}!*Matrix {
        const self = @as(*CPUBackend, @ptrCast(@alignCast(ptr)));
        // Allocate the matrix wrapper
        const matrix = try allocator.create(Matrix);
        errdefer allocator.destroy(matrix);

        // Allocate the CPU-specific data
        const cpu_data = try allocator.create(CPUMatrix);
        errdefer allocator.destroy(cpu_data);

        // Allocate the data array
        const element_count = dimensions.elementCount(rows, cols) catch
            return error.OutOfMemory;
        const data = self.takeWorkspaceBuffer(allocator, element_count) orelse blk: {
            self.storage_allocations += 1;
            break :blk try allocator.alloc(f32, element_count);
        };
        errdefer allocator.free(data);

        // Initialize with zeros
        @memset(data, 0);

        // Set up the CPU matrix data
        cpu_data.* = .{
            .data = data,
            .allocator = allocator,
            .owner = self,
        };

        // Set up the matrix
        matrix.* = .{
            .rows = rows,
            .cols = cols,
            .backend = undefined, // Will be set by the caller
            .impl_data = cpu_data,
        };
        self.stats.buffer_allocations += 1;
        self.stats.live_buffers += 1;

        return matrix;
    }

    pub fn runtimeStats(ptr: *anyopaque) backend.RuntimeStats {
        const self = @as(*CPUBackend, @ptrCast(@alignCast(ptr)));
        return self.stats;
    }

    pub fn resetRuntimeStats(ptr: *anyopaque) void {
        const self = @as(*CPUBackend, @ptrCast(@alignCast(ptr)));
        const live_buffers = self.stats.live_buffers;
        self.stats = .{ .live_buffers = live_buffers };
    }

    pub fn beginBatch(_: *anyopaque) !void {}

    pub fn endBatch(_: *anyopaque) !void {}

    pub fn synchronize(_: *anyopaque) !void {}

    /// Configures independent output-column tiles for CPU matrix products.
    /// One tile is deterministic and avoids thread startup overhead for small
    /// educational models; callers can opt into more tiles for wider layers.
    pub fn configureOutputTiles(ptr: *anyopaque, count: usize) !void {
        if (count == 0 or count > max_output_tiles) return error.InvalidOutputTileCount;
        const self = @as(*CPUBackend, @ptrCast(@alignCast(ptr)));
        self.output_tiles = count;
    }

    /// Stores an immutable, vector-width-padded copy of a matrix used as the
    /// right-hand side of inference matmuls. Mutations discard the copy so the
    /// readable row-major matrix always remains the source of truth.
    pub fn prepareInferenceWeight(_: *anyopaque, matrix: *Matrix) error{OutOfMemory}!void {
        const cpu_data = @as(*CPUMatrix, @ptrCast(@alignCast(matrix.impl_data)));
        if (cpu_data.packed_weight != null) return;
        const stride = std.mem.alignForward(usize, matrix.cols, vector_width);
        const element_count = dimensions.elementCount(matrix.rows, stride) catch return error.OutOfMemory;
        const data = try cpu_data.allocator.alloc(f32, element_count);
        errdefer cpu_data.allocator.free(data);
        @memset(data, 0);
        for (0..matrix.rows) |row| {
            @memcpy(
                data[row * stride ..][0..matrix.cols],
                cpu_data.data[row * matrix.cols ..][0..matrix.cols],
            );
        }
        cpu_data.packed_weight = .{ .data = data, .stride = stride };
    }

    pub fn deinitMatrix(_: *anyopaque, matrix: *Matrix) void {
        // First, cast to get the CPU-specific data
        const cpu_data = @as(*CPUMatrix, @ptrCast(@alignCast(matrix.impl_data)));
        const allocator = cpu_data.allocator;
        const owner = cpu_data.owner;

        invalidatePacked(cpu_data);

        // Free the data array
        owner.releaseWorkspaceBuffer(allocator, cpu_data.data);

        // Free the CPU matrix data
        allocator.destroy(cpu_data);

        // Free the matrix wrapper
        allocator.destroy(matrix);
        std.debug.assert(owner.stats.live_buffers > 0);
        owner.stats.live_buffers -= 1;
    }

    fn takeWorkspaceBuffer(self: *CPUBackend, allocator: Allocator, element_count: usize) ?[]f32 {
        if (!sameAllocator(allocator, self.allocator)) return null;
        for (self.workspace_buffers[0..self.workspace_buffer_count], 0..) |buffer, index| {
            if (buffer.len != element_count) continue;
            self.workspace_buffer_count -= 1;
            self.workspace_buffers[index] = self.workspace_buffers[self.workspace_buffer_count];
            self.workspace_reuses += 1;
            return buffer;
        }
        return null;
    }

    fn releaseWorkspaceBuffer(self: *CPUBackend, allocator: Allocator, buffer: []f32) void {
        if (buffer.len > 0 and sameAllocator(allocator, self.allocator) and
            self.workspace_buffer_count < self.workspace_buffers.len)
        {
            self.workspace_buffers[self.workspace_buffer_count] = buffer;
            self.workspace_buffer_count += 1;
        } else {
            allocator.free(buffer);
        }
    }

    pub fn getMatrixElement(_: *anyopaque, matrix: *const Matrix, row: usize, col: usize) f64 {
        if (row >= matrix.rows or col >= matrix.cols) {
            @panic("Index out of bounds");
        }

        const cpu_data = @as(*const CPUMatrix, @ptrCast(@alignCast(matrix.impl_data)));
        return @floatCast(cpu_data.data[row * matrix.cols + col]);
    }

    pub fn writeMatrixF32(_: *anyopaque, matrix: *Matrix, values: []const f32) bool {
        const cpu_data = @as(*CPUMatrix, @ptrCast(@alignCast(matrix.impl_data)));
        if (values.len != cpu_data.data.len) return false;
        invalidatePacked(cpu_data);
        @memcpy(cpu_data.data, values);
        return true;
    }

    pub fn readMatrixF32(_: *anyopaque, matrix: *const Matrix, values: []f32) bool {
        const cpu_data = @as(*const CPUMatrix, @ptrCast(@alignCast(matrix.impl_data)));
        if (values.len != cpu_data.data.len) return false;
        @memcpy(values, cpu_data.data);
        return true;
    }

    pub fn setMatrixElement(_: *anyopaque, matrix: *Matrix, row: usize, col: usize, value: f64) void {
        if (row >= matrix.rows or col >= matrix.cols) {
            @panic("Index out of bounds");
        }

        const cpu_data = @as(*CPUMatrix, @ptrCast(@alignCast(matrix.impl_data)));
        invalidatePacked(cpu_data);
        cpu_data.data[row * matrix.cols + col] = @floatCast(value);
    }

    pub fn fillMatrix(_: *anyopaque, matrix: *Matrix, value: f64) void {
        const cpu_data = @as(*CPUMatrix, @ptrCast(@alignCast(matrix.impl_data)));
        invalidatePacked(cpu_data);
        @memset(cpu_data.data, @as(f32, @floatCast(value)));
    }

    pub fn copyMatrix(ptr: *anyopaque, source: *const Matrix, allocator: Allocator) error{OutOfMemory}!*Matrix {
        _ = @as(*CPUBackend, @ptrCast(@alignCast(ptr)));

        const result = try initMatrix(ptr, allocator, source.rows, source.cols);
        errdefer deinitMatrix(undefined, result);

        const source_cpu_data = @as(*const CPUMatrix, @ptrCast(@alignCast(source.impl_data)));
        const result_cpu_data = @as(*CPUMatrix, @ptrCast(@alignCast(result.impl_data)));

        // Copy all data
        @memcpy(result_cpu_data.data, source_cpu_data.data);

        return result;
    }

    pub fn dotProduct(ptr: *anyopaque, a: *const Matrix, b: *const Matrix, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!*Matrix {
        if (a.cols != b.rows) {
            return error.DimensionMismatch;
        }

        const self = @as(*CPUBackend, @ptrCast(@alignCast(ptr)));

        const result = try initMatrix(ptr, allocator, a.rows, b.cols);
        errdefer deinitMatrix(undefined, result);

        const a_cpu = @as(*const CPUMatrix, @ptrCast(@alignCast(a.impl_data)));
        const b_cpu = @as(*const CPUMatrix, @ptrCast(@alignCast(b.impl_data)));
        const result_cpu = @as(*CPUMatrix, @ptrCast(@alignCast(result.impl_data)));

        runMatmulTiles(
            self.output_tiles,
            a_cpu.data,
            a.rows,
            a.cols,
            weightValues(b_cpu),
            weightStride(b_cpu, b.cols),
            b.cols,
            null,
            false,
            result_cpu.data,
        );

        return result;
    }

    pub fn batchedDotProduct(
        ptr: *anyopaque,
        a: *const Matrix,
        b: *const Matrix,
        allocator: Allocator,
        batch: usize,
        a_rows: usize,
        a_cols: usize,
        b_rows: usize,
        b_cols: usize,
        transpose_a: bool,
        transpose_b: bool,
    ) error{ OutOfMemory, DimensionMismatch }!*Matrix {
        const output_rows = if (transpose_a) a_cols else a_rows;
        const inner_a = if (transpose_a) a_rows else a_cols;
        const inner_b = if (transpose_b) b_cols else b_rows;
        const output_cols = if (transpose_b) b_rows else b_cols;
        const a_elements = dimensions.elementCount(a.rows, a.cols) catch
            return error.DimensionMismatch;
        const b_elements = dimensions.elementCount(b.rows, b.cols) catch
            return error.DimensionMismatch;
        if (batch == 0 or inner_a != inner_b or
            !dimensions.matches(a_elements, &.{ batch, a_rows, a_cols }) or
            !dimensions.matches(b_elements, &.{ batch, b_rows, b_cols }))
        {
            return error.DimensionMismatch;
        }

        const result_rows = dimensions.product(&.{ batch, output_rows }) catch
            return error.DimensionMismatch;
        const result = try initMatrix(ptr, allocator, result_rows, output_cols);
        errdefer deinitMatrix(undefined, result);
        const a_cpu = @as(*const CPUMatrix, @ptrCast(@alignCast(a.impl_data)));
        const b_cpu = @as(*const CPUMatrix, @ptrCast(@alignCast(b.impl_data)));
        const result_cpu = @as(*CPUMatrix, @ptrCast(@alignCast(result.impl_data)));

        if (!transpose_a and !transpose_b) {
            for (0..batch) |batch_index| {
                const a_offset = batch_index * a_rows * a_cols;
                const b_offset = batch_index * b_rows * b_cols;
                const result_offset = batch_index * output_rows * output_cols;
                for (0..output_rows) |row| {
                    const output = result_cpu.data[result_offset + row * output_cols ..][0..output_cols];
                    for (0..inner_a) |inner| {
                        const left = a_cpu.data[a_offset + row * a_cols + inner];
                        const right = b_cpu.data[b_offset + inner * b_cols ..][0..b_cols];
                        for (output, right) |*value, factor| {
                            value.* += left * factor;
                        }
                    }
                }
            }
            return result;
        }

        for (0..batch) |batch_index| {
            const a_offset = batch_index * a_rows * a_cols;
            const b_offset = batch_index * b_rows * b_cols;
            const result_offset = batch_index * output_rows * output_cols;
            for (0..output_rows) |row| {
                for (0..output_cols) |col| {
                    var sum: f32 = 0;
                    for (0..inner_a) |inner| {
                        const a_index = a_offset + if (transpose_a)
                            inner * a_cols + row
                        else
                            row * a_cols + inner;
                        const b_index = b_offset + if (transpose_b)
                            col * b_cols + inner
                        else
                            inner * b_cols + col;
                        sum += a_cpu.data[a_index] * b_cpu.data[b_index];
                    }
                    result_cpu.data[result_offset + row * output_cols + col] = sum;
                }
            }
        }
        return result;
    }

    pub fn permuteBatchHeads(ptr: *anyopaque, input: *const Matrix, allocator: Allocator, batch: usize, tokens: usize, heads: usize, width: usize, split: bool) error{ OutOfMemory, DimensionMismatch }!*Matrix {
        const input_elements = dimensions.elementCount(input.rows, input.cols) catch
            return error.DimensionMismatch;
        if (batch == 0 or tokens == 0 or heads == 0 or width == 0 or
            !dimensions.matches(input_elements, &.{ batch, tokens, heads, width }))
        {
            return error.DimensionMismatch;
        }
        const result_rows = dimensions.product(
            if (split) &.{ batch, heads, tokens } else &.{ batch, tokens },
        ) catch return error.DimensionMismatch;
        const result_cols = dimensions.product(
            if (split) &.{width} else &.{ heads, width },
        ) catch return error.DimensionMismatch;
        const result = try initMatrix(
            ptr,
            allocator,
            result_rows,
            result_cols,
        );
        errdefer deinitMatrix(undefined, result);
        const source = @as(*const CPUMatrix, @ptrCast(@alignCast(input.impl_data)));
        const destination = @as(*CPUMatrix, @ptrCast(@alignCast(result.impl_data)));
        for (0..batch) |batch_index| {
            for (0..heads) |head| {
                for (0..tokens) |token| {
                    for (0..width) |channel| {
                        const token_major = (((batch_index * tokens + token) * heads + head) * width + channel);
                        const head_major = (((batch_index * heads + head) * tokens + token) * width + channel);
                        if (split) {
                            destination.data[head_major] = source.data[token_major];
                        } else {
                            destination.data[token_major] = source.data[head_major];
                        }
                    }
                }
            }
        }
        return result;
    }

    pub fn add(ptr: *anyopaque, a: *const Matrix, b: *const Matrix, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!*Matrix {
        if (a.rows != b.rows or a.cols != b.cols) {
            return error.DimensionMismatch;
        }

        _ = @as(*CPUBackend, @ptrCast(@alignCast(ptr)));

        const result = try initMatrix(ptr, allocator, a.rows, a.cols);
        errdefer deinitMatrix(undefined, result);

        const a_cpu = @as(*const CPUMatrix, @ptrCast(@alignCast(a.impl_data)));
        const b_cpu = @as(*const CPUMatrix, @ptrCast(@alignCast(b.impl_data)));
        const result_cpu = @as(*CPUMatrix, @ptrCast(@alignCast(result.impl_data)));

        for (0..result_cpu.data.len) |i| {
            result_cpu.data[i] = a_cpu.data[i] + b_cpu.data[i];
        }

        return result;
    }

    pub fn addRowBias(ptr: *anyopaque, matrix: *const Matrix, bias: *const Matrix, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!*Matrix {
        if (bias.rows != 1 or bias.cols != matrix.cols) return error.DimensionMismatch;

        const result = try initMatrix(ptr, allocator, matrix.rows, matrix.cols);
        errdefer deinitMatrix(undefined, result);
        const input_data = @as(*const CPUMatrix, @ptrCast(@alignCast(matrix.impl_data)));
        const bias_data = @as(*const CPUMatrix, @ptrCast(@alignCast(bias.impl_data)));
        const result_data = @as(*CPUMatrix, @ptrCast(@alignCast(result.impl_data)));
        for (0..matrix.rows) |row| {
            for (0..matrix.cols) |col| {
                const index = row * matrix.cols + col;
                result_data.data[index] = input_data.data[index] + bias_data.data[col];
            }
        }
        return result;
    }

    pub fn linearBiasGelu(
        ptr: *anyopaque,
        input: *const Matrix,
        weights: *const Matrix,
        bias: *const Matrix,
        allocator: Allocator,
    ) error{ OutOfMemory, DimensionMismatch }!*Matrix {
        if (input.cols != weights.rows or bias.rows != 1 or bias.cols != weights.cols) {
            return error.DimensionMismatch;
        }

        const result = try initMatrix(ptr, allocator, input.rows, weights.cols);
        errdefer deinitMatrix(undefined, result);
        const input_data = @as(*const CPUMatrix, @ptrCast(@alignCast(input.impl_data)));
        const weight_data = @as(*const CPUMatrix, @ptrCast(@alignCast(weights.impl_data)));
        const bias_data = @as(*const CPUMatrix, @ptrCast(@alignCast(bias.impl_data)));
        const result_data = @as(*CPUMatrix, @ptrCast(@alignCast(result.impl_data)));

        const self = @as(*CPUBackend, @ptrCast(@alignCast(ptr)));
        runMatmulTiles(
            self.output_tiles,
            input_data.data,
            input.rows,
            input.cols,
            weightValues(weight_data),
            weightStride(weight_data, weights.cols),
            weights.cols,
            bias_data.data,
            true,
            result_data.data,
        );
        return result;
    }

    pub fn subtract(ptr: *anyopaque, a: *const Matrix, b: *const Matrix, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!*Matrix {
        if (a.rows != b.rows or a.cols != b.cols) {
            return error.DimensionMismatch;
        }

        _ = @as(*CPUBackend, @ptrCast(@alignCast(ptr)));

        const result = try initMatrix(ptr, allocator, a.rows, a.cols);
        errdefer deinitMatrix(undefined, result);

        const a_cpu = @as(*const CPUMatrix, @ptrCast(@alignCast(a.impl_data)));
        const b_cpu = @as(*const CPUMatrix, @ptrCast(@alignCast(b.impl_data)));
        const result_cpu = @as(*CPUMatrix, @ptrCast(@alignCast(result.impl_data)));

        for (0..result_cpu.data.len) |i| {
            result_cpu.data[i] = a_cpu.data[i] - b_cpu.data[i];
        }

        return result;
    }

    pub fn elementWiseMultiply(ptr: *anyopaque, a: *const Matrix, b: *const Matrix, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!*Matrix {
        if (a.rows != b.rows or a.cols != b.cols) {
            return error.DimensionMismatch;
        }

        _ = @as(*CPUBackend, @ptrCast(@alignCast(ptr)));

        const result = try initMatrix(ptr, allocator, a.rows, a.cols);
        errdefer deinitMatrix(undefined, result);

        const a_cpu = @as(*const CPUMatrix, @ptrCast(@alignCast(a.impl_data)));
        const b_cpu = @as(*const CPUMatrix, @ptrCast(@alignCast(b.impl_data)));
        const result_cpu = @as(*CPUMatrix, @ptrCast(@alignCast(result.impl_data)));

        for (0..result_cpu.data.len) |i| {
            result_cpu.data[i] = a_cpu.data[i] * b_cpu.data[i];
        }

        return result;
    }

    pub fn scale(ptr: *anyopaque, matrix: *const Matrix, scalar: f64, allocator: Allocator) error{OutOfMemory}!*Matrix {
        _ = @as(*CPUBackend, @ptrCast(@alignCast(ptr)));

        const result = try initMatrix(ptr, allocator, matrix.rows, matrix.cols);
        errdefer deinitMatrix(undefined, result);

        const matrix_cpu = @as(*const CPUMatrix, @ptrCast(@alignCast(matrix.impl_data)));
        const result_cpu = @as(*CPUMatrix, @ptrCast(@alignCast(result.impl_data)));

        for (0..result_cpu.data.len) |i| {
            result_cpu.data[i] = matrix_cpu.data[i] * @as(f32, @floatCast(scalar));
        }

        return result;
    }

    pub fn optimizerUpdate(
        _: *anyopaque,
        parameter: *Matrix,
        gradient: *const Matrix,
        first_moment: *Matrix,
        second_moment: *Matrix,
        total_squares: *const Matrix,
        config: backend.OptimizerUpdateConfig,
    ) error{DimensionMismatch}!void {
        if (parameter.rows != gradient.rows or parameter.cols != gradient.cols or
            parameter.rows != first_moment.rows or parameter.cols != first_moment.cols or
            parameter.rows != second_moment.rows or parameter.cols != second_moment.cols or
            total_squares.rows != 1 or total_squares.cols != 1)
        {
            return error.DimensionMismatch;
        }

        const parameter_data = @as(*CPUMatrix, @ptrCast(@alignCast(parameter.impl_data)));
        const gradient_data = @as(*const CPUMatrix, @ptrCast(@alignCast(gradient.impl_data)));
        const first_moment_data = @as(*CPUMatrix, @ptrCast(@alignCast(first_moment.impl_data)));
        const second_moment_data = @as(*CPUMatrix, @ptrCast(@alignCast(second_moment.impl_data)));
        const total_squares_data = @as(*const CPUMatrix, @ptrCast(@alignCast(total_squares.impl_data)));
        invalidatePacked(parameter_data);
        backend.applyOptimizerUpdate(
            parameter_data.data,
            gradient_data.data,
            first_moment_data.data,
            second_moment_data.data,
            total_squares_data.data[0],
            config,
        );
    }

    pub fn sumRows(ptr: *anyopaque, matrix: *const Matrix, allocator: Allocator) error{OutOfMemory}!*Matrix {
        _ = @as(*CPUBackend, @ptrCast(@alignCast(ptr)));

        const result = try initMatrix(ptr, allocator, 1, matrix.cols);
        errdefer deinitMatrix(undefined, result);

        const matrix_cpu = @as(*const CPUMatrix, @ptrCast(@alignCast(matrix.impl_data)));

        for (0..matrix.cols) |j| {
            var sum: f32 = 0;
            for (0..matrix.rows) |i| {
                sum += matrix_cpu.data[i * matrix.cols + j];
            }
            setMatrixElement(undefined, result, 0, j, sum);
        }

        return result;
    }

    pub fn transpose(ptr: *anyopaque, matrix: *const Matrix, allocator: Allocator) error{OutOfMemory}!*Matrix {
        _ = @as(*CPUBackend, @ptrCast(@alignCast(ptr)));

        const result = try initMatrix(ptr, allocator, matrix.cols, matrix.rows);
        errdefer deinitMatrix(undefined, result);

        for (0..matrix.rows) |i| {
            for (0..matrix.cols) |j| {
                const value = getMatrixElement(undefined, matrix, i, j);
                setMatrixElement(undefined, result, j, i, value);
            }
        }

        return result;
    }

    pub fn extractBatch(ptr: *anyopaque, matrix: *const Matrix, start: usize, end: usize, allocator: Allocator) error{ OutOfMemory, InvalidBatchIndices }!*Matrix {
        if (start >= matrix.rows or end > matrix.rows or start >= end) {
            return error.InvalidBatchIndices;
        }

        _ = @as(*CPUBackend, @ptrCast(@alignCast(ptr)));

        const batch_size = end - start;
        const result = try initMatrix(ptr, allocator, batch_size, matrix.cols);
        errdefer deinitMatrix(undefined, result);

        const matrix_cpu = @as(*const CPUMatrix, @ptrCast(@alignCast(matrix.impl_data)));
        const result_cpu = @as(*CPUMatrix, @ptrCast(@alignCast(result.impl_data)));

        for (0..batch_size) |i| {
            const source_offset = (start + i) * matrix.cols;
            const dest_offset = i * matrix.cols;
            @memcpy(result_cpu.data[dest_offset..(dest_offset + matrix.cols)], matrix_cpu.data[source_offset..(source_offset + matrix.cols)]);
        }

        return result;
    }

    pub fn randomize(_: *anyopaque, matrix: *Matrix, min: f64, max: f64) void {
        var prng = std.Random.DefaultPrng.init(randomSeed());
        const rand = prng.random();

        const matrix_cpu = @as(*CPUMatrix, @ptrCast(@alignCast(matrix.impl_data)));
        invalidatePacked(matrix_cpu);

        for (0..matrix_cpu.data.len) |i| {
            const rand_float = rand.float(f64);
            matrix_cpu.data[i] = @floatCast(min + (rand_float * (max - min)));
        }
    }

    pub fn applyActivation(ptr: *anyopaque, matrix: *const Matrix, activation: *const fn (f64) f64, allocator: Allocator) error{OutOfMemory}!*Matrix {
        _ = @as(*CPUBackend, @ptrCast(@alignCast(ptr)));

        const result = try initMatrix(ptr, allocator, matrix.rows, matrix.cols);
        errdefer deinitMatrix(undefined, result);

        const matrix_cpu = @as(*const CPUMatrix, @ptrCast(@alignCast(matrix.impl_data)));
        const result_cpu = @as(*CPUMatrix, @ptrCast(@alignCast(result.impl_data)));

        for (0..result_cpu.data.len) |i| {
            result_cpu.data[i] = @floatCast(activation(@floatCast(matrix_cpu.data[i])));
        }

        return result;
    }

    pub fn applySoftmax(ptr: *anyopaque, matrix: *const Matrix, allocator: Allocator) error{OutOfMemory}!*Matrix {
        _ = @as(*CPUBackend, @ptrCast(@alignCast(ptr)));

        const result = try initMatrix(ptr, allocator, matrix.rows, matrix.cols);
        errdefer deinitMatrix(undefined, result);

        // Process each row independently
        for (0..matrix.rows) |i| {
            // Find max value in row for numerical stability
            var max_val: f64 = -std.math.inf(f64);
            for (0..matrix.cols) |j| {
                max_val = @max(max_val, getMatrixElement(undefined, matrix, i, j));
            }

            // Compute exp(x - max) and sum
            var sum: f64 = 0.0;
            for (0..matrix.cols) |j| {
                const shifted = getMatrixElement(undefined, matrix, i, j) - max_val;
                const exp_val = std.math.exp(shifted);
                setMatrixElement(undefined, result, i, j, exp_val);
                sum += exp_val;
            }

            // Normalize by sum
            for (0..matrix.cols) |j| {
                const val = getMatrixElement(undefined, result, i, j) / sum;
                setMatrixElement(undefined, result, i, j, val);
            }
        }

        return result;
    }

    pub fn layerNorm(ptr: *anyopaque, matrix: *const Matrix, gamma: *const Matrix, beta: *const Matrix, epsilon: f64, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!*Matrix {
        if (gamma.rows != 1 or beta.rows != 1 or gamma.cols != matrix.cols or beta.cols != matrix.cols) {
            return error.DimensionMismatch;
        }

        const result = try initMatrix(ptr, allocator, matrix.rows, matrix.cols);
        errdefer deinitMatrix(undefined, result);
        const input_data = @as(*const CPUMatrix, @ptrCast(@alignCast(matrix.impl_data)));
        const gamma_data = @as(*const CPUMatrix, @ptrCast(@alignCast(gamma.impl_data)));
        const beta_data = @as(*const CPUMatrix, @ptrCast(@alignCast(beta.impl_data)));
        const result_data = @as(*CPUMatrix, @ptrCast(@alignCast(result.impl_data)));
        const width = @as(f32, @floatFromInt(matrix.cols));
        const epsilon_f32: f32 = @floatCast(epsilon);

        for (0..matrix.rows) |row| {
            const offset = row * matrix.cols;
            var mean: f32 = 0.0;
            for (0..matrix.cols) |col| mean += input_data.data[offset + col];
            mean /= width;

            var variance: f32 = 0.0;
            for (0..matrix.cols) |col| {
                const centered = input_data.data[offset + col] - mean;
                variance += centered * centered;
            }
            variance /= width;
            const inverse_std = 1.0 / @sqrt(variance + epsilon_f32);

            for (0..matrix.cols) |col| {
                const normalized = (input_data.data[offset + col] - mean) * inverse_std;
                result_data.data[offset + col] = normalized * gamma_data.data[col] + beta_data.data[col];
            }
        }
        return result;
    }

    pub fn layerNormBackward(ptr: *anyopaque, input: *const Matrix, gamma: *const Matrix, output_gradient: *const Matrix, epsilon: f64, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!backend.LayerNormGradients {
        if (gamma.rows != 1 or gamma.cols != input.cols or
            output_gradient.rows != input.rows or output_gradient.cols != input.cols)
        {
            return error.DimensionMismatch;
        }

        const input_gradient = try initMatrix(ptr, allocator, input.rows, input.cols);
        errdefer deinitMatrix(undefined, input_gradient);
        const gamma_gradient = try initMatrix(ptr, allocator, 1, input.cols);
        errdefer deinitMatrix(undefined, gamma_gradient);
        const beta_gradient = try initMatrix(ptr, allocator, 1, input.cols);
        errdefer deinitMatrix(undefined, beta_gradient);
        const input_data = @as(*const CPUMatrix, @ptrCast(@alignCast(input.impl_data)));
        const gamma_data = @as(*const CPUMatrix, @ptrCast(@alignCast(gamma.impl_data)));
        const output_gradient_data = @as(*const CPUMatrix, @ptrCast(@alignCast(output_gradient.impl_data)));
        const input_gradient_data = @as(*CPUMatrix, @ptrCast(@alignCast(input_gradient.impl_data)));
        const gamma_gradient_data = @as(*CPUMatrix, @ptrCast(@alignCast(gamma_gradient.impl_data)));
        const beta_gradient_data = @as(*CPUMatrix, @ptrCast(@alignCast(beta_gradient.impl_data)));
        @memset(gamma_gradient_data.data, 0);
        @memset(beta_gradient_data.data, 0);
        const width = @as(f32, @floatFromInt(input.cols));
        const epsilon_f32: f32 = @floatCast(epsilon);

        for (0..input.rows) |row| {
            const offset = row * input.cols;
            var mean: f32 = 0;
            for (0..input.cols) |col| mean += input_data.data[offset + col];
            mean /= width;
            var variance: f32 = 0;
            for (0..input.cols) |col| {
                const centered = input_data.data[offset + col] - mean;
                variance += centered * centered;
            }
            variance /= width;
            const inverse_std = 1.0 / @sqrt(variance + epsilon_f32);
            var sum_gradient: f32 = 0;
            var sum_gradient_normalized: f32 = 0;
            for (0..input.cols) |col| {
                const normalized = (input_data.data[offset + col] - mean) * inverse_std;
                const gradient = output_gradient_data.data[offset + col];
                const normalized_gradient = gradient * gamma_data.data[col];
                sum_gradient += normalized_gradient;
                sum_gradient_normalized += normalized_gradient * normalized;
                gamma_gradient_data.data[col] += gradient * normalized;
                beta_gradient_data.data[col] += gradient;
            }
            for (0..input.cols) |col| {
                const normalized = (input_data.data[offset + col] - mean) * inverse_std;
                const normalized_gradient = output_gradient_data.data[offset + col] * gamma_data.data[col];
                input_gradient_data.data[offset + col] = inverse_std / width *
                    (width * normalized_gradient - sum_gradient - normalized * sum_gradient_normalized);
            }
        }
        return .{ .input = input_gradient, .gamma = gamma_gradient, .beta = beta_gradient };
    }

    pub fn causalSelfAttention(ptr: *anyopaque, query: *const Matrix, key: *const Matrix, value: *const Matrix, heads: usize, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!*Matrix {
        if (heads == 0 or query.rows != key.rows or query.rows != value.rows or
            query.cols != key.cols or query.cols != value.cols or query.cols % heads != 0)
        {
            return error.DimensionMismatch;
        }

        const result = try initMatrix(ptr, allocator, query.rows, query.cols);
        errdefer deinitMatrix(undefined, result);
        const query_data = @as(*const CPUMatrix, @ptrCast(@alignCast(query.impl_data)));
        const key_data = @as(*const CPUMatrix, @ptrCast(@alignCast(key.impl_data)));
        const value_data = @as(*const CPUMatrix, @ptrCast(@alignCast(value.impl_data)));
        const result_data = @as(*CPUMatrix, @ptrCast(@alignCast(result.impl_data)));
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
                        score += query_data.data[query_position * query.cols + channel_offset + channel] *
                            key_data.data[key_position * key.cols + channel_offset + channel];
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
                            value_data.data[key_position * value.cols + channel_offset + channel];
                    }
                    result_data.data[query_position * result.cols + channel_offset + channel] = output;
                }
            }
        }
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
        errdefer deinitMatrix(undefined, query_gradient);
        const key_gradient = try initMatrix(ptr, allocator, key.rows, key.cols);
        errdefer deinitMatrix(undefined, key_gradient);
        const value_gradient = try initMatrix(ptr, allocator, value.rows, value.cols);
        errdefer deinitMatrix(undefined, value_gradient);
        const query_data = @as(*const CPUMatrix, @ptrCast(@alignCast(query.impl_data)));
        const key_data = @as(*const CPUMatrix, @ptrCast(@alignCast(key.impl_data)));
        const value_data = @as(*const CPUMatrix, @ptrCast(@alignCast(value.impl_data)));
        const output_gradient_data = @as(*const CPUMatrix, @ptrCast(@alignCast(output_gradient.impl_data)));
        const query_gradient_data = @as(*CPUMatrix, @ptrCast(@alignCast(query_gradient.impl_data)));
        const key_gradient_data = @as(*CPUMatrix, @ptrCast(@alignCast(key_gradient.impl_data)));
        const value_gradient_data = @as(*CPUMatrix, @ptrCast(@alignCast(value_gradient.impl_data)));
        @memset(query_gradient_data.data, 0);
        @memset(key_gradient_data.data, 0);
        @memset(value_gradient_data.data, 0);
        const probabilities = try allocator.alloc(f32, query.rows);
        defer allocator.free(probabilities);
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
                        score += query_data.data[query_position * query.cols + channel_offset + channel] *
                            key_data.data[key_position * key.cols + channel_offset + channel];
                    }
                    score *= attention_scale;
                    probabilities[key_position] = score;
                    max_score = @max(max_score, score);
                }
                var denominator: f32 = 0;
                for (0..query_position + 1) |key_position| {
                    probabilities[key_position] = @exp(probabilities[key_position] - max_score);
                    denominator += probabilities[key_position];
                }
                var weighted_probability_gradient: f32 = 0;
                for (0..query_position + 1) |key_position| {
                    probabilities[key_position] /= denominator;
                    var probability_gradient: f32 = 0;
                    for (0..head_width) |channel| {
                        probability_gradient += output_gradient_data.data[query_position * query.cols + channel_offset + channel] *
                            value_data.data[key_position * value.cols + channel_offset + channel];
                    }
                    probability_gradients[key_position] = probability_gradient;
                    weighted_probability_gradient += probabilities[key_position] * probability_gradient;
                }
                for (0..query_position + 1) |key_position| {
                    const score_gradient = probabilities[key_position] *
                        (probability_gradients[key_position] - weighted_probability_gradient);
                    for (0..head_width) |channel| {
                        const query_index = query_position * query.cols + channel_offset + channel;
                        const key_index = key_position * key.cols + channel_offset + channel;
                        query_gradient_data.data[query_index] += score_gradient * key_data.data[key_index] * attention_scale;
                        key_gradient_data.data[key_index] += score_gradient * query_data.data[query_index] * attention_scale;
                        value_gradient_data.data[key_index] += probabilities[key_position] * output_gradient_data.data[query_index];
                    }
                }
            }
        }
        return .{ .query = query_gradient, .key = key_gradient, .value = value_gradient };
    }

    pub fn embeddingLookup(ptr: *anyopaque, table: *const Matrix, indices: *const Matrix, allocator: Allocator) error{ OutOfMemory, DimensionMismatch, InvalidIndex }!*Matrix {
        if (indices.cols != 1) return error.DimensionMismatch;
        const result = try initMatrix(ptr, allocator, indices.rows, table.cols);
        errdefer deinitMatrix(undefined, result);
        const table_data = @as(*const CPUMatrix, @ptrCast(@alignCast(table.impl_data)));
        const indices_data = @as(*const CPUMatrix, @ptrCast(@alignCast(indices.impl_data)));
        const result_data = @as(*CPUMatrix, @ptrCast(@alignCast(result.impl_data)));
        for (0..indices.rows) |row| {
            const raw_index = indices_data.data[row];
            if (!std.math.isFinite(raw_index) or raw_index < 0 or @floor(raw_index) != raw_index or raw_index >= @as(f32, @floatFromInt(table.rows))) {
                return error.InvalidIndex;
            }
            const index: usize = @intFromFloat(raw_index);
            @memcpy(result_data.data[row * table.cols ..][0..table.cols], table_data.data[index * table.cols ..][0..table.cols]);
        }
        return result;
    }

    pub fn embeddingGradient(ptr: *anyopaque, indices: *const Matrix, output_gradient: *const Matrix, vocabulary_size: usize, allocator: Allocator) error{ OutOfMemory, DimensionMismatch, InvalidIndex }!*Matrix {
        if (indices.cols != 1 or indices.rows != output_gradient.rows or vocabulary_size == 0) return error.DimensionMismatch;
        const result = try initMatrix(ptr, allocator, vocabulary_size, output_gradient.cols);
        errdefer deinitMatrix(undefined, result);
        const indices_data = @as(*const CPUMatrix, @ptrCast(@alignCast(indices.impl_data)));
        const output_gradient_data = @as(*const CPUMatrix, @ptrCast(@alignCast(output_gradient.impl_data)));
        const result_data = @as(*CPUMatrix, @ptrCast(@alignCast(result.impl_data)));
        @memset(result_data.data, 0);
        for (0..indices.rows) |row| {
            const raw_index = indices_data.data[row];
            if (!std.math.isFinite(raw_index) or raw_index < 0 or @floor(raw_index) != raw_index or raw_index >= @as(f32, @floatFromInt(vocabulary_size))) {
                return error.InvalidIndex;
            }
            const index: usize = @intFromFloat(raw_index);
            for (0..output_gradient.cols) |col| {
                result_data.data[index * output_gradient.cols + col] += output_gradient_data.data[row * output_gradient.cols + col];
            }
        }
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
        errdefer deinitMatrix(undefined, result);
        const query_data = @as(*const CPUMatrix, @ptrCast(@alignCast(query.impl_data)));
        const key_data = @as(*const CPUMatrix, @ptrCast(@alignCast(key.impl_data)));
        const value_data = @as(*const CPUMatrix, @ptrCast(@alignCast(value.impl_data)));
        const key_cache_data = @as(*CPUMatrix, @ptrCast(@alignCast(key_cache.impl_data)));
        const value_cache_data = @as(*CPUMatrix, @ptrCast(@alignCast(value_cache.impl_data)));
        const result_data = @as(*CPUMatrix, @ptrCast(@alignCast(result.impl_data)));
        @memcpy(key_cache_data.data[position * query.cols ..][0..query.cols], key_data.data[0..query.cols]);
        @memcpy(value_cache_data.data[position * query.cols ..][0..query.cols], value_data.data[0..query.cols]);
        const probabilities = try allocator.alloc(f32, position + 1);
        defer allocator.free(probabilities);
        const head_width = query.cols / heads;
        const attention_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_width)));
        for (0..heads) |head| {
            const offset = head * head_width;
            var max_score = -std.math.inf(f32);
            for (0..position + 1) |key_position| {
                var score: f32 = 0;
                for (0..head_width) |channel| score += query_data.data[offset + channel] * key_cache_data.data[key_position * query.cols + offset + channel];
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
                for (0..position + 1) |key_position| output += probabilities[key_position] / denominator * value_cache_data.data[key_position * query.cols + offset + channel];
                result_data.data[offset + channel] = output;
            }
        }
        return result;
    }

    pub fn applyGLU(ptr: *anyopaque, linear_part: *const Matrix, gating_part: *const Matrix, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!*Matrix {
        if (linear_part.rows != gating_part.rows or linear_part.cols != gating_part.cols) {
            return error.DimensionMismatch;
        }

        // Apply sigmoid to the gating part
        const sigmoid_gate = try applyActivation(ptr, gating_part, sigmoid, allocator);
        defer deinitMatrix(undefined, sigmoid_gate);

        // Element-wise multiply with the linear part
        return elementWiseMultiply(ptr, linear_part, sigmoid_gate, allocator);
    }

    pub fn applySwiGLU(ptr: *anyopaque, linear_part: *const Matrix, gating_part: *const Matrix, allocator: Allocator) error{ OutOfMemory, DimensionMismatch }!*Matrix {
        if (linear_part.rows != gating_part.rows or linear_part.cols != gating_part.cols) {
            return error.DimensionMismatch;
        }

        // Apply swish to the gating part
        const swish_gate = try applyActivation(ptr, gating_part, swish, allocator);
        defer deinitMatrix(undefined, swish_gate);

        // Element-wise multiply with the linear part
        return elementWiseMultiply(ptr, linear_part, swish_gate, allocator);
    }

    fn getBackendType(_: *anyopaque) BackendType {
        return .CPU;
    }
};

const MatmulTile = struct {
    input: []const f32,
    input_rows: usize,
    input_cols: usize,
    weights: []const f32,
    weight_stride: usize,
    output_cols: usize,
    bias: ?[]const f32,
    apply_gelu: bool,
    output: []f32,
    column_start: usize,
    column_end: usize,
};

fn sameAllocator(left: Allocator, right: Allocator) bool {
    return left.ptr == right.ptr and left.vtable == right.vtable;
}

fn invalidatePacked(matrix: *CPUMatrix) void {
    if (matrix.packed_weight) |packed_weight| {
        matrix.allocator.free(packed_weight.data);
        matrix.packed_weight = null;
    }
}

fn weightValues(matrix: *const CPUMatrix) []const f32 {
    return if (matrix.packed_weight) |packed_weight| packed_weight.data else matrix.data;
}

fn weightStride(matrix: *const CPUMatrix, original_stride: usize) usize {
    return if (matrix.packed_weight) |packed_weight| packed_weight.stride else original_stride;
}

fn runMatmulTiles(
    requested_tiles: usize,
    input: []const f32,
    input_rows: usize,
    input_cols: usize,
    weights: []const f32,
    weight_stride: usize,
    output_cols: usize,
    bias: ?[]const f32,
    apply_gelu: bool,
    output: []f32,
) void {
    if (input_rows == 0 or output_cols == 0) return;
    const tile_count = @min(@max(requested_tiles, 1), output_cols);
    if (tile_count == 1) {
        executeMatmulTile(&.{
            .input = input,
            .input_rows = input_rows,
            .input_cols = input_cols,
            .weights = weights,
            .weight_stride = weight_stride,
            .output_cols = output_cols,
            .bias = bias,
            .apply_gelu = apply_gelu,
            .output = output,
            .column_start = 0,
            .column_end = output_cols,
        });
        return;
    }

    var tasks: [max_output_tiles]MatmulTile = undefined;
    const columns_per_tile = std.math.divCeil(usize, output_cols, tile_count) catch unreachable;
    var actual_tiles: usize = 0;
    for (0..tile_count) |index| {
        const column_start = index * columns_per_tile;
        if (column_start >= output_cols) break;
        tasks[actual_tiles] = .{
            .input = input,
            .input_rows = input_rows,
            .input_cols = input_cols,
            .weights = weights,
            .weight_stride = weight_stride,
            .output_cols = output_cols,
            .bias = bias,
            .apply_gelu = apply_gelu,
            .output = output,
            .column_start = column_start,
            .column_end = @min(column_start + columns_per_tile, output_cols),
        };
        actual_tiles += 1;
    }

    var threads: [max_output_tiles - 1]std.Thread = undefined;
    var spawned: usize = 0;
    var spawn_failed = false;
    for (1..actual_tiles) |index| {
        threads[spawned] = std.Thread.spawn(.{}, executeMatmulTile, .{&tasks[index]}) catch {
            spawn_failed = true;
            break;
        };
        spawned += 1;
    }
    if (spawn_failed) {
        for (threads[0..spawned]) |thread| thread.join();
        @memset(output, 0);
        executeMatmulTile(&.{
            .input = input,
            .input_rows = input_rows,
            .input_cols = input_cols,
            .weights = weights,
            .weight_stride = weight_stride,
            .output_cols = output_cols,
            .bias = bias,
            .apply_gelu = apply_gelu,
            .output = output,
            .column_start = 0,
            .column_end = output_cols,
        });
        return;
    }

    executeMatmulTile(&tasks[0]);
    for (threads[0..spawned]) |thread| thread.join();
}

fn executeMatmulTile(task: *const MatmulTile) void {
    for (0..task.input_rows) |row| {
        const output = task.output[row * task.output_cols + task.column_start .. row * task.output_cols + task.column_end];
        if (task.bias) |bias| {
            @memcpy(output, bias[task.column_start..task.column_end]);
        } else {
            @memset(output, 0);
        }
        for (0..task.input_cols) |inner| {
            const left = task.input[row * task.input_cols + inner];
            const right = task.weights[inner * task.weight_stride + task.column_start .. inner * task.weight_stride + task.column_end];
            accumulateScaled(output, right, left);
        }
        if (task.apply_gelu) {
            for (output) |*value| {
                value.* = @floatCast(@import("activation.zig").Activation.gelu(value.*));
            }
        }
    }
}

/// Readable scalar matmul used by parity tests for the packed/vectorized path.
fn matmulScalarReference(
    input: []const f32,
    input_rows: usize,
    input_cols: usize,
    weights: []const f32,
    output_cols: usize,
    output: []f32,
) void {
    @memset(output, 0);
    for (0..input_rows) |row| {
        for (0..output_cols) |column| {
            var sum: f32 = 0;
            for (0..input_cols) |inner| {
                sum += input[row * input_cols + inner] * weights[inner * output_cols + column];
            }
            output[row * output_cols + column] = sum;
        }
    }
}

// Helper activation functions needed for GLU and SwiGLU
fn sigmoid(x: f64) f64 {
    return 1.0 / (1.0 + std.math.exp(-x));
}

/// Vector-width output tiling keeps inference weights in their existing
/// contiguous row-major layout while making the packed CPU path explicit. The
/// scalar tail is also the readable reference for narrow matrices.
fn accumulateScaled(output: []f32, right: []const f32, left: f32) void {
    const Vector = @Vector(vector_width, f32);
    const scale: Vector = @splat(left);
    var column: usize = 0;
    while (column + vector_width <= output.len) : (column += vector_width) {
        const output_array: [vector_width]f32 = output[column..][0..vector_width].*;
        const right_array: [vector_width]f32 = right[column..][0..vector_width].*;
        var output_vector: Vector = output_array;
        const right_vector: Vector = right_array;
        output_vector += scale * right_vector;
        output[column..][0..vector_width].* = @as([vector_width]f32, output_vector);
    }
    for (output[column..], right[column..]) |*value, factor| value.* += left * factor;
}

test "vectorized accumulation preserves scalar tails" {
    var output = [_]f32{0} ** 11;
    const right = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
    accumulateScaled(&output, &right, 0.5);
    for (output, right) |actual, expected| {
        try std.testing.expectApproxEqAbs(expected * 0.5, actual, 0.000001);
    }
}

test "packed parallel matmul matches the scalar reference" {
    const allocator = std.testing.allocator;
    const cpu = try CPUBackend.init(allocator);
    defer cpu.deinit();
    const ptr: *anyopaque = @ptrCast(cpu);
    const input_values = [_]f32{ 1, 2, 3, -2, 0.5, 4 };
    const weight_values = [_]f32{
        1,  2, 3, 4,  5,  6,  7,  8,  9,  10, 11,
        -1, 0, 1, 2,  3,  4,  5,  6,  7,  8,  9,
        2,  1, 0, -1, -2, -3, -4, -5, -6, -7, -8,
    };
    const input = try CPUBackend.initMatrix(ptr, allocator, 2, 3);
    defer CPUBackend.deinitMatrix(ptr, input);
    const weights = try CPUBackend.initMatrix(ptr, allocator, 3, 11);
    defer CPUBackend.deinitMatrix(ptr, weights);
    try std.testing.expect(CPUBackend.writeMatrixF32(ptr, input, &input_values));
    try std.testing.expect(CPUBackend.writeMatrixF32(ptr, weights, &weight_values));

    try CPUBackend.prepareInferenceWeight(ptr, weights);
    const cpu_weights = @as(*const CPUMatrix, @ptrCast(@alignCast(weights.impl_data)));
    try std.testing.expectEqual(@as(usize, 16), cpu_weights.packed_weight.?.stride);
    try CPUBackend.configureOutputTiles(ptr, 4);
    const result = try CPUBackend.dotProduct(ptr, input, weights, allocator);
    defer CPUBackend.deinitMatrix(ptr, result);

    var expected: [22]f32 = undefined;
    matmulScalarReference(&input_values, 2, 3, &weight_values, 11, &expected);
    var actual: [22]f32 = undefined;
    try std.testing.expect(CPUBackend.readMatrixF32(ptr, result, &actual));
    for (expected, actual) |reference, optimized| {
        try std.testing.expectApproxEqAbs(reference, optimized, 1e-4);
    }
}

test "mutating a packed weight invalidates its inference copy" {
    const allocator = std.testing.allocator;
    const cpu = try CPUBackend.init(allocator);
    defer cpu.deinit();
    const ptr: *anyopaque = @ptrCast(cpu);
    const weights = try CPUBackend.initMatrix(ptr, allocator, 2, 3);
    defer CPUBackend.deinitMatrix(ptr, weights);

    try CPUBackend.prepareInferenceWeight(ptr, weights);
    const cpu_weights = @as(*CPUMatrix, @ptrCast(@alignCast(weights.impl_data)));
    try std.testing.expect(cpu_weights.packed_weight != null);
    CPUBackend.setMatrixElement(ptr, weights, 0, 0, 2);
    try std.testing.expect(cpu_weights.packed_weight == null);
    try CPUBackend.prepareInferenceWeight(ptr, weights);
    try std.testing.expect(cpu_weights.packed_weight != null);
}

test "CPU workspace reuses intermediate matrix storage" {
    const allocator = std.testing.allocator;
    const cpu = try CPUBackend.init(allocator);
    defer cpu.deinit();
    const ptr: *anyopaque = @ptrCast(cpu);

    const first = try CPUBackend.initMatrix(ptr, allocator, 3, 5);
    const first_data = @as(*CPUMatrix, @ptrCast(@alignCast(first.impl_data))).data.ptr;
    CPUBackend.deinitMatrix(ptr, first);
    const second = try CPUBackend.initMatrix(ptr, allocator, 3, 5);
    defer CPUBackend.deinitMatrix(ptr, second);
    const second_data = @as(*CPUMatrix, @ptrCast(@alignCast(second.impl_data))).data.ptr;

    try std.testing.expectEqual(first_data, second_data);
    try std.testing.expectEqual(@as(usize, 1), cpu.storage_allocations);
    try std.testing.expectEqual(@as(usize, 1), cpu.workspace_reuses);
}

fn swish(x: f64) f64 {
    const beta: f64 = 1.0;
    return x * sigmoid(beta * x);
}

/// Creates a CPU backend instance, returning an opaque pointer
pub fn createCPUBackend(allocator: Allocator) !*anyopaque {
    const cpu_backend = try CPUBackend.init(allocator);
    errdefer cpu_backend.deinit(); // Ensure cleanup on populate error if any
    return @ptrCast(cpu_backend);
}
