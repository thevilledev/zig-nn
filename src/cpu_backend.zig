const std = @import("std");
const Allocator = std.mem.Allocator;
const backend = @import("backend.zig");
const ComputeBackend = backend.ComputeBackend;
const BackendType = backend.BackendType;
const Matrix = backend.Matrix;
const randomSeed = @import("matrix.zig").randomSeed;

/// CPU-specific implementation of a matrix
const CPUMatrix = struct {
    data: []f32,
    allocator: Allocator,
};

/// CPU backend implementation
pub const CPUBackend = struct {
    allocator: Allocator,
    stats: backend.RuntimeStats,

    /// Initialize a new CPU backend
    pub fn init(allocator: Allocator) !*CPUBackend {
        const cpu_backend = try allocator.create(CPUBackend);
        cpu_backend.* = CPUBackend{
            .allocator = allocator,
            .stats = .{},
        };
        return cpu_backend;
    }

    /// Free all resources used by the CPU backend
    pub fn deinit(self: *CPUBackend) void {
        self.allocator.destroy(self);
    }

    // ComputeBackend interface implementations
    pub fn initMatrix(ptr: *anyopaque, allocator: Allocator, rows: usize, cols: usize) error{OutOfMemory}!*Matrix {
        const self = @as(*CPUBackend, @ptrCast(@alignCast(ptr)));
        // Allocate the matrix wrapper
        const matrix = try allocator.create(Matrix);
        errdefer allocator.destroy(matrix);

        // Allocate the CPU-specific data
        const cpu_data = try allocator.create(CPUMatrix);
        errdefer allocator.destroy(cpu_data);

        // Allocate the data array
        const data = try allocator.alloc(f32, rows * cols);
        errdefer allocator.free(data);

        // Initialize with zeros
        @memset(data, 0);

        // Set up the CPU matrix data
        cpu_data.* = .{
            .data = data,
            .allocator = allocator,
        };

        // Set up the matrix
        matrix.* = .{
            .rows = rows,
            .cols = cols,
            .backend = undefined, // Will be set by the caller
            .impl_data = cpu_data,
        };
        self.stats.buffer_allocations += 1;

        return matrix;
    }

    pub fn runtimeStats(ptr: *anyopaque) backend.RuntimeStats {
        const self = @as(*CPUBackend, @ptrCast(@alignCast(ptr)));
        return self.stats;
    }

    pub fn resetRuntimeStats(ptr: *anyopaque) void {
        const self = @as(*CPUBackend, @ptrCast(@alignCast(ptr)));
        self.stats = .{};
    }

    pub fn beginBatch(_: *anyopaque) !void {}

    pub fn endBatch(_: *anyopaque) !void {}

    pub fn synchronize(_: *anyopaque) !void {}

    pub fn deinitMatrix(_: *anyopaque, matrix: *Matrix) void {
        // First, cast to get the CPU-specific data
        const cpu_data = @as(*CPUMatrix, @ptrCast(@alignCast(matrix.impl_data)));
        const allocator = cpu_data.allocator;

        // Free the data array
        allocator.free(cpu_data.data);

        // Free the CPU matrix data
        allocator.destroy(cpu_data);

        // Free the matrix wrapper
        allocator.destroy(matrix);
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
        cpu_data.data[row * matrix.cols + col] = @floatCast(value);
    }

    pub fn fillMatrix(_: *anyopaque, matrix: *Matrix, value: f64) void {
        const cpu_data = @as(*CPUMatrix, @ptrCast(@alignCast(matrix.impl_data)));
        for (cpu_data.data) |*element| {
            element.* = @floatCast(value);
        }
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

        _ = @as(*CPUBackend, @ptrCast(@alignCast(ptr)));

        const result = try initMatrix(ptr, allocator, a.rows, b.cols);
        errdefer deinitMatrix(undefined, result);

        const a_cpu = @as(*const CPUMatrix, @ptrCast(@alignCast(a.impl_data)));
        const b_cpu = @as(*const CPUMatrix, @ptrCast(@alignCast(b.impl_data)));

        for (0..a.rows) |i| {
            for (0..b.cols) |j| {
                var sum: f32 = 0;
                for (0..a.cols) |k| {
                    sum += a_cpu.data[i * a.cols + k] * b_cpu.data[k * b.cols + j];
                }
                setMatrixElement(undefined, result, i, j, sum);
            }
        }

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
        if (batch == 0 or inner_a != inner_b or
            a.rows * a.cols != batch * a_rows * a_cols or
            b.rows * b.cols != batch * b_rows * b_cols)
        {
            return error.DimensionMismatch;
        }

        const result = try initMatrix(ptr, allocator, batch * output_rows, output_cols);
        errdefer deinitMatrix(undefined, result);
        const a_cpu = @as(*const CPUMatrix, @ptrCast(@alignCast(a.impl_data)));
        const b_cpu = @as(*const CPUMatrix, @ptrCast(@alignCast(b.impl_data)));
        const result_cpu = @as(*CPUMatrix, @ptrCast(@alignCast(result.impl_data)));

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

        for (0..a.rows * a.cols) |i| {
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

        for (0..a.rows * a.cols) |i| {
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

        for (0..a.rows * a.cols) |i| {
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

        for (0..matrix.rows * matrix.cols) |i| {
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

        for (0..matrix.rows * matrix.cols) |i| {
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

        for (0..matrix.rows * matrix.cols) |i| {
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

// Helper activation functions needed for GLU and SwiGLU
fn sigmoid(x: f64) f64 {
    return 1.0 / (1.0 + std.math.exp(-x));
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
