const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;
const dimensions = @import("dimensions.zig");

pub const MatrixError = error{
    IndexOutOfBounds,
    DimensionMismatch,
    DimensionOverflow,
    InvalidBatchIndices,
};

pub fn randomSeed() u64 {
    var seed: u64 = undefined;
    std.Options.debug_io.random(std.mem.asBytes(&seed));
    return seed;
}

/// Matrix struct representing a 2D array of f64 values
/// Used for linear algebra operations in neural networks
/// Stores data in row-major order for efficient memory access
pub const Matrix = struct {
    rows: usize,
    cols: usize,
    data: []f64,
    allocator: Allocator,

    /// Creates a new matrix with specified dimensions
    /// Initializes all elements to zero
    /// Memory complexity: O(rows * cols)
    /// Time complexity: O(rows * cols)
    ///
    /// Parameters:
    ///   - allocator: Memory allocator for matrix data
    ///   - rows: Number of rows in the matrix
    ///   - cols: Number of columns in the matrix
    pub fn init(allocator: Allocator, rows: usize, cols: usize) !Matrix {
        const element_count = dimensions.elementCount(rows, cols) catch
            return MatrixError.DimensionOverflow;
        const data = try allocator.alloc(f64, element_count);
        @memset(data, 0);
        return .{
            .rows = rows,
            .cols = cols,
            .data = data,
            .allocator = allocator,
        };
    }

    /// Fills all elements of the matrix with a given value
    /// Time complexity: O(rows * cols)
    ///
    /// Parameters:
    ///   - value: Value to fill the matrix with
    pub fn fill(self: *Matrix, value: f64) void {
        @memset(self.data, value);
    }

    /// Creates a deep copy of an existing matrix
    /// Essential for operations where we need to preserve the original matrix
    /// Memory complexity: O(rows * cols)
    /// Time complexity: O(rows * cols)
    ///
    /// Parameters:
    ///   - source: Matrix to copy
    ///   - allocator: Memory allocator for the new matrix
    pub fn copy(source: Matrix, allocator: Allocator) !Matrix {
        const result = try Matrix.init(allocator, source.rows, source.cols);
        @memcpy(result.data, source.data);
        return result;
    }

    /// Frees the matrix memory
    /// Should be called when matrix is no longer needed to prevent memory leaks
    /// Time complexity: O(1)
    pub fn deinit(self: Matrix) void {
        self.allocator.free(self.data);
    }

    /// Gets value at specified position (i,j)
    /// Matrix is stored in row-major order: index = i * cols + j
    /// Time complexity: O(1)
    ///
    /// Parameters:
    ///   - row: Row index i (0-based)
    ///   - col: Column index j (0-based)
    pub fn get(self: Matrix, row: usize, col: usize) MatrixError!f64 {
        if (row >= self.rows or col >= self.cols) {
            return MatrixError.IndexOutOfBounds;
        }
        return self.data[row * self.cols + col];
    }

    /// Sets value at specified position (i,j)
    /// Matrix is stored in row-major order: index = i * cols + j
    /// Time complexity: O(1)
    ///
    /// Parameters:
    ///   - row: Row index i (0-based)
    ///   - col: Column index j (0-based)
    ///   - value: Value to set at position (i,j)
    pub fn set(self: *Matrix, row: usize, col: usize, value: f64) MatrixError!void {
        if (row >= self.rows or col >= self.cols) {
            return MatrixError.IndexOutOfBounds;
        }
        self.data[row * self.cols + col] = value;
    }

    /// Fills matrix with random values in range [min, max]
    /// Uses uniform distribution: P(x) = 1/(max-min) for x in [min,max]
    /// Common initialization for neural network weights
    /// Time complexity: O(rows * cols)
    ///
    /// Parameters:
    ///   - min: Minimum value (inclusive)
    ///   - max: Maximum value (inclusive)
    pub fn randomize(self: *Matrix, min: f64, max: f64) void {
        var prng = std.Random.DefaultPrng.init(randomSeed());
        self.randomizeWith(prng.random(), min, max);
    }

    /// Fills the matrix from a caller-provided random stream.
    /// This is useful for reproducible experiments while `randomize` remains
    /// the convenient entropy-seeded default.
    pub fn randomizeWith(self: *Matrix, random: std.Random, min: f64, max: f64) void {
        for (self.data) |*element| {
            element.* = min + random.float(f64) * (max - min);
        }
    }

    /// Performs matrix multiplication (dot product): C = A × B
    /// Mathematical definition: C[i,j] = Σₖ A[i,k] × B[k,j]
    /// Where k goes from 0 to A.cols-1 (or B.rows-1)
    /// Time complexity: O(rows * cols * other.cols)
    /// Memory complexity: O(rows * other.cols)
    ///
    /// Parameters:
    ///   - other: Right-hand matrix B in A × B
    ///   - allocator: Memory allocator for result matrix
    /// Returns: Result matrix C with dimensions (self.rows × other.cols)
    pub fn dotProduct(self: Matrix, other: Matrix, allocator: Allocator) !Matrix {
        if (self.cols != other.rows) {
            return MatrixError.DimensionMismatch;
        }

        const result = try Matrix.init(allocator, self.rows, other.cols);

        for (0..self.rows) |i| {
            for (0..other.cols) |j| {
                var sum: f64 = 0;
                for (0..self.cols) |k| {
                    sum += self.data[i * self.cols + k] * other.data[k * other.cols + j];
                }
                result.data[i * result.cols + j] = sum;
            }
        }

        return result;
    }

    /// Performs element-wise matrix addition: C = A + B
    /// Mathematical definition: C[i,j] = A[i,j] + B[i,j]
    /// Time complexity: O(rows * cols)
    /// Memory complexity: O(rows * cols)
    ///
    /// Parameters:
    ///   - other: Matrix to add element-wise
    ///   - allocator: Memory allocator for result matrix
    /// Returns: Result matrix with same dimensions as inputs
    pub fn add(self: Matrix, other: Matrix, allocator: Allocator) !Matrix {
        if (self.rows != other.rows or self.cols != other.cols) {
            return MatrixError.DimensionMismatch;
        }

        const result = try Matrix.init(allocator, self.rows, self.cols);
        for (result.data, self.data, other.data) |*output, left, right| {
            output.* = left + right;
        }
        return result;
    }

    /// Performs element-wise matrix subtraction: C = A - B
    /// Mathematical definition: C[i,j] = A[i,j] - B[i,j]
    /// Time complexity: O(rows * cols)
    /// Memory complexity: O(rows * cols)
    ///
    /// Parameters:
    ///   - other: Matrix to subtract element-wise
    ///   - allocator: Memory allocator for result matrix
    /// Returns: Result matrix with same dimensions as inputs
    pub fn subtract(self: Matrix, other: Matrix, allocator: Allocator) !Matrix {
        if (self.rows != other.rows or self.cols != other.cols) {
            return MatrixError.DimensionMismatch;
        }

        const result = try Matrix.init(allocator, self.rows, self.cols);
        for (result.data, self.data, other.data) |*output, left, right| {
            output.* = left - right;
        }
        return result;
    }

    /// Performs Hadamard (element-wise) multiplication: C = A ⊙ B
    /// Mathematical definition: C[i,j] = A[i,j] × B[i,j]
    /// Common in neural network gradient calculations
    /// Time complexity: O(rows * cols)
    /// Memory complexity: O(rows * cols)
    ///
    /// Parameters:
    ///   - other: Matrix to multiply element-wise
    ///   - allocator: Memory allocator for result matrix
    /// Returns: Result matrix with same dimensions as inputs
    pub fn elementWiseMultiply(self: Matrix, other: Matrix, allocator: Allocator) !Matrix {
        if (self.rows != other.rows or self.cols != other.cols) {
            return MatrixError.DimensionMismatch;
        }

        const result = try Matrix.init(allocator, self.rows, self.cols);
        for (result.data, self.data, other.data) |*output, left, right| {
            output.* = left * right;
        }
        return result;
    }

    /// Scales matrix by a scalar value: B = αA
    /// Mathematical definition: B[i,j] = α × A[i,j]
    /// Used in gradient descent and other optimization algorithms
    /// Time complexity: O(rows * cols)
    /// Memory complexity: O(rows * cols)
    ///
    /// Parameters:
    ///   - scalar: Value α to multiply each element by
    ///   - allocator: Memory allocator for result matrix
    /// Returns: Scaled matrix with same dimensions as input
    pub fn scale(self: Matrix, scalar: f64, allocator: Allocator) !Matrix {
        const result = try Matrix.init(allocator, self.rows, self.cols);
        for (result.data, self.data) |*output, value| {
            output.* = value * scalar;
        }
        return result;
    }

    /// Computes column-wise sum of matrix elements
    /// Mathematical definition: result[0,j] = Σᵢ A[i,j]
    /// Where i goes from 0 to rows-1
    /// Used in neural network bias gradient calculations
    /// Time complexity: O(rows * cols)
    /// Memory complexity: O(cols)
    ///
    /// Parameters:
    ///   - allocator: Memory allocator for result matrix
    /// Returns: 1×cols matrix containing column sums
    pub fn sumRows(self: Matrix, allocator: Allocator) !Matrix {
        const result = try Matrix.init(allocator, 1, self.cols);

        for (0..self.cols) |j| {
            var sum: f64 = 0;
            for (0..self.rows) |i| {
                sum += self.data[i * self.cols + j];
            }
            result.data[j] = sum;
        }

        return result;
    }

    /// Computes matrix transpose: B = Aᵀ
    /// Mathematical definition: B[j,i] = A[i,j]
    /// Essential operation in backpropagation
    /// Time complexity: O(rows * cols)
    /// Memory complexity: O(rows * cols)
    ///
    /// Parameters:
    ///   - allocator: Memory allocator for result matrix
    /// Returns: Transposed matrix with dimensions (cols × rows)
    pub fn transpose(self: Matrix, allocator: Allocator) !Matrix {
        const result = try Matrix.init(allocator, self.cols, self.rows);

        for (0..self.rows) |i| {
            for (0..self.cols) |j| {
                result.data[j * result.cols + i] = self.data[i * self.cols + j];
            }
        }

        return result;
    }

    /// Extracts a batch of rows from the matrix
    /// Used for mini-batch processing in machine learning
    /// Time complexity: O(batch_size * cols)
    /// Memory complexity: O(batch_size * cols)
    ///
    /// Parameters:
    ///   - start: Starting row index (inclusive)
    ///   - end: Ending row index (exclusive)
    ///   - allocator: Memory allocator for result matrix
    /// Returns: New matrix containing the specified rows
    pub fn extractBatch(self: Matrix, start: usize, end: usize, allocator: Allocator) !Matrix {
        if (start >= self.rows or end > self.rows or start >= end) {
            return MatrixError.InvalidBatchIndices;
        }

        const batch_size = end - start;
        const batch = try Matrix.init(allocator, batch_size, self.cols);
        const source_start = start * self.cols;
        @memcpy(batch.data, self.data[source_start..][0..batch.data.len]);
        return batch;
    }
};

test "matrix rejects overflowing dimensions" {
    try testing.expectError(
        MatrixError.DimensionOverflow,
        Matrix.init(testing.allocator, std.math.maxInt(usize), 2),
    );
}

// Tests
test "matrix basic operations" {
    const allocator = testing.allocator;

    // Test initialization
    var m = try Matrix.init(allocator, 2, 3);
    defer m.deinit();

    try testing.expectEqual(@as(usize, 2), m.rows);
    try testing.expectEqual(@as(usize, 3), m.cols);

    // Test set and get
    try m.set(0, 0, 1.0);
    try m.set(0, 1, 2.0);
    try m.set(1, 0, 3.0);

    try testing.expectEqual(@as(f64, 1.0), try m.get(0, 0));
    try testing.expectEqual(@as(f64, 2.0), try m.get(0, 1));
    try testing.expectEqual(@as(f64, 3.0), try m.get(1, 0));
}

test "matrix multiplication" {
    const allocator = testing.allocator;

    var m1 = try Matrix.init(allocator, 2, 3);
    defer m1.deinit();

    var m2 = try Matrix.init(allocator, 3, 2);
    defer m2.deinit();

    // Set test values
    try m1.set(0, 0, 1.0);
    try m1.set(0, 1, 2.0);
    try m1.set(0, 2, 3.0);
    try m1.set(1, 0, 4.0);
    try m1.set(1, 1, 5.0);
    try m1.set(1, 2, 6.0);

    try m2.set(0, 0, 7.0);
    try m2.set(0, 1, 8.0);
    try m2.set(1, 0, 9.0);
    try m2.set(1, 1, 10.0);
    try m2.set(2, 0, 11.0);
    try m2.set(2, 1, 12.0);

    var result = try m1.dotProduct(m2, allocator);
    defer result.deinit();

    // Expected results: [[58, 64], [139, 154]]
    try testing.expectEqual(@as(f64, 58.0), try result.get(0, 0));
    try testing.expectEqual(@as(f64, 64.0), try result.get(0, 1));
    try testing.expectEqual(@as(f64, 139.0), try result.get(1, 0));
    try testing.expectEqual(@as(f64, 154.0), try result.get(1, 1));
}

test "matrix copy" {
    const allocator = testing.allocator;

    var m1 = try Matrix.init(allocator, 2, 2);
    defer m1.deinit();

    try m1.set(0, 0, 1.0);
    try m1.set(0, 1, 2.0);
    try m1.set(1, 0, 3.0);
    try m1.set(1, 1, 4.0);

    var m2 = try Matrix.copy(m1, allocator);
    defer m2.deinit();

    // Check dimensions
    try testing.expectEqual(@as(usize, 2), m2.rows);
    try testing.expectEqual(@as(usize, 2), m2.cols);

    // Check values
    try testing.expectEqual(@as(f64, 1.0), try m2.get(0, 0));
    try testing.expectEqual(@as(f64, 2.0), try m2.get(0, 1));
    try testing.expectEqual(@as(f64, 3.0), try m2.get(1, 0));
    try testing.expectEqual(@as(f64, 4.0), try m2.get(1, 1));

    // Modify m1 and check that m2 is not affected
    try m1.set(0, 0, 5.0);
    try testing.expectEqual(@as(f64, 5.0), try m1.get(0, 0));
    try testing.expectEqual(@as(f64, 1.0), try m2.get(0, 0));
}

test "element-wise operations" {
    const allocator = testing.allocator;

    var m1 = try Matrix.init(allocator, 2, 2);
    defer m1.deinit();

    var m2 = try Matrix.init(allocator, 2, 2);
    defer m2.deinit();

    try m1.set(0, 0, 1.0);
    try m1.set(0, 1, 2.0);
    try m1.set(1, 0, 3.0);
    try m1.set(1, 1, 4.0);

    try m2.set(0, 0, 5.0);
    try m2.set(0, 1, 6.0);
    try m2.set(1, 0, 7.0);
    try m2.set(1, 1, 8.0);

    // Test addition
    var add_result = try m1.add(m2, allocator);
    defer add_result.deinit();

    try testing.expectEqual(@as(f64, 6.0), try add_result.get(0, 0));
    try testing.expectEqual(@as(f64, 8.0), try add_result.get(0, 1));
    try testing.expectEqual(@as(f64, 10.0), try add_result.get(1, 0));
    try testing.expectEqual(@as(f64, 12.0), try add_result.get(1, 1));

    // Test subtraction
    var sub_result = try m1.subtract(m2, allocator);
    defer sub_result.deinit();

    try testing.expectEqual(@as(f64, -4.0), try sub_result.get(0, 0));
    try testing.expectEqual(@as(f64, -4.0), try sub_result.get(0, 1));
    try testing.expectEqual(@as(f64, -4.0), try sub_result.get(1, 0));
    try testing.expectEqual(@as(f64, -4.0), try sub_result.get(1, 1));

    // Test element-wise multiplication
    var mul_result = try m1.elementWiseMultiply(m2, allocator);
    defer mul_result.deinit();

    try testing.expectEqual(@as(f64, 5.0), try mul_result.get(0, 0));
    try testing.expectEqual(@as(f64, 12.0), try mul_result.get(0, 1));
    try testing.expectEqual(@as(f64, 21.0), try mul_result.get(1, 0));
    try testing.expectEqual(@as(f64, 32.0), try mul_result.get(1, 1));
}

test "matrix scaling" {
    const allocator = testing.allocator;

    var m = try Matrix.init(allocator, 2, 2);
    defer m.deinit();

    try m.set(0, 0, 1.0);
    try m.set(0, 1, 2.0);
    try m.set(1, 0, 3.0);
    try m.set(1, 1, 4.0);

    var scaled = try m.scale(2.0, allocator);
    defer scaled.deinit();

    try testing.expectEqual(@as(f64, 2.0), try scaled.get(0, 0));
    try testing.expectEqual(@as(f64, 4.0), try scaled.get(0, 1));
    try testing.expectEqual(@as(f64, 6.0), try scaled.get(1, 0));
    try testing.expectEqual(@as(f64, 8.0), try scaled.get(1, 1));
}

test "matrix sum rows" {
    const allocator = testing.allocator;

    var m = try Matrix.init(allocator, 2, 3);
    defer m.deinit();

    try m.set(0, 0, 1.0);
    try m.set(0, 1, 2.0);
    try m.set(0, 2, 3.0);
    try m.set(1, 0, 4.0);
    try m.set(1, 1, 5.0);
    try m.set(1, 2, 6.0);

    var sum = try m.sumRows(allocator);
    defer sum.deinit();

    try testing.expectEqual(@as(usize, 1), sum.rows);
    try testing.expectEqual(@as(usize, 3), sum.cols);
    try testing.expectEqual(@as(f64, 5.0), try sum.get(0, 0));
    try testing.expectEqual(@as(f64, 7.0), try sum.get(0, 1));
    try testing.expectEqual(@as(f64, 9.0), try sum.get(0, 2));
}

test "matrix transpose" {
    const allocator = testing.allocator;

    var m = try Matrix.init(allocator, 2, 3);
    defer m.deinit();

    try m.set(0, 0, 1.0);
    try m.set(0, 1, 2.0);
    try m.set(0, 2, 3.0);
    try m.set(1, 0, 4.0);
    try m.set(1, 1, 5.0);
    try m.set(1, 2, 6.0);

    var transposed = try m.transpose(allocator);
    defer transposed.deinit();

    try testing.expectEqual(@as(usize, 3), transposed.rows);
    try testing.expectEqual(@as(usize, 2), transposed.cols);
    try testing.expectEqual(@as(f64, 1.0), try transposed.get(0, 0));
    try testing.expectEqual(@as(f64, 4.0), try transposed.get(0, 1));
    try testing.expectEqual(@as(f64, 2.0), try transposed.get(1, 0));
    try testing.expectEqual(@as(f64, 5.0), try transposed.get(1, 1));
    try testing.expectEqual(@as(f64, 3.0), try transposed.get(2, 0));
    try testing.expectEqual(@as(f64, 6.0), try transposed.get(2, 1));
}

test "matrix extract batch" {
    const allocator = testing.allocator;

    var m = try Matrix.init(allocator, 4, 2);
    defer m.deinit();

    // Set test values
    try m.set(0, 0, 1.0);
    try m.set(0, 1, 2.0);
    try m.set(1, 0, 3.0);
    try m.set(1, 1, 4.0);
    try m.set(2, 0, 5.0);
    try m.set(2, 1, 6.0);
    try m.set(3, 0, 7.0);
    try m.set(3, 1, 8.0);

    // Extract middle two rows
    var batch = try m.extractBatch(1, 3, allocator);
    defer batch.deinit();

    // Check dimensions
    try testing.expectEqual(@as(usize, 2), batch.rows);
    try testing.expectEqual(@as(usize, 2), batch.cols);

    // Check values
    try testing.expectEqual(@as(f64, 3.0), try batch.get(0, 0));
    try testing.expectEqual(@as(f64, 4.0), try batch.get(0, 1));
    try testing.expectEqual(@as(f64, 5.0), try batch.get(1, 0));
    try testing.expectEqual(@as(f64, 6.0), try batch.get(1, 1));
}

test "matrix fill" {
    const allocator = testing.allocator;

    var m = try Matrix.init(allocator, 2, 3);
    defer m.deinit();

    // Fill matrix with 42.0
    m.fill(42.0);

    // Check all elements are 42.0
    for (0..m.rows) |i| {
        for (0..m.cols) |j| {
            try testing.expectEqual(@as(f64, 42.0), try m.get(i, j));
        }
    }

    // Fill matrix with -1.0
    m.fill(-1.0);

    // Check all elements are -1.0
    for (0..m.rows) |i| {
        for (0..m.cols) |j| {
            try testing.expectEqual(@as(f64, -1.0), try m.get(i, j));
        }
    }
}

test "matrix invalid inputs return errors" {
    const allocator = testing.allocator;

    var m = try Matrix.init(allocator, 2, 2);
    defer m.deinit();

    var other = try Matrix.init(allocator, 3, 1);
    defer other.deinit();

    try testing.expectError(MatrixError.IndexOutOfBounds, m.get(2, 0));
    try testing.expectError(MatrixError.IndexOutOfBounds, m.set(0, 2, 1.0));
    try testing.expectError(MatrixError.DimensionMismatch, m.add(other, allocator));
    try testing.expectError(MatrixError.DimensionMismatch, m.dotProduct(other, allocator));
    try testing.expectError(MatrixError.InvalidBatchIndices, m.extractBatch(1, 1, allocator));
}
