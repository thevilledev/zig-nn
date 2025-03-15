const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;

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
        const data = try allocator.alloc(f64, rows * cols);
        // Initialize with zeros
        @memset(data, 0);
        return Matrix{
            .rows = rows,
            .cols = cols,
            .data = data,
            .allocator = allocator,
        };
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
        var result = try Matrix.init(allocator, source.rows, source.cols);

        for (0..source.rows) |i| {
            for (0..source.cols) |j| {
                result.set(i, j, source.get(i, j));
            }
        }

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
    /// Panics if indices are out of bounds
    pub fn get(self: Matrix, row: usize, col: usize) f64 {
        if (row >= self.rows or col >= self.cols) {
            @panic("Index out of bounds");
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
    /// Panics if indices are out of bounds
    pub fn set(self: *Matrix, row: usize, col: usize, value: f64) void {
        if (row >= self.rows or col >= self.cols) {
            @panic("Index out of bounds");
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
        var prng = std.Random.DefaultPrng.init(@intCast(std.time.milliTimestamp()));
        const rand = prng.random();

        for (0..self.rows) |i| {
            for (0..self.cols) |j| {
                const rand_float = rand.float(f64);
                const value = min + (rand_float * (max - min));
                self.set(i, j, value);
            }
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
    /// Panics if matrix dimensions don't match (self.cols ≠ other.rows)
    pub fn dotProduct(self: Matrix, other: Matrix, allocator: Allocator) !Matrix {
        if (self.cols != other.rows) {
            @panic("Invalid matrix dimensions for multiplication");
        }

        var result = try Matrix.init(allocator, self.rows, other.cols);

        for (0..self.rows) |i| {
            for (0..other.cols) |j| {
                var sum: f64 = 0;
                for (0..self.cols) |k| {
                    sum += self.get(i, k) * other.get(k, j);
                }
                result.set(i, j, sum);
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
    /// Panics if matrix dimensions don't match
    pub fn add(self: Matrix, other: Matrix, allocator: Allocator) !Matrix {
        if (self.rows != other.rows or self.cols != other.cols) {
            @panic("Matrix dimensions must match for addition");
        }

        var result = try Matrix.init(allocator, self.rows, self.cols);

        for (0..self.rows) |i| {
            for (0..self.cols) |j| {
                const sum = self.get(i, j) + other.get(i, j);
                result.set(i, j, sum);
            }
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
    /// Panics if matrix dimensions don't match
    pub fn subtract(self: Matrix, other: Matrix, allocator: Allocator) !Matrix {
        if (self.rows != other.rows or self.cols != other.cols) {
            @panic("Matrix dimensions must match for subtraction");
        }

        var result = try Matrix.init(allocator, self.rows, self.cols);

        for (0..self.rows) |i| {
            for (0..self.cols) |j| {
                const diff = self.get(i, j) - other.get(i, j);
                result.set(i, j, diff);
            }
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
    /// Panics if matrix dimensions don't match
    pub fn elementWiseMultiply(self: Matrix, other: Matrix, allocator: Allocator) !Matrix {
        if (self.rows != other.rows or self.cols != other.cols) {
            @panic("Matrix dimensions must match for element-wise multiplication");
        }

        var result = try Matrix.init(allocator, self.rows, self.cols);

        for (0..self.rows) |i| {
            for (0..self.cols) |j| {
                const product = self.get(i, j) * other.get(i, j);
                result.set(i, j, product);
            }
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
        var result = try Matrix.init(allocator, self.rows, self.cols);

        for (0..self.rows) |i| {
            for (0..self.cols) |j| {
                const scaled_value = self.get(i, j) * scalar;
                result.set(i, j, scaled_value);
            }
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
        var result = try Matrix.init(allocator, 1, self.cols);

        for (0..self.cols) |j| {
            var sum: f64 = 0;
            for (0..self.rows) |i| {
                sum += self.get(i, j);
            }
            result.set(0, j, sum);
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
        var result = try Matrix.init(allocator, self.cols, self.rows);

        for (0..self.rows) |i| {
            for (0..self.cols) |j| {
                result.set(j, i, self.get(i, j));
            }
        }

        return result;
    }
};

// Tests
test "matrix basic operations" {
    const allocator = testing.allocator;

    // Test initialization
    var m = try Matrix.init(allocator, 2, 3);
    defer m.deinit();

    try testing.expectEqual(@as(usize, 2), m.rows);
    try testing.expectEqual(@as(usize, 3), m.cols);

    // Test set and get
    m.set(0, 0, 1.0);
    m.set(0, 1, 2.0);
    m.set(1, 0, 3.0);

    try testing.expectEqual(@as(f64, 1.0), m.get(0, 0));
    try testing.expectEqual(@as(f64, 2.0), m.get(0, 1));
    try testing.expectEqual(@as(f64, 3.0), m.get(1, 0));
}

test "matrix multiplication" {
    const allocator = testing.allocator;

    var m1 = try Matrix.init(allocator, 2, 3);
    defer m1.deinit();

    var m2 = try Matrix.init(allocator, 3, 2);
    defer m2.deinit();

    // Set test values
    m1.set(0, 0, 1.0);
    m1.set(0, 1, 2.0);
    m1.set(0, 2, 3.0);
    m1.set(1, 0, 4.0);
    m1.set(1, 1, 5.0);
    m1.set(1, 2, 6.0);

    m2.set(0, 0, 7.0);
    m2.set(0, 1, 8.0);
    m2.set(1, 0, 9.0);
    m2.set(1, 1, 10.0);
    m2.set(2, 0, 11.0);
    m2.set(2, 1, 12.0);

    var result = try m1.dotProduct(m2, allocator);
    defer result.deinit();

    // Expected results: [[58, 64], [139, 154]]
    try testing.expectEqual(@as(f64, 58.0), result.get(0, 0));
    try testing.expectEqual(@as(f64, 64.0), result.get(0, 1));
    try testing.expectEqual(@as(f64, 139.0), result.get(1, 0));
    try testing.expectEqual(@as(f64, 154.0), result.get(1, 1));
}

test "matrix copy" {
    const allocator = testing.allocator;

    var m1 = try Matrix.init(allocator, 2, 2);
    defer m1.deinit();

    m1.set(0, 0, 1.0);
    m1.set(0, 1, 2.0);
    m1.set(1, 0, 3.0);
    m1.set(1, 1, 4.0);

    var m2 = try Matrix.copy(m1, allocator);
    defer m2.deinit();

    // Check dimensions
    try testing.expectEqual(@as(usize, 2), m2.rows);
    try testing.expectEqual(@as(usize, 2), m2.cols);

    // Check values
    try testing.expectEqual(@as(f64, 1.0), m2.get(0, 0));
    try testing.expectEqual(@as(f64, 2.0), m2.get(0, 1));
    try testing.expectEqual(@as(f64, 3.0), m2.get(1, 0));
    try testing.expectEqual(@as(f64, 4.0), m2.get(1, 1));

    // Modify m1 and check that m2 is not affected
    m1.set(0, 0, 5.0);
    try testing.expectEqual(@as(f64, 5.0), m1.get(0, 0));
    try testing.expectEqual(@as(f64, 1.0), m2.get(0, 0));
}

test "element-wise operations" {
    const allocator = testing.allocator;

    var m1 = try Matrix.init(allocator, 2, 2);
    defer m1.deinit();

    var m2 = try Matrix.init(allocator, 2, 2);
    defer m2.deinit();

    m1.set(0, 0, 1.0);
    m1.set(0, 1, 2.0);
    m1.set(1, 0, 3.0);
    m1.set(1, 1, 4.0);

    m2.set(0, 0, 5.0);
    m2.set(0, 1, 6.0);
    m2.set(1, 0, 7.0);
    m2.set(1, 1, 8.0);

    // Test addition
    var add_result = try m1.add(m2, allocator);
    defer add_result.deinit();

    try testing.expectEqual(@as(f64, 6.0), add_result.get(0, 0));
    try testing.expectEqual(@as(f64, 8.0), add_result.get(0, 1));
    try testing.expectEqual(@as(f64, 10.0), add_result.get(1, 0));
    try testing.expectEqual(@as(f64, 12.0), add_result.get(1, 1));

    // Test subtraction
    var sub_result = try m1.subtract(m2, allocator);
    defer sub_result.deinit();

    try testing.expectEqual(@as(f64, -4.0), sub_result.get(0, 0));
    try testing.expectEqual(@as(f64, -4.0), sub_result.get(0, 1));
    try testing.expectEqual(@as(f64, -4.0), sub_result.get(1, 0));
    try testing.expectEqual(@as(f64, -4.0), sub_result.get(1, 1));

    // Test element-wise multiplication
    var mul_result = try m1.elementWiseMultiply(m2, allocator);
    defer mul_result.deinit();

    try testing.expectEqual(@as(f64, 5.0), mul_result.get(0, 0));
    try testing.expectEqual(@as(f64, 12.0), mul_result.get(0, 1));
    try testing.expectEqual(@as(f64, 21.0), mul_result.get(1, 0));
    try testing.expectEqual(@as(f64, 32.0), mul_result.get(1, 1));
}

test "matrix scaling" {
    const allocator = testing.allocator;

    var m = try Matrix.init(allocator, 2, 2);
    defer m.deinit();

    m.set(0, 0, 1.0);
    m.set(0, 1, 2.0);
    m.set(1, 0, 3.0);
    m.set(1, 1, 4.0);

    var scaled = try m.scale(2.0, allocator);
    defer scaled.deinit();

    try testing.expectEqual(@as(f64, 2.0), scaled.get(0, 0));
    try testing.expectEqual(@as(f64, 4.0), scaled.get(0, 1));
    try testing.expectEqual(@as(f64, 6.0), scaled.get(1, 0));
    try testing.expectEqual(@as(f64, 8.0), scaled.get(1, 1));
}

test "matrix sum rows" {
    const allocator = testing.allocator;

    var m = try Matrix.init(allocator, 2, 3);
    defer m.deinit();

    m.set(0, 0, 1.0);
    m.set(0, 1, 2.0);
    m.set(0, 2, 3.0);
    m.set(1, 0, 4.0);
    m.set(1, 1, 5.0);
    m.set(1, 2, 6.0);

    var sum = try m.sumRows(allocator);
    defer sum.deinit();

    try testing.expectEqual(@as(usize, 1), sum.rows);
    try testing.expectEqual(@as(usize, 3), sum.cols);
    try testing.expectEqual(@as(f64, 5.0), sum.get(0, 0));
    try testing.expectEqual(@as(f64, 7.0), sum.get(0, 1));
    try testing.expectEqual(@as(f64, 9.0), sum.get(0, 2));
}

test "matrix transpose" {
    const allocator = testing.allocator;

    var m = try Matrix.init(allocator, 2, 3);
    defer m.deinit();

    m.set(0, 0, 1.0);
    m.set(0, 1, 2.0);
    m.set(0, 2, 3.0);
    m.set(1, 0, 4.0);
    m.set(1, 1, 5.0);
    m.set(1, 2, 6.0);

    var transposed = try m.transpose(allocator);
    defer transposed.deinit();

    try testing.expectEqual(@as(usize, 3), transposed.rows);
    try testing.expectEqual(@as(usize, 2), transposed.cols);
    try testing.expectEqual(@as(f64, 1.0), transposed.get(0, 0));
    try testing.expectEqual(@as(f64, 4.0), transposed.get(0, 1));
    try testing.expectEqual(@as(f64, 2.0), transposed.get(1, 0));
    try testing.expectEqual(@as(f64, 5.0), transposed.get(1, 1));
    try testing.expectEqual(@as(f64, 3.0), transposed.get(2, 0));
    try testing.expectEqual(@as(f64, 6.0), transposed.get(2, 1));
}
