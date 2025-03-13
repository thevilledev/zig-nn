const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;

/// Matrix struct representing a 2D array of f64 values
pub const Matrix = struct {
    rows: usize,
    cols: usize,
    data: []f64,
    allocator: Allocator,

    /// Create a new matrix with specified dimensions
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

    /// Free the matrix memory
    pub fn deinit(self: Matrix) void {
        self.allocator.free(self.data);
    }

    /// Get value at specified position
    pub fn get(self: Matrix, row: usize, col: usize) f64 {
        if (row >= self.rows or col >= self.cols) {
            @panic("Index out of bounds");
        }
        return self.data[row * self.cols + col];
    }

    /// Set value at specified position
    pub fn set(self: *Matrix, row: usize, col: usize, value: f64) void {
        if (row >= self.rows or col >= self.cols) {
            @panic("Index out of bounds");
        }
        self.data[row * self.cols + col] = value;
    }

    /// Fill matrix with random values between min and max using std.Random
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

    /// Matrix multiplication
    pub fn multiply(self: Matrix, other: Matrix, allocator: Allocator) !Matrix {
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

    /// Element-wise addition
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

    /// Matrix transpose
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
    m1.set(0, 0, 1.0); m1.set(0, 1, 2.0); m1.set(0, 2, 3.0);
    m1.set(1, 0, 4.0); m1.set(1, 1, 5.0); m1.set(1, 2, 6.0);

    m2.set(0, 0, 7.0); m2.set(0, 1, 8.0);
    m2.set(1, 0, 9.0); m2.set(1, 1, 10.0);
    m2.set(2, 0, 11.0); m2.set(2, 1, 12.0);

    var result = try m1.multiply(m2, allocator);
    defer result.deinit();

    // Expected results: [[58, 64], [139, 154]]
    try testing.expectEqual(@as(f64, 58.0), result.get(0, 0));
    try testing.expectEqual(@as(f64, 64.0), result.get(0, 1));
    try testing.expectEqual(@as(f64, 139.0), result.get(1, 0));
    try testing.expectEqual(@as(f64, 154.0), result.get(1, 1));
} 