const std = @import("std");
const Matrix = @import("nn").Matrix;

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Create matrices directly using the Matrix struct
    var a = try Matrix.init(allocator, 2, 2);
    defer a.deinit();

    var b = try Matrix.init(allocator, 2, 2);
    defer b.deinit();

    // Initialize matrices
    try a.set(0, 0, 1.0);
    try a.set(0, 1, 2.0);
    try a.set(1, 0, 3.0);
    try a.set(1, 1, 4.0);

    try b.set(0, 0, 5.0);
    try b.set(0, 1, 6.0);
    try b.set(1, 0, 7.0);
    try b.set(1, 1, 8.0);

    // Print matrices
    std.debug.print("Matrix A:\n", .{});
    try printMatrix(a);

    std.debug.print("Matrix B:\n", .{});
    try printMatrix(b);

    // Perform matrix multiplication
    std.debug.print("A × B:\n", .{});
    var c = try a.dotProduct(b, allocator);
    defer c.deinit();
    try printMatrix(c);
}

/// Helper function to print a matrix
fn printMatrix(matrix: Matrix) !void {
    for (0..matrix.rows) |i| {
        for (0..matrix.cols) |j| {
            std.debug.print("{d:.1} ", .{try matrix.get(i, j)});
        }
        std.debug.print("\n", .{});
    }
}
