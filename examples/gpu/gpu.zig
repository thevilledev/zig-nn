const std = @import("std");
const nn = @import("nn");
const Matrix = nn.BackendMatrix;
const BackendType = nn.BackendType;
const BackendInstance = nn.BackendInstance;

// TODO: Currently only supports CPU backend until Metal backend is fully implemented
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    // Always use CPU backend for now until Metal backend is fully implemented
    std.debug.print("Creating CPU backend...\n", .{});
    const cpu_ptr = try nn.createCPUBackend(allocator);
    var backend_instance = BackendInstance{ .CPU = cpu_ptr };
    defer backend_instance.deinit();

    // Print the backend type
    std.debug.print("Using backend: {s}\n", .{@tagName(backend_instance.getBackendType())});

    // Create matrices using the BackendInstance (smaller for faster testing)
    var matrix_a = try Matrix.init(backend_instance, allocator, 100, 100);
    var matrix_b = try Matrix.init(backend_instance, allocator, 100, 100);

    // Initialize matrices with random values
    matrix_a.randomize(-1.0, 1.0);
    matrix_b.randomize(-1.0, 1.0);

    // Create a timer to measure performance
    var timer = try std.time.Timer.start();

    // Perform matrix operations directly on Matrix instances
    std.debug.print("Performing matrix operations...\n", .{});

    // Matrix multiplication
    timer.reset();
    var matrix_mul = try matrix_a.dotProduct(matrix_b, allocator);
    const mul_time = timer.lap();

    // Matrix addition
    timer.reset();
    var matrix_add = try matrix_a.add(matrix_b, allocator);
    const add_time = timer.lap();

    // Element-wise multiplication
    timer.reset();
    var matrix_element_mul = try matrix_a.elementWiseMultiply(matrix_b, allocator);
    const element_mul_time = timer.lap();

    // Matrix transpose
    timer.reset();
    var matrix_transpose = try matrix_a.transpose(allocator);
    const transpose_time = timer.lap();

    // Display results
    std.debug.print("\nPerformance results:\n", .{});
    std.debug.print("Matrix multiplication: {d:.2} ms\n", .{@as(f64, @floatFromInt(mul_time)) / 1_000_000.0});
    std.debug.print("Matrix addition: {d:.2} ms\n", .{@as(f64, @floatFromInt(add_time)) / 1_000_000.0});
    std.debug.print("Element-wise multiplication: {d:.2} ms\n", .{@as(f64, @floatFromInt(element_mul_time)) / 1_000_000.0});
    std.debug.print("Matrix transpose: {d:.2} ms\n", .{@as(f64, @floatFromInt(transpose_time)) / 1_000_000.0});

    // Clean up all matrices manually (in reverse order)
    matrix_transpose.deinit();
    matrix_element_mul.deinit();
    matrix_add.deinit();
    matrix_mul.deinit();
    matrix_b.deinit();
    matrix_a.deinit();

    std.debug.print("\nCPU backend demo completed successfully!\n", .{});
    return;
}
