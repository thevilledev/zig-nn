const std = @import("std");
const nn = @import("nn");
const Matrix = nn.BackendMatrix;
const BackendType = nn.BackendType;

fn nowNs() i96 {
    return std.Io.Clock.awake.now(std.Options.debug_io).toNanoseconds();
}

pub fn main() !void {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    std.debug.print("Creating requested GPU backend...\n", .{});
    const requested_backend: BackendType = if (nn.enable_cuda) .CUDA else if (nn.enable_rocm) .ROCm else .Metal;
    var backend_instance = try nn.createBackend(allocator, requested_backend);
    defer backend_instance.deinit();

    // Print the backend type
    std.debug.print("Using backend: {s}\n", .{@tagName(backend_instance.getBackendType())});

    // Create matrices using the BackendInstance (smaller for faster testing)
    var matrix_a = try Matrix.init(backend_instance, allocator, 100, 100);
    var matrix_b = try Matrix.init(backend_instance, allocator, 100, 100);

    // Initialize matrices with random values
    matrix_a.randomize(-1.0, 1.0);
    matrix_b.randomize(-1.0, 1.0);

    // Perform matrix operations directly on Matrix instances
    std.debug.print("Performing matrix operations...\n", .{});

    // Matrix multiplication
    var start = nowNs();
    var matrix_mul = try matrix_a.dotProduct(matrix_b, allocator);
    const mul_time = nowNs() - start;

    // Matrix addition
    start = nowNs();
    var matrix_add = try matrix_a.add(matrix_b, allocator);
    const add_time = nowNs() - start;

    // Element-wise multiplication
    start = nowNs();
    var matrix_element_mul = try matrix_a.elementWiseMultiply(matrix_b, allocator);
    const element_mul_time = nowNs() - start;

    // Matrix transpose
    start = nowNs();
    var matrix_transpose = try matrix_a.transpose(allocator);
    const transpose_time = nowNs() - start;

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

    std.debug.print("\nGPU backend demo completed successfully!\n", .{});
    return;
}
