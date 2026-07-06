const std = @import("std");
const nn = @import("nn");

const BackendType = nn.BackendType;
const BackendInstance = nn.BackendInstance;
const BackendMatrix = nn.BackendMatrix;

const BenchmarkCase = struct {
    size: usize,
    trials: usize,
};

const BenchmarkResult = struct {
    average_ns: f64,
    sample_error: f64,
};

const cases = [_]BenchmarkCase{
    .{ .size = 128, .trials = 3 },
    .{ .size = 256, .trials = 3 },
    .{ .size = 512, .trials = 2 },
};

fn nowNs() i96 {
    return std.Io.Clock.awake.now(std.Options.debug_io).toNanoseconds();
}

pub fn main() !void {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var cpu = try nn.createBackend(allocator, .CPU);
    defer cpu.deinit();

    const requested_backend: BackendType = if (nn.enable_cuda) .CUDA else .Metal;
    const gpu_name = if (requested_backend == .CUDA) "cuda" else "metal";
    var gpu = try nn.createBackend(allocator, requested_backend);
    defer gpu.deinit();

    if (gpu.getBackendType() != requested_backend) {
        std.debug.print(
            "{s} backend is unavailable. Run with matching -Dgpu={s} support.\n",
            .{ gpu_name, gpu_name },
        );
        return;
    }

    std.debug.print("GPU benchmark: square matrix multiplication\n", .{});
    std.debug.print("Build tip: use -Doptimize=ReleaseFast for timing.\n\n", .{});
    std.debug.print(
        "{s: >6} {s: >6} {s: >12} {s: >12} {s: >10} {s: >12}\n",
        .{ "size", "runs", "cpu ms", "gpu ms", "speedup", "sample err" },
    );

    for (cases) |case| {
        const result = try benchmarkCase(allocator, cpu, gpu, case);
        std.debug.print(
            "{d: >6} {d: >6} {d: >12.3} {d: >12.3} {d: >9.2}x {d: >12.6} ({s})\n",
            .{
                case.size,
                case.trials,
                result.cpu.average_ns / 1_000_000.0,
                result.gpu.average_ns / 1_000_000.0,
                result.cpu.average_ns / result.gpu.average_ns,
                result.gpu.sample_error,
                gpu_name,
            },
        );
    }
}

const CaseComparison = struct {
    cpu: BenchmarkResult,
    gpu: BenchmarkResult,
};

fn benchmarkCase(
    allocator: std.mem.Allocator,
    cpu: BackendInstance,
    gpu: BackendInstance,
    case: BenchmarkCase,
) !CaseComparison {
    var cpu_a = try BackendMatrix.init(cpu, allocator, case.size, case.size);
    defer cpu_a.deinit();
    var cpu_b = try BackendMatrix.init(cpu, allocator, case.size, case.size);
    defer cpu_b.deinit();
    var gpu_a = try BackendMatrix.init(gpu, allocator, case.size, case.size);
    defer gpu_a.deinit();
    var gpu_b = try BackendMatrix.init(gpu, allocator, case.size, case.size);
    defer gpu_b.deinit();

    fillInputs(cpu_a, cpu_b, gpu_a, gpu_b);

    const cpu_result = try timeDotProduct(allocator, cpu_a, cpu_b, case.trials);
    const gpu_result = try timeDotProduct(allocator, gpu_a, gpu_b, case.trials);

    const cpu_reference = try cpu_a.dotProduct(cpu_b, allocator);
    defer cpu_reference.deinit();
    const gpu_reference = try gpu_a.dotProduct(gpu_b, allocator);
    defer gpu_reference.deinit();

    return .{
        .cpu = .{
            .average_ns = cpu_result,
            .sample_error = 0.0,
        },
        .gpu = .{
            .average_ns = gpu_result,
            .sample_error = sampleMaxAbsDiff(cpu_reference, gpu_reference),
        },
    };
}

fn timeDotProduct(
    allocator: std.mem.Allocator,
    a: *const BackendMatrix,
    b: *const BackendMatrix,
    trials: usize,
) !f64 {
    var total_ns: f64 = 0.0;

    for (0..trials) |_| {
        const start = nowNs();
        const result = try a.dotProduct(b, allocator);
        const elapsed = nowNs() - start;
        result.deinit();
        total_ns += @as(f64, @floatFromInt(elapsed));
    }

    return total_ns / @as(f64, @floatFromInt(trials));
}

fn fillInputs(
    cpu_a: *BackendMatrix,
    cpu_b: *BackendMatrix,
    gpu_a: *BackendMatrix,
    gpu_b: *BackendMatrix,
) void {
    for (0..cpu_a.rows) |row| {
        for (0..cpu_a.cols) |col| {
            const a_value = deterministicValue(row, col, 31, 17);
            const b_value = deterministicValue(row, col, 13, 29);
            cpu_a.set(row, col, a_value);
            cpu_b.set(row, col, b_value);
            gpu_a.set(row, col, a_value);
            gpu_b.set(row, col, b_value);
        }
    }
}

fn deterministicValue(row: usize, col: usize, row_mul: usize, col_mul: usize) f64 {
    const mixed = (row * row_mul + col * col_mul + 7) % 997;
    return @as(f64, @floatFromInt(mixed)) / 997.0 - 0.5;
}

fn sampleMaxAbsDiff(cpu: *const BackendMatrix, gpu: *const BackendMatrix) f64 {
    var max_abs: f64 = 0.0;
    const last_row = cpu.rows - 1;
    const last_col = cpu.cols - 1;
    const samples = [_][2]usize{
        .{ 0, 0 },
        .{ 0, last_col },
        .{ last_row, 0 },
        .{ last_row, last_col },
        .{ cpu.rows / 2, cpu.cols / 2 },
        .{ cpu.rows / 3, cpu.cols / 3 },
        .{ (cpu.rows * 2) / 3, (cpu.cols * 2) / 3 },
    };

    for (samples) |sample| {
        const row = sample[0];
        const col = sample[1];
        const diff = @abs(cpu.get(row, col) - gpu.get(row, col));
        max_abs = @max(max_abs, diff);
    }

    return max_abs;
}
