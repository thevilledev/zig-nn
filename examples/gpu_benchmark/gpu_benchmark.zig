const std = @import("std");
const nn = @import("nn");

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

    var metal = try nn.createBackend(allocator, .Metal);
    defer metal.deinit();

    if (metal.getBackendType() != .Metal) {
        std.debug.print(
            "Metal backend is unavailable. Run on macOS with -Dgpu=metal.\n",
            .{},
        );
        return;
    }

    std.debug.print("GPU benchmark: square matrix multiplication\n", .{});
    std.debug.print("Build tip: use -Doptimize=ReleaseFast for timing.\n\n", .{});
    std.debug.print(
        "{s: >6} {s: >6} {s: >12} {s: >12} {s: >10} {s: >12}\n",
        .{ "size", "runs", "cpu ms", "metal ms", "speedup", "sample err" },
    );

    for (cases) |case| {
        const result = try benchmarkCase(allocator, cpu, metal, case);
        std.debug.print(
            "{d: >6} {d: >6} {d: >12.3} {d: >12.3} {d: >9.2}x {d: >12.6}\n",
            .{
                case.size,
                case.trials,
                result.cpu.average_ns / 1_000_000.0,
                result.metal.average_ns / 1_000_000.0,
                result.cpu.average_ns / result.metal.average_ns,
                result.metal.sample_error,
            },
        );
    }
}

const CaseComparison = struct {
    cpu: BenchmarkResult,
    metal: BenchmarkResult,
};

fn benchmarkCase(
    allocator: std.mem.Allocator,
    cpu: BackendInstance,
    metal: BackendInstance,
    case: BenchmarkCase,
) !CaseComparison {
    var cpu_a = try BackendMatrix.init(cpu, allocator, case.size, case.size);
    defer cpu_a.deinit();
    var cpu_b = try BackendMatrix.init(cpu, allocator, case.size, case.size);
    defer cpu_b.deinit();
    var metal_a = try BackendMatrix.init(metal, allocator, case.size, case.size);
    defer metal_a.deinit();
    var metal_b = try BackendMatrix.init(metal, allocator, case.size, case.size);
    defer metal_b.deinit();

    fillInputs(cpu_a, cpu_b, metal_a, metal_b);

    const cpu_result = try timeDotProduct(allocator, cpu_a, cpu_b, case.trials);
    const metal_result = try timeDotProduct(allocator, metal_a, metal_b, case.trials);

    const cpu_reference = try cpu_a.dotProduct(cpu_b, allocator);
    defer cpu_reference.deinit();
    const metal_reference = try metal_a.dotProduct(metal_b, allocator);
    defer metal_reference.deinit();

    return .{
        .cpu = .{
            .average_ns = cpu_result,
            .sample_error = 0.0,
        },
        .metal = .{
            .average_ns = metal_result,
            .sample_error = sampleMaxAbsDiff(cpu_reference, metal_reference),
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
    metal_a: *BackendMatrix,
    metal_b: *BackendMatrix,
) void {
    for (0..cpu_a.rows) |row| {
        for (0..cpu_a.cols) |col| {
            const a_value = deterministicValue(row, col, 31, 17);
            const b_value = deterministicValue(row, col, 13, 29);
            cpu_a.set(row, col, a_value);
            cpu_b.set(row, col, b_value);
            metal_a.set(row, col, a_value);
            metal_b.set(row, col, b_value);
        }
    }
}

fn deterministicValue(row: usize, col: usize, row_mul: usize, col_mul: usize) f64 {
    const mixed = (row * row_mul + col * col_mul + 7) % 997;
    return @as(f64, @floatFromInt(mixed)) / 997.0 - 0.5;
}

fn sampleMaxAbsDiff(cpu: *const BackendMatrix, metal: *const BackendMatrix) f64 {
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
        const diff = @abs(cpu.get(row, col) - metal.get(row, col));
        max_abs = @max(max_abs, diff);
    }

    return max_abs;
}
