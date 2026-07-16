const std = @import("std");
const builtin = @import("builtin");
const nn = @import("nn");
const events = @import("experiment_events");

const BackendType = nn.BackendType;
const BackendInstance = nn.BackendInstance;
const BackendMatrix = nn.BackendMatrix;

const Options = struct {
    format: events.Format = .human,
};

const BenchmarkCase = struct {
    size: usize,
    trials: usize,
};

const RuntimeTelemetry = struct {
    execution: nn.ExecutionStats,
    backend: nn.BackendRuntimeStats,
};

const CaseResult = struct {
    size: usize,
    trials: usize,
    cpu_ms: f64,
    accelerator_ms: f64,
    speedup: f64,
    sample_error: f64,
    telemetry: RuntimeTelemetry,
};

const cases = [_]BenchmarkCase{
    .{ .size = 128, .trials = 5 },
    .{ .size = 256, .trials = 5 },
    .{ .size = 512, .trials = 3 },
};

fn nowNs() i96 {
    return std.Io.Clock.awake.now(std.Options.debug_io).toNanoseconds();
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const options = try parseArgs(args[1..]);
    const allocator = init.gpa;

    var stdout_buffer: [16 * 1024]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writerStreaming(std.Options.debug_io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};

    var cpu = try nn.createBackend(allocator, .CPU);
    defer cpu.deinit();

    const requested_backend: BackendType = if (nn.enable_cuda) .CUDA else if (nn.enable_rocm) .ROCm else .Metal;
    var accelerator = try nn.createBackend(allocator, requested_backend);
    defer accelerator.deinit();
    if (accelerator.getBackendType() != requested_backend) return error.BackendUnavailable;
    const accelerator_name = backendName(requested_backend);

    if (options.format == .ndjson) {
        try events.emit(stdout, .{
            .v = 1,
            .type = "run_started",
            .experiment = "gpu-benchmark",
            .data = .{
                .config = .{ .cases = cases.len },
                .topology = &[_]usize{},
                .activations = &[_][]const u8{},
                .case_sizes = &[_]usize{ 128, 256, 512 },
                .execution = .{
                    .requested_backend = accelerator_name,
                    .selected_backend = accelerator_name,
                    .optimize = @tagName(builtin.mode),
                },
            },
        });
    } else {
        try stdout.print("GPU benchmark: square matrix multiplication\n", .{});
        try stdout.print("Build tip: use -Doptimize=ReleaseFast for timing.\n\n", .{});
        try stdout.print(
            "{s: >6} {s: >6} {s: >12} {s: >12} {s: >10} {s: >12}\n",
            .{ "size", "runs", "cpu ms", "gpu ms", "speedup", "sample err" },
        );
    }

    var results: [cases.len]CaseResult = undefined;
    for (cases, 0..) |case, index| {
        results[index] = try benchmarkCase(allocator, cpu, accelerator, case);
        if (options.format == .ndjson) {
            try events.emit(stdout, .{
                .v = 1,
                .type = "snapshot",
                .experiment = "gpu-benchmark",
                .step = index + 1,
                .total_steps = cases.len,
                .data = .{
                    .kind = "backend_benchmark",
                    .cases = results[0 .. index + 1],
                    .telemetry = results[index].telemetry,
                },
            });
        } else {
            const result = results[index];
            try stdout.print(
                "{d: >6} {d: >6} {d: >12.3} {d: >12.3} {d: >9.2}x {d: >12.6}\n",
                .{
                    result.size,
                    result.trials,
                    result.cpu_ms,
                    result.accelerator_ms,
                    result.speedup,
                    result.sample_error,
                },
            );
        }
    }

    if (options.format == .ndjson) {
        try events.emit(stdout, .{
            .v = 1,
            .type = "run_completed",
            .experiment = "gpu-benchmark",
            .step = cases.len,
            .total_steps = cases.len,
            .data = .{ .cases = results },
        });
    }
}

fn parseArgs(args: []const []const u8) !Options {
    var options: Options = .{};
    var index: usize = 0;
    while (index < args.len) : (index += 1) {
        if (std.mem.eql(u8, args[index], "--format")) {
            index += 1;
            if (index >= args.len) return error.MissingArgument;
            options.format = try events.parseFormat(args[index]);
        } else {
            return error.UnknownArgument;
        }
    }
    return options;
}

fn benchmarkCase(
    allocator: std.mem.Allocator,
    cpu: BackendInstance,
    accelerator: BackendInstance,
    case: BenchmarkCase,
) !CaseResult {
    var cpu_a = try BackendMatrix.init(cpu, allocator, case.size, case.size);
    defer cpu_a.deinit();
    var cpu_b = try BackendMatrix.init(cpu, allocator, case.size, case.size);
    defer cpu_b.deinit();
    var accelerator_a = try BackendMatrix.init(accelerator, allocator, case.size, case.size);
    defer accelerator_a.deinit();
    var accelerator_b = try BackendMatrix.init(accelerator, allocator, case.size, case.size);
    defer accelerator_b.deinit();

    try fillInputs(allocator, cpu_a, cpu_b, accelerator_a, accelerator_b);
    cpu.resetRuntimeStats();
    accelerator.resetRuntimeStats();

    const cpu_ns = try medianDotProduct(allocator, cpu, cpu_a, cpu_b, case.trials);
    const accelerator_ns = try medianDotProduct(allocator, accelerator, accelerator_a, accelerator_b, case.trials);
    const runtime_stats = accelerator.runtimeStats();

    const cpu_reference = try cpu_a.dotProduct(cpu_b, allocator);
    defer cpu_reference.deinit();
    const accelerator_reference = try accelerator_a.dotProduct(accelerator_b, allocator);
    defer accelerator_reference.deinit();
    try cpu.synchronize();
    try accelerator.synchronize();

    return .{
        .size = case.size,
        .trials = case.trials,
        .cpu_ms = cpu_ns / 1_000_000.0,
        .accelerator_ms = accelerator_ns / 1_000_000.0,
        .speedup = cpu_ns / accelerator_ns,
        .sample_error = sampleMaxAbsDiff(cpu_reference, accelerator_reference),
        .telemetry = telemetryFrom(runtime_stats),
    };
}

fn medianDotProduct(
    allocator: std.mem.Allocator,
    backend: BackendInstance,
    a: *const BackendMatrix,
    b: *const BackendMatrix,
    trials: usize,
) !f64 {
    var warmup = try a.dotProduct(b, allocator);
    try backend.synchronize();
    warmup.deinit();

    const timings = try allocator.alloc(f64, trials);
    defer allocator.free(timings);
    for (timings) |*timing| {
        const start = nowNs();
        var result = try a.dotProduct(b, allocator);
        try backend.synchronize();
        timing.* = @floatFromInt(nowNs() - start);
        result.deinit();
    }
    std.mem.sort(f64, timings, {}, comptime std.sort.asc(f64));
    const middle = trials / 2;
    if (trials % 2 == 1) return timings[middle];
    return (timings[middle - 1] + timings[middle]) / 2;
}

fn fillInputs(
    allocator: std.mem.Allocator,
    cpu_a: *BackendMatrix,
    cpu_b: *BackendMatrix,
    accelerator_a: *BackendMatrix,
    accelerator_b: *BackendMatrix,
) !void {
    const count = cpu_a.rows * cpu_a.cols;
    const a_values = try allocator.alloc(f32, count);
    defer allocator.free(a_values);
    const b_values = try allocator.alloc(f32, count);
    defer allocator.free(b_values);
    for (0..cpu_a.rows) |row| {
        for (0..cpu_a.cols) |column| {
            const index = row * cpu_a.cols + column;
            a_values[index] = @floatCast(deterministicValue(row, column, 31, 17));
            b_values[index] = @floatCast(deterministicValue(row, column, 13, 29));
        }
    }
    try cpu_a.writeF32(a_values);
    try cpu_b.writeF32(b_values);
    try accelerator_a.writeF32(a_values);
    try accelerator_b.writeF32(b_values);
}

fn deterministicValue(row: usize, column: usize, row_mul: usize, column_mul: usize) f64 {
    const mixed = (row * row_mul + column * column_mul + 7) % 997;
    return @as(f64, @floatFromInt(mixed)) / 997.0 - 0.5;
}

fn sampleMaxAbsDiff(cpu: *const BackendMatrix, accelerator: *const BackendMatrix) f64 {
    var max_abs: f64 = 0.0;
    const last_row = cpu.rows - 1;
    const last_column = cpu.cols - 1;
    const samples = [_][2]usize{
        .{ 0, 0 },
        .{ 0, last_column },
        .{ last_row, 0 },
        .{ last_row, last_column },
        .{ cpu.rows / 2, cpu.cols / 2 },
        .{ cpu.rows / 3, cpu.cols / 3 },
        .{ (cpu.rows * 2) / 3, (cpu.cols * 2) / 3 },
    };
    for (samples) |sample| {
        const difference = @abs(cpu.get(sample[0], sample[1]) - accelerator.get(sample[0], sample[1]));
        max_abs = @max(max_abs, difference);
    }
    return max_abs;
}

fn telemetryFrom(stats: nn.BackendRuntimeStats) RuntimeTelemetry {
    return .{
        .execution = .{
            .uploads = stats.host_to_device_transfers,
            .upload_bytes = stats.host_to_device_bytes,
            .readbacks = stats.device_to_host_transfers,
            .readback_bytes = stats.device_to_host_bytes,
            .kernels = stats.kernel_launches + stats.vendor_gemm_launches,
            .synchronizations = stats.synchronizations,
        },
        .backend = stats,
    };
}

fn backendName(backend: BackendType) []const u8 {
    return switch (backend) {
        .CPU => "cpu",
        .Metal => "metal",
        .CUDA => "cuda",
        .ROCm => "rocm",
    };
}

test "gpu benchmark accepts only the structured output flag" {
    const options = try parseArgs(&.{ "--format", "ndjson" });
    try std.testing.expectEqual(events.Format.ndjson, options.format);
    try std.testing.expectError(error.UnknownArgument, parseArgs(&.{"--trials"}));
}

test "gpu benchmark inputs are deterministic" {
    try std.testing.expectEqual(deterministicValue(3, 7, 31, 17), deterministicValue(3, 7, 31, 17));
    try std.testing.expect(deterministicValue(3, 7, 31, 17) != deterministicValue(4, 7, 31, 17));
}
