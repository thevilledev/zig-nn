const std = @import("std");
const builtin = @import("builtin");
const nn = @import("nn");
const tiny_gpt = @import("tiny_gpt");

const Allocator = std.mem.Allocator;
const BackendInstance = nn.BackendInstance;
const BackendMatrix = nn.BackendMatrix;
const Matrix = nn.Matrix;

const Options = struct {
    quick: bool = false,
    filter: ?[]const u8 = null,
};

const Timing = struct {
    warmups: usize,
    iterations: usize,
    average_ns: f64,
    min_ns: f64,
    max_ns: f64,
    checksum: f64,
};

const BackendOp = enum {
    relu,
    tanh,
    swish,
    softmax,
    glu,
    swiglu,
};

const QuantizationOp = enum {
    uniform,
    turbo,
};

fn nowNs() i96 {
    return std.Io.Clock.awake.now(std.Options.debug_io).toNanoseconds();
}

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.smp_allocator;

    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args.deinit();
    _ = args.skip();

    const options = try parseOptions(&args);
    if (options.filter) |filter| {
        if (filter.len == 0) return error.InvalidBenchmarkFilter;
    }

    var cpu = BackendInstance{ .CPU = try nn.createCPUBackend(allocator) };
    defer cpu.deinit();

    const metal = createMetalBackend(allocator);
    defer if (metal) |backend| backend.deinit();
    const cuda = createCUDABackend(allocator);
    defer if (cuda) |backend| backend.deinit();
    const rocm = createROCmBackend(allocator);
    defer if (rocm) |backend| backend.deinit();

    printHeader(options, metal != null, cuda != null, rocm != null);

    if (shouldRun(options, "matmul")) {
        try benchmarkMatmul(allocator, cpu, metal, cuda, rocm, options);
    }
    if (shouldRun(options, "activation")) {
        try benchmarkActivations(allocator, cpu, metal, cuda, rocm, options);
    }
    if (shouldRun(options, "gpu_heavy")) {
        try benchmarkGpuHeavy(allocator, metal, cuda, rocm, options);
    }
    if (shouldRunExplicit(options, "gpu_peak")) {
        try benchmarkGpuPeak(allocator, metal, cuda, rocm, options);
    }
    if (shouldRun(options, "layer_norm")) {
        try benchmarkLayerNorm(allocator, options);
    }
    if (shouldRun(options, "training")) {
        try benchmarkTraining(allocator, metal, cuda, rocm, options);
    }
    if (shouldRun(options, "tiny_gpt")) {
        try benchmarkTinyGpt(allocator, options);
    }
    if (shouldRun(options, "quantization")) {
        try benchmarkQuantization(allocator, options);
    }
}

fn parseOptions(args: *std.process.Args.Iterator) !Options {
    var options: Options = .{};
    while (args.next()) |arg_z| {
        const arg: []const u8 = arg_z;
        if (std.mem.eql(u8, arg, "--quick")) {
            options.quick = true;
        } else if (std.mem.eql(u8, arg, "--filter")) {
            const filter_z = args.next() orelse return error.MissingBenchmarkFilter;
            options.filter = filter_z;
        } else if (std.mem.startsWith(u8, arg, "--filter=")) {
            options.filter = arg["--filter=".len..];
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            printHelp();
            std.process.exit(0);
        } else {
            std.debug.print("unknown benchmark argument: {s}\n", .{arg});
            return error.InvalidBenchmarkArgument;
        }
    }
    return options;
}

fn printHelp() void {
    std.debug.print(
        \\Usage:
        \\  zig build benchmark [-- --quick] [-- --filter <suite>]
        \\
        \\Suites:
        \\  matmul        backend matrix multiplication
        \\  activation    backend activations and gated activations
        \\  gpu_heavy     GPU-only large matmul and activation workloads
        \\  gpu_peak      opt-in huge GEMM workloads for high-end GPUs
        \\  layer_norm    CPU LayerNorm forward passes
        \\  training      CPU Network.trainBatch loops
        \\  tiny_gpt      CPU TinyGPT forward passes
        \\  quantization  CPU uniform, TurboQuant, and KV-cache-shaped quantization loops
        \\
    , .{});
}

fn shouldRun(options: Options, suite: []const u8) bool {
    const filter = options.filter orelse return true;
    return std.mem.indexOf(u8, suite, filter) != null;
}

fn shouldRunExplicit(options: Options, suite: []const u8) bool {
    const filter = options.filter orelse return false;
    return std.mem.indexOf(u8, suite, filter) != null;
}

fn createMetalBackend(allocator: Allocator) ?BackendInstance {
    const ptr = nn.createMetalBackend(allocator) catch return null;
    return BackendInstance{ .Metal = ptr };
}

fn createCUDABackend(allocator: Allocator) ?BackendInstance {
    const ptr = nn.createCUDABackend(allocator) catch return null;
    return BackendInstance{ .CUDA = ptr };
}

fn createROCmBackend(allocator: Allocator) ?BackendInstance {
    const ptr = nn.createROCmBackend(allocator) catch return null;
    return BackendInstance{ .ROCm = ptr };
}

fn printHeader(options: Options, metal_available: bool, cuda_available: bool, rocm_available: bool) void {
    std.debug.print("# zig-nn benchmark suite\n", .{});
    std.debug.print("# mode={s} quick={} metal_available={} cuda_available={} rocm_available={}\n", .{ modeName(), options.quick, metal_available, cuda_available, rocm_available });
    std.debug.print("# repeatability: fixed shapes, fixed seeds, deterministic data, warmups excluded\n", .{});
    std.debug.print(
        "suite,case,backend,mode,warmups,iterations,avg_ns,min_ns,max_ns,checksum,sample_error,status\n",
        .{},
    );
}

fn modeName() []const u8 {
    return switch (builtin.mode) {
        .Debug => "Debug",
        .ReleaseSafe => "ReleaseSafe",
        .ReleaseFast => "ReleaseFast",
        .ReleaseSmall => "ReleaseSmall",
    };
}

fn benchmarkMatmul(
    allocator: Allocator,
    cpu: BackendInstance,
    metal: ?BackendInstance,
    cuda: ?BackendInstance,
    rocm: ?BackendInstance,
    options: Options,
) !void {
    const cases = [_]struct {
        m: usize,
        k: usize,
        n: usize,
        warmups: usize,
        iterations: usize,
    }{
        .{ .m = 64, .k = 64, .n = 64, .warmups = 1, .iterations = 8 },
        .{ .m = 128, .k = 128, .n = 128, .warmups = 1, .iterations = 5 },
        .{ .m = 256, .k = 256, .n = 256, .warmups = 1, .iterations = 3 },
        .{ .m = 512, .k = 512, .n = 512, .warmups = 1, .iterations = 2 },
        .{ .m = 1024, .k = 1024, .n = 1024, .warmups = 1, .iterations = 1 },
    };

    for (cases, 0..) |case, index| {
        if (options.quick and index > 0) break;

        const warmups = quickCount(options, case.warmups);
        const iterations = quickCount(options, case.iterations);

        var name_buffer: [64]u8 = undefined;
        const case_name = try std.fmt.bufPrint(&name_buffer, "{d}x{d}x{d}", .{ case.m, case.k, case.n });

        var cpu_a = try BackendMatrix.init(cpu, allocator, case.m, case.k);
        defer cpu_a.deinit();
        var cpu_b = try BackendMatrix.init(cpu, allocator, case.k, case.n);
        defer cpu_b.deinit();
        fillBackendMatrix(cpu_a, 11);
        fillBackendMatrix(cpu_b, 29);

        const cpu_timing = try timeMatmul(allocator, cpu_a, cpu_b, warmups, iterations);
        printResult("matmul", case_name, "cpu", cpu_timing, 0.0, "ok");

        if (metal) |metal_backend| {
            var metal_a = try BackendMatrix.init(metal_backend, allocator, case.m, case.k);
            defer metal_a.deinit();
            var metal_b = try BackendMatrix.init(metal_backend, allocator, case.k, case.n);
            defer metal_b.deinit();
            fillBackendMatrix(metal_a, 11);
            fillBackendMatrix(metal_b, 29);

            const metal_timing = try timeMatmul(allocator, metal_a, metal_b, warmups, iterations);
            const sample_error = try compareMatmul(allocator, cpu_a, cpu_b, metal_a, metal_b);
            printResult("matmul", case_name, "metal", metal_timing, sample_error, "ok");
        } else {
            printSkipped("matmul", case_name, "metal", warmups, iterations, "skipped_metal_unavailable");
        }

        if (cuda) |cuda_backend| {
            var cuda_a = try BackendMatrix.init(cuda_backend, allocator, case.m, case.k);
            defer cuda_a.deinit();
            var cuda_b = try BackendMatrix.init(cuda_backend, allocator, case.k, case.n);
            defer cuda_b.deinit();
            fillBackendMatrix(cuda_a, 11);
            fillBackendMatrix(cuda_b, 29);

            const cuda_timing = try timeMatmul(allocator, cuda_a, cuda_b, warmups, iterations);
            const sample_error = try compareMatmul(allocator, cpu_a, cpu_b, cuda_a, cuda_b);
            printResult("matmul", case_name, "cuda", cuda_timing, sample_error, "ok");
        } else {
            printSkipped("matmul", case_name, "cuda", warmups, iterations, "skipped_cuda_unavailable");
        }

        if (rocm) |rocm_backend| {
            var rocm_a = try BackendMatrix.init(rocm_backend, allocator, case.m, case.k);
            defer rocm_a.deinit();
            var rocm_b = try BackendMatrix.init(rocm_backend, allocator, case.k, case.n);
            defer rocm_b.deinit();
            fillBackendMatrix(rocm_a, 11);
            fillBackendMatrix(rocm_b, 29);

            const rocm_timing = try timeMatmul(allocator, rocm_a, rocm_b, warmups, iterations);
            const sample_error = try compareMatmul(allocator, cpu_a, cpu_b, rocm_a, rocm_b);
            printResult("matmul", case_name, "rocm", rocm_timing, sample_error, "ok");
        } else {
            printSkipped("matmul", case_name, "rocm", warmups, iterations, "skipped_rocm_unavailable");
        }
    }
}

fn timeMatmul(
    allocator: Allocator,
    a: *const BackendMatrix,
    b: *const BackendMatrix,
    warmups: usize,
    iterations: usize,
) !Timing {
    for (0..warmups) |_| {
        const result = try a.dotProduct(b, allocator);
        result.deinit();
    }

    var total_ns: f64 = 0.0;
    var min_ns: f64 = std.math.inf(f64);
    var max_ns: f64 = 0.0;
    var checksum: f64 = 0.0;

    for (0..iterations) |iteration| {
        const start = nowNs();
        const result = try a.dotProduct(b, allocator);
        const elapsed = nowNs() - start;
        checksum += sampleBackendMatrix(result, iteration);
        result.deinit();

        const elapsed_ns = @as(f64, @floatFromInt(elapsed));
        total_ns += elapsed_ns;
        min_ns = @min(min_ns, elapsed_ns);
        max_ns = @max(max_ns, elapsed_ns);
    }

    return .{
        .warmups = warmups,
        .iterations = iterations,
        .average_ns = total_ns / @as(f64, @floatFromInt(iterations)),
        .min_ns = min_ns,
        .max_ns = max_ns,
        .checksum = checksum,
    };
}

fn compareMatmul(
    allocator: Allocator,
    cpu_a: *const BackendMatrix,
    cpu_b: *const BackendMatrix,
    metal_a: *const BackendMatrix,
    metal_b: *const BackendMatrix,
) !f64 {
    const cpu_result = try cpu_a.dotProduct(cpu_b, allocator);
    defer cpu_result.deinit();
    const metal_result = try metal_a.dotProduct(metal_b, allocator);
    defer metal_result.deinit();
    return sampleMaxAbsDiff(cpu_result, metal_result);
}

fn benchmarkActivations(
    allocator: Allocator,
    cpu: BackendInstance,
    metal: ?BackendInstance,
    cuda: ?BackendInstance,
    rocm: ?BackendInstance,
    options: Options,
) !void {
    const cases = [_]struct {
        name: []const u8,
        op: BackendOp,
        rows: usize,
        cols: usize,
        warmups: usize,
        iterations: usize,
    }{
        .{ .name = "relu_512x512", .op = .relu, .rows = 512, .cols = 512, .warmups = 1, .iterations = 10 },
        .{ .name = "tanh_512x512", .op = .tanh, .rows = 512, .cols = 512, .warmups = 1, .iterations = 8 },
        .{ .name = "swish_512x512", .op = .swish, .rows = 512, .cols = 512, .warmups = 1, .iterations = 8 },
        .{ .name = "softmax_256x256", .op = .softmax, .rows = 256, .cols = 256, .warmups = 1, .iterations = 8 },
        .{ .name = "glu_512x512", .op = .glu, .rows = 512, .cols = 512, .warmups = 1, .iterations = 8 },
        .{ .name = "swiglu_512x512", .op = .swiglu, .rows = 512, .cols = 512, .warmups = 1, .iterations = 8 },
        .{ .name = "relu_2048x2048", .op = .relu, .rows = 2048, .cols = 2048, .warmups = 1, .iterations = 3 },
        .{ .name = "tanh_2048x2048", .op = .tanh, .rows = 2048, .cols = 2048, .warmups = 1, .iterations = 2 },
        .{ .name = "swish_2048x2048", .op = .swish, .rows = 2048, .cols = 2048, .warmups = 1, .iterations = 2 },
        .{ .name = "softmax_1024x1024", .op = .softmax, .rows = 1024, .cols = 1024, .warmups = 1, .iterations = 2 },
        .{ .name = "glu_2048x2048", .op = .glu, .rows = 2048, .cols = 2048, .warmups = 1, .iterations = 2 },
        .{ .name = "swiglu_2048x2048", .op = .swiglu, .rows = 2048, .cols = 2048, .warmups = 1, .iterations = 2 },
    };

    for (cases, 0..) |case, index| {
        if (options.quick and index > 1) break;

        const warmups = quickCount(options, case.warmups);
        const iterations = quickCount(options, case.iterations);

        var cpu_input = try BackendMatrix.init(cpu, allocator, case.rows, case.cols);
        defer cpu_input.deinit();
        var cpu_gate = try BackendMatrix.init(cpu, allocator, case.rows, case.cols);
        defer cpu_gate.deinit();
        fillBackendMatrix(cpu_input, 43);
        fillBackendMatrix(cpu_gate, 71);

        const cpu_timing = try timeActivation(allocator, case.op, cpu_input, cpu_gate, warmups, iterations);
        printResult("activation", case.name, "cpu", cpu_timing, 0.0, "ok");

        if (metal) |metal_backend| {
            var metal_input = try BackendMatrix.init(metal_backend, allocator, case.rows, case.cols);
            defer metal_input.deinit();
            var metal_gate = try BackendMatrix.init(metal_backend, allocator, case.rows, case.cols);
            defer metal_gate.deinit();
            fillBackendMatrix(metal_input, 43);
            fillBackendMatrix(metal_gate, 71);

            const metal_timing = try timeActivation(allocator, case.op, metal_input, metal_gate, warmups, iterations);
            const sample_error = try compareActivation(allocator, case.op, cpu_input, cpu_gate, metal_input, metal_gate);
            printResult("activation", case.name, "metal", metal_timing, sample_error, "ok");
        } else {
            printSkipped("activation", case.name, "metal", warmups, iterations, "skipped_metal_unavailable");
        }

        if (cuda) |cuda_backend| {
            var cuda_input = try BackendMatrix.init(cuda_backend, allocator, case.rows, case.cols);
            defer cuda_input.deinit();
            var cuda_gate = try BackendMatrix.init(cuda_backend, allocator, case.rows, case.cols);
            defer cuda_gate.deinit();
            fillBackendMatrix(cuda_input, 43);
            fillBackendMatrix(cuda_gate, 71);

            const cuda_timing = try timeActivation(allocator, case.op, cuda_input, cuda_gate, warmups, iterations);
            const sample_error = try compareActivation(allocator, case.op, cpu_input, cpu_gate, cuda_input, cuda_gate);
            printResult("activation", case.name, "cuda", cuda_timing, sample_error, "ok");
        } else {
            printSkipped("activation", case.name, "cuda", warmups, iterations, "skipped_cuda_unavailable");
        }

        if (rocm) |rocm_backend| {
            var rocm_input = try BackendMatrix.init(rocm_backend, allocator, case.rows, case.cols);
            defer rocm_input.deinit();
            var rocm_gate = try BackendMatrix.init(rocm_backend, allocator, case.rows, case.cols);
            defer rocm_gate.deinit();
            fillBackendMatrix(rocm_input, 43);
            fillBackendMatrix(rocm_gate, 71);

            const rocm_timing = try timeActivation(allocator, case.op, rocm_input, rocm_gate, warmups, iterations);
            const sample_error = try compareActivation(allocator, case.op, cpu_input, cpu_gate, rocm_input, rocm_gate);
            printResult("activation", case.name, "rocm", rocm_timing, sample_error, "ok");
        } else {
            printSkipped("activation", case.name, "rocm", warmups, iterations, "skipped_rocm_unavailable");
        }
    }
}

fn timeActivation(
    allocator: Allocator,
    op: BackendOp,
    input: *const BackendMatrix,
    gate: *const BackendMatrix,
    warmups: usize,
    iterations: usize,
) !Timing {
    for (0..warmups) |_| {
        const result = try runBackendOp(allocator, op, input, gate);
        result.deinit();
    }

    var total_ns: f64 = 0.0;
    var min_ns: f64 = std.math.inf(f64);
    var max_ns: f64 = 0.0;
    var checksum: f64 = 0.0;

    for (0..iterations) |iteration| {
        const start = nowNs();
        const result = try runBackendOp(allocator, op, input, gate);
        const elapsed = nowNs() - start;
        checksum += sampleBackendMatrix(result, iteration);
        result.deinit();

        const elapsed_ns = @as(f64, @floatFromInt(elapsed));
        total_ns += elapsed_ns;
        min_ns = @min(min_ns, elapsed_ns);
        max_ns = @max(max_ns, elapsed_ns);
    }

    return .{
        .warmups = warmups,
        .iterations = iterations,
        .average_ns = total_ns / @as(f64, @floatFromInt(iterations)),
        .min_ns = min_ns,
        .max_ns = max_ns,
        .checksum = checksum,
    };
}

fn runBackendOp(
    allocator: Allocator,
    op: BackendOp,
    input: *const BackendMatrix,
    gate: *const BackendMatrix,
) !*BackendMatrix {
    return switch (op) {
        .relu => input.applyActivation(nn.Activation.relu, allocator),
        .tanh => input.applyActivation(nn.Activation.tanh, allocator),
        .swish => input.applyActivation(nn.Activation.swish, allocator),
        .softmax => input.applySoftmax(allocator),
        .glu => input.applyGLU(gate, allocator),
        .swiglu => input.applySwiGLU(gate, allocator),
    };
}

fn compareActivation(
    allocator: Allocator,
    op: BackendOp,
    cpu_input: *const BackendMatrix,
    cpu_gate: *const BackendMatrix,
    metal_input: *const BackendMatrix,
    metal_gate: *const BackendMatrix,
) !f64 {
    const cpu_result = try runBackendOp(allocator, op, cpu_input, cpu_gate);
    defer cpu_result.deinit();
    const metal_result = try runBackendOp(allocator, op, metal_input, metal_gate);
    defer metal_result.deinit();
    return sampleMaxAbsDiff(cpu_result, metal_result);
}

fn benchmarkGpuHeavy(
    allocator: Allocator,
    metal: ?BackendInstance,
    cuda: ?BackendInstance,
    rocm: ?BackendInstance,
    options: Options,
) !void {
    const matmul_cases = [_]struct {
        name: []const u8,
        m: usize,
        k: usize,
        n: usize,
        warmups: usize,
        iterations: usize,
    }{
        .{ .name = "matmul_2048x2048x2048", .m = 2048, .k = 2048, .n = 2048, .warmups = 1, .iterations = 1 },
        .{ .name = "matmul_4096x1024x4096", .m = 4096, .k = 1024, .n = 4096, .warmups = 1, .iterations = 1 },
    };
    const activation_cases = [_]struct {
        name: []const u8,
        op: BackendOp,
        rows: usize,
        cols: usize,
        warmups: usize,
        iterations: usize,
    }{
        .{ .name = "relu_8192x8192", .op = .relu, .rows = 8192, .cols = 8192, .warmups = 1, .iterations = 1 },
        .{ .name = "softmax_4096x4096", .op = .softmax, .rows = 4096, .cols = 4096, .warmups = 1, .iterations = 1 },
        .{ .name = "swiglu_4096x4096", .op = .swiglu, .rows = 4096, .cols = 4096, .warmups = 1, .iterations = 1 },
    };

    for (matmul_cases) |case| {
        if (options.quick) {
            printGpuHeavyQuickSkipped(case.name);
            continue;
        }

        if (metal) |metal_backend| {
            try benchmarkGpuHeavyMatmulCase(allocator, metal_backend, "metal", case.name, case.m, case.k, case.n, case.warmups, case.iterations);
        } else {
            printSkipped("gpu_heavy", case.name, "metal", case.warmups, case.iterations, "skipped_metal_unavailable");
        }

        if (cuda) |cuda_backend| {
            try benchmarkGpuHeavyMatmulCase(allocator, cuda_backend, "cuda", case.name, case.m, case.k, case.n, case.warmups, case.iterations);
        } else {
            printSkipped("gpu_heavy", case.name, "cuda", case.warmups, case.iterations, "skipped_cuda_unavailable");
        }

        if (rocm) |rocm_backend| {
            try benchmarkGpuHeavyMatmulCase(allocator, rocm_backend, "rocm", case.name, case.m, case.k, case.n, case.warmups, case.iterations);
        } else {
            printSkipped("gpu_heavy", case.name, "rocm", case.warmups, case.iterations, "skipped_rocm_unavailable");
        }
    }

    for (activation_cases) |case| {
        if (options.quick) {
            printGpuHeavyQuickSkipped(case.name);
            continue;
        }

        if (metal) |metal_backend| {
            try benchmarkGpuHeavyActivationCase(allocator, metal_backend, "metal", case.name, case.op, case.rows, case.cols, case.warmups, case.iterations);
        } else {
            printSkipped("gpu_heavy", case.name, "metal", case.warmups, case.iterations, "skipped_metal_unavailable");
        }

        if (cuda) |cuda_backend| {
            try benchmarkGpuHeavyActivationCase(allocator, cuda_backend, "cuda", case.name, case.op, case.rows, case.cols, case.warmups, case.iterations);
        } else {
            printSkipped("gpu_heavy", case.name, "cuda", case.warmups, case.iterations, "skipped_cuda_unavailable");
        }

        if (rocm) |rocm_backend| {
            try benchmarkGpuHeavyActivationCase(allocator, rocm_backend, "rocm", case.name, case.op, case.rows, case.cols, case.warmups, case.iterations);
        } else {
            printSkipped("gpu_heavy", case.name, "rocm", case.warmups, case.iterations, "skipped_rocm_unavailable");
        }
    }
}

fn benchmarkGpuHeavyMatmulCase(
    allocator: Allocator,
    backend: BackendInstance,
    backend_name: []const u8,
    case_name: []const u8,
    m: usize,
    k: usize,
    n: usize,
    warmups: usize,
    iterations: usize,
) !void {
    var a = try BackendMatrix.init(backend, allocator, m, k);
    defer a.deinit();
    var b = try BackendMatrix.init(backend, allocator, k, n);
    defer b.deinit();
    fillBackendMatrix(a, 313);
    fillBackendMatrix(b, 619);

    const timing = try timeMatmul(allocator, a, b, warmups, iterations);
    printResult("gpu_heavy", case_name, backend_name, timing, 0.0, "ok");
}

fn benchmarkGpuHeavyActivationCase(
    allocator: Allocator,
    backend: BackendInstance,
    backend_name: []const u8,
    case_name: []const u8,
    op: BackendOp,
    rows: usize,
    cols: usize,
    warmups: usize,
    iterations: usize,
) !void {
    var input = try BackendMatrix.init(backend, allocator, rows, cols);
    defer input.deinit();
    fillBackendMatrix(input, 733);

    const timing = switch (op) {
        .glu, .swiglu => blk: {
            var gate = try BackendMatrix.init(backend, allocator, rows, cols);
            defer gate.deinit();
            fillBackendMatrix(gate, 947);
            break :blk try timeActivation(allocator, op, input, gate, warmups, iterations);
        },
        else => try timeActivation(allocator, op, input, input, warmups, iterations),
    };
    printResult("gpu_heavy", case_name, backend_name, timing, 0.0, "ok");
}

fn printGpuHeavyQuickSkipped(case_name: []const u8) void {
    printSkipped("gpu_heavy", case_name, "metal", 0, 0, "skipped_quick_mode");
    printSkipped("gpu_heavy", case_name, "cuda", 0, 0, "skipped_quick_mode");
    printSkipped("gpu_heavy", case_name, "rocm", 0, 0, "skipped_quick_mode");
}

fn benchmarkGpuPeak(
    allocator: Allocator,
    metal: ?BackendInstance,
    cuda: ?BackendInstance,
    rocm: ?BackendInstance,
    options: Options,
) !void {
    const cases = [_]struct {
        name: []const u8,
        m: usize,
        k: usize,
        n: usize,
        warmups: usize,
        iterations: usize,
    }{
        .{ .name = "matmul_8192x8192x8192", .m = 8192, .k = 8192, .n = 8192, .warmups = 1, .iterations = 2 },
        .{ .name = "matmul_12288x12288x12288", .m = 12288, .k = 12288, .n = 12288, .warmups = 1, .iterations = 2 },
        .{ .name = "matmul_16384x8192x16384", .m = 16384, .k = 8192, .n = 16384, .warmups = 1, .iterations = 2 },
        .{ .name = "matmul_24576x4096x24576", .m = 24576, .k = 4096, .n = 24576, .warmups = 1, .iterations = 2 },
    };

    for (cases) |case| {
        if (options.quick) {
            printGpuPeakQuickSkipped(case.name);
            continue;
        }

        if (metal) |metal_backend| {
            try benchmarkGpuPeakMatmulCase(allocator, metal_backend, "metal", case.name, case.m, case.k, case.n, case.warmups, case.iterations);
        } else {
            printSkipped("gpu_peak", case.name, "metal", case.warmups, case.iterations, "skipped_metal_unavailable");
        }

        if (cuda) |cuda_backend| {
            try benchmarkGpuPeakMatmulCase(allocator, cuda_backend, "cuda", case.name, case.m, case.k, case.n, case.warmups, case.iterations);
        } else {
            printSkipped("gpu_peak", case.name, "cuda", case.warmups, case.iterations, "skipped_cuda_unavailable");
        }

        if (rocm) |rocm_backend| {
            try benchmarkGpuPeakMatmulCase(allocator, rocm_backend, "rocm", case.name, case.m, case.k, case.n, case.warmups, case.iterations);
        } else {
            printSkipped("gpu_peak", case.name, "rocm", case.warmups, case.iterations, "skipped_rocm_unavailable");
        }
    }
}

fn benchmarkGpuPeakMatmulCase(
    allocator: Allocator,
    backend: BackendInstance,
    backend_name: []const u8,
    case_name: []const u8,
    m: usize,
    k: usize,
    n: usize,
    warmups: usize,
    iterations: usize,
) !void {
    var a = try BackendMatrix.init(backend, allocator, m, k);
    defer a.deinit();
    var b = try BackendMatrix.init(backend, allocator, k, n);
    defer b.deinit();
    fillBackendMatrix(a, 1597);
    fillBackendMatrix(b, 3251);

    const timing = try timeMatmul(allocator, a, b, warmups, iterations);
    printResult("gpu_peak", case_name, backend_name, timing, 0.0, "ok");
}

fn printGpuPeakQuickSkipped(case_name: []const u8) void {
    printSkipped("gpu_peak", case_name, "metal", 0, 0, "skipped_quick_mode");
    printSkipped("gpu_peak", case_name, "cuda", 0, 0, "skipped_quick_mode");
    printSkipped("gpu_peak", case_name, "rocm", 0, 0, "skipped_quick_mode");
}

fn benchmarkLayerNorm(allocator: Allocator, options: Options) !void {
    const cases = [_]struct {
        name: []const u8,
        rows: usize,
        cols: usize,
        warmups: usize,
        iterations: usize,
    }{
        .{ .name = "batch128_width64", .rows = 128, .cols = 64, .warmups = 1, .iterations = 10 },
        .{ .name = "batch512_width256", .rows = 512, .cols = 256, .warmups = 1, .iterations = 6 },
        .{ .name = "batch1024_width768", .rows = 1024, .cols = 768, .warmups = 1, .iterations = 3 },
        .{ .name = "batch4096_width1024", .rows = 4096, .cols = 1024, .warmups = 1, .iterations = 2 },
    };

    for (cases, 0..) |case, index| {
        if (options.quick and index > 0) break;

        var layer_norm = try nn.LayerNorm.init(allocator, case.cols);
        defer layer_norm.deinit();
        fillCpuMatrix(&layer_norm.weight, 211);
        fillCpuMatrix(&layer_norm.bias, 223);

        var input = try Matrix.init(allocator, case.rows, case.cols);
        defer input.deinit();
        fillCpuMatrix(&input, 197);

        const timing = try timeLayerNormForward(
            &layer_norm,
            input,
            quickCount(options, case.warmups),
            quickCount(options, case.iterations),
        );
        printResult("layer_norm", case.name, "cpu", timing, 0.0, "ok");
        printSkipped("layer_norm", case.name, "metal", timing.warmups, timing.iterations, "skipped_cpu_only_path");
        printSkipped("layer_norm", case.name, "cuda", timing.warmups, timing.iterations, "skipped_cpu_only_path");
        printSkipped("layer_norm", case.name, "rocm", timing.warmups, timing.iterations, "skipped_cpu_only_path");
    }
}

fn timeLayerNormForward(
    layer_norm: *const nn.LayerNorm,
    input: Matrix,
    warmups: usize,
    iterations: usize,
) !Timing {
    for (0..warmups) |_| {
        var output = try layer_norm.forward(input, input.allocator);
        output.deinit();
    }

    var total_ns: f64 = 0.0;
    var min_ns: f64 = std.math.inf(f64);
    var max_ns: f64 = 0.0;
    var checksum: f64 = 0.0;

    for (0..iterations) |iteration| {
        const start = nowNs();
        var output = try layer_norm.forward(input, input.allocator);
        const elapsed = nowNs() - start;
        checksum += sampleCpuMatrix(output, iteration);
        output.deinit();

        const elapsed_ns = @as(f64, @floatFromInt(elapsed));
        total_ns += elapsed_ns;
        min_ns = @min(min_ns, elapsed_ns);
        max_ns = @max(max_ns, elapsed_ns);
    }

    return .{
        .warmups = warmups,
        .iterations = iterations,
        .average_ns = total_ns / @as(f64, @floatFromInt(iterations)),
        .min_ns = min_ns,
        .max_ns = max_ns,
        .checksum = checksum,
    };
}

fn benchmarkTraining(
    allocator: Allocator,
    metal: ?BackendInstance,
    cuda: ?BackendInstance,
    rocm: ?BackendInstance,
    options: Options,
) !void {
    const cases = [_]struct {
        name: []const u8,
        samples: usize,
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        steps: usize,
        warmups: usize,
        iterations: usize,
    }{
        .{ .name = "batch64_32_64_16_steps8", .samples = 64, .input_size = 32, .hidden_size = 64, .output_size = 16, .steps = 8, .warmups = 1, .iterations = 5 },
        .{ .name = "batch128_64_128_32_steps4", .samples = 128, .input_size = 64, .hidden_size = 128, .output_size = 32, .steps = 4, .warmups = 1, .iterations = 3 },
        .{ .name = "batch256_128_256_64_steps4", .samples = 256, .input_size = 128, .hidden_size = 256, .output_size = 64, .steps = 4, .warmups = 1, .iterations = 1 },
    };

    for (cases, 0..) |case, index| {
        if (options.quick and index > 0) break;

        var inputs = try Matrix.init(allocator, case.samples, case.input_size);
        defer inputs.deinit();
        var targets = try Matrix.init(allocator, case.samples, case.output_size);
        defer targets.deinit();
        fillCpuMatrix(&inputs, 101);
        fillCpuMatrix(&targets, 137);

        var network = try makeTrainingNetwork(allocator, case.input_size, case.hidden_size, case.output_size);
        defer network.deinit();

        const timing = try timeTrainingLoop(
            &network,
            inputs,
            targets,
            case.steps,
            quickCount(options, case.warmups),
            quickCount(options, case.iterations),
        );
        printResult("training", case.name, "cpu", timing, 0.0, "ok");

        if (metal) |metal_backend| {
            var metal_network = try makeTrainingNetwork(allocator, case.input_size, case.hidden_size, case.output_size);
            defer metal_network.deinit();

            const metal_inputs = try BackendMatrix.fromMatrix(metal_backend, inputs, allocator);
            defer metal_inputs.deinit();
            const metal_targets = try BackendMatrix.fromMatrix(metal_backend, targets, allocator);
            defer metal_targets.deinit();

            const metal_timing = try timeTrainingLoopBackend(
                &metal_network,
                metal_inputs,
                metal_targets,
                case.steps,
                quickCount(options, case.warmups),
                quickCount(options, case.iterations),
            );
            printResult("training", case.name, "metal", metal_timing, @abs(timing.checksum - metal_timing.checksum), "ok");
        } else {
            printSkipped("training", case.name, "metal", timing.warmups, timing.iterations, "skipped_metal_unavailable");
        }

        if (cuda) |cuda_backend| {
            var cuda_network = try makeTrainingNetwork(allocator, case.input_size, case.hidden_size, case.output_size);
            defer cuda_network.deinit();

            const cuda_inputs = try BackendMatrix.fromMatrix(cuda_backend, inputs, allocator);
            defer cuda_inputs.deinit();
            const cuda_targets = try BackendMatrix.fromMatrix(cuda_backend, targets, allocator);
            defer cuda_targets.deinit();

            const cuda_timing = try timeTrainingLoopBackend(
                &cuda_network,
                cuda_inputs,
                cuda_targets,
                case.steps,
                quickCount(options, case.warmups),
                quickCount(options, case.iterations),
            );
            printResult("training", case.name, "cuda", cuda_timing, @abs(timing.checksum - cuda_timing.checksum), "ok");
        } else {
            printSkipped("training", case.name, "cuda", timing.warmups, timing.iterations, "skipped_cuda_unavailable");
        }

        if (rocm) |rocm_backend| {
            var rocm_network = try makeTrainingNetwork(allocator, case.input_size, case.hidden_size, case.output_size);
            defer rocm_network.deinit();

            const rocm_inputs = try BackendMatrix.fromMatrix(rocm_backend, inputs, allocator);
            defer rocm_inputs.deinit();
            const rocm_targets = try BackendMatrix.fromMatrix(rocm_backend, targets, allocator);
            defer rocm_targets.deinit();

            const rocm_timing = try timeTrainingLoopBackend(
                &rocm_network,
                rocm_inputs,
                rocm_targets,
                case.steps,
                quickCount(options, case.warmups),
                quickCount(options, case.iterations),
            );
            printResult("training", case.name, "rocm", rocm_timing, @abs(timing.checksum - rocm_timing.checksum), "ok");
        } else {
            printSkipped("training", case.name, "rocm", timing.warmups, timing.iterations, "skipped_rocm_unavailable");
        }
    }
}

fn makeTrainingNetwork(
    allocator: Allocator,
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
) !nn.Network {
    var network = nn.Network.init(allocator, 0.01, .MeanSquaredError);
    errdefer network.deinit();

    try network.addLayer(input_size, hidden_size, nn.Activation.relu, nn.Activation.relu_derivative);
    try network.addLayer(hidden_size, output_size, nn.Activation.linear, nn.Activation.linear_derivative);
    fillNetworkParameters(&network);

    return network;
}

fn fillNetworkParameters(network: *nn.Network) void {
    for (network.layers.items, 0..) |layer_variant, layer_index| {
        switch (layer_variant) {
            .Standard => |layer| {
                fillCpuMatrix(&layer.weights, 1_000 + layer_index * 17);
                fillCpuMatrix(&layer.bias, 2_000 + layer_index * 17);
            },
            .Gated => |layer| {
                fillCpuMatrix(&layer.linear_weights, 3_000 + layer_index * 17);
                fillCpuMatrix(&layer.linear_bias, 4_000 + layer_index * 17);
                fillCpuMatrix(&layer.gate_weights, 5_000 + layer_index * 17);
                fillCpuMatrix(&layer.gate_bias, 6_000 + layer_index * 17);
            },
        }
    }
}

fn timeTrainingLoop(
    network: *nn.Network,
    inputs: Matrix,
    targets: Matrix,
    steps: usize,
    warmups: usize,
    iterations: usize,
) !Timing {
    for (0..warmups) |_| {
        for (0..steps) |_| {
            _ = try network.trainBatch(inputs, targets);
        }
    }

    var total_ns: f64 = 0.0;
    var min_ns: f64 = std.math.inf(f64);
    var max_ns: f64 = 0.0;
    var checksum: f64 = 0.0;

    for (0..iterations) |_| {
        const start = nowNs();
        var last_loss: f64 = 0.0;
        for (0..steps) |_| {
            last_loss = try network.trainBatch(inputs, targets);
        }
        const elapsed = nowNs() - start;
        checksum += last_loss;

        const elapsed_ns = @as(f64, @floatFromInt(elapsed));
        total_ns += elapsed_ns;
        min_ns = @min(min_ns, elapsed_ns);
        max_ns = @max(max_ns, elapsed_ns);
    }

    return .{
        .warmups = warmups,
        .iterations = iterations,
        .average_ns = total_ns / @as(f64, @floatFromInt(iterations)),
        .min_ns = min_ns,
        .max_ns = max_ns,
        .checksum = checksum,
    };
}

fn timeTrainingLoopBackend(
    network: *nn.Network,
    inputs: *const BackendMatrix,
    targets: *const BackendMatrix,
    steps: usize,
    warmups: usize,
    iterations: usize,
) !Timing {
    for (0..warmups) |_| {
        for (0..steps) |_| {
            _ = try network.trainBatchBackend(inputs, targets);
        }
    }

    var total_ns: f64 = 0.0;
    var min_ns: f64 = std.math.inf(f64);
    var max_ns: f64 = 0.0;
    var checksum: f64 = 0.0;

    for (0..iterations) |_| {
        const start = nowNs();
        var last_loss: f64 = 0.0;
        for (0..steps) |_| {
            last_loss = try network.trainBatchBackend(inputs, targets);
        }
        const elapsed = nowNs() - start;
        checksum += last_loss;

        const elapsed_ns = @as(f64, @floatFromInt(elapsed));
        total_ns += elapsed_ns;
        min_ns = @min(min_ns, elapsed_ns);
        max_ns = @max(max_ns, elapsed_ns);
    }

    return .{
        .warmups = warmups,
        .iterations = iterations,
        .average_ns = total_ns / @as(f64, @floatFromInt(iterations)),
        .min_ns = min_ns,
        .max_ns = max_ns,
        .checksum = checksum,
    };
}

fn benchmarkTinyGpt(allocator: Allocator, options: Options) !void {
    const cases = [_]struct {
        name: []const u8,
        config: tiny_gpt.Config,
        seq_len: usize,
        warmups: usize,
        iterations: usize,
    }{
        .{
            .name = "layers2_heads2_embd32_seq16",
            .config = .{ .block_size = 16, .n_layer = 2, .n_head = 2, .n_embd = 32 },
            .seq_len = 16,
            .warmups = 1,
            .iterations = 5,
        },
        .{
            .name = "layers2_heads4_embd64_seq32",
            .config = .{ .block_size = 32, .n_layer = 2, .n_head = 4, .n_embd = 64 },
            .seq_len = 32,
            .warmups = 1,
            .iterations = 3,
        },
        .{
            .name = "layers4_heads4_embd128_seq64",
            .config = .{ .block_size = 64, .n_layer = 4, .n_head = 4, .n_embd = 128 },
            .seq_len = 64,
            .warmups = 1,
            .iterations = 1,
        },
    };

    for (cases, 0..) |case, index| {
        if (options.quick and index > 0) break;

        var model = try tiny_gpt.TinyGPT.init(allocator, case.config, 0x9e37_79b9);
        defer model.deinit();

        const tokens = try allocator.alloc(usize, case.seq_len);
        defer allocator.free(tokens);
        fillTokens(tokens, case.config.vocab_size);

        const timing = try timeTinyGptForward(
            &model,
            tokens,
            quickCount(options, case.warmups),
            quickCount(options, case.iterations),
        );
        printResult("tiny_gpt", case.name, "cpu", timing, 0.0, "ok");
        printSkipped("tiny_gpt", case.name, "metal", timing.warmups, timing.iterations, "skipped_cpu_only_path");
        printSkipped("tiny_gpt", case.name, "cuda", timing.warmups, timing.iterations, "skipped_cpu_only_path");
        printSkipped("tiny_gpt", case.name, "rocm", timing.warmups, timing.iterations, "skipped_cpu_only_path");
    }
}

fn timeTinyGptForward(
    model: *tiny_gpt.TinyGPT,
    tokens: []const usize,
    warmups: usize,
    iterations: usize,
) !Timing {
    for (0..warmups) |_| {
        var logits = try model.forward(tokens);
        logits.deinit();
    }

    var total_ns: f64 = 0.0;
    var min_ns: f64 = std.math.inf(f64);
    var max_ns: f64 = 0.0;
    var checksum: f64 = 0.0;

    for (0..iterations) |iteration| {
        const start = nowNs();
        var logits = try model.forward(tokens);
        const elapsed = nowNs() - start;
        checksum += sampleCpuMatrix(logits, iteration);
        logits.deinit();

        const elapsed_ns = @as(f64, @floatFromInt(elapsed));
        total_ns += elapsed_ns;
        min_ns = @min(min_ns, elapsed_ns);
        max_ns = @max(max_ns, elapsed_ns);
    }

    return .{
        .warmups = warmups,
        .iterations = iterations,
        .average_ns = total_ns / @as(f64, @floatFromInt(iterations)),
        .min_ns = min_ns,
        .max_ns = max_ns,
        .checksum = checksum,
    };
}

fn benchmarkQuantization(allocator: Allocator, options: Options) !void {
    const vector_cases = [_]struct {
        name: []const u8,
        op: QuantizationOp,
        len: usize,
        bits: u8,
        warmups: usize,
        iterations: usize,
    }{
        .{ .name = "uniform_8bit_4096", .op = .uniform, .len = 4096, .bits = 8, .warmups = 1, .iterations = 8 },
        .{ .name = "turbo_8bit_4096", .op = .turbo, .len = 4096, .bits = 8, .warmups = 1, .iterations = 6 },
        .{ .name = "turbo_8bit_16384", .op = .turbo, .len = 16_384, .bits = 8, .warmups = 1, .iterations = 4 },
        .{ .name = "uniform_8bit_262144", .op = .uniform, .len = 262_144, .bits = 8, .warmups = 1, .iterations = 2 },
        .{ .name = "turbo_8bit_262144", .op = .turbo, .len = 262_144, .bits = 8, .warmups = 1, .iterations = 1 },
    };
    const kv_cases = [_]struct {
        name: []const u8,
        tokens: usize,
        heads: usize,
        head_dim: usize,
        bits: u8,
        warmups: usize,
        iterations: usize,
    }{
        .{ .name = "kv_cache_turbo4_64t4h32d", .tokens = 64, .heads = 4, .head_dim = 32, .bits = 4, .warmups = 1, .iterations = 4 },
        .{ .name = "kv_cache_turbo4_128t8h64d", .tokens = 128, .heads = 8, .head_dim = 64, .bits = 4, .warmups = 1, .iterations = 2 },
        .{ .name = "kv_cache_turbo4_256t16h64d", .tokens = 256, .heads = 16, .head_dim = 64, .bits = 4, .warmups = 1, .iterations = 1 },
    };

    for (vector_cases, 0..) |case, index| {
        if (options.quick and index > 0) break;

        const input = try allocator.alloc(f64, case.len);
        defer allocator.free(input);
        const query = try allocator.alloc(f64, case.len);
        defer allocator.free(query);
        fillVector(input, 503);
        fillVector(query, 907);

        const timing = try timeQuantization(
            allocator,
            case.op,
            input,
            query,
            case.bits,
            quickCount(options, case.warmups),
            quickCount(options, case.iterations),
        );
        printResult("quantization", case.name, "cpu", timing, 0.0, "ok");
        printSkipped("quantization", case.name, "metal", timing.warmups, timing.iterations, "skipped_cpu_only_path");
        printSkipped("quantization", case.name, "cuda", timing.warmups, timing.iterations, "skipped_cpu_only_path");
        printSkipped("quantization", case.name, "rocm", timing.warmups, timing.iterations, "skipped_cpu_only_path");
    }

    for (kv_cases, 0..) |case, index| {
        if (options.quick and index > 0) break;

        const len = case.tokens * case.heads * case.head_dim;
        const keys = try allocator.alloc(f64, len);
        defer allocator.free(keys);
        const values = try allocator.alloc(f64, len);
        defer allocator.free(values);
        const query = try allocator.alloc(f64, len);
        defer allocator.free(query);
        fillKVCache(keys, case.heads, case.head_dim, 1_211);
        fillKVCache(values, case.heads, case.head_dim, 1_619);
        fillKVCache(query, case.heads, case.head_dim, 2_023);

        const timing = try timeKVCacheQuantization(
            allocator,
            keys,
            values,
            query,
            case.heads,
            case.head_dim,
            case.bits,
            quickCount(options, case.warmups),
            quickCount(options, case.iterations),
        );
        printResult("quantization", case.name, "cpu", timing, 0.0, "ok");
        printSkipped("quantization", case.name, "metal", timing.warmups, timing.iterations, "skipped_cpu_only_path");
        printSkipped("quantization", case.name, "cuda", timing.warmups, timing.iterations, "skipped_cpu_only_path");
        printSkipped("quantization", case.name, "rocm", timing.warmups, timing.iterations, "skipped_cpu_only_path");
    }
}

fn timeQuantization(
    allocator: Allocator,
    op: QuantizationOp,
    input: []const f64,
    query: []const f64,
    bits: u8,
    warmups: usize,
    iterations: usize,
) !Timing {
    for (0..warmups) |_| {
        _ = try runQuantization(allocator, op, input, query, bits);
    }

    var total_ns: f64 = 0.0;
    var min_ns: f64 = std.math.inf(f64);
    var max_ns: f64 = 0.0;
    var checksum: f64 = 0.0;

    for (0..iterations) |_| {
        const start = nowNs();
        const metric = try runQuantization(allocator, op, input, query, bits);
        const elapsed = nowNs() - start;
        checksum += metric;

        const elapsed_ns = @as(f64, @floatFromInt(elapsed));
        total_ns += elapsed_ns;
        min_ns = @min(min_ns, elapsed_ns);
        max_ns = @max(max_ns, elapsed_ns);
    }

    return .{
        .warmups = warmups,
        .iterations = iterations,
        .average_ns = total_ns / @as(f64, @floatFromInt(iterations)),
        .min_ns = min_ns,
        .max_ns = max_ns,
        .checksum = checksum,
    };
}

fn runQuantization(
    allocator: Allocator,
    op: QuantizationOp,
    input: []const f64,
    query: []const f64,
    bits: u8,
) !f64 {
    switch (op) {
        .uniform => {
            const encoded = try nn.Quantization.encodeUniformScalar(allocator, input, bits);
            defer encoded.deinit();

            const decoded = try encoded.decode(allocator);
            defer allocator.free(decoded);

            const metrics = try nn.Quantization.measureVectorError(input, decoded, query, encoded.payloadBits());
            return metrics.mse + metrics.max_abs_error + metrics.inner_product_error * 1e-9;
        },
        .turbo => {
            const encoded = try nn.Quantization.encodeTurboQuant(allocator, input, bits, 0x1234_5678_9abc_def0);
            defer encoded.deinit();

            const decoded = try encoded.decode(allocator);
            defer allocator.free(decoded);

            const metrics = try nn.Quantization.measureVectorError(input, decoded, query, encoded.payloadBits());
            return metrics.mse + metrics.max_abs_error + metrics.inner_product_error * 1e-9;
        },
    }
}

fn timeKVCacheQuantization(
    allocator: Allocator,
    keys: []const f64,
    values: []const f64,
    query: []const f64,
    heads: usize,
    head_dim: usize,
    bits: u8,
    warmups: usize,
    iterations: usize,
) !Timing {
    for (0..warmups) |_| {
        _ = try runKVCacheQuantization(allocator, keys, values, query, heads, head_dim, bits);
    }

    var total_ns: f64 = 0.0;
    var min_ns: f64 = std.math.inf(f64);
    var max_ns: f64 = 0.0;
    var checksum: f64 = 0.0;

    for (0..iterations) |_| {
        const start = nowNs();
        const metric = try runKVCacheQuantization(allocator, keys, values, query, heads, head_dim, bits);
        const elapsed = nowNs() - start;
        checksum += metric;

        const elapsed_ns = @as(f64, @floatFromInt(elapsed));
        total_ns += elapsed_ns;
        min_ns = @min(min_ns, elapsed_ns);
        max_ns = @max(max_ns, elapsed_ns);
    }

    return .{
        .warmups = warmups,
        .iterations = iterations,
        .average_ns = total_ns / @as(f64, @floatFromInt(iterations)),
        .min_ns = min_ns,
        .max_ns = max_ns,
        .checksum = checksum,
    };
}

fn runKVCacheQuantization(
    allocator: Allocator,
    keys: []const f64,
    values: []const f64,
    query: []const f64,
    heads: usize,
    head_dim: usize,
    bits: u8,
) !f64 {
    if (head_dim == 0 or heads == 0) return error.InvalidKVCacheShape;
    if (keys.len != values.len or keys.len != query.len) return error.InvalidKVCacheShape;
    if (keys.len % (heads * head_dim) != 0) return error.InvalidKVCacheShape;

    const vector_count = keys.len / head_dim;
    var checksum: f64 = 0.0;
    for (0..vector_count) |vector_index| {
        const start = vector_index * head_dim;
        const end = start + head_dim;
        const seed = 0x91e1_0da5_0000_0000 + @as(u64, @intCast(vector_index));

        checksum += try quantizeKVVector(allocator, keys[start..end], query[start..end], bits, seed);
        checksum += try quantizeKVVector(allocator, values[start..end], query[start..end], bits, seed ^ 0xa5a5_5a5a_f0f0_0f0f);
    }

    return checksum / @as(f64, @floatFromInt(vector_count * 2));
}

fn quantizeKVVector(
    allocator: Allocator,
    input: []const f64,
    query: []const f64,
    bits: u8,
    seed: u64,
) !f64 {
    const encoded = try nn.Quantization.encodeTurboQuant(allocator, input, bits, seed);
    defer encoded.deinit();

    const decoded = try encoded.decode(allocator);
    defer allocator.free(decoded);

    const metrics = try nn.Quantization.measureVectorError(input, decoded, query, encoded.payloadBits());
    return metrics.mse + metrics.max_abs_error + metrics.inner_product_error * 1e-6;
}

fn quickCount(options: Options, count: usize) usize {
    if (!options.quick) return count;
    return @min(count, 1);
}

fn fillBackendMatrix(matrix: *BackendMatrix, salt: usize) void {
    for (0..matrix.rows) |row| {
        for (0..matrix.cols) |col| {
            matrix.set(row, col, deterministicValue(row, col, salt));
        }
    }
}

fn fillCpuMatrix(matrix: *Matrix, salt: usize) void {
    for (matrix.data, 0..) |*value, index| {
        const row = index / matrix.cols;
        const col = index % matrix.cols;
        value.* = deterministicValue(row, col, salt);
    }
}

fn fillVector(values: []f64, salt: usize) void {
    for (values, 0..) |*value, index| {
        value.* = deterministicValue(index, index * 3 + 1, salt);
    }
}

fn fillKVCache(values: []f64, heads: usize, head_dim: usize, salt: usize) void {
    for (values, 0..) |*value, index| {
        const token = index / (heads * head_dim);
        const head = (index / head_dim) % heads;
        const dim = index % head_dim;
        const base = deterministicValue(token, dim, salt + head * 31);
        const head_scale = 1.0 + @as(f64, @floatFromInt(head)) * 0.05;
        value.* = base * head_scale;
    }
}

fn fillTokens(tokens: []usize, vocab_size: usize) void {
    for (tokens, 0..) |*token, index| {
        token.* = (index * 17 + 11) % vocab_size;
    }
}

fn deterministicValue(row: usize, col: usize, salt: usize) f64 {
    const mixed = (row * 131 + col * 313 + salt * 977 + 17) % 2003;
    const centered = @as(f64, @floatFromInt(mixed)) / 1001.0 - 1.0;
    return centered * 0.25;
}

fn sampleBackendMatrix(matrix: *const BackendMatrix, iteration: usize) f64 {
    const row = (iteration * 17 + matrix.rows / 3) % matrix.rows;
    const col = (iteration * 31 + matrix.cols / 5) % matrix.cols;
    return matrix.get(row, col);
}

fn sampleCpuMatrix(matrix: Matrix, iteration: usize) f64 {
    const row = (iteration * 17 + matrix.rows / 3) % matrix.rows;
    const col = (iteration * 31 + matrix.cols / 5) % matrix.cols;
    return matrix.data[row * matrix.cols + col];
}

fn sampleMaxAbsDiff(a: *const BackendMatrix, b: *const BackendMatrix) f64 {
    var max_abs: f64 = 0.0;
    const samples = [_][2]usize{
        .{ 0, 0 },
        .{ 0, a.cols - 1 },
        .{ a.rows - 1, 0 },
        .{ a.rows - 1, a.cols - 1 },
        .{ a.rows / 2, a.cols / 2 },
        .{ a.rows / 3, a.cols / 3 },
        .{ (a.rows * 2) / 3, (a.cols * 2) / 3 },
    };

    for (samples) |sample| {
        const diff = @abs(a.get(sample[0], sample[1]) - b.get(sample[0], sample[1]));
        max_abs = @max(max_abs, diff);
    }

    return max_abs;
}

fn printResult(
    suite: []const u8,
    case_name: []const u8,
    backend: []const u8,
    timing: Timing,
    sample_error: f64,
    status: []const u8,
) void {
    std.debug.print(
        "{s},{s},{s},{s},{d},{d},{d:.0},{d:.0},{d:.0},{d:.9},{d:.9},{s}\n",
        .{
            suite,
            case_name,
            backend,
            modeName(),
            timing.warmups,
            timing.iterations,
            timing.average_ns,
            timing.min_ns,
            timing.max_ns,
            timing.checksum,
            sample_error,
            status,
        },
    );
}

fn printSkipped(
    suite: []const u8,
    case_name: []const u8,
    backend: []const u8,
    warmups: usize,
    iterations: usize,
    status: []const u8,
) void {
    std.debug.print(
        "{s},{s},{s},{s},{d},{d},0,0,0,0,0,{s}\n",
        .{ suite, case_name, backend, modeName(), warmups, iterations, status },
    );
}
