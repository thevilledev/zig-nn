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

    printHeader(options, metal != null);

    if (shouldRun(options, "matmul")) {
        try benchmarkMatmul(allocator, cpu, metal, options);
    }
    if (shouldRun(options, "activation")) {
        try benchmarkActivations(allocator, cpu, metal, options);
    }
    if (shouldRun(options, "training")) {
        try benchmarkTraining(allocator, options);
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
        \\  training      CPU Network.trainBatch loops
        \\  tiny_gpt      CPU TinyGPT forward passes
        \\  quantization  CPU uniform and TurboQuant encode/decode/error loops
        \\
    , .{});
}

fn shouldRun(options: Options, suite: []const u8) bool {
    const filter = options.filter orelse return true;
    return std.mem.indexOf(u8, suite, filter) != null;
}

fn createMetalBackend(allocator: Allocator) ?BackendInstance {
    const ptr = nn.createMetalBackend(allocator) catch return null;
    return BackendInstance{ .Metal = ptr };
}

fn printHeader(options: Options, metal_available: bool) void {
    std.debug.print("# zig-nn benchmark suite\n", .{});
    std.debug.print("# mode={s} quick={} metal_available={}\n", .{ modeName(), options.quick, metal_available });
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

fn benchmarkTraining(allocator: Allocator, options: Options) !void {
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
    };

    for (cases, 0..) |case, index| {
        if (options.quick and index > 0) break;

        var network = try makeTrainingNetwork(allocator, case.input_size, case.hidden_size, case.output_size);
        defer network.deinit();

        var inputs = try Matrix.init(allocator, case.samples, case.input_size);
        defer inputs.deinit();
        var targets = try Matrix.init(allocator, case.samples, case.output_size);
        defer targets.deinit();
        fillCpuMatrix(&inputs, 101);
        fillCpuMatrix(&targets, 137);

        const timing = try timeTrainingLoop(
            &network,
            inputs,
            targets,
            case.steps,
            quickCount(options, case.warmups),
            quickCount(options, case.iterations),
        );
        printResult("training", case.name, "cpu", timing, 0.0, "ok");
        printSkipped("training", case.name, "metal", timing.warmups, timing.iterations, "skipped_cpu_only_path");
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
    const cases = [_]struct {
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
    };

    for (cases, 0..) |case, index| {
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
