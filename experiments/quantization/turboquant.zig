const std = @import("std");
const nn = @import("nn");

const Quantization = nn.Quantization;

const dimension = 384;
const benchmark_trials = 128;
const compression_dimension = 16384;
const compression_trials = 16;
const data_seed: u64 = 0x5442_5155_414e_5431;
const rotation_seed: u64 = 0x9e37_79b9_7f4a_7c15;

fn nowNs() i96 {
    return std.Io.Clock.awake.now(std.Options.debug_io).toNanoseconds();
}

pub fn main() !void {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const source = try allocator.alloc(f64, dimension);
    defer allocator.free(source);
    const query = try allocator.alloc(f64, dimension);
    defer allocator.free(query);
    fillExperimentVectors(source, query, data_seed);

    std.debug.print("TurboQuant paper lab\n", .{});
    std.debug.print("single vector: {d} f64 values\n\n", .{dimension});
    std.debug.print(
        "{s: <18} {s: >4} {s: >12} {s: >12} {s: >10} {s: >10}\n",
        .{ "method", "bits", "mse", "max error", "cosine", "ratio" },
    );

    try runUniform(allocator, source, query, 8);
    try runTurboQuant(allocator, source, query, 8);
    try runUniform(allocator, source, query, 4);
    try runTurboQuant(allocator, source, query, 4);

    std.debug.print("\nAggregate benchmark: {d} deterministic outlier vectors\n", .{benchmark_trials});
    std.debug.print(
        "{s: >4} {s: >12} {s: >12} {s: >11} {s: >12} {s: >12} {s: >11} {s: >10} {s: >10}\n",
        .{
            "bits",
            "base mse",
            "rot mse",
            "mse gain",
            "base inner",
            "rot inner",
            "inner gain",
            "base us",
            "rot us",
        },
    );
    try runBenchmark(allocator, 8);
    try runBenchmark(allocator, 4);

    std.debug.print(
        "\nCompression benchmark: TurboQuant 4-bit vs previous uniform 8-bit\n",
        .{},
    );
    std.debug.print(
        "{s: >7} {s: >6} {s: >6} {s: >11} {s: >12} {s: >12} {s: >11} {s: >12} {s: >12} {s: >11} {s: >10} {s: >10}\n",
        .{
            "vectors",
            "dims",
            "bits",
            "payload win",
            "base mse",
            "tq mse",
            "mse gain",
            "base inner",
            "tq inner",
            "inner gain",
            "base us",
            "tq us",
        },
    );
    try runCompressionBenchmark(allocator);
}

fn fillExperimentVectors(source: []f64, query: []f64, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();

    for (source, query, 0..) |*value, *q, i| {
        const index = @as(f64, @floatFromInt(i));
        const slow_wave = std.math.sin(index * 0.031) * 1.7;
        const fast_wave = std.math.cos(index * 0.173) * 0.45;
        const noise = (random.float(f64) - 0.5) * 0.25;
        const spike: f64 = if (i % 97 == 0)
            if ((i / 97) % 2 == 0) 8.0 else -7.5
        else
            0.0;

        value.* = slow_wave + fast_wave + noise + spike;
        q.* = std.math.sin(index * 0.097) + std.math.cos(index * 0.011) * 0.5;
    }
}

fn fillSparseOutlierVectors(source: []f64, query: []f64, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    const spike_index = seed % source.len;

    for (source, query, 0..) |*value, *q, i| {
        const index = @as(f64, @floatFromInt(i));
        const baseline = std.math.sin(index * 0.013) * 0.8 +
            std.math.cos(index * 0.047) * 0.35;
        const noise = (random.float(f64) - 0.5) * 0.08;
        const spike: f64 = if (i == spike_index) 2048.0 else 0.0;

        value.* = baseline + noise + spike;
        q.* = std.math.sin(index * 0.005) * 0.7 +
            std.math.cos(index * 0.019) * 0.3;
    }
}

const BenchTotals = struct {
    uniform_mse: f64 = 0.0,
    rotated_mse: f64 = 0.0,
    uniform_inner_error: f64 = 0.0,
    rotated_inner_error: f64 = 0.0,
    uniform_total_ns: f64 = 0.0,
    rotated_total_ns: f64 = 0.0,
};

const CompressionTotals = struct {
    uniform_mse: f64 = 0.0,
    turbo_mse: f64 = 0.0,
    uniform_inner_error: f64 = 0.0,
    turbo_inner_error: f64 = 0.0,
    uniform_total_ns: f64 = 0.0,
    turbo_total_ns: f64 = 0.0,
    uniform_payload_bits: usize = 0,
    turbo_payload_bits: usize = 0,
};

fn runUniform(
    allocator: std.mem.Allocator,
    source: []const f64,
    query: []const f64,
    bits: u8,
) !void {
    const encoded = try Quantization.encodeUniformScalar(allocator, source, bits);
    defer encoded.deinit();

    const decoded = try encoded.decode(allocator);
    defer allocator.free(decoded);

    const metrics = try Quantization.measureVectorError(
        source,
        decoded,
        query,
        encoded.payloadBits(),
    );
    printMetrics("uniform scalar", bits, metrics);
}

fn runTurboQuant(
    allocator: std.mem.Allocator,
    source: []const f64,
    query: []const f64,
    bits: u8,
) !void {
    const encoded = try Quantization.encodeTurboQuant(
        allocator,
        source,
        bits,
        rotation_seed,
    );
    defer encoded.deinit();

    const decoded = try encoded.decode(allocator);
    defer allocator.free(decoded);

    const metrics = try Quantization.measureVectorError(
        source,
        decoded,
        query,
        encoded.payloadBits(),
    );
    printMetrics("rotated scalar", bits, metrics);
}

fn runBenchmark(allocator: std.mem.Allocator, bits: u8) !void {
    var totals = BenchTotals{};

    for (0..benchmark_trials) |trial| {
        const source = try allocator.alloc(f64, dimension);
        defer allocator.free(source);
        const query = try allocator.alloc(f64, dimension);
        defer allocator.free(query);

        fillExperimentVectors(source, query, data_seed +% @as(u64, @intCast(trial)));

        var start = nowNs();
        const uniform = try Quantization.encodeUniformScalar(allocator, source, bits);
        const uniform_decoded = try uniform.decode(allocator);
        const uniform_ns = nowNs() - start;
        defer allocator.free(uniform_decoded);
        defer uniform.deinit();

        start = nowNs();
        const rotated = try Quantization.encodeTurboQuant(
            allocator,
            source,
            bits,
            rotation_seed +% @as(u64, @intCast(trial)),
        );
        const rotated_decoded = try rotated.decode(allocator);
        const rotated_ns = nowNs() - start;
        defer allocator.free(rotated_decoded);
        defer rotated.deinit();

        const uniform_metrics = try Quantization.measureVectorError(
            source,
            uniform_decoded,
            query,
            uniform.payloadBits(),
        );
        const rotated_metrics = try Quantization.measureVectorError(
            source,
            rotated_decoded,
            query,
            rotated.payloadBits(),
        );

        totals.uniform_mse += uniform_metrics.mse;
        totals.rotated_mse += rotated_metrics.mse;
        totals.uniform_inner_error += uniform_metrics.inner_product_error;
        totals.rotated_inner_error += rotated_metrics.inner_product_error;
        totals.uniform_total_ns += @as(f64, @floatFromInt(uniform_ns));
        totals.rotated_total_ns += @as(f64, @floatFromInt(rotated_ns));
    }

    const trials = @as(f64, @floatFromInt(benchmark_trials));
    const uniform_mse = totals.uniform_mse / trials;
    const rotated_mse = totals.rotated_mse / trials;
    const uniform_inner = totals.uniform_inner_error / trials;
    const rotated_inner = totals.rotated_inner_error / trials;

    std.debug.print(
        "{d: >4} {d: >12.6} {d: >12.6} {d: >10.1}% {d: >12.6} {d: >12.6} {d: >10.1}% {d: >10.2} {d: >10.2}\n",
        .{
            bits,
            uniform_mse,
            rotated_mse,
            percentGain(uniform_mse, rotated_mse),
            uniform_inner,
            rotated_inner,
            percentGain(uniform_inner, rotated_inner),
            totals.uniform_total_ns / trials / 1000.0,
            totals.rotated_total_ns / trials / 1000.0,
        },
    );
}

fn runCompressionBenchmark(allocator: std.mem.Allocator) !void {
    var totals = CompressionTotals{};

    for (0..compression_trials) |trial| {
        const source = try allocator.alloc(f64, compression_dimension);
        defer allocator.free(source);
        const query = try allocator.alloc(f64, compression_dimension);
        defer allocator.free(query);

        fillSparseOutlierVectors(
            source,
            query,
            data_seed +% @as(u64, @intCast(trial * 97)),
        );

        var start = nowNs();
        const uniform = try Quantization.encodeUniformScalar(allocator, source, 8);
        const uniform_decoded = try uniform.decode(allocator);
        const uniform_ns = nowNs() - start;
        defer allocator.free(uniform_decoded);
        defer uniform.deinit();

        start = nowNs();
        const turbo = try Quantization.encodeTurboQuant(
            allocator,
            source,
            4,
            rotation_seed +% @as(u64, @intCast(trial)),
        );
        const turbo_decoded = try turbo.decode(allocator);
        const turbo_ns = nowNs() - start;
        defer allocator.free(turbo_decoded);
        defer turbo.deinit();

        const uniform_metrics = try Quantization.measureVectorError(
            source,
            uniform_decoded,
            query,
            uniform.payloadBits(),
        );
        const turbo_metrics = try Quantization.measureVectorError(
            source,
            turbo_decoded,
            query,
            turbo.payloadBits(),
        );

        totals.uniform_mse += uniform_metrics.mse;
        totals.turbo_mse += turbo_metrics.mse;
        totals.uniform_inner_error += uniform_metrics.inner_product_error;
        totals.turbo_inner_error += turbo_metrics.inner_product_error;
        totals.uniform_total_ns += @as(f64, @floatFromInt(uniform_ns));
        totals.turbo_total_ns += @as(f64, @floatFromInt(turbo_ns));
        totals.uniform_payload_bits += uniform_metrics.quantized_bits;
        totals.turbo_payload_bits += turbo_metrics.quantized_bits;
    }

    const trials = @as(f64, @floatFromInt(compression_trials));
    const uniform_mse = totals.uniform_mse / trials;
    const turbo_mse = totals.turbo_mse / trials;
    const uniform_inner = totals.uniform_inner_error / trials;
    const turbo_inner = totals.turbo_inner_error / trials;
    const payload_win = @as(f64, @floatFromInt(totals.uniform_payload_bits)) /
        @as(f64, @floatFromInt(totals.turbo_payload_bits));

    std.debug.print(
        "{d: >7} {d: >6} {s: >6} {d: >10.2}x {d: >12.6} {d: >12.6} {d: >10.1}% {d: >12.6} {d: >12.6} {d: >10.1}% {d: >10.2} {d: >10.2}\n",
        .{
            compression_trials,
            compression_dimension,
            "8->4",
            payload_win,
            uniform_mse,
            turbo_mse,
            percentGain(uniform_mse, turbo_mse),
            uniform_inner,
            turbo_inner,
            percentGain(uniform_inner, turbo_inner),
            totals.uniform_total_ns / trials / 1000.0,
            totals.turbo_total_ns / trials / 1000.0,
        },
    );
}

fn percentGain(baseline: f64, candidate: f64) f64 {
    if (baseline == 0.0) return 0.0;
    return ((baseline - candidate) / baseline) * 100.0;
}

fn printMetrics(label: []const u8, bits: u8, metrics: Quantization.VectorMetrics) void {
    std.debug.print(
        "{s: <18} {d: >4} {d: >12.6} {d: >12.6} {d: >10.6} {d: >10.2}x\n",
        .{
            label,
            bits,
            metrics.mse,
            metrics.max_abs_error,
            metrics.cosine_similarity,
            metrics.compression_ratio,
        },
    );
}
