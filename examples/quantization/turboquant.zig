const std = @import("std");
const nn = @import("nn");

const Quantization = nn.Quantization;

const dimension = 384;
const data_seed: u64 = 0x5442_5155_414e_5431;
const rotation_seed: u64 = 0x9e37_79b9_7f4a_7c15;

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const source = try allocator.alloc(f64, dimension);
    const query = try allocator.alloc(f64, dimension);
    fillExperimentVectors(source, query);

    std.debug.print("TurboQuant paper lab\n", .{});
    std.debug.print("vector length: {d} f64 values\n\n", .{dimension});
    std.debug.print(
        "{s: <18} {s: >4} {s: >12} {s: >12} {s: >10} {s: >10}\n",
        .{ "method", "bits", "mse", "max error", "cosine", "ratio" },
    );

    try runUniform(allocator, source, query, 8);
    try runTurboQuant(allocator, source, query, 8);
    try runUniform(allocator, source, query, 4);
    try runTurboQuant(allocator, source, query, 4);
}

fn fillExperimentVectors(source: []f64, query: []f64) void {
    var prng = std.Random.DefaultPrng.init(data_seed);
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
