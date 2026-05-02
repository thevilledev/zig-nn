const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const QuantizationError = error{
    EmptyInput,
    InvalidBitWidth,
    DimensionMismatch,
    InvalidPayloadBits,
};

pub const VectorMetrics = struct {
    element_count: usize,
    mse: f64,
    max_abs_error: f64,
    cosine_similarity: f64,
    inner_product_error: f64,
    original_bits: usize,
    quantized_bits: usize,
    compression_ratio: f64,
};

pub const UniformQuantizedVector = struct {
    allocator: Allocator,
    codes: []u16,
    bits: u8,
    min: f64,
    scale: f64,

    pub fn deinit(self: UniformQuantizedVector) void {
        self.allocator.free(self.codes);
    }

    pub fn payloadBits(self: UniformQuantizedVector) usize {
        return self.codes.len * @as(usize, self.bits);
    }

    pub fn decode(self: UniformQuantizedVector, allocator: Allocator) ![]f64 {
        return decodeCodes(allocator, self.codes, self.min, self.scale);
    }
};

pub const TurboQuantizedVector = struct {
    allocator: Allocator,
    codes: []u16,
    bits: u8,
    min: f64,
    scale: f64,
    seed: u64,
    original_len: usize,
    transformed_len: usize,

    pub fn deinit(self: TurboQuantizedVector) void {
        self.allocator.free(self.codes);
    }

    pub fn payloadBits(self: TurboQuantizedVector) usize {
        return self.codes.len * @as(usize, self.bits);
    }

    pub fn decode(self: TurboQuantizedVector, allocator: Allocator) ![]f64 {
        const rotated = try decodeCodes(allocator, self.codes, self.min, self.scale);
        defer allocator.free(rotated);

        return inverseRandomHadamard(
            allocator,
            rotated,
            self.original_len,
            self.seed,
        );
    }
};

pub fn encodeUniformScalar(
    allocator: Allocator,
    input: []const f64,
    bits: u8,
) !UniformQuantizedVector {
    try validateInput(input, bits);

    const range = findRange(input);
    const scale = quantizationScale(range.min, range.max, bits);
    const codes = try encodeCodes(allocator, input, range.min, scale, bits);
    errdefer allocator.free(codes);

    return UniformQuantizedVector{
        .allocator = allocator,
        .codes = codes,
        .bits = bits,
        .min = range.min,
        .scale = scale,
    };
}

pub fn encodeTurboQuant(
    allocator: Allocator,
    input: []const f64,
    bits: u8,
    seed: u64,
) !TurboQuantizedVector {
    try validateInput(input, bits);

    const rotated = try randomHadamardTransform(allocator, input, seed);
    defer allocator.free(rotated);

    const range = findRange(rotated);
    const scale = quantizationScale(range.min, range.max, bits);
    const codes = try encodeCodes(allocator, rotated, range.min, scale, bits);
    errdefer allocator.free(codes);

    return TurboQuantizedVector{
        .allocator = allocator,
        .codes = codes,
        .bits = bits,
        .min = range.min,
        .scale = scale,
        .seed = seed,
        .original_len = input.len,
        .transformed_len = rotated.len,
    };
}

pub fn measureVectorError(
    original: []const f64,
    decoded: []const f64,
    query: ?[]const f64,
    quantized_bits: usize,
) !VectorMetrics {
    if (original.len == 0) return QuantizationError.EmptyInput;
    if (original.len != decoded.len) return QuantizationError.DimensionMismatch;
    if (quantized_bits == 0) return QuantizationError.InvalidPayloadBits;
    if (query) |q| {
        if (q.len != original.len) return QuantizationError.DimensionMismatch;
    }

    var sum_sq: f64 = 0.0;
    var max_abs_error: f64 = 0.0;
    var original_norm: f64 = 0.0;
    var decoded_norm: f64 = 0.0;
    var cross_norm: f64 = 0.0;
    var original_inner: f64 = 0.0;
    var decoded_inner: f64 = 0.0;

    for (original, decoded, 0..) |source, reconstructed, i| {
        const diff = source - reconstructed;
        const abs_error = @abs(diff);
        sum_sq += diff * diff;
        max_abs_error = @max(max_abs_error, abs_error);
        original_norm += source * source;
        decoded_norm += reconstructed * reconstructed;
        cross_norm += source * reconstructed;

        const q = if (query) |provided| provided[i] else source;
        original_inner += q * source;
        decoded_inner += q * reconstructed;
    }

    const original_bits = original.len * 64;
    return VectorMetrics{
        .element_count = original.len,
        .mse = sum_sq / @as(f64, @floatFromInt(original.len)),
        .max_abs_error = max_abs_error,
        .cosine_similarity = cosineFromNorms(original_norm, decoded_norm, cross_norm),
        .inner_product_error = @abs(original_inner - decoded_inner),
        .original_bits = original_bits,
        .quantized_bits = quantized_bits,
        .compression_ratio = @as(f64, @floatFromInt(original_bits)) /
            @as(f64, @floatFromInt(quantized_bits)),
    };
}

fn validateInput(input: []const f64, bits: u8) QuantizationError!void {
    if (input.len == 0) return QuantizationError.EmptyInput;
    if (bits == 0 or bits > 16) return QuantizationError.InvalidBitWidth;
}

fn quantizationLevels(bits: u8) u32 {
    const shift: u5 = @intCast(bits);
    return (@as(u32, 1) << shift) - 1;
}

fn quantizationScale(min: f64, max: f64, bits: u8) f64 {
    const levels = @as(f64, @floatFromInt(quantizationLevels(bits)));
    const range = max - min;
    return if (range == 0.0) 0.0 else range / levels;
}

fn encodeCodes(
    allocator: Allocator,
    input: []const f64,
    min: f64,
    scale: f64,
    bits: u8,
) ![]u16 {
    const codes = try allocator.alloc(u16, input.len);
    errdefer allocator.free(codes);

    const levels = @as(f64, @floatFromInt(quantizationLevels(bits)));
    for (input, 0..) |value, i| {
        if (scale == 0.0) {
            codes[i] = 0;
            continue;
        }

        var normalized = @round((value - min) / scale);
        normalized = @max(0.0, @min(levels, normalized));
        codes[i] = @as(u16, @intFromFloat(normalized));
    }

    return codes;
}

fn decodeCodes(
    allocator: Allocator,
    codes: []const u16,
    min: f64,
    scale: f64,
) ![]f64 {
    const decoded = try allocator.alloc(f64, codes.len);
    errdefer allocator.free(decoded);

    for (codes, 0..) |code, i| {
        decoded[i] = min + @as(f64, @floatFromInt(code)) * scale;
    }

    return decoded;
}

const Range = struct {
    min: f64,
    max: f64,
};

fn findRange(input: []const f64) Range {
    var min = input[0];
    var max = input[0];
    for (input[1..]) |value| {
        min = @min(min, value);
        max = @max(max, value);
    }
    return .{ .min = min, .max = max };
}

fn cosineFromNorms(original_norm: f64, decoded_norm: f64, cross_norm: f64) f64 {
    if (original_norm == 0.0 and decoded_norm == 0.0) return 1.0;
    if (original_norm == 0.0 or decoded_norm == 0.0) return 0.0;
    return cross_norm / @sqrt(original_norm * decoded_norm);
}

fn nextPowerOfTwo(n: usize) usize {
    var result: usize = 1;
    while (result < n) : (result *= 2) {}
    return result;
}

fn randomHadamardTransform(
    allocator: Allocator,
    input: []const f64,
    seed: u64,
) ![]f64 {
    const transformed_len = nextPowerOfTwo(input.len);
    const transformed = try allocator.alloc(f64, transformed_len);
    errdefer allocator.free(transformed);

    @memset(transformed, 0.0);
    @memcpy(transformed[0..input.len], input);

    applyRandomSigns(transformed, seed);
    hadamardInPlace(transformed);
    normalizeHadamard(transformed);

    return transformed;
}

fn inverseRandomHadamard(
    allocator: Allocator,
    rotated: []const f64,
    original_len: usize,
    seed: u64,
) ![]f64 {
    var transformed = try allocator.alloc(f64, rotated.len);
    defer allocator.free(transformed);
    @memcpy(transformed, rotated);

    hadamardInPlace(transformed);
    normalizeHadamard(transformed);
    applyRandomSigns(transformed, seed);

    const decoded = try allocator.alloc(f64, original_len);
    errdefer allocator.free(decoded);
    @memcpy(decoded, transformed[0..original_len]);
    return decoded;
}

fn applyRandomSigns(values: []f64, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();

    for (values) |*value| {
        if (random.boolean()) {
            value.* = -value.*;
        }
    }
}

fn hadamardInPlace(values: []f64) void {
    var width: usize = 1;
    while (width < values.len) : (width *= 2) {
        var start: usize = 0;
        while (start < values.len) : (start += width * 2) {
            for (0..width) |offset| {
                const a = values[start + offset];
                const b = values[start + offset + width];
                values[start + offset] = a + b;
                values[start + offset + width] = a - b;
            }
        }
    }
}

fn normalizeHadamard(values: []f64) void {
    const scale = 1.0 / @sqrt(@as(f64, @floatFromInt(values.len)));
    for (values) |*value| {
        value.* *= scale;
    }
}

test "uniform scalar quantization decodes bounded values" {
    const allocator = testing.allocator;
    const input = [_]f64{ -1.0, -0.25, 0.25, 1.0 };

    const encoded = try encodeUniformScalar(allocator, &input, 3);
    defer encoded.deinit();

    const decoded = try encoded.decode(allocator);
    defer allocator.free(decoded);

    try testing.expectEqual(input.len, decoded.len);
    try testing.expectApproxEqAbs(@as(f64, -1.0), decoded[0], 0.0001);
    try testing.expectApproxEqAbs(@as(f64, 1.0), decoded[3], 0.0001);
}

test "vector metrics detect exact reconstruction" {
    const input = [_]f64{ 1.0, -2.0, 3.0 };

    const metrics = try measureVectorError(&input, &input, null, 24);

    try testing.expectEqual(@as(usize, 3), metrics.element_count);
    try testing.expectApproxEqAbs(@as(f64, 0.0), metrics.mse, 0.0001);
    try testing.expectApproxEqAbs(@as(f64, 1.0), metrics.cosine_similarity, 0.0001);
    try testing.expectApproxEqAbs(@as(f64, 8.0), metrics.compression_ratio, 0.0001);
}

test "turbo quantized vectors are deterministic" {
    const allocator = testing.allocator;
    const input = [_]f64{ 0.25, -0.75, 1.5, 0.0, 2.25 };

    const first = try encodeTurboQuant(allocator, &input, 4, 1234);
    defer first.deinit();
    const second = try encodeTurboQuant(allocator, &input, 4, 1234);
    defer second.deinit();

    try testing.expectEqualSlices(u16, first.codes, second.codes);
    try testing.expectEqual(nextPowerOfTwo(input.len), first.transformed_len);

    const decoded = try first.decode(allocator);
    defer allocator.free(decoded);

    try testing.expectEqual(input.len, decoded.len);
    const metrics = try measureVectorError(&input, decoded, null, first.payloadBits());
    try testing.expect(metrics.mse < 0.02);
}
