const std = @import("std");

pub const SpectralError = error{
    BufferSizeMismatch,
    FeatureCountOverflow,
    InvalidCoordinate,
    InvalidLength,
    NonFiniteValue,
    NonPowerOfTwo,
};

fn validateFloat(comptime Float: type) void {
    if (Float != f32 and Float != f64) {
        @compileError("spectral operations support f32 and f64");
    }
}

/// In-place radix-2 Cooley-Tukey FFT over separate real and imaginary arrays.
pub fn fftInPlace(comptime Float: type, real: []Float, imaginary: []Float) SpectralError!void {
    comptime validateFloat(Float);
    if (real.len == 0 or real.len != imaginary.len) return error.InvalidLength;
    if (real.len & (real.len - 1) != 0) return error.NonPowerOfTwo;
    for (real, imaginary) |real_value, imaginary_value| {
        if (!std.math.isFinite(real_value) or !std.math.isFinite(imaginary_value)) {
            return error.NonFiniteValue;
        }
    }

    var reversed: usize = 0;
    for (1..real.len) |index| {
        var bit = real.len >> 1;
        while (reversed & bit != 0) : (bit >>= 1) reversed ^= bit;
        reversed ^= bit;
        if (index < reversed) {
            std.mem.swap(Float, &real[index], &real[reversed]);
            std.mem.swap(Float, &imaginary[index], &imaginary[reversed]);
        }
    }

    var length: usize = 2;
    while (true) {
        const angle = -2.0 * std.math.pi / @as(f64, @floatFromInt(length));
        const step_real: Float = @floatCast(@cos(angle));
        const step_imaginary: Float = @floatCast(@sin(angle));
        var start: usize = 0;
        while (start < real.len) : (start += length) {
            var twiddle_real: Float = 1;
            var twiddle_imaginary: Float = 0;
            for (0..length / 2) |offset| {
                const even = start + offset;
                const odd = even + length / 2;
                const odd_real = real[odd] * twiddle_real - imaginary[odd] * twiddle_imaginary;
                const odd_imaginary = real[odd] * twiddle_imaginary + imaginary[odd] * twiddle_real;
                real[odd] = real[even] - odd_real;
                imaginary[odd] = imaginary[even] - odd_imaginary;
                real[even] += odd_real;
                imaginary[even] += odd_imaginary;
                const next_real = twiddle_real * step_real - twiddle_imaginary * step_imaginary;
                twiddle_imaginary = twiddle_real * step_imaginary + twiddle_imaginary * step_real;
                twiddle_real = next_real;
            }
        }
        if (length == real.len) break;
        length *= 2;
    }
}

/// Returns the single-sided amplitude spectrum for a real, evenly sampled signal.
/// The output length must be `samples.len / 2 + 1`.
pub fn realAmplitudeSpectrum(
    allocator: std.mem.Allocator,
    comptime Float: type,
    samples: []const Float,
    output: []Float,
) (SpectralError || std.mem.Allocator.Error)!void {
    comptime validateFloat(Float);
    if (samples.len == 0) return error.InvalidLength;
    if (samples.len & (samples.len - 1) != 0) return error.NonPowerOfTwo;
    if (output.len != samples.len / 2 + 1) return error.BufferSizeMismatch;

    const real = try allocator.dupe(Float, samples);
    defer allocator.free(real);
    const imaginary = try allocator.alloc(Float, samples.len);
    defer allocator.free(imaginary);
    @memset(imaginary, 0);
    try fftInPlace(Float, real, imaginary);

    const divisor: Float = @floatFromInt(samples.len);
    for (output, 0..) |*amplitude, frequency| {
        const magnitude = @sqrt(real[frequency] * real[frequency] + imaginary[frequency] * imaginary[frequency]);
        const scale: Float = if (frequency == 0 or frequency * 2 == samples.len) 1 else 2;
        amplitude.* = scale * magnitude / divisor;
    }
}

/// Number of features emitted for a scalar coordinate: the coordinate itself,
/// followed by sine/cosine pairs for each requested harmonic band.
pub fn fourierFeatureCount(bands: usize) SpectralError!usize {
    const periodic = std.math.mul(usize, bands, 2) catch return error.FeatureCountOverflow;
    return std.math.add(usize, periodic, 1) catch return error.FeatureCountOverflow;
}

/// Encodes scalar coordinates as `[x, sin(2πx), cos(2πx), ...]`.
pub fn encodeFourierFeatures(
    comptime Float: type,
    coordinates: []const Float,
    bands: usize,
    output: []Float,
) SpectralError!void {
    comptime validateFloat(Float);
    const feature_count = try fourierFeatureCount(bands);
    const expected = std.math.mul(usize, coordinates.len, feature_count) catch
        return error.FeatureCountOverflow;
    if (output.len != expected) return error.BufferSizeMismatch;

    for (coordinates, 0..) |coordinate, row| {
        if (!std.math.isFinite(coordinate)) return error.InvalidCoordinate;
        const offset = row * feature_count;
        output[offset] = coordinate;
        for (1..bands + 1) |frequency| {
            const phase = 2 * std.math.pi * @as(Float, @floatFromInt(frequency)) * coordinate;
            output[offset + frequency * 2 - 1] = @sin(phase);
            output[offset + frequency * 2] = @cos(phase);
        }
    }
}

test "real amplitude spectrum recovers constant sine and cosine components" {
    const sample_count = 64;
    var samples: [sample_count]f64 = undefined;
    for (&samples, 0..) |*sample, index| {
        const x = @as(f64, @floatFromInt(index)) / sample_count;
        sample.* = 0.25 + 0.5 * @sin(2 * std.math.pi * 3 * x) + 0.75 * @cos(2 * std.math.pi * 7 * x);
    }
    var amplitudes: [sample_count / 2 + 1]f64 = undefined;
    try realAmplitudeSpectrum(std.testing.allocator, f64, &samples, &amplitudes);

    try std.testing.expectApproxEqAbs(@as(f64, 0.25), amplitudes[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), amplitudes[3], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 0.75), amplitudes[7], 1e-10);
    try std.testing.expect(amplitudes[2] < 1e-10);
}

test "Fourier feature encoding has a stable row-major order" {
    var encoded: [10]f32 = undefined;
    try encodeFourierFeatures(f32, &.{ 0, 0.25 }, 2, &encoded);

    try std.testing.expectEqual(@as(f32, 0), encoded[0]);
    try std.testing.expectApproxEqAbs(@as(f32, 0), encoded[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1), encoded[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1), encoded[6], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0), encoded[7], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0), encoded[8], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -1), encoded[9], 1e-6);
}

test "spectral operations validate shapes and finite values" {
    var real = [_]f32{ 1, 2, 3 };
    var imaginary = [_]f32{ 0, 0, 0 };
    try std.testing.expectError(error.NonPowerOfTwo, fftInPlace(f32, &real, &imaginary));

    var output: [3]f64 = undefined;
    try std.testing.expectError(
        error.BufferSizeMismatch,
        encodeFourierFeatures(f64, &.{ 0, 1 }, 1, &output),
    );
    var encoded: [3]f64 = undefined;
    try std.testing.expectError(
        error.InvalidCoordinate,
        encodeFourierFeatures(f64, &.{std.math.nan(f64)}, 1, &encoded),
    );
}
