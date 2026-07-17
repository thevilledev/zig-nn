const std = @import("std");
const spectral = @import("spectral.zig");

pub const target_sample_rate: u32 = 16_000;
pub const target_sample_count: usize = 16_000;
pub const default_frame_length: usize = 400;
pub const default_frame_step: usize = 160;
pub const default_fft_size: usize = 512;
pub const default_mel_bins: usize = 24;
pub const default_time_bins: usize = 16;
pub const default_feature_count: usize = default_mel_bins * default_time_bins;

pub const AudioError = error{
    InvalidWav,
    TruncatedWav,
    MissingFormatChunk,
    MissingDataChunk,
    UnsupportedWavEncoding,
    UnsupportedChannelCount,
    InvalidSampleRate,
    ClipTooLong,
    InvalidFrontendConfig,
    InvalidSampleCount,
};

pub const Waveform = struct {
    allocator: std.mem.Allocator,
    samples: []f32,
    sample_rate: u32,

    pub fn deinit(self: *Waveform) void {
        self.allocator.free(self.samples);
        self.* = undefined;
    }
};

const WavFormat = struct {
    encoding: u16,
    channels: u16,
    sample_rate: u32,
    block_align: u16,
    bits_per_sample: u16,
};

/// Decodes little-endian RIFF/WAVE PCM16 data into normalized mono samples.
/// Mono and stereo files are accepted; stereo channels are averaged.
pub fn decodeWav(allocator: std.mem.Allocator, bytes: []const u8) !Waveform {
    if (bytes.len < 12 or
        !std.mem.eql(u8, bytes[0..4], "RIFF") or
        !std.mem.eql(u8, bytes[8..12], "WAVE"))
    {
        return AudioError.InvalidWav;
    }

    const riff_size = readU32(bytes, 4);
    if (@as(u64, riff_size) + 8 > bytes.len) return AudioError.TruncatedWav;

    var format: ?WavFormat = null;
    var data: ?[]const u8 = null;
    var offset: usize = 12;
    while (offset < bytes.len) {
        if (bytes.len - offset < 8) return AudioError.TruncatedWav;
        const chunk_id = bytes[offset .. offset + 4];
        const chunk_size: usize = readU32(bytes, offset + 4);
        offset += 8;
        if (chunk_size > bytes.len - offset) return AudioError.TruncatedWav;
        const chunk = bytes[offset .. offset + chunk_size];

        if (std.mem.eql(u8, chunk_id, "fmt ")) {
            if (chunk.len < 16) return AudioError.InvalidWav;
            format = .{
                .encoding = readU16(chunk, 0),
                .channels = readU16(chunk, 2),
                .sample_rate = readU32(chunk, 4),
                .block_align = readU16(chunk, 12),
                .bits_per_sample = readU16(chunk, 14),
            };
        } else if (std.mem.eql(u8, chunk_id, "data")) {
            data = chunk;
        }

        offset += chunk_size;
        if (chunk_size & 1 == 1) {
            if (offset >= bytes.len) return AudioError.TruncatedWav;
            offset += 1;
        }
    }

    const wav_format = format orelse return AudioError.MissingFormatChunk;
    const pcm = data orelse return AudioError.MissingDataChunk;
    if (wav_format.encoding != 1 or wav_format.bits_per_sample != 16) {
        return AudioError.UnsupportedWavEncoding;
    }
    if (wav_format.channels != 1 and wav_format.channels != 2) {
        return AudioError.UnsupportedChannelCount;
    }
    if (wav_format.sample_rate == 0) return AudioError.InvalidSampleRate;
    const expected_block_align = wav_format.channels * @sizeOf(i16);
    if (wav_format.block_align != expected_block_align or pcm.len % expected_block_align != 0) {
        return AudioError.InvalidWav;
    }

    const frame_count = pcm.len / expected_block_align;
    const samples = try allocator.alloc(f32, frame_count);
    errdefer allocator.free(samples);
    for (samples, 0..) |*sample, frame| {
        var total: f32 = 0;
        for (0..wav_format.channels) |channel| {
            const sample_offset = frame * expected_block_align + channel * @sizeOf(i16);
            const raw = readI16(pcm, sample_offset);
            total += @as(f32, @floatFromInt(raw)) / 32768.0;
        }
        sample.* = total / @as(f32, @floatFromInt(wav_format.channels));
    }
    return .{
        .allocator = allocator,
        .samples = samples,
        .sample_rate = wav_format.sample_rate,
    };
}

/// Resamples a decoded clip to 16 kHz and pads it to exactly one second.
/// Longer clips are rejected so inference cannot silently discard speech.
pub fn prepareOneSecond(allocator: std.mem.Allocator, waveform: Waveform) ![]f32 {
    if (waveform.sample_rate == 0) return AudioError.InvalidSampleRate;
    const numerator = try std.math.mul(usize, waveform.samples.len, target_sample_rate);
    const resampled_count = (numerator + waveform.sample_rate / 2) / waveform.sample_rate;
    if (resampled_count > target_sample_count) return AudioError.ClipTooLong;

    const result = try allocator.alloc(f32, target_sample_count);
    @memset(result, 0);
    if (resampled_count == 0 or waveform.samples.len == 0) return result;
    if (waveform.sample_rate == target_sample_rate) {
        @memcpy(result[0..waveform.samples.len], waveform.samples);
        return result;
    }
    if (resampled_count == 1 or waveform.samples.len == 1) {
        result[0] = waveform.samples[0];
        return result;
    }

    const source_span = @as(f64, @floatFromInt(waveform.samples.len - 1));
    const output_span = @as(f64, @floatFromInt(resampled_count - 1));
    for (result[0..resampled_count], 0..) |*sample, index| {
        const position = @as(f64, @floatFromInt(index)) * source_span / output_span;
        const left: usize = @intFromFloat(@floor(position));
        const right = @min(left + 1, waveform.samples.len - 1);
        const fraction: f32 = @floatCast(position - @as(f64, @floatFromInt(left)));
        sample.* = waveform.samples[left] * (1 - fraction) + waveform.samples[right] * fraction;
    }
    return result;
}

pub const LogMelConfig = struct {
    sample_rate: u32 = target_sample_rate,
    sample_count: usize = target_sample_count,
    frame_length: usize = default_frame_length,
    frame_step: usize = default_frame_step,
    fft_size: usize = default_fft_size,
    mel_bins: usize = default_mel_bins,
    time_bins: usize = default_time_bins,
    minimum_hz: f32 = 20,
    maximum_hz: f32 = 8_000,

    pub fn featureCount(self: LogMelConfig) !usize {
        try self.validate();
        return std.math.mul(usize, self.mel_bins, self.time_bins) catch
            AudioError.InvalidFrontendConfig;
    }

    fn validate(self: LogMelConfig) !void {
        if (self.sample_rate == 0 or self.sample_count < self.frame_length or
            self.frame_length < 2 or self.frame_step == 0 or self.fft_size < self.frame_length or
            self.fft_size & (self.fft_size - 1) != 0 or self.mel_bins == 0 or self.time_bins == 0 or
            !std.math.isFinite(self.minimum_hz) or !std.math.isFinite(self.maximum_hz) or
            self.minimum_hz < 0 or self.maximum_hz <= self.minimum_hz or
            self.maximum_hz > @as(f32, @floatFromInt(self.sample_rate)) / 2)
        {
            return AudioError.InvalidFrontendConfig;
        }
        const frame_count = 1 + (self.sample_count - self.frame_length) / self.frame_step;
        if (self.time_bins > frame_count) return AudioError.InvalidFrontendConfig;
        _ = std.math.mul(usize, self.mel_bins, self.time_bins) catch
            return AudioError.InvalidFrontendConfig;
    }
};

/// Reusable host-side log-mel frontend. Window and filter weights are cached
/// so a dataset can be featurized without rebuilding them for every clip.
pub const LogMelFrontend = struct {
    allocator: std.mem.Allocator,
    config: LogMelConfig,
    window: []f32,
    filters: []f32,

    pub fn init(allocator: std.mem.Allocator, config: LogMelConfig) !LogMelFrontend {
        try config.validate();
        const window = try allocator.alloc(f32, config.frame_length);
        errdefer allocator.free(window);
        for (window, 0..) |*value, index| {
            const phase = 2.0 * std.math.pi * @as(f64, @floatFromInt(index)) /
                @as(f64, @floatFromInt(config.frame_length - 1));
            value.* = @floatCast(0.5 - 0.5 * @cos(phase));
        }

        const spectrum_bins = config.fft_size / 2 + 1;
        const filter_count = std.math.mul(usize, config.mel_bins, spectrum_bins) catch
            return AudioError.InvalidFrontendConfig;
        const filters = try allocator.alloc(f32, filter_count);
        errdefer allocator.free(filters);
        @memset(filters, 0);
        buildMelFilters(config, filters);
        return .{
            .allocator = allocator,
            .config = config,
            .window = window,
            .filters = filters,
        };
    }

    pub fn deinit(self: *LogMelFrontend) void {
        self.allocator.free(self.window);
        self.allocator.free(self.filters);
        self.* = undefined;
    }

    pub fn extract(self: LogMelFrontend, samples: []const f32) ![]f32 {
        if (samples.len != self.config.sample_count) return AudioError.InvalidSampleCount;
        const features = try self.allocator.alloc(f32, try self.config.featureCount());
        errdefer self.allocator.free(features);
        @memset(features, 0);

        const frame_count = 1 + (samples.len - self.config.frame_length) / self.config.frame_step;
        const spectrum_bins = self.config.fft_size / 2 + 1;
        const real = try self.allocator.alloc(f32, self.config.fft_size);
        defer self.allocator.free(real);
        const imaginary = try self.allocator.alloc(f32, self.config.fft_size);
        defer self.allocator.free(imaginary);
        const time_counts = try self.allocator.alloc(usize, self.config.time_bins);
        defer self.allocator.free(time_counts);
        @memset(time_counts, 0);

        for (0..frame_count) |frame| {
            @memset(real, 0);
            @memset(imaginary, 0);
            const start = frame * self.config.frame_step;
            for (0..self.config.frame_length) |index| {
                real[index] = samples[start + index] * self.window[index];
            }
            try spectral.fftInPlace(f32, real, imaginary);

            const time_bin = @min(frame * self.config.time_bins / frame_count, self.config.time_bins - 1);
            time_counts[time_bin] += 1;
            for (0..self.config.mel_bins) |mel_bin| {
                var energy: f32 = 0;
                const filter = self.filters[mel_bin * spectrum_bins .. (mel_bin + 1) * spectrum_bins];
                for (filter, 0..) |weight, frequency_bin| {
                    const power = real[frequency_bin] * real[frequency_bin] +
                        imaginary[frequency_bin] * imaginary[frequency_bin];
                    energy += weight * power;
                }
                features[time_bin * self.config.mel_bins + mel_bin] += @log(1 + energy);
            }
        }

        for (0..self.config.time_bins) |time_bin| {
            const divisor = @as(f32, @floatFromInt(time_counts[time_bin]));
            for (features[time_bin * self.config.mel_bins .. (time_bin + 1) * self.config.mel_bins]) |*value| {
                value.* /= divisor;
            }
        }
        return features;
    }
};

fn buildMelFilters(config: LogMelConfig, filters: []f32) void {
    const spectrum_bins = config.fft_size / 2 + 1;
    const minimum_mel = hzToMel(config.minimum_hz);
    const maximum_mel = hzToMel(config.maximum_hz);
    for (0..config.mel_bins) |mel_bin| {
        const left = melToHz(minimum_mel + (maximum_mel - minimum_mel) *
            @as(f32, @floatFromInt(mel_bin)) / @as(f32, @floatFromInt(config.mel_bins + 1)));
        const center = melToHz(minimum_mel + (maximum_mel - minimum_mel) *
            @as(f32, @floatFromInt(mel_bin + 1)) / @as(f32, @floatFromInt(config.mel_bins + 1)));
        const right = melToHz(minimum_mel + (maximum_mel - minimum_mel) *
            @as(f32, @floatFromInt(mel_bin + 2)) / @as(f32, @floatFromInt(config.mel_bins + 1)));
        for (0..spectrum_bins) |frequency_bin| {
            const hz = @as(f32, @floatFromInt(frequency_bin * config.sample_rate)) /
                @as(f32, @floatFromInt(config.fft_size));
            const weight = if (hz >= left and hz <= center)
                (hz - left) / @max(center - left, 1e-12)
            else if (hz > center and hz <= right)
                (right - hz) / @max(right - center, 1e-12)
            else
                0;
            filters[mel_bin * spectrum_bins + frequency_bin] = weight;
        }
    }
}

fn hzToMel(hz: f32) f32 {
    return 1127.0 * @log(1.0 + hz / 700.0);
}

fn melToHz(mel: f32) f32 {
    return 700.0 * (@exp(mel / 1127.0) - 1.0);
}

fn readU16(bytes: []const u8, offset: usize) u16 {
    return @as(u16, bytes[offset]) | @as(u16, bytes[offset + 1]) << 8;
}

fn readI16(bytes: []const u8, offset: usize) i16 {
    return @bitCast(readU16(bytes, offset));
}

fn readU32(bytes: []const u8, offset: usize) u32 {
    return @as(u32, bytes[offset]) |
        @as(u32, bytes[offset + 1]) << 8 |
        @as(u32, bytes[offset + 2]) << 16 |
        @as(u32, bytes[offset + 3]) << 24;
}

fn writeU16(bytes: []u8, offset: usize, value: u16) void {
    bytes[offset] = @truncate(value);
    bytes[offset + 1] = @truncate(value >> 8);
}

fn writeU32(bytes: []u8, offset: usize, value: u32) void {
    bytes[offset] = @truncate(value);
    bytes[offset + 1] = @truncate(value >> 8);
    bytes[offset + 2] = @truncate(value >> 16);
    bytes[offset + 3] = @truncate(value >> 24);
}

fn testWav(comptime channels: u16, sample_rate: u32, comptime samples: []const i16) [44 + samples.len * 2]u8 {
    var bytes: [44 + samples.len * 2]u8 = @splat(0);
    @memcpy(bytes[0..4], "RIFF");
    writeU32(&bytes, 4, bytes.len - 8);
    @memcpy(bytes[8..12], "WAVE");
    @memcpy(bytes[12..16], "fmt ");
    writeU32(&bytes, 16, 16);
    writeU16(&bytes, 20, 1);
    writeU16(&bytes, 22, channels);
    writeU32(&bytes, 24, sample_rate);
    writeU32(&bytes, 28, sample_rate * channels * 2);
    writeU16(&bytes, 32, channels * 2);
    writeU16(&bytes, 34, 16);
    @memcpy(bytes[36..40], "data");
    writeU32(&bytes, 40, samples.len * 2);
    for (samples, 0..) |sample, index| writeU16(&bytes, 44 + index * 2, @bitCast(sample));
    return bytes;
}

test "WAV decoder normalizes mono and downmixes stereo PCM16" {
    const testing = std.testing;
    const mono_bytes = testWav(1, 16_000, &.{ -32768, 0, 32767 });
    var mono = try decodeWav(testing.allocator, &mono_bytes);
    defer mono.deinit();
    try testing.expectEqual(@as(u32, 16_000), mono.sample_rate);
    try testing.expectApproxEqAbs(@as(f32, -1), mono.samples[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0), mono.samples[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 32767.0 / 32768.0), mono.samples[2], 1e-6);

    const stereo_bytes = testWav(2, 8_000, &.{ 32767, -32768, 16384, 16384 });
    var stereo = try decodeWav(testing.allocator, &stereo_bytes);
    defer stereo.deinit();
    try testing.expectEqual(@as(usize, 2), stereo.samples.len);
    try testing.expectApproxEqAbs(@as(f32, -1.0 / 65536.0), stereo.samples[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.5), stereo.samples[1], 1e-6);
}

test "WAV decoder skips padded unknown chunks and rejects unsupported data" {
    const testing = std.testing;
    var bytes: [56]u8 = @splat(0);
    @memcpy(bytes[0..4], "RIFF");
    writeU32(&bytes, 4, bytes.len - 8);
    @memcpy(bytes[8..12], "WAVE");
    @memcpy(bytes[12..16], "fmt ");
    writeU32(&bytes, 16, 16);
    writeU16(&bytes, 20, 1);
    writeU16(&bytes, 22, 1);
    writeU32(&bytes, 24, 16_000);
    writeU32(&bytes, 28, 32_000);
    writeU16(&bytes, 32, 2);
    writeU16(&bytes, 34, 16);
    @memcpy(bytes[36..40], "JUNK");
    writeU32(&bytes, 40, 1);
    bytes[44] = 99;
    @memcpy(bytes[46..50], "data");
    writeU32(&bytes, 50, 2);
    writeU16(&bytes, 54, @bitCast(@as(i16, 1234)));
    var waveform = try decodeWav(testing.allocator, &bytes);
    defer waveform.deinit();
    try testing.expectEqual(@as(usize, 1), waveform.samples.len);

    var unsupported = bytes;
    writeU16(&unsupported, 20, 3);
    try testing.expectError(AudioError.UnsupportedWavEncoding, decodeWav(testing.allocator, &unsupported));
    try testing.expectError(AudioError.TruncatedWav, decodeWav(testing.allocator, bytes[0 .. bytes.len - 1]));
}

test "one-second preparation resamples, pads, and rejects long clips" {
    const testing = std.testing;
    var source = [_]f32{ 0, 1 };
    const waveform: Waveform = .{ .allocator = testing.allocator, .samples = &source, .sample_rate = 8_000 };
    const prepared = try prepareOneSecond(testing.allocator, waveform);
    defer testing.allocator.free(prepared);
    try testing.expectEqual(@as(usize, target_sample_count), prepared.len);
    try testing.expectApproxEqAbs(@as(f32, 0), prepared[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.0 / 3.0), prepared[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 2.0 / 3.0), prepared[2], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1), prepared[3], 1e-6);
    try testing.expectEqual(@as(f32, 0), prepared[4]);

    const long_samples = try testing.allocator.alloc(f32, target_sample_count + 1);
    defer testing.allocator.free(long_samples);
    const long: Waveform = .{ .allocator = testing.allocator, .samples = long_samples, .sample_rate = target_sample_rate };
    try testing.expectError(AudioError.ClipTooLong, prepareOneSecond(testing.allocator, long));
}

test "FFT localizes a sinusoid" {
    const testing = std.testing;
    var real: [default_fft_size]f32 = undefined;
    var imaginary: [default_fft_size]f32 = @splat(0);
    const expected_bin: usize = 32;
    for (&real, 0..) |*value, index| {
        value.* = @floatCast(@sin(2.0 * std.math.pi * @as(f64, @floatFromInt(expected_bin * index)) /
            @as(f64, @floatFromInt(default_fft_size))));
    }
    try spectral.fftInPlace(f32, &real, &imaginary);
    var peak: usize = 0;
    var peak_power: f32 = 0;
    for (real[0 .. default_fft_size / 2], imaginary[0 .. default_fft_size / 2], 0..) |re, im, index| {
        const power = re * re + im * im;
        if (power > peak_power) {
            peak = index;
            peak_power = power;
        }
    }
    try testing.expectEqual(expected_bin, peak);
}

test "log-mel frontend is deterministic and finite for silence" {
    const testing = std.testing;
    var frontend = try LogMelFrontend.init(testing.allocator, .{});
    defer frontend.deinit();
    const samples = try testing.allocator.alloc(f32, target_sample_count);
    defer testing.allocator.free(samples);
    @memset(samples, 0);
    const first = try frontend.extract(samples);
    defer testing.allocator.free(first);
    const second = try frontend.extract(samples);
    defer testing.allocator.free(second);
    try testing.expectEqual(@as(usize, default_feature_count), first.len);
    try testing.expectEqualSlices(f32, first, second);
    for (first) |value| {
        try testing.expect(std.math.isFinite(value));
        try testing.expectEqual(@as(f32, 0), value);
    }
}

test "log-mel frontend rejects degenerate windows and empty time bins" {
    try std.testing.expectError(
        AudioError.InvalidFrontendConfig,
        LogMelFrontend.init(std.testing.allocator, .{ .frame_length = 1 }),
    );
    try std.testing.expectError(
        AudioError.InvalidFrontendConfig,
        LogMelFrontend.init(std.testing.allocator, .{ .time_bins = std.math.maxInt(usize) }),
    );
    try std.testing.expectError(
        AudioError.InvalidFrontendConfig,
        LogMelFrontend.init(std.testing.allocator, .{
            .sample_count = std.math.maxInt(usize),
            .frame_length = 2,
            .frame_step = 1,
            .fft_size = 2,
            .mel_bins = 2,
            .time_bins = std.math.maxInt(usize) - 1,
        }),
    );
}
