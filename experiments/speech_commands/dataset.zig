const std = @import("std");
const nn = @import("nn");
const model = @import("model.zig");

pub const Split = enum { train, validation, testing };

pub const Entry = struct {
    path: []u8,
    label: usize,
    split: Split,
};

pub const Normalizer = struct {
    mean: [model.feature_count]f32,
    stddev: [model.feature_count]f32,
};

pub const Dataset = struct {
    allocator: std.mem.Allocator,
    entries: []Entry,
    features: []f32,

    pub fn load(
        allocator: std.mem.Allocator,
        data_dir: []const u8,
        seed: u64,
        frontend: nn.Audio.LogMelFrontend,
    ) !Dataset {
        const io = std.Options.debug_io;
        const root = try std.Io.Dir.cwd().openDir(io, data_dir, .{});
        defer root.close(io);
        var entries: std.ArrayList(Entry) = .empty;
        defer entries.deinit(allocator);
        errdefer for (entries.items) |entry| allocator.free(entry.path);

        for (model.labels, 0..) |label, label_index| {
            const label_dir = try root.openDir(io, label, .{ .iterate = true });
            defer label_dir.close(io);
            var iterator = label_dir.iterate();
            while (try iterator.next(io)) |item| {
                if (item.kind != .file or !std.mem.endsWith(u8, item.name, ".wav")) continue;
                const path = try std.fs.path.join(allocator, &.{ data_dir, label, item.name });
                errdefer allocator.free(path);
                try entries.append(allocator, .{
                    .path = path,
                    .label = label_index,
                    .split = splitForFilename(item.name, seed),
                });
            }
        }
        if (entries.items.len == 0) return error.EmptyDataset;
        std.mem.sort(Entry, entries.items, {}, lessThanEntry);

        const owned_entries = try entries.toOwnedSlice(allocator);
        errdefer {
            for (owned_entries) |entry| allocator.free(entry.path);
            allocator.free(owned_entries);
        }
        const features = try allocator.alloc(f32, owned_entries.len * model.feature_count);
        errdefer allocator.free(features);
        for (owned_entries, 0..) |entry, index| {
            const bytes = try std.Io.Dir.cwd().readFileAlloc(io, entry.path, allocator, .limited(4 * 1024 * 1024));
            defer allocator.free(bytes);
            var waveform = try nn.Audio.decodeWav(allocator, bytes);
            defer waveform.deinit();
            const prepared = try nn.Audio.prepareOneSecond(allocator, waveform);
            defer allocator.free(prepared);
            const extracted = try frontend.extract(prepared);
            defer allocator.free(extracted);
            @memcpy(features[index * model.feature_count .. (index + 1) * model.feature_count], extracted);
        }

        var result: Dataset = .{
            .allocator = allocator,
            .entries = owned_entries,
            .features = features,
        };
        try result.validateSplits();
        return result;
    }

    pub fn deinit(self: *Dataset) void {
        for (self.entries) |entry| self.allocator.free(entry.path);
        self.allocator.free(self.entries);
        self.allocator.free(self.features);
        self.* = undefined;
    }

    pub fn indices(self: Dataset, allocator: std.mem.Allocator, split: Split) ![]usize {
        var result: std.ArrayList(usize) = .empty;
        defer result.deinit(allocator);
        for (self.entries, 0..) |entry, index| {
            if (entry.split == split) try result.append(allocator, index);
        }
        return result.toOwnedSlice(allocator);
    }

    pub fn featureRow(self: Dataset, index: usize) []f32 {
        return self.features[index * model.feature_count .. (index + 1) * model.feature_count];
    }

    pub fn computeNormalizer(self: Dataset, train_indices: []const usize) !Normalizer {
        if (train_indices.len == 0) return error.EmptyTrainingSplit;
        var result: Normalizer = .{ .mean = @splat(0), .stddev = @splat(0) };
        for (train_indices) |index| {
            for (self.featureRow(index), 0..) |value, feature| result.mean[feature] += value;
        }
        const count = @as(f32, @floatFromInt(train_indices.len));
        for (&result.mean) |*value| value.* /= count;
        for (train_indices) |index| {
            for (self.featureRow(index), 0..) |value, feature| {
                const difference = value - result.mean[feature];
                result.stddev[feature] += difference * difference;
            }
        }
        for (&result.stddev) |*value| value.* = @max(@sqrt(value.* / count), 1e-5);
        return result;
    }

    pub fn applyNormalizer(self: *Dataset, normalizer: Normalizer) void {
        for (self.features, 0..) |*value, index| {
            const feature = index % model.feature_count;
            value.* = (value.* - normalizer.mean[feature]) / normalizer.stddev[feature];
        }
    }

    fn validateSplits(self: Dataset) !void {
        var counts = [_]usize{0} ** 3;
        for (self.entries) |entry| counts[@intFromEnum(entry.split)] += 1;
        for (counts) |count| if (count == 0) return error.EmptyDatasetSplit;
    }
};

pub fn splitForFilename(filename: []const u8, seed: u64) Split {
    const basename = std.fs.path.basename(filename);
    const marker = std.mem.indexOf(u8, basename, "_nohash_");
    const speaker = if (marker) |index| basename[0..index] else basename;
    const bucket = std.hash.Wyhash.hash(seed, speaker) % 10;
    return if (bucket < 8) .train else if (bucket == 8) .validation else .testing;
}

fn lessThanEntry(_: void, left: Entry, right: Entry) bool {
    return std.mem.order(u8, left.path, right.path) == .lt;
}

test "speaker split is deterministic and prevents cross-split leakage" {
    const testing = std.testing;
    const first = splitForFilename("alice_nohash_0.wav", 42);
    try testing.expectEqual(first, splitForFilename("alice_nohash_7.wav", 42));
    try testing.expectEqual(first, splitForFilename("alice_nohash_12.wav", 42));
    try testing.expectEqual(splitForFilename("bob_nohash_0.wav", 42), splitForFilename("bob_nohash_2.wav", 42));
    try testing.expectEqual(first, splitForFilename("alice_nohash_0.wav", 42));
}
