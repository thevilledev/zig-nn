const std = @import("std");

pub const Format = enum {
    human,
    ndjson,
};

pub fn parseFormat(value: []const u8) !Format {
    if (std.mem.eql(u8, value, "human")) return .human;
    if (std.mem.eql(u8, value, "ndjson")) return .ndjson;
    return error.UnknownOutputFormat;
}

pub fn emit(writer: *std.Io.Writer, value: anytype) !void {
    try std.json.Stringify.value(value, .{}, writer);
    try writer.writeByte('\n');
    try writer.flush();
}

pub fn shouldEmitMetric(step: usize, total: usize) bool {
    return shouldEmit(step, total, 100);
}

pub fn shouldEmitSnapshot(step: usize, total: usize) bool {
    return shouldEmit(step, total, 20);
}

fn shouldEmit(step: usize, total: usize, segments: usize) bool {
    if (total == 0) return false;
    if (step == 0 or step + 1 == total) return true;
    const interval = @max(@as(usize, 1), total / segments);
    return step % interval == 0;
}

test "format parsing is explicit" {
    try std.testing.expectEqual(Format.human, try parseFormat("human"));
    try std.testing.expectEqual(Format.ndjson, try parseFormat("ndjson"));
    try std.testing.expectError(error.UnknownOutputFormat, parseFormat("json"));
}

test "cadence includes first and last step" {
    try std.testing.expect(shouldEmitMetric(0, 1000));
    try std.testing.expect(shouldEmitMetric(999, 1000));
    try std.testing.expect(shouldEmitSnapshot(50, 1000));
    try std.testing.expect(!shouldEmitSnapshot(51, 1000));
}

test "emit writes one valid JSON record and flushes it" {
    var output: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer output.deinit();

    try emit(&output.writer, .{
        .v = 1,
        .type = "metric",
        .step = 3,
        .data = .{ .loss = 0.25 },
    });

    const bytes = output.writer.buffer[0..output.writer.end];
    try std.testing.expect(bytes.len > 1);
    try std.testing.expectEqual(@as(u8, '\n'), bytes[bytes.len - 1]);
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, bytes[0 .. bytes.len - 1], .{});
    defer parsed.deinit();
    try std.testing.expectEqualStrings("metric", parsed.value.object.get("type").?.string);
}
