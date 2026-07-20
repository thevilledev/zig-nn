const std = @import("std");
const nn = @import("nn");

const Options = struct {
    model_path: ?[]const u8 = null,
    input_json: ?[]const u8 = null,
    batch_size: usize = 1,
    backend: nn.DevicePreference = .auto,
};

pub fn main(init: std.process.Init) !void {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const options = parseArgs(args[1..]) catch |err| {
        printUsage();
        if (err == error.HelpRequested) return;
        return err;
    };
    const model_path = options.model_path orelse return error.MissingModel;
    const input_json = options.input_json orelse return error.MissingInput;
    const parsed = try std.json.parseFromSlice([]f32, allocator, input_json, .{});
    defer parsed.deinit();

    var session = try nn.Inference.DenseSession.init(allocator, model_path, .{ .device = options.backend });
    defer session.deinit();
    const output = try session.predictAlloc(parsed.value, options.batch_size);
    defer allocator.free(output);

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writerStreaming(std.Options.debug_io, &buffer);
    const writer = &stdout_writer.interface;
    defer writer.flush() catch {};
    try writer.writeAll("{\"backend\":\"");
    try writer.writeAll(@tagName(session.backendType()));
    try writer.writeAll("\",\"batch_size\":");
    try writer.print("{}", .{options.batch_size});
    try writer.writeAll(",\"output_size\":");
    try writer.print("{}", .{session.outputSize()});
    try writer.writeAll(",\"predictions\":[");
    for (output, 0..) |value, index| {
        if (index > 0) try writer.writeByte(',');
        try writer.print("{d}", .{value});
    }
    try writer.writeAll("]}\n");
}

fn parseArgs(args: []const []const u8) !Options {
    var options: Options = .{};
    var index: usize = 0;
    while (index < args.len) : (index += 1) {
        const arg = args[index];
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) return error.HelpRequested;
        if (std.mem.eql(u8, arg, "--model")) {
            index += 1;
            if (index >= args.len) return error.MissingArgument;
            options.model_path = args[index];
        } else if (std.mem.eql(u8, arg, "--input")) {
            index += 1;
            if (index >= args.len) return error.MissingArgument;
            options.input_json = args[index];
        } else if (std.mem.eql(u8, arg, "--batch-size")) {
            index += 1;
            if (index >= args.len) return error.MissingArgument;
            options.batch_size = try std.fmt.parseInt(usize, args[index], 10);
        } else if (std.mem.eql(u8, arg, "--backend")) {
            index += 1;
            if (index >= args.len) return error.MissingArgument;
            options.backend = try parseBackend(args[index]);
        } else {
            return error.UnknownArgument;
        }
    }
    if (options.batch_size == 0) return error.InvalidBatchSize;
    return options;
}

fn parseBackend(value: []const u8) !nn.DevicePreference {
    if (std.mem.eql(u8, value, "none") or std.mem.eql(u8, value, "cpu")) return .cpu;
    if (std.mem.eql(u8, value, "auto")) return .auto;
    if (std.mem.eql(u8, value, "metal")) return .metal;
    if (std.mem.eql(u8, value, "cuda")) return .cuda;
    if (std.mem.eql(u8, value, "rocm")) return .rocm;
    return error.InvalidBackend;
}

fn printUsage() void {
    std.debug.print(
        \\Usage: zig build run_inference_predict -- --model <path> --input <json-array> [options]
        \\
        \\Options:
        \\  --batch-size <n>  Number of flat input rows (default: 1)
        \\  --backend <name>  cpu, auto, metal, cuda, or rocm
        \\
    , .{});
}

test "predict arguments preserve explicit backend and batch" {
    const options = try parseArgs(&.{ "--model", "model.znn", "--input", "[1,2]", "--batch-size", "2", "--backend", "cpu" });
    try std.testing.expectEqual(@as(usize, 2), options.batch_size);
    try std.testing.expectEqual(nn.DevicePreference.cpu, options.backend);
}
