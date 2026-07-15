const std = @import("std");
const nn = @import("nn");

const InferenceService = nn.InferenceService;
const http = std.http;
const json = std.json;
const net = std.Io.net;
const testing = std.testing;

const max_body_bytes = 64 * 1024;

const ServerOptions = struct {
    host: []const u8 = "127.0.0.1",
    port: u16 = 8080,
    model_path: []const u8 = "xor_model.bin",
};

const Request = struct {
    input: []f64,
    batch_size: u32 = 1,
};

const Response = struct {
    prediction: []const f64,
    confidence: f64,

    pub fn writeJson(self: Response, writer: anytype) !void {
        try writer.writeAll("{\"prediction\":[");
        for (self.prediction, 0..) |value, i| {
            if (i > 0) try writer.writeAll(",");
            try writer.print("{d}", .{value});
        }
        try writer.writeAll("],\"confidence\":");
        try writer.print("{d}", .{self.confidence});
        try writer.writeAll("}");
    }
};

const ApiError = struct {
    status: http.Status = .bad_request,
    code: []const u8,
    message: []const u8,
};

pub fn main(init: std.process.Init) !void {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    const io = std.Options.debug_io;

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writerStreaming(io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};

    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const options = parseArgs(args[1..]) catch |err| {
        try printUsage(stdout);
        if (err == error.HelpRequested) return;
        return err;
    };

    var inference = try InferenceService.init(allocator, options.model_path);
    defer inference.deinit();

    const address = try net.IpAddress.parse(options.host, options.port);
    var server = try address.listen(io, .{ .reuse_address = true });
    defer server.deinit(io);

    try stdout.print("Server listening on http://{s}:{}\n", .{ options.host, options.port });
    try stdout.print("Model: {s}\n", .{options.model_path});
    try stdout.flush();

    while (true) {
        const stream = try server.accept(io);
        defer stream.close(io);

        handleConnection(allocator, &inference, stream) catch |err| {
            std.debug.print("request failed: {}\n", .{err});
        };
    }
}

fn parseArgs(args: []const []const u8) !ServerOptions {
    var options: ServerOptions = .{};
    var i: usize = 0;

    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            return error.HelpRequested;
        } else if (std.mem.eql(u8, arg, "--model")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.model_path = args[i];
        } else if (std.mem.startsWith(u8, arg, "--model=")) {
            options.model_path = arg["--model=".len..];
        } else if (std.mem.eql(u8, arg, "--host")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.host = args[i];
        } else if (std.mem.startsWith(u8, arg, "--host=")) {
            options.host = arg["--host=".len..];
        } else if (std.mem.eql(u8, arg, "--port")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.port = try std.fmt.parseInt(u16, args[i], 10);
        } else if (std.mem.startsWith(u8, arg, "--port=")) {
            options.port = try std.fmt.parseInt(u16, arg["--port=".len..], 10);
        } else {
            return error.UnknownArgument;
        }
    }

    return options;
}

fn printUsage(writer: anytype) !void {
    try writer.writeAll(
        \\Usage: zig build run_serving -- [options]
        \\
        \\Options:
        \\  --model <path>   Model file to load (default: xor_model.bin)
        \\  --host <host>    Host to bind (default: 127.0.0.1)
        \\  --port <port>    Port to bind (default: 8080)
        \\
    );
}

fn handleConnection(
    allocator: std.mem.Allocator,
    inference: *InferenceService,
    stream: anytype,
) !void {
    const io = std.Options.debug_io;
    var read_buf: [8192]u8 = undefined;
    var write_buf: [8192]u8 = undefined;
    var stream_reader = stream.reader(io, &read_buf);
    var stream_writer = stream.writer(io, &write_buf);
    var http_server = http.Server.init(&stream_reader.interface, &stream_writer.interface);

    var request = try http_server.receiveHead();
    const target_path = pathOnly(request.head.target);

    if (request.head.method == .OPTIONS) {
        return respondJson(&request, "{}", .no_content);
    }

    if (!std.mem.eql(u8, target_path, "/predict")) {
        return respondApiError(&request, .{
            .status = .not_found,
            .code = "not_found",
            .message = "Unknown endpoint.",
        });
    }

    if (request.head.method != .POST) {
        return respondApiError(&request, .{
            .status = .method_not_allowed,
            .code = "unsupported_method",
            .message = "Use POST for /predict.",
        });
    }

    var body_buf: [max_body_bytes]u8 = undefined;
    const body = readBody(&request, body_buf[0..]) catch |err| switch (err) {
        error.RequestBodyTooLarge => {
            return respondApiError(&request, .{
                .status = .payload_too_large,
                .code = "request_body_too_large",
                .message = "Request body exceeds 64 KiB.",
            });
        },
        else => return err,
    };

    const parsed = json.parseFromSlice(Request, allocator, body, .{}) catch {
        return respondApiError(&request, .{
            .status = .bad_request,
            .code = "invalid_json",
            .message = "Request body must be valid prediction JSON.",
        });
    };
    defer parsed.deinit();

    const response_body = predictionResponse(allocator, inference, parsed.value) catch |err| switch (err) {
        error.UnsupportedBatchSize => {
            return respondApiError(&request, .{
                .status = .bad_request,
                .code = "unsupported_batch_size",
                .message = "batch_size must be 1 for /predict.",
            });
        },
        error.InvalidInputDimensions => {
            return respondApiError(&request, .{
                .status = .bad_request,
                .code = "invalid_input_dimensions",
                .message = "Input length does not match the model input size.",
            });
        },
        else => return err,
    };
    defer allocator.free(response_body);

    return respondJson(&request, response_body, .ok);
}

fn readBody(request: *http.Server.Request, body_buf: []u8) ![]const u8 {
    const content_length = request.head.content_length orelse 0;
    if (content_length > max_body_bytes) return error.RequestBodyTooLarge;

    const reader = request.readerExpectNone(body_buf);
    return reader.take(@intCast(content_length));
}

fn predictionResponse(allocator: std.mem.Allocator, inference: *InferenceService, request: Request) ![]u8 {
    if (request.batch_size != 1) return error.UnsupportedBatchSize;
    if (request.input.len != try inference.getInputSize()) return error.InvalidInputDimensions;

    const prediction = try inference.predict(request.input);
    defer allocator.free(prediction);

    var body: std.Io.Writer.Allocating = .init(allocator);
    errdefer body.deinit();
    const response = Response{
        .prediction = prediction,
        .confidence = if (prediction.len > 0) prediction[0] else 0.0,
    };
    try response.writeJson(&body.writer);
    return body.toOwnedSlice();
}

fn pathOnly(target: []const u8) []const u8 {
    if (std.mem.indexOfScalar(u8, target, '?')) |index| {
        return target[0..index];
    }
    return target;
}

fn respondJson(request: *http.Server.Request, body: []const u8, status: http.Status) !void {
    try request.respond(body, .{
        .status = status,
        .extra_headers = &[_]http.Header{
            .{ .name = "Content-Type", .value = "application/json" },
            .{ .name = "Access-Control-Allow-Origin", .value = "*" },
            .{ .name = "Access-Control-Allow-Methods", .value = "POST, OPTIONS" },
            .{ .name = "Access-Control-Allow-Headers", .value = "Content-Type" },
        },
    });
}

fn respondApiError(request: *http.Server.Request, err: ApiError) !void {
    var body_buffer: [256]u8 = undefined;
    var writer = std.Io.Writer.fixed(&body_buffer);
    try writeApiErrorJson(&writer, err);
    try respondJson(request, writer.buffered(), err.status);
}

fn writeApiErrorJson(writer: anytype, err: ApiError) !void {
    try writer.writeAll("{\"error\":{\"code\":\"");
    try writer.writeAll(err.code);
    try writer.writeAll("\",\"message\":\"");
    try writer.writeAll(err.message);
    try writer.writeAll("\"}}");
}

test "request parsing" {
    const allocator = testing.allocator;
    const json_str =
        \\{"input": [0.0, 1.0], "batch_size": 1}
    ;

    const parsed = try json.parseFromSlice(
        Request,
        allocator,
        json_str,
        .{},
    );
    defer parsed.deinit();
    const request = parsed.value;

    try testing.expectEqual(@as(usize, 2), request.input.len);
    try testing.expectEqual(@as(f64, 0.0), request.input[0]);
    try testing.expectEqual(@as(f64, 1.0), request.input[1]);
    try testing.expectEqual(@as(u32, 1), request.batch_size);
}

test "request parsing defaults batch size" {
    const parsed = try json.parseFromSlice(
        Request,
        testing.allocator,
        \\{"input": [0.0, 1.0]}
    ,
        .{},
    );
    defer parsed.deinit();

    try testing.expectEqual(@as(u32, 1), parsed.value.batch_size);
}

test "response formatting" {
    const prediction = [_]f64{0.95};

    const response_data = Response{
        .prediction = &prediction,
        .confidence = 0.95,
    };

    var buffer: [256]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buffer);
    try response_data.writeJson(&writer);
    const json_str = writer.buffered();

    const parsed = try json.parseFromSlice(
        Response,
        testing.allocator,
        json_str,
        .{},
    );
    defer parsed.deinit();

    try testing.expectEqual(@as(usize, 1), parsed.value.prediction.len);
    try testing.expectApproxEqAbs(@as(f64, 0.95), parsed.value.prediction[0], 0.000001);
    try testing.expectApproxEqAbs(@as(f64, 0.95), parsed.value.confidence, 0.000001);
}

test "argument parsing supports model host and port" {
    const args = [_][]const u8{ "--model", "model.bin", "--host=0.0.0.0", "--port", "9000" };
    const options = try parseArgs(args[0..]);

    try testing.expectEqualStrings("model.bin", options.model_path);
    try testing.expectEqualStrings("0.0.0.0", options.host);
    try testing.expectEqual(@as(u16, 9000), options.port);
}

test "pathOnly strips query strings" {
    try testing.expectEqualStrings("/predict", pathOnly("/predict?trace=1"));
    try testing.expectEqualStrings("/predict", pathOnly("/predict"));
}

test "error response formatting" {
    var buffer: [256]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buffer);
    try writeApiErrorJson(&writer, .{
        .code = "invalid_json",
        .message = "Request body must be valid prediction JSON.",
    });

    try testing.expectEqualStrings(
        "{\"error\":{\"code\":\"invalid_json\",\"message\":\"Request body must be valid prediction JSON.\"}}",
        writer.buffered(),
    );
}
