const std = @import("std");
const nn = @import("nn");
const InferenceService = nn.InferenceService;
const json = std.json;
const http = std.http;
const net = std.Io.net;
const testing = std.testing;

const Request = struct {
    input: []f64,
    batch_size: u32,
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

pub fn main() !void {
    // Initialize allocator
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    const io = std.Options.debug_io;

    // Initialize inference service
    var inference = try InferenceService.init(allocator, "xor_model.bin");
    defer inference.deinit();

    // Create server
    const address = try net.IpAddress.parse("0.0.0.0", 8080);
    var server = try address.listen(io, .{ .reuse_address = true });
    defer server.deinit(io);

    std.debug.print("Server listening on http://127.0.0.1:8080\n", .{});

    while (true) {
        const stream = try server.accept(io);
        defer stream.close(io);

        var read_buf: [8192]u8 = undefined;
        var write_buf: [8192]u8 = undefined;
        var stream_reader = stream.reader(io, &read_buf);
        var stream_writer = stream.writer(io, &write_buf);
        var http_server = http.Server.init(&stream_reader.interface, &stream_writer.interface);

        // Handle request
        var request = try http_server.receiveHead();
        std.debug.print("request size: {}\n{s}\n", .{ request.head_buffer.len, request.head_buffer });

        // Get Content-Length
        const content_length = if (request.head.content_length) |l| l else 0;
        if (content_length > read_buf.len) return error.RequestBodyTooLarge;

        var body_buf: [8192]u8 = undefined;
        const body_reader = request.readerExpectNone(&body_buf);
        const body = try body_reader.take(@intCast(content_length));

        std.debug.print("content_length: {}\n", .{content_length});
        std.debug.print("Body content: {s}\n", .{body});

        // Parse JSON request
        const parsed = try std.json.parseFromSlice(
            Request,
            allocator,
            body,
            .{},
        );
        defer parsed.deinit();
        const request_data = parsed.value;

        // Make prediction
        const prediction = try inference.predict(request_data.input);
        defer allocator.free(prediction);

        // Calculate confidence (using the first prediction value)
        const confidence = prediction[0];

        // Create response
        const response_data = Response{
            .prediction = prediction,
            .confidence = confidence,
        };

        // Serialize response to JSON
        var json_buffer: [4096]u8 = undefined;
        var json_writer = std.Io.Writer.fixed(&json_buffer);
        try response_data.writeJson(&json_writer);
        const json_response = json_writer.buffered();

        // Send response
        try request.respond(json_response, .{
            .extra_headers = &[_]http.Header{
                .{ .name = "Content-Type", .value = "application/json" },
                .{ .name = "Access-Control-Allow-Origin", .value = "*" },
            },
        });
    }
}

// Tests
test "request parsing" {
    const allocator = testing.allocator;
    const json_str =
        \\{"input": [0.0, 1.0], "batch_size": 1}
    ;

    const parsed = try std.json.parseFromSlice(
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

    // Print actual JSON for debugging
    std.debug.print("\nActual JSON: {s}\n", .{json_str});

    // Parse the JSON back to verify values
    const parsed = try json.parseFromSlice(
        Response,
        testing.allocator,
        json_str,
        .{},
    );
    defer parsed.deinit();

    // Check the actual values
    try testing.expectEqual(@as(usize, 1), parsed.value.prediction.len);
    try testing.expectApproxEqAbs(@as(f64, 0.95), parsed.value.prediction[0], 0.000001);
    try testing.expectApproxEqAbs(@as(f64, 0.95), parsed.value.confidence, 0.000001);
}
