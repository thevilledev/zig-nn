const std = @import("std");
const nn = @import("nn");
const InferenceService = nn.InferenceService;
const json = std.json;
const http = std.http;
const net = std.net;
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
            try std.fmt.format(writer, "{d}", .{value});
        }
        try writer.writeAll("],\"confidence\":");
        try std.fmt.format(writer, "{d}", .{self.confidence});
        try writer.writeAll("}");
    }
};

pub fn main() !void {
    // Initialize allocator
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize inference service
    var inference = try InferenceService.init(allocator, "xor_model.bin");
    defer inference.deinit();

    // Create server
    const address = try net.Address.resolveIp("0.0.0.0", 8080);
    var server = try address.listen(.{ .reuse_port = true });
    defer server.deinit();

    std.debug.print("Server listening on http://127.0.0.1:8080\n", .{});

    while (true) {
        var connection = try server.accept();
        defer connection.stream.close();

        var buf: [8192]u8 = undefined;
        var http_server = http.Server.init(connection, &buf);

        // Handle request
        var request = try http_server.receiveHead();
        std.debug.print("request size: {}\n{s}\n", .{ request.head_end, buf[0..request.head_end] });

        // Find the start of the body (after the double newline)
        var body_start: usize = request.head_end;
        // Skip any remaining newlines after headers
        while (body_start < buf.len and (buf[body_start] == '\r' or buf[body_start] == '\n')) {
            body_start += 1;
        }

        // Get Content-Length
        const content_length = if (request.head.content_length) |l| l else 0;

        // Debug print positions and content
        std.debug.print("head_end: {}, body_start: {}, content_length: {}\n", .{ request.head_end, body_start, content_length });
        std.debug.print("Body content (hex): ", .{});
        for (buf[body_start .. body_start + content_length]) |byte| {
            std.debug.print("{x:0>2} ", .{byte});
        }
        std.debug.print("\nBody content: {s}\n", .{buf[body_start .. body_start + content_length]});

        // Parse JSON request
        const parsed = try std.json.parseFromSlice(
            Request,
            allocator,
            buf[body_start .. body_start + content_length],
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
        var fbs = std.io.fixedBufferStream(&json_buffer);
        try response_data.writeJson(fbs.writer());
        const json_response = fbs.getWritten();

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
    var fbs = std.io.fixedBufferStream(&buffer);
    try response_data.writeJson(fbs.writer());
    const json_str = fbs.getWritten();

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
