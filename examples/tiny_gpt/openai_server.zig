const std = @import("std");
const http = std.http;
const json = std.json;
const net = std.Io.net;
const tiny = @import("tiny_gpt.zig");

const max_body_bytes = 64 * 1024;

const ServerOptions = struct {
    host: []const u8 = "127.0.0.1",
    port: u16 = 8080,
    model_path: ?[]const u8 = null,
    model_name: []const u8 = "tiny-gpt-zig",
    seed: u64 = 42,
    max_tokens: usize = 80,
    temperature: f64 = 1.0,
    top_k: usize = 8,
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

    var model = if (options.model_path) |path|
        try tiny.TinyGPT.loadFromFile(allocator, path)
    else
        try tiny.TinyGPT.init(allocator, .{}, options.seed);
    defer model.deinit();

    const address = try net.IpAddress.parse(options.host, options.port);
    var server = try address.listen(io, .{ .reuse_address = true });
    defer server.deinit(io);

    try stdout.print("Tiny GPT OpenAI-compatible server listening on http://{s}:{}\n", .{ options.host, options.port });
    if (options.model_path) |path| {
        try stdout.print("Model: {s} ({s})\n", .{ options.model_name, path });
    } else {
        try stdout.print("Model: {s} (seeded untrained checkpoint)\n", .{options.model_name});
    }
    try stdout.flush();

    while (true) {
        {
            const stream = try server.accept(io);
            defer stream.close(io);

            handleConnection(allocator, &model, options, stream) catch |err| {
                std.debug.print("request failed: {}\n", .{err});
            };
        }
    }
}

fn handleConnection(
    allocator: std.mem.Allocator,
    model: *tiny.TinyGPT,
    options: ServerOptions,
    stream: anytype,
) !void {
    const io = std.Options.debug_io;
    var read_buf: [8192]u8 = undefined;
    var write_buf: [8192]u8 = undefined;
    var stream_reader = stream.reader(io, &read_buf);
    var stream_writer = stream.writer(io, &write_buf);
    var http_server = http.Server.init(&stream_reader.interface, &stream_writer.interface);

    var request = try http_server.receiveHead();
    const target = request.head.target;
    const method = request.head.method;

    if (method == .OPTIONS) {
        return respondJson(&request, "{}", .no_content);
    }

    if (method == .GET and std.mem.eql(u8, pathOnly(target), "/v1/models")) {
        const response = try modelsResponse(allocator, options.model_name);
        defer allocator.free(response);
        return respondJson(&request, response, .ok);
    }

    if (method != .POST) {
        return respondError(&request, .method_not_allowed, "unsupported_method", "Use POST for completion endpoints.");
    }

    const path = try allocator.dupe(u8, pathOnly(target));
    defer allocator.free(path);

    var body_buf: [max_body_bytes]u8 = undefined;
    const body = try readBody(&request, body_buf[0..]);
    const parsed = json.parseFromSlice(json.Value, allocator, body, .{}) catch {
        return respondError(&request, .bad_request, "invalid_json", "Request body must be valid JSON.");
    };
    defer parsed.deinit();

    if (std.mem.eql(u8, path, "/v1/chat/completions")) {
        const response = chatCompletionResponse(allocator, model, options, parsed.value) catch |err| {
            return respondError(&request, .bad_request, "invalid_request", @errorName(err));
        };
        defer allocator.free(response);
        return respondJson(&request, response, .ok);
    }
    if (std.mem.eql(u8, path, "/v1/completions")) {
        const response = completionResponse(allocator, model, options, parsed.value) catch |err| {
            return respondError(&request, .bad_request, "invalid_request", @errorName(err));
        };
        defer allocator.free(response);
        return respondJson(&request, response, .ok);
    }

    return respondError(&request, .not_found, "not_found", "Unknown endpoint.");
}

fn readBody(request: *http.Server.Request, body_buf: []u8) ![]const u8 {
    const content_length = request.head.content_length orelse 0;
    if (content_length > max_body_bytes) return error.RequestBodyTooLarge;

    const reader = request.readerExpectNone(body_buf);
    return reader.take(@intCast(content_length));
}

fn completionResponse(
    allocator: std.mem.Allocator,
    model: *tiny.TinyGPT,
    options: ServerOptions,
    request_value: json.Value,
) ![]u8 {
    if (request_value != .object) return error.InvalidRequest;
    const object = request_value.object;
    const prompt = stringField(object, "prompt") orelse return error.MissingPrompt;
    const max_tokens = usizeField(object, "max_tokens") orelse options.max_tokens;
    const temperature = floatField(object, "temperature") orelse options.temperature;
    const top_k = usizeField(object, "top_k") orelse options.top_k;
    const response_model = stringField(object, "model") orelse options.model_name;

    const completion = try generateCompletion(allocator, model, prompt, max_tokens, temperature, top_k);
    defer allocator.free(completion);

    var out: std.Io.Writer.Allocating = .init(allocator);
    errdefer out.deinit();
    const writer = &out.writer;
    const created = unixSeconds();

    try writer.print("{{\"id\":\"cmpl-{d}\",\"object\":\"text_completion\",\"created\":{d},\"model\":", .{ created, created });
    try writeJsonString(writer, response_model);
    try writer.writeAll(",\"choices\":[{\"text\":");
    try writeJsonString(writer, completion);
    try writer.writeAll(",\"index\":0,\"finish_reason\":\"length\"}],\"usage\":{");
    try writer.print("\"prompt_tokens\":{},\"completion_tokens\":{},\"total_tokens\":{}", .{
        prompt.len,
        completion.len,
        prompt.len + completion.len,
    });
    try writer.writeAll("}}");

    return out.toOwnedSlice();
}

fn chatCompletionResponse(
    allocator: std.mem.Allocator,
    model: *tiny.TinyGPT,
    options: ServerOptions,
    request_value: json.Value,
) ![]u8 {
    if (request_value != .object) return error.InvalidRequest;
    const object = request_value.object;
    const messages_value = object.get("messages") orelse return error.MissingMessages;
    if (messages_value != .array) return error.InvalidMessages;

    const prompt = try promptFromMessages(allocator, messages_value.array.items);
    defer allocator.free(prompt);

    const max_tokens = usizeField(object, "max_tokens") orelse options.max_tokens;
    const temperature = floatField(object, "temperature") orelse options.temperature;
    const top_k = usizeField(object, "top_k") orelse options.top_k;
    const response_model = stringField(object, "model") orelse options.model_name;

    const completion = try generateCompletion(allocator, model, prompt, max_tokens, temperature, top_k);
    defer allocator.free(completion);

    var out: std.Io.Writer.Allocating = .init(allocator);
    errdefer out.deinit();
    const writer = &out.writer;
    const created = unixSeconds();

    try writer.print("{{\"id\":\"chatcmpl-{d}\",\"object\":\"chat.completion\",\"created\":{d},\"model\":", .{ created, created });
    try writeJsonString(writer, response_model);
    try writer.writeAll(",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":");
    try writeJsonString(writer, completion);
    try writer.writeAll("},\"finish_reason\":\"length\"}],\"usage\":{");
    try writer.print("\"prompt_tokens\":{},\"completion_tokens\":{},\"total_tokens\":{}", .{
        prompt.len,
        completion.len,
        prompt.len + completion.len,
    });
    try writer.writeAll("}}");

    return out.toOwnedSlice();
}

fn modelsResponse(allocator: std.mem.Allocator, model_name: []const u8) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    errdefer out.deinit();
    const writer = &out.writer;

    try writer.writeAll("{\"object\":\"list\",\"data\":[{\"id\":");
    try writeJsonString(writer, model_name);
    try writer.print(",\"object\":\"model\",\"created\":{d},\"owned_by\":\"zig-nn\"}}]}}", .{unixSeconds()});

    return out.toOwnedSlice();
}

fn generateCompletion(
    allocator: std.mem.Allocator,
    model: *tiny.TinyGPT,
    prompt: []const u8,
    max_tokens: usize,
    temperature: f64,
    top_k: usize,
) ![]u8 {
    const generated_tokens = try model.generateTokens(allocator, prompt, max_tokens, temperature, top_k);
    defer allocator.free(generated_tokens);

    const prompt_token_count = if (prompt.len == 0) 1 else prompt.len;
    return tiny.Tokenizer.decode(allocator, generated_tokens[prompt_token_count..]);
}

fn promptFromMessages(allocator: std.mem.Allocator, messages: []const json.Value) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    errdefer out.deinit();
    const writer = &out.writer;

    for (messages) |message| {
        if (message != .object) return error.InvalidMessages;
        const object = message.object;
        const role = stringField(object, "role") orelse "user";
        const content = stringField(object, "content") orelse return error.InvalidMessages;
        try writer.print("{s}: {s}\n", .{ role, content });
    }
    try writer.writeAll("assistant: ");

    return out.toOwnedSlice();
}

fn stringField(object: json.ObjectMap, name: []const u8) ?[]const u8 {
    const value = object.get(name) orelse return null;
    return switch (value) {
        .string => |text| text,
        else => null,
    };
}

fn usizeField(object: json.ObjectMap, name: []const u8) ?usize {
    const value = object.get(name) orelse return null;
    return switch (value) {
        .integer => |number| if (number >= 0) @intCast(number) else null,
        .float => |number| if (number >= 0.0) @intFromFloat(number) else null,
        else => null,
    };
}

fn floatField(object: json.ObjectMap, name: []const u8) ?f64 {
    const value = object.get(name) orelse return null;
    return switch (value) {
        .integer => |number| @floatFromInt(number),
        .float => |number| number,
        else => null,
    };
}

fn writeJsonString(writer: *std.Io.Writer, text: []const u8) !void {
    try json.Stringify.value(text, .{}, writer);
}

fn respondJson(request: *http.Server.Request, body: []const u8, status: http.Status) !void {
    try request.respond(body, .{
        .status = status,
        .extra_headers = &cors_headers,
    });
}

fn respondError(
    request: *http.Server.Request,
    status: http.Status,
    code: []const u8,
    message: []const u8,
) !void {
    var buffer: [512]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buffer);
    try writer.writeAll("{\"error\":{\"message\":");
    try writeJsonString(&writer, message);
    try writer.writeAll(",\"type\":\"invalid_request_error\",\"code\":");
    try writeJsonString(&writer, code);
    try writer.writeAll("}}");
    try respondJson(request, writer.buffered(), status);
}

const cors_headers = [_]http.Header{
    .{ .name = "Content-Type", .value = "application/json" },
    .{ .name = "Access-Control-Allow-Origin", .value = "*" },
    .{ .name = "Access-Control-Allow-Headers", .value = "Content-Type, Authorization" },
    .{ .name = "Access-Control-Allow-Methods", .value = "GET, POST, OPTIONS" },
};

fn pathOnly(target: []const u8) []const u8 {
    if (std.mem.indexOfScalar(u8, target, '?')) |idx| return target[0..idx];
    return target;
}

fn unixSeconds() i64 {
    return @divTrunc(std.Io.Clock.real.now(std.Options.debug_io).toMilliseconds(), 1000);
}

fn parseArgs(args: []const [:0]const u8) !ServerOptions {
    var options: ServerOptions = .{};
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--host")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.host = args[i];
        } else if (std.mem.eql(u8, arg, "--port")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.port = try std.fmt.parseInt(u16, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--model")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.model_path = args[i];
        } else if (std.mem.eql(u8, arg, "--model-name")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.model_name = args[i];
        } else if (std.mem.eql(u8, arg, "--seed")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.seed = try std.fmt.parseInt(u64, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--max-tokens")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.max_tokens = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--temperature")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.temperature = try std.fmt.parseFloat(f64, args[i]);
        } else if (std.mem.eql(u8, arg, "--top-k")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            options.top_k = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--help")) {
            return error.HelpRequested;
        } else {
            return error.UnknownArgument;
        }
    }
    return options;
}

fn printUsage(writer: anytype) !void {
    try writer.writeAll(
        \\Usage: zig build run_tiny_gpt_openai -- [options]
        \\
        \\Options:
        \\  --host <ip>          Bind host (default: 127.0.0.1)
        \\  --port <n>           Bind port (default: 8080)
        \\  --model <path>       TinyGPT checkpoint from `run_tiny_gpt -- --save-checkpoint`
        \\  --model-name <name>  Model id returned in OpenAI-compatible responses
        \\  --max-tokens <n>     Default completion length (default: 80)
        \\  --temperature <f>    Default sampling temperature (default: 1.0)
        \\  --top-k <n>          Default top-k sampler cutoff (default: 8)
        \\
        \\Endpoints:
        \\  GET  /v1/models
        \\  POST /v1/completions
        \\  POST /v1/chat/completions
        \\
    );
}

test "chat prompt builder formats messages" {
    const allocator = std.testing.allocator;
    const body =
        \\{"messages":[{"role":"system","content":"short"},{"role":"user","content":"hello"}]}
    ;
    const parsed = try json.parseFromSlice(json.Value, allocator, body, .{});
    defer parsed.deinit();

    const messages = parsed.value.object.get("messages").?.array.items;
    const prompt = try promptFromMessages(allocator, messages);
    defer allocator.free(prompt);

    try std.testing.expectEqualStrings("system: short\nuser: hello\nassistant: ", prompt);
}

test "pathOnly removes query string" {
    try std.testing.expectEqualStrings("/v1/models", pathOnly("/v1/models?x=1"));
    try std.testing.expectEqualStrings("/v1/completions", pathOnly("/v1/completions"));
}
