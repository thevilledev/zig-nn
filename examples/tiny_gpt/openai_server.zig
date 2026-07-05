const std = @import("std");
const http = std.http;
const json = std.json;
const net = std.Io.net;
const tiny = @import("tiny_gpt.zig");

const max_body_bytes = 64 * 1024;
const system_fingerprint = "zig-nn-tiny-gpt";

const ServerOptions = struct {
    host: []const u8 = "127.0.0.1",
    port: u16 = 8080,
    model_path: ?[]const u8 = null,
    model_name: []const u8 = "tiny-gpt-zig",
    seed: u64 = 42,
    max_tokens: usize = 80,
    temperature: f64 = 1.0,
    top_k: usize = 8,
    allow_untrained: bool = false,
};

const ApiError = struct {
    status: http.Status = .bad_request,
    code: []const u8,
    message: []const u8,
    param: ?[]const u8 = null,
};

const RequestSettings = struct {
    model_name: []const u8,
    max_tokens: usize,
    temperature: f64,
    top_k: usize,
    stop_sequences: []const []const u8 = &.{},
    owns_stop_sequences: bool = false,

    fn deinit(self: *RequestSettings, allocator: std.mem.Allocator) void {
        if (self.owns_stop_sequences) allocator.free(self.stop_sequences);
    }
};

const GeneratedText = struct {
    text: []u8,
    finish_reason: []const u8,

    fn deinit(self: GeneratedText, allocator: std.mem.Allocator) void {
        allocator.free(self.text);
    }
};

const PromptList = struct {
    items: []const []const u8,
    owned_text: []const []u8,

    fn deinit(self: PromptList, allocator: std.mem.Allocator) void {
        for (self.owned_text) |text| allocator.free(text);
        allocator.free(self.owned_text);
        allocator.free(self.items);
    }
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

    if (options.model_path == null and !options.allow_untrained) {
        try stdout.writeAll("Missing --model. Pass --allow-untrained to serve a seeded untrained model.\n\n");
        try printUsage(stdout);
        return error.MissingModel;
    }

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
    const target_path = pathOnly(target);

    if (method == .OPTIONS) {
        return respondJson(&request, "{}", .no_content);
    }

    if (std.mem.eql(u8, target_path, "/v1/models")) {
        if (method != .GET) {
            return respondApiError(&request, .{
                .status = .method_not_allowed,
                .code = "unsupported_method",
                .message = "Use GET for /v1/models.",
            });
        }
        const response = try modelsResponse(allocator, options.model_name);
        defer allocator.free(response);
        return respondJson(&request, response, .ok);
    }

    if (!std.mem.eql(u8, target_path, "/v1/chat/completions") and
        !std.mem.eql(u8, target_path, "/v1/completions"))
    {
        return respondApiError(&request, .{
            .status = .not_found,
            .code = "not_found",
            .message = "Unknown endpoint.",
        });
    }

    if (method != .POST) {
        return respondApiError(&request, .{
            .status = .method_not_allowed,
            .code = "unsupported_method",
            .message = "Use POST for completion endpoints.",
        });
    }

    const path = try allocator.dupe(u8, target_path);
    defer allocator.free(path);

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
    const parsed = json.parseFromSlice(json.Value, allocator, body, .{}) catch {
        return respondApiError(&request, .{
            .status = .bad_request,
            .code = "invalid_json",
            .message = "Request body must be valid JSON.",
        });
    };
    defer parsed.deinit();

    var api_error: ?ApiError = null;
    if (std.mem.eql(u8, path, "/v1/chat/completions")) {
        const response = chatCompletionResponse(allocator, model, options, parsed.value, &api_error) catch |err| {
            return respondApiError(&request, api_error orelse fallbackApiError(err));
        };
        defer allocator.free(response);
        return respondJson(&request, response, .ok);
    }

    const response = completionResponse(allocator, model, options, parsed.value, &api_error) catch |err| {
        return respondApiError(&request, api_error orelse fallbackApiError(err));
    };
    defer allocator.free(response);
    return respondJson(&request, response, .ok);
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
    api_error: *?ApiError,
) ![]u8 {
    if (request_value != .object) return requestFail(api_error, .{
        .code = "invalid_request",
        .message = "Request body must be a JSON object.",
    });
    const object = request_value.object;

    var settings = try parseRequestSettings(allocator, object, options, api_error);
    defer settings.deinit(allocator);

    var prompts = try parseCompletionPrompts(allocator, object, api_error);
    defer prompts.deinit(allocator);

    var generated = try allocator.alloc(GeneratedText, prompts.items.len);
    var generated_count: usize = 0;
    errdefer {
        for (generated[0..generated_count]) |item| item.deinit(allocator);
        allocator.free(generated);
    }

    var prompt_tokens: usize = 0;
    var completion_tokens: usize = 0;
    for (prompts.items, 0..) |prompt, i| {
        generated[i] = try generateCompletion(allocator, model, prompt, settings);
        generated_count += 1;
        prompt_tokens += prompt.len;
        completion_tokens += generated[i].text.len;
    }
    defer {
        for (generated[0..generated_count]) |item| item.deinit(allocator);
        allocator.free(generated);
    }

    var out: std.Io.Writer.Allocating = .init(allocator);
    errdefer out.deinit();
    const writer = &out.writer;
    const created = unixSeconds();

    try writer.print("{{\"id\":\"cmpl-{d}\",\"object\":\"text_completion\",\"created\":{d},\"model\":", .{ created, created });
    try writeJsonString(writer, settings.model_name);
    try writer.writeAll(",\"system_fingerprint\":");
    try writeJsonString(writer, system_fingerprint);
    try writer.writeAll(",\"choices\":[");
    for (generated[0..generated_count], 0..) |item, i| {
        if (i > 0) try writer.writeByte(',');
        try writer.writeAll("{\"text\":");
        try writeJsonString(writer, item.text);
        try writer.print(",\"index\":{},\"finish_reason\":", .{i});
        try writeJsonString(writer, item.finish_reason);
        try writer.writeByte('}');
    }
    try writer.writeAll("],\"usage\":{");
    try writer.print("\"prompt_tokens\":{},\"completion_tokens\":{},\"total_tokens\":{}", .{
        prompt_tokens,
        completion_tokens,
        prompt_tokens + completion_tokens,
    });
    try writer.writeAll("}}");

    return out.toOwnedSlice();
}

fn chatCompletionResponse(
    allocator: std.mem.Allocator,
    model: *tiny.TinyGPT,
    options: ServerOptions,
    request_value: json.Value,
    api_error: *?ApiError,
) ![]u8 {
    if (request_value != .object) return requestFail(api_error, .{
        .code = "invalid_request",
        .message = "Request body must be a JSON object.",
    });
    const object = request_value.object;

    var settings = try parseRequestSettings(allocator, object, options, api_error);
    defer settings.deinit(allocator);

    const messages_value = object.get("messages") orelse return requestFail(api_error, .{
        .code = "missing_required_parameter",
        .message = "Missing required parameter: messages.",
        .param = "messages",
    });
    if (messages_value != .array) return requestFail(api_error, .{
        .code = "invalid_type",
        .message = "messages must be an array.",
        .param = "messages",
    });

    const prompt = try promptFromMessages(allocator, messages_value.array.items, api_error);
    defer allocator.free(prompt);

    const generated = try generateCompletion(allocator, model, prompt, settings);
    defer generated.deinit(allocator);

    var out: std.Io.Writer.Allocating = .init(allocator);
    errdefer out.deinit();
    const writer = &out.writer;
    const created = unixSeconds();

    try writer.print("{{\"id\":\"chatcmpl-{d}\",\"object\":\"chat.completion\",\"created\":{d},\"model\":", .{ created, created });
    try writeJsonString(writer, settings.model_name);
    try writer.writeAll(",\"system_fingerprint\":");
    try writeJsonString(writer, system_fingerprint);
    try writer.writeAll(",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":");
    try writeJsonString(writer, generated.text);
    try writer.writeAll("},\"finish_reason\":");
    try writeJsonString(writer, generated.finish_reason);
    try writer.writeAll("}],\"usage\":{");
    try writer.print("\"prompt_tokens\":{},\"completion_tokens\":{},\"total_tokens\":{}", .{
        prompt.len,
        generated.text.len,
        prompt.len + generated.text.len,
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

fn parseRequestSettings(
    allocator: std.mem.Allocator,
    object: json.ObjectMap,
    options: ServerOptions,
    api_error: *?ApiError,
) !RequestSettings {
    try rejectUnsupportedSettings(object, api_error);

    const max_tokens = usizeField(object, "max_tokens") orelse options.max_tokens;
    const temperature = floatField(object, "temperature") orelse options.temperature;
    const top_k = usizeField(object, "top_k") orelse options.top_k;
    const response_model = stringField(object, "model") orelse options.model_name;

    const stop_sequences = try parseStopSequences(allocator, object, api_error);
    return .{
        .model_name = response_model,
        .max_tokens = max_tokens,
        .temperature = temperature,
        .top_k = top_k,
        .stop_sequences = stop_sequences,
        .owns_stop_sequences = true,
    };
}

fn rejectUnsupportedSettings(object: json.ObjectMap, api_error: *?ApiError) !void {
    if (boolField(object, "stream")) |stream| {
        if (stream) return requestFail(api_error, .{
            .code = "unsupported_parameter",
            .message = "stream=true is not supported by this server.",
            .param = "stream",
        });
    } else if (object.get("stream")) |_| {
        return requestFail(api_error, .{
            .code = "invalid_type",
            .message = "stream must be a boolean.",
            .param = "stream",
        });
    }

    if (object.get("top_p")) |_| {
        return requestFail(api_error, .{
            .code = "unsupported_parameter",
            .message = "top_p sampling is not supported; use top_k instead.",
            .param = "top_p",
        });
    }

    if (usizeField(object, "n")) |n| {
        if (n != 1) return requestFail(api_error, .{
            .code = "unsupported_parameter",
            .message = "Only n=1 is supported.",
            .param = "n",
        });
    } else if (object.get("n")) |_| {
        return requestFail(api_error, .{
            .code = "invalid_type",
            .message = "n must be a positive integer.",
            .param = "n",
        });
    }

    if (object.get("user")) |value| {
        if (value != .string) return requestFail(api_error, .{
            .code = "invalid_type",
            .message = "user must be a string when provided.",
            .param = "user",
        });
    }
}

fn parseStopSequences(
    allocator: std.mem.Allocator,
    object: json.ObjectMap,
    api_error: *?ApiError,
) ![]const []const u8 {
    const value = object.get("stop") orelse return allocator.alloc([]const u8, 0);
    switch (value) {
        .null => return allocator.alloc([]const u8, 0),
        .string => |text| {
            const stops = try allocator.alloc([]const u8, 1);
            stops[0] = text;
            return stops;
        },
        .array => |array| {
            const stops = try allocator.alloc([]const u8, array.items.len);
            errdefer allocator.free(stops);
            for (array.items, 0..) |item, i| {
                if (item != .string) return requestFail(api_error, .{
                    .code = "invalid_type",
                    .message = "stop array entries must be strings.",
                    .param = "stop",
                });
                stops[i] = item.string;
            }
            return stops;
        },
        else => return requestFail(api_error, .{
            .code = "invalid_type",
            .message = "stop must be a string, an array of strings, or null.",
            .param = "stop",
        }),
    }
}

fn parseCompletionPrompts(
    allocator: std.mem.Allocator,
    object: json.ObjectMap,
    api_error: *?ApiError,
) !PromptList {
    const prompt_value = object.get("prompt") orelse return requestFail(api_error, .{
        .code = "missing_required_parameter",
        .message = "Missing required parameter: prompt.",
        .param = "prompt",
    });

    switch (prompt_value) {
        .string => |text| {
            const owned_text = try allocator.alloc([]u8, 1);
            errdefer allocator.free(owned_text);
            owned_text[0] = try allocator.dupe(u8, text);
            errdefer allocator.free(owned_text[0]);

            const items = try allocator.alloc([]const u8, 1);
            errdefer allocator.free(items);
            items[0] = owned_text[0];
            return .{ .items = items, .owned_text = owned_text };
        },
        .array => |array| {
            if (array.items.len == 0) return requestFail(api_error, .{
                .code = "invalid_request",
                .message = "prompt array must contain at least one string.",
                .param = "prompt",
            });

            const owned_text = try allocator.alloc([]u8, array.items.len);
            var owned_count: usize = 0;
            errdefer {
                for (owned_text[0..owned_count]) |text| allocator.free(text);
                allocator.free(owned_text);
            }

            const items = try allocator.alloc([]const u8, array.items.len);
            errdefer allocator.free(items);

            for (array.items, 0..) |item, i| {
                if (item == .array) return requestFail(api_error, .{
                    .code = "unsupported_parameter",
                    .message = "Token-array prompts are not supported.",
                    .param = "prompt",
                });
                if (item != .string) return requestFail(api_error, .{
                    .code = "invalid_type",
                    .message = "prompt array entries must be strings.",
                    .param = "prompt",
                });
                owned_text[i] = try allocator.dupe(u8, item.string);
                owned_count += 1;
                items[i] = owned_text[i];
            }
            return .{ .items = items, .owned_text = owned_text };
        },
        else => return requestFail(api_error, .{
            .code = "invalid_type",
            .message = "prompt must be a string or an array of strings.",
            .param = "prompt",
        }),
    }
}

fn generateCompletion(
    allocator: std.mem.Allocator,
    model: *tiny.TinyGPT,
    prompt: []const u8,
    settings: RequestSettings,
) !GeneratedText {
    const generated_tokens = try model.generateTokens(
        allocator,
        prompt,
        settings.max_tokens,
        settings.temperature,
        settings.top_k,
    );
    defer allocator.free(generated_tokens);

    const prompt_token_count = if (prompt.len == 0) 1 else prompt.len;
    const raw = try tiny.Tokenizer.decode(allocator, generated_tokens[prompt_token_count..]);
    defer allocator.free(raw);

    const stop_index = firstStopIndex(raw, settings.stop_sequences);
    const end = stop_index orelse raw.len;
    const text = try allocator.dupe(u8, raw[0..end]);
    return .{
        .text = text,
        .finish_reason = if (stop_index != null) "stop" else "length",
    };
}

fn promptFromMessages(
    allocator: std.mem.Allocator,
    messages: []const json.Value,
    api_error: *?ApiError,
) ![]u8 {
    if (messages.len == 0) return requestFail(api_error, .{
        .code = "invalid_request",
        .message = "messages must contain at least one message.",
        .param = "messages",
    });

    var out: std.Io.Writer.Allocating = .init(allocator);
    errdefer out.deinit();
    const writer = &out.writer;

    for (messages) |message| {
        if (message != .object) return requestFail(api_error, .{
            .code = "invalid_type",
            .message = "Each message must be an object.",
            .param = "messages",
        });
        const object = message.object;
        const role = stringField(object, "role") orelse return requestFail(api_error, .{
            .code = "missing_required_parameter",
            .message = "Each message must include a role.",
            .param = "messages.role",
        });
        if (!isSupportedRole(role)) return requestFail(api_error, .{
            .code = "invalid_value",
            .message = "Unsupported message role.",
            .param = "messages.role",
        });

        try writer.print("{s}: ", .{role});
        const content = object.get("content") orelse return requestFail(api_error, .{
            .code = "missing_required_parameter",
            .message = "Each message must include text content.",
            .param = "messages.content",
        });
        try appendMessageContent(writer, content, api_error);
        try writer.writeByte('\n');
    }
    try writer.writeAll("assistant: ");

    return out.toOwnedSlice();
}

fn appendMessageContent(writer: *std.Io.Writer, content: json.Value, api_error: *?ApiError) !void {
    switch (content) {
        .string => |text| try writer.writeAll(text),
        .array => |parts| {
            for (parts.items) |part| {
                if (part != .object) return requestFail(api_error, .{
                    .code = "invalid_type",
                    .message = "Message content parts must be objects.",
                    .param = "messages.content",
                });
                const part_object = part.object;
                const part_type = stringField(part_object, "type") orelse return requestFail(api_error, .{
                    .code = "missing_required_parameter",
                    .message = "Message content parts must include type.",
                    .param = "messages.content.type",
                });
                if (!std.mem.eql(u8, part_type, "text")) return requestFail(api_error, .{
                    .code = "unsupported_content",
                    .message = "Only text content parts are supported.",
                    .param = "messages.content",
                });
                const text = stringField(part_object, "text") orelse return requestFail(api_error, .{
                    .code = "missing_required_parameter",
                    .message = "Text content parts must include text.",
                    .param = "messages.content.text",
                });
                try writer.writeAll(text);
            }
        },
        else => return requestFail(api_error, .{
            .code = "invalid_type",
            .message = "Message content must be a string or an array of text parts.",
            .param = "messages.content",
        }),
    }
}

fn firstStopIndex(text: []const u8, stop_sequences: []const []const u8) ?usize {
    var best: ?usize = null;
    for (stop_sequences) |stop| {
        if (stop.len == 0) continue;
        if (std.mem.indexOf(u8, text, stop)) |idx| {
            if (best == null or idx < best.?) best = idx;
        }
    }
    return best;
}

fn isSupportedRole(role: []const u8) bool {
    return std.mem.eql(u8, role, "developer") or
        std.mem.eql(u8, role, "system") or
        std.mem.eql(u8, role, "user") or
        std.mem.eql(u8, role, "assistant") or
        std.mem.eql(u8, role, "tool") or
        std.mem.eql(u8, role, "function");
}

fn stringField(object: json.ObjectMap, name: []const u8) ?[]const u8 {
    const value = object.get(name) orelse return null;
    return switch (value) {
        .string => |text| text,
        else => null,
    };
}

fn boolField(object: json.ObjectMap, name: []const u8) ?bool {
    const value = object.get(name) orelse return null;
    return switch (value) {
        .bool => |b| b,
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

fn requestFail(api_error: *?ApiError, err: ApiError) error{OpenAIRequestError} {
    api_error.* = err;
    return error.OpenAIRequestError;
}

fn fallbackApiError(err: anyerror) ApiError {
    return .{
        .status = .bad_request,
        .code = "invalid_request",
        .message = @errorName(err),
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

fn respondApiError(request: *http.Server.Request, err: ApiError) !void {
    var out: std.Io.Writer.Allocating = .init(std.heap.smp_allocator);
    defer out.deinit();
    const writer = &out.writer;

    try writer.writeAll("{\"error\":{\"message\":");
    try writeJsonString(writer, err.message);
    try writer.writeAll(",\"type\":\"invalid_request_error\",\"param\":");
    if (err.param) |param| {
        try writeJsonString(writer, param);
    } else {
        try writer.writeAll("null");
    }
    try writer.writeAll(",\"code\":");
    try writeJsonString(writer, err.code);
    try writer.writeAll("}}");

    try respondJson(request, writer.buffered(), err.status);
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
        } else if (std.mem.eql(u8, arg, "--allow-untrained")) {
            options.allow_untrained = true;
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
        \\Usage: zig build run_tiny_gpt_openai -- --model <checkpoint> [options]
        \\
        \\Options:
        \\  --host <ip>          Bind host (default: 127.0.0.1)
        \\  --port <n>           Bind port (default: 8080)
        \\  --model <path>       TinyGPT checkpoint from `run_tiny_gpt -- --save-checkpoint`
        \\  --model-name <name>  Model id returned in OpenAI-compatible responses
        \\  --max-tokens <n>     Default completion length (default: 80)
        \\  --temperature <f>    Default sampling temperature (default: 1.0)
        \\  --top-k <n>          Default top-k sampler cutoff (default: 8)
        \\  --allow-untrained    Serve a seeded untrained model when --model is omitted
        \\
        \\Endpoints:
        \\  GET  /v1/models
        \\  POST /v1/completions
        \\  POST /v1/chat/completions
        \\
    );
}

test "chat prompt builder formats supported roles and text parts" {
    const allocator = std.testing.allocator;
    const body =
        \\{"messages":[{"role":"developer","content":"short"},{"role":"user","content":[{"type":"text","text":"hello"},{"type":"text","text":" world"}]}]}
    ;
    const parsed = try json.parseFromSlice(json.Value, allocator, body, .{});
    defer parsed.deinit();

    var api_error: ?ApiError = null;
    const messages = parsed.value.object.get("messages").?.array.items;
    const prompt = try promptFromMessages(allocator, messages, &api_error);
    defer allocator.free(prompt);

    try std.testing.expectEqual(@as(?ApiError, null), api_error);
    try std.testing.expectEqualStrings("developer: short\nuser: hello world\nassistant: ", prompt);
}

test "chat prompt builder rejects multimodal content parts" {
    const allocator = std.testing.allocator;
    const body =
        \\{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"x"}}]}]}
    ;
    const parsed = try json.parseFromSlice(json.Value, allocator, body, .{});
    defer parsed.deinit();

    var api_error: ?ApiError = null;
    const messages = parsed.value.object.get("messages").?.array.items;
    try std.testing.expectError(error.OpenAIRequestError, promptFromMessages(allocator, messages, &api_error));
    try std.testing.expectEqualStrings("unsupported_content", api_error.?.code);
    try std.testing.expectEqualStrings("messages.content", api_error.?.param.?);
}

test "completion prompt parser accepts string and string arrays" {
    const allocator = std.testing.allocator;
    const body =
        \\{"prompt":["first","second"]}
    ;
    const parsed = try json.parseFromSlice(json.Value, allocator, body, .{});
    defer parsed.deinit();

    var api_error: ?ApiError = null;
    const prompts = try parseCompletionPrompts(allocator, parsed.value.object, &api_error);
    defer prompts.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 2), prompts.items.len);
    try std.testing.expectEqualStrings("first", prompts.items[0]);
    try std.testing.expectEqualStrings("second", prompts.items[1]);
}

test "completion prompt parser rejects token arrays" {
    const allocator = std.testing.allocator;
    const body =
        \\{"prompt":[[1,2,3]]}
    ;
    const parsed = try json.parseFromSlice(json.Value, allocator, body, .{});
    defer parsed.deinit();

    var api_error: ?ApiError = null;
    try std.testing.expectError(error.OpenAIRequestError, parseCompletionPrompts(allocator, parsed.value.object, &api_error));
    try std.testing.expectEqualStrings("unsupported_parameter", api_error.?.code);
    try std.testing.expectEqualStrings("prompt", api_error.?.param.?);
}

test "request settings reject unsupported compatibility fields" {
    const allocator = std.testing.allocator;
    const cases = [_]struct {
        body: []const u8,
        param: []const u8,
    }{
        .{ .body = "{\"stream\":true}", .param = "stream" },
        .{ .body = "{\"top_p\":0.9}", .param = "top_p" },
        .{ .body = "{\"n\":2}", .param = "n" },
    };

    for (cases) |case| {
        const parsed = try json.parseFromSlice(json.Value, allocator, case.body, .{});
        defer parsed.deinit();

        var api_error: ?ApiError = null;
        try std.testing.expectError(
            error.OpenAIRequestError,
            parseRequestSettings(allocator, parsed.value.object, .{}, &api_error),
        );
        try std.testing.expectEqualStrings(case.param, api_error.?.param.?);
    }
}

test "stop sequences choose earliest match" {
    const stops = [_][]const u8{ "world", "lo" };
    try std.testing.expectEqual(@as(?usize, 3), firstStopIndex("hello world", &stops));
}

test "models response has OpenAI-compatible list shape" {
    const allocator = std.testing.allocator;
    const response = try modelsResponse(allocator, "tiny-gpt-zig");
    defer allocator.free(response);

    const parsed = try json.parseFromSlice(json.Value, allocator, response, .{});
    defer parsed.deinit();

    try std.testing.expectEqualStrings("list", parsed.value.object.get("object").?.string);
    const first = parsed.value.object.get("data").?.array.items[0].object;
    try std.testing.expectEqualStrings("tiny-gpt-zig", first.get("id").?.string);
    try std.testing.expectEqualStrings("model", first.get("object").?.string);
}

test "chat completion response includes fingerprint and finish reason" {
    const allocator = std.testing.allocator;
    var model = try tiny.TinyGPT.init(allocator, .{ .block_size = 4, .n_layer = 1, .n_head = 2, .n_embd = 8 }, 123);
    defer model.deinit();

    const body =
        \\{"messages":[{"role":"user","content":"to"}],"max_tokens":1}
    ;
    const parsed = try json.parseFromSlice(json.Value, allocator, body, .{});
    defer parsed.deinit();

    var api_error: ?ApiError = null;
    const response = try chatCompletionResponse(allocator, &model, .{}, parsed.value, &api_error);
    defer allocator.free(response);

    const response_json = try json.parseFromSlice(json.Value, allocator, response, .{});
    defer response_json.deinit();

    try std.testing.expectEqualStrings("chat.completion", response_json.value.object.get("object").?.string);
    try std.testing.expectEqualStrings(system_fingerprint, response_json.value.object.get("system_fingerprint").?.string);
    const choice = response_json.value.object.get("choices").?.array.items[0].object;
    try std.testing.expect(choice.get("finish_reason").? == .string);
    try std.testing.expect(choice.get("message").?.object.get("content").? == .string);
}

test "pathOnly removes query string" {
    try std.testing.expectEqualStrings("/v1/models", pathOnly("/v1/models?x=1"));
    try std.testing.expectEqualStrings("/v1/completions", pathOnly("/v1/completions"));
}
