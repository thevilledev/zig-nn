//! Long-lived, allocation-explicit inference sessions.
//!
//! Dense and text models intentionally have separate APIs: their input and
//! output contracts are different, while device selection and telemetry are
//! shared.

const std = @import("std");
const backend_mod = @import("backend.zig");
const cpu_backend_mod = @import("cpu_backend.zig");
const decoding = @import("decoding.zig");
const network_mod = @import("network.zig");
const tensor_mod = @import("tensor.zig");
const tiny = @import("tiny_gpt.zig");
const transformer = @import("transformer.zig");

const Allocator = std.mem.Allocator;
const Device = tensor_mod.Device;
const ExecutionContext = tensor_mod.ExecutionContext;
const Tensor = tensor_mod.Tensor;

pub const SessionOptions = struct {
    device: tensor_mod.DevicePreference = .auto,
    /// Number of independent CPU output-column tiles used by inference
    /// matmuls. The default avoids thread overhead for small models.
    cpu_output_tiles: usize = 1,
};

pub const ModelKind = enum {
    dense_network,
    tiny_gpt,
};

pub const ModelInfo = struct {
    kind: ModelKind,
    version: u32,
    input_size: ?usize = null,
    output_size: ?usize = null,
    layers: ?usize = null,
    block_size: ?usize = null,
    heads: ?usize = null,
    channels: ?usize = null,
    vocabulary_size: ?usize = null,
};

pub const RuntimeSnapshot = struct {
    execution: tensor_mod.ExecutionStats,
    backend: backend_mod.RuntimeStats,
};

pub const DenseSession = struct {
    allocator: Allocator,
    network: network_mod.Network,
    device: *Device,
    context: ExecutionContext,
    snapshot: network_mod.BackendNetwork,
    input_size: usize,
    output_size: usize,

    pub fn init(allocator: Allocator, model_path: []const u8, options: SessionOptions) !DenseSession {
        var network = try network_mod.Network.loadFromFile(allocator, model_path);
        errdefer network.deinit();

        const device = try allocator.create(Device);
        errdefer allocator.destroy(device);
        device.* = try Device.init(allocator, options.device);
        errdefer device.deinit();
        try device.configureCpuOutputTiles(options.cpu_output_tiles);

        var snapshot = try network.backendSnapshot(device.backendInstance());
        errdefer snapshot.deinit();

        return .{
            .allocator = allocator,
            .network = network,
            .device = device,
            .context = ExecutionContext.init(device),
            .snapshot = snapshot,
            .input_size = try network.getInputSize(),
            .output_size = try network.getOutputSize(),
        };
    }

    pub fn deinit(self: *DenseSession) void {
        self.snapshot.deinit();
        self.network.deinit();
        self.device.deinit();
        self.allocator.destroy(self.device);
        self.* = undefined;
    }

    pub fn backendType(self: DenseSession) backend_mod.BackendType {
        return self.device.backendType();
    }

    pub fn inputSize(self: DenseSession) usize {
        return self.input_size;
    }

    pub fn outputSize(self: DenseSession) usize {
        return self.output_size;
    }

    pub fn predictInto(self: *DenseSession, input: []const f32, batch_size: usize, output: []f32) !RuntimeSnapshot {
        if (batch_size == 0) return error.EmptyInput;
        const expected_input = std.math.mul(usize, batch_size, self.input_size) catch return error.DimensionOverflow;
        const expected_output = std.math.mul(usize, batch_size, self.output_size) catch return error.DimensionOverflow;
        if (input.len != expected_input) return error.InvalidInputDimensions;
        if (output.len != expected_output) return error.InvalidOutputDimensions;

        self.context.resetStats();
        var input_tensor = try self.context.upload(&.{ batch_size, self.input_size }, input);
        defer input_tensor.deinit();

        try self.context.beginBatch();
        var batch_active = true;
        errdefer if (batch_active) self.context.endBatch() catch {};
        const output_matrix = try self.snapshot.predict(input_tensor.matrix);
        try self.context.endBatch();
        batch_active = false;

        var output_tensor: Tensor = .{
            .matrix = output_matrix,
            .shape = try tensor_mod.Shape.init(&.{ batch_size, self.output_size }),
            .allocator = self.allocator,
        };
        defer output_tensor.deinit();
        try self.context.readback(output_tensor, output);
        return self.runtimeStats();
    }

    pub fn predictAlloc(self: *DenseSession, input: []const f32, batch_size: usize) ![]f32 {
        const output_len = std.math.mul(usize, batch_size, self.output_size) catch return error.DimensionOverflow;
        const output = try self.allocator.alloc(f32, output_len);
        errdefer self.allocator.free(output);
        _ = try self.predictInto(input, batch_size, output);
        return output;
    }

    pub fn runtimeStats(self: DenseSession) RuntimeSnapshot {
        return .{ .execution = self.context.executionStats(), .backend = self.context.backendStats() };
    }
};

pub const GenerationOptions = struct {
    max_tokens: usize = 80,
    temperature: f64 = 1,
    top_k: usize = 8,
    top_p: f64 = 1,
    seed: u64 = 42,

    pub fn validate(self: GenerationOptions) !void {
        try (decoding.SamplingConfig{
            .temperature = self.temperature,
            .top_k = self.top_k,
            .top_p = self.top_p,
        }).validate();
    }
};

pub const GenerationStats = struct {
    prompt_tokens: usize,
    generated_tokens: usize,
    elapsed_ns: u64,
    backend: backend_mod.BackendType,
    runtime: RuntimeSnapshot,
};

pub const GeneratedText = struct {
    allocator: Allocator,
    text: []u8,
    stats: GenerationStats,

    pub fn deinit(self: *GeneratedText) void {
        self.allocator.free(self.text);
        self.* = undefined;
    }
};

pub const TextSession = struct {
    allocator: Allocator,
    model: tiny.Model,
    device: *Device,
    context: ExecutionContext,
    decoder: transformer.Decoder,
    cache: transformer.DecoderCache,
    token_buffer: std.ArrayList(usize),
    logits: []f32,
    scores: []f64,
    candidates: []decoding.Candidate,
    current_logits: ?Tensor,
    prng: std.Random.DefaultPrng,

    pub fn init(allocator: Allocator, model_path: []const u8, options: SessionOptions) !TextSession {
        const model = try tiny.Model.loadFromFile(allocator, model_path);
        return initModel(allocator, model, options);
    }

    /// Takes ownership of an already-created model. This keeps seeded model
    /// demos on the same persistent runtime path as loaded checkpoints.
    pub fn initModel(allocator: Allocator, model: tiny.Model, options: SessionOptions) !TextSession {
        var owned_model = model;
        errdefer owned_model.deinit();

        const device = try allocator.create(Device);
        errdefer allocator.destroy(device);
        device.* = try Device.init(allocator, options.device);
        errdefer device.deinit();
        try device.configureCpuOutputTiles(options.cpu_output_tiles);

        var context = ExecutionContext.init(device);
        var decoder_model = try owned_model.toDeviceDecoder(&context);
        errdefer decoder_model.deinit();
        var cache = try decoder_model.initCache(&context);
        errdefer cache.deinit();

        const logits = try allocator.alloc(f32, owned_model.config.vocab_size);
        errdefer allocator.free(logits);
        const scores = try allocator.alloc(f64, owned_model.config.vocab_size);
        errdefer allocator.free(scores);
        const candidates = try allocator.alloc(decoding.Candidate, owned_model.config.vocab_size);
        errdefer allocator.free(candidates);

        return .{
            .allocator = allocator,
            .model = owned_model,
            .device = device,
            .context = context,
            .decoder = decoder_model,
            .cache = cache,
            .token_buffer = .empty,
            .logits = logits,
            .scores = scores,
            .candidates = candidates,
            .current_logits = null,
            .prng = std.Random.DefaultPrng.init(42),
        };
    }

    pub fn deinit(self: *TextSession) void {
        if (self.current_logits) |*value| value.deinit();
        self.token_buffer.deinit(self.allocator);
        self.allocator.free(self.candidates);
        self.allocator.free(self.scores);
        self.allocator.free(self.logits);
        self.cache.deinit();
        self.decoder.deinit();
        self.model.deinit();
        self.device.deinit();
        self.allocator.destroy(self.device);
        self.* = undefined;
    }

    pub fn backendType(self: TextSession) backend_mod.BackendType {
        return self.device.backendType();
    }

    pub fn reset(self: *TextSession) void {
        if (self.current_logits) |*value| value.deinit();
        self.current_logits = null;
        self.cache.reset();
        self.token_buffer.clearRetainingCapacity();
        self.context.resetStats();
    }

    pub fn generate(self: *TextSession, prompt: []const u8, options: GenerationOptions, writer: anytype) !GenerationStats {
        try options.validate();
        const started = nowNs();
        const prompt_tokens = try self.prefill(prompt, options.seed, options.max_tokens);
        if (options.max_tokens == 0) return self.generationStats(prompt_tokens, 0, started);
        for (0..options.max_tokens) |generated| {
            const next = try self.decode(options, generated + 1 < options.max_tokens);
            writer.writeByte(next) catch |err| {
                if (err == error.StopGeneration) {
                    return self.generationStats(prompt_tokens, generated + 1, started);
                }
                return err;
            };
        }
        return self.generationStats(prompt_tokens, options.max_tokens, started);
    }

    /// Resets the session, tokenizes a prompt, and fills the persistent KV
    /// cache. `additional_capacity` reserves space for subsequent decode calls.
    pub fn prefill(self: *TextSession, prompt: []const u8, seed: u64, additional_capacity: usize) !usize {
        self.reset();
        self.prng = std.Random.DefaultPrng.init(seed);
        const prompt_tokens = if (prompt.len == 0) 1 else prompt.len;
        const capacity = std.math.add(usize, prompt_tokens, additional_capacity) catch return error.DimensionOverflow;
        try self.token_buffer.ensureTotalCapacity(self.allocator, capacity);
        if (prompt.len == 0) {
            try self.token_buffer.append(self.allocator, tiny.Tokenizer.encodeChar(' '));
        } else {
            for (prompt) |byte| try self.token_buffer.append(self.allocator, tiny.Tokenizer.encodeChar(byte));
        }
        const start = prompt_tokens - @min(prompt_tokens, self.model.config.block_size);
        self.current_logits = try tiny.forwardDeviceTokens(
            self.allocator,
            &self.decoder,
            &self.context,
            &self.cache,
            self.token_buffer.items[start..prompt_tokens],
            true,
        );
        return prompt_tokens;
    }

    /// Decodes one token and advances the KV cache for another call.
    pub fn decodeNext(self: *TextSession, options: GenerationOptions) !u8 {
        try options.validate();
        return self.decode(options, true);
    }

    pub fn generateAlloc(self: *TextSession, prompt: []const u8, options: GenerationOptions) !GeneratedText {
        var output: std.Io.Writer.Allocating = .init(self.allocator);
        errdefer output.deinit();
        const stats = try self.generate(prompt, options, &output.writer);
        return .{ .allocator = self.allocator, .text = try output.toOwnedSlice(), .stats = stats };
    }

    pub fn runtimeStats(self: TextSession) RuntimeSnapshot {
        return .{ .execution = self.context.executionStats(), .backend = self.context.backendStats() };
    }

    fn decode(self: *TextSession, options: GenerationOptions, prepare_next: bool) !u8 {
        const current = self.current_logits orelse return error.SessionNotPrefilled;
        try self.context.readback(current, self.logits);
        for (self.logits, self.scores) |logit, *score| score.* = logit;
        const next_token = try decoding.sampleWithWorkspace(
            self.prng.random(),
            self.scores,
            &.{},
            .{ .temperature = options.temperature, .top_k = options.top_k, .top_p = options.top_p },
            self.candidates,
        );
        try self.token_buffer.append(self.allocator, next_token);
        if (prepare_next) {
            const token_count = self.token_buffer.items.len;
            const next_logits = if (self.cache.position == self.model.config.block_size) blk: {
                const start = token_count - @min(token_count, self.model.config.block_size);
                break :blk try tiny.forwardDeviceTokens(
                    self.allocator,
                    &self.decoder,
                    &self.context,
                    &self.cache,
                    self.token_buffer.items[start..token_count],
                    true,
                );
            } else try tiny.forwardDeviceTokens(
                self.allocator,
                &self.decoder,
                &self.context,
                &self.cache,
                self.token_buffer.items[token_count - 1 .. token_count],
                false,
            );
            if (self.current_logits) |*value| value.deinit();
            self.current_logits = next_logits;
        }
        return tiny.Tokenizer.decodeToken(next_token);
    }

    fn generationStats(self: TextSession, prompt_tokens: usize, generated_tokens: usize, started: i96) GenerationStats {
        const elapsed = nowNs() - started;
        return .{
            .prompt_tokens = prompt_tokens,
            .generated_tokens = generated_tokens,
            .elapsed_ns = @intCast(@max(elapsed, 0)),
            .backend = self.backendType(),
            .runtime = self.runtimeStats(),
        };
    }
};

pub fn inspectModel(path: []const u8) !ModelInfo {
    const io = std.Options.debug_io;
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    var buffer: [256]u8 = undefined;
    var file_reader = file.readerStreaming(io, &buffer);
    return inspectReader(&file_reader.interface);
}

/// Inspects a checkpoint header without touching the filesystem. Keeping this
/// parser pure makes malformed model input inexpensive to fuzz.
pub fn inspectModelBytes(data: []const u8) !ModelInfo {
    var reader: std.Io.Reader = .fixed(data);
    return inspectReader(&reader);
}

fn inspectReader(reader: *std.Io.Reader) !ModelInfo {
    var magic: [4]u8 = undefined;
    @memcpy(&magic, try reader.take(4));
    if (std.mem.eql(u8, &magic, "ZNN\x00")) {
        const version = try reader.takeInt(u32, .little);
        if (version < 1 or version > 3) return error.UnsupportedVersion;
        _ = try reader.takeInt(i64, .little);
        if (version >= 2) {
            _ = try reader.takeInt(i64, .little);
            _ = try reader.takeByte();
        }
        return .{
            .kind = .dense_network,
            .version = version,
            .input_size = try reader.takeInt(u32, .little),
            .output_size = try reader.takeInt(u32, .little),
            .layers = try reader.takeInt(u32, .little),
        };
    }
    if (std.mem.eql(u8, &magic, "TGPT")) {
        const version = try reader.takeInt(u32, .little);
        if (version < 1 or version > 3) return error.UnsupportedVersion;
        const block_size = try reader.takeInt(u32, .little);
        const layers = try reader.takeInt(u32, .little);
        const heads = try reader.takeInt(u32, .little);
        const channels = try reader.takeInt(u32, .little);
        const vocabulary_size = try reader.takeInt(u32, .little);
        if (version >= 3 and try reader.takeInt(u64, .little) != tiny.Tokenizer.vocabularyFingerprint()) {
            return error.UnsupportedVocabulary;
        }
        return .{
            .kind = .tiny_gpt,
            .version = version,
            .layers = layers,
            .block_size = block_size,
            .heads = heads,
            .channels = channels,
            .vocabulary_size = vocabulary_size,
        };
    }
    return error.InvalidFileFormat;
}

fn nowNs() i96 {
    return std.Io.Clock.awake.now(std.Options.debug_io).toNanoseconds();
}

fn temporaryModelPath(allocator: Allocator, tmp: *const std.testing.TmpDir, filename: []const u8) ![]u8 {
    return std.fs.path.join(allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..], filename });
}

fn checkTextSessionAllocationFailure(allocator: Allocator, path: []const u8) !void {
    var session = try TextSession.init(allocator, path, .{ .device = .cpu });
    defer session.deinit();
    var storage: [4]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&storage);
    _ = try session.generate("ab", .{ .max_tokens = 1, .temperature = 0 }, &writer);
}

fn checkDenseSessionAllocationFailure(allocator: Allocator, path: []const u8) !void {
    var session = try DenseSession.init(allocator, path, .{ .device = .cpu });
    defer session.deinit();
    var output: [1]f32 = undefined;
    _ = try session.predictInto(&.{ 0.25, -0.5 }, 1, &output);
}

fn fuzzModelHeader(_: void, smith: *std.testing.Smith) !void {
    @disableInstrumentation();
    var buffer: [96]u8 = undefined;
    const len = smith.sliceWeightedBytes(&buffer, &.{
        .rangeAtMost(u8, 0, 255, 1),
        .value(u8, 0, 3),
        .rangeAtMost(u8, 'A', 'Z', 2),
    });
    _ = inspectModelBytes(buffer[0..len]) catch {};
}

fn fuzzCheckpointLoaders(_: void, smith: *std.testing.Smith) !void {
    @disableInstrumentation();
    var buffer: [512]u8 = undefined;
    const len = smith.sliceWeightedBytes(&buffer, &.{
        .rangeAtMost(u8, 0, 255, 1),
        .value(u8, 0, 3),
        .rangeAtMost(u8, 'A', 'Z', 2),
    });
    const bytes = buffer[0..len];
    if (network_mod.Network.loadFromBytes(std.testing.allocator, bytes)) |network_value| {
        var network = network_value;
        network.deinit();
    } else |_| {}
    if (tiny.Model.loadFromBytes(std.testing.allocator, bytes)) |model_value| {
        var model = model_value;
        model.deinit();
    } else |_| {}
}

test "dense session predicts into caller storage and reports CPU telemetry" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try temporaryModelPath(allocator, &tmp, "dense.znn");
    defer allocator.free(path);

    var network = network_mod.Network.init(allocator, 0.1, .MeanSquaredError);
    defer network.deinit();
    try network.addLayer(2, 1, @import("activation.zig").Activation.linear, @import("activation.zig").Activation.linear_derivative);
    try network.saveToFile(path);

    var session = try DenseSession.init(allocator, path, .{ .device = .cpu });
    defer session.deinit();
    var output: [2]f32 = undefined;
    const stats = try session.predictInto(&.{ 1, 2, 3, 4 }, 2, &output);
    try std.testing.expectEqual(backend_mod.BackendType.CPU, session.backendType());
    try std.testing.expectEqual(@as(usize, 1), stats.execution.uploads);
    try std.testing.expectEqual(@as(usize, 1), stats.execution.readbacks);
    try std.testing.expectError(error.EmptyInput, session.predictInto(&.{}, 0, &output));
    try std.testing.expectError(error.InvalidInputDimensions, session.predictInto(&.{1}, 1, output[0..1]));
    try std.testing.expectError(error.InvalidOutputDimensions, session.predictInto(&.{ 1, 2 }, 1, output[0..0]));
}

test "dense sessions validate CPU output tiling and preserve predictions" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try temporaryModelPath(allocator, &tmp, "dense-tiles.znn");
    defer allocator.free(path);

    var network = network_mod.Network.init(allocator, 0.1, .MeanSquaredError);
    defer network.deinit();
    try network.addLayer(2, 11, @import("activation.zig").Activation.linear, @import("activation.zig").Activation.linear_derivative);
    try network.saveToFile(path);

    try std.testing.expectError(
        error.InvalidOutputTileCount,
        DenseSession.init(allocator, path, .{ .device = .cpu, .cpu_output_tiles = 0 }),
    );
    try std.testing.expectError(
        error.InvalidOutputTileCount,
        DenseSession.init(allocator, path, .{ .device = .cpu, .cpu_output_tiles = 65 }),
    );

    var reference = try DenseSession.init(allocator, path, .{ .device = .cpu });
    defer reference.deinit();
    var parallel = try DenseSession.init(allocator, path, .{ .device = .cpu, .cpu_output_tiles = 4 });
    defer parallel.deinit();
    const expected = try reference.predictAlloc(&.{ 0.25, -0.5 }, 1);
    defer allocator.free(expected);
    const actual = try parallel.predictAlloc(&.{ 0.25, -0.5 }, 1);
    defer allocator.free(actual);
    for (expected, actual) |reference_value, parallel_value| {
        try std.testing.expectApproxEqAbs(reference_value, parallel_value, 1e-4);
    }
}

test "text session reuses a loaded decoder and deterministic greedy generation" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try temporaryModelPath(allocator, &tmp, "tiny.tgpt");
    defer allocator.free(path);

    var model = try tiny.Model.init(allocator, .{ .block_size = 4, .n_layer = 1, .n_head = 2, .n_embd = 8 }, 123);
    defer model.deinit();
    try model.saveToFile(path);

    var session = try TextSession.init(allocator, path, .{ .device = .cpu });
    defer session.deinit();
    var first = try session.generateAlloc("ab", .{ .max_tokens = 3, .temperature = 0 });
    defer first.deinit();
    var second = try session.generateAlloc("ab", .{ .max_tokens = 3, .temperature = 0 });
    defer second.deinit();
    try std.testing.expectEqualStrings(first.text, second.text);
    try std.testing.expectEqual(@as(usize, 3), second.stats.generated_tokens);
    try std.testing.expect(second.stats.runtime.execution.readbacks >= 3);
}

test "packed decoder logits match the scalar TinyGPT reference" {
    const allocator = std.testing.allocator;
    const config: tiny.Config = .{ .block_size = 4, .n_layer = 1, .n_head = 2, .n_embd = 8 };
    var reference_model = try tiny.Model.init(allocator, config, 321);
    defer reference_model.deinit();
    const device_model = try tiny.Model.init(allocator, config, 321);
    const tokens = [_]usize{ tiny.Tokenizer.encodeChar('a'), tiny.Tokenizer.encodeChar('b') };
    var reference_logits = try reference_model.forward(&tokens);
    defer reference_logits.deinit();

    var session = try TextSession.initModel(allocator, device_model, .{
        .device = .cpu,
        .cpu_output_tiles = 3,
    });
    defer session.deinit();
    _ = try session.prefill("ab", 42, 0);
    try session.context.readback(session.current_logits.?, session.logits);

    const reference_row = reference_logits.rows - 1;
    for (session.logits, 0..) |actual, token| {
        const expected: f32 = @floatCast(try reference_logits.get(reference_row, token));
        try std.testing.expectApproxEqAbs(expected, actual, 1e-4);
    }
}

test "text session keeps buffers and workspaces bounded for 1000 decode steps" {
    const allocator = std.testing.allocator;
    const model = try tiny.Model.init(allocator, .{ .block_size = 8, .n_layer = 1, .n_head = 1, .n_embd = 4 }, 99);
    var session = try TextSession.initModel(allocator, model, .{ .device = .cpu });
    defer session.deinit();

    _ = try session.prefill("a", 42, 1016);
    for (0..16) |_| _ = try session.decodeNext(.{ .temperature = 0 });
    const live_buffers = session.runtimeStats().backend.live_buffers;
    const cpu = switch (session.device.backendInstance()) {
        .CPU => |ptr| @as(*cpu_backend_mod.CPUBackend, @ptrCast(@alignCast(ptr))),
        else => unreachable,
    };
    const storage_allocations = cpu.storage_allocations;
    const token_capacity = session.token_buffer.capacity;
    for (0..1000) |_| {
        _ = try session.decodeNext(.{ .temperature = 0 });
        try std.testing.expectEqual(live_buffers, session.runtimeStats().backend.live_buffers);
        try std.testing.expectEqual(storage_allocations, cpu.storage_allocations);
        try std.testing.expect(session.cache.position <= session.model.config.block_size);
    }
    try std.testing.expectEqual(token_capacity, session.token_buffer.capacity);
}

test "model inspection rejects malformed and incompatible headers" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try temporaryModelPath(allocator, &tmp, "bad.bin");
    defer allocator.free(path);
    const io = std.Options.debug_io;
    const file = try std.Io.Dir.cwd().createFile(io, path, .{});
    try file.writeStreamingAll(io, "NOPE");
    file.close(io);
    try std.testing.expectError(error.InvalidFileFormat, inspectModel(path));
}

test "text session cleans up every allocation failure" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try temporaryModelPath(allocator, &tmp, "allocation.tgpt");
    defer allocator.free(path);
    var model = try tiny.Model.init(allocator, .{ .block_size = 2, .n_layer = 1, .n_head = 1, .n_embd = 4 }, 7);
    defer model.deinit();
    try model.saveToFile(path);
    try std.testing.checkAllAllocationFailures(allocator, checkTextSessionAllocationFailure, .{path});
}

test "dense session cleans up every load and inference allocation failure" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try temporaryModelPath(allocator, &tmp, "allocation.znn");
    defer allocator.free(path);
    var network = network_mod.Network.init(allocator, 0.1, .MeanSquaredError);
    defer network.deinit();
    try network.addLayer(2, 1, @import("activation.zig").Activation.linear, @import("activation.zig").Activation.linear_derivative);
    try network.saveToFile(path);
    try std.testing.checkAllAllocationFailures(allocator, checkDenseSessionAllocationFailure, .{path});
}

test "seeded dense and text inference match every available backend" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dense_path = try temporaryModelPath(allocator, &tmp, "parity.znn");
    defer allocator.free(dense_path);
    const text_path = try temporaryModelPath(allocator, &tmp, "parity.tgpt");
    defer allocator.free(text_path);

    var network = network_mod.Network.init(allocator, 0.1, .MeanSquaredError);
    defer network.deinit();
    try network.addLayer(2, 3, @import("activation.zig").Activation.relu, @import("activation.zig").Activation.relu_derivative);
    try network.addLayer(3, 1, @import("activation.zig").Activation.linear, @import("activation.zig").Activation.linear_derivative);
    try network.saveToFile(dense_path);
    var model = try tiny.Model.init(allocator, .{ .block_size = 4, .n_layer = 1, .n_head = 2, .n_embd = 8 }, 77);
    defer model.deinit();
    try model.saveToFile(text_path);

    var dense_cpu = try DenseSession.init(allocator, dense_path, .{ .device = .cpu });
    defer dense_cpu.deinit();
    const dense_expected = try dense_cpu.predictAlloc(&.{ 0.25, -0.5 }, 1);
    defer allocator.free(dense_expected);
    var text_cpu = try TextSession.init(allocator, text_path, .{ .device = .cpu });
    defer text_cpu.deinit();
    var text_expected = try text_cpu.generateAlloc("ab", .{ .max_tokens = 6, .temperature = 0, .seed = 42 });
    defer text_expected.deinit();

    for ([_]tensor_mod.DevicePreference{ .metal, .cuda, .rocm }) |preference| {
        var dense = DenseSession.init(allocator, dense_path, .{ .device = preference }) catch |err| switch (err) {
            error.BackendUnavailable => continue,
            else => return err,
        };
        defer dense.deinit();
        const dense_actual = try dense.predictAlloc(&.{ 0.25, -0.5 }, 1);
        defer allocator.free(dense_actual);
        for (dense_expected, dense_actual) |expected, actual| {
            try std.testing.expectApproxEqAbs(expected, actual, 1e-4);
        }

        var text = try TextSession.init(allocator, text_path, .{ .device = preference });
        defer text.deinit();
        var text_actual = try text.generateAlloc("ab", .{ .max_tokens = 6, .temperature = 0, .seed = 42 });
        defer text_actual.deinit();
        try std.testing.expectEqualStrings(text_expected.text, text_actual.text);
    }
}

test "text generation stops when the token sink disconnects" {
    const DisconnectingWriter = struct {
        written: usize = 0,

        pub fn writeByte(self: *@This(), _: u8) !void {
            if (self.written == 1) return error.ClientDisconnected;
            self.written += 1;
        }
    };

    const allocator = std.testing.allocator;
    const source = try tiny.Model.init(allocator, .{ .block_size = 4, .n_layer = 1, .n_head = 2, .n_embd = 8 }, 123);
    var session = try TextSession.initModel(allocator, source, .{ .device = .cpu });
    defer session.deinit();
    var writer: DisconnectingWriter = .{};

    try std.testing.expectError(
        error.ClientDisconnected,
        session.generate("ab", .{ .max_tokens = 10, .temperature = 0 }, &writer),
    );
    try std.testing.expectEqual(@as(usize, 1), writer.written);
    try std.testing.expect(session.token_buffer.items.len < 12);
}

test "checkpoint header parser is fuzzable" {
    try std.testing.fuzz({}, fuzzModelHeader, .{
        .corpus = &.{ "ZNN\x00", "TGPT", "NOPE", "TGPT\x03\x00\x00\x00" },
    });
}

test "ZNN and TGPT checkpoint loaders accept fuzzed bytes" {
    try std.testing.fuzz({}, fuzzCheckpointLoaders, .{
        .corpus = &.{ "ZNN\x00", "TGPT", "NOPE", "TGPT\x03\x00\x00\x00" },
    });
}
