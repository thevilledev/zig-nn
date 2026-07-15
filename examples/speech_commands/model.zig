const std = @import("std");
const nn = @import("nn");

pub const labels = [_][]const u8{ "down", "go", "left", "no", "right", "stop", "up", "yes" };
pub const class_count = labels.len;
pub const feature_count = nn.Audio.default_feature_count;
pub const hidden_features: usize = 64;
pub const parameter_count: usize = 4;
const checkpoint_version: u32 = 1;

pub const SpeechModel = struct {
    network: nn.Modules.Mlp,
    mean: [feature_count]f32 = @splat(0),
    stddev: [feature_count]f32 = @splat(1),

    pub fn init(context: *nn.ExecutionContext, random: std.Random) !SpeechModel {
        return .{
            .network = try nn.Modules.Mlp.init(context, .{
                .widths = &.{ feature_count, hidden_features, class_count },
                .hidden_activation = .relu,
                .output_activation = .linear,
            }, random),
        };
    }

    pub fn deinit(self: *SpeechModel) void {
        self.network.deinit();
        self.* = undefined;
    }

    pub fn parameters(self: *SpeechModel, output: *[parameter_count]*nn.Tensor) ![]*nn.Tensor {
        return self.network.parameters(output);
    }

    pub fn normalize(self: SpeechModel, features: []f32) !void {
        if (features.len % feature_count != 0) return error.DimensionMismatch;
        for (features, 0..) |*value, index| {
            const feature = index % feature_count;
            value.* = (value.* - self.mean[feature]) / self.stddev[feature];
        }
    }

    pub fn trainBatch(
        self: *SpeechModel,
        context: *nn.ExecutionContext,
        optimizer: *nn.Training.Optimizer,
        bound_parameters: []const *nn.Tensor,
        features: []const f32,
        targets: []const usize,
    ) !f32 {
        if (targets.len == 0 or features.len != targets.len * feature_count) return error.DimensionMismatch;
        try context.beginBatch();
        var batch_active = true;
        errdefer if (batch_active) context.endBatch() catch {};

        var input = try context.upload(&.{ targets.len, feature_count }, features);
        defer input.deinit();
        var forward = try self.network.forward(context, input);
        defer forward.deinit();
        const mask = try context.device.allocator.alloc(f32, targets.len);
        defer context.device.allocator.free(mask);
        @memset(mask, 1);
        var objective = try context.maskedSparseCrossEntropy(forward.output, targets, mask);
        defer objective.deinit();
        const loss = try objective.meanLoss(context);
        var gradients = try self.network.backward(context, &forward.cache, objective.gradient);
        defer gradients.deinit();
        var gradient_storage: [parameter_count]nn.Tensor = undefined;
        const parameter_gradients = try gradients.parameterGradients(&gradient_storage);
        _ = try optimizer.step(context, bound_parameters, parameter_gradients);
        try context.endBatch();
        batch_active = false;
        return loss;
    }

    pub fn predict(
        self: *const SpeechModel,
        allocator: std.mem.Allocator,
        context: *nn.ExecutionContext,
        features: []const f32,
    ) ![]f32 {
        if (features.len == 0 or features.len % feature_count != 0) return error.DimensionMismatch;
        const rows = features.len / feature_count;
        try context.beginBatch();
        var batch_active = true;
        errdefer if (batch_active) context.endBatch() catch {};
        var input = try context.upload(&.{ rows, feature_count }, features);
        defer input.deinit();
        var forward = try self.network.forward(context, input);
        defer forward.deinit();
        var probabilities = try context.softmax(forward.output);
        defer probabilities.deinit();
        try context.endBatch();
        batch_active = false;
        const values = try allocator.alloc(f32, rows * class_count);
        errdefer allocator.free(values);
        try context.readback(probabilities, values);
        return values;
    }

    pub fn save(self: *const SpeechModel, context: *nn.ExecutionContext, path: []const u8) !void {
        const io = std.Options.debug_io;
        const file = try std.Io.Dir.cwd().createFile(io, path, .{});
        defer file.close(io);
        var buffer: [4096]u8 = undefined;
        var file_writer = file.writerStreaming(io, &buffer);
        const writer = &file_writer.interface;

        try writer.writeAll("ZNSC");
        try writer.writeInt(u32, checkpoint_version, .little);
        try writer.writeInt(u32, nn.Audio.target_sample_rate, .little);
        try writer.writeInt(u32, nn.Audio.target_sample_count, .little);
        try writer.writeInt(u32, nn.Audio.default_frame_length, .little);
        try writer.writeInt(u32, nn.Audio.default_frame_step, .little);
        try writer.writeInt(u32, nn.Audio.default_fft_size, .little);
        try writer.writeInt(u32, nn.Audio.default_mel_bins, .little);
        try writer.writeInt(u32, nn.Audio.default_time_bins, .little);
        try writer.writeInt(u32, feature_count, .little);
        try writer.writeInt(u32, hidden_features, .little);
        try writer.writeInt(u32, class_count, .little);
        for (labels) |label| {
            try writer.writeByte(@intCast(label.len));
            try writer.writeAll(label);
        }
        for (self.mean) |value| try writer.writeInt(u32, @bitCast(value), .little);
        for (self.stddev) |value| try writer.writeInt(u32, @bitCast(value), .little);

        var parameter_storage: [parameter_count]*nn.Tensor = undefined;
        const bound_parameters = try @constCast(&self.network).parameters(&parameter_storage);
        for (bound_parameters) |parameter| {
            const values = try context.device.allocator.alloc(f32, parameter.elementCount());
            defer context.device.allocator.free(values);
            try context.readback(parameter.*, values);
            try writer.writeInt(u32, @intCast(values.len), .little);
            for (values) |value| try writer.writeInt(u32, @bitCast(value), .little);
        }
        try writer.flush();
    }

    pub fn load(context: *nn.ExecutionContext, path: []const u8) !SpeechModel {
        const io = std.Options.debug_io;
        const file = try std.Io.Dir.cwd().openFile(io, path, .{});
        defer file.close(io);
        var buffer: [4096]u8 = undefined;
        var file_reader = file.readerStreaming(io, &buffer);
        const reader = &file_reader.interface;

        var magic: [4]u8 = undefined;
        @memcpy(&magic, try reader.take(4));
        if (!std.mem.eql(u8, &magic, "ZNSC")) return error.InvalidCheckpoint;
        if (try reader.takeInt(u32, .little) != checkpoint_version) return error.UnsupportedCheckpointVersion;
        try expectMetadata(reader, nn.Audio.target_sample_rate);
        try expectMetadata(reader, nn.Audio.target_sample_count);
        try expectMetadata(reader, nn.Audio.default_frame_length);
        try expectMetadata(reader, nn.Audio.default_frame_step);
        try expectMetadata(reader, nn.Audio.default_fft_size);
        try expectMetadata(reader, nn.Audio.default_mel_bins);
        try expectMetadata(reader, nn.Audio.default_time_bins);
        try expectMetadata(reader, feature_count);
        try expectMetadata(reader, hidden_features);
        try expectMetadata(reader, class_count);
        for (labels) |label| {
            const length = try reader.takeByte();
            if (length != label.len or !std.mem.eql(u8, try reader.take(length), label)) {
                return error.UnsupportedVocabulary;
            }
        }

        var prng = std.Random.DefaultPrng.init(0);
        var model = try SpeechModel.init(context, prng.random());
        errdefer model.deinit();
        for (&model.mean) |*value| value.* = @bitCast(try reader.takeInt(u32, .little));
        for (&model.stddev) |*value| {
            value.* = @bitCast(try reader.takeInt(u32, .little));
            if (!std.math.isFinite(value.*) or value.* <= 0) return error.InvalidCheckpoint;
        }

        var parameter_storage: [parameter_count]*nn.Tensor = undefined;
        const bound_parameters = try model.parameters(&parameter_storage);
        for (bound_parameters) |parameter| {
            const count = try reader.takeInt(u32, .little);
            if (count != parameter.elementCount()) return error.IncompatibleCheckpoint;
            const values = try context.device.allocator.alloc(f32, count);
            defer context.device.allocator.free(values);
            for (values) |*value| value.* = @bitCast(try reader.takeInt(u32, .little));
            try parameter.writeF32(values);
        }
        return model;
    }
};

fn expectMetadata(reader: *std.Io.Reader, expected: anytype) !void {
    const actual = try reader.takeInt(u32, .little);
    if (actual != @as(u32, @intCast(expected))) return error.IncompatibleCheckpoint;
}
