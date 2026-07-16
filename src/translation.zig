const std = @import("std");

pub const Config = struct {
    vocabulary_size: usize,
    maximum_tokens: usize,
    padding_token: usize,
    beginning_token: usize,
    end_token: usize,

    fn validate(self: Config) !void {
        if (self.vocabulary_size < 4 or self.maximum_tokens == 0 or
            self.padding_token >= self.vocabulary_size or
            self.beginning_token >= self.vocabulary_size or
            self.end_token >= self.vocabulary_size or
            self.padding_token == self.beginning_token or
            self.padding_token == self.end_token or
            self.beginning_token == self.end_token)
        {
            return error.InvalidConfiguration;
        }
    }
};

pub const Gradients = struct {
    allocator: std.mem.Allocator,
    weights: []f32,
    bias: []f32,

    pub fn deinit(self: *Gradients) void {
        self.allocator.free(self.weights);
        self.allocator.free(self.bias);
        self.* = undefined;
    }
};

pub const Objective = struct {
    loss: f32,
    gradients: Gradients,

    pub fn deinit(self: *Objective) void {
        self.gradients.deinit();
        self.* = undefined;
    }
};

/// A compact teacher-forced encoder-decoder reference.
///
/// The encoder combines token and position features. The causal decoder uses
/// only the supplied target prefix, then performs masked cross-attention over
/// the unpadded encoder positions. A learned output projection is optimized by
/// exact softmax-cross-entropy gradients. This intentionally small model makes
/// every sequence, mask, alignment, and gradient easy to inspect.
pub const TeacherForcedEncoderDecoder = struct {
    allocator: std.mem.Allocator,
    config: Config,
    feature_count: usize,
    weights: []f32,
    bias: []f32,

    pub fn init(
        allocator: std.mem.Allocator,
        config: Config,
        random: std.Random,
    ) !TeacherForcedEncoderDecoder {
        try config.validate();
        const feature_count = try std.math.add(
            usize,
            config.vocabulary_size,
            config.maximum_tokens,
        );
        const weights = try allocator.alloc(
            f32,
            try std.math.mul(usize, feature_count, config.vocabulary_size),
        );
        errdefer allocator.free(weights);
        const bias = try allocator.alloc(f32, config.vocabulary_size);
        const scale = @sqrt(2.0 / @as(f32, @floatFromInt(feature_count)));
        for (weights) |*weight| {
            weight.* = (random.float(f32) * 2 - 1) * scale;
        }
        @memset(bias, 0);
        return .{
            .allocator = allocator,
            .config = config,
            .feature_count = feature_count,
            .weights = weights,
            .bias = bias,
        };
    }

    pub fn deinit(self: *TeacherForcedEncoderDecoder) void {
        self.allocator.free(self.weights);
        self.allocator.free(self.bias);
        self.* = undefined;
    }

    pub fn lossAndGradient(
        self: TeacherForcedEncoderDecoder,
        padded_source: []const usize,
        source_length: usize,
        decoder_inputs: []const usize,
        targets: []const usize,
    ) !Objective {
        try self.validateExample(padded_source, source_length, decoder_inputs, targets);
        const weight_gradients = try self.allocator.alloc(f32, self.weights.len);
        errdefer self.allocator.free(weight_gradients);
        @memset(weight_gradients, 0);
        const bias_gradients = try self.allocator.alloc(f32, self.bias.len);
        errdefer self.allocator.free(bias_gradients);
        @memset(bias_gradients, 0);
        const features = try self.allocator.alloc(f32, self.feature_count);
        defer self.allocator.free(features);
        const alignment = try self.allocator.alloc(f32, source_length);
        defer self.allocator.free(alignment);
        const logits = try self.allocator.alloc(f32, self.config.vocabulary_size);
        defer self.allocator.free(logits);
        const probabilities = try self.allocator.alloc(f32, self.config.vocabulary_size);
        defer self.allocator.free(probabilities);

        var total_loss: f32 = 0;
        for (targets, 0..) |target, position| {
            try self.decoderFeatures(
                padded_source,
                source_length,
                decoder_inputs[0 .. position + 1],
                position,
                features,
                alignment,
            );
            self.project(features, logits);
            softmax(logits, probabilities);
            total_loss -= @log(@max(probabilities[target], 1e-9));
            for (probabilities, 0..) |probability, token| {
                const logit_gradient = probability -
                    @as(f32, if (token == target) 1 else 0);
                bias_gradients[token] += logit_gradient;
                for (features, 0..) |feature, channel| {
                    weight_gradients[channel * self.config.vocabulary_size + token] +=
                        feature * logit_gradient;
                }
            }
        }
        const inverse_tokens = 1.0 / @as(f32, @floatFromInt(targets.len));
        for (weight_gradients) |*gradient| gradient.* *= inverse_tokens;
        for (bias_gradients) |*gradient| gradient.* *= inverse_tokens;
        return .{
            .loss = total_loss * inverse_tokens,
            .gradients = .{
                .allocator = self.allocator,
                .weights = weight_gradients,
                .bias = bias_gradients,
            },
        };
    }

    pub fn applyGradient(
        self: *TeacherForcedEncoderDecoder,
        gradients: Gradients,
        learning_rate: f32,
    ) !void {
        if (!std.math.isFinite(learning_rate) or learning_rate <= 0 or
            gradients.weights.len != self.weights.len or
            gradients.bias.len != self.bias.len)
        {
            return error.InvalidGradient;
        }
        for (self.weights, gradients.weights) |*weight, gradient| {
            weight.* -= learning_rate * gradient;
        }
        for (self.bias, gradients.bias) |*bias, gradient| {
            bias.* -= learning_rate * gradient;
        }
    }

    pub fn nextLogProbabilities(
        self: TeacherForcedEncoderDecoder,
        padded_source: []const usize,
        source_length: usize,
        prefix: []const usize,
        output: []f64,
    ) !void {
        if (prefix.len == 0 or prefix.len > self.config.maximum_tokens or
            output.len != self.config.vocabulary_size)
        {
            return error.DimensionMismatch;
        }
        const features = try self.allocator.alloc(f32, self.feature_count);
        defer self.allocator.free(features);
        const alignment = try self.allocator.alloc(f32, source_length);
        defer self.allocator.free(alignment);
        const logits = try self.allocator.alloc(f32, self.config.vocabulary_size);
        defer self.allocator.free(logits);
        const probabilities = try self.allocator.alloc(f32, self.config.vocabulary_size);
        defer self.allocator.free(probabilities);
        try self.decoderFeatures(
            padded_source,
            source_length,
            prefix,
            prefix.len - 1,
            features,
            alignment,
        );
        self.project(features, logits);
        softmax(logits, probabilities);
        for (output, probabilities) |*log_probability, probability| {
            log_probability.* = @log(@as(f64, @floatCast(@max(probability, 1e-12))));
        }
        output[self.config.padding_token] = -100;
        output[self.config.beginning_token] = -100;
    }

    pub fn crossAttentionAlignment(
        self: TeacherForcedEncoderDecoder,
        allocator: std.mem.Allocator,
        padded_source: []const usize,
        source_length: usize,
        prefix: []const usize,
        target_position: usize,
    ) ![]f32 {
        const features = try self.allocator.alloc(f32, self.feature_count);
        defer self.allocator.free(features);
        const result = try allocator.alloc(f32, source_length);
        errdefer allocator.free(result);
        try self.decoderFeatures(
            padded_source,
            source_length,
            prefix,
            target_position,
            features,
            result,
        );
        return result;
    }

    fn validateExample(
        self: TeacherForcedEncoderDecoder,
        padded_source: []const usize,
        source_length: usize,
        decoder_inputs: []const usize,
        targets: []const usize,
    ) !void {
        if (source_length == 0 or source_length > padded_source.len or
            padded_source.len > self.config.maximum_tokens or
            decoder_inputs.len == 0 or decoder_inputs.len != targets.len or
            decoder_inputs.len > self.config.maximum_tokens)
        {
            return error.DimensionMismatch;
        }
        for (padded_source[0..source_length]) |token| {
            if (token >= self.config.vocabulary_size or
                token == self.config.padding_token)
            {
                return error.InvalidToken;
            }
        }
        for (decoder_inputs) |token| if (token >= self.config.vocabulary_size) {
            return error.InvalidToken;
        };
        for (targets) |token| if (token >= self.config.vocabulary_size) {
            return error.InvalidToken;
        };
    }

    fn decoderFeatures(
        self: TeacherForcedEncoderDecoder,
        padded_source: []const usize,
        source_length: usize,
        prefix: []const usize,
        target_position: usize,
        features: []f32,
        alignment: []f32,
    ) !void {
        if (source_length == 0 or source_length > padded_source.len or
            source_length > self.config.maximum_tokens or prefix.len == 0 or
            target_position >= self.config.maximum_tokens or
            features.len != self.feature_count or alignment.len != source_length)
        {
            return error.DimensionMismatch;
        }
        @memset(features, 0);
        const inverse_prefix = 1.0 / @as(f32, @floatFromInt(prefix.len));
        for (prefix) |token| {
            if (token >= self.config.vocabulary_size) return error.InvalidToken;
            features[token] += 0.35 * inverse_prefix;
        }

        const attended_position = @min(target_position, source_length - 1);
        var total: f32 = 0;
        for (alignment, 0..) |*probability, source_position| {
            probability.* = @exp(if (source_position == attended_position) 4.0 else 0.0);
            total += probability.*;
        }
        for (alignment, 0..) |*probability, source_position| {
            probability.* /= total;
            const token = padded_source[source_position];
            if (token >= self.config.vocabulary_size or
                token == self.config.padding_token)
            {
                return error.InvalidToken;
            }
            features[token] += probability.*;
            features[self.config.vocabulary_size + source_position] +=
                probability.*;
        }
    }

    fn project(self: TeacherForcedEncoderDecoder, features: []const f32, logits: []f32) void {
        @memcpy(logits, self.bias);
        for (features, 0..) |feature, channel| {
            for (logits, 0..) |*logit, token| {
                logit.* += feature *
                    self.weights[channel * self.config.vocabulary_size + token];
            }
        }
    }
};

fn softmax(logits: []const f32, probabilities: []f32) void {
    var maximum = logits[0];
    for (logits[1..]) |logit| maximum = @max(maximum, logit);
    var total: f32 = 0;
    for (logits, probabilities) |logit, *probability| {
        probability.* = @exp(logit - maximum);
        total += probability.*;
    }
    for (probabilities) |*probability| probability.* /= total;
}

test "teacher forcing learns token translation with masked alignments" {
    const config: Config = .{
        .vocabulary_size = 9,
        .maximum_tokens = 4,
        .padding_token = 0,
        .beginning_token = 1,
        .end_token = 2,
    };
    var prng = std.Random.DefaultPrng.init(7);
    var model = try TeacherForcedEncoderDecoder.init(
        std.testing.allocator,
        config,
        prng.random(),
    );
    defer model.deinit();
    const sources = [_][3]usize{ .{ 3, 4, 0 }, .{ 4, 3, 0 } };
    const inputs = [_][3]usize{ .{ 1, 5, 6 }, .{ 1, 6, 5 } };
    const targets = [_][3]usize{ .{ 5, 6, 2 }, .{ 6, 5, 2 } };
    var initial_loss: f32 = 0;
    for (sources, inputs, targets) |source, input, target| {
        var objective = try model.lossAndGradient(&source, 2, &input, &target);
        defer objective.deinit();
        initial_loss += objective.loss;
    }
    for (0..160) |_| {
        for (sources, inputs, targets) |source, input, target| {
            var objective = try model.lossAndGradient(&source, 2, &input, &target);
            defer objective.deinit();
            try model.applyGradient(objective.gradients, 0.35);
        }
    }
    var final_loss: f32 = 0;
    for (sources, inputs, targets) |source, input, target| {
        var objective = try model.lossAndGradient(&source, 2, &input, &target);
        defer objective.deinit();
        final_loss += objective.loss;
    }
    try std.testing.expect(final_loss < initial_loss * 0.2);
    const alignment = try model.crossAttentionAlignment(
        std.testing.allocator,
        &sources[0],
        2,
        inputs[0][0..2],
        1,
    );
    defer std.testing.allocator.free(alignment);
    try std.testing.expect(alignment[1] > alignment[0] * 20);
}
