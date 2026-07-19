const std = @import("std");
const tensor = @import("tensor.zig");
const modules = @import("modules.zig");

const ExecutionContext = tensor.ExecutionContext;
const Tensor = tensor.Tensor;
const Linear = modules.Linear;
const LinearGradients = modules.LinearGradients;

/// One device-resident gated recurrent unit cell. The update convention is
/// `h_next = (1 - update) * candidate + update * h_previous`.
pub const Gru = struct {
    input_update: Linear,
    hidden_update: Linear,
    input_reset: Linear,
    hidden_reset: Linear,
    input_candidate: Linear,
    hidden_candidate: Linear,
    input_size: usize,
    hidden_size: usize,

    pub fn init(
        context: *ExecutionContext,
        input_size: usize,
        hidden_size: usize,
        random: std.Random,
    ) !Gru {
        if (input_size == 0 or hidden_size == 0) return error.InvalidDimension;
        var input_update = try Linear.init(context, input_size, hidden_size, random);
        errdefer input_update.deinit();
        var hidden_update = try Linear.init(context, hidden_size, hidden_size, random);
        errdefer hidden_update.deinit();
        var input_reset = try Linear.init(context, input_size, hidden_size, random);
        errdefer input_reset.deinit();
        var hidden_reset = try Linear.init(context, hidden_size, hidden_size, random);
        errdefer hidden_reset.deinit();
        var input_candidate = try Linear.init(context, input_size, hidden_size, random);
        errdefer input_candidate.deinit();
        var hidden_candidate = try Linear.init(context, hidden_size, hidden_size, random);
        errdefer hidden_candidate.deinit();
        return .{
            .input_update = input_update,
            .hidden_update = hidden_update,
            .input_reset = input_reset,
            .hidden_reset = hidden_reset,
            .input_candidate = input_candidate,
            .hidden_candidate = hidden_candidate,
            .input_size = input_size,
            .hidden_size = hidden_size,
        };
    }

    pub fn deinit(self: *Gru) void {
        self.input_update.deinit();
        self.hidden_update.deinit();
        self.input_reset.deinit();
        self.hidden_reset.deinit();
        self.input_candidate.deinit();
        self.hidden_candidate.deinit();
        self.* = undefined;
    }

    pub fn forward(
        self: *const Gru,
        context: *ExecutionContext,
        input: Tensor,
        previous_hidden: Tensor,
    ) !GruForward {
        try self.validateInputs(input, previous_hidden);

        var input_update = try self.input_update.forward(context, input);
        defer input_update.deinit();
        var hidden_update = try self.hidden_update.forward(context, previous_hidden);
        defer hidden_update.deinit();
        var pre_update = try context.add(input_update, hidden_update);
        errdefer pre_update.deinit();
        var update = try context.activate(pre_update, .sigmoid);
        errdefer update.deinit();

        var input_reset = try self.input_reset.forward(context, input);
        defer input_reset.deinit();
        var hidden_reset = try self.hidden_reset.forward(context, previous_hidden);
        defer hidden_reset.deinit();
        var pre_reset = try context.add(input_reset, hidden_reset);
        errdefer pre_reset.deinit();
        var reset = try context.activate(pre_reset, .sigmoid);
        errdefer reset.deinit();

        var input_candidate = try self.input_candidate.forward(context, input);
        defer input_candidate.deinit();
        var hidden_candidate = try self.hidden_candidate.forward(context, previous_hidden);
        errdefer hidden_candidate.deinit();
        var gated_hidden = try context.multiply(reset, hidden_candidate);
        defer gated_hidden.deinit();
        var pre_candidate = try context.add(input_candidate, gated_hidden);
        errdefer pre_candidate.deinit();
        var candidate = try context.activate(pre_candidate, .tanh);
        errdefer candidate.deinit();

        var hidden_difference = try context.subtract(previous_hidden, candidate);
        defer hidden_difference.deinit();
        var retained_hidden = try context.multiply(update, hidden_difference);
        defer retained_hidden.deinit();
        const output = try context.add(candidate, retained_hidden);
        return .{
            .output = output,
            .cache = .{
                .input = input,
                .previous_hidden = previous_hidden,
                .pre_update = pre_update,
                .update = update,
                .pre_reset = pre_reset,
                .reset = reset,
                .hidden_candidate = hidden_candidate,
                .pre_candidate = pre_candidate,
                .candidate = candidate,
            },
        };
    }

    pub fn backward(
        self: *const Gru,
        context: *ExecutionContext,
        cache: *const GruCache,
        output_gradient: Tensor,
    ) !GruGradients {
        try self.validateInputs(cache.input, cache.previous_hidden);
        if (!output_gradient.shape.eql(cache.candidate.shape)) return error.DimensionMismatch;

        var hidden_difference = try context.subtract(
            cache.previous_hidden,
            cache.candidate,
        );
        defer hidden_difference.deinit();
        var update_gradient = try context.multiply(output_gradient, hidden_difference);
        defer update_gradient.deinit();
        var direct_hidden_gradient = try context.multiply(output_gradient, cache.update);
        defer direct_hidden_gradient.deinit();
        var retained_candidate_gradient = try context.multiply(output_gradient, cache.update);
        defer retained_candidate_gradient.deinit();
        var candidate_gradient = try context.subtract(
            output_gradient,
            retained_candidate_gradient,
        );
        defer candidate_gradient.deinit();

        var candidate_derivative = try context.activationDerivative(cache.pre_candidate, .tanh);
        defer candidate_derivative.deinit();
        var pre_candidate_gradient = try context.multiply(
            candidate_gradient,
            candidate_derivative,
        );
        defer pre_candidate_gradient.deinit();
        var input_candidate = try self.input_candidate.backward(
            context,
            cache.input,
            pre_candidate_gradient,
        );
        errdefer input_candidate.deinit();
        var hidden_candidate_gradient = try context.multiply(
            pre_candidate_gradient,
            cache.reset,
        );
        defer hidden_candidate_gradient.deinit();
        var hidden_candidate = try self.hidden_candidate.backward(
            context,
            cache.previous_hidden,
            hidden_candidate_gradient,
        );
        errdefer hidden_candidate.deinit();
        var reset_gradient = try context.multiply(
            pre_candidate_gradient,
            cache.hidden_candidate,
        );
        defer reset_gradient.deinit();

        var reset_derivative = try context.activationDerivative(cache.pre_reset, .sigmoid);
        defer reset_derivative.deinit();
        var pre_reset_gradient = try context.multiply(reset_gradient, reset_derivative);
        defer pre_reset_gradient.deinit();
        var input_reset = try self.input_reset.backward(
            context,
            cache.input,
            pre_reset_gradient,
        );
        errdefer input_reset.deinit();
        var hidden_reset = try self.hidden_reset.backward(
            context,
            cache.previous_hidden,
            pre_reset_gradient,
        );
        errdefer hidden_reset.deinit();

        var update_derivative = try context.activationDerivative(cache.pre_update, .sigmoid);
        defer update_derivative.deinit();
        var pre_update_gradient = try context.multiply(update_gradient, update_derivative);
        defer pre_update_gradient.deinit();
        var input_update = try self.input_update.backward(
            context,
            cache.input,
            pre_update_gradient,
        );
        errdefer input_update.deinit();
        var hidden_update = try self.hidden_update.backward(
            context,
            cache.previous_hidden,
            pre_update_gradient,
        );
        errdefer hidden_update.deinit();

        var input_partial = try context.add(input_candidate.input, input_reset.input);
        defer input_partial.deinit();
        var input_gradient = try context.add(input_partial, input_update.input);
        errdefer input_gradient.deinit();
        var hidden_partial = try context.add(hidden_candidate.input, hidden_reset.input);
        defer hidden_partial.deinit();
        var hidden_recurrent = try context.add(hidden_partial, hidden_update.input);
        defer hidden_recurrent.deinit();
        var previous_hidden_gradient = try context.add(
            direct_hidden_gradient,
            hidden_recurrent,
        );
        errdefer previous_hidden_gradient.deinit();

        return .{
            .input = input_gradient,
            .previous_hidden = previous_hidden_gradient,
            .input_update = input_update,
            .hidden_update = hidden_update,
            .input_reset = input_reset,
            .hidden_reset = hidden_reset,
            .input_candidate = input_candidate,
            .hidden_candidate = hidden_candidate,
        };
    }

    pub fn parameterTensorCount(_: Gru) usize {
        return 12;
    }

    pub fn parameterCount(self: Gru) usize {
        return self.input_update.parameterCount() +
            self.hidden_update.parameterCount() +
            self.input_reset.parameterCount() +
            self.hidden_reset.parameterCount() +
            self.input_candidate.parameterCount() +
            self.hidden_candidate.parameterCount();
    }

    pub fn parameters(self: *Gru, output: []*Tensor) ![]*Tensor {
        if (output.len < self.parameterTensorCount()) return error.InsufficientBuffer;
        const linears = [_]*Linear{
            &self.input_update,
            &self.hidden_update,
            &self.input_reset,
            &self.hidden_reset,
            &self.input_candidate,
            &self.hidden_candidate,
        };
        var index: usize = 0;
        for (linears) |linear| {
            output[index] = &linear.weights;
            output[index + 1] = &linear.bias;
            index += 2;
        }
        return output[0..index];
    }

    fn validateInputs(self: Gru, input: Tensor, previous_hidden: Tensor) !void {
        if (input.shape.rank != 2 or previous_hidden.shape.rank != 2 or
            input.shape.dims[0] != previous_hidden.shape.dims[0] or
            input.shape.dims[1] != self.input_size or
            previous_hidden.shape.dims[1] != self.hidden_size)
        {
            return error.DimensionMismatch;
        }
        if (!input.matrix.backend.sameInstance(previous_hidden.matrix.backend)) {
            return error.BackendMismatch;
        }
    }
};

pub const GruForward = struct {
    output: Tensor,
    cache: GruCache,

    pub fn deinit(self: *GruForward) void {
        self.output.deinit();
        self.cache.deinit();
        self.* = undefined;
    }
};

pub const GruCache = struct {
    input: Tensor,
    previous_hidden: Tensor,
    pre_update: Tensor,
    update: Tensor,
    pre_reset: Tensor,
    reset: Tensor,
    hidden_candidate: Tensor,
    pre_candidate: Tensor,
    candidate: Tensor,

    pub fn deinit(self: *GruCache) void {
        self.pre_update.deinit();
        self.update.deinit();
        self.pre_reset.deinit();
        self.reset.deinit();
        self.hidden_candidate.deinit();
        self.pre_candidate.deinit();
        self.candidate.deinit();
        self.* = undefined;
    }
};

pub const GruGradients = struct {
    input: Tensor,
    previous_hidden: Tensor,
    input_update: LinearGradients,
    hidden_update: LinearGradients,
    input_reset: LinearGradients,
    hidden_reset: LinearGradients,
    input_candidate: LinearGradients,
    hidden_candidate: LinearGradients,

    pub fn deinit(self: *GruGradients) void {
        self.input.deinit();
        self.previous_hidden.deinit();
        self.input_update.deinit();
        self.hidden_update.deinit();
        self.input_reset.deinit();
        self.hidden_reset.deinit();
        self.input_candidate.deinit();
        self.hidden_candidate.deinit();
        self.* = undefined;
    }

    pub fn parameterTensorCount(_: GruGradients) usize {
        return 12;
    }

    pub fn parameterGradients(self: GruGradients, output: []Tensor) ![]Tensor {
        if (output.len < self.parameterTensorCount()) return error.InsufficientBuffer;
        const gradients = [_]LinearGradients{
            self.input_update,
            self.hidden_update,
            self.input_reset,
            self.hidden_reset,
            self.input_candidate,
            self.hidden_candidate,
        };
        var index: usize = 0;
        for (gradients) |gradient| {
            output[index] = gradient.weights;
            output[index + 1] = gradient.bias;
            index += 2;
        }
        return output[0..index];
    }
};

test "GRU exposes forward, backward, and parameter shapes" {
    const testing = std.testing;
    var device = try tensor.Device.init(testing.allocator, .cpu);
    defer device.deinit();
    var context = ExecutionContext.init(&device);
    var prng = std.Random.DefaultPrng.init(5);
    var gru = try Gru.init(&context, 2, 3, prng.random());
    defer gru.deinit();
    var input = try context.upload(&.{ 2, 2 }, &.{ 0.1, 0.2, 0.3, 0.4 });
    defer input.deinit();
    var hidden = try context.upload(&.{ 2, 3 }, &.{ 0, 0, 0, 0.1, 0.2, 0.3 });
    defer hidden.deinit();
    var forward = try gru.forward(&context, input, hidden);
    defer forward.deinit();
    try testing.expect(forward.output.shape.eql(try tensor.Shape.init(&.{ 2, 3 })));
    var output_gradient = try context.upload(&.{ 2, 3 }, &.{ 1, 1, 1, 1, 1, 1 });
    defer output_gradient.deinit();
    var gradients = try gru.backward(&context, &forward.cache, output_gradient);
    defer gradients.deinit();
    try testing.expect(gradients.input.shape.eql(input.shape));
    try testing.expect(gradients.previous_hidden.shape.eql(hidden.shape));
    try testing.expectEqual(@as(usize, 12), gru.parameterTensorCount());
    try testing.expectEqual(@as(usize, 63), gru.parameterCount());
}

test "GRU rejects hidden state from a separate backend instance" {
    const testing = std.testing;
    var first_device = try tensor.Device.init(testing.allocator, .cpu);
    defer first_device.deinit();
    var second_device = try tensor.Device.init(testing.allocator, .cpu);
    defer second_device.deinit();
    var first_context = ExecutionContext.init(&first_device);
    var second_context = ExecutionContext.init(&second_device);
    var prng = std.Random.DefaultPrng.init(7);
    var gru = try Gru.init(&first_context, 2, 2, prng.random());
    defer gru.deinit();
    var input = try first_context.upload(&.{ 1, 2 }, &.{ 0.1, 0.2 });
    defer input.deinit();
    var hidden = try second_context.upload(&.{ 1, 2 }, &.{ 0.3, 0.4 });
    defer hidden.deinit();

    try testing.expectError(
        error.BackendMismatch,
        gru.forward(&first_context, input, hidden),
    );
}

test "GRU input and recurrent gradients match finite differences" {
    const testing = std.testing;
    var device = try tensor.Device.init(testing.allocator, .cpu);
    defer device.deinit();
    var context = ExecutionContext.init(&device);
    var prng = std.Random.DefaultPrng.init(11);
    var gru = try Gru.init(&context, 2, 2, prng.random());
    defer gru.deinit();
    const input_values = [_]f32{ 0.2, -0.35 };
    var input = try context.upload(&.{ 1, 2 }, &input_values);
    defer input.deinit();
    const hidden_values = [_]f32{ 0.1, -0.2 };
    var hidden = try context.upload(&.{ 1, 2 }, &hidden_values);
    defer hidden.deinit();
    var forward = try gru.forward(&context, input, hidden);
    defer forward.deinit();
    var output_gradient = try context.upload(&.{ 1, 2 }, &.{ 1, 1 });
    defer output_gradient.deinit();
    var gradients = try gru.backward(&context, &forward.cache, output_gradient);
    defer gradients.deinit();
    var analytical: [2]f32 = undefined;
    try context.readback(gradients.input, &analytical);
    var analytical_hidden: [2]f32 = undefined;
    try context.readback(gradients.previous_hidden, &analytical_hidden);

    const epsilon: f32 = 1e-3;
    for (0..2) |input_index| {
        var plus_values = input_values;
        plus_values[input_index] += epsilon;
        var plus_input = try context.upload(&.{ 1, 2 }, &plus_values);
        defer plus_input.deinit();
        var plus_forward = try gru.forward(&context, plus_input, hidden);
        defer plus_forward.deinit();
        var plus_output: [2]f32 = undefined;
        try context.readback(plus_forward.output, &plus_output);

        var minus_values = input_values;
        minus_values[input_index] -= epsilon;
        var minus_input = try context.upload(&.{ 1, 2 }, &minus_values);
        defer minus_input.deinit();
        var minus_forward = try gru.forward(&context, minus_input, hidden);
        defer minus_forward.deinit();
        var minus_output: [2]f32 = undefined;
        try context.readback(minus_forward.output, &minus_output);
        const numerical = ((plus_output[0] + plus_output[1]) -
            (minus_output[0] + minus_output[1])) / (2 * epsilon);
        try testing.expectApproxEqAbs(numerical, analytical[input_index], 3e-3);
    }
    for (0..2) |hidden_index| {
        var plus_values = hidden_values;
        plus_values[hidden_index] += epsilon;
        var plus_hidden = try context.upload(&.{ 1, 2 }, &plus_values);
        defer plus_hidden.deinit();
        var plus_forward = try gru.forward(&context, input, plus_hidden);
        defer plus_forward.deinit();
        var plus_output: [2]f32 = undefined;
        try context.readback(plus_forward.output, &plus_output);

        var minus_values = hidden_values;
        minus_values[hidden_index] -= epsilon;
        var minus_hidden = try context.upload(&.{ 1, 2 }, &minus_values);
        defer minus_hidden.deinit();
        var minus_forward = try gru.forward(&context, input, minus_hidden);
        defer minus_forward.deinit();
        var minus_output: [2]f32 = undefined;
        try context.readback(minus_forward.output, &minus_output);
        const numerical = ((plus_output[0] + plus_output[1]) -
            (minus_output[0] + minus_output[1])) / (2 * epsilon);
        try testing.expectApproxEqAbs(numerical, analytical_hidden[hidden_index], 3e-3);
    }
}

test "GRU forward and backward stay on each available device" {
    const testing = std.testing;
    const preferences = [_]tensor.DevicePreference{ .cpu, .metal, .cuda, .rocm };
    for (preferences) |preference| {
        var device = tensor.Device.init(testing.allocator, preference) catch |err| switch (err) {
            error.BackendUnavailable => continue,
            else => return err,
        };
        defer device.deinit();
        var context = ExecutionContext.init(&device);
        var prng = std.Random.DefaultPrng.init(19);
        var gru = try Gru.init(&context, 2, 2, prng.random());
        defer gru.deinit();
        var input = try context.upload(&.{ 1, 2 }, &.{ 0.2, -0.1 });
        defer input.deinit();
        var hidden = try context.upload(&.{ 1, 2 }, &.{ 0.1, 0.3 });
        defer hidden.deinit();
        var forward = try gru.forward(&context, input, hidden);
        defer forward.deinit();
        var output_gradient = try context.upload(&.{ 1, 2 }, &.{ 1, 1 });
        defer output_gradient.deinit();
        var gradients = try gru.backward(&context, &forward.cache, output_gradient);
        defer gradients.deinit();
        try testing.expectEqual(@as(usize, 0), context.stats.readbacks);
    }
}
