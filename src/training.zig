const std = @import("std");
const tensor = @import("tensor.zig");

const BackendInstance = @import("backend.zig").BackendInstance;
const ExecutionContext = tensor.ExecutionContext;
const Tensor = tensor.Tensor;

pub const OptimizerKind = enum {
    sgd,
    momentum,
    adamw,
};

pub const OptimizerConfig = struct {
    kind: OptimizerKind = .adamw,
    learning_rate: f32,
    momentum: f32 = 0.9,
    beta1: f32 = 0.9,
    beta2: f32 = 0.999,
    epsilon: f32 = 1e-8,
    weight_decay: f32 = 0,
    max_gradient_norm: f32 = 0,
    accumulation_steps: usize = 1,

    pub fn sgd(learning_rate: f32) OptimizerConfig {
        return .{ .kind = .sgd, .learning_rate = learning_rate };
    }

    pub fn withMomentum(learning_rate: f32, coefficient: f32) OptimizerConfig {
        return .{
            .kind = .momentum,
            .learning_rate = learning_rate,
            .momentum = coefficient,
        };
    }

    pub fn adamw(learning_rate: f32) OptimizerConfig {
        return .{ .kind = .adamw, .learning_rate = learning_rate };
    }

    pub fn validate(self: OptimizerConfig) !void {
        if (!std.math.isFinite(self.learning_rate) or self.learning_rate <= 0) {
            return error.InvalidLearningRate;
        }
        if (!std.math.isFinite(self.weight_decay) or self.weight_decay < 0) {
            return error.InvalidWeightDecay;
        }
        if (!std.math.isFinite(self.max_gradient_norm) or self.max_gradient_norm < 0) {
            return error.InvalidGradientNorm;
        }
        if (self.accumulation_steps == 0) return error.InvalidAccumulationSteps;

        switch (self.kind) {
            .sgd => {},
            .momentum => {
                if (!std.math.isFinite(self.momentum) or self.momentum < 0 or self.momentum >= 1) {
                    return error.InvalidMomentum;
                }
            },
            .adamw => {
                if (!std.math.isFinite(self.beta1) or self.beta1 < 0 or self.beta1 >= 1 or
                    !std.math.isFinite(self.beta2) or self.beta2 < 0 or self.beta2 >= 1)
                {
                    return error.InvalidBeta;
                }
                if (!std.math.isFinite(self.epsilon) or self.epsilon <= 0) {
                    return error.InvalidEpsilon;
                }
            },
        }
    }
};

pub const StepResult = struct {
    updated: bool,
    pending_steps: usize,
    update_steps: usize,
};

const ParameterState = struct {
    first_moment: Tensor,
    second_moment: Tensor,
    accumulated_gradient: Tensor,

    fn deinit(self: *ParameterState) void {
        self.first_moment.deinit();
        self.second_moment.deinit();
        self.accumulated_gradient.deinit();
        self.* = undefined;
    }
};

/// Owns optimizer moments and gradient accumulators for a stable, ordered set
/// of caller-owned parameter Tensors.
pub const Optimizer = struct {
    allocator: std.mem.Allocator,
    config: OptimizerConfig,
    parameter_addresses: []usize,
    parameter_backends: []BackendInstance,
    states: []ParameterState,
    pending_steps: usize = 0,
    update_steps: usize = 0,

    pub fn init(
        context: *ExecutionContext,
        config: OptimizerConfig,
        parameters: []const *Tensor,
    ) !Optimizer {
        try config.validate();
        if (parameters.len == 0) return error.EmptyParameterList;

        const allocator = context.device.allocator;
        const addresses = try allocator.alloc(usize, parameters.len);
        errdefer allocator.free(addresses);
        const backends = try allocator.alloc(BackendInstance, parameters.len);
        errdefer allocator.free(backends);
        const states = try allocator.alloc(ParameterState, parameters.len);
        errdefer allocator.free(states);

        var initialized: usize = 0;
        errdefer for (states[0..initialized]) |*state| state.deinit();
        for (parameters, 0..) |parameter, index| {
            if (!parameter.matrix.backend.sameInstance(context.device.instance)) {
                return error.BackendMismatch;
            }
            var first_moment = try context.createTensor(parameter.shape.slice());
            errdefer first_moment.deinit();
            var second_moment = try context.createTensor(parameter.shape.slice());
            errdefer second_moment.deinit();
            const accumulated_gradient = try context.createTensor(parameter.shape.slice());
            states[index] = .{
                .first_moment = first_moment,
                .second_moment = second_moment,
                .accumulated_gradient = accumulated_gradient,
            };
            addresses[index] = @intFromPtr(parameter);
            backends[index] = parameter.matrix.backend;
            initialized += 1;
        }

        return .{
            .allocator = allocator,
            .config = config,
            .parameter_addresses = addresses,
            .parameter_backends = backends,
            .states = states,
        };
    }

    pub fn deinit(self: *Optimizer) void {
        for (self.states) |*state| state.deinit();
        self.allocator.free(self.states);
        self.allocator.free(self.parameter_addresses);
        self.allocator.free(self.parameter_backends);
        self.* = undefined;
    }

    pub fn step(
        self: *Optimizer,
        context: *ExecutionContext,
        parameters: []const *Tensor,
        gradients: []const Tensor,
    ) !StepResult {
        try self.validateBindings(context, parameters, gradients);
        for (self.states, gradients) |*state, gradient| {
            const accumulated = try context.add(state.accumulated_gradient, gradient);
            state.accumulated_gradient.deinit();
            state.accumulated_gradient = accumulated;
        }
        self.pending_steps += 1;
        if (self.pending_steps < self.config.accumulation_steps) {
            return self.result(false);
        }
        try self.applyUpdate(context, parameters);
        return self.result(true);
    }

    /// Applies a final averaged update when an epoch ends with a partial
    /// accumulation window. Returns false when no gradients are pending.
    pub fn flush(self: *Optimizer, context: *ExecutionContext, parameters: []const *Tensor) !StepResult {
        if (self.pending_steps == 0) return self.result(false);
        try self.validateParameters(context, parameters);
        try self.applyUpdate(context, parameters);
        return self.result(true);
    }

    pub fn setLearningRate(self: *Optimizer, learning_rate: f32) !void {
        if (!std.math.isFinite(learning_rate) or learning_rate <= 0) {
            return error.InvalidLearningRate;
        }
        self.config.learning_rate = learning_rate;
    }

    fn applyUpdate(self: *Optimizer, context: *ExecutionContext, parameters: []const *Tensor) !void {
        const averaged = try self.allocator.alloc(Tensor, self.states.len);
        defer self.allocator.free(averaged);
        var averaged_count: usize = 0;
        defer for (averaged[0..averaged_count]) |*gradient| gradient.deinit();

        const divisor = @as(f32, @floatFromInt(self.pending_steps));
        for (self.states, 0..) |state, index| {
            averaged[index] = try context.scale(state.accumulated_gradient, 1 / divisor);
            averaged_count += 1;
        }

        var total_squares = try context.sumSquares(averaged[0]);
        defer total_squares.deinit();
        for (averaged[1..]) |gradient| {
            var part = try context.sumSquares(gradient);
            defer part.deinit();
            const combined = try context.add(total_squares, part);
            total_squares.deinit();
            total_squares = combined;
        }

        const next_update = self.update_steps + 1;
        const update_config = self.backendConfig(next_update);
        for (parameters, averaged, self.states) |parameter, gradient, *state| {
            try context.optimizerUpdate(
                parameter,
                gradient,
                &state.first_moment,
                &state.second_moment,
                total_squares,
                update_config,
            );
        }

        for (self.states) |*state| {
            const zeroed = try context.scale(state.accumulated_gradient, 0);
            state.accumulated_gradient.deinit();
            state.accumulated_gradient = zeroed;
        }
        self.pending_steps = 0;
        self.update_steps = next_update;
    }

    fn backendConfig(self: Optimizer, update_step: usize) tensor.OptimizerUpdateConfig {
        const step_number = @as(f32, @floatFromInt(update_step));
        const kind: tensor.OptimizerUpdateKind = switch (self.config.kind) {
            .sgd => .sgd,
            .momentum => .momentum,
            .adamw => .adamw,
        };
        return .{
            .kind = kind,
            .learning_rate = self.config.learning_rate,
            .beta1 = if (self.config.kind == .momentum) self.config.momentum else self.config.beta1,
            .beta2 = self.config.beta2,
            .epsilon = self.config.epsilon,
            .weight_decay = self.config.weight_decay,
            .bias_correction1 = if (self.config.kind == .adamw)
                1 - std.math.pow(f32, self.config.beta1, step_number)
            else
                1,
            .bias_correction2 = if (self.config.kind == .adamw)
                1 - std.math.pow(f32, self.config.beta2, step_number)
            else
                1,
            .max_gradient_norm = self.config.max_gradient_norm,
        };
    }

    fn validateBindings(
        self: Optimizer,
        context: *ExecutionContext,
        parameters: []const *Tensor,
        gradients: []const Tensor,
    ) !void {
        try self.validateParameters(context, parameters);
        if (gradients.len != self.states.len) return error.ParameterCountMismatch;
        for (parameters, gradients) |parameter, gradient| {
            if (!parameter.shape.eql(gradient.shape)) return error.DimensionMismatch;
            if (!parameter.matrix.backend.sameInstance(gradient.matrix.backend)) {
                return error.BackendMismatch;
            }
        }
    }

    fn validateParameters(
        self: Optimizer,
        context: *ExecutionContext,
        parameters: []const *Tensor,
    ) !void {
        if (parameters.len != self.states.len) return error.ParameterCountMismatch;
        for (parameters, self.parameter_addresses, self.parameter_backends, self.states) |parameter, address, backend, state| {
            if (@intFromPtr(parameter) != address) return error.ParameterOrderChanged;
            if (!parameter.matrix.backend.sameInstance(backend) or
                !parameter.matrix.backend.sameInstance(context.device.instance))
            {
                return error.BackendMismatch;
            }
            if (!parameter.shape.eql(state.accumulated_gradient.shape)) return error.DimensionMismatch;
        }
    }

    fn result(self: Optimizer, updated: bool) StepResult {
        return .{
            .updated = updated,
            .pending_steps = self.pending_steps,
            .update_steps = self.update_steps,
        };
    }
};

test "optimizer validates configuration" {
    const testing = std.testing;

    try testing.expectError(error.InvalidLearningRate, OptimizerConfig.sgd(0).validate());
    try testing.expectError(error.InvalidMomentum, OptimizerConfig.withMomentum(0.1, 1).validate());
    try testing.expectError(error.InvalidAccumulationSteps, (OptimizerConfig{
        .kind = .adamw,
        .learning_rate = 0.1,
        .accumulation_steps = 0,
    }).validate());
}

test "optimizer accumulates and clips one global gradient" {
    const testing = std.testing;

    var device = try tensor.Device.init(testing.allocator, .cpu);
    defer device.deinit();
    var context = ExecutionContext.init(&device);
    var parameter = try context.upload(&.{ 1, 2 }, &.{ 1, 2 });
    defer parameter.deinit();
    var gradient = try context.upload(&.{ 1, 2 }, &.{ 3, 4 });
    defer gradient.deinit();
    const parameters = [_]*Tensor{&parameter};
    const gradients = [_]Tensor{gradient};
    var optimizer = try Optimizer.init(&context, .{
        .kind = .sgd,
        .learning_rate = 0.1,
        .max_gradient_norm = 1,
        .accumulation_steps = 2,
    }, &parameters);
    defer optimizer.deinit();

    const pending = try optimizer.step(&context, &parameters, &gradients);
    try testing.expect(!pending.updated);
    try testing.expectEqual(@as(usize, 1), pending.pending_steps);
    const updated = try optimizer.step(&context, &parameters, &gradients);
    try testing.expect(updated.updated);
    try testing.expectEqual(@as(usize, 1), updated.update_steps);
    try testing.expectEqual(@as(usize, 0), context.stats.readbacks);

    var actual: [2]f32 = undefined;
    try context.readback(parameter, &actual);
    try testing.expectApproxEqAbs(@as(f32, 0.94), actual[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 1.92), actual[1], 1e-5);
}

test "optimizer flushes partial AdamW accumulation" {
    const testing = std.testing;

    var device = try tensor.Device.init(testing.allocator, .cpu);
    defer device.deinit();
    var context = ExecutionContext.init(&device);
    var parameter = try context.upload(&.{ 1, 1 }, &.{1});
    defer parameter.deinit();
    var gradient = try context.upload(&.{ 1, 1 }, &.{2});
    defer gradient.deinit();
    const parameters = [_]*Tensor{&parameter};
    const gradients = [_]Tensor{gradient};
    var optimizer = try Optimizer.init(&context, .{
        .kind = .adamw,
        .learning_rate = 0.1,
        .weight_decay = 0.01,
        .accumulation_steps = 4,
    }, &parameters);
    defer optimizer.deinit();

    _ = try optimizer.step(&context, &parameters, &gradients);
    const flushed = try optimizer.flush(&context, &parameters);
    try testing.expect(flushed.updated);
    try testing.expectEqual(@as(usize, 1), flushed.update_steps);
    try testing.expectEqual(@as(usize, 0), context.stats.readbacks);

    var actual: [1]f32 = undefined;
    try context.readback(parameter, &actual);
    try testing.expectApproxEqAbs(@as(f32, 0.899), actual[0], 1e-5);
}

test "optimizer requires the context's exact backend instance" {
    const testing = std.testing;

    var first_device = try tensor.Device.init(testing.allocator, .cpu);
    defer first_device.deinit();
    var second_device = try tensor.Device.init(testing.allocator, .cpu);
    defer second_device.deinit();
    var first_context = ExecutionContext.init(&first_device);
    var second_context = ExecutionContext.init(&second_device);
    var parameter = try first_context.upload(&.{ 1, 1 }, &.{1});
    defer parameter.deinit();
    var foreign_parameter = try second_context.upload(&.{ 1, 1 }, &.{1});
    defer foreign_parameter.deinit();
    var foreign_gradient = try second_context.upload(&.{ 1, 1 }, &.{1});
    defer foreign_gradient.deinit();

    const foreign_parameters = [_]*Tensor{&foreign_parameter};
    try testing.expectError(
        error.BackendMismatch,
        Optimizer.init(&first_context, OptimizerConfig.sgd(0.1), &foreign_parameters),
    );

    const parameters = [_]*Tensor{&parameter};
    const gradients = [_]Tensor{foreign_gradient};
    var optimizer = try Optimizer.init(
        &first_context,
        OptimizerConfig.sgd(0.1),
        &parameters,
    );
    defer optimizer.deinit();
    try testing.expectError(
        error.BackendMismatch,
        optimizer.step(&first_context, &parameters, &gradients),
    );
}
