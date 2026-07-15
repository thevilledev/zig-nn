const std = @import("std");
const nn = @import("nn");

const state_count: usize = 5;
const action_count: usize = 2;
const hidden_features: usize = 12;
const parameter_tensors: usize = 4;
const replay_capacity: usize = 256;
const batch_size: usize = 16;
const target_sync_interval: usize = 25;
const discount: f32 = 0.9;
const default_episodes: usize = 100;
const default_seed: u64 = 42;

const Options = struct {
    backend: nn.DevicePreference = .auto,
    episodes: usize = default_episodes,
    seed: u64 = default_seed,
};

const LineWorld = struct {
    position: usize,
    steps: usize = 0,
    maximum_steps: usize = 8,

    const Step = struct {
        reward: f32,
        terminal: bool,
        reached_goal: bool,
    };

    fn init(start: usize) !LineWorld {
        if (start >= state_count - 1) return error.InvalidStart;
        return .{ .position = start };
    }

    fn step(self: *LineWorld, action: usize) !Step {
        if (action >= action_count) return error.InvalidAction;
        if (action == 0) {
            self.position -|= 1;
        } else {
            self.position = @min(self.position + 1, state_count - 1);
        }
        self.steps += 1;
        const reached_goal = self.position == state_count - 1;
        return .{
            .reward = if (reached_goal) 1 else -0.04,
            .terminal = reached_goal or self.steps >= self.maximum_steps,
            .reached_goal = reached_goal,
        };
    }
};

const Metrics = struct {
    success_rate: f32,
    average_steps: f32,
};

const ExperimentResult = struct {
    backend: nn.BackendType,
    initial: Metrics,
    final: Metrics,
    environment_steps: usize,
    optimization_steps: usize,
    replay_size: usize,
    target_syncs: usize,
    final_epsilon: f32,
    training_readbacks: usize,
    policy: [state_count - 1]usize,
    q_values: [state_count - 1][action_count]f32,
};

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const options = parseArgs(args[1..]) catch |err| {
        printUsage();
        if (err == error.HelpRequested) return;
        return err;
    };
    const result = try runExperiment(init.gpa, options);
    std.debug.print("DQN LineWorld lesson on {s}\n", .{backendName(result.backend)});
    std.debug.print(
        "task: reach state {d} using left/right actions\n" ++
            "model: one-hot state -> MLP Q-values, replay, target network\n",
        .{state_count - 1},
    );
    std.debug.print(
        "greedy success: {d:.1}% -> {d:.1}%, average steps: {d:.2}\n" ++
            "environment steps: {d}, optimizer updates: {d}, replay: {d}/{d}\n" ++
            "target syncs: {d}, epsilon: {d:.3}, host RL readbacks: {d}\n\n",
        .{
            result.initial.success_rate * 100,
            result.final.success_rate * 100,
            result.final.average_steps,
            result.environment_steps,
            result.optimization_steps,
            result.replay_size,
            replay_capacity,
            result.target_syncs,
            result.final_epsilon,
            result.training_readbacks,
        },
    );
    std.debug.print("learned greedy policy:\n", .{});
    for (result.policy, result.q_values, 0..) |action, values, state| {
        std.debug.print(
            "  state {d}: Q(left)={d:.3}, Q(right)={d:.3} -> {s}\n",
            .{ state, values[0], values[1], if (action == 0) "left" else "right" },
        );
    }
}

fn parseArgs(args: []const []const u8) !Options {
    var options: Options = .{};
    var index: usize = 0;
    while (index < args.len) : (index += 1) {
        const arg = args[index];
        if (std.mem.eql(u8, arg, "--backend")) {
            index += 1;
            if (index >= args.len) return error.MissingArgument;
            options.backend = try parseBackend(args[index]);
        } else if (std.mem.eql(u8, arg, "--episodes")) {
            index += 1;
            if (index >= args.len) return error.MissingArgument;
            options.episodes = try std.fmt.parseInt(usize, args[index], 10);
            if (options.episodes == 0) return error.InvalidEpisodeCount;
        } else if (std.mem.eql(u8, arg, "--seed")) {
            index += 1;
            if (index >= args.len) return error.MissingArgument;
            options.seed = try std.fmt.parseInt(u64, args[index], 10);
        } else if (std.mem.eql(u8, arg, "--help")) {
            return error.HelpRequested;
        } else {
            return error.UnknownArgument;
        }
    }
    return options;
}

fn parseBackend(value: []const u8) !nn.DevicePreference {
    if (std.mem.eql(u8, value, "cpu")) return .cpu;
    if (std.mem.eql(u8, value, "auto")) return .auto;
    if (std.mem.eql(u8, value, "metal")) return .metal;
    if (std.mem.eql(u8, value, "cuda")) return .cuda;
    if (std.mem.eql(u8, value, "rocm")) return .rocm;
    return error.UnknownBackend;
}

fn printUsage() void {
    std.debug.print(
        \\Usage: dqn [options]
        \\
        \\Options:
        \\  --backend <name>   cpu, auto, metal, cuda, or rocm (default: auto)
        \\  --episodes <count> training episodes (default: 100)
        \\  --seed <number>    model and exploration seed (default: 42)
        \\  --help             show this help
        \\
    , .{});
}

fn runExperiment(allocator: std.mem.Allocator, options: Options) !ExperimentResult {
    var device = try nn.Device.init(allocator, options.backend);
    defer device.deinit();
    var context = nn.ExecutionContext.init(&device);
    var online_prng = std.Random.DefaultPrng.init(options.seed);
    var online = try nn.Modules.Mlp.init(&context, .{
        .widths = &.{ state_count, hidden_features, action_count },
        .hidden_activation = .relu,
        .output_activation = .linear,
    }, online_prng.random());
    defer online.deinit();
    var target_prng = std.Random.DefaultPrng.init(options.seed);
    var target = try nn.Modules.Mlp.init(&context, .{
        .widths = &.{ state_count, hidden_features, action_count },
        .hidden_activation = .relu,
        .output_activation = .linear,
    }, target_prng.random());
    defer target.deinit();
    var parameter_buffer: [parameter_tensors]*nn.Tensor = undefined;
    const parameters = try online.parameters(&parameter_buffer);
    var optimizer = try nn.Training.Optimizer.init(&context, .{
        .kind = .adamw,
        .learning_rate = 0.01,
        .weight_decay = 0.0001,
        .max_gradient_norm = 5,
    }, parameters);
    defer optimizer.deinit();
    var replay = try nn.Reinforcement.ReplayBuffer.init(
        allocator,
        replay_capacity,
        state_count,
    );
    defer replay.deinit();
    const epsilon_schedule = try nn.Reinforcement.EpsilonSchedule.init(1, 0.05, 400);
    const initial = try evaluate(&context, &online, 24);
    context.resetStats();

    var prng = std.Random.DefaultPrng.init(options.seed ^ 0xa0761d6478bd642f);
    var random = prng.random();
    var environment_steps: usize = 0;
    var target_syncs: usize = 0;
    for (0..options.episodes) |episode| {
        var environment = try LineWorld.init(episode % (state_count - 1));
        while (true) {
            var state = oneHotState(environment.position);
            const epsilon = epsilon_schedule.value(environment_steps);
            const action = if (random.float(f32) < epsilon)
                random.uintLessThan(usize, action_count)
            else
                try greedyAction(&context, &online, &state, null);
            const transition = try environment.step(action);
            var next_state = oneHotState(environment.position);
            try replay.add(
                &state,
                action,
                transition.reward,
                &next_state,
                transition.terminal,
            );
            environment_steps += 1;

            if (replay.len >= batch_size) {
                var batch = try replay.sample(random, batch_size);
                defer batch.deinit();
                try trainBatch(
                    allocator,
                    &context,
                    &online,
                    &target,
                    &optimizer,
                    parameters,
                    batch,
                );
                if (optimizer.update_steps % target_sync_interval == 0) {
                    try copyParameters(&target, &online);
                    target_syncs += 1;
                }
            }
            if (transition.terminal) break;
        }
    }
    const training_stats = context.stats;
    const final = try evaluate(&context, &online, 40);
    var policy: [state_count - 1]usize = undefined;
    var q_values: [state_count - 1][action_count]f32 = undefined;
    for (0..state_count - 1) |state_index| {
        var state = oneHotState(state_index);
        policy[state_index] = try greedyAction(
            &context,
            &online,
            &state,
            &q_values[state_index],
        );
    }
    return .{
        .backend = device.backendType(),
        .initial = initial,
        .final = final,
        .environment_steps = environment_steps,
        .optimization_steps = optimizer.update_steps,
        .replay_size = replay.len,
        .target_syncs = target_syncs,
        .final_epsilon = epsilon_schedule.value(environment_steps),
        .training_readbacks = training_stats.readbacks,
        .policy = policy,
        .q_values = q_values,
    };
}

fn trainBatch(
    allocator: std.mem.Allocator,
    context: *nn.ExecutionContext,
    online: *const nn.Modules.Mlp,
    target: *const nn.Modules.Mlp,
    optimizer: *nn.Training.Optimizer,
    parameters: []const *nn.Tensor,
    batch: nn.Reinforcement.ReplayBatch,
) !void {
    var states = try context.upload(&.{ batch.batch_size, state_count }, batch.states);
    defer states.deinit();
    var next_states = try context.upload(&.{ batch.batch_size, state_count }, batch.next_states);
    defer next_states.deinit();
    try context.beginBatch();
    var batch_active = true;
    errdefer if (batch_active) context.endBatch() catch {};
    var current = try online.forward(context, states);
    defer current.deinit();
    var next = try target.forward(context, next_states);
    defer next.deinit();
    try context.endBatch();
    batch_active = false;

    const value_count = batch.batch_size * action_count;
    const current_values = try allocator.alloc(f32, value_count);
    defer allocator.free(current_values);
    const next_values = try allocator.alloc(f32, value_count);
    defer allocator.free(next_values);
    try context.readback(current.output, current_values);
    try context.readback(next.output, next_values);
    const output_gradient = try allocator.alloc(f32, value_count);
    defer allocator.free(output_gradient);
    @memset(output_gradient, 0);
    for (0..batch.batch_size) |index| {
        const offset = index * action_count;
        const next_maximum = @max(next_values[offset], next_values[offset + 1]);
        const target_value = try nn.Reinforcement.bellmanTarget(
            batch.rewards[index],
            next_maximum,
            batch.terminals[index],
            discount,
        );
        const action_index = offset + batch.actions[index];
        output_gradient[action_index] = 2 * (current_values[action_index] - target_value) /
            @as(f32, @floatFromInt(batch.batch_size));
    }
    var gradient = try context.upload(
        &.{ batch.batch_size, action_count },
        output_gradient,
    );
    defer gradient.deinit();

    try context.beginBatch();
    batch_active = true;
    var gradients = try online.backward(context, &current.cache, gradient);
    defer gradients.deinit();
    var gradient_buffer: [parameter_tensors]nn.Tensor = undefined;
    const parameter_gradients = try gradients.parameterGradients(&gradient_buffer);
    const update = try optimizer.step(context, parameters, parameter_gradients);
    std.debug.assert(update.updated);
    try context.endBatch();
    batch_active = false;
}

fn evaluate(
    context: *nn.ExecutionContext,
    model: *const nn.Modules.Mlp,
    episodes: usize,
) !Metrics {
    var successes: usize = 0;
    var total_steps: usize = 0;
    for (0..episodes) |episode| {
        var environment = try LineWorld.init(episode % (state_count - 1));
        while (true) {
            var state = oneHotState(environment.position);
            const action = try greedyAction(context, model, &state, null);
            const transition = try environment.step(action);
            if (transition.terminal) {
                if (transition.reached_goal) successes += 1;
                total_steps += environment.steps;
                break;
            }
        }
    }
    return .{
        .success_rate = @as(f32, @floatFromInt(successes)) /
            @as(f32, @floatFromInt(episodes)),
        .average_steps = @as(f32, @floatFromInt(total_steps)) /
            @as(f32, @floatFromInt(episodes)),
    };
}

fn greedyAction(
    context: *nn.ExecutionContext,
    model: *const nn.Modules.Mlp,
    state: *const [state_count]f32,
    values_out: ?*[action_count]f32,
) !usize {
    var input = try context.upload(&.{ 1, state_count }, state);
    defer input.deinit();
    try context.beginBatch();
    var batch_active = true;
    errdefer if (batch_active) context.endBatch() catch {};
    var forward = try model.forward(context, input);
    defer forward.deinit();
    try context.endBatch();
    batch_active = false;
    var values: [action_count]f32 = undefined;
    try context.readback(forward.output, &values);
    if (values_out) |output| output.* = values;
    return if (values[1] > values[0]) 1 else 0;
}

fn copyParameters(target: *nn.Modules.Mlp, source: *const nn.Modules.Mlp) !void {
    if (target.layers.len != source.layers.len) return error.DimensionMismatch;
    for (target.layers, source.layers) |*target_layer, source_layer| {
        if (!target_layer.weights.shape.eql(source_layer.weights.shape) or
            !target_layer.bias.shape.eql(source_layer.bias.shape))
        {
            return error.DimensionMismatch;
        }
        var weights = try source_layer.weights.copy();
        errdefer weights.deinit();
        const bias = try source_layer.bias.copy();
        target_layer.weights.deinit();
        target_layer.bias.deinit();
        target_layer.weights = weights;
        target_layer.bias = bias;
    }
}

fn oneHotState(position: usize) [state_count]f32 {
    var state = [_]f32{0} ** state_count;
    state[position] = 1;
    return state;
}

fn backendName(backend: nn.BackendType) []const u8 {
    return switch (backend) {
        .CPU => "cpu",
        .Metal => "metal",
        .CUDA => "cuda",
        .ROCm => "rocm",
    };
}

test "DQN options validate backend and episode count" {
    const options = try parseArgs(&.{ "--backend", "cpu", "--episodes", "12", "--seed", "7" });
    try std.testing.expectEqual(nn.DevicePreference.cpu, options.backend);
    try std.testing.expectEqual(@as(usize, 12), options.episodes);
    try std.testing.expectEqual(@as(u64, 7), options.seed);
    try std.testing.expectError(error.UnknownBackend, parseBackend("quantum"));
    try std.testing.expectError(error.InvalidEpisodeCount, parseArgs(&.{ "--episodes", "0" }));
}

test "LineWorld clamps left and rewards the goal" {
    var environment = try LineWorld.init(0);
    var transition = try environment.step(0);
    try std.testing.expectEqual(@as(usize, 0), environment.position);
    try std.testing.expect(!transition.terminal);
    environment = try LineWorld.init(state_count - 2);
    transition = try environment.step(1);
    try std.testing.expect(transition.terminal);
    try std.testing.expect(transition.reached_goal);
    try std.testing.expectEqual(@as(f32, 1), transition.reward);
}

test "DQN learns a shortest-path greedy policy" {
    const result = try runExperiment(std.testing.allocator, .{
        .backend = .cpu,
        .episodes = 50,
        .seed = default_seed,
    });
    try std.testing.expect(result.final.success_rate >= 0.95);
    try std.testing.expect(result.final.average_steps <= 3);
    for (result.policy) |action| try std.testing.expectEqual(@as(usize, 1), action);
    try std.testing.expect(result.optimization_steps > 100);
    try std.testing.expect(result.target_syncs > 0);
}
