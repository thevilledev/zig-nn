const std = @import("std");

/// A transition batch sampled with replacement from a replay buffer.
pub const ReplayBatch = struct {
    allocator: std.mem.Allocator,
    batch_size: usize,
    state_features: usize,
    states: []f32,
    actions: []usize,
    rewards: []f32,
    next_states: []f32,
    terminals: []bool,

    pub fn deinit(self: *ReplayBatch) void {
        self.allocator.free(self.states);
        self.allocator.free(self.actions);
        self.allocator.free(self.rewards);
        self.allocator.free(self.next_states);
        self.allocator.free(self.terminals);
        self.* = undefined;
    }

    pub fn state(self: ReplayBatch, index: usize) []const f32 {
        const offset = index * self.state_features;
        return self.states[offset..][0..self.state_features];
    }

    pub fn nextState(self: ReplayBatch, index: usize) []const f32 {
        const offset = index * self.state_features;
        return self.next_states[offset..][0..self.state_features];
    }
};

/// Fixed-capacity ring buffer for off-policy reinforcement learning.
/// Sampling uses replacement, matching the independent minibatch draws used
/// by compact DQN implementations.
pub const ReplayBuffer = struct {
    allocator: std.mem.Allocator,
    state_features: usize,
    capacity: usize,
    len: usize = 0,
    next_index: usize = 0,
    states: []f32,
    actions: []usize,
    rewards: []f32,
    next_states: []f32,
    terminals: []bool,

    pub fn init(
        allocator: std.mem.Allocator,
        capacity: usize,
        state_features: usize,
    ) !ReplayBuffer {
        if (capacity == 0 or state_features == 0) return error.InvalidDimension;
        const state_values = try std.math.mul(usize, capacity, state_features);
        const states = try allocator.alloc(f32, state_values);
        errdefer allocator.free(states);
        const actions = try allocator.alloc(usize, capacity);
        errdefer allocator.free(actions);
        const rewards = try allocator.alloc(f32, capacity);
        errdefer allocator.free(rewards);
        const next_states = try allocator.alloc(f32, state_values);
        errdefer allocator.free(next_states);
        const terminals = try allocator.alloc(bool, capacity);
        return .{
            .allocator = allocator,
            .state_features = state_features,
            .capacity = capacity,
            .states = states,
            .actions = actions,
            .rewards = rewards,
            .next_states = next_states,
            .terminals = terminals,
        };
    }

    pub fn deinit(self: *ReplayBuffer) void {
        self.allocator.free(self.states);
        self.allocator.free(self.actions);
        self.allocator.free(self.rewards);
        self.allocator.free(self.next_states);
        self.allocator.free(self.terminals);
        self.* = undefined;
    }

    pub fn add(
        self: *ReplayBuffer,
        state: []const f32,
        action: usize,
        reward: f32,
        next_state: []const f32,
        terminal: bool,
    ) !void {
        if (state.len != self.state_features or next_state.len != self.state_features) {
            return error.DimensionMismatch;
        }
        if (!std.math.isFinite(reward)) return error.InvalidReward;
        for (state, next_state) |state_value, next_value| {
            if (!std.math.isFinite(state_value) or !std.math.isFinite(next_value)) {
                return error.InvalidState;
            }
        }
        const offset = self.next_index * self.state_features;
        @memcpy(self.states[offset..][0..self.state_features], state);
        @memcpy(self.next_states[offset..][0..self.state_features], next_state);
        self.actions[self.next_index] = action;
        self.rewards[self.next_index] = reward;
        self.terminals[self.next_index] = terminal;
        self.next_index = (self.next_index + 1) % self.capacity;
        self.len = @min(self.len + 1, self.capacity);
    }

    pub fn sample(
        self: ReplayBuffer,
        random: std.Random,
        batch_size: usize,
    ) !ReplayBatch {
        if (batch_size == 0) return error.InvalidBatchSize;
        if (self.len == 0) return error.EmptyReplayBuffer;
        const state_values = try std.math.mul(usize, batch_size, self.state_features);
        const states = try self.allocator.alloc(f32, state_values);
        errdefer self.allocator.free(states);
        const actions = try self.allocator.alloc(usize, batch_size);
        errdefer self.allocator.free(actions);
        const rewards = try self.allocator.alloc(f32, batch_size);
        errdefer self.allocator.free(rewards);
        const next_states = try self.allocator.alloc(f32, state_values);
        errdefer self.allocator.free(next_states);
        const terminals = try self.allocator.alloc(bool, batch_size);

        for (0..batch_size) |batch_index| {
            const replay_index = random.uintLessThan(usize, self.len);
            const replay_offset = replay_index * self.state_features;
            const batch_offset = batch_index * self.state_features;
            @memcpy(
                states[batch_offset..][0..self.state_features],
                self.states[replay_offset..][0..self.state_features],
            );
            @memcpy(
                next_states[batch_offset..][0..self.state_features],
                self.next_states[replay_offset..][0..self.state_features],
            );
            actions[batch_index] = self.actions[replay_index];
            rewards[batch_index] = self.rewards[replay_index];
            terminals[batch_index] = self.terminals[replay_index];
        }
        return .{
            .allocator = self.allocator,
            .batch_size = batch_size,
            .state_features = self.state_features,
            .states = states,
            .actions = actions,
            .rewards = rewards,
            .next_states = next_states,
            .terminals = terminals,
        };
    }
};

/// Linear exploration schedule for epsilon-greedy policies.
pub const EpsilonSchedule = struct {
    initial: f32,
    final: f32,
    decay_steps: usize,

    pub fn init(initial: f32, final: f32, decay_steps: usize) !EpsilonSchedule {
        if (!std.math.isFinite(initial) or !std.math.isFinite(final) or
            initial < 0 or initial > 1 or final < 0 or final > initial or
            decay_steps == 0)
        {
            return error.InvalidSchedule;
        }
        return .{ .initial = initial, .final = final, .decay_steps = decay_steps };
    }

    pub fn value(self: EpsilonSchedule, step: usize) f32 {
        const progress = @min(
            @as(f32, @floatFromInt(step)) / @as(f32, @floatFromInt(self.decay_steps)),
            1,
        );
        return self.initial + (self.final - self.initial) * progress;
    }
};

/// One-step Bellman target used by Q-learning and DQN.
pub fn bellmanTarget(
    reward: f32,
    next_maximum_q: f32,
    terminal: bool,
    discount: f32,
) !f32 {
    if (!std.math.isFinite(reward) or !std.math.isFinite(next_maximum_q)) {
        return error.InvalidValue;
    }
    if (!std.math.isFinite(discount) or discount < 0 or discount > 1) {
        return error.InvalidDiscount;
    }
    return reward + if (terminal) 0 else discount * next_maximum_q;
}

test "replay buffer overwrites its oldest transitions" {
    const testing = std.testing;
    var replay = try ReplayBuffer.init(testing.allocator, 3, 2);
    defer replay.deinit();
    try replay.add(&.{ 0, 0 }, 0, 0, &.{ 0, 1 }, false);
    try replay.add(&.{ 1, 0 }, 1, 1, &.{ 1, 1 }, false);
    try replay.add(&.{ 2, 0 }, 0, 2, &.{ 2, 1 }, false);
    try replay.add(&.{ 3, 0 }, 1, 3, &.{ 3, 1 }, true);
    try testing.expectEqual(@as(usize, 3), replay.len);
    try testing.expectEqual(@as(usize, 1), replay.next_index);

    var prng = std.Random.DefaultPrng.init(7);
    var batch = try replay.sample(prng.random(), 32);
    defer batch.deinit();
    for (0..batch.batch_size) |index| {
        try testing.expect(batch.state(index)[0] >= 1);
        try testing.expectEqual(@as(f32, 1), batch.nextState(index)[1]);
    }
}

test "replay sampling is deterministic for a seeded generator" {
    const testing = std.testing;
    var replay = try ReplayBuffer.init(testing.allocator, 4, 1);
    defer replay.deinit();
    for (0..4) |index| {
        const value: f32 = @floatFromInt(index);
        try replay.add(&.{value}, index % 2, value, &.{value}, false);
    }
    var first_prng = std.Random.DefaultPrng.init(11);
    var second_prng = std.Random.DefaultPrng.init(11);
    var first = try replay.sample(first_prng.random(), 8);
    defer first.deinit();
    var second = try replay.sample(second_prng.random(), 8);
    defer second.deinit();
    try testing.expectEqualSlices(f32, first.states, second.states);
    try testing.expectEqualSlices(usize, first.actions, second.actions);
}

test "epsilon schedule decays and Bellman targets stop at terminals" {
    const testing = std.testing;
    const schedule = try EpsilonSchedule.init(1, 0.1, 100);
    try testing.expectApproxEqAbs(@as(f32, 1), schedule.value(0), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.55), schedule.value(50), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.1), schedule.value(200), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 2.8), try bellmanTarget(1, 2, false, 0.9), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1), try bellmanTarget(1, 2, true, 0.9), 1e-6);
    try testing.expectError(error.InvalidDiscount, bellmanTarget(0, 0, false, 1.1));
}
