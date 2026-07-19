const std = @import("std");
const dimensions = @import("dimensions.zig");

pub const Edge = struct {
    source: usize,
    target: usize,
};

/// Compressed-sparse-row graph storage. A disjoint union provides sparse graph
/// batches without padding node or edge tensors.
pub const CsrGraph = struct {
    allocator: std.mem.Allocator,
    node_count: usize,
    offsets: []usize,
    neighbors: []usize,

    pub fn init(
        allocator: std.mem.Allocator,
        node_count: usize,
        edges: []const Edge,
        undirected: bool,
    ) !CsrGraph {
        if (node_count == 0) return error.InvalidGraph;
        const offset_count = try dimensions.add(node_count, 1);
        const offsets = try allocator.alloc(usize, offset_count);
        errdefer allocator.free(offsets);
        @memset(offsets, 0);
        for (edges) |edge| {
            if (edge.source >= node_count or edge.target >= node_count) {
                return error.InvalidEdge;
            }
            offsets[edge.source + 1] = try dimensions.add(offsets[edge.source + 1], 1);
            if (undirected and edge.source != edge.target) {
                offsets[edge.target + 1] = try dimensions.add(offsets[edge.target + 1], 1);
            }
        }
        for (1..offsets.len) |index| {
            offsets[index] = try dimensions.add(offsets[index], offsets[index - 1]);
        }
        const neighbors = try allocator.alloc(usize, offsets[node_count]);
        errdefer allocator.free(neighbors);
        const cursor = try allocator.dupe(usize, offsets[0..node_count]);
        defer allocator.free(cursor);
        for (edges) |edge| {
            neighbors[cursor[edge.source]] = edge.target;
            cursor[edge.source] += 1;
            if (undirected and edge.source != edge.target) {
                neighbors[cursor[edge.target]] = edge.source;
                cursor[edge.target] += 1;
            }
        }
        for (0..node_count) |node| {
            std.mem.sort(usize, neighbors[offsets[node]..offsets[node + 1]], {}, std.sort.asc(usize));
        }
        return .{
            .allocator = allocator,
            .node_count = node_count,
            .offsets = offsets,
            .neighbors = neighbors,
        };
    }

    pub fn disjointUnion(
        allocator: std.mem.Allocator,
        graphs: []const CsrGraph,
    ) !CsrGraph {
        if (graphs.len == 0) return error.InvalidGraph;
        var total_nodes: usize = 0;
        var total_neighbors: usize = 0;
        for (graphs) |graph| {
            total_nodes = try dimensions.add(total_nodes, graph.node_count);
            total_neighbors = try dimensions.add(total_neighbors, graph.neighbors.len);
        }
        const offset_count = try dimensions.add(total_nodes, 1);
        const offsets = try allocator.alloc(usize, offset_count);
        errdefer allocator.free(offsets);
        const neighbors = try allocator.alloc(usize, total_neighbors);
        errdefer allocator.free(neighbors);
        offsets[0] = 0;
        var node_offset: usize = 0;
        var neighbor_offset: usize = 0;
        for (graphs) |graph| {
            for (0..graph.node_count) |node| {
                const degree = graph.offsets[node + 1] - graph.offsets[node];
                const output_node = try dimensions.add(node_offset, node);
                offsets[try dimensions.add(output_node, 1)] = try dimensions.add(neighbor_offset, degree);
                for (graph.neighbors[graph.offsets[node]..graph.offsets[node + 1]], 0..) |neighbor, index| {
                    neighbors[try dimensions.add(neighbor_offset, index)] = try dimensions.add(node_offset, neighbor);
                }
                neighbor_offset = try dimensions.add(neighbor_offset, degree);
            }
            node_offset = try dimensions.add(node_offset, graph.node_count);
        }
        return .{
            .allocator = allocator,
            .node_count = total_nodes,
            .offsets = offsets,
            .neighbors = neighbors,
        };
    }

    pub fn deinit(self: *CsrGraph) void {
        self.allocator.free(self.offsets);
        self.allocator.free(self.neighbors);
        self.* = undefined;
    }

    pub fn aggregateMean(
        self: CsrGraph,
        allocator: std.mem.Allocator,
        features: []const f32,
        feature_count: usize,
        include_self: bool,
    ) ![]f32 {
        if (feature_count == 0) return error.DimensionMismatch;
        const expected_features = try dimensions.elementCount(self.node_count, feature_count);
        if (features.len != expected_features) {
            return error.DimensionMismatch;
        }
        const output = try allocator.alloc(f32, features.len);
        errdefer allocator.free(output);
        @memset(output, 0);
        for (0..self.node_count) |node| {
            var contributors: usize = 0;
            if (include_self) {
                for (0..feature_count) |feature| {
                    output[node * feature_count + feature] +=
                        features[node * feature_count + feature];
                }
                contributors = try dimensions.add(contributors, 1);
            }
            for (self.neighbors[self.offsets[node]..self.offsets[node + 1]]) |neighbor| {
                for (0..feature_count) |feature| {
                    output[node * feature_count + feature] +=
                        features[neighbor * feature_count + feature];
                }
                contributors = try dimensions.add(contributors, 1);
            }
            if (contributors == 0) return error.IsolatedNode;
            const scale = 1.0 / @as(f32, @floatFromInt(contributors));
            for (output[node * feature_count ..][0..feature_count]) |*value| {
                value.* *= scale;
            }
        }
        return output;
    }
};

pub const ClassifierGradients = struct {
    allocator: std.mem.Allocator,
    weights: []f32,
    bias: []f32,

    pub fn deinit(self: *ClassifierGradients) void {
        self.allocator.free(self.weights);
        self.allocator.free(self.bias);
        self.* = undefined;
    }
};

pub const ClassifierObjective = struct {
    loss: f32,
    gradients: ClassifierGradients,

    pub fn deinit(self: *ClassifierObjective) void {
        self.gradients.deinit();
        self.* = undefined;
    }
};

/// One graph-convolution classification layer: mean neighborhood aggregation
/// followed by a learned linear softmax classifier.
pub const GraphConvolution = struct {
    allocator: std.mem.Allocator,
    input_features: usize,
    classes: usize,
    weights: []f32,
    bias: []f32,

    pub fn init(
        allocator: std.mem.Allocator,
        input_features: usize,
        classes: usize,
        random: std.Random,
    ) !GraphConvolution {
        if (input_features == 0 or classes < 2) return error.InvalidConfiguration;
        const weight_count = try dimensions.elementCount(input_features, classes);
        const weights = try allocator.alloc(f32, weight_count);
        errdefer allocator.free(weights);
        const bias = try allocator.alloc(f32, classes);
        const scale = @sqrt(2.0 / @as(f32, @floatFromInt(input_features)));
        for (weights) |*weight| weight.* = (random.float(f32) * 2 - 1) * scale;
        @memset(bias, 0);
        return .{
            .allocator = allocator,
            .input_features = input_features,
            .classes = classes,
            .weights = weights,
            .bias = bias,
        };
    }

    pub fn deinit(self: *GraphConvolution) void {
        self.allocator.free(self.weights);
        self.allocator.free(self.bias);
        self.* = undefined;
    }

    pub fn lossAndGradient(
        self: GraphConvolution,
        graph: CsrGraph,
        features: []const f32,
        labels: []const usize,
        training_mask: []const bool,
    ) !ClassifierObjective {
        try self.validateData(graph, features, labels, training_mask);
        const aggregated = try graph.aggregateMean(
            self.allocator,
            features,
            self.input_features,
            true,
        );
        defer self.allocator.free(aggregated);
        const weight_gradient = try self.allocator.alloc(f32, self.weights.len);
        errdefer self.allocator.free(weight_gradient);
        @memset(weight_gradient, 0);
        const bias_gradient = try self.allocator.alloc(f32, self.bias.len);
        errdefer self.allocator.free(bias_gradient);
        @memset(bias_gradient, 0);
        const logits = try self.allocator.alloc(f32, self.classes);
        defer self.allocator.free(logits);
        const probabilities = try self.allocator.alloc(f32, self.classes);
        defer self.allocator.free(probabilities);
        var loss: f32 = 0;
        var examples: usize = 0;
        for (labels, training_mask, 0..) |label, trained, node| {
            if (!trained) continue;
            self.project(aggregated[node * self.input_features ..][0..self.input_features], logits);
            softmax(logits, probabilities);
            loss -= @log(@max(probabilities[label], 1e-9));
            for (probabilities, 0..) |probability, class| {
                const gradient = probability - @as(f32, if (class == label) 1 else 0);
                bias_gradient[class] += gradient;
                for (0..self.input_features) |feature| {
                    weight_gradient[feature * self.classes + class] +=
                        aggregated[node * self.input_features + feature] * gradient;
                }
            }
            examples += 1;
        }
        if (examples == 0) return error.EmptyTrainingMask;
        const scale = 1.0 / @as(f32, @floatFromInt(examples));
        for (weight_gradient) |*gradient| gradient.* *= scale;
        for (bias_gradient) |*gradient| gradient.* *= scale;
        return .{
            .loss = loss * scale,
            .gradients = .{
                .allocator = self.allocator,
                .weights = weight_gradient,
                .bias = bias_gradient,
            },
        };
    }

    pub fn applyGradient(
        self: *GraphConvolution,
        gradients: ClassifierGradients,
        learning_rate: f32,
    ) !void {
        if (!std.math.isFinite(learning_rate) or learning_rate <= 0 or
            gradients.weights.len != self.weights.len or gradients.bias.len != self.bias.len)
        {
            return error.InvalidGradient;
        }
        for (self.weights, gradients.weights) |*weight, gradient| weight.* -= learning_rate * gradient;
        for (self.bias, gradients.bias) |*bias, gradient| bias.* -= learning_rate * gradient;
    }

    pub fn accuracy(
        self: GraphConvolution,
        graph: CsrGraph,
        features: []const f32,
        labels: []const usize,
        evaluation_mask: []const bool,
    ) !f32 {
        try self.validateData(graph, features, labels, evaluation_mask);
        const aggregated = try graph.aggregateMean(
            self.allocator,
            features,
            self.input_features,
            true,
        );
        defer self.allocator.free(aggregated);
        const logits = try self.allocator.alloc(f32, self.classes);
        defer self.allocator.free(logits);
        var correct: usize = 0;
        var examples: usize = 0;
        for (labels, evaluation_mask, 0..) |label, evaluated, node| {
            if (!evaluated) continue;
            self.project(aggregated[node * self.input_features ..][0..self.input_features], logits);
            var best: usize = 0;
            for (logits[1..], 1..) |value, class| if (value > logits[best]) {
                best = class;
            };
            if (best == label) correct += 1;
            examples += 1;
        }
        if (examples == 0) return error.EmptyEvaluationMask;
        return @as(f32, @floatFromInt(correct)) /
            @as(f32, @floatFromInt(examples));
    }

    fn validateData(
        self: GraphConvolution,
        graph: CsrGraph,
        features: []const f32,
        labels: []const usize,
        mask: []const bool,
    ) !void {
        const expected_features = try dimensions.elementCount(graph.node_count, self.input_features);
        if (features.len != expected_features or labels.len != graph.node_count or
            mask.len != graph.node_count)
        {
            return error.DimensionMismatch;
        }
        for (labels) |label| if (label >= self.classes) return error.InvalidLabel;
    }

    fn project(self: GraphConvolution, features: []const f32, logits: []f32) void {
        @memcpy(logits, self.bias);
        for (features, 0..) |value, feature| {
            for (logits, 0..) |*logit, class| {
                logit.* += value * self.weights[feature * self.classes + class];
            }
        }
    }
};

fn softmax(logits: []const f32, probabilities: []f32) void {
    var maximum = logits[0];
    for (logits[1..]) |value| maximum = @max(maximum, value);
    var total: f32 = 0;
    for (logits, probabilities) |logit, *probability| {
        probability.* = @exp(logit - maximum);
        total += probability.*;
    }
    for (probabilities) |*probability| probability.* /= total;
}

test "sparse graph batching offsets neighbor indices" {
    var first = try CsrGraph.init(std.testing.allocator, 2, &.{.{ .source = 0, .target = 1 }}, true);
    defer first.deinit();
    var second = try CsrGraph.init(std.testing.allocator, 3, &.{.{ .source = 1, .target = 2 }}, true);
    defer second.deinit();
    var batch = try CsrGraph.disjointUnion(std.testing.allocator, &.{ first, second });
    defer batch.deinit();
    try std.testing.expectEqual(@as(usize, 5), batch.node_count);
    try std.testing.expectEqualSlices(usize, &.{ 1, 0, 4, 3 }, batch.neighbors);
}

test "mean aggregation combines self and neighbor features" {
    var graph = try CsrGraph.init(std.testing.allocator, 3, &.{
        .{ .source = 0, .target = 1 },
        .{ .source = 1, .target = 2 },
    }, true);
    defer graph.deinit();
    const aggregated = try graph.aggregateMean(
        std.testing.allocator,
        &.{ 1, 0, 3, 2, 5, 4 },
        2,
        true,
    );
    defer std.testing.allocator.free(aggregated);
    try std.testing.expectEqualSlices(f32, &.{ 2, 1, 3, 2, 4, 3 }, aggregated);
}

test "graph APIs reject overflowing derived dimensions" {
    const allocator = std.testing.allocator;
    const maximum = std.math.maxInt(usize);
    try std.testing.expectError(
        error.DimensionOverflow,
        CsrGraph.init(allocator, maximum, &.{}, false),
    );

    const impossible = CsrGraph{
        .allocator = allocator,
        .node_count = maximum,
        .offsets = &.{},
        .neighbors = &.{},
    };
    try std.testing.expectError(
        error.DimensionOverflow,
        CsrGraph.disjointUnion(allocator, &.{impossible}),
    );
    try std.testing.expectError(
        error.DimensionOverflow,
        impossible.aggregateMean(allocator, &.{}, 2, true),
    );

    var prng = std.Random.DefaultPrng.init(1);
    try std.testing.expectError(
        error.DimensionOverflow,
        GraphConvolution.init(allocator, maximum, 2, prng.random()),
    );
}
