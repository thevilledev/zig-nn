const std = @import("std");
const nn = @import("nn");

const node_count: usize = 120;
const feature_count: usize = 2;
const class_count: usize = 2;
const training_steps: usize = 160;

const Result = struct {
    mlp_accuracy: f32,
    graph_accuracy: f32,
    directed_edges: usize,
    train_nodes: usize,
    test_nodes: usize,
};

pub fn main(init: std.process.Init) !void {
    const result = try runExperiment(init.gpa);
    std.debug.print("Sparse graph node classification\n", .{});
    std.debug.print(
        "batch: {d} nodes, {d} directed CSR entries, {d} train / {d} test\n" ++
            "feature-only MLP accuracy: {d:.1}%\n" ++
            "one-layer graph convolution accuracy: {d:.1}%\n",
        .{
            node_count,
            result.directed_edges,
            result.train_nodes,
            result.test_nodes,
            result.mlp_accuracy * 100,
            result.graph_accuracy * 100,
        },
    );
    std.debug.print(
        "lesson: averaging homophilous neighbors denoises the same features seen by the MLP\n",
        .{},
    );
}

fn runExperiment(allocator: std.mem.Allocator) !Result {
    const features = try allocator.alloc(f32, node_count * feature_count);
    defer allocator.free(features);
    const labels = try allocator.alloc(usize, node_count);
    defer allocator.free(labels);
    const training_mask = try allocator.alloc(bool, node_count);
    defer allocator.free(training_mask);
    const testing_mask = try allocator.alloc(bool, node_count);
    defer allocator.free(testing_mask);
    generateFeatures(features, labels, training_mask, testing_mask);
    const edges = try allocator.alloc(nn.Graph.Edge, node_count * 3);
    defer allocator.free(edges);
    generateEdges(edges);
    var graph = try nn.Graph.CsrGraph.init(allocator, node_count, edges, true);
    defer graph.deinit();
    var isolated = try nn.Graph.CsrGraph.init(allocator, node_count, &.{}, false);
    defer isolated.deinit();
    var mlp_prng = std.Random.DefaultPrng.init(23);
    var graph_prng = std.Random.DefaultPrng.init(23);
    var mlp = try nn.Graph.GraphConvolution.init(
        allocator,
        feature_count,
        class_count,
        mlp_prng.random(),
    );
    defer mlp.deinit();
    var convolution = try nn.Graph.GraphConvolution.init(
        allocator,
        feature_count,
        class_count,
        graph_prng.random(),
    );
    defer convolution.deinit();
    for (0..training_steps) |_| {
        try trainStep(&mlp, isolated, features, labels, training_mask);
        try trainStep(&convolution, graph, features, labels, training_mask);
    }
    var train_nodes: usize = 0;
    var test_nodes: usize = 0;
    for (training_mask, testing_mask) |trained, tested| {
        if (trained) train_nodes += 1;
        if (tested) test_nodes += 1;
    }
    return .{
        .mlp_accuracy = try mlp.accuracy(isolated, features, labels, testing_mask),
        .graph_accuracy = try convolution.accuracy(graph, features, labels, testing_mask),
        .directed_edges = graph.neighbors.len,
        .train_nodes = train_nodes,
        .test_nodes = test_nodes,
    };
}

fn trainStep(
    model: *nn.Graph.GraphConvolution,
    graph: nn.Graph.CsrGraph,
    features: []const f32,
    labels: []const usize,
    training_mask: []const bool,
) !void {
    var objective = try model.lossAndGradient(graph, features, labels, training_mask);
    defer objective.deinit();
    try model.applyGradient(objective.gradients, 0.25);
}

fn generateFeatures(
    features: []f32,
    labels: []usize,
    training_mask: []bool,
    testing_mask: []bool,
) void {
    var prng = std.Random.DefaultPrng.init(101);
    const random = prng.random();
    for (0..node_count) |node| {
        const label = node / (node_count / class_count);
        labels[node] = label;
        const signal: f32 = if (label == 0) -1 else 1;
        features[node * feature_count] = signal +
            (random.float(f32) * 2 - 1) * 3.4;
        features[node * feature_count + 1] =
            (random.float(f32) * 2 - 1) * 3.4;
        testing_mask[node] = node % 3 == 0;
        training_mask[node] = !testing_mask[node];
    }
}

fn generateEdges(edges: []nn.Graph.Edge) void {
    const community_size = node_count / class_count;
    var edge_index: usize = 0;
    for (0..node_count) |node| {
        const community_start = node / community_size * community_size;
        const local = node - community_start;
        edges[edge_index] = .{
            .source = node,
            .target = community_start + (local + 1) % community_size,
        };
        edge_index += 1;
        edges[edge_index] = .{
            .source = node,
            .target = community_start + (local + 2) % community_size,
        };
        edge_index += 1;
        edges[edge_index] = .{
            .source = node,
            .target = community_start + (local + 3) % community_size,
        };
        edge_index += 1;
    }
}

test "graph convolution outperforms feature-only classification" {
    const result = try runExperiment(std.testing.allocator);
    try std.testing.expect(result.graph_accuracy > 0.9);
    try std.testing.expect(result.graph_accuracy > result.mlp_accuracy + 0.2);
    try std.testing.expect(result.directed_edges > node_count);
}
