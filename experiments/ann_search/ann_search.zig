const std = @import("std");
const nn = @import("nn");

const document_count: usize = 256;
const feature_count: usize = 8;
const cluster_count: usize = 8;
const query_count: usize = 24;
const neighbors: usize = 5;

const Measurement = struct {
    probes: usize,
    recall: f32,
    average_candidates: f32,
    average_nanoseconds: f64,
};

const Result = struct {
    exact_average_nanoseconds: f64,
    narrow: Measurement,
    wide: Measurement,
};

pub fn main(init: std.process.Init) !void {
    const result = try runExperiment(init.gpa, init.io);
    std.debug.print("Approximate nearest-neighbor search with IVF\n", .{});
    std.debug.print(
        "dataset: {d} normalized documents, {d} dimensions, {d} coarse lists\n" ++
            "exact cosine: {d:.0} ns/query, {d} candidates\n",
        .{
            document_count,
            feature_count,
            cluster_count,
            result.exact_average_nanoseconds,
            document_count,
        },
    );
    printMeasurement(result.narrow);
    printMeasurement(result.wide);
    std.debug.print(
        "lesson: probing more lists raises candidate work to protect recall@{d}\n",
        .{neighbors},
    );
}

fn printMeasurement(measurement: Measurement) void {
    std.debug.print(
        "IVF probes={d}: recall@{d} {d:.1}%, {d:.1} candidates, {d:.0} ns/query\n",
        .{
            measurement.probes,
            neighbors,
            measurement.recall * 100,
            measurement.average_candidates,
            measurement.average_nanoseconds,
        },
    );
}

fn runExperiment(allocator: std.mem.Allocator, io: std.Io) !Result {
    const documents = try allocator.alloc(f32, document_count * feature_count);
    defer allocator.free(documents);
    const queries = try allocator.alloc(f32, query_count * feature_count);
    defer allocator.free(queries);
    generateDataset(documents, queries);
    var index = try nn.Retrieval.IvfIndex.init(
        allocator,
        documents,
        feature_count,
        cluster_count,
        8,
    );
    defer index.deinit();

    const exact_results = try allocator.alloc(nn.Retrieval.Neighbor, query_count * neighbors);
    defer allocator.free(exact_results);
    const exact_start = std.Io.Clock.awake.now(io);
    for (0..query_count) |query_index| {
        const exact = try nn.Retrieval.topKCosine(
            allocator,
            queries[query_index * feature_count ..][0..feature_count],
            documents,
            feature_count,
            neighbors,
        );
        defer allocator.free(exact);
        @memcpy(
            exact_results[query_index * neighbors ..][0..neighbors],
            exact,
        );
    }
    const exact_elapsed = exact_start.durationTo(std.Io.Clock.awake.now(io));
    return .{
        .exact_average_nanoseconds = @as(f64, @floatFromInt(exact_elapsed.nanoseconds)) /
            @as(f64, @floatFromInt(query_count)),
        .narrow = try measureApproximate(io, &index, queries, exact_results, 1),
        .wide = try measureApproximate(io, &index, queries, exact_results, 3),
    };
}

fn measureApproximate(
    io: std.Io,
    index: *const nn.Retrieval.IvfIndex,
    queries: []const f32,
    exact_results: []const nn.Retrieval.Neighbor,
    probes: usize,
) !Measurement {
    var recall: f32 = 0;
    var candidates: usize = 0;
    const start = std.Io.Clock.awake.now(io);
    for (0..query_count) |query_index| {
        var approximate = try index.search(
            queries[query_index * feature_count ..][0..feature_count],
            neighbors,
            probes,
        );
        defer approximate.deinit();
        recall += try nn.Retrieval.recallAtK(
            exact_results[query_index * neighbors ..][0..neighbors],
            approximate.neighbors,
        );
        candidates += approximate.candidates_scored;
    }
    const elapsed = start.durationTo(std.Io.Clock.awake.now(io));
    return .{
        .probes = probes,
        .recall = recall / @as(f32, @floatFromInt(query_count)),
        .average_candidates = @as(f32, @floatFromInt(candidates)) /
            @as(f32, @floatFromInt(query_count)),
        .average_nanoseconds = @as(f64, @floatFromInt(elapsed.nanoseconds)) /
            @as(f64, @floatFromInt(query_count)),
    };
}

fn generateDataset(documents: []f32, queries: []f32) void {
    var prng = std.Random.DefaultPrng.init(91);
    const random = prng.random();
    const documents_per_cluster = document_count / cluster_count;
    for (0..document_count) |document_index| {
        const cluster = document_index / documents_per_cluster;
        for (0..feature_count) |feature| {
            const center: f32 = if (feature == cluster) 1 else 0;
            documents[document_index * feature_count + feature] =
                center + (random.float(f32) * 2 - 1) * 0.08;
        }
    }
    for (0..query_count) |query_index| {
        const cluster = query_index % cluster_count;
        const document_index = cluster * documents_per_cluster +
            (query_index * 7 % documents_per_cluster);
        for (0..feature_count) |feature| {
            queries[query_index * feature_count + feature] =
                documents[document_index * feature_count + feature] +
                (random.float(f32) * 2 - 1) * 0.02;
        }
    }
}

test "IVF exposes recall and candidate reduction" {
    const result = try runExperiment(std.testing.allocator, std.Options.debug_io);
    try std.testing.expect(result.narrow.recall > 0.95);
    try std.testing.expect(result.narrow.average_candidates < document_count / 2);
    try std.testing.expect(result.wide.recall >= result.narrow.recall);
    try std.testing.expect(
        result.wide.average_candidates > result.narrow.average_candidates,
    );
    try std.testing.expect(result.exact_average_nanoseconds > 0);
}
