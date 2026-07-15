const std = @import("std");
const Network = @import("network.zig").Network;
const Matrix = @import("matrix.zig").Matrix;
const json = std.json;
const testing = std.testing;
const Activation = @import("activation.zig").Activation;

/// Inference service for serving neural network predictions
/// Provides a high-level interface for making predictions using a trained model
pub const InferenceService = struct {
    network: Network,
    allocator: std.mem.Allocator,

    /// Initialize a new inference service from a saved model
    /// Parameters:
    ///   - allocator: Memory allocator for the service
    ///   - model_path: Path to the saved model file
    pub fn init(allocator: std.mem.Allocator, model_path: []const u8) !InferenceService {
        const network = try Network.loadFromFile(allocator, model_path);
        return InferenceService{
            .network = network,
            .allocator = allocator,
        };
    }

    /// Free all resources used by the inference service
    pub fn deinit(self: *InferenceService) void {
        self.network.deinit();
        self.* = undefined;
    }

    /// Make a prediction for a single input
    /// Parameters:
    ///   - input: Array of input values
    /// Returns: Array of predicted values
    pub fn predict(self: *InferenceService, input: []const f64) ![]f64 {
        // Create input matrix
        var input_matrix = try Matrix.init(self.allocator, 1, input.len);
        defer input_matrix.deinit();

        // Copy input values to matrix
        for (input, 0..) |value, i| {
            try input_matrix.set(0, i, value);
        }

        // Get prediction
        var output_matrix = try self.network.predict(input_matrix);
        defer output_matrix.deinit();

        // Convert output matrix to array
        var output = try self.allocator.alloc(f64, output_matrix.cols);
        errdefer self.allocator.free(output);
        for (0..output_matrix.cols) |i| {
            output[i] = try output_matrix.get(0, i);
        }

        return output;
    }

    /// Make predictions for multiple inputs
    /// Parameters:
    ///   - inputs: Array of input arrays
    /// Returns: Array of prediction arrays
    pub fn predictBatch(self: *InferenceService, inputs: []const []const f64) ![][]f64 {
        if (inputs.len == 0) {
            return error.EmptyInput;
        }

        // Create input matrix
        var input_matrix = try Matrix.init(self.allocator, inputs.len, inputs[0].len);
        defer input_matrix.deinit();

        // Copy input values to matrix
        for (inputs, 0..) |input, i| {
            if (input.len != inputs[0].len) {
                return error.InconsistentInputSizes;
            }
            for (input, 0..) |value, j| {
                try input_matrix.set(i, j, value);
            }
        }

        // Get predictions
        var output_matrix = try self.network.predict(input_matrix);
        defer output_matrix.deinit();

        // Convert output matrix to array of arrays
        var outputs = try self.allocator.alloc([]f64, output_matrix.rows);
        var initialized_outputs: usize = 0;
        errdefer {
            for (outputs[0..initialized_outputs]) |output| {
                self.allocator.free(output);
            }
            self.allocator.free(outputs);
        }

        for (0..output_matrix.rows) |i| {
            outputs[i] = try self.allocator.alloc(f64, output_matrix.cols);
            initialized_outputs += 1;
            for (0..output_matrix.cols) |j| {
                outputs[i][j] = try output_matrix.get(i, j);
            }
        }

        return outputs;
    }

    /// Get the input size required by the model
    pub fn getInputSize(self: InferenceService) !usize {
        return self.network.getInputSize();
    }

    /// Get the output size produced by the model
    pub fn getOutputSize(self: InferenceService) !usize {
        return self.network.getOutputSize();
    }
};

// Tests
fn temporaryModelPath(allocator: std.mem.Allocator, tmp: *const testing.TmpDir, filename: []const u8) ![]u8 {
    return std.fs.path.join(allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..], filename });
}

fn checkPredictBatchAllocationFailure(allocator: std.mem.Allocator) !void {
    var service = InferenceService{
        .network = Network.init(allocator, 0.1, .MeanSquaredError),
        .allocator = allocator,
    };
    defer service.deinit();
    try service.network.addLayer(2, 2, Activation.linear, Activation.linear_derivative);

    const first = [_]f64{ 1.0, 0.5 };
    const second = [_]f64{ 0.0, 1.0 };
    const inputs = [_][]const f64{ first[0..], second[0..] };
    const outputs = try service.predictBatch(&inputs);
    defer {
        for (outputs) |output| allocator.free(output);
        allocator.free(outputs);
    }
}

test "inference service initialization and prediction" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const model_path = try temporaryModelPath(allocator, &tmp, "model.bin");
    defer allocator.free(model_path);

    // Create a simple network for testing
    var network = Network.init(allocator, 0.1, .MeanSquaredError);
    defer network.deinit();

    try network.addLayer(2, 4, Activation.relu, Activation.relu_derivative);
    try network.addLayer(4, 1, Activation.sigmoid, Activation.sigmoid_derivative);

    // Save network to file
    try network.saveToFile(model_path);

    // Create inference service
    var service = try InferenceService.init(allocator, model_path);
    defer service.deinit();

    // Test single prediction
    const input = [_]f64{ 1.0, 0.5 };
    const prediction = try service.predict(input[0..]);
    defer allocator.free(prediction);

    try testing.expectEqual(@as(usize, 1), prediction.len);
    try testing.expect(prediction[0] >= 0.0 and prediction[0] <= 1.0);

    // Test batch prediction
    var input1 = [_]f64{ 1.0, 0.5 };
    var input2 = [_]f64{ 0.0, 1.0 };

    var batch_inputs = [_][]const f64{
        input1[0..],
        input2[0..],
    };

    const predictions = try service.predictBatch(batch_inputs[0..]);
    defer {
        for (predictions) |pred| {
            allocator.free(pred);
        }
        allocator.free(predictions);
    }

    try testing.expectEqual(@as(usize, 2), predictions.len);
    try testing.expectEqual(@as(usize, 1), predictions[0].len);
    try testing.expectEqual(@as(usize, 1), predictions[1].len);
}

test "inference service error handling" {
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const model_path = try temporaryModelPath(allocator, &tmp, "model.bin");
    defer allocator.free(model_path);
    const missing_model_path = try temporaryModelPath(allocator, &tmp, "missing.bin");
    defer allocator.free(missing_model_path);

    // Test with non-existent model file
    try testing.expectError(error.FileNotFound, InferenceService.init(allocator, missing_model_path));

    // Create a simple network for testing
    var network = Network.init(allocator, 0.1, .MeanSquaredError);
    defer network.deinit();

    try network.addLayer(2, 4, Activation.relu, Activation.relu_derivative);
    try network.addLayer(4, 1, Activation.sigmoid, Activation.sigmoid_derivative);

    // Save network to file
    try network.saveToFile(model_path);

    // Test with empty batch
    var service = try InferenceService.init(allocator, model_path);
    defer service.deinit();

    var empty_batch: [0][]const f64 = .{};
    try testing.expectError(error.EmptyInput, service.predictBatch(empty_batch[0..]));

    // Test with inconsistent input sizes
    var inconsistent1 = [_]f64{ 1.0, 0.5 };
    var inconsistent2 = [_]f64{0.0};

    var inconsistent_batch = [_][]const f64{
        inconsistent1[0..],
        inconsistent2[0..], // Different size
    };

    try testing.expectError(error.InconsistentInputSizes, service.predictBatch(inconsistent_batch[0..]));
}

test "batch prediction cleans up partial allocations" {
    try testing.checkAllAllocationFailures(testing.allocator, checkPredictBatchAllocationFailure, .{});
}
