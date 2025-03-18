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
            input_matrix.set(0, i, value);
        }

        // Get prediction
        var output_matrix = try self.network.predict(input_matrix);
        defer output_matrix.deinit();

        // Convert output matrix to array
        var output = try self.allocator.alloc(f64, output_matrix.cols);
        for (0..output_matrix.cols) |i| {
            output[i] = output_matrix.get(0, i);
        }

        return output;
    }

    /// Make predictions for multiple inputs
    /// Parameters:
    ///   - inputs: Array of input arrays
    /// Returns: Array of prediction arrays
    pub fn predictBatch(self: *InferenceService, inputs: [][]const f64) ![][]f64 {
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
                input_matrix.set(i, j, value);
            }
        }

        // Get predictions
        var output_matrix = try self.network.predict(input_matrix);
        defer output_matrix.deinit();

        // Convert output matrix to array of arrays
        var outputs = try self.allocator.alloc([]f64, output_matrix.rows);
        errdefer {
            for (outputs) |output| {
                self.allocator.free(output);
            }
            self.allocator.free(outputs);
        }

        for (0..output_matrix.rows) |i| {
            outputs[i] = try self.allocator.alloc(f64, output_matrix.cols);
            for (0..output_matrix.cols) |j| {
                outputs[i][j] = output_matrix.get(i, j);
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
test "inference service initialization and prediction" {
    const allocator = testing.allocator;

    // Create a simple network for testing
    var network = Network.init(allocator, 0.1, .MeanSquaredError);
    defer network.deinit();

    try network.addLayer(2, 4, Activation.relu, Activation.relu_derivative);
    try network.addLayer(4, 1, Activation.sigmoid, Activation.sigmoid_derivative);

    // Save network to file
    try network.saveToFile("test_model.bin");

    // Create inference service
    var service = try InferenceService.init(allocator, "test_model.bin");
    defer service.deinit();

    // Test single prediction
    const input = [_]f64{ 1.0, 0.5 };
    const prediction = try service.predict(&input);
    defer allocator.free(prediction);

    try testing.expectEqual(@as(usize, 1), prediction.len);
    try testing.expect(prediction[0] >= 0.0 and prediction[0] <= 1.0);

    // Test batch prediction
    const inputs = [_][]const f64{
        &[_]f64{ 1.0, 0.5 },
        &[_]f64{ 0.0, 1.0 },
    };
    const predictions = try service.predictBatch(&inputs);
    defer {
        for (predictions) |pred| {
            allocator.free(pred);
        }
        allocator.free(predictions);
    }

    try testing.expectEqual(@as(usize, 2), predictions.len);
    try testing.expectEqual(@as(usize, 1), predictions[0].len);
    try testing.expectEqual(@as(usize, 1), predictions[1].len);

    // Clean up test file
    try std.fs.cwd().deleteFile("test_model.bin");
}

test "inference service error handling" {
    const allocator = testing.allocator;

    // Test with non-existent model file
    try testing.expectError(error.FileNotFound, InferenceService.init(allocator, "nonexistent.bin"));

    // Test with empty batch
    var service = try InferenceService.init(allocator, "test_model.bin");
    defer service.deinit();

    const empty_inputs = [_][]const f64{};
    try testing.expectError(error.EmptyInput, service.predictBatch(&empty_inputs));

    // Test with inconsistent input sizes
    const inconsistent_inputs = [_][]const f64{
        &[_]f64{ 1.0, 0.5 },
        &[_]f64{0.0}, // Different size
    };
    try testing.expectError(error.InconsistentInputSizes, service.predictBatch(&inconsistent_inputs));
}
