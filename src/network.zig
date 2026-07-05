const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const testing = std.testing;
const Layer = @import("layer.zig").Layer;
const GatedLayer = @import("layer.zig").GatedLayer;
const matrix_mod = @import("matrix.zig");
const Matrix = matrix_mod.Matrix;
const backend_mod = @import("backend.zig");
const BackendMatrix = backend_mod.Matrix;
const Activation = @import("activation.zig").Activation;

fn unixMilliTimestamp() i64 {
    return std.Io.Clock.real.now(std.Options.debug_io).toMilliseconds();
}

/// Neural Network Implementation
/// Provides a flexible neural network architecture supporting:
/// - Multiple layer types (standard and gated)
/// - Various activation functions
/// - Different loss functions (MSE and Cross Entropy)
/// - Mini-batch training
/// - Configurable learning rates
///
/// Network architecture: input → [layer_1 → layer_2 → ... → layer_n] → output
/// Each layer can be either a standard fully connected layer or a gated layer (GLU/SwiGLU)
/// Layer type enumeration for polymorphic layer handling
/// Supports two types of layers:
/// - Standard: Traditional fully connected layer
/// - Gated: Advanced layer with gating mechanism (GLU/SwiGLU)
pub const LayerType = enum {
    Standard,
    Gated,
};

/// Polymorphic layer container using tagged union
/// Enables unified interface for different layer types while
/// maintaining type safety and avoiding virtual dispatch
pub const LayerVariant = union(LayerType) {
    Standard: *Layer,
    Gated: *GatedLayer,

    /// Returns the number of input features the layer accepts
    /// Essential for validating layer connections in the network
    pub fn getInputSize(self: LayerVariant) usize {
        return switch (self) {
            .Standard => |layer| layer.getInputSize(),
            .Gated => |layer| layer.getInputSize(),
        };
    }

    /// Returns the number of output features the layer produces
    /// Used for automatic layer size validation during network construction
    pub fn getOutputSize(self: LayerVariant) usize {
        return switch (self) {
            .Standard => |layer| layer.getOutputSize(),
            .Gated => |layer| layer.getOutputSize(),
        };
    }

    /// Performs forward propagation through the layer
    /// Dispatches to appropriate layer implementation
    /// Returns: Output matrix after layer transformation
    pub fn forward(self: LayerVariant, input: Matrix) !Matrix {
        return switch (self) {
            .Standard => |layer| layer.forward(input),
            .Gated => |layer| layer.forward(input),
        };
    }

    /// Performs backend-aware inference through the layer.
    pub fn forwardBackend(self: LayerVariant, input: *const BackendMatrix) !*BackendMatrix {
        return switch (self) {
            .Standard => |layer| layer.forwardBackend(input),
            .Gated => |layer| layer.forwardBackend(input),
        };
    }

    /// Performs backward propagation through the layer
    /// Updates layer parameters using gradient descent
    /// Returns: Gradient for backpropagation to previous layer
    pub fn backward(self: LayerVariant, output_gradient: Matrix, learning_rate: f64) !Matrix {
        return switch (self) {
            .Standard => |layer| layer.backward(output_gradient, learning_rate),
            .Gated => |layer| layer.backward(output_gradient, learning_rate),
        };
    }

    /// Frees layer resources including weights, biases, and cached values
    /// Must be called to prevent memory leaks
    pub fn deinit(self: LayerVariant, allocator: Allocator) void {
        switch (self) {
            .Standard => |layer| {
                layer.deinit();
                allocator.destroy(layer);
            },
            .Gated => |layer| {
                layer.deinit();
                allocator.destroy(layer);
            },
        }
    }
};

/// Supported loss functions for network training
/// Each loss function has specific properties making it suitable for different tasks:
/// - MeanSquaredError: General purpose, good for regression
/// - CrossEntropy: Better for classification, especially with sigmoid/softmax outputs
/// - BinaryCrossEntropy: Added for explicit binary case
pub const LossFunction = enum {
    MeanSquaredError,
    CrossEntropy,
    BinaryCrossEntropy,
};

/// Neural Network implementation
/// Manages a sequence of layers and provides training functionality
/// Features:
/// - Dynamic layer addition
/// - Automatic dimension validation
/// - Forward/backward propagation
/// - Loss calculation and optimization
/// - Model persistence (save/load)
pub const Network = struct {
    layers: ArrayList(LayerVariant),
    allocator: Allocator,
    learning_rate: f64,
    loss_function: LossFunction,

    /// Initializes a new neural network
    /// Parameters:
    ///   - allocator: Memory allocator for network resources
    ///   - learning_rate: Step size for gradient descent (η)
    ///   - loss_function: Type of loss function to use
    pub fn init(allocator: Allocator, learning_rate: f64, loss_function: LossFunction) Network {
        return Network{
            .layers = .empty,
            .allocator = allocator,
            .learning_rate = learning_rate,
            .loss_function = loss_function,
        };
    }

    /// Frees all network resources
    /// Includes all layers and their associated memory
    pub fn deinit(self: *Network) void {
        for (self.layers.items) |layer| {
            layer.deinit(self.allocator);
        }
        self.layers.deinit(self.allocator);
    }

    /// Adds a standard fully connected layer to the network
    /// Parameters:
    ///   - input_size: Number of input features
    ///   - output_size: Number of neurons in the layer
    ///   - activation_fn: Activation function σ(x)
    ///   - activation_derivative_fn: Derivative σ'(x) for backpropagation
    pub fn addLayer(self: *Network, input_size: usize, output_size: usize, activation_fn: *const fn (f64) f64, activation_derivative_fn: *const fn (f64) f64) !void {
        const layer_ptr = try self.allocator.create(Layer);
        layer_ptr.* = try Layer.init(self.allocator, input_size, output_size, activation_fn, activation_derivative_fn);
        try self.layers.append(self.allocator, LayerVariant{ .Standard = layer_ptr });
    }

    /// Adds a gated layer (GLU/SwiGLU) to the network
    /// Parameters:
    ///   - input_size: Number of input features
    ///   - output_size: Number of neurons in the layer
    ///   - use_swiglu: Whether to use SwiGLU (true) or GLU (false)
    pub fn addGatedLayer(self: *Network, input_size: usize, output_size: usize, use_swiglu: bool) !void {
        const layer_ptr = try self.allocator.create(GatedLayer);
        layer_ptr.* = try GatedLayer.init(self.allocator, input_size, output_size, use_swiglu);
        try self.layers.append(self.allocator, LayerVariant{ .Gated = layer_ptr });
    }

    /// Performs forward propagation through the entire network
    /// Mathematical process: x → [layer_1 → layer_2 → ... → layer_n] → y
    /// Where each layer performs: y = σ(W·x + b)
    ///
    /// Parameters:
    ///   - input: Input matrix (batch_size × input_features)
    /// Returns: Output matrix (batch_size × output_features)
    ///
    /// Time complexity: O(batch_size * sum(layer_i_inputs * layer_i_outputs))
    /// Memory complexity: O(batch_size * max(layer_outputs))
    pub fn forward(self: Network, input: Matrix) !Matrix {
        if (self.layers.items.len == 0) {
            return error.EmptyNetwork;
        }

        var current = input;

        for (self.layers.items) |layer| {
            if (current.cols != layer.getInputSize()) {
                return error.InvalidInputDimensions;
            }

            const next = try layer.forward(current);

            if (current.data.ptr != input.data.ptr) {
                current.deinit();
            }
            current = next;
        }

        return current;
    }

    /// Performs inference through the backend-aware matrix path.
    ///
    /// The input matrix determines the backend used for all layer operations.
    /// Unlike `forward`, this path does not populate backpropagation caches and
    /// is intended for prediction/inference only.
    pub fn forwardBackend(self: Network, input: *const BackendMatrix) !*BackendMatrix {
        if (self.layers.items.len == 0) {
            return error.EmptyNetwork;
        }

        var current: *const BackendMatrix = input;
        var owned_current: ?*BackendMatrix = null;

        for (self.layers.items) |layer| {
            if (current.cols != layer.getInputSize()) {
                if (owned_current) |owned| {
                    owned.deinit();
                }
                return error.InvalidInputDimensions;
            }

            const next = layer.forwardBackend(current) catch |err| {
                if (owned_current) |owned| {
                    owned.deinit();
                }
                return err;
            };

            if (owned_current) |owned| {
                owned.deinit();
            }
            current = next;
            owned_current = next;
        }

        return owned_current.?;
    }

    /// Calculates loss between predicted and target values
    /// Supports three loss functions:
    /// 1. Mean Squared Error (MSE):
    ///    L = (1/n) * Σ(y_pred - y_true)²
    /// 2. Cross Entropy (for multi-class):
    ///    L = -(1/n) * Σ(y_true * log(y_pred))
    /// 3. Binary Cross Entropy:
    ///    L = -(1/n) * Σ(y_true * log(y_pred) + (1-y_true) * log(1-y_pred))
    ///
    /// Parameters:
    ///   - predicted: Network output (batch_size × output_features)
    ///   - target: Desired output (batch_size × output_features)
    /// Returns: Scalar loss value
    pub fn calculateLoss(self: Network, predicted: Matrix, target: Matrix) !f64 {
        if (predicted.rows != target.rows or predicted.cols != target.cols) {
            return error.DimensionMismatch;
        }

        var total_loss: f64 = 0.0;
        const n = @as(f64, @floatFromInt(predicted.rows));

        switch (self.loss_function) {
            .MeanSquaredError => {
                // MSE = (1/n) * Σ(y_pred - y_true)²
                for (0..predicted.rows) |i| {
                    for (0..predicted.cols) |j| {
                        const diff = (try predicted.get(i, j)) - (try target.get(i, j));
                        total_loss += diff * diff;
                    }
                }
                total_loss /= n;
            },
            .CrossEntropy => {
                // Multi-class cross entropy = -(1/n) * Σ(y_true * log(y_pred))
                for (0..predicted.rows) |i| {
                    for (0..predicted.cols) |j| {
                        const y_true = try target.get(i, j);
                        var y_pred = try predicted.get(i, j);

                        // Skip if true label is 0 (contributes nothing to sum)
                        if (y_true > 0.0) {
                            // Clip predictions for numerical stability
                            if (y_pred < 1e-15) y_pred = 1e-15;
                            total_loss -= y_true * @log(y_pred);
                        }
                    }
                }
                total_loss /= n;
            },
            .BinaryCrossEntropy => {
                // Binary cross entropy = -(1/n) * Σ(y_true * log(y_pred) + (1-y_true) * log(1-y_pred))
                for (0..predicted.rows) |i| {
                    for (0..predicted.cols) |j| {
                        const y_true = try target.get(i, j);
                        var y_pred = try predicted.get(i, j);

                        // Clip predictions to avoid log(0)
                        if (y_pred < 1e-15) y_pred = 1e-15;
                        if (y_pred > 1.0 - 1e-15) y_pred = 1.0 - 1e-15;

                        total_loss -= y_true * @log(y_pred) +
                            (1.0 - y_true) * @log(1.0 - y_pred);
                    }
                }
                total_loss /= n;
            },
        }

        return total_loss;
    }

    /// Calculates the gradient of the loss with respect to network outputs
    /// Mathematical formulas:
    /// 1. MSE gradient: ∂L/∂y_pred = 2(y_pred - y_true)/n
    /// 2. Cross Entropy gradient: ∂L/∂y_pred = -y_true/y_pred + (1-y_true)/(1-y_pred)
    ///
    /// Parameters:
    ///   - predicted: Network output (batch_size × output_features)
    ///   - target: Desired output (batch_size × output_features)
    /// Returns: Gradient matrix (batch_size × output_features)
    pub fn calculateLossGradient(self: Network, predicted: Matrix, target: Matrix) !Matrix {
        if (predicted.rows != target.rows or predicted.cols != target.cols) {
            return error.DimensionMismatch;
        }

        var gradient = try Matrix.init(self.allocator, predicted.rows, predicted.cols);
        const n = @as(f64, @floatFromInt(predicted.rows));

        switch (self.loss_function) {
            .MeanSquaredError => {
                // dMSE/dy_pred = 2 * (y_pred - y_true) / n
                for (0..predicted.rows) |i| {
                    for (0..predicted.cols) |j| {
                        const diff = (try predicted.get(i, j)) - (try target.get(i, j));
                        try gradient.set(i, j, 2.0 * diff / n);
                    }
                }
            },
            .CrossEntropy => {
                // dCE/dy_pred = -y_true / y_pred. Layer.backward handles the
                // row-wise softmax Jacobian, yielding y_pred - y_true for the
                // common softmax + cross entropy pairing.
                for (0..predicted.rows) |i| {
                    for (0..predicted.cols) |j| {
                        const y_true = try target.get(i, j);
                        if (y_true == 0.0) {
                            try gradient.set(i, j, 0.0);
                            continue;
                        }

                        var y_pred = try predicted.get(i, j);
                        if (y_pred < 1e-15) y_pred = 1e-15;
                        try gradient.set(i, j, -y_true / y_pred / n);
                    }
                }
            },
            .BinaryCrossEntropy => {
                // dBCE/dy_pred = -y_true/y_pred + (1-y_true)/(1-y_pred)
                for (0..predicted.rows) |i| {
                    for (0..predicted.cols) |j| {
                        const y_true = try target.get(i, j);
                        var y_pred = try predicted.get(i, j);

                        // Clip predictions to avoid division by zero
                        if (y_pred < 1e-15) y_pred = 1e-15;
                        if (y_pred > 1.0 - 1e-15) y_pred = 1.0 - 1e-15;

                        try gradient.set(i, j, (-y_true / y_pred + (1.0 - y_true) / (1.0 - y_pred)) / n);
                    }
                }
            },
        }

        return gradient;
    }

    /// Performs backward propagation through the network
    /// Updates all layer parameters using gradient descent
    /// Process:
    /// 1. Calculate loss gradient
    /// 2. Backpropagate through layers in reverse order
    /// 3. Update weights and biases using gradients
    ///
    /// Parameters:
    ///   - predicted: Network output from forward pass
    ///   - target: Desired output for training
    /// Time complexity: O(batch_size * sum(layer_i_inputs * layer_i_outputs))
    pub fn backward(self: *Network, predicted: Matrix, target: Matrix) !void {
        if (self.layers.items.len == 0) {
            return error.EmptyNetwork;
        }

        // Calculate initial gradient from loss function
        var gradient = try self.calculateLossGradient(predicted, target);
        errdefer gradient.deinit();

        // Backpropagate through layers in reverse order
        var i = self.layers.items.len;
        while (i > 0) {
            i -= 1;
            const new_gradient = try self.layers.items[i].backward(gradient, self.learning_rate);
            gradient.deinit();
            if (i > 0) {
                gradient = new_gradient;
            } else {
                new_gradient.deinit();
            }
        }
    }

    /// Train the network on a single batch of data
    pub fn trainBatch(self: *Network, inputs: Matrix, targets: Matrix) !f64 {
        // Forward pass
        var predictions = try self.forward(inputs);
        defer predictions.deinit();

        // Calculate loss
        const loss = try self.calculateLoss(predictions, targets);

        // Backward pass
        try self.backward(predictions, targets);

        return loss;
    }

    /// Train the network for a specified number of epochs
    pub fn train(self: *Network, inputs: Matrix, targets: Matrix, epochs: usize, batch_size: usize) ![]f64 {
        if (inputs.rows != targets.rows) {
            return error.DimensionMismatch;
        }

        const num_samples = inputs.rows;
        const num_batches = (num_samples + batch_size - 1) / batch_size; // Ceiling division

        // Allocate array to store loss history
        var loss_history = try self.allocator.alloc(f64, epochs);
        errdefer self.allocator.free(loss_history);

        var prng = std.Random.DefaultPrng.init(matrix_mod.randomSeed());
        const rand = prng.random();

        // Training loop
        for (0..epochs) |epoch| {
            var epoch_loss: f64 = 0.0;

            // Create a random permutation for shuffling
            var indices = try self.allocator.alloc(usize, num_samples);
            defer self.allocator.free(indices);
            for (0..num_samples) |i| {
                indices[i] = i;
            }
            rand.shuffle(usize, indices);

            // Process each batch
            for (0..num_batches) |batch| {
                const start_idx = batch * batch_size;
                const end_idx = @min(start_idx + batch_size, num_samples);
                const current_batch_size = end_idx - start_idx;

                // Create batch inputs and targets
                var batch_inputs = try Matrix.init(self.allocator, current_batch_size, inputs.cols);
                defer batch_inputs.deinit();
                var batch_targets = try Matrix.init(self.allocator, current_batch_size, targets.cols);
                defer batch_targets.deinit();

                // Fill batch with shuffled data
                for (0..current_batch_size) |i| {
                    const sample_idx = indices[start_idx + i];
                    for (0..inputs.cols) |j| {
                        try batch_inputs.set(i, j, try inputs.get(sample_idx, j));
                    }
                    for (0..targets.cols) |j| {
                        try batch_targets.set(i, j, try targets.get(sample_idx, j));
                    }
                }

                // Train on batch
                const batch_loss = try self.trainBatch(batch_inputs, batch_targets);
                epoch_loss += batch_loss * @as(f64, @floatFromInt(current_batch_size));
            }

            // Calculate average loss for the epoch
            epoch_loss /= @as(f64, @floatFromInt(num_samples));
            loss_history[epoch] = epoch_loss;
        }

        return loss_history;
    }

    /// Predict outputs for given inputs
    pub fn predict(self: Network, inputs: Matrix) !Matrix {
        return self.forward(inputs);
    }

    /// Predict outputs for backend-aware inputs.
    pub fn predictBackend(self: Network, inputs: *const BackendMatrix) !*BackendMatrix {
        return self.forwardBackend(inputs);
    }

    /// Get the input size required by the network
    pub fn getInputSize(self: Network) !usize {
        if (self.layers.items.len == 0) {
            return error.EmptyNetwork;
        }
        return self.layers.items[0].getInputSize();
    }

    /// Get the output size produced by the network
    pub fn getOutputSize(self: Network) !usize {
        if (self.layers.items.len == 0) {
            return error.EmptyNetwork;
        }
        return self.layers.items[self.layers.items.len - 1].getOutputSize();
    }

    /// Saves the network to a binary file
    /// Format:
    /// - Magic number (4 bytes): "ZNN\0"
    /// - Version (4 bytes): 2
    /// - Created at (8 bytes): Unix timestamp
    /// - Learning rate (8 bytes)
    /// - Loss function (1 byte)
    /// - Input size (4 bytes)
    /// - Output size (4 bytes)
    /// - Layer count (4 bytes)
    /// - For each layer:
    ///   * Layer type (1 byte): 0=Standard, 1=Gated
    ///   * Input size (4 bytes)
    ///   * Output size (4 bytes)
    ///   * Activation type (1 byte)
    ///   * Weight matrix (input_size * output_size * 8 bytes)
    ///   * Bias vector (output_size * 8 bytes)
    ///   * For gated layers: additional weight matrix and bias vector
    pub fn saveToFile(self: Network, filepath: []const u8) !void {
        const io = std.Options.debug_io;
        const file = try std.Io.Dir.cwd().createFile(io, filepath, .{});
        defer file.close(io);

        var buffer: [4096]u8 = undefined;
        var file_writer = file.writerStreaming(io, &buffer);
        const writer = &file_writer.interface;

        // Write magic number
        try writer.writeAll("ZNN\x00");

        // Write version
        try writer.writeInt(u32, 2, .little);

        // Write creation timestamp
        try writer.writeInt(i64, unixMilliTimestamp(), .little);

        // Write training metadata
        try writer.writeInt(i64, @as(i64, @bitCast(self.learning_rate)), .little);
        try writer.writeByte(@intFromEnum(self.loss_function));

        // Write network metadata
        const input_size = try self.getInputSize();
        const output_size = try self.getOutputSize();
        try writer.writeInt(u32, @intCast(input_size), .little);
        try writer.writeInt(u32, @intCast(output_size), .little);
        try writer.writeInt(u32, @intCast(self.layers.items.len), .little);

        // Write each layer
        for (self.layers.items) |layer| {
            switch (layer) {
                .Standard => |standard_layer| {
                    // Write layer type (Standard = 0)
                    try writer.writeByte(0);

                    // Write layer metadata
                    try writer.writeInt(u32, @intCast(standard_layer.getInputSize()), .little);
                    try writer.writeInt(u32, @intCast(standard_layer.getOutputSize()), .little);

                    // Write activation type
                    const activation_type: u8 = if (standard_layer.activation_fn == Activation.sigmoid) 0 else if (standard_layer.activation_fn == Activation.tanh) 1 else if (standard_layer.activation_fn == Activation.relu) 2 else if (standard_layer.activation_fn == Activation.softmax) 3 else 0; // Default to sigmoid
                    try writer.writeByte(activation_type);

                    // Write weights
                    for (0..standard_layer.weights.rows) |i| {
                        for (0..standard_layer.weights.cols) |j| {
                            try writer.writeInt(i64, @as(i64, @bitCast(try standard_layer.weights.get(i, j))), .little);
                        }
                    }

                    // Write biases
                    for (0..standard_layer.bias.cols) |j| {
                        try writer.writeInt(i64, @as(i64, @bitCast(try standard_layer.bias.get(0, j))), .little);
                    }
                },
                .Gated => |gated_layer| {
                    // Write layer type (Gated = 1)
                    try writer.writeByte(1);

                    // Write layer metadata
                    try writer.writeInt(u32, @intCast(gated_layer.getInputSize()), .little);
                    try writer.writeInt(u32, @intCast(gated_layer.getOutputSize()), .little);

                    // Write activation type (GLU = 4, SwiGLU = 5)
                    const activation_type: u8 = if (gated_layer.use_swiglu) 5 else 4;
                    try writer.writeByte(activation_type);

                    // Write linear weights
                    for (0..gated_layer.linear_weights.rows) |i| {
                        for (0..gated_layer.linear_weights.cols) |j| {
                            try writer.writeInt(i64, @as(i64, @bitCast(try gated_layer.linear_weights.get(i, j))), .little);
                        }
                    }

                    // Write linear biases
                    for (0..gated_layer.linear_bias.cols) |j| {
                        try writer.writeInt(i64, @as(i64, @bitCast(try gated_layer.linear_bias.get(0, j))), .little);
                    }

                    // Write gate weights
                    for (0..gated_layer.gate_weights.rows) |i| {
                        for (0..gated_layer.gate_weights.cols) |j| {
                            try writer.writeInt(i64, @as(i64, @bitCast(try gated_layer.gate_weights.get(i, j))), .little);
                        }
                    }

                    // Write gate biases
                    for (0..gated_layer.gate_bias.cols) |j| {
                        try writer.writeInt(i64, @as(i64, @bitCast(try gated_layer.gate_bias.get(0, j))), .little);
                    }
                },
            }
        }

        try writer.flush();
    }

    /// Loads a network from a binary file
    /// Returns a new Network instance with the loaded architecture and weights
    pub fn loadFromFile(allocator: Allocator, filepath: []const u8) !Network {
        const io = std.Options.debug_io;
        const file = try std.Io.Dir.cwd().openFile(io, filepath, .{});
        defer file.close(io);

        var buffer: [4096]u8 = undefined;
        var file_reader = file.readerStreaming(io, &buffer);
        const reader = &file_reader.interface;

        // Read and validate magic number
        var magic: [4]u8 = undefined;
        @memcpy(&magic, try reader.take(4));
        if (!std.mem.eql(u8, &magic, "ZNN\x00")) {
            return error.InvalidFileFormat;
        }

        // Read version
        const version = try reader.takeInt(u32, .little);
        if (version != 1 and version != 2) {
            return error.UnsupportedVersion;
        }

        // Skip creation timestamp
        _ = try reader.takeInt(i64, .little);

        var learning_rate: f64 = 0.1;
        var loss_function: LossFunction = .MeanSquaredError;
        if (version >= 2) {
            learning_rate = @as(f64, @bitCast(try reader.takeInt(i64, .little)));
            const loss_tag = try reader.takeByte();
            loss_function = switch (loss_tag) {
                0 => .MeanSquaredError,
                1 => .CrossEntropy,
                2 => .BinaryCrossEntropy,
                else => return error.InvalidFileFormat,
            };
        }

        // Read network metadata
        _ = try reader.takeInt(u32, .little); // input_size
        _ = try reader.takeInt(u32, .little); // output_size
        const layer_count = try reader.takeInt(u32, .little);

        var network = Network.init(allocator, learning_rate, loss_function);

        // Read each layer
        for (0..layer_count) |_| {
            const layer_type = try reader.takeByte();
            const layer_input_size = try reader.takeInt(u32, .little);
            const layer_output_size = try reader.takeInt(u32, .little);
            const activation_type = try reader.takeByte();

            switch (layer_type) {
                0 => { // Standard layer
                    // Determine activation function
                    const activation_fn: *const fn (f64) f64 = switch (activation_type) {
                        0 => Activation.sigmoid,
                        1 => Activation.tanh,
                        2 => Activation.relu,
                        3 => Activation.softmax,
                        else => Activation.sigmoid,
                    };
                    const activation_derivative_fn: *const fn (f64) f64 = switch (activation_type) {
                        0 => Activation.sigmoid_derivative,
                        1 => Activation.tanh_derivative,
                        2 => Activation.relu_derivative,
                        3 => Activation.softmax_derivative,
                        else => Activation.sigmoid_derivative,
                    };

                    // Create layer
                    try network.addLayer(layer_input_size, layer_output_size, activation_fn, activation_derivative_fn);

                    // Read weights
                    const layer = network.layers.items[network.layers.items.len - 1].Standard;
                    for (0..layer.weights.rows) |i| {
                        for (0..layer.weights.cols) |j| {
                            const weight = @as(f64, @bitCast(try reader.takeInt(i64, .little)));
                            try layer.weights.set(i, j, weight);
                        }
                    }

                    // Read biases
                    for (0..layer.bias.cols) |j| {
                        const bias = @as(f64, @bitCast(try reader.takeInt(i64, .little)));
                        try layer.bias.set(0, j, bias);
                    }
                },
                1 => { // Gated layer
                    const use_swiglu = activation_type == 5;
                    try network.addGatedLayer(layer_input_size, layer_output_size, use_swiglu);

                    // Read linear weights
                    const layer = network.layers.items[network.layers.items.len - 1].Gated;
                    for (0..layer.linear_weights.rows) |i| {
                        for (0..layer.linear_weights.cols) |j| {
                            const weight = @as(f64, @bitCast(try reader.takeInt(i64, .little)));
                            try layer.linear_weights.set(i, j, weight);
                        }
                    }

                    // Read linear biases
                    for (0..layer.linear_bias.cols) |j| {
                        const bias = @as(f64, @bitCast(try reader.takeInt(i64, .little)));
                        try layer.linear_bias.set(0, j, bias);
                    }

                    // Read gate weights
                    for (0..layer.gate_weights.rows) |i| {
                        for (0..layer.gate_weights.cols) |j| {
                            const weight = @as(f64, @bitCast(try reader.takeInt(i64, .little)));
                            try layer.gate_weights.set(i, j, weight);
                        }
                    }

                    // Read gate biases
                    for (0..layer.gate_bias.cols) |j| {
                        const bias = @as(f64, @bitCast(try reader.takeInt(i64, .little)));
                        try layer.gate_bias.set(0, j, bias);
                    }
                },
                else => return error.InvalidLayerType,
            }
        }

        return network;
    }
};

// Tests
test "network initialization and layer addition" {
    const allocator = testing.allocator;

    var network = Network.init(allocator, 0.1, .MeanSquaredError);
    defer network.deinit();

    try network.addLayer(2, 3, Activation.sigmoid, Activation.sigmoid_derivative);
    try network.addLayer(3, 1, Activation.sigmoid, Activation.sigmoid_derivative);

    try testing.expectEqual(@as(usize, 2), try network.getInputSize());
    try testing.expectEqual(@as(usize, 1), try network.getOutputSize());
}

test "network with gated layers" {
    const allocator = testing.allocator;

    var network = Network.init(allocator, 0.1, .MeanSquaredError);
    defer network.deinit();

    try network.addLayer(2, 3, Activation.sigmoid, Activation.sigmoid_derivative);
    try network.addGatedLayer(3, 2, false); // GLU layer
    try network.addGatedLayer(2, 1, true); // SwiGLU layer

    try testing.expectEqual(@as(usize, 2), try network.getInputSize());
    try testing.expectEqual(@as(usize, 1), try network.getOutputSize());
}

test "network forward propagation" {
    const allocator = testing.allocator;

    var network = Network.init(allocator, 0.1, .MeanSquaredError);
    defer network.deinit();

    // Create a simple network with 2 inputs, 3 hidden neurons, and 1 output
    try network.addLayer(2, 3, Activation.sigmoid, Activation.sigmoid_derivative);
    try network.addLayer(3, 1, Activation.sigmoid, Activation.sigmoid_derivative);

    // Set up input (1x2 matrix)
    var input = try Matrix.init(allocator, 1, 2);
    defer input.deinit();
    try input.set(0, 0, 1.0);
    try input.set(0, 1, 0.5);

    // Forward propagation
    var output = try network.forward(input);
    defer output.deinit();

    // Check output dimensions
    try testing.expectEqual(@as(usize, 1), output.rows);
    try testing.expectEqual(@as(usize, 1), output.cols);

    // The actual output value will depend on the random weights,
    // but it should be between 0 and 1 (sigmoid output range)
    const value = try output.get(0, 0);
    try testing.expect(value >= 0.0 and value <= 1.0);
}

test "network with mixed layer types" {
    const allocator = testing.allocator;

    var network = Network.init(allocator, 0.1, .MeanSquaredError);
    defer network.deinit();

    // Create a network with mixed layer types
    try network.addLayer(2, 4, Activation.relu, Activation.relu_derivative);
    try network.addGatedLayer(4, 3, false); // GLU layer
    try network.addGatedLayer(3, 2, true); // SwiGLU layer
    try network.addLayer(2, 1, Activation.sigmoid, Activation.sigmoid_derivative);

    // Set up input (1x2 matrix)
    var input = try Matrix.init(allocator, 1, 2);
    defer input.deinit();
    try input.set(0, 0, 1.0);
    try input.set(0, 1, 0.5);

    // Forward propagation
    var output = try network.forward(input);
    defer output.deinit();

    // Check output dimensions
    try testing.expectEqual(@as(usize, 1), output.rows);
    try testing.expectEqual(@as(usize, 1), output.cols);

    // The output should be between 0 and 1 (sigmoid output range)
    const value = try output.get(0, 0);
    try testing.expect(value >= 0.0 and value <= 1.0);
}

test "network backend inference matches cpu forward for mixed layers" {
    const allocator = testing.allocator;

    var network = Network.init(allocator, 0.1, .MeanSquaredError);
    defer network.deinit();

    try network.addLayer(2, 2, Activation.relu, Activation.relu_derivative);
    try network.addGatedLayer(2, 2, false);
    try network.addLayer(2, 1, Activation.sigmoid, Activation.sigmoid_derivative);

    const hidden = network.layers.items[0].Standard;
    try hidden.weights.set(0, 0, 0.5);
    try hidden.weights.set(0, 1, -0.2);
    try hidden.weights.set(1, 0, 0.25);
    try hidden.weights.set(1, 1, 0.75);
    try hidden.bias.set(0, 0, 0.1);
    try hidden.bias.set(0, 1, -0.3);

    const gated = network.layers.items[1].Gated;
    try gated.linear_weights.set(0, 0, 0.4);
    try gated.linear_weights.set(0, 1, -0.1);
    try gated.linear_weights.set(1, 0, 0.2);
    try gated.linear_weights.set(1, 1, 0.3);
    try gated.linear_bias.set(0, 0, 0.05);
    try gated.linear_bias.set(0, 1, -0.15);
    try gated.gate_weights.set(0, 0, -0.3);
    try gated.gate_weights.set(0, 1, 0.6);
    try gated.gate_weights.set(1, 0, 0.8);
    try gated.gate_weights.set(1, 1, -0.4);
    try gated.gate_bias.set(0, 0, 0.2);
    try gated.gate_bias.set(0, 1, -0.05);

    const output_layer = network.layers.items[2].Standard;
    try output_layer.weights.set(0, 0, 0.7);
    try output_layer.weights.set(1, 0, -0.45);
    try output_layer.bias.set(0, 0, 0.12);

    var input = try Matrix.init(allocator, 2, 2);
    defer input.deinit();
    try input.set(0, 0, 1.0);
    try input.set(0, 1, 0.5);
    try input.set(1, 0, -0.25);
    try input.set(1, 1, 0.75);

    var cpu_output = try network.forward(input);
    defer cpu_output.deinit();

    try testing.expect(hidden.last_input != null);
    const cached_first_input = try hidden.last_input.?.get(0, 0);

    var backend_instance = try backend_mod.createBackend(allocator, .CPU);
    defer backend_instance.deinit();

    const backend_input = try BackendMatrix.fromMatrix(backend_instance, input, allocator);
    defer backend_input.deinit();

    const backend_output = try network.predictBackend(backend_input);
    defer backend_output.deinit();

    var backend_cpu_output = try backend_output.toMatrix(allocator);
    defer backend_cpu_output.deinit();

    try testing.expectEqual(cpu_output.rows, backend_cpu_output.rows);
    try testing.expectEqual(cpu_output.cols, backend_cpu_output.cols);
    for (0..cpu_output.rows) |row| {
        for (0..cpu_output.cols) |col| {
            try testing.expectApproxEqAbs(try cpu_output.get(row, col), try backend_cpu_output.get(row, col), 1e-12);
        }
    }

    const build_options = @import("build_options");
    if (@import("builtin").os.tag == .macos and build_options.enable_metal) {
        var metal_backend = try backend_mod.createBackend(allocator, .Metal);
        defer metal_backend.deinit();

        if (metal_backend.getBackendType() == .Metal) {
            const metal_input = try BackendMatrix.fromMatrix(metal_backend, input, allocator);
            defer metal_input.deinit();

            const metal_output = try network.predictBackend(metal_input);
            defer metal_output.deinit();

            var metal_cpu_output = try metal_output.toMatrix(allocator);
            defer metal_cpu_output.deinit();

            try testing.expectEqual(cpu_output.rows, metal_cpu_output.rows);
            try testing.expectEqual(cpu_output.cols, metal_cpu_output.cols);
            for (0..cpu_output.rows) |row| {
                for (0..cpu_output.cols) |col| {
                    try testing.expectApproxEqAbs(try cpu_output.get(row, col), try metal_cpu_output.get(row, col), 1e-4);
                }
            }
        }
    }

    try testing.expect(hidden.last_input != null);
    try testing.expectApproxEqAbs(cached_first_input, try hidden.last_input.?.get(0, 0), 1e-12);
}

test "loss calculation" {
    const allocator = testing.allocator;

    var network = Network.init(allocator, 0.1, .MeanSquaredError);
    defer network.deinit();

    // Create predictions and targets
    var predictions = try Matrix.init(allocator, 2, 1);
    defer predictions.deinit();
    try predictions.set(0, 0, 0.7);
    try predictions.set(1, 0, 0.3);

    var targets = try Matrix.init(allocator, 2, 1);
    defer targets.deinit();
    try targets.set(0, 0, 1.0);
    try targets.set(1, 0, 0.0);

    // Calculate MSE loss
    const mse_loss = try network.calculateLoss(predictions, targets);
    // Expected: ((0.7-1.0)² + (0.3-0.0)²) / 2 = (0.09 + 0.09) / 2 = 0.09
    try testing.expectApproxEqAbs(@as(f64, 0.09), mse_loss, 0.0001);

    // Change to cross entropy loss
    network.loss_function = .CrossEntropy;
    const ce_loss = try network.calculateLoss(predictions, targets);
    // Expected: -[1.0*log(0.7) + (1-1.0)*log(1-0.7) + 0.0*log(0.3) + (1-0.0)*log(1-0.3)]/2
    try testing.expect(ce_loss > 0.0);
}

test "backpropagation and training" {
    const allocator = testing.allocator;

    var network = Network.init(allocator, 0.1, .MeanSquaredError);
    defer network.deinit();

    // Create a simple network for XOR problem
    try network.addLayer(2, 4, Activation.sigmoid, Activation.sigmoid_derivative);
    try network.addLayer(4, 1, Activation.sigmoid, Activation.sigmoid_derivative);

    // Create XOR training data
    var inputs = try Matrix.init(allocator, 4, 2);
    defer inputs.deinit();
    try inputs.set(0, 0, 0.0);
    try inputs.set(0, 1, 0.0); // [0, 0] -> 0
    try inputs.set(1, 0, 0.0);
    try inputs.set(1, 1, 1.0); // [0, 1] -> 1
    try inputs.set(2, 0, 1.0);
    try inputs.set(2, 1, 0.0); // [1, 0] -> 1
    try inputs.set(3, 0, 1.0);
    try inputs.set(3, 1, 1.0); // [1, 1] -> 0

    var targets = try Matrix.init(allocator, 4, 1);
    defer targets.deinit();
    try targets.set(0, 0, 0.0);
    try targets.set(1, 0, 1.0);
    try targets.set(2, 0, 1.0);
    try targets.set(3, 0, 0.0);

    // Train for a few epochs
    const epochs: usize = 10;
    const batch_size: usize = 2;
    const loss_history = try network.train(inputs, targets, epochs, batch_size);
    defer allocator.free(loss_history);

    // Check that loss history has the correct length
    try testing.expectEqual(epochs, loss_history.len);

    // Make predictions
    var predictions = try network.predict(inputs);
    defer predictions.deinit();

    // Check predictions dimensions
    try testing.expectEqual(@as(usize, 4), predictions.rows);
    try testing.expectEqual(@as(usize, 1), predictions.cols);

    // The predictions should be between 0 and 1
    for (0..4) |i| {
        const value = try predictions.get(i, 0);
        try testing.expect(value >= 0.0 and value <= 1.0);
    }
}

test "softmax cross entropy training step reduces loss" {
    const allocator = testing.allocator;

    var network = Network.init(allocator, 0.1, .CrossEntropy);
    defer network.deinit();

    try network.addLayer(2, 2, Activation.softmax, Activation.softmax_derivative);
    const layer = network.layers.items[0].Standard;
    layer.weights.fill(0.0);
    layer.bias.fill(0.0);

    var input = try Matrix.init(allocator, 1, 2);
    defer input.deinit();
    try input.set(0, 0, 1.0);
    try input.set(0, 1, 0.0);

    var target = try Matrix.init(allocator, 1, 2);
    defer target.deinit();
    try target.set(0, 0, 1.0);
    try target.set(0, 1, 0.0);

    var before = try network.forward(input);
    defer before.deinit();
    const before_loss = try network.calculateLoss(before, target);

    _ = try network.trainBatch(input, target);

    var after = try network.forward(input);
    defer after.deinit();
    const after_loss = try network.calculateLoss(after, target);

    try testing.expect(after_loss < before_loss);
    try testing.expect((try after.get(0, 0)) > (try before.get(0, 0)));
}

test "model persistence" {
    const allocator = testing.allocator;

    // Create a network with mixed layer types
    var network = Network.init(allocator, 0.1, .MeanSquaredError);
    defer network.deinit();

    try network.addLayer(2, 4, Activation.relu, Activation.relu_derivative);
    try network.addGatedLayer(4, 3, false); // GLU layer
    try network.addGatedLayer(3, 2, true); // SwiGLU layer
    try network.addLayer(2, 1, Activation.sigmoid, Activation.sigmoid_derivative);

    // Set up input (1x2 matrix)
    var input = try Matrix.init(allocator, 1, 2);
    defer input.deinit();
    try input.set(0, 0, 1.0);
    try input.set(0, 1, 0.5);

    // Get initial prediction
    var initial_output = try network.forward(input);
    defer initial_output.deinit();

    // Save network to file
    try network.saveToFile("test_model.bin");

    // Load network from file
    var loaded_network = try Network.loadFromFile(allocator, "test_model.bin");
    defer loaded_network.deinit();
    try testing.expectApproxEqAbs(network.learning_rate, loaded_network.learning_rate, 1e-12);
    try testing.expectEqual(network.loss_function, loaded_network.loss_function);

    // Get prediction from loaded network
    var loaded_output = try loaded_network.forward(input);
    defer loaded_output.deinit();

    // Compare predictions
    try testing.expectEqual(initial_output.rows, loaded_output.rows);
    try testing.expectEqual(initial_output.cols, loaded_output.cols);
    try testing.expectApproxEqAbs(try initial_output.get(0, 0), try loaded_output.get(0, 0), 1e-10);

    // Clean up test file
    try std.Io.Dir.cwd().deleteFile(std.Options.debug_io, "test_model.bin");
}
