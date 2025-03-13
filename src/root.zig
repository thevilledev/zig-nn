//! By convention, root.zig is the root source file when making a library. If
//! you are making an executable, the convention is to delete this file and
//! start with main.zig instead.
const std = @import("std");
const testing = std.testing;

// Import the modules
const matrix_mod = @import("matrix.zig");
const activation_mod = @import("activation.zig");
const layer_mod = @import("layer.zig");
const network_mod = @import("network.zig");

// Export the main components of the library
pub const Matrix = matrix_mod.Matrix;
pub const Activation = activation_mod.Activation;
pub const Layer = layer_mod.Layer;
pub const GatedLayer = layer_mod.GatedLayer;
pub const Network = network_mod.Network;
pub const LayerType = network_mod.LayerType;
pub const LayerVariant = network_mod.LayerVariant;

/// Simple example of creating and using a neural network
///
/// ```zig
/// // Create a neural network
/// var network = Network.init(allocator);
/// defer network.deinit();
///
/// // Add layers (input -> hidden -> output)
/// try network.addLayer(2, 3, Activation.relu, Activation.relu_derivative);
/// try network.addLayer(3, 1, Activation.sigmoid, Activation.sigmoid_derivative);
///
/// // Create input data
/// var input = try Matrix.init(allocator, 1, 2);
/// defer input.deinit();
/// input.set(0, 0, 0.5);
/// input.set(0, 1, 0.8);
///
/// // Forward pass
/// var output = try network.forward(input);
/// defer output.deinit();
///
/// // Get the result
/// const result = output.get(0, 0);
/// ```