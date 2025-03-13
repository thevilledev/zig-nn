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

pub export fn add(a: i32, b: i32) i32 {
    return a + b;
}

test "basic add functionality" {
    try testing.expect(add(3, 7) == 10);
}