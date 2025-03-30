const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const Network = @import("network.zig").Network;
const LayerVariant = @import("network.zig").LayerVariant;
const fmt = std.fmt;

/// Represents dimensions for layout calculations
const LayerDim = struct {
    input_size: usize,
    output_size: usize,
    height: usize,
    label: []const u8,
};

/// Generates an ASCII representation of the neural network.
///
/// Args:
///   network: The Network instance to visualize.
///   allocator: Allocator for memory allocation.
///
/// Returns:
///   An ArrayList(u8) containing the ASCII visualization, or an error.
pub fn visualiseNetwork(network: *const Network, allocator: Allocator) !ArrayList(u8) {
    var buffer = ArrayList(u8).init(allocator);
    const writer = buffer.writer();

    if (network.layers.items.len == 0) {
        try writer.print("Empty Network\n", .{});
        return buffer;
    }

    // 1. Gather layer dimensions and find max height
    var layer_dims = ArrayList(LayerDim).init(allocator);
    defer layer_dims.deinit();

    var max_height: usize = 0;
    var max_label_len: usize = 0;

    // --- Handle Input 'Layer' --- (Inferred from the first actual layer)
    const first_layer = network.layers.items[0];
    const input_size = first_layer.getInputSize();
    const input_label = try fmt.allocPrint(allocator, "Input (Size: {d})", .{input_size});
    try layer_dims.append(.{
        .input_size = 0, // Not applicable for input pseudo-layer
        .output_size = input_size,
        .height = input_size,
        .label = input_label,
    });
    max_height = @max(max_height, input_size);
    max_label_len = @max(max_label_len, input_label.len);

    // --- Handle Hidden & Output Layers ---
    for (network.layers.items, 1..) |layer, i| {
        const layer_input_size = layer.getInputSize();
        const layer_output_size = layer.getOutputSize();
        const layer_height = layer_output_size;
        const layer_type_str = switch (layer) {
            .Standard => "Standard",
            .Gated => "Gated",
        };
        const layer_label_prefix = if (i == network.layers.items.len) "Output" else "Layer";
        const layer_label = try fmt.allocPrint(allocator, "{s} {d}: {s} (In: {d}, Out: {d})", .{ layer_label_prefix, i, layer_type_str, layer_input_size, layer_output_size });

        try layer_dims.append(.{
            .input_size = layer_input_size,
            .output_size = layer_output_size,
            .height = layer_height,
            .label = layer_label,
        });
        max_height = @max(max_height, layer_height);
        max_label_len = @max(max_label_len, layer_label.len);
    }

    // Constants for drawing
    const neuron_str = "( )";
    const neuron_width = neuron_str.len;

    // There's a memory leak in our test because we allocate labels but don't free them
    // Let's create a free list for these allocations
    var label_free_list = ArrayList([]const u8).init(allocator);
    defer {
        for (label_free_list.items) |label| {
            allocator.free(label);
        }
        label_free_list.deinit();
    }

    // Track the labels we allocate for cleanup
    try label_free_list.append(input_label);
    for (layer_dims.items[1..]) |ld| {
        try label_free_list.append(ld.label);
    }

    // Constants for drawing - reduce spacing for compactness
    const connection_str = "-->";
    const h_spacing = 3; // Horizontal spacing between components

    // Calculate max label width to truncate if needed
    const max_display_width = 120; // Reasonable terminal width
    const avg_column_width = 25; // Estimated average space needed per layer
    const max_columns = max_display_width / avg_column_width;

    // Check if we need to split visualization into multiple rows
    const total_layers = layer_dims.items.len;
    const is_wide_network = total_layers > max_columns;

    // For wide networks, we'll show a more compact representation
    if (is_wide_network) {
        try writer.print("Network architecture: ", .{});
        // Print layer sizes in a compact format: [2]->[3]->[1]
        for (layer_dims.items, 0..) |ld, i| {
            if (i == 0) {
                try writer.print("[{d}]", .{ld.output_size});
            } else {
                const layer_type = switch (network.layers.items[i - 1]) {
                    .Standard => "S",
                    .Gated => "G",
                };
                try writer.print("-{s}->[{d}]", .{ layer_type, ld.output_size });
            }
        }
        try writer.writeByte('\n');

        // Add some details about each layer
        try writer.print("\nLayer details:\n", .{});
        for (layer_dims.items, 0..) |ld, i| {
            if (i == 0) {
                try writer.print("Input: {d} neurons\n", .{ld.output_size});
            } else {
                const layer_type = switch (network.layers.items[i - 1]) {
                    .Standard => "Standard",
                    .Gated => "Gated",
                };
                try writer.print("Layer {d}: {s} ({d}->{d})\n", .{ i, layer_type, ld.input_size, ld.output_size });
            }
        }
        return buffer;
    }

    // For narrower networks, use the visual representation
    // 1. Build header row with compact spacing
    for (layer_dims.items, 0..) |ld, layer_idx| {
        try writer.print("{s}", .{ld.label});

        if (layer_idx < layer_dims.items.len - 1) {
            try writer.writeByteNTimes(' ', h_spacing);
        }
    }
    try writer.writeByte('\n');

    // 2. Build neuron rows with connections
    for (0..max_height) |row_idx| {
        for (layer_dims.items, 0..) |ld, layer_idx| {
            const current_layer_height = ld.height;
            const is_last_layer = (layer_idx == layer_dims.items.len - 1);

            // Calculate vertical padding for centering
            const top_padding = (max_height - current_layer_height) / 2;
            const bottom_padding = max_height - current_layer_height - top_padding;

            // Check if a neuron should be drawn at this row
            const should_draw_neuron = (row_idx >= top_padding) and (row_idx < max_height - bottom_padding);

            // Add padding before neuron
            try writer.writeByteNTimes(' ', 6);

            // Draw neuron or space
            if (should_draw_neuron) {
                try writer.print("{s}", .{neuron_str});
            } else {
                try writer.writeByteNTimes(' ', neuron_width);
            }

            // Add connection if not the last layer
            if (!is_last_layer) {
                try writer.writeByteNTimes(' ', 1); // Reduced space after neuron

                // For networks with different sized layers, we need to ensure connections
                // are shown between all neurons in adjacent layers
                const next_layer = layer_dims.items[layer_idx + 1];
                const next_layer_height = next_layer.height;
                const next_top_padding = (max_height - next_layer_height) / 2;
                const next_bottom_padding = max_height - next_layer_height - next_top_padding;

                // Check if this row connects to any neuron in the next layer
                const connects_to_next = (row_idx >= next_top_padding) and
                    (row_idx < max_height - next_bottom_padding);

                // Draw connections where appropriate
                if (should_draw_neuron or connects_to_next) {
                    try writer.print("{s}", .{connection_str});
                } else {
                    try writer.writeByteNTimes(' ', connection_str.len);
                }

                try writer.writeByteNTimes(' ', h_spacing); // Reduced spacing
            }
        }
        try writer.writeByte('\n');
    }

    return buffer;
}

// TODO: Add tests later
const testing = std.testing;
const Activation = @import("activation.zig").Activation;

test "visualize simple network" {
    const allocator = testing.allocator;

    // Create a simple network: 2 -> 3 -> 1
    var network = Network.init(allocator, 0.01, .MeanSquaredError); // Learning rate and loss don't matter for viz
    defer network.deinit();

    try network.addLayer(2, 3, Activation.relu, Activation.relu_derivative); // Input: 2, Hidden: 3
    try network.addLayer(3, 1, Activation.sigmoid, Activation.sigmoid_derivative); // Hidden: 3, Output: 1

    // Generate visualization
    var viz_buffer = try visualiseNetwork(&network, allocator);
    defer viz_buffer.deinit();

    // Don't test exact string output as it's implementation dependent
    // Just verify we got some output and it contains expected elements
    try testing.expect(viz_buffer.items.len > 0);
    try testing.expect(std.mem.indexOf(u8, viz_buffer.items, "Input") != null);
    try testing.expect(std.mem.indexOf(u8, viz_buffer.items, "Layer") != null);
    try testing.expect(std.mem.indexOf(u8, viz_buffer.items, "Standard") != null);
    try testing.expect(std.mem.indexOf(u8, viz_buffer.items, "( )") != null);
}
