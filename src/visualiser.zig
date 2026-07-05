const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const fmt = std.fmt;

const Activation = @import("activation.zig").Activation;
const LayerVariant = @import("network.zig").LayerVariant;
const Network = @import("network.zig").Network;

const target_width = 100;
const card_width = 16;
const card_inner_width = card_width - 2;
const connector_width = 12;
const card_rows = 7;
const field_capacity = 48;

const Align = enum {
    left,
    center,
};

const LayerRole = enum {
    input,
    dense,
    gated,
    output,
};

const LayerInfo = struct {
    role: LayerRole = .dense,
    input_size: usize = 0,
    output_size: usize = 0,
    units: usize = 0,
    param_count: usize = 0,

    title_buf: [field_capacity]u8 = undefined,
    title_len: usize = 0,
    detail_buf: [field_capacity]u8 = undefined,
    detail_len: usize = 0,
    kind_buf: [field_capacity]u8 = undefined,
    kind_len: usize = 0,
    strip_buf: [field_capacity]u8 = undefined,
    strip_len: usize = 0,
    footer_buf: [field_capacity]u8 = undefined,
    footer_len: usize = 0,
    connection_buf: [field_capacity]u8 = undefined,
    connection_len: usize = 0,
    connection_detail_buf: [field_capacity]u8 = undefined,
    connection_detail_len: usize = 0,

    fn setTitle(self: *LayerInfo, comptime format: []const u8, args: anytype) !void {
        try setField(&self.title_buf, &self.title_len, format, args);
    }

    fn setDetail(self: *LayerInfo, comptime format: []const u8, args: anytype) !void {
        try setField(&self.detail_buf, &self.detail_len, format, args);
    }

    fn setKind(self: *LayerInfo, comptime format: []const u8, args: anytype) !void {
        try setField(&self.kind_buf, &self.kind_len, format, args);
    }

    fn setStrip(self: *LayerInfo, comptime format: []const u8, args: anytype) !void {
        try setField(&self.strip_buf, &self.strip_len, format, args);
    }

    fn setFooter(self: *LayerInfo, comptime format: []const u8, args: anytype) !void {
        try setField(&self.footer_buf, &self.footer_len, format, args);
    }

    fn setConnection(self: *LayerInfo, comptime format: []const u8, args: anytype) !void {
        try setField(&self.connection_buf, &self.connection_len, format, args);
    }

    fn setConnectionDetail(self: *LayerInfo, comptime format: []const u8, args: anytype) !void {
        try setField(&self.connection_detail_buf, &self.connection_detail_len, format, args);
    }

    fn title(self: *const LayerInfo) []const u8 {
        return self.title_buf[0..self.title_len];
    }

    fn detail(self: *const LayerInfo) []const u8 {
        return self.detail_buf[0..self.detail_len];
    }

    fn kind(self: *const LayerInfo) []const u8 {
        return self.kind_buf[0..self.kind_len];
    }

    fn strip(self: *const LayerInfo) []const u8 {
        return self.strip_buf[0..self.strip_len];
    }

    fn footer(self: *const LayerInfo) []const u8 {
        return self.footer_buf[0..self.footer_len];
    }

    fn connection(self: *const LayerInfo) []const u8 {
        return self.connection_buf[0..self.connection_len];
    }

    fn connectionDetail(self: *const LayerInfo) []const u8 {
        return self.connection_detail_buf[0..self.connection_detail_len];
    }
};

fn setField(dest: *[field_capacity]u8, len: *usize, comptime format: []const u8, args: anytype) !void {
    const rendered = try fmt.bufPrint(dest, format, args);
    len.* = rendered.len;
}

/// Generates a Unicode terminal representation of the neural network.
///
/// Args:
///   network: The Network instance to visualize.
///   allocator: Allocator for memory allocation.
///
/// Returns:
///   An ArrayList(u8) containing the terminal visualization, or an error.
pub fn visualiseNetwork(network: *const Network, allocator: Allocator) !ArrayList(u8) {
    var buffer = std.Io.Writer.Allocating.init(allocator);
    errdefer buffer.deinit();
    const writer = &buffer.writer;

    if (network.layers.items.len == 0) {
        try renderEmptyNetwork(writer);
        return buffer.toArrayList();
    }

    var layers: ArrayList(LayerInfo) = .empty;
    defer layers.deinit(allocator);

    try layers.append(allocator, try makeInputInfo(network.layers.items[0].getInputSize()));

    var total_params: usize = 0;
    for (network.layers.items, 1..) |layer, layer_index| {
        const is_output = layer_index == network.layers.items.len;
        const info = try makeLayerInfo(layer, layer_index, is_output);
        total_params += info.param_count;
        try layers.append(allocator, info);
    }

    try renderSummary(writer, network, layers.items, total_params);
    try writer.writeByte('\n');
    try renderLayerStrips(writer, layers.items);

    return buffer.toArrayList();
}

fn renderEmptyNetwork(writer: anytype) !void {
    try writer.writeAll("Topology: empty | layers: 0 | params: 0\n\n");
    try renderCardTop(writer);
    try writer.writeByte('\n');
    try renderCardContent(writer, "Empty network");
    try writer.writeByte('\n');
    try renderCardContent(writer, "no layers");
    try writer.writeByte('\n');
    try renderCardContent(writer, "addLayer first");
    try writer.writeByte('\n');
    try renderCardContent(writer, "╭─╮");
    try writer.writeByte('\n');
    try renderCardContent(writer, "0 units");
    try writer.writeByte('\n');
    try renderCardBottom(writer);
    try writer.writeByte('\n');
}

fn makeInputInfo(input_size: usize) !LayerInfo {
    var info = LayerInfo{
        .role = .input,
        .output_size = input_size,
        .units = input_size,
    };

    try info.setTitle("Input", .{});
    try info.setDetail("dim {d}", .{input_size});
    try info.setKind("source", .{});
    try setNeuronStrip(&info, input_size);
    try setUnitFooter(&info, input_size);

    return info;
}

fn makeLayerInfo(layer: LayerVariant, layer_index: usize, is_output: bool) !LayerInfo {
    const role: LayerRole = if (is_output) .output else switch (layer) {
        .Standard => .dense,
        .Gated => .gated,
    };

    var info = LayerInfo{
        .role = role,
        .input_size = layer.getInputSize(),
        .output_size = layer.getOutputSize(),
        .units = layer.getOutputSize(),
    };

    switch (layer) {
        .Standard => |dense| {
            info.param_count = (dense.getInputSize() * dense.getOutputSize()) + dense.getOutputSize();
            if (is_output) {
                try info.setTitle("Output", .{});
            } else {
                try info.setTitle("Dense {d}", .{layer_index});
            }
            try info.setDetail("{d} -> {d}", .{ dense.getInputSize(), dense.getOutputSize() });
            try info.setKind("act {s}", .{activationName(dense.activation_fn)});
            try info.setConnection("W {d}x{d}", .{ dense.getInputSize(), dense.getOutputSize() });
            try info.setConnectionDetail("{d} params", .{info.param_count});
        },
        .Gated => |gated| {
            const gate_name = if (gated.use_swiglu) "SwiGLU" else "GLU";
            info.param_count = 2 * ((gated.getInputSize() * gated.getOutputSize()) + gated.getOutputSize());
            if (is_output) {
                try info.setTitle("Output", .{});
            } else {
                try info.setTitle("Gated {d}", .{layer_index});
            }
            try info.setDetail("{d} -> {d}", .{ gated.getInputSize(), gated.getOutputSize() });
            try info.setKind("{s} gate", .{gate_name});
            try info.setConnection("{s}", .{gate_name});
            try info.setConnectionDetail("{d}p {d}x{d}x2", .{ info.param_count, gated.getInputSize(), gated.getOutputSize() });
        },
    }

    try setNeuronStrip(&info, info.units);
    try setUnitFooter(&info, info.units);

    return info;
}

fn setNeuronStrip(info: *LayerInfo, units: usize) !void {
    if (units == 0) {
        try info.setStrip("╭─╮ empty", .{});
    } else if (units <= 4) {
        var strip: [field_capacity]u8 = undefined;
        var stream = std.Io.Writer.fixed(&strip);
        try stream.writeAll("╭─╮");
        for (0..units) |_| {
            try stream.writeAll(" ●");
        }
        const written = stream.buffered();
        try info.setStrip("{s}", .{written});
    } else {
        try info.setStrip("╭─╮ ● ● ⋯ ● ●", .{});
    }
}

fn setUnitFooter(info: *LayerInfo, units: usize) !void {
    if (units > 4) {
        try info.setFooter("⋮ {d} units", .{units});
    } else if (units == 1) {
        try info.setFooter("{d} unit", .{units});
    } else {
        try info.setFooter("{d} units", .{units});
    }
}

fn renderSummary(writer: anytype, network: *const Network, layers: []const LayerInfo, total_params: usize) !void {
    var tail_buf: [80]u8 = undefined;
    const tail = try fmt.bufPrint(&tail_buf, " | layers: {d} | params: {d} | lr: {d} | loss: {s}", .{
        network.layers.items.len,
        total_params,
        network.learning_rate,
        lossName(network.loss_function),
    });

    const prefix = "Topology: ";
    const tail_width = displayWidth(tail);
    const arch_width = if (target_width > prefix.len + tail_width)
        target_width - prefix.len - tail_width
    else
        16;

    try writer.writeAll(prefix);
    try writeArchitecture(writer, layers, arch_width);
    try writer.writeAll(tail);
    try writer.writeByte('\n');
}

fn renderLayerStrips(writer: anytype, layers: []const LayerInfo) !void {
    const max_cards = @max(@as(usize, 2), (target_width + connector_width) / (card_width + connector_width));

    var start: usize = 0;
    while (start < layers.len) {
        const end = @min(start + max_cards, layers.len);
        try renderStrip(writer, layers[start..end]);

        if (end == layers.len) break;

        try writer.writeByte('\n');
        try writer.print("⋮ continued: {s} feeds {s}\n\n", .{ layers[end - 1].title(), layers[end].title() });
        start = end - 1;
    }
}

fn renderStrip(writer: anytype, layers: []const LayerInfo) !void {
    for (0..card_rows) |row| {
        for (layers, 0..) |layer, index| {
            try renderCardRow(writer, &layer, row);
            if (index + 1 < layers.len) {
                try renderConnectorRow(writer, &layers[index + 1], row);
            }
        }
        try writer.writeByte('\n');
    }
}

fn renderCardRow(writer: anytype, layer: *const LayerInfo, row: usize) !void {
    switch (row) {
        0 => try renderCardTop(writer),
        1 => try renderCardContent(writer, layer.title()),
        2 => try renderCardContent(writer, layer.detail()),
        3 => try renderCardContent(writer, layer.kind()),
        4 => try renderCardContent(writer, layer.strip()),
        5 => try renderCardContent(writer, layer.footer()),
        6 => try renderCardBottom(writer),
        else => unreachable,
    }
}

fn renderConnectorRow(writer: anytype, right: *const LayerInfo, row: usize) !void {
    const text = switch (row) {
        0 => right.connection(),
        1 => "──▶",
        2 => right.connectionDetail(),
        else => "",
    };
    try writePadded(writer, text, connector_width, .center);
}

fn renderCardTop(writer: anytype) !void {
    try writer.writeAll("┌");
    try repeatText(writer, "─", card_inner_width);
    try writer.writeAll("┐");
}

fn renderCardBottom(writer: anytype) !void {
    try writer.writeAll("└");
    try repeatText(writer, "─", card_inner_width);
    try writer.writeAll("┘");
}

fn renderCardContent(writer: anytype, text: []const u8) !void {
    try writer.writeAll("│");
    try writePadded(writer, text, card_inner_width, .left);
    try writer.writeAll("│");
}

fn writeArchitecture(writer: anytype, layers: []const LayerInfo, max_width: usize) !void {
    var used: usize = 0;
    for (layers, 0..) |layer, index| {
        var segment_buf: [32]u8 = undefined;
        const segment = if (index == 0)
            try fmt.bufPrint(&segment_buf, "{d}", .{layer.units})
        else
            try fmt.bufPrint(&segment_buf, " ──▶ {d}", .{layer.units});

        const segment_width = displayWidth(segment);
        if (used + segment_width > max_width) {
            if (used + 2 <= max_width) {
                try writer.writeAll(" ⋯");
            } else if (used + 1 <= max_width) {
                try writer.writeAll("⋯");
            }
            return;
        }

        try writer.writeAll(segment);
        used += segment_width;
    }
}

fn writePadded(writer: anytype, text: []const u8, width: usize, alignment: Align) !void {
    const text_width = displayWidth(text);
    if (text_width >= width) {
        try writeClipped(writer, text, width);
        return;
    }

    const remaining = width - text_width;
    const left_padding = switch (alignment) {
        .left => 0,
        .center => remaining / 2,
    };
    const right_padding = remaining - left_padding;

    try writer.splatByteAll(' ', left_padding);
    try writer.writeAll(text);
    try writer.splatByteAll(' ', right_padding);
}

fn writeClipped(writer: anytype, text: []const u8, width: usize) !void {
    if (width == 0) return;
    if (displayWidth(text) <= width) {
        try writer.writeAll(text);
        return;
    }

    const visible_width = width - 1;
    var used: usize = 0;
    var index: usize = 0;
    while (index < text.len and used < visible_width) {
        const next = nextCodepointIndex(text, index);
        try writer.writeAll(text[index..next]);
        used += 1;
        index = next;
    }
    try writer.writeAll("⋯");
}

fn repeatText(writer: anytype, text: []const u8, count: usize) !void {
    for (0..count) |_| {
        try writer.writeAll(text);
    }
}

fn displayWidth(text: []const u8) usize {
    var width: usize = 0;
    for (text) |byte| {
        if ((byte & 0b1100_0000) != 0b1000_0000) {
            width += 1;
        }
    }
    return width;
}

fn nextCodepointIndex(text: []const u8, index: usize) usize {
    var next = index + 1;
    while (next < text.len and (text[next] & 0b1100_0000) == 0b1000_0000) {
        next += 1;
    }
    return next;
}

fn activationName(activation_fn: *const fn (f64) f64) []const u8 {
    if (activation_fn == Activation.relu) return "relu";
    if (activation_fn == Activation.sigmoid) return "sigmoid";
    if (activation_fn == Activation.tanh) return "tanh";
    if (activation_fn == Activation.linear) return "linear";
    if (activation_fn == Activation.swish) return "swish";
    if (activation_fn == Activation.softmax) return "softmax";
    return "custom";
}

fn lossName(loss_function: @import("network.zig").LossFunction) []const u8 {
    return switch (loss_function) {
        .MeanSquaredError => "MSE",
        .CrossEntropy => "cross entropy",
        .BinaryCrossEntropy => "binary xent",
    };
}

const testing = std.testing;

fn customActivation(x: f64) f64 {
    return x * x;
}

fn customActivationDerivative(x: f64) f64 {
    return 2.0 * x;
}

fn expectContains(haystack: []const u8, needle: []const u8) !void {
    try testing.expect(std.mem.indexOf(u8, haystack, needle) != null);
}

fn expectNotContains(haystack: []const u8, needle: []const u8) !void {
    try testing.expect(std.mem.indexOf(u8, haystack, needle) == null);
}

fn expectNonEmptyLinesWithinTarget(text: []const u8) !void {
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        if (line.len == 0) continue;
        try testing.expect(displayWidth(line) <= target_width);
    }
}

test "visualize empty network" {
    const allocator = testing.allocator;

    var network = Network.init(allocator, 0.01, .MeanSquaredError);
    defer network.deinit();

    var viz_buffer = try visualiseNetwork(&network, allocator);
    defer viz_buffer.deinit(allocator);

    try expectContains(viz_buffer.items, "empty");
    try expectContains(viz_buffer.items, "┌");
    try expectContains(viz_buffer.items, "╭─╮");
    try expectNonEmptyLinesWithinTarget(viz_buffer.items);
}

test "visualize simple network with Unicode cards" {
    const allocator = testing.allocator;

    var network = Network.init(allocator, 0.01, .MeanSquaredError);
    defer network.deinit();

    try network.addLayer(2, 3, Activation.relu, Activation.relu_derivative);
    try network.addLayer(3, 1, Activation.sigmoid, Activation.sigmoid_derivative);

    var viz_buffer = try visualiseNetwork(&network, allocator);
    defer viz_buffer.deinit(allocator);

    try expectContains(viz_buffer.items, "Topology: 2 ──▶ 3 ──▶ 1");
    try expectContains(viz_buffer.items, "Dense 1");
    try expectContains(viz_buffer.items, "Output");
    try expectContains(viz_buffer.items, "W 2x3");
    try expectContains(viz_buffer.items, "──▶");
    try expectContains(viz_buffer.items, "●");
    try expectContains(viz_buffer.items, "┌──────────────┐");
    try expectContains(viz_buffer.items, "└──────────────┘");
    try expectNotContains(viz_buffer.items, "-->");
    try expectNonEmptyLinesWithinTarget(viz_buffer.items);
}

test "visualize SwiGLU gated network" {
    const allocator = testing.allocator;

    var network = Network.init(allocator, 0.1, .MeanSquaredError);
    defer network.deinit();

    try network.addLayer(2, 4, Activation.relu, Activation.relu_derivative);
    try network.addGatedLayer(4, 3, true);
    try network.addLayer(3, 1, Activation.sigmoid, Activation.sigmoid_derivative);

    var viz_buffer = try visualiseNetwork(&network, allocator);
    defer viz_buffer.deinit(allocator);

    try expectContains(viz_buffer.items, "Gated 2");
    try expectContains(viz_buffer.items, "SwiGLU");
    try expectContains(viz_buffer.items, "30p 4x3x2");
    try expectNotContains(viz_buffer.items, "-->");
    try expectNonEmptyLinesWithinTarget(viz_buffer.items);
}

test "visualize tall layer with compact neurons" {
    const allocator = testing.allocator;

    var network = Network.init(allocator, 0.1, .MeanSquaredError);
    defer network.deinit();

    try network.addLayer(4, 20, Activation.relu, Activation.relu_derivative);
    try network.addLayer(20, 4, Activation.sigmoid, Activation.sigmoid_derivative);

    var viz_buffer = try visualiseNetwork(&network, allocator);
    defer viz_buffer.deinit(allocator);

    try expectContains(viz_buffer.items, "╭─╮ ● ● ⋯ ● ●");
    try expectContains(viz_buffer.items, "⋮ 20 units");
    try expectNonEmptyLinesWithinTarget(viz_buffer.items);
}

test "visualize deep network wraps into strips" {
    const allocator = testing.allocator;

    var network = Network.init(allocator, 0.1, .MeanSquaredError);
    defer network.deinit();

    try network.addLayer(2, 10, Activation.tanh, Activation.tanh_derivative);
    try network.addLayer(10, 8, Activation.relu, Activation.relu_derivative);
    try network.addLayer(8, 6, Activation.relu, Activation.relu_derivative);
    try network.addLayer(6, 4, Activation.relu, Activation.relu_derivative);
    try network.addLayer(4, 1, Activation.sigmoid, Activation.sigmoid_derivative);

    var viz_buffer = try visualiseNetwork(&network, allocator);
    defer viz_buffer.deinit(allocator);

    try expectContains(viz_buffer.items, "⋮ continued");
    try expectContains(viz_buffer.items, "Dense 3");
    try expectContains(viz_buffer.items, "Output");
    try expectNotContains(viz_buffer.items, "Network architecture:");
    try expectNonEmptyLinesWithinTarget(viz_buffer.items);
}

test "visualize custom activation fallback" {
    const allocator = testing.allocator;

    var network = Network.init(allocator, 0.1, .MeanSquaredError);
    defer network.deinit();

    try network.addLayer(2, 2, customActivation, customActivationDerivative);
    try network.addLayer(2, 1, Activation.linear, Activation.linear_derivative);

    var viz_buffer = try visualiseNetwork(&network, allocator);
    defer viz_buffer.deinit(allocator);

    try expectContains(viz_buffer.items, "act custom");
    try expectContains(viz_buffer.items, "act linear");
    try expectNonEmptyLinesWithinTarget(viz_buffer.items);
}
