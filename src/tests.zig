const std = @import("std");
const nn = @import("root.zig");

test "public API declarations are analyzable" {
    std.testing.refAllDecls(nn);
}
