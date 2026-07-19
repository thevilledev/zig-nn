const std = @import("std");

pub const Error = error{DimensionOverflow};

/// Computes the product of a set of dimensions without allowing integer
/// overflow to turn an invalid shape into a smaller allocation.
pub fn product(factors: []const usize) Error!usize {
    if (std.mem.indexOfScalar(usize, factors, 0) != null) return 0;

    var result: usize = 1;
    for (factors) |factor| {
        result = std.math.mul(usize, result, factor) catch return error.DimensionOverflow;
    }
    return result;
}

pub fn elementCount(rows: usize, cols: usize) Error!usize {
    return product(&.{ rows, cols });
}

/// Adds two dimensions without allowing a derived extent to wrap.
pub fn add(left: usize, right: usize) Error!usize {
    return std.math.add(usize, left, right) catch error.DimensionOverflow;
}

/// Returns false for both a mismatched product and an overflowing one. This is
/// useful at API validation boundaries where either condition is a dimension
/// mismatch to the caller.
pub fn matches(expected: usize, factors: []const usize) bool {
    const actual = product(factors) catch return false;
    return actual == expected;
}

test "checked dimension product" {
    try std.testing.expectEqual(@as(usize, 24), try product(&.{ 2, 3, 4 }));
    try std.testing.expectEqual(@as(usize, 0), try product(&.{ 4, 0, 8 }));
    try std.testing.expect(matches(24, &.{ 2, 3, 4 }));
    try std.testing.expect(!matches(23, &.{ 2, 3, 4 }));
}

test "dimension product reports overflow" {
    try std.testing.expectError(error.DimensionOverflow, product(&.{ std.math.maxInt(usize), 2 }));
    try std.testing.expect(!matches(0, &.{ std.math.maxInt(usize), 2 }));
    try std.testing.expectEqual(@as(usize, 0), try product(&.{ 0, std.math.maxInt(usize), 2 }));
    try std.testing.expectEqual(@as(usize, 0), try product(&.{ std.math.maxInt(usize), 2, 0 }));
}

test "checked dimension addition" {
    try std.testing.expectEqual(@as(usize, 7), try add(3, 4));
    try std.testing.expectError(error.DimensionOverflow, add(std.math.maxInt(usize), 1));
}
