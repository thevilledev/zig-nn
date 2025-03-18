const std = @import("std");

// Although this function looks imperative, note that its job is to
// declaratively construct a build graph that will be executed by an external
// runner.
pub fn build(b: *std.Build) void {
    // Standard target options allows the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    const target = b.standardTargetOptions(.{});

    // Standard optimization options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall. Here we do not
    // set a preferred release mode, allowing the user to decide how to optimize.
    const optimize = b.standardOptimizeOption(.{});

    // This creates a "module", which represents a collection of source files alongside
    // some compilation options, such as optimization mode and linked system libraries.
    // Every executable or library we compile will be based on one or more modules.
    const lib_mod = b.createModule(.{
        // `root_source_file` is the Zig "entry point" of the module. If a module
        // only contains e.g. external object files, you can make this `null`.
        // In this case the main source file is merely a path, however, in more
        // complicated build scripts, this could be a generated file.
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Now, we will create a static library based on the module we created above.
    // This creates a `std.Build.Step.Compile`, which is the build step responsible
    // for actually invoking the compiler.
    const lib = b.addLibrary(.{
        .linkage = .static,
        .name = "zig_nn",
        .root_module = lib_mod,
    });

    // This declares intent for the library to be installed into the standard
    // location when the user invokes the "install" step (the default step when
    // running `zig build`).
    b.installArtifact(lib);

    // Create a step for building examples
    const examples_step = b.step("examples", "Build all examples");

    // Define all examples in a single place for easier maintenance
    inline for ([_]struct {
        name: []const u8,
        src: []const u8,
        description: []const u8,
    }{
        .{ .name = "gated_network", .src = "examples/gated_network/gated_network.zig", .description = "Run the gated network example" },
        .{ .name = "simple_xor", .src = "examples/simple_xor/simple_xor.zig", .description = "Run the simple XOR example" },
        .{ .name = "xor_training", .src = "examples/xor_training/xor_training.zig", .description = "Run the XOR training example with backpropagation" },
        .{ .name = "binary_classification", .src = "examples/binary_classification/binary_classification.zig", .description = "Run the binary classification example with circular decision boundary" },
        .{ .name = "regression", .src = "examples/regression/regression.zig", .description = "Run the regression example with nonlinear function approximation" },
        .{ .name = "mnist", .src = "examples/mnist/mnist.zig", .description = "Run the MNIST digit recognition example" },
        .{ .name = "serving", .src = "examples/serving/server.zig", .description = "Run the serving example" },
        // Add new examples here in the future
    }) |example| {
        // Build the example executable
        const exe = b.addExecutable(.{
            .name = example.name,
            .root_source_file = b.path(example.src),
            .target = target,
            .optimize = optimize,
        });

        // Add the library module to the example executable
        exe.root_module.addImport("nn", lib_mod);

        // Install the example executable
        b.installArtifact(exe);

        // Create a run step for the example
        const run_cmd = b.addRunArtifact(exe);
        run_cmd.step.dependOn(b.getInstallStep());

        // Add command line arguments if provided
        if (b.args) |args| {
            run_cmd.addArgs(args);
        }

        // Add a separate step to run the example
        const step_name = "run_" ++ example.name;
        const run_step = b.step(step_name, example.description);
        run_step.dependOn(&run_cmd.step);

        // Add the example to the examples step
        examples_step.dependOn(&exe.step);
    }

    // Main test step that will run all tests
    const test_step = b.step("test", "Run all unit tests");

    // Create individual test steps for each source file
    // Run them in a logical order: matrix -> activation -> layer -> network
    var prev_step = addTestStep(b, test_step, "matrix", "src/matrix.zig", null);
    prev_step = addTestStep(b, test_step, "activation", "src/activation.zig", prev_step);
    prev_step = addTestStep(b, test_step, "layer", "src/layer.zig", prev_step);
    prev_step = addTestStep(b, test_step, "network", "src/network.zig", prev_step);
    _ = addTestStep(b, test_step, "inference_service", "src/inference_service.zig", prev_step);

    // Create a step for running acceptance tests from examples
    const acceptance_test_step = b.step("test-acceptance", "Run all example acceptance tests");

    // Add test steps for all examples
    var example_prev_step: ?*std.Build.Step = null;
    inline for ([_]struct {
        name: []const u8,
        path: []const u8,
    }{
        .{ .name = "gated_network", .path = "examples/gated_network/gated_network.zig" },
        .{ .name = "simple_xor", .path = "examples/simple_xor/simple_xor.zig" },
        .{ .name = "xor_training", .path = "examples/xor_training/xor_training.zig" },
        .{ .name = "binary_classification", .path = "examples/binary_classification/binary_classification.zig" },
        .{ .name = "regression", .path = "examples/regression/regression.zig" },
        .{ .name = "mnist", .path = "examples/mnist/mnist.zig" },
        .{ .name = "serving", .path = "examples/serving/server.zig" },
    }) |example| {
        example_prev_step = addTestStep(b, acceptance_test_step, example.name, example.path, example_prev_step);
    }
}

// Helper function to create a test step for a specific file
fn addTestStep(
    b: *std.Build,
    main_test_step: *std.Build.Step,
    name: []const u8,
    path: []const u8,
    prev_step: ?*std.Build.Step,
) *std.Build.Step {
    // Create a system command to run zig test directly
    const test_cmd = b.addSystemCommand(&[_][]const u8{
        "zig", "test", path,
    });

    // Print the test name with a separator for better visibility
    const echo_step = b.addSystemCommand(&[_][]const u8{
        "echo", b.fmt("\n=== Running {s} tests ===", .{name}),
    });

    // Make sure echo runs before the test
    test_cmd.step.dependOn(&echo_step.step);

    // If there's a previous step, make this step depend on it
    // This ensures sequential execution
    if (prev_step) |step| {
        echo_step.step.dependOn(step);
    }

    // Create an individual step for this test
    const test_name = b.fmt("test-{s}", .{name});
    const test_desc = b.fmt("Run {s} tests", .{name});
    const file_test_step = b.step(test_name, test_desc);
    file_test_step.dependOn(&test_cmd.step);

    // Add this test to the main test step
    main_test_step.dependOn(&test_cmd.step);

    return &test_cmd.step;
}
