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

    // Build the gated_network example
    const gated_network_exe = b.addExecutable(.{
        .name = "gated_network",
        .root_source_file = b.path("examples/gated_network/gated_network.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Add the library module to the example executable
    gated_network_exe.root_module.addImport("zig-nn", lib_mod);

    // Install the example executable
    b.installArtifact(gated_network_exe);

    // Create a run step for the gated_network example
    const run_gated_network_cmd = b.addRunArtifact(gated_network_exe);
    run_gated_network_cmd.step.dependOn(b.getInstallStep());

    // Add a separate step to run the gated_network example
    const run_gated_network_step = b.step("run-gated-network", "Run the gated network example");
    run_gated_network_step.dependOn(&run_gated_network_cmd.step);

    // Add the gated_network example to the examples step
    examples_step.dependOn(&gated_network_exe.step);

    // Main test step that will run all tests
    const test_step = b.step("test", "Run all unit tests");

    // Create individual test steps for each source file
    // Run them in a logical order: matrix -> activation -> layer -> network
    var prev_step = addTestStep(b, test_step, "matrix", "src/matrix.zig", null);
    prev_step = addTestStep(b, test_step, "activation", "src/activation.zig", prev_step);
    prev_step = addTestStep(b, test_step, "layer", "src/layer.zig", prev_step);
    _ = addTestStep(b, test_step, "network", "src/network.zig", prev_step);
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
