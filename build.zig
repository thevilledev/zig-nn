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

    // GPU backend option
    const gpu_option = b.option(
        []const u8,
        "gpu",
        "GPU backend to use: auto, metal, cuda, rocm, or none (default: none)",
    ) orelse "none";
    const cuda_path = b.option(
        []const u8,
        "cuda-path",
        "CUDA toolkit root used for headers and NVRTC libraries (default: /usr/local/cuda)",
    ) orelse "/usr/local/cuda";
    const rocm_path = b.option(
        []const u8,
        "rocm-path",
        "ROCm toolkit root used for HIP, HIPRTC, and rocBLAS libraries (default: /opt/rocm)",
    ) orelse "/opt/rocm";

    // Determine enabled GPU backends based on target and option
    const enable_gpu = !std.mem.eql(u8, gpu_option, "none");
    const is_macos = target.result.os.tag == .macos;
    const is_linux = target.result.os.tag == .linux;

    const enable_metal = is_macos and (std.mem.eql(u8, gpu_option, "metal") or
        std.mem.eql(u8, gpu_option, "auto"));

    const enable_cuda = is_linux and (std.mem.eql(u8, gpu_option, "cuda") or
        std.mem.eql(u8, gpu_option, "auto"));
    const enable_rocm = is_linux and std.mem.eql(u8, gpu_option, "rocm");

    // Build options
    const build_options = b.addOptions();
    build_options.addOption(bool, "enable_gpu", enable_gpu);
    build_options.addOption(bool, "enable_metal", enable_metal);
    build_options.addOption(bool, "enable_cuda", enable_cuda);
    build_options.addOption(bool, "enable_rocm", enable_rocm);

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

    // Add the build options to the module
    lib_mod.addOptions("build_options", build_options);

    addMetalSupport(b, lib_mod, enable_metal, true);
    addCudaSupport(b, lib_mod, enable_cuda, true, cuda_path);
    addRocmSupport(b, lib_mod, enable_rocm, true, rocm_path);

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

    // Create a step for building experiments
    const experiments_step = b.step("experiments", "Build all experiments");
    const examples_step = b.step("examples", "Alias for the experiments build step");
    examples_step.dependOn(experiments_step);

    // Define all experiments in a single place for easier maintenance
    inline for ([_]struct {
        name: []const u8,
        src: []const u8,
        description: []const u8,
    }{
        .{ .name = "gated_network", .src = "experiments/gated_network/gated_network.zig", .description = "Run the gated network experiment" },
        .{ .name = "simple_xor", .src = "experiments/simple_xor/simple_xor.zig", .description = "Run the simple XOR experiment" },
        .{ .name = "xor_training", .src = "experiments/xor_training/xor_training.zig", .description = "Run the XOR training experiment with backpropagation" },
        .{ .name = "binary_classification", .src = "experiments/binary_classification/binary_classification.zig", .description = "Run the binary classification experiment with circular decision boundary" },
        .{ .name = "regression", .src = "experiments/regression/regression.zig", .description = "Run the regression experiment with nonlinear function approximation" },
        .{ .name = "mnist", .src = "experiments/mnist/mnist.zig", .description = "Run the MNIST digit recognition experiment" },
        .{ .name = "serving", .src = "experiments/serving/server.zig", .description = "Run the serving experiment" },
        .{ .name = "network_visualisation", .src = "experiments/network_visualisation/network_visualisation.zig", .description = "Run the network visualisation experiment" },
        .{ .name = "backend_demo", .src = "experiments/backend_demo/backend_demo.zig", .description = "Run the backend abstraction demonstration" },
        .{ .name = "backend_training", .src = "experiments/backend_training/backend_training.zig", .description = "Run the backend-aware training experiment" },
        .{ .name = "optimizer_lab", .src = "experiments/optimizer_lab/optimizer_lab.zig", .description = "Compare SGD, momentum, and AdamW on two-moons classification" },
        .{ .name = "tokenizer_lab", .src = "experiments/tokenizer_lab/tokenizer_lab.zig", .description = "Compare byte and learned BPE tokenization" },
        .{ .name = "padding_masks", .src = "experiments/padding_masks/padding_masks.zig", .description = "Learn batching and padding masks for variable-length sequences" },
        .{ .name = "word2vec", .src = "experiments/word2vec/word2vec.zig", .description = "Learn distributional word embeddings with skip-gram" },
        .{ .name = "speech_commands", .src = "experiments/speech_commands/speech_commands.zig", .description = "Train and run an eight-word speech command recognizer" },
        .{ .name = "text_classifier", .src = "experiments/text_classifier/text_classifier.zig", .description = "Classify padded text with a masked multi-head encoder" },
        .{ .name = "sequence_tagging", .src = "experiments/sequence_tagging/sequence_tagging.zig", .description = "Learn structured BIO tagging with a linear-chain CRF" },
        .{ .name = "decoding_lab", .src = "experiments/decoding_lab/decoding_lab.zig", .description = "Compare greedy, top-k, nucleus, and repetition-aware decoding" },
        .{ .name = "seq2seq", .src = "experiments/seq2seq/seq2seq.zig", .description = "Learn encoder-decoder alignment with cross-attention" },
        .{ .name = "semantic_search", .src = "experiments/semantic_search/semantic_search.zig", .description = "Learn dual-encoder retrieval with symmetric InfoNCE" },
        .{ .name = "cnn", .src = "experiments/cnn/cnn.zig", .description = "Learn image patterns with convolution and max pooling" },
        .{ .name = "autoencoder", .src = "experiments/autoencoder/autoencoder.zig", .description = "Learn denoising and latent representations with an autoencoder" },
        .{ .name = "gru_sequence", .src = "experiments/gru_sequence/gru_sequence.zig", .description = "Learn selective sequence memory with a GRU" },
        .{ .name = "transformer_encoder", .src = "experiments/transformer_encoder/transformer_encoder.zig", .description = "Learn bidirectional context with a Transformer encoder" },
        .{ .name = "dqn", .src = "experiments/dqn/dqn.zig", .description = "Learn value-based reinforcement learning with DQN" },
        .{ .name = "gpu", .src = "experiments/gpu/gpu.zig", .description = "Run the GPU experiment" },
        .{ .name = "gpu_benchmark", .src = "experiments/gpu_benchmark/gpu_benchmark.zig", .description = "Benchmark Metal backend against CPU" },
        .{ .name = "turboquant", .src = "experiments/quantization/turboquant.zig", .description = "Run the TurboQuant paper lab experiment" },
        .{ .name = "tiny_gpt", .src = "experiments/tiny_gpt/tiny_gpt.zig", .description = "Run the tiny GPT decoder-only Transformer experiment" },
        .{ .name = "tiny_gpt_openai", .src = "experiments/tiny_gpt/openai_server.zig", .description = "Run the Tiny GPT OpenAI-compatible inference server" },
        // Add new experiments here in the future
    }) |experiment| {
        const exe_mod = b.createModule(.{
            .root_source_file = b.path(experiment.src),
            .target = target,
            .optimize = optimize,
        });

        // Add the library module to the experiment executable
        exe_mod.addImport("nn", lib_mod);

        // Build the experiment executable
        const exe = b.addExecutable(.{
            .name = experiment.name,
            .root_module = exe_mod,
        });

        // Conditionally link frameworks/libraries for GPU experiments
        if (std.mem.eql(u8, experiment.name, "gpu") or std.mem.eql(u8, experiment.name, "gpu_benchmark")) {
            if (enable_metal) {
                addMetalSupport(b, exe_mod, enable_metal, false);
            }
            if (enable_cuda) {
                addCudaSupport(b, exe_mod, enable_cuda, false, cuda_path);
            }
            if (enable_rocm) {
                addRocmSupport(b, exe_mod, enable_rocm, false, rocm_path);
            }
        }

        // Install the experiment executable
        b.installArtifact(exe);

        // Create a run step for the experiment
        const run_cmd = b.addRunArtifact(exe);
        run_cmd.step.dependOn(b.getInstallStep());

        // Add command line arguments if provided
        if (b.args) |args| {
            run_cmd.addArgs(args);
        }

        // Add a separate step to run the experiment
        const step_name = "run_" ++ experiment.name;
        const run_step = b.step(step_name, experiment.description);
        run_step.dependOn(&run_cmd.step);

        // Add the experiment to the experiments step
        experiments_step.dependOn(&exe.step);
    }

    addBenchmarkStep(
        b,
        target,
        build_options,
        enable_metal,
        enable_cuda,
        enable_rocm,
        cuda_path,
        rocm_path,
        .ReleaseFast,
        "zig_nn_benchmark",
        "benchmark",
        "Run repeatable ReleaseFast CPU/GPU benchmark suite",
    );
    addBenchmarkStep(
        b,
        target,
        build_options,
        enable_metal,
        enable_cuda,
        enable_rocm,
        cuda_path,
        rocm_path,
        .Debug,
        "zig_nn_benchmark_debug",
        "benchmark-debug",
        "Run repeatable Debug CPU/GPU benchmark suite",
    );

    // Main test step that will run all tests
    const test_step = b.step("test", "Run all unit tests");

    // Create individual test steps for each source file
    // Run them in a logical order: matrix -> activation -> layer -> network
    var prev_step = addTestStep(b, test_step, "matrix", "src/matrix.zig", null, target, optimize, build_options, enable_metal, enable_cuda, enable_rocm, cuda_path, rocm_path, null);
    prev_step = addTestStep(b, test_step, "activation", "src/activation.zig", prev_step, target, optimize, build_options, enable_metal, enable_cuda, enable_rocm, cuda_path, rocm_path, null);
    prev_step = addTestStep(b, test_step, "tensor", "src/tensor.zig", prev_step, target, optimize, build_options, enable_metal, enable_cuda, enable_rocm, cuda_path, rocm_path, null);
    prev_step = addTestStep(b, test_step, "text", "src/text.zig", prev_step, target, optimize, build_options, enable_metal, enable_cuda, enable_rocm, cuda_path, rocm_path, null);
    prev_step = addTestStep(b, test_step, "embeddings", "src/embeddings.zig", prev_step, target, optimize, build_options, enable_metal, enable_cuda, enable_rocm, cuda_path, rocm_path, null);
    prev_step = addTestStep(b, test_step, "structured", "src/structured.zig", prev_step, target, optimize, build_options, enable_metal, enable_cuda, enable_rocm, cuda_path, rocm_path, null);
    prev_step = addTestStep(b, test_step, "decoding", "src/decoding.zig", prev_step, target, optimize, build_options, enable_metal, enable_cuda, enable_rocm, cuda_path, rocm_path, null);
    prev_step = addTestStep(b, test_step, "retrieval", "src/retrieval.zig", prev_step, target, optimize, build_options, enable_metal, enable_cuda, enable_rocm, cuda_path, rocm_path, null);
    prev_step = addTestStep(b, test_step, "audio", "src/audio.zig", prev_step, target, optimize, build_options, enable_metal, enable_cuda, enable_rocm, cuda_path, rocm_path, null);
    prev_step = addTestStep(b, test_step, "modules", "src/modules.zig", prev_step, target, optimize, build_options, enable_metal, enable_cuda, enable_rocm, cuda_path, rocm_path, null);
    prev_step = addTestStep(b, test_step, "training", "src/training.zig", prev_step, target, optimize, build_options, enable_metal, enable_cuda, enable_rocm, cuda_path, rocm_path, null);
    prev_step = addTestStep(b, test_step, "spatial", "src/spatial.zig", prev_step, target, optimize, build_options, enable_metal, enable_cuda, enable_rocm, cuda_path, rocm_path, null);
    prev_step = addTestStep(b, test_step, "recurrent", "src/recurrent.zig", prev_step, target, optimize, build_options, enable_metal, enable_cuda, enable_rocm, cuda_path, rocm_path, null);
    prev_step = addTestStep(b, test_step, "reinforcement", "src/reinforcement.zig", prev_step, target, optimize, build_options, enable_metal, enable_cuda, enable_rocm, cuda_path, rocm_path, null);
    prev_step = addTestStep(b, test_step, "transformer", "src/transformer.zig", prev_step, target, optimize, build_options, enable_metal, enable_cuda, enable_rocm, cuda_path, rocm_path, null);
    prev_step = addTestStep(b, test_step, "layer", "src/layer.zig", prev_step, target, optimize, build_options, enable_metal, enable_cuda, enable_rocm, cuda_path, rocm_path, null);
    prev_step = addTestStep(b, test_step, "layer_norm", "src/layer_norm.zig", prev_step, target, optimize, build_options, enable_metal, enable_cuda, enable_rocm, cuda_path, rocm_path, null);
    prev_step = addTestStep(b, test_step, "network", "src/network.zig", prev_step, target, optimize, build_options, enable_metal, enable_cuda, enable_rocm, cuda_path, rocm_path, null);
    prev_step = addTestStep(b, test_step, "inference_service", "src/inference_service.zig", prev_step, target, optimize, build_options, enable_metal, enable_cuda, enable_rocm, cuda_path, rocm_path, null);
    prev_step = addTestStep(b, test_step, "visualiser", "src/visualiser.zig", prev_step, target, optimize, build_options, enable_metal, enable_cuda, enable_rocm, cuda_path, rocm_path, null);
    prev_step = addTestStep(b, test_step, "quantization", "src/quantization.zig", prev_step, target, optimize, build_options, enable_metal, enable_cuda, enable_rocm, cuda_path, rocm_path, null);

    // Add backend-related tests
    prev_step = addTestStep(b, test_step, "backend", "src/backend.zig", prev_step, target, optimize, build_options, enable_metal, enable_cuda, enable_rocm, cuda_path, rocm_path, null);
    prev_step = addTestStep(b, test_step, "cpu_backend", "src/cpu_backend.zig", prev_step, target, optimize, build_options, enable_metal, enable_cuda, enable_rocm, cuda_path, rocm_path, null);
    prev_step = addTestStep(b, test_step, "metal_backend", "src/metal_backend.zig", prev_step, target, optimize, build_options, enable_metal, enable_cuda, enable_rocm, cuda_path, rocm_path, null);
    prev_step = addTestStep(b, test_step, "cuda_backend", "src/cuda_backend.zig", prev_step, target, optimize, build_options, enable_metal, enable_cuda, enable_rocm, cuda_path, rocm_path, null);
    _ = addTestStep(b, test_step, "rocm_backend", "src/rocm_backend.zig", prev_step, target, optimize, build_options, enable_metal, enable_cuda, enable_rocm, cuda_path, rocm_path, null);

    // Create a step for running acceptance tests from experiments
    const acceptance_test_step = b.step("test-acceptance", "Run all experiment acceptance tests");

    // Add test steps for all experiments
    var experiment_prev_step: ?*std.Build.Step = null;
    inline for ([_]struct {
        name: []const u8,
        path: []const u8,
    }{
        .{ .name = "gated_network", .path = "experiments/gated_network/gated_network.zig" },
        .{ .name = "simple_xor", .path = "experiments/simple_xor/simple_xor.zig" },
        .{ .name = "xor_training", .path = "experiments/xor_training/xor_training.zig" },
        .{ .name = "binary_classification", .path = "experiments/binary_classification/binary_classification.zig" },
        .{ .name = "regression", .path = "experiments/regression/regression.zig" },
        .{ .name = "mnist", .path = "experiments/mnist/mnist.zig" },
        .{ .name = "serving", .path = "experiments/serving/server.zig" },
        .{ .name = "network_visualisation", .path = "experiments/network_visualisation/network_visualisation.zig" },
        .{ .name = "backend_demo", .path = "experiments/backend_demo/backend_demo.zig" },
        .{ .name = "backend_training", .path = "experiments/backend_training/backend_training.zig" },
        .{ .name = "optimizer_lab", .path = "experiments/optimizer_lab/optimizer_lab.zig" },
        .{ .name = "tokenizer_lab", .path = "experiments/tokenizer_lab/tokenizer_lab.zig" },
        .{ .name = "padding_masks", .path = "experiments/padding_masks/padding_masks.zig" },
        .{ .name = "word2vec", .path = "experiments/word2vec/word2vec.zig" },
        .{ .name = "speech_commands", .path = "experiments/speech_commands/speech_commands.zig" },
        .{ .name = "text_classifier", .path = "experiments/text_classifier/text_classifier.zig" },
        .{ .name = "sequence_tagging", .path = "experiments/sequence_tagging/sequence_tagging.zig" },
        .{ .name = "decoding_lab", .path = "experiments/decoding_lab/decoding_lab.zig" },
        .{ .name = "seq2seq", .path = "experiments/seq2seq/seq2seq.zig" },
        .{ .name = "semantic_search", .path = "experiments/semantic_search/semantic_search.zig" },
        .{ .name = "cnn", .path = "experiments/cnn/cnn.zig" },
        .{ .name = "autoencoder", .path = "experiments/autoencoder/autoencoder.zig" },
        .{ .name = "gru_sequence", .path = "experiments/gru_sequence/gru_sequence.zig" },
        .{ .name = "transformer_encoder", .path = "experiments/transformer_encoder/transformer_encoder.zig" },
        .{ .name = "dqn", .path = "experiments/dqn/dqn.zig" },
        .{ .name = "gpu", .path = "experiments/gpu/gpu.zig" },
        .{ .name = "gpu_benchmark", .path = "experiments/gpu_benchmark/gpu_benchmark.zig" },
        .{ .name = "turboquant", .path = "experiments/quantization/turboquant.zig" },
        .{ .name = "tiny_gpt", .path = "experiments/tiny_gpt/tiny_gpt.zig" },
        .{ .name = "tiny_gpt_openai", .path = "experiments/tiny_gpt/openai_server.zig" },
    }) |experiment| {
        experiment_prev_step = addTestStep(b, acceptance_test_step, experiment.name, experiment.path, experiment_prev_step, target, optimize, build_options, enable_metal, enable_cuda, enable_rocm, cuda_path, rocm_path, lib_mod);
    }
}

fn addBenchmarkStep(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    build_options: *std.Build.Step.Options,
    enable_metal: bool,
    enable_cuda: bool,
    enable_rocm: bool,
    cuda_path: []const u8,
    rocm_path: []const u8,
    benchmark_optimize: std.builtin.OptimizeMode,
    exe_name: []const u8,
    step_name: []const u8,
    description: []const u8,
) void {
    const bench_nn_mod = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = benchmark_optimize,
    });
    bench_nn_mod.addOptions("build_options", build_options);
    addMetalSupport(b, bench_nn_mod, enable_metal, true);
    addCudaSupport(b, bench_nn_mod, enable_cuda, true, cuda_path);
    addRocmSupport(b, bench_nn_mod, enable_rocm, true, rocm_path);

    const bench_mod = b.createModule(.{
        .root_source_file = b.path("benchmarks/benchmark.zig"),
        .target = target,
        .optimize = benchmark_optimize,
    });
    bench_mod.addImport("nn", bench_nn_mod);

    const tiny_gpt_mod = b.createModule(.{
        .root_source_file = b.path("experiments/tiny_gpt/model.zig"),
        .target = target,
        .optimize = benchmark_optimize,
    });
    tiny_gpt_mod.addImport("nn", bench_nn_mod);
    bench_mod.addImport("tiny_gpt", tiny_gpt_mod);

    if (enable_metal) {
        addMetalSupport(b, bench_mod, enable_metal, false);
    }
    if (enable_cuda) {
        addCudaSupport(b, bench_mod, enable_cuda, false, cuda_path);
    }
    if (enable_rocm) {
        addRocmSupport(b, bench_mod, enable_rocm, false, rocm_path);
    }

    const exe = b.addExecutable(.{
        .name = exe_name,
        .root_module = bench_mod,
    });

    const run_cmd = b.addRunArtifact(exe);
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const benchmark_step = b.step(step_name, description);
    benchmark_step.dependOn(&run_cmd.step);
}

fn addMetalSupport(b: *std.Build, mod: *std.Build.Module, enable_metal: bool, include_wrapper: bool) void {
    if (!enable_metal) return;

    mod.linkFramework("Metal", .{});
    mod.linkFramework("MetalPerformanceShaders", .{});
    mod.linkFramework("Foundation", .{});
    mod.linkFramework("CoreGraphics", .{});
    mod.linkSystemLibrary("c", .{});
    mod.linkSystemLibrary("c++", .{});
    mod.linkSystemLibrary("objc", .{});

    if (include_wrapper) {
        mod.addCSourceFile(.{
            .file = b.path("src/metal/metal_wrapper.m"),
            .flags = &.{ "-Wall", "-Wextra", "-fno-objc-arc" },
        });
    }

    mod.addIncludePath(b.path("src/metal"));
}

fn addCudaSupport(b: *std.Build, mod: *std.Build.Module, enable_cuda: bool, include_wrapper: bool, cuda_path: []const u8) void {
    if (!enable_cuda) return;

    const include_path = b.fmt("{s}/include", .{cuda_path});
    const lib_path = b.fmt("{s}/lib64", .{cuda_path});
    const stub_lib_path = b.fmt("{s}/lib64/stubs", .{cuda_path});

    mod.addIncludePath(b.path("src/cuda"));
    mod.addSystemIncludePath(.{ .cwd_relative = include_path });
    mod.addLibraryPath(.{ .cwd_relative = lib_path });
    mod.addLibraryPath(.{ .cwd_relative = stub_lib_path });
    mod.addRPath(.{ .cwd_relative = lib_path });
    mod.linkSystemLibrary("cuda", .{});
    mod.linkSystemLibrary("nvrtc", .{});
    mod.linkSystemLibrary("cublas", .{});
    mod.linkSystemLibrary("c", .{});

    if (include_wrapper) {
        mod.addCSourceFile(.{
            .file = b.path("src/cuda/cuda_wrapper.c"),
            .flags = &.{ "-Wall", "-Wextra" },
        });
    }
}

fn addRocmSupport(b: *std.Build, mod: *std.Build.Module, enable_rocm: bool, include_wrapper: bool, rocm_path: []const u8) void {
    if (!enable_rocm) return;

    const include_path = b.fmt("{s}/include", .{rocm_path});
    const lib_path = b.fmt("{s}/lib", .{rocm_path});

    mod.addIncludePath(b.path("src/rocm"));
    mod.addSystemIncludePath(.{ .cwd_relative = include_path });
    mod.addLibraryPath(.{ .cwd_relative = lib_path });
    mod.addRPath(.{ .cwd_relative = lib_path });
    mod.linkSystemLibrary("amdhip64", .{});
    mod.linkSystemLibrary("hiprtc", .{});
    mod.linkSystemLibrary("rocblas", .{});
    mod.linkSystemLibrary("c", .{});
    mod.linkSystemLibrary("c++", .{});

    if (include_wrapper) {
        mod.addCSourceFile(.{
            .file = b.path("src/rocm/rocm_wrapper.c"),
            .flags = &.{ "-Wall", "-Wextra", "-D__HIP_PLATFORM_AMD__" },
        });
    }
}

// Helper function to create a test step for a specific file
fn addTestStep(
    b: *std.Build,
    main_test_step: *std.Build.Step,
    name: []const u8,
    path: []const u8,
    prev_step: ?*std.Build.Step,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    build_options: *std.Build.Step.Options,
    enable_metal: bool,
    enable_cuda: bool,
    enable_rocm: bool,
    cuda_path: []const u8,
    rocm_path: []const u8,
    nn_mod: ?*std.Build.Module,
) *std.Build.Step {
    const test_mod = b.createModule(.{
        .root_source_file = b.path(path),
        .target = target,
        .optimize = optimize,
    });
    test_mod.addOptions("build_options", build_options);
    if (nn_mod) |module| {
        test_mod.addImport("nn", module);
    }
    addMetalSupport(b, test_mod, enable_metal, nn_mod == null);
    addCudaSupport(b, test_mod, enable_cuda, nn_mod == null, cuda_path);
    addRocmSupport(b, test_mod, enable_rocm, nn_mod == null, rocm_path);

    const test_artifact = b.addTest(.{
        .root_module = test_mod,
    });
    const run_cmd = b.addRunArtifact(test_artifact);

    // Print the test name with a separator for better visibility
    const echo_step = b.addSystemCommand(&[_][]const u8{
        "echo", b.fmt("\n=== Running {s} tests ===", .{name}),
    });

    // Make sure echo runs before the test
    run_cmd.step.dependOn(&echo_step.step);

    // If there's a previous step, make this step depend on it
    // This ensures sequential execution
    if (prev_step) |step| {
        echo_step.step.dependOn(step);
    }

    // Create an individual step for this test
    const test_name = b.fmt("test-{s}", .{name});
    const test_desc = b.fmt("Run {s} tests", .{name});
    const file_test_step = b.step(test_name, test_desc);
    file_test_step.dependOn(&run_cmd.step);

    // Add this test to the main test step
    main_test_step.dependOn(&run_cmd.step);

    return &run_cmd.step;
}
