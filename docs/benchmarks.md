# Benchmarks

The benchmark suite is meant for optimization work, not pass/fail testing. It
prints CSV with fixed input shapes, fixed seeds, warmups excluded from timing,
and a checksum so obviously-elided work is visible.

## Commands

Run the human-readable `nnctl` report:

```bash
./bin/nnctl benchmark
```

That command runs the ReleaseFast benchmark with Metal enabled by default,
groups results by suite, and shows CPU vs Metal timing deltas where both
backends are available. Use `--csv` to print the raw benchmark CSV.

Run the default release-mode suite. The `benchmark` step pins the benchmark
binary and its `nn` module import to `ReleaseFast` so timings are not
accidentally collected from a Debug build:

```bash
zig build benchmark
```

Run the same suite with Metal backend cases enabled on macOS:

```bash
zig build benchmark -Dgpu=metal
```

Run Debug-mode timings for comparison:

```bash
zig build benchmark-debug
```

Run a fast smoke benchmark:

```bash
zig build benchmark -- --quick
```

Run one suite:

```bash
zig build benchmark -- --filter matmul
zig build benchmark -- --filter activation
zig build benchmark -- --filter training
zig build benchmark -- --filter tiny_gpt
zig build benchmark -- --filter quantization
```

## Coverage

- `matmul`: backend-aware `BackendMatrix.dotProduct` on CPU and Metal.
- `activation`: backend-aware ReLU, tanh, Swish, softmax, GLU, and SwiGLU on
  CPU and Metal.
- `training`: CPU `Network.trainBatch` loops. The current training path uses
  the learning-oriented `Matrix`, so Metal rows are marked `skipped_cpu_only_path`.
- `tiny_gpt`: CPU TinyGPT forward passes through the example model. TinyGPT also
  uses the learning-oriented `Matrix` path today.
- `quantization`: CPU uniform scalar quantization and TurboQuant encode,
  decode, and error measurement.

Metal rows include `sample_error`, sampled against the CPU result. The Metal
kernels currently use f32 buffers, so small non-zero errors are expected.
