# Benchmarks

The benchmark suite is meant for optimization work, not pass/fail testing. It
prints CSV with fixed input shapes, fixed seeds, warmups excluded from timing,
and a checksum so obviously-elided work is visible.

## Commands

Run the human-readable `nnctl` report:

```bash
./bin/nnctl benchmark
```

That command runs the ReleaseFast benchmark with GPU auto-detection enabled,
groups results by suite, and shows CPU vs Metal and CPU vs CUDA timing deltas
where those backends are available. Use `--csv` to print the raw benchmark CSV.

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

Run the same suite with CUDA backend cases enabled on Linux:

```bash
zig build benchmark -Dgpu=cuda
```

Run Debug-mode timings for comparison:

```bash
zig build benchmark-debug
```

Run a fast smoke benchmark:

```bash
zig build benchmark -- --quick
```

Save a raw `nnctl` CSV baseline and compare a later run against it:

```bash
nnctl benchmark --quick --csv > baseline.csv
nnctl benchmark --quick --compare baseline.csv
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

- `matmul`: backend-aware `BackendMatrix.dotProduct` on CPU, Metal, and CUDA.
- `activation`: backend-aware ReLU, tanh, Swish, softmax, GLU, and SwiGLU on
  CPU, Metal, and CUDA.
- `training`: CPU `Network.trainBatch` loops plus `Network.trainBatchBackend`
  rows for Metal and CUDA when those backends are available.
- `tiny_gpt`: CPU TinyGPT forward passes through the example model. TinyGPT also
  uses the learning-oriented `Matrix` path today.
- `quantization`: CPU uniform scalar quantization and TurboQuant encode,
  decode, and error measurement.

GPU rows include `sample_error`, sampled against the CPU result. The Metal and
CUDA kernels currently use f32 buffers, so small non-zero errors are expected.
