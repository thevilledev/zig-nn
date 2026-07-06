# Neural Network in Zig

[![Work in Progress](https://img.shields.io/badge/Status-Work%20in%20Progress-yellow)](https://github.com/thevilledev/zig-nn)

A small neural-network playground written in Zig.

This is not trying to be a production ML framework. The point is to make the
mechanics visible: matrix operations, layers, training loops, backend
boundaries, and small experiments that can be checked against real output.

## What This Repo Is For

- Learning neural-network fundamentals by implementing the pieces directly
- Practicing Zig memory management, error handling, and build tooling
- Running small examples that expose the math instead of hiding it
- Trying focused research ideas and comparing them with measured results
- Exploring a CPU-first implementation with a separate backend-aware matrix path

## Current Shape

The repo has two matrix paths:

- `Matrix` is the learning-oriented path used by the default `Network`,
  `Layer`, examples, and training code.
- `BackendMatrix` is the backend-aware path that can run operations through CPU,
  Metal, or CUDA implementations.

Metal and CUDA support currently applies to backend matrix operations, GPU
examples, backend-aware `Network.forwardBackend` / `Network.predictBackend`
inference, and `Network.trainBatchBackend` / `Network.trainBackend` training.
`BackendNetwork` snapshots provide persistent backend inference parameters, and
`BackendTrainer` provides persistent backend training parameters for standard
and gated network layers. `Network.trainBatchBackend` / `Network.trainBackend`
remain available as CPU-owned backend training paths.

## Start Here

Make sure Zig `0.16.0` and Go `1.26` or newer are installed. From the
repository root, install `nnctl` once, then use it for repo tasks.

```bash
# Install nnctl
go install ./nnctl/cmd/nnctl
nnctl help

# Build, test, and build examples
nnctl all

# Run specific examples
nnctl run simple-xor
nnctl run tiny-gpt
nnctl run backend-training

# Download data for examples
nnctl data mnist
nnctl data tiny-gpt

# Train and serve a TinyGPT checkpoint
nnctl train tiny-gpt --preset coherent-small --output tiny-gpt.bin
nnctl chat --model tiny-gpt.bin

# Build with a different optimization mode
nnctl build --mode ReleaseFast

# Run readable benchmark comparisons
nnctl benchmark
```

For setup details and direct `zig build` commands, see
[Getting Started](docs/getting-started.md).

## Guide

- [Getting Started](docs/getting-started.md) - prerequisites, build commands,
  tests, and common development tasks
- [Examples](docs/examples.md) - runnable demos and what each one is meant to
  show
- [Benchmarks](docs/benchmarks.md) - repeatable CPU, GPU, and release-mode
  benchmark commands
- [Experiments](docs/experiments.md) - research-style probes, metrics, and
  current experiment notes
- [GPU and Backend Notes](docs/gpu.md) - current backend boundaries, Metal and
  CUDA verification
- [Neural Network Architecture](docs/architecture.md) - design principles and
  component overview
- [Advanced Activation Functions](docs/activation_functions.md) - Swish, GLU,
  SwiGLU, and implementation notes
- [Research Resources](docs/research.md) - papers, official docs, datasets,
  and implementation references for topics covered by the repo

## Repository Map

- `src/` - reusable library code
- `examples/` - runnable programs and acceptance-style example tests
- `docs/` - guides and architecture notes
- `nnctl/` - Go CLI for repository tasks
- `build.zig` - Zig build graph, test steps, examples, and backend options

## License

MIT License
