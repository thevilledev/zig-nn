# Neural Network in Zig

[![Work in Progress](https://img.shields.io/badge/Status-Work%20in%20Progress-yellow)](https://github.com/thevilledev/zig-nn)

A small neural-network playground written in Zig.

This is for learning neural-network fundamentals by building the pieces
directly: matrix math, layers, training loops, small examples, focused
experiments, and a CPU-first path with separate backend-aware matrix operations
plus a device-resident f32 tensor and Transformer path for CPU, Metal, CUDA,
and ROCm.
It is not trying to be a production ML framework.

## Start Here

Install Zig `0.16.0` and Go `1.26` or newer, then from the repository root:

```bash
go install ./nnctl/cmd/nnctl
nnctl all
nnctl run quick
```

`nnctl all` builds the library, runs the tests, and builds the examples.
`nnctl run quick` runs the small CPU examples that do not need downloaded data
or a saved model.

Try a specific example or workflow:

```bash
nnctl run simple-xor
nnctl run tiny-gpt
nnctl run backend-training
nnctl data mnist
nnctl train tiny-gpt --preset coherent-small --output tiny-gpt.bin
nnctl chat --model tiny-gpt.bin
nnctl benchmark
```

For direct `zig build` commands and data files, see
[Getting Started](docs/getting-started.md). For local hooks and full-repository
checks, see [Development Environment](docs/development.md). GPU details are in
[GPU and Backend Notes](docs/gpu.md).

## Guide

- [Getting Started](docs/getting-started.md) - prerequisites, build commands,
  tests, and common development tasks
- [Development Environment](docs/development.md) - pinned tools, local hooks,
  and full-repository checks
- [Examples](docs/examples.md) - runnable demos and what each one is meant to
  show
- [Benchmarks](docs/benchmarks.md) - repeatable CPU, GPU, and release-mode
  benchmark commands
- [Experiments](docs/experiments.md) - research-style probes, metrics, and
  current experiment notes
- [GPU and Backend Notes](docs/gpu.md) - current backend boundaries, Metal,
  CUDA, and ROCm verification
- [Neural Network Architecture](docs/architecture.md) - design principles and
  component overview
- [Advanced Activation Functions](docs/activation_functions.md) - Swish, GLU,
  SwiGLU, and implementation notes
- [Research Resources](docs/research.md) - papers, official docs, datasets,
  and implementation references for topics covered by the repo

## Where To Look

Start with `examples/` when you want to see the code run. Read `src/matrix.zig`,
`src/layer.zig`, and `src/network.zig` for the core learning path. Use
[GPU and Backend Notes](docs/gpu.md) when you need the current backend
boundaries, and `nnctl/` only when you are changing the helper CLI.

## License

MIT License
