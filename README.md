# Neural Network in Zig

[![Work in Progress](https://img.shields.io/badge/Status-Work%20in%20Progress-yellow)](https://github.com/thevilledev/zig-nn)

A minimalistic neural network implementation in Zig for learning purposes.
This project focuses on implementing the pieces directly enough to inspect
the math, run examples, and compare small experiments against actual output.

## Features

- Matrix operations from scratch
- Common activation functions (Sigmoid, Linear, ReLU, Tanh)
- Advanced activation functions (Swish, GLU, SwiGLU)
- Basic feed-forward neural network architecture
- Support for gated architectures used in modern transformer models
- Mean squared error, cross entropy, and binary cross entropy losses
- Mini-batch training, model save/load, and inference service helpers
- CPU backend plus a Metal backend for backend matrix operations
- Reproducible experiments for testing research ideas in this repo

See [examples](examples/).

## Current Architecture

There are two matrix paths:

- `Matrix` powers the learning-oriented `Network`, `Layer`, examples, and
  training code.
- `BackendMatrix` powers the backend abstraction and can run operations on
  CPU or Metal.

Metal support is available for backend matrix operations and GPU examples.
The higher-level `Network` training and inference path is still CPU-only.
CUDA is present as an enum/build option, but currently falls back to CPU.

## Documentation

Detailed documentation is available in the `docs` directory:
- [Neural Network Architecture](docs/architecture.md) - Design principles and implementation details
- [Advanced Activation Functions](docs/activation_functions.md) - Detailed information about activation functions

Research experiments live beside the examples they run. The first one is:

- [TurboQuant](examples/quantization/README.md) - vector quantization metrics
  and a rotated scalar quantization experiment

## Building

Make sure you have Zig `0.16.0` installed on your system. This project is developed with the latest stable version of Zig.

For convenience, a Makefile is provided with common operations:

```bash
# Build, test, and build examples
make

# Run specific examples, for example:
make example-simple-xor
make example-turboquant

# Build with different optimization modes
make BUILD_MODE=ReleaseFast
make release  # Build with ReleaseSafe mode

# See all available commands
make help
```

## Testing

The project includes a comprehensive test suite to verify the functionality of all components. The build system is configured to run tests for each module sequentially, making it easy to identify which component has issues.

```bash
# Run all tests
make test

# Run tests for specific components, for example:
zig build test-matrix     # Run matrix operation tests
zig build test-activation # Run activation function tests
zig build test-layer      # Run neural network layer tests
zig build test-network    # Run full network tests
zig build test-quantization # Run quantization tests
```

## GPU / Metal Verification

On macOS, run:

```bash
zig build -Dgpu=metal test-metal_backend --summary all
zig build -Dgpu=metal run_gpu
```

On non-macOS targets, Metal tests are skipped or unavailable. The default
test suite still verifies CPU behavior:

```bash
zig build test --summary all
zig build test-acceptance --summary all
```

## Experiments

Experiments are small, reproducible programs that turn an idea into code and
metrics. Notes live with the runnable example, and reusable code belongs in
`src/`.

Run the TurboQuant lab with:

```bash
zig build run_turboquant
```

## Learning Goals

This project serves as a learning exercise for:

- Understanding neural network fundamentals
- Implementing mathematical operations in Zig
- Working with Zig's memory management and error handling
- Reading papers and checking ideas against measured results

## License

MIT License 
