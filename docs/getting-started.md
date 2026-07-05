# Getting Started

This project targets Zig `0.16.0`. The package records that requirement in
`build.zig.zon`.

## Prerequisites

- Zig `0.16.0`
- `make`, if you want the convenience commands
- macOS, only if you want to run the Metal backend checks

The core build has no third-party package dependencies.

## Fast Path

From the repository root:

```bash
zig version
make
make run-examples
```

`make` builds the library, runs the unit and acceptance tests, and builds all
examples. `make run-examples` runs the quick examples that do not require an
external data file or a saved model.

If a GPU toolchain is unavailable on your machine, run individual CPU examples
from [Examples](examples.md) instead of the full `run-examples` target.

## Common Make Targets

```bash
make build
make test
make examples
make run-examples
make prepare-tiny-gpt-data
make release
make format
make clean
make help
```

You can change the optimization mode used by the Makefile:

```bash
make BUILD_MODE=ReleaseFast
make release
```

## Direct Zig Commands

```bash
zig build
zig build test
zig build test-acceptance
zig build examples
```

Individual test steps are also available:

```bash
zig build test-matrix
zig build test-activation
zig build test-layer
zig build test-network
zig build test-inference_service
zig build test-visualiser
zig build test-quantization
zig build test-backend
zig build test-cpu_backend
zig build test-metal_backend
```

## Data And Model Files

Most examples are self-contained. Two examples need local files, and one has
optional sourced corpora:

- MNIST expects the dataset files. Use `scripts/download_mnist.sh`, then run
  `zig build run_mnist`.
- Serving expects a saved model named `xor_model.bin` by default. Create one
  with `zig build run_xor_training -- --output=xor_model.bin`, then run
  `zig build run_serving`.
- Tiny GPT falls back to the checked-in toy corpus by default. To use sourced
  Tiny Shakespeare or TinyStories corpora, run `make prepare-tiny-gpt-data`.

See [Examples](examples.md) for the full example list.
