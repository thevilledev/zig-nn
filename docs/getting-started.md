# Getting Started

This project targets Zig `0.16.0`. The package records that requirement in
`build.zig.zon`.

## Prerequisites

- Zig `0.16.0`
- Go `1.26` or newer, for `nnctl`
- macOS, only if you want to run the Metal backend checks

The core build has no third-party package dependencies.

## Fast Path

From the repository root:

```bash
zig version
mkdir -p bin
go build -C nnctl -o ../bin/nnctl ./cmd/nnctl
./bin/nnctl all
./bin/nnctl run quick
```

`nnctl all` builds the library, runs the unit and acceptance tests, and builds
all examples. `nnctl run quick` runs the quick examples that do not require an
external data file or a saved model.

If a GPU toolchain is unavailable on your machine, run individual CPU examples
from [Examples](examples.md) instead of the full `run-examples` target.

## Common nnctl Commands

```bash
./bin/nnctl build
./bin/nnctl test
./bin/nnctl examples
./bin/nnctl run quick
./bin/nnctl data tiny-gpt
./bin/nnctl release
./bin/nnctl fmt
./bin/nnctl clean
./bin/nnctl help
```

You can change the optimization mode used by `nnctl`:

```bash
./bin/nnctl build --mode ReleaseFast
./bin/nnctl release --mode ReleaseFast
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

- MNIST expects the dataset files. Use `./bin/nnctl data mnist`, then run
  `./bin/nnctl run mnist`.
- Serving expects a saved model named `xor_model.bin` by default. Create one
  with `./bin/nnctl run xor-training -- --output=xor_model.bin`, then run
  `./bin/nnctl run serving`.
- Tiny GPT falls back to the checked-in toy corpus by default. To use sourced
  Tiny Shakespeare or TinyStories corpora, run `./bin/nnctl data tiny-gpt`.

See [Examples](examples.md) for the full example list.
