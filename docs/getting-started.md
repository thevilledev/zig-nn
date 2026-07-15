# Getting Started

This project targets Zig `0.16.0`. The package records that requirement in
`build.zig.zon`.

## Prerequisites

- Zig `0.16.0`
- Go `1.26` or newer, for `nnctl`
- macOS, only if you want to run the Metal backend checks
- Linux with NVIDIA drivers and CUDA toolkit, only if you want to run the CUDA
  backend checks
- Linux with AMD ROCm, HIPRTC, and rocBLAS, only if you want to run the ROCm
  backend checks

The core build has no third-party package dependencies.

## Fast Path

From the repository root:

```bash
zig version
go install ./nnctl/cmd/nnctl
nnctl all
nnctl run quick
```

`nnctl all` builds the library, runs the unit and acceptance tests, and builds
all examples. `nnctl run quick` runs the quick examples that do not require an
external data file or a saved model.

`go install` writes the executable to Go's install bin directory. In a standard
Go setup that directory is already on your `PATH`.

If a GPU toolchain is unavailable on your machine, run individual CPU examples
from [Examples](examples.md) instead of the full `run-examples` target.

## Common nnctl Commands

```bash
nnctl build
nnctl test
nnctl examples
nnctl run quick
nnctl data tiny-gpt
nnctl release
nnctl fmt
nnctl clean
nnctl help
```

You can change the optimization mode used by `nnctl`:

```bash
nnctl build --mode ReleaseFast
nnctl release --mode ReleaseFast
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
zig build test-tensor
zig build test-modules
zig build test-training
zig build test-spatial
zig build test-recurrent
zig build test-reinforcement
zig build test-transformer
zig build test-layer
zig build test-network
zig build test-inference_service
zig build test-visualiser
zig build test-quantization
zig build test-backend
zig build test-cpu_backend
zig build test-metal_backend
zig build test-cuda_backend
zig build test-rocm_backend
```

## Data And Model Files

Most examples are self-contained. Two examples need local files, and one has
optional sourced corpora:

- MNIST expects the dataset files. Use `nnctl data mnist`, then run
  `nnctl run mnist`.
- Serving expects a saved model named `xor_model.bin` by default. Create one
  with `nnctl run xor-training -- --output=xor_model.bin`, then run
  `nnctl run serving`.
- Tiny GPT falls back to the checked-in toy corpus by default. To use sourced
  Tiny Shakespeare or TinyStories corpora, run `nnctl data tiny-gpt`.

See [Examples](examples.md) for the full example list.
