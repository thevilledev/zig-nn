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
all experiments. `nnctl run quick` runs the quick experiments that do not require an
external data file or a saved model.

`go install` writes the executable to Go's install bin directory. In a standard
Go setup that directory is already on your `PATH`.

If a GPU toolchain is unavailable, the CPU experiments still work and `auto`
may fall back to CPU. Choose an individual program from the
[experiment guide](../experiments/README.md); explicit `metal`, `cuda`, and
`rocm` requests require the corresponding toolchain.

## Common nnctl Commands

```bash
nnctl build
nnctl test
nnctl experiments
nnctl run quick
nnctl data tiny-gpt
nnctl data speech-commands
nnctl train speech-commands --output speech-commands.bin
nnctl release
nnctl fmt
nnctl clean
nnctl help
```

## Real-Time Learning Lab

The browser lab builds the Svelte frontend and starts the local `nnctl` server:

```bash
mise run lab
```

Open the printed `http://127.0.0.1:8091` URL. Runs execute as native Zig
processes, so the UI is local-only and uses the same implementation as the
terminal experiments.

For frontend development, run the API and Vite in separate terminals:

```bash
mise run web:install
mise run lab:api
cd web && npm run dev
```

Vite proxies `/api` to port 8091. See [Real-Time Learning Lab](learning-lab.md)
for the supported controls and event format.

To deploy a Verda CUDA worker from the same UI, configure the `nnctl/verda`
keyring credentials described in the learning-lab guide and start the server
with `nnctl lab --cloud`. Cloud control remains loopback-only, uploads committed
`HEAD`, streams the native experiment protocol over SSH, and defaults to
destroying the worker after the run.

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
zig build experiments
```

The former `nnctl examples` and `zig build examples` spellings remain as
compatibility aliases.

Individual test steps are also available:

```bash
zig build test-matrix
zig build test-activation
zig build test-tensor
zig build test-text
zig build test-embeddings
zig build test-structured
zig build test-decoding
zig build test-retrieval
zig build test-audio
zig build test-spectral
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

Most experiments are self-contained. Some workflows use local data or checkpoints:

- MNIST expects the dataset files. Use `nnctl data mnist`, then run
  `nnctl run mnist`.
- Serving expects a saved model named `xor_model.bin` by default. Create one
  with `nnctl run xor-training -- --output=xor_model.bin`, then run
  `nnctl run serving`.
- Tiny GPT falls back to the checked-in toy corpus by default. To use sourced
  Tiny Shakespeare or TinyStories corpora, run `nnctl data tiny-gpt`.
- Speech Commands needs the external Mini Speech Commands WAV dataset and a
  locally trained checkpoint. Run `nnctl data speech-commands`, then
  `nnctl train speech-commands --output speech-commands.bin`.

See [Experiments](../experiments/README.md) for the complete catalog and the
evidence each program is designed to expose.
