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
nnctl model inspect CHECKPOINT
nnctl predict --model xor_model.bin --input '[0,1]' --gpu auto
nnctl serve --model CHECKPOINT --gpu auto
nnctl release
nnctl fmt
nnctl clean
nnctl help
```

## Inference From A Checkpoint

`nnctl` recognizes existing ZNN network and TGPT TinyGPT magic headers. Inspect
a checkpoint before running it:

```bash
nnctl model inspect xor_model.bin
nnctl model inspect tiny-gpt.bin --json
```

Dense checkpoints accept a flat JSON array plus an optional batch size. The
session validates that each batch has the model's input width:

```bash
nnctl predict --model xor_model.bin --input '[0,1]' --gpu auto
nnctl predict --model dense.bin --input '[0,1,1,0]' --batch-size 2
```

`--input` accepts the complete flat JSON array; its length must equal
`batch-size * model-input-size`.

Serve either checkpoint type with one command:

```bash
nnctl serve --model xor_model.bin --gpu auto
nnctl serve --model tiny-gpt.bin --gpu auto
```

`--gpu auto` is the only selection allowed to fall back to CPU. `--gpu none`
requires CPU, while `metal`, `cuda`, and `rocm` fail if the selected backend is
not compiled or available. `nnctl chat --model tiny-gpt.bin --gpu auto` starts
the same TinyGPT serving path with the local chat client.

TinyGPT `stream: true` requests emit an SSE event for each generated character,
then a finish event and `[DONE]`. The educational v1 server processes one
generation at a time, caps request bodies at 64 KiB and `max_tokens` at 4096,
and stops generation when the streaming client disconnects.

## Cloud GPU Workers

Cloud commands default to Verda for compatibility. Use `--provider` to select
another adapter:

```bash
nnctl cloud pricing --provider verda --single-gpu
nnctl cloud pricing --provider digitalocean --single-gpu
nnctl cloud ssh-keys --provider digitalocean
nnctl cloud deploy \
  --provider digitalocean \
  --instance-type gpu-mi300x1-192gb \
  --location-code tor1 \
  --ssh-key-id <digitalocean-ssh-key-id> \
  --dry-run --json
```

Remove `--dry-run` to create the GPU Droplet, then use the same provider when
listing or destroying it. DigitalOcean deployments require an explicit account
SSH key ID; `nnctl` never attaches every account key implicitly.

```bash
nnctl cloud list --provider digitalocean
nnctl cloud destroy --provider digitalocean <droplet-id>
```

See [Benchmarks](benchmarks.md) for the managed one-command benchmark workflow
and [Real-Time Learning Lab](learning-lab.md) for keyring setup and
contract-only GPU offerings.

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

To deploy a cloud worker from the same UI, configure the provider keyring
credentials described in the learning-lab guide. Verda remains the default and
keeps its compatibility shorthand:

```bash
nnctl lab --cloud
```

Enable Verda and DigitalOcean together with explicit provider flags. The
DigitalOcean lab also requires an account SSH key ID:

```bash
nnctl lab \
  --cloud-provider verda \
  --cloud-provider digitalocean \
  --cloud-ssh-key digitalocean:<digitalocean-ssh-key-id>
```

Cloud control remains loopback-only, uploads committed `HEAD`, streams the
native experiment protocol over SSH, and defaults to a reusable worker that
requires explicit cleanup. DigitalOcean AMD GPU workers expose ROCm, while
Verda and DigitalOcean NVIDIA workers expose CUDA.

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
zig build test-inference
zig build test-tiny_gpt_model
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
- Dense serving uses a ZNN checkpoint. Create one with
  `nnctl run xor-training -- --output=xor_model.bin`, then run
  `nnctl serve --model xor_model.bin`.
- Tiny GPT falls back to the checked-in toy corpus by default. To use sourced
  Tiny Shakespeare or TinyStories corpora, run `nnctl data tiny-gpt`. Train a
  TGPT checkpoint with `nnctl train tiny-gpt --output tiny-gpt.bin`, inspect it,
  then serve it with `nnctl serve --model tiny-gpt.bin`.
- Speech Commands needs the external Mini Speech Commands WAV dataset and a
  locally trained checkpoint. Run `nnctl data speech-commands`, then
  `nnctl train speech-commands --output speech-commands.bin`.

The MNIST and Mini Speech Commands downloads use pinned HTTPS sources, enforce
compressed and extracted size limits, and verify exact SHA-256 digests before
installing files.

See [Experiments](../experiments/README.md) for the complete catalog and the
evidence each program is designed to expose.
