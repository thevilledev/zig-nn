# GPU And Backend Notes

The project currently has a CPU-first learning path and a separate
backend-aware matrix path.

## Current Backend Boundary

- `src/matrix.zig` powers the high-level `Network`, `Layer`, training examples,
  and default inference helpers.
- `src/backend.zig`, `src/cpu_backend.zig`, `src/metal_backend.zig`,
  `src/cuda_backend.zig`, and `src/rocm_backend.zig` power the backend-aware
  `BackendMatrix` API.
- Metal support is available for backend matrix operations on macOS.
- CUDA support is available for backend matrix operations on Linux with the
  NVIDIA driver, CUDA toolkit headers, and NVRTC libraries available.
- ROCm support is available for backend matrix operations on Linux with AMD
  ROCm HIP, HIPRTC, and rocBLAS libraries available.
- `Network.forwardBackend` and `Network.predictBackend` can run inference with
  `BackendMatrix` inputs through the selected backend.
- `Network.trainBatchBackend` and `Network.trainBackend` can train with
  `BackendMatrix` inputs and targets through the selected backend.
- `BackendNetwork` snapshots can run repeated backend inference without
  re-copying CPU parameters on every forward pass.
- `BackendTrainer` can train standard and gated networks while keeping
  trainable parameters on the backend across batches.
- `BackendTrainer` uses plain SGD by default and can keep momentum optimizer
  state on the backend through `Network.backendTrainerWithOptimizer`.
- The default `Network.trainBatch` and `Network.train` methods still use the
  CPU `Matrix` path.
- `Network.trainBatchBackend` and `Network.trainBackend` remain available for
  direct CPU-owned backend training calls.
- `Device`, `Tensor`, and `ExecutionContext` provide the backend-neutral f32
  runtime for rank-aware model code. Explicit `metal`, `cuda`, and `rocm`
  selections fail when unavailable; only `auto` may fall back to CPU.
- `Modules.Mlp` and `Training.Optimizer` keep forward/backward tensors,
  optimizer moments, gradient accumulation, clipping, and parameter updates on
  the selected backend.
- The autoencoder, optimizer lab, GRU sequence, and Transformer encoder lessons
  use that device path. GRU backpropagation and unmasked encoder attention are
  composed from the same tensor primitives and run on each enabled backend.
- `Spatial.Conv2d` and max pooling are intentionally CPU-first reference
  implementations. Their matching CNN lesson prioritizes readable indexing and
  gradients over accelerator kernels.
- DQN runs its MLP updates on the selected backend but intentionally reads Q
  values to the host for environment actions and Bellman-target construction.
  The example reports this boundary instead of presenting it as device-resident.
- `Transformer.Decoder` keeps embeddings, decoder blocks, optimizer updates,
  and KV caches on one backend. Linear/GELU/layer-normalization, causal
  attention, their backward passes, and embedding gradients have native Metal
  CUDA, and ROCm kernels.
- `ExecutionContext.beginBatch` and `endBatch` form an explicit synchronization
  boundary. Runtime telemetry reports allocations, transfers, kernels, vendor
  GEMMs, and synchronizations so tests can catch accidental host round trips.

## TinyGPT On A Device

The existing checkpoint and server format remains CPU `Matrix`-based. TinyGPT
splits its combined QKV checkpoint weights into the public device decoder, runs
the selected path, then merges weights back before saving or serving.

Run full-model SGD training on the automatically selected backend:

```bash
zig build run_tiny_gpt -Dgpu=auto -- \
  --train-full --backend auto --optimizer sgd --full-batch-size 1 \
  --weight-decay 0 --no-corpus-prior
```

Use `--backend metal`, `--backend cuda`, or `--backend rocm` to require that
exact accelerator. Device training currently supports SGD, one context window
per update, and no weight decay. AdamW, gradient accumulation, and multi-window
device batches remain on the CPU educational trainer for now.

With `--no-corpus-prior`, backend-selected generation uses per-layer KV caches.
The cache appends projected keys and values in place and replays the current
window only when context rollover is required.

## Metal Verification

On macOS, run:

```bash
zig build -Dgpu=metal test-metal_backend --summary all
zig build -Dgpu=metal run_gpu
```

To compare Metal against CPU matrix multiplication:

```bash
zig build run_gpu_benchmark -Dgpu=metal -Doptimize=ReleaseFast
```

The benchmark prints CPU timing, Metal timing, speedup, and a small sampled
error check.

## CUDA Verification

On Linux with an NVIDIA GPU and CUDA toolkit, run:

```bash
zig build -Dgpu=cuda test-cuda_backend --summary all
zig build -Dgpu=cuda run_gpu
```

If CUDA is installed somewhere other than `/usr/local/cuda`, pass the toolkit
root explicitly:

```bash
zig build -Dgpu=cuda -Dcuda-path=/opt/cuda test-cuda_backend --summary all
```

To compare CUDA against CPU matrix multiplication:

```bash
zig build run_gpu_benchmark -Dgpu=cuda -Doptimize=ReleaseFast
```

## ROCm Verification

On Linux with an AMD GPU and ROCm toolkit, run:

```bash
zig build -Dgpu=rocm test-rocm_backend --summary all
zig build -Dgpu=rocm run_gpu
```

If ROCm is installed somewhere other than `/opt/rocm`, pass the toolkit root
explicitly:

```bash
zig build -Dgpu=rocm -Drocm-path=/path/to/rocm test-rocm_backend --summary all
```

To compare ROCm against CPU matrix multiplication:

```bash
zig build run_gpu_benchmark -Dgpu=rocm -Doptimize=ReleaseFast
```

The broader benchmark suite emits `cpu`, `metal`, `cuda`, and `rocm` rows. On
machines without a particular GPU backend, that backend is reported as skipped.

## CPU Verification

On non-macOS targets, or when GPU support is not enabled, use the normal CPU
test paths:

```bash
zig build test --summary all
zig build test-acceptance --summary all
```

If a GPU backend is requested on a platform where it is unavailable, generic
backend creation falls back to CPU.

## Working On Backends

Use `nnctl doctor` to print local tool versions, platform-specific backend
availability, and the exact Metal/CUDA/ROCm verification commands for the
current machine.

Use `BackendMatrix` when testing backend behavior directly. Use
`Network.forwardBackend`, `Network.predictBackend`, `BackendNetwork`,
`BackendTrainer`, `Network.backendTrainerWithOptimizer`,
`Network.trainBatchBackend`, or `Network.trainBackend` when testing
backend-aware network execution. Use `Matrix`, `Network.forward`, and the
default training methods when working on the learning-oriented CPU path.

The next backend optimization targets are mixed precision, batched sequences,
and tiled/reduction-optimized attention and spatial kernels.
The current kernels prioritize explicit, testable semantics over peak hardware
throughput.
