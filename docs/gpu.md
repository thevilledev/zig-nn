# GPU And Backend Notes

The project currently has a CPU-first learning path and a separate
backend-aware matrix path.

## Current Backend Boundary

- `src/matrix.zig` powers the high-level `Network`, `Layer`, training examples,
  and default inference helpers.
- `src/backend.zig`, `src/cpu_backend.zig`, `src/metal_backend.zig`, and
  `src/cuda_backend.zig` power the backend-aware `BackendMatrix` API.
- Metal support is available for backend matrix operations on macOS.
- CUDA support is available for backend matrix operations on Linux with the
  NVIDIA driver, CUDA toolkit headers, and NVRTC libraries available.
- `Network.forwardBackend` and `Network.predictBackend` can run inference with
  `BackendMatrix` inputs through the selected backend.
- `Network.trainBatchBackend` and `Network.trainBackend` can train with
  `BackendMatrix` inputs and targets through the selected backend.
- `BackendNetwork` snapshots can run repeated backend inference without
  re-copying CPU parameters on every forward pass.
- The default `Network.trainBatch` and `Network.train` methods still use the
  CPU `Matrix` path.
- Backend training currently mirrors CPU-owned network parameters into backend
  matrices during training calls. Persistent backend parameters remain future
  performance work.

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

The broader benchmark suite emits `cpu`, `metal`, and `cuda` rows. On machines
without a particular GPU backend, that backend is reported as skipped.

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
availability, and the exact Metal/CUDA verification commands for the current
machine.

Use `BackendMatrix` when testing backend behavior directly. Use
`Network.forwardBackend`, `Network.predictBackend`, `BackendNetwork`,
`Network.trainBatchBackend`, or `Network.trainBackend` when testing
backend-aware network execution. Use `Matrix`, `Network.forward`, and the
default training methods when working on the learning-oriented CPU path.

The next major bridge is persistent backend parameters for training without
losing the readability that makes the project useful as a learning repo.
