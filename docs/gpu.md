# GPU And Backend Notes

The project currently has a CPU-first learning path and a separate
backend-aware matrix path.

## Current Backend Boundary

- `src/matrix.zig` powers the high-level `Network`, `Layer`, training examples,
  and default inference helpers.
- `src/backend.zig`, `src/cpu_backend.zig`, and `src/metal_backend.zig` power
  the backend-aware `BackendMatrix` API.
- Metal support is available for backend matrix operations on macOS.
- `Network.forwardBackend` and `Network.predictBackend` can run inference with
  `BackendMatrix` inputs through the selected backend.
- The high-level training and backpropagation path does not yet use
  `BackendMatrix`.
- CUDA is present as an enum/build option, but the library implementation uses
  the CPU backend for CUDA-tagged work today. Do not expect CUDA acceleration
  yet.

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

## CPU Verification

On non-macOS targets, or when GPU support is not enabled, use the normal CPU
test paths:

```bash
zig build test --summary all
zig build test-acceptance --summary all
```

If Metal is requested on a platform where it is unavailable, backend creation
falls back to CPU.

## Working On Backends

Use `BackendMatrix` when testing backend behavior directly. Use
`Network.forwardBackend` or `Network.predictBackend` when testing backend-aware
inference. Use `Matrix`, `Network.forward`, and training methods when working
on the learning-oriented training path.

The next major bridge is moving from inference-only backend execution to
persistent backend parameters and backend-aware training without losing the
readability that makes the project useful as a learning repo.
