# Benchmarks

The benchmark suite is meant for optimization work, not pass/fail testing. It
prints CSV with fixed input shapes, fixed seeds, warmups excluded from timing,
and a checksum so obviously-elided work is visible.

## Commands

Run the human-readable `nnctl` report:

```bash
./bin/nnctl benchmark
```

That command runs the ReleaseFast benchmark with GPU auto-detection enabled,
groups results by suite, and shows CPU vs Metal, CUDA, and ROCm timing deltas
where those backends are available. Use `--csv` to print the raw benchmark CSV.

Run the default release-mode suite. The `benchmark` step pins the benchmark
binary and its `nn` module import to `ReleaseFast` so timings are not
accidentally collected from a Debug build:

```bash
zig build benchmark
```

Run the same suite with Metal backend cases enabled on macOS:

```bash
zig build benchmark -Dgpu=metal
```

Run the same suite with CUDA backend cases enabled on Linux:

```bash
zig build benchmark -Dgpu=cuda
```

Run the same suite with ROCm backend cases enabled on Linux:

```bash
zig build benchmark -Dgpu=rocm
```

Run Debug-mode timings for comparison:

```bash
zig build benchmark-debug
```

Run a fast smoke benchmark:

```bash
zig build benchmark -- --quick
```

The default suite includes heavier workloads intended to move more rows into
millisecond-scale timings, plus a GPU-only `gpu_heavy` suite with larger matrix
and activation shapes. Use `--quick` when you only need compilation and basic
benchmark plumbing coverage; the GPU-heavy rows are skipped in quick mode.
The `gpu_peak` suite is opt-in and focuses on huge GEMM workloads for high-end
GPUs; it is skipped unless selected with `--filter gpu_peak`.

Run the high-end GPU peak suite:

```bash
zig build benchmark -Dgpu=cuda -- --filter gpu_peak
zig build benchmark -Dgpu=rocm -- --filter gpu_peak
```

Save a raw `nnctl` CSV baseline and compare a later run against it:

```bash
./bin/nnctl benchmark --quick --csv > baseline.csv
./bin/nnctl benchmark --quick --compare baseline.csv
```

Deploy a clean git snapshot to a remote benchmark host:

```bash
./bin/nnctl deploy user@gpu1:/srv/zig-nn --delete
ssh user@gpu1 'cd /srv/zig-nn && zig build benchmark -Dgpu=cuda'
```

`nnctl deploy` uses `git archive` to materialize the selected ref in a temporary
directory, then rsyncs that snapshot to the target. Local build artifacts,
dependency folders, and `.git` metadata are not copied. The default ref is
`HEAD`; pass `--ref <tree-ish>` for another branch, tag, or commit. The command
refuses to run with uncommitted changes unless `--ignore-dirty` is set, so
benchmark runs are easy to tie back to a reproducible source snapshot.

Run one suite:

```bash
zig build benchmark -- --filter matmul
zig build benchmark -- --filter activation
zig build benchmark -- --filter gpu_heavy
zig build benchmark -- --filter gpu_peak
zig build benchmark -- --filter layer_norm
zig build benchmark -- --filter training
zig build benchmark -- --filter tiny_gpt
zig build benchmark -- --filter quantization
```

## Remote GPU Benchmark Workflow

Use this process when adding a new GPU result to the comparison table. Keep each
run tied to a reproducible source snapshot, deploy only one spot worker at a
time, and destroy the worker plus any cloned OS volume as soon as the benchmark
is captured.

### NVIDIA Verda Spot Workers

Use `nnctl cloud deploy` for NVIDIA CUDA benchmarks.

1. List currently available single-GPU spot instance types and pick one GPU
   model that is not already represented in the results table:

   ```bash
   ./bin/nnctl cloud pricing --single-gpu
   ```

2. Deploy one spot worker. Without a source OS volume, `nnctl` boots the
   requested Ubuntu image and applies the embedded Verda bootstrap script:

   ```bash
   ./bin/nnctl cloud deploy \
     --instance-type <instance-type> \
     --ssh-key-id <ssh-key-id> \
     --hostname nnctl-bench-$(git rev-parse --short HEAD)-<gpu>
   ```

   When a prepared source OS volume is available, deploy from a clone instead.
   Use the current source volume name or ID from Verda; do not reuse deleted
   volume IDs from older benchmark notes.

   ```bash
   ./bin/nnctl cloud deploy \
     --instance-type <instance-type> \
     --source-os-volume-name <source-volume-name> \
     --hostname nnctl-bench-$(git rev-parse --short HEAD)-<gpu>
   ```

   Existing OS volumes may reject newly attached SSH keys. In that case, omit
   `--ssh-key-id` and use the keys already present in the cloned image. Record
   both the instance ID and the cloned OS volume ID from the deploy output.

3. Poll until the VM is running and has an IP address:

   ```bash
   ./bin/nnctl cloud list
   ```

4. Verify Zig, the CUDA toolkit, and the NVIDIA driver. The bootstrap installs
   Zig and the CUDA toolkit, but the kernel driver may still need to be
   installed on a fresh image:

   ```bash
   host=root@<ip>
   ssh -o UserKnownHostsFile=/tmp/zig-nn-bench-known_hosts \
     -o StrictHostKeyChecking=accept-new \
     "$host" 'zig version; test -f /usr/local/cuda/include/cuda.h; nvidia-smi'

   ssh -o UserKnownHostsFile=/tmp/zig-nn-bench-known_hosts "$host" \
     'apt-get install -y cuda-drivers && modprobe nvidia && nvidia-smi'
   ```

   Very old NVIDIA parts may need a legacy driver instead of `cuda-drivers`.
   Record the exact `nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version`
   output with the result.

5. Deploy a clean source snapshot and run quick, then full, CUDA benchmarks.
   Commit source changes before benchmarking when possible. If a dirty local
   patch is required, use `--ignore-dirty` and record the exact patch with the
   result.

   ```bash
   dest=/root/zig-nn-bench-$(git rev-parse --short HEAD)-<gpu>
   ./bin/nnctl deploy "$host:$dest" --ref HEAD --delete

   ssh -o UserKnownHostsFile=/tmp/zig-nn-bench-known_hosts "$host" \
     "cd '$dest' && zig build benchmark -Dgpu=cuda -- --quick"

   ssh -o UserKnownHostsFile=/tmp/zig-nn-bench-known_hosts "$host" \
     "cd '$dest' && zig build benchmark -Dgpu=cuda" \
     > /tmp/zig-nn-benchmark-$(git rev-parse --short HEAD)-<gpu>.csv

   ssh -o UserKnownHostsFile=/tmp/zig-nn-bench-known_hosts "$host" \
     "cd '$dest' && zig build benchmark -Dgpu=cuda -- --filter gpu_peak" \
     > /tmp/zig-nn-benchmark-$(git rev-parse --short HEAD)-<gpu>-peak.csv
   ```

6. Destroy the instance and the cloned OS volume reported by deploy. Do not pass
   the source OS volume as `--volume-id`:

   ```bash
   ./bin/nnctl cloud destroy <instance-id> \
     --volume-id <cloned-os-volume-id> \
     --source-os-volume-id <source-os-volume-id>
   ./bin/nnctl cloud list --all
   ```

### AMD ROCm Hosts

Run AMD ROCm benchmarks manually on a ROCm-provisioned host. Do not use
`nnctl cloud deploy` for AMD until the cloud workflow has explicit ROCm image,
driver, and cleanup support.

1. Provision the host with the expected ROCm stack and verify the GPU:

   ```bash
   rocm-smi
   rocminfo | grep -E 'Name:|gfx'
   zig version
   ```

2. Deploy the source snapshot and run quick, then full, ROCm benchmarks:

   ```bash
   host=<user>@<amd-host>
   dest=/srv/zig-nn-bench-$(git rev-parse --short HEAD)-rocm
   ./bin/nnctl deploy "$host:$dest" --ref HEAD --delete

   ssh "$host" "cd '$dest' && zig build benchmark -Dgpu=rocm -- --quick"
   ssh "$host" "cd '$dest' && zig build benchmark -Dgpu=rocm" \
     > /tmp/zig-nn-benchmark-$(git rev-parse --short HEAD)-rocm.csv
   ```

3. Record the GPU model, ISA, VRAM, ROCm version, kernel driver version, source
   commit, and raw CSV path. The committed table below should stay compact and
   should use backend absolute timings for cross-GPU comparisons.

## Coverage

- `matmul`: backend-aware `BackendMatrix.dotProduct` on CPU plus Metal, CUDA,
  and ROCm, from small smoke shapes up to 1024x1024x1024.
- `activation`: backend-aware ReLU, tanh, Swish, softmax, GLU, and SwiGLU on
  CPU, Metal, CUDA, and ROCm, including multi-million element activation cases.
- `gpu_heavy`: GPU-only Metal, CUDA, and ROCm rows for larger matrix
  multiplication and activation workloads. CPU rows and CPU sample-error
  comparisons are omitted so these cases can exceed practical CPU baseline
  sizes.
- `gpu_peak`: opt-in GPU-only Metal, CUDA, and ROCm rows for huge GEMM
  workloads sized for high-end accelerators. CPU rows and sample-error
  comparisons are omitted, and quick mode skips the suite.
- `training`: CPU `Network.trainBatch` loops plus `Network.trainBatchBackend`
  rows for Metal, CUDA, and ROCm when those backends are available.
- `tiny_gpt`: legacy CPU TinyGPT forward passes through the example model, with
  a larger four-layer decoder row in the default suite. The device decoder is
  covered by transfer/kernel/synchronization assertions and deterministic
  parity tests; add dedicated training and cached-generation timing rows before
  using this suite for end-to-end GPU claims.
- `quantization`: CPU uniform scalar quantization and TurboQuant encode,
  decode, error measurement, and KV-cache-shaped TurboQuant loops over
  per-head key/value vectors, including larger flat-vector and KV-cache rows.

GPU rows include `sample_error`, sampled against the CPU result. The Metal,
CUDA, and ROCm kernels currently use f32 buffers, so small non-zero errors are
expected.

## Remote GPU Results

These results are from Verda single-GPU spot instances and a manually
provisioned AMD ROCm host. Lower backend average time is better. CPU timings
vary by VM, so use backend absolute timings for cross-GPU comparison.

The successful CUDA rows include the `ZIG_NN_CUDA_INFINITY` NVRTC fix in
`src/cuda/kernels.cu`. The run used CUDA toolkit 13.0 and NVIDIA driver
610.43.02.

The Radeon 890M ROCm rows were captured from the same benchmark source on a
Manjaro ROCm host with ROCm `7.2.53211-9999`, kernel `6.18.32-1-MANJARO`, and
ISA `gfx1150`.

Old B300, H200, RTX PRO 6000, and MI300X rows were removed because they were
captured against older commits, different toolchains, or manual hosts. Add
older GPU models back only after rerunning the workflows above.

| GPU | Instance type | Location | Memory | Capability | Driver | Spot/h | Result |
| --- | --- | --- | ---: | ---: | --- | ---: | --- |
| NVIDIA RTX A6000 | 1A6000.10V | FIN-01 | 49140 MiB | 8.6 | 610.43.02 | 0.2135 usd | ok |
| NVIDIA A100-SXM4-40GB | 1A100.40S.22V | FIN-01 | 39936 MiB | 8.0 | 610.43.02 | 0.4515 usd | ok |
| NVIDIA A100-SXM4-80GB | 1A100.22V | FIN-03 | 81920 MiB | 8.0 | 610.43.02 | 0.6265 usd | ok |
| AMD Radeon 890M Graphics | manual ROCm host | local | 4096 MiB | gfx1150 | ROCm 7.2.53211-9999 | n/a | ok |
| Tesla V100-SXM2-16GB | 1V100.6V | FIN-01 | 16384 MiB | 7.0 | 580.173.02 | 0.0595 usd | no result: CUDA 13 NVRTC rejected the architecture flag |
| NVIDIA RTX PRO 6000 CC | 1RTXPRO6000.30V.CC | FIN-03 | n/a | n/a | 610.43.02 | 0.6747 usd | no result: driver installed, but `nvidia-smi` found no device and reboot left the worker offline |

### GPU Peak Timings

The `gpu_peak` rows are intentionally much larger than the default comparison
rows and are meant to separate integrated GPUs from high-end accelerator-class
GPUs.

The Radeon 890M ROCm baseline was captured on the Manjaro ROCm host described
above. Values are backend `avg_ns` converted to milliseconds. Lower is better.

| suite | case | Radeon 890M ROCm |
| --- | --- | ---: |
| gpu_peak | matmul_8192x8192x8192 | 3798.242 ms |
| gpu_peak | matmul_12288x12288x12288 | 11371.647 ms |
| gpu_peak | matmul_16384x8192x16384 | 6392.198 ms |
| gpu_peak | matmul_24576x4096x24576 | 5947.610 ms |

Verda spot coverage for `gpu_peak` was attempted on 2026-07-13, but no CUDA
timings were captured from the spot inventory available at that time.

| GPU | Instance type | Location | Source | Result |
| --- | --- | --- | --- | --- |
| NVIDIA A100-SXM4-80GB | 1A100.22V | FIN-01 | fresh Ubuntu image | no result: cloud-init installed Zig and CUDA headers, but no `nvidia-smi`; installing `cuda-drivers` dropped SSH and the worker disappeared from `nnctl cloud list --all` before a benchmark ran |
| NVIDIA RTX PRO 6000 Blackwell Server Edition | 1RTXPRO6000.30V.CC | FIN-03 | fresh Ubuntu image | no result: the provided FIN-03 source volume was deleted, proprietary 610.43.02 failed `RmInitAdapter`, open 610.43.02 could not keep CUDA enumeration stable, `gpu_peak --quick` reported `CUDA_ERROR_INVALID_DEVICE`, and reboot left the worker offline |

### Backend Timings

All values are backend `avg_ns` converted to microseconds. NVIDIA columns use
CUDA and the Radeon 890M column uses ROCm. Lower is better.

Do not read the fastest column as a peak hardware ranking. These rows compare
the current `zig-nn` backend path, including per-operation allocation,
synchronization, backend library behavior, and runtime overhead.

| suite | case | A6000 CUDA | A100 40GB CUDA | A100 80GB CUDA | Radeon 890M ROCm | fastest |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| matmul | 64x64x64 | 27.391 us | 32.805 us | 32.252 us | **23.055 us** | Radeon 890M ROCm |
| matmul | 128x128x128 | 79.918 us | 90.210 us | **71.844 us** | 112.966 us | A100 80GB CUDA |
| matmul | 256x256x256 | 247.126 us | 289.016 us | 213.084 us | **167.960 us** | Radeon 890M ROCm |
| matmul | 512x512x512 | 888.459 us | 953.149 us | 651.223 us | **617.898 us** | Radeon 890M ROCm |
| matmul | 1024x1024x1024 | 3675.980 us | 4026.399 us | 2758.149 us | **2549.791 us** | Radeon 890M ROCm |
| activation | relu_512x512 | 877.009 us | 935.598 us | 650.154 us | **310.732 us** | Radeon 890M ROCm |
| activation | tanh_512x512 | 874.204 us | 936.338 us | 650.950 us | **341.186 us** | Radeon 890M ROCm |
| activation | swish_512x512 | 922.134 us | 925.814 us | 652.972 us | **342.798 us** | Radeon 890M ROCm |
| activation | softmax_256x256 | 459.538 us | 626.464 us | 550.704 us | **293.018 us** | Radeon 890M ROCm |
| activation | glu_512x512 | 876.888 us | 930.491 us | 677.686 us | **356.368 us** | Radeon 890M ROCm |
| activation | swiglu_512x512 | 877.286 us | 935.728 us | 656.160 us | **356.494 us** | Radeon 890M ROCm |
| training | batch64_32_64_16_steps8 | 6224.451 us | 7250.397 us | 7144.028 us | **4181.507 us** | Radeon 890M ROCm |
| training | batch128_64_128_32_steps4 | 7807.708 us | 8752.585 us | 7495.103 us | **4100.768 us** | Radeon 890M ROCm |
| training | batch256_128_256_64_steps4 | 29589.377 us | 31790.415 us | 23932.484 us | **12383.132 us** | Radeon 890M ROCm |
| gpu_heavy | matmul_2048x2048x2048 | 14862.869 us | 16299.930 us | **11550.046 us** | 20422.924 us | A100 80GB CUDA |
| gpu_heavy | matmul_4096x1024x4096 | 56849.723 us | 61306.667 us | **43415.654 us** | 46259.500 us | A100 80GB CUDA |
| gpu_heavy | relu_8192x8192 | 221300.538 us | 234864.349 us | 161706.900 us | **85734.531 us** | Radeon 890M ROCm |
| gpu_heavy | softmax_4096x4096 | 58741.468 us | 64372.288 us | **45615.792 us** | 100296.001 us | A100 80GB CUDA |
| gpu_heavy | swiglu_4096x4096 | 55609.986 us | 59002.368 us | 41563.598 us | **15042.235 us** | Radeon 890M ROCm |

### Current Read

Radeon 890M ROCm is fastest on many small, activation, and training rows in this
suite, while A100 80GB remains ahead on the largest matrix multiplication and
4096x4096 softmax cases. Treat these rows as backend smoke and trend data rather
than peak hardware measurements, especially across discrete GPUs and the shared
memory AMD APU.
