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
run tied to a reproducible source snapshot, deploy only one worker at a time,
and destroy the worker plus any cloned OS volume as soon as the benchmark is
captured.

The one-command Verda workflow is:

```bash
./bin/nnctl cloud benchmark-deploy \
  --instance-type 1A100.22V \
  --ssh-key-id <verda-ssh-key-id>
```

`benchmark-deploy` writes the embedded Packer template, reuses a matching
golden OS volume or builds one when missing, clones the golden volume, deploys
the worker, waits for SSH, transfers a clean `HEAD` snapshot, runs a quick
smoke followed by the full CUDA benchmark, and saves a metadata-prefixed CSV in
the repository root. It then permanently deletes the worker and cloned volume;
the golden source volume is retained. Use `--market on-demand` to opt out of
the default spot market, `--filter gpu_peak` to capture one suite,
`--update-docs` to add a generated result section below, or `--keep-instance`
to retain the worker and clone for debugging. Packer is only invoked when a
reusable source OS volume is unavailable.

The detailed steps below remain useful for diagnosing or manually recovering a
run.

### NVIDIA Verda Spot Workers

Use `nnctl cloud deploy` for NVIDIA CUDA benchmarks. Prefer spot capacity, then
fall back to on-demand only when no useful spot GPU is available in a location
with a current golden source OS volume.

Current Packer-built CUDA 13.0 source OS volume from 2026-07-14:

| Location | Source OS volume ID | Image |
| --- | --- | --- |
| FIN-01 | `49caae1d-cfb7-4ef1-b657-57c364d52e59` | `ubuntu-24.04-cuda-13.0-open` |

1. List available single-GPU plans in locations with a current source OS volume.
   Pick one GPU model that is not already represented in the results table:

   ```bash
   ./bin/nnctl cloud pricing \
     --market spot \
     --single-gpu \
     --location-code FIN-01

   # Only when no useful spot GPU is available:
   ./bin/nnctl cloud pricing \
     --market on-demand \
     --single-gpu \
     --location-code FIN-01
   ```

2. Deploy one worker from a clone of the matching source OS volume. Never pass
   the source OS volume as a worker image or cleanup volume. The clone ID is
   reported as `os_volume_clone.volume_id` in the deploy JSON.

   ```bash
   market=spot
   location=FIN-01
   source_os_volume_id=49caae1d-cfb7-4ef1-b657-57c364d52e59
   instance_type=1A100.22V
   gpu_slug=a100-80gb

   ./bin/nnctl cloud deploy \
     --market "$market" \
     --instance-type "$instance_type" \
     --source-os-volume-id "$source_os_volume_id" \
     --location-code "$location" \
     --hostname "nnctl-bench-$(git rev-parse --short HEAD)-$gpu_slug" \
     --json
   ```

   The golden image must already contain persistent root `authorized_keys`.
   Passing `--ssh-key-id` should not be required for images built from the
   checked-in Packer template. Record the instance ID, IP, source OS volume ID,
   cloned OS volume ID, market, location, and price from the JSON.

3. Poll until the VM is running and has an IP address:

   ```bash
   ./bin/nnctl cloud list
   ```

4. Verify the image without installing packages on the worker. Zig, CUDA, the
   NVIDIA driver, and root SSH keys must already be bundled into the cloned OS
   volume:

   ```bash
   host=root@<ip>
   ssh -o UserKnownHostsFile=/tmp/zig-nn-bench-known_hosts \
     -o StrictHostKeyChecking=accept-new \
     "$host" 'zig version; test -f /usr/local/cuda/include/cuda.h; nvidia-smi'
   ```

   If `nvidia-smi`, `libcuda.so.1`, or a raw `cuInit(0)` probe fails, destroy
   the worker and cloned OS volume, then rebuild or choose another base image.
   Do not repair benchmark workers with `apt-get`, `modprobe`, or manual driver
   steps. CUDA 13.0 on the current open-driver image worked on A100 but failed
   raw `cuInit(0)` on RTX A6000.
   Record the exact `nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version`
   output with the result.

5. Deploy a clean source snapshot and run quick, then full, CUDA benchmarks.
   Commit source changes before benchmarking when possible. If local scratch
   files make the worktree dirty, pass `--ignore-dirty` while still deploying
   `--ref HEAD`; uncommitted files are not included in the archive.

   ```bash
   dest=/root/zig-nn-bench-$(git rev-parse --short HEAD)-$gpu_slug
   ssh_opts='ssh -o UserKnownHostsFile=/tmp/zig-nn-bench-known_hosts'

   ./bin/nnctl deploy "$host:$dest" \
     --ref HEAD \
     --delete \
     --ignore-dirty \
     --ssh "$ssh_opts"

   ssh -o UserKnownHostsFile=/tmp/zig-nn-bench-known_hosts "$host" \
     "source /etc/profile.d/zig-nn.sh && cd '$dest' && zig build benchmark -Dgpu=cuda -- --quick"

   ssh -o UserKnownHostsFile=/tmp/zig-nn-bench-known_hosts "$host" \
     "source /etc/profile.d/zig-nn.sh && cd '$dest' && zig build benchmark -Dgpu=cuda" \
     > /tmp/zig-nn-benchmark-$(git rev-parse --short HEAD)-$gpu_slug.csv

   ssh -o UserKnownHostsFile=/tmp/zig-nn-bench-known_hosts "$host" \
     "source /etc/profile.d/zig-nn.sh && cd '$dest' && zig build benchmark -Dgpu=cuda -- --filter gpu_peak" \
     > /tmp/zig-nn-benchmark-$(git rev-parse --short HEAD)-$gpu_slug-peak.csv
   ```

6. Destroy the instance and the cloned OS volume reported by deploy. Pass the
   source OS volume only as protection metadata; do not pass it as `--volume-id`:

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
- `tiny_gpt`: legacy CPU TinyGPT forward passes through the experiment model, with
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
`src/cuda/kernels.cu`. Earlier A6000, A100 40GB, and H200 rows used CUDA
toolkit 13.0 with NVIDIA driver `610.43.02`; the current A100 80GB rows were
captured from the cloned `ubuntu-24.04-cuda-13.0-open` image with NVIDIA driver
`580.126.09`.

The Radeon 890M ROCm rows were captured from the same benchmark source on a
Manjaro ROCm host with ROCm `7.2.53211-9999`, kernel `6.18.32-1-MANJARO`, and
ISA `gfx1150`.

Old B300, H200, RTX PRO 6000, and MI300X rows were removed because they were
captured against older commits, different toolchains, or manual hosts. Add
older GPU models back only after rerunning the workflows above.

| GPU | Instance type | Location | Memory | Capability | Driver | Price/h | Result |
| --- | --- | --- | ---: | ---: | --- | ---: | --- |
| NVIDIA RTX A6000 | 1A6000.10V | FIN-01 | 49140 MiB | 8.6 | 610.43.02 | 0.2135 usd | ok |
| NVIDIA A100-SXM4-40GB | 1A100.40S.22V | FIN-01 | 39936 MiB | 8.0 | 610.43.02 | 0.4515 usd | ok |
| NVIDIA A100-SXM4-80GB | 1A100.22V | FIN-01 | 81920 MiB | 8.0 | 580.126.09 | 0.6265 usd | ok |
| NVIDIA H200 | 1H200.141S.44V | FIN-02 | 143771 MiB | 9.0 | 610.43.02 | 3.4855 usd | ok: `gpu_peak` only |
| AMD Radeon 890M Graphics | manual ROCm host | local | 4096 MiB | gfx1150 | ROCm 7.2.53211-9999 | n/a | ok |
| Tesla V100-SXM2-16GB | 1V100.6V | FIN-01 | 16384 MiB | 7.0 | 580.173.02 | 0.0595 usd | no result: CUDA 13 NVRTC rejected the architecture flag |
| NVIDIA RTX PRO 6000 CC | 1RTXPRO6000.30V.CC | FIN-03 | n/a | n/a | 610.43.02 | 0.6747 usd | no result: driver installed, but `nvidia-smi` found no device and reboot left the worker offline |

### GPU Peak Timings

The `gpu_peak` rows are intentionally much larger than the default comparison
rows and are meant to separate integrated GPUs from high-end accelerator-class
GPUs.

The H200 CUDA result was captured from a FIN-02 on-demand worker. The A100 80GB
CUDA result was captured from a FIN-01 spot worker cloned from source OS volume
`49caae1d-cfb7-4ef1-b657-57c364d52e59`. The Radeon 890M ROCm baseline was
captured on the Manjaro ROCm host described above. Values are backend `avg_ns`
converted to milliseconds. Lower is better.

| suite | case | H200 CUDA | A100 80GB CUDA | Radeon 890M ROCm |
| --- | --- | ---: | ---: | ---: |
| gpu_peak | matmul_8192x8192x8192 | **175.899 ms** | 372.817 ms | 3798.242 ms |
| gpu_peak | matmul_12288x12288x12288 | **416.596 ms** | 903.061 ms | 11371.647 ms |
| gpu_peak | matmul_16384x8192x16384 | **698.731 ms** | 1494.535 ms | 6392.198 ms |
| gpu_peak | matmul_24576x4096x24576 | **1477.606 ms** | 3127.268 ms | 5947.610 ms |

Verda spot coverage for `gpu_peak` was attempted on 2026-07-13, but no CUDA
timings were captured from the spot inventory available at that time. H200 was
then captured from on-demand capacity on 2026-07-14.

| GPU | Instance type | Location | Source | Result |
| --- | --- | --- | --- | --- |
| NVIDIA H200 | 1H200.141S.44V | FIN-02 | fresh Ubuntu image, on-demand | ok: installed `cuda-drivers`, verified CUDA toolkit 13.0 and NVIDIA driver 610.43.02, captured `gpu_peak` |
| NVIDIA A100-SXM4-80GB | 1A100.22V | FIN-01 | cloned golden source OS volume `49caae1d-cfb7-4ef1-b657-57c364d52e59` | ok: deployed commit `fc6f3f6`, verified `ubuntu-24.04-cuda-13.0-open`, CUDA toolkit 13.0, NVIDIA driver 580.126.09, raw `cuInit(0)`, and `cudaGetDeviceCount`; captured default and `gpu_peak` CUDA benchmarks from clone `fc6f3fe2-c19f-4962-95af-29efeaec8500` |
| NVIDIA RTX PRO 6000 Blackwell Server Edition | 1RTXPRO6000.30V.CC | FIN-03 | fresh Ubuntu image | no result: the provided FIN-03 source volume was deleted, proprietary 610.43.02 failed `RmInitAdapter`, open 610.43.02 could not keep CUDA enumeration stable, `gpu_peak --quick` reported `CUDA_ERROR_INVALID_DEVICE`, and reboot left the worker offline |

### Backend Timings

All values are backend `avg_ns` converted to microseconds. NVIDIA columns use
CUDA and the Radeon 890M column uses ROCm. Lower is better.

Do not read the fastest column as a peak hardware ranking. These rows compare
the current `zig-nn` backend path, including per-operation allocation,
synchronization, backend library behavior, and runtime overhead.

| suite | case | A6000 CUDA | A100 40GB CUDA | A100 80GB CUDA | Radeon 890M ROCm | fastest |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| matmul | 64x64x64 | 27.391 us | 32.805 us | 40.119 us | **23.055 us** | Radeon 890M ROCm |
| matmul | 128x128x128 | **79.918 us** | 90.210 us | 115.622 us | 112.966 us | A6000 CUDA |
| matmul | 256x256x256 | 247.126 us | 289.016 us | 380.374 us | **167.960 us** | Radeon 890M ROCm |
| matmul | 512x512x512 | 888.459 us | 953.149 us | 1297.021 us | **617.898 us** | Radeon 890M ROCm |
| matmul | 1024x1024x1024 | 3675.980 us | 4026.399 us | 5330.364 us | **2549.791 us** | Radeon 890M ROCm |
| activation | relu_512x512 | 877.009 us | 935.598 us | 1272.698 us | **310.732 us** | Radeon 890M ROCm |
| activation | tanh_512x512 | 874.204 us | 936.338 us | 1279.155 us | **341.186 us** | Radeon 890M ROCm |
| activation | swish_512x512 | 922.134 us | 925.814 us | 1267.655 us | **342.798 us** | Radeon 890M ROCm |
| activation | softmax_256x256 | 459.538 us | 626.464 us | 701.267 us | **293.018 us** | Radeon 890M ROCm |
| activation | glu_512x512 | 876.888 us | 930.491 us | 1274.095 us | **356.368 us** | Radeon 890M ROCm |
| activation | swiglu_512x512 | 877.286 us | 935.728 us | 1273.437 us | **356.494 us** | Radeon 890M ROCm |
| training | batch64_32_64_16_steps8 | 6224.451 us | 7250.397 us | 8964.432 us | **4181.507 us** | Radeon 890M ROCm |
| training | batch128_64_128_32_steps4 | 7807.708 us | 8752.585 us | 11625.504 us | **4100.768 us** | Radeon 890M ROCm |
| training | batch256_128_256_64_steps4 | 29589.377 us | 31790.415 us | 43097.109 us | **12383.132 us** | Radeon 890M ROCm |
| gpu_heavy | matmul_2048x2048x2048 | **14862.869 us** | 16299.930 us | 21497.780 us | 20422.924 us | A6000 CUDA |
| gpu_heavy | matmul_4096x1024x4096 | 56849.723 us | 61306.667 us | 82033.014 us | **46259.500 us** | Radeon 890M ROCm |
| gpu_heavy | relu_8192x8192 | 221300.538 us | 234864.349 us | 316483.145 us | **85734.531 us** | Radeon 890M ROCm |
| gpu_heavy | softmax_4096x4096 | **58741.468 us** | 64372.288 us | 84358.636 us | 100296.001 us | A6000 CUDA |
| gpu_heavy | swiglu_4096x4096 | 55609.986 us | 59002.368 us | 79500.632 us | **15042.235 us** | Radeon 890M ROCm |

### Current Read

Radeon 890M ROCm is fastest on many small, activation, training, and some
`gpu_heavy` rows in this suite, while A6000 CUDA is fastest on the 128x128x128
matmul, 2048x2048x2048 `gpu_heavy` matmul, and 4096x4096 softmax rows. Treat
these rows as backend smoke and trend data rather than peak hardware
measurements, especially across different driver branches, discrete GPUs, and
the shared-memory AMD APU.
