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

Save a raw `nnctl` CSV baseline and compare a later run against it:

```bash
nnctl benchmark --quick --csv > baseline.csv
nnctl benchmark --quick --compare baseline.csv
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
zig build benchmark -- --filter layer_norm
zig build benchmark -- --filter training
zig build benchmark -- --filter tiny_gpt
zig build benchmark -- --filter quantization
```

## Remote GPU Benchmark Workflow

Use this process when adding a new GPU result to the comparison table. The goal
is to keep each run tied to a reproducible source snapshot and to avoid leaving
spot instances running after the benchmark is captured.

1. List currently available single-GPU spot instance types:

   ```bash
   ./bin/nnctl cloud pricing --single-gpu
   ```

2. Pick a single-GPU model that is not already represented in the results table.
   Prefer the cheapest available instance when several untested models are
   available.

3. Materialize the embedded Verda Packer template, build the golden OS volume,
   and record the resulting Verda OS volume ID:

   ```bash
   ./bin/nnctl cloud packer-template /tmp/zig-nn-verda-packer
   cd /tmp/zig-nn-verda-packer
   packer init .
   packer build .
   ```

4. Deploy one benchmark worker with the shared SSH key. `nnctl` can choose a
   Packer-built source OS volume by name when matching source volumes exist in
   multiple zones: it checks spot availability, picks a matching volume in an
   available location, clones that source volume, and uses the clone as the
   instance image:

   ```bash
   ./bin/nnctl cloud deploy \
     --instance-type <instance-type> \
     --source-os-volume-name <packer-os-volume-name> \
     --ssh-key-id '397644c5-cb14-4ab0-b072-d14090f881f3'
   ```

   To force a specific source volume, pass `--source-os-volume-id` instead.

5. Poll until the VM reports an IP address:

   ```bash
   ./bin/nnctl cloud list
   ```

6. SSH to the host and wait for cloud-init to finish. Some images expose the GPU
   before Zig has been installed by the startup script. Use `nvidia-smi` on CUDA
   hosts, or `rocm-smi` and `rocminfo` on ROCm hosts.

   ```bash
   host=root@<ip>
   ssh -o UserKnownHostsFile=/tmp/zig-nn-bench-known_hosts \
     -o StrictHostKeyChecking=accept-new \
     "$host" 'cloud-init status --long; zig version; nvidia-smi'
   ```

7. Deploy the current source snapshot and run a quick backend smoke test
   followed by the full ReleaseFast suite:

   ```bash
   dest=/root/zig-nn-bench-$(git rev-parse --short HEAD)-$(date -u +%Y%m%dT%H%M%SZ)
   ./bin/nnctl deploy "$host:$dest"

   ssh -o UserKnownHostsFile=/tmp/zig-nn-bench-known_hosts "$host" \
     "cd '$dest' && zig build benchmark -Dgpu=<cuda|rocm> -- --quick"

   ssh -o UserKnownHostsFile=/tmp/zig-nn-bench-known_hosts "$host" \
     "cd '$dest' && zig build benchmark -Dgpu=<cuda|rocm>"
   ```

8. Record the GPU model, instance type, source commit, command output, and raw
   CSV in the local benchmark scratchpad. The committed table below should stay
   compact and should use backend absolute timings for cross-GPU comparisons.

9. Destroy the instance and the cloned OS volume reported by deploy. Do not pass
   the golden Packer source volume as `--volume-id`:

   ```bash
   ./bin/nnctl cloud destroy <instance-id> \
     --volume-id <cloned-os-volume-id> \
     --source-os-volume-id <packer-os-volume-id>
   ./bin/nnctl cloud list --all
   ```

## Coverage

- `matmul`: backend-aware `BackendMatrix.dotProduct` on CPU plus Metal, CUDA,
  and ROCm, from small smoke shapes up to 1024x1024x1024.
- `activation`: backend-aware ReLU, tanh, Swish, softmax, GLU, and SwiGLU on
  CPU, Metal, CUDA, and ROCm, including multi-million element activation cases.
- `gpu_heavy`: GPU-only Metal, CUDA, and ROCm rows for larger matrix
  multiplication and activation workloads. CPU rows and CPU sample-error
  comparisons are omitted so these cases can exceed practical CPU baseline
  sizes.
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

## Remote GPU Results So Far

These results are from Verda single-GPU spot instances. Lower backend average
time is better. CPU timings vary by VM, so use the speedup columns only as
within-host context; use backend absolute timings for cross-GPU comparison.

The B300, A100, and RTX PRO 6000 runs were captured from
`f935dc9f1de578ca570b7342dc326ba7dd1f8db3` plus the local NVRTC
minor-architecture fallback that later landed as
`57dd73185f59f350d1296ef11078b7d3f2f888c3`. The H200 run used `57dd731`
directly. The MI300X ROCm run was captured on 2026-07-11 from
`17aaf061901c09d0b9c5320e5e530932d8c2c9b9` with ROCm
`7.2.53211-97f5574fe2`.

| GPU | Instance type | Memory | Capability/ISA | Spot/h |
| --- | --- | ---: | ---: | ---: |
| NVIDIA B300 SXM6 AC | B300 benchmark VM | 275040 MiB | 10.3 | n/a |
| NVIDIA A100-SXM4-40GB | 1A100.40S.22V | 40960 MiB | 8.0 | 0.4515 usd |
| NVIDIA RTX PRO 6000 Blackwell Server Edition | 1RTXPRO6000.30V | 97887 MiB | 12.0 | 0.6615 usd |
| NVIDIA H200 | 1H200.141S.44V | 143771 MiB | 9.0 | 1.4000 usd |
| AMD Instinct MI300X VF | MI300X benchmark VM | 196288 MiB | gfx942 | n/a |

The L40S instance type was attempted but disappeared during the first quick
benchmark start, so there is no L40S result yet.

### Backend Timings

All values are backend `avg_ns` converted to microseconds. Lower is better. The
NVIDIA columns use CUDA and the MI300X column uses ROCm, so the fastest column
compares backend plus GPU for the common rows available across runs.

| suite | case | H200 CUDA | B300 CUDA | RTX PRO 6000 CUDA | A100 40GB CUDA | MI300X ROCm | fastest |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| matmul | 64x64x64 | **17.929 us** | 21.285 us | 20.322 us | 29.253 us | 28.855 us | H200 CUDA |
| matmul | 128x128x128 | 98.038 us | 53.447 us | **50.806 us** | 98.948 us | 60.842 us | RTX PRO 6000 CUDA |
| matmul | 256x256x256 | 319.594 us | 152.891 us | 153.610 us | 337.947 us | **123.522 us** | MI300X ROCm |
| activation | relu_512x512 | 1235.105 us | 503.210 us | 555.753 us | 1273.950 us | **372.241 us** | MI300X ROCm |
| activation | tanh_512x512 | 1236.356 us | 497.784 us | 551.355 us | 1270.881 us | **380.152 us** | MI300X ROCm |
| activation | swish_512x512 | 1238.508 us | 492.604 us | 559.834 us | 1266.106 us | **369.961 us** | MI300X ROCm |
| activation | softmax_256x256 | 488.318 us | 297.040 us | **294.713 us** | 606.676 us | 485.011 us | RTX PRO 6000 CUDA |
| activation | glu_512x512 | 1236.975 us | **498.199 us** | 549.327 us | 1267.980 us | 1270.741 us | B300 CUDA |
| activation | swiglu_512x512 | 1234.974 us | **490.700 us** | 551.783 us | 1260.985 us | 1162.341 us | B300 CUDA |
| training | batch64_32_64_16_steps8 | 5544.331 us | **5460.393 us** | 5863.017 us | 8317.747 us | 5718.591 us | B300 CUDA |
| training | batch128_64_128_32_steps4 | 9428.812 us | 5471.653 us | 6111.196 us | 10718.566 us | **5446.511 us** | MI300X ROCm |

### MI300X ROCm Full-Suite Highlights

These rows were added by the full `-Dgpu=rocm` ReleaseFast run and do not yet
have refreshed NVIDIA comparison rows in this table.

| suite | case | MI300X ROCm |
| --- | --- | ---: |
| matmul | 512x512x512 | 476.433 us |
| matmul | 1024x1024x1024 | 1973.841 us |
| gpu_heavy | matmul_2048x2048x2048 | 5979.791 us |
| gpu_heavy | matmul_4096x1024x4096 | 25168.139 us |
| gpu_heavy | relu_8192x8192 | 116520.081 us |
| gpu_heavy | softmax_4096x4096 | 35956.735 us |
| gpu_heavy | swiglu_4096x4096 | 29626.441 us |
| training | batch256_128_256_64_steps4 | 15026.182 us |

### Current Read

B300 remains strong for the historical CUDA rows, especially GLU, SwiGLU, and
small training. MI300X ROCm is fastest among the common rows for 256x256x256
matmul, ReLU, tanh, Swish, and the batch128 training case. RTX PRO 6000 still
wins the 128x128x128 matmul and softmax cases, while H200 wins only the
smallest matmul in this suite. A100 40GB is consistently behind the newer GPUs
here, but remains a useful baseline for older data-center hardware.

Small training remains sensitive to launch overhead. The device runtime now
batches synchronization and uses vendor GEMM, but these historical rows predate
the Transformer tensor path. Treat them as backend smoke and trend data rather
than peak hardware or current TinyGPT measurements, and record a fresh source
commit whenever publishing new results.
