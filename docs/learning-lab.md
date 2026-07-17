# Real-Time Learning Lab

The learning lab connects the repository's educational questions to live,
inspectable evidence. A Svelte frontend displays events from native Zig
experiments, while `nnctl` owns process execution, validation, cancellation,
and the local HTTP server.

## Run The Lab

```bash
mise run lab
```

Open the printed `http://127.0.0.1:8091` URL. Only one experiment runs at a
time. Runs and their event history live in memory until `nnctl lab` exits.

To enable Verda worker deployment and remote CUDA runs, start the same local
server with cloud control explicitly enabled:

```bash
cd nnctl
go run ./cmd/nnctl lab --cloud
```

Cloud control is accepted only on a loopback bind address. The browser never
receives provider credentials or SSH private keys; the local Go process reads
Verda credentials from the OS keyring and performs provider, Git, rsync, and
SSH operations itself.

The live labs are grouped into a learning route:

- **Learning XOR:** loss, topology, and four truth-table probabilities;
- **Approximating a Curve:** training samples, the target function, and the
  network's evolving prediction;
- **Drawing a Decision Boundary:** labelled points, target circle, and learned
  probability field;
- **Learning in Frequency Space:** raw-coordinate and Fourier-feature curves,
  amplitude spectra, pointwise loss, and low/high harmonic error;
- **Comparing Optimizers:** synchronized SGD, momentum, and AdamW loss,
  accuracy, decision boundaries, and runtime telemetry;
- **When a GPU Wins:** synchronized CPU/GPU matrix timings, speedup,
  numerical error, transfers, kernels, GEMMs, and synchronization;
- **Learning Semantic Search:** InfoNCE loss, recall, reciprocal rank, a live
  query/document cosine-similarity grid, and inspectable rankings.

Each page provides bounded parameter controls, a live loss chart, snapshots
that can be scrubbed after training, interpretation prompts, and the relevant
source paths. The seed controls dataset construction, model initialization,
and shuffling through independent deterministic streams.

## Frontend Development

Run the API and Vite in separate terminals:

```bash
mise run web:install
mise run lab:api
cd web && npm run dev
```

The Vite server proxies `/api` to `127.0.0.1:8091`. Production assets are built
into the ignored `web/dist/` directory and served by `nnctl lab`.

Relevant checks are:

```bash
mise run web:lint
mise run web:check
mise run web:test
mise run web:build
```

## HTTP Interface

- `GET /api/capabilities` returns the host platform, local backends, and whether
  cloud control was enabled. Local acceleration exposes CPU and Metal on macOS.
- `GET /api/experiments` returns the allowlisted learning metadata, metric
  definitions, backend contract, and numeric parameter constraints.
- `POST /api/runs` accepts
  `{ "experiment": "...", "backend": "cpu|metal|cuda", "parameters": { ... } }`
  and returns a run ID.
- `GET /api/runs/{id}/events` streams ordered Server-Sent Events and honors
  `Last-Event-ID` for replay.
- `DELETE /api/runs/{id}` cancels the native process.

Unknown experiments, arbitrary flags, unsupported backends, unknown
parameters, invalid ranges, and concurrent starts are rejected by the Go
server. Accelerator selection is exact: an explicit Metal request either
reports Metal from the native process or fails instead of falling back to CPU.

With `--cloud`, the server also exposes:

- `GET /api/cloud/status` for keyring configuration and the committed/dirty
  repository state;
- `GET /api/cloud/options` for available single-NVIDIA-GPU prices and the
  fixed golden OS volume name;
- `GET|POST /api/cloud/workers` to inspect or asynchronously create the one
  persisted worker managed by the lab;
- `GET /api/cloud/workers/{id}/events` for worker lifecycle SSE;
- `DELETE /api/cloud/workers/{id}` to permanently destroy the instance and its
  cloned OS volume while protecting the golden source volume.

A remote run extends the normal request with an allowlisted target:

```json
{
  "experiment": "optimizer-lab",
  "backend": "cuda",
  "parameters": { "steps": 200 },
  "target": { "kind": "cloud", "worker_id": "..." },
  "acknowledge_committed_head": true
}
```

The acknowledgement is required when the local tree is dirty because cloud
runs always archive committed `HEAD`; uncommitted and untracked files are never
uploaded. The run stream adds `run_status` events for infrastructure phases,
then carries the same native experiment events as a local run.

## Cloud Setup And Cleanup

The existing Verda commands and the lab share the `nnctl/verda` keyring service
with accounts `client_id` and `client_secret`. On macOS, store credentials with:

```bash
security add-generic-password -U -s nnctl/verda -a client_id -w "$VERDA_CLIENT_ID"
security add-generic-password -U -s nnctl/verda -a client_secret -w "$VERDA_CLIENT_SECRET"
```

On Linux, store the same service/account pairs in the desktop Secret Service
keyring. Verify access with:

```bash
nnctl cloud pricing --single-gpu
```

The lab always clones `packer-verda-zig-nn-volume-root`. Authorized root keys
and the CUDA/Zig tooling are baked into that golden volume, so deploy requests
intentionally omit `ssh_key_ids`. The UI does not allow selecting another SSH
key or OS volume, and only offers GPU capacity in locations with a ready golden
volume. SSH itself uses the local process's normal identity and agent
configuration.

Workers default to spot capacity and manual cleanup so the same VM can run
multiple experiments. **Destroy worker** permanently deletes the instance and
its cloned OS volume. Opting into automatic cleanup instead destroys the worker
after the next run, after 30 minutes idle, or when the lab exits normally.

Managed-worker state is stored in `learning-lab.sqlite` under the OS user-cache
directory (`~/Library/Caches/nnctl/` on macOS and typically
`~/.cache/nnctl/` on Linux). The database contains worker and provider resource
IDs, connection metadata, and lifecycle state, but never credentials. On
restart, the lab verifies the persisted Verda instance, SSH, Zig, and CUDA
before making the worker ready for more experiments. The previous
`lab-cloud-worker.json` journal is migrated once. The UI deliberately does not
adopt or destroy unrelated Verda instances. Pass `--state-db PATH` to
`nnctl lab --cloud` to use a different database location.

The CPU-capable structured experiments can execute on the remote CPU.
Optimizer Lab and Semantic Search additionally allow explicit CUDA, while the
GPU benchmark uses CUDA on a cloud worker. CUDA requests fail if the remote
preflight or native process cannot select CUDA.

## Experiment Event Protocol

Instrumented experiments retain human output by default and accept:

```text
--format ndjson --epochs <count> --learning-rate <value> --seed <number>
```

In NDJSON mode, stdout contains one versioned JSON object per line and is
flushed after every event. Diagnostics belong on stderr. The common envelope is:

```json
{
  "v": 1,
  "type": "metric",
  "experiment": "xor-training",
  "step": 100,
  "total_steps": 10000,
  "data": { "name": "loss", "value": 0.12 }
}
```

Supported experiment event types remain:

- `run_started` for validated configuration, topology, and static evidence;
- `metric` for scalar loss samples;
- `snapshot` for a visualization-specific, step-addressable state;
- `run_completed` for final evidence and summary metrics.

Metric payloads may include a `series` name for comparisons such as three
optimizers or the raw/Fourier spectral models.
`run_started.data.execution` reports the requested backend, the selected
backend, and Zig optimization mode. Tensor and accelerator snapshots may
include cumulative or interval telemetry for tensor operations and native
backend work.

`nnctl` adds `run_id` and monotonically increasing `seq` fields, and may add
`log` or `run_failed` events around the native stream. Metrics are emitted at
roughly one-percent intervals and snapshots at roughly five-percent intervals,
including the first and final state.

## Add Another Learning Experiment

1. Give the Zig program the applicable bounded flags above and preserve its
   default human output. Backend-aware experiments must support an explicit
   `--backend` flag and report requested and selected values.
2. Import the `experiment_events` build module and use `events.emit` for strict
   stdout NDJSON plus the shared cadence helpers for metrics and snapshots.
3. Add an allowlisted specification, parameter constraints, learning question,
   interpretation, visualization key, and source paths in `nnctl/internal/lab`.
4. Add metric definitions, a typed snapshot payload, optional runtime
   telemetry, and an accessible Svelte visualization, then cover the reducer
   and component with frontend tests.
5. Verify both the normal acceptance behavior and a short structured run before
   adding the experiment to the documented learning route.

## Metal Verification

On macOS, the accelerator lessons compile Metal into the native Zig executable;
the browser only renders the resulting evidence. The lab does not use WebGPU.
Use explicit selections when validating acceleration:

```bash
zig build run_optimizer_lab -Dgpu=metal -- \
  --format ndjson --backend metal --steps 20
zig build run_gpu_benchmark -Dgpu=metal -Doptimize=ReleaseFast -- \
  --format ndjson
zig build run_semantic_search -Dgpu=metal -- \
  --format ndjson --backend metal --steps 10
```

The `run_started` event must report `selected_backend: "metal"`. Training
snapshots should show native kernel activity and zero training readbacks; later
evaluation and ranking readbacks are intentional reporting boundaries.
