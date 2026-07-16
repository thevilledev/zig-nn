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

The live labs are grouped into a learning route:

- **Learning XOR:** loss, topology, and four truth-table probabilities;
- **Approximating a Curve:** training samples, the target function, and the
  network's evolving prediction;
- **Drawing a Decision Boundary:** labelled points, target circle, and learned
  probability field;
- **Comparing Optimizers:** synchronized SGD, momentum, and AdamW loss,
  accuracy, decision boundaries, and runtime telemetry;
- **When Metal Wins:** synchronized CPU/Metal matrix timings, speedup,
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

- `GET /api/capabilities` returns the host platform and backends this lab can
  build. The initial accelerator UI exposes only CPU and Metal on macOS.
- `GET /api/experiments` returns the allowlisted learning metadata, metric
  definitions, backend contract, and numeric parameter constraints.
- `POST /api/runs` accepts
  `{ "experiment": "...", "backend": "cpu|metal", "parameters": { ... } }`
  and returns a run ID.
- `GET /api/runs/{id}/events` streams ordered Server-Sent Events and honors
  `Last-Event-ID` for replay.
- `DELETE /api/runs/{id}` cancels the native process.

Unknown experiments, arbitrary flags, unsupported backends, unknown
parameters, invalid ranges, and concurrent starts are rejected by the Go
server. Accelerator selection is exact: an explicit Metal request either
reports Metal from the native process or fails instead of falling back to CPU.

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
optimizers. `run_started.data.execution` reports the requested backend, the
selected backend, and Zig optimization mode. Tensor and accelerator snapshots
may include cumulative or interval telemetry for tensor operations and native
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
