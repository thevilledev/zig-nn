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

The first labs are:

- **Learning XOR:** loss, topology, and four truth-table probabilities;
- **Approximating a Curve:** training samples, the target function, and the
  network's evolving prediction;
- **Drawing a Decision Boundary:** labelled points, target circle, and learned
  probability field.

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
mise run web:check
mise run web:test
mise run web:build
```

## HTTP Interface

- `GET /api/experiments` returns the allowlisted learning metadata and numeric
  parameter constraints.
- `POST /api/runs` accepts `{ "experiment": "...", "parameters": { ... } }`
  and returns a run ID.
- `GET /api/runs/{id}/events` streams ordered Server-Sent Events and honors
  `Last-Event-ID` for replay.
- `DELETE /api/runs/{id}` cancels the native process.

Unknown experiments, arbitrary flags, unknown parameters, invalid ranges, and
concurrent starts are rejected by the Go server.

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

Supported experiment event types are:

- `run_started` for validated configuration, topology, and static evidence;
- `metric` for scalar loss samples;
- `snapshot` for a visualization-specific, step-addressable state;
- `run_completed` for final evidence and summary metrics.

`nnctl` adds `run_id` and monotonically increasing `seq` fields, and may add
`log` or `run_failed` events around the native stream. Metrics are emitted at
roughly one-percent intervals and snapshots at roughly five-percent intervals,
including the first and final state.

## Add Another Learning Experiment

1. Give the Zig program the four bounded flags above and preserve its default
   human output.
2. Import the `experiment_events` build module and use `events.emit` for strict
   stdout NDJSON plus the shared cadence helpers for metrics and snapshots.
3. Add an allowlisted specification, parameter constraints, learning question,
   interpretation, visualization key, and source paths in `nnctl/internal/lab`.
4. Add a typed snapshot payload and accessible Svelte visualization, then cover
   the reducer and component with frontend tests.
5. Verify both the normal acceptance behavior and a short structured run before
   adding the experiment to the documented learning route.
