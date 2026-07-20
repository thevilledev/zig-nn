# Roadmap: Consolidate, Serve, Optimize

`zig-nn` remains an educational lab with one polished inference vertical.
Unrelated model-domain experiments are paused through the approachable
inference release. Existing ZNN and TGPT checkpoint formats remain the public
boundary; no universal model container is planned for these releases.

## v0.0.5 — Confidence And Measurement

The release candidate adds measured Go and web coverage ratchets plus
risk-based Zig tests for allocation failure, fuzzed inputs, backend parity, and
malformed or incompatible checkpoints. The end-to-end inference benchmark now
covers dense batch sizes 1 and 32, TinyGPT cold load, prompt prefill, cached
decode, and generation. Its CSV format remains compatible with old 12-column
baselines while current rows declare work units and runtime counters.

Exit gate: `mise run coverage`, the Zig inference tests, and the committed
Apple M1 baseline reproduce locally. Paid CUDA and ROCm measurements remain
manual release checks.

## v0.0.6 — Approachable Inference

TinyGPT checkpoint logic is reusable library code with TGPT v1–v3
compatibility. `nn.Inference` exposes separate persistent `DenseSession` and
`TextSession` contracts with allocation-explicit prediction and generation.
The sessions own device state, uploaded weights, execution context, workspaces,
KV cache, and runtime statistics.

The supported user path is:

```bash
nnctl model inspect CHECKPOINT
nnctl predict --model MODEL.bin --input '[0,1]' --gpu auto
nnctl serve --model CHECKPOINT --gpu auto
```

`nnctl chat` uses the same TinyGPT serving path. Existing experiment servers
remain compatibility commands. The v1 text server supports JSON responses and
SSE-framed `stream=true` responses, but deliberately permits one active
generation and bounded requests rather than claiming production concurrency.

Exit gate: inspect, load, infer, and serve existing ZNN and TGPT checkpoints;
explicit unavailable devices fail, only `auto` falls back, and greedy TinyGPT
outputs remain within f32 parity tolerance.

## v0.0.7 — Fast Steady-State Inference

Loaded sessions persist across requests. TinyGPT reuses its decoder, logits,
sampling workspace, intermediate storage, and in-place KV cache. The CPU path
uses persistent contiguous inference weights, blocked/vectorized f32
accumulation, and fused linear-plus-bias-plus-GELU while retaining readable
reference compositions for parity checks.

The controlled Apple M1 baseline records a 2.11x improvement in the largest
cached-decode row over v0.0.6, no unaffected regression above 5%, and checksum
error within `1e-4`. Tests require stable live backend-buffer counts across
1,000 decode steps and prohibit device/host round trips inside decoder blocks.

Exit gate: the committed release-baseline regression test and deterministic
resource assertions pass. Wall-clock gates run only on controlled hardware.

## Deferred Until The Steady-State Path Is Proven

- mixed precision;
- GGUF or ONNX import;
- fused attention kernels;
- production multi-tenant or incremental-token serving.

After these releases, new work should be chosen by measured bottlenecks and
the educational value of the implementation, not by model-count coverage.

## Pull-Request Gates

Every pull request runs Zig formatting and unit tests, inference acceptance
tests, Go lint/race/coverage checks, and web lint/type/test/coverage checks.
Performance timings are informational in ordinary CI; deterministic resource
and transfer assertions are blocking. Public changes must update architecture,
getting-started, GPU, TinyGPT, serving, and benchmark documentation together.
