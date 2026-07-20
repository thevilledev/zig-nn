# Development Environment

Use the pinned tools in `mise.toml` when preparing a dev machine or running
shared checks.

```bash
mise install
```

## Local Hooks

The Git hook tooling is Node-based and pinned by `package-lock.json`. Install
the dependencies and hooks through the mise task:

```bash
mise run hooks:install
```

Run the same hooks across the repository with:

```bash
mise run hooks:check
```

Update hook revisions with the same seven-day cooldown used for dependency
updates:

```bash
npm run prek:update
```

## Checks

Use the smallest check that covers the change you made:

```bash
mise run zig:test
mise run zig:test-acceptance
mise run go:test
mise run go:test-race
mise run go:coverage
mise run go:lint
mise run go:vuln
mise run web:lint
mise run web:check
mise run web:test
mise run web:coverage
mise run coverage
mise run packer:validate
mise run actions:lint
mise run hooks:check
mise run ci
```

Coverage thresholds are measured ratchets, not aspirational repository-wide
numbers. Go currently requires 59.0% statement coverage. The web suite requires
89% statements, 66% branches, 91% functions, and 89% lines. `mise run coverage`
runs both reports and writes generated output under the ignored `coverage/`
directory.

Zig does not currently have a stable line-coverage gate in this toolchain.
Risk-based coverage instead includes allocation-failure checks, fuzzed ZNN and
TGPT headers, tokenizer and request-parser fuzz tests, seeded backend parity,
checkpoint versions 1–3, truncated/corrupt files, explicit-device failure,
auto fallback, deterministic sampling, KV-cache rollover, and persistent
session reuse.

Every inference change should run `mise run zig:test-acceptance` in addition to
the normal Zig suite. Wall-clock performance is gated only on controlled
release hardware, while live allocation/buffer growth and unexpected device
transfers or synchronizations are deterministic blocking assertions. See
[Benchmarks](benchmarks.md) for the committed CPU baseline and manual CUDA/ROCm
release process.

The toolchain pins both Packer and `govulncheck` in `mise.toml`. The embedded
Verda template also requires an exact plugin release, so `packer init` cannot
silently select a newer provider plugin during an image build.
