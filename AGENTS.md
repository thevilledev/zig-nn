# Agent Instructions

## Tooling

Use the commands declared in `mise.toml` instead of installing tools ad hoc. The pinned toolchain is:

- Go 1.26.1 for `nnctl`
- Zig 0.16.0
- golangci-lint 2.12.2
- zizmor 1.26.1
- Node 25.9.0 with hook dependencies pinned by `package-lock.json`

If a tool is missing, run `mise install` from the repository root. Do not run `cargo install zizmor`, install golangci-lint manually, or create per-worktree tool installs.

## Required Checks

Run the smallest check set that covers the files you changed:

- Go changes under `nnctl/`: `mise run go:lint` and `mise run go:test`
- Zig source, experiments, benchmarks, or build files: `mise run zig:test`
- Zig acceptance behavior: `mise run zig:test-acceptance`
- GitHub Actions workflow changes: `mise run actions:lint`
- Hook, formatting, or repository hygiene changes: `mise run hooks:check`
- Broad or cross-language changes: `mise run ci`

When creating a commit, always use a Conventional Commit message and verify it with:

```sh
mise run commitlint:head
```

If you are preparing changes without creating the commit yourself, say which commit message you expect to pass commitlint.

## Worktree-Friendly Rules

This repository is often used from multiple git worktrees. Keep expensive dependency and tool caches shared across worktrees:

- Prefer `mise run ...` tasks so tools come from mise's shared install/cache rather than from each worktree.
- Do not run Rust/Cargo compilation for zizmor in every worktree. Use `mise run actions:lint`, which should reuse the pinned `zizmor` binary managed by mise.
- Keep build outputs and dependency folders out of git: `.zig-cache/`, `zig-out/`, `bin/`, and `node_modules/` are local artifacts.
- Do not vendor generated caches, toolchains, downloaded models, or benchmark data into the repo.
- If a command needs a cache directory, use the default shared user cache or `/tmp`, not a path inside the worktree, unless the command already does that intentionally.

## Development Notes

- Prefer `make test` only for the `nnctl` Go tests; use `mise run go:lint` for Go linting.
- Use `npm ci` only when the Node dependencies are missing or `package-lock.json` changed. Otherwise, prefer `npx --no-install ...` through the existing npm/mise tasks.
- Keep fixes scoped to the files relevant to the request. Do not reformat unrelated Zig, Go, Markdown, or workflow files.
- If a required check cannot run, report the exact command and failure reason.
