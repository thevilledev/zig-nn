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
mise run go:lint
mise run actions:lint
mise run hooks:check
mise run ci
```
