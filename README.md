# Neural-Network Experiments in Zig

[![Work in Progress](https://img.shields.io/badge/Status-Work%20in%20Progress-yellow)](https://github.com/thevilledev/zig-nn)

`zig-nn` is an educational laboratory for learning how neural networks work by
building and running the pieces directly. It favors small, inspectable
implementations and experiments whose output shows whether an idea works.

There are two complementary paths through the code:

- The CPU fundamentals in [`src/matrix.zig`](src/matrix.zig),
  [`src/layer.zig`](src/layer.zig), and [`src/network.zig`](src/network.zig)
  keep forward passes, losses, and backpropagation easy to follow.
- The tensor runtime in [`src/tensor.zig`](src/tensor.zig), reusable modules in
  [`src/modules.zig`](src/modules.zig), and training tools in
  [`src/training.zig`](src/training.zig) support the later CPU, Metal, CUDA, and
  ROCm experiments without hiding device boundaries.

The reusable mechanics live in `src/`. The runnable programs in
[`experiments/`](experiments/README.md) connect those mechanics to a concrete
question, observable metrics, and relevant research. This is a learning
project, not a production machine-learning framework.

## Start Here

Install Zig `0.16.0` and Go `1.26` or newer, then run one small experiment:

```bash
go install ./nnctl/cmd/nnctl
nnctl run simple-xor
```

Continue with `nnctl run quick`, or choose a topic from the
[experiment guide](experiments/README.md). The complete setup, direct Zig
commands, data requirements, and full validation workflow are in
[Getting Started](docs/getting-started.md).

For a guided browser view of live training, run:

```bash
mise run lab
```

The local learning lab starts with XOR training, nonlinear regression, and
binary classification. It streams native Zig metrics and snapshots through
`nnctl`; see the [learning lab guide](docs/learning-lab.md) for the interface,
development workflow, and experiment event protocol.

## Documentation

- [Experiments](experiments/README.md) — learning routes, runnable programs,
  source links, expected evidence, and research background
- [Getting Started](docs/getting-started.md) — prerequisites, build commands,
  tests, data, and checkpoints
- [Architecture](docs/architecture.md) — how the CPU fundamentals, tensor
  runtime, backends, and model-specific components fit together
- [Research Resources](docs/research.md) — papers, datasets, official docs, and
  links back to their implementations
- [GPU and Backend Notes](docs/gpu.md) — backend boundaries and Metal, CUDA,
  and ROCm verification
- [NLP and Adjacent Roadmap](docs/nlp-roadmap.md) — current backend boundaries
  and planned learning experiments
- [Benchmarks](docs/benchmarks.md) — repeatable local and remote measurements
- [Development Environment](docs/development.md) — pinned tools, hooks, and
  repository-wide checks
- [Real-Time Learning Lab](docs/learning-lab.md) — live browser experiments,
  controls, and the structured event protocol

## License

MIT License
