# Quantization Examples

This directory contains reproducible quantization experiments. Each example
should be runnable through `zig build`, print comparable metrics, and keep
reusable code in `src/`.

## TurboQuant

Based on:

- TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate
- OpenReview: https://openreview.net/forum?id=tO3ASKZlok
- arXiv: https://arxiv.org/abs/2504.19874

Run:

```bash
zig build run_turboquant
```

The current example implements:

- uniform scalar quantization baseline
- deterministic random sign plus normalized Hadamard rotation
- scalar quantization in the rotated space
- inverse rotation and reconstruction
- vector metrics for MSE, max error, cosine similarity, inner-product error,
  and payload compression ratio

It is not a full paper reproduction yet. The online update path, QJL residual
correction, KV-cache benchmarks, and nearest-neighbor experiments are natural
future checkpoints.
