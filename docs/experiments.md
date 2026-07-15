# Experiments

Experiments are small, reproducible programs that turn an idea into code and
metrics. They should be easy to run, honest about what is implemented, and
specific about what remains future work.

The convention in this repo is:

- Put runnable experiment code under `examples/`.
- Keep reusable implementation pieces in `src/`.
- Keep notes beside the runnable example when the notes are specific to that
  experiment.
- Print metrics that make comparisons possible.

## Current Experiments

### GRU Selective Memory

Location: [examples/gru_sequence](../examples/gru_sequence/gru_sequence.zig)

Run:

```bash
nnctl run gru-sequence
```

This marks the first bit in a six-step sequence, fills the remaining steps
with random distractors, and trains a GRU to recall the marked bit from its
final state. The training loop performs full backpropagation through time and
reports held-out accuracy plus device telemetry.

### Denoising Autoencoder

Location: [examples/autoencoder](../examples/autoencoder/autoencoder.zig)

Run:

```bash
nnctl run autoencoder
```

This compresses noisy 4x4 patterns through a two-value bottleneck and learns
to reconstruct their clean forms. It reports held-out reconstruction error,
pixel accuracy, bottleneck centers for four pattern families, kernel counts,
and training-time readbacks.

### Optimizer Lab

Location: [examples/optimizer_lab](../examples/optimizer_lab/optimizer_lab.zig)

Run:

```bash
nnctl run optimizer-lab
```

This trains identically initialized MLPs on a deterministic two-moons dataset
with SGD, momentum, and AdamW. It reports loss reduction, held-out accuracy,
kernel counts, and any training-time readbacks. Learning rates are
optimizer-specific so the comparison also makes tuning part of the lesson.

### TurboQuant

Location: [examples/quantization](../examples/quantization/README.md)

Run:

```bash
nnctl run turboquant
```

This compares a uniform scalar quantization baseline with a rotated scalar
quantization variant, then reports vector reconstruction and inner-product
metrics. It is a reproducible lab, not a full paper reproduction.

### Tiny GPT

Location: [examples/tiny_gpt](../examples/tiny_gpt/README.md)

Run:

```bash
nnctl run tiny-gpt -- --prompt "to be" --tokens 80 --seed 42
```

This is a tiny decoder-only Transformer demo that uses this project's matrix
operations for projections and logits. It includes readable sampling and an
output-head-only training path, plus full-model educational training for small
checkpoints. Use `--preset coherent-small` when you want the model-only
coherence baseline with the readability prior disabled.

By default, `--corpus auto` uses a prepared TinyStories slice if it exists,
then prepared Tiny Shakespeare, then the checked-in toy corpus. Prepare the
sourced corpora with:

```bash
nnctl data tiny-gpt
nnctl train tiny-gpt --preset coherent-small --output tiny-gpt.bin
nnctl chat --model tiny-gpt.bin
```

## When Adding A New Experiment

A good experiment should answer:

- What idea is being tested?
- What command runs it?
- What metrics should a reader compare?
- Which parts are implemented now?
- Which parts are intentionally out of scope?
