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

### TurboQuant

Location: [examples/quantization](../examples/quantization/README.md)

Run:

```bash
./bin/nnctl run turboquant
```

This compares a uniform scalar quantization baseline with a rotated scalar
quantization variant, then reports vector reconstruction and inner-product
metrics. It is a reproducible lab, not a full paper reproduction.

### Tiny GPT

Location: [examples/tiny_gpt](../examples/tiny_gpt/README.md)

Run:

```bash
./bin/nnctl run tiny-gpt -- --prompt "to be" --tokens 80 --seed 42
```

This is a tiny decoder-only Transformer demo that uses this project's matrix
operations for projections and logits. It includes readable sampling and an
output-head-only training path, plus full-model educational training for small
checkpoints.

By default, `--corpus auto` uses a prepared TinyStories slice if it exists,
then prepared Tiny Shakespeare, then the checked-in toy corpus. Prepare the
sourced corpora with:

```bash
./bin/nnctl data tiny-gpt
./bin/nnctl train tiny-gpt --corpus tinystories --output tiny-gpt.bin
./bin/nnctl chat --model tiny-gpt.bin
```

## When Adding A New Experiment

A good experiment should answer:

- What idea is being tested?
- What command runs it?
- What metrics should a reader compare?
- Which parts are implemented now?
- Which parts are intentionally out of scope?
