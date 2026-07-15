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

The complete NLP progression—from BPE through CRFs, decoding,
encoder-decoder alignment, and contrastive retrieval—is summarized in the
[NLP and Adjacent Learning Roadmap](nlp-roadmap.md).

### Seq2Seq Cross-Attention

Location: [examples/seq2seq](../examples/seq2seq/seq2seq.zig)

Run:

```bash
nnctl run seq2seq
```

This trains decoder queries to translate and reverse short color sequences.
It reports held-out token/exact-sequence accuracy, verifies zero training
readbacks, and prints the target-to-source attention matrix.

### Contrastive Semantic Search

Location: [examples/semantic_search](../examples/semantic_search/semantic_search.zig)

Run:

```bash
nnctl run semantic-search
```

This trains separate query and document encoders with symmetric InfoNCE. It
reports loss, recall@1, mean reciprocal rank, device telemetry, and an exact
top-k cosine ranking for held-out query wording.

### DQN LineWorld

Location: [examples/dqn](../examples/dqn/dqn.zig)

Run:

```bash
nnctl run dqn
```

This trains a small Q-network to navigate a five-state world. It exposes the
full DQN loop: epsilon-greedy exploration, a bounded replay buffer, sampled TD
updates, a separate target network, and one-step Bellman targets. It reports
greedy success, path length, the learned Q-table view, replay/update counts,
and the readbacks used at the host reinforcement-learning boundary.

### Transformer Encoder Context

Location: [examples/transformer_encoder](../examples/transformer_encoder/transformer_encoder.zig)

Run:

```bash
nnctl run transformer-encoder
```

This trains a pre-normalized encoder block to predict the final bit from the
first token. Only the first-token loss contributes a gradient, so successful
classification demonstrates bidirectional context rather than a shortcut at
the final position. It reports held-out accuracy and verifies that the training
path stays device-resident.

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

### CNN Pattern Classifier

Location: [examples/cnn](../examples/cnn/cnn.zig)

Run:

```bash
nnctl run cnn
```

This uses the CPU-first convolution and max-pooling reference to distinguish
synthetic horizontal and vertical patterns. The small data makes kernels,
pooling choices, gradient flow, and the learned spatial bias easy to inspect.

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
