# Tiny GPT Example

This example is a tiny, readable decoder-only Transformer in the style of
Karpathy's minGPT and nanoGPT demos. It uses this project's `nn.Matrix`
operations for the linear projections and logits, while keeping the
Transformer-specific code local to the example.

Run it with:

```bash
bin/nnctl run tiny-gpt -- --prompt "to be" --tokens 80 --seed 42
```

By default the example uses `--corpus auto`: a prepared TinyStories slice if it
exists, then prepared Tiny Shakespeare, then the checked-in toy corpus. The fast
run trains only the output head on a small prefix, then samples with a small
character backoff prior. That path is an architecture/readability demo, not
evidence that the model itself has learned coherent language. To inspect the
model without the readability prior, pass `--no-corpus-prior`; to see the raw
untrained architecture sample, pass `--no-train --no-corpus-prior`.

For an end-to-end training pass over all Transformer weights, embeddings, layer
norms, and the output head, use `nnctl train tiny-gpt`. This trains multiple
next-token context windows per step, uses AdamW by default, applies a repeatable
train/eval split, prints validation loss, embeds run metadata in the checkpoint,
and can write a JSON summary for comparing runs:

```bash
bin/nnctl train tiny-gpt --corpus tinystories --output tiny-gpt.bin \
  --steps 5000 --block-size 32 --layers 4 --heads 4 --embd 64 \
  --eval-split 0.05 --lr-schedule cosine --summary-path runs/tiny-gpt.json
```

For the smallest recommended model-only coherence baseline, prepare TinyStories
and use the `coherent-small` preset. It disables the corpus prior for the sample
and expands to a 4-layer, 4-head, 128-channel, 64-character-context model with
random mini-batches and AdamW:

```bash
bin/nnctl data tiny-gpt
bin/nnctl train tiny-gpt --preset coherent-small --output tiny-gpt.bin
```

Resume a checkpoint for more training:

```bash
bin/nnctl train tiny-gpt --resume tiny-gpt.bin --steps 2000
```

Reload a checkpoint for generation:

```bash
bin/nnctl run tiny-gpt -- --load-checkpoint tiny-gpt.bin --no-train \
  --prompt "to be" --tokens 120 --no-corpus-prior
```

Run the OpenAI-compatible inference service:

```bash
bin/nnctl run tiny-gpt-openai -- --model tiny-gpt.bin --port 8080
```

The server requires `--model` by default. For quick protocol testing with a
seeded untrained model, pass `--allow-untrained`.

Or run the inference server and a small local browser chat app together:

```bash
bin/nnctl chat --model tiny-gpt.bin
```

Then call it with an OpenAI-style non-streaming request:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"tiny-gpt-zig","messages":[{"role":"user","content":"to be"}],"max_tokens":80}'
```

Prepare the sourced corpora with:

```bash
bin/nnctl data tiny-gpt
```

Then try the progression:

```bash
bin/nnctl run tiny-gpt -- --corpus toy --prompt "to be"
bin/nnctl train tiny-gpt --corpus shakespeare --output shakespeare-gpt.bin
bin/nnctl train tiny-gpt --corpus tinystories --output tinystories-gpt.bin
```

The default model is intentionally small:

- block size: 16 characters
- layers: 2
- heads: 2
- embedding size: 32
- MLP hidden size: 128
- tokenizer: a printable-ASCII character vocabulary plus newline

Better checkpoints come from matching capacity, data, and training time:

- increase `--block-size` so the model sees longer context
- increase `--layers`, `--heads`, and `--embd` for more parameters
- increase `--train-chars` and use `bin/nnctl data tiny-gpt` for real corpora
- increase `--steps` and watch both training and validation loss
- start from `--preset coherent-small` when you want model-only text, not a
  readability prior
- keep `--summary-path` outputs when comparing runs
- try `--lr-schedule cosine --warmup-steps 50 --min-learning-rate 0.001`
- resume with `--resume` instead of starting from scratch
- use `--optimizer adamw --weight-decay 0.01` unless you are comparing SGD

Implemented pieces:

- learned token and position embeddings
- pre-layernorm Transformer blocks
- multi-head causal self-attention
- GPT-style MLP with GELU
- residual connections and final layernorm
- trainable output projection head
- manual next-token backpropagation for the full tiny Transformer
- binary checkpoints for trained model weights and run metadata
- temperature and top-k autoregressive sampling
- data presets for toy, Tiny Shakespeare, TinyStories, and custom text files
- deterministic JSON run summaries for experiment comparison
- fast demo-corpus output-head training
- seeded random-window mini-batch training with averaged/clipped gradients
- OpenAI-compatible non-streaming `/v1/chat/completions`, `/v1/completions`,
  and `/v1/models` serving example
- readable sampling with a tiny character backoff prior

The full-training path is intentionally educational rather than high-throughput:
it uses explicit backward passes in the example file, trains small batches of
context windows, and keeps the model character-level and small enough to inspect.
The serving path accepts common OpenAI request fields and returns explicit JSON
errors for unsupported streaming, `n > 1`, token-array prompts, and multimodal
message content.

See [data/README.md](data/README.md) for source links, dataset notes, and the
intended story arc for improving the model over time.
