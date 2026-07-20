# TinyGPT Experiment

This experiment is a tiny, readable decoder-only Transformer in the style of
Karpathy's minGPT and nanoGPT demos. Its checkpoint/config/model implementation
lives in `src/tiny_gpt`, and `nn.Inference.TextSession` runs the reusable device
decoder on CPU, Metal, CUDA, or ROCm. Experiment imports and commands remain as
compatibility paths.

Run it with:

```bash
nnctl run tiny-gpt -- --prompt "to be" --tokens 80 --seed 42
```

By default the experiment uses `--corpus auto`: a prepared TinyStories slice if it
exists, then prepared Tiny Shakespeare, then the checked-in toy corpus. The fast
run trains only the output head on a small prefix, then samples with a small
character backoff prior. That path is an architecture/readability demo, not
evidence that the model itself has learned coherent language. To inspect the
model without the readability prior, pass `--no-corpus-prior`; to see the raw
untrained architecture sample, pass `--no-train --no-corpus-prior`.

For an end-to-end training pass over all Transformer weights, embeddings, layer
norms, and the output head, use `nnctl train tiny-gpt`. This trains multiple
next-token context windows per step, uses AdamW by default, applies a repeatable
train/eval split, prints validation loss and perplexity, embeds run metadata in
the checkpoint, and can write a JSON summary for comparing runs:

```bash
nnctl train tiny-gpt --corpus tinystories --output tiny-gpt.bin \
  --steps 5000 --block-size 32 --layers 4 --heads 4 --embd 64 \
  --eval-split 0.05 --lr-schedule cosine --summary-path runs/tiny-gpt.json
```

For the smallest recommended model-only coherence baseline, prepare TinyStories
and use the `coherent-small` preset. It disables the corpus prior for the sample
and expands to a 4-layer, 4-head, 128-channel, 64-character-context model with
random mini-batches and AdamW:

```bash
nnctl data tiny-gpt
nnctl train tiny-gpt --preset coherent-small --output tiny-gpt.bin
```

Resume a checkpoint for more training:

```bash
nnctl train tiny-gpt --resume tiny-gpt.bin --steps 2000
```

Reload a checkpoint for generation:

```bash
nnctl run tiny-gpt -- --load-checkpoint tiny-gpt.bin --no-train \
  --prompt "to be" --tokens 120 --top-k 16 --top-p 0.9 --no-corpus-prior
```

Run full-model SGD training and KV-cached generation on the selected tensor
backend:

```bash
zig build run_tiny_gpt -Dgpu=auto -- \
  --train-full --backend auto --optimizer sgd --full-batch-size 1 \
  --weight-decay 0 --no-corpus-prior --prompt "to be"
```

`--backend metal`, `--backend cuda`, and `--backend rocm` require the requested
accelerator; `--backend auto` may fall back to CPU. Device training currently
supports SGD with one context window per update and no weight decay. The legacy
trainer continues to provide AdamW, gradient clipping, and averaged
mini-batches.

Inspect and serve a trained checkpoint through the unified inference path:

```bash
nnctl model inspect tiny-gpt.bin
nnctl serve --model tiny-gpt.bin --gpu auto --port 8080
```

The compatibility spelling remains available:

```bash
nnctl run tiny-gpt-openai -- --model tiny-gpt.bin --backend auto --port 8080
```

The server requires `--model` by default. The compatibility command accepts
`--allow-untrained` for quick protocol testing with a seeded untrained model.
Only `auto` can fall back to CPU; explicit accelerator selection fails when it
is unavailable.

Or run the inference server and a small local browser chat app together:

```bash
nnctl chat --model tiny-gpt.bin --gpu auto
```

Then call it with an OpenAI-style request:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"tiny-gpt-zig","messages":[{"role":"user","content":"to be"}],"max_tokens":80}'
```

`stream: true` returns an SSE-framed completion followed by `[DONE]`:

```bash
curl -N http://127.0.0.1:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"tiny-gpt-zig","prompt":"to be","max_tokens":80,"stream":true}'
```

The v1 educational server handles one active generation at a time and bounds
request bodies. It completes generation before writing the single SSE payload,
so this protocol form is useful for client compatibility but is not yet
incremental token streaming or production multi-tenant serving.

Prepare the sourced corpora with:

```bash
nnctl data tiny-gpt
```

Then try the progression:

```bash
nnctl run tiny-gpt -- --corpus toy --prompt "to be"
nnctl train tiny-gpt --corpus shakespeare --output shakespeare-gpt.bin
nnctl train tiny-gpt --corpus tinystories --output tinystories-gpt.bin
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
- increase `--train-chars` and use `nnctl data tiny-gpt` for real corpora
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
- temperature, top-k, and nucleus top-p autoregressive sampling
- backend-selectable full-model SGD training with checkpoint-compatible QKV
  split/merge
- native per-layer KV caching for backend-selected generation
- data presets for toy, Tiny Shakespeare, TinyStories, and custom text files
- deterministic JSON run summaries for experiment comparison
- fast demo-corpus output-head training
- seeded random-window mini-batch training with averaged/clipped gradients
- OpenAI-compatible non-streaming and SSE-framed `/v1/chat/completions`,
  `/v1/completions`, and `/v1/models` serving
- readable sampling with a tiny character backoff prior

The full-training path is intentionally educational rather than high-throughput:
it uses explicit backward passes in the experiment files, trains small batches of
context windows, and keeps the model character-level and small enough to inspect.
The serving path accepts common OpenAI request fields and returns explicit JSON
errors for `n > 1`, token-array prompts, and multimodal message content. It
reuses one loaded `TextSession`, including device weights, KV cache, and
sampling workspaces, for sequential requests.

See [data/README.md](data/README.md) for source links, dataset notes, and the
intended story arc for improving the model over time.
