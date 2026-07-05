# Tiny GPT Example

This example is a tiny, readable decoder-only Transformer in the style of
Karpathy's minGPT and nanoGPT demos. It uses this project's `nn.Matrix`
operations for the linear projections and logits, while keeping the
Transformer-specific code local to the example.

Run it with:

```bash
zig build run_tiny_gpt -- --prompt "to be" --tokens 80 --seed 42
```

By default the example uses `--corpus auto`: a prepared TinyStories slice if it
exists, then prepared Tiny Shakespeare, then the checked-in toy corpus. It
trains the output head on a small prefix of that corpus, then samples with a
small character backoff prior. This keeps the demo fast while making the output
readable enough to inspect. To see the raw untrained architecture sample, pass
`--no-train --no-corpus-prior`.

For an end-to-end training pass over all Transformer weights, embeddings, layer
norms, and the output head, use `--train-full`. This trains one next-token
context window per step and can save a reusable checkpoint:

```bash
zig build run_tiny_gpt -- --corpus toy --train-full --full-train-steps 200 \
  --full-learning-rate 0.03 --save-checkpoint tiny-gpt.bin
```

Reload a checkpoint for generation:

```bash
zig build run_tiny_gpt -- --load-checkpoint tiny-gpt.bin --no-train \
  --prompt "to be" --tokens 120
```

Run the OpenAI-compatible inference service:

```bash
zig build run_tiny_gpt_openai -- --model tiny-gpt.bin --port 8080
```

The server requires `--model` by default. For quick protocol testing with a
seeded untrained model, pass `--allow-untrained`.

Then call it with an OpenAI-style non-streaming request:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"tiny-gpt-zig","messages":[{"role":"user","content":"to be"}],"max_tokens":80}'
```

Prepare the sourced corpora with:

```bash
make prepare-tiny-gpt-data
```

Then try the progression:

```bash
zig build run_tiny_gpt -- --corpus toy --prompt "to be"
zig build run_tiny_gpt -- --corpus shakespeare --prompt "ROMEO:"
zig build run_tiny_gpt -- --corpus tinystories --prompt "Once upon a time"
```

The default model is intentionally small:

- block size: 16 characters
- layers: 2
- heads: 2
- embedding size: 32
- MLP hidden size: 128
- tokenizer: a tiny built-in character vocabulary

Implemented pieces:

- learned token and position embeddings
- pre-layernorm Transformer blocks
- multi-head causal self-attention
- GPT-style MLP with GELU
- residual connections and final layernorm
- trainable output projection head
- manual next-token backpropagation for the full tiny Transformer
- binary checkpoints for trained model weights
- temperature and top-k autoregressive sampling
- data presets for toy, Tiny Shakespeare, TinyStories, and custom text files
- fast demo-corpus output-head training
- OpenAI-compatible non-streaming `/v1/chat/completions`, `/v1/completions`,
  and `/v1/models` serving example
- readable sampling with a tiny character backoff prior

The full-training path is intentionally educational rather than high-throughput:
it uses explicit backward passes in the example file, trains single context
windows, and keeps the model character-level and small enough to inspect.
The serving path accepts common OpenAI request fields and returns explicit JSON
errors for unsupported streaming, `n > 1`, token-array prompts, and multimodal
message content.

See [data/README.md](data/README.md) for source links, dataset notes, and the
intended story arc for improving the model over time.
