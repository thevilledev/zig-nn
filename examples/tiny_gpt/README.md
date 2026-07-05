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
- temperature and top-k autoregressive sampling
- data presets for toy, Tiny Shakespeare, TinyStories, and custom text files
- fast demo-corpus output-head training
- readable sampling with a tiny character backoff prior

Full Transformer training is not implemented yet. The file includes
`crossEntropyLoss`, next-token target helpers, and an output-head-only training
path so a future version has a clear place to start, but training the whole
model still needs backward passes for embeddings, layernorm, masked attention,
and the Transformer MLP, or a future autograd path in the library.

See [data/README.md](data/README.md) for source links, dataset notes, and the
intended story arc for improving the model over time.
