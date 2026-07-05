# Tiny GPT Example

This example is a tiny, readable decoder-only Transformer in the style of
Karpathy's minGPT and nanoGPT demos. It uses this project's `nn.Matrix`
operations for the linear projections and logits, while keeping the
Transformer-specific code local to the example.

Run it with:

```bash
zig build run_tiny_gpt -- --prompt "to be" --tokens 80 --seed 42
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
- tied token embedding output head
- temperature and top-k autoregressive sampling

Training is not implemented yet. The file includes `crossEntropyLoss` and
next-token target helpers so a future version has a clear place to start, but
full training still needs backward passes for embeddings, layernorm, masked
attention, and the Transformer MLP, or a future autograd path in the library.
