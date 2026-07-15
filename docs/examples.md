# Examples

Examples are small programs that make one behavior easy to inspect. Reusable
library code should live in `src/`; example-specific glue and notes should stay
with the example.

## Quick Runs

```bash
nnctl run quick
```

This command runs the quick examples. On machines without the relevant GPU
toolchain, prefer the individual CPU examples below.

## Suggested Learning Path

The examples are intentionally small enough to read beside the reusable code:

1. Start with Simple XOR, XOR training, classification, and regression for
   forward passes, losses, and backpropagation.
2. Run Optimizer Lab beside `src/modules.zig` and `src/training.zig` to compare
   the same MLP under SGD, momentum, and AdamW.
3. Use CNN for spatial inductive bias, then Autoencoder for denoising and a
   learned bottleneck representation.
4. Train Speech Commands to see WAV decoding, log-mel features, a
   speaker-disjoint split, checkpointing, and closed-set audio classification.
5. Run Tokenizer Lab to compare raw bytes with learned BPE merges and inspect
   the compression-versus-vocabulary tradeoff.
6. Use Padding Masks to batch variable-length sequences without learning from
   filler values.
7. Run Word2Vec to see how a prediction objective turns co-occurrence into
   geometric neighborhoods.
8. Use Text Classifier to combine learned embeddings, padding masks, multi-head
   attention, pooling, and sparse cross-entropy.
9. Use Sequence Tagging to compare independent token decisions with globally
   normalized CRF and Viterbi decoding.
10. Run Decoding Lab to contrast greedy choice, fixed top-k truncation, adaptive
   nucleus top-p truncation, and repetition penalties.
11. Use Seq2Seq to see decoder queries learn an alignment over separate encoder
   memory while translating and reordering tokens.
12. Run Semantic Search to train two vocabularies into a shared embedding space
    with symmetric InfoNCE and rank documents by cosine similarity.
13. Compare GRU Sequence with Transformer Encoder. The GRU carries state through
    time; the encoder uses unmasked attention to retrieve context from any token.
14. Finish with DQN to see how an MLP, replay memory, exploration, Bellman
   targets, and a target network fit into an agent-environment loop.

The CNN and CRF tagging examples use inspectable CPU-first references. The
optimizer, padding-mask, Word2Vec, speech-command, text-classifier, Seq2Seq, semantic-search,
autoencoder, GRU, Transformer encoder, and DQN models accept `--backend`; use
`nnctl` for automatic backend flags or pass `-Dgpu=auto` to direct Zig runs.
DQN intentionally crosses the host boundary for environment interaction,
discrete action choice, and Bellman-target construction. Semantic Search reads
final embeddings back once for exact cosine ranking; its InfoNCE training stays
on the selected device.

## Catalog

| Example | Zig step | nnctl command | Purpose |
| --- | --- | --- | --- |
| Simple XOR | `zig build run_simple_xor` | `nnctl run simple-xor` | Minimal forward pass through an XOR-shaped network |
| XOR training | `zig build run_xor_training` | `nnctl run xor-training` | Backpropagation on XOR and optional model output |
| Binary classification | `zig build run_binary_classification` | `nnctl run binary-classification` | Circular decision-boundary classification |
| Regression | `zig build run_regression` | `nnctl run regression` | Nonlinear function approximation |
| Gated network | `zig build run_gated_network` | `nnctl run gated-network` | GLU/SwiGLU-style gated layer behavior |
| Network visualisation | `zig build run_network_visualisation` | `nnctl run network-visualisation` | Unicode terminal topology diagram of a network shape |
| Backend demo | `zig build run_backend_demo` | `nnctl run backend-demo` | Basic matrix operations through the public API |
| Backend training | `zig build run_backend_training` | `nnctl run backend-training` | Tiny supervised training loop through `BackendMatrix` |
| Optimizer lab | `zig build run_optimizer_lab -Dgpu=auto` | `nnctl run optimizer-lab` | Device-resident MLP comparison of SGD, momentum, and AdamW |
| Tokenizer lab | `zig build run_tokenizer_lab` | `nnctl run tokenizer-lab` | Byte fallback, learned BPE merges, vocabulary growth, and round trips |
| Padding masks | `zig build run_padding_masks -Dgpu=auto` | `nnctl run padding-masks` | Batched matmul, masked softmax, dropout, and sparse loss for padded sequences |
| Word2Vec | `zig build run_word2vec -Dgpu=auto` | `nnctl run word2vec` | Skip-gram pairs, unigram negative sampling, and learned embedding neighborhoods |
| Speech commands | `zig build run_speech_commands -Dgpu=auto -Doptimize=ReleaseFast` | `nnctl run speech-commands` | Closed-set recognition of eight spoken words with log-mel features and a custom MLP |
| Text classifier | `zig build run_text_classifier -Dgpu=auto` | `nnctl run text-classifier` | Padding-aware multi-head encoder trained on contextual sentiment phrases |
| Sequence tagging | `zig build run_sequence_tagging` | `nnctl run sequence-tagging` | Forward-backward CRF training and Viterbi decoding for BIO tags |
| Decoding lab | `zig build run_decoding_lab` | `nnctl run decoding-lab` | Greedy, top-k, nucleus top-p, repetition penalties, and seeded sampling |
| Seq2Seq | `zig build run_seq2seq -Dgpu=auto` | `nnctl run seq2seq` | Cross-attention learns translation, reordering, and target-to-source alignment |
| Semantic search | `zig build run_semantic_search -Dgpu=auto` | `nnctl run semantic-search` | Symmetric InfoNCE, dual encoders, in-batch negatives, and top-k cosine retrieval |
| CNN | `zig build run_cnn` | `nnctl run cnn` | Convolution, ReLU, max-pool, and softmax on synthetic image patterns |
| Autoencoder | `zig build run_autoencoder -Dgpu=auto` | `nnctl run autoencoder` | Denoising and two-dimensional latent representations of tiny images |
| GRU sequence | `zig build run_gru_sequence -Dgpu=auto` | `nnctl run gru-sequence` | Backpropagation through time on a selective-memory task |
| Transformer encoder | `zig build run_transformer_encoder -Dgpu=auto` | `nnctl run transformer-encoder` | Unmasked attention retrieves a final-token value from the first position |
| DQN | `zig build run_dqn -Dgpu=auto` | `nnctl run dqn` | Replay-based value learning and epsilon-greedy control in LineWorld |
| GPU demo | `zig build run_gpu -Dgpu=auto` | `nnctl run gpu` | Backend matrix operations on Metal, CUDA, or ROCm |
| GPU benchmark | `zig build run_gpu_benchmark -Dgpu=auto -Doptimize=ReleaseFast` | `nnctl run gpu-benchmark` | CPU vs GPU matrix multiplication timing |
| TurboQuant | `zig build run_turboquant` | `nnctl run turboquant` | Quantization experiment with comparable metrics |
| Tiny GPT | `zig build run_tiny_gpt` | `nnctl run tiny-gpt` | Tiny decoder-only Transformer demo with corpus presets |
| Serving | `zig build run_serving` | `nnctl run serving` | HTTP prediction server for a saved XOR model |
| MNIST | `zig build run_mnist` | `nnctl run mnist` | Digit recognition with external MNIST data |

## Examples With Extra Notes

- [Quantization examples](../examples/quantization/README.md)
- [Tiny GPT example](../examples/tiny_gpt/README.md)
- [Serving example](../examples/serving/README.md)
- [Speech Commands example](../examples/speech_commands/README.md)

## External Inputs

MNIST needs downloaded data:

```bash
nnctl data mnist
nnctl run mnist
```

Speech Commands downloads an external WAV dataset, trains a checkpoint, and
then recognizes one or more clips:

```bash
nnctl data speech-commands
nnctl train speech-commands --output speech-commands.bin
nnctl run speech-commands -- --model speech-commands.bin --input clip.wav
```

The serving example needs a saved model:

```bash
nnctl run xor-training -- --output=xor_model.bin
nnctl run serving
```

Then send a request:

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"input": [0.0, 1.0], "batch_size": 1}'
```

Tiny GPT can run with the checked-in toy corpus, or with sourced corpora:

```bash
nnctl data tiny-gpt
nnctl train tiny-gpt --preset coherent-small --output tiny-gpt.bin
nnctl chat --model tiny-gpt.bin
```
