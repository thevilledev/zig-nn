# NLP And Adjacent Learning Roadmap

The NLP path is organized as small, inspectable lessons. Reusable mechanics
live in `src/`; every feature has a runnable example that exposes the data,
objective, gradients, and evaluation signal.

## Current Learning Path

| Lesson | Run | Reusable feature | What to inspect |
| --- | --- | --- | --- |
| Tokenizer lab | `nnctl run tokenizer-lab` | Byte fallback and learned byte-pair encoding | Merge learning, vocabulary growth, unknown-free byte round trips |
| Padding masks | `nnctl run padding-masks` | Batched matmul, masked softmax/backward, dropout, masked sparse loss, head layouts | Why padded values must not affect attention or pooling |
| Word2Vec | `nnctl run word2vec` | Skip-gram pairs, unigram negative sampling, two embedding tables | How prediction turns co-occurrence into vector neighborhoods |
| Speech commands | `nnctl run speech-commands` | PCM16 WAV decoding, resampling, FFT, and pooled log-mel features | Closed-set keyword recognition, speaker-disjoint evaluation, and checkpoint compatibility |
| Text classifier | `nnctl run text-classifier` | Padding-aware multi-head encoder | Contextual classification with learned embeddings and masked pooling |
| Sequence tagging | `nnctl run sequence-tagging` | Linear-chain CRF, forward-backward, Viterbi | Local emission scores versus globally valid BIO sequences |
| Decoding lab | `nnctl run decoding-lab` | Greedy, temperature, top-k, nucleus top-p, repetition penalty | How truncation changes a next-token distribution |
| Seq2Seq | `nnctl run seq2seq` | Cross-attention with query and memory gradients | Learned target-to-source alignment during translation and reordering |
| Semantic search | `nnctl run semantic-search` | Symmetric InfoNCE and deterministic top-k cosine ranking | Dual encoders, in-batch negatives, recall@1, and MRR |
| Tiny GPT | `nnctl run tiny-gpt` | Decoder-only Transformer and KV cache | Autoregressive training, checkpointing, serving, and sampling |

Most examples use tiny synthetic or checked-in datasets so the lesson remains
repeatable. Speech Commands uses a reproducible external teaching dataset.
Their metrics demonstrate mechanics, not production model quality.

## GPU Boundary

Using `Tensor` means an operation is backend-neutral; it is GPU-accelerated
only when the build enables a GPU backend and `Device` actually selects it.
Run an example with `-Dgpu=auto` through Zig or use its `nnctl` command, which
supplies the catalogued backend default.

| Area | Device-resident work | Intentional host work |
| --- | --- | --- |
| Padding and multi-head attention | Batched matmul, masked softmax and backward, head transforms, dropout, loss | Length-to-mask construction before upload |
| Word2Vec | Embedding lookup/backward and optimizer updates | Pair generation and negative sampling before upload; metric reporting |
| Speech commands | MLP forward/backward and AdamW updates | WAV decoding, log-mel extraction, dataset splitting, feature normalization, and metric reporting |
| Text classifier | Encoder forward/backward, pooling, classifier, optimizer | Tokenization and dataset assembly |
| CRF tagging | None; the readable reference is CPU-first | Forward-backward, Viterbi, and training |
| Cross-attention Seq2Seq | Projections, attention, gradients, classifier, optimizer | Synthetic sequence construction and final alignment display |
| Semantic search | Dual encoders, symmetric InfoNCE, gradients, optimizer | Final embedding readback and top-k cosine ranking |
| Decoding and BPE | None; both are discrete host algorithms | Merge learning, token choice, and sampling |

Explicit `metal`, `cuda`, and `rocm` device preferences fail if unavailable;
`auto` may fall back to CPU. See [GPU and Backend Notes](gpu.md) for toolchain
requirements and verification commands.

## Next Roadmap Set

These are the next useful increments. Each item includes the example required
to make it a learning feature rather than an isolated API.

1. **Teacher-forced encoder-decoder stack and beam search** — compose encoder,
   causal decoder, cross-attention, padding masks, and length-normalized beam
   search. Add a `machine-translation` example that compares greedy and beam
   outputs and visualizes alignments.
2. **Sequence bucketing and dynamic batches** — build batches by similar
   lengths, generate encoder/decoder masks, and accumulate gradients across
   microbatches. Extend `seq2seq` with padding-efficiency and throughput
   metrics.
3. **Persistent subword vocabularies and special-token policy** — serialize BPE
   merges, reserve BOS/EOS/PAD/UNK IDs, and make training/inference vocabulary
   compatibility explicit. Extend `tokenizer-lab` and TinyGPT checkpoint tests.
4. **Approximate nearest-neighbor indexing** — add an inspectable HNSW or
   inverted-file index with recall-versus-latency measurements. Add an
   `ann-search` example that compares exact cosine ranking with the index.
5. **CTC alignment for speech or OCR** — implement blank-aware forward-backward
   and greedy/prefix decoding. Add a `ctc-alignment` example over synthetic
   noisy character frames.
6. **One-dimensional sequence models for time series** — add causal Conv1d and
   masked forecasting losses. Add a `time-series` example with rolling-window
   evaluation and a persistence baseline.
7. **Graph message passing** — add neighborhood aggregation and sparse graph
   batches. Add a `node-classification` example that contrasts an MLP with a
   small graph convolutional network.

Mixed precision, fused attention, and accelerator-native top-k are backend
optimization tracks that can advance alongside these learning features.
