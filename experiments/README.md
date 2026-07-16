# Experiments

Each experiment turns one learning question into runnable code and observable
evidence. Reusable implementation belongs in [`src/`](../src/); experiment-only
data construction, training loops, reporting, and notes stay here.

A useful experiment should make six things clear:

1. the question it asks;
2. the command that runs it;
3. the reusable code it exercises;
4. the metric, prediction, or trace that answers the question;
5. the boundary between what is implemented and what is intentionally omitted;
6. the research or technical reference behind the idea.

## Running Experiments

Use `nnctl` to apply the catalogued optimization and backend defaults:

```bash
nnctl list experiments
nnctl run simple-xor
nnctl run quick
nnctl experiments
```

`nnctl run quick` runs the short experiments that need no downloaded dataset or
saved model. `nnctl experiments` builds every experiment without running it.
The compatibility commands `nnctl examples` and `zig build examples` remain
available, but new documentation and scripts should use “experiments.”

Every individual command maps to a stable Zig step. For example,
`nnctl run optimizer-lab` maps to `zig build run_optimizer_lab -Dgpu=auto`.
See [Getting Started](../docs/getting-started.md) for direct Zig commands and
[GPU and Backend Notes](../docs/gpu.md) for backend requirements.

XOR Training, Regression, and Binary Classification also expose structured
live evidence in the local browser lab:

```bash
mise run lab
```

Their normal terminal output remains the default. The lab invokes them with
`--format ndjson`; see the [learning lab guide](../docs/learning-lab.md) for the
event contract and extension steps.

## Suggested Learning Routes

- **Foundations:** Simple XOR → XOR Training → Binary Classification →
  Regression → Gated Network. Read each program beside `matrix.zig`,
  `layer.zig`, and `network.zig`.
- **Training and runtime:** Backend Demo → Backend Training → Optimizer Lab →
  Padding Masks → GPU Demo. This route exposes storage, transfer, optimizer,
  and device-residency decisions.
- **Vision and audio:** CNN → MNIST → Autoencoder → Speech Commands. Compare a
  readable spatial reference, a real image dataset, representation learning,
  and an external audio pipeline.
- **Language and sequences:** Tokenizer Lab → Word2Vec → Text Classifier →
  Sequence Tagging → Decoding Lab → Seq2Seq → Semantic Search, then compare GRU
  Sequence with Transformer Encoder before exploring TinyGPT.
- **Applied systems:** DQN, TurboQuant, GPU Benchmark, Serving, and the TinyGPT
  OpenAI-compatible server show where learning code meets evaluation,
  compression, performance, and interfaces.

## Foundations

| Experiment and run command | Question and evidence | Code to read | Background |
| --- | --- | --- | --- |
| **Simple XOR** — `nnctl run simple-xor` | What does an untrained forward pass do? Inspect the XOR truth-table predictions. | [entrypoint](simple_xor/simple_xor.zig), [matrix](../src/matrix.zig), [layer](../src/layer.zig), [network](../src/network.zig) | [Neural-network foundations](../docs/research.md#neural-network-foundations) |
| **XOR Training** — `nnctl run xor-training` | Can backpropagation fit XOR? Compare loss and predictions before and after training. | [entrypoint](xor_training/xor_training.zig), [network](../src/network.zig) | [Neural-network foundations](../docs/research.md#neural-network-foundations) |
| **Binary Classification** — `nnctl run binary-classification` | Can a small network learn a circular decision boundary? Inspect training loss and classification results. | [entrypoint](binary_classification/binary_classification.zig), [network](../src/network.zig) | [Neural-network foundations](../docs/research.md#neural-network-foundations) |
| **Regression** — `nnctl run regression` | Can the same primitives approximate a nonlinear function? Compare targets with predictions. | [entrypoint](regression/regression.zig), [network](../src/network.zig) | [Neural-network foundations](../docs/research.md#neural-network-foundations) |
| **Gated Network** — `nnctl run gated-network` | How do GLU and SwiGLU paths change a layer’s forward computation? Inspect the gated outputs. | [entrypoint](gated_network/gated_network.zig), [layer](../src/layer.zig), [activation](../src/activation.zig) | [Activations and gating](../docs/research.md#activations-and-gating) |
| **Network Visualisation** — `nnctl run network-visualisation` | How do several network shapes look before training? Inspect the terminal topology diagrams. | [entrypoint](network_visualisation/network_visualisation.zig), [visualiser](../src/visualiser.zig) | [Architecture](../docs/architecture.md) |

## Training and Runtime

| Experiment and run command | Question and evidence | Code to read | Background |
| --- | --- | --- | --- |
| **Backend Demo** — `nnctl run backend-demo` | Do public matrix operations behave consistently through the backend abstraction? Inspect operation results and selected backend. | [entrypoint](backend_demo/backend_demo.zig), [backend](../src/backend.zig) | [GPU and backend work](../docs/research.md#gpu-and-backend-work) |
| **Backend Training** — `nnctl run backend-training` | Can parameters remain on a backend through a small supervised loop? Inspect loss reduction and explicit synchronization. | [entrypoint](backend_training/backend_training.zig), [backend](../src/backend.zig) | [GPU and backend work](../docs/research.md#gpu-and-backend-work) |
| **Optimizer Lab** — `nnctl run optimizer-lab` | How do SGD, momentum, and AdamW behave from identical initialization? Compare loss reduction and held-out accuracy. | [entrypoint](optimizer_lab/optimizer_lab.zig), [modules](../src/modules.zig), [training](../src/training.zig) | [Optimization](../docs/research.md#optimization) |
| **Padding Masks** — `nnctl run padding-masks` | How can variable-length sequences share a batch without learning from filler values? Inspect masked probabilities, loss, and gradients. | [entrypoint](padding_masks/padding_masks.zig), [tensor](../src/tensor.zig), [transformer](../src/transformer.zig) | [Transformers](../docs/research.md#transformers-and-tiny-gpt) |
| **GPU Demo** — `nnctl run gpu` | Which backend is selected, and do basic operations agree with CPU behavior? Inspect backend identity and matrix results. | [entrypoint](gpu/gpu.zig), [backend API](../src/backend.zig), [GPU notes](../docs/gpu.md) | [GPU and backend work](../docs/research.md#gpu-and-backend-work) |

## Vision and Audio

| Experiment and run command | Question and evidence | Code to read | Background |
| --- | --- | --- | --- |
| **CNN** — `nnctl run cnn` | Does convolution learn orientation more directly than flattened features? Inspect held-out pattern accuracy and gradients. | [entrypoint](cnn/cnn.zig), [spatial layers](../src/spatial.zig) | [Representations, vision, and audio](../docs/research.md#representations-vision-and-audio) |
| **MNIST** — `nnctl run mnist` | Can the CPU fundamentals classify handwritten digits? Inspect loss and test accuracy on downloaded MNIST data. | [entrypoint](mnist/mnist.zig), [network](../src/network.zig) | [Representations, vision, and audio](../docs/research.md#representations-vision-and-audio) |
| **Autoencoder** — `nnctl run autoencoder` | Can a two-value bottleneck preserve small image families while removing noise? Inspect reconstruction error, pixel accuracy, and latent centers. | [entrypoint](autoencoder/autoencoder.zig), [modules](../src/modules.zig), [training](../src/training.zig) | [Representations, vision, and audio](../docs/research.md#representations-vision-and-audio) |
| **Speech Commands** — `nnctl run speech-commands` | Can log-mel features and a small MLP recognize eight spoken words across speakers? Inspect validation accuracy, confusion counts, and checkpoint compatibility. | [entrypoint](speech_commands/speech_commands.zig), [notes](speech_commands/README.md), [audio](../src/audio.zig), [training](../src/training.zig) | [Representations, vision, and audio](../docs/research.md#representations-vision-and-audio) |

## Language, Retrieval, and Sequence Models

| Experiment and run command | Question and evidence | Code to read | Background |
| --- | --- | --- | --- |
| **Tokenizer Lab** — `nnctl run tokenizer-lab` | What does learned BPE trade for a larger vocabulary? Compare merge growth, compression, and exact byte round trips. | [entrypoint](tokenizer_lab/tokenizer_lab.zig), [text](../src/text.zig) | [NLP and retrieval](../docs/research.md#nlp-structured-prediction-and-retrieval) |
| **Word2Vec** — `nnctl run word2vec` | Can a prediction objective turn co-occurrence into geometric neighborhoods? Inspect loss and nearest words. | [entrypoint](word2vec/word2vec.zig), [embeddings](../src/embeddings.zig) | [NLP and retrieval](../docs/research.md#nlp-structured-prediction-and-retrieval) |
| **Text Classifier** — `nnctl run text-classifier` | Can masked multi-head attention distinguish contextual sentiment? Inspect held-out accuracy and phrase predictions. | [entrypoint](text_classifier/text_classifier.zig), [embeddings](../src/embeddings.zig), [transformer](../src/transformer.zig) | [Transformers](../docs/research.md#transformers-and-tiny-gpt) |
| **Sequence Tagging** — `nnctl run sequence-tagging` | How do independent token scores become a globally valid BIO sequence? Inspect likelihood, marginals, and Viterbi output. | [entrypoint](sequence_tagging/sequence_tagging.zig), [structured prediction](../src/structured.zig) | [NLP and retrieval](../docs/research.md#nlp-structured-prediction-and-retrieval) |
| **Decoding Lab** — `nnctl run decoding-lab` | How do greedy, top-k, nucleus, temperature, and repetition penalties reshape token choice? Compare seeded outputs. | [entrypoint](decoding_lab/decoding_lab.zig), [decoding](../src/decoding.zig) | [NLP and retrieval](../docs/research.md#nlp-structured-prediction-and-retrieval) |
| **Seq2Seq** — `nnctl run seq2seq` | Can decoder queries learn translation, reordering, and source alignment? Inspect token/exact accuracy and the attention matrix. | [entrypoint](seq2seq/seq2seq.zig), [transformer](../src/transformer.zig) | [NLP and retrieval](../docs/research.md#nlp-structured-prediction-and-retrieval) |
| **Semantic Search** — `nnctl run semantic-search` | Can paired encoders learn a shared retrieval space with in-batch negatives? Inspect recall@1, MRR, and cosine rankings. | [entrypoint](semantic_search/semantic_search.zig), [retrieval](../src/retrieval.zig), [embeddings](../src/embeddings.zig) | [NLP and retrieval](../docs/research.md#nlp-structured-prediction-and-retrieval) |
| **GRU Sequence** — `nnctl run gru-sequence` | Can a recurrent state retain one marked bit through distractors? Inspect held-out accuracy and device telemetry. | [entrypoint](gru_sequence/gru_sequence.zig), [recurrent](../src/recurrent.zig) | [Sequence models and reinforcement learning](../docs/research.md#sequence-models-and-reinforcement-learning) |
| **Transformer Encoder** — `nnctl run transformer-encoder` | Can bidirectional attention retrieve a value from a distant token? Inspect held-out accuracy and training readbacks. | [entrypoint](transformer_encoder/transformer_encoder.zig), [transformer](../src/transformer.zig) | [Transformers](../docs/research.md#transformers-and-tiny-gpt) |
| **TinyGPT** — `nnctl run tiny-gpt` | How do a decoder-only Transformer, KV cache, training loop, and sampler fit together? Inspect loss and seeded generated text. | [entrypoint](tiny_gpt/tiny_gpt.zig), [notes](tiny_gpt/README.md), [model](tiny_gpt/model.zig), [transformer](../src/transformer.zig) | [Transformers and TinyGPT](../docs/research.md#transformers-and-tiny-gpt) |

## Applied Systems

| Experiment and run command | Question and evidence | Code to read | Background |
| --- | --- | --- | --- |
| **DQN** — `nnctl run dqn` | Can replay, exploration, Bellman targets, and a target network solve LineWorld? Inspect greedy success, path length, and learned Q-values. | [entrypoint](dqn/dqn.zig), [reinforcement](../src/reinforcement.zig), [modules](../src/modules.zig) | [Sequence models and reinforcement learning](../docs/research.md#sequence-models-and-reinforcement-learning) |
| **TurboQuant** — `nnctl run turboquant` | How does rotation change scalar-quantization error? Compare reconstruction, inner-product, and compression metrics. | [entrypoint](quantization/turboquant.zig), [notes](quantization/README.md), [quantization](../src/quantization.zig) | [Quantization](../docs/research.md#quantization) |
| **GPU Benchmark** — `nnctl run gpu-benchmark` | When does backend matrix multiplication beat CPU execution, and at what numerical error? Compare timings and sampled error. | [entrypoint](gpu_benchmark/gpu_benchmark.zig), [backend](../src/backend.zig), [benchmark guide](../docs/benchmarks.md) | [GPU and backend work](../docs/research.md#gpu-and-backend-work) |
| **Serving** — `nnctl run serving` | What is the smallest useful prediction service around a saved model? Inspect HTTP validation and prediction responses. | [entrypoint](serving/server.zig), [notes](serving/README.md), [inference service](../src/inference_service.zig) | [Serving and tooling](../docs/research.md#serving-and-tooling) |
| **TinyGPT OpenAI Server** — `nnctl run tiny-gpt-openai` | How can the local TinyGPT model expose OpenAI-compatible model and chat endpoints? Inspect `/v1/models` and `/v1/chat/completions` responses. | [entrypoint](tiny_gpt/openai_server.zig), [notes](tiny_gpt/README.md), [model](tiny_gpt/model.zig) | [Serving and tooling](../docs/research.md#serving-and-tooling) |

## Data and Checkpoints

Most experiments are self-contained. These workflows need local files:

### MNIST

```bash
nnctl data mnist
nnctl run mnist
```

### Speech Commands

```bash
nnctl data speech-commands
nnctl train speech-commands --output speech-commands.bin
nnctl run speech-commands -- --model speech-commands.bin --input clip.wav
```

The data command downloads the external Mini Speech Commands dataset. Training
uses a deterministic speaker-disjoint split; see the
[experiment notes](speech_commands/README.md) for labels and input details.

### Serving

```bash
nnctl run xor-training -- --output=xor_model.bin
nnctl run serving
```

The server loads `xor_model.bin` by default. Its [local README](serving/README.md)
contains request examples and the response schema.

### TinyGPT

The checked-in toy corpus works offline. Optional sourced corpora and a trained
checkpoint use:

```bash
nnctl data tiny-gpt
nnctl train tiny-gpt --preset coherent-small --output tiny-gpt.bin
nnctl chat --model tiny-gpt.bin
```

See the [TinyGPT notes](tiny_gpt/README.md) for presets, corpus selection,
checkpoint behavior, and serving commands.

## Adding an Experiment

- Start with one precise question and a deterministic small dataset when
  possible.
- Put reusable operations, modules, and algorithms in `src/`; keep scenario
  construction and reporting in the experiment directory.
- Add the executable and acceptance test to `build.zig` and the command/defaults
  to the `nnctl` experiment catalog.
- Print evidence that answers the question: a metric, comparison, prediction,
  trace, or visualization.
- State important host/device boundaries and anything deliberately left out.
- Link the entrypoint, reusable implementation, and primary reference from this
  README; add a local README only when setup or interpretation needs more room.
