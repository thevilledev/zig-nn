# Neural Network Architecture

This document maps the reusable implementation to the experiments that make
each subsystem observable. Start with the [experiment guide](../experiments/README.md)
when you prefer to learn by running code.

## Core Design Principles

1. **Memory Safety**: Utilizing Zig's memory safety features and explicit allocation
2. **Performance**: Optimized matrix operations and efficient memory usage
3. **Modularity**: Clear separation of concerns between components
4. **Extensibility**: Easy to add new layer types and activation functions

## Component Overview

### [Matrix Operations](../src/matrix.zig)
- Core mathematical operations
- Memory-efficient implementation
- Support for batch operations
- Optimized for neural network computations

Try it in [Simple XOR](../experiments/simple_xor/simple_xor.zig) and
[XOR Training](../experiments/xor_training/xor_training.zig).

### [Layers](../src/layer.zig)
Two main layer types:
1. **Standard Layer**
   - Traditional fully connected layer
   - Configurable activation functions
   - Weight and bias management

2. **Gated Layer**
   - Support for GLU and SwiGLU
   - Dual weight matrices
   - Optimized gating operations

Try the standard path in [Binary Classification](../experiments/binary_classification/binary_classification.zig)
and the gated path in [Gated Network](../experiments/gated_network/gated_network.zig).

### [Network](../src/network.zig)
- Flexible layer composition
- Dynamic architecture building
- Support for different loss functions:
  - Mean Squared Error
  - Cross Entropy
  - Binary Cross Entropy
- Mini-batch training capabilities

Try it in [Regression](../experiments/regression/regression.zig) or
[MNIST](../experiments/mnist/mnist.zig).

### [Spectral Methods](../src/spectral.zig)

- In-place radix-2 FFT for `f32` and `f64` real/imaginary buffers
- Normalized single-sided amplitude spectra for real signals
- Scalar coordinate encoding with ordered sine/cosine harmonic pairs
- Shared FFT mechanics used by the audio log-mel frontend

Try them in [Spectral Learning](../experiments/spectral_learning/spectral_learning.zig),
where the CPU `Network` compares raw coordinates with Fourier-feature inputs.

### [Tensor Runtime](../src/tensor.zig), [Modules](../src/modules.zig), And [Training](../src/training.zig)

- Rank-aware f32 tensors backed by the selected CPU, Metal, CUDA, or ROCm
  backend
- Explicit execution batches and transfer/kernel telemetry
- Reusable device-resident `Linear` and configurable `Mlp` modules with
  forward caches and parameter gradients
- Ordered optimizer bindings for SGD, momentum, and AdamW
- Gradient accumulation, global-norm clipping, weight decay, and fused backend
  parameter updates

Try them in [Optimizer Lab](../experiments/optimizer_lab/optimizer_lab.zig),
[Padding Masks](../experiments/padding_masks/padding_masks.zig), and
[Autoencoder](../experiments/autoencoder/autoencoder.zig).

### [Inference Sessions](../src/inference.zig)

Inference deliberately has two type-safe session contracts instead of one
generic model interface:

- `DenseSession` loads a ZNN network, selects one device, uploads a persistent
  backend snapshot, and exposes `predictInto` and `predictAlloc` for validated
  batches.
- `TextSession` loads a TGPT TinyGPT model, keeps its decoder, execution
  context, KV cache, logits, and sampling workspace alive, and exposes
  `prefill`, `decodeNext`, writer-based `generate`, and `generateAlloc`.

Both sessions own their selected device and expose execution/backend counters.
`GenerationOptions` makes the seed, token limit, temperature, top-k, and top-p
explicit; `GenerationStats` returns the backend, token counts, elapsed time,
and runtime snapshot. Only automatic device selection can fall back to CPU.

The library detects existing ZNN and TGPT headers without introducing another
container format. ZNN and TGPT versions 1–3 remain readable. `InferenceService`,
`run_serving`, and `run_tiny_gpt_openai` are compatibility entry points backed
by the session implementation during the transition.

### Backends

Implementations: [abstraction](../src/backend.zig), [CPU](../src/cpu_backend.zig),
[Metal](../src/metal_backend.zig), [CUDA](../src/cuda_backend.zig), and
[ROCm](../src/rocm_backend.zig). CUDA and ROCm use a shared
[compiled GPU implementation](../src/compiled_gpu_backend.zig) behind their
vendor-specific ABI adapters.

- `BackendMatrix` provides a separate backend-aware matrix API
- CPU backend implements all backend matrix operations
- Metal backend accelerates backend matrix operations on macOS
- CUDA backend accelerates backend matrix operations on Linux with NVIDIA GPUs
- ROCm backend accelerates backend matrix operations on Linux with AMD GPUs
- `Network.forwardBackend` and `Network.predictBackend` provide
  backend-aware inference without updating training caches
- `Network.trainBatchBackend` and `Network.trainBackend` provide
  backend-aware training against `BackendMatrix` inputs and targets
- `BackendNetwork` snapshots mirror network parameters once for repeated
  backend inference
- `BackendTrainer` keeps standard and gated trainable parameters on the
  backend across batches, then syncs them back to a CPU `Network` explicitly
  and can keep momentum optimizer state on the backend when configured

The default high-level `Network.forward`, `Network.trainBatch`, and
`Network.train` paths still use the CPU `Matrix` implementation. Backend
training works through explicit backend methods. Persistent backend training is
available for standard and gated networks through `BackendTrainer`; the
CPU-owned `Network.trainBatchBackend` and `Network.trainBackend` methods remain
available for direct backend-aware training calls. GPU support should be
verified through backend tests, GPU experiments, backend training experiments,
snapshot inference checks, and benchmark rows for the target machine.

Try the abstraction in [Backend Demo](../experiments/backend_demo/backend_demo.zig)
and [Backend Training](../experiments/backend_training/backend_training.zig),
then verify device selection with [GPU Demo](../experiments/gpu/gpu.zig).

### [Quantization](../src/quantization.zig)
- Uniform scalar quantization baseline
- TurboQuant-inspired rotated scalar quantization experiment
- Vector metrics for MSE, max error, cosine similarity, inner-product error,
  and compression ratio
- Intended for reproducible small experiments

Try it in the [TurboQuant experiment](../experiments/quantization/README.md).

### [Spatial Layers](../src/spatial.zig)

- CPU-first `Conv2d` reference over flattened NHWC image batches
- Explicit input, weight, and bias gradients for studying backpropagation
- Max-pool forward and backward operations with inspectable gradient routing

Try them in [CNN](../experiments/cnn/cnn.zig).

### [Audio Features](../src/audio.zig)

- PCM16 WAV decoding and mono conversion
- Linear resampling, shared FFT power spectra, mel filterbanks, and pooled log-mel
  features
- Host-side preprocessing with explicit normalization before tensor upload

Try the complete pipeline in
[Speech Commands](../experiments/speech_commands/README.md).

### [Recurrent](../src/recurrent.zig) And [Transformer](../src/transformer.zig) Models

- Device-backed GRU cell with input, recurrent, and parameter gradients for
  explicit backpropagation through time
- Causal multi-head decoder attention and KV-cache support
- Single-head unmasked self-attention composed from tensor primitives
- Padding-aware batched multi-head encoder attention with explicit token and
  attention masks
- Encoder-decoder cross-attention with separate query/memory gradients and
  inspectable alignment probabilities
- Pre-normalized encoder and decoder blocks with residual connections and GELU
  feed-forward networks

Compare [GRU Sequence](../experiments/gru_sequence/gru_sequence.zig) with
[Transformer Encoder](../experiments/transformer_encoder/transformer_encoder.zig),
then inspect cross-attention in [Seq2Seq](../experiments/seq2seq/seq2seq.zig).

### [Text](../src/text.zig), [Embeddings](../src/embeddings.zig), And [Structured Prediction](../src/structured.zig)

- Byte-level tokenization and learned byte-pair merges with exact byte fallback
- Skip-gram pair generation, unigram negative sampling, device-backed embedding
  training, and host-side cosine evaluation
- Linear-chain CRF partition functions, marginal-based gradients, and Viterbi
  decoding for structured sequence labels

Try them in [Tokenizer Lab](../experiments/tokenizer_lab/tokenizer_lab.zig),
[Word2Vec](../experiments/word2vec/word2vec.zig), and
[Sequence Tagging](../experiments/sequence_tagging/sequence_tagging.zig).

### [Decoding](../src/decoding.zig) And [Retrieval](../src/retrieval.zig)

- Deterministic greedy decoding plus temperature, top-k, nucleus top-p, and
  repetition-penalty sampling
- Device-resident symmetric InfoNCE over paired embedding batches
- Exact host-side cosine top-k with deterministic tie-breaking

Try them in [Decoding Lab](../experiments/decoding_lab/decoding_lab.zig) and
[Semantic Search](../experiments/semantic_search/semantic_search.zig).

### [Reinforcement Learning](../src/reinforcement.zig)

- Fixed-capacity transition replay with seeded minibatch sampling
- Linear epsilon-greedy exploration schedule
- Terminal-aware one-step Bellman targets
- Environment-specific interaction remains in experiments rather than the core
  library

Try it in [DQN](../experiments/dqn/dqn.zig).

## Cloud Orchestration

`nnctl` keeps cloud vocabulary separate from provider APIs. The
[provider contract](../nnctl/internal/cloud/provider.go) normalizes offerings,
accelerators, instances, deployment requests, owned resources, and destroy
results. A registry resolves a provider factory by stable name; Verda remains
the default for command compatibility, while the lab persists the selected
provider with each managed worker. Provider descriptors also carry the default
market policy, avoiding shared-command assumptions about spot, on-demand, or
future reservation models.

The base `Provider` interface contains only the lifecycle shared by GPU clouds:
catalog discovery, deploy, get/list instances, and destroy. Less universal
features use narrow capability interfaces or provider strategies, so a
neocloud adapter does not need to fake unsupported concepts:

| Capability | Contract | Current use |
| --- | --- | --- |
| SSH keys | `SSHKeyProvider` | Resolve attachable account key IDs |
| Volumes | `VolumeProvider` | Verda golden-volume discovery and cleanup |
| Image templates | `ImageTemplateFactory` | Verda Packer artifacts |
| Benchmark image | Capability plus CLI strategy | Provider-specific image preparation |

Normalized records carry `ProviderDetail` for provider-native diagnostics and
legacy output without leaking those fields into cross-provider decisions.
`ResourceRef` records ownership and preservation intent so cleanup remains
provider-controlled. Catalog results distinguish API-discovered offerings from
explicit contract offerings; this supports reserved and sales-enabled GPU
inventory without claiming it is publicly discoverable.

The [Verda adapter](../nnctl/internal/cloud/verda/provider.go) maps the existing
spot, pricing, volume-clone, and startup-script behavior onto the contract. The
[DigitalOcean adapter](../nnctl/internal/cloud/digitalocean/provider.go) maps
GPU Droplet sizes, GPU-ready images, account SSH keys, and on-demand lifecycle.
AMD offerings advertise ROCm and NVIDIA offerings advertise CUDA. The
[learning-lab manager](../nnctl/internal/lab/cloud.go) consumes only normalized
offerings and resource references, tolerates one provider catalog failing, and
dispatches recovery and cleanup to the provider stored with the worker. It
resolves every deploy request against the server-side available-offering list,
including explicitly configured contract inventory, before provisioning.

When adding a neocloud:

1. Implement `Factory` and the base `Provider`, then register the stable
   provider name in the CLI registry.
2. Map native SKUs into accelerator metadata and exact backend compatibility;
   keep fabric, reservation, quota, and contract quirks in provider details or
   provider configuration.
3. Implement only the optional capabilities the provider actually supports.
4. Add a benchmark provisioning strategy only when image preparation or
   readiness differs from the existing strategies.
5. Prove catalog normalization, dry-run behavior, resource ownership, partial
   failure, recovery, and cleanup with adapter contract tests.

This keeps the common surface small while leaving room for providers with
reserved clusters, bare metal, bid/spot markets, custom images, or asynchronous
provisioning semantics.

## Memory Management

### Allocation Strategy
- Clear ownership model
- Explicit deallocation
- Efficient matrix reuse
- Temporary buffer management

### Resource Lifecycle
1. Network initialization
2. Layer allocation
3. Forward propagation buffers
4. Backward propagation gradients
5. Clean deallocation

## Error Handling

Using Zig's error union types for handling:
- Invalid dimensions
- Memory allocation failures
- Numerical instabilities
- Configuration errors

## Performance Optimizations

1. **Matrix Operations**
   - Cache-friendly memory layout
   - Vectorized operations where possible
   - Minimal temporary allocations

2. **Training Process**
   - Efficient mini-batch processing
   - Smart gradient accumulation
   - Memory reuse between iterations

3. **Memory Layout**
   - Contiguous memory for matrices
   - Aligned allocations
   - Efficient striding

4. **GPU Backend Operations**
   - Metal compute kernels for backend matrix operations on macOS
   - CUDA kernels for backend matrix operations on Linux with NVIDIA GPUs
   - ROCm HIP kernels for backend matrix operations on Linux with AMD GPUs
   - Low-level backend probing can fall back to CPU; explicit tensor-device GPU
     requests reject that fallback

## Future Considerations

Areas for potential improvement:
1. SIMD optimizations
2. Mixed-precision tensor training
3. Batched recurrent sequences and dynamic sequence bucketing
4. Teacher-forced encoder-decoder stacks and beam search
5. Distributed training support
6. Approximate nearest-neighbor indexing
7. Quantized network inference and KV-cache experiments
