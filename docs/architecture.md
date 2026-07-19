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
