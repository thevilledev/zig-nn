# Neural Network Architecture

This document outlines the architectural design of our neural network implementation in Zig.

## Core Design Principles

1. **Memory Safety**: Utilizing Zig's memory safety features and explicit allocation
2. **Performance**: Optimized matrix operations and efficient memory usage
3. **Modularity**: Clear separation of concerns between components
4. **Extensibility**: Easy to add new layer types and activation functions

## Component Overview

### Matrix Operations (`matrix.zig`)
- Core mathematical operations
- Memory-efficient implementation
- Support for batch operations
- Optimized for neural network computations

### Layers (`layer.zig`)
Two main layer types:
1. **Standard Layer**
   - Traditional fully connected layer
   - Configurable activation functions
   - Weight and bias management

2. **Gated Layer**
   - Support for GLU and SwiGLU
   - Dual weight matrices
   - Optimized gating operations

### Network (`network.zig`)
- Flexible layer composition
- Dynamic architecture building
- Support for different loss functions:
  - Mean Squared Error
  - Cross Entropy
  - Binary Cross Entropy
- Mini-batch training capabilities

### Backends (`backend.zig`, `cpu_backend.zig`, `metal_backend.zig`, `cuda_backend.zig`, `rocm_backend.zig`)
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
verified through backend tests, GPU examples, backend training examples,
snapshot inference checks, and benchmark rows for the target machine.

### Quantization (`quantization.zig`)
- Uniform scalar quantization baseline
- TurboQuant-inspired rotated scalar quantization experiment
- Vector metrics for MSE, max error, cosine similarity, inner-product error,
  and compression ratio
- Intended for reproducible small experiments

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
   - CPU fallback when a requested GPU backend is unavailable

## Future Considerations

Areas for potential improvement:
1. SIMD optimizations
2. Optimizer state support for backend trainers
3. Distributed training support
4. Additional layer types:
   - Convolutional layers
   - Attention mechanisms
   - Residual connections
5. Backend-aware Tiny GPT inference and training experiments
6. Quantized network inference and KV-cache experiments
