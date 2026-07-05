# Research Resources

A reference list for topics covered by this repository's
[architecture](architecture.md), [examples](examples.md),
[experiments](experiments.md), [GPU notes](gpu.md), and
[activation function notes](activation_functions.md).

## Core References

- [Deep Learning](https://www.deeplearningbook.org/) - Feedforward networks,
  backpropagation, losses, optimization, and numerical computation.
- [BLAS](https://www.netlib.org/blas/) - Vector, matrix-vector, and
  matrix-matrix operation interfaces.
- [LAPACK](https://www.netlib.org/lapack/) - Dense linear algebra routines
  built around optimized BLAS usage.
- [Zig Language Reference 0.16.0](https://ziglang.org/documentation/0.16.0/) -
  Language reference for the Zig version used by the project docs.
- [Zig Standard Library](https://ziglang.org/documentation/0.16.0/std/) -
  Allocators, arrays, math functions, file I/O, and testing APIs.

## Neural Network Foundations

- [Learning representations by back-propagating errors](https://www.nature.com/articles/323533a0)
  - Backpropagation.
- [Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a.html)
  - Xavier-style initialization and gradient flow.
- [Delving Deep into Rectifiers](https://arxiv.org/abs/1502.01852) - He,
  Zhang, Ren, and Sun's rectifier initialization paper.
- [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)
  - Neural network training for document recognition and MNIST-style data.

## Activations And Gating

- [Gaussian Error Linear Units](https://arxiv.org/abs/1606.08415) - Hendrycks
  and Gimpel's GELU paper.
- [Searching for Activation Functions](https://arxiv.org/abs/1710.05941) -
  Ramachandran, Zoph, and Le's Swish paper.
- [Language Modeling with Gated Convolutional Networks](https://arxiv.org/abs/1612.08083)
  - GLU.
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) -
  SwiGLU and other gated feed-forward variants.

## Transformers And Tiny GPT

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et
  al.'s Transformer paper.
- [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
  - GPT-1.
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
  - GPT-2.
- [minGPT](https://github.com/karpathy/minGPT) - Educational GPT
  implementation.
- [nanoGPT](https://github.com/karpathy/nanoGPT) - Compact GPT training
  implementation.
- [TinyStories paper](https://arxiv.org/abs/2305.07759) - Small language
  models trained on simple stories.
- [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories)
  - Dataset card and files.

## Quantization

- [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874)
  - Random rotation, scalar quantization, inner-product error, KV-cache, and
  nearest-neighbor quantization.
- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877)
  - Integer-only neural network inference.
- [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339)
  - Transformer inference quantization and outlier-aware mixed precision.
- [GPTQ](https://arxiv.org/abs/2210.17323) - Post-training weight quantization
  for GPT-style models.
- [SmoothQuant](https://arxiv.org/abs/2211.10438) - Activation-outlier
  smoothing for efficient LLM inference.
- [A Survey of Quantization Methods for Efficient Neural Network Inference](https://arxiv.org/abs/2103.13630)
  - Quantization terminology and method taxonomy.

## GPU And Backend Work

- [Apple Metal](https://developer.apple.com/metal/) - Metal overview,
  platform support, and tooling.
- [Performing calculations on a GPU](https://developer.apple.com/documentation/metal/performing-calculations-on-a-gpu)
  - Metal command queues, buffers, compute pipelines, and dispatch.
- [Metal-cpp](https://developer.apple.com/metal/cpp/) - Official C++ interface
  for Metal.
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)
  - CUDA execution model, memory model, and programming interface.
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
  - CUDA optimization, profiling, memory hierarchy, and numerical accuracy.
- [cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html) - NVIDIA BLAS
  library documentation.

## Serving And Tooling

- [TensorFlow Serving: Flexible, High-Performance ML Serving](https://arxiv.org/abs/1712.06139)
  - Model serving architecture, versioning, model lookup, and lifecycle.
- [TensorFlow Serving docs](https://www.tensorflow.org/tfx/guide/serving) -
  Serving overview and API references.
- [NVIDIA Triton Inference Server docs](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/)
  - Model repositories, batching, backends, and inference serving.
- [Go net/http](https://pkg.go.dev/net/http) - Official HTTP package docs for
  servers and clients.
- [Go encoding/json](https://pkg.go.dev/encoding/json) - Official JSON package
  docs.

## Datasets

- [MNIST database](https://yann.lecun.com/exdb/mnist/) - Original MNIST dataset
  home.
- [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories)
  - Dataset card and files.
- [Tiny Shakespeare corpus](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)
  - Small character-level language-modeling corpus.
