# Research Resources

This bibliography connects the ideas in the
[experiment guide](../experiments/README.md) to their reusable implementations.
Experiment-specific caveats remain beside the runnable code; this page provides
the broader papers, datasets, and technical references.

## Core References

**See it in the repo:** [matrix operations](../src/matrix.zig),
[tensor runtime](../src/tensor.zig), [architecture](architecture.md), and the
[foundation experiments](../experiments/README.md#foundations).

- [Deep Learning](https://www.deeplearningbook.org/) - Feedforward networks,
  backpropagation, losses, optimization, and numerical computation.
- [BLAS](https://www.netlib.org/blas/) - Vector, matrix-vector, and
  matrix-matrix operation interfaces.
- [LAPACK](https://www.netlib.org/lapack/) - Dense linear algebra routines
  built around optimized BLAS usage.
- [Zig Language Reference 0.16.0](https://ziglang.org/documentation/0.16.0/) -
  Language reference for the Zig version used by the project.
- [Zig Standard Library](https://ziglang.org/documentation/0.16.0/std/) -
  Allocators, arrays, math functions, file I/O, and testing APIs.

## Neural Network Foundations

**See it in the repo:** [matrix](../src/matrix.zig), [layers](../src/layer.zig),
[networks](../src/network.zig), and the
[foundation experiments](../experiments/README.md#foundations).

- [Learning representations by back-propagating errors](https://www.nature.com/articles/323533a0)
  - Backpropagation.
- [Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a.html)
  - Xavier-style initialization and gradient flow.
- [Delving Deep into Rectifiers](https://arxiv.org/abs/1502.01852) - He,
  Zhang, Ren, and Sun's rectifier initialization paper.
- [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)
  - Neural-network training for document recognition and MNIST-style data.

## Optimization

**See it in the repo:** [training and optimizers](../src/training.zig),
[reusable modules](../src/modules.zig), and
[Optimizer Lab](../experiments/optimizer_lab/optimizer_lab.zig).

- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
  - Adaptive first- and second-moment optimization.
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
  - AdamW and weight decay separated from the gradient update.
- [On the importance of initialization and momentum in deep learning](https://proceedings.mlr.press/v28/sutskever13.html)
  - Momentum behavior and optimization conditioning.

## Spectral Learning

**See it in the repo:** [spectral transforms and features](../src/spectral.zig),
[network fundamentals](../src/network.zig), and
[Spectral Learning](../experiments/spectral_learning/spectral_learning.zig).

- [On the Spectral Bias of Neural Networks](https://proceedings.mlr.press/v97/rahaman19a.html)
  - Frequency-dependent learning speed and the tendency of deep networks to
    learn low-frequency structure first.
- [Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains](https://proceedings.neurips.cc/paper/2020/hash/55053683268957697aa39fba6f231c68-Abstract.html)
  - Coordinate feature mappings that expose tunable harmonic structure to an
    MLP.

## Activations And Gating

**See it in the repo:** [activation functions](../src/activation.zig),
[gated layers](../src/layer.zig),
[Gated Network](../experiments/gated_network/gated_network.zig), and the
[implementation notes](activation_functions.md).

- [Gaussian Error Linear Units](https://arxiv.org/abs/1606.08415) - Hendrycks
  and Gimpel's GELU paper.
- [Searching for Activation Functions](https://arxiv.org/abs/1710.05941) -
  Ramachandran, Zoph, and Le's Swish paper.
- [Language Modeling with Gated Convolutional Networks](https://arxiv.org/abs/1612.08083)
  - GLU.
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) -
  SwiGLU and other gated feed-forward variants.

## Representations, Vision, And Audio

**See it in the repo:** [spatial layers](../src/spatial.zig),
[audio features](../src/audio.zig), [reusable modules](../src/modules.zig), and
the [vision and audio experiments](../experiments/README.md#vision-and-audio).

- [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)
  - Convolutional networks and handwritten-digit recognition.
- [Extracting and Composing Robust Features with Denoising Autoencoders](https://arxiv.org/abs/0803.1119)
  - Corruption-based representation learning and reconstruction.
- [Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition](https://arxiv.org/abs/1804.03209)
  - Dataset construction and keyword-recognition evaluation.
- [TensorFlow simple audio recognition tutorial](https://www.tensorflow.org/tutorials/audio/simple_audio)
  - The Mini Speech Commands teaching subset and log-mel workflow used as a
  practical reference.

## Transformers And Tiny GPT

**See it in the repo:** [Transformer components](../src/transformer.zig),
[tensor primitives](../src/tensor.zig),
[TinyGPT](../experiments/tiny_gpt/README.md), and the
[language and sequence experiments](../experiments/README.md#language-retrieval-and-sequence-models).

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

## NLP, Structured Prediction, And Retrieval

**See it in the repo:** [text processing](../src/text.zig),
[embeddings](../src/embeddings.zig),
[structured prediction](../src/structured.zig), [decoding](../src/decoding.zig),
[retrieval](../src/retrieval.zig), and the
[language and sequence experiments](../experiments/README.md#language-retrieval-and-sequence-models).

- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
  - Encoder-decoder attention and learned alignments.
- [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
  - Skip-gram Word2Vec.
- [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546)
  - Negative sampling and subsampling for word representations.
- [Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data](https://repository.upenn.edu/cis_papers/159/)
  - Linear-chain conditional random fields.
- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)
  - Byte-pair encoding for subword tokenization.
- [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)
  - Nucleus top-p sampling.
- [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
  - Symmetric contrastive learning across paired encoders.

## Sequence Models And Reinforcement Learning

**See it in the repo:** [recurrent models](../src/recurrent.zig),
[reinforcement-learning utilities](../src/reinforcement.zig),
[GRU Sequence](../experiments/gru_sequence/gru_sequence.zig), and
[DQN](../experiments/dqn/dqn.zig).

- [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078)
  - The gated recurrent unit and encoder-decoder recurrence.
- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
  - Experience replay, target networks, and deep Q-learning.
- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
  - The later DQN evaluation and algorithm presentation.

## Quantization

**See it in the repo:** [quantization primitives](../src/quantization.zig) and
the [TurboQuant experiment](../experiments/quantization/README.md).

- [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874)
  - Random rotation, scalar quantization, inner-product error, KV-cache, and
  nearest-neighbor quantization.
- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877)
  - Integer-only neural-network inference.
- [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339)
  - Transformer inference quantization and outlier-aware mixed precision.
- [GPTQ](https://arxiv.org/abs/2210.17323) - Post-training weight quantization
  for GPT-style models.
- [SmoothQuant](https://arxiv.org/abs/2211.10438) - Activation-outlier
  smoothing for efficient LLM inference.
- [A Survey of Quantization Methods for Efficient Neural Network Inference](https://arxiv.org/abs/2103.13630)
  - Quantization terminology and method taxonomy.

## GPU And Backend Work

**See it in the repo:** [backend abstraction](../src/backend.zig),
[Metal](../src/metal_backend.zig), [CUDA](../src/cuda_backend.zig),
[ROCm](../src/rocm_backend.zig), the
[runtime experiments](../experiments/README.md#training-and-runtime), and the
[GPU verification guide](gpu.md).

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
- [ROCm HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/)
  - AMD HIP programming model and runtime documentation.
- [HIPRTC](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_rtc.html)
  - Runtime compilation for HIP kernels.
- [rocBLAS](https://rocm.docs.amd.com/projects/rocBLAS/en/latest/) - AMD BLAS
  library documentation.

## Serving And Tooling

**See it in the repo:** [inference service](../src/inference_service.zig),
[XOR serving](../experiments/serving/README.md), and the
[TinyGPT OpenAI-compatible server](../experiments/tiny_gpt/openai_server.zig).

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

**See them in the repo:** [MNIST](../experiments/mnist/mnist.zig),
[Speech Commands](../experiments/speech_commands/README.md), and the
[TinyGPT corpus notes](../experiments/tiny_gpt/data/README.md).

- [MNIST database](https://yann.lecun.com/exdb/mnist/) - Original MNIST dataset
  home.
- [Speech Commands dataset](https://arxiv.org/abs/1804.03209) - Dataset design,
  collection, and evaluation conventions.
- [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories)
  - Dataset card and files.
- [Tiny Shakespeare corpus](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)
  - Small character-level language-modeling corpus.
