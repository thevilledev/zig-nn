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
| GPU demo | `zig build run_gpu -Dgpu=auto` | `nnctl run gpu` | Backend matrix operations on Metal or CUDA |
| GPU benchmark | `zig build run_gpu_benchmark -Dgpu=auto -Doptimize=ReleaseFast` | `nnctl run gpu-benchmark` | CPU vs GPU matrix multiplication timing |
| TurboQuant | `zig build run_turboquant` | `nnctl run turboquant` | Quantization experiment with comparable metrics |
| Tiny GPT | `zig build run_tiny_gpt` | `nnctl run tiny-gpt` | Tiny decoder-only Transformer demo with corpus presets |
| Serving | `zig build run_serving` | `nnctl run serving` | HTTP prediction server for a saved XOR model |
| MNIST | `zig build run_mnist` | `nnctl run mnist` | Digit recognition with external MNIST data |

## Examples With Extra Notes

- [Quantization examples](../examples/quantization/README.md)
- [Tiny GPT example](../examples/tiny_gpt/README.md)
- [Serving example](../examples/serving/README.md)

## External Inputs

MNIST needs downloaded data:

```bash
nnctl data mnist
nnctl run mnist
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
