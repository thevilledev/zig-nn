# Examples

Examples are small programs that make one behavior easy to inspect. Reusable
library code should live in `src/`; example-specific glue and notes should stay
with the example.

## Quick Runs

```bash
./bin/nnctl run quick
```

This command runs the quick examples. On machines without the relevant GPU
toolchain, prefer the individual CPU examples below.

## Catalog

| Example | Zig step | nnctl command | Purpose |
| --- | --- | --- | --- |
| Simple XOR | `zig build run_simple_xor` | `./bin/nnctl run simple-xor` | Minimal forward pass through an XOR-shaped network |
| XOR training | `zig build run_xor_training` | `./bin/nnctl run xor-training` | Backpropagation on XOR and optional model output |
| Binary classification | `zig build run_binary_classification` | `./bin/nnctl run binary-classification` | Circular decision-boundary classification |
| Regression | `zig build run_regression` | `./bin/nnctl run regression` | Nonlinear function approximation |
| Gated network | `zig build run_gated_network` | `./bin/nnctl run gated-network` | GLU/SwiGLU-style gated layer behavior |
| Network visualisation | `zig build run_network_visualisation` | `./bin/nnctl run network-visualisation` | Unicode terminal topology diagram of a network shape |
| Backend demo | `zig build run_backend_demo` | `./bin/nnctl run backend-demo` | Basic matrix operations through the public API |
| GPU demo | `zig build run_gpu -Dgpu=metal` | `./bin/nnctl run gpu --gpu metal` | Metal-backed matrix operations on macOS |
| GPU benchmark | `zig build run_gpu_benchmark -Dgpu=metal -Doptimize=ReleaseFast` | `./bin/nnctl run gpu-benchmark` | CPU vs Metal matrix multiplication timing |
| TurboQuant | `zig build run_turboquant` | `./bin/nnctl run turboquant` | Quantization experiment with comparable metrics |
| Tiny GPT | `zig build run_tiny_gpt` | `./bin/nnctl run tiny-gpt` | Tiny decoder-only Transformer demo with corpus presets |
| Serving | `zig build run_serving` | `./bin/nnctl run serving` | HTTP prediction server for a saved XOR model |
| MNIST | `zig build run_mnist` | `./bin/nnctl run mnist` | Digit recognition with external MNIST data |

## Examples With Extra Notes

- [Quantization examples](../examples/quantization/README.md)
- [Tiny GPT example](../examples/tiny_gpt/README.md)
- [Serving example](../examples/serving/README.md)

## External Inputs

MNIST needs downloaded data:

```bash
./bin/nnctl data mnist
./bin/nnctl run mnist
```

The serving example needs a saved model:

```bash
./bin/nnctl run xor-training -- --output=xor_model.bin
./bin/nnctl run serving
```

Then send a request:

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"input": [0.0, 1.0], "batch_size": 1}'
```

Tiny GPT can run with the checked-in toy corpus, or with sourced corpora:

```bash
./bin/nnctl data tiny-gpt
./bin/nnctl train tiny-gpt --corpus tinystories --output tiny-gpt.bin
./bin/nnctl chat --model tiny-gpt.bin
```
