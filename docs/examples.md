# Examples

Examples are small programs that make one behavior easy to inspect. Reusable
library code should live in `src/`; example-specific glue and notes should stay
with the example.

## Quick Runs

```bash
make run-examples
```

This target runs the quick examples. On machines without the relevant GPU
toolchain, prefer the individual CPU examples below.

## Catalog

| Example | Zig step | Make target | Purpose |
| --- | --- | --- | --- |
| Simple XOR | `zig build run_simple_xor` | `make example-simple-xor` | Minimal forward pass through an XOR-shaped network |
| XOR training | `zig build run_xor_training` | `make example-xor-training` | Backpropagation on XOR and optional model output |
| Binary classification | `zig build run_binary_classification` | `make example-binary-classification` | Circular decision-boundary classification |
| Regression | `zig build run_regression` | `make example-regression` | Nonlinear function approximation |
| Gated network | `zig build run_gated_network` | `make example-gated-network` | GLU/SwiGLU-style gated layer behavior |
| Network visualisation | `zig build run_network_visualisation` | `make example-network-visualisation` | Text visualization of a network shape |
| Backend demo | `zig build run_backend_demo` | `make example-backend-demo` | Basic matrix operations through the public API |
| GPU demo | `zig build run_gpu -Dgpu=metal` | `make example-gpu-metal` | Metal-backed matrix operations on macOS |
| GPU benchmark | `zig build run_gpu_benchmark -Dgpu=metal -Doptimize=ReleaseFast` | `make example-gpu-benchmark` | CPU vs Metal matrix multiplication timing |
| TurboQuant | `zig build run_turboquant` | `make example-turboquant` | Quantization experiment with comparable metrics |
| Tiny GPT | `zig build run_tiny_gpt` | `make example-tiny-gpt` | Tiny decoder-only Transformer demo with corpus presets |
| Serving | `zig build run_serving` | `make example-serving` | HTTP prediction server for a saved XOR model |
| MNIST | `zig build run_mnist` | `make example-mnist` | Digit recognition with external MNIST data |

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
zig build run_xor_training -- --output=xor_model.bin
zig build run_serving
```

Then send a request:

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"input": [0.0, 1.0], "batch_size": 1}'
```

Tiny GPT can run with the checked-in toy corpus, or with sourced corpora:

```bash
make prepare-tiny-gpt-data
zig build run_tiny_gpt -- --corpus tinystories --prompt "Once upon a time"
```
