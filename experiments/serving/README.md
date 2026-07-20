# Neural Network Serving Experiment

This experiment wraps a trained neural network in a small HTTP prediction
server so request validation, model loading, and response behavior remain easy
to inspect.

## Prerequisites

1. A trained model saved in binary format (e.g., `xor_model.bin`)
2. The model file should be in the same directory as the server executable

You can train a simple model [in the XOR training experiment](../xor_training).

Run the following in the repository root:

```
zig build run_xor_training -- --output=xor_model.bin
```

## Building and Running

Inspect the ZNN header and start the unified server:

```bash
nnctl model inspect xor_model.bin
nnctl serve --model xor_model.bin --gpu auto
```

For one prediction without an HTTP server:

```bash
nnctl predict --model xor_model.bin --input '[0,1]' --gpu auto
```

The original experiment command remains available as a compatibility path:

```bash
zig build run_serving
Server listening on http://127.0.0.1:8080
```

You can override the model, host, or port:

```bash
zig build run_serving -- \
  --model=xor_model.bin --host=127.0.0.1 --port=8080 --backend=auto
```

Both commands load one persistent `nn.Inference.DenseSession`: the selected
device, execution context, uploaded network snapshot, and runtime counters are
reused across requests. Only `auto` may fall back to CPU; explicitly selected
Metal, CUDA, or ROCm backends fail if unavailable.

## API Usage

### Single Prediction

Send a POST request to `/predict` with a JSON body:

```json
{
    "input": [0.0, 1.0],
    "batch_size": 1
}
```

The server will respond with:

```json
{
    "prediction":[0.9833722451345307],
    "confidence":0.9833722451345307
}
```

### Example using curl

```bash
curl -X POST http://localhost:8080/predict \
     -H "Content-Type: application/json" \
     -d '{"input": [0.0, 1.0], "batch_size": 1}'
```

Validated batches from 1 through 1024 are accepted. For a two-input model, a
two-sample request is a flat array:

```bash
curl -X POST http://localhost:8080/predict \
     -H "Content-Type: application/json" \
     -d '{"input": [0.0, 1.0, 1.0, 0.0], "batch_size": 2}'
```

## Testing

Run the tests with:

```bash
zig build test-serving
```

## Notes

- The server expects the model file to be named `xor_model.bin` by default
- It listens on 127.0.0.1:8080 by default
- Input dimensions must match the model's expected input size
- `batch_size` must be between 1 and 1024
- Batches must contain exactly `batch_size * input_size` values
- The confidence value is currently derived from the first prediction value
- Error responses use a JSON `error` object with an HTTP status code
