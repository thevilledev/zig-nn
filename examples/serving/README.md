# Neural Network Serving Example

This directory contains an example implementation of an HTTP server that serves predictions from a trained neural network model.

## Prerequisites

1. A trained model saved in binary format (e.g., `xor_model.bin`)
2. The model file should be in the same directory as the server executable

You can train a simple model [in the XOR training example](../xor_training).

Run the following in the repository root:

```
zig build run_xor_training -- --output=xor_model.bin
```

## Building and Running

```bash
# Run the server
zig build run_serving
Server listening on http://127.0.0.1:8080
```

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

## Testing

Run the tests with:

```bash
zig test examples/server.zig
```

## Notes

- The server expects the model file to be named `xor_model.bin` by default
- Hardcoded to listen to 127.0.0.1 on port 8080
- Input dimensions must match the model's expected input size
- The confidence value is currently derived from the first prediction value
- The server supports concurrent connections 
