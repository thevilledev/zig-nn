# Neural Network in Zig

[![Work in Progress](https://img.shields.io/badge/Status-Work%20in%20Progress-yellow)](https://github.com/thevilledev/zig-nn)

A minimalistic neural network implementation in Zig for learning purposes. This project focuses on understanding the fundamentals of neural networks including hidden layers, activation functions, and basic training processes.

## Features

- Matrix operations from scratch
- Common activation functions (Sigmoid, ReLU, Tanh)
- Advanced activation functions (Swish, GLU, SwiGLU)
- Basic feed-forward neural network architecture
- Support for gated architectures used in modern transformer models

See [examples](examples/).

## Documentation

Detailed documentation is available in the `docs` directory:
- [Neural Network Architecture](docs/architecture.md) - Design principles and implementation details
- [Advanced Activation Functions](docs/activation_functions.md) - Detailed information about activation functions

## Building

Make sure you have Zig installed on your system. This project is developed with the latest stable version of Zig.

For convenience, a Makefile is provided with common operations:

```bash
# Build and run everything (build, test, examples)
make

# Run specific examples, for example:
make example-simple-xor

# Build with different optimization modes
make BUILD_MODE=ReleaseFast
make release  # Build with ReleaseSafe mode

# See all available commands
make help
```

## Testing

The project includes a comprehensive test suite to verify the functionality of all components. The build system is configured to run tests for each module sequentially, making it easy to identify which component has issues.

```bash
# Run all tests
make test

# Run tests for specific components, for example:
zig build test-matrix     # Run matrix operation tests
zig build test-activation # Run activation function tests
zig build test-layer      # Run neural network layer tests
zig build test-network    # Run full network tests
```

## Learning Goals

This project serves as a learning exercise for:

- Understanding neural network fundamentals
- Implementing mathematical operations in Zig
- Working with Zig's memory management and error handling

## License

MIT License 
