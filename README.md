# Neural Network in Zig

[![Work in Progress](https://img.shields.io/badge/Status-Work%20in%20Progress-yellow)](https://github.com/thevilledev/zig-nn)

A minimalistic neural network implementation in Zig for learning purposes. This project focuses on understanding the fundamentals of neural networks including hidden layers, activation functions, and basic training processes.

## Features

- Matrix operations from scratch
- Common activation functions (Sigmoid, ReLU, Tanh)
- Basic feed-forward neural network architecture

## Building

Make sure you have Zig installed on your system. This project is developed with the latest stable version of Zig.

```bash
# Clone the repository
git clone https://github.com/thevilledev/zig-nn.git
cd zig-nn

# Build the project
zig build

# Run tests
zig build test
```

## Testing

The project includes a comprehensive test suite to verify the functionality of all components. The build system is configured to run tests for each module separately, making it easy to identify which component has issues.

### Running Tests

```bash
# Run all tests
zig build test

# Run tests for specific components
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