# Zig Neural Network Project Makefile
# Configuration
ZIG ?= zig
BUILD_MODE ?= Debug
RELEASE_MODE ?= ReleaseSafe

# Directories
BUILD_DIR = zig-out
EXAMPLES_DIR = examples

# Default target
.PHONY: all
all: build test examples

# Build the project
.PHONY: build
build:
	$(ZIG) build -Doptimize=$(BUILD_MODE)

# Run all tests
.PHONY: test
test:
	$(ZIG) build test
	$(ZIG) build test-acceptance

# Build all examples
.PHONY: examples
examples:
	$(ZIG) build examples -Doptimize=$(BUILD_MODE)

# Run quick examples that do not require external data or model files
.PHONY: run-examples
run-examples: example-simple-xor example-gated-network example-network-visualisation example-backend-demo example-gpu

# Individual examples
.PHONY: example-simple-xor
example-simple-xor:
	@echo "Running Simple XOR example..."
	$(ZIG) build run_simple_xor -Doptimize=$(BUILD_MODE)

.PHONY: example-gated-network
example-gated-network:
	@echo "Running Gated Network example..."
	$(ZIG) build run_gated_network -Doptimize=$(BUILD_MODE)

.PHONY: example-xor-training
example-xor-training:
	@echo "Running XOR Training example..."
	$(ZIG) build run_xor_training -Doptimize=$(BUILD_MODE)

.PHONY: example-binary-classification
example-binary-classification:
	@echo "Running Binary Classification example..."
	$(ZIG) build run_binary_classification -Doptimize=$(BUILD_MODE)

.PHONY: example-regression
example-regression:
	@echo "Running Regression example..."
	$(ZIG) build run_regression -Doptimize=$(BUILD_MODE)

.PHONY: example-network-visualisation
example-network-visualisation:
	@echo "Running Network Visualisation example..."
	$(ZIG) build run_network_visualisation -Doptimize=$(BUILD_MODE)

.PHONY: example-network-visualization
example-network-visualization: example-network-visualisation

.PHONY: example-backend-demo
example-backend-demo:
	@echo "Running Backend Demo example..."
	$(ZIG) build run_backend_demo -Doptimize=$(BUILD_MODE)

.PHONY: example-serving
example-serving:
	@echo "Running Serving example..."
	$(ZIG) build run_serving -Doptimize=$(BUILD_MODE)

.PHONY: example-mnist
example-mnist:
	@echo "Running MNIST example..."
	$(ZIG) build run_mnist -Doptimize=$(BUILD_MODE)

.PHONY: example-gpu
example-gpu:
	@echo "Running GPU (auto) example..."
	$(ZIG) build run_gpu -Dgpu=auto -Doptimize=$(BUILD_MODE)

.PHONY: example-gpu-metal
example-gpu-metal:
	@echo "Running GPU (metal) example..."
	$(ZIG) build run_gpu -Dgpu=metal -Doptimize=$(BUILD_MODE)

# Build release version
.PHONY: release
release:
	$(ZIG) build -Doptimize=$(RELEASE_MODE)

# Clean build artifacts
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
	rm -rf .zig-cache

# Development helpers
.PHONY: format
format:
	$(ZIG) fmt .

.PHONY: help
help:
	@printf "Usage: make [target] [BUILD_MODE=Debug]\n\n"
	@printf "Core targets:\n"
	@printf "  all                         Build, test, and build examples\n"
	@printf "  build                       Build the library\n"
	@printf "  test                        Run unit and acceptance tests\n"
	@printf "  examples                    Build all examples\n"
	@printf "  run-examples                Run quick examples\n"
	@printf "  release                     Build with RELEASE_MODE\n"
	@printf "  format                      Format Zig source\n"
	@printf "  clean                       Remove build artifacts\n\n"
	@printf "Example targets:\n"
	@printf "  example-simple-xor          Run simple XOR\n"
	@printf "  example-gated-network       Run gated network\n"
	@printf "  example-xor-training        Run XOR training\n"
	@printf "  example-binary-classification Run binary classification\n"
	@printf "  example-regression          Run regression\n"
	@printf "  example-network-visualisation Run network visualisation\n"
	@printf "  example-backend-demo        Run backend demo\n"
	@printf "  example-gpu                 Run GPU auto demo\n"
	@printf "  example-gpu-metal           Run GPU Metal demo\n"
	@printf "  example-serving             Run serving example (requires model)\n"
	@printf "  example-mnist               Run MNIST example (requires data)\n"
