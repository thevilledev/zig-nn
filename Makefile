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

# Build and run all examples
.PHONY: examples
examples: example-simple-xor example-gated-network example-xor-training example-binary-classification example-regression example-network-visualization

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
