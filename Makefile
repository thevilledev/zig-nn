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

# Build and run all examples
.PHONY: examples
examples: example-simple-xor example-gated-network example-xor-training

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