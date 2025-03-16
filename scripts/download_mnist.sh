#!/bin/bash

# Create data directory if it doesn't exist
mkdir -p data

# Download MNIST dataset files
# See https://github.com/cvdfoundation/mnist for more information
curl -o data/train-images-idx3-ubyte.gz https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
curl -o data/train-labels-idx1-ubyte.gz https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
curl -o data/t10k-images-idx3-ubyte.gz https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
curl -o data/t10k-labels-idx1-ubyte.gz https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz

# Extract files
cd data
gunzip train-images-idx3-ubyte.gz
gunzip train-labels-idx1-ubyte.gz
gunzip t10k-images-idx3-ubyte.gz
gunzip t10k-labels-idx1-ubyte.gz 
