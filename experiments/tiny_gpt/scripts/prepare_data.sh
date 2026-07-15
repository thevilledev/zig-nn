#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
DATA_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/../data" && pwd)

SHAKESPEARE_URL="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
TINYSTORIES_VALID_URL="https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt"
TINYSTORIES_BYTES="${TINYSTORIES_BYTES:-1048576}"

download() {
    url="$1"
    output="$2"

    if command -v curl >/dev/null 2>&1; then
        curl -L "$url" -o "$output"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "$output" "$url"
    else
        echo "error: curl or wget is required to download data" >&2
        exit 1
    fi
}

mkdir -p "$DATA_DIR/shakespeare" "$DATA_DIR/tinystories"

if [ ! -f "$DATA_DIR/shakespeare/input.txt" ]; then
    echo "Downloading Tiny Shakespeare..."
    download "$SHAKESPEARE_URL" "$DATA_DIR/shakespeare/input.txt"
else
    echo "Tiny Shakespeare already exists."
fi

if [ ! -f "$DATA_DIR/tinystories/valid.txt" ]; then
    echo "Downloading TinyStories validation split..."
    download "$TINYSTORIES_VALID_URL" "$DATA_DIR/tinystories/valid.txt"
else
    echo "TinyStories validation split already exists."
fi

echo "Writing TinyStories ${TINYSTORIES_BYTES}-byte slice..."
head -c "$TINYSTORIES_BYTES" "$DATA_DIR/tinystories/valid.txt" > "$DATA_DIR/tinystories/tinystories_1mb.txt"

echo "Prepared Tiny GPT corpora in $DATA_DIR"
