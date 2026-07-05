package cli

import (
	"fmt"
	"io"

	"nnctl/internal/catalog"
)

func printUsage(w io.Writer) {
	fmt.Fprint(w, `Usage: nnctl [--repo DIR] [--zig PATH] <command> [args]

Commands:
  all                 Build, test, and build examples
  build               Build the Zig library
  release             Build the Zig library with release defaults
  test                Run unit and acceptance tests
  examples            Build all examples
  run <example>       Run an example, or "quick" for quick examples
  train tiny-gpt      Train a TinyGPT checkpoint
  chat                Run TinyGPT inference with a local chat app
  list [topic]        List tasks, examples, or tests
  fmt                 Format Zig source
  clean               Remove Zig build artifacts
  data mnist          Download and extract MNIST data
  doctor              Check local tool versions

Examples:
  nnctl build --mode ReleaseFast
  nnctl test matrix --summary all
  nnctl run tiny-gpt -- --prompt "to be" --tokens 80
  nnctl train tiny-gpt --output tiny-gpt.bin
  nnctl chat --model tiny-gpt.bin
  nnctl data mnist
`)
}

func printHelp(w io.Writer, args []string) {
	if len(args) == 0 {
		printUsage(w)
		return
	}
	switch catalog.NormalizeName(args[0]) {
	case "run":
		fmt.Fprint(w, `Usage: nnctl run [flags] <example> [-- example args]

Flags:
  --mode, --optimize  Zig optimization mode
  --gpu               GPU backend: none, auto, metal, or cuda

Examples:
  nnctl run simple-xor
  nnctl run gpu --gpu metal
  nnctl run tiny-gpt -- --prompt "to be" --tokens 80
  nnctl run quick
`)
	case "train":
		fmt.Fprint(w, `Usage: nnctl train tiny-gpt [flags]

Trains a TinyGPT checkpoint by delegating to the TinyGPT example with
full-model training enabled. Fresh checkpoints accept model-shape flags;
resumed checkpoints load shape from the checkpoint.

Flags:
  --output            Checkpoint path to write (default: tiny-gpt.bin)
  --resume            Resume from an existing checkpoint; saves back unless --output is set
  --corpus            auto, toy, shakespeare, or tinystories
  --corpus-path       Custom UTF-8 corpus path
  --steps             Full-model training steps (default: 5000)
  --learning-rate     Full-model learning rate (default: 0.01)
  --batch-size        Corpus windows per training step (default: 4)
  --train-chars       Training corpus characters (default: 65536)
  --validation-chars  Validation holdout characters (default: 4096)
  --block-size        Context length for new checkpoints (default: 32)
  --layers            Transformer block count for new checkpoints (default: 4)
  --heads             Attention head count for new checkpoints (default: 4)
  --embd              Embedding/channel size for new checkpoints (default: 64)
  --optimizer         sgd or adamw (default: adamw)
  --weight-decay      Full-training weight decay (default: 0.01)
  --prompt            Prompt for the post-training sample
  --tokens            Sample tokens after training (default: 120)
  --temperature       Sampling temperature (default: 0.8)
  --top-k             Top-k sampler cutoff (default: 16)
  --seed              Deterministic seed (default: 42)
  --corpus-prior      Enable corpus-prior sampling in the post-training sample
  --mode, --optimize  Zig optimization mode (default: ReleaseFast)

Examples:
  nnctl data tiny-gpt
  nnctl train tiny-gpt --corpus tinystories --output tiny-gpt.bin
  nnctl train tiny-gpt --resume tiny-gpt.bin --steps 2000
  nnctl chat --model tiny-gpt.bin
`)
	case "test":
		fmt.Fprint(w, `Usage: nnctl test [flags] [name]

Flags:
  --unit-only         Run only unit tests
  --acceptance-only   Run only acceptance tests
  --name              Individual test name
  --summary           Zig build summary mode, for example all
  --mode, --optimize  Zig optimization mode
  --gpu               GPU backend: none, auto, metal, or cuda
`)
	case "chat":
		fmt.Fprint(w, `Usage: nnctl chat --model <checkpoint> [flags]

Starts the TinyGPT OpenAI-compatible inference server and serves a small local
browser chat UI that proxies requests to it.

Flags:
  --model             TinyGPT checkpoint from nnctl train tiny-gpt
  --allow-untrained   Serve a seeded untrained model when --model is omitted
  --host              Chat app bind host (default: 127.0.0.1)
  --port              Chat app bind port (default: 8090)
  --api-host          Inference server bind host (default: 127.0.0.1)
  --api-port          Inference server bind port (default: 8080)
  --model-name        Model id returned by the inference server
  --max-tokens        Default generated tokens per reply
  --temperature       Default sampling temperature
  --top-k             TinyGPT top-k sampler cutoff
  --ready-timeout     Time to wait for inference readiness
  --mode, --optimize  Zig optimization mode

Examples:
  nnctl chat --model tiny-gpt.bin
  nnctl chat --allow-untrained --port 8090
`)
	case "data":
		printDataHelp(w)
	default:
		printUsage(w)
	}
}

func printDataHelp(w io.Writer) {
	fmt.Fprint(w, `Usage: nnctl data <command>

Commands:
  mnist               Download and extract the MNIST dataset into data/
  tiny-gpt            Download sourced Tiny GPT demo corpora

Examples:
  nnctl data mnist
  nnctl data mnist --dir /tmp/mnist --force
  nnctl data tiny-gpt
`)
}
