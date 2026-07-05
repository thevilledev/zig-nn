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
  list [topic]        List tasks, examples, or tests
  fmt                 Format Zig source
  clean               Remove Zig build artifacts
  data mnist          Download and extract MNIST data
  doctor              Check local tool versions

Examples:
  nnctl build --mode ReleaseFast
  nnctl test matrix --summary all
  nnctl run tiny-gpt -- --prompt "to be" --tokens 80
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
