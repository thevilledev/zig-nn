package cli

import (
	"context"
	"flag"
	"fmt"
	"strconv"

	"nnctl/internal/catalog"
	"nnctl/internal/flagx"
	"nnctl/internal/zig"
)

type tinyGPTTrainOptions struct {
	mode            string
	output          string
	resume          string
	corpus          string
	corpusPath      string
	steps           int
	learningRate    string
	batchSize       int
	trainChars      int
	validationChars int
	blockSize       int
	layers          int
	heads           int
	embd            int
	optimizer       string
	weightDecay     string
	prompt          string
	tokens          int
	temperature     string
	topK            int
	seed            int
	corpusPrior     bool
}

func (a *App) cmdTrain(ctx context.Context, args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("usage: nnctl train tiny-gpt [flags]")
	}
	name := catalog.NormalizeName(args[0])
	switch name {
	case "tiny-gpt", "tinygpt":
		return a.cmdTrainTinyGPT(ctx, args[1:])
	default:
		return fmt.Errorf("unknown train target %q", args[0])
	}
}

func (a *App) cmdTrainTinyGPT(ctx context.Context, args []string) error {
	opts := tinyGPTTrainOptions{
		mode:            getenvDefault("NNCTL_OPTIMIZE", "ReleaseFast"),
		output:          "tiny-gpt.bin",
		corpus:          "auto",
		steps:           5000,
		learningRate:    "0.01",
		batchSize:       4,
		trainChars:      65536,
		validationChars: 4096,
		blockSize:       32,
		layers:          4,
		heads:           4,
		embd:            64,
		optimizer:       "adamw",
		weightDecay:     "0.01",
		prompt:          "Once upon a time",
		tokens:          120,
		temperature:     "0.8",
		topK:            16,
		seed:            42,
	}

	fs := newFlagSet(a.stderr(), "train tiny-gpt")
	fs.StringVar(&opts.mode, "mode", opts.mode, "Zig optimization mode")
	fs.StringVar(&opts.mode, "optimize", opts.mode, "Zig optimization mode")
	fs.StringVar(&opts.output, "output", opts.output, "checkpoint path to write")
	fs.StringVar(&opts.resume, "resume", opts.resume, "checkpoint path to resume from")
	fs.StringVar(&opts.corpus, "corpus", opts.corpus, "auto, toy, shakespeare, or tinystories")
	fs.StringVar(&opts.corpusPath, "corpus-path", opts.corpusPath, "custom UTF-8 text corpus path")
	fs.IntVar(&opts.steps, "steps", opts.steps, "full-model training steps")
	fs.StringVar(&opts.learningRate, "learning-rate", opts.learningRate, "full-model learning rate")
	fs.IntVar(&opts.batchSize, "batch-size", opts.batchSize, "corpus windows per training step")
	fs.IntVar(&opts.trainChars, "train-chars", opts.trainChars, "training corpus characters")
	fs.IntVar(&opts.validationChars, "validation-chars", opts.validationChars, "validation holdout characters")
	fs.IntVar(&opts.blockSize, "block-size", opts.blockSize, "context length for new checkpoints")
	fs.IntVar(&opts.layers, "layers", opts.layers, "Transformer block count for new checkpoints")
	fs.IntVar(&opts.heads, "heads", opts.heads, "attention head count for new checkpoints")
	fs.IntVar(&opts.embd, "embd", opts.embd, "embedding/channel size for new checkpoints")
	fs.StringVar(&opts.optimizer, "optimizer", opts.optimizer, "sgd or adamw")
	fs.StringVar(&opts.weightDecay, "weight-decay", opts.weightDecay, "full-training weight decay")
	fs.StringVar(&opts.prompt, "prompt", opts.prompt, "prompt used for the post-training sample")
	fs.IntVar(&opts.tokens, "tokens", opts.tokens, "sample tokens after training")
	fs.StringVar(&opts.temperature, "temperature", opts.temperature, "sampling temperature")
	fs.IntVar(&opts.topK, "top-k", opts.topK, "top-k sampler cutoff")
	fs.IntVar(&opts.seed, "seed", opts.seed, "deterministic seed")
	fs.BoolVar(&opts.corpusPrior, "corpus-prior", opts.corpusPrior, "enable readable corpus-prior sampling in the post-training sample")
	if err := flagx.ParseInterspersed(fs, args, map[string]bool{
		"mode":             true,
		"optimize":         true,
		"output":           true,
		"resume":           true,
		"corpus":           true,
		"corpus-path":      true,
		"steps":            true,
		"learning-rate":    true,
		"batch-size":       true,
		"train-chars":      true,
		"validation-chars": true,
		"block-size":       true,
		"layers":           true,
		"heads":            true,
		"embd":             true,
		"optimizer":        true,
		"weight-decay":     true,
		"prompt":           true,
		"tokens":           true,
		"temperature":      true,
		"top-k":            true,
		"seed":             true,
		"corpus-prior":     false,
	}); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("train tiny-gpt does not accept positional arguments")
	}

	visited := visitedFlags(fs)
	if opts.resume != "" {
		if !visited["output"] {
			opts.output = opts.resume
		}
		for _, shapeFlag := range []string{"block-size", "layers", "heads", "embd"} {
			if visited[shapeFlag] {
				return fmt.Errorf("--%s cannot be used with --resume; the checkpoint already defines model shape", shapeFlag)
			}
		}
	}

	trainArgs, err := tinyGPTTrainArgs(opts)
	if err != nil {
		return err
	}
	return a.runZig(ctx, zig.RunArgs("run_tiny_gpt", zig.Options{Optimize: opts.mode}, trainArgs)...)
}

func tinyGPTTrainArgs(opts tinyGPTTrainOptions) ([]string, error) {
	if opts.steps < 0 || opts.batchSize <= 0 || opts.trainChars < 0 || opts.validationChars < 0 ||
		opts.tokens < 0 || opts.topK < 0 || opts.seed < 0 {
		return nil, fmt.Errorf("numeric train options must be non-negative, and batch size must be positive")
	}
	if opts.resume == "" && (opts.blockSize <= 0 || opts.layers <= 0 || opts.heads <= 0 || opts.embd <= 0) {
		return nil, fmt.Errorf("new checkpoint shape options must be positive")
	}
	if opts.output == "" && opts.resume == "" {
		return nil, fmt.Errorf("train tiny-gpt requires --output <checkpoint> or --resume <checkpoint>")
	}

	args := []string{
		"--train-full",
		"--full-train-steps", strconv.Itoa(opts.steps),
		"--full-learning-rate", opts.learningRate,
		"--full-batch-size", strconv.Itoa(opts.batchSize),
		"--train-chars", strconv.Itoa(opts.trainChars),
		"--validation-chars", strconv.Itoa(opts.validationChars),
		"--optimizer", opts.optimizer,
		"--weight-decay", opts.weightDecay,
		"--prompt", opts.prompt,
		"--tokens", strconv.Itoa(opts.tokens),
		"--temperature", opts.temperature,
		"--top-k", strconv.Itoa(opts.topK),
		"--seed", strconv.Itoa(opts.seed),
	}
	if opts.corpusPath != "" {
		args = append(args, "--corpus-path", opts.corpusPath)
	} else {
		args = append(args, "--corpus", opts.corpus)
	}
	if opts.resume != "" {
		args = append(args, "--resume-checkpoint", opts.resume)
	} else {
		args = append(args,
			"--block-size", strconv.Itoa(opts.blockSize),
			"--layers", strconv.Itoa(opts.layers),
			"--heads", strconv.Itoa(opts.heads),
			"--embd", strconv.Itoa(opts.embd),
		)
	}
	if opts.output != "" && opts.output != opts.resume {
		args = append(args, "--save-checkpoint", opts.output)
	}
	if !opts.corpusPrior {
		args = append(args, "--no-corpus-prior")
	}
	return args, nil
}

func visitedFlags(fs interface{ Visit(func(*flag.Flag)) }) map[string]bool {
	visited := map[string]bool{}
	fs.Visit(func(f *flag.Flag) {
		visited[f.Name] = true
	})
	return visited
}
