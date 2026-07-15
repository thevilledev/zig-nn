package cli

import (
	"context"
	"fmt"
	"strconv"

	"nnctl/internal/zig"
)

type tinyGPTTrainOptions struct {
	mode            string
	output          string
	resume          string
	preset          string
	corpus          string
	corpusPath      string
	steps           int
	learningRate    string
	minLearningRate string
	lrSchedule      string
	warmupSteps     int
	batchSize       int
	trainChars      int
	validationChars int
	evalSplit       string
	evalWindows     int
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
	summaryPath     string
}

type speechCommandsTrainOptions struct {
	mode         string
	gpu          string
	dataDir      string
	output       string
	epochs       int
	batchSize    int
	learningRate string
	seed         int
}

func defaultSpeechCommandsTrainOptions() speechCommandsTrainOptions {
	return speechCommandsTrainOptions{
		mode:         "ReleaseFast",
		gpu:          "auto",
		dataDir:      "data/mini_speech_commands",
		output:       "speech-commands.bin",
		epochs:       12,
		batchSize:    64,
		learningRate: "0.003",
		seed:         42,
	}
}

func defaultTinyGPTTrainOptions() tinyGPTTrainOptions {
	return tinyGPTTrainOptions{
		mode:            getenvDefault("NNCTL_OPTIMIZE", "ReleaseFast"),
		output:          "tiny-gpt.bin",
		corpus:          "auto",
		steps:           5000,
		learningRate:    "0.01",
		minLearningRate: "0.001",
		lrSchedule:      "cosine",
		warmupSteps:     50,
		batchSize:       4,
		trainChars:      65536,
		validationChars: 0,
		evalSplit:       "0.05",
		evalWindows:     32,
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
}

func applyTinyGPTTrainPreset(opts *tinyGPTTrainOptions, changed func(string) bool) error {
	switch opts.preset {
	case "", "none":
		return nil
	case "coherent-small":
		setStringIfUnchanged(&opts.corpus, "tinystories", "corpus", changed)
		setIntIfUnchanged(&opts.steps, 3000, "steps", changed)
		setStringIfUnchanged(&opts.learningRate, "0.001", "learning-rate", changed)
		setStringIfUnchanged(&opts.minLearningRate, "0.0001", "min-learning-rate", changed)
		setStringIfUnchanged(&opts.lrSchedule, "cosine", "lr-schedule", changed)
		setIntIfUnchanged(&opts.warmupSteps, 100, "warmup-steps", changed)
		setIntIfUnchanged(&opts.batchSize, 12, "batch-size", changed)
		setIntIfUnchanged(&opts.trainChars, 0, "train-chars", changed)
		setStringIfUnchanged(&opts.evalSplit, "0.05", "eval-split", changed)
		setIntIfUnchanged(&opts.evalWindows, 64, "eval-windows", changed)
		setIntIfUnchanged(&opts.blockSize, 64, "block-size", changed)
		setIntIfUnchanged(&opts.layers, 4, "layers", changed)
		setIntIfUnchanged(&opts.heads, 4, "heads", changed)
		setIntIfUnchanged(&opts.embd, 128, "embd", changed)
		setStringIfUnchanged(&opts.optimizer, "adamw", "optimizer", changed)
		setStringIfUnchanged(&opts.weightDecay, "0.01", "weight-decay", changed)
		setStringIfUnchanged(&opts.temperature, "0.8", "temperature", changed)
		setIntIfUnchanged(&opts.topK, 16, "top-k", changed)
		setBoolIfUnchanged(&opts.corpusPrior, false, "corpus-prior", changed)
		return nil
	default:
		return fmt.Errorf("unknown TinyGPT preset %q", opts.preset)
	}
}

func setStringIfUnchanged(target *string, value, flag string, changed func(string) bool) {
	if changed == nil || !changed(flag) {
		*target = value
	}
}

func setIntIfUnchanged(target *int, value int, flag string, changed func(string) bool) {
	if changed == nil || !changed(flag) {
		*target = value
	}
}

func setBoolIfUnchanged(target *bool, value bool, flag string, changed func(string) bool) {
	if changed == nil || !changed(flag) {
		*target = value
	}
}

func prepareTinyGPTResumeOptions(opts *tinyGPTTrainOptions, changed func(string) bool) error {
	if opts.resume != "" {
		if !changed("output") {
			opts.output = opts.resume
		}
		for _, shapeFlag := range []string{"block-size", "layers", "heads", "embd"} {
			if changed(shapeFlag) {
				return fmt.Errorf("--%s cannot be used with --resume; the checkpoint already defines model shape", shapeFlag)
			}
		}
	}
	return nil
}

func (a *App) runTrainTinyGPT(ctx context.Context, opts tinyGPTTrainOptions) error {
	trainArgs, err := tinyGPTTrainArgs(opts)
	if err != nil {
		return err
	}
	return a.runZig(ctx, zig.RunArgs("run_tiny_gpt", zig.Options{Optimize: opts.mode}, trainArgs)...)
}

func tinyGPTTrainArgs(opts tinyGPTTrainOptions) ([]string, error) {
	if opts.minLearningRate == "" {
		opts.minLearningRate = "0"
	}
	if opts.lrSchedule == "" {
		opts.lrSchedule = "constant"
	}
	if opts.evalSplit == "" {
		opts.evalSplit = "0"
	}
	if opts.evalWindows <= 0 {
		opts.evalWindows = 12
	}
	if opts.steps < 0 || opts.warmupSteps < 0 || opts.batchSize <= 0 || opts.trainChars < 0 || opts.validationChars < 0 ||
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
		"--min-learning-rate", opts.minLearningRate,
		"--lr-schedule", opts.lrSchedule,
		"--warmup-steps", strconv.Itoa(opts.warmupSteps),
		"--full-batch-size", strconv.Itoa(opts.batchSize),
		"--train-chars", strconv.Itoa(opts.trainChars),
		"--validation-chars", strconv.Itoa(opts.validationChars),
		"--eval-split", opts.evalSplit,
		"--eval-windows", strconv.Itoa(opts.evalWindows),
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
	if opts.summaryPath != "" {
		args = append(args, "--summary-path", opts.summaryPath)
	}
	return args, nil
}

func (a *App) runTrainSpeechCommands(ctx context.Context, opts speechCommandsTrainOptions) error {
	trainArgs, err := speechCommandsTrainArgs(opts)
	if err != nil {
		return err
	}
	return a.runZig(ctx, zig.RunArgs("run_speech_commands", zig.Options{Optimize: opts.mode, GPU: opts.gpu}, trainArgs)...)
}

func speechCommandsTrainArgs(opts speechCommandsTrainOptions) ([]string, error) {
	if opts.dataDir == "" || opts.output == "" {
		return nil, fmt.Errorf("speech command training requires --data-dir and --output")
	}
	if opts.epochs <= 0 || opts.batchSize <= 0 || opts.seed < 0 {
		return nil, fmt.Errorf("epochs and batch size must be positive, and seed must be non-negative")
	}
	backend := opts.gpu
	switch backend {
	case "", "auto":
		backend = "auto"
	case "none":
		backend = "cpu"
	case "metal", "cuda", "rocm":
	default:
		return nil, fmt.Errorf("unknown GPU backend %q", opts.gpu)
	}
	return []string{
		"--train",
		"--data-dir", opts.dataDir,
		"--output", opts.output,
		"--epochs", strconv.Itoa(opts.epochs),
		"--batch-size", strconv.Itoa(opts.batchSize),
		"--learning-rate", opts.learningRate,
		"--seed", strconv.Itoa(opts.seed),
		"--backend", backend,
	}, nil
}
