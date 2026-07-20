package cli

import (
	"context"
	"fmt"
	"runtime"
	"strings"

	"github.com/spf13/cobra"

	"nnctl/internal/catalog"
)

func (a *app) newExperimentsCommand(withRepo repoRunner) *cobra.Command {
	opts := buildOptions{
		mode: getenvDefault("NNCTL_OPTIMIZE", getenvDefault("BUILD_MODE", defaultBuildMode)),
	}
	cmd := &cobra.Command{
		Use:     "experiments",
		Aliases: []string{"examples"},
		Short:   "Build all experiments",
		Long:    "Builds all experiment executables without running them. Use nnctl run <experiment> to run one experiment.",
		Args:    cobra.NoArgs,
		RunE: withRepo(func(ctx context.Context, cmd *cobra.Command, args []string) error {
			return a.runExperiments(ctx, opts)
		}),
	}
	addModeFlags(cmd, &opts.mode)
	addGPUFlag(cmd, &opts.gpu)
	return cmd
}

func (a *app) newRunCommand(withRepo repoRunner) *cobra.Command {
	mode := ""
	gpu := ""
	cmd := &cobra.Command{
		Use:   "run <experiment>",
		Short: "Run an experiment",
		Long:  `Runs a named experiment. Flags configure the Zig build. Args after "--" are passed to the experiment executable.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			return cmd.Help()
		},
	}
	addModeFlags(cmd, &mode)
	addGPUFlag(cmd, &gpu)

	quick := &cobra.Command{
		Use:   "quick",
		Short: `Run all experiments marked as quick`,
		Args:  cobra.ArbitraryArgs,
		RunE: withRepo(func(ctx context.Context, cmd *cobra.Command, args []string) error {
			return a.runExperiment(ctx, "quick", nil, mode, gpu)
		}),
	}
	cmd.AddCommand(quick)

	for _, experiment := range catalog.SortedExperiments() {
		if experiment.Hidden {
			continue
		}
		experiment := experiment
		experimentCmd := &cobra.Command{
			Use:     experiment.Name + " [-- experiment args]",
			Aliases: experimentAliases(experiment),
			Short:   experiment.Description,
			Long:    experimentHelpLong(experiment),
			Args:    cobra.ArbitraryArgs,
			RunE: withRepo(func(ctx context.Context, cmd *cobra.Command, args []string) error {
				return a.runExperiment(ctx, experiment.Name, args, mode, gpu)
			}),
		}
		cmd.AddCommand(experimentCmd)
	}
	return cmd
}

func (a *app) newTrainCommand(withRepo repoRunner) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "train <target>",
		Short: "Train model checkpoints",
		RunE: func(cmd *cobra.Command, args []string) error {
			return cmd.Help()
		},
	}
	cmd.AddCommand(a.newTrainTinyGPTCommand(withRepo), a.newTrainSpeechCommandsCommand(withRepo))
	return cmd
}

func (a *app) newTrainSpeechCommandsCommand(withRepo repoRunner) *cobra.Command {
	opts := defaultSpeechCommandsTrainOptions()
	cmd := &cobra.Command{
		Use:     "speech-commands",
		Aliases: []string{"speechcommands"},
		Short:   "Train an eight-word speech command checkpoint",
		Long: `Trains a small device-backed MLP from scratch on Mini Speech Commands.
Prepare the official dataset first with "nnctl data speech-commands". The best
validation checkpoint is evaluated once on a speaker-disjoint held-out split.`,
		Example: `  nnctl data speech-commands
  nnctl train speech-commands --output speech-commands.bin
  nnctl run speech-commands -- --model speech-commands.bin --input clip.wav`,
		Args: cobra.NoArgs,
		RunE: withRepo(func(ctx context.Context, cmd *cobra.Command, args []string) error {
			return a.runTrainSpeechCommands(ctx, opts)
		}),
	}
	addModeFlags(cmd, &opts.mode)
	addGPUFlag(cmd, &opts.gpu)
	cmd.Flags().StringVar(&opts.dataDir, "data-dir", opts.dataDir, "Mini Speech Commands directory")
	cmd.Flags().StringVar(&opts.output, "output", opts.output, "best-validation checkpoint path")
	cmd.Flags().IntVar(&opts.epochs, "epochs", opts.epochs, "training epochs")
	cmd.Flags().IntVar(&opts.batchSize, "batch-size", opts.batchSize, "training batch size")
	cmd.Flags().StringVar(&opts.learningRate, "learning-rate", opts.learningRate, "AdamW learning rate")
	cmd.Flags().IntVar(&opts.seed, "seed", opts.seed, "split, shuffle, and initialization seed")
	cmd.Flags().SortFlags = false
	return cmd
}

func (a *app) newTrainTinyGPTCommand(withRepo repoRunner) *cobra.Command {
	opts := defaultTinyGPTTrainOptions()
	cmd := &cobra.Command{
		Use:     "tiny-gpt",
		Aliases: []string{"tinygpt"},
		Short:   "Train a TinyGPT checkpoint",
		Long: `Trains all TinyGPT Transformer weights, embeddings, layer norms, and the output head.
Fresh checkpoints accept model-shape flags; resumed checkpoints load the shape from the checkpoint.
Runs include an explicit train/eval split, validation loss, scheduled learning rates, checkpoint metadata, and optional JSON summaries.

Corpora:
  auto         Use prepared TinyStories, then Tiny Shakespeare, then toy.
  toy          Always available from experiments/tiny_gpt/data/toy.txt.
  shakespeare  Requires "nnctl data tiny-gpt".
  tinystories  Requires "nnctl data tiny-gpt".

Use --corpus-path to train on any local UTF-8 text file instead of a preset.`,
		Example: `  nnctl data tiny-gpt
  nnctl train tiny-gpt --preset coherent-small --output tiny-gpt.bin
  nnctl train tiny-gpt --corpus toy --output toy-gpt.bin
  nnctl train tiny-gpt --corpus shakespeare --output shakespeare-gpt.bin
  nnctl train tiny-gpt --corpus tinystories --output tinystories-gpt.bin
  nnctl train tiny-gpt --corpus tinystories --summary-path runs/tinystories.json
  nnctl train tiny-gpt --corpus-path /tmp/my-corpus.txt --output custom-gpt.bin
  nnctl train tiny-gpt --resume tinystories-gpt.bin --steps 2000
  nnctl chat --model tinystories-gpt.bin`,
		Args: cobra.NoArgs,
		RunE: withRepo(func(ctx context.Context, cmd *cobra.Command, args []string) error {
			if err := applyTinyGPTTrainPreset(&opts, cmd.Flags().Changed); err != nil {
				return err
			}
			if err := prepareTinyGPTResumeOptions(&opts, cmd.Flags().Changed); err != nil {
				return err
			}
			return a.runTrainTinyGPT(ctx, opts)
		}),
	}
	addTinyGPTTrainFlags(cmd, &opts)
	return cmd
}

func (a *app) newModelCommand(withRepo repoRunner) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "model",
		Short: "Inspect model checkpoints",
		RunE: func(cmd *cobra.Command, args []string) error {
			return cmd.Help()
		},
	}
	opts := modelInspectOptions{}
	inspect := &cobra.Command{
		Use:   "inspect CHECKPOINT",
		Short: "Show checkpoint type, version, and dimensions",
		Args:  cobra.ExactArgs(1),
		RunE: withRepo(func(ctx context.Context, cmd *cobra.Command, args []string) error {
			return a.runModelInspect(args[0], opts)
		}),
	}
	inspect.Flags().BoolVar(&opts.json, "json", opts.json, "print machine-readable JSON")
	cmd.AddCommand(inspect)
	return cmd
}

func (a *app) newPredictCommand(withRepo repoRunner) *cobra.Command {
	opts := defaultPredictOptions()
	cmd := &cobra.Command{
		Use:   "predict",
		Short: "Run dense checkpoint inference",
		Example: `  nnctl predict --model xor_model.bin --input '[0,1]'
  nnctl predict --model classifier.bin --input '[1,2,3,4]' --batch-size 2 --gpu cpu`,
		Args: cobra.NoArgs,
		RunE: withRepo(func(ctx context.Context, cmd *cobra.Command, args []string) error {
			return a.runPredict(ctx, opts)
		}),
	}
	addModeFlags(cmd, &opts.mode)
	addGPUFlag(cmd, &opts.backend)
	cmd.Flags().StringVar(&opts.modelPath, "model", opts.modelPath, "ZNN checkpoint to load")
	cmd.Flags().StringVar(&opts.inputJSON, "input", opts.inputJSON, "flat JSON number array")
	cmd.Flags().IntVar(&opts.batchSize, "batch-size", opts.batchSize, "number of input rows")
	cmd.Flags().SortFlags = false
	return cmd
}

func (a *app) newServeCommand(withRepo repoRunner) *cobra.Command {
	opts := defaultServeOptions()
	cmd := &cobra.Command{
		Use:   "serve",
		Short: "Serve a ZNN or TinyGPT checkpoint",
		Long:  "Detects the checkpoint format and starts the matching persistent inference service on a loopback address.",
		Example: `  nnctl serve --model xor_model.bin
  nnctl serve --model tiny-gpt.bin --gpu auto --port 8080`,
		Args: cobra.NoArgs,
		RunE: withRepo(func(ctx context.Context, cmd *cobra.Command, args []string) error {
			return a.runServe(ctx, opts)
		}),
	}
	addModeFlags(cmd, &opts.mode)
	addGPUFlag(cmd, &opts.backend)
	cmd.Flags().StringVar(&opts.modelPath, "model", opts.modelPath, "ZNN or TinyGPT checkpoint to serve")
	cmd.Flags().StringVar(&opts.modelName, "model-name", opts.modelName, "OpenAI-compatible TinyGPT model id")
	cmd.Flags().StringVar(&opts.host, "host", opts.host, "loopback bind host")
	cmd.Flags().IntVar(&opts.port, "port", opts.port, "bind port")
	cmd.Flags().SortFlags = false
	return cmd
}

func (a *app) newChatCommand(withRepo repoRunner) *cobra.Command {
	opts := defaultChatOptions()
	cmd := &cobra.Command{
		Use:   "chat",
		Short: "Run TinyGPT inference with a local chat app",
		Long:  "Starts the TinyGPT OpenAI-compatible inference server and serves a small local browser chat UI that proxies requests to it.",
		Example: `  nnctl chat --model tiny-gpt.bin
  nnctl chat --allow-untrained --port 8090`,
		Args: cobra.NoArgs,
		RunE: withRepo(func(ctx context.Context, cmd *cobra.Command, args []string) error {
			return a.runChat(ctx, opts)
		}),
	}
	addModeFlags(cmd, &opts.mode)
	addGPUFlag(cmd, &opts.backend)
	cmd.Flags().StringVar(&opts.host, "host", opts.host, "chat app bind host")
	cmd.Flags().IntVar(&opts.port, "port", opts.port, "chat app bind port")
	cmd.Flags().StringVar(&opts.apiHost, "api-host", opts.apiHost, "TinyGPT inference bind host")
	cmd.Flags().IntVar(&opts.apiPort, "api-port", opts.apiPort, "TinyGPT inference bind port")
	cmd.Flags().StringVar(&opts.modelPath, "model", opts.modelPath, "TinyGPT checkpoint to serve")
	cmd.Flags().StringVar(&opts.modelName, "model-name", opts.modelName, "OpenAI-compatible model id")
	cmd.Flags().IntVar(&opts.maxTokens, "max-tokens", opts.maxTokens, "default generated tokens per reply")
	cmd.Flags().Float64Var(&opts.temperature, "temperature", opts.temperature, "default sampling temperature")
	cmd.Flags().IntVar(&opts.topK, "top-k", opts.topK, "default TinyGPT top-k sampler cutoff")
	cmd.Flags().BoolVar(&opts.allowUntrained, "allow-untrained", opts.allowUntrained, "serve a seeded untrained model when --model is omitted")
	cmd.Flags().StringVar(&opts.readyTimeoutText, "ready-timeout", opts.readyTimeoutText, "time to wait for inference server readiness")
	cmd.Flags().SortFlags = false
	return cmd
}

func (a *app) newListCommand(root *cobra.Command) *cobra.Command {
	return &cobra.Command{
		Use:   "list [topic]",
		Short: "List tasks, experiments, or tests",
		Args:  cobra.MaximumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			topic := "tasks"
			if len(args) == 1 {
				topic = catalog.NormalizeName(args[0])
			}
			switch topic {
			case "tasks", "commands":
				return root.Help()
			case "experiments", "experiment", "examples", "example":
				return a.listExperiments()
			case "tests", "test":
				return a.listTests()
			default:
				return fmt.Errorf("unknown list topic %q", topic)
			}
		},
	}
}

func (a *app) newFormatCommand(withRepo repoRunner) *cobra.Command {
	check := false
	cmd := &cobra.Command{
		Use:     "fmt",
		Aliases: []string{"format"},
		Short:   "Format Zig source",
		Args:    cobra.NoArgs,
		RunE: withRepo(func(ctx context.Context, cmd *cobra.Command, args []string) error {
			return a.runFormat(ctx, check)
		}),
	}
	cmd.Flags().BoolVar(&check, "check", false, "check formatting without rewriting files")
	return cmd
}

func (a *app) newCleanCommand(withRepo repoRunner) *cobra.Command {
	includeTool := false
	cmd := &cobra.Command{
		Use:   "clean",
		Short: "Remove Zig build artifacts",
		Args:  cobra.NoArgs,
		RunE: withRepo(func(ctx context.Context, cmd *cobra.Command, args []string) error {
			return a.runClean(includeTool)
		}),
	}
	cmd.Flags().BoolVar(&includeTool, "tool", false, "also remove bin/nnctl")
	return cmd
}

func (a *app) newDataCommand(withRepo repoRunner) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "data <command>",
		Short: "Download or prepare experiment data",
		RunE: func(cmd *cobra.Command, args []string) error {
			return cmd.Help()
		},
	}
	cmd.AddCommand(a.newDataMNISTCommand(withRepo), a.newDataTinyGPTCommand(withRepo), a.newDataSpeechCommandsCommand(withRepo))
	return cmd
}

func (a *app) newDataSpeechCommandsCommand(withRepo repoRunner) *cobra.Command {
	dir := "data/mini_speech_commands"
	force := false
	cmd := &cobra.Command{
		Use:     "speech-commands",
		Aliases: []string{"speechcommands"},
		Short:   "Download and extract Mini Speech Commands",
		Long:    "Downloads the official eight-word Mini Speech Commands WAV dataset and extracts it safely into the repository data directory.",
		Example: `  nnctl data speech-commands
  nnctl train speech-commands --output speech-commands.bin`,
		Args: cobra.NoArgs,
		RunE: withRepo(func(ctx context.Context, cmd *cobra.Command, args []string) error {
			return a.runDataSpeechCommands(ctx, dir, force)
		}),
	}
	cmd.Flags().StringVar(&dir, "dir", dir, "dataset output directory")
	cmd.Flags().BoolVar(&force, "force", false, "replace an existing prepared dataset")
	cmd.Flags().SortFlags = false
	return cmd
}

func (a *app) newDataMNISTCommand(withRepo repoRunner) *cobra.Command {
	dir := "data"
	force := false
	cmd := &cobra.Command{
		Use:   "mnist",
		Short: "Download and extract the MNIST dataset",
		Long:  "Downloads and extracts the four MNIST IDX files expected by the MNIST experiment.",
		Args:  cobra.NoArgs,
		RunE: withRepo(func(ctx context.Context, cmd *cobra.Command, args []string) error {
			return a.runDataMNIST(ctx, dir, force)
		}),
	}
	cmd.Flags().StringVar(&dir, "dir", dir, "data directory")
	cmd.Flags().BoolVar(&force, "force", false, "overwrite existing files")
	cmd.Flags().SortFlags = false
	return cmd
}

func (a *app) newDataTinyGPTCommand(withRepo repoRunner) *cobra.Command {
	return &cobra.Command{
		Use:     "tiny-gpt",
		Aliases: []string{"tinygpt"},
		Short:   "Download sourced TinyGPT demo corpora",
		Long: `Downloads Tiny Shakespeare and a TinyStories validation split, then writes a small TinyStories slice used by the TinyGPT preset.

Prepared files:
  experiments/tiny_gpt/data/shakespeare/input.txt
  experiments/tiny_gpt/data/tinystories/valid.txt
  experiments/tiny_gpt/data/tinystories/tinystories_1mb.txt

The checked-in toy corpus works offline and does not need this command.
Set TINYSTORIES_BYTES before running to change the TinyStories slice size.`,
		Example: `  nnctl data tiny-gpt
  nnctl run tiny-gpt -- --corpus shakespeare
  nnctl train tiny-gpt --corpus tinystories --output tinystories-gpt.bin`,
		Args: cobra.NoArgs,
		RunE: withRepo(func(ctx context.Context, cmd *cobra.Command, args []string) error {
			return a.runDataTinyGPT(ctx)
		}),
	}
}

func (a *app) newDoctorCommand(withRepo repoRunner) *cobra.Command {
	return &cobra.Command{
		Use:     "doctor",
		Aliases: []string{"env"},
		Short:   "Check local tools and backend support",
		Args:    cobra.NoArgs,
		RunE: withRepo(func(ctx context.Context, cmd *cobra.Command, args []string) error {
			return a.runDoctor(ctx)
		}),
	}
}

func (a *app) newTinyGPTTopicCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "tiny-gpt",
		Short: "Show TinyGPT workflows",
		Long: `TinyGPT can be used as a quick runnable demo, a trainable checkpoint target, or a local chat model.

Common workflows:
  nnctl data tiny-gpt
  nnctl run tiny-gpt -- --corpus toy --prompt "to be"
  nnctl train tiny-gpt --preset coherent-small --output tiny-gpt.bin
  nnctl chat --model tiny-gpt.bin

Corpora:
  auto         Use prepared TinyStories, then Tiny Shakespeare, then toy.
  toy          Works offline from experiments/tiny_gpt/data/toy.txt.
  shakespeare  Requires "nnctl data tiny-gpt".
  tinystories  Requires "nnctl data tiny-gpt".

Use --corpus-path with run or train to use any local UTF-8 text file instead of a preset.`,
		Example: `  nnctl help run tiny-gpt
  nnctl help train tiny-gpt
  nnctl help data tiny-gpt`,
		RunE: func(cmd *cobra.Command, args []string) error {
			return cmd.Help()
		},
	}
}

func newVersionCommand(a *app) *cobra.Command {
	return &cobra.Command{
		Use:   "version",
		Short: "Print nnctl version",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			if _, err := fmt.Fprintf(a.stdout(), "nnctl dev (%s)\n", runtime.Version()); err != nil {
				return fmt.Errorf("write nnctl version: %w", err)
			}
			return nil
		},
	}
}

type repoRunner func(func(context.Context, *cobra.Command, []string) error) func(*cobra.Command, []string) error

func addModeFlags(cmd *cobra.Command, mode *string) {
	cmd.Flags().StringVar(mode, "mode", *mode, "Zig optimization mode")
	cmd.Flags().StringVar(mode, "optimize", *mode, "Zig optimization mode")
	cmd.Flags().SortFlags = false
}

func addGPUFlag(cmd *cobra.Command, gpu *string) {
	cmd.Flags().StringVar(gpu, "gpu", *gpu, "GPU backend: none, auto, metal, cuda, or rocm")
	cmd.Flags().SortFlags = false
}

func addTinyGPTTrainFlags(cmd *cobra.Command, opts *tinyGPTTrainOptions) {
	addModeFlags(cmd, &opts.mode)
	cmd.Flags().StringVar(&opts.output, "output", opts.output, "checkpoint path to write")
	cmd.Flags().StringVar(&opts.resume, "resume", opts.resume, "checkpoint path to resume from")
	cmd.Flags().StringVar(&opts.preset, "preset", opts.preset, "training preset: coherent-small")
	cmd.Flags().StringVar(&opts.corpus, "corpus", opts.corpus, "auto, toy, shakespeare, or tinystories")
	cmd.Flags().StringVar(&opts.corpusPath, "corpus-path", opts.corpusPath, "custom UTF-8 text corpus path")
	cmd.Flags().IntVar(&opts.steps, "steps", opts.steps, "full-model training steps")
	cmd.Flags().StringVar(&opts.learningRate, "learning-rate", opts.learningRate, "full-model learning rate")
	cmd.Flags().StringVar(&opts.minLearningRate, "min-learning-rate", opts.minLearningRate, "learning-rate floor for schedules")
	cmd.Flags().StringVar(&opts.lrSchedule, "lr-schedule", opts.lrSchedule, "constant, linear, or cosine")
	cmd.Flags().IntVar(&opts.warmupSteps, "warmup-steps", opts.warmupSteps, "linear warmup steps before LR decay")
	cmd.Flags().IntVar(&opts.batchSize, "batch-size", opts.batchSize, "corpus windows per training step")
	cmd.Flags().IntVar(&opts.trainChars, "train-chars", opts.trainChars, "training corpus characters")
	cmd.Flags().IntVar(&opts.validationChars, "validation-chars", opts.validationChars, "validation holdout characters")
	cmd.Flags().IntVar(&opts.validationChars, "eval-chars", opts.validationChars, "alias for --validation-chars")
	cmd.Flags().StringVar(&opts.evalSplit, "eval-split", opts.evalSplit, "eval fraction when validation chars are 0")
	cmd.Flags().IntVar(&opts.evalWindows, "eval-windows", opts.evalWindows, "sampled windows for train/eval loss")
	cmd.Flags().IntVar(&opts.blockSize, "block-size", opts.blockSize, "context length for new checkpoints")
	cmd.Flags().IntVar(&opts.layers, "layers", opts.layers, "Transformer block count for new checkpoints")
	cmd.Flags().IntVar(&opts.heads, "heads", opts.heads, "attention head count for new checkpoints")
	cmd.Flags().IntVar(&opts.embd, "embd", opts.embd, "embedding/channel size for new checkpoints")
	cmd.Flags().StringVar(&opts.optimizer, "optimizer", opts.optimizer, "sgd or adamw")
	cmd.Flags().StringVar(&opts.weightDecay, "weight-decay", opts.weightDecay, "full-training weight decay")
	cmd.Flags().StringVar(&opts.prompt, "prompt", opts.prompt, "prompt used for the post-training sample")
	cmd.Flags().IntVar(&opts.tokens, "tokens", opts.tokens, "sample tokens after training")
	cmd.Flags().StringVar(&opts.temperature, "temperature", opts.temperature, "sampling temperature")
	cmd.Flags().IntVar(&opts.topK, "top-k", opts.topK, "top-k sampler cutoff")
	cmd.Flags().IntVar(&opts.seed, "seed", opts.seed, "deterministic seed")
	cmd.Flags().BoolVar(&opts.corpusPrior, "corpus-prior", opts.corpusPrior, "enable readable corpus-prior sampling in the post-training sample")
	cmd.Flags().StringVar(&opts.summaryPath, "summary-path", opts.summaryPath, "write deterministic JSON run summary")
	cmd.Flags().SortFlags = false
}

func modeFlagChanged(cmd *cobra.Command) bool {
	return cmd.Flags().Changed("mode") || cmd.Flags().Changed("optimize")
}

func experimentAliases(experiment catalog.Experiment) []string {
	aliases := append([]string(nil), experiment.Aliases...)
	underscored := strings.ReplaceAll(experiment.Name, "-", "_")
	if underscored != experiment.Name {
		aliases = append(aliases, underscored)
	}
	return aliases
}

func experimentHelpLong(experiment catalog.Experiment) string {
	if catalog.NormalizeName(experiment.Name) != "tiny-gpt" {
		return experiment.Description
	}
	return `Runs the TinyGPT demo. This path does quick output-head training by default, then samples text. Pass TinyGPT flags after "--".

Common TinyGPT args:
  --corpus NAME          auto, toy, shakespeare, or tinystories
  --corpus-path PATH     use a custom UTF-8 text corpus
  --prompt TEXT          prompt to continue
  --tokens N             number of new tokens to sample
  --no-train             skip demo output-head training
  --no-corpus-prior      disable readable corpus-prior sampling
  --load-checkpoint PATH load a checkpoint before generation

Corpora:
  auto         Use prepared TinyStories, then Tiny Shakespeare, then toy.
  toy          Use the checked-in offline corpus.
  shakespeare  Use data prepared by "nnctl data tiny-gpt".
  tinystories  Use data prepared by "nnctl data tiny-gpt".

Examples:
  nnctl data tiny-gpt
  nnctl run tiny-gpt -- --corpus toy --prompt "to be"
  nnctl run tiny-gpt -- --corpus shakespeare --prompt "to be" --tokens 120
  nnctl run tiny-gpt -- --corpus-path /tmp/my-corpus.txt --prompt "Once"
  nnctl run tiny-gpt -- --help`
}
