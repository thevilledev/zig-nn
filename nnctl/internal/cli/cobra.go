package cli

import (
	"context"
	"fmt"
	"runtime"
	"strings"

	"github.com/spf13/cobra"

	"nnctl/internal/catalog"
	"nnctl/internal/repo"
)

func (a *App) newRootCommand() *cobra.Command {
	repoRoot := ""
	zigPath := getenvDefault("ZIG", "zig")
	configured := false

	configureRepo := func() error {
		if configured {
			return nil
		}
		root, err := repo.ResolveRoot(repoRoot)
		if err != nil {
			return err
		}
		a.repoRoot = root
		a.zig = zigPath
		configured = true
		return nil
	}

	withRepo := func(run func(context.Context, *cobra.Command, []string) error) func(*cobra.Command, []string) error {
		return func(cmd *cobra.Command, args []string) error {
			if err := configureRepo(); err != nil {
				return err
			}
			return run(cmd.Context(), cmd, args)
		}
	}

	root := &cobra.Command{
		Use:           "nnctl",
		Short:         "Repository task runner for zig-nn",
		Long:          "nnctl builds, tests, runs examples, prepares data, and trains or serves the TinyGPT demo in this repository.",
		SilenceErrors: true,
		SilenceUsage:  true,
		Version:       fmt.Sprintf("dev (%s)", runtime.Version()),
		RunE: func(cmd *cobra.Command, args []string) error {
			return cmd.Help()
		},
	}
	root.SetOut(a.stdout())
	root.SetErr(a.stderr())
	root.SetIn(a.stdin())
	root.SetVersionTemplate("nnctl {{.Version}}\n")
	root.CompletionOptions.DisableDefaultCmd = true
	root.PersistentFlags().SortFlags = false
	root.PersistentFlags().StringVarP(&repoRoot, "repo", "C", "", "repository root to operate on")
	root.PersistentFlags().StringVar(&zigPath, "zig", zigPath, "Zig executable to use")

	root.AddCommand(
		a.newAllCommand(withRepo),
		a.newBuildCommand(withRepo),
		a.newReleaseCommand(withRepo),
		a.newTestCommand(withRepo),
		a.newExamplesCommand(withRepo),
		a.newRunCommand(withRepo),
		a.newTrainCommand(withRepo),
		a.newChatCommand(withRepo),
		a.newListCommand(root),
		a.newFormatCommand(withRepo),
		a.newCleanCommand(withRepo),
		a.newDataCommand(withRepo),
		a.newDoctorCommand(withRepo),
		a.newTinyGPTTopicCommand(),
		newVersionCommand(a),
	)
	return root
}

func (a *App) newAllCommand(withRepo repoRunner) *cobra.Command {
	opts := buildOptions{
		mode: getenvDefault("NNCTL_OPTIMIZE", getenvDefault("BUILD_MODE", defaultBuildMode)),
	}
	cmd := &cobra.Command{
		Use:   "all",
		Short: "Build, test, and build examples",
		Long:  "Builds the library, runs unit tests, runs acceptance tests, and builds all examples.",
		Args:  cobra.NoArgs,
		RunE: withRepo(func(ctx context.Context, cmd *cobra.Command, args []string) error {
			return a.runAll(ctx, opts)
		}),
	}
	addModeFlags(cmd, &opts.mode)
	addGPUFlag(cmd, &opts.gpu)
	return cmd
}

func (a *App) newBuildCommand(withRepo repoRunner) *cobra.Command {
	opts := buildOptions{
		mode: getenvDefault("NNCTL_OPTIMIZE", getenvDefault("BUILD_MODE", defaultBuildMode)),
	}
	release := false
	cmd := &cobra.Command{
		Use:   "build",
		Short: "Build the Zig library",
		Args:  cobra.NoArgs,
		RunE: withRepo(func(ctx context.Context, cmd *cobra.Command, args []string) error {
			if release && !modeFlagChanged(cmd) {
				opts.mode = getenvDefault("NNCTL_RELEASE_OPTIMIZE", getenvDefault("RELEASE_MODE", defaultReleaseMode))
			}
			return a.runBuild(ctx, opts)
		}),
	}
	addModeFlags(cmd, &opts.mode)
	addGPUFlag(cmd, &opts.gpu)
	cmd.Flags().BoolVar(&release, "release", release, "build with the release optimization default")
	cmd.Flags().SortFlags = false
	return cmd
}

func (a *App) newReleaseCommand(withRepo repoRunner) *cobra.Command {
	opts := buildOptions{
		mode: getenvDefault("NNCTL_RELEASE_OPTIMIZE", getenvDefault("RELEASE_MODE", defaultReleaseMode)),
	}
	cmd := &cobra.Command{
		Use:   "release",
		Short: "Build the Zig library with release defaults",
		Args:  cobra.NoArgs,
		RunE: withRepo(func(ctx context.Context, cmd *cobra.Command, args []string) error {
			return a.runBuild(ctx, opts)
		}),
	}
	addModeFlags(cmd, &opts.mode)
	addGPUFlag(cmd, &opts.gpu)
	return cmd
}

func (a *App) newTestCommand(withRepo repoRunner) *cobra.Command {
	opts := testOptions{}
	cmd := &cobra.Command{
		Use:   "test [flags] [name]",
		Short: "Run unit and acceptance tests",
		Long:  `Runs unit tests and then acceptance tests. Pass a test name to run one Zig test step, such as "matrix" or "tiny-gpt".`,
		Args:  cobra.MaximumNArgs(1),
		RunE: withRepo(func(ctx context.Context, cmd *cobra.Command, args []string) error {
			if len(args) == 1 {
				opts.name = args[0]
			}
			return a.runTest(ctx, opts)
		}),
	}
	addModeFlags(cmd, &opts.mode)
	addGPUFlag(cmd, &opts.gpu)
	cmd.Flags().StringVar(&opts.summary, "summary", opts.summary, "Zig build summary mode, for example all")
	cmd.Flags().BoolVar(&opts.unitOnly, "unit-only", false, "run only unit tests")
	cmd.Flags().BoolVar(&opts.acceptanceOnly, "acceptance-only", false, "run only acceptance tests")
	cmd.Flags().StringVar(&opts.name, "name", "", "individual test name")
	cmd.Flags().SortFlags = false
	return cmd
}

func (a *App) newExamplesCommand(withRepo repoRunner) *cobra.Command {
	opts := buildOptions{
		mode: getenvDefault("NNCTL_OPTIMIZE", getenvDefault("BUILD_MODE", defaultBuildMode)),
	}
	cmd := &cobra.Command{
		Use:   "examples",
		Short: "Build all examples",
		Long:  "Builds all example executables without running them. Use nnctl run <example> to run one example.",
		Args:  cobra.NoArgs,
		RunE: withRepo(func(ctx context.Context, cmd *cobra.Command, args []string) error {
			return a.runExamples(ctx, opts)
		}),
	}
	addModeFlags(cmd, &opts.mode)
	addGPUFlag(cmd, &opts.gpu)
	return cmd
}

func (a *App) newRunCommand(withRepo repoRunner) *cobra.Command {
	mode := ""
	gpu := ""
	cmd := &cobra.Command{
		Use:   "run <example>",
		Short: "Run an example",
		Long:  `Runs a named example. Flags configure the Zig build. Args after "--" are passed to the example executable.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			return cmd.Help()
		},
	}
	addModeFlags(cmd, &mode)
	addGPUFlag(cmd, &gpu)

	quick := &cobra.Command{
		Use:   "quick",
		Short: `Run all examples marked as quick`,
		Args:  cobra.ArbitraryArgs,
		RunE: withRepo(func(ctx context.Context, cmd *cobra.Command, args []string) error {
			return a.runExample(ctx, "quick", nil, mode, gpu)
		}),
	}
	cmd.AddCommand(quick)

	for _, example := range catalog.SortedExamples() {
		if example.Hidden {
			continue
		}
		example := example
		exampleCmd := &cobra.Command{
			Use:     example.Name + " [-- example args]",
			Aliases: exampleAliases(example),
			Short:   example.Description,
			Long:    exampleHelpLong(example),
			Args:    cobra.ArbitraryArgs,
			RunE: withRepo(func(ctx context.Context, cmd *cobra.Command, args []string) error {
				return a.runExample(ctx, example.Name, args, mode, gpu)
			}),
		}
		cmd.AddCommand(exampleCmd)
	}
	return cmd
}

func (a *App) newTrainCommand(withRepo repoRunner) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "train <target>",
		Short: "Train model checkpoints",
		RunE: func(cmd *cobra.Command, args []string) error {
			return cmd.Help()
		},
	}
	cmd.AddCommand(a.newTrainTinyGPTCommand(withRepo))
	return cmd
}

func (a *App) newTrainTinyGPTCommand(withRepo repoRunner) *cobra.Command {
	opts := defaultTinyGPTTrainOptions()
	cmd := &cobra.Command{
		Use:     "tiny-gpt",
		Aliases: []string{"tinygpt"},
		Short:   "Train a TinyGPT checkpoint",
		Long: `Trains all TinyGPT Transformer weights, embeddings, layer norms, and the output head.
Fresh checkpoints accept model-shape flags; resumed checkpoints load the shape from the checkpoint.

Corpora:
  auto         Use prepared TinyStories, then Tiny Shakespeare, then toy.
  toy          Always available from examples/tiny_gpt/data/toy.txt.
  shakespeare  Requires "nnctl data tiny-gpt".
  tinystories  Requires "nnctl data tiny-gpt".

Use --corpus-path to train on any local UTF-8 text file instead of a preset.`,
		Example: `  nnctl data tiny-gpt
  nnctl train tiny-gpt --corpus toy --output toy-gpt.bin
  nnctl train tiny-gpt --corpus shakespeare --output shakespeare-gpt.bin
  nnctl train tiny-gpt --corpus tinystories --output tinystories-gpt.bin
  nnctl train tiny-gpt --corpus-path /tmp/my-corpus.txt --output custom-gpt.bin
  nnctl train tiny-gpt --resume tinystories-gpt.bin --steps 2000
  nnctl chat --model tinystories-gpt.bin`,
		Args: cobra.NoArgs,
		RunE: withRepo(func(ctx context.Context, cmd *cobra.Command, args []string) error {
			if err := prepareTinyGPTResumeOptions(&opts, cmd.Flags().Changed); err != nil {
				return err
			}
			return a.runTrainTinyGPT(ctx, opts)
		}),
	}
	addTinyGPTTrainFlags(cmd, &opts)
	return cmd
}

func (a *App) newChatCommand(withRepo repoRunner) *cobra.Command {
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

func (a *App) newListCommand(root *cobra.Command) *cobra.Command {
	return &cobra.Command{
		Use:   "list [topic]",
		Short: "List tasks, examples, or tests",
		Args:  cobra.MaximumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			topic := "tasks"
			if len(args) == 1 {
				topic = catalog.NormalizeName(args[0])
			}
			switch topic {
			case "tasks", "commands":
				return root.Help()
			case "examples", "example":
				a.listExamples()
			case "tests", "test":
				a.listTests()
			default:
				return fmt.Errorf("unknown list topic %q", topic)
			}
			return nil
		},
	}
}

func (a *App) newFormatCommand(withRepo repoRunner) *cobra.Command {
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

func (a *App) newCleanCommand(withRepo repoRunner) *cobra.Command {
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

func (a *App) newDataCommand(withRepo repoRunner) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "data <command>",
		Short: "Download or prepare example data",
		RunE: func(cmd *cobra.Command, args []string) error {
			return cmd.Help()
		},
	}
	cmd.AddCommand(a.newDataMNISTCommand(withRepo), a.newDataTinyGPTCommand(withRepo))
	return cmd
}

func (a *App) newDataMNISTCommand(withRepo repoRunner) *cobra.Command {
	dir := "data"
	force := false
	cmd := &cobra.Command{
		Use:   "mnist",
		Short: "Download and extract the MNIST dataset",
		Long:  "Downloads and extracts the four MNIST IDX files expected by the MNIST example.",
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

func (a *App) newDataTinyGPTCommand(withRepo repoRunner) *cobra.Command {
	return &cobra.Command{
		Use:     "tiny-gpt",
		Aliases: []string{"tinygpt"},
		Short:   "Download sourced TinyGPT demo corpora",
		Long: `Downloads Tiny Shakespeare and a TinyStories validation split, then writes a small TinyStories slice used by the TinyGPT preset.

Prepared files:
  examples/tiny_gpt/data/shakespeare/input.txt
  examples/tiny_gpt/data/tinystories/valid.txt
  examples/tiny_gpt/data/tinystories/tinystories_1mb.txt

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

func (a *App) newDoctorCommand(withRepo repoRunner) *cobra.Command {
	return &cobra.Command{
		Use:     "doctor",
		Aliases: []string{"env"},
		Short:   "Check local tool versions",
		Args:    cobra.NoArgs,
		RunE: withRepo(func(ctx context.Context, cmd *cobra.Command, args []string) error {
			return a.runDoctor(ctx)
		}),
	}
}

func (a *App) newTinyGPTTopicCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "tiny-gpt",
		Short: "Show TinyGPT workflows",
		Long: `TinyGPT can be used as a quick runnable demo, a trainable checkpoint target, or a local chat model.

Common workflows:
  nnctl data tiny-gpt
  nnctl run tiny-gpt -- --corpus toy --prompt "to be"
  nnctl train tiny-gpt --corpus tinystories --output tiny-gpt.bin
  nnctl chat --model tiny-gpt.bin

Corpora:
  auto         Use prepared TinyStories, then Tiny Shakespeare, then toy.
  toy          Works offline from examples/tiny_gpt/data/toy.txt.
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

func newVersionCommand(a *App) *cobra.Command {
	return &cobra.Command{
		Use:   "version",
		Short: "Print nnctl version",
		Args:  cobra.NoArgs,
		Run: func(cmd *cobra.Command, args []string) {
			fmt.Fprintf(a.stdout(), "nnctl dev (%s)\n", runtime.Version())
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
	cmd.Flags().StringVar(gpu, "gpu", *gpu, "GPU backend: none, auto, metal, or cuda")
	cmd.Flags().SortFlags = false
}

func addTinyGPTTrainFlags(cmd *cobra.Command, opts *tinyGPTTrainOptions) {
	addModeFlags(cmd, &opts.mode)
	cmd.Flags().StringVar(&opts.output, "output", opts.output, "checkpoint path to write")
	cmd.Flags().StringVar(&opts.resume, "resume", opts.resume, "checkpoint path to resume from")
	cmd.Flags().StringVar(&opts.corpus, "corpus", opts.corpus, "auto, toy, shakespeare, or tinystories")
	cmd.Flags().StringVar(&opts.corpusPath, "corpus-path", opts.corpusPath, "custom UTF-8 text corpus path")
	cmd.Flags().IntVar(&opts.steps, "steps", opts.steps, "full-model training steps")
	cmd.Flags().StringVar(&opts.learningRate, "learning-rate", opts.learningRate, "full-model learning rate")
	cmd.Flags().IntVar(&opts.batchSize, "batch-size", opts.batchSize, "corpus windows per training step")
	cmd.Flags().IntVar(&opts.trainChars, "train-chars", opts.trainChars, "training corpus characters")
	cmd.Flags().IntVar(&opts.validationChars, "validation-chars", opts.validationChars, "validation holdout characters")
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
	cmd.Flags().SortFlags = false
}

func modeFlagChanged(cmd *cobra.Command) bool {
	return cmd.Flags().Changed("mode") || cmd.Flags().Changed("optimize")
}

func exampleAliases(example catalog.Example) []string {
	aliases := append([]string(nil), example.Aliases...)
	underscored := strings.ReplaceAll(example.Name, "-", "_")
	if underscored != example.Name {
		aliases = append(aliases, underscored)
	}
	return aliases
}

func exampleHelpLong(example catalog.Example) string {
	if catalog.NormalizeName(example.Name) != "tiny-gpt" {
		return example.Description
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
