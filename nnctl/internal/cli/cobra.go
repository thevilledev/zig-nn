package cli

import (
	"context"
	"fmt"
	"runtime"
	"strings"

	"github.com/spf13/cobra"

	"nnctl/internal/catalog"
	verdacloud "nnctl/internal/cloud/verda"
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
		a.newBenchmarkCommand(withRepo),
		a.newDeployCommand(withRepo),
		a.newCloudCommand(),
		a.newExamplesCommand(withRepo),
		a.newRunCommand(withRepo),
		a.newTrainCommand(withRepo),
		a.newChatCommand(withRepo),
		a.newListCommand(root),
		a.newFormatCommand(withRepo),
		a.newCleanCommand(withRepo),
		a.newDataCommand(withRepo),
		a.newDoctorCommand(withRepo),
		a.newCompletionCommand(),
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

func (a *App) newBenchmarkCommand(withRepo repoRunner) *cobra.Command {
	opts := defaultBenchmarkOptions()
	cmd := &cobra.Command{
		Use:   "benchmark",
		Short: "Run benchmarks with readable CPU/GPU comparisons",
		Long: `Runs the Zig benchmark suite and renders the CSV as grouped tables.
The default uses ReleaseFast and GPU auto-detection so CPU, Metal, and CUDA rows
can be compared where those backends are available.
Use --csv to print the raw benchmark CSV instead. Use --compare with a previous
raw CSV file to print current-vs-baseline deltas.`,
		Example: `  nnctl benchmark
  nnctl benchmark --quick
  nnctl benchmark --filter matmul
  nnctl benchmark --gpu none --csv
  nnctl benchmark --quick --compare baseline.csv
  nnctl benchmark --debug --filter activation`,
		Args: cobra.NoArgs,
		RunE: withRepo(func(ctx context.Context, cmd *cobra.Command, args []string) error {
			return a.runBenchmark(ctx, opts)
		}),
	}
	addGPUFlag(cmd, &opts.gpu)
	cmd.Flags().StringVar(&opts.filter, "filter", opts.filter, "benchmark suite filter")
	cmd.Flags().BoolVar(&opts.quick, "quick", opts.quick, "run the smoke-sized benchmark subset")
	cmd.Flags().BoolVar(&opts.debug, "debug", opts.debug, "run benchmark-debug instead of ReleaseFast benchmark")
	cmd.Flags().BoolVar(&opts.csv, "csv", opts.csv, "print raw CSV output")
	cmd.Flags().StringVar(&opts.compare, "compare", opts.compare, "compare current results against a raw benchmark CSV file")
	cmd.Flags().SortFlags = false
	return cmd
}

func (a *App) newDeployCommand(withRepo repoRunner) *cobra.Command {
	opts := defaultDeployOptions()
	cmd := &cobra.Command{
		Use:   "deploy TARGET",
		Short: "Deploy a git snapshot with rsync",
		Long: `Creates a git archive for the selected ref, extracts it into a temporary
snapshot directory, and rsyncs that clean tree to TARGET. Build artifacts,
dependency folders, and .git metadata are not copied.

By default nnctl refuses to deploy when the working tree has uncommitted
changes, so remote benchmark runs can be tied back to a specific git ref.`,
		Example: `  nnctl deploy gpu1:~/zig-nn
  nnctl deploy user@gpu1:/srv/zig-nn --delete
  nnctl deploy gpu1:~/zig-nn --ref feature/fast-matmul
  nnctl deploy gpu1:~/zig-nn --dry-run`,
		Args: cobra.ExactArgs(1),
		RunE: withRepo(func(ctx context.Context, cmd *cobra.Command, args []string) error {
			opts.target = args[0]
			return a.runDeploy(ctx, opts)
		}),
	}
	cmd.Flags().StringVar(&opts.ref, "ref", opts.ref, "git ref or tree-ish to archive")
	cmd.Flags().BoolVar(&opts.delete, "delete", opts.delete, "delete target files missing from the snapshot")
	cmd.Flags().BoolVar(&opts.dryRun, "dry-run", opts.dryRun, "ask rsync to show changes without updating the target")
	cmd.Flags().BoolVar(&opts.ignoreDirty, "ignore-dirty", opts.ignoreDirty, "deploy --ref even when the working tree has uncommitted changes")
	cmd.Flags().StringVar(&opts.ssh, "ssh", opts.ssh, "ssh command passed to rsync -e")
	cmd.Flags().StringVar(&opts.git, "git", opts.git, "Git executable")
	cmd.Flags().StringVar(&opts.tar, "tar", opts.tar, "tar executable")
	cmd.Flags().StringVar(&opts.rsync, "rsync", opts.rsync, "rsync executable")
	cmd.Flags().SortFlags = false
	return cmd
}

func (a *App) newCloudCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "cloud <command>",
		Short: "Deploy cloud GPU benchmark workers",
		Long:  "Deploys cloud GPU benchmark workers for nnctl agents. Verda deployments are restricted to spot, single-GPU instances.",
		RunE: func(cmd *cobra.Command, args []string) error {
			return cmd.Help()
		},
	}
	cmd.AddCommand(
		a.newCloudDeployCommand(),
		a.newCloudDestroyCommand(),
		a.newCloudPackerTemplateCommand(),
		a.newCloudListCommand(),
		a.newCloudPricingCommand(),
		a.newCloudSSHKeysCommand(),
	)
	return cmd
}

func (a *App) newCloudDeployCommand() *cobra.Command {
	opts := cloudDeployOptions{
		DeployOptions: verdacloud.DefaultDeployOptions(""),
	}
	cmd := &cobra.Command{
		Use:   "deploy",
		Short: "Deploy a Verda spot GPU worker",
		Long: `Deploys one Verda GPU instance for nnctl benchmark work.
The deployment policy is intentionally narrow: spot only and single-GPU instance
types only. By default nnctl chooses the cheapest currently available spot
location for the requested instance type. Userdata defaults to the embedded
script from nnctl/internal/cloud/verda/bootstrap.sh.`,
		Example: `  nnctl cloud deploy --instance-type 1V100.6V --source-os-volume-id volume_id --ssh-key-id ssh_key_id
  nnctl cloud deploy --instance-type 1V100.6V --source-os-volume-id volume_id --dry-run --json
  nnctl cloud deploy --instance-type 1V100.6V --source-os-volume-id volume_id --location-code FIN-03
  nnctl cloud deploy --instance-type 1V100.6V --source-os-volume-id volume_id --user-data-file ./custom-bootstrap.sh`,
		Args: cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			return a.runCloudDeploy(cmd.Context(), opts)
		},
	}
	cmd.Flags().StringVar(&opts.InstanceType, "instance-type", opts.InstanceType, "Verda instance type, for example 1L40S.20V")
	cmd.Flags().StringVar(&opts.SourceOSVolumeID, "source-os-volume-id", opts.SourceOSVolumeID, "Packer-built Verda OS volume ID to clone for this instance")
	cmd.Flags().StringVar(&opts.Hostname, "hostname", opts.Hostname, "instance hostname")
	cmd.Flags().StringVar(&opts.Description, "description", opts.Description, "instance description")
	cmd.Flags().StringArrayVar(&opts.SSHKeyIDs, "ssh-key-id", opts.SSHKeyIDs, "Verda SSH key ID to attach (repeatable)")
	cmd.Flags().StringVar(&opts.LocationCode, "location-code", opts.LocationCode, "Verda location code; defaults to cheapest currently available spot location")
	cmd.Flags().StringVar(&opts.StartupScriptName, "startup-script-name", opts.StartupScriptName, "Verda startup script name")
	cmd.Flags().StringVar(&opts.userDataFile, "user-data-file", opts.userDataFile, "read userdata from a file instead of the embedded script")
	cmd.Flags().StringVar(&opts.BaseURL, "base-url", opts.BaseURL, "Verda API base URL")
	cmd.Flags().BoolVar(&opts.SkipAvailabilityCheck, "skip-availability-check", opts.SkipAvailabilityCheck, "skip spot availability check before creating the instance")
	cmd.Flags().BoolVar(&opts.DryRun, "dry-run", opts.DryRun, "print the planned spot deployment without reading keychain credentials")
	cmd.Flags().BoolVar(&opts.jsonOutput, "json", opts.jsonOutput, "print machine-readable JSON")
	cmd.Flags().SortFlags = false
	_ = cmd.MarkFlagRequired("instance-type")
	_ = cmd.MarkFlagRequired("source-os-volume-id")
	return cmd
}

func (a *App) newCloudDestroyCommand() *cobra.Command {
	opts := cloudDestroyOptions{
		DestroyOptions: verdacloud.DefaultDestroyOptions(nil),
	}
	cmd := &cobra.Command{
		Use:     "destroy INSTANCE_ID [INSTANCE_ID...]",
		Aliases: []string{"delete", "rm", "terminate"},
		Short:   "Destroy Verda instances",
		Long:    "Destroys one or more Verda instances by ID. Instance IDs can be copied from `nnctl cloud list`.",
		Example: `  nnctl cloud destroy instance_id
  nnctl cloud destroy instance_id --dry-run
  nnctl cloud destroy instance_id --permanent=false
  nnctl cloud destroy instance_id --volume-id cloned_os_volume_id --source-os-volume-id source_os_volume_id --json`,
		Args: cobra.MinimumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			opts.InstanceIDs = args
			return a.runCloudDestroy(cmd.Context(), opts)
		},
	}
	cmd.Flags().StringArrayVar(&opts.VolumeIDs, "volume-id", opts.VolumeIDs, "Verda volume ID to delete with the instance (repeatable)")
	cmd.Flags().StringVar(&opts.SourceOSVolumeID, "source-os-volume-id", opts.SourceOSVolumeID, "protect this golden Packer source OS volume ID from cleanup")
	cmd.Flags().BoolVar(&opts.DeletePermanently, "permanent", opts.DeletePermanently, "delete permanently instead of using Verda's non-permanent delete")
	cmd.Flags().StringVar(&opts.BaseURL, "base-url", opts.BaseURL, "Verda API base URL")
	cmd.Flags().BoolVar(&opts.DryRun, "dry-run", opts.DryRun, "print the planned destroy request without reading keychain credentials")
	cmd.Flags().BoolVar(&opts.jsonOutput, "json", opts.jsonOutput, "print machine-readable JSON")
	cmd.Flags().SortFlags = false
	return cmd
}

func (a *App) newCloudPackerTemplateCommand() *cobra.Command {
	opts := cloudPackerTemplateOptions{}
	cmd := &cobra.Command{
		Use:   "packer-template DIR",
		Short: "Write Verda Packer template files",
		Long:  "Writes the embedded Verda Packer template and bootstrap script used to build the golden OS volume for nnctl cloud deploy.",
		Example: `  nnctl cloud packer-template ./packer/verda
  cd ./packer/verda && packer init . && packer build .`,
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			return a.runCloudPackerTemplate(args[0], opts)
		},
	}
	cmd.Flags().BoolVar(&opts.force, "force", opts.force, "overwrite existing template files")
	cmd.Flags().SortFlags = false
	return cmd
}

func (a *App) newCloudSSHKeysCommand() *cobra.Command {
	opts := cloudSSHKeysOptions{}
	cmd := &cobra.Command{
		Use:     "ssh-keys",
		Aliases: []string{"sshkeys", "ssh-key"},
		Short:   "List Verda SSH key IDs",
		Long:    "Lists Verda SSH keys and their UUIDs. Pass one of these IDs to cloud deploy with --ssh-key-id.",
		Example: `  nnctl cloud ssh-keys
  nnctl cloud ssh-keys --json
  nnctl cloud deploy --instance-type 1L40S.20V --ssh-key-id 00000000-0000-0000-0000-000000000000`,
		Args: cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			return a.runCloudSSHKeys(cmd.Context(), opts)
		},
	}
	cmd.Flags().StringVar(&opts.baseURL, "base-url", opts.baseURL, "Verda API base URL")
	cmd.Flags().BoolVar(&opts.jsonOutput, "json", opts.jsonOutput, "print machine-readable JSON")
	cmd.Flags().SortFlags = false
	return cmd
}

func (a *App) newCloudListCommand() *cobra.Command {
	opts := cloudListOptions{}
	cmd := &cobra.Command{
		Use:   "list",
		Short: "List active Verda instances",
		Long:  "Lists active Verda instances by default. Use --all to include inactive instances or --status to query one Verda status.",
		Example: `  nnctl cloud list
  nnctl cloud list --all
  nnctl cloud list --status running
  nnctl cloud list --json`,
		Args: cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			return a.runCloudList(cmd.Context(), opts)
		},
	}
	cmd.Flags().StringVar(&opts.status, "status", opts.status, "Verda instance status filter, for example running")
	cmd.Flags().BoolVar(&opts.all, "all", opts.all, "include inactive instances")
	cmd.Flags().StringVar(&opts.baseURL, "base-url", opts.baseURL, "Verda API base URL")
	cmd.Flags().BoolVar(&opts.jsonOutput, "json", opts.jsonOutput, "print machine-readable JSON")
	cmd.Flags().SortFlags = false
	return cmd
}

func (a *App) newCloudPricingCommand() *cobra.Command {
	opts := cloudPricingOptions{
		sortBy: "price",
	}
	opts.filters.AvailableOnly = true
	cmd := &cobra.Command{
		Use:   "pricing",
		Short: "List Verda spot prices",
		Long:  "Lists Verda spot prices with filters for location, instance type, GPU model, manufacturer, and GPU count.",
		Example: `  nnctl cloud pricing
  nnctl cloud pricing --single-gpu
  nnctl cloud pricing --zone FIN-02 --model L40
  nnctl cloud pricing --instance-type 1L40S.20V --json
  nnctl cloud pricing --available-only=false --sort location`,
		Args: cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			opts.filters.LocationCodes = append(opts.filters.LocationCodes, opts.zones...)
			opts.filters.LocationCodes = sortUniqueStrings(opts.filters.LocationCodes)
			opts.filters.InstanceTypes = sortUniqueStrings(opts.filters.InstanceTypes)
			return a.runCloudPricing(cmd.Context(), opts)
		},
	}
	cmd.Flags().StringArrayVar(&opts.filters.LocationCodes, "location-code", opts.filters.LocationCodes, "Verda location code filter (repeatable)")
	cmd.Flags().StringArrayVar(&opts.zones, "zone", opts.zones, "alias for --location-code")
	cmd.Flags().StringArrayVar(&opts.filters.InstanceTypes, "instance-type", opts.filters.InstanceTypes, "Verda instance type filter (repeatable)")
	cmd.Flags().StringVar(&opts.filters.Model, "model", opts.filters.Model, "GPU model/display substring filter")
	cmd.Flags().StringVar(&opts.filters.Manufacturer, "manufacturer", opts.filters.Manufacturer, "GPU manufacturer substring filter")
	cmd.Flags().BoolVar(&opts.filters.SingleGPU, "single-gpu", opts.filters.SingleGPU, "only show single-GPU instance types")
	cmd.Flags().BoolVar(&opts.filters.AvailableOnly, "available-only", opts.filters.AvailableOnly, "only show currently available spot instance types")
	cmd.Flags().StringVar(&opts.filters.Currency, "currency", opts.filters.Currency, "Verda pricing currency")
	cmd.Flags().StringVar(&opts.sortBy, "sort", opts.sortBy, "sort by price, location, or instance-type")
	cmd.Flags().StringVar(&opts.baseURL, "base-url", opts.baseURL, "Verda API base URL")
	cmd.Flags().BoolVar(&opts.jsonOutput, "json", opts.jsonOutput, "print machine-readable JSON")
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
Runs include an explicit train/eval split, validation loss, scheduled learning rates, checkpoint metadata, and optional JSON summaries.

Corpora:
  auto         Use prepared TinyStories, then Tiny Shakespeare, then toy.
  toy          Always available from examples/tiny_gpt/data/toy.txt.
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
		Short:   "Check local tools and backend support",
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
  nnctl train tiny-gpt --preset coherent-small --output tiny-gpt.bin
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
