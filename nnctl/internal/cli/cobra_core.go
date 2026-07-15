package cli

import (
	"context"
	"fmt"
	"runtime"

	"github.com/spf13/cobra"

	"nnctl/internal/repo"
)

func (a *app) newRootCommand() *cobra.Command {
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
		Long:          "nnctl builds, tests, runs experiments, prepares data, and trains or serves the TinyGPT demo in this repository.",
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
		a.newCloudCommand(withRepo),
		a.newExperimentsCommand(withRepo),
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

func (a *app) newAllCommand(withRepo repoRunner) *cobra.Command {
	opts := buildOptions{
		mode: getenvDefault("NNCTL_OPTIMIZE", getenvDefault("BUILD_MODE", defaultBuildMode)),
	}
	cmd := &cobra.Command{
		Use:   "all",
		Short: "Build, test, and build experiments",
		Long:  "Builds the library, runs unit tests, runs acceptance tests, and builds all experiments.",
		Args:  cobra.NoArgs,
		RunE: withRepo(func(ctx context.Context, cmd *cobra.Command, args []string) error {
			return a.runAll(ctx, opts)
		}),
	}
	addModeFlags(cmd, &opts.mode)
	addGPUFlag(cmd, &opts.gpu)
	return cmd
}

func (a *app) newBuildCommand(withRepo repoRunner) *cobra.Command {
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

func (a *app) newReleaseCommand(withRepo repoRunner) *cobra.Command {
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

func (a *app) newTestCommand(withRepo repoRunner) *cobra.Command {
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

func (a *app) newBenchmarkCommand(withRepo repoRunner) *cobra.Command {
	opts := defaultBenchmarkOptions()
	cmd := &cobra.Command{
		Use:   "benchmark",
		Short: "Run benchmarks with readable CPU/GPU comparisons",
		Long: `Runs the Zig benchmark suite and renders the CSV as grouped tables.
The default uses ReleaseFast and GPU auto-detection so CPU, Metal, CUDA, and ROCm rows
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

func (a *app) newDeployCommand(withRepo repoRunner) *cobra.Command {
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
