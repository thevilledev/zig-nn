package cli

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"

	"nnctl/internal/catalog"
	"nnctl/internal/flagx"
	"nnctl/internal/zig"
)

func (a *App) cmdAll(ctx context.Context, args []string) error {
	fs := newFlagSet(a.stderr(), "all")
	mode := getenvDefault("NNCTL_OPTIMIZE", getenvDefault("BUILD_MODE", defaultBuildMode))
	gpu := ""
	fs.StringVar(&mode, "mode", mode, "Zig optimization mode")
	fs.StringVar(&mode, "optimize", mode, "Zig optimization mode")
	fs.StringVar(&gpu, "gpu", gpu, "GPU backend: none, auto, metal, or cuda")
	if err := flagx.ParseInterspersed(fs, args, map[string]bool{
		"mode":     true,
		"optimize": true,
		"gpu":      true,
	}); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("all does not accept positional arguments")
	}

	if err := a.runZig(ctx, zig.Args("build", zig.Options{Optimize: mode, GPU: gpu})...); err != nil {
		return err
	}
	if err := a.runZig(ctx, zig.Args("test", zig.Options{GPU: gpu})...); err != nil {
		return err
	}
	if err := a.runZig(ctx, zig.Args("test-acceptance", zig.Options{GPU: gpu})...); err != nil {
		return err
	}
	return a.runZig(ctx, zig.Args("examples", zig.Options{Optimize: mode, GPU: gpu})...)
}

func (a *App) cmdBuild(ctx context.Context, args []string, release bool) error {
	fs := newFlagSet(a.stderr(), "build")
	mode := getenvDefault("NNCTL_OPTIMIZE", getenvDefault("BUILD_MODE", defaultBuildMode))
	if release {
		mode = getenvDefault("NNCTL_RELEASE_OPTIMIZE", getenvDefault("RELEASE_MODE", defaultReleaseMode))
	}
	gpu := ""
	releaseFlag := release
	fs.StringVar(&mode, "mode", mode, "Zig optimization mode")
	fs.StringVar(&mode, "optimize", mode, "Zig optimization mode")
	fs.StringVar(&gpu, "gpu", gpu, "GPU backend: none, auto, metal, or cuda")
	fs.BoolVar(&releaseFlag, "release", releaseFlag, "build with the release optimization default")
	modeProvided := flagx.HasAny(args, "mode", "optimize")
	if err := flagx.ParseInterspersed(fs, args, map[string]bool{
		"mode":     true,
		"optimize": true,
		"gpu":      true,
		"release":  false,
	}); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("build does not accept positional arguments")
	}
	if releaseFlag && !release && !modeProvided {
		mode = getenvDefault("NNCTL_RELEASE_OPTIMIZE", getenvDefault("RELEASE_MODE", defaultReleaseMode))
	}
	return a.runZig(ctx, zig.Args("build", zig.Options{Optimize: mode, GPU: gpu})...)
}

func (a *App) cmdTest(ctx context.Context, args []string) error {
	fs := newFlagSet(a.stderr(), "test")
	mode := ""
	gpu := ""
	summary := ""
	unitOnly := false
	acceptanceOnly := false
	name := ""
	fs.StringVar(&mode, "mode", mode, "Zig optimization mode")
	fs.StringVar(&mode, "optimize", mode, "Zig optimization mode")
	fs.StringVar(&gpu, "gpu", gpu, "GPU backend: none, auto, metal, or cuda")
	fs.StringVar(&summary, "summary", summary, "Zig build summary mode, for example all")
	fs.BoolVar(&unitOnly, "unit-only", false, "run only unit tests")
	fs.BoolVar(&acceptanceOnly, "acceptance-only", false, "run only acceptance tests")
	fs.StringVar(&name, "name", "", "individual test name")
	if err := flagx.ParseInterspersed(fs, args, map[string]bool{
		"mode":            true,
		"optimize":        true,
		"gpu":             true,
		"summary":         true,
		"name":            true,
		"unit-only":       false,
		"acceptance-only": false,
	}); err != nil {
		return err
	}
	if fs.NArg() > 1 {
		return fmt.Errorf("test accepts at most one test name")
	}
	if fs.NArg() == 1 {
		name = fs.Arg(0)
	}
	if unitOnly && acceptanceOnly {
		return fmt.Errorf("--unit-only and --acceptance-only cannot be used together")
	}
	opts := zig.Options{Optimize: mode, GPU: gpu, Summary: summary}
	if name != "" {
		step := "test-" + catalog.ZigIdentifier(name)
		return a.runZig(ctx, zig.Args(step, opts)...)
	}
	if acceptanceOnly {
		return a.runZig(ctx, zig.Args("test-acceptance", opts)...)
	}
	if err := a.runZig(ctx, zig.Args("test", opts)...); err != nil {
		return err
	}
	if unitOnly {
		return nil
	}
	return a.runZig(ctx, zig.Args("test-acceptance", opts)...)
}

func (a *App) cmdExamples(ctx context.Context, args []string) error {
	fs := newFlagSet(a.stderr(), "examples")
	mode := getenvDefault("NNCTL_OPTIMIZE", getenvDefault("BUILD_MODE", defaultBuildMode))
	gpu := ""
	fs.StringVar(&mode, "mode", mode, "Zig optimization mode")
	fs.StringVar(&mode, "optimize", mode, "Zig optimization mode")
	fs.StringVar(&gpu, "gpu", gpu, "GPU backend: none, auto, metal, or cuda")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("examples does not accept positional arguments; use nnctl run <example>")
	}
	return a.runZig(ctx, zig.Args("examples", zig.Options{Optimize: mode, GPU: gpu})...)
}

func (a *App) cmdRun(ctx context.Context, args []string) error {
	beforePassthrough, passthrough := flagx.SplitPassthrough(args)

	fs := newFlagSet(a.stderr(), "run")
	mode := ""
	gpu := ""
	fs.StringVar(&mode, "mode", mode, "Zig optimization mode")
	fs.StringVar(&mode, "optimize", mode, "Zig optimization mode")
	fs.StringVar(&gpu, "gpu", gpu, "GPU backend: none, auto, metal, or cuda")
	if err := flagx.ParseInterspersed(fs, beforePassthrough, map[string]bool{
		"mode":     true,
		"optimize": true,
		"gpu":      true,
	}); err != nil {
		return err
	}
	if fs.NArg() != 1 {
		return fmt.Errorf("usage: nnctl run [flags] <example> [-- example args]")
	}

	name := fs.Arg(0)
	if catalog.NormalizeName(name) == "quick" {
		for _, example := range catalog.Examples() {
			if !example.Quick {
				continue
			}
			opts := exampleRunOptions(example, mode, gpu)
			if err := a.runZig(ctx, zig.RunArgs(example.Step, opts, nil)...); err != nil {
				return err
			}
		}
		return nil
	}

	example, ok := catalog.ResolveExample(name)
	if !ok {
		return fmt.Errorf("unknown example %q; run nnctl list examples", name)
	}
	opts := exampleRunOptions(example, mode, gpu)
	return a.runZig(ctx, zig.RunArgs(example.Step, opts, passthrough)...)
}

func (a *App) cmdList(args []string) error {
	topic := "tasks"
	if len(args) > 1 {
		return fmt.Errorf("list accepts at most one topic")
	}
	if len(args) == 1 {
		topic = catalog.NormalizeName(args[0])
	}

	switch topic {
	case "tasks", "commands":
		printUsage(a.stdout())
	case "examples", "example":
		for _, example := range catalog.SortedExamples() {
			if example.Hidden {
				continue
			}
			fmt.Fprintf(a.stdout(), "%-24s %s\n", example.Name, example.Description)
		}
	case "tests", "test":
		for _, test := range catalog.Tests() {
			fmt.Fprintln(a.stdout(), test)
		}
	default:
		return fmt.Errorf("unknown list topic %q", topic)
	}
	return nil
}

func (a *App) cmdFormat(ctx context.Context, args []string) error {
	fs := newFlagSet(a.stderr(), "fmt")
	check := false
	fs.BoolVar(&check, "check", false, "check formatting without rewriting files")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("fmt does not accept positional arguments")
	}
	if check {
		return a.run(ctx, a.repoRoot, a.zig, "fmt", "--check", ".")
	}
	return a.run(ctx, a.repoRoot, a.zig, "fmt", ".")
}

func (a *App) cmdClean(args []string) error {
	fs := newFlagSet(a.stderr(), "clean")
	includeTool := false
	fs.BoolVar(&includeTool, "tool", false, "also remove bin/nnctl")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("clean does not accept positional arguments")
	}
	paths := []string{"zig-out", ".zig-cache"}
	if includeTool {
		paths = append(paths, filepath.Join("bin", "nnctl"))
	}
	for _, rel := range paths {
		path := filepath.Join(a.repoRoot, rel)
		if err := os.RemoveAll(path); err != nil {
			return fmt.Errorf("remove %s: %w", rel, err)
		}
		fmt.Fprintf(a.stdout(), "removed %s\n", rel)
	}
	return nil
}

func (a *App) cmdDoctor(ctx context.Context, args []string) error {
	fs := newFlagSet(a.stderr(), "doctor")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("doctor does not accept positional arguments")
	}
	fmt.Fprintf(a.stdout(), "repo: %s\n", a.repoRoot)
	fmt.Fprintf(a.stdout(), "nnctl: %s\n", runtime.Version())
	a.printToolVersion(ctx, "go", "version")
	a.printToolVersion(ctx, a.zig, "version")
	return nil
}

func (a *App) printToolVersion(ctx context.Context, name string, args ...string) {
	path, err := exec.LookPath(name)
	if err != nil {
		fmt.Fprintf(a.stdout(), "%s: missing\n", name)
		return
	}
	cmd := exec.CommandContext(ctx, path, args...)
	cmd.Dir = a.repoRoot
	out, err := cmd.CombinedOutput()
	if err != nil {
		fmt.Fprintf(a.stdout(), "%s: %s (%v)\n", name, strings.TrimSpace(string(out)), err)
		return
	}
	fmt.Fprintf(a.stdout(), "%s: %s\n", name, strings.TrimSpace(string(out)))
}

func (a *App) runZig(ctx context.Context, args ...string) error {
	return a.run(ctx, a.repoRoot, a.zig, args...)
}

func (a *App) run(ctx context.Context, dir, name string, args ...string) error {
	fmt.Fprintf(a.stderr(), "==> %s\n", zig.CommandString(name, args))
	cmd := exec.CommandContext(ctx, name, args...)
	cmd.Dir = dir
	cmd.Stdin = a.stdin()
	cmd.Stdout = a.stdout()
	cmd.Stderr = a.stderr()
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("%s failed: %w", name, err)
	}
	return nil
}

func exampleRunOptions(example catalog.Example, mode, gpu string) zig.Options {
	if mode == "" {
		mode = example.DefaultOptimize
	}
	if mode == "" {
		mode = getenvDefault("NNCTL_OPTIMIZE", getenvDefault("BUILD_MODE", defaultBuildMode))
	}
	if gpu == "" {
		gpu = example.DefaultGPU
	}
	return zig.Options{Optimize: mode, GPU: gpu}
}
