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
	"nnctl/internal/zig"
)

type buildOptions struct {
	mode string
	gpu  string
}

type testOptions struct {
	mode           string
	gpu            string
	summary        string
	unitOnly       bool
	acceptanceOnly bool
	name           string
}

func (a *App) runAll(ctx context.Context, opts buildOptions) error {
	if err := a.runZig(ctx, zig.Args("build", zig.Options{Optimize: opts.mode, GPU: opts.gpu})...); err != nil {
		return err
	}
	if err := a.runZig(ctx, zig.Args("test", zig.Options{GPU: opts.gpu})...); err != nil {
		return err
	}
	if err := a.runZig(ctx, zig.Args("test-acceptance", zig.Options{GPU: opts.gpu})...); err != nil {
		return err
	}
	return a.runZig(ctx, zig.Args("examples", zig.Options{Optimize: opts.mode, GPU: opts.gpu})...)
}

func (a *App) runBuild(ctx context.Context, opts buildOptions) error {
	return a.runZig(ctx, zig.Args("build", zig.Options{Optimize: opts.mode, GPU: opts.gpu})...)
}

func (a *App) runTest(ctx context.Context, opts testOptions) error {
	if opts.unitOnly && opts.acceptanceOnly {
		return fmt.Errorf("--unit-only and --acceptance-only cannot be used together")
	}

	zigOpts := zig.Options{Optimize: opts.mode, GPU: opts.gpu, Summary: opts.summary}
	if opts.name != "" {
		step := "test-" + catalog.ZigIdentifier(opts.name)
		return a.runZig(ctx, zig.Args(step, zigOpts)...)
	}
	if opts.acceptanceOnly {
		return a.runZig(ctx, zig.Args("test-acceptance", zigOpts)...)
	}
	if err := a.runZig(ctx, zig.Args("test", zigOpts)...); err != nil {
		return err
	}
	if opts.unitOnly {
		return nil
	}
	return a.runZig(ctx, zig.Args("test-acceptance", zigOpts)...)
}

func (a *App) runExamples(ctx context.Context, opts buildOptions) error {
	return a.runZig(ctx, zig.Args("examples", zig.Options{Optimize: opts.mode, GPU: opts.gpu})...)
}

func (a *App) runExample(ctx context.Context, name string, passthrough []string, mode, gpu string) error {
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

func (a *App) listExamples() {
	for _, example := range catalog.SortedExamples() {
		if example.Hidden {
			continue
		}
		fmt.Fprintf(a.stdout(), "%-24s %s\n", example.Name, example.Description)
	}
}

func (a *App) listTests() {
	for _, test := range catalog.Tests() {
		fmt.Fprintln(a.stdout(), test)
	}
}

func (a *App) runFormat(ctx context.Context, check bool) error {
	if check {
		return a.run(ctx, a.repoRoot, a.zig, "fmt", "--check", ".")
	}
	return a.run(ctx, a.repoRoot, a.zig, "fmt", ".")
}

func (a *App) runClean(includeTool bool) error {
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

func (a *App) runDoctor(ctx context.Context) error {
	fmt.Fprintf(a.stdout(), "repo: %s\n", a.repoRoot)
	fmt.Fprintf(a.stdout(), "nnctl: %s\n", runtime.Version())
	fmt.Fprintf(a.stdout(), "platform: %s/%s\n", runtime.GOOS, runtime.GOARCH)
	a.printToolVersion(ctx, "go", "version")
	a.printToolVersion(ctx, a.zig, "version")
	a.printBackendDoctor()
	return nil
}

func (a *App) printBackendDoctor() {
	cudaPath := getenvDefault("CUDA_PATH", "/usr/local/cuda")
	fmt.Fprintf(a.stdout(), "metal: %s\n", metalDoctorStatus(runtime.GOOS))
	fmt.Fprintf(a.stdout(), "cuda: %s\n", cudaDoctorStatus(runtime.GOOS, cudaPath, pathExists))
	fmt.Fprintf(a.stdout(), "verify metal: zig build -Dgpu=metal test-metal_backend --summary all\n")
	fmt.Fprintf(a.stdout(), "verify cuda: zig build -Dgpu=cuda -Dcuda-path=%s test-cuda_backend --summary all\n", cudaPath)
	fmt.Fprintf(a.stdout(), "benchmark gpu: nnctl benchmark --gpu auto --quick\n")
}

func metalDoctorStatus(goos string) string {
	if goos == "darwin" {
		return "available through -Dgpu=metal"
	}
	return "unavailable on this OS; Metal requires macOS"
}

func cudaDoctorStatus(goos, cudaPath string, exists func(string) bool) string {
	if goos != "linux" {
		return "build option unavailable on this OS; CUDA backend currently targets Linux"
	}
	if exists(cudaPath) {
		return fmt.Sprintf("toolkit candidate found at %s", cudaPath)
	}
	return fmt.Sprintf("toolkit not found at %s; pass -Dcuda-path or set CUDA_PATH", cudaPath)
}

func pathExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
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
