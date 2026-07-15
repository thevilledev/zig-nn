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

func (a *app) runAll(ctx context.Context, opts buildOptions) error {
	if err := a.runZig(ctx, zig.Args("build", zig.Options{Optimize: opts.mode, GPU: opts.gpu})...); err != nil {
		return err
	}
	if err := a.runZig(ctx, zig.Args("test", zig.Options{GPU: opts.gpu})...); err != nil {
		return err
	}
	if err := a.runZig(ctx, zig.Args("test-acceptance", zig.Options{GPU: opts.gpu})...); err != nil {
		return err
	}
	return a.runZig(ctx, zig.Args("experiments", zig.Options{Optimize: opts.mode, GPU: opts.gpu})...)
}

func (a *app) runBuild(ctx context.Context, opts buildOptions) error {
	return a.runZig(ctx, zig.Args("build", zig.Options{Optimize: opts.mode, GPU: opts.gpu})...)
}

func (a *app) runTest(ctx context.Context, opts testOptions) error {
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

func (a *app) runExperiments(ctx context.Context, opts buildOptions) error {
	return a.runZig(ctx, zig.Args("experiments", zig.Options{Optimize: opts.mode, GPU: opts.gpu})...)
}

func (a *app) runExperiment(ctx context.Context, name string, passthrough []string, mode, gpu string) error {
	if catalog.NormalizeName(name) == "quick" {
		for _, experiment := range catalog.Experiments() {
			if !experiment.Quick {
				continue
			}
			opts := experimentRunOptions(experiment, mode, gpu)
			if err := a.runZig(ctx, zig.RunArgs(experiment.Step, opts, nil)...); err != nil {
				return err
			}
		}
		return nil
	}

	experiment, ok := catalog.ResolveExperiment(name)
	if !ok {
		return fmt.Errorf("unknown experiment %q; run nnctl list experiments", name)
	}
	opts := experimentRunOptions(experiment, mode, gpu)
	return a.runZig(ctx, zig.RunArgs(experiment.Step, opts, passthrough)...)
}

func (a *app) listExperiments() {
	for _, experiment := range catalog.SortedExperiments() {
		if experiment.Hidden {
			continue
		}
		fmt.Fprintf(a.stdout(), "%-24s %s\n", experiment.Name, experiment.Description)
	}
}

func (a *app) listTests() {
	for _, test := range catalog.Tests() {
		fmt.Fprintln(a.stdout(), test)
	}
}

func (a *app) runFormat(ctx context.Context, check bool) error {
	if check {
		return a.run(ctx, a.repoRoot, a.zig, "fmt", "--check", ".")
	}
	return a.run(ctx, a.repoRoot, a.zig, "fmt", ".")
}

func (a *app) runClean(includeTool bool) error {
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

func (a *app) runDoctor(ctx context.Context) error {
	fmt.Fprintf(a.stdout(), "repo: %s\n", a.repoRoot)
	fmt.Fprintf(a.stdout(), "nnctl: %s\n", runtime.Version())
	fmt.Fprintf(a.stdout(), "platform: %s/%s\n", runtime.GOOS, runtime.GOARCH)
	a.printToolVersion(ctx, "go", "version")
	a.printToolVersion(ctx, a.zig, "version")
	a.printBackendDoctor()
	return nil
}

func (a *app) printBackendDoctor() {
	cudaPath := getenvDefault("CUDA_PATH", "/usr/local/cuda")
	rocmPath := getenvDefault("ROCM_PATH", "/opt/rocm")
	fmt.Fprintf(a.stdout(), "metal: %s\n", metalDoctorStatus(runtime.GOOS))
	fmt.Fprintf(a.stdout(), "cuda: %s\n", cudaDoctorStatus(runtime.GOOS, cudaPath, pathExists))
	fmt.Fprintf(a.stdout(), "rocm: %s\n", rocmDoctorStatus(runtime.GOOS, rocmPath, pathExists))
	fmt.Fprintf(a.stdout(), "verify metal: zig build -Dgpu=metal test-metal_backend --summary all\n")
	fmt.Fprintf(a.stdout(), "verify cuda: zig build -Dgpu=cuda -Dcuda-path=%s test-cuda_backend --summary all\n", cudaPath)
	fmt.Fprintf(a.stdout(), "verify rocm: zig build -Dgpu=rocm -Drocm-path=%s test-rocm_backend --summary all\n", rocmPath)
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

func rocmDoctorStatus(goos, rocmPath string, exists func(string) bool) string {
	if goos != "linux" {
		return "build option unavailable on this OS; ROCm backend currently targets Linux"
	}
	if exists(rocmPath) {
		return fmt.Sprintf("toolkit candidate found at %s", rocmPath)
	}
	return fmt.Sprintf("toolkit not found at %s; pass -Drocm-path or set ROCM_PATH", rocmPath)
}

func pathExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

func (a *app) printToolVersion(ctx context.Context, name string, args ...string) {
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

func (a *app) runZig(ctx context.Context, args ...string) error {
	return a.run(ctx, a.repoRoot, a.zig, args...)
}

func (a *app) run(ctx context.Context, dir, name string, args ...string) error {
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

func experimentRunOptions(experiment catalog.Experiment, mode, gpu string) zig.Options {
	if mode == "" {
		mode = experiment.DefaultOptimize
	}
	if mode == "" {
		mode = getenvDefault("NNCTL_OPTIMIZE", getenvDefault("BUILD_MODE", defaultBuildMode))
	}
	if gpu == "" {
		gpu = experiment.DefaultGPU
	}
	return zig.Options{Optimize: mode, GPU: gpu}
}
