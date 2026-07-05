package nnctl

import (
	"compress/gzip"
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
)

const (
	defaultBuildMode   = "Debug"
	defaultReleaseMode = "ReleaseSafe"
)

type App struct {
	Stdin  io.Reader
	Stdout io.Writer
	Stderr io.Writer

	repoRoot string
	zig      string
}

func Main(args []string) int {
	app := &App{
		Stdin:  os.Stdin,
		Stdout: os.Stdout,
		Stderr: os.Stderr,
	}
	if err := app.Run(context.Background(), args); err != nil {
		fmt.Fprintln(app.stderr(), "error:", err)
		return 1
	}
	return 0
}

func (a *App) Run(ctx context.Context, args []string) error {
	global := flag.NewFlagSet("nnctl", flag.ContinueOnError)
	global.SetOutput(a.stderr())

	repo := ""
	zig := getenvDefault("ZIG", "zig")
	global.StringVar(&repo, "repo", "", "repository root to operate on")
	global.StringVar(&repo, "C", "", "repository root to operate on")
	global.StringVar(&zig, "zig", zig, "Zig executable to use")

	if err := global.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			printUsage(a.stdout())
			return nil
		}
		return err
	}

	rest := global.Args()
	if len(rest) == 0 {
		printUsage(a.stdout())
		return nil
	}

	command := rest[0]
	commandArgs := rest[1:]
	if command == "help" || command == "-h" || command == "--help" {
		printHelp(a.stdout(), commandArgs)
		return nil
	}
	if command == "version" {
		fmt.Fprintf(a.stdout(), "nnctl dev (%s)\n", runtime.Version())
		return nil
	}

	root, err := resolveRepoRoot(repo)
	if err != nil {
		return err
	}
	a.repoRoot = root
	a.zig = zig

	switch command {
	case "all":
		return a.cmdAll(ctx, commandArgs)
	case "build":
		return a.cmdBuild(ctx, commandArgs, false)
	case "release":
		return a.cmdBuild(ctx, commandArgs, true)
	case "test":
		return a.cmdTest(ctx, commandArgs)
	case "examples":
		return a.cmdExamples(ctx, commandArgs)
	case "run":
		return a.cmdRun(ctx, commandArgs)
	case "list":
		return a.cmdList(commandArgs)
	case "fmt", "format":
		return a.cmdFormat(ctx, commandArgs)
	case "clean":
		return a.cmdClean(commandArgs)
	case "data":
		return a.cmdData(ctx, commandArgs)
	case "doctor", "env":
		return a.cmdDoctor(ctx, commandArgs)
	default:
		return fmt.Errorf("unknown command %q; run nnctl help", command)
	}
}

func (a *App) stdout() io.Writer {
	if a.Stdout == nil {
		return io.Discard
	}
	return a.Stdout
}

func (a *App) stderr() io.Writer {
	if a.Stderr == nil {
		return io.Discard
	}
	return a.Stderr
}

func (a *App) stdin() io.Reader {
	if a.Stdin == nil {
		return strings.NewReader("")
	}
	return a.Stdin
}

func (a *App) cmdAll(ctx context.Context, args []string) error {
	fs := newFlagSet(a.stderr(), "all")
	mode := getenvDefault("NNCTL_OPTIMIZE", getenvDefault("BUILD_MODE", defaultBuildMode))
	gpu := ""
	fs.StringVar(&mode, "mode", mode, "Zig optimization mode")
	fs.StringVar(&mode, "optimize", mode, "Zig optimization mode")
	fs.StringVar(&gpu, "gpu", gpu, "GPU backend: none, auto, metal, or cuda")
	if err := parseInterspersed(fs, args, map[string]bool{
		"mode":     true,
		"optimize": true,
		"gpu":      true,
	}); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("all does not accept positional arguments")
	}

	if err := a.runZig(ctx, zigArgs("build", zigOptions{optimize: mode, gpu: gpu})...); err != nil {
		return err
	}
	if err := a.runZig(ctx, zigArgs("test", zigOptions{gpu: gpu})...); err != nil {
		return err
	}
	if err := a.runZig(ctx, zigArgs("test-acceptance", zigOptions{gpu: gpu})...); err != nil {
		return err
	}
	return a.runZig(ctx, zigArgs("examples", zigOptions{optimize: mode, gpu: gpu})...)
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
	modeProvided := hasAnyFlag(args, "mode", "optimize")
	if err := parseInterspersed(fs, args, map[string]bool{
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
	return a.runZig(ctx, zigArgs("build", zigOptions{optimize: mode, gpu: gpu})...)
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
	if err := parseInterspersed(fs, args, map[string]bool{
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
	opts := zigOptions{optimize: mode, gpu: gpu, summary: summary}
	if name != "" {
		step := "test-" + zigIdentifier(name)
		return a.runZig(ctx, zigArgs(step, opts)...)
	}
	if acceptanceOnly {
		return a.runZig(ctx, zigArgs("test-acceptance", opts)...)
	}
	if err := a.runZig(ctx, zigArgs("test", opts)...); err != nil {
		return err
	}
	if unitOnly {
		return nil
	}
	return a.runZig(ctx, zigArgs("test-acceptance", opts)...)
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
	return a.runZig(ctx, zigArgs("examples", zigOptions{optimize: mode, gpu: gpu})...)
}

func (a *App) cmdRun(ctx context.Context, args []string) error {
	beforePassthrough, passthrough := splitPassthrough(args)

	fs := newFlagSet(a.stderr(), "run")
	mode := ""
	gpu := ""
	fs.StringVar(&mode, "mode", mode, "Zig optimization mode")
	fs.StringVar(&mode, "optimize", mode, "Zig optimization mode")
	fs.StringVar(&gpu, "gpu", gpu, "GPU backend: none, auto, metal, or cuda")
	if err := parseInterspersed(fs, beforePassthrough, map[string]bool{
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
	if normalizeName(name) == "quick" {
		for _, example := range examples {
			if !example.Quick {
				continue
			}
			opts := exampleRunOptions(example, mode, gpu)
			if err := a.runZig(ctx, zigRunArgs(example.Step, opts, nil)...); err != nil {
				return err
			}
		}
		return nil
	}

	example, ok := resolveExample(name)
	if !ok {
		return fmt.Errorf("unknown example %q; run nnctl list examples", name)
	}
	opts := exampleRunOptions(example, mode, gpu)
	return a.runZig(ctx, zigRunArgs(example.Step, opts, passthrough)...)
}

func (a *App) cmdList(args []string) error {
	topic := "tasks"
	if len(args) > 1 {
		return fmt.Errorf("list accepts at most one topic")
	}
	if len(args) == 1 {
		topic = normalizeName(args[0])
	}

	switch topic {
	case "tasks", "commands":
		printUsage(a.stdout())
	case "examples", "example":
		for _, example := range sortedExamples() {
			if example.Hidden {
				continue
			}
			fmt.Fprintf(a.stdout(), "%-24s %s\n", example.Name, example.Description)
		}
	case "tests", "test":
		for _, test := range tests {
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

func (a *App) cmdData(ctx context.Context, args []string) error {
	if len(args) == 0 {
		printDataHelp(a.stdout())
		return nil
	}
	switch args[0] {
	case "mnist":
		return a.cmdDataMNIST(ctx, args[1:])
	case "tiny-gpt", "tinygpt":
		return a.cmdDataTinyGPT(ctx, args[1:])
	case "help", "-h", "--help":
		printDataHelp(a.stdout())
		return nil
	default:
		return fmt.Errorf("unknown data command %q", args[0])
	}
}

func (a *App) cmdDataMNIST(ctx context.Context, args []string) error {
	fs := newFlagSet(a.stderr(), "data mnist")
	dir := "data"
	force := false
	fs.StringVar(&dir, "dir", dir, "data directory")
	fs.BoolVar(&force, "force", false, "overwrite existing files")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("data mnist does not accept positional arguments")
	}
	if !filepath.IsAbs(dir) {
		dir = filepath.Join(a.repoRoot, dir)
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("create data directory: %w", err)
	}
	for _, file := range mnistFiles {
		if err := a.downloadGzip(ctx, file.URL, filepath.Join(dir, file.Name), force); err != nil {
			return err
		}
	}
	return nil
}

func (a *App) cmdDataTinyGPT(ctx context.Context, args []string) error {
	fs := newFlagSet(a.stderr(), "data tiny-gpt")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("data tiny-gpt does not accept positional arguments")
	}
	script := filepath.Join(a.repoRoot, "examples", "tiny_gpt", "scripts", "prepare_data.sh")
	if !fileExists(script) {
		return fmt.Errorf("missing Tiny GPT data script at %s", script)
	}
	return a.run(ctx, a.repoRoot, script)
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

func (a *App) downloadGzip(ctx context.Context, url, dest string, force bool) error {
	if !force {
		if _, err := os.Stat(dest); err == nil {
			fmt.Fprintf(a.stdout(), "exists %s\n", filepath.Base(dest))
			return nil
		} else if !errors.Is(err, os.ErrNotExist) {
			return fmt.Errorf("stat %s: %w", dest, err)
		}
	}

	fmt.Fprintf(a.stdout(), "download %s\n", filepath.Base(dest))
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("download %s: %w", url, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download %s: %s", url, resp.Status)
	}

	gz, err := gzip.NewReader(resp.Body)
	if err != nil {
		return fmt.Errorf("open gzip stream for %s: %w", url, err)
	}
	defer gz.Close()

	tmp := dest + ".tmp"
	out, err := os.OpenFile(tmp, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		return fmt.Errorf("create %s: %w", tmp, err)
	}
	if _, err := io.Copy(out, gz); err != nil {
		out.Close()
		os.Remove(tmp)
		return fmt.Errorf("write %s: %w", dest, err)
	}
	if err := out.Close(); err != nil {
		os.Remove(tmp)
		return fmt.Errorf("close %s: %w", tmp, err)
	}
	if err := os.Rename(tmp, dest); err != nil {
		os.Remove(tmp)
		return fmt.Errorf("rename %s: %w", dest, err)
	}
	return nil
}

func (a *App) runZig(ctx context.Context, args ...string) error {
	return a.run(ctx, a.repoRoot, a.zig, args...)
}

func (a *App) run(ctx context.Context, dir, name string, args ...string) error {
	fmt.Fprintf(a.stderr(), "==> %s\n", commandString(name, args))
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

type zigOptions struct {
	optimize string
	gpu      string
	summary  string
}

func zigArgs(step string, opts zigOptions) []string {
	args := []string{"build"}
	if step != "" {
		args = append(args, step)
	}
	if opts.gpu != "" {
		args = append(args, "-Dgpu="+opts.gpu)
	}
	if opts.optimize != "" {
		args = append(args, "-Doptimize="+opts.optimize)
	}
	if opts.summary != "" {
		args = append(args, "--summary", opts.summary)
	}
	return args
}

func zigRunArgs(step string, opts zigOptions, passthrough []string) []string {
	args := zigArgs(step, opts)
	if len(passthrough) > 0 {
		args = append(args, "--")
		args = append(args, passthrough...)
	}
	return args
}

type exampleDef struct {
	Name            string
	Step            string
	Description     string
	DefaultGPU      string
	DefaultOptimize string
	Quick           bool
	Hidden          bool
	Aliases         []string
}

var examples = []exampleDef{
	{Name: "simple-xor", Step: "run_simple_xor", Description: "Run simple XOR", Quick: true},
	{Name: "gated-network", Step: "run_gated_network", Description: "Run gated network", Quick: true},
	{Name: "xor-training", Step: "run_xor_training", Description: "Run XOR training with backpropagation"},
	{Name: "binary-classification", Step: "run_binary_classification", Description: "Run binary classification"},
	{Name: "regression", Step: "run_regression", Description: "Run nonlinear regression"},
	{Name: "network-visualisation", Step: "run_network_visualisation", Description: "Run network visualisation", Quick: true, Aliases: []string{"network-visualization"}},
	{Name: "backend-demo", Step: "run_backend_demo", Description: "Run backend abstraction demo", Quick: true},
	{Name: "serving", Step: "run_serving", Description: "Run serving example"},
	{Name: "mnist", Step: "run_mnist", Description: "Run MNIST example"},
	{Name: "gpu", Step: "run_gpu", Description: "Run GPU example", DefaultGPU: "auto", Quick: true},
	{Name: "gpu-metal", Step: "run_gpu", Description: "Run GPU example with Metal", DefaultGPU: "metal"},
	{Name: "gpu-benchmark", Step: "run_gpu_benchmark", Description: "Benchmark Metal backend against CPU", DefaultGPU: "metal", DefaultOptimize: "ReleaseFast"},
	{Name: "turboquant", Step: "run_turboquant", Description: "Run TurboQuant paper lab", Quick: true},
	{Name: "tiny-gpt", Step: "run_tiny_gpt", Description: "Run tiny GPT decoder-only Transformer", Quick: true},
}

var tests = []string{
	"matrix",
	"activation",
	"layer",
	"network",
	"inference-service",
	"visualiser",
	"quantization",
	"backend",
	"cpu-backend",
	"metal-backend",
	"gated-network",
	"simple-xor",
	"xor-training",
	"binary-classification",
	"regression",
	"mnist",
	"serving",
	"network-visualisation",
	"backend-demo",
	"gpu",
	"gpu-benchmark",
	"turboquant",
	"tiny-gpt",
}

type mnistFile struct {
	Name string
	URL  string
}

var mnistFiles = []mnistFile{
	{Name: "train-images-idx3-ubyte", URL: "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz"},
	{Name: "train-labels-idx1-ubyte", URL: "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz"},
	{Name: "t10k-images-idx3-ubyte", URL: "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz"},
	{Name: "t10k-labels-idx1-ubyte", URL: "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz"},
}

func resolveExample(name string) (exampleDef, bool) {
	normalized := normalizeName(name)
	for _, example := range examples {
		if normalizeName(example.Name) == normalized {
			return example, true
		}
		for _, alias := range example.Aliases {
			if normalizeName(alias) == normalized {
				return example, true
			}
		}
	}
	return exampleDef{}, false
}

func exampleRunOptions(example exampleDef, mode, gpu string) zigOptions {
	if mode == "" {
		mode = example.DefaultOptimize
	}
	if mode == "" {
		mode = getenvDefault("NNCTL_OPTIMIZE", getenvDefault("BUILD_MODE", defaultBuildMode))
	}
	if gpu == "" {
		gpu = example.DefaultGPU
	}
	return zigOptions{optimize: mode, gpu: gpu}
}

func resolveRepoRoot(repo string) (string, error) {
	if repo != "" {
		abs, err := filepath.Abs(repo)
		if err != nil {
			return "", err
		}
		if !isRepoRoot(abs) {
			return "", fmt.Errorf("%s does not look like the zig-nn repo root", abs)
		}
		return abs, nil
	}
	cwd, err := os.Getwd()
	if err != nil {
		return "", err
	}
	for {
		if isRepoRoot(cwd) {
			return cwd, nil
		}
		parent := filepath.Dir(cwd)
		if parent == cwd {
			return "", fmt.Errorf("could not find repo root from %s", cwd)
		}
		cwd = parent
	}
}

func isRepoRoot(dir string) bool {
	return fileExists(filepath.Join(dir, "build.zig")) && fileExists(filepath.Join(dir, "build.zig.zon"))
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

func splitPassthrough(args []string) ([]string, []string) {
	for i, arg := range args {
		if arg == "--" {
			return args[:i], args[i+1:]
		}
	}
	return args, nil
}

func newFlagSet(out io.Writer, name string) *flag.FlagSet {
	fs := flag.NewFlagSet(name, flag.ContinueOnError)
	fs.SetOutput(out)
	return fs
}

func parseInterspersed(fs *flag.FlagSet, args []string, valueFlags map[string]bool) error {
	reordered, err := reorderInterspersed(args, valueFlags)
	if err != nil {
		return err
	}
	return fs.Parse(reordered)
}

func reorderInterspersed(args []string, valueFlags map[string]bool) ([]string, error) {
	flags := make([]string, 0, len(args))
	positionals := make([]string, 0, len(args))
	for i := 0; i < len(args); i++ {
		arg := args[i]
		if !strings.HasPrefix(arg, "-") || arg == "-" {
			positionals = append(positionals, arg)
			continue
		}

		name, hasInlineValue := parseFlagName(arg)
		wantsValue, known := valueFlags[name]
		if !known {
			flags = append(flags, arg)
			continue
		}

		flags = append(flags, arg)
		if wantsValue && !hasInlineValue {
			if i+1 >= len(args) {
				return nil, fmt.Errorf("flag needs an argument: -%s", name)
			}
			i++
			flags = append(flags, args[i])
		}
	}
	return append(flags, positionals...), nil
}

func parseFlagName(arg string) (string, bool) {
	arg = strings.TrimLeft(arg, "-")
	if arg == "" {
		return "", false
	}
	if index := strings.IndexByte(arg, '='); index >= 0 {
		return arg[:index], true
	}
	return arg, false
}

func hasAnyFlag(args []string, names ...string) bool {
	for _, arg := range args {
		if !strings.HasPrefix(arg, "-") || arg == "-" {
			continue
		}
		name, _ := parseFlagName(arg)
		for _, candidate := range names {
			if name == candidate {
				return true
			}
		}
	}
	return false
}

func getenvDefault(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}

func normalizeName(name string) string {
	name = strings.TrimSpace(strings.ToLower(name))
	name = strings.TrimPrefix(name, "example-")
	name = strings.ReplaceAll(name, "_", "-")
	return name
}

func zigIdentifier(name string) string {
	return strings.ReplaceAll(normalizeName(name), "-", "_")
}

func commandString(name string, args []string) string {
	parts := append([]string{name}, args...)
	for i, part := range parts {
		parts[i] = quoteShell(part)
	}
	return strings.Join(parts, " ")
}

func quoteShell(s string) string {
	if s == "" {
		return `""`
	}
	if strings.ContainsAny(s, " \t\n\"'`$\\") {
		return strconv.Quote(s)
	}
	return s
}

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
	switch normalizeName(args[0]) {
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

func sortedExamples() []exampleDef {
	sorted := append([]exampleDef(nil), examples...)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Name < sorted[j].Name
	})
	return sorted
}
