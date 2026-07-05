package cli

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"runtime"
	"strings"

	"nnctl/internal/repo"
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

	repoRoot := ""
	zig := getenvDefault("ZIG", "zig")
	global.StringVar(&repoRoot, "repo", "", "repository root to operate on")
	global.StringVar(&repoRoot, "C", "", "repository root to operate on")
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

	root, err := repo.ResolveRoot(repoRoot)
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
	case "train":
		return a.cmdTrain(ctx, commandArgs)
	case "chat":
		return a.cmdChat(ctx, commandArgs)
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

func newFlagSet(out io.Writer, name string) *flag.FlagSet {
	fs := flag.NewFlagSet(name, flag.ContinueOnError)
	fs.SetOutput(out)
	return fs
}

func getenvDefault(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}
