package cli

import (
	"context"
	"fmt"
	"io"
	"os"
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
	root := a.newRootCommand()
	root.SetArgs(args)
	return root.ExecuteContext(ctx)
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

func getenvDefault(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}
