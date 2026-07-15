// Package cli implements the nnctl command-line interface.
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

type app struct {
	Stdin  io.Reader
	Stdout io.Writer
	Stderr io.Writer

	repoRoot string
	zig      string
}

// Main runs nnctl with the process standard streams and returns its exit code.
func Main(args []string) int {
	app := &app{
		Stdin:  os.Stdin,
		Stdout: os.Stdout,
		Stderr: os.Stderr,
	}
	if err := app.execute(context.Background(), args); err != nil {
		fmt.Fprintln(app.stderr(), "error:", err)
		return 1
	}
	return 0
}

func (a *app) execute(ctx context.Context, args []string) error {
	root := a.newRootCommand()
	root.SetArgs(args)
	return root.ExecuteContext(ctx)
}

func (a *app) stdout() io.Writer {
	if a.Stdout == nil {
		return io.Discard
	}
	return a.Stdout
}

func (a *app) stderr() io.Writer {
	if a.Stderr == nil {
		return io.Discard
	}
	return a.Stderr
}

func (a *app) stdin() io.Reader {
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
