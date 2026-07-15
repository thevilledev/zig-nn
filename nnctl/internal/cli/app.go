// Package cli implements the nnctl command-line interface.
package cli

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
)

const (
	defaultBuildMode   = "Debug"
	defaultReleaseMode = "ReleaseSafe"
)

type app struct {
	stdinReader  io.Reader
	stdoutWriter io.Writer
	stderrWriter io.Writer
	httpClient   httpDoer

	repoRoot string
	zig      string
}

type httpDoer interface {
	Do(*http.Request) (*http.Response, error)
}

func newApp(stdin io.Reader, stdout, stderr io.Writer) *app {
	return &app{
		stdinReader:  stdin,
		stdoutWriter: stdout,
		stderrWriter: stderr,
		httpClient:   http.DefaultClient,
	}
}

// Main runs nnctl with the process standard streams and returns its exit code.
func Main(args []string) int {
	app := newApp(os.Stdin, os.Stdout, os.Stderr)
	if err := app.execute(context.Background(), args); err != nil {
		_, _ = fmt.Fprintln(app.stderr(), "error:", err)
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
	if a.stdoutWriter == nil {
		return io.Discard
	}
	return a.stdoutWriter
}

func (a *app) stderr() io.Writer {
	if a.stderrWriter == nil {
		return io.Discard
	}
	return a.stderrWriter
}

func (a *app) stdin() io.Reader {
	if a.stdinReader == nil {
		return strings.NewReader("")
	}
	return a.stdinReader
}

func (a *app) http() httpDoer {
	if a.httpClient == nil {
		return http.DefaultClient
	}
	return a.httpClient
}

func getenvDefault(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}
