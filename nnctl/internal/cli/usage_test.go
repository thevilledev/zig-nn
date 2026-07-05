package cli

import (
	"bytes"
	"context"
	"strings"
	"testing"
)

func TestDirectHelpAvailableForEveryCommand(t *testing.T) {
	tests := [][]string{
		{"all", "--help"},
		{"build", "--help"},
		{"release", "--help"},
		{"test", "--help"},
		{"examples", "--help"},
		{"run", "--help"},
		{"run", "tiny-gpt", "--help"},
		{"train", "--help"},
		{"train", "tiny-gpt", "--help"},
		{"chat", "--help"},
		{"list", "--help"},
		{"fmt", "--help"},
		{"format", "--help"},
		{"clean", "--help"},
		{"data", "--help"},
		{"data", "mnist", "--help"},
		{"data", "tiny-gpt", "--help"},
		{"doctor", "--help"},
		{"env", "--help"},
		{"version", "--help"},
		{"help", "tiny-gpt"},
	}

	for _, args := range tests {
		t.Run(strings.Join(args, " "), func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			app := &App{Stdout: &stdout, Stderr: &stderr}

			if err := app.Run(context.Background(), args); err != nil {
				t.Fatalf("Run() error = %v", err)
			}
			if !strings.Contains(stdout.String(), "Usage:") || !strings.Contains(stdout.String(), "nnctl") {
				t.Fatalf("stdout did not contain help usage:\n%s", stdout.String())
			}
			if strings.Contains(stderr.String(), "flag: help requested") {
				t.Fatalf("help leaked flag.ErrHelp to stderr:\n%s", stderr.String())
			}
		})
	}
}

func TestNestedHelpContainsTinyGPTCorpusWorkflow(t *testing.T) {
	var stdout bytes.Buffer
	app := &App{Stdout: &stdout}

	if err := app.Run(context.Background(), []string{"help", "train", "tiny-gpt"}); err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	help := stdout.String()
	for _, want := range []string{
		"--corpus",
		"auto",
		"toy",
		"shakespeare",
		"tinystories",
		"--corpus-path",
		"--eval-split",
		"--lr-schedule",
		"--summary-path",
		"nnctl data tiny-gpt",
	} {
		if !strings.Contains(help, want) {
			t.Fatalf("TinyGPT train help missing %q:\n%s", want, help)
		}
	}
}

func TestHelpAfterRunPassthroughSeparatorReachesExample(t *testing.T) {
	var stdout, stderr bytes.Buffer
	app := &App{Stdout: &stdout, Stderr: &stderr}

	if err := app.Run(context.Background(), []string{"--zig", "/bin/echo", "run", "tiny-gpt", "--", "--help"}); err != nil {
		t.Fatalf("Run() error = %v\nstderr:\n%s", err, stderr.String())
	}
	if strings.Contains(stdout.String(), "Usage: nnctl run tiny-gpt") {
		t.Fatalf("passthrough --help was captured by Cobra help:\n%s", stdout.String())
	}
	if !strings.Contains(stdout.String(), "--help") {
		t.Fatalf("passthrough --help did not reach the example command:\nstdout:\n%s\nstderr:\n%s", stdout.String(), stderr.String())
	}
}
