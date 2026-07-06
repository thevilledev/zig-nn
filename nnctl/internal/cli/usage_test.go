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
		{"benchmark", "--help"},
		{"cloud", "--help"},
		{"cloud", "deploy", "--help"},
		{"cloud", "destroy", "--help"},
		{"cloud", "list", "--help"},
		{"cloud", "pricing", "--help"},
		{"cloud", "ssh-keys", "--help"},
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
		{"completion", "--help"},
		{"completion", "bash", "--help"},
		{"completion", "zsh", "--help"},
		{"completion", "fish", "--help"},
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
		"--preset",
		"coherent-small",
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

func TestCompletionScripts(t *testing.T) {
	tests := []struct {
		name string
		args []string
		want []string
	}{
		{
			name: "bash",
			args: []string{"completion", "bash"},
			want: []string{"# bash completion V2 for nnctl", "__complete"},
		},
		{
			name: "zsh",
			args: []string{"completion", "zsh"},
			want: []string{"#compdef nnctl", "__complete"},
		},
		{
			name: "fish",
			args: []string{"completion", "fish"},
			want: []string{"complete -c nnctl", "__complete"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			app := &App{Stdout: &stdout, Stderr: &stderr}

			if err := app.Run(context.Background(), tt.args); err != nil {
				t.Fatalf("Run() error = %v\nstderr:\n%s", err, stderr.String())
			}
			for _, want := range tt.want {
				if !strings.Contains(stdout.String(), want) {
					t.Fatalf("completion output missing %q:\n%s", want, stdout.String())
				}
			}
			if stderr.Len() != 0 {
				t.Fatalf("completion wrote to stderr:\n%s", stderr.String())
			}
		})
	}
}

func TestCompletionDoesNotResolveRepo(t *testing.T) {
	var stdout, stderr bytes.Buffer
	app := &App{Stdout: &stdout, Stderr: &stderr}

	if err := app.Run(context.Background(), []string{"--repo", "/definitely/not/zig-nn", "completion", "bash"}); err != nil {
		t.Fatalf("Run() error = %v\nstderr:\n%s", err, stderr.String())
	}
	if !strings.Contains(stdout.String(), "# bash completion V2 for nnctl") {
		t.Fatalf("completion output missing bash script:\n%s", stdout.String())
	}
}
