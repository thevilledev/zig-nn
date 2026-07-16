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
		{"deploy", "--help"},
		{"cloud", "--help"},
		{"cloud", "benchmark-deploy", "--help"},
		{"cloud", "benchmark", "--help"},
		{"cloud", "deploy", "--help"},
		{"cloud", "volume", "--help"},
		{"cloud", "volumes", "--help"},
		{"cloud", "destroy", "--help"},
		{"cloud", "packer-template", "--help"},
		{"cloud", "list", "--help"},
		{"cloud", "pricing", "--help"},
		{"cloud", "ssh-keys", "--help"},
		{"experiments", "--help"},
		{"examples", "--help"},
		{"run", "--help"},
		{"run", "tiny-gpt", "--help"},
		{"train", "--help"},
		{"train", "tiny-gpt", "--help"},
		{"train", "speech-commands", "--help"},
		{"chat", "--help"},
		{"lab", "--help"},
		{"list", "--help"},
		{"fmt", "--help"},
		{"format", "--help"},
		{"clean", "--help"},
		{"data", "--help"},
		{"data", "mnist", "--help"},
		{"data", "tiny-gpt", "--help"},
		{"data", "speech-commands", "--help"},
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
			app := &app{stdoutWriter: &stdout, stderrWriter: &stderr}

			if err := app.execute(context.Background(), args); err != nil {
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

func TestExperimentListCompatibilityAliases(t *testing.T) {
	outputs := make([]string, 0, 4)
	for _, topic := range []string{"experiments", "experiment", "examples", "example"} {
		var stdout bytes.Buffer
		app := &app{stdoutWriter: &stdout}
		if err := app.execute(context.Background(), []string{"list", topic}); err != nil {
			t.Fatalf("nnctl list %s: %v", topic, err)
		}
		outputs = append(outputs, stdout.String())
	}
	for index := 1; index < len(outputs); index++ {
		if outputs[index] != outputs[0] {
			t.Fatalf("list alias output differs:\nprimary:\n%s\nalias:\n%s", outputs[0], outputs[index])
		}
	}
}

func TestUnknownExperimentPointsToPrimaryListCommand(t *testing.T) {
	app := &app{}
	err := app.runExperiment(context.Background(), "definitely-missing", nil, "", "")
	if err == nil || !strings.Contains(err.Error(), "nnctl list experiments") {
		t.Fatalf("runExperiment() error = %v", err)
	}
}

func TestNestedHelpContainsTinyGPTCorpusWorkflow(t *testing.T) {
	var stdout bytes.Buffer
	app := &app{stdoutWriter: &stdout}

	if err := app.execute(context.Background(), []string{"help", "train", "tiny-gpt"}); err != nil {
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

func TestHelpAfterRunPassthroughSeparatorReachesExperiment(t *testing.T) {
	var stdout, stderr bytes.Buffer
	app := &app{stdoutWriter: &stdout, stderrWriter: &stderr}

	if err := app.execute(context.Background(), []string{"--zig", "/bin/echo", "run", "tiny-gpt", "--", "--help"}); err != nil {
		t.Fatalf("Run() error = %v\nstderr:\n%s", err, stderr.String())
	}
	if strings.Contains(stdout.String(), "Usage: nnctl run tiny-gpt") {
		t.Fatalf("passthrough --help was captured by Cobra help:\n%s", stdout.String())
	}
	if !strings.Contains(stdout.String(), "--help") {
		t.Fatalf("passthrough --help did not reach the experiment command:\nstdout:\n%s\nstderr:\n%s", stdout.String(), stderr.String())
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
			app := &app{stdoutWriter: &stdout, stderrWriter: &stderr}

			if err := app.execute(context.Background(), tt.args); err != nil {
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
	app := &app{stdoutWriter: &stdout, stderrWriter: &stderr}

	if err := app.execute(context.Background(), []string{"--repo", "/definitely/not/zig-nn", "completion", "bash"}); err != nil {
		t.Fatalf("Run() error = %v\nstderr:\n%s", err, stderr.String())
	}
	if !strings.Contains(stdout.String(), "# bash completion V2 for nnctl") {
		t.Fatalf("completion output missing bash script:\n%s", stdout.String())
	}
}
