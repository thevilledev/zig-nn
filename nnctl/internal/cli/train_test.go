package cli

import (
	"reflect"
	"testing"
)

func TestSpeechCommandsTrainArgs(t *testing.T) {
	args, err := speechCommandsTrainArgs(defaultSpeechCommandsTrainOptions())
	if err != nil {
		t.Fatal(err)
	}
	want := []string{
		"--train", "--data-dir", "data/mini_speech_commands", "--output", "speech-commands.bin",
		"--epochs", "12", "--batch-size", "64", "--learning-rate", "0.003",
		"--seed", "42", "--backend", "auto",
	}
	if !reflect.DeepEqual(args, want) {
		t.Fatalf("speechCommandsTrainArgs() = %#v, want %#v", args, want)
	}

	opts := defaultSpeechCommandsTrainOptions()
	opts.gpu = "none"
	args, err = speechCommandsTrainArgs(opts)
	if err != nil {
		t.Fatal(err)
	}
	if got := args[len(args)-1]; got != "cpu" {
		t.Fatalf("backend = %q, want cpu", got)
	}
}

func TestTinyGPTTrainArgsFreshCheckpoint(t *testing.T) {
	got, err := tinyGPTTrainArgs(tinyGPTTrainOptions{
		output:          "model.bin",
		corpus:          "tinystories",
		steps:           10,
		learningRate:    "0.01",
		minLearningRate: "0.001",
		lrSchedule:      "cosine",
		warmupSteps:     2,
		batchSize:       2,
		trainChars:      1024,
		evalSplit:       "0.1",
		evalWindows:     16,
		blockSize:       16,
		layers:          2,
		heads:           2,
		embd:            32,
		optimizer:       "adamw",
		weightDecay:     "0.01",
		prompt:          "Once",
		tokens:          20,
		temperature:     "0.8",
		topK:            8,
		seed:            7,
		summaryPath:     "runs/tinystories.json",
	})
	if err != nil {
		t.Fatal(err)
	}

	wantContains := []string{
		"--train-full",
		"--lr-schedule", "cosine",
		"--warmup-steps", "2",
		"--eval-split", "0.1",
		"--eval-windows", "16",
		"--corpus", "tinystories",
		"--block-size", "16",
		"--layers", "2",
		"--heads", "2",
		"--embd", "32",
		"--save-checkpoint", "model.bin",
		"--no-corpus-prior",
		"--summary-path", "runs/tinystories.json",
	}
	if !containsOrdered(got, wantContains) {
		t.Fatalf("args did not contain expected ordered subsequence\n got: %#v\nwant subsequence: %#v", got, wantContains)
	}
}

func TestTinyGPTTrainCoherentSmallPreset(t *testing.T) {
	opts := defaultTinyGPTTrainOptions()
	opts.output = "model.bin"
	opts.preset = "coherent-small"

	if err := applyTinyGPTTrainPreset(&opts, func(string) bool { return false }); err != nil {
		t.Fatal(err)
	}
	got, err := tinyGPTTrainArgs(opts)
	if err != nil {
		t.Fatal(err)
	}

	wantContains := []string{
		"--full-train-steps", "3000",
		"--full-learning-rate", "0.001",
		"--min-learning-rate", "0.0001",
		"--lr-schedule", "cosine",
		"--warmup-steps", "100",
		"--full-batch-size", "12",
		"--train-chars", "0",
		"--eval-split", "0.05",
		"--eval-windows", "64",
		"--optimizer", "adamw",
		"--weight-decay", "0.01",
		"--temperature", "0.8",
		"--top-k", "16",
		"--corpus", "tinystories",
		"--block-size", "64",
		"--layers", "4",
		"--heads", "4",
		"--embd", "128",
		"--no-corpus-prior",
	}
	if !containsOrdered(got, wantContains) {
		t.Fatalf("coherent-small args missing expected subsequence\n got: %#v\nwant subsequence: %#v", got, wantContains)
	}
}

func TestTinyGPTTrainCoherentSmallPresetAllowsExplicitOverrides(t *testing.T) {
	opts := defaultTinyGPTTrainOptions()
	opts.output = "model.bin"
	opts.preset = "coherent-small"
	opts.corpus = "toy"
	opts.batchSize = 2
	opts.topK = 4
	opts.corpusPrior = true

	changed := map[string]bool{
		"corpus":       true,
		"batch-size":   true,
		"top-k":        true,
		"corpus-prior": true,
	}
	if err := applyTinyGPTTrainPreset(&opts, func(name string) bool { return changed[name] }); err != nil {
		t.Fatal(err)
	}
	got, err := tinyGPTTrainArgs(opts)
	if err != nil {
		t.Fatal(err)
	}

	for _, want := range [][]string{
		{"--corpus", "toy"},
		{"--full-batch-size", "2"},
		{"--top-k", "4"},
	} {
		if !containsOrdered(got, want) {
			t.Fatalf("explicit override missing %v from args: %#v", want, got)
		}
	}
	if containsString(got, "--no-corpus-prior") {
		t.Fatalf("explicit --corpus-prior override should not emit --no-corpus-prior: %#v", got)
	}
}

func TestTinyGPTTrainArgsResumeSavesBackByDefault(t *testing.T) {
	got, err := tinyGPTTrainArgs(tinyGPTTrainOptions{
		output:          "model.bin",
		resume:          "model.bin",
		corpus:          "toy",
		steps:           1,
		learningRate:    "0.01",
		minLearningRate: "0",
		lrSchedule:      "constant",
		batchSize:       1,
		trainChars:      64,
		validationChars: 0,
		evalWindows:     1,
		optimizer:       "sgd",
		weightDecay:     "0",
		prompt:          "to",
		tokens:          4,
		temperature:     "1.0",
		topK:            4,
		seed:            1,
	})
	if err != nil {
		t.Fatal(err)
	}

	if containsOrdered(got, []string{"--save-checkpoint", "model.bin"}) {
		t.Fatalf("resume-to-same checkpoint should use --resume-checkpoint without a duplicate save flag: %#v", got)
	}
	if !containsOrdered(got, []string{"--resume-checkpoint", "model.bin"}) {
		t.Fatalf("missing resume checkpoint args: %#v", got)
	}
}

func TestTinyGPTTrainPresetRejectsUnknownValue(t *testing.T) {
	opts := defaultTinyGPTTrainOptions()
	opts.preset = "large-chaos"
	if err := applyTinyGPTTrainPreset(&opts, func(string) bool { return false }); err == nil {
		t.Fatal("expected unknown preset error")
	}
}

func TestTinyGPTTrainArgsRejectsMissingOutput(t *testing.T) {
	_, err := tinyGPTTrainArgs(tinyGPTTrainOptions{
		steps:        1,
		learningRate: "0.01",
		batchSize:    1,
		blockSize:    1,
		layers:       1,
		heads:        1,
		embd:         1,
		tokens:       1,
		topK:         1,
		seed:         1,
	})
	if err == nil {
		t.Fatal("expected missing output error")
	}
}

func containsString(items []string, needle string) bool {
	for _, item := range items {
		if item == needle {
			return true
		}
	}
	return false
}

func containsOrdered(haystack, needle []string) bool {
	if len(needle) == 0 {
		return true
	}
	if len(needle) > len(haystack) {
		return false
	}
	pos := 0
	for _, item := range haystack {
		if item == needle[pos] {
			pos++
			if pos == len(needle) {
				return true
			}
		}
	}
	return false
}
