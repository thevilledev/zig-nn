package cli

import (
	"testing"
)

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
