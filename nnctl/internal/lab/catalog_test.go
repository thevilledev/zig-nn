package lab

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestLearningCatalogUsesRunnableExperiments(t *testing.T) {
	for _, spec := range Experiments() {
		if spec.Step == "" {
			t.Fatalf("experiment %s has no Zig step", spec.ID)
		}
		if len(spec.Parameters) != 3 {
			t.Fatalf("experiment %s has %d parameters", spec.ID, len(spec.Parameters))
		}
	}
}

func TestBuildArgumentsDefaultsAndValidation(t *testing.T) {
	spec, ok := ResolveExperiment("xor-training")
	if !ok {
		t.Fatal("xor-training not found")
	}
	args, err := BuildArguments(spec, map[string]json.Number{"epochs": json.Number("250")})
	if err != nil {
		t.Fatal(err)
	}
	joined := strings.Join(args, " ")
	for _, want := range []string{"--format ndjson", "--epochs 250", "--learning-rate 0.3", "--seed 42"} {
		if !strings.Contains(joined, want) {
			t.Fatalf("arguments %q missing %q", joined, want)
		}
	}

	if _, err := BuildArguments(spec, map[string]json.Number{"epochs": json.Number("99")}); err == nil {
		t.Fatal("expected lower-bound error")
	}
	if _, err := BuildArguments(spec, map[string]json.Number{"seed": json.Number("1.5")}); err == nil {
		t.Fatal("expected integer error")
	}
	if _, err := BuildArguments(spec, map[string]json.Number{"command": json.Number("1")}); err == nil {
		t.Fatal("expected unknown parameter error")
	}
}
