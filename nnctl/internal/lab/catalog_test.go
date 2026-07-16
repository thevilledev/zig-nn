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
		if len(spec.Backends) == 0 || spec.DefaultBackend == "" {
			t.Fatalf("experiment %s has no backend contract", spec.ID)
		}
		if spec.Parameters == nil || spec.Metrics == nil {
			t.Fatalf("experiment %s exposes a null collection", spec.ID)
		}
	}
}

func TestBuildRunOptionsDefaultsAndValidation(t *testing.T) {
	spec, ok := ResolveExperiment("xor-training")
	if !ok {
		t.Fatal("xor-training not found")
	}
	options, err := BuildRunOptions(spec, "", map[string]json.Number{"epochs": json.Number("250")})
	if err != nil {
		t.Fatal(err)
	}
	if options.Backend != "cpu" {
		t.Fatalf("backend = %q", options.Backend)
	}
	joined := strings.Join(options.Arguments, " ")
	for _, want := range []string{"--format ndjson", "--epochs 250", "--learning-rate 0.3", "--seed 42"} {
		if !strings.Contains(joined, want) {
			t.Fatalf("arguments %q missing %q", joined, want)
		}
	}

	if _, err := BuildRunOptions(spec, "cpu", map[string]json.Number{"epochs": json.Number("99")}); err == nil {
		t.Fatal("expected lower-bound error")
	}
	if _, err := BuildRunOptions(spec, "cpu", map[string]json.Number{"seed": json.Number("1.5")}); err == nil {
		t.Fatal("expected integer error")
	}
	if _, err := BuildRunOptions(spec, "cpu", map[string]json.Number{"command": json.Number("1")}); err == nil {
		t.Fatal("expected unknown parameter error")
	}
	if _, err := BuildRunOptions(spec, "metal", nil); err == nil {
		t.Fatal("expected unsupported backend error")
	}

	optimizer, _ := ResolveExperiment("optimizer-lab")
	metal, err := BuildRunOptions(optimizer, "metal", nil)
	if err != nil || !strings.Contains(strings.Join(metal.Arguments, " "), "--backend metal") {
		t.Fatalf("metal options = %#v, %v", metal, err)
	}
}
