package catalog

import "testing"

func TestResolveExampleAliases(t *testing.T) {
	example, ok := ResolveExample("example-network-visualization")
	if !ok {
		t.Fatal("expected alias to resolve")
	}
	if example.Step != "run_network_visualisation" {
		t.Fatalf("unexpected step: %s", example.Step)
	}
}

func TestResolveTinyGPTOpenAIExample(t *testing.T) {
	example, ok := ResolveExample("tiny_gpt_openai")
	if !ok {
		t.Fatal("expected tiny_gpt_openai to resolve")
	}
	if example.Step != "run_tiny_gpt_openai" {
		t.Fatalf("unexpected step: %s", example.Step)
	}
}

func TestResolveOptimizerLabExample(t *testing.T) {
	example, ok := ResolveExample("optimizer_lab")
	if !ok {
		t.Fatal("expected optimizer_lab to resolve")
	}
	if example.Step != "run_optimizer_lab" {
		t.Fatalf("unexpected step: %s", example.Step)
	}
	if example.DefaultGPU != "auto" || !example.Quick {
		t.Fatalf("unexpected optimizer lab defaults: %+v", example)
	}
}
