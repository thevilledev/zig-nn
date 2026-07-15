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

func TestResolveCNNExample(t *testing.T) {
	example, ok := ResolveExample("cnn")
	if !ok {
		t.Fatal("expected cnn to resolve")
	}
	if example.Step != "run_cnn" || example.Quick {
		t.Fatalf("unexpected cnn example: %+v", example)
	}
}

func TestResolveAutoencoderExample(t *testing.T) {
	example, ok := ResolveExample("autoencoder")
	if !ok {
		t.Fatal("expected autoencoder to resolve")
	}
	if example.Step != "run_autoencoder" || example.DefaultGPU != "auto" || example.Quick {
		t.Fatalf("unexpected autoencoder example: %+v", example)
	}
}

func TestResolveGRUSequenceExample(t *testing.T) {
	example, ok := ResolveExample("gru_sequence")
	if !ok {
		t.Fatal("expected gru_sequence to resolve")
	}
	if example.Step != "run_gru_sequence" || example.DefaultGPU != "auto" || example.Quick {
		t.Fatalf("unexpected gru sequence example: %+v", example)
	}
}

func TestResolveTransformerEncoderExample(t *testing.T) {
	example, ok := ResolveExample("transformer_encoder")
	if !ok {
		t.Fatal("expected transformer_encoder to resolve")
	}
	if example.Step != "run_transformer_encoder" || example.DefaultGPU != "auto" || example.Quick {
		t.Fatalf("unexpected transformer encoder example: %+v", example)
	}
}

func TestResolveDQNExample(t *testing.T) {
	example, ok := ResolveExample("dqn")
	if !ok {
		t.Fatal("expected dqn to resolve")
	}
	if example.Step != "run_dqn" || example.DefaultGPU != "auto" || example.Quick {
		t.Fatalf("unexpected dqn example: %+v", example)
	}
}
