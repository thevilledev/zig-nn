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

func TestResolveTokenizerLabExample(t *testing.T) {
	example, ok := ResolveExample("tokenizer_lab")
	if !ok {
		t.Fatal("expected tokenizer_lab to resolve")
	}
	if example.Step != "run_tokenizer_lab" || !example.Quick {
		t.Fatalf("unexpected tokenizer lab example: %+v", example)
	}
}

func TestResolvePaddingMasksExample(t *testing.T) {
	example, ok := ResolveExample("padding_masks")
	if !ok {
		t.Fatal("expected padding_masks to resolve")
	}
	if example.Step != "run_padding_masks" || example.DefaultGPU != "auto" || !example.Quick {
		t.Fatalf("unexpected padding masks example: %+v", example)
	}
}

func TestResolveWord2VecExample(t *testing.T) {
	example, ok := ResolveExample("word2vec")
	if !ok {
		t.Fatal("expected word2vec to resolve")
	}
	if example.Step != "run_word2vec" || example.DefaultGPU != "auto" || example.Quick {
		t.Fatalf("unexpected word2vec example: %+v", example)
	}
}

func TestResolveTextClassifierExample(t *testing.T) {
	example, ok := ResolveExample("text_classifier")
	if !ok {
		t.Fatal("expected text_classifier to resolve")
	}
	if example.Step != "run_text_classifier" || example.DefaultGPU != "auto" || example.Quick {
		t.Fatalf("unexpected text classifier example: %+v", example)
	}
}

func TestResolveSequenceTaggingExample(t *testing.T) {
	example, ok := ResolveExample("sequence_tagging")
	if !ok {
		t.Fatal("expected sequence_tagging to resolve")
	}
	if example.Step != "run_sequence_tagging" || !example.Quick {
		t.Fatalf("unexpected sequence tagging example: %+v", example)
	}
}

func TestResolveDecodingLabExample(t *testing.T) {
	example, ok := ResolveExample("decoding_lab")
	if !ok {
		t.Fatal("expected decoding_lab to resolve")
	}
	if example.Step != "run_decoding_lab" || !example.Quick {
		t.Fatalf("unexpected decoding lab example: %+v", example)
	}
}

func TestResolveSeq2SeqExample(t *testing.T) {
	example, ok := ResolveExample("seq2seq")
	if !ok {
		t.Fatal("expected seq2seq to resolve")
	}
	if example.Step != "run_seq2seq" || example.DefaultGPU != "auto" || example.Quick {
		t.Fatalf("unexpected seq2seq example: %+v", example)
	}
}

func TestResolveSemanticSearchExample(t *testing.T) {
	example, ok := ResolveExample("semantic_search")
	if !ok {
		t.Fatal("expected semantic_search to resolve")
	}
	if example.Step != "run_semantic_search" || example.DefaultGPU != "auto" || !example.Quick {
		t.Fatalf("unexpected semantic search example: %+v", example)
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
