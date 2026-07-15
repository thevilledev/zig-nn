package catalog

import "testing"

func TestResolveExperimentAliases(t *testing.T) {
	for _, name := range []string{"experiment-network-visualization", "example-network-visualization"} {
		experiment, ok := ResolveExperiment(name)
		if !ok {
			t.Fatalf("expected %q alias to resolve", name)
		}
		if experiment.Step != "run_network_visualisation" {
			t.Fatalf("unexpected step for %q: %s", name, experiment.Step)
		}
	}
}

func TestResolveTinyGPTOpenAIExperiment(t *testing.T) {
	experiment, ok := ResolveExperiment("tiny_gpt_openai")
	if !ok {
		t.Fatal("expected tiny_gpt_openai to resolve")
	}
	if experiment.Step != "run_tiny_gpt_openai" {
		t.Fatalf("unexpected step: %s", experiment.Step)
	}
}

func TestResolveOptimizerLabExperiment(t *testing.T) {
	experiment, ok := ResolveExperiment("optimizer_lab")
	if !ok {
		t.Fatal("expected optimizer_lab to resolve")
	}
	if experiment.Step != "run_optimizer_lab" {
		t.Fatalf("unexpected step: %s", experiment.Step)
	}
	if experiment.DefaultGPU != "auto" || !experiment.Quick {
		t.Fatalf("unexpected optimizer lab defaults: %+v", experiment)
	}
}

func TestResolveTokenizerLabExperiment(t *testing.T) {
	experiment, ok := ResolveExperiment("tokenizer_lab")
	if !ok {
		t.Fatal("expected tokenizer_lab to resolve")
	}
	if experiment.Step != "run_tokenizer_lab" || !experiment.Quick {
		t.Fatalf("unexpected tokenizer lab experiment: %+v", experiment)
	}
}

func TestResolvePaddingMasksExperiment(t *testing.T) {
	experiment, ok := ResolveExperiment("padding_masks")
	if !ok {
		t.Fatal("expected padding_masks to resolve")
	}
	if experiment.Step != "run_padding_masks" || experiment.DefaultGPU != "auto" || !experiment.Quick {
		t.Fatalf("unexpected padding masks experiment: %+v", experiment)
	}
}

func TestResolveWord2VecExperiment(t *testing.T) {
	experiment, ok := ResolveExperiment("word2vec")
	if !ok {
		t.Fatal("expected word2vec to resolve")
	}
	if experiment.Step != "run_word2vec" || experiment.DefaultGPU != "auto" || experiment.Quick {
		t.Fatalf("unexpected word2vec experiment: %+v", experiment)
	}
}

func TestResolveTextClassifierExperiment(t *testing.T) {
	experiment, ok := ResolveExperiment("text_classifier")
	if !ok {
		t.Fatal("expected text_classifier to resolve")
	}
	if experiment.Step != "run_text_classifier" || experiment.DefaultGPU != "auto" || experiment.Quick {
		t.Fatalf("unexpected text classifier experiment: %+v", experiment)
	}
}

func TestResolveSequenceTaggingExperiment(t *testing.T) {
	experiment, ok := ResolveExperiment("sequence_tagging")
	if !ok {
		t.Fatal("expected sequence_tagging to resolve")
	}
	if experiment.Step != "run_sequence_tagging" || !experiment.Quick {
		t.Fatalf("unexpected sequence tagging experiment: %+v", experiment)
	}
}

func TestResolveDecodingLabExperiment(t *testing.T) {
	experiment, ok := ResolveExperiment("decoding_lab")
	if !ok {
		t.Fatal("expected decoding_lab to resolve")
	}
	if experiment.Step != "run_decoding_lab" || !experiment.Quick {
		t.Fatalf("unexpected decoding lab experiment: %+v", experiment)
	}
}

func TestResolveSeq2SeqExperiment(t *testing.T) {
	experiment, ok := ResolveExperiment("seq2seq")
	if !ok {
		t.Fatal("expected seq2seq to resolve")
	}
	if experiment.Step != "run_seq2seq" || experiment.DefaultGPU != "auto" || experiment.Quick {
		t.Fatalf("unexpected seq2seq experiment: %+v", experiment)
	}
}

func TestResolveSemanticSearchExperiment(t *testing.T) {
	experiment, ok := ResolveExperiment("semantic_search")
	if !ok {
		t.Fatal("expected semantic_search to resolve")
	}
	if experiment.Step != "run_semantic_search" || experiment.DefaultGPU != "auto" || !experiment.Quick {
		t.Fatalf("unexpected semantic search experiment: %+v", experiment)
	}
}

func TestResolveCNNExperiment(t *testing.T) {
	experiment, ok := ResolveExperiment("cnn")
	if !ok {
		t.Fatal("expected cnn to resolve")
	}
	if experiment.Step != "run_cnn" || experiment.Quick {
		t.Fatalf("unexpected cnn experiment: %+v", experiment)
	}
}

func TestResolveAutoencoderExperiment(t *testing.T) {
	experiment, ok := ResolveExperiment("autoencoder")
	if !ok {
		t.Fatal("expected autoencoder to resolve")
	}
	if experiment.Step != "run_autoencoder" || experiment.DefaultGPU != "auto" || experiment.Quick {
		t.Fatalf("unexpected autoencoder experiment: %+v", experiment)
	}
}

func TestResolveGRUSequenceExperiment(t *testing.T) {
	experiment, ok := ResolveExperiment("gru_sequence")
	if !ok {
		t.Fatal("expected gru_sequence to resolve")
	}
	if experiment.Step != "run_gru_sequence" || experiment.DefaultGPU != "auto" || experiment.Quick {
		t.Fatalf("unexpected gru sequence experiment: %+v", experiment)
	}
}

func TestResolveTransformerEncoderExperiment(t *testing.T) {
	experiment, ok := ResolveExperiment("transformer_encoder")
	if !ok {
		t.Fatal("expected transformer_encoder to resolve")
	}
	if experiment.Step != "run_transformer_encoder" || experiment.DefaultGPU != "auto" || experiment.Quick {
		t.Fatalf("unexpected transformer encoder experiment: %+v", experiment)
	}
}

func TestResolveDQNExperiment(t *testing.T) {
	experiment, ok := ResolveExperiment("dqn")
	if !ok {
		t.Fatal("expected dqn to resolve")
	}
	if experiment.Step != "run_dqn" || experiment.DefaultGPU != "auto" || experiment.Quick {
		t.Fatalf("unexpected dqn experiment: %+v", experiment)
	}
}
