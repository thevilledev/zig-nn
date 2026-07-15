package catalog

import (
	"sort"
	"strings"
)

type Example struct {
	Name            string
	Step            string
	Description     string
	DefaultGPU      string
	DefaultOptimize string
	Quick           bool
	Hidden          bool
	Aliases         []string
}

var examples = []Example{
	{Name: "simple-xor", Step: "run_simple_xor", Description: "Run simple XOR", Quick: true},
	{Name: "gated-network", Step: "run_gated_network", Description: "Run gated network", Quick: true},
	{Name: "xor-training", Step: "run_xor_training", Description: "Run XOR training with backpropagation"},
	{Name: "binary-classification", Step: "run_binary_classification", Description: "Run binary classification"},
	{Name: "regression", Step: "run_regression", Description: "Run nonlinear regression"},
	{Name: "network-visualisation", Step: "run_network_visualisation", Description: "Run network visualisation", Quick: true, Aliases: []string{"network-visualization"}},
	{Name: "backend-demo", Step: "run_backend_demo", Description: "Run backend abstraction demo", Quick: true},
	{Name: "backend-training", Step: "run_backend_training", Description: "Run backend-aware training demo", Quick: true},
	{Name: "optimizer-lab", Step: "run_optimizer_lab", Description: "Compare SGD, momentum, and AdamW", DefaultGPU: "auto", Quick: true},
	{Name: "tokenizer-lab", Step: "run_tokenizer_lab", Description: "Compare byte and learned BPE tokenization", Quick: true},
	{Name: "padding-masks", Step: "run_padding_masks", Description: "Learn batching and padding masks", DefaultGPU: "auto", Quick: true},
	{Name: "word2vec", Step: "run_word2vec", Description: "Learn skip-gram word embeddings", DefaultGPU: "auto"},
	{Name: "text-classifier", Step: "run_text_classifier", Description: "Classify padded text with masked multi-head attention", DefaultGPU: "auto"},
	{Name: "sequence-tagging", Step: "run_sequence_tagging", Description: "Learn structured BIO tagging with a CRF", Quick: true},
	{Name: "decoding-lab", Step: "run_decoding_lab", Description: "Compare greedy, top-k, and nucleus decoding", Quick: true},
	{Name: "cnn", Step: "run_cnn", Description: "Learn image patterns with convolution and pooling"},
	{Name: "autoencoder", Step: "run_autoencoder", Description: "Learn denoising and latent representations", DefaultGPU: "auto"},
	{Name: "gru-sequence", Step: "run_gru_sequence", Description: "Learn selective sequence memory with a GRU", DefaultGPU: "auto"},
	{Name: "transformer-encoder", Step: "run_transformer_encoder", Description: "Learn bidirectional context with a Transformer encoder", DefaultGPU: "auto"},
	{Name: "dqn", Step: "run_dqn", Description: "Learn value-based reinforcement learning with DQN", DefaultGPU: "auto"},
	{Name: "serving", Step: "run_serving", Description: "Run serving example"},
	{Name: "mnist", Step: "run_mnist", Description: "Run MNIST example"},
	{Name: "gpu", Step: "run_gpu", Description: "Run GPU example", DefaultGPU: "auto", Quick: true},
	{Name: "gpu-metal", Step: "run_gpu", Description: "Run GPU example with Metal", DefaultGPU: "metal"},
	{Name: "gpu-cuda", Step: "run_gpu", Description: "Run GPU example with CUDA", DefaultGPU: "cuda"},
	{Name: "gpu-rocm", Step: "run_gpu", Description: "Run GPU example with ROCm", DefaultGPU: "rocm"},
	{Name: "gpu-benchmark", Step: "run_gpu_benchmark", Description: "Benchmark GPU backend against CPU", DefaultGPU: "auto", DefaultOptimize: "ReleaseFast"},
	{Name: "turboquant", Step: "run_turboquant", Description: "Run TurboQuant paper lab", Quick: true},
	{Name: "tiny-gpt", Step: "run_tiny_gpt", Description: "Run tiny GPT decoder-only Transformer", Quick: true},
	{Name: "tiny-gpt-openai", Step: "run_tiny_gpt_openai", Description: "Run Tiny GPT OpenAI-compatible server"},
}

var tests = []string{
	"matrix",
	"activation",
	"layer",
	"network",
	"inference-service",
	"visualiser",
	"quantization",
	"backend",
	"cpu-backend",
	"metal-backend",
	"cuda-backend",
	"rocm-backend",
	"gated-network",
	"simple-xor",
	"xor-training",
	"binary-classification",
	"regression",
	"mnist",
	"serving",
	"network-visualisation",
	"backend-demo",
	"backend-training",
	"optimizer-lab",
	"tokenizer-lab",
	"padding-masks",
	"word2vec",
	"text-classifier",
	"sequence-tagging",
	"decoding-lab",
	"cnn",
	"autoencoder",
	"gru-sequence",
	"transformer-encoder",
	"dqn",
	"gpu",
	"gpu-benchmark",
	"turboquant",
	"tiny-gpt",
	"tiny-gpt-openai",
}

func Examples() []Example {
	return append([]Example(nil), examples...)
}

func SortedExamples() []Example {
	sorted := Examples()
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Name < sorted[j].Name
	})
	return sorted
}

func Tests() []string {
	return append([]string(nil), tests...)
}

func ResolveExample(name string) (Example, bool) {
	normalized := NormalizeName(name)
	for _, example := range examples {
		if NormalizeName(example.Name) == normalized {
			return example, true
		}
		for _, alias := range example.Aliases {
			if NormalizeName(alias) == normalized {
				return example, true
			}
		}
	}
	return Example{}, false
}

func NormalizeName(name string) string {
	name = strings.TrimSpace(strings.ToLower(name))
	name = strings.TrimPrefix(name, "example-")
	name = strings.ReplaceAll(name, "_", "-")
	return name
}

func ZigIdentifier(name string) string {
	return strings.ReplaceAll(NormalizeName(name), "-", "_")
}
