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
	{Name: "serving", Step: "run_serving", Description: "Run serving example"},
	{Name: "mnist", Step: "run_mnist", Description: "Run MNIST example"},
	{Name: "gpu", Step: "run_gpu", Description: "Run GPU example", DefaultGPU: "auto", Quick: true},
	{Name: "gpu-metal", Step: "run_gpu", Description: "Run GPU example with Metal", DefaultGPU: "metal"},
	{Name: "gpu-benchmark", Step: "run_gpu_benchmark", Description: "Benchmark Metal backend against CPU", DefaultGPU: "metal", DefaultOptimize: "ReleaseFast"},
	{Name: "turboquant", Step: "run_turboquant", Description: "Run TurboQuant paper lab", Quick: true},
	{Name: "tiny-gpt", Step: "run_tiny_gpt", Description: "Run tiny GPT decoder-only Transformer", Quick: true},
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
	"gated-network",
	"simple-xor",
	"xor-training",
	"binary-classification",
	"regression",
	"mnist",
	"serving",
	"network-visualisation",
	"backend-demo",
	"gpu",
	"gpu-benchmark",
	"turboquant",
	"tiny-gpt",
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
