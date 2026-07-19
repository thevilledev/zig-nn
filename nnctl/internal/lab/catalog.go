package lab

import (
	"encoding/json"
	"fmt"
	"math"
	"strconv"

	"nnctl/internal/catalog"
)

type ParameterKind string

const (
	ParameterInteger ParameterKind = "integer"
	ParameterNumber  ParameterKind = "number"
)

type ParameterSpec struct {
	Name    string        `json:"name"`
	Label   string        `json:"label"`
	Help    string        `json:"help"`
	Kind    ParameterKind `json:"kind"`
	Default float64       `json:"default"`
	Min     float64       `json:"min"`
	Max     float64       `json:"max"`
	Step    float64       `json:"step"`
	Flag    string        `json:"-"`
}

type MetricSpec struct {
	Name  string `json:"name"`
	Label string `json:"label"`
}

type ExperimentSpec struct {
	ID             string          `json:"id"`
	Category       string          `json:"category"`
	Title          string          `json:"title"`
	Description    string          `json:"description"`
	Question       string          `json:"question"`
	Observe        []string        `json:"observe"`
	Interpretation []string        `json:"interpretation"`
	Visualization  string          `json:"visualization"`
	Sources        []string        `json:"sources"`
	Parameters     []ParameterSpec `json:"parameters"`
	Metrics        []MetricSpec    `json:"metrics"`
	Backends       []string        `json:"backends"`
	DefaultBackend string          `json:"default_backend"`
	Step           string          `json:"-"`
	BackendFlag    bool            `json:"-"`
	Optimize       string          `json:"-"`
}

var learningSpecs = []ExperimentSpec{
	{
		ID:          "xor-training",
		Category:    "Foundations",
		Title:       "Learning XOR",
		Description: "Watch a small multilayer network turn four contradictory-looking examples into a nonlinear rule.",
		Question:    "How does backpropagation make a network solve a problem that no single straight boundary can represent?",
		Observe:     []string{"Whether loss falls steadily", "How the four output probabilities separate", "Why hidden nonlinear layers are necessary"},
		Interpretation: []string{
			"Falling loss means the weights are making the four predictions agree more closely with the XOR truth table.",
			"Predictions near 0.5 are uncertain; successful training pushes the positive cases toward 1 and the negative cases toward 0.",
		},
		Visualization: "xor_predictions",
		Sources:       []string{"experiments/xor_training/xor_training.zig", "src/network.zig", "src/layer.zig"},
		Parameters: []ParameterSpec{
			integerParameter("epochs", "Epochs", "Complete passes over the four XOR examples.", 10_000, 100, 20_000, 100, "--epochs"),
			numberParameter("learning_rate", "Learning rate", "Size of each gradient update.", 0.3, 0.001, 1, 0.001, "--learning-rate"),
			integerParameter("seed", "Seed", "Reproduces initialization and shuffling.", 42, 0, math.MaxUint32, 1, "--seed"),
		},
		Metrics:        []MetricSpec{{Name: "loss", Label: "Training loss"}},
		Backends:       []string{"cpu"},
		DefaultBackend: "cpu",
	},
	{
		ID:          "regression",
		Category:    "Foundations",
		Title:       "Approximating a Curve",
		Description: "See a ReLU network approximate the nonlinear function f(x) = x² sin(x).",
		Question:    "How does a network made from simple piecewise-linear units approximate a smooth nonlinear function?",
		Observe:     []string{"How quickly the prediction reaches the central region", "Whether errors remain near the range edges", "The relationship between loss and curve shape"},
		Interpretation: []string{
			"The predicted curve is assembled from many locally linear ReLU responses rather than storing the formula itself.",
			"Large edge errors are a form of underfitting: the network has less useful evidence or capacity in those regions.",
		},
		Visualization: "regression_curve",
		Sources:       []string{"experiments/regression/regression.zig", "src/network.zig", "src/activation.zig"},
		Parameters: []ParameterSpec{
			integerParameter("epochs", "Epochs", "Complete passes over the generated samples.", 500, 10, 2_000, 10, "--epochs"),
			numberParameter("learning_rate", "Learning rate", "Size of each gradient update.", 0.001, 0.00001, 0.1, 0.00001, "--learning-rate"),
			integerParameter("seed", "Seed", "Reproduces the dataset and initialization.", 42, 0, math.MaxUint32, 1, "--seed"),
		},
		Metrics:        []MetricSpec{{Name: "loss", Label: "Training loss"}},
		Backends:       []string{"cpu"},
		DefaultBackend: "cpu",
	},
	{
		ID:          "binary-classification",
		Category:    "Foundations",
		Title:       "Drawing a Decision Boundary",
		Description: "Watch a classifier learn that points inside a circle belong to a different class.",
		Question:    "How do layers of neurons bend a simple input plane into a nonlinear classification boundary?",
		Observe:     []string{"Where predicted confidence changes", "How the 0.5 contour approaches the target circle", "Whether low loss also produces a useful boundary"},
		Interpretation: []string{
			"The probability field shows confidence, while the transition around 0.5 is the actual decision boundary.",
			"A sharp but misplaced boundary can still be confidently wrong; compare it with both the samples and the target circle.",
		},
		Visualization: "decision_boundary",
		Sources:       []string{"experiments/binary_classification/binary_classification.zig", "src/network.zig", "src/layer.zig"},
		Parameters: []ParameterSpec{
			integerParameter("epochs", "Epochs", "Complete passes over the generated points.", 1_000, 10, 5_000, 10, "--epochs"),
			numberParameter("learning_rate", "Learning rate", "Size of each gradient update.", 0.001, 0.00001, 0.1, 0.00001, "--learning-rate"),
			integerParameter("seed", "Seed", "Reproduces the dataset and initialization.", 42, 0, math.MaxUint32, 1, "--seed"),
		},
		Metrics:        []MetricSpec{{Name: "loss", Label: "Training loss"}},
		Backends:       []string{"cpu"},
		DefaultBackend: "cpu",
	},
	{
		ID:          "spectral-learning",
		Category:    "Spectral methods",
		Title:       "Learning in Frequency Space",
		Description: "Compare a coordinate MLP with the same hidden network fed explicit Fourier features while both learn a signal with low, middle, and high frequencies.",
		Question:    "Why does a coordinate network learn smooth structure before fine detail, and how do Fourier features change that behavior?",
		Observe: []string{
			"Whether the raw model's low-frequency error falls before its high-frequency error",
			"How explicit harmonic inputs change the learned curve and amplitude spectrum",
			"Whether falling pointwise loss also recovers the intended frequency components",
		},
		Interpretation: []string{
			"The spectrum separates the learned function into frequency bins, making progress on broad shape and fine oscillation visible independently.",
			"The models share their hidden widths, optimizer, samples, and update count, but Fourier encoding widens the first layer; this is a representation comparison rather than a parameter-matched benchmark.",
		},
		Visualization: "spectral_learning",
		Sources:       []string{"experiments/spectral_learning/spectral_learning.zig", "src/spectral.zig", "src/network.zig"},
		Parameters: []ParameterSpec{
			integerParameter("steps", "Steps", "Full-batch updates applied to both models.", 1_000, 100, 3_000, 100, "--steps"),
			numberParameter("learning_rate", "Learning rate", "Shared SGD step size for both models.", 0.01, 0.0001, 0.1, 0.0001, "--learning-rate"),
			integerParameter("fourier_bands", "Fourier bands", "Sine/cosine input pairs supplied to the encoded model.", 9, 1, 16, 1, "--fourier-bands"),
			integerParameter("seed", "Seed", "Reproduces both model initializations.", 42, 0, math.MaxUint32, 1, "--seed"),
		},
		Metrics: []MetricSpec{
			{Name: "loss", Label: "Training loss"},
			{Name: "harmonic_error", Label: "Target harmonic amplitude error"},
		},
		Backends:       []string{"cpu"},
		DefaultBackend: "cpu",
	},
	{
		ID:          "optimizer-lab",
		Category:    "Training",
		Title:       "Comparing Optimizers",
		Description: "Train identical two-moons classifiers with SGD, momentum, and AdamW while keeping their updates on one selected backend.",
		Question:    "How do optimizer state and update rules change the path a network takes toward the same decision boundary?",
		Observe:     []string{"How quickly each loss series falls", "Whether optimizer speed also improves held-out accuracy", "Which work remains device-resident between reports"},
		Interpretation: []string{
			"The models start with identical parameters, so their different paths come from the optimizer updates rather than initialization.",
			"Telemetry covers the training interval before each report; evaluation readbacks are deliberately kept outside that interval.",
		},
		Visualization: "optimizer_comparison",
		Sources:       []string{"experiments/optimizer_lab/optimizer_lab.zig", "src/training.zig", "src/modules.zig"},
		Parameters: []ParameterSpec{
			integerParameter("steps", "Steps", "Full-batch updates applied to every optimizer.", 200, 20, 600, 10, "--steps"),
			integerParameter("seed", "Seed", "Reproduces the dataset and shared initialization.", 42, 0, math.MaxUint32, 1, "--seed"),
		},
		Metrics: []MetricSpec{
			{Name: "loss", Label: "Training loss"},
			{Name: "accuracy", Label: "Held-out accuracy"},
		},
		Backends:       []string{"cpu", "metal", "cuda", "rocm"},
		DefaultBackend: "cpu",
		BackendFlag:    true,
	},
	{
		ID:          "gpu-benchmark",
		Category:    "Accelerators",
		Title:       "When a GPU Wins",
		Description: "Compare complete CPU and GPU matrix multiplications at several sizes, including synchronization and a numerical agreement check.",
		Question:    "When does enough parallel work offset the cost of preparing and synchronizing a GPU operation?",
		Observe:     []string{"Whether small matrices favor CPU execution", "Where the timing curves cross", "How many transfers, kernels, and synchronizations the GPU performs"},
		Interpretation: []string{
			"A GPU is not automatically faster: launch and synchronization overhead can dominate small operations.",
			"Speedup is meaningful only alongside the sampled numerical error and the exact backend selected by the native process.",
		},
		Visualization:  "backend_benchmark",
		Sources:        []string{"experiments/gpu_benchmark/gpu_benchmark.zig", "src/backend.zig"},
		Metrics:        []MetricSpec{},
		Backends:       []string{"metal", "cuda", "rocm"},
		DefaultBackend: "metal",
		Optimize:       "ReleaseFast",
	},
	{
		ID:          "semantic-search",
		Category:    "Language",
		Title:       "Learning Semantic Search",
		Description: "Watch paired query and document encoders turn a dense similarity grid into useful retrieval rankings.",
		Question:    "How do in-batch negatives teach related texts to become closer than every mismatched pair?",
		Observe:     []string{"Whether the correct diagonal becomes dominant", "How recall and reciprocal rank respond to falling loss", "Where embeddings cross from device training to host ranking"},
		Interpretation: []string{
			"Each row compares one held-out query with every document; a strong diagonal means the correct document outranks the mismatches.",
			"Encoder training is backend-neutral, while final top-k ranking intentionally happens after one explicit embedding readback.",
		},
		Visualization: "semantic_similarity",
		Sources:       []string{"experiments/semantic_search/semantic_search.zig", "src/retrieval.zig", "src/embeddings.zig"},
		Parameters: []ParameterSpec{
			integerParameter("steps", "Steps", "Full-batch contrastive updates.", 40, 10, 200, 10, "--steps"),
			integerParameter("seed", "Seed", "Reproduces both encoder initializations.", 42, 0, math.MaxUint32, 1, "--seed"),
		},
		Metrics: []MetricSpec{
			{Name: "loss", Label: "InfoNCE loss"},
			{Name: "recall_at_one", Label: "Recall at one"},
			{Name: "mean_reciprocal_rank", Label: "Mean reciprocal rank"},
		},
		Backends:       []string{"cpu", "metal", "cuda", "rocm"},
		DefaultBackend: "cpu",
		BackendFlag:    true,
	},
}

func integerParameter(name, label, help string, defaultValue, minValue, maxValue, step float64, flag string) ParameterSpec {
	return ParameterSpec{Name: name, Label: label, Help: help, Kind: ParameterInteger, Default: defaultValue, Min: minValue, Max: maxValue, Step: step, Flag: flag}
}

func numberParameter(name, label, help string, defaultValue, minValue, maxValue, step float64, flag string) ParameterSpec {
	return ParameterSpec{Name: name, Label: label, Help: help, Kind: ParameterNumber, Default: defaultValue, Min: minValue, Max: maxValue, Step: step, Flag: flag}
}

func Experiments() []ExperimentSpec {
	specs := make([]ExperimentSpec, len(learningSpecs))
	for i := range learningSpecs {
		specs[i] = learningSpecs[i]
		if specs[i].Parameters == nil {
			specs[i].Parameters = []ParameterSpec{}
		}
		if specs[i].Metrics == nil {
			specs[i].Metrics = []MetricSpec{}
		}
		experiment, ok := catalog.ResolveExperiment(specs[i].ID)
		if ok {
			specs[i].Step = experiment.Step
		}
	}
	return specs
}

func ResolveExperiment(id string) (ExperimentSpec, bool) {
	for _, spec := range Experiments() {
		if spec.ID == catalog.NormalizeName(id) {
			return spec, true
		}
	}
	return ExperimentSpec{}, false
}

type RunTarget string

const (
	RunTargetLocal RunTarget = "local"
	RunTargetCloud RunTarget = "cloud"
)

type RunOptions struct {
	Backend          string
	Arguments        []string
	Target           RunTarget
	WorkerID         string
	AcknowledgeDirty bool
}

func BuildRunOptions(spec ExperimentSpec, backend string, values map[string]json.Number) (RunOptions, error) {
	if backend == "" {
		backend = spec.DefaultBackend
	}
	if !contains(spec.Backends, backend) {
		return RunOptions{}, fmt.Errorf("backend %q is not supported by %s", backend, spec.ID)
	}
	parameters := make(map[string]ParameterSpec, len(spec.Parameters))
	for _, parameter := range spec.Parameters {
		parameters[parameter.Name] = parameter
	}
	for name := range values {
		if _, ok := parameters[name]; !ok {
			return RunOptions{}, fmt.Errorf("unknown parameter %q", name)
		}
	}

	args := []string{"--format", "ndjson"}
	if spec.BackendFlag {
		args = append(args, "--backend", backend)
	}
	for _, parameter := range spec.Parameters {
		value := parameter.Default
		if supplied, ok := values[parameter.Name]; ok {
			parsed, err := supplied.Float64()
			if err != nil || math.IsNaN(parsed) || math.IsInf(parsed, 0) {
				return RunOptions{}, fmt.Errorf("parameter %s must be a finite number", parameter.Name)
			}
			value = parsed
		}
		if value < parameter.Min || value > parameter.Max {
			return RunOptions{}, fmt.Errorf("parameter %s must be between %s and %s", parameter.Name, formatNumber(parameter.Min), formatNumber(parameter.Max))
		}
		if parameter.Kind == ParameterInteger && math.Trunc(value) != value {
			return RunOptions{}, fmt.Errorf("parameter %s must be an integer", parameter.Name)
		}
		args = append(args, parameter.Flag, formatNumber(value))
	}
	return RunOptions{Backend: backend, Arguments: args, Target: RunTargetLocal}, nil
}

func contains(values []string, wanted string) bool {
	for _, value := range values {
		if value == wanted {
			return true
		}
	}
	return false
}

func formatNumber(value float64) string {
	return strconv.FormatFloat(value, 'f', -1, 64)
}
