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

type ExperimentSpec struct {
	ID             string          `json:"id"`
	Title          string          `json:"title"`
	Description    string          `json:"description"`
	Question       string          `json:"question"`
	Observe        []string        `json:"observe"`
	Interpretation []string        `json:"interpretation"`
	Visualization  string          `json:"visualization"`
	Sources        []string        `json:"sources"`
	Parameters     []ParameterSpec `json:"parameters"`
	Step           string          `json:"-"`
}

var learningSpecs = []ExperimentSpec{
	{
		ID:          "xor-training",
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
	},
	{
		ID:          "regression",
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
	},
	{
		ID:          "binary-classification",
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

func BuildArguments(spec ExperimentSpec, values map[string]json.Number) ([]string, error) {
	parameters := make(map[string]ParameterSpec, len(spec.Parameters))
	for _, parameter := range spec.Parameters {
		parameters[parameter.Name] = parameter
	}
	for name := range values {
		if _, ok := parameters[name]; !ok {
			return nil, fmt.Errorf("unknown parameter %q", name)
		}
	}

	args := []string{"--format", "ndjson"}
	for _, parameter := range spec.Parameters {
		value := parameter.Default
		if supplied, ok := values[parameter.Name]; ok {
			parsed, err := supplied.Float64()
			if err != nil || math.IsNaN(parsed) || math.IsInf(parsed, 0) {
				return nil, fmt.Errorf("parameter %s must be a finite number", parameter.Name)
			}
			value = parsed
		}
		if value < parameter.Min || value > parameter.Max {
			return nil, fmt.Errorf("parameter %s must be between %s and %s", parameter.Name, formatNumber(parameter.Min), formatNumber(parameter.Max))
		}
		if parameter.Kind == ParameterInteger && math.Trunc(value) != value {
			return nil, fmt.Errorf("parameter %s must be an integer", parameter.Name)
		}
		args = append(args, parameter.Flag, formatNumber(value))
	}
	return args, nil
}

func formatNumber(value float64) string {
	return strconv.FormatFloat(value, 'f', -1, 64)
}
