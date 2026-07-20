package cli

import (
	"bufio"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"nnctl/internal/localhttp"
	"nnctl/internal/zig"
)

type checkpointKind string

const (
	checkpointDense   checkpointKind = "dense_network"
	checkpointTinyGPT checkpointKind = "tiny_gpt"
)

type checkpointInfo struct {
	Kind           checkpointKind `json:"kind"`
	Version        uint32         `json:"version"`
	InputSize      uint32         `json:"input_size,omitempty"`
	OutputSize     uint32         `json:"output_size,omitempty"`
	Layers         uint32         `json:"layers,omitempty"`
	BlockSize      uint32         `json:"block_size,omitempty"`
	Heads          uint32         `json:"heads,omitempty"`
	Channels       uint32         `json:"channels,omitempty"`
	VocabularySize uint32         `json:"vocabulary_size,omitempty"`
}

type modelInspectOptions struct {
	json bool
}

type predictOptions struct {
	mode      string
	backend   string
	modelPath string
	inputJSON string
	batchSize int
}

type serveOptions struct {
	mode      string
	backend   string
	modelPath string
	modelName string
	host      string
	port      int
}

func defaultPredictOptions() predictOptions {
	return predictOptions{mode: defaultReleaseMode, backend: "auto", batchSize: 1}
}

func defaultServeOptions() serveOptions {
	return serveOptions{mode: defaultReleaseMode, backend: "auto", modelName: "tiny-gpt-zig", host: "127.0.0.1", port: 8080}
}

func (a *app) runModelInspect(path string, opts modelInspectOptions) error {
	resolved := resolveModelPath(a.repoRoot, path)
	info, err := inspectCheckpoint(resolved)
	if err != nil {
		return err
	}
	if opts.json {
		encoder := json.NewEncoder(a.stdout())
		encoder.SetEscapeHTML(false)
		if err := encoder.Encode(info); err != nil {
			return fmt.Errorf("write model information: %w", err)
		}
		return nil
	}
	output := newErrorWriter(a.stdout())
	output.printf("kind: %s\n", info.Kind)
	output.printf("version: %d\n", info.Version)
	switch info.Kind {
	case checkpointDense:
		output.printf("input size: %d\n", info.InputSize)
		output.printf("output size: %d\n", info.OutputSize)
		output.printf("layers: %d\n", info.Layers)
	case checkpointTinyGPT:
		output.printf("block size: %d\n", info.BlockSize)
		output.printf("layers: %d\n", info.Layers)
		output.printf("heads: %d\n", info.Heads)
		output.printf("channels: %d\n", info.Channels)
		output.printf("vocabulary size: %d\n", info.VocabularySize)
	}
	return output.Err()
}

func (a *app) runPredict(ctx context.Context, opts predictOptions) error {
	if err := validateInferenceBackend(opts.backend); err != nil {
		return err
	}
	if opts.modelPath == "" {
		return fmt.Errorf("predict requires --model")
	}
	if opts.inputJSON == "" {
		return fmt.Errorf("predict requires --input")
	}
	if opts.batchSize <= 0 {
		return fmt.Errorf("--batch-size must be positive")
	}
	var values []float32
	if err := json.Unmarshal([]byte(opts.inputJSON), &values); err != nil {
		return fmt.Errorf("--input must be a JSON number array: %w", err)
	}
	if len(values) == 0 {
		return fmt.Errorf("--input must not be empty")
	}
	args := []string{
		"--model", opts.modelPath,
		"--input", opts.inputJSON,
		"--batch-size", fmt.Sprint(opts.batchSize),
		"--backend", runtimeBackend(opts.backend),
	}
	return a.runZig(ctx, zig.RunArgs("run_inference_predict", zig.Options{
		Optimize: opts.mode,
		GPU:      buildBackend(opts.backend),
	}, args)...)
}

func (a *app) runServe(ctx context.Context, opts serveOptions) error {
	if err := validateInferenceBackend(opts.backend); err != nil {
		return err
	}
	if opts.modelPath == "" {
		return fmt.Errorf("serve requires --model")
	}
	if !localhttp.IsLoopbackHost(opts.host) {
		return fmt.Errorf("--host must be a loopback address; use an SSH tunnel for remote access")
	}
	if opts.port <= 0 || opts.port > 65535 {
		return fmt.Errorf("--port must be between 1 and 65535")
	}
	info, err := inspectCheckpoint(resolveModelPath(a.repoRoot, opts.modelPath))
	if err != nil {
		return err
	}
	step := "run_serving"
	args := []string{"--model", opts.modelPath, "--host", opts.host, "--port", fmt.Sprint(opts.port), "--backend", runtimeBackend(opts.backend)}
	if info.Kind == checkpointTinyGPT {
		step = "run_tiny_gpt_openai"
		args = append(args, "--model-name", opts.modelName)
	}
	return a.runZig(ctx, zig.RunArgs(step, zig.Options{Optimize: opts.mode, GPU: buildBackend(opts.backend)}, args)...)
}

func inspectCheckpoint(path string) (checkpointInfo, error) {
	file, err := os.Open(path)
	if err != nil {
		return checkpointInfo{}, fmt.Errorf("open model: %w", err)
	}
	defer func() { _ = file.Close() }()
	reader := bufio.NewReader(file)
	var magic [4]byte
	if _, err := io.ReadFull(reader, magic[:]); err != nil {
		return checkpointInfo{}, fmt.Errorf("read model header: %w", err)
	}
	var version uint32
	if err := binary.Read(reader, binary.LittleEndian, &version); err != nil {
		return checkpointInfo{}, fmt.Errorf("read model version: %w", err)
	}
	switch string(magic[:]) {
	case "ZNN\x00":
		if version < 1 || version > 3 {
			return checkpointInfo{}, fmt.Errorf("unsupported ZNN version %d", version)
		}
		if _, err := io.CopyN(io.Discard, reader, 8); err != nil {
			return checkpointInfo{}, fmt.Errorf("read ZNN timestamp: %w", err)
		}
		if version >= 2 {
			if _, err := io.CopyN(io.Discard, reader, 9); err != nil {
				return checkpointInfo{}, fmt.Errorf("read ZNN training metadata: %w", err)
			}
		}
		values := [3]uint32{}
		if err := binary.Read(reader, binary.LittleEndian, &values); err != nil {
			return checkpointInfo{}, fmt.Errorf("read ZNN dimensions: %w", err)
		}
		return checkpointInfo{Kind: checkpointDense, Version: version, InputSize: values[0], OutputSize: values[1], Layers: values[2]}, nil
	case "TGPT":
		if version < 1 || version > 3 {
			return checkpointInfo{}, fmt.Errorf("unsupported TGPT version %d", version)
		}
		values := [5]uint32{}
		if err := binary.Read(reader, binary.LittleEndian, &values); err != nil {
			return checkpointInfo{}, fmt.Errorf("read TGPT dimensions: %w", err)
		}
		return checkpointInfo{
			Kind: checkpointTinyGPT, Version: version, BlockSize: values[0], Layers: values[1],
			Heads: values[2], Channels: values[3], VocabularySize: values[4],
		}, nil
	default:
		return checkpointInfo{}, fmt.Errorf("unsupported model format %q", strings.TrimRight(string(magic[:]), "\x00"))
	}
}

func resolveModelPath(repoRoot, path string) string {
	if filepath.IsAbs(path) {
		return path
	}
	return filepath.Join(repoRoot, path)
}

func validateInferenceBackend(value string) error {
	switch value {
	case "none", "cpu", "auto", "metal", "cuda", "rocm":
		return nil
	default:
		return fmt.Errorf("unsupported backend %q", value)
	}
}

func buildBackend(value string) string {
	if value == "cpu" {
		return "none"
	}
	return value
}

func runtimeBackend(value string) string {
	if value == "none" {
		return "cpu"
	}
	return value
}
