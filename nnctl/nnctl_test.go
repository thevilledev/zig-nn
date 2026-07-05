package nnctl

import (
	"reflect"
	"testing"
)

func TestResolveExampleAliases(t *testing.T) {
	example, ok := resolveExample("example-network-visualization")
	if !ok {
		t.Fatal("expected alias to resolve")
	}
	if example.Step != "run_network_visualisation" {
		t.Fatalf("unexpected step: %s", example.Step)
	}
}

func TestZigArgs(t *testing.T) {
	got := zigArgs("examples", zigOptions{
		optimize: "ReleaseFast",
		gpu:      "metal",
		summary:  "all",
	})
	want := []string{"build", "examples", "-Dgpu=metal", "-Doptimize=ReleaseFast", "--summary", "all"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("zigArgs() = %#v, want %#v", got, want)
	}
}

func TestZigRunArgsPassesArgumentsAfterSeparator(t *testing.T) {
	got := zigRunArgs("run_tiny_gpt", zigOptions{optimize: "Debug"}, []string{"--prompt", "to be"})
	want := []string{"build", "run_tiny_gpt", "-Doptimize=Debug", "--", "--prompt", "to be"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("zigRunArgs() = %#v, want %#v", got, want)
	}
}

func TestSplitPassthrough(t *testing.T) {
	before, after := splitPassthrough([]string{"--mode", "Debug", "tiny-gpt", "--", "--tokens", "80"})
	if !reflect.DeepEqual(before, []string{"--mode", "Debug", "tiny-gpt"}) {
		t.Fatalf("before = %#v", before)
	}
	if !reflect.DeepEqual(after, []string{"--tokens", "80"}) {
		t.Fatalf("after = %#v", after)
	}
}

func TestReorderInterspersed(t *testing.T) {
	got, err := reorderInterspersed([]string{"gpu", "--gpu", "metal", "--mode=ReleaseFast"}, map[string]bool{
		"gpu":  true,
		"mode": true,
	})
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"--gpu", "metal", "--mode=ReleaseFast", "gpu"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("reorderInterspersed() = %#v, want %#v", got, want)
	}
}
