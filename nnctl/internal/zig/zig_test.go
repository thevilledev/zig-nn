package zig

import (
	"reflect"
	"testing"
)

func TestArgs(t *testing.T) {
	got := Args("examples", Options{
		Optimize: "ReleaseFast",
		GPU:      "metal",
		Summary:  "all",
	})
	want := []string{"build", "examples", "-Dgpu=metal", "-Doptimize=ReleaseFast", "--summary", "all"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Args() = %#v, want %#v", got, want)
	}
}

func TestRunArgsPassesArgumentsAfterSeparator(t *testing.T) {
	got := RunArgs("run_tiny_gpt", Options{Optimize: "Debug"}, []string{"--prompt", "to be"})
	want := []string{"build", "run_tiny_gpt", "-Doptimize=Debug", "--", "--prompt", "to be"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("RunArgs() = %#v, want %#v", got, want)
	}
}
