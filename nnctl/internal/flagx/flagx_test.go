package flagx

import (
	"reflect"
	"testing"
)

func TestSplitPassthrough(t *testing.T) {
	before, after := SplitPassthrough([]string{"--mode", "Debug", "tiny-gpt", "--", "--tokens", "80"})
	if !reflect.DeepEqual(before, []string{"--mode", "Debug", "tiny-gpt"}) {
		t.Fatalf("before = %#v", before)
	}
	if !reflect.DeepEqual(after, []string{"--tokens", "80"}) {
		t.Fatalf("after = %#v", after)
	}
}

func TestReorderInterspersed(t *testing.T) {
	got, err := ReorderInterspersed([]string{"gpu", "--gpu", "metal", "--mode=ReleaseFast"}, map[string]bool{
		"gpu":  true,
		"mode": true,
	})
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"--gpu", "metal", "--mode=ReleaseFast", "gpu"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("ReorderInterspersed() = %#v, want %#v", got, want)
	}
}
