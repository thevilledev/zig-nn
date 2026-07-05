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
