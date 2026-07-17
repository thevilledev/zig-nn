package cli

import (
	"strings"
	"testing"
)

func TestIsLoopbackLabHost(t *testing.T) {
	tests := map[string]bool{
		"127.0.0.1": true,
		"::1":       true,
		"localhost": true,
		"0.0.0.0":   false,
		"10.0.0.2":  false,
		"example":   false,
	}
	for host, expected := range tests {
		if actual := isLoopbackLabHost(host); actual != expected {
			t.Errorf("isLoopbackLabHost(%q) = %t, want %t", host, actual, expected)
		}
	}
}

func TestRunLabRequiresLoopbackHost(t *testing.T) {
	t.Parallel()
	a := newApp(nil, nil, nil)
	opts := defaultLabOptions()
	opts.host = "0.0.0.0"
	if err := a.runLab(t.Context(), opts); err == nil || !strings.Contains(err.Error(), "loopback") {
		t.Fatalf("runLab() error = %v", err)
	}
}
