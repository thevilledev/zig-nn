package cli

import "testing"

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
