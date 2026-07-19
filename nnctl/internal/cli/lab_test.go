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

func TestParseLabCloudOfferings(t *testing.T) {
	parsed, err := parseLabCloudOfferings(
		[]string{"digitalocean:gpu-mi325x1-256gb-contracted@nyc3"},
		[]string{"digitalocean", "verda"},
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(parsed["digitalocean"]) != 1 || parsed["digitalocean"][0] != "gpu-mi325x1-256gb-contracted@nyc3" {
		t.Fatalf("parsed offerings = %#v", parsed)
	}
	_, err = parseLabCloudOfferings([]string{"missing:size@region"}, []string{"verda"})
	if err == nil || !strings.Contains(err.Error(), "not enabled") {
		t.Fatalf("disabled provider error = %v", err)
	}
}

func TestParseLabCloudSSHKeys(t *testing.T) {
	parsed, err := parseLabCloudSSHKeys([]string{"digitalocean:17"}, []string{"digitalocean"})
	if err != nil {
		t.Fatal(err)
	}
	if len(parsed["digitalocean"]) != 1 || parsed["digitalocean"][0] != "17" {
		t.Fatalf("parsed SSH keys = %#v", parsed)
	}
	_, err = parseLabCloudSSHKeys([]string{"missing:17"}, []string{"verda"})
	if err == nil || !strings.Contains(err.Error(), "not enabled") {
		t.Fatalf("disabled provider error = %v", err)
	}
}
