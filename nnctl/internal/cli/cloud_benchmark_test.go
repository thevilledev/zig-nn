package cli

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	verdacloud "nnctl/internal/cloud/verda"
)

func TestNormalizeCloudBenchmarkDeployOptionsDefaultsSourceName(t *testing.T) {
	opts := defaultCloudBenchmarkDeployOptions()
	opts.InstanceType = "1A100.22V"

	if err := normalizeCloudBenchmarkDeployOptions(&opts); err != nil {
		t.Fatalf("normalizeCloudBenchmarkDeployOptions() error = %v", err)
	}
	if opts.SourceOSVolumeName != defaultCloudBenchmarkSourceName {
		t.Fatalf("SourceOSVolumeName = %q, want %q", opts.SourceOSVolumeName, defaultCloudBenchmarkSourceName)
	}
	if opts.Market != verdacloud.PricingMarketSpot || opts.backend != "cuda" {
		t.Fatalf("unexpected defaults: market=%q backend=%q", opts.Market, opts.backend)
	}
	if opts.Image != defaultCloudBenchmarkPackerImage {
		t.Fatalf("Image = %q, want %q", opts.Image, defaultCloudBenchmarkPackerImage)
	}
}

func TestNormalizeCloudBenchmarkDeployOptionsAllowsSourceID(t *testing.T) {
	opts := defaultCloudBenchmarkDeployOptions()
	opts.InstanceType = "1A100.22V"
	opts.SourceOSVolumeID = "vol-golden"

	if err := normalizeCloudBenchmarkDeployOptions(&opts); err != nil {
		t.Fatalf("normalizeCloudBenchmarkDeployOptions() error = %v", err)
	}
	if opts.SourceOSVolumeName != "" {
		t.Fatalf("SourceOSVolumeName = %q, want empty", opts.SourceOSVolumeName)
	}
}

func TestReusableCloudBenchmarkSourceVolumeUsesNewestActiveOSVolume(t *testing.T) {
	volumes := []verdacloud.Volume{
		{ID: "vol-old", Name: defaultCloudBenchmarkSourceName, Location: "FIN-01", IsOSVolume: true, CreatedAt: "2026-07-10T00:00:00Z"},
		{ID: "vol-new", Name: defaultCloudBenchmarkSourceName, Location: "fin-01", IsOSVolume: true, CreatedAt: "2026-07-15T00:00:00Z"},
		{ID: "vol-data", Name: defaultCloudBenchmarkSourceName, Location: "FIN-01", CreatedAt: "2026-07-16T00:00:00Z"},
		{ID: "vol-deleted", Name: defaultCloudBenchmarkSourceName, Location: "FIN-01", IsOSVolume: true, Status: "deleted", CreatedAt: "2026-07-17T00:00:00Z"},
	}

	got, ok := reusableCloudBenchmarkSourceVolume(volumes, defaultCloudBenchmarkSourceName, "FIN-01")
	if !ok {
		t.Fatal("reusableCloudBenchmarkSourceVolume() did not find a volume")
	}
	if got.ID != "vol-new" || got.Location != "FIN-01" {
		t.Fatalf("volume = %#v, want newest active OS volume", got)
	}
}

func TestEnsureCloudPackerTemplatePreservesExistingFilesUnlessRefreshed(t *testing.T) {
	dir := t.TempDir()
	templatePath := filepath.Join(dir, "ubuntu.pkr.hcl")
	if err := os.WriteFile(templatePath, []byte("custom template\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	written, err := ensureCloudPackerTemplate(dir, false)
	if err != nil {
		t.Fatalf("ensureCloudPackerTemplate() error = %v", err)
	}
	if !reflect.DeepEqual(written, []string{filepath.Join(dir, "bootstrap.sh")}) {
		t.Fatalf("written = %#v", written)
	}
	content, err := os.ReadFile(templatePath)
	if err != nil {
		t.Fatal(err)
	}
	if string(content) != "custom template\n" {
		t.Fatalf("existing template was overwritten: %q", content)
	}

	written, err = ensureCloudPackerTemplate(dir, true)
	if err != nil {
		t.Fatalf("ensureCloudPackerTemplate(force) error = %v", err)
	}
	if !reflect.DeepEqual(written, []string{templatePath}) {
		t.Fatalf("force written = %#v", written)
	}
	content, err = os.ReadFile(templatePath)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(content), `variable "artifact_volume_name"`) {
		t.Fatalf("refreshed template missing configurable artifact name:\n%s", content)
	}
}

func TestCloudBenchmarkPackerBuildArgsConfigureMissingVolume(t *testing.T) {
	dir := t.TempDir()
	if _, err := ensureCloudPackerTemplate(dir, false); err != nil {
		t.Fatal(err)
	}
	opts := defaultCloudBenchmarkDeployOptions()
	opts.InstanceType = "1H200.141S.44V"
	opts.SourceOSVolumeName = "zig-nn-h200-golden"
	opts.Market = verdacloud.PricingMarketOnDemand

	got, err := cloudBenchmarkPackerBuildArgs(filepath.Join(dir, "ubuntu.pkr.hcl"), "/tmp/keys", opts, "FIN-03")
	if err != nil {
		t.Fatalf("cloudBenchmarkPackerBuildArgs() error = %v", err)
	}
	joined := strings.Join(got, "\n")
	for _, want := range []string{
		"authorized_keys_file=/tmp/keys",
		"artifact_volume_name=zig-nn-h200-golden",
		"build_location_code=FIN-03",
		"build_market=on-demand",
		`artifact_volume_location_codes=["FIN-03"]`,
	} {
		if !strings.Contains(joined, want) {
			t.Fatalf("Packer args missing %q:\n%#v", want, got)
		}
	}
}

func TestRemoteBenchmarkCommandQuotesRemoteInputs(t *testing.T) {
	got := remoteBenchmarkCommand("/tmp/bench dir", "cuda", "gpu_peak'; touch /tmp/bad; echo '", false)
	for _, want := range []string{
		"cd '/tmp/bench dir'",
		"-Dgpu='cuda'",
		`--filter 'gpu_peak'"'"'; touch /tmp/bad; echo '"'"''`,
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("remote command missing %q:\n%s", want, got)
		}
	}
}

func TestUpdateCloudBenchmarkDocsIsIdempotent(t *testing.T) {
	dir := t.TempDir()
	docsPath := filepath.Join(dir, "docs", "benchmarks.md")
	csvPath := filepath.Join(dir, "benchmark-a100.csv")
	if err := os.MkdirAll(filepath.Dir(docsPath), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(docsPath, []byte("# Benchmarks\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	metadata := cloudBenchmarkMetadata{
		Timestamp:    time.Date(2026, 7, 15, 12, 0, 0, 0, time.UTC),
		Commit:       "abc123",
		InstanceType: "1A100.22V",
		Location:     "FIN-01",
		Market:       "spot",
		Backend:      "cuda",
		GPU:          "NVIDIA A100",
	}
	rows := []benchmarkRow{
		{Suite: "matmul", Case: "64x64x64", Backend: "cuda", AverageNS: 12345, Status: "ok"},
		{Suite: "matmul", Case: "64x64x64", Backend: "cpu", AverageNS: 50000, Status: "ok"},
	}

	if err := updateCloudBenchmarkDocs(docsPath, csvPath, metadata, rows); err != nil {
		t.Fatalf("updateCloudBenchmarkDocs() error = %v", err)
	}
	if err := updateCloudBenchmarkDocs(docsPath, csvPath, metadata, rows); err != nil {
		t.Fatalf("second updateCloudBenchmarkDocs() error = %v", err)
	}
	content, err := os.ReadFile(docsPath)
	if err != nil {
		t.Fatal(err)
	}
	text := string(content)
	if strings.Count(text, "nnctl-cloud-benchmark:benchmark-a100.csv:start") != 1 {
		t.Fatalf("generated block was duplicated:\n%s", text)
	}
	for _, want := range []string{"## Automated Cloud Benchmark Runs", "NVIDIA A100", "12.345 us", "../benchmark-a100.csv"} {
		if !strings.Contains(text, want) {
			t.Fatalf("docs missing %q:\n%s", want, text)
		}
	}
	if strings.Contains(text, "50.000 us") {
		t.Fatalf("docs included non-selected backend:\n%s", text)
	}
}
