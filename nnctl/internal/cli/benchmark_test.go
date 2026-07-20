package cli

import (
	"bytes"
	"errors"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestParseBenchmarkCSVSkipsComments(t *testing.T) {
	rows, err := parseBenchmarkCSV([]byte(`# zig-nn benchmark suite
# mode=ReleaseFast quick=true metal_available=true
suite,case,backend,mode,warmups,iterations,avg_ns,min_ns,max_ns,checksum,sample_error,status
matmul,64x64x64,cpu,ReleaseFast,1,1,1000000,900000,1100000,0.1,0,ok
matmul,64x64x64,metal,ReleaseFast,1,1,500000,450000,550000,0.1,0.000000014,ok
training,batch,cpu,ReleaseFast,1,1,2000000,2000000,2000000,0.2,0,ok
training,batch,metal,ReleaseFast,1,1,0,0,0,0,0,skipped_cpu_only_path
training,batch,rocm,ReleaseFast,1,1,0,0,0,0,0,skipped_cpu_only_path
`))
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 5 {
		t.Fatalf("len(rows) = %d, want 5", len(rows))
	}
	if rows[1].Backend != "metal" || rows[1].AverageNS != 500000 || rows[1].SampleError == 0 {
		t.Fatalf("unexpected parsed metal row: %#v", rows[1])
	}
}

func TestParseBenchmarkCSVAcceptsWorkUnitsAndLegacyRows(t *testing.T) {
	rows, err := parseBenchmarkCSV([]byte(`suite,case,backend,mode,warmups,iterations,avg_ns,min_ns,max_ns,checksum,sample_error,work_units,work_unit,status
inference,tiny_decode,cpu,ReleaseFast,1,2,500000000,490000000,510000000,1,0,32,token,ok
`))
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 1 || rows[0].WorkUnits != 32 || rows[0].WorkUnit != "token" {
		t.Fatalf("unexpected work-unit row: %#v", rows)
	}
	if got := formatBenchmarkTime(&rows[0]); !strings.Contains(got, "64.0 token/s") {
		t.Fatalf("formatBenchmarkTime() = %q", got)
	}
}

func TestParseBenchmarkCSVAcceptsInferenceTelemetry(t *testing.T) {
	rows, err := parseBenchmarkCSV([]byte(`suite,case,backend,mode,warmups,iterations,avg_ns,min_ns,max_ns,checksum,sample_error,work_units,work_unit,allocations,live_buffers,h2d_transfers,d2h_transfers,kernel_launches,vendor_gemm_launches,synchronizations,status
inference,tiny_decode,cuda,ReleaseFast,1,2,500000,490000,510000,1,0,1,token,8,42,0,1,20,7,1,ok
`))
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 1 || rows[0].Allocations != 8 || rows[0].LiveBuffers != 42 || rows[0].VendorGEMMLaunches != 7 || rows[0].Synchronizations != 1 {
		t.Fatalf("unexpected telemetry row: %#v", rows)
	}
	if got := formatBenchmarkTelemetry(&rows[0]); !strings.Contains(got, "gemm=7") {
		t.Fatalf("formatBenchmarkTelemetry() = %q", got)
	}
}

func TestCommittedInferenceBaselinesMeetReleaseGate(t *testing.T) {
	root := filepath.Join("..", "..", "..")
	baseline := readBenchmarkFixture(t, filepath.Join(root, "benchmarks", "baselines", "inference-apple-m1-v0.0.6.csv"))
	optimized := readBenchmarkFixture(t, filepath.Join(root, "benchmarks", "baselines", "inference-apple-m1-v0.0.7.csv"))
	baselineByKey := make(map[string]benchmarkRow, len(baseline))
	for _, row := range baseline {
		baselineByKey[benchmarkRowKey(row)] = row
	}

	for _, current := range optimized {
		previous, ok := baselineByKey[benchmarkRowKey(current)]
		if !ok {
			t.Fatalf("optimized row has no baseline: %s", benchmarkRowKey(current))
		}
		if current.AverageNS > previous.AverageNS*1.05 {
			t.Errorf("%s regressed by more than 5%%: %.0f -> %.0f ns", current.Case, previous.AverageNS, current.AverageNS)
		}
		if math.Abs(current.Checksum-previous.Checksum) > 1e-4 {
			t.Errorf("%s checksum changed beyond f32 tolerance", current.Case)
		}
		if current.LiveBuffers <= 0 {
			t.Errorf("%s does not record live-buffer telemetry", current.Case)
		}
	}

	decodeKey := benchmarkRowKey(benchmarkRow{Suite: "inference", Case: "tiny_cached_decode", Backend: "cpu", Mode: "ReleaseFast"})
	previous := baselineByKey[decodeKey]
	var current benchmarkRow
	for _, row := range optimized {
		if benchmarkRowKey(row) == decodeKey {
			current = row
			break
		}
	}
	if speedup := previous.AverageNS / current.AverageNS; speedup < 1.5 {
		t.Fatalf("cached decode speedup = %.2fx, want at least 1.50x", speedup)
	}
}

func readBenchmarkFixture(t *testing.T, path string) []benchmarkRow {
	t.Helper()
	contents, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	rows, err := parseBenchmarkCSV(contents)
	if err != nil {
		t.Fatal(err)
	}
	return rows
}

func TestPrintBenchmarkReportShowsDiffsAndSkips(t *testing.T) {
	rows := []benchmarkRow{
		{Suite: "matmul", Case: "64x64x64", Backend: "cpu", Mode: "ReleaseFast", AverageNS: 1_000_000, Status: "ok"},
		{Suite: "matmul", Case: "64x64x64", Backend: "metal", Mode: "ReleaseFast", AverageNS: 500_000, SampleError: 0.000000014, Status: "ok"},
		{Suite: "training", Case: "batch", Backend: "cpu", Mode: "ReleaseFast", AverageNS: 2_000_000, Status: "ok"},
		{Suite: "training", Case: "batch", Backend: "metal", Mode: "ReleaseFast", Status: "skipped_cpu_only_path"},
		{Suite: "training", Case: "batch", Backend: "rocm", Mode: "ReleaseFast", Status: "skipped_rocm_unavailable"},
	}

	var out bytes.Buffer
	if err := printBenchmarkReport(&out, rows); err != nil {
		t.Fatalf("printBenchmarkReport() error = %v", err)
	}
	report := out.String()

	for _, want := range []string{
		"Benchmark report",
		"Mode: ReleaseFast",
		"matmul",
		"64x64x64",
		"1.000 ms",
		"500.000 us",
		"-50.0%",
		"2.00x",
		"1.40e-08",
		"training",
		"metal: skipped_cpu_only_path",
		"rocm: skipped_rocm_unavailable",
	} {
		if !strings.Contains(report, want) {
			t.Fatalf("report missing %q:\n%s", want, report)
		}
	}
}

func TestPrintBenchmarkComparisonShowsBaselineDeltas(t *testing.T) {
	current := []benchmarkRow{
		{Suite: "matmul", Case: "64x64x64", Backend: "cpu", Mode: "ReleaseFast", AverageNS: 1_200_000, Status: "ok"},
		{Suite: "matmul", Case: "64x64x64", Backend: "metal", Mode: "ReleaseFast", AverageNS: 400_000, Status: "ok"},
		{Suite: "training", Case: "batch", Backend: "cuda", Mode: "ReleaseFast", Status: "skipped_cuda_unavailable"},
		{Suite: "tiny_gpt", Case: "small", Backend: "metal", Mode: "ReleaseFast", Status: "skipped_cpu_only_path"},
		{Suite: "tiny_gpt", Case: "small", Backend: "cpu", Mode: "ReleaseFast", AverageNS: 750_000, Status: "ok"},
	}
	baseline := []benchmarkRow{
		{Suite: "matmul", Case: "64x64x64", Backend: "cpu", Mode: "ReleaseFast", AverageNS: 1_000_000, Status: "ok"},
		{Suite: "matmul", Case: "64x64x64", Backend: "metal", Mode: "ReleaseFast", AverageNS: 500_000, Status: "ok"},
		{Suite: "training", Case: "batch", Backend: "cuda", Mode: "ReleaseFast", AverageNS: 2_000_000, Status: "ok"},
		{Suite: "tiny_gpt", Case: "small", Backend: "metal", Mode: "ReleaseFast", Status: "skipped_cpu_only_path"},
		{Suite: "quantization", Case: "uniform", Backend: "cpu", Mode: "ReleaseFast", AverageNS: 100_000, Status: "ok"},
	}

	var out bytes.Buffer
	if err := printBenchmarkComparison(&out, current, baseline); err != nil {
		t.Fatalf("printBenchmarkComparison() error = %v", err)
	}
	report := out.String()

	for _, want := range []string{
		"Benchmark comparison",
		"Mode: ReleaseFast",
		"matmul",
		"64x64x64",
		"+20.0%",
		"-20.0%",
		"current: skipped_cuda_unavailable",
		"tiny_gpt",
		"skipped_cpu_only_path",
		"new",
		"quantization",
		"missing_current",
	} {
		if !strings.Contains(report, want) {
			t.Fatalf("comparison missing %q:\n%s", want, report)
		}
	}
}

func TestPrintBenchmarkReportReturnsWriteFailure(t *testing.T) {
	wantErr := errors.New("report unavailable")
	err := printBenchmarkReport(testErrorWriter{err: wantErr}, []benchmarkRow{{Mode: "ReleaseFast"}})
	if !errors.Is(err, wantErr) {
		t.Fatalf("printBenchmarkReport() error = %v, want %v", err, wantErr)
	}
}
