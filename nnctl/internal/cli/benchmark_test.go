package cli

import (
	"bytes"
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
`))
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 4 {
		t.Fatalf("len(rows) = %d, want 4", len(rows))
	}
	if rows[1].Backend != "metal" || rows[1].AverageNS != 500000 || rows[1].SampleError == 0 {
		t.Fatalf("unexpected parsed metal row: %#v", rows[1])
	}
}

func TestPrintBenchmarkReportShowsDiffsAndSkips(t *testing.T) {
	rows := []benchmarkRow{
		{Suite: "matmul", Case: "64x64x64", Backend: "cpu", Mode: "ReleaseFast", AverageNS: 1_000_000, Status: "ok"},
		{Suite: "matmul", Case: "64x64x64", Backend: "metal", Mode: "ReleaseFast", AverageNS: 500_000, SampleError: 0.000000014, Status: "ok"},
		{Suite: "training", Case: "batch", Backend: "cpu", Mode: "ReleaseFast", AverageNS: 2_000_000, Status: "ok"},
		{Suite: "training", Case: "batch", Backend: "metal", Mode: "ReleaseFast", Status: "skipped_cpu_only_path"},
	}

	var out bytes.Buffer
	printBenchmarkReport(&out, rows)
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
	} {
		if !strings.Contains(report, want) {
			t.Fatalf("report missing %q:\n%s", want, report)
		}
	}
}
