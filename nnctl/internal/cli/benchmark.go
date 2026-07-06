package cli

import (
	"bytes"
	"context"
	"encoding/csv"
	"fmt"
	"math"
	"os/exec"
	"strconv"
	"strings"
	"text/tabwriter"

	"nnctl/internal/zig"
)

type benchmarkOptions struct {
	gpu    string
	filter string
	quick  bool
	debug  bool
	csv    bool
}

type benchmarkRow struct {
	Suite       string
	Case        string
	Backend     string
	Mode        string
	Warmups     int
	Iterations  int
	AverageNS   float64
	MinNS       float64
	MaxNS       float64
	Checksum    float64
	SampleError float64
	Status      string
}

func defaultBenchmarkOptions() benchmarkOptions {
	return benchmarkOptions{
		gpu: "metal",
	}
}

func (a *App) runBenchmark(ctx context.Context, opts benchmarkOptions) error {
	step := "benchmark"
	if opts.debug {
		step = "benchmark-debug"
	}

	passthrough := make([]string, 0, 3)
	if opts.quick {
		passthrough = append(passthrough, "--quick")
	}
	if opts.filter != "" {
		passthrough = append(passthrough, "--filter", opts.filter)
	}

	args := zig.RunArgs(step, zig.Options{GPU: opts.gpu}, passthrough)
	fmt.Fprintf(a.stderr(), "==> %s\n", zig.CommandString(a.zig, args))

	cmd := exec.CommandContext(ctx, a.zig, args...)
	cmd.Dir = a.repoRoot
	cmd.Stdin = a.stdin()
	output, err := cmd.CombinedOutput()
	if err != nil {
		if len(output) > 0 {
			fmt.Fprint(a.stderr(), string(output))
		}
		return fmt.Errorf("%s failed: %w", a.zig, err)
	}

	if opts.csv {
		fmt.Fprint(a.stdout(), string(output))
		return nil
	}

	rows, err := parseBenchmarkCSV(output)
	if err != nil {
		if len(output) > 0 {
			fmt.Fprint(a.stderr(), string(output))
		}
		return err
	}

	printBenchmarkReport(a.stdout(), rows)
	return nil
}

func parseBenchmarkCSV(output []byte) ([]benchmarkRow, error) {
	lines := strings.Split(string(output), "\n")
	csvLines := make([]string, 0, len(lines))
	inCSV := false

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		if strings.HasPrefix(line, "suite,case,backend,mode,") {
			inCSV = true
		}
		if inCSV {
			csvLines = append(csvLines, line)
		}
	}
	if len(csvLines) == 0 {
		return nil, fmt.Errorf("benchmark output did not contain CSV rows")
	}

	reader := csv.NewReader(strings.NewReader(strings.Join(csvLines, "\n")))
	records, err := reader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("parse benchmark CSV: %w", err)
	}
	if len(records) < 2 {
		return nil, fmt.Errorf("benchmark output did not contain result rows")
	}

	rows := make([]benchmarkRow, 0, len(records)-1)
	for i, record := range records[1:] {
		row, err := parseBenchmarkRecord(record)
		if err != nil {
			return nil, fmt.Errorf("parse benchmark row %d: %w", i+2, err)
		}
		rows = append(rows, row)
	}
	return rows, nil
}

func parseBenchmarkRecord(record []string) (benchmarkRow, error) {
	if len(record) != 12 {
		return benchmarkRow{}, fmt.Errorf("expected 12 fields, got %d", len(record))
	}

	warmups, err := strconv.Atoi(record[4])
	if err != nil {
		return benchmarkRow{}, fmt.Errorf("warmups: %w", err)
	}
	iterations, err := strconv.Atoi(record[5])
	if err != nil {
		return benchmarkRow{}, fmt.Errorf("iterations: %w", err)
	}
	averageNS, err := strconv.ParseFloat(record[6], 64)
	if err != nil {
		return benchmarkRow{}, fmt.Errorf("avg_ns: %w", err)
	}
	minNS, err := strconv.ParseFloat(record[7], 64)
	if err != nil {
		return benchmarkRow{}, fmt.Errorf("min_ns: %w", err)
	}
	maxNS, err := strconv.ParseFloat(record[8], 64)
	if err != nil {
		return benchmarkRow{}, fmt.Errorf("max_ns: %w", err)
	}
	checksum, err := strconv.ParseFloat(record[9], 64)
	if err != nil {
		return benchmarkRow{}, fmt.Errorf("checksum: %w", err)
	}
	sampleError, err := strconv.ParseFloat(record[10], 64)
	if err != nil {
		return benchmarkRow{}, fmt.Errorf("sample_error: %w", err)
	}

	return benchmarkRow{
		Suite:       record[0],
		Case:        record[1],
		Backend:     record[2],
		Mode:        record[3],
		Warmups:     warmups,
		Iterations:  iterations,
		AverageNS:   averageNS,
		MinNS:       minNS,
		MaxNS:       maxNS,
		Checksum:    checksum,
		SampleError: sampleError,
		Status:      record[11],
	}, nil
}

func printBenchmarkReport(dst interface{ Write([]byte) (int, error) }, rows []benchmarkRow) {
	fmt.Fprintf(dst, "Benchmark report\n")
	if len(rows) > 0 {
		fmt.Fprintf(dst, "Mode: %s\n", rows[0].Mode)
	}
	fmt.Fprintf(dst, "\n")

	type pair struct {
		cpu   *benchmarkRow
		metal *benchmarkRow
	}

	suiteOrder := make([]string, 0)
	caseOrder := map[string][]string{}
	pairs := map[string]map[string]*pair{}

	for i := range rows {
		row := &rows[i]
		if _, ok := pairs[row.Suite]; !ok {
			pairs[row.Suite] = map[string]*pair{}
			suiteOrder = append(suiteOrder, row.Suite)
		}
		if _, ok := pairs[row.Suite][row.Case]; !ok {
			pairs[row.Suite][row.Case] = &pair{}
			caseOrder[row.Suite] = append(caseOrder[row.Suite], row.Case)
		}

		switch row.Backend {
		case "cpu":
			pairs[row.Suite][row.Case].cpu = row
		case "metal":
			pairs[row.Suite][row.Case].metal = row
		}
	}

	for _, suite := range suiteOrder {
		fmt.Fprintf(dst, "%s\n", suite)
		var table bytes.Buffer
		tw := tabwriter.NewWriter(&table, 0, 0, 2, ' ', 0)
		fmt.Fprintln(tw, "case\tcpu avg\tmetal avg\tdiff\tspeedup\terror\tstatus")
		for _, caseName := range caseOrder[suite] {
			pair := pairs[suite][caseName]
			fmt.Fprintf(
				tw,
				"%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
				caseName,
				formatBenchmarkTime(pair.cpu),
				formatBenchmarkTime(pair.metal),
				formatBenchmarkDiff(pair.cpu, pair.metal),
				formatBenchmarkSpeedup(pair.cpu, pair.metal),
				formatBenchmarkError(pair.metal),
				formatBenchmarkStatus(pair.cpu, pair.metal),
			)
		}
		_ = tw.Flush()
		fmt.Fprint(dst, table.String())
		fmt.Fprintln(dst)
	}
}

func formatBenchmarkTime(row *benchmarkRow) string {
	if row == nil || row.Status != "ok" {
		return "-"
	}
	return formatDurationNS(row.AverageNS)
}

func formatBenchmarkDiff(cpu, metal *benchmarkRow) string {
	if !comparableBenchmarkRows(cpu, metal) {
		return "-"
	}
	delta := (metal.AverageNS - cpu.AverageNS) / cpu.AverageNS * 100.0
	return fmt.Sprintf("%+.1f%%", delta)
}

func formatBenchmarkSpeedup(cpu, metal *benchmarkRow) string {
	if !comparableBenchmarkRows(cpu, metal) {
		return "-"
	}
	return fmt.Sprintf("%.2fx", cpu.AverageNS/metal.AverageNS)
}

func formatBenchmarkError(row *benchmarkRow) string {
	if row == nil || row.Status != "ok" {
		return "-"
	}
	if row.SampleError == 0 {
		return "0"
	}
	return fmt.Sprintf("%.2e", row.SampleError)
}

func formatBenchmarkStatus(cpu, metal *benchmarkRow) string {
	if cpu == nil && metal == nil {
		return "missing"
	}
	if cpu != nil && cpu.Status != "ok" {
		return "cpu: " + cpu.Status
	}
	if metal == nil {
		return "metal: missing"
	}
	if metal.Status != "ok" {
		return "metal: " + metal.Status
	}
	return "ok"
}

func comparableBenchmarkRows(cpu, metal *benchmarkRow) bool {
	return cpu != nil &&
		metal != nil &&
		cpu.Status == "ok" &&
		metal.Status == "ok" &&
		cpu.AverageNS > 0 &&
		metal.AverageNS > 0 &&
		!math.IsNaN(cpu.AverageNS) &&
		!math.IsNaN(metal.AverageNS)
}

func formatDurationNS(ns float64) string {
	switch {
	case ns >= 1_000_000_000:
		return fmt.Sprintf("%.3f s", ns/1_000_000_000)
	case ns >= 1_000_000:
		return fmt.Sprintf("%.3f ms", ns/1_000_000)
	case ns >= 1_000:
		return fmt.Sprintf("%.3f us", ns/1_000)
	default:
		return fmt.Sprintf("%.0f ns", ns)
	}
}
