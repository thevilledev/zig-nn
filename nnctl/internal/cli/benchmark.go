package cli

import (
	"bytes"
	"context"
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"
	"strings"
	"text/tabwriter"

	"nnctl/internal/process"
	"nnctl/internal/zig"
)

type benchmarkOptions struct {
	gpu     string
	filter  string
	quick   bool
	debug   bool
	csv     bool
	compare string
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
		gpu: "auto",
	}
}

func (a *app) runBenchmark(ctx context.Context, opts benchmarkOptions) error {
	if opts.csv && opts.compare != "" {
		return fmt.Errorf("--csv cannot be used with --compare")
	}

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
	if _, err := fmt.Fprintf(a.stderr(), "==> %s\n", zig.CommandString(a.zig, args)); err != nil {
		return fmt.Errorf("write benchmark command: %w", err)
	}

	cmd := process.CommandContext(ctx, a.zig, args...)
	cmd.Dir = a.repoRoot
	cmd.Stdin = a.stdin()
	output, err := cmd.CombinedOutput()
	if err != nil {
		runErr := fmt.Errorf("%s failed: %w", a.zig, err)
		if len(output) > 0 {
			if writeErr := writeBenchmarkOutput(a.stderr(), output); writeErr != nil {
				return errors.Join(runErr, writeErr)
			}
		}
		return runErr
	}

	if opts.csv {
		return writeBenchmarkOutput(a.stdout(), output)
	}

	rows, err := parseBenchmarkCSV(output)
	if err != nil {
		if len(output) > 0 {
			if writeErr := writeBenchmarkOutput(a.stderr(), output); writeErr != nil {
				return errors.Join(err, writeErr)
			}
		}
		return err
	}

	if opts.compare != "" {
		baselineOutput, err := os.ReadFile(opts.compare)
		if err != nil {
			return fmt.Errorf("read benchmark baseline: %w", err)
		}
		baselineRows, err := parseBenchmarkCSV(baselineOutput)
		if err != nil {
			return fmt.Errorf("parse benchmark baseline: %w", err)
		}
		return printBenchmarkComparison(a.stdout(), rows, baselineRows)
	}

	return printBenchmarkReport(a.stdout(), rows)
}

func writeBenchmarkOutput(dst io.Writer, output []byte) error {
	if _, err := io.Copy(dst, bytes.NewReader(output)); err != nil {
		return fmt.Errorf("write benchmark output: %w", err)
	}
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

func printBenchmarkReport(dst io.Writer, rows []benchmarkRow) error {
	output := newErrorWriter(dst)
	output.printf("Benchmark report\n")
	if len(rows) > 0 {
		output.printf("Mode: %s\n", rows[0].Mode)
	}
	output.printf("\n")

	type pair struct {
		cpu   *benchmarkRow
		metal *benchmarkRow
		cuda  *benchmarkRow
		rocm  *benchmarkRow
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
		case "cuda":
			pairs[row.Suite][row.Case].cuda = row
		case "rocm":
			pairs[row.Suite][row.Case].rocm = row
		}
	}

	for _, suite := range suiteOrder {
		output.printf("%s\n", suite)
		var table bytes.Buffer
		tw := tabwriter.NewWriter(&table, 0, 0, 2, ' ', 0)
		tableOutput := newErrorWriter(tw)
		tableOutput.printf("case\tcpu avg\tmetal avg\tmetal diff\tmetal speedup\tcuda avg\tcuda diff\tcuda speedup\trocm avg\trocm diff\trocm speedup\terror\tstatus\n")
		for _, caseName := range caseOrder[suite] {
			pair := pairs[suite][caseName]
			tableOutput.printf(
				"%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
				caseName,
				formatBenchmarkTime(pair.cpu),
				formatBenchmarkTime(pair.metal),
				formatBenchmarkDiff(pair.cpu, pair.metal),
				formatBenchmarkSpeedup(pair.cpu, pair.metal),
				formatBenchmarkTime(pair.cuda),
				formatBenchmarkDiff(pair.cpu, pair.cuda),
				formatBenchmarkSpeedup(pair.cpu, pair.cuda),
				formatBenchmarkTime(pair.rocm),
				formatBenchmarkDiff(pair.cpu, pair.rocm),
				formatBenchmarkSpeedup(pair.cpu, pair.rocm),
				formatBenchmarkErrors(pair.metal, pair.cuda, pair.rocm),
				formatBenchmarkStatus(pair.cpu, pair.metal, pair.cuda, pair.rocm),
			)
		}
		if err := tableOutput.Err(); err != nil {
			return fmt.Errorf("format benchmark report: %w", err)
		}
		if err := tw.Flush(); err != nil {
			return fmt.Errorf("format benchmark report: %w", err)
		}
		output.printf("%s\n", table.String())
	}
	if err := output.Err(); err != nil {
		return fmt.Errorf("write benchmark report: %w", err)
	}
	return nil
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

func formatBenchmarkErrors(metal, cuda, rocm *benchmarkRow) string {
	metalError := formatBenchmarkError(metal)
	cudaError := formatBenchmarkError(cuda)
	rocmError := formatBenchmarkError(rocm)

	errors := make([]string, 0, 3)
	if metalError != "-" {
		errors = append(errors, "metal="+metalError)
	}
	if cudaError != "-" {
		errors = append(errors, "cuda="+cudaError)
	}
	if rocmError != "-" {
		errors = append(errors, "rocm="+rocmError)
	}
	if len(errors) == 0 {
		return "-"
	}
	if len(errors) == 1 {
		if _, value, ok := strings.Cut(errors[0], "="); ok {
			return value
		}
		return errors[0]
	}
	return strings.Join(errors, " ")
}

func formatBenchmarkStatus(cpu, metal, cuda, rocm *benchmarkRow) string {
	if cpu == nil && metal == nil && cuda == nil && rocm == nil {
		return "missing"
	}
	if cpu != nil && cpu.Status != "ok" {
		return "cpu: " + cpu.Status
	}
	statuses := make([]string, 0, 3)
	if metal != nil && metal.Status != "ok" {
		statuses = append(statuses, "metal: "+metal.Status)
	}
	if cuda != nil && cuda.Status != "ok" {
		statuses = append(statuses, "cuda: "+cuda.Status)
	}
	if rocm != nil && rocm.Status != "ok" {
		statuses = append(statuses, "rocm: "+rocm.Status)
	}
	if len(statuses) > 0 {
		return strings.Join(statuses, ", ")
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

func printBenchmarkComparison(dst io.Writer, current, baseline []benchmarkRow) error {
	output := newErrorWriter(dst)
	output.printf("Benchmark comparison\n")
	if len(current) > 0 {
		output.printf("Mode: %s\n", current[0].Mode)
	}
	output.printf("\n")

	baselineByKey := make(map[string]*benchmarkRow, len(baseline))
	for i := range baseline {
		row := &baseline[i]
		baselineByKey[benchmarkRowKey(*row)] = row
	}

	seen := make(map[string]bool, len(current))
	var table bytes.Buffer
	tw := tabwriter.NewWriter(&table, 0, 0, 2, ' ', 0)
	tableOutput := newErrorWriter(tw)
	tableOutput.printf("suite\tcase\tbackend\tcurrent\tbaseline\tdelta\tstatus\n")

	for i := range current {
		row := &current[i]
		key := benchmarkRowKey(*row)
		seen[key] = true
		base := baselineByKey[key]
		tableOutput.printf(
			"%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
			row.Suite,
			row.Case,
			row.Backend,
			formatBenchmarkTime(row),
			formatBenchmarkTime(base),
			formatBenchmarkComparisonDelta(row, base),
			formatBenchmarkComparisonStatus(row, base),
		)
	}

	for i := range baseline {
		row := &baseline[i]
		key := benchmarkRowKey(*row)
		if seen[key] {
			continue
		}
		tableOutput.printf(
			"%s\t%s\t%s\t-\t%s\t-\tmissing_current\n",
			row.Suite,
			row.Case,
			row.Backend,
			formatBenchmarkTime(row),
		)
	}

	if err := tableOutput.Err(); err != nil {
		return fmt.Errorf("format benchmark comparison: %w", err)
	}
	if err := tw.Flush(); err != nil {
		return fmt.Errorf("format benchmark comparison: %w", err)
	}
	output.printf("%s", table.String())
	if err := output.Err(); err != nil {
		return fmt.Errorf("write benchmark comparison: %w", err)
	}
	return nil
}

func benchmarkRowKey(row benchmarkRow) string {
	return strings.Join([]string{row.Mode, row.Suite, row.Case, row.Backend}, "\x00")
}

func formatBenchmarkComparisonDelta(current, baseline *benchmarkRow) string {
	if !comparableBenchmarkRows(baseline, current) {
		return "-"
	}
	delta := (current.AverageNS - baseline.AverageNS) / baseline.AverageNS * 100.0
	return fmt.Sprintf("%+.1f%%", delta)
}

func formatBenchmarkComparisonStatus(current, baseline *benchmarkRow) string {
	switch {
	case current == nil && baseline == nil:
		return "missing"
	case current == nil:
		return "missing_current"
	case baseline == nil:
		return "new"
	case current.Status != "ok" && current.Status == baseline.Status:
		return current.Status
	case current.Status != "ok":
		return "current: " + current.Status
	case baseline.Status != "ok":
		return "baseline: " + baseline.Status
	default:
		return "ok"
	}
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
