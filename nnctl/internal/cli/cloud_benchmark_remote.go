package cli

import (
	"bytes"
	"context"
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"nnctl/internal/cloud/verda"
	"nnctl/internal/zig"
)

func waitForCloudBenchmarkInstance(ctx context.Context, client cloudBenchmarkClient, instanceID string, timeout, poll time.Duration) (verda.Instance, error) {
	deadline := time.Now().Add(timeout)
	for {
		instances, err := client.ListInstances(ctx, "")
		if err != nil {
			return verda.Instance{}, fmt.Errorf("list Verda instances while waiting for %s: %w", instanceID, err)
		}
		for _, instance := range instances {
			if instance.ID != instanceID {
				continue
			}
			status := strings.ToLower(strings.TrimSpace(instance.Status))
			if cloudBenchmarkInstanceFailed(status) {
				return verda.Instance{}, fmt.Errorf("benchmark instance %s entered terminal status %q", instanceID, instance.Status)
			}
			if status == "running" && instance.IP != nil && strings.TrimSpace(*instance.IP) != "" {
				return instance, nil
			}
		}
		if time.Now().After(deadline) {
			return verda.Instance{}, fmt.Errorf("timed out waiting for benchmark instance %s to be running with an IP", instanceID)
		}
		timer := time.NewTimer(poll)
		select {
		case <-ctx.Done():
			timer.Stop()
			return verda.Instance{}, ctx.Err()
		case <-timer.C:
		}
	}
}

func cloudBenchmarkInstanceFailed(status string) bool {
	switch status {
	case "deleted", "deleting", "error", "failed", "canceled", "cancelled":
		return true
	default:
		return false
	}
}

func waitForCloudBenchmarkSSH(ctx context.Context, ssh string, baseArgs []string, host string, timeout, poll time.Duration) error {
	deadline := time.Now().Add(timeout)
	var lastErr error
	for {
		args := append(append([]string{}, baseArgs...), host, "true")
		cmd := exec.CommandContext(ctx, ssh, args...)
		cmd.Stdout = io.Discard
		cmd.Stderr = io.Discard
		if err := cmd.Run(); err == nil {
			return nil
		} else {
			lastErr = err
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("timed out waiting for SSH on %s: %w", host, lastErr)
		}
		timer := time.NewTimer(poll)
		select {
		case <-ctx.Done():
			timer.Stop()
			return ctx.Err()
		case <-timer.C:
		}
	}
}

func cloudBenchmarkSSHArgs(knownHosts string) []string {
	return []string{
		"-o", "UserKnownHostsFile=" + knownHosts,
		"-o", "StrictHostKeyChecking=accept-new",
		"-o", "ConnectTimeout=10",
	}
}

func remoteBenchmarkCommand(remoteDir, backend, filter string, quick bool) string {
	parts := []string{
		"set -eu",
		"if [ -f /etc/profile.d/zig-nn.sh ]; then . /etc/profile.d/zig-nn.sh; fi",
		"cd " + shellQuote(remoteDir),
		"zig build benchmark -Dgpu=" + shellQuote(backend),
	}
	if quick {
		parts[len(parts)-1] += " -- --quick"
		if filter != "" {
			parts[len(parts)-1] += " --filter " + shellQuote(filter)
		}
	} else if filter != "" {
		parts[len(parts)-1] += " -- --filter " + shellQuote(filter)
	}
	return strings.Join(parts, "; ")
}

func remoteBenchmarkMetadataCommand() string {
	return "if command -v zig >/dev/null 2>&1; then printf 'zig='; zig version; fi; " +
		"if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv,noheader,nounits | sed 's/^/nvidia=/'; fi"
}

func (a *app) runCloudBenchmarkSSH(ctx context.Context, ssh string, baseArgs []string, host, remoteCommand string, stdout io.Writer) error {
	args := append(append([]string{}, baseArgs...), host, remoteCommand)
	fmt.Fprintf(a.stderr(), "==> %s\n", zig.CommandString(ssh, args))
	cmd := exec.CommandContext(ctx, ssh, args...)
	cmd.Stdin = a.stdin()
	cmd.Stdout = stdout
	cmd.Stderr = a.stderr()
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("%s failed: %w", ssh, err)
	}
	return nil
}

func (a *app) captureCloudBenchmarkSSH(ctx context.Context, ssh string, baseArgs []string, host, remoteCommand string) ([]byte, error) {
	args := append(append([]string{}, baseArgs...), host, remoteCommand)
	fmt.Fprintf(a.stderr(), "==> %s\n", zig.CommandString(ssh, args))
	cmd := exec.CommandContext(ctx, ssh, args...)
	cmd.Stdin = a.stdin()
	output, err := cmd.CombinedOutput()
	if err != nil {
		if len(output) > 0 {
			fmt.Fprint(a.stderr(), string(output))
		}
		return nil, fmt.Errorf("%s failed: %w", ssh, err)
	}
	return output, nil
}

func (a *app) runCloudBenchmarkCommand(ctx context.Context, dir string, env []string, name string, args ...string) error {
	fmt.Fprintf(a.stderr(), "==> %s\n", zig.CommandString(name, args))
	cmd := exec.CommandContext(ctx, name, args...)
	cmd.Dir = dir
	cmd.Env = env
	cmd.Stdin = a.stdin()
	cmd.Stdout = a.stdout()
	cmd.Stderr = a.stderr()
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("%s failed: %w", name, err)
	}
	return nil
}

func cleanupCloudBenchmark(ctx context.Context, client cloudBenchmarkClient, baseURL, instanceID, cloneID, sourceID string) error {
	volumeIDs := []string(nil)
	if strings.TrimSpace(cloneID) != "" {
		volumeIDs = []string{cloneID}
	}
	_, err := verda.Destroy(ctx, client, verda.DestroyOptions{
		InstanceIDs:       []string{instanceID},
		VolumeIDs:         volumeIDs,
		SourceOSVolumeID:  sourceID,
		DeletePermanently: true,
		BaseURL:           baseURL,
	})
	return err
}

func writeCloudBenchmarkCSV(path string, metadata cloudBenchmarkMetadata, output []byte) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create benchmark CSV directory: %w", err)
	}
	var content bytes.Buffer
	writeCloudBenchmarkMetadata(&content, metadata)
	content.Write(output)
	if content.Len() > 0 && content.Bytes()[content.Len()-1] != '\n' {
		content.WriteByte('\n')
	}
	if err := os.WriteFile(path, content.Bytes(), 0o644); err != nil {
		return fmt.Errorf("write benchmark CSV: %w", err)
	}
	return nil
}

func writeCloudBenchmarkMetadata(dst io.Writer, metadata cloudBenchmarkMetadata) {
	values := []struct {
		name  string
		value string
	}{
		{"captured", metadata.Timestamp.UTC().Format(time.RFC3339)},
		{"provider", metadata.Provider},
		{"instance_id", metadata.InstanceID},
		{"instance_type", metadata.InstanceType},
		{"location", metadata.Location},
		{"market", metadata.Market},
		{"price_per_hour", fmt.Sprintf("%g", metadata.PricePerHour)},
		{"source_os_volume_id", metadata.SourceVolumeID},
		{"cloned_os_volume_id", metadata.ClonedVolumeID},
		{"commit", metadata.Commit},
		{"backend", metadata.Backend},
		{"zig", metadata.ZigVersion},
		{"gpu", metadata.GPU},
		{"memory_mib", metadata.MemoryMiB},
		{"compute", metadata.ComputeCapability},
		{"driver", metadata.Driver},
	}
	for _, item := range values {
		if strings.TrimSpace(item.value) != "" {
			fmt.Fprintf(dst, "# %s=%s\n", item.name, strings.TrimSpace(item.value))
		}
	}
}

func mergeCloudBenchmarkRemoteMetadata(metadata *cloudBenchmarkMetadata, output []byte) {
	for _, line := range strings.Split(string(output), "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "zig=") {
			metadata.ZigVersion = strings.TrimSpace(strings.TrimPrefix(line, "zig="))
			continue
		}
		if !strings.HasPrefix(line, "nvidia=") {
			continue
		}
		records, err := csv.NewReader(strings.NewReader(strings.TrimPrefix(line, "nvidia="))).Read()
		if err != nil || len(records) < 4 {
			continue
		}
		metadata.GPU = strings.TrimSpace(records[0])
		metadata.MemoryMiB = strings.TrimSpace(records[1])
		metadata.ComputeCapability = strings.TrimSpace(records[2])
		metadata.Driver = strings.TrimSpace(records[3])
	}
}

func updateCloudBenchmarkDocs(path, csvPath string, metadata cloudBenchmarkMetadata, rows []benchmarkRow) error {
	content, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read benchmark docs: %w", err)
	}
	relativeCSV, err := filepath.Rel(filepath.Dir(path), csvPath)
	if err != nil {
		return fmt.Errorf("make benchmark CSV docs path: %w", err)
	}
	relativeCSV = filepath.ToSlash(relativeCSV)
	key := filepath.Base(csvPath)
	startMarker := "<!-- nnctl-cloud-benchmark:" + key + ":start -->"
	endMarker := "<!-- nnctl-cloud-benchmark:" + key + ":end -->"
	block := renderCloudBenchmarkDocsBlock(startMarker, endMarker, relativeCSV, metadata, rows)

	text := string(content)
	if start := strings.Index(text, startMarker); start >= 0 {
		end := strings.Index(text[start:], endMarker)
		if end < 0 {
			return fmt.Errorf("benchmark docs marker %q is missing its end marker", startMarker)
		}
		end += start + len(endMarker)
		text = text[:start] + block + text[end:]
	} else {
		if !strings.Contains(text, cloudBenchmarkDocsHeading) {
			text = strings.TrimRight(text, "\n") + "\n\n" + cloudBenchmarkDocsHeading + "\n\n"
			text += "This section is generated by `nnctl cloud benchmark-deploy --update-docs`.\n"
		}
		text = strings.TrimRight(text, "\n") + "\n\n" + block + "\n"
	}
	info, err := os.Stat(path)
	if err != nil {
		return fmt.Errorf("stat benchmark docs: %w", err)
	}
	if err := os.WriteFile(path, []byte(text), info.Mode().Perm()); err != nil {
		return fmt.Errorf("update benchmark docs: %w", err)
	}
	return nil
}

func renderCloudBenchmarkDocsBlock(startMarker, endMarker, relativeCSV string, metadata cloudBenchmarkMetadata, rows []benchmarkRow) string {
	var out strings.Builder
	fmt.Fprintln(&out, startMarker)
	title := firstNonEmpty(metadata.GPU, metadata.InstanceType)
	fmt.Fprintf(&out, "### %s — %s\n\n", markdownCell(title), metadata.Timestamp.UTC().Format("2006-01-02"))
	fmt.Fprintf(&out, "Verda `%s` `%s` run in `%s` from commit `%s`; raw [CSV](%s).\n\n",
		markdownCell(metadata.Market), markdownCell(metadata.InstanceType), markdownCell(metadata.Location), markdownCell(metadata.Commit), relativeCSV)
	fmt.Fprintln(&out, "| suite | case | backend | average | status |")
	fmt.Fprintln(&out, "| --- | --- | --- | ---: | --- |")
	for i := range rows {
		row := &rows[i]
		if row.Backend != metadata.Backend || row.Status != "ok" {
			continue
		}
		fmt.Fprintf(&out, "| %s | %s | %s | %s | %s |\n",
			markdownCell(row.Suite), markdownCell(row.Case), markdownCell(row.Backend), formatDurationNS(row.AverageNS), markdownCell(row.Status))
	}
	fmt.Fprint(&out, endMarker)
	return out.String()
}

func defaultCloudBenchmarkOutputPath(repoRoot string, metadata cloudBenchmarkMetadata) string {
	name := firstNonEmpty(metadata.GPU, metadata.InstanceType, "cloud")
	filename := fmt.Sprintf("benchmark-%s-%s-%s-%s.csv",
		slugCloudBenchmark(name), slugCloudBenchmark(metadata.Backend), slugCloudBenchmark(metadata.Commit), metadata.Timestamp.UTC().Format("20060102T150405Z"))
	return filepath.Join(repoRoot, filename)
}

func gitRevision(ctx context.Context, repoRoot, git, ref string) (string, error) {
	cmd := exec.CommandContext(ctx, git, "rev-parse", "--short=12", ref)
	cmd.Dir = repoRoot
	output, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("resolve git ref %q: %s: %w", ref, strings.TrimSpace(string(output)), err)
	}
	return strings.TrimSpace(string(output)), nil
}

func environmentWith(current []string, values map[string]string) []string {
	env := make([]string, 0, len(current)+len(values))
	for _, item := range current {
		name, _, ok := strings.Cut(item, "=")
		if ok {
			if _, replace := values[name]; replace {
				continue
			}
		}
		env = append(env, item)
	}
	for name, value := range values {
		env = append(env, name+"="+value)
	}
	return env
}

func resolveRepoPath(repoRoot, path string) string {
	if filepath.IsAbs(path) {
		return filepath.Clean(path)
	}
	return filepath.Join(repoRoot, path)
}

func commandString(name string, args []string) string {
	parts := []string{shellQuote(name)}
	for _, arg := range args {
		parts = append(parts, shellQuote(arg))
	}
	return strings.Join(parts, " ")
}

func shellQuote(value string) string {
	return "'" + strings.ReplaceAll(value, "'", `'"'"'`) + "'"
}

func slugCloudBenchmark(value string) string {
	value = strings.ToLower(strings.TrimSpace(value))
	var out strings.Builder
	lastDash := false
	for _, r := range value {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			out.WriteRune(r)
			lastDash = false
		} else if !lastDash && out.Len() > 0 {
			out.WriteByte('-')
			lastDash = true
		}
	}
	return strings.Trim(out.String(), "-")
}

func markdownCell(value string) string {
	return strings.ReplaceAll(strings.TrimSpace(value), "|", `\|`)
}
