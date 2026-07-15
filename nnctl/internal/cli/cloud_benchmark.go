package cli

import (
	"bytes"
	"context"
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"

	verdacloud "nnctl/internal/cloud/verda"
	"nnctl/internal/zig"
)

const (
	defaultCloudBenchmarkPackerDir       = "packer/verda"
	defaultCloudBenchmarkSourceName      = "packer-verda-zig-nn-volume-root"
	defaultCloudBenchmarkPackerType      = "CPU.4V.16G"
	defaultCloudBenchmarkPackerImage     = "ubuntu-24.04-cuda-13.0-open"
	defaultCloudBenchmarkBackend         = "cuda"
	defaultCloudBenchmarkSSHUser         = "root"
	defaultCloudBenchmarkDocsPath        = "docs/benchmarks.md"
	defaultCloudBenchmarkTimeout         = 20 * time.Minute
	defaultCloudBenchmarkPollInterval    = 10 * time.Second
	cloudBenchmarkDocsHeading            = "## Automated Cloud Benchmark Runs"
	cloudBenchmarkPackerBuildLocation    = "FIN-01"
	cloudBenchmarkPackerArtifactLocation = "FIN-02"
)

type cloudBenchmarkDeployOptions struct {
	verdacloud.DeployOptions
	packerDir             string
	packer                string
	packerInstanceType    string
	authorizedKeysFile    string
	refreshPackerTemplate bool
	backend               string
	filter                string
	quick                 bool
	skipSmoke             bool
	ref                   string
	ignoreDirty           bool
	remoteDir             string
	sshUser               string
	ssh                   string
	rsync                 string
	git                   string
	tar                   string
	output                string
	updateDocs            bool
	docsPath              string
	keepInstance          bool
	timeout               time.Duration
	pollInterval          time.Duration
}

type cloudBenchmarkClient interface {
	verdacloud.Client
	verdacloud.DestroyClient
	ListInstances(context.Context, string) ([]verdacloud.Instance, error)
	ListSSHKeys(context.Context) ([]verdacloud.SSHKey, error)
}

type cloudBenchmarkMetadata struct {
	Timestamp         time.Time
	Commit            string
	Provider          string
	InstanceID        string
	InstanceType      string
	Location          string
	Market            string
	PricePerHour      float64
	SourceVolumeID    string
	ClonedVolumeID    string
	Backend           string
	GPU               string
	MemoryMiB         string
	ComputeCapability string
	Driver            string
	ZigVersion        string
}

func defaultCloudBenchmarkDeployOptions() cloudBenchmarkDeployOptions {
	deployOpts := verdacloud.DefaultDeployOptions("")
	deployOpts.Image = defaultCloudBenchmarkPackerImage
	return cloudBenchmarkDeployOptions{
		DeployOptions:      deployOpts,
		packerDir:          defaultCloudBenchmarkPackerDir,
		packer:             "packer",
		packerInstanceType: defaultCloudBenchmarkPackerType,
		backend:            defaultCloudBenchmarkBackend,
		ref:                "HEAD",
		sshUser:            defaultCloudBenchmarkSSHUser,
		ssh:                "ssh",
		rsync:              "rsync",
		git:                "git",
		tar:                "tar",
		docsPath:           defaultCloudBenchmarkDocsPath,
		timeout:            defaultCloudBenchmarkTimeout,
		pollInterval:       defaultCloudBenchmarkPollInterval,
	}
}

func (a *App) runCloudBenchmarkDeploy(ctx context.Context, opts cloudBenchmarkDeployOptions) (runErr error) {
	if err := normalizeCloudBenchmarkDeployOptions(&opts); err != nil {
		return err
	}

	if !opts.ignoreDirty {
		status, err := a.gitStatusPorcelain(ctx, opts.git)
		if err != nil {
			return err
		}
		if status != "" {
			return fmt.Errorf("working tree has uncommitted changes; commit them or pass --ignore-dirty to benchmark %s anyway", opts.ref)
		}
	}

	commit, err := gitRevision(ctx, a.repoRoot, opts.git, opts.ref)
	if err != nil {
		return err
	}
	stamp := time.Now().UTC()
	if opts.authorizedKeysFile != "" {
		opts.authorizedKeysFile = resolveRepoPath(a.repoRoot, opts.authorizedKeysFile)
	}

	packerDir := resolveRepoPath(a.repoRoot, opts.packerDir)
	written, err := ensureCloudPackerTemplate(packerDir, opts.refreshPackerTemplate)
	if err != nil {
		return err
	}
	for _, path := range written {
		fmt.Fprintf(a.stderr(), "==> wrote %s\n", path)
	}
	if len(written) == 0 {
		fmt.Fprintf(a.stderr(), "==> using Packer template in %s\n", packerDir)
	}

	client, creds, err := a.newVerdaSDKClientWithCredentials(ctx, opts.BaseURL)
	if err != nil {
		return err
	}

	workflowDir, err := os.MkdirTemp("", "nnctl-cloud-benchmark-*")
	if err != nil {
		return fmt.Errorf("create cloud benchmark work directory: %w", err)
	}
	defer os.RemoveAll(workflowDir)

	sourceVolume, err := a.prepareCloudBenchmarkSourceVolume(ctx, client, creds, packerDir, workflowDir, &opts)
	if err != nil {
		return err
	}
	if opts.LocationCode != "" && !strings.EqualFold(opts.LocationCode, sourceVolume.Location) {
		return fmt.Errorf("source OS volume %s is in %s, but --location-code requested %s", sourceVolume.ID, sourceVolume.Location, opts.LocationCode)
	}
	opts.SourceOSVolumeID = sourceVolume.ID
	opts.SourceOSVolumeName = ""
	opts.LocationCode = sourceVolume.Location
	fmt.Fprintf(a.stderr(), "==> using source OS volume %s (%s in %s)\n", sourceVolume.ID, sourceVolume.Name, sourceVolume.Location)

	result, err := verdacloud.Deploy(ctx, client, opts.DeployOptions)
	if err != nil {
		return err
	}
	if result.Instance == nil || strings.TrimSpace(result.Instance.ID) == "" {
		return fmt.Errorf("verda deployment did not return an instance")
	}

	instanceID := result.Instance.ID
	cloneID := ""
	if result.OSVolumeClone != nil {
		cloneID = result.OSVolumeClone.VolumeID
	}
	if !opts.keepInstance {
		defer func() {
			cleanupCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Minute)
			defer cancel()
			cleanupErr := cleanupCloudBenchmark(cleanupCtx, client, opts.BaseURL, instanceID, cloneID, sourceVolume.ID)
			if cleanupErr != nil {
				runErr = errors.Join(runErr, cleanupErr)
				return
			}
			fmt.Fprintf(a.stderr(), "==> destroyed benchmark instance %s and cloned OS volume %s\n", instanceID, cloneID)
		}()
	} else {
		fmt.Fprintf(a.stderr(), "==> keeping benchmark instance %s and cloned OS volume %s\n", instanceID, cloneID)
	}

	instance, err := waitForCloudBenchmarkInstance(ctx, client, instanceID, opts.timeout, opts.pollInterval)
	if err != nil {
		return err
	}
	host := opts.sshUser + "@" + strings.TrimSpace(*instance.IP)
	knownHosts := filepath.Join(workflowDir, "known_hosts")
	sshArgs := cloudBenchmarkSSHArgs(knownHosts)
	if err := waitForCloudBenchmarkSSH(ctx, opts.ssh, sshArgs, host, opts.timeout, opts.pollInterval); err != nil {
		return err
	}

	remoteDir := strings.TrimSpace(opts.remoteDir)
	if remoteDir == "" {
		remoteRoot := "/tmp"
		if opts.sshUser == "root" {
			remoteRoot = "/root"
		}
		remoteDir = remoteRoot + "/zig-nn-bench-" + slugCloudBenchmark(commit) + "-" + stamp.Format("20060102T150405Z")
	}
	deployOpts := defaultDeployOptions()
	deployOpts.target = host + ":" + remoteDir
	deployOpts.ref = opts.ref
	deployOpts.delete = true
	deployOpts.ignoreDirty = true
	deployOpts.ssh = commandString(opts.ssh, sshArgs)
	deployOpts.git = opts.git
	deployOpts.tar = opts.tar
	deployOpts.rsync = opts.rsync
	if err := a.runDeploy(ctx, deployOpts); err != nil {
		return err
	}

	if !opts.quick && !opts.skipSmoke {
		fmt.Fprintln(a.stderr(), "==> running remote quick benchmark smoke test")
		if err := a.runCloudBenchmarkSSH(ctx, opts.ssh, sshArgs, host, remoteBenchmarkCommand(remoteDir, opts.backend, "", true), a.stdout()); err != nil {
			return fmt.Errorf("remote quick benchmark failed: %w", err)
		}
	}

	fmt.Fprintln(a.stderr(), "==> running remote benchmark")
	benchmarkOutput, err := a.captureCloudBenchmarkSSH(ctx, opts.ssh, sshArgs, host, remoteBenchmarkCommand(remoteDir, opts.backend, opts.filter, opts.quick))
	if err != nil {
		return fmt.Errorf("remote benchmark failed: %w", err)
	}
	rows, err := parseBenchmarkCSV(benchmarkOutput)
	if err != nil {
		return err
	}

	pricePerHour := instance.PricePerHour
	if pricePerHour == 0 && result.Placement != nil {
		pricePerHour = result.Placement.SpotPrice
	}
	metadata := cloudBenchmarkMetadata{
		Timestamp:      stamp,
		Commit:         commit,
		Provider:       verdacloud.ProviderName,
		InstanceID:     instanceID,
		InstanceType:   opts.InstanceType,
		Location:       firstNonEmpty(instance.Location, sourceVolume.Location),
		Market:         opts.Market,
		PricePerHour:   pricePerHour,
		SourceVolumeID: sourceVolume.ID,
		ClonedVolumeID: cloneID,
		Backend:        opts.backend,
	}
	if result.InstanceType != nil {
		metadata.GPU = result.InstanceType.Model
	}
	if remoteMetadata, metadataErr := a.captureCloudBenchmarkSSH(ctx, opts.ssh, sshArgs, host, remoteBenchmarkMetadataCommand()); metadataErr != nil {
		fmt.Fprintf(a.stderr(), "warning: remote benchmark metadata could not be collected: %v\n", metadataErr)
	} else {
		mergeCloudBenchmarkRemoteMetadata(&metadata, remoteMetadata)
	}

	outputPath := strings.TrimSpace(opts.output)
	if outputPath == "" {
		outputPath = defaultCloudBenchmarkOutputPath(a.repoRoot, metadata)
	} else {
		outputPath = resolveRepoPath(a.repoRoot, outputPath)
	}
	if err := writeCloudBenchmarkCSV(outputPath, metadata, benchmarkOutput); err != nil {
		return err
	}
	fmt.Fprintf(a.stdout(), "benchmark CSV: %s\n", outputPath)

	if opts.updateDocs {
		docsPath := resolveRepoPath(a.repoRoot, opts.docsPath)
		if err := updateCloudBenchmarkDocs(docsPath, outputPath, metadata, rows); err != nil {
			return err
		}
		fmt.Fprintf(a.stdout(), "updated benchmark docs: %s\n", docsPath)
	}
	return nil
}

func normalizeCloudBenchmarkDeployOptions(opts *cloudBenchmarkDeployOptions) error {
	opts.InstanceType = strings.TrimSpace(opts.InstanceType)
	opts.SourceOSVolumeID = strings.TrimSpace(opts.SourceOSVolumeID)
	opts.SourceOSVolumeName = strings.TrimSpace(opts.SourceOSVolumeName)
	opts.Market = strings.ToLower(strings.TrimSpace(opts.Market))
	opts.LocationCode = strings.ToUpper(strings.TrimSpace(opts.LocationCode))
	opts.packerDir = strings.TrimSpace(opts.packerDir)
	opts.packer = strings.TrimSpace(opts.packer)
	opts.packerInstanceType = strings.TrimSpace(opts.packerInstanceType)
	opts.backend = strings.ToLower(strings.TrimSpace(opts.backend))
	opts.ref = strings.TrimSpace(opts.ref)
	opts.sshUser = strings.TrimSpace(opts.sshUser)
	opts.ssh = strings.TrimSpace(opts.ssh)
	opts.rsync = strings.TrimSpace(opts.rsync)
	opts.git = strings.TrimSpace(opts.git)
	opts.tar = strings.TrimSpace(opts.tar)
	opts.docsPath = strings.TrimSpace(opts.docsPath)

	if opts.InstanceType == "" {
		return fmt.Errorf("instance type is required")
	}
	if opts.SourceOSVolumeID != "" && opts.SourceOSVolumeName != "" {
		return fmt.Errorf("source OS volume ID and name cannot be combined")
	}
	if opts.SourceOSVolumeID == "" && opts.SourceOSVolumeName == "" {
		opts.SourceOSVolumeName = defaultCloudBenchmarkSourceName
	}
	if opts.Market != verdacloud.PricingMarketSpot && opts.Market != verdacloud.PricingMarketOnDemand {
		return fmt.Errorf("benchmark deploy market must be spot or on-demand")
	}
	switch opts.backend {
	case "cuda", "rocm", "none":
	default:
		return fmt.Errorf("benchmark GPU backend must be cuda, rocm, or none")
	}
	if opts.packerDir == "" || opts.packer == "" || opts.packerInstanceType == "" {
		return fmt.Errorf("packer directory, executable, and instance type must be configured")
	}
	if opts.ref == "" {
		return fmt.Errorf("--ref must not be empty")
	}
	if opts.sshUser == "" || opts.ssh == "" || opts.rsync == "" || opts.git == "" || opts.tar == "" {
		return fmt.Errorf("SSH user, ssh, rsync, git, and tar must be configured")
	}
	if opts.timeout <= 0 || opts.pollInterval <= 0 {
		return fmt.Errorf("timeout and poll interval must be positive")
	}
	if opts.updateDocs && opts.docsPath == "" {
		return fmt.Errorf("--docs must not be empty with --update-docs")
	}
	if err := verdacloud.ValidateSSHKeyIDs(opts.SSHKeyIDs); err != nil {
		return err
	}
	return nil
}

func (a *App) prepareCloudBenchmarkSourceVolume(
	ctx context.Context,
	client cloudBenchmarkClient,
	creds verdacloud.Credentials,
	packerDir string,
	workflowDir string,
	opts *cloudBenchmarkDeployOptions,
) (verdacloud.Volume, error) {
	if opts.SourceOSVolumeID != "" {
		volume, err := client.GetVolume(ctx, opts.SourceOSVolumeID)
		if err != nil {
			return verdacloud.Volume{}, fmt.Errorf("get source Verda OS volume %s: %w", opts.SourceOSVolumeID, err)
		}
		return normalizeCloudBenchmarkSourceVolume(volume)
	}

	placements, err := client.GetPlacementOptions(ctx, opts.InstanceType, opts.Market == verdacloud.PricingMarketSpot)
	if err != nil {
		return verdacloud.Volume{}, fmt.Errorf("list Verda %s placements for %q: %w", opts.Market, opts.InstanceType, err)
	}
	sortCloudBenchmarkPlacements(placements)
	volumes, err := client.ListVolumes(ctx)
	if err != nil {
		return verdacloud.Volume{}, fmt.Errorf("list Verda volumes: %w", err)
	}
	if opts.LocationCode == "" {
		for _, placement := range placements {
			if source, ok := reusableCloudBenchmarkSourceVolume(volumes, opts.SourceOSVolumeName, placement.LocationCode); ok {
				return source, nil
			}
		}
	}
	targetLocation, err := cloudBenchmarkTargetLocation(opts.LocationCode, placements)
	if err != nil {
		return verdacloud.Volume{}, err
	}
	if source, ok := reusableCloudBenchmarkSourceVolume(volumes, opts.SourceOSVolumeName, targetLocation); ok {
		return source, nil
	}

	fmt.Fprintf(a.stderr(), "==> no reusable OS volume named %q in %s; building it with Packer\n", opts.SourceOSVolumeName, targetLocation)
	keysFile, err := a.cloudBenchmarkAuthorizedKeysFile(ctx, client, workflowDir, opts.authorizedKeysFile, opts.SSHKeyIDs)
	if err != nil {
		return verdacloud.Volume{}, err
	}
	templatePath := filepath.Join(packerDir, "ubuntu.pkr.hcl")
	buildArgs, err := cloudBenchmarkPackerBuildArgs(templatePath, keysFile, *opts, targetLocation)
	if err != nil {
		return verdacloud.Volume{}, err
	}
	packerEnv := environmentWith(os.Environ(), map[string]string{
		"VERDA_CLIENT_ID":     creds.ClientID,
		"VERDA_CLIENT_SECRET": creds.ClientSecret,
	})
	if err := a.runCloudBenchmarkCommand(ctx, packerDir, packerEnv, opts.packer, "init", "."); err != nil {
		return verdacloud.Volume{}, err
	}
	if err := a.runCloudBenchmarkCommand(ctx, packerDir, packerEnv, opts.packer, buildArgs...); err != nil {
		return verdacloud.Volume{}, err
	}

	deadline := time.Now().Add(opts.timeout)
	for {
		volumes, listErr := client.ListVolumes(ctx)
		if listErr == nil {
			if source, ok := reusableCloudBenchmarkSourceVolume(volumes, opts.SourceOSVolumeName, targetLocation); ok {
				return source, nil
			}
		}
		if time.Now().After(deadline) {
			if listErr != nil {
				return verdacloud.Volume{}, fmt.Errorf("wait for Packer OS volume %q in %s: %w", opts.SourceOSVolumeName, targetLocation, listErr)
			}
			return verdacloud.Volume{}, fmt.Errorf("packer completed but OS volume %q did not appear in %s", opts.SourceOSVolumeName, targetLocation)
		}
		timer := time.NewTimer(opts.pollInterval)
		select {
		case <-ctx.Done():
			timer.Stop()
			return verdacloud.Volume{}, ctx.Err()
		case <-timer.C:
		}
	}
}

func (a *App) cloudBenchmarkAuthorizedKeysFile(ctx context.Context, client cloudBenchmarkClient, workflowDir, configured string, ids []string) (string, error) {
	if strings.TrimSpace(configured) != "" {
		path, err := filepath.Abs(configured)
		if err != nil {
			return "", fmt.Errorf("resolve authorized keys file: %w", err)
		}
		if _, err := os.Stat(path); err != nil {
			return "", fmt.Errorf("read authorized keys file: %w", err)
		}
		return path, nil
	}
	if len(ids) == 0 {
		return "", fmt.Errorf("building the golden OS volume requires --authorized-keys-file or at least one --ssh-key-id")
	}

	keys, err := client.ListSSHKeys(ctx)
	if err != nil {
		return "", fmt.Errorf("list Verda SSH keys for Packer image: %w", err)
	}
	byID := make(map[string]verdacloud.SSHKey, len(keys))
	for _, key := range keys {
		byID[strings.TrimSpace(key.ID)] = key
	}
	var content strings.Builder
	for _, id := range ids {
		key, ok := byID[id]
		if !ok {
			return "", fmt.Errorf("verda SSH key %s was not found; run `nnctl cloud ssh-keys`", id)
		}
		publicKey := strings.TrimSpace(key.PublicKey)
		if publicKey == "" {
			return "", fmt.Errorf("verda SSH key %s did not include a public key; pass --authorized-keys-file", id)
		}
		content.WriteString(publicKey)
		content.WriteByte('\n')
	}
	path := filepath.Join(workflowDir, "authorized_keys")
	if err := os.WriteFile(path, []byte(content.String()), 0o600); err != nil {
		return "", fmt.Errorf("write temporary authorized keys file: %w", err)
	}
	return path, nil
}

func ensureCloudPackerTemplate(outputDir string, force bool) ([]string, error) {
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		return nil, fmt.Errorf("create Packer template directory: %w", err)
	}
	var written []string
	for _, file := range verdacloud.PackerTemplateFiles() {
		path := filepath.Join(outputDir, file.Path)
		current, err := os.ReadFile(path)
		if err == nil && !force {
			continue
		}
		if err != nil && !os.IsNotExist(err) {
			return nil, fmt.Errorf("read %s: %w", path, err)
		}
		if err == nil && bytes.Equal(current, []byte(file.Content)) {
			continue
		}
		if err := os.WriteFile(path, []byte(file.Content), 0o644); err != nil {
			return nil, fmt.Errorf("write %s: %w", path, err)
		}
		written = append(written, path)
	}
	return written, nil
}

func cloudBenchmarkPackerBuildArgs(templatePath, keysFile string, opts cloudBenchmarkDeployOptions, targetLocation string) ([]string, error) {
	template, err := os.ReadFile(templatePath)
	if err != nil {
		return nil, fmt.Errorf("read Packer template: %w", err)
	}
	args := []string{
		"build",
		"-var", "authorized_keys_file=" + keysFile,
		"-var", "instance_type=" + opts.packerInstanceType,
		"-var", "image=" + opts.Image,
		"-var", "hostname=packer-verda-zig-nn-" + strings.ToLower(targetLocation),
	}
	needsConfigurableTemplate := opts.SourceOSVolumeName != defaultCloudBenchmarkSourceName ||
		(targetLocation != cloudBenchmarkPackerBuildLocation && targetLocation != cloudBenchmarkPackerArtifactLocation) ||
		opts.Market != verdacloud.PricingMarketSpot ||
		strings.TrimSpace(opts.BaseURL) != ""
	if needsConfigurableTemplate {
		required := []string{`variable "artifact_volume_name"`, `variable "build_location_code"`, `variable "build_market"`, `variable "artifact_volume_location_codes"`, `variable "base_url"`}
		for _, declaration := range required {
			if !bytes.Contains(template, []byte(declaration)) {
				return nil, fmt.Errorf("existing %s does not support the requested source name, location, or base URL; pass --refresh-packer-template", templatePath)
			}
		}
		args = append(args,
			"-var", "artifact_volume_name="+opts.SourceOSVolumeName,
			"-var", "build_location_code="+targetLocation,
			"-var", "build_market="+opts.Market,
			"-var", fmt.Sprintf(`artifact_volume_location_codes=[%q]`, targetLocation),
			"-var", "base_url="+opts.BaseURL,
		)
	}
	return append(args, "."), nil
}

func reusableCloudBenchmarkSourceVolume(volumes []verdacloud.Volume, name, location string) (verdacloud.Volume, bool) {
	var matches []verdacloud.Volume
	for _, volume := range volumes {
		if strings.TrimSpace(volume.Name) != name || !volume.IsOSVolume || !strings.EqualFold(volume.Location, location) || cloudBenchmarkVolumeDeleted(volume) {
			continue
		}
		normalized, err := normalizeCloudBenchmarkSourceVolume(volume)
		if err == nil {
			matches = append(matches, normalized)
		}
	}
	if len(matches) == 0 {
		return verdacloud.Volume{}, false
	}
	sort.SliceStable(matches, func(i, j int) bool {
		if matches[i].CreatedAt != matches[j].CreatedAt {
			return matches[i].CreatedAt > matches[j].CreatedAt
		}
		return matches[i].ID < matches[j].ID
	})
	return matches[0], true
}

func normalizeCloudBenchmarkSourceVolume(volume verdacloud.Volume) (verdacloud.Volume, error) {
	volume.ID = strings.TrimSpace(volume.ID)
	volume.Name = strings.TrimSpace(volume.Name)
	volume.Location = strings.ToUpper(strings.TrimSpace(volume.Location))
	if volume.ID == "" || volume.Location == "" {
		return verdacloud.Volume{}, fmt.Errorf("source OS volume did not report an ID and location")
	}
	return volume, nil
}

func cloudBenchmarkVolumeDeleted(volume verdacloud.Volume) bool {
	if volume.IsPermanentlyDeleted || strings.TrimSpace(volume.DeletedAt) != "" {
		return true
	}
	switch strings.ToLower(strings.TrimSpace(volume.Status)) {
	case "deleted", "deleting":
		return true
	default:
		return false
	}
}

func sortCloudBenchmarkPlacements(placements []verdacloud.SpotPlacement) {
	sort.SliceStable(placements, func(i, j int) bool {
		left, right := placements[i], placements[j]
		if left.PriceKnown != right.PriceKnown {
			return left.PriceKnown
		}
		if left.PriceKnown && left.SpotPrice != right.SpotPrice {
			return left.SpotPrice < right.SpotPrice
		}
		return left.LocationCode < right.LocationCode
	})
}

func cloudBenchmarkTargetLocation(requested string, placements []verdacloud.SpotPlacement) (string, error) {
	requested = strings.ToUpper(strings.TrimSpace(requested))
	if requested != "" {
		for _, placement := range placements {
			if strings.EqualFold(placement.LocationCode, requested) {
				return requested, nil
			}
		}
		return "", fmt.Errorf("requested instance type is not currently available in %s", requested)
	}
	if len(placements) == 0 {
		return "", fmt.Errorf("requested instance type is not currently available in any location")
	}
	return strings.ToUpper(strings.TrimSpace(placements[0].LocationCode)), nil
}

func waitForCloudBenchmarkInstance(ctx context.Context, client cloudBenchmarkClient, instanceID string, timeout, poll time.Duration) (verdacloud.Instance, error) {
	deadline := time.Now().Add(timeout)
	for {
		instances, err := client.ListInstances(ctx, "")
		if err != nil {
			return verdacloud.Instance{}, fmt.Errorf("list Verda instances while waiting for %s: %w", instanceID, err)
		}
		for _, instance := range instances {
			if instance.ID != instanceID {
				continue
			}
			status := strings.ToLower(strings.TrimSpace(instance.Status))
			if cloudBenchmarkInstanceFailed(status) {
				return verdacloud.Instance{}, fmt.Errorf("benchmark instance %s entered terminal status %q", instanceID, instance.Status)
			}
			if status == "running" && instance.IP != nil && strings.TrimSpace(*instance.IP) != "" {
				return instance, nil
			}
		}
		if time.Now().After(deadline) {
			return verdacloud.Instance{}, fmt.Errorf("timed out waiting for benchmark instance %s to be running with an IP", instanceID)
		}
		timer := time.NewTimer(poll)
		select {
		case <-ctx.Done():
			timer.Stop()
			return verdacloud.Instance{}, ctx.Err()
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

func (a *App) runCloudBenchmarkSSH(ctx context.Context, ssh string, baseArgs []string, host, remoteCommand string, stdout io.Writer) error {
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

func (a *App) captureCloudBenchmarkSSH(ctx context.Context, ssh string, baseArgs []string, host, remoteCommand string) ([]byte, error) {
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

func (a *App) runCloudBenchmarkCommand(ctx context.Context, dir string, env []string, name string, args ...string) error {
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
	_, err := verdacloud.Destroy(ctx, client, verdacloud.DestroyOptions{
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
