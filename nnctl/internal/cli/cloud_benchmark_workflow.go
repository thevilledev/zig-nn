package cli

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"nnctl/internal/cloud/verda"
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
	cloudBenchmarkPhaseCount             = 9
	cloudBenchmarkPhasePrepare           = 1
	cloudBenchmarkPhaseImage             = 2
	cloudBenchmarkPhaseDeploy            = 3
	cloudBenchmarkPhaseReady             = 4
	cloudBenchmarkPhaseUpload            = 5
	cloudBenchmarkPhaseSmoke             = 6
	cloudBenchmarkPhaseRun               = 7
	cloudBenchmarkPhaseResults           = 8
	cloudBenchmarkPhaseCleanup           = 9
)

type cloudBenchmarkDeployOptions struct {
	verda.DeployOptions
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
	verda.Client
	verda.DestroyClient
	ListInstances(context.Context, string) ([]verda.Instance, error)
	ListSSHKeys(context.Context) ([]verda.SSHKey, error)
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

type cloudBenchmarkProgress struct {
	dst     io.Writer
	started bool
}

func newCloudBenchmarkProgress(dst io.Writer) *cloudBenchmarkProgress {
	return &cloudBenchmarkProgress{dst: dst}
}

func (p *cloudBenchmarkProgress) phase(number int, title string) {
	if p.started {
		fmt.Fprintln(p.dst)
	}
	fmt.Fprintf(p.dst, "[%d/%d] %s\n", number, cloudBenchmarkPhaseCount, title)
	p.started = true
}

func (p *cloudBenchmarkProgress) detail(format string, args ...any) {
	fmt.Fprintf(p.dst, "      "+format+"\n", args...)
}

func defaultCloudBenchmarkDeployOptions() cloudBenchmarkDeployOptions {
	deployOpts := verda.DefaultDeployOptions("")
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

func (a *app) runCloudBenchmarkDeploy(ctx context.Context, opts cloudBenchmarkDeployOptions) (runErr error) {
	if err := normalizeCloudBenchmarkDeployOptions(&opts); err != nil {
		return err
	}
	progress := newCloudBenchmarkProgress(a.stderr())
	progress.phase(cloudBenchmarkPhasePrepare, "Prepare benchmark workflow")
	progress.detail("Checking git state and resolving %s", opts.ref)

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
	progress.detail("Revision: %s", commit)
	if opts.authorizedKeysFile != "" {
		opts.authorizedKeysFile = resolveRepoPath(a.repoRoot, opts.authorizedKeysFile)
	}

	progress.phase(cloudBenchmarkPhaseImage, "Resolve golden OS volume")
	packerDir := resolveRepoPath(a.repoRoot, opts.packerDir)
	written, err := ensureCloudPackerTemplate(packerDir, opts.refreshPackerTemplate)
	if err != nil {
		return err
	}
	for _, path := range written {
		progress.detail("Wrote Packer file: %s", path)
	}
	if len(written) == 0 {
		progress.detail("Packer template: %s", packerDir)
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

	sourceVolume, err := a.prepareCloudBenchmarkSourceVolume(ctx, client, creds, packerDir, workflowDir, &opts, progress)
	if err != nil {
		return err
	}
	if opts.LocationCode != "" && !strings.EqualFold(opts.LocationCode, sourceVolume.Location) {
		return fmt.Errorf("source OS volume %s is in %s, but --location-code requested %s", sourceVolume.ID, sourceVolume.Location, opts.LocationCode)
	}
	opts.SourceOSVolumeID = sourceVolume.ID
	opts.SourceOSVolumeName = ""
	opts.LocationCode = sourceVolume.Location
	progress.detail("Using volume: %s (%s in %s)", sourceVolume.ID, sourceVolume.Name, sourceVolume.Location)

	progress.phase(cloudBenchmarkPhaseDeploy, "Deploy cloud worker")
	progress.detail("Instance: %s, market: %s, location: %s", opts.InstanceType, opts.Market, opts.LocationCode)
	result, err := verda.Deploy(ctx, client, opts.DeployOptions)
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
	progress.detail("Worker: %s, cloned volume: %s", instanceID, cloneID)
	defer func() {
		progress.phase(cloudBenchmarkPhaseCleanup, "Clean up cloud resources")
		if opts.keepInstance {
			progress.detail("Skipped because --keep-instance was set")
			progress.detail("Worker: %s, cloned volume: %s", instanceID, cloneID)
			return
		}
		progress.detail("Destroying worker %s and cloned volume %s", instanceID, cloneID)
		cleanupCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Minute)
		cleanupErr := cleanupCloudBenchmark(cleanupCtx, client, opts.BaseURL, instanceID, cloneID, sourceVolume.ID)
		cancel()
		if cleanupErr != nil {
			progress.detail("Cleanup failed: %v", cleanupErr)
			runErr = errors.Join(runErr, cleanupErr)
			return
		}
		progress.detail("Cleanup complete")
	}()

	progress.phase(cloudBenchmarkPhaseReady, "Wait for worker readiness")
	progress.detail("Waiting for an IP address and SSH access")
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
	progress.detail("Worker ready: %s", host)

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
	progress.phase(cloudBenchmarkPhaseUpload, "Upload source snapshot")
	progress.detail("%s -> %s", opts.ref, deployOpts.target)
	if err := a.runDeploy(ctx, deployOpts); err != nil {
		return err
	}

	progress.phase(cloudBenchmarkPhaseSmoke, "Run quick smoke benchmark")
	if opts.quick {
		progress.detail("Skipped because --quick makes the captured run smoke-sized")
	} else if opts.skipSmoke {
		progress.detail("Skipped because --skip-smoke was set")
	} else {
		if err := a.runCloudBenchmarkSSH(ctx, opts.ssh, sshArgs, host, remoteBenchmarkCommand(remoteDir, opts.backend, "", true), a.stdout()); err != nil {
			return fmt.Errorf("remote quick benchmark failed: %w", err)
		}
		progress.detail("Smoke benchmark passed")
	}

	progress.phase(cloudBenchmarkPhaseRun, "Run captured benchmark")
	if opts.filter != "" {
		progress.detail("Backend: %s, filter: %s, quick: %t", opts.backend, opts.filter, opts.quick)
	} else {
		progress.detail("Backend: %s, filter: all default suites, quick: %t", opts.backend, opts.quick)
	}
	benchmarkOutput, err := a.captureCloudBenchmarkSSH(ctx, opts.ssh, sshArgs, host, remoteBenchmarkCommand(remoteDir, opts.backend, opts.filter, opts.quick))
	if err != nil {
		return fmt.Errorf("remote benchmark failed: %w", err)
	}

	progress.phase(cloudBenchmarkPhaseResults, "Collect and save results")
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
		Provider:       verda.ProviderName,
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
		progress.detail("Warning: remote metadata could not be collected: %v", metadataErr)
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
	progress.detail("CSV: %s", outputPath)
	fmt.Fprintf(a.stdout(), "benchmark CSV: %s\n", outputPath)

	if opts.updateDocs {
		docsPath := resolveRepoPath(a.repoRoot, opts.docsPath)
		if err := updateCloudBenchmarkDocs(docsPath, outputPath, metadata, rows); err != nil {
			return err
		}
		progress.detail("Docs: %s", docsPath)
		fmt.Fprintf(a.stdout(), "updated benchmark docs: %s\n", docsPath)
	}
	return nil
}

func normalizeCloudBenchmarkDeployOptions(opts *cloudBenchmarkDeployOptions) error {
	normalizeCloudBenchmarkStrings(opts)
	if opts.SourceOSVolumeID == "" && opts.SourceOSVolumeName == "" {
		opts.SourceOSVolumeName = defaultCloudBenchmarkSourceName
	}
	return validateCloudBenchmarkDeployOptions(*opts)
}

func normalizeCloudBenchmarkStrings(opts *cloudBenchmarkDeployOptions) {
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
}

func validateCloudBenchmarkDeployOptions(opts cloudBenchmarkDeployOptions) error {
	if opts.InstanceType == "" {
		return fmt.Errorf("instance type is required")
	}
	if opts.SourceOSVolumeID != "" && opts.SourceOSVolumeName != "" {
		return fmt.Errorf("source OS volume ID and name cannot be combined")
	}
	if opts.Market != verda.PricingMarketSpot && opts.Market != verda.PricingMarketOnDemand {
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
	if err := verda.ValidateSSHKeyIDs(opts.SSHKeyIDs); err != nil {
		return err
	}
	return nil
}
