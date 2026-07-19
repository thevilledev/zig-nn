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

	cloudcore "nnctl/internal/cloud"
	"nnctl/internal/cloud/digitalocean"
	"nnctl/internal/cloud/verda"
	cloudworkflow "nnctl/internal/cloud/workflow"
)

type cloudBenchmarkOperations struct {
	gitStatus            func(context.Context, string) (string, error)
	gitRevision          func(context.Context, string, string) (string, error)
	ensurePackerTemplate func(string, bool) ([]string, error)
	newClient            func(context.Context, string) (cloudBenchmarkClient, verda.Credentials, error)
	prepareSourceVolume  func(context.Context, cloudBenchmarkClient, verda.Credentials, string, string, *cloudBenchmarkDeployOptions, *cloudBenchmarkProgress) (verda.Volume, error)
	deploy               func(context.Context, verda.Client, verda.DeployOptions) (*verda.DeployResult, error)
	waitInstance         func(context.Context, cloudBenchmarkClient, string, time.Duration, time.Duration) (verda.Instance, error)
	waitSSH              func(context.Context, string, []string, string, time.Duration, time.Duration) error
	waitCommand          func(context.Context, string, []string, string, string, time.Duration, time.Duration) error
	upload               func(context.Context, deployOptions) error
	runSSH               func(context.Context, string, []string, string, string, io.Writer) error
	captureSSH           func(context.Context, string, []string, string, string) ([]byte, error)
	cleanup              func(context.Context, cloudBenchmarkClient, string, string, string) error
}

func newCloudBenchmarkOperations(a *app) cloudBenchmarkOperations {
	return cloudBenchmarkOperations{
		gitStatus: a.gitStatusPorcelain,
		gitRevision: func(ctx context.Context, git, ref string) (string, error) {
			return gitRevision(ctx, a.repoRoot, git, ref)
		},
		ensurePackerTemplate: ensureCloudPackerTemplate,
		newClient: func(ctx context.Context, baseURL string) (cloudBenchmarkClient, verda.Credentials, error) {
			client, credentials, err := a.newVerdaSDKClientWithCredentials(ctx, baseURL)
			return client, credentials, err
		},
		prepareSourceVolume: a.prepareCloudBenchmarkSourceVolume,
		deploy:              verda.Deploy,
		waitInstance:        waitForCloudBenchmarkInstance,
		waitSSH:             waitForCloudBenchmarkSSH,
		waitCommand:         cloudworkflow.WaitCommand,
		upload:              a.runDeploy,
		runSSH:              a.runCloudBenchmarkSSH,
		captureSSH:          a.captureCloudBenchmarkSSH,
		cleanup:             cleanupCloudBenchmark,
	}
}

type cloudBenchmarkRun struct {
	app      *app
	ctx      context.Context
	opts     cloudBenchmarkDeployOptions
	progress *cloudBenchmarkProgress
	ops      cloudBenchmarkOperations

	commit          string
	stamp           time.Time
	packerDir       string
	workflowDir     string
	client          cloudBenchmarkClient
	credentials     verda.Credentials
	sourceVolume    verda.Volume
	result          *verda.DeployResult
	instance        verda.Instance
	instanceID      string
	cloneID         string
	host            string
	sshArgs         []string
	remoteDir       string
	benchmarkOutput []byte
	rows            []benchmarkRow
	metadata        cloudBenchmarkMetadata
	strategy        cloudBenchmarkStrategy
}

type cloudBenchmarkStrategy interface {
	prepareImage(*cloudBenchmarkRun) error
	deploy(*cloudBenchmarkRun) error
	wait(*cloudBenchmarkRun) error
	metadata(*cloudBenchmarkRun) cloudBenchmarkMetadata
	cleanup(context.Context, *cloudBenchmarkRun) error
	resources(*cloudBenchmarkRun) string
}

type verdaCloudBenchmarkStrategy struct{}

func newCloudBenchmarkRun(a *app, ctx context.Context, opts cloudBenchmarkDeployOptions) *cloudBenchmarkRun {
	return &cloudBenchmarkRun{
		app:      a,
		ctx:      ctx,
		opts:     opts,
		progress: newCloudBenchmarkProgress(a.stderr()),
		ops:      newCloudBenchmarkOperations(a),
	}
}

func (a *app) runCloudBenchmarkDeploy(ctx context.Context, opts cloudBenchmarkDeployOptions) (runErr error) {
	factory, err := a.cloudProviderFactory(opts.provider)
	if err != nil {
		return err
	}
	if !factory.Descriptor().Supports(cloudcore.CapabilityBenchmarkImage) {
		return unsupportedCloudCapability(factory.Descriptor(), cloudcore.CapabilityBenchmarkImage)
	}
	if err := normalizeCloudBenchmarkDeployOptionsForProvider(&opts, factory.Descriptor().Name); err != nil {
		return err
	}
	run := newCloudBenchmarkRun(a, ctx, opts)
	switch factory.Descriptor().Name {
	case verda.ProviderName:
		run.strategy = verdaCloudBenchmarkStrategy{}
	case digitalocean.ProviderName:
		run.strategy = &digitalOceanCloudBenchmarkStrategy{}
	default:
		return fmt.Errorf("cloud provider %q does not implement benchmark deployment", factory.Descriptor().Name)
	}
	if err := run.prepare(); err != nil {
		return err
	}
	if err := run.prepareProviderImage(); err != nil {
		return err
	}
	defer func() {
		runErr = errors.Join(runErr, run.removeWorkflowDir())
	}()
	if err := run.deployWorker(); err != nil {
		return err
	}
	defer func() {
		runErr = errors.Join(runErr, run.cleanupResources())
	}()

	for _, step := range []func() error{
		run.waitForWorker,
		run.uploadSource,
		run.runSmokeBenchmark,
		run.runCapturedBenchmark,
		run.collectResults,
	} {
		if err := step(); err != nil {
			return err
		}
	}
	return nil
}

func (r *cloudBenchmarkRun) selectedStrategy() cloudBenchmarkStrategy {
	if r.strategy == nil {
		return verdaCloudBenchmarkStrategy{}
	}
	return r.strategy
}

func (r *cloudBenchmarkRun) prepareProviderImage() error {
	return r.selectedStrategy().prepareImage(r)
}

func (verdaCloudBenchmarkStrategy) prepareImage(r *cloudBenchmarkRun) error {
	return r.resolveSourceVolume()
}

func (r *cloudBenchmarkRun) prepare() error {
	r.progress.phase(cloudBenchmarkPhasePrepare, "Prepare benchmark workflow")
	r.progress.detail("Checking git state and resolving %s", r.opts.ref)
	if err := r.progress.Err(); err != nil {
		return err
	}
	if !r.opts.ignoreDirty {
		status, err := r.ops.gitStatus(r.ctx, r.opts.git)
		if err != nil {
			return err
		}
		if status != "" {
			return fmt.Errorf("working tree has uncommitted changes; commit them or pass --ignore-dirty to benchmark %s anyway", r.opts.ref)
		}
	}

	commit, err := r.ops.gitRevision(r.ctx, r.opts.git, r.opts.ref)
	if err != nil {
		return err
	}
	r.commit = commit
	r.stamp = r.app.currentTime().UTC()
	r.progress.detail("Revision: %s", r.commit)
	if r.opts.authorizedKeysFile != "" {
		r.opts.authorizedKeysFile = resolveRepoPath(r.app.repoRoot, r.opts.authorizedKeysFile)
	}
	return r.progress.Err()
}

func (r *cloudBenchmarkRun) resolveSourceVolume() (runErr error) {
	r.progress.phase(cloudBenchmarkPhaseImage, "Resolve golden OS volume")
	if err := r.progress.Err(); err != nil {
		return err
	}
	r.packerDir = resolveRepoPath(r.app.repoRoot, r.opts.packerDir)
	written, err := r.ops.ensurePackerTemplate(r.packerDir, r.opts.refreshPackerTemplate)
	if err != nil {
		return err
	}
	if err := r.reportPackerTemplate(written); err != nil {
		return err
	}

	r.client, r.credentials, err = r.ops.newClient(r.ctx, r.opts.baseURL)
	if err != nil {
		return err
	}
	if err := r.ensureWorkflowDir(); err != nil {
		return err
	}
	defer func() {
		if runErr != nil {
			runErr = errors.Join(runErr, r.removeWorkflowDir())
		}
	}()

	r.sourceVolume, err = r.ops.prepareSourceVolume(
		r.ctx,
		r.client,
		r.credentials,
		r.packerDir,
		r.workflowDir,
		&r.opts,
		r.progress,
	)
	if err != nil {
		return err
	}
	if r.opts.LocationCode != "" && !strings.EqualFold(r.opts.LocationCode, r.sourceVolume.Location) {
		return fmt.Errorf("source OS volume %s is in %s, but --location-code requested %s", r.sourceVolume.ID, r.sourceVolume.Location, r.opts.LocationCode)
	}
	r.opts.SourceOSVolumeID = r.sourceVolume.ID
	r.opts.SourceOSVolumeName = ""
	r.opts.LocationCode = r.sourceVolume.Location
	r.progress.detail("Using volume: %s (%s in %s)", r.sourceVolume.ID, r.sourceVolume.Name, r.sourceVolume.Location)
	return r.progress.Err()
}

func (r *cloudBenchmarkRun) ensureWorkflowDir() error {
	if r.workflowDir != "" {
		return nil
	}
	directory, err := os.MkdirTemp("", "nnctl-cloud-benchmark-*")
	if err != nil {
		return fmt.Errorf("create cloud benchmark work directory: %w", err)
	}
	r.workflowDir = directory
	return nil
}

func (r *cloudBenchmarkRun) reportPackerTemplate(written []string) error {
	for _, path := range written {
		r.progress.detail("Wrote Packer file: %s", path)
	}
	if len(written) == 0 {
		r.progress.detail("Packer template: %s", r.packerDir)
	}
	return r.progress.Err()
}

func (r *cloudBenchmarkRun) deployWorker() error {
	return r.selectedStrategy().deploy(r)
}

func (verdaCloudBenchmarkStrategy) deploy(r *cloudBenchmarkRun) error {
	return r.deployVerdaWorker()
}

func (r *cloudBenchmarkRun) deployVerdaWorker() error {
	r.progress.phase(cloudBenchmarkPhaseDeploy, "Deploy cloud worker")
	r.progress.detail("Instance: %s, market: %s, location: %s", r.opts.InstanceType, r.opts.Market, r.opts.LocationCode)
	if err := r.progress.Err(); err != nil {
		return err
	}
	result, err := r.ops.deploy(r.ctx, r.client, r.opts.DeployOptions)
	if err != nil {
		return err
	}
	if result == nil || result.Instance == nil || strings.TrimSpace(result.Instance.ID) == "" {
		return fmt.Errorf("verda deployment did not return an instance")
	}
	r.result = result
	r.instanceID = result.Instance.ID
	if result.OSVolumeClone != nil {
		r.cloneID = result.OSVolumeClone.VolumeID
	}
	r.progress.detail("Worker: %s, cloned volume: %s", r.instanceID, r.cloneID)
	return r.progress.Err()
}

func (r *cloudBenchmarkRun) waitForWorker() error {
	return r.selectedStrategy().wait(r)
}

func (verdaCloudBenchmarkStrategy) wait(r *cloudBenchmarkRun) error {
	return r.waitForVerdaWorker()
}

func (r *cloudBenchmarkRun) waitForVerdaWorker() error {
	r.progress.phase(cloudBenchmarkPhaseReady, "Wait for worker readiness")
	r.progress.detail("Waiting for an IP address and SSH access")
	if err := r.progress.Err(); err != nil {
		return err
	}
	instance, err := r.ops.waitInstance(r.ctx, r.client, r.instanceID, r.opts.timeout, r.opts.pollInterval)
	if err != nil {
		return err
	}
	r.instance = instance
	r.host, err = cloudworkflow.SSHDestination(r.opts.sshUser, *instance.IP)
	if err != nil {
		return err
	}
	r.sshArgs = cloudBenchmarkSSHArgs(filepath.Join(r.workflowDir, "known_hosts"))
	if err := r.ops.waitSSH(r.ctx, r.opts.ssh, r.sshArgs, r.host, r.opts.timeout, r.opts.pollInterval); err != nil {
		return err
	}
	r.progress.detail("Worker ready: %s", r.host)
	return r.progress.Err()
}

func (r *cloudBenchmarkRun) uploadSource() error {
	r.remoteDir = strings.TrimSpace(r.opts.remoteDir)
	if r.remoteDir == "" {
		r.remoteDir = r.defaultRemoteDir()
	}
	deployOpts := defaultDeployOptions()
	deployOpts.target = r.host + ":" + r.remoteDir
	deployOpts.ref = r.opts.ref
	deployOpts.delete = true
	deployOpts.ignoreDirty = true
	deployOpts.ssh = commandString(r.opts.ssh, r.sshArgs)
	deployOpts.git = r.opts.git
	deployOpts.tar = r.opts.tar
	deployOpts.rsync = r.opts.rsync
	r.progress.phase(cloudBenchmarkPhaseUpload, "Upload source snapshot")
	r.progress.detail("%s -> %s", r.opts.ref, deployOpts.target)
	if err := r.progress.Err(); err != nil {
		return err
	}
	return r.ops.upload(r.ctx, deployOpts)
}

func (r *cloudBenchmarkRun) defaultRemoteDir() string {
	root := "/tmp"
	if r.opts.sshUser == "root" {
		root = "/root"
	}
	return root + "/zig-nn-bench-" + slugCloudBenchmark(r.commit) + "-" + r.stamp.Format("20060102T150405Z")
}

func (r *cloudBenchmarkRun) runSmokeBenchmark() error {
	r.progress.phase(cloudBenchmarkPhaseSmoke, "Run quick smoke benchmark")
	if err := r.progress.Err(); err != nil {
		return err
	}
	if r.opts.quick {
		r.progress.detail("Skipped because --quick makes the captured run smoke-sized")
		return r.progress.Err()
	}
	if r.opts.skipSmoke {
		r.progress.detail("Skipped because --skip-smoke was set")
		return r.progress.Err()
	}
	command := remoteBenchmarkCommand(r.remoteDir, r.opts.backend, "", true)
	if err := r.ops.runSSH(r.ctx, r.opts.ssh, r.sshArgs, r.host, command, r.app.stdout()); err != nil {
		return fmt.Errorf("remote quick benchmark failed: %w", err)
	}
	r.progress.detail("Smoke benchmark passed")
	return r.progress.Err()
}

func (r *cloudBenchmarkRun) runCapturedBenchmark() error {
	r.progress.phase(cloudBenchmarkPhaseRun, "Run captured benchmark")
	if r.opts.filter != "" {
		r.progress.detail("Backend: %s, filter: %s, quick: %t", r.opts.backend, r.opts.filter, r.opts.quick)
	} else {
		r.progress.detail("Backend: %s, filter: all default suites, quick: %t", r.opts.backend, r.opts.quick)
	}
	if err := r.progress.Err(); err != nil {
		return err
	}
	command := remoteBenchmarkCommand(r.remoteDir, r.opts.backend, r.opts.filter, r.opts.quick)
	output, err := r.ops.captureSSH(r.ctx, r.opts.ssh, r.sshArgs, r.host, command)
	if err != nil {
		return fmt.Errorf("remote benchmark failed: %w", err)
	}
	r.benchmarkOutput = output
	return nil
}

func (r *cloudBenchmarkRun) collectResults() error {
	r.progress.phase(cloudBenchmarkPhaseResults, "Collect and save results")
	if err := r.progress.Err(); err != nil {
		return err
	}
	rows, err := parseBenchmarkCSV(r.benchmarkOutput)
	if err != nil {
		return err
	}
	r.rows = rows
	r.metadata = r.selectedStrategy().metadata(r)
	if err := r.collectRemoteMetadata(); err != nil {
		return err
	}

	outputPath := strings.TrimSpace(r.opts.output)
	if outputPath == "" {
		outputPath = defaultCloudBenchmarkOutputPath(r.app.repoRoot, r.metadata)
	} else {
		outputPath = resolveRepoPath(r.app.repoRoot, outputPath)
	}
	if err := writeCloudBenchmarkCSV(outputPath, r.metadata, r.benchmarkOutput); err != nil {
		return err
	}
	r.progress.detail("CSV: %s", outputPath)
	if _, err := fmt.Fprintf(r.app.stdout(), "benchmark CSV: %s\n", outputPath); err != nil {
		return fmt.Errorf("write benchmark CSV path: %w", err)
	}

	if !r.opts.updateDocs {
		return r.progress.Err()
	}
	docsPath := resolveRepoPath(r.app.repoRoot, r.opts.docsPath)
	if err := updateCloudBenchmarkDocs(docsPath, outputPath, r.metadata, r.rows); err != nil {
		return err
	}
	r.progress.detail("Docs: %s", docsPath)
	if _, err := fmt.Fprintf(r.app.stdout(), "updated benchmark docs: %s\n", docsPath); err != nil {
		return fmt.Errorf("write benchmark docs path: %w", err)
	}
	return r.progress.Err()
}

func (verdaCloudBenchmarkStrategy) metadata(r *cloudBenchmarkRun) cloudBenchmarkMetadata {
	pricePerHour := r.instance.PricePerHour
	if pricePerHour == 0 && r.result.Placement != nil {
		pricePerHour = r.result.Placement.SpotPrice
	}
	metadata := cloudBenchmarkMetadata{
		Timestamp:      r.stamp,
		Commit:         r.commit,
		Provider:       verda.ProviderName,
		InstanceID:     r.instanceID,
		InstanceType:   r.opts.InstanceType,
		Location:       firstNonEmpty(r.instance.Location, r.sourceVolume.Location),
		Market:         r.opts.Market,
		PricePerHour:   pricePerHour,
		SourceVolumeID: r.sourceVolume.ID,
		ClonedVolumeID: r.cloneID,
		Backend:        r.opts.backend,
	}
	if r.result.InstanceType != nil {
		metadata.GPU = r.result.InstanceType.Model
	}
	return metadata
}

func (r *cloudBenchmarkRun) collectRemoteMetadata() error {
	command := remoteBenchmarkMetadataCommand()
	metadata, err := r.ops.captureSSH(r.ctx, r.opts.ssh, r.sshArgs, r.host, command)
	if err != nil {
		r.progress.detail("Warning: remote metadata could not be collected: %v", err)
		return r.progress.Err()
	}
	mergeCloudBenchmarkRemoteMetadata(&r.metadata, metadata)
	return nil
}

func (r *cloudBenchmarkRun) cleanupResources() error {
	r.progress.phase(cloudBenchmarkPhaseCleanup, "Clean up cloud resources")
	progressErr := r.progress.Err()
	if r.opts.keepInstance {
		r.progress.detail("Skipped because --keep-instance was set")
		r.progress.detail("Retained: %s", r.selectedStrategy().resources(r))
		return errors.Join(progressErr, r.progress.Err())
	}
	r.progress.detail("Destroying %s", r.selectedStrategy().resources(r))
	progressErr = errors.Join(progressErr, r.progress.Err())
	cleanupCtx, cancel := context.WithTimeout(context.WithoutCancel(r.ctx), 5*time.Minute)
	defer cancel()
	err := r.selectedStrategy().cleanup(cleanupCtx, r)
	if err != nil {
		r.progress.detail("Cleanup failed: %v", err)
		return errors.Join(progressErr, err, r.progress.Err())
	}
	r.progress.detail("Cleanup complete")
	return errors.Join(progressErr, r.progress.Err())
}

func (verdaCloudBenchmarkStrategy) cleanup(ctx context.Context, r *cloudBenchmarkRun) error {
	return r.ops.cleanup(ctx, r.client, r.instanceID, r.cloneID, r.sourceVolume.ID)
}

func (verdaCloudBenchmarkStrategy) resources(r *cloudBenchmarkRun) string {
	return fmt.Sprintf("worker %s and cloned volume %s", r.instanceID, r.cloneID)
}

func (r *cloudBenchmarkRun) removeWorkflowDir() error {
	if r.workflowDir == "" {
		return nil
	}
	path := r.workflowDir
	r.workflowDir = ""
	if err := os.RemoveAll(path); err != nil {
		return fmt.Errorf("remove cloud benchmark work directory: %w", err)
	}
	return nil
}
