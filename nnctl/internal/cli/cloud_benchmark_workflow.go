package cli

import (
	"context"
	"fmt"
	"io"
	"strings"
	"time"

	"nnctl/internal/cloud/verda"
	cloudworkflow "nnctl/internal/cloud/workflow"
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
	provider              string
	baseURL               string
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
	output  *errorWriter
	started bool
}

func newCloudBenchmarkProgress(dst io.Writer) *cloudBenchmarkProgress {
	return &cloudBenchmarkProgress{output: newErrorWriter(dst)}
}

func (p *cloudBenchmarkProgress) phase(number int, title string) {
	if p.started {
		p.output.printf("\n")
	}
	p.output.printf("[%d/%d] %s\n", number, cloudBenchmarkPhaseCount, title)
	p.started = true
}

func (p *cloudBenchmarkProgress) detail(format string, args ...any) {
	p.output.printf("      "+format+"\n", args...)
}

func (p *cloudBenchmarkProgress) Err() error {
	if err := p.output.Err(); err != nil {
		return fmt.Errorf("write cloud benchmark progress: %w", err)
	}
	return nil
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

func normalizeCloudBenchmarkDeployOptions(opts *cloudBenchmarkDeployOptions) error {
	normalizeCloudBenchmarkStrings(opts)
	if opts.SourceOSVolumeID == "" && opts.SourceOSVolumeName == "" {
		opts.SourceOSVolumeName = defaultCloudBenchmarkSourceName
	}
	return validateCloudBenchmarkDeployOptions(*opts)
}

func normalizeCloudBenchmarkStrings(opts *cloudBenchmarkDeployOptions) {
	opts.baseURL = strings.TrimSpace(opts.baseURL)
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
	for _, validate := range []func(cloudBenchmarkDeployOptions) error{
		validateCloudBenchmarkSelection,
		validateCloudBenchmarkTools,
	} {
		if err := validate(opts); err != nil {
			return err
		}
	}
	return verda.ValidateSSHKeyIDs(opts.SSHKeyIDs)
}

func validateCloudBenchmarkSelection(opts cloudBenchmarkDeployOptions) error {
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
		return nil
	default:
		return fmt.Errorf("benchmark GPU backend must be cuda, rocm, or none")
	}
}

func validateCloudBenchmarkTools(opts cloudBenchmarkDeployOptions) error {
	if !allConfigured(opts.packerDir, opts.packer, opts.packerInstanceType) {
		return fmt.Errorf("packer directory, executable, and instance type must be configured")
	}
	if opts.ref == "" {
		return fmt.Errorf("--ref must not be empty")
	}
	if !allConfigured(opts.sshUser, opts.ssh, opts.rsync, opts.git, opts.tar) {
		return fmt.Errorf("SSH user, ssh, rsync, git, and tar must be configured")
	}
	if err := cloudworkflow.ValidateSSHUser(opts.sshUser); err != nil {
		return err
	}
	if opts.timeout <= 0 || opts.pollInterval <= 0 {
		return fmt.Errorf("timeout and poll interval must be positive")
	}
	if opts.updateDocs && opts.docsPath == "" {
		return fmt.Errorf("--docs must not be empty with --update-docs")
	}
	return nil
}

func allConfigured(values ...string) bool {
	for _, value := range values {
		if value == "" {
			return false
		}
	}
	return true
}
