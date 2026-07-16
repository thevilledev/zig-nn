package lab

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"time"

	"nnctl/internal/cloud/verda"
	cloudworkflow "nnctl/internal/cloud/workflow"
	"nnctl/internal/zig"
)

const (
	defaultCloudSourceVolumeName = "packer-verda-zig-nn-volume-root"
	defaultCloudSSHUser          = "root"
	defaultCloudTimeout          = 20 * time.Minute
	defaultCloudPollInterval     = 10 * time.Second
	defaultCloudIdleTimeout      = 30 * time.Minute
)

type CloudWorkerState string

const (
	CloudWorkerProvisioning CloudWorkerState = "provisioning"
	CloudWorkerReady        CloudWorkerState = "ready"
	CloudWorkerBusy         CloudWorkerState = "busy"
	CloudWorkerDestroying   CloudWorkerState = "destroying"
	CloudWorkerFailed       CloudWorkerState = "failed"
	CloudWorkerDestroyed    CloudWorkerState = "destroyed"
)

type CloudWorker struct {
	ID               string           `json:"id"`
	Provider         string           `json:"provider"`
	State            CloudWorkerState `json:"state"`
	Message          string           `json:"message"`
	InstanceID       string           `json:"instance_id,omitempty"`
	InstanceType     string           `json:"instance_type"`
	Hostname         string           `json:"hostname,omitempty"`
	Location         string           `json:"location,omitempty"`
	Market           string           `json:"market"`
	IP               string           `json:"ip,omitempty"`
	Backends         []string         `json:"backends"`
	PricePerHour     float64          `json:"price_per_hour,omitempty"`
	Currency         string           `json:"currency,omitempty"`
	AutoDestroy      bool             `json:"auto_destroy"`
	CreatedAt        time.Time        `json:"created_at"`
	ReadyAt          *time.Time       `json:"ready_at,omitempty"`
	ExpiresAt        *time.Time       `json:"expires_at,omitempty"`
	RepositoryCommit string           `json:"repository_commit,omitempty"`
}

type managedCloudWorker struct {
	CloudWorker
	SourceVolumeID string `json:"source_volume_id,omitempty"`
	CloneVolumeID  string `json:"clone_volume_id,omitempty"`
	KnownHosts     string `json:"known_hosts,omitempty"`
	Host           string `json:"host,omitempty"`
	BaseURL        string `json:"base_url,omitempty"`
	recovered      bool
}

type CloudStatus struct {
	Enabled    bool                          `json:"enabled"`
	Configured bool                          `json:"configured"`
	Provider   string                        `json:"provider"`
	Error      string                        `json:"error,omitempty"`
	Repository cloudworkflow.RepositoryState `json:"repository"`
}

type CloudOptions struct {
	Prices             []verda.InstancePrice `json:"prices"`
	Provider           string                `json:"provider"`
	SourceOSVolumeName string                `json:"source_os_volume_name"`
}

type CloudDeployRequest struct {
	InstanceType string `json:"instance_type"`
	Market       string `json:"market"`
	LocationCode string `json:"location_code,omitempty"`
	AutoDestroy  *bool  `json:"auto_destroy,omitempty"`
}

type cloudClientFactory func(context.Context, string) (cloudworkflow.Client, error)

type CloudManagerOptions struct {
	RepoRoot      string
	BaseURL       string
	SSH           string
	Git           string
	Tar           string
	Rsync         string
	SSHUser       string
	Mode          string
	JournalPath   string
	Timeout       time.Duration
	PollInterval  time.Duration
	IdleTimeout   time.Duration
	ClientFactory cloudClientFactory
	Now           func() time.Time
}

type CloudManager struct {
	repoRoot     string
	baseURL      string
	ssh          string
	git          string
	tar          string
	rsync        string
	sshUser      string
	mode         string
	journalPath  string
	timeout      time.Duration
	pollInterval time.Duration
	idleTimeout  time.Duration
	newClient    cloudClientFactory
	now          func() time.Time

	mu                sync.Mutex
	destroyMu         sync.Mutex
	worker            *managedCloudWorker
	workerSubscribers map[chan CloudWorker]struct{}
	ctx               context.Context
	cancel            context.CancelFunc
	wg                sync.WaitGroup
	closed            bool
	provisionWorkerID string
	provisionCancel   context.CancelFunc
	provisionDone     chan struct{}
}

func NewCloudManager(options CloudManagerOptions) (*CloudManager, error) {
	if strings.TrimSpace(options.RepoRoot) == "" {
		return nil, errors.New("cloud manager repository root is required")
	}
	if options.SSH == "" {
		options.SSH = "ssh"
	}
	if options.Git == "" {
		options.Git = "git"
	}
	if options.Tar == "" {
		options.Tar = "tar"
	}
	if options.Rsync == "" {
		options.Rsync = "rsync"
	}
	if options.SSHUser == "" {
		options.SSHUser = defaultCloudSSHUser
	}
	if options.Timeout <= 0 {
		options.Timeout = defaultCloudTimeout
	}
	if options.PollInterval <= 0 {
		options.PollInterval = defaultCloudPollInterval
	}
	if options.IdleTimeout <= 0 {
		options.IdleTimeout = defaultCloudIdleTimeout
	}
	if options.ClientFactory == nil {
		options.ClientFactory = cloudworkflow.NewVerdaClient
	}
	if options.Now == nil {
		options.Now = time.Now
	}
	if options.JournalPath == "" {
		cache, err := os.UserCacheDir()
		if err != nil {
			return nil, fmt.Errorf("resolve user cache for cloud worker journal: %w", err)
		}
		options.JournalPath = filepath.Join(cache, "nnctl", "lab-cloud-worker.json")
	}
	manager := &CloudManager{
		repoRoot:          options.RepoRoot,
		baseURL:           strings.TrimSpace(options.BaseURL),
		ssh:               options.SSH,
		git:               options.Git,
		tar:               options.Tar,
		rsync:             options.Rsync,
		sshUser:           options.SSHUser,
		mode:              options.Mode,
		journalPath:       options.JournalPath,
		timeout:           options.Timeout,
		pollInterval:      options.PollInterval,
		idleTimeout:       options.IdleTimeout,
		newClient:         options.ClientFactory,
		now:               options.Now,
		workerSubscribers: make(map[chan CloudWorker]struct{}),
	}
	manager.ctx, manager.cancel = context.WithCancel(context.Background())
	if err := manager.loadJournal(); err != nil {
		return nil, err
	}
	return manager, nil
}

func (m *CloudManager) Status(ctx context.Context) CloudStatus {
	status := CloudStatus{Enabled: true, Provider: verda.ProviderName}
	repository, err := cloudworkflow.InspectRepository(ctx, m.repoRoot, m.git)
	if err != nil {
		status.Error = err.Error()
		return status
	}
	status.Repository = repository
	if _, err := m.newClient(ctx, m.baseURL); err != nil {
		status.Error = err.Error()
		return status
	}
	status.Configured = true
	return status
}

func (m *CloudManager) Options(ctx context.Context) (CloudOptions, error) {
	client, err := m.newClient(ctx, m.baseURL)
	if err != nil {
		return CloudOptions{}, err
	}
	prices, err := client.ListInstancePrices(ctx, verda.PricingFilters{
		GPUCounts:     []int{1},
		Manufacturer:  "nvidia",
		Market:        verda.PricingMarketAll,
		AvailableOnly: true,
	})
	if err != nil {
		return CloudOptions{}, fmt.Errorf("list Verda prices: %w", err)
	}
	volumes, err := verda.ListVolumes(ctx, client, verda.ListVolumesOptions{})
	if err != nil {
		return CloudOptions{}, fmt.Errorf("list Verda volumes: %w", err)
	}
	goldenLocations := make(map[string]struct{})
	for _, volume := range volumes {
		if strings.EqualFold(strings.TrimSpace(volume.Name), defaultCloudSourceVolumeName) && volume.IsOSVolume && cloudVolumeReady(volume.Status) {
			goldenLocations[strings.ToUpper(strings.TrimSpace(volume.Location))] = struct{}{}
		}
	}
	compatiblePrices := make([]verda.InstancePrice, 0, len(prices))
	for _, price := range prices {
		if _, ok := goldenLocations[strings.ToUpper(strings.TrimSpace(price.LocationCode))]; ok {
			compatiblePrices = append(compatiblePrices, price)
		}
	}
	prices = compatiblePrices
	verda.SortInstancePrices(prices, "price")
	return CloudOptions{Prices: prices, Provider: verda.ProviderName, SourceOSVolumeName: defaultCloudSourceVolumeName}, nil
}

func cloudVolumeReady(status string) bool {
	switch strings.ToLower(strings.TrimSpace(status)) {
	case "available", "created", "ready", "detached", "in-use", "in_use":
		return true
	default:
		return false
	}
}

func (m *CloudManager) Deploy(ctx context.Context, request CloudDeployRequest) (CloudWorker, error) {
	request.InstanceType = strings.TrimSpace(request.InstanceType)
	request.Market = strings.ToLower(strings.TrimSpace(request.Market))
	request.LocationCode = strings.ToUpper(strings.TrimSpace(request.LocationCode))
	if request.InstanceType == "" {
		return CloudWorker{}, errors.New("instance type is required")
	}
	if request.Market == "" {
		request.Market = verda.PricingMarketSpot
	}
	if request.Market != verda.PricingMarketSpot && request.Market != verda.PricingMarketOnDemand {
		return CloudWorker{}, errors.New("market must be spot or on-demand")
	}
	if err := ctx.Err(); err != nil {
		return CloudWorker{}, err
	}
	autoDestroy := true
	if request.AutoDestroy != nil {
		autoDestroy = *request.AutoDestroy
	}

	m.mu.Lock()
	if m.closed {
		m.mu.Unlock()
		return CloudWorker{}, errors.New("cloud manager is closed")
	}
	if m.worker != nil && m.worker.State != CloudWorkerDestroyed {
		m.mu.Unlock()
		return CloudWorker{}, errors.New("a cloud worker is already managed by this lab")
	}
	now := m.now().UTC()
	worker := &managedCloudWorker{CloudWorker: CloudWorker{
		ID:           newRunID(),
		Provider:     verda.ProviderName,
		State:        CloudWorkerProvisioning,
		Message:      "Connecting to Verda",
		InstanceType: request.InstanceType,
		Market:       request.Market,
		Backends:     []string{"cpu", "cuda"},
		AutoDestroy:  autoDestroy,
		CreatedAt:    now,
	}}
	m.worker = worker
	provisionContext, provisionCancel := context.WithCancel(m.ctx)
	provisionDone := make(chan struct{})
	m.provisionWorkerID = worker.ID
	m.provisionCancel = provisionCancel
	m.provisionDone = provisionDone
	m.wg.Add(1)
	created := worker.CloudWorker
	created.Backends = append([]string(nil), worker.Backends...)
	m.mu.Unlock()

	go func() {
		defer func() {
			provisionCancel()
			close(provisionDone)
			m.mu.Lock()
			if m.provisionWorkerID == worker.ID {
				m.provisionWorkerID = ""
				m.provisionCancel = nil
				m.provisionDone = nil
			}
			m.mu.Unlock()
			m.wg.Done()
		}()
		m.provision(provisionContext, worker.ID, request)
	}()
	return created, nil
}

func (m *CloudManager) provision(ctx context.Context, workerID string, request CloudDeployRequest) {
	ctx, cancel := context.WithTimeout(ctx, m.timeout)
	defer cancel()
	client, err := m.newClient(ctx, m.baseURL)
	if err != nil {
		m.failWorker(workerID, err)
		return
	}
	m.updateWorker(workerID, func(worker *managedCloudWorker) {
		worker.Message = "Cloning the golden OS volume and creating a Verda instance"
	})
	deployOptions := verda.DefaultDeployOptions(request.InstanceType)
	deployOptions.SourceOSVolumeName = defaultCloudSourceVolumeName
	deployOptions.Market = request.Market
	deployOptions.LocationCode = request.LocationCode
	deployOptions.Hostname = "nnctl-lab-" + workerID[:8]
	deployOptions.Description = "nnctl learning lab cloud worker " + workerID
	deployOptions.BaseURL = m.baseURL
	result, err := verda.Deploy(ctx, client, deployOptions)
	if err != nil {
		m.failWorker(workerID, err)
		return
	}
	if result.Instance == nil || strings.TrimSpace(result.Instance.ID) == "" {
		m.failWorker(workerID, errors.New("verda deployment did not return an instance"))
		return
	}
	instanceID := result.Instance.ID
	cloneID := ""
	if result.OSVolumeClone != nil {
		cloneID = result.OSVolumeClone.VolumeID
	}
	sourceID := result.SourceOSVolumeID
	m.updateWorker(workerID, func(worker *managedCloudWorker) {
		worker.InstanceID = instanceID
		worker.Hostname = result.Instance.Hostname
		worker.SourceVolumeID = sourceID
		worker.CloneVolumeID = cloneID
		worker.BaseURL = m.baseURL
		worker.Message = "Waiting for an IP address"
		if result.Placement != nil {
			worker.PricePerHour = result.Placement.SpotPrice
			worker.Currency = result.Placement.Currency
		}
	})
	if err := m.persistJournal(); err != nil {
		m.cleanupFailedProvision(workerID, client, err)
		return
	}

	instance, err := cloudworkflow.WaitInstance(ctx, client, instanceID, m.timeout, m.pollInterval)
	if err != nil {
		m.cleanupFailedProvision(workerID, client, err)
		return
	}
	knownHosts, err := m.knownHostsPath(workerID)
	if err != nil {
		m.cleanupFailedProvision(workerID, client, err)
		return
	}
	host := m.sshUser + "@" + strings.TrimSpace(*instance.IP)
	sshArgs := cloudworkflow.SSHArgs(knownHosts)
	m.updateWorker(workerID, func(worker *managedCloudWorker) {
		worker.IP = strings.TrimSpace(*instance.IP)
		worker.Location = instance.Location
		worker.PricePerHour = firstCloudPrice(instance.PricePerHour, worker.PricePerHour)
		worker.Currency = firstCloudString(instance.Currency, worker.Currency)
		worker.KnownHosts = knownHosts
		worker.Host = host
		worker.Message = "Waiting for SSH"
	})
	if err := m.persistJournal(); err != nil {
		m.cleanupFailedProvision(workerID, client, err)
		return
	}
	if err := cloudworkflow.WaitSSH(ctx, m.ssh, sshArgs, host, m.timeout, m.pollInterval); err != nil {
		m.cleanupFailedProvision(workerID, client, err)
		return
	}
	m.updateWorker(workerID, func(worker *managedCloudWorker) {
		worker.Message = "Verifying Zig and CUDA"
	})
	preflight := "if [ -f /etc/profile.d/zig-nn.sh ]; then . /etc/profile.d/zig-nn.sh; fi; zig version; nvidia-smi >/dev/null"
	if err := cloudworkflow.WaitCommand(ctx, m.ssh, sshArgs, host, preflight, m.timeout, m.pollInterval); err != nil {
		m.cleanupFailedProvision(workerID, client, err)
		return
	}
	ready := m.now().UTC()
	expires := ready.Add(m.idleTimeout)
	m.updateWorker(workerID, func(worker *managedCloudWorker) {
		worker.State = CloudWorkerReady
		worker.Message = "Ready for experiments"
		worker.ReadyAt = &ready
		if worker.AutoDestroy {
			worker.ExpiresAt = &expires
		}
	})
	if err := m.persistJournal(); err != nil {
		m.cleanupFailedProvision(workerID, client, err)
		return
	}
	m.wg.Add(1)
	go func() {
		defer m.wg.Done()
		m.destroyIdleWorker(m.ctx, workerID, expires)
	}()
}

func (m *CloudManager) destroyIdleWorker(ctx context.Context, workerID string, expires time.Time) {
	delay := time.Until(expires)
	if delay > 0 {
		timer := time.NewTimer(delay)
		defer timer.Stop()
		select {
		case <-ctx.Done():
			return
		case <-timer.C:
		}
	}
	worker, ok := m.managedWorker(workerID)
	if !ok || !worker.AutoDestroy || worker.State != CloudWorkerReady || worker.ExpiresAt == nil || !worker.ExpiresAt.Equal(expires) {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	_ = m.Destroy(ctx, workerID)
}

func firstCloudPrice(values ...float64) float64 {
	for _, value := range values {
		if value != 0 {
			return value
		}
	}
	return 0
}

func firstCloudString(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

func (m *CloudManager) cleanupFailedProvision(workerID string, client cloudworkflow.Client, cause error) {
	worker, ok := m.managedWorker(workerID)
	if ok && worker.InstanceID != "" {
		cleanupCtx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
		cleanupErr := cloudworkflow.Destroy(cleanupCtx, client, worker.InstanceID, worker.CloneVolumeID, worker.SourceVolumeID, worker.BaseURL)
		cancel()
		cause = errors.Join(cause, cleanupErr)
		if cleanupErr == nil {
			m.updateWorker(workerID, func(current *managedCloudWorker) {
				current.InstanceID = ""
				current.CloneVolumeID = ""
			})
		}
	}
	m.failWorker(workerID, cause)
}

func (m *CloudManager) failWorker(workerID string, err error) {
	m.updateWorker(workerID, func(worker *managedCloudWorker) {
		worker.State = CloudWorkerFailed
		worker.Message = err.Error()
	})
	_ = m.persistJournal()
}

func (m *CloudManager) updateWorker(workerID string, update func(*managedCloudWorker)) {
	m.mu.Lock()
	if m.worker == nil || m.worker.ID != workerID {
		m.mu.Unlock()
		return
	}
	update(m.worker)
	worker := m.worker.CloudWorker
	worker.Backends = append([]string(nil), worker.Backends...)
	subscribers := make([]chan CloudWorker, 0, len(m.workerSubscribers))
	for subscriber := range m.workerSubscribers {
		subscribers = append(subscribers, subscriber)
	}
	m.mu.Unlock()
	for _, subscriber := range subscribers {
		select {
		case subscriber <- worker:
		default:
		}
	}
}

func (m *CloudManager) managedWorker(workerID string) (managedCloudWorker, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.worker == nil || m.worker.ID != workerID {
		return managedCloudWorker{}, false
	}
	return *m.worker, true
}

func (m *CloudManager) Workers() []CloudWorker {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.worker == nil {
		return []CloudWorker{}
	}
	worker := m.worker.CloudWorker
	worker.Backends = append([]string(nil), worker.Backends...)
	return []CloudWorker{worker}
}

type CloudWorkerSubscription struct {
	Initial CloudWorker
	Events  <-chan CloudWorker
	close   func()
}

func (s CloudWorkerSubscription) Close() {
	if s.close != nil {
		s.close()
	}
}

func (m *CloudManager) SubscribeWorker(workerID string) (CloudWorkerSubscription, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.worker == nil || m.worker.ID != workerID {
		return CloudWorkerSubscription{}, errors.New("cloud worker not found")
	}
	channel := make(chan CloudWorker, 32)
	m.workerSubscribers[channel] = struct{}{}
	initial := m.worker.CloudWorker
	initial.Backends = append([]string(nil), initial.Backends...)
	return CloudWorkerSubscription{
		Initial: initial,
		Events:  channel,
		close: func() {
			m.mu.Lock()
			delete(m.workerSubscribers, channel)
			m.mu.Unlock()
		},
	}, nil
}

func (m *CloudManager) ValidateWorker(workerID, backend string) error {
	worker, ok := m.managedWorker(workerID)
	if !ok {
		return errors.New("cloud worker not found")
	}
	if worker.State != CloudWorkerReady {
		return fmt.Errorf("cloud worker is %s", worker.State)
	}
	if !slices.Contains(worker.Backends, backend) {
		return fmt.Errorf("backend %q is unavailable on cloud worker", backend)
	}
	if worker.recovered {
		return errors.New("recovered cloud worker must be destroyed before creating another worker")
	}
	return nil
}

func (m *CloudManager) Destroy(ctx context.Context, workerID string) error {
	m.destroyMu.Lock()
	defer m.destroyMu.Unlock()

	worker, ok := m.managedWorker(workerID)
	if !ok {
		return nil
	}
	if worker.State == CloudWorkerDestroyed {
		return nil
	}
	m.updateWorker(workerID, func(current *managedCloudWorker) {
		current.State = CloudWorkerDestroying
		current.Message = "Stopping provisioning and destroying cloud resources"
	})
	m.mu.Lock()
	provisionCancel := m.provisionCancel
	provisionDone := m.provisionDone
	if m.provisionWorkerID != workerID {
		provisionCancel = nil
		provisionDone = nil
	}
	m.mu.Unlock()
	if provisionCancel != nil {
		provisionCancel()
	}
	if provisionDone != nil {
		select {
		case <-provisionDone:
		case <-ctx.Done():
			return fmt.Errorf("wait for cloud provisioning to stop: %w", ctx.Err())
		}
	}
	worker, ok = m.managedWorker(workerID)
	if !ok || worker.State == CloudWorkerDestroyed {
		return nil
	}
	m.updateWorker(workerID, func(current *managedCloudWorker) {
		current.State = CloudWorkerDestroying
		current.Message = "Destroying instance and cloned volume"
	})
	var err error
	if worker.InstanceID != "" {
		var client cloudworkflow.Client
		client, err = m.newClient(ctx, worker.BaseURL)
		if err == nil {
			err = cloudworkflow.Destroy(ctx, client, worker.InstanceID, worker.CloneVolumeID, worker.SourceVolumeID, worker.BaseURL)
		}
	}
	if err != nil {
		m.failWorker(workerID, err)
		return err
	}
	m.updateWorker(workerID, func(current *managedCloudWorker) {
		current.State = CloudWorkerDestroyed
		current.Message = "Destroyed"
	})
	if worker.KnownHosts != "" {
		_ = os.Remove(worker.KnownHosts)
	}
	return m.clearJournal()
}

func (m *CloudManager) Execute(
	ctx context.Context,
	spec ExperimentSpec,
	options RunOptions,
	stdoutLine func([]byte) error,
	stderrLine func(string),
) error {
	if err := m.ValidateWorker(options.WorkerID, options.Backend); err != nil {
		return err
	}
	worker, _ := m.managedWorker(options.WorkerID)
	m.updateWorker(worker.ID, func(current *managedCloudWorker) {
		current.State = CloudWorkerBusy
		current.Message = "Preparing remote experiment"
	})
	defer func() {
		if worker.AutoDestroy {
			cleanupCtx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
			if err := m.Destroy(cleanupCtx, worker.ID); err != nil {
				stderrLine("cloud cleanup failed: " + err.Error())
			}
			cancel()
			return
		}
		m.updateWorker(worker.ID, func(current *managedCloudWorker) {
			if current.State == CloudWorkerBusy {
				current.State = CloudWorkerReady
				current.Message = "Ready for another experiment"
			}
		})
	}()

	repository, err := cloudworkflow.InspectRepository(ctx, m.repoRoot, m.git)
	if err != nil {
		return err
	}
	if repository.Dirty && !options.AcknowledgeDirty {
		return errors.New("working tree has uncommitted changes; acknowledge that the cloud run uses committed HEAD")
	}
	remoteDir := "/root/zig-nn-lab-" + repository.Revision[:12]
	stderrLine("cloud: uploading committed revision " + repository.Revision)
	sshArgs := cloudworkflow.SSHArgs(worker.KnownHosts)
	if err := cloudworkflow.UploadSnapshot(ctx, cloudworkflow.UploadOptions{
		RepoRoot: m.repoRoot,
		Target:   worker.Host + ":" + remoteDir,
		Ref:      repository.Revision,
		SSH:      cloudworkflow.CommandString(m.ssh, sshArgs),
		Git:      m.git,
		Tar:      m.tar,
		Rsync:    m.rsync,
	}); err != nil {
		return err
	}
	m.updateWorker(worker.ID, func(current *managedCloudWorker) {
		current.RepositoryCommit = repository.Revision
		current.Message = "Running " + spec.Title
	})
	mode := m.mode
	if mode == "" {
		mode = "Debug"
	}
	if spec.Optimize != "" {
		mode = spec.Optimize
	}
	gpu := ""
	if options.Backend != "cpu" {
		gpu = options.Backend
	}
	args := zig.RunArgs(spec.Step, zig.Options{Optimize: mode, GPU: gpu}, options.Arguments)
	command := strings.Join([]string{
		"set -eu",
		"if [ -f /etc/profile.d/zig-nn.sh ]; then . /etc/profile.d/zig-nn.sh; fi",
		"cd " + cloudworkflow.ShellQuote(remoteDir),
		"exec " + cloudworkflow.CommandString("zig", args),
	}, "; ")
	return cloudworkflow.StreamSSH(ctx, m.ssh, sshArgs, worker.Host, command, stdoutLine, stderrLine)
}

type RoutingExecutor struct {
	Local Executor
	Cloud *CloudManager
}

func (e RoutingExecutor) Execute(
	ctx context.Context,
	spec ExperimentSpec,
	options RunOptions,
	stdout func([]byte) error,
	stderr func(string),
) error {
	if options.Target == RunTargetCloud {
		if e.Cloud == nil {
			return errors.New("cloud execution is disabled")
		}
		return e.Cloud.Execute(ctx, spec, options, stdout, stderr)
	}
	return e.Local.Execute(ctx, spec, options, stdout, stderr)
}

func (m *CloudManager) knownHostsPath(workerID string) (string, error) {
	directory := filepath.Join(filepath.Dir(m.journalPath), "known-hosts")
	if err := os.MkdirAll(directory, 0o700); err != nil {
		return "", fmt.Errorf("create known-hosts directory: %w", err)
	}
	return filepath.Join(directory, workerID), nil
}

func (m *CloudManager) persistJournal() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.worker == nil || m.worker.State == CloudWorkerDestroyed {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(m.journalPath), 0o700); err != nil {
		return fmt.Errorf("create cloud journal directory: %w", err)
	}
	data, err := json.MarshalIndent(m.worker, "", "  ")
	if err != nil {
		return fmt.Errorf("encode cloud worker journal: %w", err)
	}
	temporary := m.journalPath + ".tmp"
	if err := os.WriteFile(temporary, append(data, '\n'), 0o600); err != nil {
		return fmt.Errorf("write cloud worker journal: %w", err)
	}
	if err := os.Rename(temporary, m.journalPath); err != nil {
		return fmt.Errorf("replace cloud worker journal: %w", err)
	}
	return nil
}

func (m *CloudManager) loadJournal() error {
	data, err := os.ReadFile(m.journalPath)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("read cloud worker journal: %w", err)
	}
	var worker managedCloudWorker
	if err := json.Unmarshal(data, &worker); err != nil {
		return fmt.Errorf("decode cloud worker journal: %w", err)
	}
	worker.State = CloudWorkerFailed
	worker.Message = "Recovered from a previous lab session; destroy this worker before creating another"
	worker.recovered = true
	m.worker = &worker
	return nil
}

func (m *CloudManager) clearJournal() error {
	if err := os.Remove(m.journalPath); err != nil && !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("remove cloud worker journal: %w", err)
	}
	return nil
}

func (m *CloudManager) Close() error {
	m.mu.Lock()
	m.closed = true
	cancel := m.cancel
	m.mu.Unlock()
	if cancel != nil {
		cancel()
	}
	m.wg.Wait()
	m.mu.Lock()
	worker := m.worker
	m.mu.Unlock()
	if worker == nil || worker.State == CloudWorkerDestroyed || !worker.AutoDestroy {
		return nil
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	return m.Destroy(ctx, worker.ID)
}
