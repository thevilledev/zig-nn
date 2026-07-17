package lab

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"nnctl/internal/cloud/verda"
	cloudworkflow "nnctl/internal/cloud/workflow"
)

const (
	defaultCloudSourceVolumeName = "packer-verda-zig-nn-volume-root"
	defaultCloudSSHUser          = "root"
	defaultCloudTimeout          = 20 * time.Minute
	defaultCloudPollInterval     = 10 * time.Second
	defaultCloudIdleTimeout      = 30 * time.Minute
	cloudPersistenceTimeout      = 5 * time.Second
	cloudCleanupTimeout          = 5 * time.Minute
	cloudPreflightCommand        = "if [ -f /etc/profile.d/zig-nn.sh ]; then . /etc/profile.d/zig-nn.sh; fi; zig version; nvidia-smi >/dev/null"
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
	BaseURL        string `json:"base_url,omitempty"`
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
	Parent            context.Context
	RepoRoot          string
	BaseURL           string
	SSH               string
	Git               string
	Tar               string
	Rsync             string
	SSHUser           string
	Mode              string
	DatabasePath      string
	LegacyJournalPath string
	Timeout           time.Duration
	PollInterval      time.Duration
	IdleTimeout       time.Duration
	ClientFactory     cloudClientFactory
	Now               func() time.Time
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
	databasePath string
	store        *cloudStore
	timeout      time.Duration
	pollInterval time.Duration
	idleTimeout  time.Duration
	newClient    cloudClientFactory
	now          func() time.Time

	mu                sync.Mutex
	closeMu           sync.Mutex
	destroyMu         sync.Mutex
	persistenceMu     sync.Mutex
	worker            *managedCloudWorker
	workerSubscribers map[chan CloudWorker]struct{}
	ctx               context.Context
	cancel            context.CancelFunc
	wg                sync.WaitGroup
	closed            bool
	closeErr          error
	provisionWorkerID string
	provisionCancel   context.CancelFunc
	provisionDone     chan struct{}
}

func NewCloudManager(options CloudManagerOptions) (*CloudManager, error) {
	parent := options.Parent
	if parent == nil {
		parent = context.Background()
	}
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
	options.SSHUser = strings.TrimSpace(options.SSHUser)
	if err := cloudworkflow.ValidateSSHUser(options.SSHUser); err != nil {
		return nil, err
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
	if options.DatabasePath == "" {
		cache, err := os.UserCacheDir()
		if err != nil {
			return nil, fmt.Errorf("resolve user cache for cloud state: %w", err)
		}
		options.DatabasePath = filepath.Join(cache, "nnctl", cloudStateDatabaseName)
	}
	if options.LegacyJournalPath == "" {
		options.LegacyJournalPath = filepath.Join(filepath.Dir(options.DatabasePath), "lab-cloud-worker.json")
	}
	setupCtx, setupCancel := context.WithTimeout(parent, cloudPersistenceTimeout)
	defer setupCancel()
	store, err := openCloudStore(setupCtx, options.DatabasePath)
	if err != nil {
		return nil, err
	}
	if err := migrateCloudJournal(setupCtx, store, options.LegacyJournalPath); err != nil {
		_ = store.Close()
		return nil, err
	}
	managerCtx, managerCancel := context.WithCancel(parent)
	manager := &CloudManager{
		repoRoot:          options.RepoRoot,
		baseURL:           strings.TrimSpace(options.BaseURL),
		ssh:               options.SSH,
		git:               options.Git,
		tar:               options.Tar,
		rsync:             options.Rsync,
		sshUser:           options.SSHUser,
		mode:              options.Mode,
		databasePath:      options.DatabasePath,
		store:             store,
		timeout:           options.Timeout,
		pollInterval:      options.PollInterval,
		idleTimeout:       options.IdleTimeout,
		newClient:         options.ClientFactory,
		now:               options.Now,
		workerSubscribers: make(map[chan CloudWorker]struct{}),
		ctx:               managerCtx,
		cancel:            managerCancel,
	}
	if err := manager.loadWorker(); err != nil {
		managerCancel()
		_ = store.Close()
		return nil, err
	}
	manager.recoverWorker()
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
	autoDestroy := false
	if request.AutoDestroy != nil {
		autoDestroy = *request.AutoDestroy
	}

	m.mu.Lock()
	if m.closed || m.ctx.Err() != nil {
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
	m.updateProvisioningWorker(workerID, func(worker *managedCloudWorker) {
		worker.Message = "Cloning the golden OS volume and creating a Verda instance"
	})
	deployOptions := verda.DefaultDeployOptions(request.InstanceType)
	deployOptions.SourceOSVolumeName = defaultCloudSourceVolumeName
	deployOptions.Market = request.Market
	deployOptions.LocationCode = request.LocationCode
	deployOptions.Hostname = "nnctl-lab-" + workerID[:8]
	deployOptions.Description = "nnctl learning lab cloud worker " + workerID
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
		if worker.State == CloudWorkerProvisioning {
			worker.Message = "Waiting for an IP address"
		}
		if result.Placement != nil {
			worker.PricePerHour = result.Placement.SpotPrice
			worker.Currency = result.Placement.Currency
		}
	})
	if err := m.persistWorker(); err != nil {
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
	host, err := cloudworkflow.SSHDestination(m.sshUser, *instance.IP)
	if err != nil {
		m.cleanupFailedProvision(workerID, client, err)
		return
	}
	sshArgs := cloudworkflow.SSHArgs(knownHosts)
	m.updateWorker(workerID, func(worker *managedCloudWorker) {
		worker.IP = strings.TrimSpace(*instance.IP)
		worker.Location = instance.Location
		worker.PricePerHour = firstCloudPrice(instance.PricePerHour, worker.PricePerHour)
		worker.Currency = firstCloudString(instance.Currency, worker.Currency)
		if worker.State == CloudWorkerProvisioning {
			worker.Message = "Waiting for SSH"
		}
	})
	if err := m.persistWorker(); err != nil {
		m.cleanupFailedProvision(workerID, client, err)
		return
	}
	if err := cloudworkflow.WaitSSH(ctx, m.ssh, sshArgs, host, m.timeout, m.pollInterval); err != nil {
		m.cleanupFailedProvision(workerID, client, err)
		return
	}
	m.updateProvisioningWorker(workerID, func(worker *managedCloudWorker) {
		worker.Message = "Verifying Zig and CUDA"
	})
	if err := cloudworkflow.WaitCommand(ctx, m.ssh, sshArgs, host, cloudPreflightCommand, m.timeout, m.pollInterval); err != nil {
		m.cleanupFailedProvision(workerID, client, err)
		return
	}
	ready := m.now().UTC()
	expires := ready.Add(m.idleTimeout)
	if _, err := m.transitionWorker(workerID, []CloudWorkerState{CloudWorkerProvisioning}, CloudWorkerReady, "Ready for experiments", func(worker *managedCloudWorker) {
		worker.ReadyAt = &ready
		if worker.AutoDestroy {
			worker.ExpiresAt = &expires
		}
	}); err != nil {
		return
	}
	if err := m.persistWorker(); err != nil {
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
	ctx, cancel := m.cleanupContext()
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
		cleanupCtx, cancel := m.cleanupContext()
		cleanupErr := cloudworkflow.Destroy(cleanupCtx, client, worker.InstanceID, worker.CloneVolumeID, worker.SourceVolumeID)
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
	if worker.State != CloudWorkerDestroying {
		var err error
		worker, err = m.transitionWorker(
			workerID,
			[]CloudWorkerState{CloudWorkerProvisioning, CloudWorkerReady, CloudWorkerBusy, CloudWorkerFailed},
			CloudWorkerDestroying,
			"Stopping provisioning and destroying cloud resources",
			nil,
		)
		if err != nil {
			return err
		}
	}
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
	worker, err := m.transitionWorker(
		workerID,
		[]CloudWorkerState{CloudWorkerDestroying},
		CloudWorkerDestroying,
		"Destroying instance and cloned volume",
		nil,
	)
	if err != nil {
		return err
	}
	if err := m.persistWorker(); err != nil {
		return err
	}
	err = nil
	if worker.InstanceID != "" {
		var client cloudworkflow.Client
		client, err = m.newClient(ctx, worker.BaseURL)
		if err == nil {
			err = cloudworkflow.Destroy(ctx, client, worker.InstanceID, worker.CloneVolumeID, worker.SourceVolumeID)
		}
	}
	if err != nil {
		if _, transitionErr := m.transitionWorker(
			workerID,
			[]CloudWorkerState{CloudWorkerDestroying},
			CloudWorkerFailed,
			err.Error(),
			nil,
		); transitionErr == nil {
			_ = m.persistWorker()
		}
		return err
	}
	if _, err := m.transitionWorker(
		workerID,
		[]CloudWorkerState{CloudWorkerDestroying},
		CloudWorkerDestroyed,
		"Destroyed",
		nil,
	); err != nil {
		return err
	}
	_ = os.Remove(filepath.Join(filepath.Dir(m.databasePath), "known-hosts", worker.ID))
	return m.clearWorker(worker.ID)
}
