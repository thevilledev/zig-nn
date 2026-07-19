package lab

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	cloudcore "nnctl/internal/cloud"
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
	cloudPreflightBaseCommand    = "if [ -f /etc/profile.d/zig-nn.sh ]; then . /etc/profile.d/zig-nn.sh; fi; zig version"
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
	Resources      []cloudcore.ResourceRef `json:"resources,omitempty"`
	SourceVolumeID string                  `json:"source_volume_id,omitempty"`
	CloneVolumeID  string                  `json:"clone_volume_id,omitempty"`
	BaseURL        string                  `json:"base_url,omitempty"`
}

type CloudStatus struct {
	Enabled    bool                          `json:"enabled"`
	Configured bool                          `json:"configured"`
	Provider   string                        `json:"provider"`
	Error      string                        `json:"error,omitempty"`
	Repository cloudworkflow.RepositoryState `json:"repository"`
	Providers  []CloudProviderStatus         `json:"providers,omitempty"`
}

type CloudProviderStatus struct {
	Name         string                 `json:"name"`
	DisplayName  string                 `json:"display_name"`
	Configured   bool                   `json:"configured"`
	Error        string                 `json:"error,omitempty"`
	Capabilities []cloudcore.Capability `json:"capabilities"`
}

type CloudOptions struct {
	Prices             []verda.InstancePrice `json:"prices"`
	Provider           string                `json:"provider"`
	SourceOSVolumeName string                `json:"source_os_volume_name"`
	Offerings          []cloudcore.Offering  `json:"offerings"`
	Errors             map[string]string     `json:"errors,omitempty"`
}

type CloudDeployRequest struct {
	Provider     string `json:"provider,omitempty"`
	InstanceType string `json:"instance_type"`
	Market       string `json:"market"`
	LocationCode string `json:"location_code,omitempty"`
	AutoDestroy  *bool  `json:"auto_destroy,omitempty"`
}

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
	Providers         []CloudProviderConfig
	Now               func() time.Time
}

type CloudManager struct {
	repoRoot     string
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
	providers    map[string]cloudProviderRuntime
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
	if options.Now == nil {
		options.Now = time.Now
	}
	providers, err := buildCloudProviders(options)
	if err != nil {
		return nil, err
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
		providers:         providers,
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
	status := CloudStatus{Enabled: true, Provider: m.primaryProviderName()}
	repository, err := cloudworkflow.InspectRepository(ctx, m.repoRoot, m.git)
	if err != nil {
		status.Error = err.Error()
		return status
	}
	status.Repository = repository
	for _, name := range m.providerNames() {
		runtime := m.providers[name]
		descriptor := runtime.factory.Descriptor()
		configuration := runtime.factory.Status(ctx, runtime.configuration)
		providerStatus := CloudProviderStatus{
			Name: name, DisplayName: descriptor.DisplayName,
			Configured: configuration.Configured, Error: configuration.Error,
			Capabilities: append([]cloudcore.Capability(nil), descriptor.Capabilities...),
		}
		status.Providers = append(status.Providers, providerStatus)
		if configuration.Configured {
			status.Configured = true
		}
		if name == status.Provider && configuration.Error != "" {
			status.Error = configuration.Error
		}
	}
	return status
}

func (m *CloudManager) Options(ctx context.Context) (CloudOptions, error) {
	result := CloudOptions{
		Provider: m.primaryProviderName(), SourceOSVolumeName: defaultCloudSourceVolumeName,
		Errors: make(map[string]string),
	}
	for _, name := range m.providerNames() {
		provider, err := m.provider(ctx, name)
		if err != nil {
			result.Errors[name] = err.Error()
			continue
		}
		offerings, err := m.labOfferings(ctx, name, provider)
		if err != nil {
			result.Errors[name] = err.Error()
			continue
		}
		if name == verda.ProviderName {
			for _, offering := range offerings {
				var price verda.InstancePrice
				if err := json.Unmarshal(offering.ProviderDetail, &price); err != nil {
					return CloudOptions{}, fmt.Errorf("decode Verda lab offering: %w", err)
				}
				result.Prices = append(result.Prices, price)
			}
		}
		result.Offerings = append(result.Offerings, offerings...)
	}
	if err := ctx.Err(); err != nil {
		return CloudOptions{}, err
	}
	verda.SortInstancePrices(result.Prices, "price")
	sort.SliceStable(result.Offerings, func(left, right int) bool {
		a, b := result.Offerings[left], result.Offerings[right]
		if a.PriceKnown != b.PriceKnown {
			return a.PriceKnown
		}
		if a.PriceKnown && a.PricePerHour != b.PricePerHour {
			return a.PricePerHour < b.PricePerHour
		}
		if a.Provider != b.Provider {
			return a.Provider < b.Provider
		}
		return a.ID < b.ID
	})
	if len(result.Errors) == 0 {
		result.Errors = nil
	}
	return result, nil
}

func (m *CloudManager) labOfferings(
	ctx context.Context,
	name string,
	provider cloudcore.Provider,
) ([]cloudcore.Offering, error) {
	offerings, err := provider.Offerings(ctx, cloudcore.OfferingFilters{
		GPUCounts: []int{1}, AvailableOnly: true,
	})
	if err != nil {
		return nil, err
	}
	if name == verda.ProviderName {
		return compatibleVerdaOfferings(ctx, provider, offerings)
	}
	return offerings, nil
}

func compatibleVerdaOfferings(
	ctx context.Context,
	provider cloudcore.Provider,
	offerings []cloudcore.Offering,
) ([]cloudcore.Offering, error) {
	volumeProvider, ok := provider.(cloudcore.VolumeProvider)
	if !ok {
		return nil, errors.New("verda provider does not support volume discovery")
	}
	volumes, err := volumeProvider.Volumes(ctx, cloudcore.VolumeListOptions{})
	if err != nil {
		return nil, fmt.Errorf("list Verda volumes: %w", err)
	}
	goldenLocations := make(map[string]struct{})
	for _, volume := range volumes {
		if strings.EqualFold(strings.TrimSpace(volume.Name), defaultCloudSourceVolumeName) && volume.Bootable && cloudVolumeReady(volume.State) {
			goldenLocations[strings.ToUpper(strings.TrimSpace(volume.Location))] = struct{}{}
		}
	}
	compatible := make([]cloudcore.Offering, 0, len(offerings))
	for _, offering := range offerings {
		if _, ok := goldenLocations[strings.ToUpper(strings.TrimSpace(offering.Location))]; ok {
			compatible = append(compatible, offering)
		}
	}
	return compatible, nil
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
	request.Provider = strings.ToLower(strings.TrimSpace(request.Provider))
	if request.Provider == "" {
		request.Provider = m.primaryProviderName()
	}
	request.InstanceType = strings.TrimSpace(request.InstanceType)
	request.Market = strings.ToLower(strings.TrimSpace(request.Market))
	request.LocationCode = strings.TrimSpace(request.LocationCode)
	if request.InstanceType == "" {
		return CloudWorker{}, errors.New("instance type is required")
	}
	runtime, err := m.providerRuntime(request.Provider)
	if err != nil {
		return CloudWorker{}, err
	}
	if request.Market == "" && request.Provider == verda.ProviderName {
		request.Market = verda.PricingMarketSpot
	}
	if err := ctx.Err(); err != nil {
		return CloudWorker{}, err
	}
	provider, err := m.provider(ctx, request.Provider)
	if err != nil {
		return CloudWorker{}, err
	}
	offering, err := m.resolveLabOffering(ctx, request, provider)
	if err != nil {
		return CloudWorker{}, err
	}
	request.InstanceType = offering.ID
	request.LocationCode = offering.Location
	request.Market = offering.Market
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
	backends := append([]string(nil), offering.Backends...)
	if len(backends) == 0 {
		backends = []string{"cpu"}
	}
	displayName := runtime.factory.Descriptor().DisplayName
	worker := &managedCloudWorker{CloudWorker: CloudWorker{
		ID:           newRunID(),
		Provider:     request.Provider,
		State:        CloudWorkerProvisioning,
		Message:      "Connecting to " + displayName,
		InstanceType: request.InstanceType,
		Market:       request.Market,
		Backends:     backends,
		PricePerHour: offering.PricePerHour,
		Currency:     offering.Currency,
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

func (m *CloudManager) resolveLabOffering(
	ctx context.Context,
	request CloudDeployRequest,
	provider cloudcore.Provider,
) (cloudcore.Offering, error) {
	offerings, err := m.labOfferings(ctx, request.Provider, provider)
	if err != nil {
		return cloudcore.Offering{}, fmt.Errorf("list %s lab offerings: %w", request.Provider, err)
	}
	matches := make([]cloudcore.Offering, 0, 1)
	for _, offering := range offerings {
		if !strings.EqualFold(offering.Provider, request.Provider) ||
			!strings.EqualFold(offering.ID, request.InstanceType) {
			continue
		}
		if request.LocationCode != "" && !strings.EqualFold(offering.Location, request.LocationCode) {
			continue
		}
		if request.Market != "" && !strings.EqualFold(offering.Market, request.Market) {
			continue
		}
		matches = append(matches, offering)
	}
	if len(matches) == 0 {
		return cloudcore.Offering{}, fmt.Errorf(
			"cloud offering %s:%s is not enabled and available for the requested location and market",
			request.Provider,
			request.InstanceType,
		)
	}
	if len(matches) > 1 {
		return cloudcore.Offering{}, fmt.Errorf(
			"cloud offering %s:%s is ambiguous; specify location_code and market",
			request.Provider,
			request.InstanceType,
		)
	}
	return matches[0], nil
}

func (m *CloudManager) provision(ctx context.Context, workerID string, request CloudDeployRequest) {
	ctx, cancel := context.WithTimeout(ctx, m.timeout)
	defer cancel()
	provider, err := m.provider(ctx, request.Provider)
	if err != nil {
		m.failWorker(workerID, err)
		return
	}
	m.updateProvisioningWorker(workerID, func(worker *managedCloudWorker) {
		worker.Message = "Creating a " + provider.Descriptor().DisplayName + " instance"
	})
	providerOptions := map[string]string{}
	if request.Provider == verda.ProviderName {
		providerOptions[verda.OptionSourceOSVolumeName] = defaultCloudSourceVolumeName
	}
	result, err := provider.Deploy(ctx, cloudcore.DeployRequest{
		OfferingID:  request.InstanceType,
		Market:      request.Market,
		Location:    request.LocationCode,
		Name:        "nnctl-lab-" + workerID[:8],
		Description: "nnctl learning lab cloud worker " + workerID,
		Options:     providerOptions,
	})
	if err != nil {
		m.failWorker(workerID, err)
		return
	}
	if result.Instance == nil || strings.TrimSpace(result.Instance.ID) == "" {
		m.failWorker(workerID, fmt.Errorf("%s deployment did not return an instance", request.Provider))
		return
	}
	instanceID := result.Instance.ID
	resources := cloneResourceRefs(result.Resources)
	m.updateWorker(workerID, func(worker *managedCloudWorker) {
		worker.InstanceID = instanceID
		worker.Hostname = result.Instance.Name
		worker.Resources = resources
		worker.SourceVolumeID, worker.CloneVolumeID = legacyVerdaResourceIDs(resources)
		worker.Backends = append([]string(nil), result.Instance.Backends...)
		if len(worker.Backends) == 0 {
			worker.Backends = []string{"cpu"}
		}
		if worker.State == CloudWorkerProvisioning {
			worker.Message = "Waiting for an IP address"
		}
		worker.PricePerHour = result.Instance.PricePerHour
		worker.Currency = result.Instance.Currency
	})
	if err := m.persistWorker(); err != nil {
		m.cleanupFailedProvision(workerID, provider, err)
		return
	}

	instance, err := cloudworkflow.WaitProviderInstance(ctx, provider, instanceID, m.timeout, m.pollInterval)
	if err != nil {
		m.cleanupFailedProvision(workerID, provider, err)
		return
	}
	knownHosts, err := m.knownHostsPath(workerID)
	if err != nil {
		m.cleanupFailedProvision(workerID, provider, err)
		return
	}
	host, err := cloudworkflow.SSHDestination(m.sshUser, instance.PublicIP)
	if err != nil {
		m.cleanupFailedProvision(workerID, provider, err)
		return
	}
	sshArgs := cloudworkflow.SSHArgs(knownHosts)
	m.updateWorker(workerID, func(worker *managedCloudWorker) {
		worker.IP = strings.TrimSpace(instance.PublicIP)
		worker.Location = instance.Location
		worker.PricePerHour = firstCloudPrice(instance.PricePerHour, worker.PricePerHour)
		worker.Currency = firstCloudString(instance.Currency, worker.Currency)
		if len(instance.Backends) > 0 {
			worker.Backends = append([]string(nil), instance.Backends...)
		}
		if worker.State == CloudWorkerProvisioning {
			worker.Message = "Waiting for SSH"
		}
	})
	if err := m.persistWorker(); err != nil {
		m.cleanupFailedProvision(workerID, provider, err)
		return
	}
	if err := cloudworkflow.WaitSSH(ctx, m.ssh, sshArgs, host, m.timeout, m.pollInterval); err != nil {
		m.cleanupFailedProvision(workerID, provider, err)
		return
	}
	m.updateProvisioningWorker(workerID, func(worker *managedCloudWorker) {
		worker.Message = "Verifying Zig and accelerator runtime"
	})
	worker, _ := m.managedWorker(workerID)
	if err := cloudworkflow.WaitCommand(ctx, m.ssh, sshArgs, host, cloudPreflightCommandFor(worker.Backends), m.timeout, m.pollInterval); err != nil {
		m.cleanupFailedProvision(workerID, provider, err)
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
		m.cleanupFailedProvision(workerID, provider, err)
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

func cloudPreflightCommandFor(backends []string) string {
	command := cloudPreflightBaseCommand
	for _, backend := range backends {
		switch backend {
		case "cuda":
			return command + "; nvidia-smi >/dev/null"
		case "rocm":
			return command + "; rocminfo >/dev/null"
		}
	}
	return command
}

func (m *CloudManager) cleanupFailedProvision(workerID string, provider cloudcore.Provider, cause error) {
	worker, ok := m.managedWorker(workerID)
	if ok && worker.InstanceID != "" {
		cleanupCtx, cancel := m.cleanupContext()
		_, cleanupErr := provider.Destroy(cleanupCtx, cloudcore.DestroyRequest{
			InstanceIDs: []string{worker.InstanceID}, Resources: worker.Resources, Permanent: true,
		})
		cancel()
		cause = errors.Join(cause, cleanupErr)
		if cleanupErr == nil {
			m.updateWorker(workerID, func(current *managedCloudWorker) {
				current.InstanceID = ""
				current.Resources = nil
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
		var provider cloudcore.Provider
		provider, err = m.provider(ctx, worker.Provider)
		if err == nil {
			_, err = provider.Destroy(ctx, cloudcore.DestroyRequest{
				InstanceIDs: []string{worker.InstanceID}, Resources: worker.Resources, Permanent: true,
			})
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
