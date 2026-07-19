package verda

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strconv"
	"strings"

	cloudcore "nnctl/internal/cloud"
)

const (
	OptionSourceOSVolumeID          = "source-os-volume-id"
	OptionSourceOSVolumeName        = "source-os-volume-name"
	OptionClonedOSVolumeID          = "cloned-os-volume-id"
	OptionStartupScriptName         = "startup-script-name"
	OptionSkipAvailabilityCheck     = "skip-availability-check"
	OptionKeepClonedVolumeOnFailure = "keep-cloned-os-volume-on-failure"
	ResourceOSVolume                = "verda-os-volume"
)

type ProviderClient interface {
	Client
	DestroyClient
	VolumeListClient
	VolumePurgeClient
	ListInstances(context.Context, string) ([]Instance, error)
	ListSSHKeys(context.Context) ([]SSHKey, error)
	ListInstancePrices(context.Context, PricingFilters) ([]InstancePrice, error)
}

type ProviderFactory struct {
	CredentialStore CredentialStore
	ClientOptions   ClientOptions
	NewClient       func(Credentials, ClientOptions) (ProviderClient, error)
}

func (ProviderFactory) Descriptor() cloudcore.Descriptor {
	return providerDescriptor()
}

func (f ProviderFactory) Status(ctx context.Context, _ cloudcore.Configuration) cloudcore.ConfigurationStatus {
	_, err := f.credentialStore().Credentials(ctx)
	if err != nil {
		return cloudcore.ConfigurationStatus{Error: err.Error()}
	}
	return cloudcore.ConfigurationStatus{Configured: true}
}

func (f ProviderFactory) New(ctx context.Context, config cloudcore.Configuration) (cloudcore.Provider, error) {
	if config.Unauthenticated {
		return NewCloudProvider(nil), nil
	}
	credentials, err := f.credentialStore().Credentials(ctx)
	if err != nil {
		return nil, err
	}
	options := f.ClientOptions
	if strings.TrimSpace(config.BaseURL) != "" {
		options.BaseURL = config.BaseURL
	}
	if strings.TrimSpace(config.UserAgent) != "" {
		options.UserAgent = config.UserAgent
	}
	newClient := f.NewClient
	if newClient == nil {
		newClient = func(credentials Credentials, options ClientOptions) (ProviderClient, error) {
			return NewSDKClient(credentials, options)
		}
	}
	client, err := newClient(credentials, options)
	if err != nil {
		return nil, fmt.Errorf("create Verda client: %w", err)
	}
	return NewCloudProvider(client), nil
}

func (f ProviderFactory) credentialStore() CredentialStore {
	if f.CredentialStore != nil {
		return f.CredentialStore
	}
	return KeyringCredentialStore{}
}

type CloudProvider struct {
	client ProviderClient
}

func NewCloudProvider(client ProviderClient) *CloudProvider {
	return &CloudProvider{client: client}
}

func (*CloudProvider) Descriptor() cloudcore.Descriptor {
	return providerDescriptor()
}

func providerDescriptor() cloudcore.Descriptor {
	return cloudcore.Descriptor{
		Name:        ProviderName,
		DisplayName: "Verda",
		Capabilities: []cloudcore.Capability{
			cloudcore.CapabilityCompute,
			cloudcore.CapabilityPricing,
			cloudcore.CapabilitySSHKeys,
			cloudcore.CapabilityVolumes,
			cloudcore.CapabilityImageTemplate,
			cloudcore.CapabilityBenchmarkImage,
		},
	}
}

func (p *CloudProvider) Offerings(ctx context.Context, filters cloudcore.OfferingFilters) ([]cloudcore.Offering, error) {
	if p == nil || p.client == nil {
		return nil, errors.New("verda provider client is required")
	}
	pricingFilters := PricingFilters{
		InstanceTypes: filters.IDs,
		LocationCodes: filters.Locations,
		Manufacturer:  filters.Manufacturer,
		Model:         filters.Model,
		GPUCounts:     filters.GPUCounts,
		AvailableOnly: filters.AvailableOnly,
		Currency:      filters.Currency,
		Market:        PricingMarketAll,
	}
	if len(filters.Markets) == 1 {
		pricingFilters.Market = filters.Markets[0]
	}
	prices, err := p.client.ListInstancePrices(ctx, pricingFilters)
	if err != nil {
		return nil, fmt.Errorf("list Verda offerings: %w", err)
	}
	offerings := make([]cloudcore.Offering, 0, len(prices))
	for _, price := range prices {
		if len(filters.Markets) > 1 && !containsFold(filters.Markets, price.Market) {
			continue
		}
		offering, err := offeringFromPrice(price)
		if err != nil {
			return nil, err
		}
		offerings = append(offerings, offering)
	}
	return offerings, nil
}

func (p *CloudProvider) Deploy(ctx context.Context, request cloudcore.DeployRequest) (*cloudcore.Deployment, error) {
	options, err := deployOptionsFromRequest(request)
	if err != nil {
		return nil, err
	}
	var client Client
	if !request.DryRun {
		if p == nil || p.client == nil {
			return nil, errors.New("verda provider client is required")
		}
		client = p.client
	}
	result, err := Deploy(ctx, client, options)
	if err != nil {
		return nil, err
	}
	return deploymentFromResult(request, result)
}

func (p *CloudProvider) Instance(ctx context.Context, id string) (cloudcore.Instance, error) {
	if p == nil || p.client == nil {
		return cloudcore.Instance{}, errors.New("verda provider client is required")
	}
	id = strings.TrimSpace(id)
	if id == "" {
		return cloudcore.Instance{}, errors.New("instance ID is required")
	}
	instances, err := p.client.ListInstances(ctx, "")
	if err != nil {
		return cloudcore.Instance{}, fmt.Errorf("list Verda instances: %w", err)
	}
	for _, instance := range instances {
		if instance.ID == id {
			return instanceFromProvider(instance)
		}
	}
	return cloudcore.Instance{}, fmt.Errorf("verda instance %s was not found", id)
}

func (p *CloudProvider) Instances(ctx context.Context, options cloudcore.ListOptions) ([]cloudcore.Instance, error) {
	if p == nil || p.client == nil {
		return nil, errors.New("verda provider client is required")
	}
	instances, err := p.client.ListInstances(ctx, strings.TrimSpace(options.Status))
	if err != nil {
		return nil, fmt.Errorf("list Verda instances: %w", err)
	}
	result := make([]cloudcore.Instance, 0, len(instances))
	for _, instance := range instances {
		mapped, mapErr := instanceFromProvider(instance)
		if mapErr != nil {
			return nil, mapErr
		}
		if !options.All && options.Status == "" && mapped.State.Terminal() {
			continue
		}
		result = append(result, mapped)
	}
	return result, nil
}

func (p *CloudProvider) Destroy(ctx context.Context, request cloudcore.DestroyRequest) (*cloudcore.DestroyResult, error) {
	options, err := destroyOptionsFromRequest(request)
	if err != nil {
		return nil, err
	}
	var client DestroyClient
	if !request.DryRun {
		if p == nil || p.client == nil {
			return nil, errors.New("verda provider client is required")
		}
		client = p.client
	}
	result, err := Destroy(ctx, client, options)
	if err != nil {
		return nil, err
	}
	detail, err := json.Marshal(result)
	if err != nil {
		return nil, fmt.Errorf("encode Verda destroy result: %w", err)
	}
	return &cloudcore.DestroyResult{
		Provider:       ProviderName,
		DryRun:         result.DryRun,
		Request:        request,
		ProviderDetail: detail,
	}, nil
}

func (p *CloudProvider) SSHKeys(ctx context.Context) ([]cloudcore.SSHKey, error) {
	if p == nil || p.client == nil {
		return nil, errors.New("verda provider client is required")
	}
	keys, err := p.client.ListSSHKeys(ctx)
	if err != nil {
		return nil, fmt.Errorf("list Verda SSH keys: %w", err)
	}
	result := make([]cloudcore.SSHKey, 0, len(keys))
	for _, key := range keys {
		detail, marshalErr := json.Marshal(key)
		if marshalErr != nil {
			return nil, fmt.Errorf("encode Verda SSH key: %w", marshalErr)
		}
		result = append(result, cloudcore.SSHKey{
			Provider: ProviderName, ID: key.ID, Name: key.Name,
			Fingerprint: key.Fingerprint, ProviderDetail: detail,
		})
	}
	return result, nil
}

func (p *CloudProvider) Volumes(ctx context.Context, options cloudcore.VolumeListOptions) ([]cloudcore.Volume, error) {
	if p == nil || p.client == nil {
		return nil, errors.New("verda provider client is required")
	}
	volumes, err := ListVolumes(ctx, p.client, ListVolumesOptions{IncludeDeleted: options.IncludeDeleted})
	if err != nil {
		return nil, err
	}
	result := make([]cloudcore.Volume, 0, len(volumes))
	for _, volume := range volumes {
		mapped, mapErr := volumeFromProvider(volume)
		if mapErr != nil {
			return nil, mapErr
		}
		result = append(result, mapped)
	}
	return result, nil
}

func (p *CloudProvider) PurgeVolumes(ctx context.Context, request cloudcore.VolumePurgeRequest) (*cloudcore.VolumePurgeResult, error) {
	var client VolumePurgeClient
	if !request.DryRun {
		if p == nil || p.client == nil {
			return nil, errors.New("verda provider client is required")
		}
		client = p.client
	}
	result, err := PurgeVolumes(ctx, client, PurgeVolumesOptions{
		VolumeIDs:  request.IDs,
		AllDeleted: request.AllDeleted,
		DryRun:     request.DryRun,
	})
	if err != nil {
		return nil, err
	}
	detail, err := json.Marshal(result)
	if err != nil {
		return nil, fmt.Errorf("encode Verda volume purge result: %w", err)
	}
	purged := make([]string, 0, len(result.Results))
	for _, item := range result.Results {
		purged = append(purged, item.VolumeID)
	}
	return &cloudcore.VolumePurgeResult{
		Provider:       ProviderName,
		DryRun:         result.DryRun,
		Request:        request,
		PurgedIDs:      purged,
		ProviderDetail: detail,
	}, nil
}

func (*CloudProvider) ImageTemplateFiles() []cloudcore.TemplateFile {
	return verdaTemplateFiles()
}

func (ProviderFactory) ImageTemplateFiles() []cloudcore.TemplateFile {
	return verdaTemplateFiles()
}

func verdaTemplateFiles() []cloudcore.TemplateFile {
	files := PackerTemplateFiles()
	result := make([]cloudcore.TemplateFile, 0, len(files))
	for _, file := range files {
		result = append(result, cloudcore.TemplateFile{Path: file.Path, Content: file.Content})
	}
	return result
}

func deployOptionsFromRequest(request cloudcore.DeployRequest) (DeployOptions, error) {
	options := DefaultDeployOptions(request.OfferingID)
	options.LocationCode = request.Location
	if strings.TrimSpace(request.Market) != "" {
		options.Market = request.Market
	}
	if strings.TrimSpace(request.Image) != "" {
		options.Image = request.Image
	}
	options.Hostname = request.Name
	options.Description = request.Description
	options.SSHKeyIDs = append([]string(nil), request.SSHKeyIDs...)
	options.UserDataScript = request.UserData
	options.DryRun = request.DryRun
	options.SourceOSVolumeID = request.Options[OptionSourceOSVolumeID]
	options.SourceOSVolumeName = request.Options[OptionSourceOSVolumeName]
	options.ClonedOSVolumeID = request.Options[OptionClonedOSVolumeID]
	options.StartupScriptName = request.Options[OptionStartupScriptName]
	var err error
	if options.SkipAvailabilityCheck, err = boolOption(request.Options, OptionSkipAvailabilityCheck); err != nil {
		return DeployOptions{}, err
	}
	if options.KeepClonedOSVolumeOnFailure, err = boolOption(request.Options, OptionKeepClonedVolumeOnFailure); err != nil {
		return DeployOptions{}, err
	}
	return options, nil
}

func boolOption(options map[string]string, name string) (bool, error) {
	raw := strings.TrimSpace(options[name])
	if raw == "" {
		return false, nil
	}
	value, err := strconv.ParseBool(raw)
	if err != nil {
		return false, fmt.Errorf("verda option %s must be true or false", name)
	}
	return value, nil
}

func destroyOptionsFromRequest(request cloudcore.DestroyRequest) (DestroyOptions, error) {
	options := DestroyOptions{
		InstanceIDs:       append([]string(nil), request.InstanceIDs...),
		DeletePermanently: request.Permanent,
		DryRun:            request.DryRun,
	}
	for _, resource := range request.Resources {
		if resource.Kind != ResourceOSVolume || strings.TrimSpace(resource.ID) == "" {
			continue
		}
		if resource.Preserve {
			if options.SourceOSVolumeID != "" && options.SourceOSVolumeID != resource.ID {
				return DestroyOptions{}, errors.New("verda cleanup contains more than one preserved source OS volume")
			}
			options.SourceOSVolumeID = resource.ID
			continue
		}
		options.VolumeIDs = append(options.VolumeIDs, resource.ID)
	}
	return options, nil
}

func offeringFromPrice(price InstancePrice) (cloudcore.Offering, error) {
	detail, err := json.Marshal(price)
	if err != nil {
		return cloudcore.Offering{}, fmt.Errorf("encode Verda offering: %w", err)
	}
	return cloudcore.Offering{
		Provider:       ProviderName,
		ID:             price.InstanceType,
		DisplayName:    price.DisplayName,
		Location:       price.LocationCode,
		Market:         price.Market,
		Accelerator:    cloudcore.Accelerator{Manufacturer: strings.ToLower(price.Manufacturer), Model: price.Model, Count: price.GPUCount},
		Backends:       backendsForAccelerator(price.Manufacturer, price.Model, price.GPUCount),
		PricePerHour:   price.PricePerHour,
		PriceKnown:     price.PriceKnown,
		Currency:       strings.ToUpper(price.Currency),
		Available:      price.Available,
		Discoverable:   true,
		ProviderDetail: detail,
	}, nil
}

func instanceFromProvider(instance Instance) (cloudcore.Instance, error) {
	detail, err := json.Marshal(instance)
	if err != nil {
		return cloudcore.Instance{}, fmt.Errorf("encode Verda instance: %w", err)
	}
	ip := ""
	if instance.IP != nil {
		ip = strings.TrimSpace(*instance.IP)
	}
	market := PricingMarketOnDemand
	if instance.IsSpot {
		market = PricingMarketSpot
	}
	return cloudcore.Instance{
		Provider:       ProviderName,
		ID:             instance.ID,
		Name:           instance.Hostname,
		State:          instanceState(instance.Status),
		PublicIP:       ip,
		OfferingID:     instance.InstanceType,
		Location:       instance.Location,
		Market:         market,
		Backends:       backendsForInstanceType(instance.InstanceType),
		PricePerHour:   instance.PricePerHour,
		Currency:       strings.ToUpper(instance.Currency),
		ProviderDetail: detail,
	}, nil
}

func deploymentFromResult(request cloudcore.DeployRequest, result *DeployResult) (*cloudcore.Deployment, error) {
	detail, err := json.Marshal(result)
	if err != nil {
		return nil, fmt.Errorf("encode Verda deployment result: %w", err)
	}
	deployment := &cloudcore.Deployment{
		Provider:       ProviderName,
		DryRun:         result.DryRun,
		Request:        request,
		ProviderDetail: detail,
	}
	if result.Instance != nil {
		instance, mapErr := instanceFromProvider(*result.Instance)
		if mapErr != nil {
			return nil, mapErr
		}
		if result.InstanceType != nil {
			instance.Backends = backendsForAccelerator("", result.InstanceType.Model, result.InstanceType.GPUCount)
		}
		deployment.Instance = &instance
	}
	if result.SourceOSVolumeID != "" {
		deployment.Resources = append(deployment.Resources, cloudcore.ResourceRef{Kind: ResourceOSVolume, ID: result.SourceOSVolumeID, Preserve: true})
	}
	seen := make(map[string]bool)
	if result.OSVolumeClone != nil && result.OSVolumeClone.VolumeID != "" {
		seen[result.OSVolumeClone.VolumeID] = true
		deployment.Resources = append(deployment.Resources, cloudcore.ResourceRef{Kind: ResourceOSVolume, ID: result.OSVolumeClone.VolumeID})
	}
	cloneLocations := make([]string, 0, len(result.ClonedOSVolumeIDs))
	for location := range result.ClonedOSVolumeIDs {
		cloneLocations = append(cloneLocations, location)
	}
	sort.Strings(cloneLocations)
	for _, location := range cloneLocations {
		id := result.ClonedOSVolumeIDs[location]
		if id == "" || seen[id] {
			continue
		}
		seen[id] = true
		deployment.Resources = append(deployment.Resources, cloudcore.ResourceRef{Kind: ResourceOSVolume, ID: id, Metadata: map[string]string{"location": location}})
	}
	return deployment, nil
}

func volumeFromProvider(volume Volume) (cloudcore.Volume, error) {
	detail, err := json.Marshal(volume)
	if err != nil {
		return cloudcore.Volume{}, fmt.Errorf("encode Verda volume: %w", err)
	}
	return cloudcore.Volume{
		Provider:       ProviderName,
		ID:             volume.ID,
		Name:           volume.Name,
		State:          volume.Status,
		Location:       volume.Location,
		SizeGiB:        volume.Size,
		Bootable:       volume.IsOSVolume,
		Deleted:        volume.DeletedAt != "" && !volume.IsPermanentlyDeleted,
		ProviderDetail: detail,
	}, nil
}

func instanceState(status string) cloudcore.InstanceState {
	switch strings.ToLower(strings.TrimSpace(status)) {
	case "creating", "initializing", "pending", "starting":
		return cloudcore.InstancePending
	case "active", "ready", "running":
		return cloudcore.InstanceRunning
	case "off", "stopped", "stopping", "shutoff":
		return cloudcore.InstanceStopped
	case "deleting", "destroying":
		return cloudcore.InstanceDestroying
	case "deleted", "destroyed":
		return cloudcore.InstanceDestroyed
	case "canceled", "cancelled", "error", "failed":
		return cloudcore.InstanceFailed
	default:
		return cloudcore.InstanceUnknown
	}
}

func backendsForAccelerator(manufacturer, model string, count int) []string {
	backends := []string{"cpu"}
	if count <= 0 {
		return backends
	}
	identity := strings.ToLower(manufacturer + " " + model)
	if strings.Contains(identity, "amd") || strings.Contains(identity, "instinct") || strings.Contains(identity, "mi2") || strings.Contains(identity, "mi3") {
		return append(backends, "rocm")
	}
	return append(backends, "cuda")
}

func backendsForInstanceType(instanceType string) []string {
	identity := strings.ToLower(instanceType)
	for _, gpu := range []string{"a100", "h100", "h200", "l40", "v100", "rtx", "mi2", "mi3"} {
		if strings.Contains(identity, gpu) {
			return backendsForAccelerator("", identity, 1)
		}
	}
	return []string{"cpu"}
}

var (
	_ cloudcore.Factory               = ProviderFactory{}
	_ cloudcore.Provider              = (*CloudProvider)(nil)
	_ cloudcore.SSHKeyProvider        = (*CloudProvider)(nil)
	_ cloudcore.VolumeProvider        = (*CloudProvider)(nil)
	_ cloudcore.ImageTemplateProvider = (*CloudProvider)(nil)
	_ cloudcore.ImageTemplateFactory  = ProviderFactory{}
)
