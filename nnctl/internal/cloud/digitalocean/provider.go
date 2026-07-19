package digitalocean

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

type ProviderFactory struct {
	CredentialStore CredentialStore
	ClientOptions   ClientOptions
	NewClient       func(string, ClientOptions) (Client, error)
}

func (ProviderFactory) Descriptor() cloudcore.Descriptor {
	return providerDescriptor()
}

func (f ProviderFactory) Status(ctx context.Context, _ cloudcore.Configuration) cloudcore.ConfigurationStatus {
	_, err := f.credentialStore().Token(ctx)
	if err != nil {
		return cloudcore.ConfigurationStatus{Error: err.Error()}
	}
	return cloudcore.ConfigurationStatus{Configured: true}
}

func (f ProviderFactory) New(ctx context.Context, configuration cloudcore.Configuration) (cloudcore.Provider, error) {
	manual, err := parseManualOfferings(configuration.Options[OptionOfferings])
	if err != nil {
		return nil, err
	}
	defaultSSHKeyIDs := compactStrings(strings.Split(configuration.Options[OptionSSHKeyIDs], ","))
	if configuration.Unauthenticated {
		return NewCloudProvider(nil, manual, defaultSSHKeyIDs), nil
	}
	token, err := f.credentialStore().Token(ctx)
	if err != nil {
		return nil, err
	}
	options := f.ClientOptions
	if strings.TrimSpace(configuration.BaseURL) != "" {
		options.BaseURL = configuration.BaseURL
	}
	if strings.TrimSpace(configuration.UserAgent) != "" {
		options.UserAgent = configuration.UserAgent
	}
	newClient := f.NewClient
	if newClient == nil {
		newClient = func(token string, options ClientOptions) (Client, error) {
			return NewClient(token, options)
		}
	}
	client, err := newClient(token, options)
	if err != nil {
		return nil, fmt.Errorf("create DigitalOcean client: %w", err)
	}
	return NewCloudProvider(client, manual, defaultSSHKeyIDs), nil
}

func (f ProviderFactory) credentialStore() CredentialStore {
	if f.CredentialStore != nil {
		return f.CredentialStore
	}
	return KeyringCredentialStore{}
}

type CloudProvider struct {
	client          Client
	manualOfferings []cloudcore.Offering
	defaultSSHKeys  []string
}

func NewCloudProvider(client Client, manualOfferings []cloudcore.Offering, defaultSSHKeys []string) *CloudProvider {
	return &CloudProvider{
		client: client, manualOfferings: cloneOfferings(manualOfferings),
		defaultSSHKeys: compactStrings(defaultSSHKeys),
	}
}

func (*CloudProvider) Descriptor() cloudcore.Descriptor {
	return providerDescriptor()
}

func providerDescriptor() cloudcore.Descriptor {
	return cloudcore.Descriptor{
		Name: ProviderName, DisplayName: "DigitalOcean",
		Capabilities: []cloudcore.Capability{
			cloudcore.CapabilityCompute,
			cloudcore.CapabilityPricing,
			cloudcore.CapabilitySSHKeys,
			cloudcore.CapabilityBenchmarkImage,
		},
	}
}

func (p *CloudProvider) Offerings(ctx context.Context, filters cloudcore.OfferingFilters) ([]cloudcore.Offering, error) {
	if p == nil || p.client == nil {
		return nil, errors.New("DigitalOcean provider client is required")
	}
	sizes, err := p.client.ListSizes(ctx)
	if err != nil {
		return nil, fmt.Errorf("list DigitalOcean sizes: %w", err)
	}
	result := make([]cloudcore.Offering, 0)
	seen := make(map[string]struct{})
	for _, size := range sizes {
		if size.GPUInfo == nil || size.GPUInfo.Count <= 0 {
			continue
		}
		plan, mapErr := planFromSize(size)
		if mapErr != nil {
			return nil, mapErr
		}
		for _, region := range size.Regions {
			offering := offeringFromPlan(plan, region, size.Available, true)
			if !offeringMatches(offering, filters) {
				continue
			}
			key := offeringKey(offering)
			seen[key] = struct{}{}
			result = append(result, offering)
		}
	}
	manual := append(cloneOfferings(p.manualOfferings), explicitContractOfferings(filters)...)
	for _, offering := range manual {
		if !offeringMatches(offering, filters) {
			continue
		}
		key := offeringKey(offering)
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		result = append(result, offering)
	}
	sort.SliceStable(result, func(left, right int) bool {
		if result[left].PriceKnown != result[right].PriceKnown {
			return result[left].PriceKnown
		}
		if result[left].PriceKnown && result[left].PricePerHour != result[right].PricePerHour {
			return result[left].PricePerHour < result[right].PricePerHour
		}
		if result[left].ID != result[right].ID {
			return result[left].ID < result[right].ID
		}
		return result[left].Location < result[right].Location
	})
	return result, nil
}

func (p *CloudProvider) Deploy(ctx context.Context, request cloudcore.DeployRequest) (*cloudcore.Deployment, error) {
	normalized, plan, create, err := normalizeDeployRequest(request)
	if err != nil {
		return nil, err
	}
	if request.DryRun {
		return &cloudcore.Deployment{Provider: ProviderName, DryRun: true, Request: normalized}, nil
	}
	if p == nil || p.client == nil {
		return nil, errors.New("DigitalOcean provider client is required")
	}
	if len(create.SSHKeys) == 0 {
		create.SSHKeys = append([]string(nil), p.defaultSSHKeys...)
		normalized.SSHKeyIDs = append([]string(nil), p.defaultSSHKeys...)
	}
	if len(create.SSHKeys) == 0 {
		return nil, errors.New("DigitalOcean deployment requires --ssh-key-id or a configured cloud SSH key")
	}
	droplet, err := p.client.CreateDroplet(ctx, create)
	if err != nil {
		return nil, fmt.Errorf("create DigitalOcean GPU Droplet: %w", err)
	}
	instance, err := instanceFromDroplet(droplet)
	if err != nil {
		return nil, err
	}
	if instance.OfferingID == "" {
		instance.OfferingID = plan.Slug
	}
	if instance.Location == "" {
		instance.Location = normalized.Location
	}
	if instance.PricePerHour == 0 {
		instance.PricePerHour = plan.Price
	}
	instance.Currency = CurrencyUSD
	detail, err := json.Marshal(struct {
		Request CreateDropletRequest `json:"request"`
		Droplet Droplet              `json:"droplet"`
	}{Request: create, Droplet: droplet})
	if err != nil {
		return nil, fmt.Errorf("encode DigitalOcean deployment details: %w", err)
	}
	return &cloudcore.Deployment{
		Provider: ProviderName, Request: normalized, Instance: &instance,
		ProviderDetail: detail,
	}, nil
}

func (p *CloudProvider) Instance(ctx context.Context, id string) (cloudcore.Instance, error) {
	if p == nil || p.client == nil {
		return cloudcore.Instance{}, errors.New("DigitalOcean provider client is required")
	}
	id, err := normalizeDropletID(id)
	if err != nil {
		return cloudcore.Instance{}, err
	}
	droplet, err := p.client.GetDroplet(ctx, id)
	if err != nil {
		return cloudcore.Instance{}, fmt.Errorf("get DigitalOcean Droplet %s: %w", id, err)
	}
	return instanceFromDroplet(droplet)
}

func (p *CloudProvider) Instances(ctx context.Context, options cloudcore.ListOptions) ([]cloudcore.Instance, error) {
	if p == nil || p.client == nil {
		return nil, errors.New("DigitalOcean provider client is required")
	}
	droplets, err := p.client.ListDroplets(ctx)
	if err != nil {
		return nil, fmt.Errorf("list DigitalOcean Droplets: %w", err)
	}
	status := strings.ToLower(strings.TrimSpace(options.Status))
	result := make([]cloudcore.Instance, 0, len(droplets))
	for _, droplet := range droplets {
		if !isGPUDroplet(droplet) {
			continue
		}
		instance, mapErr := instanceFromDroplet(droplet)
		if mapErr != nil {
			return nil, mapErr
		}
		if status != "" && status != strings.ToLower(strings.TrimSpace(droplet.Status)) && status != string(instance.State) {
			continue
		}
		if !options.All && status == "" && instance.State.Terminal() {
			continue
		}
		result = append(result, instance)
	}
	return result, nil
}

func (p *CloudProvider) Destroy(ctx context.Context, request cloudcore.DestroyRequest) (*cloudcore.DestroyResult, error) {
	ids := make([]string, 0, len(request.InstanceIDs))
	for _, rawID := range request.InstanceIDs {
		id, err := normalizeDropletID(rawID)
		if err != nil {
			return nil, err
		}
		ids = append(ids, id)
	}
	if len(ids) == 0 {
		return nil, errors.New("at least one DigitalOcean Droplet ID is required")
	}
	normalized := request
	normalized.InstanceIDs = ids
	result := &cloudcore.DestroyResult{Provider: ProviderName, DryRun: request.DryRun, Request: normalized}
	if request.DryRun {
		return result, nil
	}
	if p == nil || p.client == nil {
		return nil, errors.New("DigitalOcean provider client is required")
	}
	for _, id := range ids {
		if err := p.client.DeleteDroplet(ctx, id); err != nil {
			if result.Errors == nil {
				result.Errors = make(map[string]string)
			}
			result.Errors[id] = err.Error()
		}
	}
	if len(result.Errors) > 0 {
		return result, fmt.Errorf("destroy DigitalOcean Droplets: %s", joinDestroyErrors(result.Errors))
	}
	return result, nil
}

func (p *CloudProvider) SSHKeys(ctx context.Context) ([]cloudcore.SSHKey, error) {
	if p == nil || p.client == nil {
		return nil, errors.New("DigitalOcean provider client is required")
	}
	keys, err := p.client.ListSSHKeys(ctx)
	if err != nil {
		return nil, fmt.Errorf("list DigitalOcean SSH keys: %w", err)
	}
	result := make([]cloudcore.SSHKey, 0, len(keys))
	for _, key := range keys {
		detail, marshalErr := json.Marshal(key)
		if marshalErr != nil {
			return nil, fmt.Errorf("encode DigitalOcean SSH key: %w", marshalErr)
		}
		result = append(result, cloudcore.SSHKey{
			Provider: ProviderName, ID: strconv.Itoa(key.ID), Name: key.Name,
			Fingerprint: key.Fingerprint, ProviderDetail: detail,
		})
	}
	return result, nil
}

func normalizeDeployRequest(request cloudcore.DeployRequest) (cloudcore.DeployRequest, gpuPlan, CreateDropletRequest, error) {
	request.OfferingID = strings.ToLower(strings.TrimSpace(request.OfferingID))
	request.Location = strings.ToLower(strings.TrimSpace(request.Location))
	request.Market = strings.ToLower(strings.TrimSpace(request.Market))
	request.Image = strings.TrimSpace(request.Image)
	request.Name = strings.TrimSpace(request.Name)
	request.SSHKeyIDs = compactStrings(request.SSHKeyIDs)
	if request.OfferingID == "" {
		return cloudcore.DeployRequest{}, gpuPlan{}, CreateDropletRequest{}, errors.New("DigitalOcean size slug is required")
	}
	if request.Location == "" {
		return cloudcore.DeployRequest{}, gpuPlan{}, CreateDropletRequest{}, errors.New("DigitalOcean region is required")
	}
	if request.Market == "" {
		request.Market = MarketOnDemand
	}
	if request.Market != MarketOnDemand {
		return cloudcore.DeployRequest{}, gpuPlan{}, CreateDropletRequest{}, errors.New("DigitalOcean GPU Droplets support only the on-demand market")
	}
	plan, err := planForSlug(request.OfferingID)
	if err != nil {
		return cloudcore.DeployRequest{}, gpuPlan{}, CreateDropletRequest{}, err
	}
	if request.Image == "" {
		request.Image = imageForPlan(plan)
	}
	if request.Name == "" {
		request.Name = "nnctl-" + strings.TrimPrefix(plan.Slug, "gpu-")
	}
	if strings.TrimSpace(request.UserData) == "" {
		request.UserData = DefaultUserDataScript
	}
	create := CreateDropletRequest{
		Name: request.Name, Region: request.Location, Size: plan.Slug, Image: request.Image,
		SSHKeys: append([]string(nil), request.SSHKeyIDs...), UserData: request.UserData,
		PublicNetworking: true, Tags: []string{"nnctl"},
	}
	return request, plan, create, nil
}

func instanceFromDroplet(droplet Droplet) (cloudcore.Instance, error) {
	id := ""
	if droplet.ID > 0 {
		id = strconv.Itoa(droplet.ID)
	}
	slug := strings.ToLower(strings.TrimSpace(droplet.SizeSlug))
	if slug == "" {
		slug = strings.ToLower(strings.TrimSpace(droplet.Size.Slug))
	}
	var plan gpuPlan
	var err error
	if droplet.GPUInfo != nil {
		size := droplet.Size
		size.Slug = slug
		size.GPUInfo = droplet.GPUInfo
		plan, err = planFromSize(size)
	} else {
		plan, err = planForSlug(slug)
	}
	if err != nil {
		return cloudcore.Instance{}, fmt.Errorf("map DigitalOcean Droplet %s: %w", id, err)
	}
	price := droplet.Size.PriceHourly
	if price == 0 {
		price = plan.Price
	}
	detail, err := json.Marshal(droplet)
	if err != nil {
		return cloudcore.Instance{}, fmt.Errorf("encode DigitalOcean Droplet: %w", err)
	}
	return cloudcore.Instance{
		Provider: ProviderName, ID: id, Name: droplet.Name,
		State: mapDropletState(droplet.Status), PublicIP: publicIP(droplet.Networks),
		OfferingID: slug, Location: strings.ToLower(strings.TrimSpace(droplet.Region.Slug)),
		Market: MarketOnDemand, Backends: backendsForPlan(plan),
		PricePerHour: price, Currency: CurrencyUSD, ProviderDetail: detail,
	}, nil
}

func mapDropletState(status string) cloudcore.InstanceState {
	switch strings.ToLower(strings.TrimSpace(status)) {
	case "new":
		return cloudcore.InstancePending
	case "active":
		return cloudcore.InstanceRunning
	case "off":
		return cloudcore.InstanceStopped
	case "archive", "deleted":
		return cloudcore.InstanceDestroyed
	default:
		return cloudcore.InstanceUnknown
	}
}

func publicIP(networks Networks) string {
	for _, network := range networks.V4 {
		if strings.EqualFold(network.Type, "public") && strings.TrimSpace(network.IPAddress) != "" {
			return strings.TrimSpace(network.IPAddress)
		}
	}
	for _, network := range networks.V6 {
		if strings.EqualFold(network.Type, "public") && strings.TrimSpace(network.IPAddress) != "" {
			return strings.TrimSpace(network.IPAddress)
		}
	}
	return ""
}

func isGPUDroplet(droplet Droplet) bool {
	return droplet.GPUInfo != nil || droplet.Size.GPUInfo != nil ||
		strings.HasPrefix(strings.ToLower(strings.TrimSpace(droplet.SizeSlug)), "gpu-") ||
		strings.HasPrefix(strings.ToLower(strings.TrimSpace(droplet.Size.Slug)), "gpu-")
}

func offeringMatches(offering cloudcore.Offering, filters cloudcore.OfferingFilters) bool {
	if filters.AvailableOnly && !offering.Available {
		return false
	}
	if len(filters.IDs) > 0 && !containsFold(filters.IDs, offering.ID) {
		return false
	}
	if len(filters.Locations) > 0 && !containsFold(filters.Locations, offering.Location) {
		return false
	}
	if len(filters.Markets) > 0 && !containsFold(filters.Markets, offering.Market) && !containsFold(filters.Markets, "all") {
		return false
	}
	if len(filters.GPUCounts) > 0 && !containsInt(filters.GPUCounts, offering.Accelerator.Count) {
		return false
	}
	if filters.Manufacturer != "" && !strings.Contains(strings.ToLower(offering.Accelerator.Manufacturer), strings.ToLower(strings.TrimSpace(filters.Manufacturer))) {
		return false
	}
	if filters.Model != "" {
		haystack := strings.ToLower(strings.Join([]string{offering.ID, offering.DisplayName, offering.Accelerator.Model}, " "))
		if !strings.Contains(haystack, strings.ToLower(strings.TrimSpace(filters.Model))) {
			return false
		}
	}
	if filters.Currency != "" && !strings.EqualFold(filters.Currency, offering.Currency) {
		return false
	}
	return true
}

func explicitContractOfferings(filters cloudcore.OfferingFilters) []cloudcore.Offering {
	if len(filters.IDs) == 0 || len(filters.Locations) == 0 {
		return nil
	}
	var result []cloudcore.Offering
	for _, id := range filters.IDs {
		plan, ok := gpuPlans[strings.ToLower(strings.TrimSpace(id))]
		if !ok || !plan.Contract {
			continue
		}
		for _, location := range filters.Locations {
			result = append(result, offeringFromPlan(plan, location, true, false))
		}
	}
	return result
}

func normalizeDropletID(raw string) (string, error) {
	id := strings.TrimSpace(raw)
	numeric, err := strconv.ParseInt(id, 10, 64)
	if err != nil || numeric <= 0 {
		return "", fmt.Errorf("DigitalOcean Droplet ID %q must be a positive integer", raw)
	}
	return strconv.FormatInt(numeric, 10), nil
}

func joinDestroyErrors(values map[string]string) string {
	ids := make([]string, 0, len(values))
	for id := range values {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	parts := make([]string, 0, len(ids))
	for _, id := range ids {
		parts = append(parts, id+": "+values[id])
	}
	return strings.Join(parts, "; ")
}

func cloneOfferings(values []cloudcore.Offering) []cloudcore.Offering {
	result := make([]cloudcore.Offering, len(values))
	for index, value := range values {
		result[index] = value
		result[index].Backends = append([]string(nil), value.Backends...)
	}
	return result
}

func offeringKey(offering cloudcore.Offering) string {
	return strings.ToLower(offering.ID + "@" + offering.Location)
}

func compactStrings(values []string) []string {
	result := make([]string, 0, len(values))
	seen := make(map[string]struct{})
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value == "" {
			continue
		}
		key := strings.ToLower(value)
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		result = append(result, value)
	}
	return result
}

func containsFold(values []string, target string) bool {
	for _, value := range values {
		if strings.EqualFold(strings.TrimSpace(value), strings.TrimSpace(target)) {
			return true
		}
	}
	return false
}

func containsInt(values []int, target int) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}

var (
	_ cloudcore.Factory        = ProviderFactory{}
	_ cloudcore.Provider       = (*CloudProvider)(nil)
	_ cloudcore.SSHKeyProvider = (*CloudProvider)(nil)
)
