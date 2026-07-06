package verdacloud

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"sort"
	"strings"

	sdkverda "github.com/verda-cloud/verdacloud-sdk-go/pkg/verda"
)

type ClientOptions struct {
	BaseURL   string
	UserAgent string
}

type SDKClient struct {
	client *sdkverda.Client
}

func NewSDKClient(creds Credentials, opts ClientOptions) (*SDKClient, error) {
	clientOpts := []sdkverda.ClientOption{
		sdkverda.WithClientID(creds.ClientID),
		sdkverda.WithClientSecret(creds.ClientSecret),
	}
	if opts.BaseURL != "" {
		clientOpts = append(clientOpts, sdkverda.WithBaseURL(opts.BaseURL))
	}
	if opts.UserAgent != "" {
		clientOpts = append(clientOpts, sdkverda.WithUserAgent(opts.UserAgent))
	}

	client, err := sdkverda.NewClient(clientOpts...)
	if err != nil {
		return nil, err
	}
	return &SDKClient{client: client}, nil
}

func (c *SDKClient) GetInstanceType(ctx context.Context, instanceType string, spot bool, locationCode string) (InstanceType, error) {
	instanceTypes, err := c.getInstanceTypes(ctx, spot, locationCode, "")
	if err != nil && locationCode != "" {
		instanceTypes, err = c.getInstanceTypes(ctx, spot, "", "")
	}
	if err != nil {
		return InstanceType{}, err
	}

	for _, info := range instanceTypes {
		if strings.EqualFold(info.InstanceType, instanceType) {
			return instanceTypeFromSDK(info), nil
		}
	}
	return InstanceType{}, fmt.Errorf("verda instance type %q was not returned by /instance-types", instanceType)
}

func (c *SDKClient) GetSpotPlacementOptions(ctx context.Context, instanceType string) ([]SpotPlacement, error) {
	availabilities, err := c.client.InstanceAvailability.GetAllAvailabilities(ctx, true, "")
	if err != nil {
		return nil, err
	}

	var options []SpotPlacement
	for _, availability := range availabilities {
		for _, availableType := range availability.Availabilities {
			if strings.EqualFold(availableType, instanceType) {
				option := SpotPlacement{LocationCode: availability.LocationCode}
				instanceTypeInfo, err := c.GetInstanceType(ctx, instanceType, true, availability.LocationCode)
				if err == nil {
					option.SpotPrice = instanceTypeInfo.SpotPrice
					option.PriceKnown = instanceTypeInfo.PriceKnown
					option.Currency = instanceTypeInfo.Currency
				}
				options = append(options, option)
				break
			}
		}
	}
	return options, nil
}

func (c *SDKClient) ListInstances(ctx context.Context, status string) ([]Instance, error) {
	instances, err := c.client.Instances.Get(ctx, strings.ToLower(strings.TrimSpace(status)))
	if err != nil {
		return nil, err
	}

	result := make([]Instance, 0, len(instances))
	for _, instance := range instances {
		result = append(result, instanceFromSDK(instance))
	}
	sort.SliceStable(result, func(i, j int) bool {
		if result[i].CreatedAt != result[j].CreatedAt {
			return result[i].CreatedAt > result[j].CreatedAt
		}
		return result[i].ID < result[j].ID
	})
	return result, nil
}

func (c *SDKClient) ListSpotPrices(ctx context.Context, filters PricingFilters) ([]SpotPrice, error) {
	filters = NormalizePricingFilters(filters)
	availability, err := c.spotAvailabilityByLocation(ctx)
	if err != nil {
		return nil, err
	}

	locationCodes := filters.LocationCodes
	if len(locationCodes) == 0 {
		for locationCode := range availability {
			locationCodes = append(locationCodes, locationCode)
		}
		sort.Strings(locationCodes)
	}

	seen := map[string]bool{}
	var prices []SpotPrice
	for _, locationCode := range locationCodes {
		instanceTypes, err := c.getInstanceTypes(ctx, true, locationCode, filters.Currency)
		if err != nil {
			return nil, fmt.Errorf("get spot prices for %s: %w", locationCode, err)
		}
		for _, info := range instanceTypes {
			available := availability[locationCode][strings.ToUpper(info.InstanceType)]
			price := spotPriceFromInstanceType(info, locationCode, available)
			if !SpotPriceMatchesFilters(price, filters) {
				continue
			}
			key := strings.ToUpper(price.LocationCode) + "\x00" + strings.ToUpper(price.InstanceType)
			if seen[key] {
				continue
			}
			seen[key] = true
			prices = append(prices, price)
		}
	}
	return prices, nil
}

func (c *SDKClient) spotAvailabilityByLocation(ctx context.Context) (map[string]map[string]bool, error) {
	availabilities, err := c.client.InstanceAvailability.GetAllAvailabilities(ctx, true, "")
	if err != nil {
		return nil, err
	}

	result := map[string]map[string]bool{}
	for _, availability := range availabilities {
		locationCode := strings.ToUpper(strings.TrimSpace(availability.LocationCode))
		if locationCode == "" {
			continue
		}
		if result[locationCode] == nil {
			result[locationCode] = map[string]bool{}
		}
		for _, instanceType := range availability.Availabilities {
			result[locationCode][strings.ToUpper(instanceType)] = true
		}
	}
	return result, nil
}

func (c *SDKClient) getInstanceTypes(ctx context.Context, spot bool, locationCode, currency string) ([]sdkverda.InstanceTypeInfo, error) {
	params := url.Values{}
	if spot {
		params.Set("is_spot", "true")
	}
	if locationCode != "" {
		params.Set("location_code", locationCode)
	}
	if currency != "" {
		params.Set("currency", currency)
	}

	path := "/instance-types"
	if len(params) > 0 {
		path += "?" + params.Encode()
	}

	var instanceTypes []sdkverda.InstanceTypeInfo
	req, err := c.client.NewRequest(ctx, http.MethodGet, path, nil)
	if err != nil {
		return nil, err
	}
	if _, err := c.client.Do(req, &instanceTypes); err != nil {
		return nil, err
	}
	return instanceTypes, nil
}

func instanceTypeFromSDK(info sdkverda.InstanceTypeInfo) InstanceType {
	price, known := spotPriceFromSDK(info)
	return InstanceType{
		InstanceType: info.InstanceType,
		DisplayName:  info.DisplayName,
		Model:        info.Model,
		GPUCount:     info.GPU.NumberOfGPUs,
		SpotPrice:    price,
		PriceKnown:   known,
		Currency:     info.Currency,
	}
}

func spotPriceFromInstanceType(info sdkverda.InstanceTypeInfo, locationCode string, available bool) SpotPrice {
	price, known := spotPriceFromSDK(info)
	return SpotPrice{
		InstanceType: info.InstanceType,
		DisplayName:  info.DisplayName,
		Model:        info.Model,
		Manufacturer: info.Manufacturer,
		GPUCount:     info.GPU.NumberOfGPUs,
		LocationCode: locationCode,
		SpotPrice:    price,
		PriceKnown:   known,
		Currency:     info.Currency,
		Available:    available,
	}
}

func spotPriceFromSDK(info sdkverda.InstanceTypeInfo) (float64, bool) {
	for _, price := range []float64{
		info.SpotPrice.Float64(),
		info.DynamicPrice.Float64(),
		info.PricePerHour.Float64(),
	} {
		if price > 0 {
			return price, true
		}
	}
	return 0, false
}

func (c *SDKClient) ListSSHKeys(ctx context.Context) ([]SSHKey, error) {
	keys, err := c.client.SSHKeys.GetAllSSHKeys(ctx)
	if err != nil {
		return nil, err
	}

	result := make([]SSHKey, 0, len(keys))
	for _, key := range keys {
		result = append(result, SSHKey{
			ID:          key.ID,
			Name:        key.Name,
			Fingerprint: key.Fingerprint,
			PublicKey:   key.PublicKey,
		})
	}
	return result, nil
}

func (c *SDKClient) CreateStartupScript(ctx context.Context, name, script string) (StartupScript, error) {
	created, err := c.client.StartupScripts.AddStartupScript(ctx, &sdkverda.CreateStartupScriptRequest{
		Name:   name,
		Script: script,
	})
	if err != nil {
		return StartupScript{}, err
	}
	return StartupScript{
		ID:   created.ID,
		Name: created.Name,
	}, nil
}

func (c *SDKClient) CreateInstance(ctx context.Context, req CreateInstanceRequest) (Instance, error) {
	sdkReq := sdkverda.CreateInstanceRequest{
		InstanceType: req.InstanceType,
		Image:        req.Image,
		Hostname:     req.Hostname,
		Description:  req.Description,
		SSHKeyIDs:    req.SSHKeyIDs,
		LocationCode: req.LocationCode,
		Contract:     req.Contract,
		IsSpot:       req.IsSpot,
	}
	if req.StartupScriptID != "" {
		sdkReq.StartupScriptID = &req.StartupScriptID
	}

	created, err := c.client.Instances.Create(ctx, sdkReq)
	if err != nil {
		return Instance{}, err
	}
	return instanceFromSDK(*created), nil
}

func (c *SDKClient) DestroyInstances(ctx context.Context, req DestroyInstanceRequest) ([]DestroyInstanceResult, error) {
	results, err := c.client.Instances.Action(ctx, sdkverda.InstanceActionRequest{
		Action:            sdkverda.ActionDelete,
		ID:                req.InstanceIDs,
		VolumeIDs:         req.VolumeIDs,
		DeletePermanently: req.DeletePermanently,
	})
	if err != nil {
		return nil, err
	}

	destroyResults := make([]DestroyInstanceResult, 0, len(results))
	for _, result := range results {
		destroyResults = append(destroyResults, DestroyInstanceResult{
			InstanceID: result.InstanceID,
			Status:     result.Status,
			Error:      result.Error,
			StatusCode: result.StatusCode,
		})
	}
	return destroyResults, nil
}

func instanceFromSDK(instance sdkverda.Instance) Instance {
	createdAt := ""
	if !instance.CreatedAt.IsZero() {
		createdAt = instance.CreatedAt.Format("2006-01-02T15:04:05Z07:00")
	}
	return Instance{
		ID:              instance.ID,
		Hostname:        instance.Hostname,
		Status:          instance.Status,
		IP:              instance.IP,
		InstanceType:    instance.InstanceType,
		Image:           instance.Image,
		Location:        instance.Location,
		IsSpot:          instance.IsSpot,
		StartupScriptID: instance.StartupScriptID,
		CreatedAt:       createdAt,
		PricePerHour:    instance.PricePerHour.Float64(),
	}
}
