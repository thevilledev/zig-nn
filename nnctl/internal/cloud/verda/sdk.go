package verdacloud

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"time"

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

func (c *SDKClient) ListInstancePrices(ctx context.Context, filters PricingFilters) ([]InstancePrice, error) {
	filters = NormalizePricingFilters(filters)
	if err := ValidatePricingFilters(filters); err != nil {
		return nil, err
	}

	seen := map[string]bool{}
	var prices []InstancePrice
	for _, market := range pricingMarkets(filters.Market) {
		spot := market == PricingMarketSpot
		availability, err := c.availabilityByLocation(ctx, spot)
		if err != nil {
			return nil, err
		}

		locationCodes := filters.LocationCodes
		if len(locationCodes) == 0 {
			locationCodes = make([]string, 0, len(availability))
			for locationCode := range availability {
				locationCodes = append(locationCodes, locationCode)
			}
			sort.Strings(locationCodes)
		}

		for _, locationCode := range locationCodes {
			instanceTypes, err := c.getInstanceTypes(ctx, spot, locationCode, filters.Currency)
			if err != nil {
				return nil, fmt.Errorf("get %s prices for %s: %w", market, locationCode, err)
			}
			for _, info := range instanceTypes {
				available := availability[locationCode][strings.ToUpper(info.InstanceType)]
				price := instancePriceFromInstanceType(info, locationCode, market, available)
				if !InstancePriceMatchesFilters(price, filters) {
					continue
				}
				key := strings.Join([]string{
					strings.ToUpper(price.Market),
					strings.ToUpper(price.LocationCode),
					strings.ToUpper(price.InstanceType),
				}, "\x00")
				if seen[key] {
					continue
				}
				seen[key] = true
				prices = append(prices, price)
			}
		}
	}
	return prices, nil
}

func (c *SDKClient) availabilityByLocation(ctx context.Context, spot bool) (map[string]map[string]bool, error) {
	availabilities, err := c.client.InstanceAvailability.GetAllAvailabilities(ctx, spot, "")
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

func instancePriceFromInstanceType(info sdkverda.InstanceTypeInfo, locationCode, market string, available bool) InstancePrice {
	price, known := instancePriceFromSDK(info, market == PricingMarketSpot)
	return InstancePrice{
		InstanceType: info.InstanceType,
		DisplayName:  info.DisplayName,
		Model:        info.Model,
		Manufacturer: info.Manufacturer,
		GPUCount:     info.GPU.NumberOfGPUs,
		LocationCode: locationCode,
		Market:       market,
		IsSpot:       market == PricingMarketSpot,
		PricePerHour: price,
		PriceKnown:   known,
		Currency:     info.Currency,
		Available:    available,
	}
}

func instancePriceFromSDK(info sdkverda.InstanceTypeInfo, spot bool) (float64, bool) {
	if spot {
		return spotPriceFromSDK(info)
	}
	price := info.PricePerHour.Float64()
	return price, price > 0
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

func (c *SDKClient) CloneVolume(ctx context.Context, req CloneVolumeRequest) (Volume, error) {
	actionReq := cloneVolumeActionRequest(req)
	if actionReq.ID == "" {
		return Volume{}, fmt.Errorf("source volume ID is required")
	}
	if actionReq.Name == "" {
		return Volume{}, fmt.Errorf("clone volume name is required")
	}

	body, err := json.Marshal(actionReq)
	if err != nil {
		return Volume{}, fmt.Errorf("marshal clone volume request: %w", err)
	}
	httpReq, err := c.client.NewRequest(ctx, http.MethodPut, "/volumes", bytes.NewReader(body))
	if err != nil {
		return Volume{}, err
	}

	var volumeIDs []string
	if _, err := c.client.Do(httpReq, &volumeIDs); err != nil {
		return Volume{}, err
	}
	createdID := ""
	if len(volumeIDs) > 0 {
		createdID = strings.TrimSpace(volumeIDs[0])
	}
	if createdID == "" {
		return Volume{}, fmt.Errorf("no volume ID returned from clone operation")
	}

	return Volume{
		ID:       createdID,
		Name:     actionReq.Name,
		Location: actionReq.LocationCode,
	}, nil
}

func cloneVolumeActionRequest(req CloneVolumeRequest) sdkverda.VolumeActionRequest {
	return sdkverda.VolumeActionRequest{
		ID:           strings.TrimSpace(req.SourceVolumeID),
		Action:       sdkverda.VolumeActionClone,
		Name:         strings.TrimSpace(req.Name),
		LocationCode: strings.TrimSpace(req.LocationCode),
	}
}

func (c *SDKClient) WaitVolumeReady(ctx context.Context, volumeID string) (Volume, error) {
	ctx, cancel := context.WithTimeout(ctx, VolumeReadyTimeout)
	defer cancel()

	for {
		volume, err := c.GetVolume(ctx, volumeID)
		if err != nil {
			return Volume{}, err
		}
		if volumeReadyStatus(volume.Status) {
			return volume, nil
		}
		if volumeFailedStatus(volume.Status) {
			return Volume{}, fmt.Errorf("volume %s entered terminal status %q", volumeID, volume.Status)
		}

		timer := time.NewTimer(VolumeReadyPoll)
		select {
		case <-ctx.Done():
			timer.Stop()
			return Volume{}, fmt.Errorf("timed out waiting for volume %s to become ready: %w", volumeID, ctx.Err())
		case <-timer.C:
		}
	}
}

func (c *SDKClient) GetVolume(ctx context.Context, volumeID string) (Volume, error) {
	volume, err := c.client.Volumes.GetVolume(ctx, volumeID)
	if err != nil {
		return Volume{}, err
	}
	return volumeFromSDK(*volume), nil
}

func (c *SDKClient) DeleteVolume(ctx context.Context, volumeID string, force bool) error {
	return c.client.Volumes.DeleteVolume(ctx, volumeID, force)
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

func volumeFromSDK(volume sdkverda.Volume) Volume {
	return Volume{
		ID:         volume.ID,
		Name:       volume.Name,
		Status:     volume.Status,
		Location:   volume.Location,
		IsOSVolume: volume.IsOSVolume,
	}
}

func volumeReadyStatus(status string) bool {
	switch strings.ToLower(strings.TrimSpace(status)) {
	case sdkverda.VolumeStatusCreated, sdkverda.VolumeStatusDetached:
		return true
	default:
		return false
	}
}

func volumeFailedStatus(status string) bool {
	switch strings.ToLower(strings.TrimSpace(status)) {
	case sdkverda.VolumeStatusDeleted, sdkverda.VolumeStatusDeleting, sdkverda.VolumeStatusCanceled, sdkverda.VolumeStatusCanceling:
		return true
	default:
		return false
	}
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
