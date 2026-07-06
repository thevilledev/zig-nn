package verdacloud

import (
	"context"
	"fmt"
	"strings"
)

type DestroyOptions struct {
	InstanceIDs       []string
	VolumeIDs         []string
	DeletePermanently bool
	BaseURL           string
	DryRun            bool
}

type DestroyInstanceRequest struct {
	InstanceIDs       []string `json:"instance_ids"`
	VolumeIDs         []string `json:"volume_ids,omitempty"`
	DeletePermanently bool     `json:"delete_permanently"`
}

type DestroyInstanceResult struct {
	InstanceID string `json:"instance_id"`
	Status     string `json:"status"`
	Error      string `json:"error,omitempty"`
	StatusCode int    `json:"status_code,omitempty"`
}

type DestroyResult struct {
	Provider string                  `json:"provider"`
	DryRun   bool                    `json:"dry_run"`
	Request  DestroyInstanceRequest  `json:"request"`
	Results  []DestroyInstanceResult `json:"results,omitempty"`
}

type DestroyClient interface {
	DestroyInstances(context.Context, DestroyInstanceRequest) ([]DestroyInstanceResult, error)
}

func DefaultDestroyOptions(instanceIDs []string) DestroyOptions {
	return DestroyOptions{
		InstanceIDs:       instanceIDs,
		DeletePermanently: true,
	}
}

func Destroy(ctx context.Context, client DestroyClient, opts DestroyOptions) (*DestroyResult, error) {
	normalized, err := opts.normalized()
	if err != nil {
		return nil, err
	}

	req := DestroyInstanceRequest{
		InstanceIDs:       normalized.InstanceIDs,
		VolumeIDs:         normalized.VolumeIDs,
		DeletePermanently: normalized.DeletePermanently,
	}
	result := &DestroyResult{
		Provider: ProviderName,
		DryRun:   normalized.DryRun,
		Request:  req,
	}
	if normalized.DryRun {
		return result, nil
	}
	if client == nil {
		return nil, fmt.Errorf("verda client is required")
	}

	results, err := client.DestroyInstances(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("destroy Verda instances: %w", err)
	}
	result.Results = results
	if err := validateDestroyResults(results); err != nil {
		return nil, err
	}
	return result, nil
}

func (o DestroyOptions) normalized() (DestroyOptions, error) {
	o.InstanceIDs = compactStrings(o.InstanceIDs)
	o.VolumeIDs = compactStrings(o.VolumeIDs)
	o.BaseURL = strings.TrimSpace(o.BaseURL)

	if len(o.InstanceIDs) == 0 {
		return DestroyOptions{}, fmt.Errorf("at least one instance ID is required")
	}
	return o, nil
}

func validateDestroyResults(results []DestroyInstanceResult) error {
	for _, result := range results {
		if result.Error != "" {
			return fmt.Errorf("destroy Verda instance %s: %s", result.InstanceID, result.Error)
		}
		if result.StatusCode >= 400 {
			return fmt.Errorf("destroy Verda instance %s failed with status %d", result.InstanceID, result.StatusCode)
		}
	}
	return nil
}
