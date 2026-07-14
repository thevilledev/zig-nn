package verdacloud

import (
	"context"
	"fmt"
	"sort"
	"strings"
)

type ListVolumesOptions struct {
	IncludeDeleted bool
}

type PurgeVolumesOptions struct {
	VolumeIDs []string
	BaseURL   string
	DryRun    bool
}

type PurgeVolumesRequest struct {
	VolumeIDs []string `json:"volume_ids"`
}

type PurgeVolumeResult struct {
	VolumeID string `json:"volume_id"`
	Status   string `json:"status"`
}

type PurgeVolumesResult struct {
	Provider string              `json:"provider"`
	DryRun   bool                `json:"dry_run"`
	Request  PurgeVolumesRequest `json:"request"`
	Results  []PurgeVolumeResult `json:"results,omitempty"`
}

type VolumeListClient interface {
	ListVolumes(context.Context) ([]Volume, error)
	ListDeletedVolumes(context.Context) ([]Volume, error)
}

type VolumePurgeClient interface {
	ListDeletedVolumes(context.Context) ([]Volume, error)
	DeleteVolume(context.Context, string, bool) error
}

func ListVolumes(ctx context.Context, client VolumeListClient, opts ListVolumesOptions) ([]Volume, error) {
	if client == nil {
		return nil, fmt.Errorf("verda client is required")
	}

	volumes, err := client.ListVolumes(ctx)
	if err != nil {
		return nil, fmt.Errorf("list Verda volumes: %w", err)
	}
	volumes = activeVolumes(volumes)

	if opts.IncludeDeleted {
		deleted, err := client.ListDeletedVolumes(ctx)
		if err != nil {
			return nil, fmt.Errorf("list deleted Verda volumes: %w", err)
		}
		volumes = append(volumes, restorableDeletedVolumes(deleted)...)
	}

	sortVolumes(volumes)
	return volumes, nil
}

func PurgeVolumes(ctx context.Context, client VolumePurgeClient, opts PurgeVolumesOptions) (*PurgeVolumesResult, error) {
	normalized, err := opts.normalized()
	if err != nil {
		return nil, err
	}

	result := &PurgeVolumesResult{
		Provider: ProviderName,
		DryRun:   normalized.DryRun,
		Request:  PurgeVolumesRequest{VolumeIDs: normalized.VolumeIDs},
	}
	if normalized.DryRun {
		return result, nil
	}
	if client == nil {
		return nil, fmt.Errorf("verda client is required")
	}

	deleted, err := client.ListDeletedVolumes(ctx)
	if err != nil {
		return nil, fmt.Errorf("list deleted Verda volumes: %w", err)
	}
	purgeable := purgeableVolumeIDs(deleted)
	for _, volumeID := range normalized.VolumeIDs {
		if !purgeable[volumeID] {
			return nil, fmt.Errorf("verda volume %s is not in trash; run `nnctl cloud volume --include-deleted` to find purgeable volumes", volumeID)
		}
	}

	for _, volumeID := range normalized.VolumeIDs {
		if err := client.DeleteVolume(ctx, volumeID, true); err != nil {
			return nil, fmt.Errorf("purge Verda volume %s: %w", volumeID, err)
		}
		result.Results = append(result.Results, PurgeVolumeResult{
			VolumeID: volumeID,
			Status:   "purged",
		})
	}
	return result, nil
}

func (o PurgeVolumesOptions) normalized() (PurgeVolumesOptions, error) {
	o.VolumeIDs = compactStrings(o.VolumeIDs)
	o.BaseURL = strings.TrimSpace(o.BaseURL)
	if len(o.VolumeIDs) == 0 {
		return PurgeVolumesOptions{}, fmt.Errorf("at least one volume ID is required")
	}
	return o, nil
}

func activeVolumes(volumes []Volume) []Volume {
	active := volumes[:0]
	for _, volume := range volumes {
		if isDeletedVolume(volume) {
			continue
		}
		active = append(active, volume)
	}
	return active
}

func restorableDeletedVolumes(volumes []Volume) []Volume {
	restorable := volumes[:0]
	for _, volume := range volumes {
		if volume.IsPermanentlyDeleted {
			continue
		}
		restorable = append(restorable, volume)
	}
	return restorable
}

func purgeableVolumeIDs(volumes []Volume) map[string]bool {
	purgeable := map[string]bool{}
	for _, volume := range restorableDeletedVolumes(volumes) {
		volumeID := strings.TrimSpace(volume.ID)
		if volumeID != "" {
			purgeable[volumeID] = true
		}
	}
	return purgeable
}

func sortVolumes(volumes []Volume) {
	sort.SliceStable(volumes, func(i, j int) bool {
		leftTime := firstNonEmptyString(volumes[i].DeletedAt, volumes[i].CreatedAt)
		rightTime := firstNonEmptyString(volumes[j].DeletedAt, volumes[j].CreatedAt)
		if leftTime != rightTime {
			return leftTime > rightTime
		}
		if volumes[i].Location != volumes[j].Location {
			return volumes[i].Location < volumes[j].Location
		}
		if volumes[i].Name != volumes[j].Name {
			return volumes[i].Name < volumes[j].Name
		}
		return volumes[i].ID < volumes[j].ID
	})
}

func isDeletedVolume(volume Volume) bool {
	if volume.IsPermanentlyDeleted || strings.TrimSpace(volume.DeletedAt) != "" {
		return true
	}
	switch strings.ToLower(strings.TrimSpace(volume.Status)) {
	case "deleted", "deleting":
		return true
	default:
		return false
	}
}
