package verda

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
	VolumeIDs  []string
	AllDeleted bool
	BaseURL    string
	DryRun     bool
}

type PurgeVolumesRequest struct {
	VolumeIDs  []string `json:"volume_ids,omitempty"`
	AllDeleted bool     `json:"all_deleted,omitempty"`
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
	PermanentDeleteVolume(context.Context, string) error
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
		Request: PurgeVolumesRequest{
			VolumeIDs:  normalized.VolumeIDs,
			AllDeleted: normalized.AllDeleted,
		},
	}
	if normalized.DryRun {
		return result, nil
	}
	if client == nil {
		return nil, fmt.Errorf("verda client is required")
	}

	if normalized.AllDeleted {
		deleted, err := client.ListDeletedVolumes(ctx)
		if err != nil {
			return nil, fmt.Errorf("list deleted Verda volumes: %w", err)
		}
		normalized.VolumeIDs = volumeIDsFromVolumes(restorableDeletedVolumes(deleted))
		result.Request.VolumeIDs = normalized.VolumeIDs
	}

	for _, volumeID := range normalized.VolumeIDs {
		if err := client.PermanentDeleteVolume(ctx, volumeID); err != nil {
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
	if o.AllDeleted && len(o.VolumeIDs) > 0 {
		return PurgeVolumesOptions{}, fmt.Errorf("volume IDs cannot be combined with purge all")
	}
	if !o.AllDeleted && len(o.VolumeIDs) == 0 {
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

func volumeIDsFromVolumes(volumes []Volume) []string {
	ids := make([]string, 0, len(volumes))
	for _, volume := range volumes {
		volumeID := strings.TrimSpace(volume.ID)
		if volumeID != "" {
			ids = append(ids, volumeID)
		}
	}
	sort.Strings(ids)
	return ids
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
