package verdacloud

import (
	"context"
	"errors"
	"strings"
	"testing"
)

func TestListVolumesExcludesDeletedByDefault(t *testing.T) {
	client := &fakeVolumeClient{
		volumes: []Volume{
			{ID: "vol-active", Status: "detached"},
			{ID: "vol-deleted", Status: "deleted"},
		},
		deletedVolumes: []Volume{{ID: "vol-trash", Status: "deleted"}},
	}

	volumes, err := ListVolumes(context.Background(), client, ListVolumesOptions{})
	if err != nil {
		t.Fatal(err)
	}

	if len(volumes) != 1 || volumes[0].ID != "vol-active" {
		t.Fatalf("unexpected volumes: %#v", volumes)
	}
	if client.listDeletedCalled {
		t.Fatal("ListDeletedVolumes was called without IncludeDeleted")
	}
}

func TestListVolumesIncludesRestorableDeleted(t *testing.T) {
	client := &fakeVolumeClient{
		volumes: []Volume{{ID: "vol-active", Status: "detached"}},
		deletedVolumes: []Volume{
			{ID: "vol-trash", Status: "deleted", DeletedAt: "2026-07-14T10:00:00Z"},
			{ID: "vol-gone", Status: "deleted", IsPermanentlyDeleted: true},
		},
	}

	volumes, err := ListVolumes(context.Background(), client, ListVolumesOptions{IncludeDeleted: true})
	if err != nil {
		t.Fatal(err)
	}

	ids := volumeIDSet(volumes)
	if !ids["vol-active"] || !ids["vol-trash"] {
		t.Fatalf("expected active and trash volumes: %#v", volumes)
	}
	if ids["vol-gone"] {
		t.Fatalf("permanently deleted volume was listed: %#v", volumes)
	}
}

func TestPurgeVolumesDryRunBuildsRequestWithoutClient(t *testing.T) {
	result, err := PurgeVolumes(context.Background(), nil, PurgeVolumesOptions{
		VolumeIDs: []string{" vol-1 ", "", "vol-2"},
		DryRun:    true,
	})
	if err != nil {
		t.Fatal(err)
	}

	if !result.DryRun || result.Provider != ProviderName {
		t.Fatalf("unexpected result header: %#v", result)
	}
	if len(result.Request.VolumeIDs) != 2 || result.Request.VolumeIDs[0] != "vol-1" || result.Request.VolumeIDs[1] != "vol-2" {
		t.Fatalf("unexpected request: %#v", result.Request)
	}
}

func TestPurgeVolumesRequiresVolumeInTrash(t *testing.T) {
	client := &fakeVolumeClient{
		deletedVolumes: []Volume{{ID: "vol-trash", Status: "deleted"}},
	}

	_, err := PurgeVolumes(context.Background(), client, PurgeVolumesOptions{VolumeIDs: []string{"vol-active"}})
	if err == nil {
		t.Fatal("expected purge refusal")
	}
	if !strings.Contains(err.Error(), "not in trash") {
		t.Fatalf("error did not explain trash requirement: %v", err)
	}
	if len(client.deleteCalls) != 0 {
		t.Fatalf("DeleteVolume was called: %#v", client.deleteCalls)
	}
}

func TestPurgeVolumesSkipsPermanentlyDeletedTrashEntries(t *testing.T) {
	client := &fakeVolumeClient{
		deletedVolumes: []Volume{{ID: "vol-gone", Status: "deleted", IsPermanentlyDeleted: true}},
	}

	_, err := PurgeVolumes(context.Background(), client, PurgeVolumesOptions{VolumeIDs: []string{"vol-gone"}})
	if err == nil {
		t.Fatal("expected purge refusal")
	}
	if !strings.Contains(err.Error(), "not in trash") {
		t.Fatalf("error did not explain trash requirement: %v", err)
	}
}

func TestPurgeVolumesCallsPermanentDelete(t *testing.T) {
	client := &fakeVolumeClient{
		deletedVolumes: []Volume{
			{ID: "vol-trash-1", Status: "deleted"},
			{ID: "vol-trash-2", Status: "deleted"},
		},
	}

	result, err := PurgeVolumes(context.Background(), client, PurgeVolumesOptions{
		VolumeIDs: []string{"vol-trash-1", "vol-trash-2"},
	})
	if err != nil {
		t.Fatal(err)
	}

	if len(client.deleteCalls) != 2 {
		t.Fatalf("unexpected delete calls: %#v", client.deleteCalls)
	}
	for _, call := range client.deleteCalls {
		if !call.force {
			t.Fatalf("delete call did not use force=true: %#v", call)
		}
	}
	if len(result.Results) != 2 || result.Results[0].Status != "purged" || result.Results[1].VolumeID != "vol-trash-2" {
		t.Fatalf("unexpected purge result: %#v", result.Results)
	}
}

func TestPurgeVolumesReturnsDeleteError(t *testing.T) {
	client := &fakeVolumeClient{
		deletedVolumes: []Volume{{ID: "vol-trash", Status: "deleted"}},
		deleteErr:      errors.New("api exploded"),
	}

	_, err := PurgeVolumes(context.Background(), client, PurgeVolumesOptions{VolumeIDs: []string{"vol-trash"}})
	if err == nil {
		t.Fatal("expected delete error")
	}
	if !strings.Contains(err.Error(), "api exploded") {
		t.Fatalf("error did not include delete error: %v", err)
	}
}

func TestPurgeVolumesRequiresVolumeID(t *testing.T) {
	_, err := PurgeVolumes(context.Background(), nil, PurgeVolumesOptions{})
	if err == nil {
		t.Fatal("expected missing volume ID error")
	}
	if !strings.Contains(err.Error(), "volume ID") {
		t.Fatalf("error did not mention volume ID: %v", err)
	}
}

func volumeIDSet(volumes []Volume) map[string]bool {
	ids := map[string]bool{}
	for _, volume := range volumes {
		ids[volume.ID] = true
	}
	return ids
}

type fakeVolumeClient struct {
	volumes           []Volume
	deletedVolumes    []Volume
	deleteCalls       []fakeVolumeDeleteCall
	deleteErr         error
	listDeletedCalled bool
}

type fakeVolumeDeleteCall struct {
	volumeID string
	force    bool
}

func (c *fakeVolumeClient) ListVolumes(context.Context) ([]Volume, error) {
	return c.volumes, nil
}

func (c *fakeVolumeClient) ListDeletedVolumes(context.Context) ([]Volume, error) {
	c.listDeletedCalled = true
	return c.deletedVolumes, nil
}

func (c *fakeVolumeClient) DeleteVolume(_ context.Context, volumeID string, force bool) error {
	c.deleteCalls = append(c.deleteCalls, fakeVolumeDeleteCall{
		volumeID: volumeID,
		force:    force,
	})
	return c.deleteErr
}
