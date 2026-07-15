package verda

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

func TestPurgeAllDeletedVolumesDryRunBuildsRequestWithoutClient(t *testing.T) {
	result, err := PurgeVolumes(context.Background(), nil, PurgeVolumesOptions{
		AllDeleted: true,
		DryRun:     true,
	})
	if err != nil {
		t.Fatal(err)
	}

	if !result.DryRun || !result.Request.AllDeleted {
		t.Fatalf("unexpected result: %#v", result)
	}
	if len(result.Request.VolumeIDs) != 0 {
		t.Fatalf("dry-run purge-all should not resolve IDs: %#v", result.Request.VolumeIDs)
	}
}

func TestPurgeVolumesCallsPermanentDelete(t *testing.T) {
	client := &fakeVolumeClient{}

	result, err := PurgeVolumes(context.Background(), client, PurgeVolumesOptions{
		VolumeIDs: []string{"vol-active", "vol-trash"},
	})
	if err != nil {
		t.Fatal(err)
	}

	if len(client.permanentDeleteCalls) != 2 {
		t.Fatalf("unexpected delete calls: %#v", client.permanentDeleteCalls)
	}
	if client.permanentDeleteCalls[0] != "vol-active" || client.permanentDeleteCalls[1] != "vol-trash" {
		t.Fatalf("unexpected delete calls: %#v", client.permanentDeleteCalls)
	}
	if client.listDeletedCalled {
		t.Fatal("ListDeletedVolumes should not be required before purge")
	}
	if len(result.Results) != 2 || result.Results[0].Status != "purged" || result.Results[1].VolumeID != "vol-trash" {
		t.Fatalf("unexpected purge result: %#v", result.Results)
	}
}

func TestPurgeAllDeletedVolumesCallsPermanentDelete(t *testing.T) {
	client := &fakeVolumeClient{
		deletedVolumes: []Volume{
			{ID: "vol-trash-b", Status: "deleted"},
			{ID: "vol-gone", Status: "deleted", IsPermanentlyDeleted: true},
			{ID: "", Status: "deleted"},
			{ID: "vol-trash-a", Status: "deleted"},
		},
	}

	result, err := PurgeVolumes(context.Background(), client, PurgeVolumesOptions{
		AllDeleted: true,
	})
	if err != nil {
		t.Fatal(err)
	}

	if !client.listDeletedCalled {
		t.Fatal("ListDeletedVolumes was not called")
	}
	if len(client.permanentDeleteCalls) != 2 {
		t.Fatalf("unexpected delete calls: %#v", client.permanentDeleteCalls)
	}
	if client.permanentDeleteCalls[0] != "vol-trash-a" || client.permanentDeleteCalls[1] != "vol-trash-b" {
		t.Fatalf("deleted IDs were not sorted/filtered: %#v", client.permanentDeleteCalls)
	}
	if !result.Request.AllDeleted || len(result.Request.VolumeIDs) != 2 || result.Request.VolumeIDs[0] != "vol-trash-a" || result.Request.VolumeIDs[1] != "vol-trash-b" {
		t.Fatalf("unexpected request: %#v", result.Request)
	}
	if len(result.Results) != 2 || result.Results[1].VolumeID != "vol-trash-b" {
		t.Fatalf("unexpected purge result: %#v", result.Results)
	}
}

func TestPurgeAllDeletedVolumesWithEmptyTrash(t *testing.T) {
	client := &fakeVolumeClient{
		deletedVolumes: []Volume{{ID: "vol-gone", IsPermanentlyDeleted: true}},
	}

	result, err := PurgeVolumes(context.Background(), client, PurgeVolumesOptions{
		AllDeleted: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(client.permanentDeleteCalls) != 0 {
		t.Fatalf("unexpected delete calls: %#v", client.permanentDeleteCalls)
	}
	if len(result.Results) != 0 || len(result.Request.VolumeIDs) != 0 {
		t.Fatalf("unexpected result: %#v", result)
	}
}

func TestPurgeVolumesReturnsDeleteError(t *testing.T) {
	client := &fakeVolumeClient{
		deleteErr: errors.New("api exploded"),
	}

	_, err := PurgeVolumes(context.Background(), client, PurgeVolumesOptions{VolumeIDs: []string{"vol-active"}})
	if err == nil {
		t.Fatal("expected delete error")
	}
	if !strings.Contains(err.Error(), "api exploded") {
		t.Fatalf("error did not include delete error: %v", err)
	}
}

func TestPurgeAllDeletedVolumesReturnsListError(t *testing.T) {
	client := &fakeVolumeClient{
		listDeletedErr: errors.New("api exploded"),
	}

	_, err := PurgeVolumes(context.Background(), client, PurgeVolumesOptions{AllDeleted: true})
	if err == nil {
		t.Fatal("expected list error")
	}
	if !strings.Contains(err.Error(), "api exploded") {
		t.Fatalf("error did not include list error: %v", err)
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

func TestPurgeVolumesRejectsIDsWithAllDeleted(t *testing.T) {
	_, err := PurgeVolumes(context.Background(), nil, PurgeVolumesOptions{
		VolumeIDs:  []string{"vol-1"},
		AllDeleted: true,
	})
	if err == nil {
		t.Fatal("expected all-deleted with volume IDs error")
	}
	if !strings.Contains(err.Error(), "purge all") {
		t.Fatalf("error did not mention purge all: %v", err)
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
	volumes              []Volume
	deletedVolumes       []Volume
	permanentDeleteCalls []string
	deleteErr            error
	listDeletedErr       error
	listDeletedCalled    bool
}

func (c *fakeVolumeClient) ListVolumes(context.Context) ([]Volume, error) {
	return c.volumes, nil
}

func (c *fakeVolumeClient) ListDeletedVolumes(context.Context) ([]Volume, error) {
	c.listDeletedCalled = true
	if c.listDeletedErr != nil {
		return nil, c.listDeletedErr
	}
	return c.deletedVolumes, nil
}

func (c *fakeVolumeClient) PermanentDeleteVolume(_ context.Context, volumeID string) error {
	c.permanentDeleteCalls = append(c.permanentDeleteCalls, volumeID)
	return c.deleteErr
}
