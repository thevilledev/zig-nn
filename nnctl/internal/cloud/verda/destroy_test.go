package verda

import (
	"context"
	"strings"
	"testing"
)

func TestNormalizeDestroyOptionsDefaultsToPermanentDelete(t *testing.T) {
	opts, err := DefaultDestroyOptions([]string{" inst-1 ", "", "inst-2"}).normalized()
	if err != nil {
		t.Fatal(err)
	}

	if len(opts.InstanceIDs) != 2 || opts.InstanceIDs[0] != "inst-1" || opts.InstanceIDs[1] != "inst-2" {
		t.Fatalf("InstanceIDs were not compacted: %#v", opts.InstanceIDs)
	}
	if !opts.DeletePermanently {
		t.Fatal("DeletePermanently = false, want true")
	}
}

func TestDestroyDryRunBuildsRequestWithoutClient(t *testing.T) {
	opts := DefaultDestroyOptions([]string{"inst-1"})
	opts.DryRun = true

	result, err := Destroy(context.Background(), nil, opts)
	if err != nil {
		t.Fatal(err)
	}

	if !result.DryRun {
		t.Fatal("DryRun = false")
	}
	if len(result.Request.InstanceIDs) != 1 || result.Request.InstanceIDs[0] != "inst-1" {
		t.Fatalf("unexpected request: %#v", result.Request)
	}
	if !result.Request.DeletePermanently {
		t.Fatal("DeletePermanently = false, want true")
	}
}

func TestDestroyCallsClient(t *testing.T) {
	client := &fakeDestroyClient{
		results: []DestroyInstanceResult{{InstanceID: "inst-1", Status: "success"}},
	}
	opts := DefaultDestroyOptions([]string{"inst-1"})
	opts.VolumeIDs = []string{"vol-1"}
	opts.DeletePermanently = false

	result, err := Destroy(context.Background(), client, opts)
	if err != nil {
		t.Fatal(err)
	}

	if len(result.Results) != 1 || result.Results[0].InstanceID != "inst-1" {
		t.Fatalf("unexpected results: %#v", result.Results)
	}
	if len(client.request.InstanceIDs) != 1 || client.request.InstanceIDs[0] != "inst-1" {
		t.Fatalf("unexpected instance IDs: %#v", client.request.InstanceIDs)
	}
	if len(client.request.VolumeIDs) != 1 || client.request.VolumeIDs[0] != "vol-1" {
		t.Fatalf("unexpected volume IDs: %#v", client.request.VolumeIDs)
	}
	if client.request.DeletePermanently {
		t.Fatal("DeletePermanently = true, want false")
	}
}

func TestDestroyRejectsSourceOSVolumeIDAsCleanupVolume(t *testing.T) {
	opts := DefaultDestroyOptions([]string{"inst-1"})
	opts.SourceOSVolumeID = "vol-golden"
	opts.VolumeIDs = []string{"vol-clone-1", "vol-golden"}

	_, err := Destroy(context.Background(), nil, opts)
	if err == nil {
		t.Fatal("expected source volume cleanup error")
	}
	if !strings.Contains(err.Error(), "source_os_volume_id") {
		t.Fatalf("error did not mention source_os_volume_id: %v", err)
	}
}

func TestDestroyRequiresInstanceID(t *testing.T) {
	_, err := Destroy(context.Background(), nil, DefaultDestroyOptions(nil))
	if err == nil {
		t.Fatal("expected missing instance ID error")
	}
	if !strings.Contains(err.Error(), "instance ID") {
		t.Fatalf("error did not mention instance ID: %v", err)
	}
}

func TestDestroyReturnsFailedActionResult(t *testing.T) {
	client := &fakeDestroyClient{
		results: []DestroyInstanceResult{{InstanceID: "inst-1", Status: "error", Error: "no such instance"}},
	}

	_, err := Destroy(context.Background(), client, DefaultDestroyOptions([]string{"inst-1"}))
	if err == nil {
		t.Fatal("expected failed action result error")
	}
	if !strings.Contains(err.Error(), "no such instance") {
		t.Fatalf("error did not include action error: %v", err)
	}
}

type fakeDestroyClient struct {
	request DestroyInstanceRequest
	results []DestroyInstanceResult
}

func (c *fakeDestroyClient) DestroyInstances(_ context.Context, req DestroyInstanceRequest) ([]DestroyInstanceResult, error) {
	c.request = req
	return c.results, nil
}
