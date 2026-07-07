package verdacloud

import (
	"context"
	"errors"
	"os"
	"strings"
	"testing"
	"time"
)

func TestDefaultUserDataScriptMatchesBootstrap(t *testing.T) {
	bootstrap, err := os.ReadFile("bootstrap.sh")
	if err != nil {
		t.Fatal(err)
	}
	if DefaultUserDataScript != string(bootstrap) {
		t.Fatal("DefaultUserDataScript must match bootstrap.sh")
	}
}

func TestNormalizeDeployOptionsDefaultsToSpotFinlandPolicyInputs(t *testing.T) {
	base := DefaultDeployOptions("1V100.6V")
	base.SourceOSVolumeID = "vol-golden"
	opts, err := base.normalized(time.Date(2026, 7, 6, 12, 34, 56, 0, time.UTC))
	if err != nil {
		t.Fatal(err)
	}

	if opts.SourceOSVolumeID != "vol-golden" {
		t.Fatalf("SourceOSVolumeID = %q, want vol-golden", opts.SourceOSVolumeID)
	}
	if opts.Hostname != "nnctl-bench-1v100-6v-20260706123456" {
		t.Fatalf("Hostname = %q", opts.Hostname)
	}
	if opts.StartupScriptName != opts.Hostname+"-userdata" {
		t.Fatalf("StartupScriptName = %q", opts.StartupScriptName)
	}
	if !strings.Contains(opts.UserDataScript, "cuda-toolkit-13-0") {
		t.Fatalf("UserDataScript did not contain expected bootstrap marker")
	}
}

func TestDeployDryRunBuildsSpotAutoPlacementRequestWithoutClient(t *testing.T) {
	opts := DefaultDeployOptions("1V100.6V")
	opts.SourceOSVolumeID = "vol-golden"
	opts.DryRun = true

	result, err := Deploy(context.Background(), nil, opts)
	if err != nil {
		t.Fatal(err)
	}

	if !result.DryRun {
		t.Fatal("DryRun = false")
	}
	if result.Policy.LocationCode != "" || result.Policy.LocationSelection != AutoLocationSelection || !result.Policy.SpotOnly || !result.Policy.SingleGPU {
		t.Fatalf("unexpected policy: %#v", result.Policy)
	}
	if !result.Request.IsSpot || result.Request.Contract != SpotContract || result.Request.LocationCode != "" {
		t.Fatalf("request did not enforce spot auto-placement policy: %#v", result.Request)
	}
	if result.SourceOSVolumeID != "vol-golden" {
		t.Fatalf("SourceOSVolumeID = %q", result.SourceOSVolumeID)
	}
	if result.OSVolumeClone == nil || result.OSVolumeClone.SourceVolumeID != "vol-golden" {
		t.Fatalf("unexpected OSVolumeClone: %#v", result.OSVolumeClone)
	}
}

func TestDeployRejectsMultiGPUInstanceTypes(t *testing.T) {
	client := &fakeClient{
		instanceType: InstanceType{InstanceType: "8H100.80S", GPUCount: 8},
	}

	opts := DefaultDeployOptions("8H100.80S")
	opts.SourceOSVolumeID = "vol-golden"
	_, err := Deploy(context.Background(), client, opts)
	if err == nil {
		t.Fatal("expected multi-GPU instance type error")
	}
	if !strings.Contains(err.Error(), "single-GPU") {
		t.Fatalf("error did not mention single-GPU policy: %v", err)
	}
	if client.createdScript {
		t.Fatal("startup script should not be created for rejected instance type")
	}
}

func TestDeployCreatesCheapestAvailableSpotInstance(t *testing.T) {
	client := &fakeClient{
		instanceType: InstanceType{InstanceType: "1V100.6V", GPUCount: 1},
		placements: []SpotPlacement{
			{LocationCode: FinlandLocationCode, SpotPrice: 2, PriceKnown: true, Currency: "eur"},
			{LocationCode: "NOR-01", SpotPrice: 1, PriceKnown: true, Currency: "eur"},
		},
		script:       StartupScript{ID: "script-1", Name: "userdata"},
		clonedVolume: Volume{ID: "vol-clone-1", Name: "worker-os", Status: "cloning", Location: "NOR-01"},
		readyVolume:  Volume{ID: "vol-clone-1", Name: "worker-os", Status: "detached", Location: "NOR-01"},
		instance:     Instance{ID: "inst-1", Hostname: "worker", Status: "new", InstanceType: "1V100.6V", Location: "NOR-01", IsSpot: true},
	}
	opts := DefaultDeployOptions("1V100.6V")
	opts.SourceOSVolumeID = "vol-golden"
	opts.Hostname = "worker"
	opts.SSHKeyIDs = []string{
		"",
		"11111111-1111-1111-1111-111111111111",
		" 22222222-2222-2222-2222-222222222222 ",
	}

	result, err := Deploy(context.Background(), client, opts)
	if err != nil {
		t.Fatal(err)
	}

	if result.Instance == nil || result.Instance.ID != "inst-1" {
		t.Fatalf("unexpected result instance: %#v", result.Instance)
	}
	if !client.placementsListed {
		t.Fatal("placement options were not listed")
	}
	if result.Policy.LocationCode != "NOR-01" {
		t.Fatalf("LocationCode = %q, want NOR-01", result.Policy.LocationCode)
	}
	if result.Placement == nil || result.Placement.SpotPrice != 1 {
		t.Fatalf("unexpected placement: %#v", result.Placement)
	}
	if client.createdScriptName != "worker-userdata" {
		t.Fatalf("createdScriptName = %q", client.createdScriptName)
	}
	req := client.createRequest
	if req.LocationCode != "NOR-01" || !req.IsSpot || req.Contract != SpotContract {
		t.Fatalf("request did not enforce spot placement policy: %#v", req)
	}
	if req.StartupScriptID != "script-1" {
		t.Fatalf("StartupScriptID = %q", req.StartupScriptID)
	}
	if req.Image != "vol-clone-1" {
		t.Fatalf("Image = %q, want cloned OS volume ID", req.Image)
	}
	if len(req.SSHKeyIDs) != 2 || req.SSHKeyIDs[0] != "11111111-1111-1111-1111-111111111111" || req.SSHKeyIDs[1] != "22222222-2222-2222-2222-222222222222" {
		t.Fatalf("SSHKeyIDs were not compacted: %#v", req.SSHKeyIDs)
	}
	if len(client.cloneRequests) != 1 || client.cloneRequests[0].SourceVolumeID != "vol-golden" || client.cloneRequests[0].Name != "worker-os" || client.cloneRequests[0].LocationCode != "NOR-01" {
		t.Fatalf("unexpected clone requests: %#v", client.cloneRequests)
	}
	if client.waitedVolumeID != "vol-clone-1" {
		t.Fatalf("waitedVolumeID = %q", client.waitedVolumeID)
	}
	if result.OSVolumeClone == nil || result.OSVolumeClone.VolumeID != "vol-clone-1" || result.OSVolumeClone.Status != "detached" {
		t.Fatalf("unexpected clone result: %#v", result.OSVolumeClone)
	}
	if result.ClonedOSVolumeIDs["inst-1"] != "vol-clone-1" {
		t.Fatalf("unexpected clone tracking: %#v", result.ClonedOSVolumeIDs)
	}
}

func TestDeployCreatesExplicitLocationSpotInstance(t *testing.T) {
	client := &fakeClient{
		instanceType: InstanceType{InstanceType: "1V100.6V", GPUCount: 1},
		placements: []SpotPlacement{
			{LocationCode: FinlandLocationCode, SpotPrice: 2, PriceKnown: true, Currency: "eur"},
			{LocationCode: "NOR-01", SpotPrice: 1, PriceKnown: true, Currency: "eur"},
		},
		script:       StartupScript{ID: "script-1", Name: "userdata"},
		clonedVolume: Volume{ID: "vol-clone-1", Name: "worker-os", Status: "cloning", Location: FinlandLocationCode},
		readyVolume:  Volume{ID: "vol-clone-1", Name: "worker-os", Status: "detached", Location: FinlandLocationCode},
		instance:     Instance{ID: "inst-1", Hostname: "worker", Status: "new", InstanceType: "1V100.6V", Location: FinlandLocationCode, IsSpot: true},
	}
	opts := DefaultDeployOptions("1V100.6V")
	opts.SourceOSVolumeID = "vol-golden"
	opts.Hostname = "worker"
	opts.LocationCode = "fin-03"

	result, err := Deploy(context.Background(), client, opts)
	if err != nil {
		t.Fatal(err)
	}

	if result.Policy.LocationCode != FinlandLocationCode || result.Policy.LocationSelection != ExplicitLocation {
		t.Fatalf("unexpected policy: %#v", result.Policy)
	}
	if result.Placement == nil || result.Placement.LocationCode != FinlandLocationCode {
		t.Fatalf("unexpected placement: %#v", result.Placement)
	}
	if client.createRequest.LocationCode != FinlandLocationCode {
		t.Fatalf("LocationCode = %q, want %s", client.createRequest.LocationCode, FinlandLocationCode)
	}
	if client.createRequest.Image != "vol-clone-1" {
		t.Fatalf("Image = %q, want cloned OS volume ID", client.createRequest.Image)
	}
}

func TestDeployRejectsAutoPlacementWithoutAvailabilityCheck(t *testing.T) {
	client := &fakeClient{instanceType: InstanceType{InstanceType: "1V100.6V", GPUCount: 1}}
	opts := DefaultDeployOptions("1V100.6V")
	opts.SourceOSVolumeID = "vol-golden"
	opts.SkipAvailabilityCheck = true

	_, err := Deploy(context.Background(), client, opts)
	if err == nil {
		t.Fatal("expected auto-placement skip availability error")
	}
	if !strings.Contains(err.Error(), "--location-code") {
		t.Fatalf("error did not mention --location-code: %v", err)
	}
}

func TestDeployRejectsNonUUIDSSHKeyID(t *testing.T) {
	opts := DefaultDeployOptions("1V100.6V")
	opts.SourceOSVolumeID = "vol-golden"
	opts.DryRun = true
	opts.SSHKeyIDs = []string{"ville+mba@vesilehto.fi"}

	_, err := Deploy(context.Background(), nil, opts)
	if err == nil {
		t.Fatal("expected non-UUID SSH key id error")
	}
	if !strings.Contains(err.Error(), "nnctl cloud ssh-keys") {
		t.Fatalf("error did not point to ssh key listing command: %v", err)
	}
}

func TestDeployRequiresSourceOSVolumeID(t *testing.T) {
	opts := DefaultDeployOptions("1V100.6V")
	opts.DryRun = true

	_, err := Deploy(context.Background(), nil, opts)
	if err == nil {
		t.Fatal("expected missing source_os_volume_id error")
	}
	if !strings.Contains(err.Error(), "source_os_volume_id") {
		t.Fatalf("error did not mention source_os_volume_id: %v", err)
	}
}

func TestDeployDeletesOnlyClonedVolumeWhenCreateInstanceFails(t *testing.T) {
	client := &fakeClient{
		instanceType: InstanceType{InstanceType: "1V100.6V", GPUCount: 1},
		placements:   []SpotPlacement{{LocationCode: FinlandLocationCode, SpotPrice: 2, PriceKnown: true}},
		script:       StartupScript{ID: "script-1", Name: "userdata"},
		clonedVolume: Volume{ID: "vol-clone-1", Name: "worker-os", Status: "cloning", Location: FinlandLocationCode},
		readyVolume:  Volume{ID: "vol-clone-1", Name: "worker-os", Status: "detached", Location: FinlandLocationCode},
		createErr:    errors.New("capacity disappeared"),
	}
	opts := DefaultDeployOptions("1V100.6V")
	opts.SourceOSVolumeID = "vol-golden"
	opts.Hostname = "worker"
	opts.LocationCode = FinlandLocationCode

	_, err := Deploy(context.Background(), client, opts)
	if err == nil {
		t.Fatal("expected create instance error")
	}
	if len(client.deletedVolumeIDs) != 1 || client.deletedVolumeIDs[0] != "vol-clone-1" {
		t.Fatalf("unexpected deleted volumes: %#v", client.deletedVolumeIDs)
	}
	if strings.Contains(strings.Join(client.deletedVolumeIDs, ","), "vol-golden") {
		t.Fatalf("source volume was deleted: %#v", client.deletedVolumeIDs)
	}
}

type fakeClient struct {
	instanceType      InstanceType
	placements        []SpotPlacement
	script            StartupScript
	clonedVolume      Volume
	readyVolume       Volume
	instance          Instance
	placementsListed  bool
	createdScript     bool
	createdScriptName string
	cloneRequests     []CloneVolumeRequest
	waitedVolumeID    string
	deletedVolumeIDs  []string
	createRequest     CreateInstanceRequest
	createErr         error
}

func (c *fakeClient) GetInstanceType(context.Context, string, bool, string) (InstanceType, error) {
	return c.instanceType, nil
}

func (c *fakeClient) GetSpotPlacementOptions(context.Context, string) ([]SpotPlacement, error) {
	c.placementsListed = true
	return append([]SpotPlacement(nil), c.placements...), nil
}

func (c *fakeClient) CreateStartupScript(_ context.Context, name, _ string) (StartupScript, error) {
	c.createdScript = true
	c.createdScriptName = name
	return c.script, nil
}

func (c *fakeClient) CloneVolume(_ context.Context, req CloneVolumeRequest) (Volume, error) {
	c.cloneRequests = append(c.cloneRequests, req)
	if c.clonedVolume.ID == "" {
		c.clonedVolume = Volume{ID: "vol-clone", Name: req.Name, Status: "cloning", Location: req.LocationCode}
	}
	return c.clonedVolume, nil
}

func (c *fakeClient) WaitVolumeReady(_ context.Context, volumeID string) (Volume, error) {
	c.waitedVolumeID = volumeID
	if c.readyVolume.ID == "" {
		c.readyVolume = Volume{ID: volumeID, Status: "detached"}
	}
	return c.readyVolume, nil
}

func (c *fakeClient) DeleteVolume(_ context.Context, volumeID string, _ bool) error {
	c.deletedVolumeIDs = append(c.deletedVolumeIDs, volumeID)
	return nil
}

func (c *fakeClient) CreateInstance(_ context.Context, req CreateInstanceRequest) (Instance, error) {
	c.createRequest = req
	if c.createErr != nil {
		return Instance{}, c.createErr
	}
	return c.instance, nil
}
