package verdacloud

import (
	"context"
	"strings"
	"testing"
	"time"
)

func TestNormalizeDeployOptionsDefaultsToSpotFinlandPolicyInputs(t *testing.T) {
	opts, err := DefaultDeployOptions("1V100.6V").normalized(time.Date(2026, 7, 6, 12, 34, 56, 0, time.UTC))
	if err != nil {
		t.Fatal(err)
	}

	if opts.Image != DefaultImage {
		t.Fatalf("Image = %q, want %q", opts.Image, DefaultImage)
	}
	if opts.Hostname != "nnctl-bench-1v100-6v-20260706123456" {
		t.Fatalf("Hostname = %q", opts.Hostname)
	}
	if opts.StartupScriptName != opts.Hostname+"-userdata" {
		t.Fatalf("StartupScriptName = %q", opts.StartupScriptName)
	}
	if !strings.Contains(opts.UserDataScript, "nnctl") {
		t.Fatalf("UserDataScript did not contain expected bootstrap marker")
	}
}

func TestDeployDryRunBuildsSpotAutoPlacementRequestWithoutClient(t *testing.T) {
	opts := DefaultDeployOptions("1V100.6V")
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
}

func TestDeployRejectsMultiGPUInstanceTypes(t *testing.T) {
	client := &fakeClient{
		instanceType: InstanceType{InstanceType: "8H100.80S", GPUCount: 8},
	}

	_, err := Deploy(context.Background(), client, DefaultDeployOptions("8H100.80S"))
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
		script:   StartupScript{ID: "script-1", Name: "userdata"},
		instance: Instance{ID: "inst-1", Hostname: "worker", Status: "new", InstanceType: "1V100.6V", Location: "NOR-01", IsSpot: true},
	}
	opts := DefaultDeployOptions("1V100.6V")
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
	if len(req.SSHKeyIDs) != 2 || req.SSHKeyIDs[0] != "11111111-1111-1111-1111-111111111111" || req.SSHKeyIDs[1] != "22222222-2222-2222-2222-222222222222" {
		t.Fatalf("SSHKeyIDs were not compacted: %#v", req.SSHKeyIDs)
	}
}

func TestDeployCreatesExplicitLocationSpotInstance(t *testing.T) {
	client := &fakeClient{
		instanceType: InstanceType{InstanceType: "1V100.6V", GPUCount: 1},
		placements: []SpotPlacement{
			{LocationCode: FinlandLocationCode, SpotPrice: 2, PriceKnown: true, Currency: "eur"},
			{LocationCode: "NOR-01", SpotPrice: 1, PriceKnown: true, Currency: "eur"},
		},
		script:   StartupScript{ID: "script-1", Name: "userdata"},
		instance: Instance{ID: "inst-1", Hostname: "worker", Status: "new", InstanceType: "1V100.6V", Location: FinlandLocationCode, IsSpot: true},
	}
	opts := DefaultDeployOptions("1V100.6V")
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
}

func TestDeployRejectsAutoPlacementWithoutAvailabilityCheck(t *testing.T) {
	client := &fakeClient{instanceType: InstanceType{InstanceType: "1V100.6V", GPUCount: 1}}
	opts := DefaultDeployOptions("1V100.6V")
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

type fakeClient struct {
	instanceType      InstanceType
	placements        []SpotPlacement
	script            StartupScript
	instance          Instance
	placementsListed  bool
	createdScript     bool
	createdScriptName string
	createRequest     CreateInstanceRequest
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

func (c *fakeClient) CreateInstance(_ context.Context, req CreateInstanceRequest) (Instance, error) {
	c.createRequest = req
	return c.instance, nil
}
