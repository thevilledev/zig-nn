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
	bootstrap, err := os.ReadFile("packer/bootstrap.sh")
	if err != nil {
		t.Fatal(err)
	}
	if DefaultUserDataScript != string(bootstrap) {
		t.Fatal("DefaultUserDataScript must match packer/bootstrap.sh")
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
	if opts.Market != PricingMarketSpot {
		t.Fatalf("Market = %q, want spot", opts.Market)
	}
	if opts.Hostname != "nnctl-bench-1v100-6v-20260706123456" {
		t.Fatalf("Hostname = %q", opts.Hostname)
	}
	if opts.StartupScriptName != "" {
		t.Fatalf("StartupScriptName = %q, want empty", opts.StartupScriptName)
	}
	if opts.UserDataScript != "" {
		t.Fatalf("UserDataScript should be empty when source volume is set")
	}
}

func TestNormalizeDeployOptionsAcceptsOnDemandMarketAlias(t *testing.T) {
	base := DefaultDeployOptions("1H200.141S.44V")
	base.Market = "on_demand"
	opts, err := base.normalized(time.Date(2026, 7, 6, 12, 34, 56, 0, time.UTC))
	if err != nil {
		t.Fatal(err)
	}

	if opts.Market != PricingMarketOnDemand {
		t.Fatalf("Market = %q, want on-demand", opts.Market)
	}
}

func TestNormalizeDeployOptionsRejectsInvalidMarket(t *testing.T) {
	base := DefaultDeployOptions("1V100.6V")
	base.Market = "reserved"
	_, err := base.normalized(time.Date(2026, 7, 6, 12, 34, 56, 0, time.UTC))
	if err == nil {
		t.Fatal("expected invalid market error")
	}
	if !strings.Contains(err.Error(), "spot or on-demand") {
		t.Fatalf("error did not mention deploy markets: %v", err)
	}
}

func TestNormalizeDeployOptionsDefaultsUserDataWithoutSourceOSVolume(t *testing.T) {
	base := DefaultDeployOptions("1V100.6V")
	opts, err := base.normalized(time.Date(2026, 7, 6, 12, 34, 56, 0, time.UTC))
	if err != nil {
		t.Fatal(err)
	}

	if opts.SourceOSVolumeID != "" {
		t.Fatalf("SourceOSVolumeID = %q, want empty", opts.SourceOSVolumeID)
	}
	if opts.Image != DefaultDeployImage {
		t.Fatalf("Image = %q, want %s", opts.Image, DefaultDeployImage)
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
	if result.Policy.LocationCode != "" || result.Policy.LocationSelection != SourceOSVolumeLocation || !result.Policy.SourceOSVolumeLocked || !result.Policy.SpotOnly || !result.Policy.AllowsCPU || result.Policy.MaxGPUCount != MaxDeployGPUCount || !result.Policy.CleanupClonedOSVolumeOnFailure {
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
	if result.StartupScript != nil {
		t.Fatalf("unexpected default startup script: %#v", result.StartupScript)
	}
}

func TestDeployDryRunBuildsOnDemandRequestWithoutClient(t *testing.T) {
	opts := DefaultDeployOptions("1H200.141S.44V")
	opts.Market = PricingMarketOnDemand
	opts.DryRun = true

	result, err := Deploy(context.Background(), nil, opts)
	if err != nil {
		t.Fatal(err)
	}

	if result.Policy.Market != PricingMarketOnDemand || result.Policy.SpotOnly {
		t.Fatalf("unexpected policy: %#v", result.Policy)
	}
	if result.Policy.LocationSelection != MarketPlacementLocation {
		t.Fatalf("LocationSelection = %q, want %s", result.Policy.LocationSelection, MarketPlacementLocation)
	}
	if result.Request.IsSpot || result.Request.Contract != OnDemandContract {
		t.Fatalf("request did not use on-demand contract: %#v", result.Request)
	}
	if result.Request.Image != DefaultDeployImage {
		t.Fatalf("Image = %q, want %s", result.Request.Image, DefaultDeployImage)
	}
}

func TestDeployDryRunWithoutSourceOSVolumeUsesDefaultImageAndUserData(t *testing.T) {
	opts := DefaultDeployOptions("1V100.6V")
	opts.DryRun = true

	result, err := Deploy(context.Background(), nil, opts)
	if err != nil {
		t.Fatal(err)
	}

	if result.Policy.LocationSelection != SpotPlacementLocation || result.Policy.SourceOSVolumeLocked {
		t.Fatalf("unexpected policy: %#v", result.Policy)
	}
	if result.Request.Image != DefaultDeployImage || result.Request.LocationCode != "" {
		t.Fatalf("unexpected request: %#v", result.Request)
	}
	if result.SourceOSVolumeID != "" || result.OSVolumeClone != nil {
		t.Fatalf("unexpected source volume plan: source=%q clone=%#v", result.SourceOSVolumeID, result.OSVolumeClone)
	}
	if result.StartupScript == nil || result.StartupScript.Name == "" {
		t.Fatalf("expected planned startup script: %#v", result.StartupScript)
	}
}

func TestDeployRejectsMultiGPUInstanceTypes(t *testing.T) {
	client := &fakeClient{
		instanceType: InstanceType{InstanceType: "8H100.80S", GPUCount: 8},
		sourceVolume: Volume{ID: "vol-golden", Location: FinlandLocationCode, IsOSVolume: true},
	}

	opts := DefaultDeployOptions("8H100.80S")
	opts.SourceOSVolumeID = "vol-golden"
	_, err := Deploy(context.Background(), client, opts)
	if err == nil {
		t.Fatal("expected multi-GPU instance type error")
	}
	if !strings.Contains(err.Error(), "CPU-only and single-GPU") {
		t.Fatalf("error did not mention CPU/single-GPU policy: %v", err)
	}
	if client.createdScript {
		t.Fatal("startup script should not be created for rejected instance type")
	}
}

func TestDeployCreatesSpotCPUInstanceInSourceOSVolumeLocation(t *testing.T) {
	client := &fakeClient{
		sourceVolume: Volume{ID: "vol-golden", Name: "golden", Status: "detached", Location: FinlandLocationCode, IsOSVolume: true},
		instanceType: InstanceType{InstanceType: "4C.16M", GPUCount: 0},
		placements: []SpotPlacement{
			{LocationCode: FinlandLocationCode, SpotPrice: 1, PriceKnown: true, Currency: "eur"},
		},
		clonedVolume: Volume{ID: "vol-clone-1", Name: "worker-os", Status: "cloning", Location: FinlandLocationCode},
		readyVolume:  Volume{ID: "vol-clone-1", Name: "worker-os", Status: "detached", Location: FinlandLocationCode},
		instance:     Instance{ID: "inst-1", Hostname: "worker", Status: "new", InstanceType: "4C.16M", Location: FinlandLocationCode, IsSpot: true},
	}
	opts := DefaultDeployOptions("4C.16M")
	opts.SourceOSVolumeID = "vol-golden"
	opts.Hostname = "worker"

	result, err := Deploy(context.Background(), client, opts)
	if err != nil {
		t.Fatal(err)
	}

	if result.InstanceType == nil || result.InstanceType.GPUCount != 0 {
		t.Fatalf("unexpected instance type result: %#v", result.InstanceType)
	}
	if result.Policy.MaxGPUCount != MaxDeployGPUCount || !result.Policy.AllowsCPU {
		t.Fatalf("unexpected policy: %#v", result.Policy)
	}
	if client.createRequest.InstanceType != "4C.16M" || client.createRequest.LocationCode != FinlandLocationCode {
		t.Fatalf("unexpected create request: %#v", client.createRequest)
	}
	if client.createdScript || result.StartupScript != nil {
		t.Fatalf("unexpected startup script: client=%t result=%#v", client.createdScript, result.StartupScript)
	}
}

func TestDeployCreatesSpotInstanceInSourceOSVolumeLocation(t *testing.T) {
	client := &fakeClient{
		sourceVolume: Volume{ID: "vol-golden", Name: "golden", Status: "detached", Location: FinlandLocationCode, IsOSVolume: true},
		instanceType: InstanceType{InstanceType: "1V100.6V", GPUCount: 1},
		placements: []SpotPlacement{
			{LocationCode: FinlandLocationCode, SpotPrice: 2, PriceKnown: true, Currency: "eur"},
			{LocationCode: "NOR-01", SpotPrice: 1, PriceKnown: true, Currency: "eur"},
		},
		clonedVolume: Volume{ID: "vol-clone-1", Name: "worker-os", Status: "cloning", Location: FinlandLocationCode},
		readyVolume:  Volume{ID: "vol-clone-1", Name: "worker-os", Status: "detached", Location: FinlandLocationCode},
		instance:     Instance{ID: "inst-1", Hostname: "worker", Status: "new", InstanceType: "1V100.6V", Location: FinlandLocationCode, IsSpot: true},
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
	if client.gotVolumeID != "vol-golden" {
		t.Fatalf("gotVolumeID = %q", client.gotVolumeID)
	}
	if result.SourceOSVolume == nil || result.SourceOSVolume.Location != FinlandLocationCode {
		t.Fatalf("unexpected source volume: %#v", result.SourceOSVolume)
	}
	if result.Policy.LocationCode != FinlandLocationCode || result.Policy.LocationSelection != SourceOSVolumeLocation || !result.Policy.SourceOSVolumeLocked {
		t.Fatalf("unexpected location policy: %#v", result.Policy)
	}
	if result.Placement == nil || result.Placement.LocationCode != FinlandLocationCode || result.Placement.SpotPrice != 2 {
		t.Fatalf("unexpected placement: %#v", result.Placement)
	}
	req := client.createRequest
	if req.LocationCode != FinlandLocationCode || !req.IsSpot || req.Contract != SpotContract {
		t.Fatalf("request did not enforce spot placement policy: %#v", req)
	}
	if client.createdScript || req.StartupScriptID != "" || result.StartupScript != nil {
		t.Fatalf("unexpected startup script: client=%t req=%#v result=%#v", client.createdScript, req, result.StartupScript)
	}
	if req.Image != "vol-clone-1" {
		t.Fatalf("Image = %q, want cloned OS volume ID", req.Image)
	}
	if len(req.SSHKeyIDs) != 2 || req.SSHKeyIDs[0] != "11111111-1111-1111-1111-111111111111" || req.SSHKeyIDs[1] != "22222222-2222-2222-2222-222222222222" {
		t.Fatalf("SSHKeyIDs were not compacted: %#v", req.SSHKeyIDs)
	}
	if len(client.cloneRequests) != 1 || client.cloneRequests[0].SourceVolumeID != "vol-golden" || client.cloneRequests[0].Name != "worker-os" || client.cloneRequests[0].LocationCode != FinlandLocationCode {
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

func TestDeploySelectsSourceOSVolumeByNameInAvailableLocation(t *testing.T) {
	client := &fakeClient{
		sourceVolumes: []Volume{
			{ID: "vol-fin", Name: "golden", Status: "detached", Location: FinlandLocationCode, IsOSVolume: true},
			{ID: "vol-nor", Name: "golden", Status: "detached", Location: "NOR-01", IsOSVolume: true},
			{ID: "vol-data", Name: "golden", Status: "detached", Location: "NOR-01"},
		},
		instanceType: InstanceType{InstanceType: "1V100.6V", GPUCount: 1},
		placements: []SpotPlacement{
			{LocationCode: FinlandLocationCode, SpotPrice: 2, PriceKnown: true, Currency: "eur"},
			{LocationCode: "NOR-01", SpotPrice: 1, PriceKnown: true, Currency: "eur"},
		},
		clonedVolume: Volume{ID: "vol-clone-1", Name: "worker-os", Status: "cloning", Location: "NOR-01"},
		readyVolume:  Volume{ID: "vol-clone-1", Name: "worker-os", Status: "detached", Location: "NOR-01"},
		instance:     Instance{ID: "inst-1", Hostname: "worker", Status: "new", InstanceType: "1V100.6V", Location: "NOR-01", IsSpot: true},
	}
	opts := DefaultDeployOptions("1V100.6V")
	opts.SourceOSVolumeName = "golden"
	opts.Hostname = "worker"

	result, err := Deploy(context.Background(), client, opts)
	if err != nil {
		t.Fatal(err)
	}

	if !client.volumesListed {
		t.Fatal("volumes were not listed")
	}
	if client.gotVolumeID != "" {
		t.Fatalf("GetVolume should not be used for source name resolution, got %q", client.gotVolumeID)
	}
	if result.SourceOSVolumeID != "vol-nor" || result.SourceOSVolumeName != "golden" {
		t.Fatalf("unexpected resolved source volume: id=%q name=%q", result.SourceOSVolumeID, result.SourceOSVolumeName)
	}
	if result.SourceOSVolume == nil || result.SourceOSVolume.Location != "NOR-01" {
		t.Fatalf("unexpected source volume: %#v", result.SourceOSVolume)
	}
	if result.Policy.LocationCode != "NOR-01" || result.Policy.LocationSelection != SourceOSVolumeLocation || !result.Policy.SourceOSVolumeLocked {
		t.Fatalf("unexpected location policy: %#v", result.Policy)
	}
	if result.Placement == nil || result.Placement.LocationCode != "NOR-01" || result.Placement.SpotPrice != 1 {
		t.Fatalf("unexpected placement: %#v", result.Placement)
	}
	if len(client.cloneRequests) != 1 || client.cloneRequests[0].SourceVolumeID != "vol-nor" || client.cloneRequests[0].LocationCode != "NOR-01" {
		t.Fatalf("unexpected clone requests: %#v", client.cloneRequests)
	}
	if client.createRequest.LocationCode != "NOR-01" || client.createRequest.Image != "vol-clone-1" {
		t.Fatalf("unexpected create request: %#v", client.createRequest)
	}
}

func TestDeploySourceOSVolumeNameHonorsExplicitLocation(t *testing.T) {
	client := &fakeClient{
		sourceVolumes: []Volume{
			{ID: "vol-fin", Name: "golden", Status: "detached", Location: FinlandLocationCode, IsOSVolume: true},
			{ID: "vol-nor", Name: "golden", Status: "detached", Location: "NOR-01", IsOSVolume: true},
		},
		instanceType: InstanceType{InstanceType: "1V100.6V", GPUCount: 1},
		placements:   []SpotPlacement{{LocationCode: FinlandLocationCode, SpotPrice: 2, PriceKnown: true, Currency: "eur"}},
		clonedVolume: Volume{ID: "vol-clone-1", Name: "worker-os", Status: "cloning", Location: FinlandLocationCode},
		readyVolume:  Volume{ID: "vol-clone-1", Name: "worker-os", Status: "detached", Location: FinlandLocationCode},
		instance:     Instance{ID: "inst-1", Hostname: "worker", Status: "new", InstanceType: "1V100.6V", Location: FinlandLocationCode, IsSpot: true},
	}
	opts := DefaultDeployOptions("1V100.6V")
	opts.SourceOSVolumeName = "golden"
	opts.Hostname = "worker"
	opts.LocationCode = strings.ToLower(FinlandLocationCode)

	result, err := Deploy(context.Background(), client, opts)
	if err != nil {
		t.Fatal(err)
	}

	if result.SourceOSVolumeID != "vol-fin" {
		t.Fatalf("SourceOSVolumeID = %q, want vol-fin", result.SourceOSVolumeID)
	}
	if result.Policy.LocationCode != FinlandLocationCode || result.Policy.LocationSelection != ExplicitLocation {
		t.Fatalf("unexpected location policy: %#v", result.Policy)
	}
	if len(client.cloneRequests) != 1 || client.cloneRequests[0].SourceVolumeID != "vol-fin" || client.cloneRequests[0].LocationCode != FinlandLocationCode {
		t.Fatalf("unexpected clone requests: %#v", client.cloneRequests)
	}
}

func TestDeployRejectsAmbiguousSourceOSVolumeNameInAvailableLocation(t *testing.T) {
	client := &fakeClient{
		sourceVolumes: []Volume{
			{ID: "vol-nor-a", Name: "golden", Status: "detached", Location: "NOR-01", IsOSVolume: true},
			{ID: "vol-nor-b", Name: "golden", Status: "detached", Location: "NOR-01", IsOSVolume: true},
		},
		instanceType: InstanceType{InstanceType: "1V100.6V", GPUCount: 1},
		placements:   []SpotPlacement{{LocationCode: "NOR-01", SpotPrice: 1, PriceKnown: true, Currency: "eur"}},
	}
	opts := DefaultDeployOptions("1V100.6V")
	opts.SourceOSVolumeName = "golden"

	_, err := Deploy(context.Background(), client, opts)
	if err == nil {
		t.Fatal("expected ambiguous source volume name error")
	}
	if !strings.Contains(err.Error(), "multiple OS volumes in NOR-01") || !strings.Contains(err.Error(), "vol-nor-a") || !strings.Contains(err.Error(), "vol-nor-b") {
		t.Fatalf("error did not mention ambiguous volumes: %v", err)
	}
	if len(client.cloneRequests) != 0 || client.createRequest.InstanceType != "" {
		t.Fatalf("deploy should stop before clone/create: clone=%#v create=%#v", client.cloneRequests, client.createRequest)
	}
}

func TestDeployRejectsSourceOSVolumeNameWithSkipAvailabilityAndMultipleMatches(t *testing.T) {
	client := &fakeClient{
		sourceVolumes: []Volume{
			{ID: "vol-fin", Name: "golden", Status: "detached", Location: FinlandLocationCode, IsOSVolume: true},
			{ID: "vol-nor", Name: "golden", Status: "detached", Location: "NOR-01", IsOSVolume: true},
		},
		instanceType: InstanceType{InstanceType: "1V100.6V", GPUCount: 1},
	}
	opts := DefaultDeployOptions("1V100.6V")
	opts.SourceOSVolumeName = "golden"
	opts.SkipAvailabilityCheck = true

	_, err := Deploy(context.Background(), client, opts)
	if err == nil {
		t.Fatal("expected skip availability ambiguity error")
	}
	if !strings.Contains(err.Error(), "--location-code") || !strings.Contains(err.Error(), "availability checks") {
		t.Fatalf("error did not explain how to disambiguate: %v", err)
	}
}

func TestDeployCreatesSourceOSVolumeInstanceWithCustomUserData(t *testing.T) {
	client := &fakeClient{
		sourceVolume: Volume{ID: "vol-golden", Name: "golden", Status: "detached", Location: FinlandLocationCode, IsOSVolume: true},
		instanceType: InstanceType{InstanceType: "1V100.6V", GPUCount: 1},
		placements:   []SpotPlacement{{LocationCode: FinlandLocationCode, SpotPrice: 2, PriceKnown: true, Currency: "eur"}},
		script:       StartupScript{ID: "script-1", Name: "userdata"},
		clonedVolume: Volume{ID: "vol-clone-1", Name: "worker-os", Status: "cloning", Location: FinlandLocationCode},
		readyVolume:  Volume{ID: "vol-clone-1", Name: "worker-os", Status: "detached", Location: FinlandLocationCode},
		instance:     Instance{ID: "inst-1", Hostname: "worker", Status: "new", InstanceType: "1V100.6V", Location: FinlandLocationCode, IsSpot: true},
	}
	opts := DefaultDeployOptions("1V100.6V")
	opts.SourceOSVolumeID = "vol-golden"
	opts.Hostname = "worker"
	opts.UserDataScript = "#!/bin/sh\necho custom\n"

	result, err := Deploy(context.Background(), client, opts)
	if err != nil {
		t.Fatal(err)
	}

	if client.createdScriptName != "worker-userdata" {
		t.Fatalf("createdScriptName = %q", client.createdScriptName)
	}
	if !strings.Contains(client.createdScriptContent, "echo custom") {
		t.Fatalf("created script did not include custom userdata: %q", client.createdScriptContent)
	}
	if client.createRequest.StartupScriptID != "script-1" {
		t.Fatalf("StartupScriptID = %q", client.createRequest.StartupScriptID)
	}
	if result.StartupScript == nil || result.StartupScript.ID != "script-1" {
		t.Fatalf("unexpected startup script result: %#v", result.StartupScript)
	}
}

func TestDeployCreatesSpotInstanceFromDefaultImageWithUserData(t *testing.T) {
	client := &fakeClient{
		instanceType: InstanceType{InstanceType: "1V100.6V", GPUCount: 1},
		placements: []SpotPlacement{
			{LocationCode: "NOR-01", SpotPrice: 1, PriceKnown: true, Currency: "eur"},
			{LocationCode: FinlandLocationCode, SpotPrice: 2, PriceKnown: true, Currency: "eur"},
		},
		script:   StartupScript{ID: "script-1", Name: "userdata"},
		instance: Instance{ID: "inst-1", Hostname: "worker", Status: "new", InstanceType: "1V100.6V", Location: "NOR-01", IsSpot: true},
	}
	opts := DefaultDeployOptions("1V100.6V")
	opts.Hostname = "worker"

	result, err := Deploy(context.Background(), client, opts)
	if err != nil {
		t.Fatal(err)
	}

	if client.gotVolumeID != "" || len(client.cloneRequests) != 0 || client.waitedVolumeID != "" {
		t.Fatalf("unexpected volume operations: got=%q clone=%#v waited=%q", client.gotVolumeID, client.cloneRequests, client.waitedVolumeID)
	}
	if client.createRequest.Image != DefaultDeployImage {
		t.Fatalf("Image = %q, want %s", client.createRequest.Image, DefaultDeployImage)
	}
	if client.createRequest.LocationCode != "NOR-01" {
		t.Fatalf("LocationCode = %q, want NOR-01", client.createRequest.LocationCode)
	}
	if client.createRequest.StartupScriptID != "script-1" {
		t.Fatalf("StartupScriptID = %q", client.createRequest.StartupScriptID)
	}
	if !strings.Contains(client.createdScriptContent, "cuda-toolkit-13-0") {
		t.Fatalf("created script did not include default userdata marker")
	}
	if result.SourceOSVolumeID != "" || result.SourceOSVolume != nil || result.OSVolumeClone != nil || len(result.ClonedOSVolumeIDs) != 0 {
		t.Fatalf("unexpected source volume result: %#v", result)
	}
	if result.Policy.LocationCode != "NOR-01" || result.Policy.LocationSelection != SpotPlacementLocation || result.Policy.SourceOSVolumeLocked {
		t.Fatalf("unexpected policy: %#v", result.Policy)
	}
}

func TestDeployCreatesOnDemandInstanceFromDefaultImageWithUserData(t *testing.T) {
	client := &fakeClient{
		instanceType: InstanceType{InstanceType: "1H200.141S.44V", GPUCount: 1},
		placements: []SpotPlacement{
			{LocationCode: "FIN-02", Market: PricingMarketOnDemand, SpotPrice: 4, PriceKnown: true, Currency: "usd"},
		},
		script:   StartupScript{ID: "script-1", Name: "userdata"},
		instance: Instance{ID: "inst-1", Hostname: "worker", Status: "new", InstanceType: "1H200.141S.44V", Location: "FIN-02"},
	}
	opts := DefaultDeployOptions("1H200.141S.44V")
	opts.Market = PricingMarketOnDemand
	opts.Hostname = "worker"
	opts.LocationCode = "fin-02"

	result, err := Deploy(context.Background(), client, opts)
	if err != nil {
		t.Fatal(err)
	}

	if len(client.gotInstanceTypeSpot) != 1 || client.gotInstanceTypeSpot[0] {
		t.Fatalf("GetInstanceType spot args = %#v, want [false]", client.gotInstanceTypeSpot)
	}
	if len(client.gotPlacementSpot) != 1 || client.gotPlacementSpot[0] {
		t.Fatalf("GetPlacementOptions spot args = %#v, want [false]", client.gotPlacementSpot)
	}
	if result.Policy.Market != PricingMarketOnDemand || result.Policy.SpotOnly {
		t.Fatalf("unexpected policy: %#v", result.Policy)
	}
	if result.Placement == nil || result.Placement.LocationCode != "FIN-02" || result.Placement.SpotPrice != 4 {
		t.Fatalf("unexpected placement: %#v", result.Placement)
	}
	if client.createRequest.IsSpot || client.createRequest.Contract != OnDemandContract {
		t.Fatalf("request did not use on-demand contract: %#v", client.createRequest)
	}
	if client.createRequest.Image != DefaultDeployImage || client.createRequest.LocationCode != "FIN-02" {
		t.Fatalf("unexpected create request: %#v", client.createRequest)
	}
	if client.createRequest.StartupScriptID != "script-1" {
		t.Fatalf("StartupScriptID = %q", client.createRequest.StartupScriptID)
	}
}

func TestDeployCreatesExplicitLocationSpotInstance(t *testing.T) {
	client := &fakeClient{
		sourceVolume: Volume{ID: "vol-golden", Name: "golden", Status: "detached", Location: FinlandLocationCode, IsOSVolume: true},
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

func TestDeployRejectsExplicitLocationDifferentFromSourceOSVolumeLocation(t *testing.T) {
	client := &fakeClient{
		sourceVolume: Volume{ID: "vol-golden", Name: "golden", Status: "detached", Location: FinlandLocationCode, IsOSVolume: true},
		instanceType: InstanceType{InstanceType: "1V100.6V", GPUCount: 1},
	}
	opts := DefaultDeployOptions("1V100.6V")
	opts.SourceOSVolumeID = "vol-golden"
	opts.LocationCode = "NOR-01"

	_, err := Deploy(context.Background(), client, opts)
	if err == nil {
		t.Fatal("expected source volume location mismatch error")
	}
	if !strings.Contains(err.Error(), "source_os_volume_id") || !strings.Contains(err.Error(), FinlandLocationCode) || !strings.Contains(err.Error(), "NOR-01") {
		t.Fatalf("error did not include source and requested locations: %v", err)
	}
	if client.createdScript {
		t.Fatal("startup script should not be created for mismatched source volume location")
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

func TestDeployRejectsClonedOSVolumeWithoutSourceOSVolume(t *testing.T) {
	opts := DefaultDeployOptions("1V100.6V")
	opts.DryRun = true
	opts.ClonedOSVolumeID = "vol-clone-1"

	_, err := Deploy(context.Background(), nil, opts)
	if err == nil {
		t.Fatal("expected cloned_os_volume_id without source_os_volume_id error")
	}
	if !strings.Contains(err.Error(), "cloned_os_volume_id") || !strings.Contains(err.Error(), "source_os_volume_id") {
		t.Fatalf("error did not mention cloned/source volume IDs: %v", err)
	}
}

func TestDeployDeletesOnlyClonedVolumeWhenCreateInstanceFails(t *testing.T) {
	client := &fakeClient{
		sourceVolume: Volume{ID: "vol-golden", Name: "golden", Status: "detached", Location: FinlandLocationCode, IsOSVolume: true},
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
	if !strings.Contains(err.Error(), "cleaned up cloned OS volume vol-clone-1") {
		t.Fatalf("error did not mention cloned volume cleanup: %v", err)
	}
	if strings.Contains(strings.Join(client.deletedVolumeIDs, ","), "vol-golden") {
		t.Fatalf("source volume was deleted: %#v", client.deletedVolumeIDs)
	}
}

func TestDeployKeepsClonedVolumeWhenCreateInstanceFailsAndCleanupDisabled(t *testing.T) {
	client := &fakeClient{
		sourceVolume: Volume{ID: "vol-golden", Name: "golden", Status: "detached", Location: FinlandLocationCode, IsOSVolume: true},
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
	opts.KeepClonedOSVolumeOnFailure = true

	_, err := Deploy(context.Background(), client, opts)
	if err == nil {
		t.Fatal("expected create instance error")
	}
	if len(client.deletedVolumeIDs) != 0 {
		t.Fatalf("unexpected deleted volumes: %#v", client.deletedVolumeIDs)
	}
	if !strings.Contains(err.Error(), "cloned OS volume vol-clone-1 was kept") {
		t.Fatalf("error did not mention kept cloned volume: %v", err)
	}
}

func TestDeployReusesClonedOSVolume(t *testing.T) {
	client := &fakeClient{
		sourceVolume: Volume{ID: "vol-golden", Name: "golden", Status: "detached", Location: FinlandLocationCode, IsOSVolume: true},
		instanceType: InstanceType{InstanceType: "1V100.6V", GPUCount: 1},
		placements:   []SpotPlacement{{LocationCode: FinlandLocationCode, SpotPrice: 2, PriceKnown: true}},
		script:       StartupScript{ID: "script-1", Name: "userdata"},
		readyVolume:  Volume{ID: "vol-reuse-1", Name: "kept-os", Status: "detached", Location: FinlandLocationCode},
		instance:     Instance{ID: "inst-1", Hostname: "worker", Status: "new", InstanceType: "1V100.6V", Location: FinlandLocationCode, IsSpot: true},
	}
	opts := DefaultDeployOptions("1V100.6V")
	opts.SourceOSVolumeID = "vol-golden"
	opts.ClonedOSVolumeID = "vol-reuse-1"
	opts.Hostname = "worker"
	opts.LocationCode = FinlandLocationCode

	result, err := Deploy(context.Background(), client, opts)
	if err != nil {
		t.Fatal(err)
	}

	if len(client.cloneRequests) != 0 {
		t.Fatalf("unexpected clone requests: %#v", client.cloneRequests)
	}
	if client.waitedVolumeID != "vol-reuse-1" {
		t.Fatalf("waitedVolumeID = %q, want vol-reuse-1", client.waitedVolumeID)
	}
	if client.createRequest.Image != "vol-reuse-1" {
		t.Fatalf("Image = %q, want reused cloned OS volume ID", client.createRequest.Image)
	}
	if result.OSVolumeClone == nil || result.OSVolumeClone.VolumeID != "vol-reuse-1" || !result.OSVolumeClone.Reused {
		t.Fatalf("unexpected OSVolumeClone: %#v", result.OSVolumeClone)
	}
	if result.ClonedOSVolumeIDs["inst-1"] != "vol-reuse-1" {
		t.Fatalf("unexpected clone tracking: %#v", result.ClonedOSVolumeIDs)
	}
}

func TestDeployRejectsReusedClonedOSVolumeLocationMismatch(t *testing.T) {
	client := &fakeClient{
		sourceVolume: Volume{ID: "vol-golden", Name: "golden", Status: "detached", Location: FinlandLocationCode, IsOSVolume: true},
		instanceType: InstanceType{InstanceType: "1V100.6V", GPUCount: 1},
		placements:   []SpotPlacement{{LocationCode: FinlandLocationCode, SpotPrice: 2, PriceKnown: true}},
		script:       StartupScript{ID: "script-1", Name: "userdata"},
		readyVolume:  Volume{ID: "vol-reuse-1", Name: "kept-os", Status: "detached", Location: "NOR-01"},
	}
	opts := DefaultDeployOptions("1V100.6V")
	opts.SourceOSVolumeID = "vol-golden"
	opts.ClonedOSVolumeID = "vol-reuse-1"
	opts.Hostname = "worker"
	opts.LocationCode = FinlandLocationCode

	_, err := Deploy(context.Background(), client, opts)
	if err == nil {
		t.Fatal("expected reused clone location mismatch")
	}
	if !strings.Contains(err.Error(), "cloned_os_volume_id vol-reuse-1 is in NOR-01") {
		t.Fatalf("error did not mention reused clone location: %v", err)
	}
	if len(client.deletedVolumeIDs) != 0 {
		t.Fatalf("reused cloned volume should not be deleted: %#v", client.deletedVolumeIDs)
	}
}

type fakeClient struct {
	sourceVolume         Volume
	sourceVolumes        []Volume
	instanceType         InstanceType
	placements           []SpotPlacement
	script               StartupScript
	clonedVolume         Volume
	readyVolume          Volume
	instance             Instance
	placementsListed     bool
	volumesListed        bool
	gotVolumeID          string
	gotInstanceTypeSpot  []bool
	gotPlacementSpot     []bool
	createdScript        bool
	createdScriptName    string
	createdScriptContent string
	cloneRequests        []CloneVolumeRequest
	waitedVolumeID       string
	deletedVolumeIDs     []string
	createRequest        CreateInstanceRequest
	createErr            error
}

func (c *fakeClient) GetVolume(_ context.Context, volumeID string) (Volume, error) {
	c.gotVolumeID = volumeID
	if c.sourceVolume.ID == "" {
		c.sourceVolume = Volume{ID: volumeID, Location: FinlandLocationCode, IsOSVolume: true}
	}
	return c.sourceVolume, nil
}

func (c *fakeClient) ListVolumes(context.Context) ([]Volume, error) {
	c.volumesListed = true
	return append([]Volume(nil), c.sourceVolumes...), nil
}

func (c *fakeClient) GetInstanceType(_ context.Context, _ string, spot bool, _ string) (InstanceType, error) {
	c.gotInstanceTypeSpot = append(c.gotInstanceTypeSpot, spot)
	return c.instanceType, nil
}

func (c *fakeClient) GetPlacementOptions(_ context.Context, _ string, spot bool) ([]SpotPlacement, error) {
	c.placementsListed = true
	c.gotPlacementSpot = append(c.gotPlacementSpot, spot)
	return append([]SpotPlacement(nil), c.placements...), nil
}

func (c *fakeClient) CreateStartupScript(_ context.Context, name, script string) (StartupScript, error) {
	c.createdScript = true
	c.createdScriptName = name
	c.createdScriptContent = script
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
