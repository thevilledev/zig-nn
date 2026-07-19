package verda

import (
	"context"
	"errors"
	"reflect"
	"testing"

	cloudcore "nnctl/internal/cloud"
)

type failingCredentialStore struct{}

func (failingCredentialStore) Credentials(context.Context) (Credentials, error) {
	return Credentials{}, errors.New("credentials unavailable")
}

func TestProviderFactoryReportsCredentialStatus(t *testing.T) {
	status := (ProviderFactory{CredentialStore: failingCredentialStore{}}).Status(t.Context(), cloudcore.Configuration{})
	if status.Configured || status.Error != "credentials unavailable" {
		t.Fatalf("Status() = %#v", status)
	}
	if descriptor := (ProviderFactory{}).Descriptor(); descriptor.Name != ProviderName || !descriptor.Supports(cloudcore.CapabilityVolumes) {
		t.Fatalf("Descriptor() = %#v", descriptor)
	}
}

func TestOfferingFromPriceMapsAccelerator(t *testing.T) {
	offering, err := offeringFromPrice(InstancePrice{
		InstanceType: "1A100.22V", Model: "A100", Manufacturer: "NVIDIA",
		GPUCount: 1, LocationCode: "FIN-02", Market: PricingMarketSpot,
		PricePerHour: 1.25, PriceKnown: true, Currency: "eur", Available: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if offering.Provider != ProviderName || offering.Accelerator.Manufacturer != "nvidia" || !reflect.DeepEqual(offering.Backends, []string{"cpu", "cuda"}) {
		t.Fatalf("offering = %#v", offering)
	}
	if !offering.Discoverable || offering.Currency != "EUR" {
		t.Fatalf("offering metadata = %#v", offering)
	}
}

func TestBackendsForAMDAccelerator(t *testing.T) {
	if got := backendsForAccelerator("AMD", "MI300X", 1); !reflect.DeepEqual(got, []string{"cpu", "rocm"}) {
		t.Fatalf("backends = %#v", got)
	}
}

func TestDeployOptionsFromRequestMapsProviderOptions(t *testing.T) {
	options, err := deployOptionsFromRequest(cloudcore.DeployRequest{
		OfferingID: "1H200.141S.44V",
		Location:   "FIN-02",
		Market:     PricingMarketOnDemand,
		SSHKeyIDs:  []string{"key-1"},
		DryRun:     true,
		Options: map[string]string{
			OptionSourceOSVolumeID:          "source-1",
			OptionSkipAvailabilityCheck:     "true",
			OptionKeepClonedVolumeOnFailure: "true",
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if options.InstanceType != "1H200.141S.44V" || options.SourceOSVolumeID != "source-1" || !options.SkipAvailabilityCheck || !options.KeepClonedOSVolumeOnFailure {
		t.Fatalf("options = %#v", options)
	}
}

func TestDeploymentFromResultPreservesGoldenVolume(t *testing.T) {
	deployment, err := deploymentFromResult(cloudcore.DeployRequest{OfferingID: "1A100.22V"}, &DeployResult{
		Provider:         ProviderName,
		SourceOSVolumeID: "source-1",
		OSVolumeClone:    &OSVolumeClone{VolumeID: "clone-1"},
		InstanceType:     &InstanceType{Model: "A100", GPUCount: 1},
		Instance:         &Instance{ID: "instance-1", Status: "running", InstanceType: "1A100.22V"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if deployment.Instance == nil || !reflect.DeepEqual(deployment.Instance.Backends, []string{"cpu", "cuda"}) {
		t.Fatalf("instance = %#v", deployment.Instance)
	}
	wantResources := []cloudcore.ResourceRef{
		{Kind: ResourceOSVolume, ID: "source-1", Preserve: true},
		{Kind: ResourceOSVolume, ID: "clone-1"},
	}
	if !reflect.DeepEqual(deployment.Resources, wantResources) {
		t.Fatalf("resources = %#v", deployment.Resources)
	}
}

func TestDestroyOptionsProtectsProviderResources(t *testing.T) {
	options, err := destroyOptionsFromRequest(cloudcore.DestroyRequest{
		InstanceIDs: []string{"instance-1"},
		Resources: []cloudcore.ResourceRef{
			{Kind: ResourceOSVolume, ID: "source-1", Preserve: true},
			{Kind: ResourceOSVolume, ID: "clone-1"},
		},
		Permanent: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if options.SourceOSVolumeID != "source-1" || !reflect.DeepEqual(options.VolumeIDs, []string{"clone-1"}) {
		t.Fatalf("options = %#v", options)
	}
}
