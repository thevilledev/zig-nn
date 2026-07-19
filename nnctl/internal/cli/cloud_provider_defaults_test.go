package cli

import (
	"context"
	"io"
	"strings"
	"testing"

	cloudcore "nnctl/internal/cloud"
)

func TestCloudCommandsUseProviderDefaultMarketWithoutProviderSpecificDefaults(t *testing.T) {
	provider := &recordingCloudProvider{descriptor: cloudcore.Descriptor{
		Name: "future-neocloud", DisplayName: "Future Neocloud", DefaultMarket: "reserved-capacity",
		Capabilities: []cloudcore.Capability{cloudcore.CapabilityCompute, cloudcore.CapabilityPricing},
	}}
	registry, err := cloudcore.NewRegistry(recordingCloudFactory{provider: provider})
	if err != nil {
		t.Fatal(err)
	}
	app := &app{cloudRegistry: registry, stdoutWriter: io.Discard, stderrWriter: io.Discard}

	deploy := defaultCloudDeployOptions()
	deploy.provider = provider.descriptor.Name
	deploy.InstanceType = "future-accelerator-1"
	deploy.DryRun = true
	if err := app.runCloudDeploy(t.Context(), deploy); err != nil {
		t.Fatal(err)
	}
	if provider.deployRequest.Market != "reserved-capacity" {
		t.Fatalf("deploy market = %q, want provider default", provider.deployRequest.Market)
	}
	if provider.deployRequest.Image != "" {
		t.Fatalf("deploy image = %q, want no cross-provider image default", provider.deployRequest.Image)
	}
	if len(provider.deployRequest.Options) != 0 {
		t.Fatalf("deploy options = %#v, want no cross-provider options", provider.deployRequest.Options)
	}

	pricing := cloudPricingOptions{provider: provider.descriptor.Name, sortBy: "price", jsonOutput: true}
	if err := app.runCloudPricing(t.Context(), pricing); err != nil {
		t.Fatal(err)
	}
	if len(provider.offeringFilters.Markets) != 1 || provider.offeringFilters.Markets[0] != "reserved-capacity" {
		t.Fatalf("pricing markets = %#v, want provider default", provider.offeringFilters.Markets)
	}
}

func TestCloudPricingPassesThroughProviderSpecificMarket(t *testing.T) {
	provider := &recordingCloudProvider{descriptor: cloudcore.Descriptor{
		Name: "future-neocloud", DisplayName: "Future Neocloud", DefaultMarket: "reserved-capacity",
		Capabilities: []cloudcore.Capability{cloudcore.CapabilityCompute, cloudcore.CapabilityPricing},
	}}
	registry, err := cloudcore.NewRegistry(recordingCloudFactory{provider: provider})
	if err != nil {
		t.Fatal(err)
	}
	app := &app{cloudRegistry: registry, stdoutWriter: io.Discard, stderrWriter: io.Discard}

	pricing := cloudPricingOptions{
		provider: provider.descriptor.Name, market: "Interruptible-VRAM", sortBy: "price", jsonOutput: true,
	}
	if err := app.runCloudPricing(t.Context(), pricing); err != nil {
		t.Fatal(err)
	}
	if len(provider.offeringFilters.Markets) != 1 || provider.offeringFilters.Markets[0] != "interruptible-vram" {
		t.Fatalf("pricing markets = %#v, want provider-specific market", provider.offeringFilters.Markets)
	}
}

func TestCloudCommandsRejectVerdaResourceFlagsForAnotherProvider(t *testing.T) {
	provider := &recordingCloudProvider{descriptor: cloudcore.Descriptor{
		Name: "future-neocloud", DisplayName: "Future Neocloud",
		Capabilities: []cloudcore.Capability{cloudcore.CapabilityCompute},
	}}
	registry, err := cloudcore.NewRegistry(recordingCloudFactory{provider: provider})
	if err != nil {
		t.Fatal(err)
	}
	app := &app{cloudRegistry: registry, stdoutWriter: io.Discard, stderrWriter: io.Discard}

	deploy := defaultCloudDeployOptions()
	deploy.provider = provider.descriptor.Name
	deploy.InstanceType = "future-accelerator-1"
	deploy.SourceOSVolumeID = "verda-volume"
	deploy.DryRun = true
	if err := app.runCloudDeploy(t.Context(), deploy); err == nil || !strings.Contains(err.Error(), "verda volume") {
		t.Fatalf("runCloudDeploy() error = %v", err)
	}

	destroy := cloudDestroyOptions{
		provider: provider.descriptor.Name, InstanceIDs: []string{"instance-1"},
		VolumeIDs: []string{"verda-volume"}, DeletePermanently: true, DryRun: true,
	}
	if err := app.runCloudDestroy(t.Context(), destroy); err == nil || !strings.Contains(err.Error(), "verda volume") {
		t.Fatalf("runCloudDestroy() error = %v", err)
	}
}

type recordingCloudFactory struct {
	provider *recordingCloudProvider
}

func (f recordingCloudFactory) Descriptor() cloudcore.Descriptor {
	return f.provider.descriptor
}

func (recordingCloudFactory) Status(context.Context, cloudcore.Configuration) cloudcore.ConfigurationStatus {
	return cloudcore.ConfigurationStatus{Configured: true}
}

func (f recordingCloudFactory) New(context.Context, cloudcore.Configuration) (cloudcore.Provider, error) {
	return f.provider, nil
}

type recordingCloudProvider struct {
	descriptor      cloudcore.Descriptor
	deployRequest   cloudcore.DeployRequest
	offeringFilters cloudcore.OfferingFilters
}

func (p *recordingCloudProvider) Descriptor() cloudcore.Descriptor {
	return p.descriptor
}

func (p *recordingCloudProvider) Offerings(_ context.Context, filters cloudcore.OfferingFilters) ([]cloudcore.Offering, error) {
	p.offeringFilters = filters
	return nil, nil
}

func (p *recordingCloudProvider) Deploy(_ context.Context, request cloudcore.DeployRequest) (*cloudcore.Deployment, error) {
	p.deployRequest = request
	return &cloudcore.Deployment{Provider: p.descriptor.Name, DryRun: request.DryRun, Request: request}, nil
}

func (*recordingCloudProvider) Instance(context.Context, string) (cloudcore.Instance, error) {
	return cloudcore.Instance{}, nil
}

func (*recordingCloudProvider) Instances(context.Context, cloudcore.ListOptions) ([]cloudcore.Instance, error) {
	return nil, nil
}

func (p *recordingCloudProvider) Destroy(_ context.Context, request cloudcore.DestroyRequest) (*cloudcore.DestroyResult, error) {
	return &cloudcore.DestroyResult{Provider: p.descriptor.Name, DryRun: request.DryRun, Request: request}, nil
}
