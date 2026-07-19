package cli

import (
	"context"
	"slices"
	"strings"
	"testing"
	"time"

	cloudcore "nnctl/internal/cloud"
	"nnctl/internal/cloud/digitalocean"
)

func TestDigitalOceanCloudBenchmarkStrategyUsesROCmAndCleansUp(t *testing.T) {
	provider := &fakeDigitalOceanBenchmarkProvider{}
	registry, err := cloudcore.NewRegistry(fakeDigitalOceanBenchmarkFactory{provider: provider})
	if err != nil {
		t.Fatal(err)
	}
	app := &app{repoRoot: t.TempDir(), cloudRegistry: registry}
	opts := defaultCloudBenchmarkDeployOptions()
	opts.provider = digitalocean.ProviderName
	opts.InstanceType = "gpu-mi300x1-192gb"
	opts.LocationCode = "tor1"
	opts.SSHKeyIDs = []string{"17"}
	if err := normalizeCloudBenchmarkDeployOptionsForProvider(&opts, digitalocean.ProviderName); err != nil {
		t.Fatal(err)
	}
	run := newCloudBenchmarkRun(app, t.Context(), opts)
	strategy := &digitalOceanCloudBenchmarkStrategy{}
	run.strategy = strategy
	run.progress = newCloudBenchmarkProgress(&strings.Builder{})
	run.ops.waitSSH = func(context.Context, string, []string, string, time.Duration, time.Duration) error {
		return nil
	}
	run.ops.waitCommand = func(_ context.Context, _ string, _ []string, host, command string, _, _ time.Duration) error {
		if host != "root@192.0.2.42" || !strings.Contains(command, "rocminfo") {
			t.Fatalf("preflight host=%q command=%q", host, command)
		}
		return nil
	}

	if err := run.prepareProviderImage(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = run.removeWorkflowDir() })
	if run.opts.backend != "rocm" || strategy.planned.Request.Image != digitalocean.AMDImage {
		t.Fatalf("planned benchmark: backend=%q deployment=%#v", run.opts.backend, strategy.planned)
	}
	if err := run.deployWorker(); err != nil {
		t.Fatal(err)
	}
	if err := run.waitForWorker(); err != nil {
		t.Fatal(err)
	}
	metadata := strategy.metadata(run)
	if metadata.Provider != digitalocean.ProviderName || metadata.GPU != "MI300X" || metadata.MemoryMiB != "196608" {
		t.Fatalf("metadata = %#v", metadata)
	}
	if err := strategy.cleanup(t.Context(), run); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(provider.destroyed, []string{"42"}) {
		t.Fatalf("destroyed = %#v", provider.destroyed)
	}
}

type fakeDigitalOceanBenchmarkFactory struct {
	provider cloudcore.Provider
}

func (fakeDigitalOceanBenchmarkFactory) Descriptor() cloudcore.Descriptor {
	return cloudcore.Descriptor{
		Name: digitalocean.ProviderName, DisplayName: "DigitalOcean",
		Capabilities: []cloudcore.Capability{cloudcore.CapabilityCompute, cloudcore.CapabilityBenchmarkImage},
	}
}

func (fakeDigitalOceanBenchmarkFactory) Status(context.Context, cloudcore.Configuration) cloudcore.ConfigurationStatus {
	return cloudcore.ConfigurationStatus{Configured: true}
}

func (f fakeDigitalOceanBenchmarkFactory) New(context.Context, cloudcore.Configuration) (cloudcore.Provider, error) {
	return f.provider, nil
}

type fakeDigitalOceanBenchmarkProvider struct {
	destroyed []string
}

func (fakeDigitalOceanBenchmarkProvider) Descriptor() cloudcore.Descriptor {
	return (fakeDigitalOceanBenchmarkFactory{}).Descriptor()
}

func (fakeDigitalOceanBenchmarkProvider) Offerings(context.Context, cloudcore.OfferingFilters) ([]cloudcore.Offering, error) {
	return []cloudcore.Offering{{
		Provider: digitalocean.ProviderName, ID: "gpu-mi300x1-192gb", Location: "tor1",
		Market:      digitalocean.MarketOnDemand,
		Accelerator: cloudcore.Accelerator{Manufacturer: "amd", Model: "MI300X", Count: 1, MemoryMiB: 192 * 1024},
		Backends:    []string{"cpu", "rocm"}, PricePerHour: 1.99, PriceKnown: true,
		Currency: digitalocean.CurrencyUSD, Available: true, Discoverable: true,
	}}, nil
}

func (fakeDigitalOceanBenchmarkProvider) Deploy(_ context.Context, request cloudcore.DeployRequest) (*cloudcore.Deployment, error) {
	if request.Image == "" {
		request.Image = digitalocean.AMDImage
	}
	deployment := &cloudcore.Deployment{
		Provider: digitalocean.ProviderName, DryRun: request.DryRun, Request: request,
	}
	if !request.DryRun {
		deployment.Instance = &cloudcore.Instance{
			Provider: digitalocean.ProviderName, ID: "42", State: cloudcore.InstancePending,
			OfferingID: request.OfferingID, Location: request.Location,
			Market: digitalocean.MarketOnDemand, Backends: []string{"cpu", "rocm"},
		}
	}
	return deployment, nil
}

func (fakeDigitalOceanBenchmarkProvider) Instance(context.Context, string) (cloudcore.Instance, error) {
	return cloudcore.Instance{
		Provider: digitalocean.ProviderName, ID: "42", State: cloudcore.InstanceRunning,
		PublicIP: "192.0.2.42", OfferingID: "gpu-mi300x1-192gb", Location: "tor1",
		Market: digitalocean.MarketOnDemand, Backends: []string{"cpu", "rocm"}, PricePerHour: 1.99,
	}, nil
}

func (fakeDigitalOceanBenchmarkProvider) Instances(context.Context, cloudcore.ListOptions) ([]cloudcore.Instance, error) {
	return nil, nil
}

func (f *fakeDigitalOceanBenchmarkProvider) Destroy(_ context.Context, request cloudcore.DestroyRequest) (*cloudcore.DestroyResult, error) {
	f.destroyed = append(f.destroyed, request.InstanceIDs...)
	return &cloudcore.DestroyResult{Provider: digitalocean.ProviderName, Request: request}, nil
}

var (
	_ cloudcore.Factory  = fakeDigitalOceanBenchmarkFactory{}
	_ cloudcore.Provider = (*fakeDigitalOceanBenchmarkProvider)(nil)
)
