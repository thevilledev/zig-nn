package lab

import (
	"context"
	"errors"
	"path/filepath"
	"testing"

	cloudcore "nnctl/internal/cloud"
)

func TestCloudManagerReportsProviderStatusAndPartialOptions(t *testing.T) {
	repoRoot, err := filepath.Abs("../../..")
	if err != nil {
		t.Fatal(err)
	}
	manager, err := NewCloudManager(CloudManagerOptions{
		RepoRoot:     repoRoot,
		DatabasePath: filepath.Join(t.TempDir(), "state.sqlite"),
		Providers: []CloudProviderConfig{
			{
				Factory: fakeCloudProviderFactory{
					descriptor: cloudcore.Descriptor{
						Name:         "digitalocean",
						DisplayName:  "DigitalOcean",
						Capabilities: []cloudcore.Capability{cloudcore.CapabilityCompute, cloudcore.CapabilityPricing},
					},
					configured: true,
					provider: fakeCoreCloudProvider{offerings: []cloudcore.Offering{{
						Provider: "digitalocean", ID: "gpu-mi300x1-192gb", Location: "tor1",
						Accelerator: cloudcore.Accelerator{Manufacturer: "amd", Model: "MI300X", Count: 1},
						Backends:    []string{"cpu", "rocm"}, Available: true, Discoverable: true,
					}}},
				},
			},
			{
				Factory: fakeCloudProviderFactory{
					descriptor: cloudcore.Descriptor{Name: "future-neocloud", DisplayName: "Future Neocloud"},
					configured: true,
					provider:   fakeCoreCloudProvider{offeringsErr: errors.New("catalog unavailable")},
				},
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = manager.Close() })

	status := manager.Status(t.Context())
	if !status.Configured || status.Provider != "digitalocean" || len(status.Providers) != 2 {
		t.Fatalf("cloud status = %#v", status)
	}
	if status.Providers[0].Name != "digitalocean" || status.Providers[0].DisplayName != "DigitalOcean" {
		t.Fatalf("first provider status = %#v", status.Providers[0])
	}

	options, err := manager.Options(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if len(options.Offerings) != 1 || options.Offerings[0].Backends[1] != "rocm" {
		t.Fatalf("cloud offerings = %#v", options.Offerings)
	}
	if options.Errors["future-neocloud"] != "catalog unavailable" {
		t.Fatalf("provider errors = %#v", options.Errors)
	}
}

type fakeCloudProviderFactory struct {
	descriptor cloudcore.Descriptor
	configured bool
	statusErr  string
	provider   cloudcore.Provider
	newErr     error
}

func (f fakeCloudProviderFactory) Descriptor() cloudcore.Descriptor {
	return f.descriptor
}

func (f fakeCloudProviderFactory) Status(context.Context, cloudcore.Configuration) cloudcore.ConfigurationStatus {
	return cloudcore.ConfigurationStatus{Configured: f.configured, Error: f.statusErr}
}

func (f fakeCloudProviderFactory) New(context.Context, cloudcore.Configuration) (cloudcore.Provider, error) {
	return f.provider, f.newErr
}

type fakeCoreCloudProvider struct {
	offerings    []cloudcore.Offering
	offeringsErr error
}

func (f fakeCoreCloudProvider) Descriptor() cloudcore.Descriptor {
	return cloudcore.Descriptor{Name: "fake", DisplayName: "Fake"}
}

func (f fakeCoreCloudProvider) Offerings(context.Context, cloudcore.OfferingFilters) ([]cloudcore.Offering, error) {
	return append([]cloudcore.Offering(nil), f.offerings...), f.offeringsErr
}

func (fakeCoreCloudProvider) Deploy(context.Context, cloudcore.DeployRequest) (*cloudcore.Deployment, error) {
	return nil, errors.New("not implemented")
}

func (fakeCoreCloudProvider) Instance(context.Context, string) (cloudcore.Instance, error) {
	return cloudcore.Instance{}, errors.New("not implemented")
}

func (fakeCoreCloudProvider) Instances(context.Context, cloudcore.ListOptions) ([]cloudcore.Instance, error) {
	return nil, errors.New("not implemented")
}

func (fakeCoreCloudProvider) Destroy(context.Context, cloudcore.DestroyRequest) (*cloudcore.DestroyResult, error) {
	return nil, errors.New("not implemented")
}

var _ cloudcore.Factory = fakeCloudProviderFactory{}
var _ cloudcore.Provider = fakeCoreCloudProvider{}
