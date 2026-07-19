package cloud

import (
	"context"
	"strings"
	"testing"
)

type fakeFactory struct {
	name string
}

func (f fakeFactory) Descriptor() Descriptor {
	return Descriptor{Name: f.name, DisplayName: strings.ToUpper(f.name), Capabilities: []Capability{CapabilityCompute}}
}

func (fakeFactory) Status(context.Context, Configuration) ConfigurationStatus {
	return ConfigurationStatus{Configured: true}
}

func (fakeFactory) New(context.Context, Configuration) (Provider, error) {
	return nil, nil
}

func TestRegistryNormalizesAndSortsProviders(t *testing.T) {
	registry, err := NewRegistry(fakeFactory{name: " Verda "}, fakeFactory{name: "digitalocean"})
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(registry.Names(), ","); got != "digitalocean,verda" {
		t.Fatalf("Names() = %q", got)
	}
	if _, err := registry.Factory(" VERDA "); err != nil {
		t.Fatalf("Factory() error = %v", err)
	}
}

func TestRegistryRejectsDuplicateProvider(t *testing.T) {
	_, err := NewRegistry(fakeFactory{name: "verda"}, fakeFactory{name: "VERDA"})
	if err == nil || !strings.Contains(err.Error(), "more than once") {
		t.Fatalf("NewRegistry() error = %v", err)
	}
}

func TestRegistryReportsAvailableProviders(t *testing.T) {
	registry, err := NewRegistry(fakeFactory{name: "verda"})
	if err != nil {
		t.Fatal(err)
	}
	_, err = registry.Factory("missing")
	if err == nil || !strings.Contains(err.Error(), "available providers: verda") {
		t.Fatalf("Factory() error = %v", err)
	}
}

func TestDescriptorSupportsCapability(t *testing.T) {
	descriptor := Descriptor{Capabilities: []Capability{CapabilityCompute, CapabilityPricing}}
	if !descriptor.Supports(CapabilityPricing) || descriptor.Supports(CapabilityVolumes) {
		t.Fatalf("unexpected capability result")
	}
}

func TestInstanceTerminal(t *testing.T) {
	if !InstanceDestroyed.Terminal() || !InstanceFailed.Terminal() {
		t.Fatal("terminal states were not recognized")
	}
	if InstanceRunning.Terminal() || InstanceUnknown.Terminal() {
		t.Fatal("non-terminal state was recognized as terminal")
	}
}
