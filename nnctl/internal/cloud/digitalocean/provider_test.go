package digitalocean

import (
	"context"
	"errors"
	"slices"
	"strconv"
	"strings"
	"testing"

	cloudcore "nnctl/internal/cloud"
)

func TestProviderFactoryReportsCredentialStatus(t *testing.T) {
	factory := ProviderFactory{CredentialStore: fakeCredentialStore{err: errors.New("missing token")}}
	status := factory.Status(t.Context(), cloudcore.Configuration{})
	if status.Configured || !strings.Contains(status.Error, "missing token") {
		t.Fatalf("status = %#v", status)
	}
	descriptor := factory.Descriptor()
	if descriptor.Name != ProviderName || !descriptor.Supports(cloudcore.CapabilityBenchmarkImage) || descriptor.Supports(cloudcore.CapabilityVolumes) {
		t.Fatalf("descriptor = %#v", descriptor)
	}
}

func TestCloudProviderMapsGPUOfferingsAndContractEntries(t *testing.T) {
	manual, err := parseManualOfferings("gpu-mi325x1-256gb-contracted@nyc3")
	if err != nil {
		t.Fatal(err)
	}
	client := &fakeClient{sizes: []Size{
		{Slug: "s-1vcpu-1gb", Available: true, Regions: []string{"nyc3"}},
		{
			Slug: "gpu-mi300x1-192gb", PriceHourly: 1.99, Available: true,
			Regions: []string{"tor1", "nyc3"},
			GPUInfo: &GPUInfo{Count: 1, Model: "amd_mi300x", VRAM: UnitValue{Amount: 192, Unit: "gib"}},
		},
	}}
	provider := NewCloudProvider(client, manual, nil)
	offerings, err := provider.Offerings(t.Context(), cloudcore.OfferingFilters{
		Manufacturer: "amd", GPUCounts: []int{1}, AvailableOnly: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(offerings) != 3 {
		t.Fatalf("offerings = %#v", offerings)
	}
	if offerings[0].ID != "gpu-mi300x1-192gb" || offerings[0].Accelerator.Model != "MI300X" || offerings[0].Accelerator.MemoryMiB != 192*1024 {
		t.Fatalf("self-service offering = %#v", offerings[0])
	}
	if offerings[0].Backends[1] != "rocm" || !offerings[0].Discoverable || !offerings[0].PriceKnown {
		t.Fatalf("self-service offering metadata = %#v", offerings[0])
	}
	if offerings[2].ID != "gpu-mi325x1-256gb-contracted" || offerings[2].Discoverable {
		t.Fatalf("contract offering = %#v", offerings[2])
	}

	explicit, err := provider.Offerings(t.Context(), cloudcore.OfferingFilters{
		IDs: []string{"gpu-mi350x1-288gb-contracted"}, Locations: []string{"atl1"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(explicit) != 1 || explicit[0].Location != "atl1" || explicit[0].Discoverable {
		t.Fatalf("explicit contract offering = %#v", explicit)
	}
}

func TestCloudProviderDeploysAMDWithAccountSSHKeys(t *testing.T) {
	client := &fakeClient{
		created: Droplet{
			ID: 42, Name: "nnctl-mi300x", Status: "active", SizeSlug: "gpu-mi300x1-192gb",
			Region:   Region{Slug: "tor1"},
			Networks: Networks{V4: []Network{{IPAddress: "10.0.0.2", Type: "private"}, {IPAddress: "192.0.2.42", Type: "public"}}},
		},
	}
	provider := NewCloudProvider(client, nil, []string{"17", "23"})
	deployment, err := provider.Deploy(t.Context(), cloudcore.DeployRequest{
		OfferingID: "gpu-mi300x1-192gb", Location: "tor1", Name: "nnctl-mi300x",
	})
	if err != nil {
		t.Fatal(err)
	}
	if deployment.Instance == nil || deployment.Instance.ID != "42" || deployment.Instance.PublicIP != "192.0.2.42" {
		t.Fatalf("deployment = %#v", deployment)
	}
	if deployment.Instance.Backends[1] != "rocm" || deployment.Instance.Currency != CurrencyUSD {
		t.Fatalf("instance metadata = %#v", deployment.Instance)
	}
	if client.createRequest.Image != AMDImage || client.createRequest.Region != "tor1" {
		t.Fatalf("create request = %#v", client.createRequest)
	}
	if !slices.Equal(client.createRequest.SSHKeys, []int{17, 23}) || !strings.Contains(client.createRequest.UserData, "ZIG_VERSION=\"0.16.0\"") {
		t.Fatalf("create authentication/bootstrap = %#v", client.createRequest)
	}
	if len(client.createRequest.Tags) != 0 {
		t.Fatalf("create request unexpectedly requires tag:create scope: %#v", client.createRequest.Tags)
	}

	result, err := provider.Destroy(t.Context(), cloudcore.DestroyRequest{InstanceIDs: []string{"42"}, Permanent: true})
	if err != nil {
		t.Fatal(err)
	}
	if result.Provider != ProviderName || strings.Join(client.deleted, ",") != "42" {
		t.Fatalf("destroy result = %#v, deleted = %#v", result, client.deleted)
	}
}

func TestCloudProviderDryRunDoesNotRequireCredentials(t *testing.T) {
	provider := NewCloudProvider(nil, nil, nil)
	deployment, err := provider.Deploy(t.Context(), cloudcore.DeployRequest{
		OfferingID: "gpu-h200x1-141gb", Location: "nyc3", DryRun: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !deployment.DryRun || deployment.Request.Image != NVIDIAX1Image || deployment.Request.Market != MarketOnDemand {
		t.Fatalf("dry-run deployment = %#v", deployment)
	}
}

func TestCloudProviderCleansUpDropletWhenCreateResponseCannotBeMapped(t *testing.T) {
	client := &fakeClient{created: Droplet{ID: 42, SizeSlug: "unknown-gpu-size"}}
	provider := NewCloudProvider(client, nil, []string{"17"})
	_, err := provider.Deploy(t.Context(), cloudcore.DeployRequest{
		OfferingID: "gpu-mi300x1-192gb", Location: "tor1",
	})
	if err == nil || !strings.Contains(err.Error(), "map DigitalOcean Droplet 42") {
		t.Fatalf("Deploy() error = %v", err)
	}
	if strings.Join(client.deleted, ",") != "42" {
		t.Fatalf("cleanup deleted = %#v", client.deleted)
	}
}

func TestCloudProviderRejectsNonNumericSSHKeyID(t *testing.T) {
	provider := NewCloudProvider(nil, nil, nil)
	_, err := provider.Deploy(t.Context(), cloudcore.DeployRequest{
		OfferingID: "gpu-mi300x1-192gb", Location: "tor1",
		SSHKeyIDs: []string{"fingerprint"}, DryRun: true,
	})
	if err == nil || !strings.Contains(err.Error(), "positive integer") {
		t.Fatalf("Deploy() error = %v", err)
	}
}

func TestCloudProviderRejectsUnsupportedProviderOptions(t *testing.T) {
	provider := NewCloudProvider(nil, nil, nil)
	_, err := provider.Deploy(t.Context(), cloudcore.DeployRequest{
		OfferingID: "gpu-mi300x1-192gb", Location: "tor1", DryRun: true,
		Options: map[string]string{"source-os-volume-id": "volume-1"},
	})
	if err == nil || !strings.Contains(err.Error(), `does not support provider option "source-os-volume-id"`) {
		t.Fatalf("Deploy() error = %v", err)
	}
}

func TestCloudProviderRejectsNonPermanentOrAuxiliaryDestroy(t *testing.T) {
	provider := NewCloudProvider(nil, nil, nil)
	if _, err := provider.Destroy(t.Context(), cloudcore.DestroyRequest{
		InstanceIDs: []string{"42"}, DryRun: true,
	}); err == nil || !strings.Contains(err.Error(), "permanent must be true") {
		t.Fatalf("non-permanent Destroy() error = %v", err)
	}
	if _, err := provider.Destroy(t.Context(), cloudcore.DestroyRequest{
		InstanceIDs: []string{"42"}, Permanent: true, DryRun: true,
		Resources: []cloudcore.ResourceRef{{Kind: "volume", ID: "volume-1"}},
	}); err == nil || !strings.Contains(err.Error(), "auxiliary resource") {
		t.Fatalf("resource Destroy() error = %v", err)
	}
}

func TestCloudProviderListsOnlyGPUDroplets(t *testing.T) {
	provider := NewCloudProvider(&fakeClient{droplets: []Droplet{
		{ID: 1, Name: "cpu", Status: "active", SizeSlug: "s-1vcpu-1gb"},
		{ID: 2, Name: "gpu", Status: "off", SizeSlug: "gpu-h100x1-80gb", Region: Region{Slug: "nyc3"}},
	}}, nil, nil)
	instances, err := provider.Instances(t.Context(), cloudcore.ListOptions{All: true})
	if err != nil {
		t.Fatal(err)
	}
	if len(instances) != 1 || instances[0].ID != "2" || instances[0].State != cloudcore.InstanceStopped {
		t.Fatalf("instances = %#v", instances)
	}
}

type fakeCredentialStore struct {
	token string
	err   error
}

func (f fakeCredentialStore) Token(context.Context) (string, error) {
	return f.token, f.err
}

type fakeClient struct {
	sizes         []Size
	created       Droplet
	droplets      []Droplet
	sshKeys       []SSHKey
	createRequest CreateDropletRequest
	deleted       []string
}

func (f *fakeClient) ListSizes(context.Context) ([]Size, error) {
	return append([]Size(nil), f.sizes...), nil
}

func (f *fakeClient) CreateDroplet(_ context.Context, request CreateDropletRequest) (Droplet, error) {
	f.createRequest = request
	return f.created, nil
}

func (f *fakeClient) GetDroplet(_ context.Context, id string) (Droplet, error) {
	for _, droplet := range f.droplets {
		if id == strconv.Itoa(droplet.ID) {
			return droplet, nil
		}
	}
	return Droplet{}, errors.New("not found")
}

func (f *fakeClient) ListDroplets(context.Context) ([]Droplet, error) {
	return append([]Droplet(nil), f.droplets...), nil
}

func (f *fakeClient) DeleteDroplet(_ context.Context, id string) error {
	f.deleted = append(f.deleted, id)
	return nil
}

func (f *fakeClient) ListSSHKeys(context.Context) ([]SSHKey, error) {
	return append([]SSHKey(nil), f.sshKeys...), nil
}

var _ Client = (*fakeClient)(nil)
