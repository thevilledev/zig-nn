package lab

import (
	"context"
	"errors"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	cloudcore "nnctl/internal/cloud"
	"nnctl/internal/cloud/digitalocean"
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

func TestCloudManagerRunsAndRecoversDigitalOceanROCmWorker(t *testing.T) {
	repoRoot, err := filepath.Abs("../../..")
	if err != nil {
		t.Fatal(err)
	}
	transport := writeFakeCloudTransport(t)
	client := &fakeDigitalOceanLabClient{}
	database := filepath.Join(t.TempDir(), "state.sqlite")
	factory := digitalocean.ProviderFactory{
		CredentialStore: fakeDigitalOceanLabTokenStore{},
		NewClient: func(token string, _ digitalocean.ClientOptions) (digitalocean.Client, error) {
			if token != "test-token" {
				t.Fatalf("DigitalOcean token = %q", token)
			}
			return client, nil
		},
	}
	newManager := func() *CloudManager {
		manager, managerErr := NewCloudManager(CloudManagerOptions{
			RepoRoot: repoRoot, SSH: transport.ssh, Rsync: transport.rsync,
			DatabasePath: database, Timeout: cloudTestTimeout, PollInterval: time.Millisecond,
			Providers: []CloudProviderConfig{{
				Factory: factory,
				Configuration: cloudcore.Configuration{Options: map[string]string{
					cloudcore.OptionSSHKeyIDs: "17",
				}},
			}},
		})
		if managerErr != nil {
			t.Fatal(managerErr)
		}
		return manager
	}

	manager := newManager()
	_, err = manager.Deploy(t.Context(), CloudDeployRequest{
		InstanceType: "gpu-mi325x1-256gb-contracted", LocationCode: "nyc3",
	})
	if err == nil || !strings.Contains(err.Error(), "is not enabled and available") {
		t.Fatalf("unconfigured contract deploy error = %v", err)
	}
	if client.createCount() != 0 {
		t.Fatal("rejected contract offering created a Droplet")
	}

	created, err := manager.Deploy(t.Context(), CloudDeployRequest{
		InstanceType: "gpu-mi300x1-192gb", LocationCode: "tor1",
	})
	if err != nil {
		t.Fatal(err)
	}
	worker := waitForCloudWorkerState(t, manager, created.ID, CloudWorkerReady)
	if worker.Provider != digitalocean.ProviderName || worker.Market != digitalocean.MarketOnDemand ||
		!slices.Contains(worker.Backends, "rocm") {
		t.Fatalf("DigitalOcean worker = %#v", worker)
	}
	request := client.lastCreateRequest()
	if !slices.Equal(request.SSHKeys, []int{17}) || request.Image != digitalocean.AMDImage ||
		!strings.Contains(request.UserData, `ZIG_VERSION="0.16.0"`) {
		t.Fatalf("DigitalOcean create request = %#v", request)
	}

	spec, _ := ResolveExperiment("optimizer-lab")
	runOptions, err := BuildRunOptions(spec, "rocm", nil)
	if err != nil {
		t.Fatal(err)
	}
	runOptions.Target = RunTargetCloud
	runOptions.WorkerID = worker.ID
	runOptions.AcknowledgeDirty = true
	var output [][]byte
	if err := manager.Execute(t.Context(), spec, runOptions, func(line []byte) error {
		output = append(output, append([]byte(nil), line...))
		return nil
	}, func(string) {}); err != nil {
		t.Fatal(err)
	}
	if len(output) != 2 || !strings.Contains(string(output[0]), `"selected_backend":"rocm"`) {
		t.Fatalf("ROCm remote output = %q", output)
	}
	waitForCloudWorkerState(t, manager, worker.ID, CloudWorkerReady)
	if err := manager.Close(); err != nil {
		t.Fatal(err)
	}

	recovered := newManager()
	recoveredWorker := waitForCloudWorkerState(t, recovered, worker.ID, CloudWorkerReady)
	if recoveredWorker.Provider != digitalocean.ProviderName || !slices.Contains(recoveredWorker.Backends, "rocm") {
		t.Fatalf("recovered DigitalOcean worker = %#v", recoveredWorker)
	}
	if err := recovered.Destroy(t.Context(), recoveredWorker.ID); err != nil {
		t.Fatal(err)
	}
	if err := recovered.Close(); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(client.deletedIDs(), []string{"42"}) {
		t.Fatalf("deleted DigitalOcean Droplets = %#v", client.deletedIDs())
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

type fakeDigitalOceanLabTokenStore struct{}

func (fakeDigitalOceanLabTokenStore) Token(context.Context) (string, error) {
	return "test-token", nil
}

type fakeDigitalOceanLabClient struct {
	mu             sync.Mutex
	createRequests []digitalocean.CreateDropletRequest
	deleted        []string
}

func (*fakeDigitalOceanLabClient) ListSizes(context.Context) ([]digitalocean.Size, error) {
	return []digitalocean.Size{{
		Slug: "gpu-mi300x1-192gb", PriceHourly: 1.99, Available: true,
		Regions: []string{"tor1"},
		GPUInfo: &digitalocean.GPUInfo{
			Count: 1, Model: "amd_mi300x",
			VRAM: digitalocean.UnitValue{Amount: 192, Unit: "gib"},
		},
	}}, nil
}

func (c *fakeDigitalOceanLabClient) CreateDroplet(_ context.Context, request digitalocean.CreateDropletRequest) (digitalocean.Droplet, error) {
	c.mu.Lock()
	c.createRequests = append(c.createRequests, request)
	c.mu.Unlock()
	return digitalocean.Droplet{
		ID: 42, Name: request.Name, Status: "new", SizeSlug: request.Size,
		Region: digitalocean.Region{Slug: request.Region},
	}, nil
}

func (*fakeDigitalOceanLabClient) GetDroplet(context.Context, string) (digitalocean.Droplet, error) {
	return digitalocean.Droplet{
		ID: 42, Name: "nnctl-lab", Status: "active", SizeSlug: "gpu-mi300x1-192gb",
		Region: digitalocean.Region{Slug: "tor1"},
		Networks: digitalocean.Networks{V4: []digitalocean.Network{{
			IPAddress: "192.0.2.42", Type: "public",
		}}},
	}, nil
}

func (*fakeDigitalOceanLabClient) ListDroplets(context.Context) ([]digitalocean.Droplet, error) {
	return nil, nil
}

func (c *fakeDigitalOceanLabClient) DeleteDroplet(_ context.Context, id string) error {
	c.mu.Lock()
	c.deleted = append(c.deleted, id)
	c.mu.Unlock()
	return nil
}

func (*fakeDigitalOceanLabClient) ListSSHKeys(context.Context) ([]digitalocean.SSHKey, error) {
	return nil, nil
}

func (c *fakeDigitalOceanLabClient) createCount() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.createRequests)
}

func (c *fakeDigitalOceanLabClient) lastCreateRequest() digitalocean.CreateDropletRequest {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.createRequests[len(c.createRequests)-1]
}

func (c *fakeDigitalOceanLabClient) deletedIDs() []string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]string(nil), c.deleted...)
}

var _ digitalocean.CredentialStore = fakeDigitalOceanLabTokenStore{}
var _ digitalocean.Client = (*fakeDigitalOceanLabClient)(nil)
