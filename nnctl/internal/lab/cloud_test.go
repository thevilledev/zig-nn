package lab

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"nnctl/internal/cloud/verda"
	cloudworkflow "nnctl/internal/cloud/workflow"
)

func TestNewCloudManagerRejectsUnsafeSSHUser(t *testing.T) {
	t.Parallel()
	_, err := NewCloudManager(CloudManagerOptions{
		RepoRoot: t.TempDir(),
		SSHUser:  "-oProxyCommand=evil",
	})
	if err == nil || !strings.Contains(err.Error(), "SSH user") {
		t.Fatalf("NewCloudManager() error = %v", err)
	}
}

const cloudTestTimeout = 10 * time.Second

func TestCloudManagerDeployExecuteDestroyAndRecover(t *testing.T) {
	repoRoot, err := filepath.Abs("../../..")
	if err != nil {
		t.Fatal(err)
	}
	transport := writeFakeCloudTransport(t)
	client := newFakeCloudClient()
	database := filepath.Join(t.TempDir(), "state.sqlite")
	manager, err := NewCloudManager(CloudManagerOptions{
		RepoRoot:      repoRoot,
		SSH:           transport.ssh,
		Rsync:         transport.rsync,
		DatabasePath:  database,
		Timeout:       cloudTestTimeout,
		PollInterval:  time.Millisecond,
		ClientFactory: func(context.Context, string) (cloudworkflow.Client, error) { return client, nil },
	})
	if err != nil {
		t.Fatal(err)
	}

	status := manager.Status(context.Background())
	if !status.Enabled || !status.Configured || status.Repository.Revision == "" {
		t.Fatalf("cloud status = %#v", status)
	}
	options, err := manager.Options(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(options.Prices) != 1 || options.SourceOSVolumeName != defaultCloudSourceVolumeName {
		t.Fatalf("cloud options = %#v", options)
	}

	autoDestroy := false
	created, err := manager.Deploy(context.Background(), CloudDeployRequest{
		InstanceType: "1A100.22V",
		Market:       verda.PricingMarketSpot,
		LocationCode: "FIN-02",
		AutoDestroy:  &autoDestroy,
	})
	if err != nil {
		t.Fatal(err)
	}
	worker := waitForCloudWorkerState(t, manager, created.ID, CloudWorkerReady)
	if worker.IP != "192.0.2.10" || !strings.Contains(worker.Message, "Ready") {
		t.Fatalf("ready worker = %#v", worker)
	}
	client.mu.Lock()
	createRequest := client.createRequest
	cloneRequest := client.cloneRequest
	cloneCalls := client.cloneCalls
	client.mu.Unlock()
	if createRequest.Image != "clone-1" || createRequest.StartupScriptID != "" {
		t.Fatalf("golden-volume create request = %#v", createRequest)
	}
	if len(createRequest.SSHKeyIDs) != 0 {
		t.Fatalf("golden-volume create request attached SSH keys: %#v", createRequest.SSHKeyIDs)
	}
	if cloneCalls != 1 {
		t.Fatalf("golden-volume deployment cloned %d OS volumes", cloneCalls)
	}
	if cloneRequest.SourceVolumeID != "source-1" || cloneRequest.LocationCode != "FIN-02" {
		t.Fatalf("golden-volume clone request = %#v", cloneRequest)
	}

	spec, _ := ResolveExperiment("optimizer-lab")
	runOptions, err := BuildRunOptions(spec, "cuda", nil)
	if err != nil {
		t.Fatal(err)
	}
	runOptions.Target = RunTargetCloud
	runOptions.WorkerID = worker.ID
	runOptions.AcknowledgeDirty = true
	var stdout [][]byte
	var diagnostics []string
	if err := manager.Execute(context.Background(), spec, runOptions, func(line []byte) error {
		stdout = append(stdout, append([]byte(nil), line...))
		return nil
	}, func(line string) {
		diagnostics = append(diagnostics, line)
	}); err != nil {
		t.Fatal(err)
	}
	if len(stdout) != 2 || !strings.Contains(string(stdout[0]), `"type":"run_started"`) {
		t.Fatalf("remote stdout = %q", stdout)
	}
	if len(diagnostics) == 0 || !strings.Contains(strings.Join(diagnostics, "\n"), "uploading committed revision") {
		t.Fatalf("remote diagnostics = %#v", diagnostics)
	}
	worker = waitForCloudWorkerState(t, manager, created.ID, CloudWorkerReady)
	if worker.RepositoryCommit == "" {
		t.Fatalf("worker did not record repository revision: %#v", worker)
	}

	if err := manager.Close(); err != nil {
		t.Fatal(err)
	}
	recovered, err := NewCloudManager(CloudManagerOptions{
		RepoRoot:      repoRoot,
		DatabasePath:  database,
		SSH:           transport.ssh,
		Rsync:         transport.rsync,
		Timeout:       cloudTestTimeout,
		PollInterval:  time.Millisecond,
		ClientFactory: func(context.Context, string) (cloudworkflow.Client, error) { return client, nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	recoveredWorker := waitForCloudWorkerState(t, recovered, created.ID, CloudWorkerReady)
	if !strings.Contains(recoveredWorker.Message, "Reconnected") {
		t.Fatalf("recovered worker = %#v", recoveredWorker)
	}
	runOptions.WorkerID = recoveredWorker.ID
	if err := recovered.Execute(context.Background(), spec, runOptions, func([]byte) error { return nil }, func(string) {}); err != nil {
		t.Fatal(err)
	}
	waitForCloudWorkerState(t, recovered, recoveredWorker.ID, CloudWorkerReady)
	if err := recovered.Destroy(context.Background(), recoveredWorker.ID); err != nil {
		t.Fatal(err)
	}
	persisted, err := recovered.store.Load(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if persisted != nil {
		t.Fatalf("destroyed worker remained in SQLite: %#v", persisted)
	}
	if err := recovered.Close(); err != nil {
		t.Fatal(err)
	}
	client.mu.Lock()
	destroyed := append([]verda.DestroyInstanceRequest(nil), client.destroyed...)
	client.mu.Unlock()
	if len(destroyed) != 1 || destroyed[0].InstanceIDs[0] != "instance-1" || len(destroyed[0].VolumeIDs) != 1 || destroyed[0].VolumeIDs[0] != "clone-1" {
		t.Fatalf("destroy requests = %#v", destroyed)
	}
}

func TestCloudManagerDefaultsToReusableWorker(t *testing.T) {
	repoRoot, _ := filepath.Abs("../../..")
	transport := writeFakeCloudTransport(t)
	client := newFakeCloudClient()
	manager, err := NewCloudManager(CloudManagerOptions{
		RepoRoot:      repoRoot,
		SSH:           transport.ssh,
		Rsync:         transport.rsync,
		DatabasePath:  filepath.Join(t.TempDir(), "state.sqlite"),
		Timeout:       cloudTestTimeout,
		PollInterval:  time.Millisecond,
		ClientFactory: func(context.Context, string) (cloudworkflow.Client, error) { return client, nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	created, err := manager.Deploy(context.Background(), CloudDeployRequest{
		InstanceType: "1A100.22V",
		Market:       "spot",
		LocationCode: "FIN-02",
	})
	if err != nil {
		t.Fatal(err)
	}
	worker := waitForCloudWorkerState(t, manager, created.ID, CloudWorkerReady)
	if worker.AutoDestroy {
		t.Fatal("worker did not default to manual cleanup")
	}
	spec, _ := ResolveExperiment("optimizer-lab")
	options, _ := BuildRunOptions(spec, "cuda", nil)
	options.Target = RunTargetCloud
	options.WorkerID = worker.ID
	options.AcknowledgeDirty = true
	for range 2 {
		if err := manager.Execute(context.Background(), spec, options, func([]byte) error { return nil }, func(string) {}); err != nil {
			t.Fatal(err)
		}
		waitForCloudWorkerState(t, manager, created.ID, CloudWorkerReady)
	}
	client.mu.Lock()
	destroyed := len(client.destroyed)
	client.mu.Unlock()
	if destroyed != 0 {
		t.Fatalf("reusable worker was destroyed after a run")
	}
	if err := manager.Destroy(context.Background(), created.ID); err != nil {
		t.Fatal(err)
	}
}

func TestCloudManagerDoesNotPersistFailedWorkerWithoutResources(t *testing.T) {
	repoRoot, _ := filepath.Abs("../../..")
	database := filepath.Join(t.TempDir(), "state.sqlite")
	client := newFakeCloudClient()
	client.cloneErr = errors.New("clone failed")
	manager, err := NewCloudManager(CloudManagerOptions{
		RepoRoot:      repoRoot,
		DatabasePath:  database,
		Timeout:       cloudTestTimeout,
		PollInterval:  time.Millisecond,
		ClientFactory: func(context.Context, string) (cloudworkflow.Client, error) { return client, nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	created, err := manager.Deploy(context.Background(), CloudDeployRequest{
		InstanceType: "1A100.22V",
		Market:       "spot",
		LocationCode: "FIN-02",
	})
	if err != nil {
		t.Fatal(err)
	}
	waitForCloudWorkerState(t, manager, created.ID, CloudWorkerFailed)
	if err := manager.Close(); err != nil {
		t.Fatal(err)
	}

	reopened, err := NewCloudManager(CloudManagerOptions{
		RepoRoot:      repoRoot,
		DatabasePath:  database,
		ClientFactory: func(context.Context, string) (cloudworkflow.Client, error) { return client, nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = reopened.Close() }()
	if workers := reopened.Workers(); len(workers) != 0 {
		t.Fatalf("resource-free failure was persisted: %#v", workers)
	}
}

func TestCloudManagerDestroysIdleAutomaticWorker(t *testing.T) {
	repoRoot, _ := filepath.Abs("../../..")
	transport := writeFakeCloudTransport(t)
	client := newFakeCloudClient()
	manager, err := NewCloudManager(CloudManagerOptions{
		RepoRoot:      repoRoot,
		SSH:           transport.ssh,
		DatabasePath:  filepath.Join(t.TempDir(), "state.sqlite"),
		Timeout:       cloudTestTimeout,
		PollInterval:  time.Millisecond,
		IdleTimeout:   150 * time.Millisecond,
		ClientFactory: func(context.Context, string) (cloudworkflow.Client, error) { return client, nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	autoDestroy := true
	created, err := manager.Deploy(context.Background(), CloudDeployRequest{
		InstanceType: "1A100.22V",
		Market:       "spot",
		LocationCode: "FIN-02",
		AutoDestroy:  &autoDestroy,
	})
	if err != nil {
		t.Fatal(err)
	}
	waitForCloudWorkerState(t, manager, created.ID, CloudWorkerReady)
	waitForCloudWorkerState(t, manager, created.ID, CloudWorkerDestroyed)
}

func TestCloudManagerCancelsProvisioningBeforeDestroy(t *testing.T) {
	repoRoot, _ := filepath.Abs("../../..")
	transport := writeFakeCloudTransport(t)
	client := newFakeCloudClient()
	client.listInstancesBlock = make(chan struct{})
	manager, err := NewCloudManager(CloudManagerOptions{
		RepoRoot:      repoRoot,
		SSH:           transport.ssh,
		DatabasePath:  filepath.Join(t.TempDir(), "state.sqlite"),
		Timeout:       cloudTestTimeout,
		PollInterval:  time.Millisecond,
		ClientFactory: func(context.Context, string) (cloudworkflow.Client, error) { return client, nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	autoDestroy := true
	created, err := manager.Deploy(context.Background(), CloudDeployRequest{
		InstanceType: "1A100.22V",
		Market:       "spot",
		LocationCode: "FIN-02",
		AutoDestroy:  &autoDestroy,
	})
	if err != nil {
		t.Fatal(err)
	}
	waitForCloudWorkerInstance(t, manager, created.ID)
	if err := manager.Destroy(context.Background(), created.ID); err != nil {
		t.Fatal(err)
	}
	waitForCloudWorkerState(t, manager, created.ID, CloudWorkerDestroyed)
	client.mu.Lock()
	destroyed := append([]verda.DestroyInstanceRequest(nil), client.destroyed...)
	client.mu.Unlock()
	if len(destroyed) != 1 || destroyed[0].InstanceIDs[0] != "instance-1" {
		t.Fatalf("destroy requests = %#v", destroyed)
	}
}

func TestCloudManagerClosesAutomaticWorker(t *testing.T) {
	repoRoot, _ := filepath.Abs("../../..")
	transport := writeFakeCloudTransport(t)
	client := newFakeCloudClient()
	manager, err := NewCloudManager(CloudManagerOptions{
		RepoRoot:      repoRoot,
		SSH:           transport.ssh,
		DatabasePath:  filepath.Join(t.TempDir(), "state.sqlite"),
		Timeout:       cloudTestTimeout,
		PollInterval:  time.Millisecond,
		ClientFactory: func(context.Context, string) (cloudworkflow.Client, error) { return client, nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	autoDestroy := true
	created, err := manager.Deploy(context.Background(), CloudDeployRequest{
		InstanceType: "1A100.22V",
		Market:       "spot",
		LocationCode: "FIN-02",
		AutoDestroy:  &autoDestroy,
	})
	if err != nil {
		t.Fatal(err)
	}
	waitForCloudWorkerState(t, manager, created.ID, CloudWorkerReady)
	if err := manager.Close(); err != nil {
		t.Fatal(err)
	}
	waitForCloudWorkerState(t, manager, created.ID, CloudWorkerDestroyed)
}

func waitForCloudWorkerState(t *testing.T, manager *CloudManager, id string, state CloudWorkerState) CloudWorker {
	t.Helper()
	deadline := time.Now().Add(cloudTestTimeout)
	for time.Now().Before(deadline) {
		workers := manager.Workers()
		if len(workers) == 1 && workers[0].ID == id && workers[0].State == state {
			return workers[0]
		}
		if len(workers) == 1 && workers[0].State == CloudWorkerFailed && state != CloudWorkerFailed {
			t.Fatalf("worker failed while waiting for %s: %#v", state, workers[0])
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatalf("timed out waiting for worker %s state %s: %#v", id, state, manager.Workers())
	return CloudWorker{}
}

func waitForCloudWorkerInstance(t *testing.T, manager *CloudManager, id string) CloudWorker {
	t.Helper()
	deadline := time.Now().Add(cloudTestTimeout)
	for time.Now().Before(deadline) {
		workers := manager.Workers()
		if len(workers) == 1 && workers[0].ID == id && workers[0].InstanceID != "" {
			return workers[0]
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatalf("timed out waiting for worker %s instance: %#v", id, manager.Workers())
	return CloudWorker{}
}

type fakeCloudTransport struct {
	ssh   string
	rsync string
}

func writeFakeCloudTransport(t *testing.T) fakeCloudTransport {
	t.Helper()
	directory := t.TempDir()
	ssh := filepath.Join(directory, "ssh")
	sshScript := `#!/bin/sh
case "$*" in
  *run_optimizer_lab*)
    printf '%s\n' '{"v":1,"type":"run_started","experiment":"optimizer-lab","data":{"config":{},"execution":{"requested_backend":"cuda","selected_backend":"cuda","optimize":"Debug"}}}'
    printf '%s\n' '{"v":1,"type":"run_completed","experiment":"optimizer-lab","step":1,"total_steps":1,"data":{"final_loss":0.1}}'
    ;;
esac
exit 0
`
	if err := os.WriteFile(ssh, []byte(sshScript), 0o700); err != nil {
		t.Fatal(err)
	}
	rsync := filepath.Join(directory, "rsync")
	if err := os.WriteFile(rsync, []byte("#!/bin/sh\nexit 0\n"), 0o700); err != nil {
		t.Fatal(err)
	}
	return fakeCloudTransport{ssh: ssh, rsync: rsync}
}

type fakeCloudClient struct {
	mu                 sync.Mutex
	instance           verda.Instance
	createRequest      verda.CreateInstanceRequest
	cloneRequest       verda.CloneVolumeRequest
	cloneCalls         int
	cloneErr           error
	destroyed          []verda.DestroyInstanceRequest
	listInstancesBlock <-chan struct{}
}

func newFakeCloudClient() *fakeCloudClient {
	return &fakeCloudClient{}
}

func (c *fakeCloudClient) GetVolume(context.Context, string) (verda.Volume, error) {
	return verda.Volume{ID: "source-1", Name: defaultCloudSourceVolumeName, Status: "detached", Location: "FIN-02", IsOSVolume: true}, nil
}

func (c *fakeCloudClient) ListVolumes(context.Context) ([]verda.Volume, error) {
	return []verda.Volume{
		{ID: "source-1", Name: defaultCloudSourceVolumeName, Status: "detached", Location: "FIN-02", IsOSVolume: true},
		{ID: "source-2", Name: defaultCloudSourceVolumeName, Status: "creating", Location: "FIN-03", IsOSVolume: true},
		{ID: "data-1", Name: "data", Status: "detached", Location: "FIN-02"},
	}, nil
}

func (c *fakeCloudClient) ListDeletedVolumes(context.Context) ([]verda.Volume, error) {
	return nil, nil
}

func (c *fakeCloudClient) GetInstanceType(context.Context, string, bool, string) (verda.InstanceType, error) {
	return verda.InstanceType{InstanceType: "1A100.22V", Model: "A100", GPUCount: 1, SpotPrice: 1.25, PriceKnown: true, Currency: "EUR"}, nil
}

func (c *fakeCloudClient) GetPlacementOptions(context.Context, string, bool) ([]verda.SpotPlacement, error) {
	return []verda.SpotPlacement{{LocationCode: "FIN-02", Market: "spot", SpotPrice: 1.25, PriceKnown: true, Currency: "EUR"}}, nil
}

func (c *fakeCloudClient) CreateStartupScript(context.Context, string, string) (verda.StartupScript, error) {
	return verda.StartupScript{ID: "script-1"}, nil
}

func (c *fakeCloudClient) CloneVolume(_ context.Context, request verda.CloneVolumeRequest) (verda.Volume, error) {
	c.mu.Lock()
	c.cloneRequest = request
	c.cloneCalls++
	err := c.cloneErr
	c.mu.Unlock()
	if err != nil {
		return verda.Volume{}, err
	}
	return verda.Volume{ID: "clone-1", Name: request.Name, Status: "cloning", Location: request.LocationCode}, nil
}

func (c *fakeCloudClient) WaitVolumeReady(context.Context, string) (verda.Volume, error) {
	return verda.Volume{ID: "clone-1", Status: "detached", Location: "FIN-02"}, nil
}

func (c *fakeCloudClient) DeleteVolume(context.Context, string, bool) error { return nil }

func (c *fakeCloudClient) CreateInstance(_ context.Context, request verda.CreateInstanceRequest) (verda.Instance, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.createRequest = request
	ip := "192.0.2.10"
	c.instance = verda.Instance{
		ID: "instance-1", Hostname: request.Hostname, Status: "running", IP: &ip,
		InstanceType: request.InstanceType, Location: request.LocationCode, IsSpot: request.IsSpot,
		PricePerHour: 1.25, Currency: "EUR",
	}
	return c.instance, nil
}

func (c *fakeCloudClient) DestroyInstances(_ context.Context, request verda.DestroyInstanceRequest) ([]verda.DestroyInstanceResult, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.destroyed = append(c.destroyed, request)
	return []verda.DestroyInstanceResult{{InstanceID: request.InstanceIDs[0], Status: "success"}}, nil
}

func (c *fakeCloudClient) ListInstances(ctx context.Context, _ string) ([]verda.Instance, error) {
	c.mu.Lock()
	instance := c.instance
	block := c.listInstancesBlock
	c.mu.Unlock()
	if block != nil {
		select {
		case <-block:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	if instance.ID == "" {
		return nil, nil
	}
	return []verda.Instance{instance}, nil
}

func (c *fakeCloudClient) ListSSHKeys(context.Context) ([]verda.SSHKey, error) {
	return []verda.SSHKey{{
		ID: "00000000-0000-0000-0000-000000000001", Name: "lab", Fingerprint: "SHA256:test", PublicKey: "ssh-ed25519 private-browser-boundary-test",
	}}, nil
}

func (c *fakeCloudClient) ListInstancePrices(context.Context, verda.PricingFilters) ([]verda.InstancePrice, error) {
	return []verda.InstancePrice{
		{
			InstanceType: "1A100.22V", Model: "A100", Manufacturer: "NVIDIA", GPUCount: 1,
			LocationCode: "FIN-02", Market: "spot", IsSpot: true, PricePerHour: 1.25,
			PriceKnown: true, Currency: "EUR", Available: true,
		},
		{
			InstanceType: "1H100.80V", Model: "H100", Manufacturer: "NVIDIA", GPUCount: 1,
			LocationCode: "FIN-03", Market: "spot", IsSpot: true, PricePerHour: 2.5,
			PriceKnown: true, Currency: "EUR", Available: true,
		},
	}, nil
}
