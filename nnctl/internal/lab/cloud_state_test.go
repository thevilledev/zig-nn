package lab

import (
	"context"
	"errors"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
)

func TestCloudWorkerTransitionRejectsStaleState(t *testing.T) {
	manager := newCloudStateTestManager(CloudWorkerProvisioning)

	if _, err := manager.transitionWorker(
		"worker-1",
		[]CloudWorkerState{CloudWorkerProvisioning},
		CloudWorkerDestroying,
		"Destroying",
		nil,
	); err != nil {
		t.Fatal(err)
	}
	if _, err := manager.transitionWorker(
		"worker-1",
		[]CloudWorkerState{CloudWorkerProvisioning},
		CloudWorkerReady,
		"Ready",
		nil,
	); err == nil {
		t.Fatal("stale provisioning transition unexpectedly succeeded")
	}

	worker, ok := manager.managedWorker("worker-1")
	if !ok || worker.State != CloudWorkerDestroying || worker.Message != "Destroying" {
		t.Fatalf("worker after stale transition = %#v", worker)
	}
}

func TestCloudManagerBeginExecutionClaimsWorkerAtomically(t *testing.T) {
	manager := newCloudStateTestManager(CloudWorkerReady)
	const callers = 32
	start := make(chan struct{})
	var successes atomic.Int32
	var wait sync.WaitGroup
	wait.Add(callers)
	for range callers {
		go func() {
			defer wait.Done()
			<-start
			if _, err := manager.beginExecution("worker-1", "cuda"); err == nil {
				successes.Add(1)
			}
		}()
	}
	close(start)
	wait.Wait()

	if got := successes.Load(); got != 1 {
		t.Fatalf("successful execution claims = %d, want 1", got)
	}
	worker, ok := manager.managedWorker("worker-1")
	if !ok || worker.State != CloudWorkerBusy {
		t.Fatalf("claimed worker = %#v", worker)
	}
}

func TestNewCloudManagerHonorsCanceledParent(t *testing.T) {
	parent, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := NewCloudManager(CloudManagerOptions{
		Parent:       parent,
		RepoRoot:     t.TempDir(),
		DatabasePath: filepath.Join(t.TempDir(), "state.sqlite"),
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("NewCloudManager() error = %v, want context canceled", err)
	}
}

func TestCloudManagerCloseIsIdempotent(t *testing.T) {
	manager, err := NewCloudManager(CloudManagerOptions{
		Parent:       context.Background(),
		RepoRoot:     t.TempDir(),
		DatabasePath: filepath.Join(t.TempDir(), "state.sqlite"),
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := manager.Close(); err != nil {
		t.Fatal(err)
	}
	if err := manager.Close(); err != nil {
		t.Fatalf("second Close() error = %v", err)
	}
}

func newCloudStateTestManager(state CloudWorkerState) *CloudManager {
	return &CloudManager{
		worker: &managedCloudWorker{CloudWorker: CloudWorker{
			ID:       "worker-1",
			State:    state,
			Backends: []string{"cpu", "cuda"},
		}},
		workerSubscribers: make(map[chan CloudWorker]struct{}),
		ctx:               context.Background(),
	}
}
