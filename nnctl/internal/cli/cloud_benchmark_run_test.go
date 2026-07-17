package cli

import (
	"context"
	"errors"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"nnctl/internal/cloud/verda"
)

func TestCloudBenchmarkRunPrepareUsesInjectedDependencies(t *testing.T) {
	stamp := time.Date(2026, time.July, 15, 12, 30, 0, 0, time.FixedZone("EEST", 3*60*60))
	var progress strings.Builder
	a := &app{
		repoRoot:     t.TempDir(),
		stderrWriter: &progress,
		now:          func() time.Time { return stamp },
	}
	opts := defaultCloudBenchmarkDeployOptions()
	opts.ref = "main"
	opts.authorizedKeysFile = "keys/authorized_keys"
	run := newCloudBenchmarkRun(a, context.Background(), opts)
	run.ops.gitStatus = func(context.Context, string) (string, error) {
		return "", nil
	}
	run.ops.gitRevision = func(_ context.Context, git, ref string) (string, error) {
		if git != "git" || ref != "main" {
			t.Fatalf("gitRevision(%q, %q), want git and main", git, ref)
		}
		return "abc123", nil
	}

	if err := run.prepare(); err != nil {
		t.Fatalf("prepare() error = %v", err)
	}
	if run.commit != "abc123" {
		t.Fatalf("commit = %q, want abc123", run.commit)
	}
	if !run.stamp.Equal(stamp.UTC()) {
		t.Fatalf("stamp = %s, want %s", run.stamp, stamp.UTC())
	}
	wantKeys := filepath.Join(a.repoRoot, "keys/authorized_keys")
	if run.opts.authorizedKeysFile != wantKeys {
		t.Fatalf("authorizedKeysFile = %q, want %q", run.opts.authorizedKeysFile, wantKeys)
	}
	if !strings.Contains(progress.String(), "Revision: abc123") {
		t.Fatalf("progress = %q, want resolved revision", progress.String())
	}
}

func TestCloudBenchmarkRunDeployRejectsEmptyResult(t *testing.T) {
	var progress strings.Builder
	run := &cloudBenchmarkRun{
		ctx:      context.Background(),
		opts:     defaultCloudBenchmarkDeployOptions(),
		progress: newCloudBenchmarkProgress(&progress),
	}
	run.ops.deploy = func(context.Context, verda.Client, verda.DeployOptions) (*verda.DeployResult, error) {
		return nil, nil
	}

	err := run.deployWorker()
	if err == nil || !strings.Contains(err.Error(), "did not return an instance") {
		t.Fatalf("deployWorker() error = %v, want missing instance error", err)
	}
}

func TestCloudBenchmarkRunCleanupReturnsProviderFailure(t *testing.T) {
	wantErr := errors.New("destroy worker")
	var progress strings.Builder
	run := &cloudBenchmarkRun{
		ctx:        context.Background(),
		opts:       cloudBenchmarkDeployOptions{},
		progress:   newCloudBenchmarkProgress(&progress),
		instanceID: "instance-1",
		cloneID:    "volume-1",
	}
	run.sourceVolume.ID = "source-1"
	run.ops.cleanup = func(ctx context.Context, _ cloudBenchmarkClient, instanceID, cloneID, sourceID string) error {
		if _, ok := ctx.Deadline(); !ok {
			t.Fatal("cleanup context has no deadline")
		}
		if instanceID != run.instanceID || cloneID != run.cloneID || sourceID != run.sourceVolume.ID {
			t.Fatalf("cleanup IDs = %q, %q, %q", instanceID, cloneID, sourceID)
		}
		return wantErr
	}

	err := run.cleanupResources()
	if !errors.Is(err, wantErr) {
		t.Fatalf("cleanupResources() error = %v, want %v", err, wantErr)
	}
	if !strings.Contains(progress.String(), "Cleanup failed: destroy worker") {
		t.Fatalf("progress = %q, want cleanup failure", progress.String())
	}
}

func TestCloudBenchmarkRunCleanupHonorsKeepInstance(t *testing.T) {
	called := false
	var progress strings.Builder
	run := &cloudBenchmarkRun{
		ctx:        context.Background(),
		opts:       cloudBenchmarkDeployOptions{keepInstance: true},
		progress:   newCloudBenchmarkProgress(&progress),
		instanceID: "instance-1",
		cloneID:    "volume-1",
	}
	run.ops.cleanup = func(context.Context, cloudBenchmarkClient, string, string, string) error {
		called = true
		return nil
	}

	if err := run.cleanupResources(); err != nil {
		t.Fatalf("cleanupResources() error = %v", err)
	}
	if called {
		t.Fatal("cleanup provider called with keepInstance set")
	}
	if !strings.Contains(progress.String(), "--keep-instance") {
		t.Fatalf("progress = %q, want skip reason", progress.String())
	}
}
