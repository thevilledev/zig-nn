package cli

import (
	"reflect"
	"testing"
)

func TestDeployArchiveArgsUsesGitArchiveForRef(t *testing.T) {
	opts := defaultDeployOptions()
	opts.ref = "feature/fast-matmul"

	got := deployArchiveArgs(opts, "/tmp/repo.tar")
	want := []string{"archive", "--format=tar", "--output", "/tmp/repo.tar", "feature/fast-matmul"}

	if !reflect.DeepEqual(got, want) {
		t.Fatalf("deployArchiveArgs() = %#v, want %#v", got, want)
	}
}

func TestDeployRsyncArgsCopiesSnapshotContents(t *testing.T) {
	opts := defaultDeployOptions()
	opts.target = "gpu1:/srv/zig-nn"
	opts.delete = true
	opts.dryRun = true
	opts.ssh = "ssh -i ~/.ssh/bench"

	got := deployRsyncArgs(opts, "/tmp/nnctl-deploy/snapshot")
	want := []string{
		"-az",
		"--delete",
		"--dry-run",
		"--itemize-changes",
		"-e",
		"ssh -i ~/.ssh/bench",
		"/tmp/nnctl-deploy/snapshot/",
		"gpu1:/srv/zig-nn",
	}

	if !reflect.DeepEqual(got, want) {
		t.Fatalf("deployRsyncArgs() = %#v, want %#v", got, want)
	}
}

func TestDeployRsyncArgsDoesNotDoubleAppendSeparator(t *testing.T) {
	opts := defaultDeployOptions()
	opts.target = "gpu1:/srv/zig-nn"

	got := deployRsyncArgs(opts, "/tmp/nnctl-deploy/snapshot/")
	want := []string{"-az", "/tmp/nnctl-deploy/snapshot/", "gpu1:/srv/zig-nn"}

	if !reflect.DeepEqual(got, want) {
		t.Fatalf("deployRsyncArgs() = %#v, want %#v", got, want)
	}
}
