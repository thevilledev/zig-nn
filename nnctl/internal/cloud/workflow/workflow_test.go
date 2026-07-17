package workflow

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestUploadSnapshotBuildsSafeSharedCommands(t *testing.T) {
	t.Parallel()
	type call struct {
		dir  string
		name string
		args []string
	}
	var calls []call
	runner := func(_ context.Context, dir, name string, args ...string) error {
		calls = append(calls, call{dir: dir, name: name, args: append([]string(nil), args...)})
		return nil
	}
	repoRoot := t.TempDir()
	err := UploadSnapshot(t.Context(), UploadOptions{
		RepoRoot: repoRoot,
		Target:   "root@192.0.2.10:/srv/zig-nn",
		Ref:      "--option-like-ref",
		SSH:      "ssh -i key",
		Git:      "test-git",
		Tar:      "test-tar",
		Rsync:    "test-rsync",
		Delete:   true,
		DryRun:   true,
		Runner:   runner,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(calls) != 3 {
		t.Fatalf("commands = %#v, want git, tar, and rsync", calls)
	}
	archive := calls[0].args[3]
	if calls[0].dir != repoRoot || calls[0].name != "test-git" || !reflect.DeepEqual(calls[0].args, []string{
		"archive", "--format=tar", "--output", archive, "--", "--option-like-ref",
	}) {
		t.Fatalf("git command = %#v", calls[0])
	}
	snapshot := filepath.Join(filepath.Dir(archive), "snapshot")
	if calls[1].name != "test-tar" || !reflect.DeepEqual(calls[1].args, []string{"-xf", archive, "-C", snapshot}) {
		t.Fatalf("tar command = %#v", calls[1])
	}
	wantRsync := []string{
		"-az", "--delete", "--dry-run", "--itemize-changes", "-e", "ssh -i key", "--",
		withTrailingSeparator(snapshot), "root@192.0.2.10:/srv/zig-nn",
	}
	if calls[2].name != "test-rsync" || !reflect.DeepEqual(calls[2].args, wantRsync) {
		t.Fatalf("rsync command = %#v, want %#v", calls[2], wantRsync)
	}
	if _, err := os.Stat(filepath.Dir(archive)); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("upload work directory still exists: %v", err)
	}
}

func TestSSHDestination(t *testing.T) {
	t.Parallel()
	for _, test := range []struct {
		name    string
		user    string
		ip      string
		want    string
		wantErr string
	}{
		{name: "IPv4", user: "root", ip: "192.0.2.10", want: "root@192.0.2.10"},
		{name: "IPv6", user: "runner", ip: "2001:db8::10", want: "runner@2001:db8::10"},
		{name: "invalid user", user: "-oProxyCommand", ip: "192.0.2.10", wantErr: "SSH user"},
		{name: "host name", user: "root", ip: "worker.example", wantErr: "IP address"},
		{name: "option injection", user: "root", ip: "-oProxyCommand=evil", wantErr: "IP address"},
		{name: "loopback", user: "root", ip: "127.0.0.1", wantErr: "IP address"},
	} {
		t.Run(test.name, func(t *testing.T) {
			got, err := SSHDestination(test.user, test.ip)
			if test.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), test.wantErr) {
					t.Fatalf("SSHDestination() error = %v, want %q", err, test.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if got != test.want {
				t.Fatalf("SSHDestination() = %q, want %q", got, test.want)
			}
		})
	}
}
