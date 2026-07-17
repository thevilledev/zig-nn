//go:build unix

package process

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

func TestCommandContextCancelsProcessGroup(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithCancel(t.Context())
	pidFile := filepath.Join(t.TempDir(), "child.pid")
	cmd := CommandContext(ctx, "sh", "-c", "sleep 60 & echo $! > \"$1\"; wait", "sh", pidFile)
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	childPID := waitForChildPID(t, pidFile)
	cancel()
	if err := cmd.Wait(); err == nil {
		t.Fatal("canceled process exited successfully")
	}
	deadline := time.Now().Add(2 * time.Second)
	for {
		err := syscall.Kill(childPID, 0)
		if errors.Is(err, syscall.ESRCH) {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("child process %d survived context cancellation: %v", childPID, err)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func waitForChildPID(t *testing.T, path string) int {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for {
		data, err := os.ReadFile(path)
		if err == nil {
			pid, parseErr := strconv.Atoi(strings.TrimSpace(string(data)))
			if parseErr != nil {
				t.Fatalf("parse child PID: %v", parseErr)
			}
			return pid
		}
		if !errors.Is(err, os.ErrNotExist) {
			t.Fatal(err)
		}
		if time.Now().After(deadline) {
			t.Fatal("child process did not start")
		}
		time.Sleep(10 * time.Millisecond)
	}
}
