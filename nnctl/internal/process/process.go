// Package process creates cancellable subprocesses with bounded cleanup.
package process

import (
	"context"
	"os/exec"
	"time"
)

const waitDelay = 5 * time.Second

// CommandContext returns an exec.Cmd with bounded, platform-specific cleanup.
// Unix cancellation terminates the command's complete process group.
func CommandContext(ctx context.Context, name string, args ...string) *exec.Cmd {
	cmd := exec.CommandContext(ctx, name, args...)
	configureTreeCancellation(cmd)
	cmd.WaitDelay = waitDelay
	return cmd
}
