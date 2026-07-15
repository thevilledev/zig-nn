package cli

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

type deployOptions struct {
	target      string
	ref         string
	delete      bool
	dryRun      bool
	ignoreDirty bool
	ssh         string
	git         string
	tar         string
	rsync       string
}

func defaultDeployOptions() deployOptions {
	return deployOptions{
		ref:   "HEAD",
		git:   "git",
		tar:   "tar",
		rsync: "rsync",
	}
}

func (a *app) runDeploy(ctx context.Context, opts deployOptions) (runErr error) {
	if strings.TrimSpace(opts.target) == "" {
		return fmt.Errorf("deploy target is required")
	}
	if strings.TrimSpace(opts.ref) == "" {
		return fmt.Errorf("--ref must not be empty")
	}
	if opts.git == "" || opts.tar == "" || opts.rsync == "" {
		return fmt.Errorf("git, tar, and rsync executables must be configured")
	}

	if !opts.ignoreDirty {
		status, err := a.gitStatusPorcelain(ctx, opts.git)
		if err != nil {
			return err
		}
		if status != "" {
			return fmt.Errorf("working tree has uncommitted changes; commit them or pass --ignore-dirty to deploy %s anyway", opts.ref)
		}
	}

	tmpDir, err := os.MkdirTemp("", "nnctl-deploy-*")
	if err != nil {
		return fmt.Errorf("create deploy snapshot dir: %w", err)
	}
	defer func() {
		if err := os.RemoveAll(tmpDir); err != nil {
			runErr = errors.Join(runErr, fmt.Errorf("remove deploy snapshot dir: %w", err))
		}
	}()

	snapshotDir := filepath.Join(tmpDir, "snapshot")
	if err := os.Mkdir(snapshotDir, 0o755); err != nil {
		return fmt.Errorf("create deploy snapshot: %w", err)
	}

	archivePath := filepath.Join(tmpDir, "repo.tar")
	if err := a.run(ctx, a.repoRoot, opts.git, deployArchiveArgs(opts, archivePath)...); err != nil {
		return err
	}
	if err := a.run(ctx, a.repoRoot, opts.tar, "-xf", archivePath, "-C", snapshotDir); err != nil {
		return err
	}
	if err := a.run(ctx, a.repoRoot, opts.rsync, deployRsyncArgs(opts, snapshotDir)...); err != nil {
		return err
	}

	if opts.dryRun {
		if _, err := fmt.Fprintf(a.stdout(), "dry run complete for git snapshot %s to %s\n", opts.ref, opts.target); err != nil {
			return fmt.Errorf("write deploy result: %w", err)
		}
		return nil
	}
	if _, err := fmt.Fprintf(a.stdout(), "deployed git snapshot %s to %s\n", opts.ref, opts.target); err != nil {
		return fmt.Errorf("write deploy result: %w", err)
	}
	return nil
}

func (a *app) gitStatusPorcelain(ctx context.Context, git string) (string, error) {
	cmd := exec.CommandContext(ctx, git, "status", "--porcelain", "--untracked-files=all")
	cmd.Dir = a.repoRoot
	output, err := cmd.CombinedOutput()
	if err != nil {
		text := strings.TrimSpace(string(output))
		if text == "" {
			return "", fmt.Errorf("git status failed: %w", err)
		}
		return "", fmt.Errorf("git status failed: %s: %w", text, err)
	}
	return strings.TrimSpace(string(output)), nil
}

func deployArchiveArgs(opts deployOptions, archivePath string) []string {
	return []string{"archive", "--format=tar", "--output", archivePath, opts.ref}
}

func deployRsyncArgs(opts deployOptions, snapshotDir string) []string {
	args := []string{"-az"}
	if opts.delete {
		args = append(args, "--delete")
	}
	if opts.dryRun {
		args = append(args, "--dry-run", "--itemize-changes")
	}
	if opts.ssh != "" {
		args = append(args, "-e", opts.ssh)
	}
	return append(args, withTrailingPathSeparator(snapshotDir), opts.target)
}

func withTrailingPathSeparator(path string) string {
	separator := string(os.PathSeparator)
	if strings.HasSuffix(path, separator) {
		return path
	}
	return path + separator
}
