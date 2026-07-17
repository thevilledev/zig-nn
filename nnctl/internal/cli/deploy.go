package cli

import (
	"context"
	"fmt"
	"strings"

	cloudworkflow "nnctl/internal/cloud/workflow"
	"nnctl/internal/process"
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

func (a *app) runDeploy(ctx context.Context, opts deployOptions) error {
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

	if err := cloudworkflow.UploadSnapshot(ctx, cloudworkflow.UploadOptions{
		RepoRoot: a.repoRoot,
		Target:   opts.target,
		Ref:      opts.ref,
		SSH:      opts.ssh,
		Git:      opts.git,
		Tar:      opts.tar,
		Rsync:    opts.rsync,
		Delete:   opts.delete,
		DryRun:   opts.dryRun,
		Runner:   a.run,
	}); err != nil {
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
	cmd := process.CommandContext(ctx, git, "status", "--porcelain", "--untracked-files=all")
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
