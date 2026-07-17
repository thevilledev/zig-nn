package lab

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"

	cloudworkflow "nnctl/internal/cloud/workflow"
	"nnctl/internal/zig"
)

func (m *CloudManager) Execute(
	ctx context.Context,
	spec ExperimentSpec,
	options RunOptions,
	stdoutLine func([]byte) error,
	stderrLine func(string),
) error {
	worker, err := m.beginExecution(options.WorkerID, options.Backend)
	if err != nil {
		return err
	}
	defer func() {
		m.finishExecution(worker, stderrLine)
	}()
	if err := m.persistWorker(); err != nil {
		return err
	}

	repository, err := cloudworkflow.InspectRepository(ctx, m.repoRoot, m.git)
	if err != nil {
		return err
	}
	if repository.Dirty && !options.AcknowledgeDirty {
		return errors.New("working tree has uncommitted changes; acknowledge that the cloud run uses committed HEAD")
	}
	remoteDir := "/root/zig-nn-lab-" + repository.Revision[:12]
	stderrLine("cloud: uploading committed revision " + repository.Revision)
	host, err := cloudworkflow.SSHDestination(m.sshUser, worker.IP)
	if err != nil {
		return err
	}
	knownHosts, err := m.knownHostsPath(worker.ID)
	if err != nil {
		return err
	}
	sshArgs := cloudworkflow.SSHArgs(knownHosts)
	if err := cloudworkflow.UploadSnapshot(ctx, cloudworkflow.UploadOptions{
		RepoRoot: m.repoRoot,
		Target:   host + ":" + remoteDir,
		Ref:      repository.Revision,
		SSH:      cloudworkflow.CommandString(m.ssh, sshArgs),
		Git:      m.git,
		Tar:      m.tar,
		Rsync:    m.rsync,
		Delete:   true,
	}); err != nil {
		return err
	}
	if _, err := m.updateWorkerInState(worker.ID, CloudWorkerBusy, func(current *managedCloudWorker) {
		current.RepositoryCommit = repository.Revision
		current.Message = "Running " + spec.Title
	}); err != nil {
		return err
	}
	if err := m.persistWorker(); err != nil {
		return err
	}
	mode := m.mode
	if mode == "" {
		mode = "Debug"
	}
	if spec.Optimize != "" {
		mode = spec.Optimize
	}
	gpu := ""
	if options.Backend != "cpu" {
		gpu = options.Backend
	}
	args := zig.RunArgs(spec.Step, zig.Options{Optimize: mode, GPU: gpu}, options.Arguments)
	command := strings.Join([]string{
		"set -eu",
		"if [ -f /etc/profile.d/zig-nn.sh ]; then . /etc/profile.d/zig-nn.sh; fi",
		"cd " + cloudworkflow.ShellQuote(remoteDir),
		"exec " + cloudworkflow.CommandString("zig", args),
	}, "; ")
	return cloudworkflow.StreamSSH(ctx, m.ssh, sshArgs, host, command, stdoutLine, stderrLine)
}

func (m *CloudManager) beginExecution(workerID, backend string) (managedCloudWorker, error) {
	return m.mutateWorker(workerID, func(current *managedCloudWorker) error {
		if m.closed || m.ctx.Err() != nil {
			return errors.New("cloud manager is closed")
		}
		if current.State != CloudWorkerReady {
			return fmt.Errorf("cloud worker is %s", current.State)
		}
		if !slices.Contains(current.Backends, backend) {
			return fmt.Errorf("backend %q is unavailable on cloud worker", backend)
		}
		current.State = CloudWorkerBusy
		current.Message = "Preparing remote experiment"
		return nil
	})
}

func (m *CloudManager) finishExecution(worker managedCloudWorker, stderrLine func(string)) {
	if worker.AutoDestroy {
		cleanupCtx, cancel := m.cleanupContext()
		if err := m.Destroy(cleanupCtx, worker.ID); err != nil {
			stderrLine("cloud cleanup failed: " + err.Error())
		}
		cancel()
		return
	}
	if _, err := m.transitionWorker(
		worker.ID,
		[]CloudWorkerState{CloudWorkerBusy},
		CloudWorkerReady,
		"Ready for another experiment",
		nil,
	); err != nil {
		return
	}
	if err := m.persistWorker(); err != nil {
		stderrLine("cloud state persistence failed: " + err.Error())
	}
}

type RoutingExecutor struct {
	Local Executor
	Cloud *CloudManager
}

func (e RoutingExecutor) Execute(
	ctx context.Context,
	spec ExperimentSpec,
	options RunOptions,
	stdout func([]byte) error,
	stderr func(string),
) error {
	if options.Target == RunTargetCloud {
		if e.Cloud == nil {
			return errors.New("cloud execution is disabled")
		}
		return e.Cloud.Execute(ctx, spec, options, stdout, stderr)
	}
	return e.Local.Execute(ctx, spec, options, stdout, stderr)
}
