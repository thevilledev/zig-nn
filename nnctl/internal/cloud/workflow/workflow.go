// Package workflow contains reusable cloud-worker transport operations shared
// by interactive frontends and command-line workflows.
package workflow

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"net/netip"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	cloudcore "nnctl/internal/cloud"
	"nnctl/internal/cloud/verda"
	"nnctl/internal/process"
)

// Client is the subset of the Verda API needed by an interactive worker.
type Client interface {
	verda.Client
	verda.DestroyClient
	ListInstances(context.Context, string) ([]verda.Instance, error)
	ListSSHKeys(context.Context) ([]verda.SSHKey, error)
	ListInstancePrices(context.Context, verda.PricingFilters) ([]verda.InstancePrice, error)
	ListDeletedVolumes(context.Context) ([]verda.Volume, error)
}

type InstanceLister interface {
	ListInstances(context.Context, string) ([]verda.Instance, error)
}

var sshUserPattern = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_.-]{0,254}$`)

// ValidateSSHUser rejects values that OpenSSH could parse as destination syntax.
func ValidateSSHUser(user string) error {
	if !sshUserPattern.MatchString(strings.TrimSpace(user)) {
		return fmt.Errorf("SSH user must contain only letters, digits, underscores, dots, or hyphens and start with a letter or underscore")
	}
	return nil
}

// SSHDestination builds an OpenSSH destination from validated provider data.
func SSHDestination(user, rawIP string) (string, error) {
	user = strings.TrimSpace(user)
	if err := ValidateSSHUser(user); err != nil {
		return "", err
	}
	address, err := netip.ParseAddr(strings.TrimSpace(rawIP))
	if err != nil || address.Zone() != "" || !address.IsGlobalUnicast() {
		return "", fmt.Errorf("cloud provider returned an invalid unicast IP address")
	}
	return user + "@" + address.Unmap().String(), nil
}

// NewVerdaClient reads credentials from the OS keyring. Credentials never
// cross the HTTP boundary used by the learning lab.
func NewVerdaClient(ctx context.Context, baseURL string) (Client, error) {
	credentials, err := (verda.KeyringCredentialStore{}).Credentials(ctx)
	if err != nil {
		return nil, err
	}
	client, err := verda.NewSDKClient(credentials, verda.ClientOptions{
		BaseURL:   strings.TrimSpace(baseURL),
		UserAgent: "nnctl-lab",
	})
	if err != nil {
		return nil, fmt.Errorf("create Verda client: %w", err)
	}
	return client, nil
}

// RepositoryState describes the exact revision uploaded to a worker.
type RepositoryState struct {
	Revision string `json:"revision"`
	Dirty    bool   `json:"dirty"`
}

func InspectRepository(ctx context.Context, root, git string) (RepositoryState, error) {
	if strings.TrimSpace(git) == "" {
		git = "git"
	}
	revision, err := commandOutput(ctx, root, git, "rev-parse", "HEAD")
	if err != nil {
		return RepositoryState{}, fmt.Errorf("resolve HEAD: %w", err)
	}
	status, err := commandOutput(ctx, root, git, "status", "--porcelain", "--untracked-files=all")
	if err != nil {
		return RepositoryState{}, fmt.Errorf("inspect working tree: %w", err)
	}
	return RepositoryState{Revision: revision, Dirty: status != ""}, nil
}

func commandOutput(ctx context.Context, dir, name string, args ...string) (string, error) {
	cmd := process.CommandContext(ctx, name, args...)
	cmd.Dir = dir
	output, err := cmd.CombinedOutput()
	if err != nil {
		message := strings.TrimSpace(string(output))
		if message == "" {
			return "", err
		}
		return "", fmt.Errorf("%s: %w", message, err)
	}
	return strings.TrimSpace(string(output)), nil
}

func WaitInstance(ctx context.Context, client InstanceLister, instanceID string, timeout, poll time.Duration) (verda.Instance, error) {
	deadline := time.Now().Add(timeout)
	for {
		instances, err := client.ListInstances(ctx, "")
		if err != nil {
			return verda.Instance{}, fmt.Errorf("list Verda instances while waiting for %s: %w", instanceID, err)
		}
		for _, instance := range instances {
			if instance.ID != instanceID {
				continue
			}
			status := strings.ToLower(strings.TrimSpace(instance.Status))
			if terminalInstanceStatus(status) {
				return verda.Instance{}, fmt.Errorf("instance %s entered terminal status %q", instanceID, instance.Status)
			}
			if status == "running" && instance.IP != nil && strings.TrimSpace(*instance.IP) != "" {
				return instance, nil
			}
		}
		if time.Now().After(deadline) {
			return verda.Instance{}, fmt.Errorf("timed out waiting for instance %s to be running with an IP", instanceID)
		}
		if err := wait(ctx, poll); err != nil {
			return verda.Instance{}, err
		}
	}
}

func WaitProviderInstance(
	ctx context.Context,
	provider cloudcore.Provider,
	instanceID string,
	timeout time.Duration,
	poll time.Duration,
) (cloudcore.Instance, error) {
	deadline := time.Now().Add(timeout)
	for {
		instance, err := provider.Instance(ctx, instanceID)
		if err != nil {
			return cloudcore.Instance{}, fmt.Errorf("get cloud instance %s while waiting: %w", instanceID, err)
		}
		if instance.State.Terminal() {
			return cloudcore.Instance{}, fmt.Errorf("instance %s entered terminal state %q", instanceID, instance.State)
		}
		if instance.State == cloudcore.InstanceRunning && strings.TrimSpace(instance.PublicIP) != "" {
			return instance, nil
		}
		if time.Now().After(deadline) {
			return cloudcore.Instance{}, fmt.Errorf("timed out waiting for instance %s to be running with an IP", instanceID)
		}
		if err := wait(ctx, poll); err != nil {
			return cloudcore.Instance{}, err
		}
	}
}

func terminalInstanceStatus(status string) bool {
	switch status {
	case "deleted", "deleting", "error", "failed", "canceled", "cancelled":
		return true
	default:
		return false
	}
}

func wait(ctx context.Context, duration time.Duration) error {
	timer := time.NewTimer(duration)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

func SSHArgs(knownHosts string) []string {
	return []string{
		"-o", "UserKnownHostsFile=" + knownHosts,
		"-o", "StrictHostKeyChecking=accept-new",
		"-o", "ConnectTimeout=10",
	}
}

func WaitSSH(ctx context.Context, ssh string, baseArgs []string, host string, timeout, poll time.Duration) error {
	deadline := time.Now().Add(timeout)
	var lastErr error
	for {
		args := append(append([]string{}, baseArgs...), host, "true")
		cmd := process.CommandContext(ctx, ssh, args...)
		cmd.Stdout = io.Discard
		cmd.Stderr = io.Discard
		if err := cmd.Run(); err == nil {
			return nil
		} else {
			lastErr = err
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("timed out waiting for SSH on %s: %w", host, lastErr)
		}
		if err := wait(ctx, poll); err != nil {
			return err
		}
	}
}

// WaitCommand retries a remote readiness command until it succeeds. This is
// needed for userdata-based instances where SSH can become available before
// Zig and the GPU toolchain have finished installing.
func WaitCommand(ctx context.Context, ssh string, baseArgs []string, host, command string, timeout, poll time.Duration) error {
	deadline := time.Now().Add(timeout)
	var lastErr error
	for {
		lastErr = RunSSH(ctx, ssh, baseArgs, host, command, io.Discard, io.Discard)
		if lastErr == nil {
			return nil
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("timed out waiting for remote toolchain on %s: %w", host, lastErr)
		}
		if err := wait(ctx, poll); err != nil {
			return err
		}
	}
}

func RunSSH(ctx context.Context, ssh string, baseArgs []string, host, command string, stdout, stderr io.Writer) error {
	args := append(append([]string{}, baseArgs...), host, command)
	cmd := process.CommandContext(ctx, ssh, args...)
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("%s failed: %w", ssh, err)
	}
	return nil
}

// StreamSSH scans remote stdout and stderr independently so native NDJSON can
// feed the lab protocol while diagnostics remain human readable.
func StreamSSH(
	ctx context.Context,
	ssh string,
	baseArgs []string,
	host string,
	command string,
	stdoutLine func([]byte) error,
	stderrLine func(string),
) error {
	runCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	args := append(append([]string{}, baseArgs...), host, command)
	cmd := process.CommandContext(runCtx, ssh, args...)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("capture remote stdout: %w", err)
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return fmt.Errorf("capture remote stderr: %w", err)
	}
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("start remote experiment: %w", err)
	}

	results := make(chan error, 2)
	go func() {
		err := scan(stdout, stdoutLine)
		if err != nil {
			cancel()
		}
		results <- err
	}()
	go func() {
		results <- scan(stderr, func(line []byte) error {
			stderrLine(string(line))
			return nil
		})
	}()
	scanErr := errors.Join(<-results, <-results)
	waitErr := cmd.Wait()
	if scanErr != nil {
		return scanErr
	}
	if waitErr != nil {
		return fmt.Errorf("remote experiment failed: %w", waitErr)
	}
	return nil
}

func scan(reader io.Reader, visit func([]byte) error) error {
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 64*1024), 1024*1024)
	for scanner.Scan() {
		line := append([]byte(nil), scanner.Bytes()...)
		if err := visit(line); err != nil {
			return err
		}
	}
	return scanner.Err()
}

type UploadOptions struct {
	RepoRoot string
	Target   string
	Ref      string
	SSH      string
	Git      string
	Tar      string
	Rsync    string
	Delete   bool
	DryRun   bool
	Runner   CommandRunner
}

// CommandRunner executes one snapshot preparation or transfer command.
type CommandRunner func(context.Context, string, string, ...string) error

// UploadSnapshot transfers only a git archive, never local build products,
// untracked files, credentials, or .git metadata.
func UploadSnapshot(ctx context.Context, options UploadOptions) (runErr error) {
	if strings.TrimSpace(options.RepoRoot) == "" || strings.TrimSpace(options.Target) == "" {
		return errors.New("repository root and upload target are required")
	}
	if options.Ref == "" {
		options.Ref = "HEAD"
	}
	if options.Git == "" {
		options.Git = "git"
	}
	if options.Tar == "" {
		options.Tar = "tar"
	}
	if options.Rsync == "" {
		options.Rsync = "rsync"
	}
	runner := options.Runner
	if runner == nil {
		runner = run
	}
	tmpDir, err := os.MkdirTemp("", "nnctl-cloud-upload-*")
	if err != nil {
		return fmt.Errorf("create upload work directory: %w", err)
	}
	defer func() {
		if err := os.RemoveAll(tmpDir); err != nil {
			runErr = errors.Join(runErr, fmt.Errorf("remove upload work directory: %w", err))
		}
	}()

	snapshotDir := filepath.Join(tmpDir, "snapshot")
	if err := os.Mkdir(snapshotDir, 0o700); err != nil {
		return fmt.Errorf("create snapshot directory: %w", err)
	}
	archive := filepath.Join(tmpDir, "repo.tar")
	if err := runner(ctx, options.RepoRoot, options.Git, "archive", "--format=tar", "--output", archive, "--", options.Ref); err != nil {
		return err
	}
	if err := runner(ctx, options.RepoRoot, options.Tar, "-xf", archive, "-C", snapshotDir); err != nil {
		return err
	}
	args := []string{"-az"}
	if options.Delete {
		args = append(args, "--delete")
	}
	if options.DryRun {
		args = append(args, "--dry-run", "--itemize-changes")
	}
	if options.SSH != "" {
		args = append(args, "-e", options.SSH)
	}
	args = append(args, "--", withTrailingSeparator(snapshotDir), options.Target)
	return runner(ctx, options.RepoRoot, options.Rsync, args...)
}

func run(ctx context.Context, dir, name string, args ...string) error {
	cmd := process.CommandContext(ctx, name, args...)
	cmd.Dir = dir
	output, err := cmd.CombinedOutput()
	if err == nil {
		return nil
	}
	message := strings.TrimSpace(string(output))
	if message == "" {
		return fmt.Errorf("%s failed: %w", name, err)
	}
	return fmt.Errorf("%s failed: %s: %w", name, message, err)
}

func withTrailingSeparator(path string) string {
	if strings.HasSuffix(path, string(os.PathSeparator)) {
		return path
	}
	return path + string(os.PathSeparator)
}

func ShellQuote(value string) string {
	return "'" + strings.ReplaceAll(value, "'", `'"'"'`) + "'"
}

func CommandString(name string, args []string) string {
	parts := []string{ShellQuote(name)}
	for _, arg := range args {
		parts = append(parts, ShellQuote(arg))
	}
	return strings.Join(parts, " ")
}

func Destroy(ctx context.Context, client verda.DestroyClient, instanceID, cloneID, sourceID string) error {
	volumes := []string(nil)
	if strings.TrimSpace(cloneID) != "" {
		volumes = []string{cloneID}
	}
	_, err := verda.Destroy(ctx, client, verda.DestroyOptions{
		InstanceIDs:       []string{instanceID},
		VolumeIDs:         volumes,
		SourceOSVolumeID:  sourceID,
		DeletePermanently: true,
	})
	return err
}
