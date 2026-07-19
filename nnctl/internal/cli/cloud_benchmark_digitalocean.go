package cli

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	cloudcore "nnctl/internal/cloud"
	"nnctl/internal/cloud/digitalocean"
	cloudworkflow "nnctl/internal/cloud/workflow"
)

type digitalOceanCloudBenchmarkStrategy struct {
	provider   cloudcore.Provider
	planned    *cloudcore.Deployment
	deployment *cloudcore.Deployment
	instance   cloudcore.Instance
	offering   cloudcore.Offering
}

func (s *digitalOceanCloudBenchmarkStrategy) prepareImage(r *cloudBenchmarkRun) (runErr error) {
	r.progress.phase(cloudBenchmarkPhaseImage, "Resolve GPU-ready image")
	if err := r.progress.Err(); err != nil {
		return err
	}
	provider, err := r.app.cloudProvider(r.ctx, r.opts.provider, r.opts.baseURL, "nnctl-benchmark", false)
	if err != nil {
		return err
	}
	s.provider = provider
	offerings, err := provider.Offerings(r.ctx, cloudcore.OfferingFilters{
		IDs: []string{r.opts.InstanceType}, Locations: []string{r.opts.LocationCode},
		Markets: []string{digitalocean.MarketOnDemand}, AvailableOnly: true,
	})
	if err != nil {
		return err
	}
	for _, offering := range offerings {
		if strings.EqualFold(offering.ID, r.opts.InstanceType) && strings.EqualFold(offering.Location, r.opts.LocationCode) {
			s.offering = offering
			break
		}
	}
	if s.offering.ID == "" {
		return fmt.Errorf("DigitalOcean offering %s is not available in %s", r.opts.InstanceType, r.opts.LocationCode)
	}
	if !r.opts.backendSet {
		switch {
		case slices.Contains(s.offering.Backends, "rocm"):
			r.opts.backend = "rocm"
		case slices.Contains(s.offering.Backends, "cuda"):
			r.opts.backend = "cuda"
		}
	}
	if r.opts.backend != "none" && !slices.Contains(s.offering.Backends, r.opts.backend) {
		return fmt.Errorf(
			"backend %q is unavailable for DigitalOcean %s; supported backends: %s",
			r.opts.backend,
			r.opts.InstanceType,
			strings.Join(s.offering.Backends, ", "),
		)
	}
	planned, err := provider.Deploy(r.ctx, s.deployRequest(r, true))
	if err != nil {
		return err
	}
	s.planned = planned
	if err := r.ensureWorkflowDir(); err != nil {
		return err
	}
	defer func() {
		if runErr != nil {
			runErr = errors.Join(runErr, r.removeWorkflowDir())
		}
	}()
	r.progress.detail("Image: %s", planned.Request.Image)
	r.progress.detail(
		"Accelerator: %d× %s, backend: %s",
		s.offering.Accelerator.Count,
		s.offering.Accelerator.Model,
		r.opts.backend,
	)
	return r.progress.Err()
}

func (s *digitalOceanCloudBenchmarkStrategy) deploy(r *cloudBenchmarkRun) error {
	r.progress.phase(cloudBenchmarkPhaseDeploy, "Deploy cloud worker")
	r.progress.detail("Instance: %s, market: %s, location: %s", r.opts.InstanceType, r.opts.Market, r.opts.LocationCode)
	if err := r.progress.Err(); err != nil {
		return err
	}
	deployment, err := s.provider.Deploy(r.ctx, s.deployRequest(r, false))
	if err != nil {
		return err
	}
	if deployment == nil || deployment.Instance == nil || strings.TrimSpace(deployment.Instance.ID) == "" {
		return errors.New("DigitalOcean deployment did not return an instance")
	}
	s.deployment = deployment
	s.instance = *deployment.Instance
	r.instanceID = deployment.Instance.ID
	r.progress.detail("Worker: %s", r.instanceID)
	return r.progress.Err()
}

func (s *digitalOceanCloudBenchmarkStrategy) wait(r *cloudBenchmarkRun) error {
	r.progress.phase(cloudBenchmarkPhaseReady, "Wait for worker readiness")
	r.progress.detail("Waiting for an IP address, cloud-init, and SSH access")
	if err := r.progress.Err(); err != nil {
		return err
	}
	instance, err := cloudworkflow.WaitProviderInstance(
		r.ctx,
		s.provider,
		r.instanceID,
		r.opts.timeout,
		r.opts.pollInterval,
	)
	if err != nil {
		return err
	}
	s.instance = instance
	r.host, err = cloudworkflow.SSHDestination(r.opts.sshUser, instance.PublicIP)
	if err != nil {
		return err
	}
	r.sshArgs = cloudBenchmarkSSHArgs(filepath.Join(r.workflowDir, "known_hosts"))
	if err := r.ops.waitSSH(r.ctx, r.opts.ssh, r.sshArgs, r.host, r.opts.timeout, r.opts.pollInterval); err != nil {
		return err
	}
	if err := r.ops.waitCommand(
		r.ctx,
		r.opts.ssh,
		r.sshArgs,
		r.host,
		digitalOceanBenchmarkPreflight(r.opts.backend),
		r.opts.timeout,
		r.opts.pollInterval,
	); err != nil {
		return err
	}
	r.progress.detail("Worker ready: %s", r.host)
	return r.progress.Err()
}

func (s *digitalOceanCloudBenchmarkStrategy) metadata(r *cloudBenchmarkRun) cloudBenchmarkMetadata {
	price := firstCloudBenchmarkPrice(s.instance.PricePerHour, s.offering.PricePerHour)
	return cloudBenchmarkMetadata{
		Timestamp: r.stamp, Commit: r.commit, Provider: digitalocean.ProviderName,
		InstanceID: r.instanceID, InstanceType: r.opts.InstanceType,
		Location: firstNonEmpty(s.instance.Location, r.opts.LocationCode),
		Market:   digitalocean.MarketOnDemand, PricePerHour: price,
		Backend: r.opts.backend, GPU: s.offering.Accelerator.Model,
		MemoryMiB: positiveIntString(s.offering.Accelerator.MemoryMiB),
	}
}

func (s *digitalOceanCloudBenchmarkStrategy) cleanup(ctx context.Context, r *cloudBenchmarkRun) error {
	_, err := s.provider.Destroy(ctx, cloudcore.DestroyRequest{
		InstanceIDs: []string{r.instanceID}, Permanent: true,
	})
	return err
}

func (*digitalOceanCloudBenchmarkStrategy) resources(r *cloudBenchmarkRun) string {
	return "DigitalOcean Droplet " + r.instanceID
}

func (s *digitalOceanCloudBenchmarkStrategy) deployRequest(r *cloudBenchmarkRun, dryRun bool) cloudcore.DeployRequest {
	image := strings.TrimSpace(r.opts.Image)
	if image == "" && s.planned != nil {
		image = s.planned.Request.Image
	}
	return cloudcore.DeployRequest{
		OfferingID:  r.opts.InstanceType,
		Location:    r.opts.LocationCode,
		Market:      digitalocean.MarketOnDemand,
		Image:       image,
		Name:        r.opts.Hostname,
		Description: r.opts.Description,
		SSHKeyIDs:   append([]string(nil), r.opts.SSHKeyIDs...),
		DryRun:      dryRun,
	}
}

func digitalOceanBenchmarkPreflight(backend string) string {
	command := "if [ -f /etc/profile.d/zig-nn.sh ]; then . /etc/profile.d/zig-nn.sh; fi; zig version"
	switch backend {
	case "cuda":
		return command + "; nvidia-smi >/dev/null"
	case "rocm":
		return command + "; rocminfo >/dev/null"
	default:
		return command
	}
}

func firstCloudBenchmarkPrice(values ...float64) float64 {
	for _, value := range values {
		if value != 0 {
			return value
		}
	}
	return 0
}

func positiveIntString(value int) string {
	if value <= 0 {
		return ""
	}
	return strconv.Itoa(value)
}

var _ cloudBenchmarkStrategy = (*digitalOceanCloudBenchmarkStrategy)(nil)
