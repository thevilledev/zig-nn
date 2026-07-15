package cli

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"nnctl/internal/cloud/verda"
)

func (a *app) prepareCloudBenchmarkSourceVolume(
	ctx context.Context,
	client cloudBenchmarkClient,
	creds verda.Credentials,
	packerDir string,
	workflowDir string,
	opts *cloudBenchmarkDeployOptions,
	progress *cloudBenchmarkProgress,
) (verda.Volume, error) {
	if opts.SourceOSVolumeID != "" {
		volume, err := client.GetVolume(ctx, opts.SourceOSVolumeID)
		if err != nil {
			return verda.Volume{}, fmt.Errorf("get source Verda OS volume %s: %w", opts.SourceOSVolumeID, err)
		}
		return normalizeCloudBenchmarkSourceVolume(volume)
	}

	placements, err := client.GetPlacementOptions(ctx, opts.InstanceType, opts.Market == verda.PricingMarketSpot)
	if err != nil {
		return verda.Volume{}, fmt.Errorf("list Verda %s placements for %q: %w", opts.Market, opts.InstanceType, err)
	}
	sortCloudBenchmarkPlacements(placements)
	volumes, err := client.ListVolumes(ctx)
	if err != nil {
		return verda.Volume{}, fmt.Errorf("list Verda volumes: %w", err)
	}
	source, targetLocation, reusable, err := resolveCloudBenchmarkSourceVolume(*opts, placements, volumes)
	if err != nil {
		return verda.Volume{}, err
	}
	if reusable {
		return source, nil
	}
	return a.buildCloudBenchmarkSourceVolume(ctx, client, creds, packerDir, workflowDir, opts, progress, targetLocation)
}

func resolveCloudBenchmarkSourceVolume(opts cloudBenchmarkDeployOptions, placements []verda.SpotPlacement, volumes []verda.Volume) (verda.Volume, string, bool, error) {
	if opts.LocationCode == "" {
		for _, placement := range placements {
			if source, ok := reusableCloudBenchmarkSourceVolume(volumes, opts.SourceOSVolumeName, placement.LocationCode); ok {
				return source, placement.LocationCode, true, nil
			}
		}
	}
	targetLocation, err := cloudBenchmarkTargetLocation(opts.LocationCode, placements)
	if err != nil {
		return verda.Volume{}, "", false, err
	}
	if source, ok := reusableCloudBenchmarkSourceVolume(volumes, opts.SourceOSVolumeName, targetLocation); ok {
		return source, targetLocation, true, nil
	}
	return verda.Volume{}, targetLocation, false, nil
}

func (a *app) buildCloudBenchmarkSourceVolume(
	ctx context.Context,
	client cloudBenchmarkClient,
	creds verda.Credentials,
	packerDir string,
	workflowDir string,
	opts *cloudBenchmarkDeployOptions,
	progress *cloudBenchmarkProgress,
	targetLocation string,
) (verda.Volume, error) {
	progress.detail("No reusable volume named %q in %s", opts.SourceOSVolumeName, targetLocation)
	progress.detail("Building a golden volume with Packer")
	keysFile, err := a.cloudBenchmarkAuthorizedKeysFile(ctx, client, workflowDir, opts.authorizedKeysFile, opts.SSHKeyIDs)
	if err != nil {
		return verda.Volume{}, err
	}
	templatePath := filepath.Join(packerDir, "ubuntu.pkr.hcl")
	buildArgs, err := cloudBenchmarkPackerBuildArgs(templatePath, keysFile, *opts, targetLocation)
	if err != nil {
		return verda.Volume{}, err
	}
	packerEnv := environmentWith(os.Environ(), map[string]string{
		"VERDA_CLIENT_ID":     creds.ClientID,
		"VERDA_CLIENT_SECRET": creds.ClientSecret,
	})
	if err := a.runCloudBenchmarkCommand(ctx, packerDir, packerEnv, opts.packer, "init", "."); err != nil {
		return verda.Volume{}, err
	}
	if err := a.runCloudBenchmarkCommand(ctx, packerDir, packerEnv, opts.packer, buildArgs...); err != nil {
		return verda.Volume{}, err
	}
	return waitForCloudBenchmarkSourceVolume(ctx, client, opts.SourceOSVolumeName, targetLocation, opts.timeout, opts.pollInterval)
}

func waitForCloudBenchmarkSourceVolume(ctx context.Context, client cloudBenchmarkClient, name, location string, timeout, poll time.Duration) (verda.Volume, error) {
	deadline := time.Now().Add(timeout)
	for {
		volumes, listErr := client.ListVolumes(ctx)
		if listErr == nil {
			if source, ok := reusableCloudBenchmarkSourceVolume(volumes, name, location); ok {
				return source, nil
			}
		}
		if time.Now().After(deadline) {
			if listErr != nil {
				return verda.Volume{}, fmt.Errorf("wait for Packer OS volume %q in %s: %w", name, location, listErr)
			}
			return verda.Volume{}, fmt.Errorf("packer completed but OS volume %q did not appear in %s", name, location)
		}
		timer := time.NewTimer(poll)
		select {
		case <-ctx.Done():
			timer.Stop()
			return verda.Volume{}, ctx.Err()
		case <-timer.C:
		}
	}
}

func (a *app) cloudBenchmarkAuthorizedKeysFile(ctx context.Context, client cloudBenchmarkClient, workflowDir, configured string, ids []string) (string, error) {
	if strings.TrimSpace(configured) != "" {
		path, err := filepath.Abs(configured)
		if err != nil {
			return "", fmt.Errorf("resolve authorized keys file: %w", err)
		}
		if _, err := os.Stat(path); err != nil {
			return "", fmt.Errorf("read authorized keys file: %w", err)
		}
		return path, nil
	}
	if len(ids) == 0 {
		return "", fmt.Errorf("building the golden OS volume requires --authorized-keys-file or at least one --ssh-key-id")
	}

	keys, err := client.ListSSHKeys(ctx)
	if err != nil {
		return "", fmt.Errorf("list Verda SSH keys for Packer image: %w", err)
	}
	byID := make(map[string]verda.SSHKey, len(keys))
	for _, key := range keys {
		byID[strings.TrimSpace(key.ID)] = key
	}
	var content strings.Builder
	for _, id := range ids {
		key, ok := byID[id]
		if !ok {
			return "", fmt.Errorf("verda SSH key %s was not found; run `nnctl cloud ssh-keys`", id)
		}
		publicKey := strings.TrimSpace(key.PublicKey)
		if publicKey == "" {
			return "", fmt.Errorf("verda SSH key %s did not include a public key; pass --authorized-keys-file", id)
		}
		content.WriteString(publicKey)
		content.WriteByte('\n')
	}
	path := filepath.Join(workflowDir, "authorized_keys")
	if err := os.WriteFile(path, []byte(content.String()), 0o600); err != nil {
		return "", fmt.Errorf("write temporary authorized keys file: %w", err)
	}
	return path, nil
}

func ensureCloudPackerTemplate(outputDir string, force bool) ([]string, error) {
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		return nil, fmt.Errorf("create Packer template directory: %w", err)
	}
	var written []string
	for _, file := range verda.PackerTemplateFiles() {
		path := filepath.Join(outputDir, file.Path)
		current, err := os.ReadFile(path)
		if err == nil && !force {
			continue
		}
		if err != nil && !os.IsNotExist(err) {
			return nil, fmt.Errorf("read %s: %w", path, err)
		}
		if err == nil && bytes.Equal(current, []byte(file.Content)) {
			continue
		}
		if err := os.WriteFile(path, []byte(file.Content), 0o644); err != nil {
			return nil, fmt.Errorf("write %s: %w", path, err)
		}
		written = append(written, path)
	}
	return written, nil
}

func cloudBenchmarkPackerBuildArgs(templatePath, keysFile string, opts cloudBenchmarkDeployOptions, targetLocation string) ([]string, error) {
	template, err := os.ReadFile(templatePath)
	if err != nil {
		return nil, fmt.Errorf("read Packer template: %w", err)
	}
	args := []string{
		"build",
		"-var", "authorized_keys_file=" + keysFile,
		"-var", "instance_type=" + opts.packerInstanceType,
		"-var", "image=" + opts.Image,
		"-var", "hostname=packer-verda-zig-nn-" + strings.ToLower(targetLocation),
	}
	needsConfigurableTemplate := opts.SourceOSVolumeName != defaultCloudBenchmarkSourceName ||
		(targetLocation != cloudBenchmarkPackerBuildLocation && targetLocation != cloudBenchmarkPackerArtifactLocation) ||
		opts.Market != verda.PricingMarketSpot ||
		strings.TrimSpace(opts.BaseURL) != ""
	if needsConfigurableTemplate {
		required := []string{`variable "artifact_volume_name"`, `variable "build_location_code"`, `variable "build_market"`, `variable "artifact_volume_location_codes"`, `variable "base_url"`}
		for _, declaration := range required {
			if !bytes.Contains(template, []byte(declaration)) {
				return nil, fmt.Errorf("existing %s does not support the requested source name, location, or base URL; pass --refresh-packer-template", templatePath)
			}
		}
		args = append(args,
			"-var", "artifact_volume_name="+opts.SourceOSVolumeName,
			"-var", "build_location_code="+targetLocation,
			"-var", "build_market="+opts.Market,
			"-var", fmt.Sprintf(`artifact_volume_location_codes=[%q]`, targetLocation),
			"-var", "base_url="+opts.BaseURL,
		)
	}
	return append(args, "."), nil
}

func reusableCloudBenchmarkSourceVolume(volumes []verda.Volume, name, location string) (verda.Volume, bool) {
	var matches []verda.Volume
	for _, volume := range volumes {
		if strings.TrimSpace(volume.Name) != name || !volume.IsOSVolume || !strings.EqualFold(volume.Location, location) || cloudBenchmarkVolumeDeleted(volume) {
			continue
		}
		normalized, err := normalizeCloudBenchmarkSourceVolume(volume)
		if err == nil {
			matches = append(matches, normalized)
		}
	}
	if len(matches) == 0 {
		return verda.Volume{}, false
	}
	sort.SliceStable(matches, func(i, j int) bool {
		if matches[i].CreatedAt != matches[j].CreatedAt {
			return matches[i].CreatedAt > matches[j].CreatedAt
		}
		return matches[i].ID < matches[j].ID
	})
	return matches[0], true
}

func normalizeCloudBenchmarkSourceVolume(volume verda.Volume) (verda.Volume, error) {
	volume.ID = strings.TrimSpace(volume.ID)
	volume.Name = strings.TrimSpace(volume.Name)
	volume.Location = strings.ToUpper(strings.TrimSpace(volume.Location))
	if volume.ID == "" || volume.Location == "" {
		return verda.Volume{}, fmt.Errorf("source OS volume did not report an ID and location")
	}
	return volume, nil
}

func cloudBenchmarkVolumeDeleted(volume verda.Volume) bool {
	if volume.IsPermanentlyDeleted || strings.TrimSpace(volume.DeletedAt) != "" {
		return true
	}
	switch strings.ToLower(strings.TrimSpace(volume.Status)) {
	case "deleted", "deleting":
		return true
	default:
		return false
	}
}

func sortCloudBenchmarkPlacements(placements []verda.SpotPlacement) {
	sort.SliceStable(placements, func(i, j int) bool {
		left, right := placements[i], placements[j]
		if left.PriceKnown != right.PriceKnown {
			return left.PriceKnown
		}
		if left.PriceKnown && left.SpotPrice != right.SpotPrice {
			return left.SpotPrice < right.SpotPrice
		}
		return left.LocationCode < right.LocationCode
	})
}

func cloudBenchmarkTargetLocation(requested string, placements []verda.SpotPlacement) (string, error) {
	requested = strings.ToUpper(strings.TrimSpace(requested))
	if requested != "" {
		for _, placement := range placements {
			if strings.EqualFold(placement.LocationCode, requested) {
				return requested, nil
			}
		}
		return "", fmt.Errorf("requested instance type is not currently available in %s", requested)
	}
	if len(placements) == 0 {
		return "", fmt.Errorf("requested instance type is not currently available in any location")
	}
	return strings.ToUpper(strings.TrimSpace(placements[0].LocationCode)), nil
}
