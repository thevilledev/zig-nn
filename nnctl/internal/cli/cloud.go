package cli

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"text/tabwriter"

	verdacloud "nnctl/internal/cloud/verda"
)

type cloudDeployOptions struct {
	verdacloud.DeployOptions
	userDataFile string
	jsonOutput   bool
}

type cloudSSHKeysOptions struct {
	baseURL    string
	jsonOutput bool
}

type cloudVolumesOptions struct {
	verdacloud.PurgeVolumesOptions
	includeDeleted bool
	purge          bool
	jsonOutput     bool
}

type cloudDestroyOptions struct {
	verdacloud.DestroyOptions
	jsonOutput bool
}

type cloudPackerTemplateOptions struct {
	force bool
}

type cloudListOptions struct {
	baseURL    string
	status     string
	all        bool
	jsonOutput bool
}

type cloudPricingOptions struct {
	filters    verdacloud.PricingFilters
	zones      []string
	gpuCounts  []int
	singleGPU  bool
	allGPU     bool
	baseURL    string
	sortBy     string
	jsonOutput bool
}

func (a *App) newVerdaSDKClient(ctx context.Context, baseURL string) (*verdacloud.SDKClient, error) {
	client, _, err := a.newVerdaSDKClientWithCredentials(ctx, baseURL)
	return client, err
}

func (a *App) newVerdaSDKClientWithCredentials(ctx context.Context, baseURL string) (*verdacloud.SDKClient, verdacloud.Credentials, error) {
	creds, err := (verdacloud.KeyringCredentialStore{}).Credentials(ctx)
	if err != nil {
		return nil, verdacloud.Credentials{}, err
	}
	client, err := verdacloud.NewSDKClient(creds, verdacloud.ClientOptions{
		BaseURL:   baseURL,
		UserAgent: "nnctl",
	})
	if err != nil {
		return nil, verdacloud.Credentials{}, fmt.Errorf("create Verda client: %w", err)
	}
	return client, creds, nil
}

func (a *App) runCloudDeploy(ctx context.Context, opts cloudDeployOptions) error {
	deployOpts := opts.DeployOptions
	if opts.userDataFile != "" {
		script, err := os.ReadFile(opts.userDataFile)
		if err != nil {
			return fmt.Errorf("read userdata file: %w", err)
		}
		deployOpts.UserDataScript = string(script)
	}

	var client verdacloud.Client
	if !deployOpts.DryRun {
		sdkClient, err := a.newVerdaSDKClient(ctx, deployOpts.BaseURL)
		if err != nil {
			return err
		}
		client = sdkClient
	}

	result, err := verdacloud.Deploy(ctx, client, deployOpts)
	if err != nil {
		return err
	}
	return a.printCloudDeployResult(result, opts.jsonOutput)
}

func (a *App) runCloudSSHKeys(ctx context.Context, opts cloudSSHKeysOptions) error {
	client, err := a.newVerdaSDKClient(ctx, opts.baseURL)
	if err != nil {
		return err
	}
	keys, err := client.ListSSHKeys(ctx)
	if err != nil {
		return fmt.Errorf("list Verda SSH keys: %w", err)
	}
	return a.printCloudSSHKeys(keys, opts.jsonOutput)
}

func (a *App) runCloudVolumes(ctx context.Context, opts cloudVolumesOptions) error {
	if opts.purge || opts.AllDeleted {
		var client verdacloud.VolumePurgeClient
		if !opts.DryRun {
			sdkClient, err := a.newVerdaSDKClient(ctx, opts.BaseURL)
			if err != nil {
				return err
			}
			client = sdkClient
		}

		result, err := verdacloud.PurgeVolumes(ctx, client, opts.PurgeVolumesOptions)
		if err != nil {
			return err
		}
		return a.printCloudPurgeResult(result, opts.jsonOutput)
	}

	client, err := a.newVerdaSDKClient(ctx, opts.BaseURL)
	if err != nil {
		return err
	}
	volumes, err := verdacloud.ListVolumes(ctx, client, verdacloud.ListVolumesOptions{
		IncludeDeleted: opts.includeDeleted,
	})
	if err != nil {
		return err
	}
	return a.printCloudVolumes(volumes, opts.jsonOutput)
}

func (a *App) runCloudDestroy(ctx context.Context, opts cloudDestroyOptions) error {
	var client verdacloud.DestroyClient
	if !opts.DryRun {
		sdkClient, err := a.newVerdaSDKClient(ctx, opts.BaseURL)
		if err != nil {
			return err
		}
		client = sdkClient
	}

	result, err := verdacloud.Destroy(ctx, client, opts.DestroyOptions)
	if err != nil {
		return err
	}
	return a.printCloudDestroyResult(result, opts.jsonOutput)
}

func (a *App) runCloudPackerTemplate(outputDir string, opts cloudPackerTemplateOptions) error {
	written, err := writeCloudPackerTemplate(outputDir, opts.force)
	if err != nil {
		return err
	}
	for _, path := range written {
		fmt.Fprintf(a.stdout(), "wrote %s\n", path)
	}
	return nil
}

func (a *App) runCloudList(ctx context.Context, opts cloudListOptions) error {
	client, err := a.newVerdaSDKClient(ctx, opts.baseURL)
	if err != nil {
		return err
	}
	instances, err := client.ListInstances(ctx, opts.status)
	if err != nil {
		return fmt.Errorf("list Verda instances: %w", err)
	}
	if opts.status == "" && !opts.all {
		instances = activeCloudInstances(instances)
	}
	return a.printCloudInstances(instances, opts.jsonOutput)
}

func writeCloudPackerTemplate(outputDir string, force bool) ([]string, error) {
	outputDir = strings.TrimSpace(outputDir)
	if outputDir == "" {
		return nil, fmt.Errorf("output directory is required")
	}
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		return nil, fmt.Errorf("create Packer template directory: %w", err)
	}

	var written []string
	for _, file := range verdacloud.PackerTemplateFiles() {
		path := filepath.Join(outputDir, file.Path)
		flag := os.O_WRONLY | os.O_CREATE
		if force {
			flag |= os.O_TRUNC
		} else {
			flag |= os.O_EXCL
		}
		handle, err := os.OpenFile(path, flag, 0o644)
		if err != nil {
			if os.IsExist(err) {
				return nil, fmt.Errorf("%s already exists; pass --force to overwrite", path)
			}
			return nil, fmt.Errorf("create %s: %w", path, err)
		}
		if _, err := handle.WriteString(file.Content); err != nil {
			_ = handle.Close()
			return nil, fmt.Errorf("write %s: %w", path, err)
		}
		if err := handle.Close(); err != nil {
			return nil, fmt.Errorf("close %s: %w", path, err)
		}
		written = append(written, path)
	}
	return written, nil
}

func (a *App) runCloudPricing(ctx context.Context, opts cloudPricingOptions) error {
	opts.filters = verdacloud.NormalizePricingFilters(opts.filters)
	if err := verdacloud.ValidatePricingFilters(opts.filters); err != nil {
		return err
	}

	client, err := a.newVerdaSDKClient(ctx, opts.baseURL)
	if err != nil {
		return err
	}
	prices, err := client.ListInstancePrices(ctx, opts.filters)
	if err != nil {
		return fmt.Errorf("list Verda prices: %w", err)
	}
	verdacloud.SortInstancePrices(prices, opts.sortBy)
	return a.printCloudPricing(prices, opts.jsonOutput)
}

func (a *App) printCloudSSHKeys(keys []verdacloud.SSHKey, jsonOutput bool) error {
	if jsonOutput {
		enc := json.NewEncoder(a.stdout())
		enc.SetIndent("", "  ")
		return enc.Encode(keys)
	}
	if len(keys) == 0 {
		fmt.Fprintln(a.stdout(), "no Verda SSH keys found")
		return nil
	}

	w := tabwriter.NewWriter(a.stdout(), 0, 0, 2, ' ', 0)
	fmt.Fprintln(w, "ID\tNAME\tFINGERPRINT")
	for _, key := range keys {
		fmt.Fprintf(w, "%s\t%s\t%s\n", key.ID, key.Name, key.Fingerprint)
	}
	return w.Flush()
}

func (a *App) printCloudVolumes(volumes []verdacloud.Volume, jsonOutput bool) error {
	if jsonOutput {
		enc := json.NewEncoder(a.stdout())
		enc.SetIndent("", "  ")
		return enc.Encode(volumes)
	}
	if len(volumes) == 0 {
		fmt.Fprintln(a.stdout(), "no Verda volumes found")
		return nil
	}

	w := tabwriter.NewWriter(a.stdout(), 0, 0, 2, ' ', 0)
	fmt.Fprintln(w, "ID\tNAME\tSTATUS\tTYPE\tSIZE\tLOCATION\tOS\tDELETED AT")
	for _, volume := range volumes {
		fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%s\t%s\t%t\t%s\n",
			volume.ID,
			firstNonEmpty(volume.Name),
			firstNonEmpty(volume.Status),
			firstNonEmpty(volume.Type),
			formatCloudVolumeSize(volume.Size),
			firstNonEmpty(volume.Location),
			volume.IsOSVolume,
			firstNonEmpty(volume.DeletedAt),
		)
	}
	return w.Flush()
}

func (a *App) printCloudPurgeResult(result *verdacloud.PurgeVolumesResult, jsonOutput bool) error {
	if jsonOutput {
		enc := json.NewEncoder(a.stdout())
		enc.SetIndent("", "  ")
		return enc.Encode(result)
	}

	if result.Request.AllDeleted {
		if result.DryRun {
			fmt.Fprintln(a.stdout(), "dry run: would purge all deleted Verda volumes")
			return nil
		}
		if len(result.Results) == 0 {
			fmt.Fprintln(a.stdout(), "no deleted Verda volumes found")
			return nil
		}
		fmt.Fprintln(a.stdout(), "purged all deleted Verda volumes")
		for _, actionResult := range result.Results {
			fmt.Fprintf(a.stdout(), "%s: %s\n", actionResult.VolumeID, firstNonEmpty(actionResult.Status, "purged"))
		}
		return nil
	}

	ids := strings.Join(result.Request.VolumeIDs, ", ")
	if result.DryRun {
		fmt.Fprintf(a.stdout(), "dry run: would purge Verda volume%s %s\n", pluralSuffix(result.Request.VolumeIDs), ids)
		return nil
	}
	fmt.Fprintf(a.stdout(), "purged Verda volume%s %s\n", pluralSuffix(result.Request.VolumeIDs), ids)
	for _, actionResult := range result.Results {
		fmt.Fprintf(a.stdout(), "%s: %s\n", actionResult.VolumeID, firstNonEmpty(actionResult.Status, "purged"))
	}
	return nil
}

func (a *App) printCloudDestroyResult(result *verdacloud.DestroyResult, jsonOutput bool) error {
	if jsonOutput {
		enc := json.NewEncoder(a.stdout())
		enc.SetIndent("", "  ")
		return enc.Encode(result)
	}

	ids := strings.Join(result.Request.InstanceIDs, ", ")
	if result.DryRun {
		fmt.Fprintf(a.stdout(), "dry run: would destroy Verda instance%s %s\n", pluralSuffix(result.Request.InstanceIDs), ids)
	} else {
		fmt.Fprintf(a.stdout(), "destroy requested for Verda instance%s %s\n", pluralSuffix(result.Request.InstanceIDs), ids)
	}
	fmt.Fprintf(a.stdout(), "delete permanently: %t\n", result.Request.DeletePermanently)
	if result.SourceOSVolumeID != "" {
		fmt.Fprintf(a.stdout(), "protected source os volume: %s\n", result.SourceOSVolumeID)
	}
	if len(result.Request.VolumeIDs) > 0 {
		fmt.Fprintf(a.stdout(), "volume ids: %s\n", strings.Join(result.Request.VolumeIDs, ", "))
	}
	for _, actionResult := range result.Results {
		fmt.Fprintf(a.stdout(), "%s: %s\n", actionResult.InstanceID, firstNonEmpty(actionResult.Status, "accepted"))
	}
	return nil
}

func (a *App) printCloudInstances(instances []verdacloud.Instance, jsonOutput bool) error {
	if jsonOutput {
		enc := json.NewEncoder(a.stdout())
		enc.SetIndent("", "  ")
		return enc.Encode(instances)
	}
	if len(instances) == 0 {
		fmt.Fprintln(a.stdout(), "no Verda instances found")
		return nil
	}

	w := tabwriter.NewWriter(a.stdout(), 0, 0, 2, ' ', 0)
	fmt.Fprintln(w, "ID\tHOSTNAME\tSTATUS\tTYPE\tLOCATION\tSPOT\tIP")
	for _, instance := range instances {
		ip := ""
		if instance.IP != nil {
			ip = *instance.IP
		}
		fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%s\t%t\t%s\n",
			instance.ID,
			instance.Hostname,
			instance.Status,
			instance.InstanceType,
			instance.Location,
			instance.IsSpot,
			ip,
		)
	}
	return w.Flush()
}

func (a *App) printCloudPricing(prices []verdacloud.InstancePrice, jsonOutput bool) error {
	if jsonOutput {
		enc := json.NewEncoder(a.stdout())
		enc.SetIndent("", "  ")
		return enc.Encode(prices)
	}
	if len(prices) == 0 {
		fmt.Fprintln(a.stdout(), "no Verda prices matched")
		return nil
	}

	w := tabwriter.NewWriter(a.stdout(), 0, 0, 2, ' ', 0)
	fmt.Fprintln(w, "LOCATION\tMARKET\tINSTANCE TYPE\tGPUS\tMODEL\tPRICE/H\tCURRENCY\tAVAILABLE")
	for _, price := range prices {
		fmt.Fprintf(w, "%s\t%s\t%s\t%d\t%s\t%s\t%s\t%t\n",
			price.LocationCode,
			price.Market,
			price.InstanceType,
			price.GPUCount,
			firstNonEmpty(price.Model, price.DisplayName),
			formatCloudPrice(price),
			price.Currency,
			price.Available,
		)
	}
	return w.Flush()
}

func (a *App) printCloudDeployResult(result *verdacloud.DeployResult, jsonOutput bool) error {
	if jsonOutput {
		enc := json.NewEncoder(a.stdout())
		enc.SetIndent("", "  ")
		return enc.Encode(result)
	}

	if result.DryRun {
		location := result.Policy.LocationCode
		if location == "" {
			location = result.Policy.LocationSelection
		}
		fmt.Fprintf(a.stdout(), "dry run: would deploy %s as a %s instance in %s\n", result.Request.InstanceType, result.Policy.Market, location)
		fmt.Fprintf(a.stdout(), "hostname: %s\n", result.Request.Hostname)
		if result.Request.Image != "" {
			fmt.Fprintf(a.stdout(), "image: %s\n", result.Request.Image)
		}
		if result.SourceOSVolumeID != "" {
			fmt.Fprintf(a.stdout(), "source os volume: %s\n", result.SourceOSVolumeID)
		}
		if result.SourceOSVolumeName != "" {
			fmt.Fprintf(a.stdout(), "source os volume name: %s\n", result.SourceOSVolumeName)
		}
		if result.Policy.SourceOSVolumeLocked {
			if result.Policy.LocationCode == "" {
				fmt.Fprintf(a.stdout(), "location locked by source os volume: resolved during deploy\n")
			} else {
				fmt.Fprintf(a.stdout(), "location locked by source os volume: %s\n", result.Policy.LocationCode)
			}
		}
		if result.OSVolumeClone != nil {
			if result.OSVolumeClone.Reused {
				fmt.Fprintf(a.stdout(), "reused cloned os volume: %s\n", result.OSVolumeClone.VolumeID)
			} else {
				fmt.Fprintf(a.stdout(), "cloned os volume name: %s\n", result.OSVolumeClone.Name)
			}
		}
		if result.StartupScript != nil {
			fmt.Fprintf(a.stdout(), "startup script: embedded userdata\n")
		} else {
			fmt.Fprintf(a.stdout(), "startup script: none\n")
		}
		return nil
	}

	instance := result.Instance
	if instance == nil {
		return fmt.Errorf("deploy result did not include an instance")
	}
	fmt.Fprintf(a.stdout(), "created Verda instance %s\n", instance.ID)
	fmt.Fprintf(a.stdout(), "hostname: %s\n", instance.Hostname)
	fmt.Fprintf(a.stdout(), "status: %s\n", instance.Status)
	fmt.Fprintf(a.stdout(), "instance type: %s\n", instance.InstanceType)
	fmt.Fprintf(a.stdout(), "location: %s\n", result.Policy.LocationCode)
	fmt.Fprintf(a.stdout(), "market: %s\n", result.Policy.Market)
	if result.Policy.SourceOSVolumeLocked {
		fmt.Fprintf(a.stdout(), "location locked by source os volume: %s\n", result.Policy.LocationCode)
	}
	if result.Placement != nil && result.Placement.PriceKnown {
		currency := result.Placement.Currency
		if currency == "" {
			currency = "unknown currency"
		}
		fmt.Fprintf(a.stdout(), "price: %.4f %s/hour\n", result.Placement.SpotPrice, currency)
	}
	fmt.Fprintf(a.stdout(), "spot: %t\n", instance.IsSpot)
	if instance.IP != nil && *instance.IP != "" {
		fmt.Fprintf(a.stdout(), "ip: %s\n", *instance.IP)
	}
	if result.OSVolumeClone != nil {
		fmt.Fprintf(a.stdout(), "source os volume: %s\n", result.OSVolumeClone.SourceVolumeID)
		if result.SourceOSVolumeName != "" {
			fmt.Fprintf(a.stdout(), "source os volume name: %s\n", result.SourceOSVolumeName)
		}
		if result.SourceOSVolume != nil {
			fmt.Fprintf(a.stdout(), "source os volume location: %s\n", result.SourceOSVolume.Location)
		}
		if result.OSVolumeClone.Reused {
			fmt.Fprintf(a.stdout(), "reused cloned os volume: %s\n", result.OSVolumeClone.VolumeID)
		} else {
			fmt.Fprintf(a.stdout(), "cloned os volume: %s\n", result.OSVolumeClone.VolumeID)
		}
	}
	if result.StartupScript != nil {
		fmt.Fprintf(a.stdout(), "startup script: %s\n", result.StartupScript.ID)
	}
	return nil
}

func activeCloudInstances(instances []verdacloud.Instance) []verdacloud.Instance {
	active := instances[:0]
	for _, instance := range instances {
		if isActiveCloudStatus(instance.Status) {
			active = append(active, instance)
		}
	}
	return active
}

func isActiveCloudStatus(status string) bool {
	switch strings.ToLower(strings.TrimSpace(status)) {
	case "new", "ordered", "provisioning", "validating", "running", "pending":
		return true
	default:
		return false
	}
}

func sortUniqueStrings(values []string) []string {
	values = append([]string(nil), values...)
	for i := range values {
		values[i] = strings.TrimSpace(values[i])
	}
	sort.Strings(values)
	unique := values[:0]
	for _, value := range values {
		if value == "" {
			continue
		}
		if len(unique) == 0 || unique[len(unique)-1] != value {
			unique = append(unique, value)
		}
	}
	return unique
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return "-"
}

func pluralSuffix[T any](values []T) string {
	if len(values) == 1 {
		return ""
	}
	return "s"
}

func resolveCloudPricingGPUCounts(opts cloudPricingOptions, singleGPUChanged, gpuCountChanged bool) ([]int, error) {
	if opts.allGPU {
		if gpuCountChanged {
			return nil, fmt.Errorf("--all-gpu-counts cannot be combined with --gpu-count")
		}
		if singleGPUChanged && opts.singleGPU {
			return nil, fmt.Errorf("--all-gpu-counts cannot be combined with --single-gpu")
		}
		return nil, nil
	}
	if gpuCountChanged {
		return sortUniqueInts(opts.gpuCounts), nil
	}
	if singleGPUChanged {
		if opts.singleGPU {
			return []int{1}, nil
		}
		return nil, nil
	}
	return []int{1}, nil
}

func sortUniqueInts(values []int) []int {
	values = append([]int(nil), values...)
	sort.Ints(values)
	unique := values[:0]
	for _, value := range values {
		if len(unique) == 0 || unique[len(unique)-1] != value {
			unique = append(unique, value)
		}
	}
	return unique
}

func formatCloudPrice(price verdacloud.InstancePrice) string {
	if !price.PriceKnown {
		return "-"
	}
	return fmt.Sprintf("%.4f", price.PricePerHour)
}

func formatCloudVolumeSize(size int) string {
	if size == 0 {
		return "-"
	}
	return fmt.Sprintf("%d", size)
}
