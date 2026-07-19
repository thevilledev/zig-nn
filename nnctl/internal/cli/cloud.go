package cli

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"text/tabwriter"

	cloudcore "nnctl/internal/cloud"
	"nnctl/internal/cloud/verda"
)

type cloudDeployOptions struct {
	verda.DeployOptions
	userDataFile string
	baseURL      string
	jsonOutput   bool
	provider     string
}

type cloudSSHKeysOptions struct {
	baseURL    string
	jsonOutput bool
	provider   string
}

type cloudVolumesOptions struct {
	verda.PurgeVolumesOptions
	includeDeleted bool
	purge          bool
	baseURL        string
	jsonOutput     bool
	provider       string
}

type cloudDestroyOptions struct {
	verda.DestroyOptions
	baseURL    string
	jsonOutput bool
	provider   string
}

type cloudPackerTemplateOptions struct {
	force    bool
	provider string
}

type cloudListOptions struct {
	baseURL    string
	status     string
	all        bool
	jsonOutput bool
	provider   string
}

type cloudPricingOptions struct {
	filters    verda.PricingFilters
	zones      []string
	gpuCounts  []int
	singleGPU  bool
	allGPU     bool
	baseURL    string
	sortBy     string
	jsonOutput bool
	provider   string
}

func (a *app) newVerdaSDKClientWithCredentials(ctx context.Context, baseURL string) (*verda.SDKClient, verda.Credentials, error) {
	creds, err := (verda.KeyringCredentialStore{}).Credentials(ctx)
	if err != nil {
		return nil, verda.Credentials{}, err
	}
	client, err := verda.NewSDKClient(creds, verda.ClientOptions{
		BaseURL:   baseURL,
		UserAgent: "nnctl",
	})
	if err != nil {
		return nil, verda.Credentials{}, fmt.Errorf("create Verda client: %w", err)
	}
	return client, creds, nil
}

func (a *app) runCloudDeploy(ctx context.Context, opts cloudDeployOptions) error {
	userData := opts.UserDataScript
	if opts.userDataFile != "" {
		script, err := os.ReadFile(opts.userDataFile)
		if err != nil {
			return fmt.Errorf("read userdata file: %w", err)
		}
		userData = string(script)
	}
	provider, err := a.cloudProvider(ctx, opts.provider, opts.baseURL, "nnctl", opts.DryRun)
	if err != nil {
		return err
	}
	deployment, err := provider.Deploy(ctx, cloudcore.DeployRequest{
		OfferingID:  opts.InstanceType,
		Location:    opts.LocationCode,
		Market:      opts.Market,
		Image:       opts.Image,
		Name:        opts.Hostname,
		Description: opts.Description,
		SSHKeyIDs:   opts.SSHKeyIDs,
		UserData:    userData,
		DryRun:      opts.DryRun,
		Options: map[string]string{
			verda.OptionSourceOSVolumeID:          opts.SourceOSVolumeID,
			verda.OptionSourceOSVolumeName:        opts.SourceOSVolumeName,
			verda.OptionClonedOSVolumeID:          opts.ClonedOSVolumeID,
			verda.OptionStartupScriptName:         opts.StartupScriptName,
			verda.OptionSkipAvailabilityCheck:     strconv.FormatBool(opts.SkipAvailabilityCheck),
			verda.OptionKeepClonedVolumeOnFailure: strconv.FormatBool(opts.KeepClonedOSVolumeOnFailure),
		},
	})
	if err != nil {
		return err
	}
	return a.printProviderDeployResult(deployment, opts.jsonOutput)
}

func (a *app) runCloudSSHKeys(ctx context.Context, opts cloudSSHKeysOptions) error {
	provider, err := a.cloudProvider(ctx, opts.provider, opts.baseURL, "nnctl", false)
	if err != nil {
		return err
	}
	keyProvider, ok := provider.(cloudcore.SSHKeyProvider)
	if !ok {
		return unsupportedCloudCapability(provider.Descriptor(), cloudcore.CapabilitySSHKeys)
	}
	keys, err := keyProvider.SSHKeys(ctx)
	if err != nil {
		return err
	}
	return a.printProviderSSHKeys(provider.Descriptor(), keys, opts.jsonOutput)
}

func (a *app) runCloudVolumes(ctx context.Context, opts cloudVolumesOptions) error {
	provider, err := a.cloudProvider(ctx, opts.provider, opts.baseURL, "nnctl", opts.DryRun)
	if err != nil {
		return err
	}
	volumeProvider, ok := provider.(cloudcore.VolumeProvider)
	if !ok {
		return unsupportedCloudCapability(provider.Descriptor(), cloudcore.CapabilityVolumes)
	}
	if opts.purge || opts.AllDeleted {
		result, err := volumeProvider.PurgeVolumes(ctx, cloudcore.VolumePurgeRequest{
			IDs: opts.VolumeIDs, AllDeleted: opts.AllDeleted, DryRun: opts.DryRun,
		})
		if err != nil {
			return err
		}
		return a.printProviderPurgeResult(result, opts.jsonOutput)
	}
	volumes, err := volumeProvider.Volumes(ctx, cloudcore.VolumeListOptions{
		IncludeDeleted: opts.includeDeleted,
	})
	if err != nil {
		return err
	}
	return a.printProviderVolumes(provider.Descriptor(), volumes, opts.jsonOutput)
}

func (a *app) runCloudDestroy(ctx context.Context, opts cloudDestroyOptions) error {
	provider, err := a.cloudProvider(ctx, opts.provider, opts.baseURL, "nnctl", opts.DryRun)
	if err != nil {
		return err
	}
	resources := make([]cloudcore.ResourceRef, 0, len(opts.VolumeIDs)+1)
	if opts.SourceOSVolumeID != "" {
		resources = append(resources, cloudcore.ResourceRef{Kind: verda.ResourceOSVolume, ID: opts.SourceOSVolumeID, Preserve: true})
	}
	for _, id := range opts.VolumeIDs {
		resources = append(resources, cloudcore.ResourceRef{Kind: verda.ResourceOSVolume, ID: id})
	}
	result, err := provider.Destroy(ctx, cloudcore.DestroyRequest{
		InstanceIDs: opts.InstanceIDs,
		Resources:   resources,
		Permanent:   opts.DeletePermanently,
		DryRun:      opts.DryRun,
	})
	if err != nil {
		return err
	}
	return a.printProviderDestroyResult(result, opts.jsonOutput)
}

func (a *app) runCloudPackerTemplate(outputDir string, opts cloudPackerTemplateOptions) error {
	factory, err := a.cloudProviderFactory(opts.provider)
	if err != nil {
		return err
	}
	templates, ok := factory.(cloudcore.ImageTemplateFactory)
	if !ok {
		return unsupportedCloudCapability(factory.Descriptor(), cloudcore.CapabilityImageTemplate)
	}
	written, err := writeCloudTemplateFiles(outputDir, opts.force, templates.ImageTemplateFiles())
	if err != nil {
		return err
	}
	output := newErrorWriter(a.stdout())
	for _, path := range written {
		output.printf("wrote %s\n", path)
	}
	if err := output.Err(); err != nil {
		return fmt.Errorf("write Packer template paths: %w", err)
	}
	return nil
}

func (a *app) runCloudList(ctx context.Context, opts cloudListOptions) error {
	provider, err := a.cloudProvider(ctx, opts.provider, opts.baseURL, "nnctl", false)
	if err != nil {
		return err
	}
	instances, err := provider.Instances(ctx, cloudcore.ListOptions{Status: opts.status, All: opts.all})
	if err != nil {
		return err
	}
	return a.printProviderInstances(provider.Descriptor(), instances, opts.jsonOutput)
}

func writeCloudTemplateFiles(outputDir string, force bool, files []cloudcore.TemplateFile) ([]string, error) {
	outputDir = strings.TrimSpace(outputDir)
	if outputDir == "" {
		return nil, fmt.Errorf("output directory is required")
	}
	if err := os.MkdirAll(outputDir, 0o700); err != nil {
		return nil, fmt.Errorf("create Packer template directory: %w", err)
	}

	var written []string
	for _, file := range files {
		path := filepath.Join(outputDir, file.Path)
		flag := os.O_WRONLY | os.O_CREATE
		if force {
			flag |= os.O_TRUNC
		} else {
			flag |= os.O_EXCL
		}
		handle, err := os.OpenFile(path, flag, 0o600)
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

func (a *app) runCloudPricing(ctx context.Context, opts cloudPricingOptions) error {
	opts.filters = verda.NormalizePricingFilters(opts.filters)
	if err := verda.ValidatePricingFilters(opts.filters); err != nil {
		return err
	}

	provider, err := a.cloudProvider(ctx, opts.provider, opts.baseURL, "nnctl", false)
	if err != nil {
		return err
	}
	markets := []string{opts.filters.Market}
	if opts.filters.Market == verda.PricingMarketAll {
		markets = nil
	}
	offerings, err := provider.Offerings(ctx, cloudcore.OfferingFilters{
		IDs:           opts.filters.InstanceTypes,
		Locations:     opts.filters.LocationCodes,
		Markets:       markets,
		Model:         opts.filters.Model,
		Manufacturer:  opts.filters.Manufacturer,
		GPUCounts:     opts.filters.GPUCounts,
		AvailableOnly: opts.filters.AvailableOnly,
		Currency:      opts.filters.Currency,
	})
	if err != nil {
		return err
	}
	return a.printProviderOfferings(provider.Descriptor(), offerings, opts.sortBy, opts.jsonOutput)
}

func (a *app) printCloudSSHKeys(keys []verda.SSHKey, jsonOutput bool) error {
	if jsonOutput {
		enc := json.NewEncoder(a.stdout())
		enc.SetIndent("", "  ")
		return enc.Encode(keys)
	}
	if len(keys) == 0 {
		if _, err := fmt.Fprintln(a.stdout(), "no Verda SSH keys found"); err != nil {
			return fmt.Errorf("write Verda SSH keys: %w", err)
		}
		return nil
	}

	w := tabwriter.NewWriter(a.stdout(), 0, 0, 2, ' ', 0)
	output := newErrorWriter(w)
	output.printf("ID\tNAME\tFINGERPRINT\n")
	for _, key := range keys {
		output.printf("%s\t%s\t%s\n", key.ID, key.Name, key.Fingerprint)
	}
	if err := output.Err(); err != nil {
		return fmt.Errorf("write Verda SSH keys: %w", err)
	}
	return w.Flush()
}

func (a *app) printCloudVolumes(volumes []verda.Volume, jsonOutput bool) error {
	if jsonOutput {
		enc := json.NewEncoder(a.stdout())
		enc.SetIndent("", "  ")
		return enc.Encode(volumes)
	}
	if len(volumes) == 0 {
		if _, err := fmt.Fprintln(a.stdout(), "no Verda volumes found"); err != nil {
			return fmt.Errorf("write Verda volumes: %w", err)
		}
		return nil
	}

	w := tabwriter.NewWriter(a.stdout(), 0, 0, 2, ' ', 0)
	output := newErrorWriter(w)
	output.printf("ID\tNAME\tSTATUS\tTYPE\tSIZE\tLOCATION\tOS\tDELETED AT\n")
	for _, volume := range volumes {
		output.printf("%s\t%s\t%s\t%s\t%s\t%s\t%t\t%s\n",
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
	if err := output.Err(); err != nil {
		return fmt.Errorf("write Verda volumes: %w", err)
	}
	return w.Flush()
}

func (a *app) printCloudPurgeResult(result *verda.PurgeVolumesResult, jsonOutput bool) error {
	if jsonOutput {
		enc := json.NewEncoder(a.stdout())
		enc.SetIndent("", "  ")
		return enc.Encode(result)
	}

	output := newErrorWriter(a.stdout())
	if result.Request.AllDeleted {
		if result.DryRun {
			output.printf("dry run: would purge all deleted Verda volumes\n")
			return cloudOutputError("purge result", output.Err())
		}
		if len(result.Results) == 0 {
			output.printf("no deleted Verda volumes found\n")
			return cloudOutputError("purge result", output.Err())
		}
		output.printf("purged all deleted Verda volumes\n")
		for _, actionResult := range result.Results {
			output.printf("%s: %s\n", actionResult.VolumeID, firstNonEmpty(actionResult.Status, "purged"))
		}
		return cloudOutputError("purge result", output.Err())
	}

	ids := strings.Join(result.Request.VolumeIDs, ", ")
	if result.DryRun {
		output.printf("dry run: would purge Verda volume%s %s\n", pluralSuffix(result.Request.VolumeIDs), ids)
		return cloudOutputError("purge result", output.Err())
	}
	output.printf("purged Verda volume%s %s\n", pluralSuffix(result.Request.VolumeIDs), ids)
	for _, actionResult := range result.Results {
		output.printf("%s: %s\n", actionResult.VolumeID, firstNonEmpty(actionResult.Status, "purged"))
	}
	return cloudOutputError("purge result", output.Err())
}

func (a *app) printCloudDestroyResult(result *verda.DestroyResult, jsonOutput bool) error {
	if jsonOutput {
		enc := json.NewEncoder(a.stdout())
		enc.SetIndent("", "  ")
		return enc.Encode(result)
	}

	output := newErrorWriter(a.stdout())
	ids := strings.Join(result.Request.InstanceIDs, ", ")
	if result.DryRun {
		output.printf("dry run: would destroy Verda instance%s %s\n", pluralSuffix(result.Request.InstanceIDs), ids)
	} else {
		output.printf("destroy requested for Verda instance%s %s\n", pluralSuffix(result.Request.InstanceIDs), ids)
	}
	output.printf("delete permanently: %t\n", result.Request.DeletePermanently)
	if result.SourceOSVolumeID != "" {
		output.printf("protected source os volume: %s\n", result.SourceOSVolumeID)
	}
	if len(result.Request.VolumeIDs) > 0 {
		output.printf("volume ids: %s\n", strings.Join(result.Request.VolumeIDs, ", "))
	}
	for _, actionResult := range result.Results {
		output.printf("%s: %s\n", actionResult.InstanceID, firstNonEmpty(actionResult.Status, "accepted"))
	}
	return cloudOutputError("destroy result", output.Err())
}

func (a *app) printCloudInstances(instances []verda.Instance, jsonOutput bool) error {
	if jsonOutput {
		enc := json.NewEncoder(a.stdout())
		enc.SetIndent("", "  ")
		return enc.Encode(instances)
	}
	if len(instances) == 0 {
		_, err := fmt.Fprintln(a.stdout(), "no Verda instances found")
		return cloudOutputError("instances", err)
	}

	w := tabwriter.NewWriter(a.stdout(), 0, 0, 2, ' ', 0)
	output := newErrorWriter(w)
	output.printf("ID\tHOSTNAME\tSTATUS\tTYPE\tLOCATION\tSPOT\tIP\n")
	for _, instance := range instances {
		ip := ""
		if instance.IP != nil {
			ip = *instance.IP
		}
		output.printf("%s\t%s\t%s\t%s\t%s\t%t\t%s\n",
			instance.ID,
			instance.Hostname,
			instance.Status,
			instance.InstanceType,
			instance.Location,
			instance.IsSpot,
			ip,
		)
	}
	if err := cloudOutputError("instances", output.Err()); err != nil {
		return err
	}
	return w.Flush()
}

func (a *app) printCloudPricing(prices []verda.InstancePrice, jsonOutput bool) error {
	if jsonOutput {
		enc := json.NewEncoder(a.stdout())
		enc.SetIndent("", "  ")
		return enc.Encode(prices)
	}
	if len(prices) == 0 {
		_, err := fmt.Fprintln(a.stdout(), "no Verda prices matched")
		return cloudOutputError("prices", err)
	}

	w := tabwriter.NewWriter(a.stdout(), 0, 0, 2, ' ', 0)
	output := newErrorWriter(w)
	output.printf("LOCATION\tMARKET\tINSTANCE TYPE\tGPUS\tMODEL\tPRICE/H\tCURRENCY\tAVAILABLE\n")
	for _, price := range prices {
		output.printf("%s\t%s\t%s\t%d\t%s\t%s\t%s\t%t\n",
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
	if err := cloudOutputError("prices", output.Err()); err != nil {
		return err
	}
	return w.Flush()
}

func cloudOutputError(name string, err error) error {
	if err == nil {
		return nil
	}
	return fmt.Errorf("write Verda %s: %w", name, err)
}

func (a *app) printCloudDeployResult(result *verda.DeployResult, jsonOutput bool) error {
	if jsonOutput {
		enc := json.NewEncoder(a.stdout())
		enc.SetIndent("", "  ")
		return enc.Encode(result)
	}

	out := newErrorWriter(a.stdout())
	if result.DryRun {
		printCloudDeployDryRun(out, result)
		return out.Err()
	}

	instance := result.Instance
	if instance == nil {
		return fmt.Errorf("deploy result did not include an instance")
	}
	printCloudDeployInstance(out, result, instance)
	return out.Err()
}

func printCloudDeployDryRun(out *errorWriter, result *verda.DeployResult) {
	location := firstNonEmpty(result.Policy.LocationCode, result.Policy.LocationSelection)
	out.printf("dry run: would deploy %s as a %s instance in %s\n", result.Request.InstanceType, result.Policy.Market, location)
	out.printf("hostname: %s\n", result.Request.Hostname)
	if result.Request.Image != "" {
		out.printf("image: %s\n", result.Request.Image)
	}
	if result.SourceOSVolumeID != "" {
		out.printf("source os volume: %s\n", result.SourceOSVolumeID)
	}
	if result.SourceOSVolumeName != "" {
		out.printf("source os volume name: %s\n", result.SourceOSVolumeName)
	}
	if result.Policy.SourceOSVolumeLocked {
		lockedLocation := result.Policy.LocationCode
		if lockedLocation == "" {
			lockedLocation = "resolved during deploy"
		}
		out.printf("location locked by source os volume: %s\n", lockedLocation)
	}
	if result.OSVolumeClone != nil {
		if result.OSVolumeClone.Reused {
			out.printf("reused cloned os volume: %s\n", result.OSVolumeClone.VolumeID)
		} else {
			out.printf("cloned os volume name: %s\n", result.OSVolumeClone.Name)
		}
	}
	if result.StartupScript != nil {
		out.printf("startup script: embedded userdata\n")
	} else {
		out.printf("startup script: none\n")
	}
}

func printCloudDeployInstance(out *errorWriter, result *verda.DeployResult, instance *verda.Instance) {
	out.printf("created Verda instance %s\n", instance.ID)
	out.printf("hostname: %s\n", instance.Hostname)
	out.printf("status: %s\n", instance.Status)
	out.printf("instance type: %s\n", instance.InstanceType)
	out.printf("location: %s\n", result.Policy.LocationCode)
	out.printf("market: %s\n", result.Policy.Market)
	if result.Policy.SourceOSVolumeLocked {
		out.printf("location locked by source os volume: %s\n", result.Policy.LocationCode)
	}
	printCloudDeployPrice(out, result.Placement)
	out.printf("spot: %t\n", instance.IsSpot)
	if instance.IP != nil && *instance.IP != "" {
		out.printf("ip: %s\n", *instance.IP)
	}
	printCloudDeployVolume(out, result)
	if result.StartupScript != nil {
		out.printf("startup script: %s\n", result.StartupScript.ID)
	}
}

func printCloudDeployPrice(out *errorWriter, placement *verda.SpotPlacement) {
	if placement == nil || !placement.PriceKnown {
		return
	}
	currency := firstNonEmpty(placement.Currency, "unknown currency")
	out.printf("price: %.4f %s/hour\n", placement.SpotPrice, currency)
}

func printCloudDeployVolume(out *errorWriter, result *verda.DeployResult) {
	if result.OSVolumeClone != nil {
		out.printf("source os volume: %s\n", result.OSVolumeClone.SourceVolumeID)
		if result.SourceOSVolumeName != "" {
			out.printf("source os volume name: %s\n", result.SourceOSVolumeName)
		}
		if result.SourceOSVolume != nil {
			out.printf("source os volume location: %s\n", result.SourceOSVolume.Location)
		}
		if result.OSVolumeClone.Reused {
			out.printf("reused cloned os volume: %s\n", result.OSVolumeClone.VolumeID)
		} else {
			out.printf("cloned os volume: %s\n", result.OSVolumeClone.VolumeID)
		}
	}
}

func activeCloudInstances(instances []verda.Instance) []verda.Instance {
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

func formatCloudPrice(price verda.InstancePrice) string {
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
