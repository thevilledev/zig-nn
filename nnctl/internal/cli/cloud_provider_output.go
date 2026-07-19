package cli

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"text/tabwriter"

	cloudcore "nnctl/internal/cloud"
	"nnctl/internal/cloud/verda"
)

func unsupportedCloudCapability(descriptor cloudcore.Descriptor, capability cloudcore.Capability) error {
	return fmt.Errorf("cloud provider %q does not support %s", descriptor.Name, capability)
}

func (a *app) printProviderDeployResult(result *cloudcore.Deployment, jsonOutput bool) error {
	if result.Provider == verda.ProviderName {
		providerResult, err := decodeCloudProviderDetail[verda.DeployResult](result.ProviderDetail, result.Provider, "deploy")
		if err != nil {
			return err
		}
		return a.printCloudDeployResult(providerResult, jsonOutput)
	}
	if jsonOutput {
		return writeCloudJSON(a.stdout(), result)
	}
	output := newErrorWriter(a.stdout())
	if result.DryRun {
		output.printf("dry run: would deploy %s on %s\n", result.Request.OfferingID, result.Provider)
		output.printf("location: %s\n", firstNonEmpty(result.Request.Location))
		return output.Err()
	}
	if result.Instance == nil {
		return fmt.Errorf("%s deploy result did not include an instance", result.Provider)
	}
	output.printf("created %s instance %s\n", result.Provider, result.Instance.ID)
	output.printf("status: %s\n", result.Instance.State)
	output.printf("instance type: %s\n", result.Instance.OfferingID)
	output.printf("location: %s\n", firstNonEmpty(result.Instance.Location))
	if result.Instance.PublicIP != "" {
		output.printf("ip: %s\n", result.Instance.PublicIP)
	}
	return output.Err()
}

func (a *app) printProviderSSHKeys(
	descriptor cloudcore.Descriptor,
	keys []cloudcore.SSHKey,
	jsonOutput bool,
) error {
	if descriptor.Name == verda.ProviderName {
		providerKeys := make([]verda.SSHKey, 0, len(keys))
		for _, key := range keys {
			providerKey, err := decodeCloudProviderDetail[verda.SSHKey](key.ProviderDetail, descriptor.Name, "SSH key")
			if err != nil {
				return err
			}
			providerKeys = append(providerKeys, *providerKey)
		}
		return a.printCloudSSHKeys(providerKeys, jsonOutput)
	}
	if jsonOutput {
		return writeCloudJSON(a.stdout(), keys)
	}
	if len(keys) == 0 {
		_, err := fmt.Fprintf(a.stdout(), "no %s SSH keys found\n", descriptor.DisplayName)
		return err
	}
	w := tabwriter.NewWriter(a.stdout(), 0, 0, 2, ' ', 0)
	output := newErrorWriter(w)
	output.printf("ID\tNAME\tFINGERPRINT\n")
	for _, key := range keys {
		output.printf("%s\t%s\t%s\n", key.ID, key.Name, key.Fingerprint)
	}
	if err := output.Err(); err != nil {
		return err
	}
	return w.Flush()
}

func (a *app) printProviderVolumes(
	descriptor cloudcore.Descriptor,
	volumes []cloudcore.Volume,
	jsonOutput bool,
) error {
	if descriptor.Name == verda.ProviderName {
		providerVolumes := make([]verda.Volume, 0, len(volumes))
		for _, volume := range volumes {
			providerVolume, err := decodeCloudProviderDetail[verda.Volume](volume.ProviderDetail, descriptor.Name, "volume")
			if err != nil {
				return err
			}
			providerVolumes = append(providerVolumes, *providerVolume)
		}
		return a.printCloudVolumes(providerVolumes, jsonOutput)
	}
	if jsonOutput {
		return writeCloudJSON(a.stdout(), volumes)
	}
	if len(volumes) == 0 {
		_, err := fmt.Fprintf(a.stdout(), "no %s volumes found\n", descriptor.DisplayName)
		return err
	}
	w := tabwriter.NewWriter(a.stdout(), 0, 0, 2, ' ', 0)
	output := newErrorWriter(w)
	output.printf("ID\tNAME\tSTATUS\tSIZE\tLOCATION\tBOOTABLE\tDELETED\n")
	for _, volume := range volumes {
		output.printf("%s\t%s\t%s\t%s\t%s\t%t\t%t\n",
			volume.ID, firstNonEmpty(volume.Name), firstNonEmpty(volume.State),
			formatCloudVolumeSize(volume.SizeGiB), firstNonEmpty(volume.Location),
			volume.Bootable, volume.Deleted,
		)
	}
	if err := output.Err(); err != nil {
		return err
	}
	return w.Flush()
}

func (a *app) printProviderPurgeResult(result *cloudcore.VolumePurgeResult, jsonOutput bool) error {
	if result.Provider == verda.ProviderName {
		providerResult, err := decodeCloudProviderDetail[verda.PurgeVolumesResult](result.ProviderDetail, result.Provider, "volume purge")
		if err != nil {
			return err
		}
		return a.printCloudPurgeResult(providerResult, jsonOutput)
	}
	if jsonOutput {
		return writeCloudJSON(a.stdout(), result)
	}
	output := newErrorWriter(a.stdout())
	if result.DryRun {
		output.printf("dry run: would purge %s volumes\n", result.Provider)
	} else {
		output.printf("purged %s volume ids: %s\n", result.Provider, strings.Join(result.PurgedIDs, ", "))
	}
	return output.Err()
}

func (a *app) printProviderDestroyResult(result *cloudcore.DestroyResult, jsonOutput bool) error {
	if result.Provider == verda.ProviderName {
		providerResult, err := decodeCloudProviderDetail[verda.DestroyResult](result.ProviderDetail, result.Provider, "destroy")
		if err != nil {
			return err
		}
		return a.printCloudDestroyResult(providerResult, jsonOutput)
	}
	if jsonOutput {
		return writeCloudJSON(a.stdout(), result)
	}
	output := newErrorWriter(a.stdout())
	action := "destroy requested for"
	if result.DryRun {
		action = "dry run: would destroy"
	}
	output.printf("%s %s instance%s %s\n", action, result.Provider, pluralSuffix(result.Request.InstanceIDs), strings.Join(result.Request.InstanceIDs, ", "))
	output.printf("delete permanently: %t\n", result.Request.Permanent)
	return output.Err()
}

func (a *app) printProviderInstances(
	descriptor cloudcore.Descriptor,
	instances []cloudcore.Instance,
	jsonOutput bool,
) error {
	if descriptor.Name == verda.ProviderName {
		providerInstances := make([]verda.Instance, 0, len(instances))
		for _, instance := range instances {
			providerInstance, err := decodeCloudProviderDetail[verda.Instance](instance.ProviderDetail, descriptor.Name, "instance")
			if err != nil {
				return err
			}
			providerInstances = append(providerInstances, *providerInstance)
		}
		return a.printCloudInstances(providerInstances, jsonOutput)
	}
	if jsonOutput {
		return writeCloudJSON(a.stdout(), instances)
	}
	if len(instances) == 0 {
		_, err := fmt.Fprintf(a.stdout(), "no %s instances found\n", descriptor.DisplayName)
		return err
	}
	w := tabwriter.NewWriter(a.stdout(), 0, 0, 2, ' ', 0)
	output := newErrorWriter(w)
	output.printf("ID\tNAME\tSTATUS\tTYPE\tLOCATION\tMARKET\tIP\n")
	for _, instance := range instances {
		output.printf("%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
			instance.ID, firstNonEmpty(instance.Name), instance.State,
			instance.OfferingID, firstNonEmpty(instance.Location),
			firstNonEmpty(instance.Market), instance.PublicIP,
		)
	}
	if err := output.Err(); err != nil {
		return err
	}
	return w.Flush()
}

func (a *app) printProviderOfferings(
	descriptor cloudcore.Descriptor,
	offerings []cloudcore.Offering,
	sortBy string,
	jsonOutput bool,
) error {
	if descriptor.Name == verda.ProviderName {
		prices := make([]verda.InstancePrice, 0, len(offerings))
		for _, offering := range offerings {
			price, err := decodeCloudProviderDetail[verda.InstancePrice](offering.ProviderDetail, descriptor.Name, "offering")
			if err != nil {
				return err
			}
			prices = append(prices, *price)
		}
		verda.SortInstancePrices(prices, sortBy)
		return a.printCloudPricing(prices, jsonOutput)
	}
	sortCloudOfferings(offerings, sortBy)
	if jsonOutput {
		return writeCloudJSON(a.stdout(), offerings)
	}
	if len(offerings) == 0 {
		_, err := fmt.Fprintf(a.stdout(), "no %s offerings matched\n", descriptor.DisplayName)
		return err
	}
	w := tabwriter.NewWriter(a.stdout(), 0, 0, 2, ' ', 0)
	output := newErrorWriter(w)
	output.printf("LOCATION\tMARKET\tINSTANCE TYPE\tGPUS\tMODEL\tPRICE/H\tCURRENCY\tAVAILABLE\n")
	for _, offering := range offerings {
		output.printf("%s\t%s\t%s\t%d\t%s\t%s\t%s\t%t\n",
			offering.Location, firstNonEmpty(offering.Market), offering.ID,
			offering.Accelerator.Count, firstNonEmpty(offering.Accelerator.Model, offering.DisplayName),
			formatProviderPrice(offering), offering.Currency, offering.Available,
		)
	}
	if err := output.Err(); err != nil {
		return err
	}
	return w.Flush()
}

func sortCloudOfferings(offerings []cloudcore.Offering, sortBy string) {
	sort.SliceStable(offerings, func(left, right int) bool {
		a, b := offerings[left], offerings[right]
		switch strings.ToLower(strings.TrimSpace(sortBy)) {
		case "instance-type", "type":
			return a.ID < b.ID
		case "location", "location-code", "zone":
			return a.Location < b.Location
		case "market", "contract":
			return a.Market < b.Market
		default:
			if a.PriceKnown != b.PriceKnown {
				return a.PriceKnown
			}
			if a.PriceKnown && a.PricePerHour != b.PricePerHour {
				return a.PricePerHour < b.PricePerHour
			}
			return a.ID < b.ID
		}
	})
}

func formatProviderPrice(offering cloudcore.Offering) string {
	if !offering.PriceKnown {
		return "-"
	}
	return fmt.Sprintf("%.4f", offering.PricePerHour)
}

func writeCloudJSON(output interface{ Write([]byte) (int, error) }, value any) error {
	encoder := json.NewEncoder(output)
	encoder.SetIndent("", "  ")
	return encoder.Encode(value)
}
