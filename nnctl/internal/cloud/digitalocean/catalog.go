package digitalocean

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"

	cloudcore "nnctl/internal/cloud"
)

const (
	MarketOnDemand  = "on-demand"
	CurrencyUSD     = "USD"
	AMDImage        = "gpu-amd-base"
	NVIDIAX1Image   = "gpu-h100x1-base"
	NVIDIAX8Image   = "gpu-h100x8-base"
	OptionOfferings = cloudcore.OptionOfferings
	OptionSSHKeyIDs = cloudcore.OptionSSHKeyIDs
)

type gpuPlan struct {
	Slug         string
	Manufacturer string
	Model        string
	Count        int
	MemoryMiB    int
	Contract     bool
	Price        float64
}

var gpuPlans = map[string]gpuPlan{
	"gpu-mi300x1-192gb":                    {Slug: "gpu-mi300x1-192gb", Manufacturer: "amd", Model: "MI300X", Count: 1, MemoryMiB: 192 * 1024, Price: 1.99},
	"gpu-mi300x8-1536gb":                   {Slug: "gpu-mi300x8-1536gb", Manufacturer: "amd", Model: "MI300X", Count: 8, MemoryMiB: 1536 * 1024, Price: 15.92},
	"gpu-h100x1-80gb":                      {Slug: "gpu-h100x1-80gb", Manufacturer: "nvidia", Model: "H100", Count: 1, MemoryMiB: 80 * 1024},
	"gpu-h100x8-640gb":                     {Slug: "gpu-h100x8-640gb", Manufacturer: "nvidia", Model: "H100", Count: 8, MemoryMiB: 640 * 1024},
	"gpu-h200x1-141gb":                     {Slug: "gpu-h200x1-141gb", Manufacturer: "nvidia", Model: "H200", Count: 1, MemoryMiB: 141 * 1024},
	"gpu-h200x8-1128gb":                    {Slug: "gpu-h200x8-1128gb", Manufacturer: "nvidia", Model: "H200", Count: 8, MemoryMiB: 1128 * 1024},
	"gpu-l40sx1-48gb":                      {Slug: "gpu-l40sx1-48gb", Manufacturer: "nvidia", Model: "L40S", Count: 1, MemoryMiB: 48 * 1024},
	"gpu-4000adax1-20gb":                   {Slug: "gpu-4000adax1-20gb", Manufacturer: "nvidia", Model: "RTX 4000", Count: 1, MemoryMiB: 20 * 1024},
	"gpu-6000adax1-48gb":                   {Slug: "gpu-6000adax1-48gb", Manufacturer: "nvidia", Model: "RTX 6000", Count: 1, MemoryMiB: 48 * 1024},
	"gpu-mi300x1-192gb-contracted":         {Slug: "gpu-mi300x1-192gb-contracted", Manufacturer: "amd", Model: "MI300X", Count: 1, MemoryMiB: 192 * 1024, Contract: true, Price: 1.99},
	"gpu-mi300x8-1536gb-contracted":        {Slug: "gpu-mi300x8-1536gb-contracted", Manufacturer: "amd", Model: "MI300X", Count: 8, MemoryMiB: 1536 * 1024, Contract: true, Price: 15.92},
	"gpu-mi325x1-256gb-contracted":         {Slug: "gpu-mi325x1-256gb-contracted", Manufacturer: "amd", Model: "MI325X", Count: 1, MemoryMiB: 256 * 1024, Contract: true},
	"gpu-mi325x8-2048gb-contracted":        {Slug: "gpu-mi325x8-2048gb-contracted", Manufacturer: "amd", Model: "MI325X", Count: 8, MemoryMiB: 2048 * 1024, Contract: true},
	"gpu-mi350x1-288gb-contracted":         {Slug: "gpu-mi350x1-288gb-contracted", Manufacturer: "amd", Model: "MI350X", Count: 1, MemoryMiB: 288 * 1024, Contract: true},
	"gpu-mi350x8-2304gb-contracted":        {Slug: "gpu-mi350x8-2304gb-contracted", Manufacturer: "amd", Model: "MI350X", Count: 8, MemoryMiB: 2304 * 1024, Contract: true},
	"gpu-b300x1-288gb-contracted":          {Slug: "gpu-b300x1-288gb-contracted", Manufacturer: "nvidia", Model: "B300", Count: 1, MemoryMiB: 288 * 1024, Contract: true},
	"gpu-b300x8-2304gb-contracted":         {Slug: "gpu-b300x8-2304gb-contracted", Manufacturer: "nvidia", Model: "B300", Count: 8, MemoryMiB: 2304 * 1024, Contract: true},
	"gpu-h100x8-640gb-contracted":          {Slug: "gpu-h100x8-640gb-contracted", Manufacturer: "nvidia", Model: "H100", Count: 8, MemoryMiB: 640 * 1024, Contract: true},
	"gpu-h200x8-1128gb-contracted":         {Slug: "gpu-h200x8-1128gb-contracted", Manufacturer: "nvidia", Model: "H200", Count: 8, MemoryMiB: 1128 * 1024, Contract: true},
	"gpu-mi300x8-1536gb-fabric-contracted": {Slug: "gpu-mi300x8-1536gb-fabric-contracted", Manufacturer: "amd", Model: "MI300X", Count: 8, MemoryMiB: 1536 * 1024, Contract: true},
	"gpu-mi325x8-2048gb-fabric-contracted": {Slug: "gpu-mi325x8-2048gb-fabric-contracted", Manufacturer: "amd", Model: "MI325X", Count: 8, MemoryMiB: 2048 * 1024, Contract: true},
	"gpu-mi350x8-2304gb-fabric-contracted": {Slug: "gpu-mi350x8-2304gb-fabric-contracted", Manufacturer: "amd", Model: "MI350X", Count: 8, MemoryMiB: 2304 * 1024, Contract: true},
	"gpu-h200x8-1128gb-fabric-contracted":  {Slug: "gpu-h200x8-1128gb-fabric-contracted", Manufacturer: "nvidia", Model: "H200", Count: 8, MemoryMiB: 1128 * 1024, Contract: true},
}

var gpuSlugPattern = regexp.MustCompile(`^gpu-([a-z0-9]+)x([0-9]+)-([0-9]+)gb(?:-[a-z0-9-]+)?$`)

func planForSlug(slug string) (gpuPlan, error) {
	slug = strings.ToLower(strings.TrimSpace(slug))
	if plan, ok := gpuPlans[slug]; ok {
		return plan, nil
	}
	matches := gpuSlugPattern.FindStringSubmatch(slug)
	if len(matches) != 4 {
		return gpuPlan{}, fmt.Errorf("DigitalOcean offering %q is not a GPU Droplet size slug", slug)
	}
	count, countErr := strconv.Atoi(matches[2])
	memoryGiB, memoryErr := strconv.Atoi(matches[3])
	if countErr != nil || memoryErr != nil || count <= 0 || memoryGiB <= 0 {
		return gpuPlan{}, fmt.Errorf("DigitalOcean offering %q has invalid GPU metadata", slug)
	}
	manufacturer, model := acceleratorFromModel(matches[1])
	if manufacturer == "" {
		return gpuPlan{}, fmt.Errorf("DigitalOcean offering %q has an unknown GPU model", slug)
	}
	return gpuPlan{
		Slug: slug, Manufacturer: manufacturer, Model: model,
		Count: count, MemoryMiB: memoryGiB * 1024,
		Contract: strings.Contains(slug, "contracted"),
	}, nil
}

func planFromSize(size Size) (gpuPlan, error) {
	plan, known := gpuPlans[strings.ToLower(strings.TrimSpace(size.Slug))]
	if !known {
		var err error
		plan, err = planForSlug(size.Slug)
		if err != nil {
			if size.GPUInfo == nil {
				return gpuPlan{}, err
			}
			manufacturer, model := acceleratorFromModel(size.GPUInfo.Model)
			if manufacturer == "" {
				return gpuPlan{}, err
			}
			plan = gpuPlan{Slug: strings.ToLower(strings.TrimSpace(size.Slug)), Manufacturer: manufacturer, Model: model}
		}
	}
	if size.GPUInfo == nil || size.GPUInfo.Count <= 0 {
		return gpuPlan{}, fmt.Errorf("DigitalOcean size %q does not include GPU metadata", size.Slug)
	}
	manufacturer, model := acceleratorFromModel(size.GPUInfo.Model)
	if manufacturer != "" {
		plan.Manufacturer = manufacturer
		plan.Model = model
	}
	plan.Count = size.GPUInfo.Count
	if memory := unitValueMiB(size.GPUInfo.VRAM); memory > 0 {
		plan.MemoryMiB = memory
	}
	plan.Price = size.PriceHourly
	return plan, nil
}

func acceleratorFromModel(value string) (manufacturer, model string) {
	normalized := strings.ToLower(strings.NewReplacer("-", "_", " ", "_").Replace(strings.TrimSpace(value)))
	normalized = strings.TrimPrefix(normalized, "gpu_")
	switch {
	case strings.Contains(normalized, "mi300x"):
		return "amd", "MI300X"
	case strings.Contains(normalized, "mi325x"):
		return "amd", "MI325X"
	case strings.Contains(normalized, "mi350x"):
		return "amd", "MI350X"
	case strings.Contains(normalized, "h100"):
		return "nvidia", "H100"
	case strings.Contains(normalized, "h200"):
		return "nvidia", "H200"
	case strings.Contains(normalized, "b300"):
		return "nvidia", "B300"
	case strings.Contains(normalized, "l40s"):
		return "nvidia", "L40S"
	case strings.Contains(normalized, "4000ada"):
		return "nvidia", "RTX 4000"
	case strings.Contains(normalized, "6000ada"):
		return "nvidia", "RTX 6000"
	default:
		return "", ""
	}
}

func unitValueMiB(value UnitValue) int {
	switch strings.ToLower(strings.TrimSpace(value.Unit)) {
	case "gib", "gb":
		return value.Amount * 1024
	case "mib", "mb":
		return value.Amount
	default:
		return 0
	}
}

func imageForPlan(plan gpuPlan) string {
	if plan.Manufacturer == "amd" {
		return AMDImage
	}
	if plan.Count > 1 {
		return NVIDIAX8Image
	}
	return NVIDIAX1Image
}

func backendsForPlan(plan gpuPlan) []string {
	if plan.Manufacturer == "amd" {
		return []string{"cpu", "rocm"}
	}
	return []string{"cpu", "cuda"}
}

func offeringFromPlan(plan gpuPlan, location string, available, discoverable bool) cloudcore.Offering {
	return cloudcore.Offering{
		Provider: ProviderName, ID: plan.Slug, DisplayName: plan.Model,
		Location: strings.ToLower(strings.TrimSpace(location)), Market: MarketOnDemand,
		Accelerator: cloudcore.Accelerator{
			Manufacturer: plan.Manufacturer, Model: plan.Model,
			Count: plan.Count, MemoryMiB: plan.MemoryMiB,
		},
		Backends: backendsForPlan(plan), PricePerHour: plan.Price,
		PriceKnown: plan.Price > 0, Currency: CurrencyUSD,
		Available: available, Discoverable: discoverable,
	}
}

func parseManualOfferings(raw string) ([]cloudcore.Offering, error) {
	if strings.TrimSpace(raw) == "" {
		return nil, nil
	}
	var result []cloudcore.Offering
	seen := make(map[string]struct{})
	for _, item := range strings.Split(raw, ",") {
		parts := strings.Split(strings.TrimSpace(item), "@")
		if len(parts) != 2 || strings.TrimSpace(parts[0]) == "" || strings.TrimSpace(parts[1]) == "" {
			return nil, fmt.Errorf("DigitalOcean offering %q must use SIZE@REGION", item)
		}
		plan, err := planForSlug(parts[0])
		if err != nil {
			return nil, err
		}
		key := plan.Slug + "@" + strings.ToLower(strings.TrimSpace(parts[1]))
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		result = append(result, offeringFromPlan(plan, parts[1], true, false))
	}
	return result, nil
}
