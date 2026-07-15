package verda

import (
	"fmt"
	"sort"
	"strings"
)

const (
	PricingMarketSpot     = "spot"
	PricingMarketOnDemand = "on-demand"
	PricingMarketAll      = "all"
)

type InstancePrice struct {
	InstanceType string  `json:"instance_type"`
	DisplayName  string  `json:"display_name,omitempty"`
	Model        string  `json:"model,omitempty"`
	Manufacturer string  `json:"manufacturer,omitempty"`
	GPUCount     int     `json:"gpu_count"`
	LocationCode string  `json:"location_code"`
	Market       string  `json:"market"`
	IsSpot       bool    `json:"is_spot"`
	PricePerHour float64 `json:"price_per_hour,omitempty"`
	PriceKnown   bool    `json:"price_known"`
	Currency     string  `json:"currency,omitempty"`
	Available    bool    `json:"available"`
}

type PricingFilters struct {
	InstanceTypes []string
	LocationCodes []string
	Model         string
	Manufacturer  string
	GPUCounts     []int
	Market        string
	AvailableOnly bool
	Currency      string
}

func NormalizePricingFilters(filters PricingFilters) PricingFilters {
	filters.InstanceTypes = compactStrings(filters.InstanceTypes)
	for i := range filters.InstanceTypes {
		filters.InstanceTypes[i] = strings.ToUpper(filters.InstanceTypes[i])
	}
	filters.LocationCodes = compactStrings(filters.LocationCodes)
	for i := range filters.LocationCodes {
		filters.LocationCodes[i] = strings.ToUpper(filters.LocationCodes[i])
	}
	filters.Model = strings.ToLower(strings.TrimSpace(filters.Model))
	filters.Manufacturer = strings.ToLower(strings.TrimSpace(filters.Manufacturer))
	filters.GPUCounts = compactInts(filters.GPUCounts)
	sort.Ints(filters.GPUCounts)
	filters.Market = normalizePricingMarketName(filters.Market)
	filters.Currency = strings.TrimSpace(filters.Currency)
	return filters
}

func ValidatePricingFilters(filters PricingFilters) error {
	filters = NormalizePricingFilters(filters)
	switch filters.Market {
	case PricingMarketSpot, PricingMarketOnDemand, PricingMarketAll:
	default:
		return fmt.Errorf("pricing market must be spot, on-demand, or all")
	}
	for _, count := range filters.GPUCounts {
		if count < 0 {
			return fmt.Errorf("gpu count filter must be zero or greater")
		}
	}
	return nil
}

func InstancePriceMatchesFilters(price InstancePrice, filters PricingFilters) bool {
	filters = NormalizePricingFilters(filters)
	if filters.AvailableOnly && !price.Available {
		return false
	}
	if len(filters.GPUCounts) > 0 && !containsInt(filters.GPUCounts, price.GPUCount) {
		return false
	}
	if filters.Market != PricingMarketAll && filters.Market != "" && price.Market != filters.Market {
		return false
	}
	if len(filters.InstanceTypes) > 0 && !containsFold(filters.InstanceTypes, price.InstanceType) {
		return false
	}
	if len(filters.LocationCodes) > 0 && !containsFold(filters.LocationCodes, price.LocationCode) {
		return false
	}
	if filters.Model != "" {
		haystack := strings.ToLower(strings.Join([]string{price.Model, price.DisplayName, price.InstanceType}, " "))
		if !strings.Contains(haystack, filters.Model) {
			return false
		}
	}
	if filters.Manufacturer != "" && !strings.Contains(strings.ToLower(price.Manufacturer), filters.Manufacturer) {
		return false
	}
	return true
}

func SortInstancePrices(prices []InstancePrice, sortBy string) {
	sortBy = strings.TrimSpace(strings.ToLower(sortBy))
	sort.SliceStable(prices, func(i, j int) bool {
		left := prices[i]
		right := prices[j]
		switch sortBy {
		case "instance-type", "type":
			if !strings.EqualFold(left.InstanceType, right.InstanceType) {
				return left.InstanceType < right.InstanceType
			}
			if left.LocationCode != right.LocationCode {
				return left.LocationCode < right.LocationCode
			}
			return left.Market < right.Market
		case "location", "location-code", "zone":
			if left.LocationCode != right.LocationCode {
				return left.LocationCode < right.LocationCode
			}
			if left.InstanceType != right.InstanceType {
				return left.InstanceType < right.InstanceType
			}
			return left.Market < right.Market
		case "market", "contract":
			if left.Market != right.Market {
				return left.Market < right.Market
			}
			if left.PriceKnown != right.PriceKnown {
				return left.PriceKnown
			}
			if left.PriceKnown && left.PricePerHour != right.PricePerHour {
				return left.PricePerHour < right.PricePerHour
			}
			return left.InstanceType < right.InstanceType
		default:
			if left.PriceKnown != right.PriceKnown {
				return left.PriceKnown
			}
			if left.PriceKnown && left.PricePerHour != right.PricePerHour {
				return left.PricePerHour < right.PricePerHour
			}
			if left.LocationCode != right.LocationCode {
				return left.LocationCode < right.LocationCode
			}
			if left.InstanceType != right.InstanceType {
				return left.InstanceType < right.InstanceType
			}
			return left.Market < right.Market
		}
	})
}

func pricingMarkets(market string) []string {
	switch normalizePricingMarketName(market) {
	case PricingMarketAll:
		return []string{PricingMarketSpot, PricingMarketOnDemand}
	case PricingMarketOnDemand:
		return []string{PricingMarketOnDemand}
	default:
		return []string{PricingMarketSpot}
	}
}

func normalizePricingMarketName(market string) string {
	switch strings.ToLower(strings.TrimSpace(market)) {
	case "", PricingMarketSpot:
		return PricingMarketSpot
	case "on-demand", "ondemand", "on_demand", "fixed":
		return PricingMarketOnDemand
	case PricingMarketAll, "any":
		return PricingMarketAll
	default:
		return strings.ToLower(strings.TrimSpace(market))
	}
}

func containsFold(values []string, needle string) bool {
	for _, value := range values {
		if strings.EqualFold(value, needle) {
			return true
		}
	}
	return false
}

func containsInt(values []int, needle int) bool {
	for _, value := range values {
		if value == needle {
			return true
		}
	}
	return false
}

func compactInts(values []int) []int {
	seen := map[int]bool{}
	var compact []int
	for _, value := range values {
		if seen[value] {
			continue
		}
		seen[value] = true
		compact = append(compact, value)
	}
	return compact
}
