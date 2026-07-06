package verdacloud

import (
	"sort"
	"strings"
)

type SpotPrice struct {
	InstanceType string  `json:"instance_type"`
	DisplayName  string  `json:"display_name,omitempty"`
	Model        string  `json:"model,omitempty"`
	Manufacturer string  `json:"manufacturer,omitempty"`
	GPUCount     int     `json:"gpu_count"`
	LocationCode string  `json:"location_code"`
	SpotPrice    float64 `json:"spot_price,omitempty"`
	PriceKnown   bool    `json:"price_known"`
	Currency     string  `json:"currency,omitempty"`
	Available    bool    `json:"available"`
}

type PricingFilters struct {
	InstanceTypes []string
	LocationCodes []string
	Model         string
	Manufacturer  string
	SingleGPU     bool
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
	filters.Currency = strings.TrimSpace(filters.Currency)
	return filters
}

func SpotPriceMatchesFilters(price SpotPrice, filters PricingFilters) bool {
	filters = NormalizePricingFilters(filters)
	if filters.AvailableOnly && !price.Available {
		return false
	}
	if filters.SingleGPU && price.GPUCount != 1 {
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

func SortSpotPrices(prices []SpotPrice, sortBy string) {
	sortBy = strings.TrimSpace(strings.ToLower(sortBy))
	sort.SliceStable(prices, func(i, j int) bool {
		left := prices[i]
		right := prices[j]
		switch sortBy {
		case "instance-type", "type":
			if !strings.EqualFold(left.InstanceType, right.InstanceType) {
				return left.InstanceType < right.InstanceType
			}
			return left.LocationCode < right.LocationCode
		case "location", "location-code", "zone":
			if left.LocationCode != right.LocationCode {
				return left.LocationCode < right.LocationCode
			}
			return left.InstanceType < right.InstanceType
		default:
			if left.PriceKnown != right.PriceKnown {
				return left.PriceKnown
			}
			if left.PriceKnown && left.SpotPrice != right.SpotPrice {
				return left.SpotPrice < right.SpotPrice
			}
			if left.LocationCode != right.LocationCode {
				return left.LocationCode < right.LocationCode
			}
			return left.InstanceType < right.InstanceType
		}
	})
}

func containsFold(values []string, needle string) bool {
	for _, value := range values {
		if strings.EqualFold(value, needle) {
			return true
		}
	}
	return false
}
