package verdacloud

import "testing"

func TestInstancePriceMatchesFilters(t *testing.T) {
	price := InstancePrice{
		InstanceType: "1L40S.20V",
		DisplayName:  "L40S",
		Model:        "L40S",
		Manufacturer: "NVIDIA",
		GPUCount:     1,
		LocationCode: "FIN-02",
		Market:       PricingMarketSpot,
		IsSpot:       true,
		Available:    true,
	}

	tests := []struct {
		name    string
		filters PricingFilters
		want    bool
	}{
		{name: "empty", filters: PricingFilters{}, want: true},
		{name: "instance", filters: PricingFilters{InstanceTypes: []string{"1l40s.20v"}}, want: true},
		{name: "location", filters: PricingFilters{LocationCodes: []string{"fin-02"}}, want: true},
		{name: "model", filters: PricingFilters{Model: "l40"}, want: true},
		{name: "manufacturer", filters: PricingFilters{Manufacturer: "nvidia"}, want: true},
		{name: "gpu count", filters: PricingFilters{GPUCounts: []int{1}}, want: true},
		{name: "market", filters: PricingFilters{Market: PricingMarketSpot}, want: true},
		{name: "all markets", filters: PricingFilters{Market: PricingMarketAll}, want: true},
		{name: "different location", filters: PricingFilters{LocationCodes: []string{"FIN-03"}}, want: false},
		{name: "cpu only", filters: PricingFilters{GPUCounts: []int{0}}, want: false},
		{name: "on demand", filters: PricingFilters{Market: PricingMarketOnDemand}, want: false},
		{name: "unavailable", filters: PricingFilters{AvailableOnly: true}, want: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := InstancePriceMatchesFilters(price, tt.filters); got != tt.want {
				t.Fatalf("InstancePriceMatchesFilters() = %t, want %t", got, tt.want)
			}
		})
	}
}

func TestSortInstancePricesByPricePrefersKnownPrices(t *testing.T) {
	prices := []InstancePrice{
		{InstanceType: "b", LocationCode: "FIN-03"},
		{InstanceType: "a", LocationCode: "FIN-02", PricePerHour: 2, PriceKnown: true},
		{InstanceType: "c", LocationCode: "FIN-01", PricePerHour: 1, PriceKnown: true},
	}

	SortInstancePrices(prices, "price")

	if prices[0].InstanceType != "c" || prices[1].InstanceType != "a" || prices[2].InstanceType != "b" {
		t.Fatalf("unexpected sort order: %#v", prices)
	}
}

func TestNormalizePricingFiltersAcceptsOnDemandAliases(t *testing.T) {
	filters := NormalizePricingFilters(PricingFilters{
		Market:    "on_demand",
		GPUCounts: []int{1, 0, 1},
	})

	if filters.Market != PricingMarketOnDemand {
		t.Fatalf("Market = %q", filters.Market)
	}
	if len(filters.GPUCounts) != 2 || filters.GPUCounts[0] != 0 || filters.GPUCounts[1] != 1 {
		t.Fatalf("GPUCounts = %#v", filters.GPUCounts)
	}
}
