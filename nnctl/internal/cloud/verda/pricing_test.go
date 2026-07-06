package verdacloud

import "testing"

func TestSpotPriceMatchesFilters(t *testing.T) {
	price := SpotPrice{
		InstanceType: "1L40S.20V",
		DisplayName:  "L40S",
		Model:        "L40S",
		Manufacturer: "NVIDIA",
		GPUCount:     1,
		LocationCode: "FIN-02",
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
		{name: "single gpu", filters: PricingFilters{SingleGPU: true}, want: true},
		{name: "different location", filters: PricingFilters{LocationCodes: []string{"FIN-03"}}, want: false},
		{name: "unavailable", filters: PricingFilters{AvailableOnly: true}, want: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := SpotPriceMatchesFilters(price, tt.filters); got != tt.want {
				t.Fatalf("SpotPriceMatchesFilters() = %t, want %t", got, tt.want)
			}
		})
	}
}

func TestSortSpotPricesByPricePrefersKnownPrices(t *testing.T) {
	prices := []SpotPrice{
		{InstanceType: "b", LocationCode: "FIN-03"},
		{InstanceType: "a", LocationCode: "FIN-02", SpotPrice: 2, PriceKnown: true},
		{InstanceType: "c", LocationCode: "FIN-01", SpotPrice: 1, PriceKnown: true},
	}

	SortSpotPrices(prices, "price")

	if prices[0].InstanceType != "c" || prices[1].InstanceType != "a" || prices[2].InstanceType != "b" {
		t.Fatalf("unexpected sort order: %#v", prices)
	}
}
