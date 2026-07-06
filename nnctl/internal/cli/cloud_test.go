package cli

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"testing"

	verdacloud "nnctl/internal/cloud/verda"
)

func TestCloudDeployDryRunJSON(t *testing.T) {
	var stdout, stderr bytes.Buffer
	app := &App{Stdout: &stdout, Stderr: &stderr}

	err := app.Run(context.Background(), []string{"cloud", "deploy", "--instance-type", "1V100.6V", "--dry-run", "--json"})
	if err != nil {
		t.Fatalf("Run() error = %v\nstderr:\n%s", err, stderr.String())
	}

	var result struct {
		Provider string `json:"provider"`
		DryRun   bool   `json:"dry_run"`
		Policy   struct {
			LocationCode      string `json:"location_code"`
			LocationSelection string `json:"location_selection"`
			SpotOnly          bool   `json:"spot_only"`
			SingleGPU         bool   `json:"single_gpu"`
		} `json:"policy"`
		Request struct {
			InstanceType string `json:"instance_type"`
			LocationCode string `json:"location_code"`
			Contract     string `json:"contract"`
			IsSpot       bool   `json:"is_spot"`
		} `json:"request"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		t.Fatalf("invalid JSON output: %v\n%s", err, stdout.String())
	}

	if result.Provider != "verda" || !result.DryRun {
		t.Fatalf("unexpected result header: %#v", result)
	}
	if result.Request.InstanceType != "1V100.6V" || result.Request.LocationCode != "" || result.Request.Contract != "SPOT" || !result.Request.IsSpot {
		t.Fatalf("request did not enforce spot auto-placement deploy: %#v", result.Request)
	}
	if result.Policy.LocationCode != "" || result.Policy.LocationSelection != "cheapest_available_spot" || !result.Policy.SpotOnly || !result.Policy.SingleGPU {
		t.Fatalf("policy was not encoded: %#v", result.Policy)
	}
}

func TestCloudDeployRejectsNonUUIDSSHKeyID(t *testing.T) {
	var stdout, stderr bytes.Buffer
	app := &App{Stdout: &stdout, Stderr: &stderr}

	err := app.Run(context.Background(), []string{"cloud", "deploy", "--instance-type", "1V100.6V", "--dry-run", "--ssh-key-id", "ville+mba@vesilehto.fi"})
	if err == nil {
		t.Fatal("expected non-UUID SSH key id error")
	}
	if !strings.Contains(err.Error(), "nnctl cloud ssh-keys") {
		t.Fatalf("error did not point to ssh key listing command: %v", err)
	}
}

func TestCloudDeployRequiresInstanceTypeFlag(t *testing.T) {
	var stdout, stderr bytes.Buffer
	app := &App{Stdout: &stdout, Stderr: &stderr}

	err := app.Run(context.Background(), []string{"cloud", "deploy", "--dry-run"})
	if err == nil {
		t.Fatal("expected missing instance type error")
	}
	if !strings.Contains(err.Error(), "instance-type") {
		t.Fatalf("error did not mention instance-type: %v", err)
	}
}

func TestPrintCloudSSHKeysPlainText(t *testing.T) {
	var stdout bytes.Buffer
	app := &App{Stdout: &stdout}

	err := app.printCloudSSHKeys([]verdacloud.SSHKey{
		{ID: "11111111-1111-1111-1111-111111111111", Name: "ville+mba@vesilehto.fi", Fingerprint: "SHA256:abc"},
	}, false)
	if err != nil {
		t.Fatal(err)
	}

	output := stdout.String()
	for _, want := range []string{
		"ID",
		"NAME",
		"FINGERPRINT",
		"11111111-1111-1111-1111-111111111111",
		"ville+mba@vesilehto.fi",
		"SHA256:abc",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("ssh key output missing %q:\n%s", want, output)
		}
	}
}

func TestPrintCloudSSHKeysJSON(t *testing.T) {
	var stdout bytes.Buffer
	app := &App{Stdout: &stdout}

	err := app.printCloudSSHKeys([]verdacloud.SSHKey{
		{ID: "11111111-1111-1111-1111-111111111111", Name: "key", Fingerprint: "SHA256:abc"},
	}, true)
	if err != nil {
		t.Fatal(err)
	}

	var keys []verdacloud.SSHKey
	if err := json.Unmarshal(stdout.Bytes(), &keys); err != nil {
		t.Fatalf("invalid JSON output: %v\n%s", err, stdout.String())
	}
	if len(keys) != 1 || keys[0].ID != "11111111-1111-1111-1111-111111111111" {
		t.Fatalf("unexpected keys: %#v", keys)
	}
}

func TestPrintCloudInstancesPlainText(t *testing.T) {
	var stdout bytes.Buffer
	app := &App{Stdout: &stdout}
	ip := "203.0.113.10"

	err := app.printCloudInstances([]verdacloud.Instance{
		{ID: "inst-1", Hostname: "worker", Status: "running", InstanceType: "1L40S.20V", Location: "FIN-02", IsSpot: true, IP: &ip},
	}, false)
	if err != nil {
		t.Fatal(err)
	}

	output := stdout.String()
	for _, want := range []string{"ID", "HOSTNAME", "STATUS", "inst-1", "worker", "running", "1L40S.20V", "FIN-02", "203.0.113.10"} {
		if !strings.Contains(output, want) {
			t.Fatalf("instance output missing %q:\n%s", want, output)
		}
	}
}

func TestActiveCloudInstancesFiltersInactiveStatuses(t *testing.T) {
	instances := []verdacloud.Instance{
		{ID: "running", Status: "running"},
		{ID: "pending", Status: "pending"},
		{ID: "offline", Status: "offline"},
		{ID: "error", Status: "error"},
	}

	active := activeCloudInstances(instances)
	if len(active) != 2 || active[0].ID != "running" || active[1].ID != "pending" {
		t.Fatalf("unexpected active instances: %#v", active)
	}
}

func TestPrintCloudPricingPlainText(t *testing.T) {
	var stdout bytes.Buffer
	app := &App{Stdout: &stdout}

	err := app.printCloudPricing([]verdacloud.SpotPrice{
		{LocationCode: "FIN-02", InstanceType: "1L40S.20V", GPUCount: 1, Model: "L40S", SpotPrice: 0.42, PriceKnown: true, Currency: "eur", Available: true},
	}, false)
	if err != nil {
		t.Fatal(err)
	}

	output := stdout.String()
	for _, want := range []string{"LOCATION", "INSTANCE TYPE", "SPOT/H", "FIN-02", "1L40S.20V", "L40S", "0.4200", "eur", "true"} {
		if !strings.Contains(output, want) {
			t.Fatalf("pricing output missing %q:\n%s", want, output)
		}
	}
}

func TestPrintCloudPricingJSON(t *testing.T) {
	var stdout bytes.Buffer
	app := &App{Stdout: &stdout}

	err := app.printCloudPricing([]verdacloud.SpotPrice{
		{LocationCode: "FIN-02", InstanceType: "1L40S.20V", GPUCount: 1, Model: "L40S", SpotPrice: 0.42, PriceKnown: true, Currency: "eur", Available: true},
	}, true)
	if err != nil {
		t.Fatal(err)
	}

	var prices []verdacloud.SpotPrice
	if err := json.Unmarshal(stdout.Bytes(), &prices); err != nil {
		t.Fatalf("invalid JSON output: %v\n%s", err, stdout.String())
	}
	if len(prices) != 1 || prices[0].InstanceType != "1L40S.20V" || prices[0].LocationCode != "FIN-02" {
		t.Fatalf("unexpected prices: %#v", prices)
	}
}
