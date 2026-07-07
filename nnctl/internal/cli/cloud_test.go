package cli

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"

	verdacloud "nnctl/internal/cloud/verda"
)

func TestCloudDeployDryRunJSON(t *testing.T) {
	var stdout, stderr bytes.Buffer
	app := &App{Stdout: &stdout, Stderr: &stderr}

	err := app.Run(context.Background(), []string{"cloud", "deploy", "--instance-type", "1V100.6V", "--source-os-volume-id", "vol-golden", "--dry-run", "--json"})
	if err != nil {
		t.Fatalf("Run() error = %v\nstderr:\n%s", err, stderr.String())
	}

	var result struct {
		Provider string `json:"provider"`
		DryRun   bool   `json:"dry_run"`
		Policy   struct {
			LocationCode         string `json:"location_code"`
			LocationSelection    string `json:"location_selection"`
			SourceOSVolumeLocked bool   `json:"source_os_volume_locked"`
			SpotOnly             bool   `json:"spot_only"`
			SingleGPU            bool   `json:"single_gpu"`
		} `json:"policy"`
		SourceOSVolumeID string `json:"source_os_volume_id"`
		Request          struct {
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
	if result.SourceOSVolumeID != "vol-golden" {
		t.Fatalf("SourceOSVolumeID = %q", result.SourceOSVolumeID)
	}
	if result.Policy.LocationCode != "" || result.Policy.LocationSelection != "source_os_volume_location" || !result.Policy.SourceOSVolumeLocked || !result.Policy.SpotOnly || !result.Policy.SingleGPU {
		t.Fatalf("policy was not encoded: %#v", result.Policy)
	}
}

func TestCloudDeployRejectsNonUUIDSSHKeyID(t *testing.T) {
	var stdout, stderr bytes.Buffer
	app := &App{Stdout: &stdout, Stderr: &stderr}

	err := app.Run(context.Background(), []string{"cloud", "deploy", "--instance-type", "1V100.6V", "--source-os-volume-id", "vol-golden", "--dry-run", "--ssh-key-id", "ville+mba@vesilehto.fi"})
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

func TestCloudDeployRequiresSourceOSVolumeIDFlag(t *testing.T) {
	var stdout, stderr bytes.Buffer
	app := &App{Stdout: &stdout, Stderr: &stderr}

	err := app.Run(context.Background(), []string{"cloud", "deploy", "--instance-type", "1V100.6V", "--dry-run"})
	if err == nil {
		t.Fatal("expected missing source-os-volume-id error")
	}
	if !strings.Contains(err.Error(), "source-os-volume-id") {
		t.Fatalf("error did not mention source-os-volume-id: %v", err)
	}
}

func TestCloudDestroyDryRunJSON(t *testing.T) {
	var stdout, stderr bytes.Buffer
	app := &App{Stdout: &stdout, Stderr: &stderr}

	err := app.Run(context.Background(), []string{"cloud", "destroy", "inst-1", "inst-2", "--dry-run", "--permanent=false", "--volume-id", "vol-1", "--json"})
	if err != nil {
		t.Fatalf("Run() error = %v\nstderr:\n%s", err, stderr.String())
	}

	var result struct {
		Provider string `json:"provider"`
		DryRun   bool   `json:"dry_run"`
		Request  struct {
			InstanceIDs       []string `json:"instance_ids"`
			VolumeIDs         []string `json:"volume_ids"`
			DeletePermanently bool     `json:"delete_permanently"`
		} `json:"request"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		t.Fatalf("invalid JSON output: %v\n%s", err, stdout.String())
	}

	if result.Provider != "verda" || !result.DryRun {
		t.Fatalf("unexpected result header: %#v", result)
	}
	if len(result.Request.InstanceIDs) != 2 || result.Request.InstanceIDs[0] != "inst-1" || result.Request.InstanceIDs[1] != "inst-2" {
		t.Fatalf("unexpected instance ids: %#v", result.Request.InstanceIDs)
	}
	if len(result.Request.VolumeIDs) != 1 || result.Request.VolumeIDs[0] != "vol-1" {
		t.Fatalf("unexpected volume ids: %#v", result.Request.VolumeIDs)
	}
	if result.Request.DeletePermanently {
		t.Fatal("DeletePermanently = true, want false")
	}
}

func TestCloudDestroyDefaultsToPermanentDryRun(t *testing.T) {
	var stdout, stderr bytes.Buffer
	app := &App{Stdout: &stdout, Stderr: &stderr}

	err := app.Run(context.Background(), []string{"cloud", "destroy", "inst-1", "--dry-run", "--json"})
	if err != nil {
		t.Fatalf("Run() error = %v\nstderr:\n%s", err, stderr.String())
	}

	var result verdacloud.DestroyResult
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		t.Fatalf("invalid JSON output: %v\n%s", err, stdout.String())
	}
	if !result.Request.DeletePermanently {
		t.Fatal("DeletePermanently = false, want true")
	}
}

func TestCloudPackerTemplateWritesEmbeddedFiles(t *testing.T) {
	var stdout, stderr bytes.Buffer
	app := &App{Stdout: &stdout, Stderr: &stderr}
	dir := t.TempDir()

	err := app.Run(context.Background(), []string{"cloud", "packer-template", dir})
	if err != nil {
		t.Fatalf("Run() error = %v\nstderr:\n%s", err, stderr.String())
	}

	for _, file := range []struct {
		name string
		want string
	}{
		{name: "ubuntu.pkr.hcl", want: `source "verda-instance" "ubuntu"`},
		{name: "bootstrap.sh", want: "cuda-toolkit-13-0"},
	} {
		content, err := os.ReadFile(dir + "/" + file.name)
		if err != nil {
			t.Fatalf("read generated %s: %v", file.name, err)
		}
		if !strings.Contains(string(content), file.want) {
			t.Fatalf("%s missing %q:\n%s", file.name, file.want, string(content))
		}
	}
	if !strings.Contains(stdout.String(), "ubuntu.pkr.hcl") || !strings.Contains(stdout.String(), "bootstrap.sh") {
		t.Fatalf("stdout did not list written files:\n%s", stdout.String())
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

func TestPrintCloudDestroyResultPlainText(t *testing.T) {
	var stdout bytes.Buffer
	app := &App{Stdout: &stdout}

	err := app.printCloudDestroyResult(&verdacloud.DestroyResult{
		Provider: "verda",
		Request: verdacloud.DestroyInstanceRequest{
			InstanceIDs:       []string{"inst-1"},
			DeletePermanently: true,
		},
		Results: []verdacloud.DestroyInstanceResult{
			{InstanceID: "inst-1", Status: "success"},
		},
	}, false)
	if err != nil {
		t.Fatal(err)
	}

	output := stdout.String()
	for _, want := range []string{"destroy requested", "inst-1", "delete permanently: true", "success"} {
		if !strings.Contains(output, want) {
			t.Fatalf("destroy output missing %q:\n%s", want, output)
		}
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
