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
			Market               string `json:"market"`
			SpotOnly             bool   `json:"spot_only"`
			AllowsCPU            bool   `json:"allows_cpu"`
			MaxGPUCount          int    `json:"max_gpu_count"`
		} `json:"policy"`
		SourceOSVolumeID string `json:"source_os_volume_id"`
		Request          struct {
			InstanceType string `json:"instance_type"`
			Image        string `json:"image"`
			LocationCode string `json:"location_code"`
			Contract     string `json:"contract"`
			IsSpot       bool   `json:"is_spot"`
		} `json:"request"`
		StartupScript *verdacloud.StartupScript `json:"startup_script"`
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
	if result.Request.Image != "" {
		t.Fatalf("Image = %q, want empty source-volume dry-run image", result.Request.Image)
	}
	if result.SourceOSVolumeID != "vol-golden" {
		t.Fatalf("SourceOSVolumeID = %q", result.SourceOSVolumeID)
	}
	if result.Policy.LocationCode != "" || result.Policy.LocationSelection != "source_os_volume_location" || result.Policy.Market != "spot" || !result.Policy.SourceOSVolumeLocked || !result.Policy.SpotOnly || !result.Policy.AllowsCPU || result.Policy.MaxGPUCount != 1 {
		t.Fatalf("policy was not encoded: %#v", result.Policy)
	}
	if result.StartupScript != nil {
		t.Fatalf("unexpected default startup script: %#v", result.StartupScript)
	}
}

func TestCloudDeployDryRunJSONOnDemand(t *testing.T) {
	var stdout, stderr bytes.Buffer
	app := &App{Stdout: &stdout, Stderr: &stderr}

	err := app.Run(context.Background(), []string{"cloud", "deploy", "--instance-type", "1H200.141S.44V", "--market", "on-demand", "--location-code", "FIN-02", "--dry-run", "--json"})
	if err != nil {
		t.Fatalf("Run() error = %v\nstderr:\n%s", err, stderr.String())
	}

	var result struct {
		Policy struct {
			LocationSelection string `json:"location_selection"`
			Market            string `json:"market"`
			SpotOnly          bool   `json:"spot_only"`
		} `json:"policy"`
		Request struct {
			Contract string `json:"contract"`
			IsSpot   bool   `json:"is_spot"`
		} `json:"request"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		t.Fatalf("invalid JSON output: %v\n%s", err, stdout.String())
	}

	if result.Policy.Market != "on-demand" || result.Policy.SpotOnly {
		t.Fatalf("unexpected policy: %#v", result.Policy)
	}
	if result.Policy.LocationSelection != "explicit" {
		t.Fatalf("LocationSelection = %q, want explicit", result.Policy.LocationSelection)
	}
	if result.Request.IsSpot || result.Request.Contract != "PAY_AS_YOU_GO" {
		t.Fatalf("request did not use on-demand market: %#v", result.Request)
	}
}

func TestCloudDeployDryRunJSONCanReuseClonedOSVolume(t *testing.T) {
	var stdout, stderr bytes.Buffer
	app := &App{Stdout: &stdout, Stderr: &stderr}

	err := app.Run(context.Background(), []string{
		"cloud",
		"deploy",
		"--instance-type",
		"1V100.6V",
		"--source-os-volume-id",
		"vol-golden",
		"--cloned-os-volume-id",
		"vol-clone-1",
		"--keep-cloned-os-volume-on-failure",
		"--dry-run",
		"--json",
	})
	if err != nil {
		t.Fatalf("Run() error = %v\nstderr:\n%s", err, stderr.String())
	}

	var result struct {
		Policy struct {
			CleanupClonedOSVolumeOnFailure bool `json:"cleanup_cloned_os_volume_on_failure"`
		} `json:"policy"`
		OSVolumeClone struct {
			VolumeID string `json:"volume_id"`
			Reused   bool   `json:"reused"`
		} `json:"os_volume_clone"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		t.Fatalf("invalid JSON output: %v\n%s", err, stdout.String())
	}

	if result.Policy.CleanupClonedOSVolumeOnFailure {
		t.Fatalf("cleanup policy = true, want false")
	}
	if result.OSVolumeClone.VolumeID != "vol-clone-1" || !result.OSVolumeClone.Reused {
		t.Fatalf("unexpected OS clone plan: %#v", result.OSVolumeClone)
	}
}

func TestCloudDeployDryRunJSONWithSourceOSVolumeName(t *testing.T) {
	var stdout, stderr bytes.Buffer
	app := &App{Stdout: &stdout, Stderr: &stderr}

	err := app.Run(context.Background(), []string{"cloud", "deploy", "--instance-type", "1V100.6V", "--source-os-volume-name", "golden", "--dry-run", "--json"})
	if err != nil {
		t.Fatalf("Run() error = %v\nstderr:\n%s", err, stderr.String())
	}

	var result struct {
		Policy struct {
			LocationSelection    string `json:"location_selection"`
			SourceOSVolumeLocked bool   `json:"source_os_volume_locked"`
		} `json:"policy"`
		SourceOSVolumeID   string `json:"source_os_volume_id"`
		SourceOSVolumeName string `json:"source_os_volume_name"`
		Request            struct {
			Image        string `json:"image"`
			LocationCode string `json:"location_code"`
		} `json:"request"`
		OSVolumeClone *verdacloud.OSVolumeClone `json:"os_volume_clone"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		t.Fatalf("invalid JSON output: %v\n%s", err, stdout.String())
	}

	if result.SourceOSVolumeID != "" || result.SourceOSVolumeName != "golden" {
		t.Fatalf("unexpected source volume fields: id=%q name=%q", result.SourceOSVolumeID, result.SourceOSVolumeName)
	}
	if result.Policy.LocationSelection != "source_os_volume_location" || !result.Policy.SourceOSVolumeLocked {
		t.Fatalf("unexpected policy: %#v", result.Policy)
	}
	if result.Request.Image != "" || result.Request.LocationCode != "" {
		t.Fatalf("unexpected request: %#v", result.Request)
	}
	if result.OSVolumeClone == nil || result.OSVolumeClone.SourceVolumeID != "" || result.OSVolumeClone.Name == "" {
		t.Fatalf("unexpected OS clone plan: %#v", result.OSVolumeClone)
	}
}

func TestCloudDeployAcceptsSourceVolumeNameAlias(t *testing.T) {
	var stdout, stderr bytes.Buffer
	app := &App{Stdout: &stdout, Stderr: &stderr}

	err := app.Run(context.Background(), []string{"cloud", "deploy", "--instance-type", "1V100.6V", "--source-volume-name", "golden", "--dry-run", "--json"})
	if err != nil {
		t.Fatalf("Run() error = %v\nstderr:\n%s", err, stderr.String())
	}

	var result struct {
		SourceOSVolumeName string `json:"source_os_volume_name"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		t.Fatalf("invalid JSON output: %v\n%s", err, stdout.String())
	}
	if result.SourceOSVolumeName != "golden" {
		t.Fatalf("SourceOSVolumeName = %q, want golden", result.SourceOSVolumeName)
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

func TestCloudDeployDryRunWithoutSourceOSVolumeUsesDefaultImage(t *testing.T) {
	var stdout, stderr bytes.Buffer
	app := &App{Stdout: &stdout, Stderr: &stderr}

	err := app.Run(context.Background(), []string{"cloud", "deploy", "--instance-type", "1V100.6V", "--dry-run", "--json"})
	if err != nil {
		t.Fatalf("Run() error = %v\nstderr:\n%s", err, stderr.String())
	}

	var result struct {
		Policy struct {
			LocationSelection    string `json:"location_selection"`
			SourceOSVolumeLocked bool   `json:"source_os_volume_locked"`
		} `json:"policy"`
		Request struct {
			Image string `json:"image"`
		} `json:"request"`
		SourceOSVolumeID string                    `json:"source_os_volume_id"`
		OSVolumeClone    *verdacloud.OSVolumeClone `json:"os_volume_clone"`
		StartupScript    *verdacloud.StartupScript `json:"startup_script"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		t.Fatalf("invalid JSON output: %v\n%s", err, stdout.String())
	}

	if result.Request.Image != verdacloud.DefaultDeployImage {
		t.Fatalf("Image = %q, want %s", result.Request.Image, verdacloud.DefaultDeployImage)
	}
	if result.Policy.LocationSelection != "spot_placement" || result.Policy.SourceOSVolumeLocked {
		t.Fatalf("unexpected policy: %#v", result.Policy)
	}
	if result.SourceOSVolumeID != "" || result.OSVolumeClone != nil {
		t.Fatalf("unexpected source volume plan: source=%q clone=%#v", result.SourceOSVolumeID, result.OSVolumeClone)
	}
	if result.StartupScript == nil {
		t.Fatal("expected planned default startup script")
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

func TestCloudVolumePurgeDryRunJSON(t *testing.T) {
	var stdout, stderr bytes.Buffer
	app := &App{Stdout: &stdout, Stderr: &stderr}

	err := app.Run(context.Background(), []string{"cloud", "volume", "vol-1", "vol-2", "--purge", "--dry-run", "--json"})
	if err != nil {
		t.Fatalf("Run() error = %v\nstderr:\n%s", err, stderr.String())
	}

	var result verdacloud.PurgeVolumesResult
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		t.Fatalf("invalid JSON output: %v\n%s", err, stdout.String())
	}
	if result.Provider != "verda" || !result.DryRun {
		t.Fatalf("unexpected result header: %#v", result)
	}
	if len(result.Request.VolumeIDs) != 2 || result.Request.VolumeIDs[0] != "vol-1" || result.Request.VolumeIDs[1] != "vol-2" {
		t.Fatalf("unexpected volume ids: %#v", result.Request.VolumeIDs)
	}
}

func TestCloudVolumePurgeAllDryRunJSON(t *testing.T) {
	var stdout, stderr bytes.Buffer
	app := &App{Stdout: &stdout, Stderr: &stderr}

	err := app.Run(context.Background(), []string{"cloud", "volume", "--purge-all", "--dry-run", "--json"})
	if err != nil {
		t.Fatalf("Run() error = %v\nstderr:\n%s", err, stderr.String())
	}

	var result verdacloud.PurgeVolumesResult
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		t.Fatalf("invalid JSON output: %v\n%s", err, stdout.String())
	}
	if result.Provider != "verda" || !result.DryRun {
		t.Fatalf("unexpected result header: %#v", result)
	}
	if !result.Request.AllDeleted {
		t.Fatalf("AllDeleted = false, want true: %#v", result.Request)
	}
	if len(result.Request.VolumeIDs) != 0 {
		t.Fatalf("unexpected volume ids: %#v", result.Request.VolumeIDs)
	}
}

func TestCloudVolumePurgeAllRejectsArgs(t *testing.T) {
	var stdout, stderr bytes.Buffer
	app := &App{Stdout: &stdout, Stderr: &stderr}

	err := app.Run(context.Background(), []string{"cloud", "volume", "vol-1", "--purge-all"})
	if err == nil {
		t.Fatal("expected volume ID with purge-all error")
	}
	if !strings.Contains(err.Error(), "volume IDs cannot be combined with --purge-all") {
		t.Fatalf("error did not explain purge-all args: %v", err)
	}
}

func TestCloudVolumePurgeAllRejectsPurge(t *testing.T) {
	var stdout, stderr bytes.Buffer
	app := &App{Stdout: &stdout, Stderr: &stderr}

	err := app.Run(context.Background(), []string{"cloud", "volume", "vol-1", "--purge", "--purge-all"})
	if err == nil {
		t.Fatal("expected conflicting purge flags error")
	}
	if !strings.Contains(err.Error(), "--purge and --purge-all cannot be combined") {
		t.Fatalf("error did not explain conflicting purge flags: %v", err)
	}
}

func TestCloudVolumeRejectsArgsWithoutPurge(t *testing.T) {
	var stdout, stderr bytes.Buffer
	app := &App{Stdout: &stdout, Stderr: &stderr}

	err := app.Run(context.Background(), []string{"cloud", "volume", "vol-1"})
	if err == nil {
		t.Fatal("expected volume ID without purge error")
	}
	if !strings.Contains(err.Error(), "volume IDs require --purge") {
		t.Fatalf("error did not explain list mode args: %v", err)
	}
}

func TestCloudVolumeDryRunRequiresPurge(t *testing.T) {
	var stdout, stderr bytes.Buffer
	app := &App{Stdout: &stdout, Stderr: &stderr}

	err := app.Run(context.Background(), []string{"cloud", "volume", "--dry-run"})
	if err == nil {
		t.Fatal("expected dry-run without purge error")
	}
	if !strings.Contains(err.Error(), "--dry-run requires --purge or --purge-all") {
		t.Fatalf("error did not explain dry-run requirement: %v", err)
	}
}

func TestPrintCloudVolumesPlainText(t *testing.T) {
	var stdout bytes.Buffer
	app := &App{Stdout: &stdout}

	err := app.printCloudVolumes([]verdacloud.Volume{
		{
			ID:         "vol-active",
			Name:       "worker-os",
			Status:     "detached",
			Type:       "NVMe",
			Size:       100,
			Location:   "FIN-03",
			IsOSVolume: true,
		},
		{
			ID:        "vol-deleted",
			Name:      "old-worker",
			Status:    "deleted",
			Type:      "NVMe",
			Size:      50,
			Location:  "FIN-03",
			DeletedAt: "2026-07-14T10:00:00Z",
		},
	}, false)
	if err != nil {
		t.Fatal(err)
	}

	out := stdout.String()
	for _, want := range []string{"ID", "DELETED AT", "vol-active", "worker-os", "vol-deleted", "2026-07-14T10:00:00Z"} {
		if !strings.Contains(out, want) {
			t.Fatalf("volume table missing %q:\n%s", want, out)
		}
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
		{name: "ubuntu.pkr.hcl", want: `variable "authorized_keys_file"`},
		{name: "ubuntu.pkr.hcl", want: `ssh_clear_authorized_keys = true`},
		{name: "ubuntu.pkr.hcl", want: `/tmp/verda_authorized_keys`},
		{name: "ubuntu.pkr.hcl", want: `post-processor "manifest"`},
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

	err := app.printCloudPricing([]verdacloud.InstancePrice{
		{LocationCode: "FIN-02", Market: "spot", IsSpot: true, InstanceType: "1L40S.20V", GPUCount: 1, Model: "L40S", PricePerHour: 0.42, PriceKnown: true, Currency: "eur", Available: true},
	}, false)
	if err != nil {
		t.Fatal(err)
	}

	output := stdout.String()
	for _, want := range []string{"LOCATION", "MARKET", "INSTANCE TYPE", "PRICE/H", "FIN-02", "spot", "1L40S.20V", "L40S", "0.4200", "eur", "true"} {
		if !strings.Contains(output, want) {
			t.Fatalf("pricing output missing %q:\n%s", want, output)
		}
	}
}

func TestPrintCloudPricingJSON(t *testing.T) {
	var stdout bytes.Buffer
	app := &App{Stdout: &stdout}

	err := app.printCloudPricing([]verdacloud.InstancePrice{
		{LocationCode: "FIN-02", Market: "spot", IsSpot: true, InstanceType: "1L40S.20V", GPUCount: 1, Model: "L40S", PricePerHour: 0.42, PriceKnown: true, Currency: "eur", Available: true},
	}, true)
	if err != nil {
		t.Fatal(err)
	}

	var prices []verdacloud.InstancePrice
	if err := json.Unmarshal(stdout.Bytes(), &prices); err != nil {
		t.Fatalf("invalid JSON output: %v\n%s", err, stdout.String())
	}
	if len(prices) != 1 || prices[0].InstanceType != "1L40S.20V" || prices[0].LocationCode != "FIN-02" || prices[0].Market != "spot" {
		t.Fatalf("unexpected prices: %#v", prices)
	}
}

func TestResolveCloudPricingGPUCounts(t *testing.T) {
	tests := []struct {
		name             string
		opts             cloudPricingOptions
		singleGPUChanged bool
		gpuCountChanged  bool
		want             []int
		wantErr          bool
	}{
		{name: "default single gpu", want: []int{1}},
		{name: "explicit cpu", opts: cloudPricingOptions{gpuCounts: []int{0}}, gpuCountChanged: true, want: []int{0}},
		{name: "explicit multi gpu", opts: cloudPricingOptions{gpuCounts: []int{8, 1, 8}}, gpuCountChanged: true, want: []int{1, 8}},
		{name: "all gpu counts", opts: cloudPricingOptions{allGPU: true}, want: nil},
		{name: "single gpu false disables default", opts: cloudPricingOptions{singleGPU: false}, singleGPUChanged: true, want: nil},
		{name: "all conflicts with gpu count", opts: cloudPricingOptions{allGPU: true, gpuCounts: []int{0}}, gpuCountChanged: true, wantErr: true},
		{name: "all conflicts with single gpu", opts: cloudPricingOptions{allGPU: true, singleGPU: true}, singleGPUChanged: true, wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := resolveCloudPricingGPUCounts(tt.opts, tt.singleGPUChanged, tt.gpuCountChanged)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if len(got) != len(tt.want) {
				t.Fatalf("GPUCounts = %#v, want %#v", got, tt.want)
			}
			for i := range got {
				if got[i] != tt.want[i] {
					t.Fatalf("GPUCounts = %#v, want %#v", got, tt.want)
				}
			}
		})
	}
}
