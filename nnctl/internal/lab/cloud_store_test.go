package lab

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestCloudStorePersistsManagedWorker(t *testing.T) {
	path := filepath.Join(t.TempDir(), "state.sqlite")
	store, err := openCloudStore(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	worker := managedCloudWorker{
		CloudWorker: CloudWorker{
			ID:           "worker-1",
			Provider:     "verda",
			State:        CloudWorkerReady,
			Message:      "Ready for experiments",
			InstanceID:   "instance-1",
			InstanceType: "1A100.22V",
			Backends:     []string{"cpu", "cuda"},
		},
		SourceVolumeID: "source-1",
		CloneVolumeID:  "clone-1",
		Host:           "root@192.0.2.10",
	}
	if err := store.Save(context.Background(), worker); err != nil {
		t.Fatal(err)
	}
	loaded, err := store.Load(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if loaded == nil || loaded.ID != worker.ID || loaded.InstanceID != worker.InstanceID || loaded.CloneVolumeID != worker.CloneVolumeID {
		t.Fatalf("loaded worker = %#v", loaded)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode().Perm() != 0o600 {
		t.Fatalf("database permissions = %o", info.Mode().Perm())
	}
	var payload string
	if err := store.db.QueryRow("SELECT CAST(worker_json AS TEXT) FROM cloud_workers WHERE id = ?", worker.ID).Scan(&payload); err != nil {
		t.Fatal(err)
	}
	for _, forbidden := range []string{"client_secret", "client_id", "credential"} {
		if strings.Contains(strings.ToLower(payload), forbidden) {
			t.Fatalf("worker state contains %q: %s", forbidden, payload)
		}
	}
}

func TestCloudStoreMigratesLegacyJournal(t *testing.T) {
	directory := t.TempDir()
	legacyPath := filepath.Join(directory, "lab-cloud-worker.json")
	worker := managedCloudWorker{
		CloudWorker:   CloudWorker{ID: "worker-1", State: CloudWorkerReady, InstanceID: "instance-1"},
		CloneVolumeID: "clone-1",
	}
	data, err := json.Marshal(worker)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(legacyPath, data, 0o600); err != nil {
		t.Fatal(err)
	}
	store, err := openCloudStore(context.Background(), filepath.Join(directory, "state.sqlite"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	if err := migrateCloudJournal(context.Background(), store, legacyPath); err != nil {
		t.Fatal(err)
	}
	loaded, err := store.Load(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if loaded == nil || loaded.ID != worker.ID || loaded.InstanceID != worker.InstanceID {
		t.Fatalf("migrated worker = %#v", loaded)
	}
	if _, err := os.Stat(legacyPath); !os.IsNotExist(err) {
		t.Fatalf("legacy journal was not removed: %v", err)
	}
}
