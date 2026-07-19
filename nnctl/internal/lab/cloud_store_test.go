package lab

import (
	"context"
	"database/sql"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	cloudcore "nnctl/internal/cloud"
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
			IP:           "192.0.2.10",
			Backends:     []string{"cpu", "cuda"},
		},
		Resources:      []cloudcore.ResourceRef{{Kind: "volume", ID: "clone-1"}},
		SourceVolumeID: "source-1",
		CloneVolumeID:  "clone-1",
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
	if err := store.db.QueryRowContext(t.Context(), "SELECT CAST(worker_json AS TEXT) FROM cloud_workers WHERE id = ?", worker.ID).Scan(&payload); err != nil {
		t.Fatal(err)
	}
	for _, forbidden := range []string{"client_secret", "client_id", "credential"} {
		if strings.Contains(strings.ToLower(payload), forbidden) {
			t.Fatalf("worker state contains %q: %s", forbidden, payload)
		}
	}
	var provider, resources string
	if err := store.db.QueryRowContext(t.Context(), "SELECT provider, CAST(resources_json AS TEXT) FROM cloud_workers WHERE id = ?", worker.ID).Scan(&provider, &resources); err != nil {
		t.Fatal(err)
	}
	if provider != "verda" || !strings.Contains(resources, "clone-1") {
		t.Fatalf("provider columns = %q, %q", provider, resources)
	}
}

func TestCloudStoreMigratesVersionOneSchema(t *testing.T) {
	path := filepath.Join(t.TempDir(), "state.sqlite")
	db, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatal(err)
	}
	worker := managedCloudWorker{
		CloudWorker:   CloudWorker{ID: "worker-1", State: CloudWorkerReady, InstanceID: "instance-1"},
		CloneVolumeID: "clone-1",
	}
	payload, err := json.Marshal(worker)
	if err != nil {
		t.Fatal(err)
	}
	statements := []string{
		`CREATE TABLE cloud_workers (
			id TEXT PRIMARY KEY,
			instance_id TEXT NOT NULL,
			clone_volume_id TEXT NOT NULL,
			source_volume_id TEXT NOT NULL,
			state TEXT NOT NULL,
			worker_json BLOB NOT NULL,
			updated_at INTEGER NOT NULL
		)`,
		"PRAGMA user_version = 1",
	}
	for _, statement := range statements {
		if _, err := db.ExecContext(t.Context(), statement); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := db.ExecContext(t.Context(), `
		INSERT INTO cloud_workers (
			id, instance_id, clone_volume_id, source_volume_id,
			state, worker_json, updated_at
		) VALUES (?, ?, ?, ?, ?, ?, ?)
	`, worker.ID, worker.InstanceID, worker.CloneVolumeID, worker.SourceVolumeID, worker.State, payload, 1); err != nil {
		t.Fatal(err)
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}

	store, err := openCloudStore(t.Context(), path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	var version int
	if err := store.db.QueryRowContext(t.Context(), "PRAGMA user_version").Scan(&version); err != nil {
		t.Fatal(err)
	}
	if version != 2 {
		t.Fatalf("schema version = %d", version)
	}
	for _, column := range []string{"provider", "resources_json"} {
		exists, err := store.hasColumn(t.Context(), "cloud_workers", column)
		if err != nil {
			t.Fatal(err)
		}
		if !exists {
			t.Fatalf("migrated schema is missing %s", column)
		}
	}
	loaded, err := store.Load(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if loaded == nil || loaded.ID != worker.ID || loaded.CloneVolumeID != worker.CloneVolumeID {
		t.Fatalf("migrated worker = %#v", loaded)
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
