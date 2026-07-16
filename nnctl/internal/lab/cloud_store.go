package lab

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"

	_ "modernc.org/sqlite"
)

const cloudStateDatabaseName = "learning-lab.sqlite"

type cloudStore struct {
	db *sql.DB
}

func openCloudStore(ctx context.Context, path string) (*cloudStore, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return nil, fmt.Errorf("create cloud state directory: %w", err)
	}
	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, fmt.Errorf("open cloud state database: %w", err)
	}
	db.SetMaxOpenConns(1)
	store := &cloudStore{db: db}
	if err := store.initialize(ctx); err != nil {
		_ = db.Close()
		return nil, err
	}
	if err := os.Chmod(path, 0o600); err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("protect cloud state database: %w", err)
	}
	return store, nil
}

func (s *cloudStore) initialize(ctx context.Context) error {
	statements := []string{
		"PRAGMA busy_timeout = 5000",
		"PRAGMA journal_mode = WAL",
		`CREATE TABLE IF NOT EXISTS cloud_workers (
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
		if _, err := s.db.ExecContext(ctx, statement); err != nil {
			return fmt.Errorf("initialize cloud state database: %w", err)
		}
	}
	return nil
}

func (s *cloudStore) Save(ctx context.Context, worker managedCloudWorker) error {
	data, err := json.Marshal(worker)
	if err != nil {
		return fmt.Errorf("encode cloud worker state: %w", err)
	}
	_, err = s.db.ExecContext(ctx, `
		INSERT INTO cloud_workers (
			id, instance_id, clone_volume_id, source_volume_id,
			state, worker_json, updated_at
		) VALUES (?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT(id) DO UPDATE SET
			instance_id = excluded.instance_id,
			clone_volume_id = excluded.clone_volume_id,
			source_volume_id = excluded.source_volume_id,
			state = excluded.state,
			worker_json = excluded.worker_json,
			updated_at = excluded.updated_at
	`, worker.ID, worker.InstanceID, worker.CloneVolumeID, worker.SourceVolumeID,
		worker.State, data, time.Now().UTC().UnixNano())
	if err != nil {
		return fmt.Errorf("save cloud worker state: %w", err)
	}
	return nil
}

func (s *cloudStore) Load(ctx context.Context) (*managedCloudWorker, error) {
	var id string
	var data []byte
	err := s.db.QueryRowContext(ctx, `
		SELECT id, worker_json
		FROM cloud_workers
		ORDER BY updated_at DESC
		LIMIT 1
	`).Scan(&id, &data)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("load cloud worker state: %w", err)
	}
	var worker managedCloudWorker
	if err := json.Unmarshal(data, &worker); err != nil {
		return nil, fmt.Errorf("decode cloud worker state: %w", err)
	}
	if worker.ID == "" || worker.ID != id {
		return nil, errors.New("cloud worker state has an invalid ID")
	}
	return &worker, nil
}

func (s *cloudStore) Delete(ctx context.Context, workerID string) error {
	if _, err := s.db.ExecContext(ctx, "DELETE FROM cloud_workers WHERE id = ?", workerID); err != nil {
		return fmt.Errorf("delete cloud worker state: %w", err)
	}
	return nil
}

func (s *cloudStore) Close() error {
	if s == nil || s.db == nil {
		return nil
	}
	return s.db.Close()
}

func migrateCloudJournal(ctx context.Context, store *cloudStore, path string) error {
	if path == "" {
		return nil
	}
	current, err := store.Load(ctx)
	if err != nil || current != nil {
		return err
	}
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("read legacy cloud worker journal: %w", err)
	}
	var worker managedCloudWorker
	if err := json.Unmarshal(data, &worker); err != nil {
		return fmt.Errorf("decode legacy cloud worker journal: %w", err)
	}
	if worker.ID != "" && (worker.InstanceID != "" || worker.CloneVolumeID != "") {
		if err := store.Save(ctx, worker); err != nil {
			return err
		}
	}
	if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("remove legacy cloud worker journal: %w", err)
	}
	return nil
}
