package lab

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	cloudcore "nnctl/internal/cloud"
	"nnctl/internal/cloud/verda"
	cloudworkflow "nnctl/internal/cloud/workflow"
)

func (m *CloudManager) knownHostsPath(workerID string) (string, error) {
	directory := filepath.Join(filepath.Dir(m.databasePath), "known-hosts")
	if err := os.MkdirAll(directory, 0o700); err != nil {
		return "", fmt.Errorf("create known-hosts directory: %w", err)
	}
	return filepath.Join(directory, workerID), nil
}

func (m *CloudManager) persistWorker() error {
	m.persistenceMu.Lock()
	defer m.persistenceMu.Unlock()
	m.mu.Lock()
	if m.worker == nil {
		m.mu.Unlock()
		return nil
	}
	worker := cloneManagedCloudWorker(*m.worker)
	m.mu.Unlock()
	if worker.State == CloudWorkerDestroyed || (worker.InstanceID == "" && len(worker.Resources) == 0 && worker.CloneVolumeID == "") {
		return m.withPersistenceContext(func(ctx context.Context) error { return m.store.Delete(ctx, worker.ID) })
	}
	return m.withPersistenceContext(func(ctx context.Context) error { return m.store.Save(ctx, worker) })
}

func (m *CloudManager) loadWorker() error {
	m.persistenceMu.Lock()
	defer m.persistenceMu.Unlock()
	var worker *managedCloudWorker
	err := m.withPersistenceContext(func(ctx context.Context) error {
		var loadErr error
		worker, loadErr = m.store.Load(ctx)
		return loadErr
	})
	if err != nil || worker == nil {
		return err
	}
	if worker.Provider == "" {
		worker.Provider = verda.ProviderName
	}
	if len(worker.Resources) == 0 {
		if worker.SourceVolumeID != "" {
			worker.Resources = append(worker.Resources, cloudcore.ResourceRef{
				Kind: verda.ResourceOSVolume, ID: worker.SourceVolumeID, Preserve: true,
			})
		}
		if worker.CloneVolumeID != "" {
			worker.Resources = append(worker.Resources, cloudcore.ResourceRef{
				Kind: verda.ResourceOSVolume, ID: worker.CloneVolumeID,
			})
		}
	}
	if worker.InstanceID == "" && len(worker.Resources) == 0 {
		return m.withPersistenceContext(func(ctx context.Context) error { return m.store.Delete(ctx, worker.ID) })
	}
	worker.State = CloudWorkerProvisioning
	worker.Message = "Reconnecting to persisted cloud worker"
	worker.ExpiresAt = nil
	m.worker = worker
	return nil
}

func (m *CloudManager) recoverWorker() {
	m.mu.Lock()
	worker := managedCloudWorker{}
	ok := m.worker != nil
	if ok {
		worker = cloneManagedCloudWorker(*m.worker)
	}
	if !ok || worker.InstanceID == "" {
		m.mu.Unlock()
		return
	}
	ctx, cancel := context.WithTimeout(m.ctx, m.timeout)
	done := make(chan struct{})
	m.provisionWorkerID = worker.ID
	m.provisionCancel = cancel
	m.provisionDone = done
	m.mu.Unlock()
	m.wg.Add(1)
	go func() {
		defer func() {
			cancel()
			close(done)
			m.mu.Lock()
			if m.provisionWorkerID == worker.ID {
				m.provisionWorkerID = ""
				m.provisionCancel = nil
				m.provisionDone = nil
			}
			m.mu.Unlock()
			m.wg.Done()
		}()
		if err := m.reconnectWorker(ctx, worker.ID); err != nil {
			if errors.Is(ctx.Err(), context.Canceled) {
				return
			}
			m.failWorker(worker.ID, err)
		}
	}()
}

func (m *CloudManager) reconnectWorker(ctx context.Context, workerID string) error {
	worker, ok := m.managedWorker(workerID)
	if !ok {
		return errors.New("persisted cloud worker not found")
	}
	provider, err := m.provider(ctx, worker.Provider)
	if err != nil {
		return err
	}
	instance, err := provider.Instance(ctx, worker.InstanceID)
	if err != nil {
		return fmt.Errorf("get persisted %s instance %s: %w", worker.Provider, worker.InstanceID, err)
	}
	if instance.State != cloudcore.InstanceRunning || strings.TrimSpace(instance.PublicIP) == "" {
		ready, err := cloudworkflow.WaitProviderInstance(ctx, provider, worker.InstanceID, m.timeout, m.pollInterval)
		if err != nil {
			return err
		}
		instance = ready
	}
	knownHosts, err := m.knownHostsPath(workerID)
	if err != nil {
		return err
	}
	ip := strings.TrimSpace(instance.PublicIP)
	host, err := cloudworkflow.SSHDestination(m.sshUser, ip)
	if err != nil {
		return err
	}
	if !m.updateProvisioningWorker(workerID, func(current *managedCloudWorker) {
		current.IP = ip
		current.Hostname = firstCloudString(instance.Name, current.Hostname)
		current.Location = firstCloudString(instance.Location, current.Location)
		current.PricePerHour = firstCloudPrice(instance.PricePerHour, current.PricePerHour)
		current.Currency = firstCloudString(instance.Currency, current.Currency)
		if len(instance.Backends) > 0 {
			current.Backends = append([]string(nil), instance.Backends...)
		}
		current.Message = "Verifying persisted cloud worker"
	}) {
		return errors.New("cloud worker is no longer reconnecting")
	}
	sshArgs := cloudworkflow.SSHArgs(knownHosts)
	if err := cloudworkflow.WaitSSH(ctx, m.ssh, sshArgs, host, m.timeout, m.pollInterval); err != nil {
		return err
	}
	backends := instance.Backends
	if len(backends) == 0 {
		backends = worker.Backends
	}
	if err := cloudworkflow.WaitCommand(ctx, m.ssh, sshArgs, host, cloudPreflightCommandFor(backends), m.timeout, m.pollInterval); err != nil {
		return err
	}
	ready := m.now().UTC()
	expires := ready.Add(m.idleTimeout)
	if _, err := m.transitionWorker(workerID, []CloudWorkerState{CloudWorkerProvisioning}, CloudWorkerReady, "Reconnected and ready for experiments", func(current *managedCloudWorker) {
		current.ReadyAt = &ready
		if current.AutoDestroy {
			current.ExpiresAt = &expires
		}
	}); err != nil {
		return err
	}
	if err := m.persistWorker(); err != nil {
		return err
	}
	if worker.AutoDestroy {
		m.wg.Add(1)
		go func() {
			defer m.wg.Done()
			m.destroyIdleWorker(m.ctx, workerID, expires)
		}()
	}
	return nil
}

func (m *CloudManager) clearWorker(workerID string) error {
	if workerID == "" {
		return nil
	}
	m.persistenceMu.Lock()
	defer m.persistenceMu.Unlock()
	return m.withPersistenceContext(func(ctx context.Context) error { return m.store.Delete(ctx, workerID) })
}

func (m *CloudManager) Close() error {
	m.closeMu.Lock()
	defer m.closeMu.Unlock()
	m.mu.Lock()
	if m.closed {
		err := m.closeErr
		m.mu.Unlock()
		return err
	}
	m.closed = true
	cancel := m.cancel
	m.mu.Unlock()
	if cancel != nil {
		cancel()
	}
	m.wg.Wait()
	m.mu.Lock()
	worker := m.worker
	m.mu.Unlock()
	var destroyErr error
	if worker != nil && worker.State != CloudWorkerDestroyed && worker.AutoDestroy {
		ctx, destroyCancel := m.cleanupContext()
		destroyErr = m.Destroy(ctx, worker.ID)
		destroyCancel()
	}
	closeErr := errors.Join(destroyErr, m.store.Close())
	m.mu.Lock()
	m.closeErr = closeErr
	m.mu.Unlock()
	return closeErr
}

func (m *CloudManager) cleanupContext() (context.Context, context.CancelFunc) {
	return context.WithTimeout(context.WithoutCancel(m.ctx), cloudCleanupTimeout)
}

func (m *CloudManager) withPersistenceContext(operation func(context.Context) error) error {
	ctx, cancel := context.WithTimeout(context.WithoutCancel(m.ctx), cloudPersistenceTimeout)
	defer cancel()
	return operation(ctx)
}
