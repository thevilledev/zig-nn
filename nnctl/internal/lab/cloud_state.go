package lab

import (
	"errors"
	"fmt"
	"slices"

	cloudcore "nnctl/internal/cloud"
	"nnctl/internal/cloud/verda"
)

func (m *CloudManager) failWorker(workerID string, err error) {
	if _, transitionErr := m.transitionWorker(
		workerID,
		[]CloudWorkerState{CloudWorkerProvisioning},
		CloudWorkerFailed,
		err.Error(),
		nil,
	); transitionErr == nil {
		_ = m.persistWorker()
	}
}

func (m *CloudManager) updateWorker(workerID string, update func(*managedCloudWorker)) {
	_, _ = m.mutateWorker(workerID, func(worker *managedCloudWorker) error {
		update(worker)
		return nil
	})
}

func (m *CloudManager) updateProvisioningWorker(workerID string, update func(*managedCloudWorker)) bool {
	_, err := m.updateWorkerInState(workerID, CloudWorkerProvisioning, update)
	return err == nil
}

func (m *CloudManager) updateWorkerInState(
	workerID string,
	state CloudWorkerState,
	update func(*managedCloudWorker),
) (managedCloudWorker, error) {
	return m.mutateWorker(workerID, func(worker *managedCloudWorker) error {
		if worker.State != state {
			return fmt.Errorf("cloud worker is %s", worker.State)
		}
		update(worker)
		return nil
	})
}

func (m *CloudManager) transitionWorker(
	workerID string,
	from []CloudWorkerState,
	to CloudWorkerState,
	message string,
	update func(*managedCloudWorker),
) (managedCloudWorker, error) {
	return m.mutateWorker(workerID, func(worker *managedCloudWorker) error {
		if !slices.Contains(from, worker.State) {
			return fmt.Errorf("cloud worker is %s", worker.State)
		}
		if update != nil {
			update(worker)
		}
		worker.State = to
		worker.Message = message
		return nil
	})
}

func (m *CloudManager) mutateWorker(
	workerID string,
	update func(*managedCloudWorker) error,
) (managedCloudWorker, error) {
	m.mu.Lock()
	if m.worker == nil || m.worker.ID != workerID {
		m.mu.Unlock()
		return managedCloudWorker{}, errors.New("cloud worker not found")
	}
	if err := update(m.worker); err != nil {
		m.mu.Unlock()
		return managedCloudWorker{}, err
	}
	worker := cloneManagedCloudWorker(*m.worker)
	subscribers := make([]chan CloudWorker, 0, len(m.workerSubscribers))
	for subscriber := range m.workerSubscribers {
		subscribers = append(subscribers, subscriber)
	}
	m.mu.Unlock()
	for _, subscriber := range subscribers {
		select {
		case subscriber <- worker.CloudWorker:
		default:
		}
	}
	return worker, nil
}

func cloneManagedCloudWorker(worker managedCloudWorker) managedCloudWorker {
	worker.Backends = append([]string(nil), worker.Backends...)
	worker.Resources = cloneResourceRefs(worker.Resources)
	return worker
}

func cloneResourceRefs(resources []cloudcore.ResourceRef) []cloudcore.ResourceRef {
	cloned := make([]cloudcore.ResourceRef, len(resources))
	for index, resource := range resources {
		cloned[index] = resource
		if resource.Metadata != nil {
			cloned[index].Metadata = make(map[string]string, len(resource.Metadata))
			for key, value := range resource.Metadata {
				cloned[index].Metadata[key] = value
			}
		}
	}
	return cloned
}

func legacyVerdaResourceIDs(resources []cloudcore.ResourceRef) (sourceID, cloneID string) {
	for _, resource := range resources {
		if resource.Kind != verda.ResourceOSVolume {
			continue
		}
		if resource.Preserve {
			sourceID = resource.ID
		} else if cloneID == "" {
			cloneID = resource.ID
		}
	}
	return sourceID, cloneID
}

func (m *CloudManager) managedWorker(workerID string) (managedCloudWorker, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.worker == nil || m.worker.ID != workerID {
		return managedCloudWorker{}, false
	}
	return cloneManagedCloudWorker(*m.worker), true
}

func (m *CloudManager) Workers() []CloudWorker {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.worker == nil {
		return []CloudWorker{}
	}
	worker := m.worker.CloudWorker
	worker.Backends = append([]string(nil), worker.Backends...)
	return []CloudWorker{worker}
}

type CloudWorkerSubscription struct {
	Initial CloudWorker
	Events  <-chan CloudWorker
	close   func()
}

func (s CloudWorkerSubscription) Close() {
	if s.close != nil {
		s.close()
	}
}

func (m *CloudManager) SubscribeWorker(workerID string) (CloudWorkerSubscription, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.worker == nil || m.worker.ID != workerID {
		return CloudWorkerSubscription{}, errors.New("cloud worker not found")
	}
	channel := make(chan CloudWorker, 32)
	m.workerSubscribers[channel] = struct{}{}
	initial := m.worker.CloudWorker
	initial.Backends = append([]string(nil), initial.Backends...)
	return CloudWorkerSubscription{
		Initial: initial,
		Events:  channel,
		close: func() {
			m.mu.Lock()
			delete(m.workerSubscribers, channel)
			m.mu.Unlock()
		},
	}, nil
}

func (m *CloudManager) ValidateWorker(workerID, backend string) error {
	worker, ok := m.managedWorker(workerID)
	if !ok {
		return errors.New("cloud worker not found")
	}
	if worker.State != CloudWorkerReady {
		return fmt.Errorf("cloud worker is %s", worker.State)
	}
	if !slices.Contains(worker.Backends, backend) {
		return fmt.Errorf("backend %q is unavailable on cloud worker", backend)
	}
	return nil
}
