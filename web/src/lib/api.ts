import type {
  Backend,
  Capabilities,
  CloudDeployRequest,
  CloudOptions,
  CloudStatus,
  CloudWorker,
  ExecutionTarget,
  ExperimentSpec,
  RunEvent
} from './types';

interface APIErrorBody {
  error?: { message?: string };
}

async function parseResponse<T>(response: Response): Promise<T> {
  const body = (await response.json()) as T & APIErrorBody;
  if (!response.ok) throw new Error(body.error?.message ?? `Request failed (${response.status})`);
  return body;
}

export async function loadExperiments(): Promise<ExperimentSpec[]> {
  const experiments = await parseResponse<ExperimentSpec[]>(await fetch('/api/experiments'));
  return experiments.map((experiment) => ({
    ...experiment,
    parameters: experiment.parameters ?? [],
    metrics: experiment.metrics ?? []
  }));
}

export async function loadCapabilities(): Promise<Capabilities> {
  return parseResponse<Capabilities>(await fetch('/api/capabilities'));
}

export async function startRun(
  experiment: string,
  backend: Backend,
  parameters: Record<string, number>,
  target: ExecutionTarget = 'local',
  workerID = '',
  acknowledgeCommittedHead = false
): Promise<string> {
  const request: Record<string, unknown> = { experiment, backend, parameters };
  if (target === 'cloud') {
    request.target = { kind: 'cloud', worker_id: workerID };
    request.acknowledge_committed_head = acknowledgeCommittedHead;
  }
  const body = await parseResponse<{ id: string }>(
    await fetch('/api/runs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request)
    })
  );
  return body.id;
}

export async function loadCloudStatus(): Promise<CloudStatus> {
  return parseResponse<CloudStatus>(await fetch('/api/cloud/status'));
}

export async function loadCloudOptions(): Promise<CloudOptions> {
  return parseResponse<CloudOptions>(await fetch('/api/cloud/options'));
}

export async function loadCloudWorkers(): Promise<CloudWorker[]> {
  return parseResponse<CloudWorker[]>(await fetch('/api/cloud/workers'));
}

export async function deployCloudWorker(request: CloudDeployRequest): Promise<CloudWorker> {
  return parseResponse<CloudWorker>(
    await fetch('/api/cloud/workers', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request)
    })
  );
}

export async function destroyCloudWorker(id: string): Promise<void> {
  const response = await fetch(`/api/cloud/workers/${id}`, { method: 'DELETE' });
  if (!response.ok) {
    const body = (await response.json()) as APIErrorBody;
    throw new Error(body.error?.message ?? `Destroy failed (${response.status})`);
  }
}

export function connectCloudWorker(
  id: string,
  onWorker: (worker: CloudWorker) => void,
  onConnectionError: () => void
): () => void {
  const source = new EventSource(`/api/cloud/workers/${id}/events`);
  source.addEventListener('worker', (message) => {
    const worker = JSON.parse((message as MessageEvent<string>).data) as CloudWorker;
    onWorker(worker);
    if (worker.state === 'destroyed') source.close();
  });
  source.onerror = onConnectionError;
  return () => source.close();
}

export async function cancelRun(id: string): Promise<void> {
  const response = await fetch(`/api/runs/${id}`, { method: 'DELETE' });
  if (!response.ok) {
    const body = (await response.json()) as APIErrorBody;
    throw new Error(body.error?.message ?? `Cancel failed (${response.status})`);
  }
}

const eventTypes: RunEvent['type'][] = ['run_started', 'metric', 'snapshot', 'run_completed', 'run_failed', 'log', 'run_status'];

export function connectRun(
  id: string,
  onEvent: (event: RunEvent) => void,
  onConnectionError: () => void
): () => void {
  const source = new EventSource(`/api/runs/${id}/events`);
  const handle = (message: Event) => {
    const event = JSON.parse((message as MessageEvent<string>).data) as RunEvent;
    onEvent(event);
    if (event.type === 'run_completed' || event.type === 'run_failed') source.close();
  };
  for (const type of eventTypes) source.addEventListener(type, handle);
  source.onerror = onConnectionError;
  return () => source.close();
}
