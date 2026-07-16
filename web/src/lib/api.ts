import type { ExperimentSpec, RunEvent } from './types';

interface APIErrorBody {
  error?: { message?: string };
}

async function parseResponse<T>(response: Response): Promise<T> {
  const body = (await response.json()) as T & APIErrorBody;
  if (!response.ok) throw new Error(body.error?.message ?? `Request failed (${response.status})`);
  return body;
}

export async function loadExperiments(): Promise<ExperimentSpec[]> {
  return parseResponse<ExperimentSpec[]>(await fetch('/api/experiments'));
}

export async function startRun(experiment: string, parameters: Record<string, number>): Promise<string> {
  const body = await parseResponse<{ id: string }>(
    await fetch('/api/runs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ experiment, parameters })
    })
  );
  return body.id;
}

export async function cancelRun(id: string): Promise<void> {
  const response = await fetch(`/api/runs/${id}`, { method: 'DELETE' });
  if (!response.ok) {
    const body = (await response.json()) as APIErrorBody;
    throw new Error(body.error?.message ?? `Cancel failed (${response.status})`);
  }
}

const eventTypes: RunEvent['type'][] = ['run_started', 'metric', 'snapshot', 'run_completed', 'run_failed', 'log'];

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
