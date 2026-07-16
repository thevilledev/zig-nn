import { fireEvent, render, screen, waitFor } from '@testing-library/svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import App from './App.svelte';
import type { CloudOptions, CloudStatus, CloudWorker, ExperimentSpec, RunEvent } from './lib/types';

const experiment: ExperimentSpec = {
  id: 'xor-training',
  category: 'Foundations',
  title: 'Learning XOR',
  description: 'Learn XOR.',
  question: 'How does XOR become learnable?',
  observe: ['Loss falls'],
  interpretation: ['Read the probabilities.'],
  visualization: 'xor_predictions',
  sources: ['experiments/xor_training/xor_training.zig'],
  metrics: [{ name: 'loss', label: 'Training loss' }],
  backends: ['cpu'],
  default_backend: 'cpu',
  parameters: [
    { name: 'epochs', label: 'Epochs', help: 'Training passes.', kind: 'integer', default: 100, min: 100, max: 20000, step: 100 },
    { name: 'learning_rate', label: 'Learning rate', help: 'Update size.', kind: 'number', default: 0.3, min: 0.001, max: 1, step: 0.001 },
    { name: 'seed', label: 'Seed', help: 'Reproducibility.', kind: 'integer', default: 42, min: 0, max: 1000, step: 1 }
  ]
};

const capabilities = { platform: 'darwin', backends: ['cpu', 'metal'] } as const;

const cloudStatus: CloudStatus = {
  enabled: true,
  configured: true,
  provider: 'verda',
  repository: { revision: '0123456789abcdef', dirty: false }
};

const cloudOptions: CloudOptions = {
  provider: 'verda',
  source_os_volume_name: 'packer-verda-zig-nn-volume-root',
  prices: [
    {
      instance_type: '1A100.22V',
      model: 'A100',
      manufacturer: 'NVIDIA',
      gpu_count: 1,
      location_code: 'FIN-02',
      market: 'spot',
      is_spot: true,
      price_per_hour: 1.25,
      price_known: true,
      currency: 'EUR',
      available: true
    }
  ]
};

const readyWorker: CloudWorker = {
  id: 'worker-1',
  provider: 'verda',
  state: 'ready',
  message: 'Ready for experiments',
  instance_id: 'instance-1',
  instance_type: '1A100.22V',
  location: 'FIN-02',
  market: 'spot',
  backends: ['cpu', 'cuda'],
  price_per_hour: 1.25,
  currency: 'EUR',
  auto_destroy: false,
  created_at: '2026-07-16T08:00:00Z'
};

const benchmark: ExperimentSpec = {
  id: 'gpu-benchmark',
  category: 'Accelerators',
  title: 'When Metal Wins',
  description: 'Compare CPU and Metal.',
  question: 'When does Metal win?',
  observe: ['The crossover point'],
  interpretation: ['Compare synchronized timings.'],
  visualization: 'backend_benchmark',
  sources: ['experiments/gpu_benchmark/gpu_benchmark.zig'],
  metrics: [],
  backends: ['metal'],
  default_backend: 'metal',
  parameters: null as unknown as ExperimentSpec['parameters']
};

const cudaExperiment: ExperimentSpec = {
  ...experiment,
  id: 'optimizer-lab',
  category: 'Training',
  title: 'Comparing Optimizers',
  description: 'Compare optimizers.',
  question: 'Which optimizer converges?',
  visualization: 'optimizer_comparison',
  metrics: [],
  backends: ['cpu', 'cuda'],
  default_backend: 'cpu',
  parameters: []
};

class FakeEventSource {
  static latest: FakeEventSource | null = null;
  listeners = new Map<string, Array<(event: MessageEvent<string>) => void>>();
  onerror: (() => void) | null = null;

  constructor(public url: string) {
    FakeEventSource.latest = this;
  }

  addEventListener(type: string, listener: (event: MessageEvent<string>) => void) {
    this.listeners.set(type, [...(this.listeners.get(type) ?? []), listener]);
  }

  close() {}

  emit(event: RunEvent) {
    const message = new MessageEvent(event.type, { data: JSON.stringify(event) });
    for (const listener of this.listeners.get(event.type) ?? []) listener(message);
  }
}

function response(value: unknown, status = 200): Response {
  return new Response(value === null ? null : JSON.stringify(value), {
    status,
    headers: value === null ? undefined : { 'Content-Type': 'application/json' }
  });
}

function runEvent(type: RunEvent['type'], seq: number, data: unknown, step?: number): RunEvent {
  return { v: 1, run_id: 'run-1', seq, type, experiment: experiment.id, step, total_steps: 100, data };
}

afterEach(() => {
  FakeEventSource.latest = null;
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe('App', () => {
  it('loads guided content and exposes labelled native controls', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValueOnce(response([experiment])).mockResolvedValueOnce(response(capabilities)));
    render(App);

    expect(await screen.findByRole('heading', { name: 'Learning XOR', level: 1 })).toBeTruthy();
    expect(screen.getByText('How does XOR become learnable?')).toBeTruthy();
    expect(screen.getByRole('spinbutton', { name: /Epochs/ })).toBeTruthy();
    expect(screen.getByRole('spinbutton', { name: /Learning rate/ })).toBeTruthy();
    expect(screen.getByRole('spinbutton', { name: /Seed/ })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Run experiment' })).toBeTruthy();
    expect(screen.getByLabelText('Run timeline')).toBeTruthy();
  });

  it('shows live progress and supports snapshot timeline scrubbing', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValueOnce(response([experiment])).mockResolvedValueOnce(response(capabilities)).mockResolvedValueOnce(response({ id: 'run-1' }, 202))
    );
    vi.stubGlobal('EventSource', FakeEventSource);
    render(App);

    await screen.findByRole('heading', { name: 'Learning XOR', level: 1 });
    await fireEvent.click(screen.getByRole('button', { name: 'Run experiment' }));
    await waitFor(() => expect(FakeEventSource.latest?.url).toBe('/api/runs/run-1/events'));

    FakeEventSource.latest!.emit(runEvent('run_started', 1, { config: {}, topology: [2, 6, 4, 1], activations: ['tanh', 'tanh', 'sigmoid'] }));
    FakeEventSource.latest!.emit(runEvent('snapshot', 2, { kind: 'xor_predictions', predictions: [] }, 0));
    FakeEventSource.latest!.emit(runEvent('metric', 3, { name: 'loss', value: 0.25 }, 10));
    FakeEventSource.latest!.emit(runEvent('snapshot', 4, { kind: 'xor_predictions', predictions: [] }, 10));

    await waitFor(() => expect(screen.getByLabelText('Run 10 percent complete')).toBeTruthy());
    const timeline = screen.getByLabelText('Run timeline') as HTMLInputElement;
    expect(timeline.value).toBe('1');
    await fireEvent.input(timeline, { target: { value: '0' } });
    expect((screen.getByRole('checkbox', { name: 'Follow live' }) as HTMLInputElement).checked).toBe(false);
    expect(screen.getByText('0 / 100')).toBeTruthy();
  });

  it('calls cancellation and renders native failure details', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(response([experiment]))
      .mockResolvedValueOnce(response(capabilities))
      .mockResolvedValueOnce(response({ id: 'run-1' }, 202))
      .mockResolvedValueOnce(response(null, 204));
    vi.stubGlobal('fetch', fetchMock);
    vi.stubGlobal('EventSource', FakeEventSource);
    render(App);

    await screen.findByRole('heading', { name: 'Learning XOR', level: 1 });
    await fireEvent.click(screen.getByRole('button', { name: 'Run experiment' }));
    await waitFor(() => expect(FakeEventSource.latest).not.toBeNull());
    await fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
    await waitFor(() => expect(fetchMock).toHaveBeenLastCalledWith('/api/runs/run-1', { method: 'DELETE' }));

    FakeEventSource.latest!.emit(runEvent('run_failed', 1, { message: 'run cancelled', cancelled: true }));
    expect((await screen.findByRole('alert')).textContent).toContain('run cancelled');
    expect(screen.getByText('cancelled')).toBeTruthy();
  });

  it('submits only parameters belonging to the selected experiment', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(response([experiment, benchmark]))
      .mockResolvedValueOnce(response(capabilities))
      .mockResolvedValueOnce(response({ id: 'run-1' }, 202));
    vi.stubGlobal('fetch', fetchMock);
    vi.stubGlobal('EventSource', FakeEventSource);
    render(App);

    await screen.findByRole('heading', { name: 'Learning XOR', level: 1 });
    await fireEvent.change(screen.getByRole('combobox', { name: 'Experiment' }), { target: { value: benchmark.id } });
    await screen.findByRole('heading', { name: 'When Metal Wins', level: 1 });
    await fireEvent.click(screen.getByRole('button', { name: 'Run experiment' }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(3));
    const request = fetchMock.mock.calls[2]?.[1] as RequestInit;
    expect(JSON.parse(String(request.body))).toEqual({ experiment: benchmark.id, backend: 'metal', parameters: {} });
  });

  it('deploys a cloud worker and submits a remote experiment target', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(response([experiment, cudaExperiment]))
      .mockResolvedValueOnce(response({ ...capabilities, cloud_enabled: true }))
      .mockResolvedValueOnce(response(cloudStatus))
      .mockResolvedValueOnce(response([]))
      .mockResolvedValueOnce(response(cloudOptions))
      .mockResolvedValueOnce(response(readyWorker, 202))
      .mockResolvedValueOnce(response(cloudStatus))
      .mockResolvedValueOnce(response({ id: 'run-1' }, 202));
    vi.stubGlobal('fetch', fetchMock);
    vi.stubGlobal('EventSource', FakeEventSource);
    render(App);

    await screen.findByRole('heading', { name: 'Learning XOR', level: 1 });
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(5));
    await fireEvent.change(screen.getByRole('combobox', { name: /Execution target/ }), { target: { value: 'cloud' } });
    expect(await screen.findByRole('heading', { name: 'Cloud worker' })).toBeTruthy();
    expect(screen.queryByRole('combobox', { name: 'SSH key' })).toBeNull();
    expect(screen.queryByRole('combobox', { name: 'Golden OS volume' })).toBeNull();
    expect(screen.getByText('packer-verda-zig-nn-volume-root')).toBeTruthy();
    expect((screen.getByRole('checkbox', { name: /Destroy after the next run/ }) as HTMLInputElement).checked).toBe(false);
    await fireEvent.click(screen.getByRole('button', { name: 'Deploy worker' }));

    await waitFor(() => expect(screen.getByText('Ready for experiments')).toBeTruthy());
    const deployRequest = fetchMock.mock.calls[5]?.[1] as RequestInit;
    expect(JSON.parse(String(deployRequest.body))).toEqual({
      instance_type: '1A100.22V',
      market: 'spot',
      location_code: 'FIN-02',
      auto_destroy: false
    });

    await fireEvent.change(screen.getByRole('combobox', { name: 'Experiment' }), { target: { value: cudaExperiment.id } });
    await screen.findByRole('heading', { name: 'Comparing Optimizers', level: 1 });
    await fireEvent.change(screen.getByRole('combobox', { name: /Backend/ }), { target: { value: 'cuda' } });

    await fireEvent.click(screen.getByRole('button', { name: 'Run experiment' }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(8));
    const runRequest = fetchMock.mock.calls[7]?.[1] as RequestInit;
    expect(JSON.parse(String(runRequest.body))).toMatchObject({
      experiment: 'optimizer-lab',
      backend: 'cuda',
      target: { kind: 'cloud', worker_id: 'worker-1' },
      acknowledge_committed_head: false
    });
  });

  it('keeps a recovered worker visible when cloud credentials are unavailable', async () => {
    const recoveredWorker: CloudWorker = {
      ...readyWorker,
      state: 'failed',
      message: 'Recovered from a previous lab session; destroy this worker before creating another',
      auto_destroy: false
    };
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(response([experiment]))
      .mockResolvedValueOnce(response({ ...capabilities, cloud_enabled: true }))
      .mockResolvedValueOnce(response({ ...cloudStatus, configured: false, error: 'Verda credentials are unavailable' }))
      .mockResolvedValueOnce(response([recoveredWorker]));
    vi.stubGlobal('fetch', fetchMock);
    vi.stubGlobal('EventSource', FakeEventSource);
    render(App);

    await screen.findByRole('heading', { name: 'Learning XOR', level: 1 });
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(4));
    await fireEvent.change(screen.getByRole('combobox', { name: /Execution target/ }), { target: { value: 'cloud' } });

    expect(await screen.findByText(recoveredWorker.message)).toBeTruthy();
    expect(screen.getByRole('alert').textContent).toContain('Verda credentials are unavailable');
    expect(screen.getByRole('button', { name: 'Destroy worker' })).toBeTruthy();
  });
});
