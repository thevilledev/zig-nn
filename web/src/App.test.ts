import { fireEvent, render, screen, waitFor } from '@testing-library/svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import App from './App.svelte';
import type { ExperimentSpec, RunEvent } from './lib/types';

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
});
