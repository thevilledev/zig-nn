import { describe, expect, it } from 'vitest';
import { initialRunState, reduceRunEvent } from './run-state';
import type { RunEvent } from './types';

function event(type: RunEvent['type'], seq: number, data: unknown, step?: number): RunEvent {
  return { v: 1, run_id: 'run-1', seq, type, experiment: 'xor-training', step, total_steps: 100, data };
}

describe('reduceRunEvent', () => {
  it('collects ordered metrics and snapshots and ignores replayed events', () => {
    let state = initialRunState('starting');
    state = reduceRunEvent(state, event('run_started', 1, { config: {}, topology: [2, 1], activations: ['sigmoid'] }));
    state = reduceRunEvent(state, event('metric', 2, { name: 'loss', value: 0.75 }, 10));
    const afterMetric = state;
    state = reduceRunEvent(state, event('metric', 2, { name: 'loss', value: 99 }, 10));
    expect(state).toBe(afterMetric);
    state = reduceRunEvent(state, event('snapshot', 3, { kind: 'xor_predictions', predictions: [] }, 10));
    state = reduceRunEvent(state, event('run_completed', 4, { final_loss: 0.75 }, 100));

    expect(state.status).toBe('completed');
    expect(state.metrics).toEqual([{ step: 10, value: 0.75 }]);
    expect(state.snapshots).toHaveLength(1);
    expect(state.lastSequence).toBe(4);
  });

  it('retains bounded logs and distinguishes cancellation from failure', () => {
    let state = initialRunState('running');
    for (let seq = 1; seq <= 110; seq++) state = reduceRunEvent(state, event('log', seq, { message: `line ${seq}` }));
    expect(state.logs).toHaveLength(100);
    state = reduceRunEvent(state, event('run_failed', 111, { message: 'run cancelled', cancelled: true }));
    expect(state.status).toBe('cancelled');
    expect(state.error).toBe('run cancelled');
  });
});
