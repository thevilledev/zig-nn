import type {
  LogData,
  MetricData,
  RunEvent,
  RunFailureData,
  RunStartedData,
  RunState,
  SnapshotData
} from './types';

export function initialRunState(status: RunState['status'] = 'idle'): RunState {
  return {
    id: null,
    status,
    lastSequence: 0,
    step: 0,
    totalSteps: 0,
    started: null,
    metrics: [],
    snapshots: [],
    logs: [],
    result: null,
    error: null
  };
}

export function reduceRunEvent(state: RunState, event: RunEvent): RunState {
  if (event.seq <= state.lastSequence) return state;
  const next: RunState = {
    ...state,
    id: event.run_id,
    lastSequence: event.seq,
    step: event.step ?? state.step,
    totalSteps: event.total_steps ?? state.totalSteps
  };

  switch (event.type) {
    case 'run_started':
      return { ...next, status: 'running', started: event.data as RunStartedData };
    case 'metric': {
      const metric = event.data as MetricData;
      if (metric.name !== 'loss' || event.step === undefined) return next;
      return { ...next, metrics: [...state.metrics, { step: event.step, value: metric.value }] };
    }
    case 'snapshot':
      if (event.step === undefined) return next;
      return {
        ...next,
        snapshots: [...state.snapshots, { step: event.step, total: event.total_steps ?? state.totalSteps, data: event.data as SnapshotData }]
      };
    case 'log':
      return { ...next, logs: [...state.logs.slice(-99), (event.data as LogData).message] };
    case 'run_completed':
      return { ...next, status: 'completed', result: event.data as Record<string, unknown> };
    case 'run_failed': {
      const failure = event.data as RunFailureData;
      return { ...next, status: failure.cancelled ? 'cancelled' : 'failed', error: failure.message };
    }
  }
}
