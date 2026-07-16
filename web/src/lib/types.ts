export type ParameterKind = 'integer' | 'number';

export interface ParameterSpec {
  name: string;
  label: string;
  help: string;
  kind: ParameterKind;
  default: number;
  min: number;
  max: number;
  step: number;
}

export interface ExperimentSpec {
  id: string;
  title: string;
  description: string;
  question: string;
  observe: string[];
  interpretation: string[];
  visualization: SnapshotKind;
  sources: string[];
  parameters: ParameterSpec[];
}

export interface Point {
  x: number;
  y: number;
}

export interface Sample extends Point {
  label: number;
}

export interface Probability extends Point {
  value: number;
}

export interface XorPrediction {
  x1: number;
  x2: number;
  predicted: number;
  expected: number;
}

export interface RunStartedData {
  config: Record<string, number>;
  topology: number[];
  activations: string[];
  target_curve?: Point[];
  training_samples?: Point[];
  samples?: Sample[];
  boundary_radius?: number;
  grid_size?: number;
}

export interface MetricData {
  name: string;
  value: number;
}

export type SnapshotKind = 'xor_predictions' | 'regression_curve' | 'decision_boundary';

export type SnapshotData =
  | { kind: 'xor_predictions'; predictions: XorPrediction[] }
  | { kind: 'regression_curve'; predictions: Point[] }
  | { kind: 'decision_boundary'; probabilities: Probability[] };

export interface RunFailureData {
  message: string;
  cancelled?: boolean;
}

export interface LogData {
  message: string;
}

export interface RunEvent<T = unknown> {
  v: number;
  run_id: string;
  seq: number;
  type: 'run_started' | 'metric' | 'snapshot' | 'run_completed' | 'run_failed' | 'log';
  experiment: string;
  step?: number;
  total_steps?: number;
  data: T;
}

export interface MetricPoint {
  step: number;
  value: number;
}

export interface SnapshotPoint {
  step: number;
  total: number;
  data: SnapshotData;
}

export type RunStatus = 'idle' | 'starting' | 'running' | 'completed' | 'failed' | 'cancelled';

export interface RunState {
  id: string | null;
  status: RunStatus;
  lastSequence: number;
  step: number;
  totalSteps: number;
  started: RunStartedData | null;
  metrics: MetricPoint[];
  snapshots: SnapshotPoint[];
  logs: string[];
  result: Record<string, unknown> | null;
  error: string | null;
}
