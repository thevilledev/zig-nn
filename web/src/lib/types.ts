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
  category: string;
  title: string;
  description: string;
  question: string;
  observe: string[];
  interpretation: string[];
  visualization: SnapshotKind;
  sources: string[];
  parameters: ParameterSpec[];
  metrics: MetricSpec[];
  backends: Backend[];
  default_backend: Backend;
}

export type Backend = 'cpu' | 'metal' | 'cuda' | 'rocm';

export interface Capabilities {
  platform: string;
  backends: Backend[];
}

export interface MetricSpec {
  name: string;
  label: string;
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
  execution?: ExecutionMetadata;
  target_curve?: Point[];
  training_samples?: Point[];
  samples?: Sample[];
  boundary_radius?: number;
  grid_size?: number;
  queries?: string[];
  documents?: string[];
}

export interface MetricData {
  name: string;
  value: number;
  series?: string;
}

export interface ExecutionMetadata {
  requested_backend: Backend;
  selected_backend: Backend;
  optimize: string;
}

export interface ExecutionStats {
  uploads: number;
  upload_bytes: number;
  readbacks: number;
  readback_bytes: number;
  kernels: number;
  synchronizations: number;
}

export interface BackendStats {
  buffer_allocations: number;
  host_to_device_transfers: number;
  host_to_device_bytes: number;
  device_to_host_transfers: number;
  device_to_host_bytes: number;
  kernel_launches: number;
  vendor_gemm_launches: number;
  synchronizations: number;
}

export interface RuntimeTelemetry {
  execution: ExecutionStats;
  backend: BackendStats;
}

export type SnapshotKind =
  | 'xor_predictions'
  | 'regression_curve'
  | 'decision_boundary'
  | 'optimizer_comparison'
  | 'backend_benchmark'
  | 'semantic_similarity';

export interface OptimizerBoundary {
  name: string;
  predictions: Probability[];
}

export interface BenchmarkCaseResult {
  size: number;
  trials: number;
  cpu_ms: number;
  accelerator_ms: number;
  speedup: number;
  sample_error: number;
  telemetry: RuntimeTelemetry;
}

export type SnapshotData =
  | { kind: 'xor_predictions'; predictions: XorPrediction[] }
  | { kind: 'regression_curve'; predictions: Point[] }
  | { kind: 'decision_boundary'; probabilities: Probability[] }
  | { kind: 'optimizer_comparison'; optimizers: OptimizerBoundary[]; telemetry: RuntimeTelemetry }
  | { kind: 'backend_benchmark'; cases: BenchmarkCaseResult[]; telemetry: RuntimeTelemetry }
  | { kind: 'semantic_similarity'; similarities: number[]; rows: number; columns: number; telemetry: RuntimeTelemetry };

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
  series: string;
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
  metrics: Record<string, MetricPoint[]>;
  snapshots: SnapshotPoint[];
  logs: string[];
  result: Record<string, unknown> | null;
  error: string | null;
}
