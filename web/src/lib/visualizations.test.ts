import { render, screen } from '@testing-library/svelte';
import { describe, expect, it } from 'vitest';
import DecisionBoundary from './DecisionBoundary.svelte';
import BackendBenchmark from './BackendBenchmark.svelte';
import ExecutionPanel from './ExecutionPanel.svelte';
import OptimizerComparison from './OptimizerComparison.svelte';
import RegressionVisualization from './RegressionVisualization.svelte';
import SemanticSimilarity from './SemanticSimilarity.svelte';
import SpectralLearning from './SpectralLearning.svelte';
import XorVisualization from './XorVisualization.svelte';
import type { RunStartedData } from './types';

const telemetry = {
  execution: { uploads: 2, upload_bytes: 64, readbacks: 0, readback_bytes: 0, kernels: 12, synchronizations: 1 },
  backend: {
    buffer_allocations: 4,
    host_to_device_transfers: 2,
    host_to_device_bytes: 64,
    device_to_host_transfers: 0,
    device_to_host_bytes: 0,
    kernel_launches: 6,
    vendor_gemm_launches: 2,
    synchronizations: 1
  }
};

describe('learning visualizations', () => {
  it('renders XOR predictions with an accessible summary', () => {
    render(XorVisualization, {
      started: { config: {}, topology: [2, 6, 4, 1], activations: ['tanh', 'tanh', 'sigmoid'] },
      snapshot: {
        step: 10,
        total: 100,
        data: { kind: 'xor_predictions', predictions: [{ x1: 0, x2: 1, expected: 1, predicted: 0.82 }] }
      }
    });
    expect(screen.getByRole('img', { name: /^XOR network predictions/ })).toBeTruthy();
    expect(screen.getByText('0.820')).toBeTruthy();
  });

  it('renders regression curves and decision-boundary sample shapes', () => {
    const regression = render(RegressionVisualization, {
      started: { config: {}, topology: [1, 1], activations: ['linear'], target_curve: [{ x: -1, y: -1 }, { x: 1, y: 1 }] },
      snapshot: { step: 1, total: 1, data: { kind: 'regression_curve', predictions: [{ x: -1, y: -0.5 }, { x: 1, y: 0.5 }] } }
    });
    expect(screen.getByRole('img', { name: /^Target and learned regression curves/ })).toBeTruthy();
    regression.unmount();

    render(DecisionBoundary, {
      started: { config: {}, topology: [2, 1], activations: ['sigmoid'], samples: [{ x: 0, y: 0, label: 1 }], grid_size: 1 },
      snapshot: { step: 1, total: 1, data: { kind: 'decision_boundary', probabilities: [{ x: 0, y: 0, value: 0.9 }] } }
    });
    expect(screen.getByRole('img', { name: /^Learned circular decision boundary/ })).toBeTruthy();
  });

  it('renders optimizer, accelerator, and execution evidence', () => {
    const started: RunStartedData = {
      config: {},
      topology: [2, 16, 1],
      activations: ['tanh', 'sigmoid'],
      samples: [{ x: 0, y: 0, label: 1 }],
      grid_size: 1,
      execution: { requested_backend: 'metal', selected_backend: 'metal', optimize: 'ReleaseFast' }
    };
    const optimizer = render(OptimizerComparison, {
      started,
      snapshot: {
        step: 2,
        total: 10,
        data: { kind: 'optimizer_comparison', optimizers: [{ name: 'sgd', predictions: [{ x: 0, y: 0, value: 0.9 }] }], telemetry }
      }
    });
    expect(screen.getByRole('img', { name: /sgd decision boundary/ })).toBeTruthy();
    optimizer.unmount();

    const benchmark = render(BackendBenchmark, {
      started,
      snapshot: {
        step: 1,
        total: 1,
        data: {
          kind: 'backend_benchmark',
          cases: [{ size: 128, trials: 5, cpu_ms: 2, accelerator_ms: 1, speedup: 2, sample_error: 1e-6, telemetry }],
          telemetry
        }
      }
    });
    expect(screen.getByRole('img', { name: /timing comparison/ })).toBeTruthy();
    benchmark.unmount();

    render(ExecutionPanel, {
      started,
      snapshot: { step: 1, total: 1, data: { kind: 'backend_benchmark', cases: [], telemetry } }
    });
    expect(screen.getByText('ReleaseFast')).toBeTruthy();
    expect(screen.getByText('Vendor GEMMs')).toBeTruthy();
  });

  it('renders semantic similarity and interactive query rankings', () => {
    render(SemanticSimilarity, {
      started: {
        config: {}, topology: [2, 2, 2], activations: ['linear'], queries: ['cat query', 'dog query'], documents: ['cat doc', 'dog doc']
      },
      snapshot: {
        step: 4,
        total: 4,
        data: { kind: 'semantic_similarity', similarities: [0.9, 0.1, 0.2, 0.8], rows: 2, columns: 2, telemetry }
      }
    });
    expect(screen.getByRole('img', { name: /Cosine similarity heatmap/ })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'cat query' })).toBeTruthy();
    expect(screen.getAllByText('cat doc')).toHaveLength(2);
  });

  it('renders spectral curves and frequency amplitudes with model context', () => {
    render(SpectralLearning, {
      started: {
        config: { steps: 1000, learning_rate: 0.01, fourier_bands: 9, seed: 42 },
        topology: [1, 32, 32, 1],
        activations: ['relu', 'relu', 'linear'],
        target_curve: [{ x: 0, y: 0 }, { x: 0.5, y: 0 }],
        target_spectrum: [0, 1 / 3, 0, 1 / 3, 0, 0, 0, 0, 0, 1 / 3],
        spectrum_limit: 9,
        models: [
          { name: 'raw', input_representation: 'coordinate', topology: [1, 32, 32, 1], parameter_count: 1153 },
          { name: 'fourier', input_representation: 'coordinate plus harmonic pairs', topology: [19, 32, 32, 1], parameter_count: 1729 }
        ]
      },
      snapshot: {
        step: 200,
        total: 1000,
        data: {
          kind: 'spectral_learning',
          series: [
            { name: 'raw', curve: [{ x: 0, y: 0.1 }], amplitudes: [0, 0.3, 0, 0.1, 0, 0, 0, 0, 0, 0.02] },
            { name: 'fourier', curve: [{ x: 0, y: 0.02 }], amplitudes: [0, 0.33, 0, 0.32, 0, 0, 0, 0, 0, 0.3] }
          ]
        }
      }
    });
    expect(screen.getByRole('img', { name: /Target and learned multi-frequency functions/ })).toBeTruthy();
    expect(screen.getByRole('img', { name: /Target and learned amplitude spectra/ })).toBeTruthy();
    expect(screen.getByText(/raw 1,153 parameters/)).toBeTruthy();
  });
});
