import { render, screen } from '@testing-library/svelte';
import { describe, expect, it } from 'vitest';
import DecisionBoundary from './DecisionBoundary.svelte';
import RegressionVisualization from './RegressionVisualization.svelte';
import XorVisualization from './XorVisualization.svelte';

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
});
