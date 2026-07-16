<script lang="ts">
  import type { Component } from 'svelte';
  import BackendBenchmark from './BackendBenchmark.svelte';
  import DecisionBoundary from './DecisionBoundary.svelte';
  import OptimizerComparison from './OptimizerComparison.svelte';
  import RegressionVisualization from './RegressionVisualization.svelte';
  import SemanticSimilarity from './SemanticSimilarity.svelte';
  import XorVisualization from './XorVisualization.svelte';
  import type { RunStartedData, SnapshotKind, SnapshotPoint } from './types';

  type VisualizationProps = { started: RunStartedData | null; snapshot?: SnapshotPoint };

  let { kind, started, snapshot }: VisualizationProps & { kind: SnapshotKind } = $props();

  const registry = {
    xor_predictions: XorVisualization,
    regression_curve: RegressionVisualization,
    decision_boundary: DecisionBoundary,
    optimizer_comparison: OptimizerComparison,
    backend_benchmark: BackendBenchmark,
    semantic_similarity: SemanticSimilarity
  } satisfies Record<SnapshotKind, Component<VisualizationProps>>;

  let Visual = $derived(registry[kind]);
</script>

<Visual {started} {snapshot} />
