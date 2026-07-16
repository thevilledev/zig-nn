<script lang="ts">
  import type { Probability, RunStartedData, Sample, SnapshotPoint } from './types';

  let { started, snapshot }: { started: RunStartedData | null; snapshot?: SnapshotPoint } = $props();
  let samples = $derived(started?.samples ?? ([] as Sample[]));
  let probabilities = $derived(snapshot?.data.kind === 'decision_boundary' ? snapshot.data.probabilities : ([] as Probability[]));
  let gridSize = $derived(started?.grid_size ?? 32);
  const plotLeft = 48;
  const plotTop = 18;
  const plotSize = 420;
  const cellSize = $derived(plotSize / gridSize);
  const xScale = (x: number) => plotLeft + ((x + 1) / 2) * plotSize;
  const yScale = (y: number) => plotTop + (1 - (y + 1) / 2) * plotSize;
</script>

<svg class="experiment-chart boundary-chart" viewBox="0 0 500 490" role="img" aria-labelledby="boundary-title boundary-desc">
  <title id="boundary-title">Learned circular decision boundary</title>
  <desc id="boundary-desc">A probability field with circle and diamond samples. The dashed circle is the target boundary.</desc>
  <rect class="plot-background" x={plotLeft} y={plotTop} width={plotSize} height={plotSize} />
  {#each probabilities as probability}
    <rect
      class:probability-high={probability.value >= 0.5}
      class:probability-low={probability.value < 0.5}
      x={xScale(probability.x) - cellSize / 2}
      y={yScale(probability.y) - cellSize / 2}
      width={cellSize + 0.5}
      height={cellSize + 0.5}
      opacity={0.08 + Math.abs(probability.value - 0.5) * 0.72}
    />
  {/each}
  <circle class="expected-boundary" cx={xScale(0)} cy={yScale(0)} r={plotSize * 0.25} />
  {#each samples as sample}
    {#if sample.label === 1}
      <rect class="sample-positive" x={xScale(sample.x) - 3} y={yScale(sample.y) - 3} width="6" height="6" transform={`rotate(45 ${xScale(sample.x)} ${yScale(sample.y)})`} />
    {:else}
      <circle class="sample-negative" cx={xScale(sample.x)} cy={yScale(sample.y)} r="3" />
    {/if}
  {/each}
  <rect class="plot-outline" x={plotLeft} y={plotTop} width={plotSize} height={plotSize} />
  <text class="axis-label" x={plotLeft} y="466">−1</text>
  <text class="axis-label" x={plotLeft + plotSize} y="466" text-anchor="end">1</text>
  <text class="axis-label" x="28" y={plotTop + 5}>1</text>
  <text class="axis-label" x="24" y={plotTop + plotSize}>−1</text>
  {#if probabilities.length === 0}<text class="empty-label" x="258" y="238" text-anchor="middle">The probability field appears when a run starts</text>{/if}
</svg>
