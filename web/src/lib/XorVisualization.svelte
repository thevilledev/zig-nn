<script lang="ts">
  import type { RunStartedData, SnapshotPoint, XorPrediction } from './types';

  let { started, snapshot }: { started: RunStartedData | null; snapshot?: SnapshotPoint } = $props();
  let predictions = $derived(
    snapshot?.data.kind === 'xor_predictions' ? snapshot.data.predictions : ([] as XorPrediction[])
  );
  let topology = $derived(started?.topology ?? [2, 6, 4, 1]);
  const plotLeft = 150;
  const plotWidth = 500;
</script>

<svg class="experiment-chart xor-chart" viewBox="0 0 720 390" role="img" aria-labelledby="xor-title xor-desc">
  <title id="xor-title">XOR network predictions</title>
  <desc id="xor-desc">A four-layer network and probability bars for each row of the XOR truth table.</desc>

  <g aria-label="Network topology">
    {#each topology as units, index}
      {@const x = 90 + index * 180}
      {#if index < topology.length - 1}<line class="topology-link" x1={x + 34} y1="62" x2={x + 146} y2="62" />{/if}
      <circle class="topology-node" cx={x} cy="62" r="31" />
      <text class="topology-count" x={x} y="59" text-anchor="middle">{units}</text>
      <text class="topology-label" x={x} y="78" text-anchor="middle">{index === 0 ? 'input' : index === topology.length - 1 ? 'output' : `hidden ${index}`}</text>
    {/each}
  </g>

  <text class="axis-label" x={plotLeft} y="128">0 probability</text>
  <text class="axis-label" x={plotLeft + plotWidth} y="128" text-anchor="end">1 probability</text>
  {#each predictions as prediction, index}
    {@const y = 156 + index * 53}
    <text class="row-label" x="22" y={y + 19}>[{prediction.x1}, {prediction.x2}] → {prediction.expected}</text>
    <rect class="probability-track" x={plotLeft} y={y} width={plotWidth} height="25" rx="4" />
    <rect class="probability-value" x={plotLeft} y={y} width={Math.max(1, prediction.predicted * plotWidth)} height="25" rx="4" />
    <line class="target-marker" x1={plotLeft + prediction.expected * plotWidth} y1={y - 5} x2={plotLeft + prediction.expected * plotWidth} y2={y + 30} />
    <text class="probability-label" x={Math.min(plotLeft + prediction.predicted * plotWidth + 8, 681)} y={y + 19}>{prediction.predicted.toFixed(3)}</text>
  {/each}
  {#if predictions.length === 0}<text class="empty-label" x="360" y="235" text-anchor="middle">Predictions appear when a run starts</text>{/if}
</svg>
