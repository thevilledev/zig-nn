<script lang="ts">
  import type { MetricPoint } from './types';

  let { points = [] }: { points?: MetricPoint[] } = $props();

  const width = 720;
  const height = 210;
  const left = 62;
  const right = 18;
  const top = 24;
  const bottom = 42;

  let maxStep = $derived(Math.max(1, ...points.map((point) => point.step)));
  let maxValue = $derived(Math.max(1e-9, ...points.map((point) => point.value)) * 1.05);
  let line = $derived(
    points
      .map((point, index) => {
        const x = left + (point.step / maxStep) * (width - left - right);
        const y = top + (1 - point.value / maxValue) * (height - top - bottom);
        return `${index === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`;
      })
      .join(' ')
  );
  let current = $derived(points.at(-1));
</script>

<section class="chart-section" aria-labelledby="loss-heading">
  <div class="section-heading">
    <h3 id="loss-heading">Training loss</h3>
    {#if current}<span class="live-value">{current.value.toPrecision(4)}</span>{/if}
  </div>
  <svg class="loss-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-labelledby="loss-title loss-desc">
    <title id="loss-title">Loss over training steps</title>
    <desc id="loss-desc">{points.length ? `Loss currently ${current?.value.toPrecision(4)} at step ${current?.step}.` : 'Loss will appear after training starts.'}</desc>
    <line class="axis" x1={left} y1={height - bottom} x2={width - right} y2={height - bottom} />
    <line class="axis" x1={left} y1={top} x2={left} y2={height - bottom} />
    {#each [0, 0.5, 1] as fraction}
      <line class="grid" x1={left} y1={top + fraction * (height - top - bottom)} x2={width - right} y2={top + fraction * (height - top - bottom)} />
    {/each}
    <text class="axis-label" x={left} y={height - 14}>0</text>
    <text class="axis-label" x={width - right} y={height - 14} text-anchor="end">step {maxStep}</text>
    <text class="axis-label" x={left - 10} y={top + 5} text-anchor="end">{maxValue.toPrecision(3)}</text>
    <text class="axis-label" x={left - 10} y={height - bottom + 5} text-anchor="end">0</text>
    {#if line}<path class="loss-line" d={line} />{/if}
  </svg>
</section>
