<script lang="ts">
  import type { MetricPoint, MetricSpec } from './types';

  let { spec, points = [] }: { spec: MetricSpec; points?: MetricPoint[] } = $props();

  const width = 720;
  const height = 210;
  const left = 62;
  const right = 18;
  const top = 24;
  const bottom = 42;
  const colors = ['var(--series-1)', 'var(--series-2)', 'var(--series-3)', 'var(--primary)'];

  let grouped = $derived(groupPoints(points));
  let maxStep = $derived(Math.max(1, ...points.map((point) => point.step)));
  let maxValue = $derived(Math.max(1e-9, ...points.map((point) => point.value)) * 1.05);
  let current = $derived(points.at(-1));

  function groupPoints(values: MetricPoint[]): Array<{ name: string; points: MetricPoint[] }> {
    const groups = new Map<string, MetricPoint[]>();
    for (const point of values) groups.set(point.series, [...(groups.get(point.series) ?? []), point]);
    return [...groups].map(([name, seriesPoints]) => ({ name, points: seriesPoints }));
  }

  function pathFor(values: MetricPoint[]): string {
    return values
      .map((point, index) => {
        const x = left + (point.step / maxStep) * (width - left - right);
        const y = top + (1 - point.value / maxValue) * (height - top - bottom);
        return `${index === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`;
      })
      .join(' ');
  }
</script>

<section class="chart-section" aria-labelledby={`metric-${spec.name}`}>
  <div class="section-heading">
    <h3 id={`metric-${spec.name}`}>{spec.label}</h3>
    {#if current}<span class="live-value">{current.value.toPrecision(4)}</span>{/if}
  </div>
  {#if grouped.length > 1}
    <div class="metric-legend" aria-label={`${spec.label} series`}>
      {#each grouped as series, index}<span><i style={`background:${colors[index % colors.length]}`}></i>{series.name}</span>{/each}
    </div>
  {/if}
  <svg class="loss-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-labelledby={`${spec.name}-title ${spec.name}-desc`}>
    <title id={`${spec.name}-title`}>{spec.label} over training steps</title>
    <desc id={`${spec.name}-desc`}>{points.length ? `${spec.label} currently ${current?.value.toPrecision(4)} at step ${current?.step}.` : `${spec.label} will appear after training starts.`}</desc>
    <line class="axis" x1={left} y1={height - bottom} x2={width - right} y2={height - bottom} />
    <line class="axis" x1={left} y1={top} x2={left} y2={height - bottom} />
    {#each [0, 0.5, 1] as fraction}
      <line class="grid" x1={left} y1={top + fraction * (height - top - bottom)} x2={width - right} y2={top + fraction * (height - top - bottom)} />
    {/each}
    <text class="axis-label" x={left} y={height - 14}>0</text>
    <text class="axis-label" x={width - right} y={height - 14} text-anchor="end">step {maxStep}</text>
    <text class="axis-label" x={left - 10} y={top + 5} text-anchor="end">{maxValue.toPrecision(3)}</text>
    <text class="axis-label" x={left - 10} y={height - bottom + 5} text-anchor="end">0</text>
    {#each grouped as series, index}
      <path class="metric-line" style={`stroke:${colors[index % colors.length]}`} d={pathFor(series.points)} />
    {/each}
  </svg>
</section>
