<script lang="ts">
  import type { Point, RunStartedData, SnapshotPoint } from './types';

  let { started, snapshot }: { started: RunStartedData | null; snapshot?: SnapshotPoint } = $props();
  let target = $derived(started?.target_curve ?? ([] as Point[]));
  let samples = $derived(started?.training_samples ?? ([] as Point[]));
  let predictions = $derived(snapshot?.data.kind === 'regression_curve' ? snapshot.data.predictions : ([] as Point[]));
  let allPoints = $derived([...target, ...predictions, ...samples]);
  let minY = $derived(Math.min(-1, ...allPoints.map((point) => point.y)));
  let maxY = $derived(Math.max(1, ...allPoints.map((point) => point.y)));

  const width = 720;
  const height = 390;
  const left = 62;
  const right = 22;
  const top = 22;
  const bottom = 44;
  const xScale = (x: number) => left + ((x + 4) / 8) * (width - left - right);
  const yScale = (y: number) => top + ((maxY - y) / (maxY - minY)) * (height - top - bottom);
  const pathFor = (points: Point[]) => points.map((point, index) => `${index ? 'L' : 'M'} ${xScale(point.x).toFixed(2)} ${yScale(point.y).toFixed(2)}`).join(' ');
  let targetPath = $derived(pathFor(target));
  let predictionPath = $derived(pathFor(predictions));
</script>

<svg class="experiment-chart regression-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-labelledby="regression-title regression-desc">
  <title id="regression-title">Target and learned regression curves</title>
  <desc id="regression-desc">Training samples, the target x squared sine x curve, and the network prediction at the selected step.</desc>
  <line class="axis" x1={left} y1={height - bottom} x2={width - right} y2={height - bottom} />
  <line class="axis" x1={left} y1={top} x2={left} y2={height - bottom} />
  {#if minY <= 0 && maxY >= 0}<line class="grid" x1={left} y1={yScale(0)} x2={width - right} y2={yScale(0)} />{/if}
  {#each [-4, -2, 0, 2, 4] as tick}
    <text class="axis-label" x={xScale(tick)} y={height - 16} text-anchor="middle">{tick}</text>
  {/each}
  <text class="axis-label" x={left - 10} y={top + 5} text-anchor="end">{maxY.toFixed(1)}</text>
  <text class="axis-label" x={left - 10} y={height - bottom + 5} text-anchor="end">{minY.toFixed(1)}</text>
  {#each samples as sample}<circle class="training-sample" cx={xScale(sample.x)} cy={yScale(sample.y)} r="2.4" />{/each}
  {#if targetPath}<path class="target-curve" d={targetPath} />{/if}
  {#if predictionPath}<path class="prediction-curve" d={predictionPath} />{/if}
  <g class="legend" transform="translate(475 28)">
    <line class="target-curve" x1="0" y1="0" x2="28" y2="0" /><text x="36" y="5">target</text>
    <line class="prediction-curve" x1="105" y1="0" x2="133" y2="0" /><text x="141" y="5">prediction</text>
  </g>
  {#if predictions.length === 0}<text class="empty-label" x="380" y="205" text-anchor="middle">The learned curve appears when a run starts</text>{/if}
</svg>
