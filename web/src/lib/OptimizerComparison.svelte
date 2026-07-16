<script lang="ts">
  import type { OptimizerBoundary, RunStartedData, Sample, SnapshotPoint } from './types';

  let { started, snapshot }: { started: RunStartedData | null; snapshot?: SnapshotPoint } = $props();
  let samples = $derived(started?.samples ?? ([] as Sample[]));
  let optimizers = $derived(snapshot?.data.kind === 'optimizer_comparison' ? snapshot.data.optimizers : ([] as OptimizerBoundary[]));
  let gridSize = $derived(started?.grid_size ?? 24);
  const left = 22;
  const top = 14;
  const plotWidth = 196;
  const plotHeight = 172;
  const xScale = (x: number) => left + ((x + 1.6) / 3.2) * plotWidth;
  const yScale = (y: number) => top + (1 - (y + 1.2) / 2.4) * plotHeight;
</script>

<section class="optimizer-visual" aria-labelledby="optimizer-boundaries-heading">
  <div class="section-heading">
    <h3 id="optimizer-boundaries-heading">Decision boundaries</h3>
    <span>{snapshot ? `step ${snapshot.step}` : 'waiting for a run'}</span>
  </div>
  <div class="optimizer-grid">
    {#each optimizers as optimizer}
      <figure>
        <svg viewBox="0 0 240 205" role="img" aria-label={`${optimizer.name} decision boundary at step ${snapshot?.step ?? 0}`}>
          <rect class="plot-background" x={left} y={top} width={plotWidth} height={plotHeight} />
          {#each optimizer.predictions as probability}
            <rect
              class:probability-high={probability.value >= 0.5}
              class:probability-low={probability.value < 0.5}
              x={xScale(probability.x) - plotWidth / gridSize / 2}
              y={yScale(probability.y) - plotHeight / gridSize / 2}
              width={plotWidth / gridSize + 0.5}
              height={plotHeight / gridSize + 0.5}
              opacity={0.08 + Math.abs(probability.value - 0.5) * 0.72}
            />
          {/each}
          {#each samples as sample}
            {#if sample.label === 1}
              <rect class="sample-positive" x={xScale(sample.x) - 2.5} y={yScale(sample.y) - 2.5} width="5" height="5" transform={`rotate(45 ${xScale(sample.x)} ${yScale(sample.y)})`} />
            {:else}
              <circle class="sample-negative" cx={xScale(sample.x)} cy={yScale(sample.y)} r="2.5" />
            {/if}
          {/each}
          <rect class="plot-outline" x={left} y={top} width={plotWidth} height={plotHeight} />
        </svg>
        <figcaption>{optimizer.name}</figcaption>
      </figure>
    {:else}
      <p class="empty-visual">The three boundaries appear when a run starts.</p>
    {/each}
  </div>
</section>
