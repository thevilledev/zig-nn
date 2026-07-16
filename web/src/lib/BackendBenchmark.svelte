<script lang="ts">
  import type { BenchmarkCaseResult, RunStartedData, SnapshotPoint } from './types';

  let { started, snapshot }: { started: RunStartedData | null; snapshot?: SnapshotPoint } = $props();
  let cases = $derived(snapshot?.data.kind === 'backend_benchmark' ? snapshot.data.cases : ([] as BenchmarkCaseResult[]));
  let accelerator = $derived(started?.execution?.selected_backend ?? 'accelerator');
  let maxMs = $derived(Math.max(0.001, ...cases.flatMap((item) => [item.cpu_ms, item.accelerator_ms])));
  const width = 720;
  const left = 90;
  const chartWidth = 600;
  const rowHeight = 72;
</script>

<section class="benchmark-visual" aria-labelledby="benchmark-heading">
  <div class="section-heading">
    <h3 id="benchmark-heading">CPU versus {accelerator}</h3>
    <span>median synchronized time</span>
  </div>
  {#if cases.length}
    <svg class="benchmark-chart" viewBox={`0 0 ${width} ${cases.length * rowHeight + 34}`} role="img" aria-label={`CPU and ${accelerator} matrix multiplication timing comparison`}>
      {#each cases as item, index}
        <text class="row-label" x="0" y={index * rowHeight + 25}>{item.size}²</text>
        <rect class="cpu-bar" x={left} y={index * rowHeight + 8} width={(item.cpu_ms / maxMs) * chartWidth} height="18" />
        <rect class="accelerator-bar" x={left} y={index * rowHeight + 32} width={(item.accelerator_ms / maxMs) * chartWidth} height="18" />
        <text class="bar-label" x={left + 6} y={index * rowHeight + 22}>CPU {item.cpu_ms.toFixed(2)} ms</text>
        <text class="bar-label" x={left + 6} y={index * rowHeight + 46}>{accelerator} {item.accelerator_ms.toFixed(2)} ms</text>
        <text class="speedup-label" x={width - 2} y={index * rowHeight + 27} text-anchor="end">{item.speedup.toFixed(2)}×</text>
      {/each}
    </svg>
    <div class="benchmark-table-wrap">
      <table class="benchmark-table">
        <thead><tr><th>Size</th><th>Trials</th><th>CPU</th><th>{accelerator}</th><th>Speedup</th><th>Max sampled error</th></tr></thead>
        <tbody>
          {#each cases as item}
            <tr>
              <th>{item.size} × {item.size}</th><td>{item.trials}</td><td>{item.cpu_ms.toFixed(3)} ms</td><td>{item.accelerator_ms.toFixed(3)} ms</td><td>{item.speedup.toFixed(2)}×</td><td>{item.sample_error.toExponential(2)}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  {:else}
    <p class="empty-visual">Benchmark cases appear as the synchronized CPU and accelerator runs complete.</p>
  {/if}
</section>
