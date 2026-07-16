<script lang="ts">
  import type { RunStartedData, SnapshotPoint } from './types';

  let { started, snapshot }: { started: RunStartedData | null; snapshot?: SnapshotPoint } = $props();
  let selectedQuery = $state(0);
  let queries = $derived(started?.queries ?? []);
  let documents = $derived(started?.documents ?? []);
  let data = $derived(snapshot?.data.kind === 'semantic_similarity' ? snapshot.data : undefined);
  let rankings = $derived(rankDocuments(data?.similarities ?? [], data?.columns ?? 0, selectedQuery, documents));
  const cell = 58;
  const left = 145;
  const top = 92;

  function rankDocuments(values: number[], columns: number, row: number, labels: string[]) {
    if (!columns) return [];
    return labels
      .map((label, column) => ({ label, score: values[row * columns + column] ?? 0, correct: row === column }))
      .sort((a, b) => b.score - a.score);
  }

  function color(score: number): string {
    const normalized = Math.max(0, Math.min(1, (score + 1) / 2));
    return `color-mix(in srgb, var(--series-1) ${Math.round(normalized * 92)}%, var(--surface))`;
  }
</script>

<section class="semantic-visual" aria-labelledby="semantic-heading">
  <div class="section-heading">
    <h3 id="semantic-heading">Query/document similarity</h3>
    <span>{snapshot ? `step ${snapshot.step}` : 'waiting for a run'}</span>
  </div>
  {#if data && queries.length && documents.length}
    <div class="similarity-scroll">
      <svg class="similarity-chart" viewBox={`0 0 ${left + data.columns * cell + 12} ${top + data.rows * cell + 12}`} role="img" aria-label="Cosine similarity heatmap for held-out queries and documents">
        {#each documents as document, column}
          <text class="heatmap-column" transform={`translate(${left + column * cell + cell / 2} ${top - 8}) rotate(-36)`} text-anchor="start">{document}</text>
        {/each}
        {#each queries as query, row}
          <text class="heatmap-row" x={left - 10} y={top + row * cell + cell / 2 + 4} text-anchor="end">{query}</text>
          {#each documents as _, column}
            {@const score = data.similarities[row * data.columns + column] ?? 0}
            <rect
              class:correct-pair={row === column}
              x={left + column * cell}
              y={top + row * cell}
              width={cell - 3}
              height={cell - 3}
              fill={color(score)}
            />
            <text class="heatmap-value" x={left + column * cell + (cell - 3) / 2} y={top + row * cell + cell / 2 + 4} text-anchor="middle">{score.toFixed(2)}</text>
          {/each}
        {/each}
      </svg>
    </div>
    <div class="ranking-panel">
      <div class="query-tabs" aria-label="Inspect query ranking">
        {#each queries as query, index}<button class:active={selectedQuery === index} type="button" onclick={() => (selectedQuery = index)}>{query}</button>{/each}
      </div>
      <ol>
        {#each rankings as item}<li class:correct-ranking={item.correct}><span>{item.label}</span><strong>{item.score.toFixed(3)}</strong></li>{/each}
      </ol>
    </div>
  {:else}
    <p class="empty-visual">The similarity grid appears when dual-encoder training starts.</p>
  {/if}
</section>
