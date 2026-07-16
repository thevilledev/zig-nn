<script lang="ts">
  import { onDestroy, onMount } from 'svelte';
  import { cancelRun, connectRun, loadCapabilities, loadExperiments, startRun } from './lib/api';
  import ExecutionPanel from './lib/ExecutionPanel.svelte';
  import MetricChart from './lib/MetricChart.svelte';
  import Visualization from './lib/Visualization.svelte';
  import { initialRunState, reduceRunEvent } from './lib/run-state';
  import type { Backend, Capabilities, ExperimentSpec, RunState } from './lib/types';

  let experiments = $state<ExperimentSpec[]>([]);
  let capabilities = $state<Capabilities>({ platform: '', backends: ['cpu'] });
  let selectedID = $state('');
  let backend = $state<Backend>('cpu');
  let parameters = $state<Record<string, number>>({});
  let run = $state<RunState>(initialRunState());
  let loadingError = $state<string | null>(null);
  let actionError = $state<string | null>(null);
  let connectionMessage = $state('');
  let followLive = $state(true);
  let snapshotIndex = $state(-1);
  let closeStream: (() => void) | null = null;

  let selected = $derived(experiments.find((experiment) => experiment.id === selectedID) ?? null);
  let availableBackends = $derived(selected?.backends.filter((candidate) => capabilities.backends.includes(candidate)) ?? []);
  let selectedSnapshot = $derived(snapshotIndex >= 0 ? run.snapshots[snapshotIndex] : undefined);
  let latestLoss = $derived(run.metrics.loss?.at(-1)?.value);
  let progress = $derived(run.totalSteps ? Math.min(100, (run.step / run.totalSteps) * 100) : 0);
  let running = $derived(run.status === 'starting' || run.status === 'running');

  $effect(() => {
    const count = run.snapshots.length;
    if (followLive && count > 0) snapshotIndex = count - 1;
  });

  onMount(async () => {
    try {
      [experiments, capabilities] = await Promise.all([loadExperiments(), loadCapabilities()]);
      const requested = new URLSearchParams(window.location.search).get('experiment');
      selectExperiment(experiments.some((experiment) => experiment.id === requested) ? requested! : experiments[0]?.id ?? '');
    } catch (error) {
      loadingError = messageFor(error);
    }
  });

  onDestroy(() => closeStream?.());

  function selectExperiment(id: string) {
    closeStream?.();
    closeStream = null;
    selectedID = id;
    const experiment = experiments.find((candidate) => candidate.id === id);
    const supported = experiment?.backends.filter((candidate) => capabilities.backends.includes(candidate)) ?? [];
    backend = supported.includes(experiment?.default_backend ?? 'cpu') ? experiment!.default_backend : (supported[0] ?? 'cpu');
    parameters = Object.fromEntries((experiment?.parameters ?? []).map((parameter) => [parameter.name, parameter.default]));
    run = initialRunState();
    actionError = null;
    connectionMessage = '';
    followLive = true;
    snapshotIndex = -1;
    if (id) history.replaceState(null, '', `?experiment=${encodeURIComponent(id)}`);
  }

  function setParameter(name: string, value: string) {
    parameters = { ...parameters, [name]: Number(value) };
  }

  async function beginRun(event: SubmitEvent) {
    event.preventDefault();
    if (!selected || running) return;
    closeStream?.();
    run = initialRunState('starting');
    actionError = null;
    connectionMessage = 'Starting native experiment…';
    followLive = true;
    snapshotIndex = -1;
    try {
      const runParameters = Object.fromEntries(
        (selected.parameters ?? []).map((parameter) => [parameter.name, parameters[parameter.name] ?? parameter.default])
      );
      const id = await startRun(selected.id, backend, runParameters);
      run = { ...run, id };
      closeStream = connectRun(
        id,
        (event) => {
          run = reduceRunEvent(run, event);
          connectionMessage = '';
        },
        () => {
          if (running) connectionMessage = 'Reconnecting to the event stream…';
        }
      );
    } catch (error) {
      actionError = messageFor(error);
      run = { ...run, status: 'failed', error: actionError };
      connectionMessage = '';
    }
  }

  async function stopRun() {
    if (!run.id || !running) return;
    actionError = null;
    connectionMessage = 'Cancelling…';
    try {
      await cancelRun(run.id);
    } catch (error) {
      actionError = messageFor(error);
      connectionMessage = '';
    }
  }

  function chooseSnapshot(value: string) {
    snapshotIndex = Number(value);
    followLive = false;
  }

  function messageFor(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
  }
</script>

<svelte:head>
  <title>{selected ? `${selected.title} · zig-nn lab` : 'zig-nn learning lab'}</title>
</svelte:head>

<header class="site-header">
  <div>
    <a class="brand" href="/">zig-nn <span>learning lab</span></a>
    <p>Run the implementation. Watch the evidence. Read the code.</p>
  </div>
  {#if experiments.length}
    <label class="experiment-picker">
      <span>Experiment</span>
      <select value={selectedID} disabled={running} onchange={(event) => selectExperiment(event.currentTarget.value)}>
        {#each experiments as experiment}<option value={experiment.id}>{experiment.title}</option>{/each}
      </select>
    </label>
  {/if}
</header>

<main>
  {#if loadingError}
    <section class="notice error" role="alert"><h1>Could not load the lab</h1><p>{loadingError}</p></section>
  {:else if !selected}
    <section class="loading" aria-live="polite">Loading experiments…</section>
  {:else}
    <section class="lesson-intro">
      <p class="eyebrow">{selected.category} · native Zig experiment</p>
      <h1>{selected.title}</h1>
      <p class="lede">{selected.description}</p>
      <div class="question-block">
        <span>Learning question</span>
        <p>{selected.question}</p>
      </div>
    </section>

    <div class="lab-layout">
      <aside class="lesson-sidebar">
        <form class="controls" onsubmit={beginRun}>
          <div class="section-heading"><h2>Run controls</h2><span>{backend}</span></div>
          {#if selected.backends.length > 1}
            <label class="field">
              <span>Backend</span>
              <select bind:value={backend} disabled={running} aria-describedby="help-backend">
                {#each selected.backends as candidate}<option value={candidate} disabled={!capabilities.backends.includes(candidate)}>{candidate}{capabilities.backends.includes(candidate) ? '' : ' (unavailable)'}</option>{/each}
              </select>
              <small id="help-backend">Explicit selection: accelerator requests never silently fall back to CPU.</small>
            </label>
          {:else if !availableBackends.length}
            <p class="backend-unavailable">{selected.backends[0]} is unavailable on {capabilities.platform || 'this platform'}.</p>
          {/if}
          {#each selected.parameters as parameter}
            <label class="field">
              <span>{parameter.label}</span>
              <input
                type="number"
                min={parameter.min}
                max={parameter.max}
                step={parameter.step}
                value={parameters[parameter.name] ?? parameter.default}
                disabled={running}
                oninput={(event) => setParameter(parameter.name, event.currentTarget.value)}
                aria-describedby={`help-${parameter.name}`}
                required
              />
              <small id={`help-${parameter.name}`}>{parameter.help}</small>
            </label>
          {/each}
          <div class="actions">
            <button class="primary" type="submit" disabled={running || !availableBackends.length}>Run experiment</button>
            <button type="button" onclick={stopRun} disabled={!running}>Cancel</button>
          </div>
        </form>

        <section class="observe" aria-labelledby="observe-heading">
          <h2 id="observe-heading">What to observe</h2>
          <ul>{#each selected.observe as item}<li>{item}</li>{/each}</ul>
        </section>

        <section class="sources" aria-labelledby="sources-heading">
          <h2 id="sources-heading">Read alongside</h2>
          <ul>{#each selected.sources as source}<li><code>{source}</code></li>{/each}</ul>
        </section>
      </aside>

      <section class="workspace" aria-label="Experiment evidence">
        <div class="run-status" aria-live="polite">
          <div><span>Status</span><strong class:status-error={run.status === 'failed'}>{connectionMessage || run.status}</strong></div>
          <div><span>Selected step</span><strong>{selectedSnapshot?.step ?? run.step} / {run.totalSteps || '—'}</strong></div>
          <div><span>Backend</span><strong>{run.started?.execution?.selected_backend ?? backend}</strong></div>
          <div><span>Latest loss</span><strong>{latestLoss === undefined ? '—' : latestLoss.toPrecision(4)}</strong></div>
        </div>
        <div class="progress-track" aria-label={`Run ${progress.toFixed(0)} percent complete`}><span style={`width: ${progress}%`}></span></div>
        {#if actionError || run.error}<p class="inline-error" role="alert">{actionError || run.error}</p>{/if}

        <div class="primary-visual">
          <Visualization kind={selected.visualization} started={run.started} snapshot={selectedSnapshot} />
        </div>

        <div class="timeline-controls">
          <label for="timeline">Run timeline</label>
          <input
            id="timeline"
            type="range"
            min="0"
            max={Math.max(0, run.snapshots.length - 1)}
            value={Math.max(0, snapshotIndex)}
            disabled={run.snapshots.length < 2}
            oninput={(event) => chooseSnapshot(event.currentTarget.value)}
          />
          <label class="follow"><input type="checkbox" bind:checked={followLive} /> Follow live</label>
        </div>

        <ExecutionPanel started={run.started} snapshot={selectedSnapshot} />

        {#each selected.metrics as metric}
          <MetricChart spec={metric} points={run.metrics[metric.name] ?? []} />
        {/each}

        <section class="interpretation" aria-labelledby="interpretation-heading">
          <h2 id="interpretation-heading">How to read the evidence</h2>
          {#each selected.interpretation as paragraph}<p>{paragraph}</p>{/each}
        </section>

        {#if run.logs.length}
          <details class="diagnostics">
            <summary>Build and runtime diagnostics ({run.logs.length})</summary>
            <pre>{run.logs.join('\n')}</pre>
          </details>
        {/if}
      </section>
    </div>
  {/if}
</main>

<footer>Local-only · events and run history remain in this process</footer>
