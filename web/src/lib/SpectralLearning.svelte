<script lang="ts">
  import type { Point, RunStartedData, SnapshotPoint, SpectralSeries } from './types';

  let { started, snapshot }: { started: RunStartedData | null; snapshot?: SnapshotPoint } = $props();
  let targetCurve = $derived(started?.target_curve ?? ([] as Point[]));
  let targetSpectrum = $derived(started?.target_spectrum ?? ([] as number[]));
  let series = $derived(snapshot?.data.kind === 'spectral_learning' ? snapshot.data.series : ([] as SpectralSeries[]));
  let allCurvePoints = $derived([...targetCurve, ...series.flatMap((item) => item.curve)]);
  let minY = $derived(Math.min(-1, ...allCurvePoints.map((point) => point.y)));
  let maxY = $derived(Math.max(1, ...allCurvePoints.map((point) => point.y)));
  let maximumAmplitude = $derived(Math.max(0.4, ...targetSpectrum, ...series.flatMap((item) => item.amplitudes)) * 1.08);
  let spectrumLimit = $derived(started?.spectrum_limit ?? Math.max(0, targetSpectrum.length - 1));
  let frequencyTicks = $derived(Array.from({ length: spectrumLimit + 1 }, (_, frequency) => frequency).filter((frequency) => spectrumLimit <= 10 || frequency % 2 === 0));
  let modelSummary = $derived((started?.models ?? []).map((model) => `${model.name} ${model.parameter_count.toLocaleString()} parameters`).join(' · '));

  const width = 720;
  const curveHeight = 300;
  const spectrumHeight = 260;
  const left = 56;
  const right = 24;
  const top = 28;
  const bottom = 42;

  const xScale = (x: number) => left + x * (width - left - right);
  const curveYScale = (y: number) => top + ((maxY - y) / (maxY - minY)) * (curveHeight - top - bottom);
  const frequencyScale = (frequency: number) => left + (frequency / Math.max(1, spectrumLimit)) * (width - left - right);
  const amplitudeScale = (amplitude: number) => top + (1 - amplitude / maximumAmplitude) * (spectrumHeight - top - bottom);
  const curvePath = (points: Point[]) => points.map((point, index) => `${index ? 'L' : 'M'} ${xScale(point.x).toFixed(2)} ${curveYScale(point.y).toFixed(2)}`).join(' ');
  const spectrumPath = (amplitudes: number[]) => amplitudes.map((amplitude, frequency) => `${frequency ? 'L' : 'M'} ${frequencyScale(frequency).toFixed(2)} ${amplitudeScale(amplitude).toFixed(2)}`).join(' ');
  const seriesClass = (name: string) => name === 'fourier' ? 'spectral-fourier' : 'spectral-raw';
</script>

<section class="spectral-visual" aria-labelledby="spectral-heading">
  <div class="section-heading">
    <h3 id="spectral-heading">Function and frequency reconstruction</h3>
    <span>{snapshot ? `step ${snapshot.step}` : 'waiting for a run'}</span>
  </div>
  {#if modelSummary}<p class="spectral-model-summary">{modelSummary}</p>{/if}

  {#if targetCurve.length}
    <svg class="experiment-chart spectral-curve-chart" viewBox={`0 0 ${width} ${curveHeight}`} role="img" aria-labelledby="spectral-curve-title spectral-curve-desc">
      <title id="spectral-curve-title">Target and learned multi-frequency functions</title>
      <desc id="spectral-curve-desc">The target function and the raw-coordinate and Fourier-feature network predictions at step {snapshot?.step ?? 0}.</desc>
      <line class="axis" x1={left} y1={curveHeight - bottom} x2={width - right} y2={curveHeight - bottom} />
      <line class="axis" x1={left} y1={top} x2={left} y2={curveHeight - bottom} />
      <line class="grid" x1={left} y1={curveYScale(0)} x2={width - right} y2={curveYScale(0)} />
      {#each [0, 0.25, 0.5, 0.75, 1] as tick}
        <text class="axis-label" x={xScale(tick)} y={curveHeight - 14} text-anchor="middle">{tick.toFixed(2)}</text>
      {/each}
      <text class="axis-label" x={left - 9} y={top + 5} text-anchor="end">{maxY.toFixed(1)}</text>
      <text class="axis-label" x={left - 9} y={curveHeight - bottom + 5} text-anchor="end">{minY.toFixed(1)}</text>
      <text class="axis-label" x={(left + width - right) / 2} y={curveHeight - 1} text-anchor="middle">coordinate x</text>
      <path class="spectral-line spectral-target" d={curvePath(targetCurve)} />
      {#each series as item}<path class={`spectral-line ${seriesClass(item.name)}`} d={curvePath(item.curve)} />{/each}
      <g class="legend spectral-legend" transform="translate(360 20)">
        <line class="spectral-line spectral-target" x1="0" y1="0" x2="25" y2="0" /><text x="31" y="4">target</text>
        <line class="spectral-line spectral-raw" x1="92" y1="0" x2="117" y2="0" /><text x="123" y="4">raw</text>
        <line class="spectral-line spectral-fourier" x1="175" y1="0" x2="200" y2="0" /><text x="206" y="4">Fourier</text>
      </g>
    </svg>

    <svg class="experiment-chart spectral-spectrum-chart" viewBox={`0 0 ${width} ${spectrumHeight}`} role="img" aria-labelledby="spectral-spectrum-title spectral-spectrum-desc">
      <title id="spectral-spectrum-title">Target and learned amplitude spectra</title>
      <desc id="spectral-spectrum-desc">Amplitude by frequency bin for the target and both network predictions. The target contains equal components at bins one, three, and nine.</desc>
      <line class="axis" x1={left} y1={spectrumHeight - bottom} x2={width - right} y2={spectrumHeight - bottom} />
      <line class="axis" x1={left} y1={top} x2={left} y2={spectrumHeight - bottom} />
      {#each [0, 0.5, 1] as fraction}
        <line class="grid" x1={left} y1={top + fraction * (spectrumHeight - top - bottom)} x2={width - right} y2={top + fraction * (spectrumHeight - top - bottom)} />
      {/each}
      {#each frequencyTicks as frequency}
        <text class="axis-label" x={frequencyScale(frequency)} y={spectrumHeight - 14} text-anchor="middle">{frequency}</text>
      {/each}
      <text class="axis-label" x={left - 9} y={top + 5} text-anchor="end">{maximumAmplitude.toFixed(2)}</text>
      <text class="axis-label" x={left - 9} y={spectrumHeight - bottom + 5} text-anchor="end">0</text>
      <text class="axis-label" x={(left + width - right) / 2} y={spectrumHeight - 1} text-anchor="middle">frequency bin</text>
      <path class="spectral-line spectral-target" d={spectrumPath(targetSpectrum)} />
      {#each targetSpectrum as amplitude, frequency}<circle class="spectral-marker spectral-target-marker" cx={frequencyScale(frequency)} cy={amplitudeScale(amplitude)} r="3" />{/each}
      {#each series as item}
        <path class={`spectral-line ${seriesClass(item.name)}`} d={spectrumPath(item.amplitudes)} />
        {#each item.amplitudes as amplitude, frequency}
          {#if item.name === 'fourier'}
            <rect class="spectral-marker spectral-fourier-marker" x={frequencyScale(frequency) - 2.8} y={amplitudeScale(amplitude) - 2.8} width="5.6" height="5.6" />
          {:else}
            <circle class="spectral-marker spectral-raw-marker" cx={frequencyScale(frequency)} cy={amplitudeScale(amplitude)} r="2.8" />
          {/if}
        {/each}
      {/each}
    </svg>
  {:else}
    <p class="empty-visual">The target, learned curves, and amplitude spectra appear when a run starts.</p>
  {/if}
</section>
