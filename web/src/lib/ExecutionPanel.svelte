<script lang="ts">
  import type { RunStartedData, SnapshotPoint } from './types';

  let { started, snapshot }: { started: RunStartedData | null; snapshot?: SnapshotPoint } = $props();
  let execution = $derived(started?.execution);
  let telemetry = $derived(snapshot && 'telemetry' in snapshot.data ? snapshot.data.telemetry : undefined);

  function bytes(value: number): string {
    if (value < 1024) return `${value} B`;
    if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KiB`;
    return `${(value / (1024 * 1024)).toFixed(1)} MiB`;
  }
</script>

{#if execution}
  <section class="execution-panel" aria-labelledby="execution-heading">
    <div class="section-heading">
      <h3 id="execution-heading">Native execution</h3>
      <span class:backend-metal={execution.selected_backend === 'metal'}>{execution.selected_backend}</span>
    </div>
    <dl class="execution-summary">
      <div><dt>Requested</dt><dd>{execution.requested_backend}</dd></div>
      <div><dt>Selected</dt><dd>{execution.selected_backend}</dd></div>
      <div><dt>Build mode</dt><dd>{execution.optimize}</dd></div>
    </dl>
    {#if telemetry}
      <p class="telemetry-note">Device activity during the latest training or benchmark interval</p>
      <dl class="telemetry-grid">
        <div><dt>Tensor kernels</dt><dd>{telemetry.execution.kernels}</dd></div>
        <div><dt>Native kernels</dt><dd>{telemetry.backend.kernel_launches}</dd></div>
        <div><dt>Vendor GEMMs</dt><dd>{telemetry.backend.vendor_gemm_launches}</dd></div>
        <div><dt>Uploads</dt><dd>{telemetry.execution.uploads} · {bytes(telemetry.execution.upload_bytes)}</dd></div>
        <div><dt>Readbacks</dt><dd>{telemetry.execution.readbacks} · {bytes(telemetry.execution.readback_bytes)}</dd></div>
        <div><dt>Synchronizations</dt><dd>{telemetry.backend.synchronizations}</dd></div>
      </dl>
    {/if}
  </section>
{/if}
