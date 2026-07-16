<script lang="ts">
  import { onDestroy } from 'svelte';
  import type { CloudDeployRequest, CloudOptions, CloudPrice, CloudStatus, CloudWorker } from './types';

  let {
    status,
    options,
    worker,
    loading,
    error,
    onDeploy,
    onDestroyWorker
  }: {
    status: CloudStatus | null;
    options: CloudOptions | null;
    worker: CloudWorker | null;
    loading: boolean;
    error: string | null;
    onDeploy: (request: CloudDeployRequest) => void | Promise<void>;
    onDestroyWorker: (id: string) => void | Promise<void>;
  } = $props();

  let selectedPriceKey = $state('');
  let autoDestroy = $state(true);
  let now = $state(Date.now());
  const clock = window.setInterval(() => (now = Date.now()), 10_000);
  onDestroy(() => window.clearInterval(clock));

  let activeWorker = $derived(worker?.state === 'destroyed' ? null : worker);
  let selectedPrice = $derived(options?.prices.find((price) => priceKey(price) === selectedPriceKey) ?? null);

  $effect(() => {
    if (!selectedPriceKey && options?.prices.length) {
      selectedPriceKey = priceKey(options.prices.find((price) => price.market === 'spot') ?? options.prices[0]);
    }
  });

  function priceKey(price: CloudPrice): string {
    return `${price.market}:${price.location_code}:${price.instance_type}`;
  }

  function priceLabel(price: CloudPrice): string {
    const gpu = price.model || price.display_name || price.instance_type;
    const amount = price.price_known ? `${price.price_per_hour?.toFixed(3)} ${price.currency || ''}/h` : 'price unavailable';
    return `${gpu} · ${price.instance_type} · ${price.location_code} · ${price.market} · ${amount}`;
  }

  async function deploy() {
    if (!selectedPrice) return;
    await onDeploy({
      instance_type: selectedPrice.instance_type,
      market: selectedPrice.market,
      location_code: selectedPrice.location_code,
      auto_destroy: autoDestroy
    });
  }

  function accruedCost(value: CloudWorker): string {
    if (!value.price_per_hour) return '—';
    const hours = Math.max(0, now - Date.parse(value.created_at)) / 3_600_000;
    return `${(hours * value.price_per_hour).toFixed(3)} ${value.currency || ''}`;
  }
</script>

<section class="cloud-panel" aria-labelledby="cloud-heading">
  <div class="section-heading">
    <h2 id="cloud-heading">Cloud worker</h2>
    <span>{status?.provider ?? 'Verda'}</span>
  </div>

  {#if !status}
    <p class="cloud-note">Loading cloud configuration…</p>
  {:else if activeWorker}
    <div class="worker-summary" aria-live="polite">
      <div><span>Status</span><strong class:status-error={activeWorker.state === 'failed'}>{activeWorker.state}</strong></div>
      <div><span>Worker</span><strong>{activeWorker.instance_type}</strong></div>
      <div><span>Location</span><strong>{activeWorker.location || 'resolving'}</strong></div>
      <div><span>Price</span><strong>{activeWorker.price_per_hour ? `${activeWorker.price_per_hour.toFixed(3)} ${activeWorker.currency || ''}/h` : '—'}</strong></div>
      <div><span>Estimated cost</span><strong>{accruedCost(activeWorker)}</strong></div>
      <div><span>Cleanup</span><strong>{activeWorker.auto_destroy ? 'after run / exit' : 'manual'}</strong></div>
      {#if activeWorker.expires_at}<div><span>Idle expiry</span><strong>{new Date(activeWorker.expires_at).toLocaleTimeString()}</strong></div>{/if}
    </div>
    <p class="cloud-message">{activeWorker.message}</p>
    {#if !status.configured}
      <p class="inline-error" role="alert">{status.error || 'Restore the Verda credentials in the OS keyring to destroy this recovered worker.'}</p>
    {/if}
    {#if activeWorker.repository_commit}<p class="cloud-note">Running committed revision <code>{activeWorker.repository_commit.slice(0, 12)}</code>.</p>{/if}
    <button
      class="destroy"
      type="button"
      disabled={loading || activeWorker.state === 'destroying'}
      onclick={() => onDestroyWorker(activeWorker!.id)}
    >Destroy worker</button>
  {:else if !status.configured}
    <p class="inline-error" role="alert">{status.error || 'Verda credentials are not configured in the OS keyring.'}</p>
  {:else if options}
    <label class="cloud-field">
      <span>GPU instance</span>
      <select bind:value={selectedPriceKey} disabled={loading || !options.prices.length}>
        {#each options.prices as price}<option value={priceKey(price)}>{priceLabel(price)}</option>{/each}
      </select>
    </label>
    <p class="cloud-note">Workers always clone <code>{options.source_os_volume_name}</code>. Its authorized keys and CUDA/Zig tooling are baked in, so the lab does not attach SSH key IDs.</p>
    {#if !options.prices.length}<p class="inline-error" role="alert">No available GPU instances share a location with the ready golden volume.</p>{/if}
    <label class="cleanup-choice"><input type="checkbox" bind:checked={autoDestroy} /> Automatically destroy after the run or when the lab exits</label>
    <button
      class="primary"
      type="button"
      disabled={loading || !selectedPrice}
      onclick={deploy}
    >Deploy worker</button>
  {:else}
    <p class="cloud-note">Loading available GPU instances…</p>
  {/if}

  {#if error}<p class="inline-error" role="alert">{error}</p>{/if}
</section>
