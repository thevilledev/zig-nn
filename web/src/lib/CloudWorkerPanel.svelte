<script lang="ts">
  import { onDestroy } from 'svelte';
  import type { CloudDeployRequest, CloudOffering, CloudOptions, CloudStatus, CloudWorker } from './types';

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

  let selectedProvider = $state('');
  let selectedOfferingKey = $state('');
  let autoDestroy = $state(false);
  let now = $state(Date.now());
  const clock = window.setInterval(() => (now = Date.now()), 10_000);
  onDestroy(() => window.clearInterval(clock));

  let activeWorker = $derived(worker?.state === 'destroyed' ? null : worker);
  let offerings = $derived(normalizeOfferings(options));
  let configuredProviders = $derived(
    status?.providers?.filter((provider) => provider.configured) ?? (status?.configured ? [{ name: status.provider, display_name: status.provider, configured: true, capabilities: [] }] : [])
  );
  let providerOfferings = $derived(offerings.filter((offering) => offering.provider === selectedProvider));
  let selectedOffering = $derived(providerOfferings.find((offering) => offeringKey(offering) === selectedOfferingKey) ?? null);
  let selectedProviderStatus = $derived(status?.providers?.find((provider) => provider.name === selectedProvider));
  let activeProviderStatus = $derived(status?.providers?.find((provider) => provider.name === activeWorker?.provider));

  $effect(() => {
    if (!selectedProvider || !configuredProviders.some((provider) => provider.name === selectedProvider)) {
      selectedProvider = configuredProviders.find((provider) => offerings.some((offering) => offering.provider === provider.name))?.name ?? configuredProviders[0]?.name ?? '';
      selectedOfferingKey = '';
    }
  });

  $effect(() => {
    if (!providerOfferings.some((offering) => offeringKey(offering) === selectedOfferingKey)) {
      const preferred = providerOfferings.find((offering) => offering.market === 'spot') ?? providerOfferings[0];
      selectedOfferingKey = preferred ? offeringKey(preferred) : '';
    }
  });

  function normalizeOfferings(value: CloudOptions | null): CloudOffering[] {
    if (!value) return [];
    if (value.offerings?.length) return value.offerings;
    return value.prices.map((price) => ({
      provider: value.provider,
      id: price.instance_type,
      display_name: price.display_name,
      location: price.location_code,
      market: price.market,
      accelerator: { manufacturer: price.manufacturer, model: price.model, count: price.gpu_count },
      backends: price.gpu_count ? ['cpu', 'cuda'] : ['cpu'],
      price_per_hour: price.price_per_hour,
      price_known: price.price_known,
      currency: price.currency,
      available: price.available,
      discoverable: true
    }));
  }

  function offeringKey(offering: CloudOffering): string {
    return `${offering.provider}:${offering.market || ''}:${offering.location}:${offering.id}`;
  }

  function offeringLabel(offering: CloudOffering): string {
    const gpu = offering.accelerator.model || offering.display_name || offering.id;
    const count = offering.accelerator.count > 1 ? `${offering.accelerator.count}× ` : '';
    const market = offering.market ? ` · ${offering.market}` : '';
    const amount = offering.price_known ? `${offering.price_per_hour?.toFixed(3)} ${offering.currency || ''}/h` : 'price unavailable';
    return `${count}${gpu} · ${offering.id} · ${offering.location}${market} · ${amount}`;
  }

  function providerLabel(name: string): string {
    return status?.providers?.find((provider) => provider.name === name)?.display_name || name;
  }

  async function deploy() {
    if (!selectedOffering) return;
    await onDeploy({
      provider: selectedOffering.provider,
      instance_type: selectedOffering.id,
      market: selectedOffering.market || '',
      location_code: selectedOffering.location,
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
    <span>{activeWorker ? providerLabel(activeWorker.provider) : selectedProvider ? providerLabel(selectedProvider) : 'Cloud'}</span>
  </div>

  {#if !status}
    <p class="cloud-note">Loading cloud configuration…</p>
  {:else if activeWorker}
    <div class="worker-summary" aria-live="polite">
      <div><span>Provider</span><strong>{providerLabel(activeWorker.provider)}</strong></div>
      <div><span>Status</span><strong class:status-error={activeWorker.state === 'failed'}>{activeWorker.state}</strong></div>
      <div><span>Worker</span><strong>{activeWorker.instance_type}</strong></div>
      <div><span>Location</span><strong>{activeWorker.location || 'resolving'}</strong></div>
      <div><span>Price</span><strong>{activeWorker.price_per_hour ? `${activeWorker.price_per_hour.toFixed(3)} ${activeWorker.currency || ''}/h` : '—'}</strong></div>
      <div><span>Estimated cost</span><strong>{accruedCost(activeWorker)}</strong></div>
      <div><span>Cleanup</span><strong>{activeWorker.auto_destroy ? 'after run / exit' : 'reusable / manual'}</strong></div>
      {#if activeWorker.expires_at}<div><span>Idle expiry</span><strong>{new Date(activeWorker.expires_at).toLocaleTimeString()}</strong></div>{/if}
    </div>
    <p class="cloud-message">{activeWorker.message}</p>
    {#if (activeProviderStatus && !activeProviderStatus.configured) || (!status.providers?.length && !status.configured)}
      <p class="inline-error" role="alert">{activeProviderStatus?.error || status.error || `Restore ${providerLabel(activeWorker.provider)} credentials to destroy this recovered worker.`}</p>
    {/if}
    {#if activeWorker.repository_commit}<p class="cloud-note">Running committed revision <code>{activeWorker.repository_commit.slice(0, 12)}</code>.</p>{/if}
    <button
      class="destroy"
      type="button"
      disabled={loading || activeWorker.state === 'destroying'}
      onclick={() => onDestroyWorker(activeWorker!.id)}
    >Destroy worker</button>
  {:else if !status.configured}
    <p class="inline-error" role="alert">{status.error || 'No enabled cloud provider has configured credentials.'}</p>
  {:else if options}
    {#if configuredProviders.length > 1}
      <label class="cloud-field">
        <span>Provider</span>
        <select bind:value={selectedProvider} disabled={loading}>
          {#each configuredProviders as provider}<option value={provider.name}>{provider.display_name}</option>{/each}
        </select>
      </label>
    {/if}
    <label class="cloud-field">
      <span>GPU instance</span>
      <select bind:value={selectedOfferingKey} disabled={loading || !providerOfferings.length}>
        {#each providerOfferings as offering}<option value={offeringKey(offering)}>{offeringLabel(offering)}</option>{/each}
      </select>
    </label>
    {#if selectedProvider === 'verda'}
      <p class="cloud-note">Workers clone <code>{options.source_os_volume_name}</code>, including its authorized keys and CUDA/Zig tooling.</p>
    {/if}
    {#if selectedProviderStatus?.error || options.errors?.[selectedProvider]}
      <p class="inline-error" role="alert">{selectedProviderStatus?.error || options.errors?.[selectedProvider]}</p>
    {:else if !providerOfferings.length}
      <p class="inline-error" role="alert">No compatible GPU offerings are currently available from this provider.</p>
    {/if}
    <label class="cleanup-choice"><input type="checkbox" bind:checked={autoDestroy} /> Destroy after the next run or when the lab exits</label>
    <p class="cloud-note">Leave cleanup unchecked to reuse this worker for multiple experiments and across lab restarts.</p>
    <button
      class="primary"
      type="button"
      disabled={loading || !selectedOffering}
      onclick={deploy}
    >Deploy worker</button>
  {:else}
    <p class="cloud-note">Loading available GPU instances…</p>
  {/if}

  {#if error}<p class="inline-error" role="alert">{error}</p>{/if}
</section>
