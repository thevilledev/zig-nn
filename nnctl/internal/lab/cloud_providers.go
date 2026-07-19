package lab

import (
	"context"
	"fmt"
	"sort"
	"strings"

	cloudcore "nnctl/internal/cloud"
	"nnctl/internal/cloud/verda"
	cloudworkflow "nnctl/internal/cloud/workflow"
)

type CloudProviderConfig struct {
	Factory       cloudcore.Factory
	Configuration cloudcore.Configuration
}

type cloudProviderRuntime struct {
	factory       cloudcore.Factory
	configuration cloudcore.Configuration
}

func buildCloudProviders(options CloudManagerOptions) (map[string]cloudProviderRuntime, error) {
	configured := options.Providers
	if len(configured) == 0 {
		factory := cloudcore.Factory(verda.ProviderFactory{})
		if options.ClientFactory != nil {
			factory = legacyVerdaFactory{newClient: options.ClientFactory}
		}
		configured = []CloudProviderConfig{{
			Factory: factory,
			Configuration: cloudcore.Configuration{
				BaseURL: options.BaseURL, UserAgent: "nnctl-lab",
			},
		}}
	}
	providers := make(map[string]cloudProviderRuntime, len(configured))
	for _, configuredProvider := range configured {
		if configuredProvider.Factory == nil {
			return nil, fmt.Errorf("cloud provider factory is required")
		}
		descriptor := configuredProvider.Factory.Descriptor()
		name := strings.ToLower(strings.TrimSpace(descriptor.Name))
		if name == "" {
			return nil, fmt.Errorf("cloud provider name is required")
		}
		if _, exists := providers[name]; exists {
			return nil, fmt.Errorf("cloud provider %q is configured more than once", name)
		}
		configuration := configuredProvider.Configuration
		if configuration.UserAgent == "" {
			configuration.UserAgent = "nnctl-lab"
		}
		providers[name] = cloudProviderRuntime{factory: configuredProvider.Factory, configuration: configuration}
	}
	return providers, nil
}

func (m *CloudManager) providerNames() []string {
	names := make([]string, 0, len(m.providers))
	for name := range m.providers {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

func (m *CloudManager) primaryProviderName() string {
	if _, ok := m.providers[verda.ProviderName]; ok {
		return verda.ProviderName
	}
	names := m.providerNames()
	if len(names) == 0 {
		return ""
	}
	return names[0]
}

func (m *CloudManager) providerRuntime(name string) (cloudProviderRuntime, error) {
	name = strings.ToLower(strings.TrimSpace(name))
	if name == "" {
		name = m.primaryProviderName()
	}
	provider, ok := m.providers[name]
	if !ok {
		return cloudProviderRuntime{}, fmt.Errorf("cloud provider %q is not enabled; enabled providers: %s", name, strings.Join(m.providerNames(), ", "))
	}
	return provider, nil
}

func (m *CloudManager) provider(ctx context.Context, name string) (cloudcore.Provider, error) {
	runtime, err := m.providerRuntime(name)
	if err != nil {
		return nil, err
	}
	provider, err := runtime.factory.New(ctx, runtime.configuration)
	if err != nil {
		return nil, fmt.Errorf("configure %s cloud provider: %w", runtime.factory.Descriptor().DisplayName, err)
	}
	return provider, nil
}

type legacyVerdaFactory struct {
	newClient cloudClientFactory
}

func (legacyVerdaFactory) Descriptor() cloudcore.Descriptor {
	return (verda.ProviderFactory{}).Descriptor()
}

func (f legacyVerdaFactory) Status(ctx context.Context, configuration cloudcore.Configuration) cloudcore.ConfigurationStatus {
	_, err := f.newClient(ctx, configuration.BaseURL)
	if err != nil {
		return cloudcore.ConfigurationStatus{Error: err.Error()}
	}
	return cloudcore.ConfigurationStatus{Configured: true}
}

func (f legacyVerdaFactory) New(ctx context.Context, configuration cloudcore.Configuration) (cloudcore.Provider, error) {
	client, err := f.newClient(ctx, configuration.BaseURL)
	if err != nil {
		return nil, err
	}
	providerClient, ok := client.(verda.ProviderClient)
	if !ok {
		return nil, fmt.Errorf("legacy Verda client does not implement the provider lifecycle")
	}
	return verda.NewCloudProvider(providerClient), nil
}

var _ cloudcore.Factory = legacyVerdaFactory{}

// Keep the compatibility factory signature until callers migrate to
// CloudProviderConfig.
type cloudClientFactory func(context.Context, string) (cloudworkflow.Client, error)
