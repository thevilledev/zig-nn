package cli

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	cloudcore "nnctl/internal/cloud"
	"nnctl/internal/cloud/verda"
)

const defaultCloudProvider = verda.ProviderName

type cloudProviderSelection struct {
	name string
}

func (a *app) cloudProviders() (*cloudcore.Registry, error) {
	if a.cloudRegistry != nil {
		return a.cloudRegistry, nil
	}
	return cloudcore.NewRegistry(verda.ProviderFactory{})
}

func (a *app) cloudProviderFactory(name string) (cloudcore.Factory, error) {
	registry, err := a.cloudProviders()
	if err != nil {
		return nil, err
	}
	if strings.TrimSpace(name) == "" {
		name = defaultCloudProvider
	}
	return registry.Factory(name)
}

func (a *app) cloudProvider(
	ctx context.Context,
	name string,
	baseURL string,
	userAgent string,
	unauthenticated bool,
) (cloudcore.Provider, error) {
	factory, err := a.cloudProviderFactory(name)
	if err != nil {
		return nil, err
	}
	provider, err := factory.New(ctx, cloudcore.Configuration{
		BaseURL: baseURL, UserAgent: userAgent, Unauthenticated: unauthenticated,
	})
	if err != nil {
		return nil, fmt.Errorf("configure %s cloud provider: %w", factory.Descriptor().DisplayName, err)
	}
	return provider, nil
}

func decodeCloudProviderDetail[T any](detail json.RawMessage, provider, operation string) (*T, error) {
	if len(detail) == 0 {
		return nil, fmt.Errorf("%s %s result did not include provider details", provider, operation)
	}
	var result T
	if err := json.Unmarshal(detail, &result); err != nil {
		return nil, fmt.Errorf("decode %s %s result: %w", provider, operation, err)
	}
	return &result, nil
}
