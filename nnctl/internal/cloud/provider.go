// Package cloud defines the provider-neutral compute contracts used by nnctl.
package cloud

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

type Capability string

const (
	CapabilityCompute        Capability = "compute"
	CapabilityPricing        Capability = "pricing"
	CapabilitySSHKeys        Capability = "ssh-keys"
	CapabilityVolumes        Capability = "volumes"
	CapabilityImageTemplate  Capability = "image-template"
	CapabilityBenchmarkImage Capability = "benchmark-image"
)

type Descriptor struct {
	Name          string       `json:"name"`
	DisplayName   string       `json:"display_name"`
	DefaultMarket string       `json:"default_market,omitempty"`
	Capabilities  []Capability `json:"capabilities"`
}

func (d Descriptor) Supports(capability Capability) bool {
	for _, candidate := range d.Capabilities {
		if candidate == capability {
			return true
		}
	}
	return false
}

type Configuration struct {
	BaseURL         string
	UserAgent       string
	Unauthenticated bool
	Options         map[string]string
}

const (
	// OptionOfferings configures explicit OFFERING@LOCATION entries that a
	// provider cannot discover from its public catalog, such as contract plans.
	OptionOfferings = "offerings"
	// OptionSSHKeyIDs configures server-side SSH keys without exposing their
	// identifiers to the browser-facing lab API.
	OptionSSHKeyIDs = "ssh-key-ids"
)

type ConfigurationStatus struct {
	Configured bool   `json:"configured"`
	Error      string `json:"error,omitempty"`
}

type Accelerator struct {
	Manufacturer string `json:"manufacturer,omitempty"`
	Model        string `json:"model,omitempty"`
	Count        int    `json:"count"`
	MemoryMiB    int    `json:"memory_mib,omitempty"`
}

type Offering struct {
	Provider       string          `json:"provider"`
	ID             string          `json:"id"`
	DisplayName    string          `json:"display_name,omitempty"`
	Location       string          `json:"location"`
	Market         string          `json:"market,omitempty"`
	Accelerator    Accelerator     `json:"accelerator"`
	Backends       []string        `json:"backends"`
	PricePerHour   float64         `json:"price_per_hour,omitempty"`
	PriceKnown     bool            `json:"price_known"`
	Currency       string          `json:"currency,omitempty"`
	Available      bool            `json:"available"`
	Discoverable   bool            `json:"discoverable"`
	ProviderDetail json.RawMessage `json:"provider_details,omitempty"`
}

type OfferingFilters struct {
	IDs           []string
	Locations     []string
	Markets       []string
	Manufacturer  string
	Model         string
	GPUCounts     []int
	AvailableOnly bool
	Currency      string
}

type InstanceState string

const (
	InstancePending    InstanceState = "pending"
	InstanceRunning    InstanceState = "running"
	InstanceStopped    InstanceState = "stopped"
	InstanceDestroying InstanceState = "destroying"
	InstanceDestroyed  InstanceState = "destroyed"
	InstanceFailed     InstanceState = "failed"
	InstanceUnknown    InstanceState = "unknown"
)

func (s InstanceState) Terminal() bool {
	switch s {
	case InstanceDestroyed, InstanceFailed:
		return true
	default:
		return false
	}
}

type Instance struct {
	Provider       string          `json:"provider"`
	ID             string          `json:"id"`
	Name           string          `json:"name,omitempty"`
	State          InstanceState   `json:"state"`
	PublicIP       string          `json:"public_ip,omitempty"`
	OfferingID     string          `json:"offering_id,omitempty"`
	Location       string          `json:"location,omitempty"`
	Market         string          `json:"market,omitempty"`
	Backends       []string        `json:"backends,omitempty"`
	PricePerHour   float64         `json:"price_per_hour,omitempty"`
	Currency       string          `json:"currency,omitempty"`
	ProviderDetail json.RawMessage `json:"provider_details,omitempty"`
}

type ResourceRef struct {
	Kind     string            `json:"kind"`
	ID       string            `json:"id"`
	Preserve bool              `json:"preserve,omitempty"`
	Metadata map[string]string `json:"metadata,omitempty"`
}

type DeployRequest struct {
	OfferingID  string            `json:"offering_id"`
	Location    string            `json:"location,omitempty"`
	Market      string            `json:"market,omitempty"`
	Image       string            `json:"image,omitempty"`
	Name        string            `json:"name,omitempty"`
	Description string            `json:"description,omitempty"`
	SSHKeyIDs   []string          `json:"ssh_key_ids,omitempty"`
	UserData    string            `json:"user_data,omitempty"`
	Labels      map[string]string `json:"labels,omitempty"`
	DryRun      bool              `json:"dry_run,omitempty"`
	Options     map[string]string `json:"options,omitempty"`
}

type Deployment struct {
	Provider       string          `json:"provider"`
	DryRun         bool            `json:"dry_run"`
	Request        DeployRequest   `json:"request"`
	Instance       *Instance       `json:"instance,omitempty"`
	Resources      []ResourceRef   `json:"resources,omitempty"`
	ProviderDetail json.RawMessage `json:"provider_details,omitempty"`
}

type ListOptions struct {
	Status string
	All    bool
}

type DestroyRequest struct {
	InstanceIDs []string      `json:"instance_ids"`
	Resources   []ResourceRef `json:"resources,omitempty"`
	Permanent   bool          `json:"permanent,omitempty"`
	DryRun      bool          `json:"dry_run,omitempty"`
}

type DestroyResult struct {
	Provider       string            `json:"provider"`
	DryRun         bool              `json:"dry_run"`
	Request        DestroyRequest    `json:"request"`
	Errors         map[string]string `json:"errors,omitempty"`
	ProviderDetail json.RawMessage   `json:"provider_details,omitempty"`
}

// Provider is the common SSH-addressable compute lifecycle. Optional cloud
// features are exposed by separate capability interfaces.
type Provider interface {
	Descriptor() Descriptor
	Offerings(context.Context, OfferingFilters) ([]Offering, error)
	Deploy(context.Context, DeployRequest) (*Deployment, error)
	Instance(context.Context, string) (Instance, error)
	Instances(context.Context, ListOptions) ([]Instance, error)
	Destroy(context.Context, DestroyRequest) (*DestroyResult, error)
}

type Factory interface {
	Descriptor() Descriptor
	Status(context.Context, Configuration) ConfigurationStatus
	New(context.Context, Configuration) (Provider, error)
}

type SSHKey struct {
	Provider       string          `json:"provider"`
	ID             string          `json:"id"`
	Name           string          `json:"name"`
	Fingerprint    string          `json:"fingerprint,omitempty"`
	ProviderDetail json.RawMessage `json:"provider_details,omitempty"`
}

type SSHKeyProvider interface {
	SSHKeys(context.Context) ([]SSHKey, error)
}

type ImageRequest struct {
	Offering Offering
	Image    string
	Purpose  string
}

type PreparedImage struct {
	Image     string
	Resources []ResourceRef
}

type ImagePreparer interface {
	PrepareImage(context.Context, ImageRequest) (PreparedImage, error)
}

type Volume struct {
	Provider       string          `json:"provider"`
	ID             string          `json:"id"`
	Name           string          `json:"name,omitempty"`
	State          string          `json:"state,omitempty"`
	Location       string          `json:"location,omitempty"`
	SizeGiB        int             `json:"size_gib,omitempty"`
	Bootable       bool            `json:"bootable,omitempty"`
	Deleted        bool            `json:"deleted,omitempty"`
	ProviderDetail json.RawMessage `json:"provider_details,omitempty"`
}

type VolumeListOptions struct {
	IncludeDeleted bool
}

type VolumePurgeRequest struct {
	IDs        []string `json:"ids,omitempty"`
	AllDeleted bool     `json:"all_deleted,omitempty"`
	DryRun     bool     `json:"dry_run,omitempty"`
}

type VolumePurgeResult struct {
	Provider       string             `json:"provider"`
	DryRun         bool               `json:"dry_run"`
	Request        VolumePurgeRequest `json:"request"`
	PurgedIDs      []string           `json:"purged_ids,omitempty"`
	ProviderDetail json.RawMessage    `json:"provider_details,omitempty"`
}

type VolumeProvider interface {
	Volumes(context.Context, VolumeListOptions) ([]Volume, error)
	PurgeVolumes(context.Context, VolumePurgeRequest) (*VolumePurgeResult, error)
}

type TemplateFile struct {
	Path    string
	Content string
}

type ImageTemplateProvider interface {
	ImageTemplateFiles() []TemplateFile
}

type ImageTemplateFactory interface {
	ImageTemplateFiles() []TemplateFile
}

type Registry struct {
	factories map[string]Factory
}

func NewRegistry(factories ...Factory) (*Registry, error) {
	registry := &Registry{factories: make(map[string]Factory, len(factories))}
	for _, factory := range factories {
		if factory == nil {
			return nil, fmt.Errorf("cloud provider factory is nil")
		}
		name, err := normalizeProviderName(factory.Descriptor().Name)
		if err != nil {
			return nil, err
		}
		if _, exists := registry.factories[name]; exists {
			return nil, fmt.Errorf("cloud provider %q is registered more than once", name)
		}
		registry.factories[name] = factory
	}
	return registry, nil
}

func (r *Registry) Factory(name string) (Factory, error) {
	normalized, err := normalizeProviderName(name)
	if err != nil {
		return nil, err
	}
	factory, ok := r.factories[normalized]
	if !ok {
		return nil, fmt.Errorf("unknown cloud provider %q; available providers: %s", normalized, strings.Join(r.Names(), ", "))
	}
	return factory, nil
}

func (r *Registry) Names() []string {
	names := make([]string, 0, len(r.factories))
	for name := range r.factories {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

func normalizeProviderName(name string) (string, error) {
	name = strings.ToLower(strings.TrimSpace(name))
	if name == "" {
		return "", fmt.Errorf("cloud provider name is required")
	}
	for index, char := range name {
		if (char >= 'a' && char <= 'z') || (index > 0 && char >= '0' && char <= '9') || (index > 0 && char == '-') {
			continue
		}
		return "", fmt.Errorf("cloud provider name %q must use lowercase letters, digits, and hyphens", name)
	}
	return name, nil
}
