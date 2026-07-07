package verdacloud

import (
	"context"
	"errors"
	"fmt"
	"regexp"
	"strings"
	"time"

	sdkverda "github.com/verda-cloud/verdacloud-sdk-go/pkg/verda"
)

const (
	ProviderName           = "verda"
	FinlandLocationCode    = sdkverda.LocationFIN03
	ExplicitLocation       = "explicit"
	SourceOSVolumeLocation = "source_os_volume_location"
	SpotContract           = "SPOT"
	DefaultDescription     = "nnctl GPU benchmark worker"
	VolumeReadyTimeout     = 20 * time.Minute
	VolumeReadyPoll        = 5 * time.Second
)

type DeployOptions struct {
	InstanceType                string
	SourceOSVolumeID            string
	ClonedOSVolumeID            string
	Hostname                    string
	Description                 string
	SSHKeyIDs                   []string
	LocationCode                string
	StartupScriptName           string
	UserDataScript              string
	BaseURL                     string
	DryRun                      bool
	SkipAvailabilityCheck       bool
	KeepClonedOSVolumeOnFailure bool
}

type InstanceType struct {
	InstanceType string  `json:"instance_type"`
	DisplayName  string  `json:"display_name,omitempty"`
	Model        string  `json:"model,omitempty"`
	GPUCount     int     `json:"gpu_count"`
	SpotPrice    float64 `json:"spot_price,omitempty"`
	PriceKnown   bool    `json:"price_known,omitempty"`
	Currency     string  `json:"currency,omitempty"`
}

type StartupScript struct {
	ID   string `json:"id"`
	Name string `json:"name"`
}

type CloneVolumeRequest struct {
	SourceVolumeID string `json:"source_volume_id"`
	Name           string `json:"name"`
	LocationCode   string `json:"location_code,omitempty"`
}

type Volume struct {
	ID         string `json:"id"`
	Name       string `json:"name,omitempty"`
	Status     string `json:"status,omitempty"`
	Location   string `json:"location,omitempty"`
	IsOSVolume bool   `json:"is_os_volume,omitempty"`
}

type OSVolumeClone struct {
	SourceVolumeID string `json:"source_volume_id"`
	VolumeID       string `json:"volume_id,omitempty"`
	Name           string `json:"name"`
	Status         string `json:"status,omitempty"`
	LocationCode   string `json:"location_code,omitempty"`
	Reused         bool   `json:"reused,omitempty"`
}

type SpotPlacement struct {
	LocationCode string  `json:"location_code"`
	SpotPrice    float64 `json:"spot_price,omitempty"`
	PriceKnown   bool    `json:"price_known"`
	Currency     string  `json:"currency,omitempty"`
}

type CreateInstanceRequest struct {
	InstanceType    string   `json:"instance_type"`
	Image           string   `json:"image"`
	Hostname        string   `json:"hostname"`
	Description     string   `json:"description"`
	SSHKeyIDs       []string `json:"ssh_key_ids,omitempty"`
	LocationCode    string   `json:"location_code"`
	Contract        string   `json:"contract"`
	IsSpot          bool     `json:"is_spot"`
	StartupScriptID string   `json:"startup_script_id,omitempty"`
}

type Instance struct {
	ID              string  `json:"id"`
	Hostname        string  `json:"hostname"`
	Status          string  `json:"status"`
	IP              *string `json:"ip,omitempty"`
	InstanceType    string  `json:"instance_type"`
	Image           string  `json:"image"`
	Location        string  `json:"location"`
	IsSpot          bool    `json:"is_spot"`
	StartupScriptID *string `json:"startup_script_id,omitempty"`
	CreatedAt       string  `json:"created_at,omitempty"`
	PricePerHour    float64 `json:"price_per_hour,omitempty"`
	Currency        string  `json:"currency,omitempty"`
}

type DeployResult struct {
	Provider          string                `json:"provider"`
	DryRun            bool                  `json:"dry_run"`
	Policy            DeployPolicy          `json:"policy"`
	SourceOSVolumeID  string                `json:"source_os_volume_id"`
	SourceOSVolume    *Volume               `json:"source_os_volume,omitempty"`
	Request           CreateInstanceRequest `json:"request"`
	InstanceType      *InstanceType         `json:"instance_type,omitempty"`
	Placement         *SpotPlacement        `json:"placement,omitempty"`
	OSVolumeClone     *OSVolumeClone        `json:"os_volume_clone,omitempty"`
	ClonedOSVolumeIDs map[string]string     `json:"cloned_os_volume_ids,omitempty"`
	StartupScript     *StartupScript        `json:"startup_script,omitempty"`
	Instance          *Instance             `json:"instance,omitempty"`
}

type DeployPolicy struct {
	LocationCode                   string `json:"location_code,omitempty"`
	LocationSelection              string `json:"location_selection"`
	SourceOSVolumeLocked           bool   `json:"source_os_volume_locked"`
	SpotOnly                       bool   `json:"spot_only"`
	SingleGPU                      bool   `json:"single_gpu"`
	CleanupClonedOSVolumeOnFailure bool   `json:"cleanup_cloned_os_volume_on_failure"`
}

type Client interface {
	GetVolume(context.Context, string) (Volume, error)
	GetInstanceType(context.Context, string, bool, string) (InstanceType, error)
	GetSpotPlacementOptions(context.Context, string) ([]SpotPlacement, error)
	CreateStartupScript(context.Context, string, string) (StartupScript, error)
	CloneVolume(context.Context, CloneVolumeRequest) (Volume, error)
	WaitVolumeReady(context.Context, string) (Volume, error)
	DeleteVolume(context.Context, string, bool) error
	CreateInstance(context.Context, CreateInstanceRequest) (Instance, error)
}

func DefaultDeployOptions(instanceType string) DeployOptions {
	return DeployOptions{
		InstanceType:   instanceType,
		Description:    DefaultDescription,
		UserDataScript: DefaultUserDataScript,
	}
}

func Deploy(ctx context.Context, client Client, opts DeployOptions) (*DeployResult, error) {
	normalized, err := opts.normalized(time.Now().UTC())
	if err != nil {
		return nil, err
	}

	req := CreateInstanceRequest{
		InstanceType: normalized.InstanceType,
		Hostname:     normalized.Hostname,
		Description:  normalized.Description,
		SSHKeyIDs:    normalized.SSHKeyIDs,
		LocationCode: normalized.LocationCode,
		Contract:     SpotContract,
		IsSpot:       true,
	}
	locationSelection := ExplicitLocation
	if normalized.LocationCode == "" {
		locationSelection = SourceOSVolumeLocation
	}

	result := &DeployResult{
		Provider: ProviderName,
		DryRun:   normalized.DryRun,
		Policy: DeployPolicy{
			LocationCode:                   normalized.LocationCode,
			LocationSelection:              locationSelection,
			SourceOSVolumeLocked:           true,
			SpotOnly:                       true,
			SingleGPU:                      true,
			CleanupClonedOSVolumeOnFailure: !normalized.KeepClonedOSVolumeOnFailure,
		},
		SourceOSVolumeID: normalized.SourceOSVolumeID,
		Request:          req,
		OSVolumeClone: &OSVolumeClone{
			SourceVolumeID: normalized.SourceOSVolumeID,
			VolumeID:       normalized.ClonedOSVolumeID,
			Name:           osVolumeCloneName(normalized.Hostname),
			LocationCode:   normalized.LocationCode,
			Reused:         normalized.ClonedOSVolumeID != "",
		},
	}
	if normalized.DryRun {
		return result, nil
	}
	if client == nil {
		return nil, fmt.Errorf("verda client is required")
	}

	sourceVolume, err := getSourceOSVolume(ctx, client, normalized.SourceOSVolumeID)
	if err != nil {
		return nil, err
	}
	result.SourceOSVolume = &sourceVolume
	if normalized.LocationCode == "" {
		req.LocationCode = sourceVolume.Location
		result.Request = req
		result.Policy.LocationCode = sourceVolume.Location
		result.OSVolumeClone.LocationCode = sourceVolume.Location
	} else if !strings.EqualFold(normalized.LocationCode, sourceVolume.Location) {
		return nil, fmt.Errorf("source_os_volume_id %s is in %s, but --location-code requested %s", normalized.SourceOSVolumeID, sourceVolume.Location, normalized.LocationCode)
	}

	instanceType, err := client.GetInstanceType(ctx, normalized.InstanceType, true, req.LocationCode)
	if err != nil {
		return nil, fmt.Errorf("get Verda instance type %q: %w", normalized.InstanceType, err)
	}
	if instanceType.GPUCount != 1 {
		return nil, fmt.Errorf("instance type %q has %d GPUs; nnctl cloud deploy only accepts single-GPU instance types", normalized.InstanceType, instanceType.GPUCount)
	}
	result.InstanceType = &instanceType

	if !normalized.SkipAvailabilityCheck {
		placement, err := requireSpotPlacement(ctx, client, normalized.InstanceType, req.LocationCode)
		if err != nil {
			return nil, err
		}
		result.Placement = &placement
	}

	script, err := client.CreateStartupScript(ctx, normalized.StartupScriptName, normalized.UserDataScript)
	if err != nil {
		return nil, fmt.Errorf("create Verda startup script: %w", err)
	}
	result.StartupScript = &script
	req.StartupScriptID = script.ID
	result.Request = req

	clone, cloneCreated, err := prepareOSVolume(ctx, client, normalized, req.LocationCode)
	if err != nil {
		return nil, err
	}
	result.OSVolumeClone = &clone
	req.Image = clone.VolumeID
	result.Request = req

	instance, err := client.CreateInstance(ctx, req)
	if err != nil {
		return nil, createInstanceFailureError(ctx, client, normalized.SourceOSVolumeID, clone, cloneCreated, normalized.KeepClonedOSVolumeOnFailure, err)
	}
	result.Instance = &instance
	result.ClonedOSVolumeIDs = map[string]string{instance.ID: clone.VolumeID}
	return result, nil
}

func (o DeployOptions) normalized(now time.Time) (DeployOptions, error) {
	o.InstanceType = strings.TrimSpace(o.InstanceType)
	o.SourceOSVolumeID = strings.TrimSpace(o.SourceOSVolumeID)
	o.ClonedOSVolumeID = strings.TrimSpace(o.ClonedOSVolumeID)
	o.Hostname = strings.TrimSpace(o.Hostname)
	o.Description = strings.TrimSpace(o.Description)
	o.StartupScriptName = strings.TrimSpace(o.StartupScriptName)
	o.UserDataScript = strings.TrimSpace(o.UserDataScript)
	o.LocationCode = strings.ToUpper(strings.TrimSpace(o.LocationCode))
	o.BaseURL = strings.TrimSpace(o.BaseURL)

	if o.InstanceType == "" {
		return DeployOptions{}, fmt.Errorf("instance type is required")
	}
	if o.SourceOSVolumeID == "" {
		return DeployOptions{}, fmt.Errorf("source_os_volume_id is required")
	}
	if o.Description == "" {
		return DeployOptions{}, fmt.Errorf("description is required")
	}
	if o.UserDataScript == "" {
		return DeployOptions{}, fmt.Errorf("userdata script is required")
	}
	if o.Hostname == "" {
		o.Hostname = defaultHostname(o.InstanceType, now)
	}
	if o.StartupScriptName == "" {
		o.StartupScriptName = o.Hostname + "-userdata"
	}
	o.SSHKeyIDs = compactStrings(o.SSHKeyIDs)
	if err := ValidateSSHKeyIDs(o.SSHKeyIDs); err != nil {
		return DeployOptions{}, err
	}
	return o, nil
}

func getSourceOSVolume(ctx context.Context, client Client, sourceVolumeID string) (Volume, error) {
	volume, err := client.GetVolume(ctx, sourceVolumeID)
	if err != nil {
		return Volume{}, fmt.Errorf("get source Verda OS volume %s: %w", sourceVolumeID, err)
	}
	volume.ID = strings.TrimSpace(volume.ID)
	if volume.ID == "" {
		volume.ID = sourceVolumeID
	}
	volume.Location = strings.ToUpper(strings.TrimSpace(volume.Location))
	if volume.Location == "" {
		return Volume{}, fmt.Errorf("source_os_volume_id %s did not report a location", sourceVolumeID)
	}
	return volume, nil
}

func prepareOSVolume(ctx context.Context, client Client, opts DeployOptions, locationCode string) (OSVolumeClone, bool, error) {
	if opts.ClonedOSVolumeID != "" {
		clone, err := reuseClonedOSVolume(ctx, client, opts.SourceOSVolumeID, opts.ClonedOSVolumeID, opts.Hostname, locationCode)
		return clone, false, err
	}
	clone, err := cloneOSVolume(ctx, client, opts.SourceOSVolumeID, opts.Hostname, locationCode, opts.KeepClonedOSVolumeOnFailure)
	return clone, true, err
}

func cloneOSVolume(ctx context.Context, client Client, sourceVolumeID, hostname, locationCode string, keepOnFailure bool) (OSVolumeClone, error) {
	cloneReq := CloneVolumeRequest{
		SourceVolumeID: sourceVolumeID,
		Name:           osVolumeCloneName(hostname),
		LocationCode:   locationCode,
	}
	created, err := client.CloneVolume(ctx, cloneReq)
	if err != nil {
		return OSVolumeClone{}, fmt.Errorf("clone Verda OS volume %s: %w", sourceVolumeID, err)
	}

	clone := OSVolumeClone{
		SourceVolumeID: sourceVolumeID,
		VolumeID:       created.ID,
		Name:           cloneReq.Name,
		Status:         created.Status,
		LocationCode:   firstNonEmptyString(created.Location, cloneReq.LocationCode),
	}
	if clone.VolumeID == "" {
		return OSVolumeClone{}, fmt.Errorf("clone Verda OS volume %s returned an empty volume ID", sourceVolumeID)
	}

	ready, err := client.WaitVolumeReady(ctx, clone.VolumeID)
	if err != nil {
		if keepOnFailure {
			return OSVolumeClone{}, fmt.Errorf("wait for cloned OS volume %s: %w; cloned OS volume %s was kept", clone.VolumeID, err, clone.VolumeID)
		}
		if cleanupErr := deleteClonedOSVolume(ctx, client, sourceVolumeID, clone.VolumeID); cleanupErr != nil {
			return OSVolumeClone{}, errors.Join(
				fmt.Errorf("wait for cloned OS volume %s: %w", clone.VolumeID, err),
				fmt.Errorf("cleanup cloned OS volume %s: %w", clone.VolumeID, cleanupErr),
			)
		}
		return OSVolumeClone{}, fmt.Errorf("wait for cloned OS volume %s: %w", clone.VolumeID, err)
	}
	clone.Status = ready.Status
	clone.LocationCode = firstNonEmptyString(ready.Location, clone.LocationCode)
	return clone, nil
}

func reuseClonedOSVolume(ctx context.Context, client Client, sourceVolumeID, clonedVolumeID, hostname, locationCode string) (OSVolumeClone, error) {
	clonedVolumeID = strings.TrimSpace(clonedVolumeID)
	if clonedVolumeID == "" {
		return OSVolumeClone{}, fmt.Errorf("cloned_os_volume_id is required")
	}
	if strings.EqualFold(clonedVolumeID, sourceVolumeID) {
		return OSVolumeClone{}, fmt.Errorf("cloned_os_volume_id must not equal source_os_volume_id %s", sourceVolumeID)
	}

	ready, err := client.WaitVolumeReady(ctx, clonedVolumeID)
	if err != nil {
		return OSVolumeClone{}, fmt.Errorf("wait for reused cloned OS volume %s: %w", clonedVolumeID, err)
	}
	ready.ID = strings.TrimSpace(ready.ID)
	if ready.ID == "" {
		ready.ID = clonedVolumeID
	}
	ready.Location = strings.ToUpper(strings.TrimSpace(ready.Location))
	if ready.Location != "" && locationCode != "" && !strings.EqualFold(ready.Location, locationCode) {
		return OSVolumeClone{}, fmt.Errorf("cloned_os_volume_id %s is in %s, but deploy location is %s", clonedVolumeID, ready.Location, locationCode)
	}

	return OSVolumeClone{
		SourceVolumeID: sourceVolumeID,
		VolumeID:       ready.ID,
		Name:           firstNonEmptyString(ready.Name, osVolumeCloneName(hostname)),
		Status:         ready.Status,
		LocationCode:   firstNonEmptyString(ready.Location, locationCode),
		Reused:         true,
	}, nil
}

func createInstanceFailureError(ctx context.Context, client Client, sourceVolumeID string, clone OSVolumeClone, cloneCreated bool, keepClone bool, err error) error {
	baseErr := fmt.Errorf("create Verda instance: %w", err)
	if clone.VolumeID == "" {
		return baseErr
	}
	if !cloneCreated {
		return fmt.Errorf("%w; reused cloned OS volume %s was kept", baseErr, clone.VolumeID)
	}
	if keepClone {
		return fmt.Errorf("%w; cloned OS volume %s was kept", baseErr, clone.VolumeID)
	}
	if cleanupErr := deleteClonedOSVolume(ctx, client, sourceVolumeID, clone.VolumeID); cleanupErr != nil {
		return errors.Join(
			baseErr,
			fmt.Errorf("cleanup cloned OS volume %s: %w", clone.VolumeID, cleanupErr),
		)
	}
	return fmt.Errorf("%w; cleaned up cloned OS volume %s", baseErr, clone.VolumeID)
}

func deleteClonedOSVolume(ctx context.Context, client Client, sourceVolumeID, cloneVolumeID string) error {
	if cloneVolumeID == "" || cloneVolumeID == sourceVolumeID {
		return nil
	}
	return client.DeleteVolume(ctx, cloneVolumeID, true)
}

func requireSpotPlacement(ctx context.Context, client Client, instanceType, locationCode string) (SpotPlacement, error) {
	options, err := client.GetSpotPlacementOptions(ctx, instanceType)
	if err != nil {
		return SpotPlacement{}, fmt.Errorf("list Verda spot placement options for %q: %w", instanceType, err)
	}
	for _, option := range options {
		if strings.EqualFold(option.LocationCode, locationCode) {
			return option, nil
		}
	}
	return SpotPlacement{}, fmt.Errorf("verda spot instance type %q is not currently available in %s", instanceType, locationCode)
}

var hostnameInvalid = regexp.MustCompile(`[^a-z0-9-]+`)
var sshKeyIDPattern = regexp.MustCompile(`^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$`)

func defaultHostname(instanceType string, now time.Time) string {
	slug := strings.ToLower(instanceType)
	slug = strings.ReplaceAll(slug, ".", "-")
	slug = hostnameInvalid.ReplaceAllString(slug, "-")
	slug = strings.Trim(slug, "-")
	if slug == "" {
		slug = "gpu"
	}
	if len(slug) > 24 {
		slug = strings.Trim(slug[:24], "-")
	}
	name := fmt.Sprintf("nnctl-bench-%s-%s", slug, now.Format("20060102150405"))
	if len(name) <= 63 {
		return name
	}
	return strings.Trim(name[:63], "-")
}

func osVolumeCloneName(hostname string) string {
	name := strings.Trim(strings.ToLower(hostname), "-")
	name = hostnameInvalid.ReplaceAllString(name, "-")
	name = strings.Trim(name, "-")
	if name == "" {
		name = "nnctl-bench"
	}
	name += "-os"
	if len(name) <= 63 {
		return name
	}
	return strings.Trim(name[:63], "-")
}

func firstNonEmptyString(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

func compactStrings(values []string) []string {
	var compact []string
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value != "" {
			compact = append(compact, value)
		}
	}
	return compact
}

func ValidateSSHKeyIDs(ids []string) error {
	for _, id := range ids {
		if !sshKeyIDPattern.MatchString(id) {
			return fmt.Errorf("verda SSH key %q is not a UUID; run `nnctl cloud ssh-keys` to list SSH key IDs", id)
		}
	}
	return nil
}
