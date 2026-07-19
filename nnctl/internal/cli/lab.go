package cli

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/spf13/cobra"

	cloudcore "nnctl/internal/cloud"
	"nnctl/internal/cloud/verda"
	learninglab "nnctl/internal/lab"
	"nnctl/internal/localhttp"
)

type labOptions struct {
	mode           string
	host           string
	port           int
	assets         string
	apiOnly        bool
	cloud          bool
	cloudProviders []string
	cloudOfferings []string
	cloudSSHKeys   []string
	cloudBaseURL   string
	stateDB        string
}

func defaultLabOptions() labOptions {
	return labOptions{
		mode: getenvDefault("NNCTL_OPTIMIZE", getenvDefault("BUILD_MODE", defaultBuildMode)),
		host: "127.0.0.1",
		port: 8091,
	}
}

func (a *app) newLabCommand(withRepo repoRunner) *cobra.Command {
	opts := defaultLabOptions()
	cmd := &cobra.Command{
		Use:   "lab",
		Short: "Run the real-time neural-network learning lab",
		Long:  "Build experiments on demand, stream typed training events, and serve the local Svelte learning interface.",
		Args:  cobra.NoArgs,
		RunE: withRepo(func(ctx context.Context, cmd *cobra.Command, args []string) error {
			return a.runLab(ctx, opts)
		}),
	}
	addModeFlags(cmd, &opts.mode)
	cmd.Flags().StringVar(&opts.host, "host", opts.host, "learning lab bind host")
	cmd.Flags().IntVar(&opts.port, "port", opts.port, "learning lab bind port")
	cmd.Flags().StringVar(&opts.assets, "assets", opts.assets, "built frontend directory (default: web/dist)")
	cmd.Flags().BoolVar(&opts.apiOnly, "api-only", opts.apiOnly, "serve only the API for Vite development")
	cmd.Flags().BoolVar(&opts.cloud, "cloud", opts.cloud, "enable Verda cloud workers (compatibility shorthand)")
	cmd.Flags().StringSliceVar(&opts.cloudProviders, "cloud-provider", opts.cloudProviders, "enable a cloud provider (repeatable)")
	cmd.Flags().StringSliceVar(&opts.cloudOfferings, "cloud-offering", opts.cloudOfferings, "add PROVIDER:OFFERING@LOCATION to a provider catalog (repeatable)")
	cmd.Flags().StringSliceVar(&opts.cloudSSHKeys, "cloud-ssh-key", opts.cloudSSHKeys, "attach PROVIDER:SSH_KEY_ID to cloud workers (repeatable)")
	cmd.Flags().StringVar(&opts.cloudBaseURL, "cloud-base-url", opts.cloudBaseURL, "Verda API base URL")
	cmd.Flags().StringVar(&opts.stateDB, "state-db", opts.stateDB, "SQLite lab state database path")
	cmd.Flags().SortFlags = false
	return cmd
}

func (a *app) runLab(ctx context.Context, opts labOptions) (runErr error) {
	if opts.port <= 0 || opts.port > 65535 {
		return fmt.Errorf("--port must be between 1 and 65535")
	}
	if !isLoopbackLabHost(opts.host) {
		return fmt.Errorf("--host must be a loopback address; use an SSH tunnel for remote access")
	}
	assets := opts.assets
	if assets == "" {
		assets = filepath.Join(a.repoRoot, "web", "dist")
	} else if !filepath.IsAbs(assets) {
		assets = filepath.Join(a.repoRoot, assets)
	}

	localExecutor := learninglab.CommandExecutor{
		RepoRoot: a.repoRoot,
		Zig:      a.zig,
		Mode:     opts.mode,
	}
	var cloud *learninglab.CloudManager
	var err error
	var executor learninglab.Executor = localExecutor
	providerNames := sortUniqueStrings(opts.cloudProviders)
	if opts.cloud && len(providerNames) == 0 {
		providerNames = []string{verda.ProviderName}
	}
	if len(providerNames) > 0 {
		cloud, err = a.newLabCloudManager(ctx, opts, providerNames)
		if err != nil {
			return err
		}
		defer func() { runErr = errors.Join(runErr, cloud.Close()) }()
		executor = learninglab.RoutingExecutor{Local: localExecutor, Cloud: cloud}
	}
	manager := learninglab.NewManager(ctx, executor)
	defer manager.Close()
	handler, err := learninglab.NewServerWithOptions(manager, assets, learninglab.ServerOptions{
		APIOnly: opts.apiOnly,
		Cloud:   cloud,
	})
	if err != nil {
		return err
	}
	address := net.JoinHostPort(opts.host, strconv.Itoa(opts.port))
	server := localhttp.NewServer(address, handler.Handler())

	if _, err := fmt.Fprintf(a.stdout(), "learning lab: http://%s\n", address); err != nil {
		return fmt.Errorf("write learning lab endpoint: %w", err)
	}
	shutdownDone := make(chan struct{})
	stopShutdown := context.AfterFunc(ctx, func() {
		defer close(shutdownDone)
		shutdownCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
		defer cancel()
		_ = server.Shutdown(shutdownCtx)
	})
	defer func() {
		if !stopShutdown() {
			<-shutdownDone
		}
	}()
	if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		return fmt.Errorf("learning lab server: %w", err)
	}
	return nil
}

func (a *app) newLabCloudManager(
	ctx context.Context,
	opts labOptions,
	providerNames []string,
) (*learninglab.CloudManager, error) {
	offeringsByProvider, err := parseLabCloudOfferings(opts.cloudOfferings, providerNames)
	if err != nil {
		return nil, err
	}
	sshKeysByProvider, err := parseLabCloudSSHKeys(opts.cloudSSHKeys, providerNames)
	if err != nil {
		return nil, err
	}
	stateDB := opts.stateDB
	if stateDB != "" && !filepath.IsAbs(stateDB) {
		stateDB = filepath.Join(a.repoRoot, stateDB)
	}
	providerConfigs := make([]learninglab.CloudProviderConfig, 0, len(providerNames))
	for _, name := range providerNames {
		factory, factoryErr := a.cloudProviderFactory(name)
		if factoryErr != nil {
			return nil, factoryErr
		}
		configuration := cloudcore.Configuration{UserAgent: "nnctl-lab"}
		if offerings := offeringsByProvider[factory.Descriptor().Name]; len(offerings) > 0 {
			setCloudConfigurationOption(&configuration, cloudcore.OptionOfferings, strings.Join(offerings, ","))
		}
		if sshKeys := sshKeysByProvider[factory.Descriptor().Name]; len(sshKeys) > 0 {
			setCloudConfigurationOption(&configuration, cloudcore.OptionSSHKeyIDs, strings.Join(sshKeys, ","))
		}
		if factory.Descriptor().Name == verda.ProviderName {
			configuration.BaseURL = opts.cloudBaseURL
		}
		providerConfigs = append(providerConfigs, learninglab.CloudProviderConfig{
			Factory: factory, Configuration: configuration,
		})
	}
	return learninglab.NewCloudManager(learninglab.CloudManagerOptions{
		Parent:       ctx,
		RepoRoot:     a.repoRoot,
		Mode:         opts.mode,
		DatabasePath: stateDB,
		Providers:    providerConfigs,
	})
}

func setCloudConfigurationOption(configuration *cloudcore.Configuration, key, value string) {
	if configuration.Options == nil {
		configuration.Options = make(map[string]string)
	}
	configuration.Options[key] = value
}

func parseLabCloudOfferings(values, enabledProviders []string) (map[string][]string, error) {
	result := make(map[string][]string)
	enabled := make(map[string]struct{}, len(enabledProviders))
	for _, provider := range enabledProviders {
		enabled[strings.ToLower(strings.TrimSpace(provider))] = struct{}{}
	}
	for _, raw := range values {
		parts := strings.SplitN(strings.TrimSpace(raw), ":", 2)
		if len(parts) != 2 || strings.TrimSpace(parts[0]) == "" || strings.TrimSpace(parts[1]) == "" || !strings.Contains(parts[1], "@") {
			return nil, fmt.Errorf("--cloud-offering %q must use PROVIDER:OFFERING@LOCATION", raw)
		}
		provider := strings.ToLower(strings.TrimSpace(parts[0]))
		if _, ok := enabled[provider]; !ok {
			return nil, fmt.Errorf("--cloud-offering provider %q is not enabled", provider)
		}
		result[provider] = append(result[provider], strings.TrimSpace(parts[1]))
	}
	return result, nil
}

func parseLabCloudSSHKeys(values, enabledProviders []string) (map[string][]string, error) {
	result := make(map[string][]string)
	enabled := make(map[string]struct{}, len(enabledProviders))
	for _, provider := range enabledProviders {
		enabled[strings.ToLower(strings.TrimSpace(provider))] = struct{}{}
	}
	for _, raw := range values {
		parts := strings.SplitN(strings.TrimSpace(raw), ":", 2)
		if len(parts) != 2 || strings.TrimSpace(parts[0]) == "" || strings.TrimSpace(parts[1]) == "" {
			return nil, fmt.Errorf("--cloud-ssh-key %q must use PROVIDER:SSH_KEY_ID", raw)
		}
		provider := strings.ToLower(strings.TrimSpace(parts[0]))
		if _, ok := enabled[provider]; !ok {
			return nil, fmt.Errorf("--cloud-ssh-key provider %q is not enabled", provider)
		}
		result[provider] = append(result[provider], strings.TrimSpace(parts[1]))
	}
	return result, nil
}

func isLoopbackLabHost(host string) bool {
	return localhttp.IsLoopbackHost(host)
}
