package cli

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"path/filepath"
	"strconv"
	"time"

	"github.com/spf13/cobra"

	learninglab "nnctl/internal/lab"
	"nnctl/internal/localhttp"
)

type labOptions struct {
	mode         string
	host         string
	port         int
	assets       string
	apiOnly      bool
	cloud        bool
	cloudBaseURL string
	stateDB      string
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
	cmd.Flags().BoolVar(&opts.cloud, "cloud", opts.cloud, "enable local Verda worker deployment and remote experiment execution")
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
	var executor learninglab.Executor = localExecutor
	if opts.cloud {
		stateDB := opts.stateDB
		if stateDB != "" && !filepath.IsAbs(stateDB) {
			stateDB = filepath.Join(a.repoRoot, stateDB)
		}
		var err error
		cloud, err = learninglab.NewCloudManager(learninglab.CloudManagerOptions{
			RepoRoot:     a.repoRoot,
			BaseURL:      opts.cloudBaseURL,
			Mode:         opts.mode,
			DatabasePath: stateDB,
		})
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

func isLoopbackLabHost(host string) bool {
	return localhttp.IsLoopbackHost(host)
}
