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
)

type labOptions struct {
	mode    string
	host    string
	port    int
	assets  string
	apiOnly bool
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
	cmd.Flags().SortFlags = false
	return cmd
}

func (a *app) runLab(ctx context.Context, opts labOptions) error {
	if opts.port <= 0 || opts.port > 65535 {
		return fmt.Errorf("--port must be between 1 and 65535")
	}
	assets := opts.assets
	if assets == "" {
		assets = filepath.Join(a.repoRoot, "web", "dist")
	} else if !filepath.IsAbs(assets) {
		assets = filepath.Join(a.repoRoot, assets)
	}

	manager := learninglab.NewManager(learninglab.CommandExecutor{
		RepoRoot: a.repoRoot,
		Zig:      a.zig,
		Mode:     opts.mode,
	})
	defer manager.Close()
	handler, err := learninglab.NewServer(manager, assets, opts.apiOnly)
	if err != nil {
		return err
	}
	address := net.JoinHostPort(opts.host, strconv.Itoa(opts.port))
	server := &http.Server{
		Addr:              address,
		Handler:           handler.Handler(),
		ReadHeaderTimeout: 5 * time.Second,
	}

	if _, err := fmt.Fprintf(a.stdout(), "learning lab: http://%s\n", address); err != nil {
		return fmt.Errorf("write learning lab endpoint: %w", err)
	}
	go func() {
		<-ctx.Done()
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = server.Shutdown(shutdownCtx)
	}()
	if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		return fmt.Errorf("learning lab server: %w", err)
	}
	return nil
}
