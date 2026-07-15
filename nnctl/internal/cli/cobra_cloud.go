package cli

import (
	"context"
	"fmt"

	"github.com/spf13/cobra"

	"nnctl/internal/cloud/verda"
)

func (a *app) newCloudCommand(withRepo repoRunner) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "cloud <command>",
		Short: "Deploy cloud benchmark workers",
		Long:  "Deploys cloud benchmark workers for nnctl agents. Verda deployments default to spot CPU-only or single-GPU instances.",
		RunE: func(cmd *cobra.Command, args []string) error {
			return cmd.Help()
		},
	}
	cmd.AddCommand(
		a.newCloudBenchmarkDeployCommand(withRepo),
		a.newCloudDeployCommand(),
		a.newCloudVolumeCommand(),
		a.newCloudDestroyCommand(),
		a.newCloudPackerTemplateCommand(),
		a.newCloudListCommand(),
		a.newCloudPricingCommand(),
		a.newCloudSSHKeysCommand(),
	)
	return cmd
}

func (a *app) newCloudBenchmarkDeployCommand(withRepo repoRunner) *cobra.Command {
	opts := defaultCloudBenchmarkDeployOptions()
	cmd := &cobra.Command{
		Use:     "benchmark-deploy",
		Aliases: []string{"benchmark"},
		Short:   "Provision a worker and capture a cloud benchmark",
		Long: `Runs the complete Verda benchmark workflow with one command. It writes the
embedded Packer template, reuses a matching golden OS volume or builds one when
needed, clones the volume, deploys a spot or on-demand worker, waits for SSH,
deploys a clean git snapshot, runs the benchmark, and saves a local CSV.

The worker and cloned OS volume are destroyed after the CSV is captured, even
when a later workflow step fails. The golden source volume is always retained.
Use --keep-instance to retain the worker and clone. Packer is only required when
no reusable source OS volume is available.`,
		Example: `  nnctl cloud benchmark-deploy --instance-type 1A100.22V --ssh-key-id ssh_key_id
  nnctl cloud benchmark-deploy --instance-type 1H200.141S.44V --market on-demand --location-code FIN-02 --ssh-key-id ssh_key_id
  nnctl cloud benchmark-deploy --instance-type 1A100.22V --filter gpu_peak --output a100-peak.csv --update-docs
  nnctl cloud benchmark --instance-type 1A100.22V --source-os-volume-id volume_id --skip-smoke`,
		Args: cobra.NoArgs,
		RunE: withRepo(func(ctx context.Context, cmd *cobra.Command, args []string) error {
			return a.runCloudBenchmarkDeploy(ctx, opts)
		}),
	}
	cmd.Flags().StringVar(&opts.InstanceType, "instance-type", opts.InstanceType, "Verda benchmark worker instance type")
	cmd.Flags().StringVar(&opts.SourceOSVolumeID, "source-os-volume-id", opts.SourceOSVolumeID, "existing golden source OS volume ID; skips name lookup and Packer")
	cmd.Flags().StringVar(&opts.SourceOSVolumeName, "source-os-volume-name", opts.SourceOSVolumeName, "golden source OS volume name; defaults to "+defaultCloudBenchmarkSourceName)
	cmd.Flags().StringVar(&opts.SourceOSVolumeName, "source-volume-name", opts.SourceOSVolumeName, "alias for --source-os-volume-name")
	cmd.Flags().StringVar(&opts.Market, "market", opts.Market, "worker market: spot or on-demand")
	cmd.Flags().StringVar(&opts.LocationCode, "location-code", opts.LocationCode, "Verda location; defaults to the cheapest available location with a reusable source volume")
	cmd.Flags().StringVar(&opts.Image, "image", opts.Image, "base Verda image used only when Packer must build the golden volume")
	cmd.Flags().StringVar(&opts.Hostname, "hostname", opts.Hostname, "benchmark worker hostname")
	cmd.Flags().StringVar(&opts.Description, "description", opts.Description, "benchmark worker description")
	cmd.Flags().StringArrayVar(&opts.SSHKeyIDs, "ssh-key-id", opts.SSHKeyIDs, "Verda SSH key ID to attach and bake into a new golden volume (repeatable)")
	cmd.Flags().StringVar(&opts.BaseURL, "base-url", opts.BaseURL, "Verda API base URL")
	cmd.Flags().StringVar(&opts.packerDir, "packer-dir", opts.packerDir, "directory for the generated Packer template")
	cmd.Flags().StringVar(&opts.packer, "packer", opts.packer, "Packer executable")
	cmd.Flags().StringVar(&opts.packerInstanceType, "packer-instance-type", opts.packerInstanceType, "instance type used to build a missing golden volume")
	cmd.Flags().StringVar(&opts.authorizedKeysFile, "authorized-keys-file", opts.authorizedKeysFile, "authorized_keys file to bake when Packer builds a golden volume")
	cmd.Flags().BoolVar(&opts.refreshPackerTemplate, "refresh-packer-template", opts.refreshPackerTemplate, "overwrite existing Packer template files with the embedded versions")
	cmd.Flags().StringVar(&opts.backend, "gpu", opts.backend, "remote benchmark GPU backend: cuda, rocm, or none")
	cmd.Flags().StringVar(&opts.filter, "filter", opts.filter, "benchmark suite filter")
	cmd.Flags().BoolVar(&opts.quick, "quick", opts.quick, "capture only the smoke-sized benchmark subset")
	cmd.Flags().BoolVar(&opts.skipSmoke, "skip-smoke", opts.skipSmoke, "skip the quick preflight before a full benchmark")
	cmd.Flags().StringVar(&opts.ref, "ref", opts.ref, "git ref or tree-ish to deploy")
	cmd.Flags().BoolVar(&opts.ignoreDirty, "ignore-dirty", opts.ignoreDirty, "deploy --ref even when the working tree has uncommitted changes")
	cmd.Flags().StringVar(&opts.remoteDir, "remote-dir", opts.remoteDir, "remote snapshot directory; defaults to a unique path")
	cmd.Flags().StringVar(&opts.sshUser, "ssh-user", opts.sshUser, "SSH user for the benchmark worker")
	cmd.Flags().StringVar(&opts.ssh, "ssh", opts.ssh, "SSH executable")
	cmd.Flags().StringVar(&opts.rsync, "rsync", opts.rsync, "rsync executable")
	cmd.Flags().StringVar(&opts.git, "git", opts.git, "Git executable")
	cmd.Flags().StringVar(&opts.tar, "tar", opts.tar, "tar executable")
	cmd.Flags().StringVar(&opts.output, "output", opts.output, "local CSV path; defaults to a metadata-based filename in the repository root")
	cmd.Flags().BoolVar(&opts.updateDocs, "update-docs", opts.updateDocs, "add or update a generated benchmark result in the benchmark docs")
	cmd.Flags().StringVar(&opts.docsPath, "docs", opts.docsPath, "benchmark Markdown file used by --update-docs")
	cmd.Flags().BoolVar(&opts.keepInstance, "keep-instance", opts.keepInstance, "keep the worker and cloned OS volume after the run")
	cmd.Flags().DurationVar(&opts.timeout, "timeout", opts.timeout, "timeout for image, instance, and SSH readiness")
	cmd.Flags().DurationVar(&opts.pollInterval, "poll-interval", opts.pollInterval, "interval between readiness checks")
	cmd.Flags().SortFlags = false
	_ = cmd.MarkFlagRequired("instance-type")
	return cmd
}

func (a *app) newCloudVolumeCommand() *cobra.Command {
	opts := cloudVolumesOptions{}
	cmd := &cobra.Command{
		Use:     "volume [VOLUME_ID...]",
		Aliases: []string{"volumes"},
		Short:   "List or purge Verda volumes",
		Long:    "Lists active Verda volumes by default. Use --include-deleted to include deleted volumes that are still in trash. Use --purge with volume IDs to permanently delete volumes or images, or --purge-all to permanently delete every deleted volume.",
		Example: `  nnctl cloud volume
  nnctl cloud volume --include-deleted
  nnctl cloud volume --json
  nnctl cloud volume volume_id --purge --dry-run
  nnctl cloud volume volume_id --purge --json
  nnctl cloud volume --purge-all`,
		Args: func(cmd *cobra.Command, args []string) error {
			if opts.purge && opts.AllDeleted {
				return fmt.Errorf("--purge and --purge-all cannot be combined")
			}
			if opts.AllDeleted {
				if len(args) > 0 {
					return fmt.Errorf("volume IDs cannot be combined with --purge-all")
				}
				return nil
			}
			if opts.purge {
				return cobra.MinimumNArgs(1)(cmd, args)
			}
			if opts.DryRun {
				return fmt.Errorf("--dry-run requires --purge or --purge-all")
			}
			if len(args) > 0 {
				return fmt.Errorf("volume IDs require --purge")
			}
			return nil
		},
		RunE: func(cmd *cobra.Command, args []string) error {
			opts.VolumeIDs = args
			return a.runCloudVolumes(cmd.Context(), opts)
		},
	}
	cmd.Flags().StringVar(&opts.BaseURL, "base-url", opts.BaseURL, "Verda API base URL")
	cmd.Flags().BoolVar(&opts.includeDeleted, "include-deleted", opts.includeDeleted, "include deleted volumes that have not been permanently deleted")
	cmd.Flags().BoolVar(&opts.purge, "purge", opts.purge, "permanently delete volume IDs instead of listing volumes")
	cmd.Flags().BoolVar(&opts.AllDeleted, "purge-all", opts.AllDeleted, "permanently delete all deleted volumes instead of listing volumes")
	cmd.Flags().BoolVar(&opts.DryRun, "dry-run", opts.DryRun, "print the planned purge request without reading keychain credentials")
	cmd.Flags().BoolVar(&opts.jsonOutput, "json", opts.jsonOutput, "print machine-readable JSON")
	cmd.Flags().SortFlags = false
	return cmd
}

func (a *app) newCloudDeployCommand() *cobra.Command {
	opts := cloudDeployOptions{
		DeployOptions: verda.DefaultDeployOptions(""),
	}
	cmd := &cobra.Command{
		Use:   "deploy",
		Short: "Deploy a Verda benchmark worker",
		Long: `Deploys one Verda instance for nnctl benchmark work.
The deployment policy is intentionally narrow: CPU-only and single-GPU instance
types accepted, with spot as the default market. When --source-os-volume-id is set, nnctl
locks placement to that source volume location and clones it as the instance
image. When --source-os-volume-name is set, nnctl chooses a matching source OS
volume in an available market location. Without a source volume, nnctl boots
--image and applies the embedded userdata script from
nnctl/internal/cloud/verda/packer/bootstrap.sh.`,
		Example: `  nnctl cloud deploy --instance-type 1V100.6V --source-os-volume-name golden-ubuntu --ssh-key-id ssh_key_id
  nnctl cloud deploy --instance-type 1V100.6V --source-os-volume-id volume_id --ssh-key-id ssh_key_id
  nnctl cloud deploy --instance-type 1H200.141S.44V --market on-demand --location-code FIN-02 --ssh-key-id ssh_key_id
  nnctl cloud deploy --instance-type 1V100.6V --ssh-key-id ssh_key_id
  nnctl cloud deploy --instance-type 1V100.6V --source-os-volume-id volume_id --dry-run --json
  nnctl cloud deploy --instance-type 1V100.6V --source-os-volume-id volume_id --location-code FIN-03
  nnctl cloud deploy --instance-type 1V100.6V --source-os-volume-id source_volume_id --cloned-os-volume-id cloned_volume_id
  nnctl cloud deploy --instance-type 1V100.6V --source-os-volume-id volume_id --user-data-file ./custom-bootstrap.sh`,
		Args: cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			return a.runCloudDeploy(cmd.Context(), opts)
		},
	}
	cmd.Flags().StringVar(&opts.InstanceType, "instance-type", opts.InstanceType, "Verda instance type, for example 1L40S.20V")
	cmd.Flags().StringVar(&opts.SourceOSVolumeID, "source-os-volume-id", opts.SourceOSVolumeID, "Packer-built Verda source OS volume ID used for location lock and cloning")
	cmd.Flags().StringVar(&opts.SourceOSVolumeName, "source-os-volume-name", opts.SourceOSVolumeName, "Packer-built Verda source OS volume name; picks a matching volume in an available market location")
	cmd.Flags().StringVar(&opts.SourceOSVolumeName, "source-volume-name", opts.SourceOSVolumeName, "alias for --source-os-volume-name")
	cmd.Flags().StringVar(&opts.ClonedOSVolumeID, "cloned-os-volume-id", opts.ClonedOSVolumeID, "existing cloned OS volume ID to use as the instance image instead of creating a new clone")
	cmd.Flags().StringVar(&opts.Market, "market", opts.Market, "deployment market: spot or on-demand")
	cmd.Flags().StringVar(&opts.Image, "image", opts.Image, "Verda image to boot when source OS volume flags are omitted")
	cmd.Flags().StringVar(&opts.Hostname, "hostname", opts.Hostname, "instance hostname")
	cmd.Flags().StringVar(&opts.Description, "description", opts.Description, "instance description")
	cmd.Flags().StringArrayVar(&opts.SSHKeyIDs, "ssh-key-id", opts.SSHKeyIDs, "Verda SSH key ID to attach (repeatable)")
	cmd.Flags().StringVar(&opts.LocationCode, "location-code", opts.LocationCode, "Verda location code; defaults to source OS volume location or market placement")
	cmd.Flags().StringVar(&opts.StartupScriptName, "startup-script-name", opts.StartupScriptName, "Verda startup script name")
	cmd.Flags().StringVar(&opts.userDataFile, "user-data-file", opts.userDataFile, "read userdata from a file; defaults to the embedded script only without source OS volume flags")
	cmd.Flags().StringVar(&opts.BaseURL, "base-url", opts.BaseURL, "Verda API base URL")
	cmd.Flags().BoolVar(&opts.SkipAvailabilityCheck, "skip-availability-check", opts.SkipAvailabilityCheck, "skip market availability check before creating the instance")
	cmd.Flags().BoolVar(&opts.KeepClonedOSVolumeOnFailure, "keep-cloned-os-volume-on-failure", opts.KeepClonedOSVolumeOnFailure, "keep a newly cloned OS volume when instance creation fails")
	cmd.Flags().BoolVar(&opts.DryRun, "dry-run", opts.DryRun, "print the planned deployment without reading keychain credentials")
	cmd.Flags().BoolVar(&opts.jsonOutput, "json", opts.jsonOutput, "print machine-readable JSON")
	cmd.Flags().SortFlags = false
	_ = cmd.MarkFlagRequired("instance-type")
	return cmd
}

func (a *app) newCloudDestroyCommand() *cobra.Command {
	opts := cloudDestroyOptions{
		DestroyOptions: verda.DefaultDestroyOptions(nil),
	}
	cmd := &cobra.Command{
		Use:     "destroy INSTANCE_ID [INSTANCE_ID...]",
		Aliases: []string{"delete", "rm", "terminate"},
		Short:   "Destroy Verda instances",
		Long:    "Destroys one or more Verda instances by ID. Instance IDs can be copied from `nnctl cloud list`.",
		Example: `  nnctl cloud destroy instance_id
  nnctl cloud destroy instance_id --dry-run
  nnctl cloud destroy instance_id --permanent=false
  nnctl cloud destroy instance_id --volume-id cloned_os_volume_id --source-os-volume-id source_os_volume_id --json`,
		Args: cobra.MinimumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			opts.InstanceIDs = args
			return a.runCloudDestroy(cmd.Context(), opts)
		},
	}
	cmd.Flags().StringArrayVar(&opts.VolumeIDs, "volume-id", opts.VolumeIDs, "Verda volume ID to delete with the instance (repeatable)")
	cmd.Flags().StringVar(&opts.SourceOSVolumeID, "source-os-volume-id", opts.SourceOSVolumeID, "protect this golden Packer source OS volume ID from cleanup")
	cmd.Flags().BoolVar(&opts.DeletePermanently, "permanent", opts.DeletePermanently, "delete permanently instead of using Verda's non-permanent delete")
	cmd.Flags().StringVar(&opts.BaseURL, "base-url", opts.BaseURL, "Verda API base URL")
	cmd.Flags().BoolVar(&opts.DryRun, "dry-run", opts.DryRun, "print the planned destroy request without reading keychain credentials")
	cmd.Flags().BoolVar(&opts.jsonOutput, "json", opts.jsonOutput, "print machine-readable JSON")
	cmd.Flags().SortFlags = false
	return cmd
}

func (a *app) newCloudPackerTemplateCommand() *cobra.Command {
	opts := cloudPackerTemplateOptions{}
	cmd := &cobra.Command{
		Use:   "packer-template DIR",
		Short: "Write Verda Packer template files",
		Long:  "Writes the embedded Verda Packer template and bootstrap script used to build the golden OS volume for nnctl cloud deploy.",
		Example: `  nnctl cloud packer-template ./packer/verda
  cd ./packer/verda && packer init . && packer build .`,
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			return a.runCloudPackerTemplate(args[0], opts)
		},
	}
	cmd.Flags().BoolVar(&opts.force, "force", opts.force, "overwrite existing template files")
	cmd.Flags().SortFlags = false
	return cmd
}

func (a *app) newCloudSSHKeysCommand() *cobra.Command {
	opts := cloudSSHKeysOptions{}
	cmd := &cobra.Command{
		Use:     "ssh-keys",
		Aliases: []string{"sshkeys", "ssh-key"},
		Short:   "List Verda SSH key IDs",
		Long:    "Lists Verda SSH keys and their UUIDs. Pass one of these IDs to cloud deploy with --ssh-key-id.",
		Example: `  nnctl cloud ssh-keys
  nnctl cloud ssh-keys --json
  nnctl cloud deploy --instance-type 1L40S.20V --ssh-key-id 00000000-0000-0000-0000-000000000000`,
		Args: cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			return a.runCloudSSHKeys(cmd.Context(), opts)
		},
	}
	cmd.Flags().StringVar(&opts.baseURL, "base-url", opts.baseURL, "Verda API base URL")
	cmd.Flags().BoolVar(&opts.jsonOutput, "json", opts.jsonOutput, "print machine-readable JSON")
	cmd.Flags().SortFlags = false
	return cmd
}

func (a *app) newCloudListCommand() *cobra.Command {
	opts := cloudListOptions{}
	cmd := &cobra.Command{
		Use:   "list",
		Short: "List active Verda instances",
		Long:  "Lists active Verda instances by default. Use --all to include inactive instances or --status to query one Verda status.",
		Example: `  nnctl cloud list
  nnctl cloud list --all
  nnctl cloud list --status running
  nnctl cloud list --json`,
		Args: cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			return a.runCloudList(cmd.Context(), opts)
		},
	}
	cmd.Flags().StringVar(&opts.status, "status", opts.status, "Verda instance status filter, for example running")
	cmd.Flags().BoolVar(&opts.all, "all", opts.all, "include inactive instances")
	cmd.Flags().StringVar(&opts.baseURL, "base-url", opts.baseURL, "Verda API base URL")
	cmd.Flags().BoolVar(&opts.jsonOutput, "json", opts.jsonOutput, "print machine-readable JSON")
	cmd.Flags().SortFlags = false
	return cmd
}

func (a *app) newCloudPricingCommand() *cobra.Command {
	opts := cloudPricingOptions{
		sortBy: "price",
	}
	opts.filters.AvailableOnly = true
	opts.filters.Market = verda.PricingMarketSpot
	cmd := &cobra.Command{
		Use:   "pricing",
		Short: "List Verda instance prices",
		Long:  "Lists Verda instance prices with filters for market, location, instance type, GPU model, manufacturer, and GPU count.",
		Example: `  nnctl cloud pricing
  nnctl cloud pricing --market on-demand --gpu-count 0
  nnctl cloud pricing --market all --all-gpu-counts
  nnctl cloud pricing --zone FIN-02 --model L40
  nnctl cloud pricing --instance-type 1L40S.20V --json
  nnctl cloud pricing --available-only=false --sort location`,
		Args: cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			opts.filters.LocationCodes = append(opts.filters.LocationCodes, opts.zones...)
			opts.filters.LocationCodes = sortUniqueStrings(opts.filters.LocationCodes)
			opts.filters.InstanceTypes = sortUniqueStrings(opts.filters.InstanceTypes)
			gpuCounts, err := resolveCloudPricingGPUCounts(
				opts,
				cmd.Flags().Changed("single-gpu"),
				cmd.Flags().Changed("gpu-count"),
			)
			if err != nil {
				return err
			}
			opts.filters.GPUCounts = gpuCounts
			return a.runCloudPricing(cmd.Context(), opts)
		},
	}
	cmd.Flags().StringArrayVar(&opts.filters.LocationCodes, "location-code", opts.filters.LocationCodes, "Verda location code filter (repeatable)")
	cmd.Flags().StringArrayVar(&opts.zones, "zone", opts.zones, "alias for --location-code")
	cmd.Flags().StringArrayVar(&opts.filters.InstanceTypes, "instance-type", opts.filters.InstanceTypes, "Verda instance type filter (repeatable)")
	cmd.Flags().StringVar(&opts.filters.Market, "market", opts.filters.Market, "pricing market: spot, on-demand, or all")
	cmd.Flags().StringVar(&opts.filters.Model, "model", opts.filters.Model, "GPU model/display substring filter")
	cmd.Flags().StringVar(&opts.filters.Manufacturer, "manufacturer", opts.filters.Manufacturer, "GPU manufacturer substring filter")
	cmd.Flags().BoolVar(&opts.singleGPU, "single-gpu", opts.singleGPU, "only show single-GPU instance types")
	cmd.Flags().IntSliceVar(&opts.gpuCounts, "gpu-count", opts.gpuCounts, "GPU count filter; use 0 for CPU-only types (repeatable or comma-separated)")
	cmd.Flags().BoolVar(&opts.allGPU, "all-gpu-counts", opts.allGPU, "show CPU, single-GPU, and multi-GPU instance types")
	cmd.Flags().BoolVar(&opts.filters.AvailableOnly, "available-only", opts.filters.AvailableOnly, "only show currently available instance types")
	cmd.Flags().StringVar(&opts.filters.Currency, "currency", opts.filters.Currency, "Verda pricing currency")
	cmd.Flags().StringVar(&opts.sortBy, "sort", opts.sortBy, "sort by price, market, location, or instance-type")
	cmd.Flags().StringVar(&opts.baseURL, "base-url", opts.baseURL, "Verda API base URL")
	cmd.Flags().BoolVar(&opts.jsonOutput, "json", opts.jsonOutput, "print machine-readable JSON")
	cmd.Flags().SortFlags = false
	return cmd
}
