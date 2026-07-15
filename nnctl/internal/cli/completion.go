package cli

import (
	"fmt"

	"github.com/spf13/cobra"
)

func (a *app) newCompletionCommand() *cobra.Command {
	noDesc := false
	cmd := &cobra.Command{
		Use:   "completion",
		Short: "Generate shell completion scripts",
		Long: `Generate shell completion scripts for bash, zsh, or fish.

Print the script to stdout so it can be sourced directly or written into your shell's completion directory.`,
		Args:              cobra.NoArgs,
		ValidArgsFunction: cobra.NoFileCompletions,
		RunE: func(cmd *cobra.Command, args []string) error {
			return cmd.Help()
		},
	}
	cmd.PersistentFlags().BoolVar(&noDesc, "no-descriptions", false, "disable completion descriptions")
	cmd.PersistentFlags().SortFlags = false

	cmd.AddCommand(
		a.newBashCompletionCommand(&noDesc),
		a.newZshCompletionCommand(&noDesc),
		a.newFishCompletionCommand(&noDesc),
	)
	return cmd
}

func (a *app) newBashCompletionCommand(noDesc *bool) *cobra.Command {
	return &cobra.Command{
		Use:                   "bash",
		Short:                 "Generate the bash completion script",
		Long:                  completionLong("bash", "source <(nnctl completion bash)", "nnctl completion bash > /etc/bash_completion.d/nnctl"),
		Args:                  cobra.NoArgs,
		DisableFlagsInUseLine: true,
		ValidArgsFunction:     cobra.NoFileCompletions,
		RunE: func(cmd *cobra.Command, args []string) error {
			return cmd.Root().GenBashCompletionV2(a.stdout(), !*noDesc)
		},
	}
}

func (a *app) newZshCompletionCommand(noDesc *bool) *cobra.Command {
	return &cobra.Command{
		Use:                   "zsh",
		Short:                 "Generate the zsh completion script",
		Long:                  completionLong("zsh", "source <(nnctl completion zsh)", `nnctl completion zsh > "${fpath[1]}/_nnctl"`),
		Args:                  cobra.NoArgs,
		DisableFlagsInUseLine: true,
		ValidArgsFunction:     cobra.NoFileCompletions,
		RunE: func(cmd *cobra.Command, args []string) error {
			if *noDesc {
				return cmd.Root().GenZshCompletionNoDesc(a.stdout())
			}
			return cmd.Root().GenZshCompletion(a.stdout())
		},
	}
}

func (a *app) newFishCompletionCommand(noDesc *bool) *cobra.Command {
	return &cobra.Command{
		Use:                   "fish",
		Short:                 "Generate the fish completion script",
		Long:                  completionLong("fish", "nnctl completion fish | source", "nnctl completion fish > ~/.config/fish/completions/nnctl.fish"),
		Args:                  cobra.NoArgs,
		DisableFlagsInUseLine: true,
		ValidArgsFunction:     cobra.NoFileCompletions,
		RunE: func(cmd *cobra.Command, args []string) error {
			return cmd.Root().GenFishCompletion(a.stdout(), !*noDesc)
		},
	}
}

func completionLong(shell, currentSession, everySession string) string {
	return fmt.Sprintf(`Generate the autocompletion script for %s.

To load completions in your current shell session:

  %s

To load completions for every new session, write the generated script once:

  %s`, shell, currentSession, everySession)
}
