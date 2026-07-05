package zig

import (
	"strconv"
	"strings"
)

type Options struct {
	Optimize string
	GPU      string
	Summary  string
}

func Args(step string, opts Options) []string {
	args := []string{"build"}
	if step != "" {
		args = append(args, step)
	}
	if opts.GPU != "" {
		args = append(args, "-Dgpu="+opts.GPU)
	}
	if opts.Optimize != "" {
		args = append(args, "-Doptimize="+opts.Optimize)
	}
	if opts.Summary != "" {
		args = append(args, "--summary", opts.Summary)
	}
	return args
}

func RunArgs(step string, opts Options, passthrough []string) []string {
	args := Args(step, opts)
	if len(passthrough) > 0 {
		args = append(args, "--")
		args = append(args, passthrough...)
	}
	return args
}

func CommandString(name string, args []string) string {
	parts := append([]string{name}, args...)
	for i, part := range parts {
		parts[i] = quoteShell(part)
	}
	return strings.Join(parts, " ")
}

func quoteShell(s string) string {
	if s == "" {
		return `""`
	}
	if strings.ContainsAny(s, " \t\n\"'`$\\") {
		return strconv.Quote(s)
	}
	return s
}
