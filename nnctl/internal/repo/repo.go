// Package repo locates and validates zig-nn repository roots.
package repo

import (
	"fmt"
	"os"
	"path/filepath"
)

func ResolveRoot(repo string) (string, error) {
	if repo != "" {
		abs, err := filepath.Abs(repo)
		if err != nil {
			return "", err
		}
		if !IsRoot(abs) {
			return "", fmt.Errorf("%s does not look like the zig-nn repo root", abs)
		}
		return abs, nil
	}
	cwd, err := os.Getwd()
	if err != nil {
		return "", err
	}
	for {
		if IsRoot(cwd) {
			return cwd, nil
		}
		parent := filepath.Dir(cwd)
		if parent == cwd {
			return "", fmt.Errorf("could not find repo root from %s", cwd)
		}
		cwd = parent
	}
}

func IsRoot(dir string) bool {
	return FileExists(filepath.Join(dir, "build.zig")) && FileExists(filepath.Join(dir, "build.zig.zon"))
}

func FileExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}
