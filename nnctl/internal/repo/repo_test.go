package repo

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestResolveRootUsesExplicitRepository(t *testing.T) {
	root := newRepository(t)
	parent := filepath.Dir(root)
	relative, err := filepath.Rel(parent, root)
	if err != nil {
		t.Fatalf("make repository path relative: %v", err)
	}
	t.Chdir(parent)

	got, err := ResolveRoot(relative)
	if err != nil {
		t.Fatalf("ResolveRoot() error = %v", err)
	}
	if got != root {
		t.Fatalf("ResolveRoot() = %q, want %q", got, root)
	}
}

func TestResolveRootWalksParentDirectories(t *testing.T) {
	root := newRepository(t)
	nested := filepath.Join(root, "one", "two")
	if err := os.MkdirAll(nested, 0o755); err != nil {
		t.Fatalf("create nested directory: %v", err)
	}
	t.Chdir(nested)

	got, err := ResolveRoot("")
	if err != nil {
		t.Fatalf("ResolveRoot() error = %v", err)
	}
	if got != root {
		t.Fatalf("ResolveRoot() = %q, want %q", got, root)
	}
}

func TestResolveRootRejectsDirectoryWithoutManifest(t *testing.T) {
	dir := t.TempDir()

	_, err := ResolveRoot(dir)
	if err == nil || !strings.Contains(err.Error(), "does not look like") {
		t.Fatalf("ResolveRoot() error = %v", err)
	}
}

func TestIsRootRequiresBothBuildFiles(t *testing.T) {
	dir := t.TempDir()
	if IsRoot(dir) {
		t.Fatal("IsRoot() = true without build files")
	}

	writeFile(t, filepath.Join(dir, "build.zig"))
	if IsRoot(dir) {
		t.Fatal("IsRoot() = true without build.zig.zon")
	}

	writeFile(t, filepath.Join(dir, "build.zig.zon"))
	if !IsRoot(dir) {
		t.Fatal("IsRoot() = false with both build files")
	}
}

func TestFileExistsRejectsDirectories(t *testing.T) {
	dir := t.TempDir()
	if FileExists(dir) {
		t.Fatal("FileExists() = true for a directory")
	}

	path := filepath.Join(dir, "file")
	writeFile(t, path)
	if !FileExists(path) {
		t.Fatal("FileExists() = false for a regular file")
	}
}

func newRepository(t *testing.T) string {
	t.Helper()

	root := t.TempDir()
	writeFile(t, filepath.Join(root, "build.zig"))
	writeFile(t, filepath.Join(root, "build.zig.zon"))
	return root
}

func writeFile(t *testing.T, path string) {
	t.Helper()

	if err := os.WriteFile(path, nil, 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}
