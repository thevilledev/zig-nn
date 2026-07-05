package cli

import (
	"compress/gzip"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"

	"nnctl/internal/repo"
)

type mnistFile struct {
	Name string
	URL  string
}

var mnistFiles = []mnistFile{
	{Name: "train-images-idx3-ubyte", URL: "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz"},
	{Name: "train-labels-idx1-ubyte", URL: "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz"},
	{Name: "t10k-images-idx3-ubyte", URL: "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz"},
	{Name: "t10k-labels-idx1-ubyte", URL: "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz"},
}

func (a *App) runDataMNIST(ctx context.Context, dir string, force bool) error {
	if !filepath.IsAbs(dir) {
		dir = filepath.Join(a.repoRoot, dir)
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("create data directory: %w", err)
	}
	for _, file := range mnistFiles {
		if err := a.downloadGzip(ctx, file.URL, filepath.Join(dir, file.Name), force); err != nil {
			return err
		}
	}
	return nil
}

func (a *App) runDataTinyGPT(ctx context.Context) error {
	script := filepath.Join(a.repoRoot, "examples", "tiny_gpt", "scripts", "prepare_data.sh")
	if !repo.FileExists(script) {
		return fmt.Errorf("missing Tiny GPT data script at %s", script)
	}
	return a.run(ctx, a.repoRoot, script)
}

func (a *App) downloadGzip(ctx context.Context, url, dest string, force bool) error {
	if !force {
		if _, err := os.Stat(dest); err == nil {
			fmt.Fprintf(a.stdout(), "exists %s\n", filepath.Base(dest))
			return nil
		} else if !errors.Is(err, os.ErrNotExist) {
			return fmt.Errorf("stat %s: %w", dest, err)
		}
	}

	fmt.Fprintf(a.stdout(), "download %s\n", filepath.Base(dest))
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("download %s: %w", url, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download %s: %s", url, resp.Status)
	}

	gz, err := gzip.NewReader(resp.Body)
	if err != nil {
		return fmt.Errorf("open gzip stream for %s: %w", url, err)
	}
	defer gz.Close()

	tmp := dest + ".tmp"
	out, err := os.OpenFile(tmp, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		return fmt.Errorf("create %s: %w", tmp, err)
	}
	if _, err := io.Copy(out, gz); err != nil {
		out.Close()
		os.Remove(tmp)
		return fmt.Errorf("write %s: %w", dest, err)
	}
	if err := out.Close(); err != nil {
		os.Remove(tmp)
		return fmt.Errorf("close %s: %w", tmp, err)
	}
	if err := os.Rename(tmp, dest); err != nil {
		os.Remove(tmp)
		return fmt.Errorf("rename %s: %w", dest, err)
	}
	return nil
}
