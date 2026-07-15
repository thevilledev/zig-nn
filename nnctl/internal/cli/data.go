package cli

import (
	"archive/zip"
	"compress/gzip"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"strings"

	"nnctl/internal/repo"
)

const miniSpeechCommandsURL = "https://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip"

var speechCommandLabels = []string{"down", "go", "left", "no", "right", "stop", "up", "yes"}

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

func (a *app) runDataMNIST(ctx context.Context, dir string, force bool) error {
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

func (a *app) runDataTinyGPT(ctx context.Context) error {
	script := filepath.Join(a.repoRoot, "experiments", "tiny_gpt", "scripts", "prepare_data.sh")
	if !repo.FileExists(script) {
		return fmt.Errorf("missing Tiny GPT data script at %s", script)
	}
	return a.run(ctx, a.repoRoot, script)
}

func (a *app) runDataSpeechCommands(ctx context.Context, dir string, force bool) error {
	if !filepath.IsAbs(dir) {
		dir = filepath.Join(a.repoRoot, dir)
	}
	return a.prepareSpeechCommands(ctx, miniSpeechCommandsURL, dir, force)
}

func (a *app) prepareSpeechCommands(ctx context.Context, url, dir string, force bool) error {
	if err := validateSpeechCommandsDir(dir); err == nil && !force {
		if _, err := fmt.Fprintf(a.stdout(), "exists %s\n", filepath.Base(dir)); err != nil {
			return fmt.Errorf("report existing speech commands data: %w", err)
		}
		return nil
	}
	if _, err := os.Stat(dir); err == nil && !force {
		return fmt.Errorf("speech commands directory %s exists but is incomplete; use --force to replace it", dir)
	} else if err != nil && !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("stat %s: %w", dir, err)
	}

	parent := filepath.Dir(dir)
	if err := os.MkdirAll(parent, 0o755); err != nil {
		return fmt.Errorf("create data parent: %w", err)
	}
	if _, err := fmt.Fprintln(a.stdout(), "download mini_speech_commands.zip"); err != nil {
		return fmt.Errorf("report speech commands download: %w", err)
	}
	archivePath, err := a.downloadSpeechCommandsArchive(ctx, url, parent)
	if err != nil {
		return err
	}
	defer func() { _ = os.Remove(archivePath) }()

	extracted, err := os.MkdirTemp(parent, ".mini-speech-commands-extract-*")
	if err != nil {
		return fmt.Errorf("create temporary extraction directory: %w", err)
	}
	defer func() { _ = os.RemoveAll(extracted) }()
	if err := extractSpeechCommandsZip(archivePath, extracted); err != nil {
		return err
	}
	if err := validateSpeechCommandsDir(extracted); err != nil {
		return fmt.Errorf("validate extracted speech commands: %w", err)
	}
	if err := replaceDirectory(extracted, dir, force); err != nil {
		return err
	}
	if _, err := fmt.Fprintf(a.stdout(), "prepared %s\n", dir); err != nil {
		return fmt.Errorf("report prepared speech commands: %w", err)
	}
	return nil
}

func (a *app) downloadSpeechCommandsArchive(ctx context.Context, url, parent string) (string, error) {
	archive, err := os.CreateTemp(parent, ".mini-speech-commands-*.zip")
	if err != nil {
		return "", fmt.Errorf("create temporary archive: %w", err)
	}
	archivePath := archive.Name()
	archiveClosed := false
	keepArchive := false
	defer func() {
		if !archiveClosed {
			_ = archive.Close()
		}
		if !keepArchive {
			_ = os.Remove(archivePath)
		}
	}()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return "", fmt.Errorf("create speech commands request: %w", err)
	}
	resp, err := a.http().Do(req)
	if err != nil {
		return "", fmt.Errorf("download %s: %w", url, err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("download %s: %s", url, resp.Status)
	}
	if _, err := io.Copy(archive, resp.Body); err != nil {
		return "", fmt.Errorf("write temporary speech commands archive: %w", err)
	}
	if err := archive.Close(); err != nil {
		return "", fmt.Errorf("close temporary speech commands archive: %w", err)
	}
	archiveClosed = true
	keepArchive = true
	return archivePath, nil
}

func extractSpeechCommandsZip(archivePath, destination string) error {
	reader, err := zip.OpenReader(archivePath)
	if err != nil {
		return fmt.Errorf("open speech commands archive: %w", err)
	}
	defer func() { _ = reader.Close() }()

	for _, file := range reader.File {
		if err := extractSpeechCommandsEntry(file, destination); err != nil {
			return err
		}
	}
	return nil
}

func extractSpeechCommandsEntry(file *zip.File, destination string) error {
	target, skip, err := speechCommandsEntryTarget(file.Name, destination)
	if err != nil || skip {
		return err
	}
	if file.FileInfo().IsDir() {
		if err := os.MkdirAll(target, 0o755); err != nil {
			return fmt.Errorf("create extracted directory: %w", err)
		}
		return nil
	}
	if !file.Mode().IsRegular() {
		return fmt.Errorf("unsupported zip entry %q", file.Name)
	}
	if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
		return fmt.Errorf("create extracted file parent: %w", err)
	}
	return copySpeechCommandsEntry(file, target)
}

func speechCommandsEntryTarget(name, destination string) (target string, skip bool, err error) {
	if strings.Contains(name, "\\") {
		return "", false, fmt.Errorf("unsafe zip entry %q", name)
	}
	clean := path.Clean(name)
	if clean == "__MACOSX" || strings.HasPrefix(clean, "__MACOSX/") || clean == "mini_speech_commands" {
		return "", true, nil
	}
	const root = "mini_speech_commands/"
	if path.IsAbs(clean) || strings.HasPrefix(clean, "../") || !strings.HasPrefix(clean, root) {
		return "", false, fmt.Errorf("unsafe zip entry %q", name)
	}
	relative := strings.TrimPrefix(clean, root)
	if relative == "" {
		return "", true, nil
	}
	target = filepath.Join(destination, filepath.FromSlash(relative))
	contained, relErr := filepath.Rel(destination, target)
	if relErr != nil || contained == ".." || strings.HasPrefix(contained, ".."+string(filepath.Separator)) {
		return "", false, fmt.Errorf("unsafe zip entry %q", name)
	}
	return target, false, nil
}

func copySpeechCommandsEntry(file *zip.File, target string) error {
	source, err := file.Open()
	if err != nil {
		return fmt.Errorf("open zip entry %q: %w", file.Name, err)
	}
	output, err := os.OpenFile(target, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		_ = source.Close()
		return fmt.Errorf("create extracted file %q: %w", target, err)
	}
	_, copyErr := io.Copy(output, source)
	closeErr := output.Close()
	sourceErr := source.Close()
	if copyErr != nil {
		return fmt.Errorf("extract %q: %w", file.Name, copyErr)
	}
	if closeErr != nil {
		return fmt.Errorf("close %q: %w", target, closeErr)
	}
	if sourceErr != nil {
		return fmt.Errorf("close zip entry %q: %w", file.Name, sourceErr)
	}
	return nil
}

func validateSpeechCommandsDir(dir string) error {
	for _, label := range speechCommandLabels {
		info, err := os.Stat(filepath.Join(dir, label))
		if err != nil {
			return fmt.Errorf("missing %s label directory: %w", label, err)
		}
		if !info.IsDir() {
			return fmt.Errorf("%s label path is not a directory", label)
		}
	}
	return nil
}

func replaceDirectory(source, destination string, force bool) error {
	if _, err := os.Stat(destination); errors.Is(err, os.ErrNotExist) {
		if err := os.Rename(source, destination); err != nil {
			return fmt.Errorf("install speech commands data: %w", err)
		}
		return nil
	} else if err != nil {
		return fmt.Errorf("stat speech commands destination: %w", err)
	}
	if !force {
		return fmt.Errorf("speech commands destination already exists")
	}

	backup, err := os.MkdirTemp(filepath.Dir(destination), ".mini-speech-commands-backup-*")
	if err != nil {
		return fmt.Errorf("reserve speech commands backup: %w", err)
	}
	if err := os.Remove(backup); err != nil {
		return fmt.Errorf("prepare speech commands backup: %w", err)
	}
	if err := os.Rename(destination, backup); err != nil {
		return fmt.Errorf("backup speech commands data: %w", err)
	}
	if err := os.Rename(source, destination); err != nil {
		installErr := fmt.Errorf("install replacement speech commands data: %w", err)
		if restoreErr := os.Rename(backup, destination); restoreErr != nil {
			return errors.Join(installErr, fmt.Errorf("restore speech commands backup: %w", restoreErr))
		}
		return installErr
	}
	if err := os.RemoveAll(backup); err != nil {
		return fmt.Errorf("remove speech commands backup: %w", err)
	}
	return nil
}

func (a *app) downloadGzip(ctx context.Context, url, dest string, force bool) error {
	if !force {
		if _, err := os.Stat(dest); err == nil {
			if _, err := fmt.Fprintf(a.stdout(), "exists %s\n", filepath.Base(dest)); err != nil {
				return fmt.Errorf("report existing data file: %w", err)
			}
			return nil
		} else if !errors.Is(err, os.ErrNotExist) {
			return fmt.Errorf("stat %s: %w", dest, err)
		}
	}

	if _, err := fmt.Fprintf(a.stdout(), "download %s\n", filepath.Base(dest)); err != nil {
		return fmt.Errorf("report data download: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return fmt.Errorf("create data request: %w", err)
	}
	resp, err := a.http().Do(req)
	if err != nil {
		return fmt.Errorf("download %s: %w", url, err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download %s: %s", url, resp.Status)
	}

	gz, err := gzip.NewReader(resp.Body)
	if err != nil {
		return fmt.Errorf("open gzip stream for %s: %w", url, err)
	}
	defer func() { _ = gz.Close() }()

	tmp := dest + ".tmp"
	out, err := os.OpenFile(tmp, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		return fmt.Errorf("create %s: %w", tmp, err)
	}
	outClosed := false
	installed := false
	defer func() {
		if !outClosed {
			_ = out.Close()
		}
		if !installed {
			_ = os.Remove(tmp)
		}
	}()
	if _, err := io.Copy(out, gz); err != nil {
		return fmt.Errorf("write %s: %w", dest, err)
	}
	if err := out.Close(); err != nil {
		return fmt.Errorf("close %s: %w", tmp, err)
	}
	outClosed = true
	if err := os.Rename(tmp, dest); err != nil {
		return fmt.Errorf("rename %s: %w", dest, err)
	}
	installed = true
	return nil
}
