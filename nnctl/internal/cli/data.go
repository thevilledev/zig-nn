package cli

import (
	"archive/zip"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path"
	"path/filepath"
	"strings"

	"nnctl/internal/repo"
)

const (
	maxSpeechCommandsEntries        = 20_000
	maxSpeechCommandsEntryBytes     = 8 * 1024 * 1024
	maxSpeechCommandsExtractedBytes = 300 * 1024 * 1024
	maxMNISTDownloadBytes           = 16 * 1024 * 1024
)

var miniSpeechCommandsArtifact = downloadArtifact{
	URL:              "https://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
	SHA256:           "49650f2341b26d886b46b3f4fb8fed59e30300b17550f1ee4a768b3106cf93a0",
	MaxDownloadBytes: 200 * 1024 * 1024,
}

var speechCommandLabels = []string{"down", "go", "left", "no", "right", "stop", "up", "yes"}

type mnistFile struct {
	Name     string
	Artifact gzipArtifact
}

var mnistFiles = []mnistFile{
	{Name: "train-images-idx3-ubyte", Artifact: newMNISTArtifact(
		"https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
		"440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
		"ba891046e6505d7aadcbbe25680a0738ad16aec93bde7f9b65e87a2fc25776db",
		48*1024*1024,
	)},
	{Name: "train-labels-idx1-ubyte", Artifact: newMNISTArtifact(
		"https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
		"3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
		"65a50cbbf4e906d70832878ad85ccda5333a97f0f4c3dd2ef09a8a9eef7101c5",
		1024*1024,
	)},
	{Name: "t10k-images-idx3-ubyte", Artifact: newMNISTArtifact(
		"https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
		"8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
		"0fa7898d509279e482958e8ce81c8e77db3f2f8254e26661ceb7762c4d494ce7",
		8*1024*1024,
	)},
	{Name: "t10k-labels-idx1-ubyte", Artifact: newMNISTArtifact(
		"https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
		"f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
		"ff7bcfd416de33731a308c3f266cc351222c34898ecbeaf847f06e48f7ec33f2",
		1024*1024,
	)},
}

type gzipArtifact struct {
	downloadArtifact
	ExtractedSHA256   string
	MaxExtractedBytes int64
}

func newMNISTArtifact(url, checksum, extractedChecksum string, maxExtractedBytes int64) gzipArtifact {
	return gzipArtifact{
		downloadArtifact:  downloadArtifact{URL: url, SHA256: checksum, MaxDownloadBytes: maxMNISTDownloadBytes},
		ExtractedSHA256:   extractedChecksum,
		MaxExtractedBytes: maxExtractedBytes,
	}
}

func (artifact gzipArtifact) validate() error {
	if err := artifact.downloadArtifact.validate(); err != nil {
		return err
	}
	if !validSHA256(artifact.ExtractedSHA256) {
		return fmt.Errorf("extracted artifact SHA-256 must contain 64 hexadecimal characters")
	}
	if artifact.MaxExtractedBytes <= 0 {
		return fmt.Errorf("artifact extraction limit must be positive")
	}
	return nil
}

func (a *app) runDataMNIST(ctx context.Context, dir string, force bool) error {
	if !filepath.IsAbs(dir) {
		dir = filepath.Join(a.repoRoot, dir)
	}
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return fmt.Errorf("create data directory: %w", err)
	}
	for _, file := range mnistFiles {
		if err := a.downloadGzip(ctx, file.Artifact, filepath.Join(dir, file.Name), force); err != nil {
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
	return a.prepareSpeechCommands(ctx, miniSpeechCommandsArtifact, dir, force)
}

func (a *app) prepareSpeechCommands(ctx context.Context, artifact downloadArtifact, dir string, force bool) error {
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
	if err := os.MkdirAll(parent, 0o700); err != nil {
		return fmt.Errorf("create data parent: %w", err)
	}
	if _, err := fmt.Fprintln(a.stdout(), "download mini_speech_commands.zip"); err != nil {
		return fmt.Errorf("report speech commands download: %w", err)
	}
	archivePath, err := a.downloadSpeechCommandsArchive(ctx, artifact, parent)
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

func (a *app) downloadSpeechCommandsArchive(ctx context.Context, artifact downloadArtifact, parent string) (string, error) {
	return a.downloadArtifact(ctx, artifact, parent, ".mini-speech-commands-*.zip")
}

func extractSpeechCommandsZip(archivePath, destination string) error {
	reader, err := zip.OpenReader(archivePath)
	if err != nil {
		return fmt.Errorf("open speech commands archive: %w", err)
	}
	defer func() { _ = reader.Close() }()

	if len(reader.File) > maxSpeechCommandsEntries {
		return fmt.Errorf("speech commands archive contains %d entries, limit is %d", len(reader.File), maxSpeechCommandsEntries)
	}
	seen := make(map[string]struct{}, len(reader.File))
	var extractedBytes int64
	for _, file := range reader.File {
		target, skip, err := speechCommandsEntryTarget(file.Name, destination)
		if err != nil || skip {
			if err != nil {
				return err
			}
			continue
		}
		if _, duplicate := seen[target]; duplicate {
			return fmt.Errorf("duplicate zip entry %q", file.Name)
		}
		seen[target] = struct{}{}
		if file.FileInfo().IsDir() {
			if err := os.MkdirAll(target, 0o700); err != nil {
				return fmt.Errorf("create extracted directory: %w", err)
			}
			continue
		}
		if !file.Mode().IsRegular() {
			return fmt.Errorf("unsupported zip entry %q", file.Name)
		}
		if file.UncompressedSize64 > uint64(maxSpeechCommandsEntryBytes) {
			return fmt.Errorf("zip entry %q exceeds %d bytes", file.Name, maxSpeechCommandsEntryBytes)
		}
		if err := os.MkdirAll(filepath.Dir(target), 0o700); err != nil {
			return fmt.Errorf("create extracted file parent: %w", err)
		}
		remaining := maxSpeechCommandsExtractedBytes - extractedBytes
		if remaining <= 0 {
			return fmt.Errorf("speech commands archive exceeds %d extracted bytes", maxSpeechCommandsExtractedBytes)
		}
		limit := min(int64(maxSpeechCommandsEntryBytes), remaining)
		written, err := copySpeechCommandsEntry(file, target, limit)
		if err != nil {
			return err
		}
		extractedBytes += written
	}
	return nil
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

func copySpeechCommandsEntry(file *zip.File, target string, maxBytes int64) (int64, error) {
	source, err := file.Open()
	if err != nil {
		return 0, fmt.Errorf("open zip entry %q: %w", file.Name, err)
	}
	output, err := os.OpenFile(target, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
	if err != nil {
		_ = source.Close()
		return 0, fmt.Errorf("create extracted file %q: %w", target, err)
	}
	written, copyErr := io.Copy(output, &io.LimitedReader{R: source, N: maxBytes + 1})
	closeErr := output.Close()
	sourceErr := source.Close()
	if copyErr != nil {
		return 0, fmt.Errorf("extract %q: %w", file.Name, copyErr)
	}
	if written > maxBytes {
		return 0, fmt.Errorf("zip entry %q exceeds %d bytes", file.Name, maxBytes)
	}
	if closeErr != nil {
		return 0, fmt.Errorf("close %q: %w", target, closeErr)
	}
	if sourceErr != nil {
		return 0, fmt.Errorf("close zip entry %q: %w", file.Name, sourceErr)
	}
	return written, nil
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

func (a *app) downloadGzip(ctx context.Context, artifact gzipArtifact, dest string, force bool) error {
	if err := artifact.validate(); err != nil {
		return err
	}
	reused, err := a.reuseGzipArtifact(dest, artifact.ExtractedSHA256, force)
	if err != nil || reused {
		return err
	}
	if _, err := fmt.Fprintf(a.stdout(), "download %s\n", filepath.Base(dest)); err != nil {
		return fmt.Errorf("report data download: %w", err)
	}
	archivePath, err := a.downloadArtifact(ctx, artifact.downloadArtifact, filepath.Dir(dest), ".mnist-*.gz")
	if err != nil {
		return err
	}
	defer func() { _ = os.Remove(archivePath) }()
	return installGzipArtifact(artifact, archivePath, dest)
}

func (a *app) reuseGzipArtifact(dest, checksum string, force bool) (bool, error) {
	if force {
		return false, nil
	}
	if _, err := os.Stat(dest); errors.Is(err, os.ErrNotExist) {
		return false, nil
	} else if err != nil {
		return false, fmt.Errorf("stat %s: %w", dest, err)
	}
	valid, err := fileHasSHA256(dest, checksum)
	if err != nil {
		return false, fmt.Errorf("verify %s: %w", dest, err)
	}
	if !valid {
		return false, fmt.Errorf("existing %s has an unexpected SHA-256; use --force to replace it", dest)
	}
	if _, err := fmt.Fprintf(a.stdout(), "exists %s\n", filepath.Base(dest)); err != nil {
		return false, fmt.Errorf("report existing data file: %w", err)
	}
	return true, nil
}

func installGzipArtifact(artifact gzipArtifact, archivePath, dest string) error {
	archive, err := os.Open(archivePath)
	if err != nil {
		return fmt.Errorf("open downloaded artifact: %w", err)
	}
	defer func() { _ = archive.Close() }()
	gz, err := gzip.NewReader(archive)
	if err != nil {
		return fmt.Errorf("open gzip stream for %s: %w", artifact.URL, err)
	}
	defer func() { _ = gz.Close() }()

	out, err := os.CreateTemp(filepath.Dir(dest), "."+filepath.Base(dest)+"-*")
	if err != nil {
		return fmt.Errorf("create temporary %s: %w", dest, err)
	}
	tmp := out.Name()
	installed := false
	defer func() {
		if out != nil {
			_ = out.Close()
		}
		if !installed {
			_ = os.Remove(tmp)
		}
	}()
	hash := sha256.New()
	written, err := io.Copy(io.MultiWriter(out, hash), &io.LimitedReader{R: gz, N: artifact.MaxExtractedBytes + 1})
	if err != nil {
		return fmt.Errorf("write %s: %w", dest, err)
	}
	if written > artifact.MaxExtractedBytes {
		return fmt.Errorf("extract %s exceeds %d bytes", artifact.URL, artifact.MaxExtractedBytes)
	}
	actual := hex.EncodeToString(hash.Sum(nil))
	if !strings.EqualFold(actual, artifact.ExtractedSHA256) {
		return fmt.Errorf("extracted %s has SHA-256 %s, want %s", artifact.URL, actual, artifact.ExtractedSHA256)
	}
	if err := out.Close(); err != nil {
		return fmt.Errorf("close %s: %w", tmp, err)
	}
	out = nil
	if err := os.Chmod(tmp, 0o600); err != nil {
		return fmt.Errorf("set permissions on %s: %w", tmp, err)
	}
	if err := os.Rename(tmp, dest); err != nil {
		return fmt.Errorf("rename %s: %w", dest, err)
	}
	installed = true
	return nil
}
