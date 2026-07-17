package cli

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
)

type downloadArtifact struct {
	URL              string
	SHA256           string
	MaxDownloadBytes int64
}

func (artifact downloadArtifact) validate() error {
	if strings.TrimSpace(artifact.URL) == "" {
		return fmt.Errorf("artifact URL is required")
	}
	if !validSHA256(artifact.SHA256) {
		return fmt.Errorf("artifact SHA-256 must contain 64 hexadecimal characters")
	}
	if artifact.MaxDownloadBytes <= 0 {
		return fmt.Errorf("artifact download limit must be positive")
	}
	return nil
}

func validSHA256(value string) bool {
	checksum, err := hex.DecodeString(value)
	return err == nil && len(checksum) == sha256.Size
}

func (a *app) downloadArtifact(ctx context.Context, artifact downloadArtifact, parent, pattern string) (path string, resultErr error) {
	if err := artifact.validate(); err != nil {
		return "", err
	}
	file, err := os.CreateTemp(parent, pattern)
	if err != nil {
		return "", fmt.Errorf("create temporary artifact: %w", err)
	}
	tempPath := file.Name()
	path = tempPath
	defer func() {
		if file != nil {
			if err := file.Close(); err != nil && resultErr == nil {
				resultErr = fmt.Errorf("close temporary artifact: %w", err)
			}
		}
		if resultErr != nil {
			_ = os.Remove(tempPath)
		}
	}()

	request, err := http.NewRequestWithContext(ctx, http.MethodGet, artifact.URL, nil)
	if err != nil {
		return "", fmt.Errorf("create artifact request: %w", err)
	}
	response, err := a.http().Do(request)
	if err != nil {
		return "", fmt.Errorf("download %s: %w", artifact.URL, err)
	}
	defer func() { _ = response.Body.Close() }()
	if response.StatusCode != http.StatusOK {
		return "", fmt.Errorf("download %s: %s", artifact.URL, response.Status)
	}
	if response.ContentLength > artifact.MaxDownloadBytes {
		return "", fmt.Errorf("download %s exceeds %d bytes", artifact.URL, artifact.MaxDownloadBytes)
	}

	hash := sha256.New()
	limited := &io.LimitedReader{R: response.Body, N: artifact.MaxDownloadBytes + 1}
	written, err := io.Copy(io.MultiWriter(file, hash), limited)
	if err != nil {
		return "", fmt.Errorf("write temporary artifact: %w", err)
	}
	if written > artifact.MaxDownloadBytes {
		return "", fmt.Errorf("download %s exceeds %d bytes", artifact.URL, artifact.MaxDownloadBytes)
	}
	actual := hex.EncodeToString(hash.Sum(nil))
	if !strings.EqualFold(actual, artifact.SHA256) {
		return "", fmt.Errorf("download %s has SHA-256 %s, want %s", artifact.URL, actual, artifact.SHA256)
	}
	if err := file.Close(); err != nil {
		return "", fmt.Errorf("close temporary artifact: %w", err)
	}
	file = nil
	return path, nil
}

func fileHasSHA256(path, expected string) (bool, error) {
	file, err := os.Open(path)
	if err != nil {
		return false, err
	}
	defer func() { _ = file.Close() }()
	hash := sha256.New()
	if _, err := io.Copy(hash, file); err != nil {
		return false, err
	}
	return strings.EqualFold(hex.EncodeToString(hash.Sum(nil)), expected), nil
}
