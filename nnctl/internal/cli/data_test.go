package cli

import (
	"archive/zip"
	"bytes"
	"compress/gzip"
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestPrepareSpeechCommandsExtractsIdempotentlyAndForcesReplacement(t *testing.T) {
	var archive []byte
	archive = speechCommandsFixture(t, "first")
	requests := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		requests++
		_, _ = w.Write(archive)
	}))
	defer server.Close()

	destination := filepath.Join(t.TempDir(), "mini_speech_commands")
	var stdout bytes.Buffer
	app := &app{stdoutWriter: &stdout}
	if err := app.prepareSpeechCommands(context.Background(), server.URL, destination, false); err != nil {
		t.Fatalf("prepareSpeechCommands() error = %v", err)
	}
	contents, err := os.ReadFile(filepath.Join(destination, "yes", "sample.wav"))
	if err != nil || string(contents) != "first" {
		t.Fatalf("extracted contents = %q, err = %v", contents, err)
	}
	if err := app.prepareSpeechCommands(context.Background(), server.URL, destination, false); err != nil {
		t.Fatalf("idempotent prepareSpeechCommands() error = %v", err)
	}
	if requests != 1 {
		t.Fatalf("requests = %d, want 1 after idempotent call", requests)
	}

	archive = speechCommandsFixture(t, "second")
	if err := app.prepareSpeechCommands(context.Background(), server.URL, destination, true); err != nil {
		t.Fatalf("forced prepareSpeechCommands() error = %v", err)
	}
	contents, err = os.ReadFile(filepath.Join(destination, "yes", "sample.wav"))
	if err != nil || string(contents) != "second" {
		t.Fatalf("replacement contents = %q, err = %v", contents, err)
	}
	if requests != 2 {
		t.Fatalf("requests = %d, want 2 after forced call", requests)
	}
	if !strings.Contains(stdout.String(), "prepared") || !strings.Contains(stdout.String(), "exists") {
		t.Fatalf("stdout missing lifecycle messages:\n%s", stdout.String())
	}
}

func TestPrepareSpeechCommandsRejectsTraversal(t *testing.T) {
	var buffer bytes.Buffer
	writer := zip.NewWriter(&buffer)
	entry, err := writer.Create("mini_speech_commands/../../escape")
	if err != nil {
		t.Fatal(err)
	}
	_, _ = entry.Write([]byte("bad"))
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write(buffer.Bytes())
	}))
	defer server.Close()

	app := &app{}
	destination := filepath.Join(t.TempDir(), "mini_speech_commands")
	err = app.prepareSpeechCommands(context.Background(), server.URL, destination, false)
	if err == nil || !strings.Contains(err.Error(), "unsafe zip entry") {
		t.Fatalf("prepareSpeechCommands() error = %v, want unsafe zip entry", err)
	}
	if _, statErr := os.Stat(filepath.Join(filepath.Dir(destination), "escape")); !os.IsNotExist(statErr) {
		t.Fatalf("traversal target exists, stat error = %v", statErr)
	}
}

func TestRunDataTinyGPTUsesExperimentsPath(t *testing.T) {
	repoRoot := t.TempDir()
	app := &app{repoRoot: repoRoot}
	err := app.runDataTinyGPT(context.Background())
	want := filepath.Join(repoRoot, "experiments", "tiny_gpt", "scripts", "prepare_data.sh")
	if err == nil || !strings.Contains(err.Error(), want) {
		t.Fatalf("runDataTinyGPT() error = %v, want path %s", err, want)
	}
}

func TestDownloadGzipUsesConfiguredHTTPClient(t *testing.T) {
	var compressed bytes.Buffer
	writer := gzip.NewWriter(&compressed)
	if _, err := writer.Write([]byte("mnist")); err != nil {
		t.Fatalf("compress fixture: %v", err)
	}
	if err := writer.Close(); err != nil {
		t.Fatalf("close fixture: %v", err)
	}

	requested := false
	client := &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		requested = true
		if request.Context() != t.Context() {
			t.Fatal("download request did not preserve the caller context")
		}
		return &http.Response{
			StatusCode: http.StatusOK,
			Status:     "200 OK",
			Body:       io.NopCloser(bytes.NewReader(compressed.Bytes())),
		}, nil
	})}
	app := &app{httpClient: client}
	destination := filepath.Join(t.TempDir(), "mnist")

	if err := app.downloadGzip(t.Context(), "https://example.test/mnist.gz", destination, false); err != nil {
		t.Fatalf("downloadGzip() error = %v", err)
	}
	if !requested {
		t.Fatal("configured HTTP client was not used")
	}
	contents, err := os.ReadFile(destination)
	if err != nil {
		t.Fatalf("read downloaded data: %v", err)
	}
	if string(contents) != "mnist" {
		t.Fatalf("downloaded data = %q, want mnist", contents)
	}
}

func TestDownloadGzipReportsOutputFailure(t *testing.T) {
	destination := filepath.Join(t.TempDir(), "mnist")
	if err := os.WriteFile(destination, nil, 0o644); err != nil {
		t.Fatalf("create existing data file: %v", err)
	}
	wantErr := errors.New("output unavailable")
	app := &app{stdoutWriter: testErrorWriter{err: wantErr}}

	err := app.downloadGzip(t.Context(), "https://example.test/mnist.gz", destination, false)
	if !errors.Is(err, wantErr) {
		t.Fatalf("downloadGzip() error = %v, want %v", err, wantErr)
	}
}

type testErrorWriter struct {
	err error
}

func (w testErrorWriter) Write([]byte) (int, error) {
	return 0, w.err
}

func speechCommandsFixture(t *testing.T, contents string) []byte {
	t.Helper()
	var buffer bytes.Buffer
	writer := zip.NewWriter(&buffer)
	metadata, err := writer.Create("__MACOSX/._mini_speech_commands")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := metadata.Write([]byte("ignored")); err != nil {
		t.Fatal(err)
	}
	for _, label := range speechCommandLabels {
		entry, err := writer.Create("mini_speech_commands/" + label + "/sample.wav")
		if err != nil {
			t.Fatal(err)
		}
		if _, err := entry.Write([]byte(contents)); err != nil {
			t.Fatal(err)
		}
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	return append([]byte(nil), buffer.Bytes()...)
}
