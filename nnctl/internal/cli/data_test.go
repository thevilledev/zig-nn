package cli

import (
	"archive/zip"
	"bytes"
	"context"
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
	app := &App{Stdout: &stdout}
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

	app := &App{}
	destination := filepath.Join(t.TempDir(), "mini_speech_commands")
	err = app.prepareSpeechCommands(context.Background(), server.URL, destination, false)
	if err == nil || !strings.Contains(err.Error(), "unsafe zip entry") {
		t.Fatalf("prepareSpeechCommands() error = %v, want unsafe zip entry", err)
	}
	if _, statErr := os.Stat(filepath.Join(filepath.Dir(destination), "escape")); !os.IsNotExist(statErr) {
		t.Fatalf("traversal target exists, stat error = %v", statErr)
	}
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
