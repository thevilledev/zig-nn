package lab

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestServerCatalogRunAndSSEReplay(t *testing.T) {
	manager := NewManager(successfulExecutor())
	defer manager.Close()
	server, err := NewServer(manager, "", true)
	if err != nil {
		t.Fatal(err)
	}
	catalogResponse := httptest.NewRecorder()
	server.Handler().ServeHTTP(catalogResponse, httptest.NewRequest(http.MethodGet, "/api/experiments", nil))
	var specs []ExperimentSpec
	if err := json.NewDecoder(catalogResponse.Body).Decode(&specs); err != nil {
		t.Fatal(err)
	}
	if len(specs) != 6 || specs[0].Step != "" {
		t.Fatalf("catalog response = %#v", specs)
	}
	capabilitiesResponse := httptest.NewRecorder()
	server.Handler().ServeHTTP(capabilitiesResponse, httptest.NewRequest(http.MethodGet, "/api/capabilities", nil))
	var capabilities Capabilities
	if err := json.NewDecoder(capabilitiesResponse.Body).Decode(&capabilities); err != nil || !contains(capabilities.Backends, "cpu") {
		t.Fatalf("capabilities = %#v, %v", capabilities, err)
	}

	runResponse := httptest.NewRecorder()
	server.Handler().ServeHTTP(runResponse, httptest.NewRequest(http.MethodPost, "/api/runs", strings.NewReader(`{"experiment":"xor-training","parameters":{"epochs":100}}`)))
	if runResponse.Code != http.StatusAccepted {
		t.Fatalf("POST /api/runs = %d: %s", runResponse.Code, runResponse.Body.String())
	}
	var created map[string]string
	if err := json.NewDecoder(runResponse.Body).Decode(&created); err != nil {
		t.Fatal(err)
	}
	eventsResponse := httptest.NewRecorder()
	server.Handler().ServeHTTP(eventsResponse, httptest.NewRequest(http.MethodGet, "/api/runs/"+created["id"]+"/events", nil))
	text := eventsResponse.Body.String()
	for _, want := range []string{"event: run_started", "event: metric", "event: run_completed", `"run_id":"` + created["id"] + `"`} {
		if !strings.Contains(text, want) {
			t.Fatalf("SSE response missing %q:\n%s", want, text)
		}
	}

	replayRequest := httptest.NewRequest(http.MethodGet, "/api/runs/"+created["id"]+"/events", nil)
	replayRequest.Header.Set("Last-Event-ID", "1")
	replayResponse := httptest.NewRecorder()
	server.Handler().ServeHTTP(replayResponse, replayRequest)
	if strings.Contains(replayResponse.Body.String(), "id: 1\n") || !strings.Contains(replayResponse.Body.String(), "id: 2\n") {
		t.Fatalf("replay did not resume after sequence 1:\n%s", replayResponse.Body.String())
	}
}

func TestServerRejectsInvalidInputs(t *testing.T) {
	manager := NewManager(successfulExecutor())
	defer manager.Close()
	server, _ := NewServer(manager, "", true)

	tests := []struct {
		body string
		want int
	}{
		{`{"experiment":"unknown","parameters":{}}`, http.StatusNotFound},
		{`{"experiment":"xor-training","parameters":{"epochs":2}}`, http.StatusBadRequest},
		{`{"experiment":"xor-training","parameters":{"command":1}}`, http.StatusBadRequest},
		{`{"experiment":"xor-training","backend":"metal","parameters":{}}`, http.StatusBadRequest},
		{`{"experiment":"xor-training","parameters":{},"extra":true}`, http.StatusBadRequest},
	}
	for _, test := range tests {
		request := httptest.NewRequest(http.MethodPost, "/api/runs", strings.NewReader(test.body))
		response := httptest.NewRecorder()
		server.Handler().ServeHTTP(response, request)
		if response.Code != test.want {
			t.Fatalf("body %s: status %d, want %d: %s", test.body, response.Code, test.want, response.Body.String())
		}
	}

	oversized := `{"experiment":"xor-training","parameters":{}}` + strings.Repeat(" ", 65*1024)
	response := httptest.NewRecorder()
	server.Handler().ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/api/runs", strings.NewReader(oversized)))
	if response.Code != http.StatusBadRequest {
		t.Fatalf("oversized request status = %d", response.Code)
	}
}

func TestServerReturnsConflictWhileRunActive(t *testing.T) {
	release := make(chan struct{})
	manager := NewManager(ExecutorFunc(func(_ context.Context, _ ExperimentSpec, _ RunOptions, _ func([]byte) error, _ func(string)) error {
		<-release
		return nil
	}))
	server, _ := NewServer(manager, "", true)
	requestBody := []byte(`{"experiment":"xor-training","parameters":{}}`)
	first := httptest.NewRecorder()
	server.Handler().ServeHTTP(first, httptest.NewRequest(http.MethodPost, "/api/runs", bytes.NewReader(requestBody)))
	if first.Code != http.StatusAccepted {
		t.Fatalf("first run status = %d", first.Code)
	}
	second := httptest.NewRecorder()
	server.Handler().ServeHTTP(second, httptest.NewRequest(http.MethodPost, "/api/runs", bytes.NewReader(requestBody)))
	if second.Code != http.StatusConflict {
		t.Fatalf("second run status = %d: %s", second.Code, second.Body.String())
	}
	close(release)
	manager.Close()
}

func TestSPAHandlerServesAssetsAndFallback(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "index.html"), []byte("<main>learning lab</main>"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "app.js"), []byte("export {}"), 0o600); err != nil {
		t.Fatal(err)
	}
	manager := NewManager(successfulExecutor())
	defer manager.Close()
	server, err := NewServer(manager, dir, false)
	if err != nil {
		t.Fatal(err)
	}
	for _, route := range []string{"/", "/experiments/xor", "/app.js"} {
		response := httptest.NewRecorder()
		server.Handler().ServeHTTP(response, httptest.NewRequest(http.MethodGet, route, nil))
		if response.Code != http.StatusOK {
			t.Fatalf("GET %s = %d", route, response.Code)
		}
	}
}

func successfulExecutor() Executor {
	return ExecutorFunc(func(_ context.Context, spec ExperimentSpec, _ RunOptions, stdout func([]byte) error, _ func(string)) error {
		for _, event := range []map[string]any{
			{"v": 1, "type": "run_started", "experiment": spec.ID, "data": map[string]any{"config": map[string]any{}}},
			{"v": 1, "type": "metric", "experiment": spec.ID, "step": 1, "total_steps": 1, "data": map[string]any{"name": "loss", "value": 0.1}},
			{"v": 1, "type": "run_completed", "experiment": spec.ID, "step": 1, "total_steps": 1, "data": map[string]any{"final_loss": 0.1}},
		} {
			line, _ := json.Marshal(event)
			if err := stdout(line); err != nil {
				return err
			}
		}
		return nil
	})
}
