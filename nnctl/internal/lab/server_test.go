package lab

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	cloudworkflow "nnctl/internal/cloud/workflow"
)

func TestServerCatalogRunAndSSEReplay(t *testing.T) {
	manager := NewManager(t.Context(), successfulExecutor())
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
	if len(specs) != 7 || specs[0].Step != "" {
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
	manager := NewManager(t.Context(), successfulExecutor())
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

	crossOriginCancel := httptest.NewRequest(http.MethodDelete, "/api/runs/missing", nil)
	crossOriginCancel.Host = "attacker.example"
	crossOriginCancel.Header.Set("Origin", "https://attacker.example")
	response = httptest.NewRecorder()
	server.Handler().ServeHTTP(response, crossOriginCancel)
	if response.Code != http.StatusForbidden {
		t.Fatalf("cross-origin cancel status = %d: %s", response.Code, response.Body.String())
	}
}

func TestServerReturnsConflictWhileRunActive(t *testing.T) {
	release := make(chan struct{})
	manager := NewManager(t.Context(), ExecutorFunc(func(_ context.Context, _ ExperimentSpec, _ RunOptions, _ func([]byte) error, _ func(string)) error {
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

func TestServerCloudWorkerAndRemoteRunLifecycle(t *testing.T) {
	repoRoot, _ := filepath.Abs("../../..")
	transport := writeFakeCloudTransport(t)
	client := newFakeCloudClient()
	cloud, err := NewCloudManager(CloudManagerOptions{
		RepoRoot:      repoRoot,
		SSH:           transport.ssh,
		Rsync:         transport.rsync,
		DatabasePath:  filepath.Join(t.TempDir(), "state.sqlite"),
		Timeout:       cloudTestTimeout,
		PollInterval:  time.Millisecond,
		ClientFactory: func(context.Context, string) (cloudworkflow.Client, error) { return client, nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = cloud.Close() }()
	manager := NewManager(t.Context(), RoutingExecutor{Local: successfulExecutor(), Cloud: cloud})
	defer manager.Close()
	server, err := NewServerWithOptions(manager, "", ServerOptions{APIOnly: true, Cloud: cloud})
	if err != nil {
		t.Fatal(err)
	}

	statusResponse := httptest.NewRecorder()
	server.Handler().ServeHTTP(statusResponse, httptest.NewRequest(http.MethodGet, "/api/cloud/status", nil))
	if statusResponse.Code != http.StatusOK || !strings.Contains(statusResponse.Body.String(), `"configured":true`) {
		t.Fatalf("cloud status = %d: %s", statusResponse.Code, statusResponse.Body.String())
	}
	optionsResponse := httptest.NewRecorder()
	server.Handler().ServeHTTP(optionsResponse, httptest.NewRequest(http.MethodGet, "/api/cloud/options", nil))
	if optionsResponse.Code != http.StatusOK || !strings.Contains(optionsResponse.Body.String(), "1A100.22V") || !strings.Contains(optionsResponse.Body.String(), defaultCloudSourceVolumeName) {
		t.Fatalf("cloud options = %d: %s", optionsResponse.Code, optionsResponse.Body.String())
	}
	if strings.Contains(optionsResponse.Body.String(), "ssh_keys") || strings.Contains(optionsResponse.Body.String(), "volumes") {
		t.Fatalf("cloud options exposed removed SSH-key or volume selectors: %s", optionsResponse.Body.String())
	}

	deployBody := `{
		"instance_type":"1A100.22V",
		"market":"spot",
		"location_code":"FIN-02",
		"auto_destroy":false
	}`
	crossOriginRequest := httptest.NewRequest(http.MethodPost, "/api/cloud/workers", strings.NewReader(deployBody))
	crossOriginRequest.Header.Set("Origin", "https://attacker.example")
	crossOriginRequest.Host = "attacker.example"
	crossOriginResponse := httptest.NewRecorder()
	server.Handler().ServeHTTP(crossOriginResponse, crossOriginRequest)
	if crossOriginResponse.Code != http.StatusForbidden {
		t.Fatalf("cross-origin deploy = %d: %s", crossOriginResponse.Code, crossOriginResponse.Body.String())
	}
	legacyResponse := httptest.NewRecorder()
	legacyBody := `{"instance_type":"1A100.22V","market":"spot","ssh_key_id":"00000000-0000-0000-0000-000000000001"}`
	server.Handler().ServeHTTP(legacyResponse, httptest.NewRequest(http.MethodPost, "/api/cloud/workers", strings.NewReader(legacyBody)))
	if legacyResponse.Code != http.StatusBadRequest {
		t.Fatalf("legacy SSH-key deploy = %d: %s", legacyResponse.Code, legacyResponse.Body.String())
	}
	deployResponse := httptest.NewRecorder()
	deployRequest := httptest.NewRequest(http.MethodPost, "/api/cloud/workers", strings.NewReader(deployBody))
	deployRequest.Host = "127.0.0.1:8091"
	deployRequest.Header.Set("Origin", "http://127.0.0.1:8091")
	server.Handler().ServeHTTP(deployResponse, deployRequest)
	if deployResponse.Code != http.StatusAccepted {
		t.Fatalf("deploy = %d: %s", deployResponse.Code, deployResponse.Body.String())
	}
	var created CloudWorker
	if err := json.NewDecoder(deployResponse.Body).Decode(&created); err != nil {
		t.Fatal(err)
	}
	waitForCloudWorkerState(t, cloud, created.ID, CloudWorkerReady)

	runBody := fmt.Sprintf(`{
		"experiment":"optimizer-lab",
		"backend":"cuda",
		"parameters":{"steps":20},
		"target":{"kind":"cloud","worker_id":%q},
		"acknowledge_committed_head":true
	}`, created.ID)
	runResponse := httptest.NewRecorder()
	server.Handler().ServeHTTP(runResponse, httptest.NewRequest(http.MethodPost, "/api/runs", strings.NewReader(runBody)))
	if runResponse.Code != http.StatusAccepted {
		t.Fatalf("cloud run = %d: %s", runResponse.Code, runResponse.Body.String())
	}
	var run map[string]string
	if err := json.NewDecoder(runResponse.Body).Decode(&run); err != nil {
		t.Fatal(err)
	}
	eventsResponse := httptest.NewRecorder()
	server.Handler().ServeHTTP(eventsResponse, httptest.NewRequest(http.MethodGet, "/api/runs/"+run["id"]+"/events", nil))
	for _, expected := range []string{"event: run_status", "event: run_started", "event: run_completed", `"selected_backend":"cuda"`} {
		if !strings.Contains(eventsResponse.Body.String(), expected) {
			t.Fatalf("cloud events missing %q:\n%s", expected, eventsResponse.Body.String())
		}
	}

	destroyResponse := httptest.NewRecorder()
	server.Handler().ServeHTTP(destroyResponse, httptest.NewRequest(http.MethodDelete, "/api/cloud/workers/"+created.ID, nil))
	if destroyResponse.Code != http.StatusNoContent {
		t.Fatalf("destroy = %d: %s", destroyResponse.Code, destroyResponse.Body.String())
	}
	workerEvents := httptest.NewRecorder()
	server.Handler().ServeHTTP(workerEvents, httptest.NewRequest(http.MethodGet, "/api/cloud/workers/"+created.ID+"/events", nil))
	if !strings.Contains(workerEvents.Body.String(), "event: worker") || !strings.Contains(workerEvents.Body.String(), `"state":"destroyed"`) {
		t.Fatalf("worker events = %s", workerEvents.Body.String())
	}
}

func TestSPAHandlerServesAssetsAndFallback(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "index.html"), []byte("<main>learning lab</main>"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "app.js"), []byte("export {}"), 0o600); err != nil {
		t.Fatal(err)
	}
	manager := NewManager(t.Context(), successfulExecutor())
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
