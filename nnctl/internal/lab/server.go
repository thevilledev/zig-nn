package lab

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"

	"nnctl/internal/localhttp"
)

type Server struct {
	manager      *Manager
	cloud        *CloudManager
	capabilities Capabilities
	handler      http.Handler
}

type Capabilities struct {
	Platform     string   `json:"platform"`
	Backends     []string `json:"backends"`
	CloudEnabled bool     `json:"cloud_enabled"`
}

func NewServer(manager *Manager, assets string, apiOnly bool) (*Server, error) {
	return NewServerWithOptions(manager, assets, ServerOptions{APIOnly: apiOnly})
}

type ServerOptions struct {
	APIOnly bool
	Cloud   *CloudManager
}

func NewServerWithOptions(manager *Manager, assets string, options ServerOptions) (*Server, error) {
	capabilities := hostCapabilities()
	capabilities.CloudEnabled = options.Cloud != nil
	server := &Server{manager: manager, cloud: options.Cloud, capabilities: capabilities}
	mux := http.NewServeMux()
	mux.HandleFunc("/api/capabilities", server.handleCapabilities)
	mux.HandleFunc("/api/experiments", server.handleExperiments)
	mux.HandleFunc("/api/runs", server.handleRuns)
	mux.HandleFunc("/api/runs/", server.handleRun)
	mux.HandleFunc("/api/cloud/status", server.handleCloudStatus)
	mux.HandleFunc("/api/cloud/options", server.handleCloudOptions)
	mux.HandleFunc("/api/cloud/workers", server.handleCloudWorkers)
	mux.HandleFunc("/api/cloud/workers/", server.handleCloudWorker)
	if options.APIOnly {
		mux.HandleFunc("/", http.NotFound)
	} else {
		spa, err := newSPAHandler(assets)
		if err != nil {
			return nil, err
		}
		mux.Handle("/", spa)
	}
	server.handler = localhttp.SecurityHeaders(mux)
	return server, nil
}

func hostCapabilities() Capabilities {
	backends := []string{"cpu"}
	if runtime.GOOS == "darwin" {
		backends = append(backends, "metal")
	}
	return Capabilities{Platform: runtime.GOOS, Backends: backends}
}

func (s *Server) handleCapabilities(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w, http.MethodGet)
		return
	}
	writeJSON(w, http.StatusOK, s.capabilities)
}

func (s *Server) Handler() http.Handler {
	return s.handler
}

func (s *Server) handleExperiments(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w, http.MethodGet)
		return
	}
	writeJSON(w, http.StatusOK, Experiments())
}

type runRequest struct {
	Experiment               string                 `json:"experiment"`
	Backend                  string                 `json:"backend"`
	Parameters               map[string]json.Number `json:"parameters"`
	Target                   runTargetRequest       `json:"target"`
	AcknowledgeCommittedHead bool                   `json:"acknowledge_committed_head"`
}

type runTargetRequest struct {
	Kind     RunTarget `json:"kind"`
	WorkerID string    `json:"worker_id,omitempty"`
}

func (s *Server) handleRuns(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		methodNotAllowed(w, http.MethodPost)
		return
	}
	if !requireSameOrigin(w, r) {
		return
	}
	defer func() { _ = r.Body.Close() }()
	decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, 64*1024))
	decoder.UseNumber()
	decoder.DisallowUnknownFields()
	var request runRequest
	if err := decoder.Decode(&request); err != nil {
		writeAPIError(w, http.StatusBadRequest, "invalid_request", "request body must be valid JSON")
		return
	}
	if err := ensureJSONEOF(decoder); err != nil {
		writeAPIError(w, http.StatusBadRequest, "invalid_request", "request body must contain one JSON object")
		return
	}
	spec, ok := ResolveExperiment(request.Experiment)
	if !ok {
		writeAPIError(w, http.StatusNotFound, "unknown_experiment", "experiment is not available in the learning lab")
		return
	}
	options, err := BuildRunOptions(spec, request.Backend, request.Parameters)
	if err != nil {
		writeAPIError(w, http.StatusBadRequest, "invalid_parameters", err.Error())
		return
	}
	if request.Target.Kind == "" {
		request.Target.Kind = RunTargetLocal
	}
	switch request.Target.Kind {
	case RunTargetLocal:
		if request.Target.WorkerID != "" {
			writeAPIError(w, http.StatusBadRequest, "invalid_target", "local runs must not include a worker ID")
			return
		}
		if !contains(s.capabilities.Backends, options.Backend) {
			writeAPIError(w, http.StatusBadRequest, "unsupported_backend", fmt.Sprintf("backend %q is not available on %s", options.Backend, s.capabilities.Platform))
			return
		}
	case RunTargetCloud:
		if s.cloud == nil {
			writeAPIError(w, http.StatusBadRequest, "cloud_disabled", "cloud execution is disabled; start nnctl lab with --cloud")
			return
		}
		if err := s.cloud.ValidateWorker(request.Target.WorkerID, options.Backend); err != nil {
			writeAPIError(w, http.StatusConflict, "cloud_worker_unavailable", err.Error())
			return
		}
		options.WorkerID = request.Target.WorkerID
		options.AcknowledgeDirty = request.AcknowledgeCommittedHead
	default:
		writeAPIError(w, http.StatusBadRequest, "invalid_target", "target kind must be local or cloud")
		return
	}
	options.Target = request.Target.Kind
	id, err := s.manager.Start(spec, options)
	if errors.Is(err, ErrBusy) {
		writeAPIError(w, http.StatusConflict, "run_busy", err.Error())
		return
	}
	if err != nil {
		writeAPIError(w, http.StatusInternalServerError, "start_failed", err.Error())
		return
	}
	writeJSON(w, http.StatusAccepted, map[string]string{"id": id})
}

func (s *Server) handleCloudStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w, http.MethodGet)
		return
	}
	if s.cloud == nil {
		writeJSON(w, http.StatusOK, CloudStatus{Enabled: false, Provider: "verda"})
		return
	}
	writeJSON(w, http.StatusOK, s.cloud.Status(r.Context()))
}

func (s *Server) handleCloudOptions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w, http.MethodGet)
		return
	}
	if s.cloud == nil {
		writeAPIError(w, http.StatusNotFound, "cloud_disabled", "cloud execution is disabled")
		return
	}
	options, err := s.cloud.Options(r.Context())
	if err != nil {
		writeAPIError(w, http.StatusBadGateway, "cloud_options_failed", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, options)
}

func (s *Server) handleCloudWorkers(w http.ResponseWriter, r *http.Request) {
	if s.cloud == nil {
		writeAPIError(w, http.StatusNotFound, "cloud_disabled", "cloud execution is disabled")
		return
	}
	switch r.Method {
	case http.MethodGet:
		writeJSON(w, http.StatusOK, s.cloud.Workers())
	case http.MethodPost:
		if !requireSameOrigin(w, r) {
			return
		}
		defer func() { _ = r.Body.Close() }()
		decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, 16*1024))
		decoder.DisallowUnknownFields()
		var request CloudDeployRequest
		if err := decoder.Decode(&request); err != nil || ensureJSONEOF(decoder) != nil {
			writeAPIError(w, http.StatusBadRequest, "invalid_request", "request body must be one valid cloud worker object")
			return
		}
		worker, err := s.cloud.Deploy(r.Context(), request)
		if err != nil {
			writeAPIError(w, http.StatusConflict, "cloud_deploy_failed", err.Error())
			return
		}
		writeJSON(w, http.StatusAccepted, worker)
	default:
		w.Header().Set("Allow", http.MethodGet+", "+http.MethodPost)
		writeAPIError(w, http.StatusMethodNotAllowed, "method_not_allowed", "method not allowed")
	}
}

func (s *Server) handleCloudWorker(w http.ResponseWriter, r *http.Request) {
	if s.cloud == nil {
		writeAPIError(w, http.StatusNotFound, "cloud_disabled", "cloud execution is disabled")
		return
	}
	remainder := strings.Trim(strings.TrimPrefix(r.URL.Path, "/api/cloud/workers/"), "/")
	parts := strings.Split(remainder, "/")
	if len(parts) == 2 && parts[0] != "" && parts[1] == "events" {
		if r.Method != http.MethodGet {
			methodNotAllowed(w, http.MethodGet)
			return
		}
		s.handleCloudWorkerEvents(w, r, parts[0])
		return
	}
	if len(parts) != 1 || parts[0] == "" {
		http.NotFound(w, r)
		return
	}
	id := parts[0]
	if r.Method != http.MethodDelete {
		methodNotAllowed(w, http.MethodDelete)
		return
	}
	if !requireSameOrigin(w, r) {
		return
	}
	ctx, cancel := context.WithTimeout(context.WithoutCancel(r.Context()), 5*time.Minute)
	defer cancel()
	if err := s.cloud.Destroy(ctx, id); err != nil {
		writeAPIError(w, http.StatusBadGateway, "cloud_destroy_failed", err.Error())
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func (s *Server) handleCloudWorkerEvents(w http.ResponseWriter, r *http.Request, workerID string) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeAPIError(w, http.StatusInternalServerError, "streaming_unavailable", "response does not support streaming")
		return
	}
	subscription, err := s.cloud.SubscribeWorker(workerID)
	if err != nil {
		writeAPIError(w, http.StatusNotFound, "cloud_worker_not_found", err.Error())
		return
	}
	defer subscription.Close()
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	if err := writeCloudWorkerSSE(w, subscription.Initial); err != nil {
		return
	}
	flusher.Flush()
	if subscription.Initial.State == CloudWorkerDestroyed {
		return
	}
	keepalive := time.NewTicker(15 * time.Second)
	defer keepalive.Stop()
	for {
		select {
		case worker := <-subscription.Events:
			if err := writeCloudWorkerSSE(w, worker); err != nil {
				return
			}
			flusher.Flush()
			if worker.State == CloudWorkerDestroyed {
				return
			}
		case <-keepalive.C:
			if _, err := io.WriteString(w, ": keepalive\n\n"); err != nil {
				return
			}
			flusher.Flush()
		case <-r.Context().Done():
			return
		}
	}
}

func writeCloudWorkerSSE(w io.Writer, worker CloudWorker) error {
	encoded, err := json.Marshal(worker)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(w, "event: worker\ndata: %s\n\n", encoded)
	return err
}

func (s *Server) handleRun(w http.ResponseWriter, r *http.Request) {
	remainder := strings.Trim(strings.TrimPrefix(r.URL.Path, "/api/runs/"), "/")
	parts := strings.Split(remainder, "/")
	if len(parts) == 1 && parts[0] != "" {
		if r.Method != http.MethodDelete {
			methodNotAllowed(w, http.MethodDelete)
			return
		}
		if !requireSameOrigin(w, r) {
			return
		}
		if err := s.manager.Cancel(parts[0]); errors.Is(err, ErrRunNotFound) {
			writeAPIError(w, http.StatusNotFound, "run_not_found", err.Error())
			return
		} else if err != nil {
			writeAPIError(w, http.StatusInternalServerError, "cancel_failed", err.Error())
			return
		}
		w.WriteHeader(http.StatusNoContent)
		return
	}
	if len(parts) == 2 && parts[0] != "" && parts[1] == "events" {
		if r.Method != http.MethodGet {
			methodNotAllowed(w, http.MethodGet)
			return
		}
		s.handleEvents(w, r, parts[0])
		return
	}
	http.NotFound(w, r)
}

func (s *Server) handleEvents(w http.ResponseWriter, r *http.Request, id string) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeAPIError(w, http.StatusInternalServerError, "streaming_unavailable", "response does not support streaming")
		return
	}
	after, err := parseLastEventID(r.Header.Get("Last-Event-ID"))
	if err != nil {
		writeAPIError(w, http.StatusBadRequest, "invalid_event_id", err.Error())
		return
	}
	subscription, err := s.manager.Subscribe(id, after)
	if errors.Is(err, ErrRunNotFound) {
		writeAPIError(w, http.StatusNotFound, "run_not_found", err.Error())
		return
	}
	if err != nil {
		writeAPIError(w, http.StatusInternalServerError, "subscribe_failed", err.Error())
		return
	}
	defer subscription.Close()

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	lastSequence := after
	for _, event := range subscription.Replay {
		if err := writeSSE(w, event); err != nil {
			return
		}
		lastSequence = event.Sequence
	}
	flusher.Flush()

	keepalive := time.NewTicker(15 * time.Second)
	defer keepalive.Stop()
	for {
		select {
		case event := <-subscription.Events:
			if event.Sequence <= lastSequence {
				continue
			}
			if err := writeSSE(w, event); err != nil {
				return
			}
			lastSequence = event.Sequence
			flusher.Flush()
		case <-subscription.Done:
			remaining, remainingErr := s.manager.EventsSince(id, lastSequence)
			if remainingErr != nil {
				return
			}
			for _, event := range remaining {
				if err := writeSSE(w, event); err != nil {
					return
				}
			}
			flusher.Flush()
			return
		case <-keepalive.C:
			if _, err := io.WriteString(w, ": keepalive\n\n"); err != nil {
				return
			}
			flusher.Flush()
		case <-r.Context().Done():
			return
		}
	}
}

func parseLastEventID(value string) (uint64, error) {
	if value == "" {
		return 0, nil
	}
	parsed, err := strconv.ParseUint(value, 10, 64)
	if err != nil {
		return 0, fmt.Errorf("Last-Event-ID must be an unsigned integer")
	}
	return parsed, nil
}

func writeSSE(w io.Writer, event Event) error {
	encoded, err := json.Marshal(event)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(w, "id: %d\nevent: %s\ndata: %s\n\n", event.Sequence, event.Type, encoded)
	return err
}

func ensureJSONEOF(decoder *json.Decoder) error {
	var extra any
	if err := decoder.Decode(&extra); errors.Is(err, io.EOF) {
		return nil
	} else {
		return err
	}
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}

func writeAPIError(w http.ResponseWriter, status int, code, message string) {
	writeJSON(w, status, map[string]any{"error": map[string]string{"code": code, "message": message}})
}

func methodNotAllowed(w http.ResponseWriter, allowed string) {
	w.Header().Set("Allow", allowed)
	writeAPIError(w, http.StatusMethodNotAllowed, "method_not_allowed", "method not allowed")
}

func requireSameOrigin(w http.ResponseWriter, r *http.Request) bool {
	if !localhttp.SameOrigin(r) {
		writeAPIError(w, http.StatusForbidden, "cross_origin_request", "cross-origin state changes are not allowed")
		return false
	}
	return true
}

func newSPAHandler(assets string) (http.Handler, error) {
	info, err := os.Stat(filepath.Join(assets, "index.html"))
	if err != nil || info.IsDir() {
		return nil, fmt.Errorf("learning lab assets not found at %s; run mise run web:build", assets)
	}
	root := os.DirFS(assets)
	files := http.FileServer(http.FS(root))
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		clean := strings.TrimPrefix(path.Clean("/"+r.URL.Path), "/")
		if clean == "." || clean == "" {
			files.ServeHTTP(w, r)
			return
		}
		if info, statErr := fs.Stat(root, clean); statErr == nil && !info.IsDir() {
			files.ServeHTTP(w, r)
			return
		}
		r.URL.Path = "/"
		files.ServeHTTP(w, r)
	}), nil
}
