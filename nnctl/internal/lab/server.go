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
)

type Server struct {
	manager      *Manager
	capabilities Capabilities
	handler      http.Handler
}

type Capabilities struct {
	Platform string   `json:"platform"`
	Backends []string `json:"backends"`
}

func NewServer(manager *Manager, assets string, apiOnly bool) (*Server, error) {
	server := &Server{manager: manager, capabilities: hostCapabilities()}
	mux := http.NewServeMux()
	mux.HandleFunc("/api/capabilities", server.handleCapabilities)
	mux.HandleFunc("/api/experiments", server.handleExperiments)
	mux.HandleFunc("/api/runs", server.handleRuns)
	mux.HandleFunc("/api/runs/", server.handleRun)
	if apiOnly {
		mux.HandleFunc("/", http.NotFound)
	} else {
		spa, err := newSPAHandler(assets)
		if err != nil {
			return nil, err
		}
		mux.Handle("/", spa)
	}
	server.handler = securityHeaders(mux)
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
	Experiment string                 `json:"experiment"`
	Backend    string                 `json:"backend"`
	Parameters map[string]json.Number `json:"parameters"`
}

func (s *Server) handleRuns(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		methodNotAllowed(w, http.MethodPost)
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
	if !contains(s.capabilities.Backends, options.Backend) {
		writeAPIError(w, http.StatusBadRequest, "unsupported_backend", fmt.Sprintf("backend %q is not available on %s", options.Backend, s.capabilities.Platform))
		return
	}
	id, err := s.manager.Start(context.Background(), spec, options)
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

func (s *Server) handleRun(w http.ResponseWriter, r *http.Request) {
	remainder := strings.Trim(strings.TrimPrefix(r.URL.Path, "/api/runs/"), "/")
	parts := strings.Split(remainder, "/")
	if len(parts) == 1 && parts[0] != "" {
		if r.Method != http.MethodDelete {
			methodNotAllowed(w, http.MethodDelete)
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

func securityHeaders(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Security-Policy", "default-src 'self'; connect-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self'; base-uri 'none'; frame-ancestors 'none'")
		w.Header().Set("Referrer-Policy", "no-referrer")
		w.Header().Set("X-Content-Type-Options", "nosniff")
		next.ServeHTTP(w, r)
	})
}
