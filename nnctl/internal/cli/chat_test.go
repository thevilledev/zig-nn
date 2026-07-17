package cli

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestChatProxyAddsDefaultsAndForwardsResponse(t *testing.T) {
	var upstreamPayload map[string]any
	client := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("unexpected upstream path: %s", r.URL.Path)
		}
		if r.Method != http.MethodPost {
			t.Fatalf("unexpected upstream method: %s", r.Method)
		}
		if err := json.NewDecoder(r.Body).Decode(&upstreamPayload); err != nil {
			t.Fatalf("decode upstream payload: %v", err)
		}
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body:       io.NopCloser(strings.NewReader(`{"choices":[{"message":{"content":"ok"}}]}`)),
		}, nil
	})}

	proxy := chatProxy{
		apiBaseURL:  "http://example.test",
		modelName:   "tiny-gpt-zig",
		maxTokens:   64,
		temperature: 0.7,
		client:      client,
	}

	request := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(`{"messages":[{"role":"user","content":"hi"}]}`))
	request.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()

	proxy.ServeHTTP(response, request)

	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
	}
	if got := upstreamPayload["model"]; got != "tiny-gpt-zig" {
		t.Fatalf("model = %#v", got)
	}
	if got := upstreamPayload["max_tokens"]; got != float64(64) {
		t.Fatalf("max_tokens = %#v", got)
	}
	if got := upstreamPayload["temperature"]; got != float64(0.7) {
		t.Fatalf("temperature = %#v", got)
	}
	if !strings.Contains(response.Body.String(), `"content":"ok"`) {
		t.Fatalf("unexpected response body: %s", response.Body.String())
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) {
	return f(r)
}

func TestChatProxyRejectsMissingMessages(t *testing.T) {
	proxy := chatProxy{apiBaseURL: "http://127.0.0.1:1"}
	request := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(`{}`))
	request.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()

	proxy.ServeHTTP(response, request)

	if response.Code != http.StatusBadRequest {
		t.Fatalf("status = %d", response.Code)
	}
}

func TestServeChatApp(t *testing.T) {
	response := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, "/", nil)

	serveChatApp(response, request)

	if response.Code != http.StatusOK {
		t.Fatalf("status = %d", response.Code)
	}
	if !strings.Contains(response.Body.String(), "zig-nn TinyGPT Chat") {
		t.Fatalf("chat page did not render expected title")
	}
	if !strings.Contains(response.Body.String(), "/app.js") {
		t.Fatalf("chat page does not load its script")
	}

	scriptResponse := httptest.NewRecorder()
	serveChatApp(scriptResponse, httptest.NewRequest(http.MethodGet, "/app.js", nil))
	if scriptResponse.Code != http.StatusOK || !strings.Contains(scriptResponse.Body.String(), "/api/chat") {
		t.Fatalf("chat script does not post to proxy endpoint")
	}
}

func TestChatProxyRejectsUnsafeRequests(t *testing.T) {
	t.Parallel()
	proxy := chatProxy{apiBaseURL: "http://127.0.0.1:1", maxTokens: 64, temperature: 1}
	tests := []struct {
		name        string
		body        string
		contentType string
		origin      string
		host        string
		want        int
	}{
		{name: "wrong content type", body: `{}`, contentType: "text/plain", want: http.StatusUnsupportedMediaType},
		{name: "unknown field", body: `{"messages":[{"role":"user","content":"hi"}],"stream":true}`, contentType: "application/json", want: http.StatusBadRequest},
		{name: "excess tokens", body: `{"messages":[{"role":"user","content":"hi"}],"max_tokens":513}`, contentType: "application/json", want: http.StatusBadRequest},
		{name: "invalid role", body: `{"messages":[{"role":"tool","content":"hi"}]}`, contentType: "application/json", want: http.StatusBadRequest},
		{name: "cross origin", body: `{"messages":[{"role":"user","content":"hi"}]}`, contentType: "application/json", origin: "https://attacker.example", host: "attacker.example", want: http.StatusForbidden},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			request := httptest.NewRequestWithContext(t.Context(), http.MethodPost, "/api/chat", strings.NewReader(test.body))
			request.Header.Set("Content-Type", test.contentType)
			request.Header.Set("Origin", test.origin)
			if test.host != "" {
				request.Host = test.host
			}
			response := httptest.NewRecorder()
			proxy.ServeHTTP(response, request)
			if response.Code != test.want {
				t.Fatalf("status = %d, want %d: %s", response.Code, test.want, response.Body.String())
			}
		})
	}
}

func TestValidateChatOptionsRequiresLoopback(t *testing.T) {
	t.Parallel()
	opts := defaultChatOptions()
	opts.allowUntrained = true
	opts.host = "0.0.0.0"
	if _, err := validateChatOptions(opts); err == nil || !strings.Contains(err.Error(), "loopback") {
		t.Fatalf("validateChatOptions() error = %v", err)
	}
	opts = defaultChatOptions()
	opts.allowUntrained = true
	opts.apiHost = "example.test"
	if _, err := validateChatOptions(opts); err == nil || !strings.Contains(err.Error(), "loopback") {
		t.Fatalf("validateChatOptions() error = %v", err)
	}
}
