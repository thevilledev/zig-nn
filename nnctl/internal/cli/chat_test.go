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
	if !strings.Contains(response.Body.String(), "/api/chat") {
		t.Fatalf("chat page does not post to proxy endpoint")
	}
}
