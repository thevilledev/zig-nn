package localhttp

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestIsLoopbackHost(t *testing.T) {
	t.Parallel()
	for _, test := range []struct {
		host string
		want bool
	}{
		{host: "localhost", want: true},
		{host: "127.0.0.1", want: true},
		{host: "::1", want: true},
		{host: "0.0.0.0", want: false},
		{host: "example.test", want: false},
	} {
		if got := IsLoopbackHost(test.host); got != test.want {
			t.Errorf("IsLoopbackHost(%q) = %t, want %t", test.host, got, test.want)
		}
	}
}

func TestRequireSameOrigin(t *testing.T) {
	t.Parallel()
	for _, test := range []struct {
		name   string
		origin string
		host   string
		want   bool
	}{
		{name: "local client", host: "127.0.0.1:8090", want: true},
		{name: "same origin", origin: "http://127.0.0.1:8090", host: "127.0.0.1:8090", want: true},
		{name: "other port", origin: "http://127.0.0.1:8080", host: "127.0.0.1:8090", want: false},
		{name: "remote origin", origin: "https://attacker.example", host: "attacker.example", want: false},
		{name: "origin with path", origin: "http://127.0.0.1:8090/path", host: "127.0.0.1:8090", want: false},
	} {
		t.Run(test.name, func(t *testing.T) {
			request := httptest.NewRequestWithContext(t.Context(), http.MethodPost, "/", nil)
			request.Host = test.host
			request.Header.Set("Origin", test.origin)
			response := httptest.NewRecorder()
			if got := RequireSameOrigin(response, request); got != test.want {
				t.Fatalf("RequireSameOrigin() = %t, want %t", got, test.want)
			}
			if !test.want && response.Code != http.StatusForbidden {
				t.Fatalf("status = %d, want %d", response.Code, http.StatusForbidden)
			}
		})
	}
}

func TestSecurityHeaders(t *testing.T) {
	t.Parallel()
	handler := SecurityHeaders(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	}))
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, httptest.NewRequestWithContext(t.Context(), http.MethodGet, "/", nil))
	for _, name := range []string{"Content-Security-Policy", "Referrer-Policy", "X-Content-Type-Options"} {
		if response.Header().Get(name) == "" {
			t.Errorf("missing %s", name)
		}
	}
}
