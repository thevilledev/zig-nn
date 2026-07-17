// Package localhttp provides shared safeguards for nnctl's local web servers.
package localhttp

import (
	"net"
	"net/http"
	"net/url"
	"strings"
	"time"
)

const (
	ReadHeaderTimeout = 5 * time.Second
	IdleTimeout       = 60 * time.Second
	MaxHeaderBytes    = 32 * 1024
)

// IsLoopbackHost reports whether host names the local machine without DNS.
func IsLoopbackHost(host string) bool {
	host = strings.TrimSpace(host)
	if strings.EqualFold(host, "localhost") {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

// NewServer returns an HTTP server with conservative local-service defaults.
// WriteTimeout is intentionally unset because the learning lab streams SSE.
func NewServer(address string, handler http.Handler) *http.Server {
	return &http.Server{
		Addr:              address,
		Handler:           handler,
		ReadHeaderTimeout: ReadHeaderTimeout,
		IdleTimeout:       IdleTimeout,
		MaxHeaderBytes:    MaxHeaderBytes,
	}
}

// RequireSameOrigin rejects browser mutations originating outside the local
// service. Requests without Origin remain available to local CLI clients.
func RequireSameOrigin(w http.ResponseWriter, r *http.Request) bool {
	if SameOrigin(r) {
		return true
	}
	http.Error(w, "cross-origin request forbidden", http.StatusForbidden)
	return false
}

// SameOrigin reports whether a request has no browser Origin or has an exact
// loopback origin matching its Host header.
func SameOrigin(r *http.Request) bool {
	origin := strings.TrimSpace(r.Header.Get("Origin"))
	if origin == "" {
		return true
	}
	parsed, err := url.Parse(origin)
	if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") ||
		parsed.User != nil || parsed.Path != "" || parsed.RawQuery != "" || parsed.Fragment != "" ||
		!strings.EqualFold(parsed.Host, r.Host) || !IsLoopbackHost(parsed.Hostname()) {
		return false
	}
	return true
}

// SecurityHeaders applies browser protections shared by the lab and chat UI.
func SecurityHeaders(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Security-Policy", "default-src 'self'; connect-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self'; base-uri 'none'; frame-ancestors 'none'")
		w.Header().Set("Referrer-Policy", "no-referrer")
		w.Header().Set("X-Content-Type-Options", "nosniff")
		next.ServeHTTP(w, r)
	})
}
