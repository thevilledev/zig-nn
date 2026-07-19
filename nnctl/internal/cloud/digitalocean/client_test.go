package digitalocean

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestAPIClientUsesBearerAuthenticationAndV2Endpoints(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if request.Header.Get("Authorization") != "Bearer test-token" || request.Header.Get("User-Agent") != "nnctl-test" {
			t.Errorf("headers = %#v", request.Header)
		}
		writer.Header().Set("Content-Type", "application/json")
		switch request.URL.Path {
		case "/v2/sizes":
			if request.URL.Query().Get("per_page") != "200" {
				t.Errorf("sizes query = %s", request.URL.RawQuery)
			}
			_, _ = writer.Write([]byte(`{"sizes":[{"slug":"gpu-mi300x1-192gb","available":true,"regions":["tor1"],"gpu_info":{"count":1,"model":"amd_mi300x","vram":{"amount":192,"unit":"gib"}}}]}`))
		case "/v2/droplets":
			if request.Method == http.MethodPost {
				var create CreateDropletRequest
				if err := json.NewDecoder(request.Body).Decode(&create); err != nil {
					t.Error(err)
				}
				_, _ = writer.Write([]byte(`{"droplet":{"id":42,"status":"new","size_slug":"gpu-mi300x1-192gb"}}`))
				return
			}
			_, _ = writer.Write([]byte(`{"droplets":[]}`))
		case "/v2/droplets/42":
			if request.Method == http.MethodDelete {
				writer.WriteHeader(http.StatusNoContent)
				return
			}
			_, _ = writer.Write([]byte(`{"droplet":{"id":42,"status":"active","size_slug":"gpu-mi300x1-192gb"}}`))
		case "/v2/account/keys":
			_, _ = writer.Write([]byte(`{"ssh_keys":[{"id":17,"name":"workstation","fingerprint":"aa:bb"}]}`))
		default:
			http.NotFound(writer, request)
		}
	}))
	t.Cleanup(server.Close)
	client, err := NewClient("test-token", ClientOptions{BaseURL: server.URL + "/v2", UserAgent: "nnctl-test"})
	if err != nil {
		t.Fatal(err)
	}
	if sizes, err := client.ListSizes(t.Context()); err != nil || len(sizes) != 1 {
		t.Fatalf("sizes = %#v, %v", sizes, err)
	}
	if _, err := client.CreateDroplet(t.Context(), CreateDropletRequest{Name: "worker", Size: "gpu-mi300x1-192gb"}); err != nil {
		t.Fatal(err)
	}
	if droplet, err := client.GetDroplet(t.Context(), "42"); err != nil || droplet.Status != "active" {
		t.Fatalf("droplet = %#v, %v", droplet, err)
	}
	if keys, err := client.ListSSHKeys(t.Context()); err != nil || len(keys) != 1 {
		t.Fatalf("keys = %#v, %v", keys, err)
	}
	if err := client.DeleteDroplet(t.Context(), "42"); err != nil {
		t.Fatal(err)
	}
}

func TestAPIClientReturnsStructuredErrors(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "application/json")
		writer.WriteHeader(http.StatusUnauthorized)
		_, _ = writer.Write([]byte(`{"id":"unauthorized","message":"Unable to authenticate you.","request_id":"request-1"}`))
	}))
	t.Cleanup(server.Close)
	client, err := NewClient("test-token", ClientOptions{BaseURL: server.URL})
	if err != nil {
		t.Fatal(err)
	}
	_, err = client.ListDroplets(context.Background())
	if err == nil || !strings.Contains(err.Error(), "401") || !strings.Contains(err.Error(), "request-1") {
		t.Fatalf("error = %v", err)
	}
}
