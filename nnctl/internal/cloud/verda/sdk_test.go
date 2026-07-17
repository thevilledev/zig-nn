package verda

import (
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"testing"
	"time"
)

func TestNewSDKClientRejectsUnsafeBaseURLs(t *testing.T) {
	t.Parallel()
	credentials := Credentials{ClientID: "client", ClientSecret: "secret"}
	for _, baseURL := range []string{
		"api.verda.example/v1",
		"http://api.verda.example/v1",
		"http://localhost:8080/v1",
		"https://user:password@api.verda.example/v1",
		"https://api.verda.example/v1?target=other",
		"file:///tmp/verda",
	} {
		t.Run(baseURL, func(t *testing.T) {
			if _, err := NewSDKClient(credentials, ClientOptions{BaseURL: baseURL}); err == nil {
				t.Fatalf("NewSDKClient() accepted unsafe base URL %q", baseURL)
			}
		})
	}
}

func TestNewSDKClientHardensHTTPTransport(t *testing.T) {
	t.Parallel()
	original := &http.Client{Timeout: 12 * time.Second}
	client, err := NewSDKClient(
		Credentials{ClientID: "client", ClientSecret: "secret"},
		ClientOptions{BaseURL: " http://127.0.0.1:8080/v1/ ", HTTPClient: original},
	)
	if err != nil {
		t.Fatal(err)
	}
	if client.client.BaseURL != "http://127.0.0.1:8080/v1" {
		t.Fatalf("base URL = %q", client.client.BaseURL)
	}
	if client.client.HTTPClient == original {
		t.Fatal("NewSDKClient() mutated the caller's HTTP client")
	}
	if client.client.HTTPClient.Timeout != 12*time.Second {
		t.Fatalf("timeout = %s, want 12s", client.client.HTTPClient.Timeout)
	}
	redirect := &http.Request{}
	if err := client.client.HTTPClient.CheckRedirect(redirect, nil); !errors.Is(err, http.ErrUseLastResponse) {
		t.Fatalf("redirect error = %v, want %v", err, http.ErrUseLastResponse)
	}
	if original.CheckRedirect != nil {
		t.Fatal("caller's redirect policy was modified")
	}
}

func TestNewSDKClientAppliesDefaultHTTPTimeout(t *testing.T) {
	t.Parallel()
	client, err := NewSDKClient(
		Credentials{ClientID: "client", ClientSecret: "secret"},
		ClientOptions{},
	)
	if err != nil {
		t.Fatal(err)
	}
	if client.client.HTTPClient.Timeout != DefaultHTTPTimeout {
		t.Fatalf("timeout = %s, want %s", client.client.HTTPClient.Timeout, DefaultHTTPTimeout)
	}
	if !strings.HasPrefix(client.client.BaseURL, "https://") {
		t.Fatalf("default base URL = %q, want HTTPS", client.client.BaseURL)
	}
}

func TestCloneVolumeActionRequestUsesLocationCode(t *testing.T) {
	actionReq := cloneVolumeActionRequest(CloneVolumeRequest{
		SourceVolumeID: "vol-source",
		Name:           "vol-clone",
		LocationCode:   "FIN-01",
	})

	payloadBytes, err := json.Marshal(actionReq)
	if err != nil {
		t.Fatalf("marshal clone volume action request: %v", err)
	}

	var payload map[string]string
	if err := json.Unmarshal(payloadBytes, &payload); err != nil {
		t.Fatalf("unmarshal clone volume action request: %v", err)
	}

	if payload["id"] != "vol-source" {
		t.Fatalf("id = %q, want vol-source", payload["id"])
	}
	if payload["action"] != "clone" {
		t.Fatalf("action = %q, want clone", payload["action"])
	}
	if payload["name"] != "vol-clone" {
		t.Fatalf("name = %q, want vol-clone", payload["name"])
	}
	if payload["location_code"] != "FIN-01" {
		t.Fatalf("location_code = %q, want FIN-01", payload["location_code"])
	}
	if payload["type"] != "" {
		t.Fatalf("type = %q, want empty", payload["type"])
	}
}

func TestPermanentDeleteVolumeActionRequestUsesPermanentDelete(t *testing.T) {
	actionReq := permanentDeleteVolumeActionRequest(" vol-trash ")

	payloadBytes, err := json.Marshal(actionReq)
	if err != nil {
		t.Fatalf("marshal permanent delete volume action request: %v", err)
	}

	var payload map[string]any
	if err := json.Unmarshal(payloadBytes, &payload); err != nil {
		t.Fatalf("unmarshal permanent delete volume action request: %v", err)
	}

	if payload["id"] != "vol-trash" {
		t.Fatalf("id = %q, want vol-trash", payload["id"])
	}
	if payload["action"] != "delete" {
		t.Fatalf("action = %q, want delete", payload["action"])
	}
	if payload["is_permanent"] != true {
		t.Fatalf("is_permanent = %v, want true", payload["is_permanent"])
	}
}
