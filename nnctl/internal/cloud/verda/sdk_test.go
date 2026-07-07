package verdacloud

import (
	"encoding/json"
	"testing"
)

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
