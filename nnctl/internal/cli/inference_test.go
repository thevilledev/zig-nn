package cli

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

func TestInspectCheckpointRecognizesDenseAndTinyGPT(t *testing.T) {
	dir := t.TempDir()
	dense := filepath.Join(dir, "dense.znn")
	var denseData bytes.Buffer
	denseData.WriteString("ZNN\x00")
	_ = binary.Write(&denseData, binary.LittleEndian, uint32(3))
	_ = binary.Write(&denseData, binary.LittleEndian, int64(0))
	_ = binary.Write(&denseData, binary.LittleEndian, int64(0))
	denseData.WriteByte(0)
	_ = binary.Write(&denseData, binary.LittleEndian, [3]uint32{2, 1, 3})
	if err := os.WriteFile(dense, denseData.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	info, err := inspectCheckpoint(dense)
	if err != nil {
		t.Fatal(err)
	}
	if info.Kind != checkpointDense || info.InputSize != 2 || info.OutputSize != 1 || info.Layers != 3 {
		t.Fatalf("unexpected dense info: %#v", info)
	}

	tiny := filepath.Join(dir, "tiny.tgpt")
	var tinyData bytes.Buffer
	tinyData.WriteString("TGPT")
	_ = binary.Write(&tinyData, binary.LittleEndian, uint32(3))
	_ = binary.Write(&tinyData, binary.LittleEndian, [5]uint32{64, 4, 4, 128, 96})
	if err := os.WriteFile(tiny, tinyData.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	info, err = inspectCheckpoint(tiny)
	if err != nil {
		t.Fatal(err)
	}
	if info.Kind != checkpointTinyGPT || info.BlockSize != 64 || info.Channels != 128 {
		t.Fatalf("unexpected TinyGPT info: %#v", info)
	}
}

func TestInspectCheckpointRejectsMalformedInput(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bad.bin")
	if err := os.WriteFile(path, []byte("NOPE"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := inspectCheckpoint(path); err == nil {
		t.Fatal("inspectCheckpoint() unexpectedly accepted malformed input")
	}
}

func TestInferenceBackendMapping(t *testing.T) {
	if err := validateInferenceBackend("metal"); err != nil {
		t.Fatal(err)
	}
	if err := validateInferenceBackend("bogus"); err == nil {
		t.Fatal("validateInferenceBackend() unexpectedly accepted bogus backend")
	}
	if buildBackend("cpu") != "none" || runtimeBackend("none") != "cpu" {
		t.Fatal("CPU backend mapping is inconsistent")
	}
}
