package cli

import (
	"strings"
	"testing"
)

func TestMetalDoctorStatus(t *testing.T) {
	if got := metalDoctorStatus("darwin"); !strings.Contains(got, "-Dgpu=metal") {
		t.Fatalf("metalDoctorStatus(darwin) = %q", got)
	}
	if got := metalDoctorStatus("linux"); !strings.Contains(got, "requires macOS") {
		t.Fatalf("metalDoctorStatus(linux) = %q", got)
	}
}

func TestCUDADoctorStatus(t *testing.T) {
	exists := func(path string) bool { return path == "/opt/cuda" }

	if got := cudaDoctorStatus("darwin", "/opt/cuda", exists); !strings.Contains(got, "targets Linux") {
		t.Fatalf("cudaDoctorStatus(darwin) = %q", got)
	}
	if got := cudaDoctorStatus("linux", "/opt/cuda", exists); !strings.Contains(got, "found at /opt/cuda") {
		t.Fatalf("cudaDoctorStatus(linux found) = %q", got)
	}
	if got := cudaDoctorStatus("linux", "/missing", exists); !strings.Contains(got, "pass -Dcuda-path") {
		t.Fatalf("cudaDoctorStatus(linux missing) = %q", got)
	}
}

func TestROCmDoctorStatus(t *testing.T) {
	exists := func(path string) bool { return path == "/opt/rocm" }

	if got := rocmDoctorStatus("darwin", "/opt/rocm", exists); !strings.Contains(got, "targets Linux") {
		t.Fatalf("rocmDoctorStatus(darwin) = %q", got)
	}
	if got := rocmDoctorStatus("linux", "/opt/rocm", exists); !strings.Contains(got, "found at /opt/rocm") {
		t.Fatalf("rocmDoctorStatus(linux found) = %q", got)
	}
	if got := rocmDoctorStatus("linux", "/missing", exists); !strings.Contains(got, "pass -Drocm-path") {
		t.Fatalf("rocmDoctorStatus(linux missing) = %q", got)
	}
}
