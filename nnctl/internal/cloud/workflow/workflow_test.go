package workflow

import (
	"strings"
	"testing"
)

func TestSSHDestination(t *testing.T) {
	t.Parallel()
	for _, test := range []struct {
		name    string
		user    string
		ip      string
		want    string
		wantErr string
	}{
		{name: "IPv4", user: "root", ip: "192.0.2.10", want: "root@192.0.2.10"},
		{name: "IPv6", user: "runner", ip: "2001:db8::10", want: "runner@2001:db8::10"},
		{name: "invalid user", user: "-oProxyCommand", ip: "192.0.2.10", wantErr: "SSH user"},
		{name: "host name", user: "root", ip: "worker.example", wantErr: "IP address"},
		{name: "option injection", user: "root", ip: "-oProxyCommand=evil", wantErr: "IP address"},
		{name: "loopback", user: "root", ip: "127.0.0.1", wantErr: "IP address"},
	} {
		t.Run(test.name, func(t *testing.T) {
			got, err := SSHDestination(test.user, test.ip)
			if test.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), test.wantErr) {
					t.Fatalf("SSHDestination() error = %v, want %q", err, test.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if got != test.want {
				t.Fatalf("SSHDestination() = %q, want %q", got, test.want)
			}
		})
	}
}
