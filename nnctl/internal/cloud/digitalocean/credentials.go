package digitalocean

import (
	"context"
	"fmt"
	"strings"

	"github.com/zalando/go-keyring"
)

const (
	KeyringService      = "nnctl/digitalocean"
	KeyringTokenAccount = "token"
)

type CredentialStore interface {
	Token(context.Context) (string, error)
}

type KeyringCredentialStore struct {
	Service string
}

func (s KeyringCredentialStore) Token(ctx context.Context) (string, error) {
	_ = ctx
	service := strings.TrimSpace(s.Service)
	if service == "" {
		service = KeyringService
	}
	token, err := keyring.Get(service, KeyringTokenAccount)
	if err != nil {
		return "", fmt.Errorf(
			"read DigitalOcean token from keyring service %q account %q: %w",
			service,
			KeyringTokenAccount,
			err,
		)
	}
	token = strings.TrimSpace(token)
	if token == "" {
		return "", fmt.Errorf("DigitalOcean token in keyring service %q must not be empty", service)
	}
	return token, nil
}
