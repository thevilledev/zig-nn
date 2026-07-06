package verdacloud

import (
	"context"
	"fmt"
	"strings"

	"github.com/zalando/go-keyring"
)

const (
	KeyringService             = "nnctl/verda"
	KeyringClientIDAccount     = "client_id"
	KeyringClientSecretAccount = "client_secret"
)

type Credentials struct {
	ClientID     string
	ClientSecret string
}

type CredentialStore interface {
	Credentials(context.Context) (Credentials, error)
}

type KeyringCredentialStore struct {
	Service string
}

func (s KeyringCredentialStore) Credentials(ctx context.Context) (Credentials, error) {
	_ = ctx
	service := strings.TrimSpace(s.Service)
	if service == "" {
		service = KeyringService
	}

	clientID, err := keyring.Get(service, KeyringClientIDAccount)
	if err != nil {
		return Credentials{}, fmt.Errorf("read Verda client id from keyring service %q account %q: %w", service, KeyringClientIDAccount, err)
	}
	clientSecret, err := keyring.Get(service, KeyringClientSecretAccount)
	if err != nil {
		return Credentials{}, fmt.Errorf("read Verda client secret from keyring service %q account %q: %w", service, KeyringClientSecretAccount, err)
	}

	creds := Credentials{
		ClientID:     strings.TrimSpace(clientID),
		ClientSecret: strings.TrimSpace(clientSecret),
	}
	if creds.ClientID == "" || creds.ClientSecret == "" {
		return Credentials{}, fmt.Errorf("Verda credentials in keyring service %q must not be empty", service)
	}
	return creds, nil
}
