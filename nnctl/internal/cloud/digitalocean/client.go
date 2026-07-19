package digitalocean

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
)

const (
	DefaultBaseURL = "https://api.digitalocean.com/v2"
	pageSize       = 200
	maxResponse    = 8 << 20
)

type ClientOptions struct {
	BaseURL   string
	UserAgent string
	HTTP      HTTPDoer
}

type HTTPDoer interface {
	Do(*http.Request) (*http.Response, error)
}

type Client interface {
	ListSizes(context.Context) ([]Size, error)
	CreateDroplet(context.Context, CreateDropletRequest) (Droplet, error)
	GetDroplet(context.Context, string) (Droplet, error)
	ListDroplets(context.Context) ([]Droplet, error)
	DeleteDroplet(context.Context, string) error
	ListSSHKeys(context.Context) ([]SSHKey, error)
}

type APIClient struct {
	token     string
	baseURL   *url.URL
	userAgent string
	http      HTTPDoer
}

type Size struct {
	Slug         string   `json:"slug"`
	Memory       int      `json:"memory"`
	VCPUs        int      `json:"vcpus"`
	Disk         int      `json:"disk"`
	PriceHourly  float64  `json:"price_hourly"`
	PriceMonthly float64  `json:"price_monthly"`
	Regions      []string `json:"regions"`
	Available    bool     `json:"available"`
	Description  string   `json:"description"`
	GPUInfo      *GPUInfo `json:"gpu_info,omitempty"`
}

type GPUInfo struct {
	Count int       `json:"count"`
	Model string    `json:"model"`
	VRAM  UnitValue `json:"vram"`
}

type UnitValue struct {
	Amount int    `json:"amount"`
	Unit   string `json:"unit"`
}

type Droplet struct {
	ID       int      `json:"id"`
	Name     string   `json:"name"`
	Status   string   `json:"status"`
	SizeSlug string   `json:"size_slug"`
	Size     Size     `json:"size"`
	Region   Region   `json:"region"`
	Networks Networks `json:"networks"`
	Image    Image    `json:"image"`
	GPUInfo  *GPUInfo `json:"gpu_info,omitempty"`
}

type Region struct {
	Slug      string   `json:"slug"`
	Name      string   `json:"name"`
	Available bool     `json:"available"`
	Sizes     []string `json:"sizes"`
}

type Networks struct {
	V4 []Network `json:"v4"`
	V6 []Network `json:"v6"`
}

type Network struct {
	IPAddress string `json:"ip_address"`
	Type      string `json:"type"`
}

type Image struct {
	ID           int      `json:"id"`
	Name         string   `json:"name"`
	Slug         string   `json:"slug"`
	Distribution string   `json:"distribution"`
	Status       string   `json:"status"`
	Regions      []string `json:"regions"`
}

type SSHKey struct {
	ID          int    `json:"id"`
	Fingerprint string `json:"fingerprint"`
	PublicKey   string `json:"public_key,omitempty"`
	Name        string `json:"name"`
}

type CreateDropletRequest struct {
	Name             string   `json:"name"`
	Region           string   `json:"region"`
	Size             string   `json:"size"`
	Image            string   `json:"image"`
	SSHKeys          []int    `json:"ssh_keys,omitempty"`
	UserData         string   `json:"user_data,omitempty"`
	Backups          bool     `json:"backups"`
	IPv6             bool     `json:"ipv6"`
	Monitoring       bool     `json:"monitoring"`
	PublicNetworking bool     `json:"public_networking"`
	Tags             []string `json:"tags,omitempty"`
}

type APIError struct {
	StatusCode int
	ID         string `json:"id"`
	Message    string `json:"message"`
	RequestID  string `json:"request_id,omitempty"`
}

func (e *APIError) Error() string {
	message := strings.TrimSpace(e.Message)
	if message == "" {
		message = http.StatusText(e.StatusCode)
	}
	if e.RequestID != "" {
		return fmt.Sprintf("DigitalOcean API returned %d: %s (request %s)", e.StatusCode, message, e.RequestID)
	}
	return fmt.Sprintf("DigitalOcean API returned %d: %s", e.StatusCode, message)
}

func NewClient(token string, options ClientOptions) (*APIClient, error) {
	token = strings.TrimSpace(token)
	if token == "" {
		return nil, errors.New("DigitalOcean API token is required")
	}
	baseURL := strings.TrimSpace(options.BaseURL)
	if baseURL == "" {
		baseURL = DefaultBaseURL
	}
	parsed, err := url.Parse(baseURL)
	if err != nil {
		return nil, fmt.Errorf("parse DigitalOcean API base URL: %w", err)
	}
	if parsed.Scheme == "" || parsed.Host == "" || parsed.User != nil {
		return nil, errors.New("DigitalOcean API base URL must be an absolute HTTP(S) URL without user information")
	}
	parsed.Path = strings.TrimRight(parsed.Path, "/")
	parsed.RawQuery = ""
	parsed.Fragment = ""
	httpClient := options.HTTP
	if httpClient == nil {
		httpClient = http.DefaultClient
	}
	return &APIClient{
		token: token, baseURL: parsed, userAgent: strings.TrimSpace(options.UserAgent), http: httpClient,
	}, nil
}

func (c *APIClient) ListSizes(ctx context.Context) ([]Size, error) {
	var sizes []Size
	err := c.listPages(ctx, "/sizes", "sizes", func(data json.RawMessage) error {
		var page []Size
		if err := json.Unmarshal(data, &page); err != nil {
			return err
		}
		sizes = append(sizes, page...)
		return nil
	})
	return sizes, err
}

func (c *APIClient) CreateDroplet(ctx context.Context, request CreateDropletRequest) (Droplet, error) {
	var response struct {
		Droplet Droplet `json:"droplet"`
	}
	if err := c.doJSON(ctx, http.MethodPost, "/droplets", request, &response); err != nil {
		return Droplet{}, err
	}
	return response.Droplet, nil
}

func (c *APIClient) GetDroplet(ctx context.Context, id string) (Droplet, error) {
	var response struct {
		Droplet Droplet `json:"droplet"`
	}
	if err := c.doJSON(ctx, http.MethodGet, "/droplets/"+url.PathEscape(id), nil, &response); err != nil {
		return Droplet{}, err
	}
	return response.Droplet, nil
}

func (c *APIClient) ListDroplets(ctx context.Context) ([]Droplet, error) {
	var droplets []Droplet
	err := c.listPages(ctx, "/droplets", "droplets", func(data json.RawMessage) error {
		var page []Droplet
		if err := json.Unmarshal(data, &page); err != nil {
			return err
		}
		droplets = append(droplets, page...)
		return nil
	})
	return droplets, err
}

func (c *APIClient) DeleteDroplet(ctx context.Context, id string) error {
	return c.doJSON(ctx, http.MethodDelete, "/droplets/"+url.PathEscape(id), nil, nil)
}

func (c *APIClient) ListSSHKeys(ctx context.Context) ([]SSHKey, error) {
	var keys []SSHKey
	err := c.listPages(ctx, "/account/keys", "ssh_keys", func(data json.RawMessage) error {
		var page []SSHKey
		if err := json.Unmarshal(data, &page); err != nil {
			return err
		}
		keys = append(keys, page...)
		return nil
	})
	return keys, err
}

func (c *APIClient) listPages(
	ctx context.Context,
	path string,
	field string,
	appendPage func(json.RawMessage) error,
) error {
	for page := 1; ; page++ {
		separator := "?"
		if strings.Contains(path, "?") {
			separator = "&"
		}
		endpoint := path + separator + "page=" + strconv.Itoa(page) + "&per_page=" + strconv.Itoa(pageSize)
		var response map[string]json.RawMessage
		if err := c.doJSON(ctx, http.MethodGet, endpoint, nil, &response); err != nil {
			return err
		}
		data, ok := response[field]
		if !ok {
			return fmt.Errorf("DigitalOcean API response did not include %q", field)
		}
		var count []json.RawMessage
		if err := json.Unmarshal(data, &count); err != nil {
			return fmt.Errorf("decode DigitalOcean %s page: %w", field, err)
		}
		if err := appendPage(data); err != nil {
			return fmt.Errorf("decode DigitalOcean %s: %w", field, err)
		}
		if len(count) < pageSize {
			return nil
		}
	}
}

func (c *APIClient) doJSON(ctx context.Context, method, endpoint string, body, output any) error {
	var requestBody io.Reader
	if body != nil {
		encoded, err := json.Marshal(body)
		if err != nil {
			return fmt.Errorf("encode DigitalOcean API request: %w", err)
		}
		requestBody = bytes.NewReader(encoded)
	}
	target := *c.baseURL
	target.Path = strings.TrimRight(c.baseURL.Path, "/") + strings.SplitN(endpoint, "?", 2)[0]
	if parts := strings.SplitN(endpoint, "?", 2); len(parts) == 2 {
		target.RawQuery = parts[1]
	}
	request, err := http.NewRequestWithContext(ctx, method, target.String(), requestBody)
	if err != nil {
		return fmt.Errorf("create DigitalOcean API request: %w", err)
	}
	request.Header.Set("Accept", "application/json")
	request.Header.Set("Authorization", "Bearer "+c.token)
	if body != nil {
		request.Header.Set("Content-Type", "application/json")
	}
	if c.userAgent != "" {
		request.Header.Set("User-Agent", c.userAgent)
	}
	response, err := c.http.Do(request)
	if err != nil {
		return fmt.Errorf("call DigitalOcean API: %w", err)
	}
	defer func() { _ = response.Body.Close() }()
	payload, err := io.ReadAll(io.LimitReader(response.Body, maxResponse+1))
	if err != nil {
		return fmt.Errorf("read DigitalOcean API response: %w", err)
	}
	if len(payload) > maxResponse {
		return errors.New("DigitalOcean API response exceeded 8 MiB")
	}
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		apiErr := &APIError{StatusCode: response.StatusCode}
		_ = json.Unmarshal(payload, apiErr)
		return apiErr
	}
	if output == nil || len(bytes.TrimSpace(payload)) == 0 {
		return nil
	}
	if err := json.Unmarshal(payload, output); err != nil {
		return fmt.Errorf("decode DigitalOcean API response: %w", err)
	}
	return nil
}

var _ Client = (*APIClient)(nil)
