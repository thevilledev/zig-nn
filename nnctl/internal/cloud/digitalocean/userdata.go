package digitalocean

import _ "embed"

// DefaultUserDataScript installs the repository's pinned Zig toolchain on top
// of DigitalOcean's GPU-ready images. The image supplies CUDA or ROCm.
//
//go:embed bootstrap.sh
var DefaultUserDataScript string
