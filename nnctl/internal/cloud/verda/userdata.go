package verdacloud

import _ "embed"

// DefaultUserDataScript embeds bootstrap.sh so deployed benchmark workers get
// the same CUDA, Zig, and linker setup that is checked into this package.
//
//go:embed bootstrap.sh
var DefaultUserDataScript string
