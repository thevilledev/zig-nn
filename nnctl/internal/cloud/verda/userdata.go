package verda

import _ "embed"

// DefaultUserDataScript embeds packer/bootstrap.sh so deployed benchmark
// workers get the same CUDA, Zig, and linker setup used by Packer.
//
//go:embed packer/bootstrap.sh
var DefaultUserDataScript string
