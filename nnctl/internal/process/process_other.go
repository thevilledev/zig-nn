//go:build !unix

package process

import "os/exec"

func configureTreeCancellation(_ *exec.Cmd) {}
