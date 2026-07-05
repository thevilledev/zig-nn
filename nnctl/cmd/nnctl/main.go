package main

import (
	"os"

	"nnctl"
)

func main() {
	os.Exit(nnctl.Main(os.Args[1:]))
}
