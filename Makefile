GO ?= go
NNCTL ?= bin/nnctl
PREK ?= npx --no-install prek

.PHONY: all build nnctl test run precommit hooks clean help

all: build

build: nnctl

nnctl:
	mkdir -p $(dir $(NNCTL))
	$(GO) build -C nnctl -o ../$(NNCTL) ./cmd/nnctl

test:
	$(GO) test ./nnctl/...

run:
	$(GO) run ./nnctl/cmd/nnctl --help

precommit:
	$(PREK) run --all-files

hooks:
	$(PREK) install -f

clean:
	rm -f $(NNCTL)

help:
	@printf "Usage: make [target]\n\n"
	@printf "Targets:\n"
	@printf "  build    Build ./bin/nnctl\n"
	@printf "  nnctl    Build ./bin/nnctl\n"
	@printf "  test     Run nnctl Go tests\n"
	@printf "  run      Show nnctl help via go run\n"
	@printf "  precommit Run prek hooks on all files\n"
	@printf "  hooks    Install prek git hooks\n"
	@printf "  clean    Remove ./bin/nnctl\n"
