GO ?= go
NNCTL ?= bin/nnctl

.PHONY: all build nnctl test run clean help

all: build

build: nnctl

nnctl:
	mkdir -p $(dir $(NNCTL))
	$(GO) build -C nnctl -o ../$(NNCTL) ./cmd/nnctl

test:
	$(GO) test ./nnctl/...

run:
	$(GO) run ./nnctl/cmd/nnctl --help

clean:
	rm -f $(NNCTL)

help:
	@printf "Usage: make [target]\n\n"
	@printf "Targets:\n"
	@printf "  build    Build ./bin/nnctl\n"
	@printf "  nnctl    Build ./bin/nnctl\n"
	@printf "  test     Run nnctl Go tests\n"
	@printf "  run      Show nnctl help via go run\n"
	@printf "  clean    Remove ./bin/nnctl\n"
