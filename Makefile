GO ?= go
FUZZTIME ?= 30s

.PHONY: build test cover lint fuzz bench tidy

build:
	$(GO) build -o bin/likeness ./cmd/likeness

test:
	$(GO) vet ./...
	$(GO) test -race -count=1 ./...

cover:
	$(GO) test -count=1 -coverprofile=coverage.out ./...
	$(GO) tool cover -func=coverage.out | tail -1

lint:
	golangci-lint run

fuzz:
	$(GO) test ./internal/vision -run '^$$' -fuzz FuzzDecode -fuzztime $(FUZZTIME)
	$(GO) test ./internal/vision -run '^$$' -fuzz FuzzExifOrientation -fuzztime $(FUZZTIME)

bench:
	$(GO) test ./... -run '^$$' -bench . -benchmem

tidy:
	$(GO) mod tidy
	gofmt -l -w cmd index internal *.go
