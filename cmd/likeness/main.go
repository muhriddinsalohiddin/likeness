// Command likeness is the face recognition server and its tooling.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"runtime"
	"syscall"

	"github.com/muhriddinsalohiddin/likeness"
)

const usage = `usage: likeness <command>

commands:
  serve     run the HTTP API; see 'likeness serve -h'
  version   print the version and exit
`

func main() {
	if len(os.Args) < 2 {
		fmt.Fprint(os.Stderr, usage)
		os.Exit(2)
	}
	switch os.Args[1] {
	case "serve":
		os.Exit(runServe(os.Args[2:]))
	case "version", "-v", "--version":
		fmt.Printf("likeness %s (%s, %s/%s)\n", likeness.Version, runtime.Version(), runtime.GOOS, runtime.GOARCH)
	case "help", "-h", "--help":
		fmt.Print(usage)
	default:
		fmt.Fprintf(os.Stderr, "likeness: unknown command %q\n\n%s", os.Args[1], usage)
		os.Exit(2)
	}
}

// runServe serves until the process is told to stop, and returns its exit
// status.
func runServe(args []string) int {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	go func() {
		// Once the first signal has started a graceful stop, a second one
		// ends the process the ordinary way.
		<-ctx.Done()
		stop()
	}()

	err := serve(ctx, args, os.Getenv, os.Stderr, nil)
	switch {
	case err == nil, errors.Is(err, flag.ErrHelp):
		return 0
	case errors.Is(err, errUsage):
		return 2
	}
	fmt.Fprintln(os.Stderr, "likeness serve:", err)
	return 1
}
