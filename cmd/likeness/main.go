// Command likeness is the face recognition server and its tooling.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"runtime"
	"syscall"
	"text/tabwriter"

	"github.com/muhriddinsalohiddin/likeness"
	"github.com/muhriddinsalohiddin/likeness/internal/cli"
	"github.com/muhriddinsalohiddin/likeness/internal/cli/bench"
	"github.com/muhriddinsalohiddin/likeness/internal/cli/enrol"
	"github.com/muhriddinsalohiddin/likeness/internal/cli/eval"
	"github.com/muhriddinsalohiddin/likeness/internal/cli/load"
	"github.com/muhriddinsalohiddin/likeness/internal/cli/serve"
	"github.com/muhriddinsalohiddin/likeness/internal/cli/setup"
)

type command struct {
	name    string
	run     cli.Command
	summary string
}

// commands are listed in the order a new user meets them.
var commands = []command{
	{"setup", setup.Run, "install the runtime library and the models"},
	{"enrol", enrol.Run, "enrol many people from pictures on disk"},
	{"serve", serve.Run, "run the HTTP API"},
	{"eval", eval.Run, "measure accuracy on labelled pictures and work out thresholds"},
	{"bench", bench.Run, "measure how fast this machine analyses pictures"},
	{"load", load.Run, "send searches to a running server and report how it holds up"},
}

// aliases are other spellings people type.
var aliases = map[string]string{"enroll": "enrol"}

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	go func() {
		// The first signal asks a command to wind down; once it has been
		// taken, a second one ends the process the ordinary way.
		<-ctx.Done()
		stop()
	}()
	status := run(ctx, os.Args[1:], cli.Env{Getenv: os.Getenv, Stdout: os.Stdout, Stderr: os.Stderr})
	stop()
	os.Exit(status)
}

// run carries out one command line and returns the exit status: 0 for
// success, 1 for failure, 2 for a command line that makes no sense.
func run(ctx context.Context, args []string, env cli.Env) int {
	if len(args) == 0 {
		usage(env.Stderr)
		return 2
	}
	name, rest := args[0], args[1:]
	if spelt, ok := aliases[name]; ok {
		name = spelt
	}
	switch name {
	case "version", "-v", "--version":
		say(env.Stdout, "likeness %s (%s, %s/%s)\n", likeness.Version, runtime.Version(), runtime.GOOS, runtime.GOARCH)
		return 0
	case "help", "-h", "--help":
		usage(env.Stdout)
		return 0
	}
	for _, c := range commands {
		if c.name != name {
			continue
		}
		err := c.run(ctx, rest, env)
		switch {
		case err == nil, errors.Is(err, flag.ErrHelp):
			return 0
		case errors.Is(err, cli.ErrUsage):
			return 2
		case errors.Is(err, context.Canceled) && ctx.Err() != nil:
			return 130 // stopped by a signal, as a shell would report it
		}
		say(env.Stderr, "likeness %s: %v\n", name, err)
		return 1
	}
	say(env.Stderr, "likeness: unknown command %q\n\n", args[0])
	usage(env.Stderr)
	return 2
}

func usage(w io.Writer) {
	say(w, "usage: likeness <command> [flags]\n\ncommands:\n")
	tw := tabwriter.NewWriter(w, 0, 0, 3, ' ', 0)
	for _, c := range commands {
		say(tw, "  %s\t%s\n", c.name, c.summary)
	}
	say(tw, "  version\tprint the version\n")
	_ = tw.Flush()
	say(w, "\nRun 'likeness <command> -h' for the flags of a command.\n")
}

func say(w io.Writer, format string, args ...any) { _, _ = fmt.Fprintf(w, format, args...) }
