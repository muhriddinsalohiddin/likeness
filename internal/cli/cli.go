// Package cli holds what the likeness subcommands have in common: how they
// are called, how they complain about a wrong command line, and how they
// get at the engine and at secrets.
package cli

import (
	"context"
	"encoding/base64"
	"encoding/hex"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/muhriddinsalohiddin/likeness"
	"github.com/muhriddinsalohiddin/likeness/store"
)

// Env is the outside world as a command sees it. main fills it in from the
// process, tests with buffers.
type Env struct {
	Getenv func(string) string
	Stdout io.Writer
	Stderr io.Writer
}

// Command runs one subcommand with the arguments that follow its name. It
// returns nil, flag.ErrHelp once it has printed its usage, ErrUsage once it
// has explained what is wrong with the command line, or the error that
// stopped it. It returns promptly when ctx is done.
type Command func(ctx context.Context, args []string, env Env) error

// ErrUsage marks a command line that was wrong and has been explained on
// Env.Stderr. The process exits with status 2.
var ErrUsage = errors.New("usage")

// Flags returns the flag set of a command. name is the command as typed,
// "likeness serve"; usage is printed above the list of flags and footer
// below it.
func Flags(name string, env Env, usage, footer string) *flag.FlagSet {
	fs := flag.NewFlagSet(name, flag.ContinueOnError)
	fs.SetOutput(env.Stderr)
	fs.Usage = func() {
		say(env.Stderr, "%s", usage)
		fs.PrintDefaults()
		say(env.Stderr, "%s", footer)
	}
	return fs
}

// Parse reads the command line. A mistake has been described by the flag
// package when Parse returns ErrUsage; -h has printed the usage when it
// returns flag.ErrHelp.
func Parse(fs *flag.FlagSet, args []string) error {
	switch err := fs.Parse(args); {
	case err == nil:
		return nil
	case errors.Is(err, flag.ErrHelp):
		return err
	}
	return ErrUsage
}

// Usage says what is wrong with a command line and returns ErrUsage.
func Usage(fs *flag.FlagSet, format string, args ...any) error {
	say(fs.Output(), "%s: %s\nRun '%s -h' for the flags.\n", fs.Name(), fmt.Sprintf(format, args...), fs.Name())
	return ErrUsage
}

func say(w io.Writer, format string, args ...any) { _, _ = fmt.Fprintf(w, format, args...) }

// Engine is what the commands need of the face engine.
type Engine interface {
	Analyze(ctx context.Context, picture []byte, o likeness.Options) (*likeness.Result, error)
	Info() likeness.Info
	Close()
}

// OpenEngine loads the models. Tests put a stand-in here.
var OpenEngine = func(cfg likeness.Config) (Engine, error) {
	e, err := likeness.Open(cfg)
	if err != nil {
		return nil, err
	}
	return e, nil
}

// EngineFlags adds the flags that choose the models, --preset, --home and
// --threads, and returns where their values end up.
func EngineFlags(fs *flag.FlagSet) *likeness.Config {
	cfg := &likeness.Config{}
	fs.StringVar(&cfg.Preset, "preset", "default", "models to use: "+strings.Join(likeness.Presets(), ", "))
	fs.StringVar(&cfg.Home, "home", "", "`directory` with the runtime library and the models (default $LIKENESS_HOME)")
	fs.IntVar(&cfg.Threads, "threads", 0, "processor threads an analysis may use (default all)")
	return cfg
}

// Secret reads a secret from the environment variable name, or from the
// file that the variable name_FILE points to: the way container platforms
// hand secrets over.
func Secret(getenv func(string) string, name string) (string, error) {
	value, file := getenv(name), getenv(name+"_FILE")
	switch {
	case value != "" && file != "":
		return "", fmt.Errorf("%s and %s_FILE are both set; use one", name, name)
	case file != "":
		b, err := os.ReadFile(file)
		if err != nil {
			return "", fmt.Errorf("%s_FILE: %w", name, err)
		}
		return strings.TrimSpace(string(b)), nil
	}
	return value, nil
}

// StoreKeyVar names the variable that holds the key a store is encrypted
// with: 32 bytes, as 64 hex digits or in base64.
const StoreKeyVar = "LIKENESS_STORE_KEY"

// StoreKey returns the key the store is encrypted with, nil if none is
// set. The key is never repeated in an error.
func StoreKey(getenv func(string) string) ([]byte, error) {
	text, err := Secret(getenv, StoreKeyVar)
	if text = strings.TrimSpace(text); err != nil || text == "" {
		return nil, err
	}
	if len(text) == 2*store.KeySize {
		if key, err := hex.DecodeString(text); err == nil {
			return key, nil
		}
	}
	for _, enc := range []*base64.Encoding{base64.StdEncoding, base64.URLEncoding, base64.RawStdEncoding, base64.RawURLEncoding} {
		if key, err := enc.DecodeString(text); err == nil && len(key) == store.KeySize {
			return key, nil
		}
	}
	return nil, fmt.Errorf("%s has to be %d bytes, written as %d hex digits or in base64", StoreKeyVar, store.KeySize, 2*store.KeySize)
}

// Terminal reports whether w is a terminal, where a line of progress can be
// redrawn in place, as opposed to a file, a pipe or a buffer.
func Terminal(w io.Writer) bool {
	f, ok := w.(*os.File)
	if !ok {
		return false
	}
	st, err := f.Stat()
	return err == nil && st.Mode()&os.ModeCharDevice != 0
}
