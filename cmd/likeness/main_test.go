package main

import (
	"bytes"
	"context"
	"errors"
	"flag"
	"strings"
	"testing"

	"github.com/muhriddinsalohiddin/likeness"
	"github.com/muhriddinsalohiddin/likeness/internal/cli"
)

// try runs a command line and returns the exit status and what was printed.
func try(t *testing.T, ctx context.Context, args ...string) (status int, stdout, stderr string) {
	t.Helper()
	var out, errs bytes.Buffer
	status = run(ctx, args, cli.Env{Getenv: func(string) string { return "" }, Stdout: &out, Stderr: &errs})
	return status, out.String(), errs.String()
}

func TestRun(t *testing.T) {
	ctx := context.Background()

	status, out, _ := try(t, ctx, "version")
	if status != 0 || !strings.HasPrefix(out, "likeness "+likeness.Version) {
		t.Errorf("version: %d %q", status, out)
	}

	status, out, _ = try(t, ctx, "help")
	if status != 0 || !strings.Contains(out, "usage: likeness <command>") {
		t.Errorf("help: %d %q", status, out)
	}
	for _, c := range commands {
		if !strings.Contains(out, "  "+c.name+" ") || c.summary == "" {
			t.Errorf("the help does not describe %q:\n%s", c.name, out)
		}
	}

	status, _, errs := try(t, ctx)
	if status != 2 || !strings.Contains(errs, "usage: likeness") {
		t.Errorf("no command: %d %q", status, errs)
	}
	status, _, errs = try(t, ctx, "fly")
	if status != 2 || !strings.Contains(errs, `unknown command "fly"`) {
		t.Errorf("an unknown command: %d %q", status, errs)
	}
}

func TestExitStatus(t *testing.T) {
	real := commands
	defer func() { commands = real }()
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()

	for _, c := range []struct {
		name   string
		ctx    context.Context
		err    error
		status int
		says   string
	}{
		{"fine", context.Background(), nil, 0, ""},
		{"asked for help", context.Background(), flag.ErrHelp, 0, ""},
		{"wrong command line", context.Background(), cli.ErrUsage, 2, ""},
		{"failed", context.Background(), errors.New("the disk is full"), 1, "likeness try: the disk is full\n"},
		{"interrupted", cancelled, context.Canceled, 130, ""},
		{"gave up by itself", context.Background(), context.Canceled, 1, "likeness try: context canceled\n"},
	} {
		commands = []command{{"try", func(context.Context, []string, cli.Env) error { return c.err }, "try something"}}
		status, _, errs := try(t, c.ctx, "try")
		if status != c.status || errs != c.says {
			t.Errorf("%s: status %d, stderr %q; want %d, %q", c.name, status, errs, c.status, c.says)
		}
	}
}

func TestCommandsGetTheirArguments(t *testing.T) {
	real, realAliases := commands, aliases
	defer func() { commands, aliases = real, realAliases }()
	var got []string
	commands = []command{{"try", func(_ context.Context, args []string, _ cli.Env) error { got = args; return nil }, "try something"}}
	aliases = map[string]string{"attempt": "try"}

	for _, name := range []string{"try", "attempt"} {
		got = nil
		if status, _, _ := try(t, context.Background(), name, "--fast", "now"); status != 0 || strings.Join(got, " ") != "--fast now" {
			t.Errorf("%s: status %d, arguments %q", name, status, got)
		}
	}
}

// Every real command explains itself and refuses a flag it does not have.
func TestEveryCommandHasHelp(t *testing.T) {
	for _, c := range commands {
		status, _, errs := try(t, context.Background(), c.name, "-h")
		if status != 0 || !strings.Contains(errs, "usage: likeness "+c.name) {
			t.Errorf("%s -h: status %d, %q", c.name, status, errs)
		}
		if status, _, _ := try(t, context.Background(), c.name, "--no-such-flag"); status != 2 {
			t.Errorf("%s with an unknown flag: status %d", c.name, status)
		}
	}
}
