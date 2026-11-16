// Package enrol is the likeness enrol command: it enrols many people at
// once, from pictures on disk straight into a store file.
//
// A picture is held to the rules the server applies to a face enrolled over
// HTTP, so a store seeded here is no different from one filled through the
// API.
package enrol

import (
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"strings"

	"github.com/muhriddinsalohiddin/likeness"
	"github.com/muhriddinsalohiddin/likeness/internal/cli"
	"github.com/muhriddinsalohiddin/likeness/internal/gallery"
	"github.com/muhriddinsalohiddin/likeness/store"
)

const usage = `usage: likeness enrol --store FILE [flags] DIR
       likeness enrol --store FILE [flags] --list FILE

Enrols many people at once, from pictures on disk straight into the store
that 'likeness serve' works on. Stop the server first: a store is held by
one process at a time.

DIR holds one sub-directory per person. The name of the sub-directory is the
subject id, and the jpg, jpeg, png and webp files in it are that person's
pictures, taken in the order of their names; other files, and names that
start with a dot, are ignored. With --list the pictures are named by a CSV
file of lines "subject,path" instead.

A picture has to pass the checks the API makes when a face is enrolled. One
that is enrolled already is left alone, so the same import can be run again.

flags:
`

const usageEnvironment = `
environment:
  LIKENESS_STORE_KEY     32 bytes in hex or base64: the key of an encrypted store
  LIKENESS_HOME          where the runtime library and the models are

LIKENESS_STORE_KEY can be read from a file instead, named by
LIKENESS_STORE_KEY_FILE.
`

// defaultMaxTemplates is the server's bound on the templates of a subject.
const defaultMaxTemplates = 10

type options struct {
	engine *likeness.Config
	store  string
	key    []byte
	// dir or list names the pictures; one of them is set.
	dir, list string
	report    string
	workers   int
	// maxTemplates is 0 for no bound, as the gallery wants it.
	maxTemplates int
	// search and verify are 0 where the model's own threshold applies.
	search, verify float32
	replace        bool
	allowDuplicate bool
	allowMismatch  bool
	dryRun         bool
}

// parse reads the command line and the environment.
func parse(args []string, env cli.Env) (*options, error) {
	o := &options{}
	var search, verify float64
	flags := cli.Flags("likeness enrol", env, usage, usageEnvironment)
	flags.StringVar(&o.store, "store", "", "`file` that keeps the enrolled faces; made if it does not exist (required)")
	flags.StringVar(&o.list, "list", "", "CSV `file` of lines subject,path; a path is relative to the directory of the file")
	o.engine = cli.EngineFlags(flags)
	flags.IntVar(&o.workers, "workers", 2, "pictures analysed at a time")
	flags.BoolVar(&o.replace, "replace", false, "drop the templates a subject had before: its first accepted picture takes their place")
	flags.BoolVar(&o.allowDuplicate, "allow-duplicate", false, "enrol a face that matches another subject: twins exist")
	flags.BoolVar(&o.allowMismatch, "allow-mismatch", false, "enrol a face that does not look like the subject's earlier pictures")
	flags.IntVar(&o.maxTemplates, "max-templates", 0, "templates a subject may hold, -1 for no bound (default 10)")
	flags.Float64Var(&search, "search-threshold", 0, "similarity from which a face is taken for another subject's (default the model's)")
	flags.Float64Var(&verify, "verify-threshold", 0, "similarity below which a face does not match its own subject (default the model's)")
	flags.BoolVar(&o.dryRun, "dry-run", false, "analyse and check everything, write nothing")
	flags.StringVar(&o.report, "report", "", "write a CSV `file` with a line per picture: what became of it and why")
	if err := cli.Parse(flags, args); err != nil {
		return nil, err
	}

	switch {
	case flags.NArg() > 1 && strings.HasPrefix(flags.Arg(1), "-"):
		return nil, cli.Usage(flags, "flags go before the directory: put %s in front of %s", flags.Arg(1), flags.Arg(0))
	case flags.NArg() > 1:
		return nil, cli.Usage(flags, "unexpected argument %q", flags.Arg(1))
	case o.store == "":
		return nil, cli.Usage(flags, "--store is required: the file that keeps the enrolled faces")
	case flags.NArg() == 0 && o.list == "":
		return nil, cli.Usage(flags, "name the directory that holds the pictures, or a list of them with --list")
	case flags.NArg() == 1 && o.list != "":
		return nil, cli.Usage(flags, "a directory and --list both name the pictures; use one")
	case o.workers < 1:
		return nil, cli.Usage(flags, "--workers is at least 1")
	case o.engine.Threads < 0:
		return nil, cli.Usage(flags, "--threads cannot be negative")
	case search < 0 || search > 1 || verify < 0 || verify > 1:
		return nil, cli.Usage(flags, "a threshold is a similarity: above 0 and at most 1")
	}
	o.dir = flags.Arg(0)
	o.search, o.verify = float32(search), float32(verify)
	switch {
	case o.maxTemplates == 0:
		o.maxTemplates = defaultMaxTemplates
	case o.maxTemplates < 0:
		o.maxTemplates = 0
	}

	key, err := cli.StoreKey(env.Getenv)
	if err != nil {
		return nil, err
	}
	o.key = key
	return o, nil
}

// rules are what an enrolment has to respect: the server's, with the
// engine's thresholds where the command line names none.
func (o *options) rules(info likeness.Info) gallery.EnrolOptions {
	rules := gallery.EnrolOptions{
		MaxTemplates:   o.maxTemplates,
		DuplicateAbove: info.SearchThreshold,
		MismatchBelow:  info.VerifyThreshold,
	}
	if o.search > 0 {
		rules.DuplicateAbove = o.search
	}
	if o.verify > 0 {
		rules.MismatchBelow = o.verify
	}
	if o.allowDuplicate {
		rules.DuplicateAbove = 0
	}
	if o.allowMismatch {
		rules.MismatchBelow = 0
	}
	return rules
}

// open opens the store as the server does. A dry run opens it read-only
// and does not make one that is missing: the gallery is then nil.
func (o *options) open(ctx context.Context, info likeness.Info) (*gallery.Gallery, error) {
	st, err := store.Open(o.store, store.Options{Dim: info.Dim, Fingerprint: info.Fingerprint, Key: o.key, ReadOnly: o.dryRun})
	if err != nil {
		if o.dryRun && errors.Is(err, fs.ErrNotExist) {
			return nil, nil
		}
		return nil, fmt.Errorf("%s: %w", o.store, describeStoreError(err, info))
	}
	faces, err := gallery.Open(ctx, st, gallery.Options{})
	if err != nil {
		_ = st.Close()
		return nil, fmt.Errorf("%s: %w", o.store, err)
	}
	return faces, nil
}

// describeStoreError says what to do about a store that would not open.
func describeStoreError(err error, info likeness.Info) error {
	switch {
	case errors.Is(err, store.ErrBusy):
		return fmt.Errorf("%w: stop the server first, a store is held by one process at a time", err)
	case errors.Is(err, store.ErrMismatch):
		return fmt.Errorf("%w; it cannot be used with the %s models (%s): "+
			"pick the preset it was made with, or another file", err, info.Preset, info.Fingerprint)
	case errors.Is(err, store.ErrKey):
		return fmt.Errorf("%w: check %s", err, cli.StoreKeyVar)
	}
	return err
}

// Run is the enrol command. It returns nil once it has gone through all the
// pictures, even if some were turned down, and an error if none of them is
// in the store at the end: a wrong directory should not look like success.
func Run(ctx context.Context, args []string, env cli.Env) (err error) {
	o, err := parse(args, env)
	if err != nil {
		return err
	}
	pictures, notes, err := o.gather()
	if err != nil {
		return err
	}
	remark(env.Stderr, notes)
	if len(pictures) == 0 {
		if o.list != "" {
			return fmt.Errorf("%s: no lines of the form subject,path", o.list)
		}
		return fmt.Errorf("%s: no pictures found: it has to hold one sub-directory per subject, "+
			"with that subject's jpg, jpeg, png or webp files in it", o.dir)
	}

	rep, err := createReport(o.report)
	if err != nil {
		return err
	}
	defer func() {
		if failed := rep.close(); err == nil && failed != nil {
			err = fmt.Errorf("%s: %w", o.report, failed)
		}
	}()

	eng, err := cli.OpenEngine(*o.engine)
	if err != nil {
		return err
	}
	defer eng.Close()
	info := eng.Info()

	faces, err := o.open(ctx, info)
	if err != nil {
		return err
	}
	if faces != nil {
		defer func() {
			if failed := faces.Close(); err == nil && failed != nil {
				err = fmt.Errorf("%s: %w", o.store, failed)
			}
		}()
	}

	b := &batch{
		engine: eng, faces: faces, rules: o.rules(info),
		replace: o.replace, dryRun: o.dryRun, workers: o.workers,
		report: rep, progress: newProgress(env.Stderr, len(pictures)),
		replaced: make(map[string]bool), added: make(map[string]int),
		count: make(map[outcome]int), rejects: make(map[string]int),
	}
	before := b.subjects()
	err = b.run(ctx, pictures)
	b.progress.finish()
	b.summarise(env.Stdout, len(pictures), before)
	switch {
	case err != nil:
		return err
	case b.count[enrolled]+b.count[wouldEnrol]+b.count[alreadyEnrolled] > 0:
		return nil
	case o.dryRun:
		return fmt.Errorf("none of the %d pictures would be enrolled", len(pictures))
	}
	return fmt.Errorf("none of the %d pictures was enrolled", len(pictures))
}

// summarise prints what became of the pictures.
func (b *batch) summarise(w io.Writer, total, before int) {
	row := func(label string, n int) { say(w, "%-18s %d\n", label, n) }
	say(w, "%-18s %d of %d\n", "pictures read", b.done, total)
	if b.dryRun {
		row(string(wouldEnrol), b.count[wouldEnrol])
	} else {
		row(string(enrolled), b.count[enrolled])
	}
	row(string(alreadyEnrolled), b.count[alreadyEnrolled])
	row(string(turnedDown), b.count[turnedDown])
	for _, reason := range sortedKeys(b.rejects) {
		row("  "+reason, b.rejects[reason])
	}
	row("duplicates", b.count[duplicate])
	row("mismatches", b.count[mismatch])
	row(string(full), b.count[full])
	row(string(unreadable), b.count[unreadable])
	for _, o := range []outcome{invalidSubject, invalidLine} {
		if n := b.count[o]; n > 0 {
			row(string(o), n)
		}
	}
	switch {
	case !b.dryRun:
		say(w, "%-18s %d before, %d after\n", "subjects in store", before, b.subjects())
	case b.faces == nil:
		say(w, "%-18s 0: the file does not exist, and a dry run does not make it\n", "subjects in store")
	default:
		say(w, "%-18s %d, as before: a dry run writes nothing\n", "subjects in store", before)
	}
	if b.dryRun {
		say(w, "The pictures were checked against the store, not against each other: a duplicate\n"+
			"or a mismatch between two pictures of this batch shows only in a real run.\n")
	}
	if b.done < total {
		say(w, "Stopped after %d of %d pictures. What is enrolled stays; the same command carries\n"+
			"on from there.\n", b.done, total)
	}
}

func say(w io.Writer, format string, args ...any) { _, _ = fmt.Fprintf(w, format, args...) }
