// Package setup is the `likeness setup` command: it makes a machine ready
// to run a preset by installing the ONNX Runtime library and the preset's
// models where the engine looks for them.
package setup

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"time"

	"github.com/muhriddinsalohiddin/likeness"
	"github.com/muhriddinsalohiddin/likeness/internal/cli"
	"github.com/muhriddinsalohiddin/likeness/internal/home"
	"github.com/muhriddinsalohiddin/likeness/internal/models"
)

const usage = `usage: likeness setup [flags]

Makes this machine ready to run a preset: installs the ONNX Runtime library
and the preset's two models, checks every file against its SHA-256 and
loads the models once to see that they work. Files that are in place
already are left alone, so the command can be run again at any time.

flags:
`

const environment = `
environment:
  LIKENESS_HOME          where to install when --home is not given
  LIKENESS_ORT_LIBRARY   an ONNX Runtime library of your own; setup then installs none
  HTTPS_PROXY, NO_PROXY  the proxy the downloads go through

On a machine without internet access, 'likeness setup --list' names the
files to download elsewhere; copy them into a directory and pass it as --from.
`

const (
	homeVar    = "LIKENESS_HOME"
	libraryVar = "LIKENESS_ORT_LIBRARY"
)

// source is where the command gets what it installs. Run uses the
// catalogue of internal/models and the network; tests put small ones here.
type source struct {
	presets []models.Preset
	// runtime returns the ONNX Runtime library for this machine, if the
	// catalogue has one.
	runtime func() (models.Artifact, bool)
	client  *http.Client
	// wait pauses between two attempts at a download.
	wait func(context.Context, time.Duration) error
}

// Run is the setup command: it installs what the presets named in args
// need and loads them once, to see that they work.
func Run(ctx context.Context, args []string, env cli.Env) error {
	s := source{presets: models.Presets, runtime: models.HostRuntime, client: newClient(), wait: wait}
	return s.run(ctx, args, env)
}

func (s source) run(ctx context.Context, args []string, env cli.Env) error {
	var (
		names, dir, from       string
		accept, listOnly, load bool
	)
	fs := cli.Flags("likeness setup", env, usage, s.footer())
	fs.StringVar(&names, "preset", "default", "`names` of the presets to install, separated by commas")
	fs.StringVar(&dir, "home", "", "`directory` to install into (default $LIKENESS_HOME, or a likeness directory in the user's cache)")
	fs.BoolVar(&accept, "accept-licence", false, "confirm that models licensed for non-commercial research only will be used for nothing else")
	fs.StringVar(&from, "from", "", "take the files from this `directory` instead of downloading them")
	fs.BoolVar(&listOnly, "list", false, "print what the presets need and whether it is installed, and change nothing")
	fs.BoolVar(&load, "check", true, "load the models once they are installed, to see that they work")
	if err := cli.Parse(fs, args); err != nil {
		return err
	}
	if fs.NArg() > 0 {
		return cli.Usage(fs, "unexpected argument %q", fs.Arg(0))
	}
	presets, err := s.choose(names)
	if err != nil {
		return cli.Usage(fs, "%v", err)
	}
	if from != "" {
		if st, err := os.Stat(from); err != nil || !st.IsDir() {
			return cli.Usage(fs, "--from %s is not a directory", from)
		}
	}

	given := dir != ""
	if dir == "" {
		dir = env.Getenv(homeVar)
	}
	if dir == "" {
		dir = home.Dir()
	}
	if abs, err := filepath.Abs(dir); err == nil {
		dir = abs
	}

	var items []item
	// note says why no runtime library is among the items, when none is.
	note := ""
	library, known := s.runtime()
	switch own := env.Getenv(libraryVar); {
	case own != "":
		note = fmt.Sprintf("%s names %s, so setup installs none", libraryVar, own)
	case known:
		items = append(items, item{
			what:     "ONNX Runtime " + models.RuntimeVersion + " library",
			artifact: library,
			dst:      filepath.Join(dir, "lib", home.LibraryName()),
		})
	case listOnly:
		note = noRuntime().Error()
	default:
		return noRuntime()
	}
	items = append(items, modelItems(presets, dir)...)

	if listOnly {
		list(env.Stdout, dir, items, note)
		return nil
	}

	var restricted []string
	for _, p := range presets {
		if p.NonCommercial() && !accept {
			notice(env.Stderr, p)
			restricted = append(restricted, p.Name)
		}
	}
	if len(restricted) > 0 {
		return cli.Usage(fs, "--accept-licence is required for %s", strings.Join(restricted, ", "))
	}

	if note != "" {
		say(env.Stderr, "ONNX Runtime library: %s\n", note)
	}
	in := installer{client: s.client, attempts: attempts, wait: s.wait, log: env.Stderr, redraw: cli.Terminal(env.Stderr)}
	if from != "" {
		in.offline(from)
	}
	if err := in.install(ctx, items); err != nil {
		return err
	}

	for _, p := range presets {
		if !load {
			say(env.Stdout, "%s: installed in %s, not checked\n", p.Name, dir)
			continue
		}
		if err := check(env.Stdout, p.Name, dir); err != nil {
			return err
		}
	}
	if given {
		say(env.Stdout, "The other commands find these files with --home %s, or with %s set to it.\n", dir, homeVar)
	}
	return nil
}

func (s source) footer() string {
	var b strings.Builder
	b.WriteString("\npresets:\n")
	for _, p := range s.presets {
		fmt.Fprintf(&b, "  %-10s %s\n", p.Name, p.Description)
	}
	return b.String() + environment
}

// choose finds the presets that a list of names separated by commas asks
// for, each of them once.
func (s source) choose(list string) ([]models.Preset, error) {
	known := make([]string, len(s.presets))
	for i, p := range s.presets {
		known[i] = p.Name
	}
	var chosen []models.Preset
	for _, name := range strings.Split(list, ",") {
		name = strings.TrimSpace(name)
		i := slices.Index(known, name)
		switch {
		case name == "":
			return nil, fmt.Errorf("--preset takes one name, or several separated by commas: %s", strings.Join(known, ", "))
		case i < 0:
			return nil, fmt.Errorf("unknown preset %q; the presets are %s", name, strings.Join(known, ", "))
		case !slices.ContainsFunc(chosen, func(p models.Preset) bool { return p.Name == name }):
			chosen = append(chosen, s.presets[i])
		}
	}
	return chosen, nil
}

// noRuntime is the error of a platform that the catalogue has no runtime
// library for.
func noRuntime() error {
	return fmt.Errorf("the catalogue has no ONNX Runtime build for %s/%s: install the library yourself and set %s to its path",
		runtime.GOOS, runtime.GOARCH, libraryVar)
}

// modelItems lists the model files of the presets under dir, each once.
func modelItems(presets []models.Preset, dir string) []item {
	roles := [...]string{"detector", "embedder"}
	var items []item
	for _, p := range presets {
		for i, m := range []*models.Manifest{p.Detector, p.Embedder} {
			dst := filepath.Join(dir, "models", m.File)
			if slices.ContainsFunc(items, func(it item) bool { return it.dst == dst }) {
				continue
			}
			items = append(items, item{
				what:     roles[i] + " model " + m.Name,
				artifact: m.Artifact,
				dst:      dst,
				licence:  m.License + ", from " + m.Source,
			})
		}
	}
	return items
}

// notice tells who may use the models of a preset, and where they come
// from, before anything is fetched.
func notice(w io.Writer, p models.Preset) {
	say(w, "The %s preset uses models that are licensed for non-commercial research only:\n\n", p.Name)
	for _, m := range []*models.Manifest{p.Detector, p.Embedder} {
		if m.NonCommercial {
			say(w, "  %s: %s licence, from %s\n    %s\n", m.Name, m.License, m.Source, m.Artifact.URL)
		}
	}
	say(w, "\nPass --accept-licence if that is all you will use them for.\n\n")
}

// list prints what the presets need and what of it is there already.
func list(w io.Writer, dir string, items []item, note string) {
	say(w, "likeness home: %s\n", dir)
	if note != "" {
		say(w, "\nONNX Runtime library: %s\n", note)
	}
	var urls []string
	for _, it := range items {
		a := it.artifact
		say(w, "\n%s: %s\n", it.what, state(it))
		say(w, "  url      %s\n", a.URL)
		if a.Member != "" {
			say(w, "  member   %s\n", a.Member)
		}
		say(w, "  size     %d bytes (%s)\n", a.Size, size(a.Size))
		say(w, "  sha256   %s\n", a.SHA256)
		if it.licence != "" {
			say(w, "  licence  %s\n", it.licence)
		}
		say(w, "  goes to  %s\n", it.dst)
		if !slices.Contains(urls, a.URL) {
			urls = append(urls, a.URL)
		}
	}
	say(w, "\nThe size and the sha256 of an item are those of the installed file. Where a\n"+
		"member is named, that file is one out of the archive at the url.\n")
	say(w, "\nFor a machine without internet access, download on one that has it:\n\n")
	for _, u := range urls {
		say(w, "  curl -fLO %s\n", u)
	}
	say(w, "\nthen carry the files over in a directory and run setup with --from DIRECTORY.\n")
}

// state says in a word or two whether an item is in place.
func state(it item) string {
	switch err := models.Verify(it.dst, it.artifact); {
	case err == nil:
		return "installed"
	case errors.Is(err, os.ErrNotExist):
		return "missing"
	case errors.Is(err, models.ErrChecksum):
		return "damaged, setup will replace it"
	default:
		return "unreadable: " + err.Error()
	}
}

// check loads a preset the way the other commands will, and says what
// came up.
func check(w io.Writer, preset, dir string) error {
	eng, err := cli.OpenEngine(likeness.Config{Preset: preset, Home: dir})
	if err != nil {
		return fmt.Errorf("the %s preset is installed in %s but does not load: %w", preset, dir, err)
	}
	info := eng.Info()
	eng.Close()
	say(w, "%s: ready, loaded with ONNX Runtime %s, detector %s and embedder %s\n",
		preset, info.Runtime, info.Detector, info.Embedder)
	return nil
}
