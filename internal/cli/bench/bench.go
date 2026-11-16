// Package bench is the `likeness bench` command: it measures, on the
// machine it runs on, how long each stage of a request takes and how many
// pictures can be analysed at once before the latency suffers.
package bench

import (
	"context"
	"fmt"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/muhriddinsalohiddin/likeness"
	"github.com/muhriddinsalohiddin/likeness/internal/cli"
)

const usage = `usage: likeness bench [flags]

Measures how long a search request takes on this machine, stage by stage,
and how many pictures it can analyse at once before the latency suffers:
what to set 'likeness serve --max-inflight' to.

flags:
`

const footer = `
Three things are measured, each for --duration: one picture at a time,
searches in a gallery of random faces, and several pictures at once. With
the defaults that takes about half a minute. The report goes to standard
output, notes on progress to standard error.

environment:
  LIKENESS_HOME   where the runtime library and the models are
`

const (
	// maxGallery is the largest gallery --gallery accepts.
	maxGallery = 1_000_000
	// maxInflight is the highest level --inflight accepts; levels above
	// the number of processors are left out anyway.
	maxInflight = 4096
)

// numCPU counts the processors of the machine. Tests put another machine
// here.
var numCPU = runtime.NumCPU

type options struct {
	engine *likeness.Config
	image  string
	json   bool
	plan   plan
}

// counts is a flag holding whole numbers separated by commas, 1,2,4,8, each
// from 1 to most. It keeps them in ascending order, each once.
type counts struct {
	list []int
	most int
}

func (c *counts) String() string {
	texts := make([]string, len(c.list))
	for i, n := range c.list {
		texts[i] = strconv.Itoa(n)
	}
	return strings.Join(texts, ",")
}

func (c *counts) Set(text string) error {
	fields := strings.Split(text, ",")
	list := make([]int, 0, len(fields))
	for _, field := range fields {
		n, err := strconv.Atoi(strings.TrimSpace(field))
		if err != nil || n < 1 || n > c.most {
			return fmt.Errorf("%q is not a whole number from 1 to %d; write a list like 1,2,4", field, c.most)
		}
		list = append(list, n)
	}
	slices.Sort(list)
	c.list = slices.Compact(list)
	return nil
}

// parse reads the command line. Levels of --inflight that the machine has
// no processors for are left out, with a note on env.Stderr.
func parse(args []string, env cli.Env) (*options, error) {
	o := &options{}
	galleries := counts{list: []int{10_000}, most: maxGallery}
	inflight := counts{list: []int{1, 2, 4, 8}, most: maxInflight}
	fs := cli.Flags("likeness bench", env, usage, footer)
	o.engine = cli.EngineFlags(fs)
	fs.StringVar(&o.image, "image", "", "picture `file` to analyse (default a built-in 1280x720 frame with one face)")
	fs.DurationVar(&o.plan.duration, "duration", 5*time.Second, "how long each measurement runs")
	fs.Var(&galleries, "gallery", "`subjects` in the gallery that is searched, a million at most; 1000,10000,100000 measures several sizes")
	fs.Var(&inflight, "inflight", "`pictures` analysed at once: the levels to try")
	fs.DurationVar(&o.plan.target, "target", 100*time.Millisecond, "longest that 99 pictures in 100 may take, analysis and search")
	fs.BoolVar(&o.json, "json", false, "print the report as one JSON document, with durations in milliseconds")
	if err := cli.Parse(fs, args); err != nil {
		return nil, err
	}

	switch {
	case fs.NArg() > 0:
		return nil, cli.Usage(fs, "unexpected argument %q", fs.Arg(0))
	case o.engine.Threads < 0:
		return nil, cli.Usage(fs, "--threads cannot be negative")
	case o.plan.duration <= 0 || o.plan.target <= 0:
		return nil, cli.Usage(fs, "--duration and --target have to be longer than nothing: 5s, 100ms")
	}
	o.plan.galleries = galleries.list

	// The levels are in ascending order: those the machine has processors
	// for come first.
	cpus := numCPU()
	fits, _ := slices.BinarySearch(inflight.list, cpus+1)
	if fits == 0 {
		return nil, cli.Usage(fs, "this machine has %d processors: every level of --inflight %s is beyond it", cpus, &inflight)
	}
	if beyond := inflight.list[fits:]; len(beyond) > 0 {
		note(env, "leaving out --inflight %s: this machine has %d processors", &counts{list: beyond}, cpus)
	}
	o.plan.inflight = inflight.list[:fits]
	return o, nil
}

// note tells the operator what is happening. It writes to standard error,
// so that standard output holds the report and nothing else.
func note(env cli.Env, format string, args ...any) {
	_, _ = fmt.Fprintf(env.Stderr, "likeness bench: "+format+"\n", args...)
}

// Run measures and prints the report. It is a cli.Command.
func Run(ctx context.Context, args []string, env cli.Env) error {
	o, err := parse(args, env)
	if err != nil {
		return err
	}
	pic, err := load(o.image)
	if err != nil {
		return err
	}

	eng, err := cli.OpenEngine(*o.engine)
	if err != nil {
		return err
	}
	defer eng.Close()

	rep, err := measure(ctx, eng, pic, o.plan, func(format string, args ...any) { note(env, format, args...) })
	if err != nil {
		return err
	}
	return rep.write(env.Stdout, o.json)
}
