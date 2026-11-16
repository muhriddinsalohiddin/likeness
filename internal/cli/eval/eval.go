// Package eval is the `likeness eval` command: it embeds a labelled set of
// pictures, measures how well a preset tells the people in it apart, and
// works out the thresholds to run the server with.
package eval

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/muhriddinsalohiddin/likeness"
	"github.com/muhriddinsalohiddin/likeness/internal/cli"
	measure "github.com/muhriddinsalohiddin/likeness/internal/eval"
)

const usage = `usage: likeness eval [flags] DIR

Measures how well a preset tells people apart on your own pictures, and
works out the thresholds to run 'likeness serve' with. DIR holds one
directory per person with that person's pictures in it, the way LFW and
most face datasets are laid out:

  DIR/<person>/<picture>.jpg   (or .jpeg, .png, .webp)

flags:
`

const footer = `
Every picture is compared with every other one, as a verification compares
two, and the first picture of each person is enrolled in a gallery that the
other pictures are then searched in. One person in every --holdout is kept
out of the gallery, to see what a search makes of a stranger.

The right search threshold depends on how many people are enrolled: among
many, some stranger always looks a little like the person at the camera.
The last table of the report gives the threshold for each --gallery-size.

environment:
  LIKENESS_HOME   where the runtime library and the models are
`

// maxGallery bounds --gallery-size. Nothing measures a gallery this large;
// the bound keeps the arithmetic in range.
const maxGallery = 1_000_000_000

type options struct {
	dir    string
	engine *likeness.Config
	// policy is the name of the quality policy and analysis what it
	// stands for.
	policy   string
	analysis likeness.Options
	workers  int
	cache    string
	json     string
	holdout  int
	// galleries are the gallery sizes to work thresholds out for.
	galleries    sizes
	maxPeople    int
	maxPerPerson int
}

// sizes is a flag holding gallery sizes, written as 1000,10000.
type sizes []int

func (s *sizes) String() string {
	texts := make([]string, len(*s))
	for i, n := range *s {
		texts[i] = strconv.Itoa(n)
	}
	return strings.Join(texts, ",")
}

func (s *sizes) Set(text string) error {
	var list sizes
	for _, field := range strings.Split(text, ",") {
		n, err := strconv.Atoi(strings.TrimSpace(field))
		if err != nil || n < 1 || n > maxGallery {
			return fmt.Errorf("%q is not a gallery size; write a list of numbers of people like 1000,10000", field)
		}
		list = append(list, n)
	}
	*s = list
	return nil
}

// analysisFor returns the engine options a --policy stands for.
func analysisFor(policy string) (likeness.Options, bool) {
	switch policy {
	case "search":
		return likeness.ForSearch(), true
	case "enrol":
		// A dataset picture may have bystanders in it; they are no reason
		// to turn it down.
		o := likeness.ForEnrolment()
		o.Faces = likeness.Largest
		return o, true
	case "none":
		return likeness.Options{}, true
	}
	return likeness.Options{}, false
}

// parse reads the command line.
func parse(args []string, env cli.Env) (*options, error) {
	o := &options{galleries: sizes{1000, 10000, 100000}}
	fs := cli.Flags("likeness eval", env, usage, footer)
	o.engine = cli.EngineFlags(fs)
	fs.StringVar(&o.policy, "policy", "search", "quality a picture must have to count: search, enrol or none")
	fs.IntVar(&o.workers, "workers", 2, "pictures analysed at the same time")
	fs.StringVar(&o.cache, "cache", "", "`file` that keeps the vectors, so that the next run need not analyse the pictures again")
	fs.IntVar(&o.holdout, "holdout", 5, "keep every `n`th person out of the gallery, as strangers to search for; 0 keeps nobody out")
	fs.Var(&o.galleries, "gallery-size", "numbers of enrolled `people` to work a search threshold out for")
	fs.StringVar(&o.json, "json", "", "`file` to write the report to as JSON as well, with rates as fractions")
	fs.IntVar(&o.maxPeople, "max-people", 0, "take only the first `n` people, for a quick look (default all)")
	fs.IntVar(&o.maxPerPerson, "max-per-person", 0, "take only the first `n` pictures of a person (default all)")
	if err := cli.Parse(fs, args); err != nil {
		return nil, err
	}

	problem := ""
	analysis, valid := analysisFor(o.policy)
	switch {
	case fs.NArg() == 0:
		problem = "name the directory with the pictures: one directory per person in it"
	case fs.NArg() > 1 && strings.HasPrefix(fs.Arg(1), "-"):
		problem = fmt.Sprintf("flags go before the directory: move %s in front of %s", fs.Arg(1), fs.Arg(0))
	case fs.NArg() > 1:
		problem = fmt.Sprintf("unexpected argument %q: one directory at a time", fs.Arg(1))
	case fs.Arg(0) == "":
		problem = "the directory has an empty name"
	case !valid:
		problem = "--policy is search, enrol or none"
	case o.engine.Threads < 0:
		problem = "--threads cannot be negative"
	case o.workers < 1:
		problem = "--workers is at least 1"
	case o.holdout < 0:
		problem = "--holdout cannot be negative"
	case o.holdout == 1:
		problem = "--holdout 1 would keep everybody out of the gallery; use 2 or more, or 0 to keep nobody out"
	case o.maxPeople < 0, o.maxPerPerson < 0:
		problem = "--max-people and --max-per-person cannot be negative"
	case o.maxPeople == 1:
		problem = "--max-people 1 would leave nobody to tell that person from; use 2 or more"
	}
	if problem != "" {
		return nil, cli.Usage(fs, "%s", problem)
	}
	o.dir, o.analysis = fs.Arg(0), analysis
	return o, nil
}

// Run evaluates the pictures in a directory and prints the report. It is a
// cli.Command.
func Run(ctx context.Context, args []string, env cli.Env) (err error) {
	o, err := parse(args, env)
	if err != nil {
		return err
	}
	// What can go wrong cheaply goes wrong before the models are loaded
	// and the pictures analysed.
	set, err := scan(o.dir, o.maxPeople, o.maxPerPerson)
	if err != nil {
		return err
	}
	for _, out := range []struct{ flag, file string }{{"--cache", o.cache}, {"--json", o.json}} {
		if out.file == "" {
			continue
		}
		if at, err := os.Stat(filepath.Dir(out.file)); err != nil || !at.IsDir() {
			return fmt.Errorf("%s %s: the directory %s does not exist", out.flag, out.file, filepath.Dir(out.file))
		}
	}

	eng, err := cli.OpenEngine(*o.engine)
	if err != nil {
		return err
	}
	defer eng.Close()
	info := eng.Info()

	held, err := openCache(o.cache, basis{
		Fingerprint: info.Fingerprint,
		Detector:    info.Detector,
		Policy:      fmt.Sprintf("%s %+v", o.policy, o.analysis),
	}, env.Stderr)
	if err != nil {
		return err
	}
	outcomes, err := embed(ctx, eng, set.pictures(), o.analysis, o.workers, held, env.Stderr)
	// Whatever stopped the run, the pictures analysed so far need not be
	// analysed again. A cache that cannot be written does not hold the
	// report back, but the run does not pass for a success either.
	if unsaved := held.save(); unsaved != nil {
		unsaved = fmt.Errorf("the cache was not saved: %w", unsaved)
		defer func() {
			if err == nil {
				err = unsaved
				return
			}
			note(env.Stderr, "%v", unsaved)
		}()
	}
	if err != nil {
		return err
	}

	people, counts := set.gather(outcomes)
	if err := counts.enough(o.policy); err != nil {
		return err
	}
	scores, err := measure.Pairs(ctx, people, 0)
	if err != nil {
		return err
	}
	trial, err := measure.Identify(ctx, people, o.holdout, 0)
	if err != nil {
		return err
	}

	r := compile(o, info, counts, scores, trial)
	if _, err := env.Stdout.Write(r.text()); err != nil {
		return fmt.Errorf("writing the report: %w", err)
	}
	if o.json != "" {
		return r.writeJSON(o.json)
	}
	return nil
}
