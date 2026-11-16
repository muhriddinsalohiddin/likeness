package eval

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/muhriddinsalohiddin/likeness"
	"github.com/muhriddinsalohiddin/likeness/internal/cli"
	"github.com/muhriddinsalohiddin/likeness/internal/enginetest"
)

// engines are the stand-in engines a test's runs have opened, in order.
type engines struct {
	mu     sync.Mutex
	opened []*enginetest.Fake
	// gate, if set, holds the "slow:" pictures of every engine.
	gate chan struct{}
}

// calls is the number of pictures the engines have analysed between them.
func (e *engines) calls() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	n := 0
	for _, f := range e.opened {
		n += f.Calls()
	}
	return n
}

func (e *engines) allClosed() bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	for _, f := range e.opened {
		if !f.Closed() {
			return false
		}
	}
	return true
}

// useFake puts stand-in engines in place of the real one: a new one for
// every run, as each run closes its own.
func useFake(t *testing.T) *engines {
	t.Helper()
	e := &engines{}
	real := cli.OpenEngine
	cli.OpenEngine = func(likeness.Config) (cli.Engine, error) {
		e.mu.Lock()
		defer e.mu.Unlock()
		f := &enginetest.Fake{Gate: e.gate}
		e.opened = append(e.opened, f)
		return f, nil
	}
	t.Cleanup(func() { cli.OpenEngine = real })
	return e
}

func write(t *testing.T, file, text string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(file), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(file, []byte(text), 0o644); err != nil {
		t.Fatal(err)
	}
}

// crowd makes a dataset for the stand-in engine: people called p01, p02
// and so on, each with the given number of pictures, 1.jpg, 2.jpg and so
// on.
func crowd(t *testing.T, people, pictures int) string {
	t.Helper()
	dir := t.TempDir()
	for p := 1; p <= people; p++ {
		for k := 1; k <= pictures; k++ {
			name := fmt.Sprintf("p%02d", p)
			write(t, filepath.Join(dir, name, fmt.Sprintf("%d.jpg", k)), fmt.Sprintf("face:%s#%d", name, k))
		}
	}
	return dir
}

// eval runs the command and returns what it printed.
func eval(ctx context.Context, args ...string) (stdout, stderr string, err error) {
	var out, errs bytes.Buffer
	err = Run(ctx, args, cli.Env{Getenv: func(string) string { return "" }, Stdout: &out, Stderr: &errs})
	return out.String(), errs.String(), err
}

// evalJSON runs the command with --json and returns the report it wrote
// next to what it printed.
func evalJSON(t *testing.T, args ...string) (*report, string) {
	t.Helper()
	file := filepath.Join(t.TempDir(), "report.json")
	stdout, stderr, err := eval(context.Background(), append([]string{"--json", file}, args...)...)
	if err != nil {
		t.Fatalf("eval %v: %v\n%s", args, err, stderr)
	}
	doc, err := os.ReadFile(file)
	if err != nil {
		t.Fatal(err)
	}
	r := &report{}
	dec := json.NewDecoder(bytes.NewReader(doc))
	dec.DisallowUnknownFields()
	if err := dec.Decode(r); err != nil {
		t.Fatalf("the JSON report: %v\n%s", err, doc)
	}
	return r, stdout
}

func TestParse(t *testing.T) {
	env := cli.Env{Stdout: io.Discard, Stderr: io.Discard}
	o, err := parse([]string{"faces"}, env)
	if err != nil {
		t.Fatal(err)
	}
	if o.dir != "faces" || o.engine.Preset != "default" || o.policy != "search" || o.workers != 2 || o.cache != "" ||
		o.json != "" || o.holdout != 5 || o.galleries.String() != "1000,10000,100000" || o.maxPeople != 0 || o.maxPerPerson != 0 {
		t.Errorf("defaults: %+v", o)
	}
	if o.analysis != likeness.ForSearch() {
		t.Errorf("the search policy: %+v", o.analysis)
	}

	o, err = parse([]string{
		"-preset", "buffalo_s", "-home", "/opt/likeness", "-threads", "4", "-policy", "enrol", "-workers", "6",
		"-cache", "vectors.cache", "-holdout", "0", "-gallery-size", "500, 2000", "-json", "report.json",
		"-max-people", "100", "-max-per-person", "3", "/data/lfw",
	}, env)
	if err != nil {
		t.Fatal(err)
	}
	if o.dir != "/data/lfw" || o.engine.Preset != "buffalo_s" || o.engine.Home != "/opt/likeness" || o.engine.Threads != 4 ||
		o.policy != "enrol" || o.workers != 6 || o.cache != "vectors.cache" || o.json != "report.json" || o.holdout != 0 ||
		o.galleries.String() != "500,2000" || o.maxPeople != 100 || o.maxPerPerson != 3 {
		t.Errorf("flags: %+v", o)
	}
	// a reference picture is strict about quality, but a bystander in a
	// dataset picture is no reason to turn it down
	if o.analysis.Policy != likeness.EnrolmentPolicy() || o.analysis.Faces != likeness.Largest {
		t.Errorf("the enrolment policy: %+v", o.analysis)
	}

	o, err = parse([]string{"--policy", "none", "--", "-odd-name"}, env)
	if err != nil || o.dir != "-odd-name" || o.analysis != (likeness.Options{}) {
		t.Errorf("no policy, and a directory that looks like a flag: %+v %v", o, err)
	}
}

func TestParseRefuses(t *testing.T) {
	for name, c := range map[string]struct {
		args []string
		says string
	}{
		"no directory":       {nil, "name the directory with the pictures"},
		"an empty directory": {[]string{""}, "empty name"},
		"two directories":    {[]string{"faces", "more"}, `unexpected argument "more"`},
		"a flag at the end":  {[]string{"faces", "--cache", "c"}, "flags go before the directory: move --cache in front of faces"},
		"unknown flag":       {[]string{"--fast", "faces"}, "flag provided but not defined"},
		"policy":             {[]string{"--policy", "strict", "faces"}, "--policy is search, enrol or none"},
		"negative threads":   {[]string{"--threads", "-2", "faces"}, "--threads cannot be negative"},
		"no workers":         {[]string{"--workers", "0", "faces"}, "--workers is at least 1"},
		"negative holdout":   {[]string{"--holdout", "-1", "faces"}, "--holdout cannot be negative"},
		"everybody held out": {[]string{"--holdout", "1", "faces"}, "would keep everybody out of the gallery"},
		"gallery of nobody":  {[]string{"--gallery-size", "0", "faces"}, `"0" is not a gallery size`},
		"gallery in words":   {[]string{"--gallery-size", "1000,many", "faces"}, `"many" is not a gallery size`},
		"gallery too large":  {[]string{"--gallery-size", "5000000000", "faces"}, "is not a gallery size"},
		"an empty gallery":   {[]string{"--gallery-size", "1000,,2000", "faces"}, `"" is not a gallery size`},
		"negative people":    {[]string{"--max-people", "-1", "faces"}, "cannot be negative"},
		"negative pictures":  {[]string{"--max-per-person", "-1", "faces"}, "cannot be negative"},
		"one person":         {[]string{"--max-people", "1", "faces"}, "would leave nobody to tell that person from"},
	} {
		stdout, stderr, err := eval(context.Background(), c.args...)
		if !errors.Is(err, cli.ErrUsage) {
			t.Errorf("%s: %v", name, err)
		}
		if !strings.Contains(stderr, c.says) || stdout != "" {
			t.Errorf("%s: explained with %q, want %q", name, stderr, c.says)
		}
	}

	stdout, stderr, err := eval(context.Background(), "-h")
	if !errors.Is(err, flag.ErrHelp) || stdout != "" {
		t.Errorf("-h: %v", err)
	}
	for _, want := range []string{
		"usage: likeness eval [flags] DIR", "DIR/<person>/<picture>.jpg", "-cache file", "-gallery-size people",
		"(default 1000,10000,100000)", "-holdout n", "-policy", "-preset", "LIKENESS_HOME",
	} {
		if !strings.Contains(stderr, want) {
			t.Errorf("the help lacks %q:\n%s", want, stderr)
		}
	}
}

// evaluated is a dataset with a little of everything, and what a run over
// it has to count: 20 people with four good pictures each, one of whom has
// a subdirectory; five pictures that cannot be used; and one person with
// no usable picture at all.
func evaluated(t *testing.T) string {
	t.Helper()
	dir := crowd(t, 20, 4)
	write(t, filepath.Join(dir, "p01", "5.jpg"), "reject:blurry")
	write(t, filepath.Join(dir, "p02", "5.jpg"), "reject:no_face")
	write(t, filepath.Join(dir, "p02", "6.jpg"), "reject:blurry")
	write(t, filepath.Join(dir, "p03", "5.jpg"), "error:corrupt")
	write(t, filepath.Join(dir, "zed", "1.jpg"), "reject:blurry")
	if err := os.Rename(filepath.Join(dir, "p04", "4.jpg"), filepath.Join(dir, "p04", "4.JPEG")); err != nil {
		t.Fatal(err)
	}
	return dir
}

func TestEval(t *testing.T) {
	fakes := useFake(t)
	dir := evaluated(t)
	r, stdout := evalJSON(t, "--gallery-size", "1,2,1000", dir)
	t.Logf("the report:\n%s", stdout)

	if fakes.calls() != 85 || !fakes.allClosed() {
		t.Errorf("%d pictures analysed, want 85; engine closed: %v", fakes.calls(), fakes.allClosed())
	}
	if r.Version != likeness.Version || r.Preset != "fake" || r.Detector != "fake-detector" || r.Embedder != "fake-embedder" ||
		r.Fingerprint != "fake:embedder:0000000000000000" || r.Dataset != dir || r.Policy != "search" {
		t.Errorf("the header: %+v", r)
	}
	if r.People != 21 || r.Pictures != 85 || r.UsablePeople != 20 || r.Usable != 80 {
		t.Errorf("%d people with %d pictures, %d and %d usable; want 21, 85, 20 and 80", r.People, r.Pictures, r.UsablePeople, r.Usable)
	}
	if len(r.TurnedDown) != 2 || r.TurnedDown["blurry"] != 3 || r.TurnedDown["no_face"] != 1 {
		t.Errorf("turned down: %v, want blurry 3 and no_face 1", r.TurnedDown)
	}
	if len(r.Unreadable) != 1 || r.Unreadable["corrupt_image"] != 1 {
		t.Errorf("unreadable: %v, want corrupt_image 1", r.Unreadable)
	}

	// 80 pictures make 3160 pairs, six of them within each person
	v := r.Verification
	if v.Genuine.Pairs != 120 || v.Impostor.Pairs != 3040 {
		t.Fatalf("%d genuine and %d impostor pairs, want 120 and 3040", v.Genuine.Pairs, v.Impostor.Pairs)
	}
	// the stand-in's pictures of one person are about 0.8 alike, those of
	// two people not at all
	if v.Genuine.Mean < 0.7 || v.Genuine.Mean > 0.9 || v.Impostor.Mean < -0.05 || v.Impostor.Mean > 0.05 ||
		v.Genuine.StdDev <= 0 || v.Genuine.StdDev > 0.1 || v.Impostor.StdDev < 0.08 || v.Impostor.StdDev > 0.18 {
		t.Errorf("the distributions do not separate: genuine %+v, impostor %+v", v.Genuine, v.Impostor)
	}
	if v.EER != 0 || v.EERThreshold < 0.3 || v.EERThreshold > 0.75 {
		t.Errorf("an equal error rate of %v at %v, want none, between the two kinds of pair", v.EER, v.EERThreshold)
	}
	if v.Current != (operating{Threshold: 0.5}) {
		t.Errorf("at the preset's verify threshold: %+v, want no errors at 0.5", v.Current)
	}
	if len(v.Targets) != 5 {
		t.Fatalf("%d FMR targets, want 5", len(v.Targets))
	}
	for n, want := range []struct {
		fmr       float64
		needed    uint64
		supported bool
	}{{1e-2, 1000, true}, {1e-3, 10000, false}, {1e-4, 100000, false}, {1e-5, 1000000, false}, {1e-6, 10000000, false}} {
		got := v.Targets[n]
		if got.FMR != want.fmr || got.PairsNeeded != want.needed || got.Supported != want.supported ||
			(got.Threshold != nil) != want.supported || (got.FNMR != nil) != want.supported {
			t.Errorf("the FMR target %v: %+v", want.fmr, got)
		}
	}
	// 1% of 3040 impostor pairs may pass: the threshold sits in the upper
	// tail of the impostors and well below the genuine pairs
	if at := v.Targets[0]; *at.Threshold < 0.2 || *at.Threshold > 0.45 || *at.FNMR != 0 {
		t.Errorf("for an FMR of 1%%: threshold %v, FNMR %v", *at.Threshold, *at.FNMR)
	}

	// every fifth of the 20 usable people is kept out of the gallery
	i := r.Identification
	if i.Holdout != 5 || i.Gallery != 16 || i.LeftOut != 4 || i.Mated != 48 || i.NonMated != 16 {
		t.Errorf("identification: %+v, want a gallery of 16, 4 left out, 48 and 16 probes", i)
	}
	if i.Rank1 == nil || *i.Rank1 != 1 {
		t.Errorf("rank-1 hit rate: %v, want 1", i.Rank1)
	}
	if i.Current.Threshold != 0.7 || i.Current.FNIR == nil || i.Current.FPIR == nil || *i.Current.FPIR != 0 {
		t.Errorf("at the preset's search threshold: %+v", i.Current)
	}

	// galleries of 1 and 2 need 995 and 1990 impostor pairs for an FPIR of
	// 1%, which there are; nothing else asked about is within reach
	if len(r.Galleries) != 6 {
		t.Fatalf("%d gallery settings, want 6", len(r.Galleries))
	}
	for n, want := range []struct {
		gallery   int
		fpir      float64
		needed    uint64
		supported bool
	}{
		{1, 0.01, 995, true}, {1, 0.001, 9995, false}, {2, 0.01, 1990, true}, {2, 0.001, 19990, false},
		{1000, 0.01, 994992, false}, {1000, 0.001, 9995000, false},
	} {
		got := r.Galleries[n]
		if got.Gallery != want.gallery || got.FPIR != want.fpir || got.PairsNeeded != want.needed || got.Supported != want.supported ||
			(got.Threshold != nil) != want.supported || (got.FNMR != nil) != want.supported {
			t.Errorf("a gallery of %d at an FPIR of %v: %+v", want.gallery, want.fpir, got)
		}
	}
	one, two := r.Galleries[0], r.Galleries[2]
	if *one.Threshold > *two.Threshold || *two.Threshold > 0.6 {
		t.Errorf("thresholds %v for a gallery of 1 and %v for 2: a larger gallery needs a higher one", *one.Threshold, *two.Threshold)
	}
	// the searches are tried at the thresholds that were worked out
	if len(i.Thresholds) == 0 || len(i.Thresholds) > 2 || i.Thresholds[0].Threshold != *one.Threshold ||
		i.Thresholds[len(i.Thresholds)-1].Threshold != *two.Threshold {
		t.Errorf("searches tried at %+v, want at %v and %v", i.Thresholds, *one.Threshold, *two.Threshold)
	}

	// the advice is for the largest gallery and the strictest FMR that
	// the pictures can speak for
	a := r.Advice
	if a.Search == nil || a.Search.Gallery != 2 || a.Search.FPIR != 0.01 || *a.Search.Threshold != *two.Threshold {
		t.Errorf("the search advice: %+v", a.Search)
	}
	if a.Verify == nil || a.Verify.FMR != 1e-2 || *a.Verify.Threshold != *v.Targets[0].Threshold {
		t.Errorf("the verify advice: %+v", a.Verify)
	}

	// The text says what the JSON says: written out again from the JSON,
	// it is the same text.
	if again := string(r.text()); again != stdout {
		t.Errorf("the report printed and the one in the JSON differ.\nprinted:\n%s\nfrom the JSON:\n%s", stdout, again)
	}
	for _, want := range []string{
		"likeness " + likeness.Version + "\n",
		"fake: fake-detector and fake-embedder",
		"fake:embedder:0000000000000000",
		dir,
		"85, of 21 people",
		"80, of 20 people, under the search policy",
		"blurry 3, no_face 1",
		"corrupt_image 1",
		"Equal error rate: 0%, at the threshold " + fixed(v.EERThreshold),
		"The preset's verify threshold, 0.500: FMR 0%, FNMR 0%.",
		"not enough pairs: takes 10000, there are 3040",
		"a gallery of 16 people",
		"48 probes of enrolled people, 16 probes of the 4 people kept out of the gallery (--holdout 5)",
		"Rank-1 hit rate: 100%",
		"the preset's search threshold",
		"not enough pairs: takes 994992, there are 3040",
		"For a gallery of 2 people, use --search-threshold " + fixed(*two.Threshold) + ".",
		"For verification, use --verify-threshold " + fixed(*a.Verify.Threshold) + ": an FMR of 1%, the lowest these pictures",
		"These figures hold for pictures like the ones evaluated.",
	} {
		if !strings.Contains(stdout, want) {
			t.Errorf("the report lacks %q", want)
		}
	}
}

func TestEvalHoldout(t *testing.T) {
	useFake(t)
	// ten people: three pictures each, but the second has one and the
	// tenth two
	dir := crowd(t, 10, 3)
	for _, gone := range []string{"p02/2.jpg", "p02/3.jpg", "p10/3.jpg"} {
		if err := os.Remove(filepath.Join(dir, gone)); err != nil {
			t.Fatal(err)
		}
	}
	for _, c := range []struct {
		holdout                           string
		gallery, leftOut, mated, nonMated int
	}{
		// p05 and p10 are kept out: their 3 and 2 pictures are strangers,
		// and of the eight enrolled, seven have two pictures to spare
		{"5", 8, 2, 14, 5},
		// p03, p06 and p09 are kept out
		{"3", 7, 3, 11, 9},
		// every other person, p02 with its one picture among them
		{"2", 5, 5, 10, 12},
		{"0", 10, 0, 17, 0},
		{"11", 10, 0, 17, 0},
	} {
		r, stdout := evalJSON(t, "--holdout", c.holdout, dir)
		i := r.Identification
		if i.Gallery != c.gallery || i.LeftOut != c.leftOut || i.Mated != c.mated || i.NonMated != c.nonMated {
			t.Errorf("--holdout %s: a gallery of %d with %d left out, %d mated and %d non-mated probes; want %d, %d, %d and %d",
				c.holdout, i.Gallery, i.LeftOut, i.Mated, i.NonMated, c.gallery, c.leftOut, c.mated, c.nonMated)
		}
		// without strangers there is no FPIR to report
		if (i.Current.FPIR == nil) != (c.nonMated == 0) || i.Current.FNIR == nil {
			t.Errorf("--holdout %s: at the preset's threshold %+v", c.holdout, i.Current)
		}
		if c.holdout == "0" && (strings.Contains(stdout, "kept out") || !strings.Contains(stdout, "17 probes of enrolled people\n")) {
			t.Errorf("--holdout 0: the report speaks of people kept out:\n%s", stdout)
		}
	}
}

func TestEvalLimits(t *testing.T) {
	fakes := useFake(t)
	dir := crowd(t, 6, 5)
	for _, c := range []struct {
		args                    []string
		people, pictures, calls int
	}{
		{[]string{"--max-people", "4"}, 4, 20, 20},
		{[]string{"--max-per-person", "2"}, 6, 12, 12},
		{[]string{"--max-people", "3", "--max-per-person", "3"}, 3, 9, 9},
		{[]string{"--max-people", "50", "--max-per-person", "50"}, 6, 30, 30},
	} {
		before := fakes.calls()
		r, _ := evalJSON(t, append(c.args, dir)...)
		if r.People != c.people || r.Pictures != c.pictures || r.Usable != c.pictures || fakes.calls()-before != c.calls {
			t.Errorf("%v: %d people, %d pictures, %d analysed; want %d, %d, %d",
				c.args, r.People, r.Pictures, fakes.calls()-before, c.people, c.pictures, c.calls)
		}
	}

	// the first people and the first pictures, by name
	set, err := scan(dir, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	pictures := set.pictures()
	names := make([]string, 0, len(pictures))
	for _, p := range pictures {
		names = append(names, p.name)
	}
	if got := strings.Join(names, " "); got != "p01/1.jpg p01/2.jpg p02/1.jpg p02/2.jpg" {
		t.Errorf("the first two pictures of the first two people: %s", got)
	}
}

func TestEvalTooLittle(t *testing.T) {
	fakes := useFake(t)
	empty := t.TempDir()
	flat := t.TempDir()
	write(t, filepath.Join(flat, "anna.jpg"), "face:anna")
	write(t, filepath.Join(flat, "boris.jpg"), "face:boris")
	alone := t.TempDir()
	write(t, filepath.Join(alone, "anna", "1.jpg"), "face:anna#1")
	write(t, filepath.Join(alone, "anna", "2.jpg"), "face:anna#2")
	write(t, filepath.Join(alone, "notes", "readme.txt"), "not a picture")
	file := filepath.Join(t.TempDir(), "faces.jpg")
	write(t, file, "face:anna")

	for name, c := range map[string]struct {
		dir  string
		says []string
	}{
		"an empty directory":     {empty, []string{"no pictures in " + empty, "one directory per person", filepath.Join(empty, "<person>", "<picture>.jpg")}},
		"pictures side by side":  {flat, []string{"no pictures in " + flat, "one directory per person"}},
		"one person":             {alone, []string{"pictures of one person only, anna", "it takes two", "one directory per person"}},
		"a directory not there":  {filepath.Join(empty, "nowhere"), []string{"nowhere"}},
		"a file for a directory": {file, []string{"faces.jpg"}},
	} {
		stdout, _, err := eval(context.Background(), c.dir)
		if err == nil || errors.Is(err, cli.ErrUsage) || stdout != "" {
			t.Errorf("%s: %v", name, err)
			continue
		}
		for _, want := range c.says {
			if !strings.Contains(err.Error(), want) {
				t.Errorf("%s: %q does not say %q", name, err, want)
			}
		}
	}
	if _, _, err := eval(context.Background(), filepath.Join(empty, "nowhere")); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("a directory that is not there: %v", err)
	}
	if fakes.calls() != 0 || len(fakes.opened) != 0 {
		t.Errorf("the engine was opened %d times for directories with nothing to evaluate", len(fakes.opened))
	}

	// enough pictures, but not enough of them usable
	unusable := t.TempDir()
	write(t, filepath.Join(unusable, "anna", "1.jpg"), "face:anna#1")
	write(t, filepath.Join(unusable, "anna", "2.jpg"), "face:anna#2")
	write(t, filepath.Join(unusable, "boris", "1.jpg"), "reject:blurry")
	write(t, filepath.Join(unusable, "boris", "2.jpg"), "reject:too_dark")
	write(t, filepath.Join(unusable, "clara", "1.jpg"), "error:corrupt")
	_, _, err := eval(context.Background(), unusable)
	for _, want := range []string{
		"2 of the 5 pictures in " + unusable + " can be used, of 1 of the 3 people", "it takes two people",
		"turned down: blurry 1, too_dark 1; unreadable: corrupt_image 1", "--policy none takes every face",
	} {
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("one usable person: %v does not say %q", err, want)
		}
	}

	single := t.TempDir()
	write(t, filepath.Join(single, "anna", "1.jpg"), "face:anna")
	write(t, filepath.Join(single, "boris", "1.jpg"), "face:boris")
	write(t, filepath.Join(single, "boris", "2.jpg"), "reject:no_face")
	_, _, err = eval(context.Background(), "--policy", "none", single)
	for _, want := range []string{"nobody in " + single + " has two usable pictures", "several pictures", "turned down: no_face 1"} {
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("one picture each: %v does not say %q", err, want)
		}
	}
	if err != nil && strings.Contains(err.Error(), "--policy none takes") {
		t.Errorf("--policy none is advised to somebody who passed it: %v", err)
	}
	if !fakes.allClosed() || len(fakes.opened) != 2 {
		t.Errorf("%d engines opened, all closed: %v", len(fakes.opened), fakes.allClosed())
	}
}

func TestEvalFailures(t *testing.T) {
	dir := crowd(t, 3, 2)

	// no models
	real := cli.OpenEngine
	cli.OpenEngine = func(likeness.Config) (cli.Engine, error) { return nil, errors.New("no models here") }
	_, _, err := eval(context.Background(), dir)
	cli.OpenEngine = real
	if err == nil || !strings.Contains(err.Error(), "no models here") {
		t.Errorf("an engine that will not open: %v", err)
	}

	fakes := useFake(t)
	// files that have nowhere to go are found out before any work is done
	for _, flag := range []string{"--json", "--cache"} {
		_, _, err := eval(context.Background(), flag, filepath.Join(dir, "nowhere", "file"), dir)
		if err == nil || !strings.Contains(err.Error(), flag) || !strings.Contains(err.Error(), "does not exist") {
			t.Errorf("%s in a directory that is not there: %v", flag, err)
		}
	}
	if len(fakes.opened) != 0 {
		t.Error("the engine was opened although the output had nowhere to go")
	}

	// the report cannot be written
	err = Run(context.Background(), []string{dir}, cli.Env{Stdout: failing{}, Stderr: io.Discard})
	if err == nil || !strings.Contains(err.Error(), "writing the report") {
		t.Errorf("a report that cannot be written: %v", err)
	}
	// the JSON cannot be written: its name is taken by a directory
	_, _, err = eval(context.Background(), "--json", dir, dir)
	if err == nil {
		t.Error("a JSON report was written over a directory")
	}
	if !fakes.allClosed() || len(fakes.opened) != 2 {
		t.Errorf("%d engines opened, all closed: %v", len(fakes.opened), fakes.allClosed())
	}
}

// failing is a writer that cannot write.
type failing struct{}

func (failing) Write([]byte) (int, error) { return 0, errors.New("the pipe is closed") }

func TestEvalUnreadable(t *testing.T) {
	fakes := useFake(t)
	dir := crowd(t, 3, 3)
	cache := filepath.Join(t.TempDir(), "cache")
	// pictures the engine fails on for reasons of its own, and one of
	// each kind it cannot read
	for k := 1; k <= 12; k++ {
		write(t, filepath.Join(dir, "p01", fmt.Sprintf("broken-%02d.png", k)), fmt.Sprintf("error:fault %d", k))
	}
	write(t, filepath.Join(dir, "p02", "a.webp"), "error:unsupported")
	write(t, filepath.Join(dir, "p02", "b.webp"), "error:large")
	write(t, filepath.Join(dir, "p02", "c.webp"), "error:small")
	write(t, filepath.Join(dir, "p02", "d.webp"), "not a picture the stand-in knows")

	file := filepath.Join(t.TempDir(), "report.json")
	_, stderr, err := eval(context.Background(), "--cache", cache, "--json", file, dir)
	if err != nil {
		t.Fatal(err)
	}
	doc, _ := os.ReadFile(file)
	r := &report{}
	if err := json.Unmarshal(doc, r); err != nil {
		t.Fatal(err)
	}
	if r.Pictures != 25 || r.Usable != 9 || len(r.Unreadable) != 4 || r.Unreadable["engine_error"] != 12 ||
		r.Unreadable["unsupported_image"] != 2 || r.Unreadable["image_too_large"] != 1 || r.Unreadable["image_too_small"] != 1 {
		t.Errorf("%d pictures, %d usable, unreadable: %v", r.Pictures, r.Usable, r.Unreadable)
	}
	// the first ten are named with what went wrong, the rest counted
	if n := strings.Count(stderr, "likeness eval: p0"); n != 10 {
		t.Errorf("%d unreadable pictures named, want 10:\n%s", n, stderr)
	}
	for _, want := range []string{"likeness eval: p01/broken-01.png: enginetest: the engine broke: fault 1\n", "likeness eval: and 6 more pictures that could not be read\n"} {
		if !strings.Contains(stderr, want) {
			t.Errorf("stderr lacks %q:\n%s", want, stderr)
		}
	}

	// What is wrong with a picture is remembered; what went wrong in the
	// engine is tried again.
	before := fakes.calls()
	if _, _, err := eval(context.Background(), "--cache", cache, dir); err != nil {
		t.Fatal(err)
	}
	if again := fakes.calls() - before; again != 12 {
		t.Errorf("%d pictures analysed again, want the 12 the engine failed on", again)
	}

	if os.Getuid() == 0 {
		return // root reads what nobody may read
	}
	dir = crowd(t, 2, 2)
	locked := filepath.Join(dir, "p02", "locked.jpg")
	write(t, locked, "face:p02#9")
	if err := os.Chmod(locked, 0); err != nil {
		t.Fatal(err)
	}
	stdout, stderr, err := eval(context.Background(), dir)
	if err != nil || !strings.Contains(stdout, "unreadable_file 1") || !strings.Contains(stderr, "p02/locked.jpg: open ") {
		t.Errorf("a file that may not be read: %v\n%s\n%s", err, stdout, stderr)
	}
}

func TestEvalCancelled(t *testing.T) {
	fakes := useFake(t)
	fakes.gate = make(chan struct{})
	dir := t.TempDir()
	write(t, filepath.Join(dir, "anna", "1.jpg"), "face:anna#1")
	write(t, filepath.Join(dir, "anna", "2.jpg"), "face:anna#2")
	write(t, filepath.Join(dir, "boris", "1.jpg"), "face:boris#1")
	write(t, filepath.Join(dir, "boris", "2.jpg"), "slow:boris#2")
	write(t, filepath.Join(dir, "boris", "3.jpg"), "slow:boris#3")
	cache := filepath.Join(t.TempDir(), "cache")

	// already cancelled: nothing is analysed
	stopped, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, err := eval(stopped, "--cache", cache, dir); !errors.Is(err, context.Canceled) {
		t.Errorf("with a context already cancelled: %v", err)
	}
	if fakes.calls() != 0 {
		t.Errorf("%d pictures analysed after the context was cancelled", fakes.calls())
	}
	if _, err := os.Stat(cache); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("a cache with nothing in it was written: %v", err)
	}

	// cancelled while the fourth picture is being analysed
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	type result struct {
		stdout, stderr string
		err            error
	}
	done := make(chan result, 1)
	go func() {
		stdout, stderr, err := eval(ctx, "--workers", "1", "--cache", cache, dir)
		done <- result{stdout, stderr, err}
	}()
	for deadline := time.Now().Add(10 * time.Second); fakes.calls() < 4; time.Sleep(time.Millisecond) {
		if time.Now().After(deadline) {
			t.Fatal("the fourth picture was never reached")
		}
	}
	cancel()
	var got result
	select {
	case got = <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("eval did not stop when its context was cancelled")
	}
	if !errors.Is(got.err, context.Canceled) || got.stdout != "" {
		t.Errorf("cancelled half-way: %v, and a report of %d bytes", got.err, len(got.stdout))
	}
	if !strings.Contains(got.stderr, "stopped after 3 of 5 pictures") {
		t.Errorf("stderr does not say how far it came:\n%s", got.stderr)
	}
	if !fakes.allClosed() {
		t.Error("the engine was left open")
	}

	// what was analysed before the stop is in the cache: the next run
	// analyses the two pictures that were not
	close(fakes.gate)
	before := fakes.calls()
	stdout, stderr, err := eval(context.Background(), "--cache", cache, dir)
	if err != nil || !strings.Contains(stdout, "5, of 2 people") {
		t.Fatalf("the run after: %v\n%s\n%s", err, stdout, stderr)
	}
	if again := fakes.calls() - before; again != 2 {
		t.Errorf("%d pictures analysed after the stop, want 2", again)
	}
	if !strings.Contains(stderr, "2 pictures analysed in ") || !strings.Contains(stderr, "; 3 more from the cache") {
		t.Errorf("stderr does not say where the pictures came from:\n%s", stderr)
	}
}
