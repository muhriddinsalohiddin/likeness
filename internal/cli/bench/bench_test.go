package bench

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/muhriddinsalohiddin/likeness"
	"github.com/muhriddinsalohiddin/likeness/index"
	"github.com/muhriddinsalohiddin/likeness/internal/cli"
	"github.com/muhriddinsalohiddin/likeness/internal/enginetest"
)

// The tests run against a stand-in engine whose "pictures" are lines of
// text (see enginetest), so they need no models. It reports the same stage
// times for every picture: decode 2 ms, detect 3 ms, align 1 ms, embed 4 ms.

// stand is the stand-in engine of a test and what the command opened it
// with.
type stand struct {
	*enginetest.Fake
	opened int
	cfg    likeness.Config
}

// useFake puts fake in place of the real engine.
func useFake(t *testing.T, fake *enginetest.Fake) *stand {
	t.Helper()
	s := &stand{Fake: fake}
	useEngine(t, func(cfg likeness.Config) (cli.Engine, error) {
		s.opened++
		s.cfg = cfg
		return fake, nil
	})
	return s
}

func useEngine(t *testing.T, open func(likeness.Config) (cli.Engine, error)) {
	t.Helper()
	real := cli.OpenEngine
	cli.OpenEngine = open
	t.Cleanup(func() { cli.OpenEngine = real })
}

// onMachine makes the tests see a machine with the given number of
// processors, whatever they run on.
func onMachine(t *testing.T, cpus int) {
	t.Helper()
	real := numCPU
	numCPU = func() int { return cpus }
	t.Cleanup(func() { numCPU = real })
}

// pictureFile writes a picture for the stand-in engine and returns its
// path.
func pictureFile(t *testing.T, text string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "picture.txt")
	if err := os.WriteFile(path, []byte(text), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

// run calls the command and returns what it printed.
func run(ctx context.Context, args ...string) (stdout, stderr string, err error) {
	var out, notes bytes.Buffer
	err = Run(ctx, args, cli.Env{Getenv: func(string) string { return "" }, Stdout: &out, Stderr: &notes})
	return out.String(), notes.String(), err
}

// quick are the flags of a run that is over in a moment.
func quick(file string, more ...string) []string {
	return append([]string{"--image", file, "--duration", "20ms", "--gallery", "100,300", "--inflight", "1,2", "--target", "10s"}, more...)
}

// row returns the cells of the line of a table that starts with first.
func row(t *testing.T, table, first string) []string {
	t.Helper()
	for _, line := range strings.Split(table, "\n") {
		if cells := strings.Fields(line); len(cells) > 0 && cells[0] == first {
			return cells
		}
	}
	t.Fatalf("no line starts with %q in:\n%s", first, table)
	return nil
}

func TestTextReport(t *testing.T) {
	engine := useFake(t, &enginetest.Fake{})
	onMachine(t, 4)
	file := pictureFile(t, "face:anna")

	stdout, stderr, err := run(context.Background(), quick(file, "--preset", "buffalo_s", "--home", "/opt/likeness", "--threads", "3")...)
	if err != nil {
		t.Fatalf("%v\n%s", err, stderr)
	}
	if want := (likeness.Config{Preset: "buffalo_s", Home: "/opt/likeness", Threads: 3}); engine.cfg != want {
		t.Errorf("the engine was opened with %+v, want %+v", engine.cfg, want)
	}
	if !engine.Closed() {
		t.Error("the engine was left open")
	}

	sections := strings.Split(strings.TrimSpace(stdout), "\n\n")
	if len(sections) != 5 {
		t.Fatalf("the report has %d sections, want a header, three tables and a recommendation:\n%s", len(sections), stdout)
	}
	header, stages, search, levels, advice := sections[0], sections[1], sections[2], sections[3], sections[4]

	for _, want := range []string{
		"likeness " + likeness.Version, "fake: fake-detector and fake-embedder", "vectors of 64 values",
		"ONNX Runtime 0.0.0, 1 thread\n", runtime.GOOS + "/" + runtime.GOARCH, "4 processors", runtime.Version(),
		file + ", 1280x720, 9 bytes",
	} {
		if !strings.Contains(header, want) {
			t.Errorf("the header lacks %q:\n%s", want, header)
		}
	}

	if !strings.HasPrefix(stages, "One picture at a time: ") || !strings.Contains(stages, " in 20ms\n") {
		t.Errorf("the first table is headed:\n%s", stages)
	}
	if got := strings.Join(row(t, stages, "stage"), "|"); got != "stage|p50|ms|p90|ms|p99|ms|max|ms" {
		t.Errorf("the columns of the stages: %s", got)
	}
	for name, ms := range map[string]string{"decode": "2.00", "detect": "3.00", "align": "1.00", "embed": "4.00"} {
		want := strings.Join([]string{name, ms, ms, ms, ms}, " ")
		if got := strings.Join(row(t, stages, name), " "); got != want {
			t.Errorf("stage %s: %q, want %q", name, got, want)
		}
	}
	if total := row(t, stages, "total"); len(total) != 5 || number(t, total[1]) > number(t, total[4]) {
		t.Errorf("the total: %q", total)
	}

	if !strings.HasPrefix(search, "Search: ") {
		t.Errorf("the second table is headed:\n%s", search)
	}
	for _, gallery := range []string{"100", "300"} {
		cells := row(t, search, gallery)
		if len(cells) != 5 || number(t, cells[1]) < 1 || number(t, cells[2]) > number(t, cells[3]) || cells[4] != index.Kernel() {
			t.Errorf("searching %s subjects: %q", gallery, cells)
		}
	}

	if !strings.HasPrefix(levels, "Several at once: ") {
		t.Errorf("the third table is headed:\n%s", levels)
	}
	for _, level := range []string{"1", "2"} {
		// the largest gallery is the one searched
		cells := row(t, levels, level)
		if len(cells) != 6 || cells[1] != "300" || number(t, cells[2]) < 1 || number(t, cells[3]) <= 0 || number(t, cells[4]) > number(t, cells[5]) {
			t.Errorf("%s at once: %q", level, cells)
		}
	}

	if !strings.HasPrefix(advice, "Recommendation: run the server with --max-inflight 2\n") ||
		!strings.Contains(advice, "within the 10000 ms asked for") {
		t.Errorf("the recommendation:\n%s", advice)
	}

	// what is happening is told on the side, not in the report
	for _, want := range []string{"warming up", "one picture at a time for 20ms", "searching a gallery of 100\n", "searching a gallery of 300\n", "2 at once for 20ms"} {
		if !strings.Contains(stderr, want) {
			t.Errorf("standard error lacks %q:\n%s", want, stderr)
		}
		if strings.Contains(stdout, want) {
			t.Errorf("the report has %q in it", want)
		}
	}
}

func number(t *testing.T, text string) float64 {
	t.Helper()
	n, err := strconv.ParseFloat(text, 64)
	if err != nil {
		t.Fatalf("%q is not a number", text)
	}
	return n
}

func TestJSONReport(t *testing.T) {
	engine := useFake(t, &enginetest.Fake{})
	onMachine(t, 4)
	file := pictureFile(t, "face:anna")

	stdout, stderr, err := run(context.Background(), quick(file, "--json")...)
	if err != nil {
		t.Fatalf("%v\n%s", err, stderr)
	}
	if !engine.Closed() {
		t.Error("the engine was left open")
	}
	if !strings.Contains(stderr, "warming up") {
		t.Errorf("standard error: %q", stderr)
	}

	type spread struct {
		P50 float64 `json:"p50_ms"`
		P90 float64 `json:"p90_ms"`
		P99 float64 `json:"p99_ms"`
		Max float64 `json:"max_ms"`
	}
	var got struct {
		Version string `json:"version"`
		Engine  struct {
			Preset   string `json:"preset"`
			Detector string `json:"detector"`
			Embedder string `json:"embedder"`
			Dim      int    `json:"dim"`
			Runtime  string `json:"runtime"`
			Threads  int    `json:"threads"`
		} `json:"engine"`
		Machine struct {
			OS   string `json:"os"`
			Arch string `json:"arch"`
			CPUs int    `json:"cpus"`
			Go   string `json:"go"`
		} `json:"machine"`
		Picture struct {
			Name   string `json:"name"`
			Width  int    `json:"width"`
			Height int    `json:"height"`
			Bytes  int    `json:"bytes"`
		} `json:"picture"`
		Duration float64 `json:"duration_ms"`
		Stages   struct {
			Pictures int    `json:"pictures"`
			Decode   spread `json:"decode"`
			Detect   spread `json:"detect"`
			Align    spread `json:"align"`
			Embed    spread `json:"embed"`
			Total    spread `json:"total"`
		} `json:"stages"`
		Search []struct {
			Gallery  int     `json:"gallery"`
			Searches int     `json:"searches"`
			Kernel   string  `json:"kernel"`
			P50      float64 `json:"p50_ms"`
			P99      float64 `json:"p99_ms"`
		} `json:"search"`
		Inflight []struct {
			Inflight  int     `json:"inflight"`
			Gallery   int     `json:"gallery"`
			Pictures  int     `json:"pictures"`
			PerSecond float64 `json:"pictures_per_second"`
			P50       float64 `json:"p50_ms"`
			P99       float64 `json:"p99_ms"`
		} `json:"inflight"`
		Recommendation struct {
			Target      float64 `json:"target_ms"`
			MaxInflight int     `json:"max_inflight"`
			PerSecond   float64 `json:"pictures_per_second"`
			P99         float64 `json:"p99_ms"`
		} `json:"recommendation"`
	}
	// one document, and no field in it that is not named above
	dec := json.NewDecoder(strings.NewReader(stdout))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&got); err != nil {
		t.Fatalf("%v\n%s", err, stdout)
	}
	if dec.More() {
		t.Errorf("something follows the document:\n%s", stdout)
	}

	if got.Version != likeness.Version || got.Engine.Preset != "fake" || got.Engine.Detector != "fake-detector" ||
		got.Engine.Embedder != "fake-embedder" || got.Engine.Dim != enginetest.Dim || got.Engine.Runtime != "0.0.0" || got.Engine.Threads != 1 {
		t.Errorf("the engine: %+v", got.Engine)
	}
	if m := got.Machine; m.OS != runtime.GOOS || m.Arch != runtime.GOARCH || m.CPUs != 4 || m.Go != runtime.Version() {
		t.Errorf("the machine: %+v", m)
	}
	if p := got.Picture; p.Name != file || p.Width != 1280 || p.Height != 720 || p.Bytes != len("face:anna") {
		t.Errorf("the picture: %+v", p)
	}
	if got.Duration != 20 {
		t.Errorf("the duration: %v ms, want 20", got.Duration)
	}

	s := got.Stages
	for name, c := range map[string]struct {
		got  spread
		want float64
	}{"decode": {s.Decode, 2}, "detect": {s.Detect, 3}, "align": {s.Align, 1}, "embed": {s.Embed, 4}} {
		if c.got != (spread{c.want, c.want, c.want, c.want}) {
			t.Errorf("stage %s: %+v, want %v ms throughout", name, c.got, c.want)
		}
	}
	if s.Pictures < 1 || s.Total.P50 <= 0 || s.Total.P50 > s.Total.P90 || s.Total.P90 > s.Total.P99 || s.Total.P99 > s.Total.Max {
		t.Errorf("the total of %d pictures: %+v", s.Pictures, s.Total)
	}

	if len(got.Search) != 2 || got.Search[0].Gallery != 100 || got.Search[1].Gallery != 300 {
		t.Fatalf("the galleries searched: %+v", got.Search)
	}
	for _, g := range got.Search {
		if g.Searches < 1 || g.Searches > maxSearches || g.Kernel != index.Kernel() || g.P50 <= 0 || g.P50 > g.P99 {
			t.Errorf("searching %d subjects: %+v", g.Gallery, g)
		}
	}

	if len(got.Inflight) != 2 || got.Inflight[0].Inflight != 1 || got.Inflight[1].Inflight != 2 {
		t.Fatalf("the levels: %+v", got.Inflight)
	}
	for _, l := range got.Inflight {
		if l.Gallery != 300 || l.Pictures < l.Inflight || l.PerSecond <= 0 || l.P50 <= 0 || l.P50 > l.P99 {
			t.Errorf("%d at once: %+v", l.Inflight, l)
		}
	}
	top := got.Inflight[1]
	if r := got.Recommendation; r.Target != 10000 || r.MaxInflight != 2 || r.PerSecond != top.PerSecond || r.P99 != top.P99 {
		t.Errorf("the recommendation: %+v, want the level %+v", r, top)
	}
}

func TestRecommendationFollowsTheTarget(t *testing.T) {
	onMachine(t, 8)
	file := pictureFile(t, "slow:anna")
	args := []string{"--image", file, "--duration", "30ms", "--gallery", "50", "--inflight", "3"}

	// a picture takes 3 ms at the least: far inside one target, outside the other
	engine := useFake(t, &enginetest.Fake{Delay: 3 * time.Millisecond})
	stdout, stderr, err := run(context.Background(), append(args, "--target", "10s")...)
	if err != nil {
		t.Fatalf("%v\n%s", err, stderr)
	}
	if !strings.Contains(stdout, "run the server with --max-inflight 3\n") {
		t.Errorf("a target that is easy to keep:\n%s", stdout)
	}
	if engine.Peak() != 3 {
		t.Errorf("%d pictures were analysed at once, want 3", engine.Peak())
	}

	useFake(t, &enginetest.Fake{Delay: 3 * time.Millisecond})
	stdout, stderr, err = run(context.Background(), append(args, "--target", "1ms")...)
	if err != nil {
		t.Fatalf("%v\n%s", err, stderr)
	}
	if !strings.Contains(stdout, "Recommendation: none. No level keeps the 99th percentile within 1 ms") ||
		!strings.Contains(stdout, "with 3 at once") || strings.Contains(stdout, "--max-inflight") {
		t.Errorf("a target that cannot be kept:\n%s", stdout)
	}
}

func TestLevelsBeyondTheMachine(t *testing.T) {
	useFake(t, &enginetest.Fake{})
	onMachine(t, 2)
	file := pictureFile(t, "face:anna")

	// the default levels are 1, 2, 4 and 8
	stdout, stderr, err := run(context.Background(), "--image", file, "--duration", "10ms", "--gallery", "50", "--json")
	if err != nil {
		t.Fatalf("%v\n%s", err, stderr)
	}
	if !strings.Contains(stderr, "leaving out --inflight 4,8: this machine has 2 processors") {
		t.Errorf("standard error does not say what was left out:\n%s", stderr)
	}
	var got struct {
		Inflight []struct {
			Inflight int `json:"inflight"`
		} `json:"inflight"`
	}
	if err := json.Unmarshal([]byte(stdout), &got); err != nil {
		t.Fatal(err)
	}
	if len(got.Inflight) != 2 || got.Inflight[0].Inflight != 1 || got.Inflight[1].Inflight != 2 {
		t.Errorf("the levels measured on two processors: %+v", got.Inflight)
	}
}

func TestWrongCommandLine(t *testing.T) {
	onMachine(t, 4)
	useEngine(t, func(likeness.Config) (cli.Engine, error) {
		t.Error("the engine was opened for a command line that is wrong")
		return &enginetest.Fake{}, nil
	})
	for name, c := range map[string]struct {
		args []string
		says string
	}{
		"stray argument":    {[]string{"now"}, `unexpected argument "now"`},
		"unknown flag":      {[]string{"--fast"}, "flag provided but not defined"},
		"negative threads":  {[]string{"--threads", "-2"}, "--threads cannot be negative"},
		"no duration":       {[]string{"--duration", "0s"}, "longer than nothing"},
		"negative duration": {[]string{"--duration", "-1s"}, "longer than nothing"},
		"not a duration":    {[]string{"--duration", "soon"}, "invalid value"},
		"no target":         {[]string{"--target", "0"}, "longer than nothing"},
		"empty gallery":     {[]string{"--gallery", "0"}, "is not a whole number from 1 to 1000000"},
		"huge gallery":      {[]string{"--gallery", "1000,1000001"}, `"1000001" is not a whole number`},
		"gallery in words":  {[]string{"--gallery", "many"}, "write a list like 1,2,4"},
		"gallery cut short": {[]string{"--gallery", "1000,"}, "is not a whole number"},
		"level zero":        {[]string{"--inflight", "1,0"}, "is not a whole number from 1 to"},
		"negative level":    {[]string{"--inflight", "-2"}, "is not a whole number"},
		"levels too high":   {[]string{"--inflight", "8,16"}, "this machine has 4 processors: every level of --inflight 8,16 is beyond it"},
	} {
		stdout, stderr, err := run(context.Background(), c.args...)
		if !errors.Is(err, cli.ErrUsage) {
			t.Errorf("%s: %v", name, err)
		}
		if !strings.Contains(stderr, c.says) {
			t.Errorf("%s: explained with %q, want %q", name, stderr, c.says)
		}
		if stdout != "" {
			t.Errorf("%s: printed %q", name, stdout)
		}
	}

	stdout, stderr, err := run(context.Background(), "-h")
	if !errors.Is(err, flag.ErrHelp) || stdout != "" {
		t.Errorf("-h: %v, printed %q", err, stdout)
	}
	for _, want := range []string{
		"usage: likeness bench", "--max-inflight", "-preset", "-home directory", "-threads", "-image file", "-duration",
		"-gallery subjects", "(default 10000)", "-inflight pictures", "(default 1,2,4,8)", "-target", "(default 100ms)", "-json",
		"LIKENESS_HOME",
	} {
		if !strings.Contains(stderr, want) {
			t.Errorf("the help lacks %q", want)
		}
	}
}

func TestCounts(t *testing.T) {
	c := counts{most: 64}
	if err := c.Set(" 8,2, 4 ,2,64"); err != nil || c.String() != "2,4,8,64" {
		t.Errorf("a list out of order with a repeat: %q %v", c.String(), err)
	}
	for _, text := range []string{"", ",", "2,,4", "65", "0", "-1", "1.5", "two", "1;2"} {
		if err := c.Set(text); err == nil {
			t.Errorf("%q: accepted as %q", text, c.String())
		}
	}
	if c.String() != "2,4,8,64" {
		t.Errorf("a list that was refused changed the flag to %q", c.String())
	}
	if (&counts{}).String() != "" {
		t.Error("an empty list is not written as nothing")
	}
}

func TestPictureWithoutAFace(t *testing.T) {
	onMachine(t, 4)
	for text, reason := range map[string]string{
		"reject:no_face":        "no_face",
		"reject:blurry":         "blurry",
		"reject:multiple_faces": "multiple_faces",
	} {
		engine := useFake(t, &enginetest.Fake{})
		file := pictureFile(t, text)
		stdout, _, err := run(context.Background(), quick(file)...)
		if err == nil || !strings.Contains(err.Error(), "no usable face in the picture ("+reason+")") ||
			!strings.Contains(err.Error(), file) || !strings.Contains(err.Error(), "--image") {
			t.Errorf("%s: %v", text, err)
		}
		if stdout != "" {
			t.Errorf("%s: a report was printed:\n%s", text, stdout)
		}
		if engine.Calls() != 1 {
			t.Errorf("%s: the picture was analysed %d times, want once", text, engine.Calls())
		}
		if !engine.Closed() {
			t.Errorf("%s: the engine was left open", text)
		}
	}
}

func TestPictureThatCannotBeRead(t *testing.T) {
	onMachine(t, 4)

	// the engine cannot make sense of it
	engine := useFake(t, &enginetest.Fake{})
	file := pictureFile(t, "error:corrupt")
	_, _, err := run(context.Background(), quick(file)...)
	if !errors.Is(err, likeness.ErrCorruptImage) || !strings.Contains(err.Error(), file) {
		t.Errorf("a damaged picture: %v", err)
	}
	if !engine.Closed() {
		t.Error("a damaged picture: the engine was left open")
	}

	// the file is not there: no need to load the models to find out
	engine = useFake(t, &enginetest.Fake{})
	missing := filepath.Join(t.TempDir(), "nobody.jpg")
	stdout, _, err := run(context.Background(), quick(missing)...)
	if !errors.Is(err, os.ErrNotExist) || !strings.Contains(err.Error(), "--image") || !strings.Contains(err.Error(), missing) {
		t.Errorf("a missing picture: %v", err)
	}
	if stdout != "" || engine.opened != 0 {
		t.Errorf("a missing picture: printed %q, opened the engine %d times", stdout, engine.opened)
	}
}

func TestEngineThatWillNotOpen(t *testing.T) {
	onMachine(t, 4)
	noModels := errors.New("no models here")
	useEngine(t, func(likeness.Config) (cli.Engine, error) { return nil, noModels })
	stdout, _, err := run(context.Background(), quick(pictureFile(t, "face:anna"))...)
	if !errors.Is(err, noModels) || stdout != "" {
		t.Errorf("%v, printed %q", err, stdout)
	}
}

// tiring is an engine that gives up after a number of pictures.
type tiring struct {
	*enginetest.Fake
	left atomic.Int64
}

var errWornOut = errors.New("the engine is worn out")

func tiringAfter(pictures int, fake *enginetest.Fake) *tiring {
	e := &tiring{Fake: fake}
	e.left.Store(int64(pictures))
	return e
}

func (e *tiring) Analyze(ctx context.Context, picture []byte, o likeness.Options) (*likeness.Result, error) {
	if e.left.Add(-1) < 0 {
		return nil, errWornOut
	}
	return e.Fake.Analyze(ctx, picture, o)
}

func TestEngineThatFailsHalfway(t *testing.T) {
	onMachine(t, 4)
	file := pictureFile(t, "face:anna")
	// after the first picture, while warming up; and a little later
	for _, pictures := range []int{1, 3} {
		engine := tiringAfter(pictures, &enginetest.Fake{})
		useEngine(t, func(likeness.Config) (cli.Engine, error) { return engine, nil })
		stdout, _, err := run(context.Background(), quick(file)...)
		if !errors.Is(err, errWornOut) || !strings.Contains(err.Error(), file) {
			t.Errorf("after %d pictures: %v", pictures, err)
		}
		if stdout != "" {
			t.Errorf("after %d pictures: half a report was printed:\n%s", pictures, stdout)
		}
		if !engine.Closed() {
			t.Errorf("after %d pictures: the engine was left open", pictures)
		}
	}
}

func TestCancelled(t *testing.T) {
	onMachine(t, 4)
	file := pictureFile(t, "slow:anna")

	// the engine holds the first picture until the run is called off
	engine := useFake(t, &enginetest.Fake{Gate: make(chan struct{})})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	type result struct {
		stdout string
		err    error
	}
	done := make(chan result, 1)
	go func() {
		stdout, _, err := run(ctx, quick(file, "--duration", "1h")...)
		done <- result{stdout, err}
	}()
	for start := time.Now(); engine.Calls() == 0; time.Sleep(time.Millisecond) {
		if time.Since(start) > 10*time.Second {
			t.Fatal("the picture never reached the engine")
		}
	}
	cancel()
	select {
	case r := <-done:
		if !errors.Is(r.err, context.Canceled) || r.stdout != "" {
			t.Errorf("%v, printed %q", r.err, r.stdout)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("the command did not stop")
	}
	if !engine.Closed() {
		t.Error("the engine was left open")
	}

	// out of time rather than called off
	engine = useFake(t, &enginetest.Fake{Gate: make(chan struct{})})
	ctx, cancel = context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	if _, _, err := run(ctx, quick(file, "--duration", "1h")...); !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("out of time: %v", err)
	}
	if !engine.Closed() {
		t.Error("out of time: the engine was left open")
	}
}

// full is a standard output that takes nothing.
type full struct{}

var errFull = errors.New("no space left")

func (full) Write([]byte) (int, error) { return 0, errFull }

func TestReportThatCannotBeWritten(t *testing.T) {
	engine := useFake(t, &enginetest.Fake{})
	onMachine(t, 4)
	var notes bytes.Buffer
	err := Run(context.Background(), quick(pictureFile(t, "face:anna")), cli.Env{Stdout: full{}, Stderr: &notes})
	if !errors.Is(err, errFull) {
		t.Errorf("%v", err)
	}
	if !engine.Closed() {
		t.Error("the engine was left open")
	}
}
