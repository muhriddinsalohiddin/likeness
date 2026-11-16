package enrol

import (
	"bytes"
	"context"
	"encoding/csv"
	"encoding/hex"
	"errors"
	"flag"
	"fmt"
	"io"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/muhriddinsalohiddin/likeness"
	"github.com/muhriddinsalohiddin/likeness/internal/cli"
	"github.com/muhriddinsalohiddin/likeness/internal/enginetest"
	"github.com/muhriddinsalohiddin/likeness/internal/gallery"
	"github.com/muhriddinsalohiddin/likeness/internal/server"
	"github.com/muhriddinsalohiddin/likeness/store"
)

var ctx = context.Background()

// environment makes a getenv out of pairs of names and values.
func environment(pairs ...string) func(string) string {
	vars := make(map[string]string)
	for i := 0; i < len(pairs); i += 2 {
		vars[pairs[i]] = pairs[i+1]
	}
	return func(name string) string { return vars[name] }
}

// fakes are the stand-in engines the command has opened, in order.
type fakes struct {
	mu     sync.Mutex
	opened []*enginetest.Fake
	// cfg is what the last engine was opened with.
	cfg likeness.Config
	// gate and delay are given to the engines opened from here on: what
	// their "slow:" pictures wait for.
	gate  chan struct{}
	delay time.Duration
}

// useFake puts stand-in engines in place of the real one.
func useFake(t *testing.T) *fakes {
	t.Helper()
	f := &fakes{}
	real := cli.OpenEngine
	cli.OpenEngine = func(cfg likeness.Config) (cli.Engine, error) {
		f.mu.Lock()
		defer f.mu.Unlock()
		e := &enginetest.Fake{Gate: f.gate, Delay: f.delay}
		f.opened, f.cfg = append(f.opened, e), cfg
		return e, nil
	}
	t.Cleanup(func() { cli.OpenEngine = real })
	return f
}

func (f *fakes) count() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.opened)
}

func (f *fakes) last() *enginetest.Fake {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.opened[len(f.opened)-1]
}

func (f *fakes) allClosed() bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	for _, e := range f.opened {
		if !e.Closed() {
			return false
		}
	}
	return len(f.opened) > 0
}

// tree makes a directory out of pairs of a file name and its content. A
// name that ends in a slash is an empty directory.
func tree(t *testing.T, files ...string) string {
	t.Helper()
	dir := t.TempDir()
	plant(t, dir, files...)
	return dir
}

func plant(t *testing.T, dir string, files ...string) {
	t.Helper()
	for i := 0; i < len(files); i += 2 {
		path := filepath.Join(dir, filepath.FromSlash(files[i]))
		if strings.HasSuffix(files[i], "/") {
			if err := os.MkdirAll(path, 0o755); err != nil {
				t.Fatal(err)
			}
			continue
		}
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(files[i+1]), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

// ran is what a run printed and returned.
type ran struct {
	stdout, stderr string
	err            error
}

func run(t *testing.T, getenv func(string) string, args ...string) ran {
	t.Helper()
	var stdout, stderr bytes.Buffer
	err := Run(ctx, args, cli.Env{Getenv: getenv, Stdout: &stdout, Stderr: &stderr})
	return ran{stdout.String(), stderr.String(), err}
}

// summary checks rows of the summary: pairs of a label and what follows it.
func (r ran) summary(t *testing.T, pairs ...string) {
	t.Helper()
	rows := make(map[string]string)
	for _, line := range strings.Split(r.stdout, "\n") {
		if len(line) > 19 {
			rows[strings.TrimSpace(line[:18])] = line[19:]
		}
	}
	for i := 0; i < len(pairs); i += 2 {
		if got, ok := rows[pairs[i]]; !ok || got != pairs[i+1] {
			t.Errorf("the summary has %q for %q, want %q\n%s", got, pairs[i], pairs[i+1], r.stdout)
		}
	}
}

// inspect opens the store as a server would, hands the gallery to look and
// lets go of the file again.
func inspect(t *testing.T, path string, key []byte, look func(g *gallery.Gallery)) {
	t.Helper()
	info := (&enginetest.Fake{}).Info()
	st, err := store.Open(path, store.Options{Dim: info.Dim, Fingerprint: info.Fingerprint, Key: key})
	if err != nil {
		t.Fatalf("the store does not open: %v", err)
	}
	g, err := gallery.Open(ctx, st, gallery.Options{})
	if err != nil {
		st.Close()
		t.Fatal(err)
	}
	defer g.Close()
	look(g)
}

// held checks which subjects the store holds, and how many templates each
// of them has.
func held(t *testing.T, path string, want map[string]int) {
	t.Helper()
	inspect(t, path, nil, func(g *gallery.Gallery) {
		t.Helper()
		page, err := g.Subjects(ctx, "", 1000)
		if err != nil {
			t.Fatal(err)
		}
		got := make(map[string]int)
		for _, s := range page {
			got[s.ID] = s.Templates
		}
		if !maps.Equal(got, want) {
			t.Errorf("the store holds %v, want %v", got, want)
		}
	})
}

// score is how close the subject's best template is to the picture of name.
func score(t *testing.T, g *gallery.Gallery, subject, name string) float32 {
	t.Helper()
	m, ok, err := g.Compare(subject, enginetest.Vector(name))
	if err != nil || !ok {
		t.Fatalf("%s is not enrolled: %v", subject, err)
	}
	return m.Score
}

// readReport returns the lines of a report, header included.
func readReport(t *testing.T, path string) [][]string {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	lines, err := csv.NewReader(f).ReadAll()
	if err != nil {
		t.Fatal(err)
	}
	if len(lines) == 0 || !slices.Equal(lines[0], reportHeader) {
		t.Fatalf("the report has no header: %v", lines)
	}
	return lines
}

// outcomes is the outcome column of a report.
func outcomes(t *testing.T, path string) []string {
	t.Helper()
	lines := readReport(t, path)[1:]
	out := make([]string, len(lines))
	for i, line := range lines {
		out[i] = line[2]
	}
	return out
}

func TestEnrol(t *testing.T) {
	engines := useFake(t)
	dir := tree(t,
		"anna/1.jpg", "face:anna",
		"anna/2.PNG", "face:anna#2",
		"anna/notes.txt", "not a picture",
		"anna/.hidden.jpg", "face:ghost",
		"anna/more/3.jpg", "face:anna#3",
		"boris/portrait.jpeg", "face:boris",
		"carla/a.webp", "face:carla",
		".cache/1.jpg", "face:ghost",
		"stray.jpg", "face:stray",
	)
	work := t.TempDir()
	path, report := filepath.Join(work, "faces.db"), filepath.Join(work, "report.csv")

	r := run(t, environment(), "--store", path, "--report", report, dir)
	if r.err != nil {
		t.Fatalf("%v\n%s", r.err, r.stderr)
	}
	const want = `pictures read      4 of 4
enrolled           4
already enrolled   0
turned down        0
duplicates         0
mismatches         0
full               0
unreadable         0
subjects in store  0 before, 3 after
`
	if r.stdout != want {
		t.Errorf("the summary:\n%s\nwant:\n%s", r.stdout, want)
	}
	if r.stderr != "" {
		t.Errorf("a short run with nothing wrong said: %s", r.stderr)
	}
	if !engines.allClosed() || engines.last().Calls() != 4 {
		t.Errorf("the engine: closed %v, %d pictures analysed", engines.allClosed(), engines.last().Calls())
	}

	held(t, path, map[string]int{"anna": 2, "boris": 1, "carla": 1})
	inspect(t, path, nil, func(g *gallery.Gallery) {
		for subject, name := range map[string]string{"anna": "anna#2", "boris": "boris", "carla": "carla"} {
			if s := score(t, g, subject, name); s < sameAbove {
				t.Errorf("%s does not hold the face %s: %.4f", subject, name, s)
			}
		}
		// the metadata is what the server stores for its own templates
		res, err := (&enginetest.Fake{}).Analyze(ctx, []byte("face:anna"), likeness.ForEnrolment())
		if err != nil {
			t.Fatal(err)
		}
		meta, err := server.TemplateMeta(res.Faces[0].Quality)
		if err != nil {
			t.Fatal(err)
		}
		anna, err := g.Subject(ctx, "anna")
		if err != nil || len(anna.Templates) != 2 {
			t.Fatalf("%+v %v", anna, err)
		}
		for _, tm := range anna.Templates {
			if !bytes.Equal(tm.Meta, meta) || !bytes.Contains(tm.Meta, []byte(`"quality"`)) {
				t.Errorf("template %d carries %s, want %s", tm.ID, tm.Meta, meta)
			}
		}
	})

	lines := readReport(t, report)
	wantLines := [][]string{
		reportHeader,
		{"anna", filepath.Join(dir, "anna", "1.jpg"), "enrolled", "", "", "", "1"},
		{"anna", filepath.Join(dir, "anna", "2.PNG"), "enrolled", "", "", "", "2"},
		{"boris", filepath.Join(dir, "boris", "portrait.jpeg"), "enrolled", "", "", "", "3"},
		{"carla", filepath.Join(dir, "carla", "a.webp"), "enrolled", "", "", "", "4"},
	}
	if len(lines) != len(wantLines) {
		t.Fatalf("the report has %d lines, want %d: %v", len(lines), len(wantLines), lines)
	}
	for i := range lines {
		if !slices.Equal(lines[i], wantLines[i]) {
			t.Errorf("line %d of the report is %q, want %q", i+1, lines[i], wantLines[i])
		}
	}

	// the same import again changes nothing
	r = run(t, environment(), "--store", path, "--report", report, dir)
	if r.err != nil {
		t.Fatal(r.err)
	}
	r.summary(t, "pictures read", "4 of 4", "enrolled", "0", "already enrolled", "4", "subjects in store", "3 before, 3 after")
	held(t, path, map[string]int{"anna": 2, "boris": 1, "carla": 1})
	for i, line := range readReport(t, report)[1:] {
		if line[2] != "already enrolled" || line[6] != fmt.Sprint(i+1) {
			t.Errorf("the second run reports %q", line)
		}
	}

	// and a later batch adds what is new
	plant(t, dir, "anna/3.jpg", "face:anna#3", "dora/1.jpg", "face:dora")
	r = run(t, environment(), "--store", path, dir)
	if r.err != nil {
		t.Fatal(r.err)
	}
	r.summary(t, "pictures read", "6 of 6", "enrolled", "2", "already enrolled", "4", "subjects in store", "3 before, 4 after")
	held(t, path, map[string]int{"anna": 3, "boris": 1, "carla": 1, "dora": 1})
}

func TestEnrolTurnsPicturesDown(t *testing.T) {
	useFake(t)
	dir := tree(t,
		"anna/1.jpg", "face:anna",
		"anna/2.jpg", "reject:blurry",
		"boris/1.jpg", "reject:no_face",
		"boris/2.jpg", "faces:boris,carla",
		"boris/3.jpg", "reject:blurry",
		"carla/1.jpg", "error:corrupt",
		"carla/2.jpg", "a text file with the wrong name",
		"dora/1.jpg", "error:small",
		"dora/2.jpg", "error:large",
		"dora/3.jpg/", "",
		"dora/4.jpg", "",
	)
	if err := os.Truncate(filepath.Join(dir, "dora", "4.jpg"), maxPicture+1); err != nil {
		t.Fatal(err)
	}
	work := t.TempDir()
	path, report := filepath.Join(work, "faces.db"), filepath.Join(work, "report.csv")

	r := run(t, environment(), "--store", path, "--report", report, dir)
	if r.err != nil {
		t.Fatalf("pictures that were turned down failed the run: %v", r.err)
	}
	r.summary(t, "pictures read", "10 of 10", "enrolled", "1", "turned down", "4",
		"blurry", "2", "multiple_faces", "1", "no_face", "1", "unreadable", "5", "subjects in store", "0 before, 1 after")
	if !strings.Contains(r.stdout, "turned down        4\n  blurry           2\n  multiple_faces   1\n  no_face          1\nduplicates") {
		t.Errorf("the reasons are not listed under the count:\n%s", r.stdout)
	}
	held(t, path, map[string]int{"anna": 1})

	want := [][2]string{
		{"enrolled", ""},
		{"turned down", "blurry"},
		{"turned down", "no_face"},
		{"turned down", "multiple_faces"},
		{"turned down", "blurry"},
		{"unreadable", "damaged or cut short"},
		{"unreadable", "not a JPEG, PNG or WebP image"},
		{"unreadable", "too small to hold a face"},
		{"unreadable", "more pixels than the engine accepts"},
		{"unreadable", "larger than 64 MiB"},
	}
	lines := readReport(t, report)[1:]
	if len(lines) != len(want) {
		t.Fatalf("the report has %d pictures, want %d: %v", len(lines), len(want), lines)
	}
	for i, line := range lines {
		if line[2] != want[i][0] || line[3] != want[i][1] {
			t.Errorf("%s: %q because of %q, want %q because of %q", line[1], line[2], line[3], want[i][0], want[i][1])
		}
	}
}

func TestEnrolNothing(t *testing.T) {
	engines := useFake(t)
	path := filepath.Join(t.TempDir(), "faces.db")

	// every picture turned down: the run completed, and still it failed
	r := run(t, environment(), "--store", path, tree(t, "anna/1.jpg", "reject:blurry", "boris/1.jpg", "error:corrupt"))
	if r.err == nil || !strings.Contains(r.err.Error(), "none of the 2 pictures was enrolled") {
		t.Errorf("nothing enrolled: %v", r.err)
	}
	r.summary(t, "pictures read", "2 of 2", "enrolled", "0", "turned down", "1", "unreadable", "1")
	if !engines.allClosed() {
		t.Error("the engine was left open")
	}
	held(t, path, map[string]int{})

	// the wrong directory is noticed before the models are loaded
	opened := engines.count()
	fresh := filepath.Join(t.TempDir(), "faces.db")
	for name, dir := range map[string]string{
		"an empty directory":             tree(t),
		"pictures with no sub-directory": tree(t, "1.jpg", "face:anna", "2.jpg", "face:boris"),
		"sub-directories of other files": tree(t, "anna/cv.pdf", "a file", "boris/", ""),
	} {
		r := run(t, environment(), "--store", fresh, dir)
		if r.err == nil || !strings.Contains(r.err.Error(), "no pictures found") || !strings.Contains(r.err.Error(), "one sub-directory per subject") {
			t.Errorf("%s: %v", name, r.err)
		}
	}
	if r := run(t, environment(), "--store", fresh, filepath.Join(t.TempDir(), "missing")); !errors.Is(r.err, os.ErrNotExist) {
		t.Errorf("a directory that is not there: %v", r.err)
	}
	if r := run(t, environment(), "--store", fresh, filepath.Join(tree(t, "anna", "a file"), "anna")); r.err == nil {
		t.Error("a file was taken for a directory")
	}
	if _, err := os.Stat(fresh); !errors.Is(err, os.ErrNotExist) || engines.count() != opened {
		t.Errorf("a run without pictures made a store (%v) or loaded the models (%d times)", err, engines.count()-opened)
	}
}

func TestEnrolReplace(t *testing.T) {
	useFake(t)
	path := filepath.Join(t.TempDir(), "faces.db")
	if r := run(t, environment(), "--store", path, tree(t, "anna/1.jpg", "face:anna", "boris/1.jpg", "face:boris")); r.err != nil {
		t.Fatal(r.err)
	}

	// the first picture that is accepted replaces, the others are added
	dir := tree(t,
		"anna/1.jpg", "reject:blurry",
		"anna/2.jpg", "face:anna#2",
		"anna/3.jpg", "face:anna#3",
		"anna/4.jpg", "face:anna#2",
	)
	report := filepath.Join(t.TempDir(), "report.csv")
	for range 2 {
		r := run(t, environment(), "--store", path, "--replace", "--report", report, dir)
		if r.err != nil {
			t.Fatal(r.err)
		}
		if got, want := outcomes(t, report), []string{"turned down", "enrolled", "enrolled", "already enrolled"}; !slices.Equal(got, want) {
			t.Errorf("--replace: %q, want %q", got, want)
		}
		held(t, path, map[string]int{"anna": 2, "boris": 1})
		inspect(t, path, nil, func(g *gallery.Gallery) {
			if s := score(t, g, "anna", "anna"); s >= sameAbove {
				t.Errorf("the template anna had before is still there: %.4f", s)
			}
		})
	}

	// without it they pile up
	if r := run(t, environment(), "--store", path, tree(t, "anna/1.jpg", "face:anna")); r.err != nil {
		t.Fatal(r.err)
	}
	held(t, path, map[string]int{"anna": 3, "boris": 1})

	// a picture the subject holds already replaces the others all the same
	r := run(t, environment(), "--store", path, "--replace", "--report", report, tree(t, "anna/1.jpg", "face:anna"))
	if r.err != nil {
		t.Fatal(r.err)
	}
	if got := outcomes(t, report); !slices.Equal(got, []string{"enrolled"}) {
		t.Errorf("--replace with a picture enrolled before: %q", got)
	}
	held(t, path, map[string]int{"anna": 1, "boris": 1})
}

func TestEnrolDuplicate(t *testing.T) {
	useFake(t)
	dir := tree(t, "anna/1.jpg", "face:anna", "boris/1.jpg", "face:anna#2", "carla/1.jpg", "face:carla")
	work := t.TempDir()
	path, report := filepath.Join(work, "faces.db"), filepath.Join(work, "report.csv")

	r := run(t, environment(), "--store", path, "--report", report, dir)
	if r.err != nil {
		t.Fatal(r.err)
	}
	r.summary(t, "enrolled", "2", "duplicates", "1", "subjects in store", "0 before, 2 after")
	held(t, path, map[string]int{"anna": 1, "carla": 1})
	inspect(t, path, nil, func(g *gallery.Gallery) {
		want := []string{"boris", filepath.Join(dir, "boris", "1.jpg"), "duplicate", "", fmt.Sprintf("%.4f", score(t, g, "anna", "anna#2")), "anna", ""}
		if line := readReport(t, report)[2]; !slices.Equal(line, want) {
			t.Errorf("the report says %q, want %q", line, want)
		}
	})

	// twins exist
	r = run(t, environment(), "--store", path, "--allow-duplicate", dir)
	if r.err != nil {
		t.Fatal(r.err)
	}
	r.summary(t, "enrolled", "1", "already enrolled", "2", "duplicates", "0")
	held(t, path, map[string]int{"anna": 1, "boris": 1, "carla": 1})

	// a stricter threshold lets the same face through
	fresh := filepath.Join(work, "strict.db")
	r = run(t, environment(), "--store", fresh, "--search-threshold", "0.95", dir)
	if r.err != nil {
		t.Fatal(r.err)
	}
	r.summary(t, "enrolled", "3", "duplicates", "0")
}

func TestEnrolMismatch(t *testing.T) {
	useFake(t)
	dir := tree(t, "anna/1.jpg", "face:anna", "anna/2.jpg", "face:boris", "anna/3.jpg", "face:anna#2")
	work := t.TempDir()
	path, report := filepath.Join(work, "faces.db"), filepath.Join(work, "report.csv")

	r := run(t, environment(), "--store", path, "--report", report, dir)
	if r.err != nil {
		t.Fatal(r.err)
	}
	r.summary(t, "enrolled", "2", "mismatches", "1")
	held(t, path, map[string]int{"anna": 2})
	inspect(t, path, nil, func(g *gallery.Gallery) {
		want := []string{"anna", filepath.Join(dir, "anna", "2.jpg"), "mismatch", "", fmt.Sprintf("%.4f", score(t, g, "anna", "boris")), "", ""}
		if line := readReport(t, report)[2]; !slices.Equal(line, want) {
			t.Errorf("the report says %q, want %q", line, want)
		}
	})

	// people change
	r = run(t, environment(), "--store", path, "--allow-mismatch", dir)
	if r.err != nil {
		t.Fatal(r.err)
	}
	r.summary(t, "enrolled", "1", "already enrolled", "2", "mismatches", "0")
	held(t, path, map[string]int{"anna": 3})

	// a stricter threshold turns another picture of anna away
	fresh := filepath.Join(work, "strict.db")
	r = run(t, environment(), "--store", fresh, "--verify-threshold", "0.9", "--report", report, dir)
	if r.err != nil {
		t.Fatal(r.err)
	}
	if got, want := outcomes(t, report), []string{"enrolled", "mismatch", "mismatch"}; !slices.Equal(got, want) {
		t.Errorf("--verify-threshold 0.9: %q, want %q", got, want)
	}
}

func TestEnrolMaxTemplates(t *testing.T) {
	useFake(t)
	files := []string{"anna/01.jpg", "face:anna"}
	for i := 2; i <= 12; i++ {
		files = append(files, fmt.Sprintf("anna/%02d.jpg", i), fmt.Sprintf("face:anna#%d", i))
	}
	dir := tree(t, files...)
	work := t.TempDir()
	path, report := filepath.Join(work, "faces.db"), filepath.Join(work, "report.csv")

	r := run(t, environment(), "--store", path, "--max-templates", "2", "--report", report, dir)
	if r.err != nil {
		t.Fatal(r.err)
	}
	r.summary(t, "enrolled", "2", "full", "10")
	if got := outcomes(t, report); !slices.Equal(got[:3], []string{"enrolled", "enrolled", "full"}) {
		t.Errorf("--max-templates 2: %q", got)
	}

	// the server's bound when none is given
	r = run(t, environment(), "--store", path, dir)
	if r.err != nil {
		t.Fatal(r.err)
	}
	r.summary(t, "enrolled", "8", "already enrolled", "2", "full", "2")
	held(t, path, map[string]int{"anna": 10})

	r = run(t, environment(), "--store", path, "--max-templates", "-1", dir)
	if r.err != nil {
		t.Fatal(r.err)
	}
	r.summary(t, "enrolled", "2", "already enrolled", "10", "full", "0")
	held(t, path, map[string]int{"anna": 12})
}

func TestEnrolDryRun(t *testing.T) {
	engines := useFake(t)
	work := t.TempDir()
	path, report := filepath.Join(work, "faces.db"), filepath.Join(work, "report.csv")
	seed := tree(t, "anna/1.jpg", "face:anna", "anna/2.jpg", "face:anna#2", "boris/1.jpg", "face:boris", "carla/1.jpg", "reject:blurry")

	// before there is a store
	r := run(t, environment(), "--store", path, "--dry-run", "--report", report, seed)
	if r.err != nil {
		t.Fatal(r.err)
	}
	if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("a dry run made the store: %v", err)
	}
	r.summary(t, "pictures read", "4 of 4", "would be enrolled", "3", "turned down", "1",
		"subjects in store", "0: the file does not exist, and a dry run does not make it")
	if strings.Contains(r.stdout, "\nenrolled ") || !strings.Contains(r.stdout, "not against each other") {
		t.Errorf("the summary of a dry run:\n%s", r.stdout)
	}
	if got, want := outcomes(t, report), []string{"would be enrolled", "would be enrolled", "would be enrolled", "turned down"}; !slices.Equal(got, want) {
		t.Errorf("the report of a dry run: %q, want %q", got, want)
	}
	if !engines.allClosed() {
		t.Error("the engine was left open")
	}
	if r := run(t, environment(), "--store", path, "--dry-run", tree(t, "anna/1.jpg", "reject:blurry")); r.err == nil ||
		!strings.Contains(r.err.Error(), "none of the 1 pictures would be enrolled") {
		t.Errorf("a dry run in which nothing passes: %v", r.err)
	}

	// against what a store holds
	if r := run(t, environment(), "--store", path, seed); r.err != nil {
		t.Fatal(r.err)
	}
	before, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	dir := tree(t,
		"anna/1.jpg", "face:anna",
		"anna/2.jpg", "face:anna#3",
		"anna/3.jpg", "face:emil",
		"anna/4.jpg", "face:anna#4",
		"dora/1.jpg", "face:boris#2",
		"emil/1.jpg", "face:emil",
	)
	for name, c := range map[string]struct {
		flags []string
		want  []string
	}{
		"the server's rules": {nil,
			[]string{"already enrolled", "would be enrolled", "mismatch", "would be enrolled", "duplicate", "would be enrolled"}},
		"allowing both": {[]string{"--allow-duplicate", "--allow-mismatch"},
			[]string{"already enrolled", "would be enrolled", "would be enrolled", "would be enrolled", "would be enrolled", "would be enrolled"}},
		"three templates each": {[]string{"--max-templates", "3"},
			[]string{"already enrolled", "would be enrolled", "mismatch", "full", "duplicate", "would be enrolled"}},
		"replacing": {[]string{"--replace"},
			[]string{"would be enrolled", "would be enrolled", "would be enrolled", "would be enrolled", "duplicate", "would be enrolled"}},
		"replacing, three templates each": {[]string{"--replace", "--max-templates", "3"},
			[]string{"would be enrolled", "would be enrolled", "would be enrolled", "full", "duplicate", "would be enrolled"}},
	} {
		args := append([]string{"--store", path, "--dry-run", "--report", report}, c.flags...)
		r := run(t, environment(), append(args, dir)...)
		if r.err != nil {
			t.Fatalf("%s: %v", name, r.err)
		}
		if got := outcomes(t, report); !slices.Equal(got, c.want) {
			t.Errorf("%s: %q, want %q", name, got, c.want)
		}
		r.summary(t, "subjects in store", "2, as before: a dry run writes nothing")
		if dup := readReport(t, report)[5]; c.flags == nil && (dup[5] != "boris" || dup[4] == "" || dup[6] != "") {
			t.Errorf("the duplicate a dry run found: %q", dup)
		}
	}

	after, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(before, after) {
		t.Error("dry runs changed the store file")
	}
	held(t, path, map[string]int{"anna": 2, "boris": 1})
}

func TestEnrolInvalidSubjects(t *testing.T) {
	engines := useFake(t)
	long := strings.Repeat("x", store.MaxSubject+1)
	dir := tree(t,
		"anna maria/1.jpg", "face:anna",
		"anna maria/2.jpg", "face:anna#2",
		"boris/1.jpg", "face:boris",
		"empty/", "",
		long+"/1.jpg", "face:carla",
	)
	work := t.TempDir()
	path, report := filepath.Join(work, "faces.db"), filepath.Join(work, "report.csv")

	r := run(t, environment(), "--store", path, "--report", report, dir)
	if r.err != nil {
		t.Fatal(r.err)
	}
	for _, want := range []string{
		"likeness enrol: " + filepath.Join(dir, "anna maria") + ": the name cannot be a subject id: contains ' '\n",
		"likeness enrol: " + filepath.Join(dir, "empty") + ": no pictures\n",
		"likeness enrol: " + filepath.Join(dir, long) + ": the name cannot be a subject id: longer than 128 bytes\n",
	} {
		if !strings.Contains(r.stderr, want) {
			t.Errorf("stderr lacks %q:\n%s", want, r.stderr)
		}
	}
	r.summary(t, "pictures read", "4 of 4", "enrolled", "1", "invalid subject", "3")
	held(t, path, map[string]int{"boris": 1})
	if n := engines.last().Calls(); n != 1 {
		t.Errorf("%d pictures were analysed, want only the one with a usable subject", n)
	}
	lines := readReport(t, report)
	if want := []string{"anna maria", filepath.Join(dir, "anna maria", "1.jpg"), "invalid subject", "contains ' '", "", "", ""}; !slices.Equal(lines[1], want) {
		t.Errorf("the report says %q, want %q", lines[1], want)
	}

	// many of them do not fill the screen
	files := make([]string, 0, 2*14)
	for i := range 14 {
		files = append(files, fmt.Sprintf("no %d/1.jpg", i), "face:anna")
	}
	r = run(t, environment(), "--store", path, tree(t, files...))
	if r.err == nil || strings.Count(r.stderr, "cannot be a subject id") != 10 || !strings.Contains(r.stderr, "and 4 more of the kind") {
		t.Errorf("fourteen bad names: %v\n%s", r.err, r.stderr)
	}
}

func TestEnrolEncryptedStore(t *testing.T) {
	engines := useFake(t)
	dir := tree(t, "anna/1.jpg", "face:anna")
	work := t.TempDir()
	path := filepath.Join(work, "faces.db")
	key := strings.Repeat("0123456789abcdef", 4)
	raw, err := hex.DecodeString(key)
	if err != nil {
		t.Fatal(err)
	}

	if r := run(t, environment("LIKENESS_STORE_KEY", key), "--store", path, dir); r.err != nil {
		t.Fatal(r.err)
	}
	inspect(t, path, raw, func(g *gallery.Gallery) {
		if !g.Info().Store.Encrypted || !g.Has("anna") {
			t.Errorf("the store: %+v", g.Info())
		}
	})

	// without the key, or with another, the store stays shut
	for name, env := range map[string]func(string) string{
		"no key":      environment(),
		"another key": environment("LIKENESS_STORE_KEY", strings.Repeat("f", 64)),
	} {
		for _, mode := range [][]string{nil, {"--dry-run"}} {
			r := run(t, env, append(mode, "--store", path, dir)...)
			if !errors.Is(r.err, store.ErrKey) || !strings.Contains(r.err.Error(), "LIKENESS_STORE_KEY") || strings.Contains(r.err.Error(), key) {
				t.Errorf("%s %v: %v", name, mode, r.err)
			}
		}
	}

	// the key from a file, as container secrets arrive
	secret := filepath.Join(work, "store-key")
	if err := os.WriteFile(secret, []byte(key+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	r := run(t, environment("LIKENESS_STORE_KEY_FILE", secret), "--store", path, dir)
	if r.err != nil {
		t.Fatal(r.err)
	}
	r.summary(t, "already enrolled", "1")

	// a key that is not one is refused without being repeated
	opened := engines.count()
	r = run(t, environment("LIKENESS_STORE_KEY", "hunter2"), "--store", path, dir)
	if r.err == nil || !strings.Contains(r.err.Error(), "LIKENESS_STORE_KEY") || strings.Contains(r.err.Error()+r.stderr, "hunter2") || engines.count() != opened {
		t.Errorf("a key of the wrong form: %v", r.err)
	}
	if !engines.allClosed() {
		t.Error("an engine was left open")
	}
}

func TestEnrolStoreInUse(t *testing.T) {
	engines := useFake(t)
	path := filepath.Join(t.TempDir(), "faces.db")
	info := (&enginetest.Fake{}).Info()
	serving, err := store.Open(path, store.Options{Dim: info.Dim, Fingerprint: info.Fingerprint})
	if err != nil {
		t.Fatal(err)
	}
	defer serving.Close()

	r := run(t, environment(), "--store", path, tree(t, "anna/1.jpg", "face:anna"))
	if !errors.Is(r.err, store.ErrBusy) || !strings.Contains(r.err.Error(), "stop the server first") {
		t.Errorf("a store that a server holds: %v", r.err)
	}
	if r.stdout != "" || !engines.allClosed() {
		t.Errorf("a run that never started printed %q; engine closed: %v", r.stdout, engines.allClosed())
	}
}

func TestEnrolAnotherModelsStore(t *testing.T) {
	useFake(t)
	path := filepath.Join(t.TempDir(), "faces.db")
	st, err := store.Open(path, store.Options{Dim: enginetest.Dim, Fingerprint: "some:other:model"})
	if err != nil {
		t.Fatal(err)
	}
	st.Close()

	dir := tree(t, "anna/1.jpg", "face:anna")
	for _, mode := range [][]string{nil, {"--dry-run"}} {
		r := run(t, environment(), append(mode, "--store", path, dir)...)
		if !errors.Is(r.err, store.ErrMismatch) || !strings.Contains(r.err.Error(), "preset it was made with") {
			t.Errorf("a store made by another model %v: %v", mode, r.err)
		}
	}
}

// watch is a writer that tells when a text has gone through it.
type watch struct {
	mu   sync.Mutex
	text strings.Builder
	want string
	seen chan struct{}
}

func (w *watch) Write(p []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	had := strings.Contains(w.text.String(), w.want)
	w.text.Write(p)
	if !had && strings.Contains(w.text.String(), w.want) {
		close(w.seen)
	}
	return len(p), nil
}

func TestEnrolCancelled(t *testing.T) {
	every := plainEvery
	plainEvery = 0 // a line after every picture
	t.Cleanup(func() { plainEvery = every })

	engines := useFake(t)
	engines.gate = make(chan struct{}) // the "slow:" picture never comes back
	dir := tree(t, "anna/1.jpg", "face:anna", "boris/1.jpg", "face:boris", "carla/1.jpg", "slow:carla", "dora/1.jpg", "face:dora")
	path := filepath.Join(t.TempDir(), "faces.db")

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	var stdout bytes.Buffer
	stderr := &watch{want: "2 of 4 pictures", seen: make(chan struct{})}
	done := make(chan error, 1)
	go func() {
		done <- Run(ctx, []string{"--store", path, "--workers", "1", dir}, cli.Env{Getenv: environment(), Stdout: &stdout, Stderr: stderr})
	}()
	select {
	case <-stderr.seen:
	case err := <-done:
		t.Fatalf("the run ended before it was interrupted: %v", err)
	case <-time.After(10 * time.Second):
		t.Fatal("the run made no progress")
	}
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Errorf("an interrupted run returned %v", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("the run did not stop")
	}

	if !strings.Contains(stdout.String(), "pictures read      2 of 4\nenrolled           2\n") ||
		!strings.Contains(stdout.String(), "Stopped after 2 of 4 pictures") {
		t.Errorf("the summary of an interrupted run:\n%s", stdout.String())
	}
	if !engines.allClosed() {
		t.Error("the engine was left open")
	}
	held(t, path, map[string]int{"anna": 1, "boris": 1})

	// the same command carries on
	plant(t, dir, "carla/1.jpg", "face:carla")
	r := run(t, environment(), "--store", path, dir)
	if r.err != nil {
		t.Fatal(r.err)
	}
	r.summary(t, "enrolled", "2", "already enrolled", "2")
	held(t, path, map[string]int{"anna": 1, "boris": 1, "carla": 1, "dora": 1})
}

func TestEnrolEngineFailure(t *testing.T) {
	engines := useFake(t)
	dir := tree(t, "anna/1.jpg", "face:anna", "boris/1.jpg", "error:broken", "carla/1.jpg", "face:carla")
	path := filepath.Join(t.TempDir(), "faces.db")

	// not the picture's fault: the run ends there, and says where
	r := run(t, environment(), "--store", path, "--workers", "1", dir)
	if r.err == nil || !strings.Contains(r.err.Error(), filepath.Join(dir, "boris", "1.jpg")) || !strings.Contains(r.err.Error(), "the engine broke") {
		t.Errorf("an engine that fails: %v", r.err)
	}
	if !strings.Contains(r.stdout, "Stopped after 1 of 3 pictures") {
		t.Errorf("the summary:\n%s", r.stdout)
	}
	if !engines.allClosed() {
		t.Error("the engine was left open")
	}
	held(t, path, map[string]int{"anna": 1})

	// no models
	real := cli.OpenEngine
	cli.OpenEngine = func(likeness.Config) (cli.Engine, error) { return nil, errors.New("no models here") }
	defer func() { cli.OpenEngine = real }()
	fresh := filepath.Join(t.TempDir(), "faces.db")
	r = run(t, environment(), "--store", fresh, dir)
	if r.err == nil || !strings.Contains(r.err.Error(), "no models here") {
		t.Errorf("an engine that will not open: %v", r.err)
	}
	if _, err := os.Stat(fresh); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("a store was made for an engine that never opened: %v", err)
	}
}

func TestRefusal(t *testing.T) {
	for name, c := range map[string]struct {
		err  error
		want verdict
	}{
		"another subject's face": {fmt.Errorf("enrol: %w", &gallery.DuplicateError{Subject: "anna", Score: 0.91}),
			verdict{outcome: duplicate, other: "anna", score: 0.91}},
		"not the subject's face": {&gallery.MismatchError{Score: 0.2}, verdict{outcome: mismatch, score: 0.2}},
		"no room":                {gallery.ErrTooManyTemplates, verdict{outcome: full}},
		"a bad id":               {store.ValidSubject("a b"), verdict{outcome: invalidSubject, reason: "contains ' '"}},
	} {
		if got, err := refusal(c.err); err != nil || got != c.want {
			t.Errorf("%s: %+v %v, want %+v", name, got, err, c.want)
		}
	}
	// anything else is not about the picture
	full := errors.New("no space left on device")
	if got, err := refusal(full); !errors.Is(err, full) || got != (verdict{}) {
		t.Errorf("a store that fails: %+v %v", got, err)
	}
}

func TestEnrolReportFailure(t *testing.T) {
	engines := useFake(t)
	work := t.TempDir()
	path := filepath.Join(work, "faces.db")
	r := run(t, environment(), "--store", path, "--report", filepath.Join(work, "missing", "report.csv"), tree(t, "anna/1.jpg", "face:anna"))
	if !errors.Is(r.err, os.ErrNotExist) || engines.count() != 0 {
		t.Errorf("a report that cannot be made: %v, %d engines opened", r.err, engines.count())
	}

	// what did not reach the file is an error when the report is closed
	rep, err := createReport(filepath.Join(work, "report.csv"))
	if err != nil {
		t.Fatal(err)
	}
	rep.file.Close()
	rep.add(&picture{subject: "anna", path: "1.jpg"}, verdict{outcome: enrolled, template: 1})
	if err := rep.close(); err == nil {
		t.Error("a report that could not be written closed without an error")
	}
}

func TestEnrolUsage(t *testing.T) {
	engines := useFake(t)
	for name, c := range map[string]struct {
		args []string
		says string
	}{
		"no store":          {[]string{"photos"}, "--store is required"},
		"no pictures":       {[]string{"--store", "f.db"}, "name the directory that holds the pictures"},
		"two sources":       {[]string{"--store", "f.db", "--list", "staff.csv", "photos"}, "use one"},
		"two directories":   {[]string{"--store", "f.db", "photos", "more"}, `unexpected argument "more"`},
		"flags at the end":  {[]string{"photos", "--store", "f.db"}, "flags go before the directory: put --store in front of photos"},
		"unknown flag":      {[]string{"--store", "f.db", "--fast", "photos"}, "flag provided but not defined"},
		"no workers":        {[]string{"--store", "f.db", "--workers", "0", "photos"}, "--workers is at least 1"},
		"workers in words":  {[]string{"--store", "f.db", "--workers", "many", "photos"}, "invalid value"},
		"negative threads":  {[]string{"--store", "f.db", "--threads", "-2", "photos"}, "cannot be negative"},
		"search threshold":  {[]string{"--store", "f.db", "--search-threshold", "60", "photos"}, "a threshold is a similarity"},
		"verify threshold":  {[]string{"--store", "f.db", "--verify-threshold", "-0.1", "photos"}, "a threshold is a similarity"},
		"templates in text": {[]string{"--store", "f.db", "--max-templates", "few", "photos"}, "invalid value"},
	} {
		r := run(t, environment(), c.args...)
		if !errors.Is(r.err, cli.ErrUsage) {
			t.Errorf("%s: %v", name, r.err)
		}
		if !strings.Contains(r.stderr, c.says) || !strings.Contains(r.stderr, "likeness enrol") {
			t.Errorf("%s: explained with %q, want %q", name, r.stderr, c.says)
		}
		if r.stdout != "" {
			t.Errorf("%s: printed %q", name, r.stdout)
		}
	}

	r := run(t, environment(), "-h")
	if !errors.Is(r.err, flag.ErrHelp) {
		t.Errorf("-h: %v", r.err)
	}
	for _, want := range []string{
		"usage: likeness enrol --store FILE [flags] DIR", "Stop the server first", "-store file", "-list file", "-preset",
		"-workers int", "-replace", "-allow-duplicate", "-allow-mismatch", "-max-templates", "-dry-run", "-report file",
		"LIKENESS_STORE_KEY", "_FILE",
	} {
		if !strings.Contains(r.stderr, want) {
			t.Errorf("the help lacks %q", want)
		}
	}
	if engines.count() != 0 {
		t.Errorf("a wrong command line loaded the models %d times", engines.count())
	}
}

func TestEnrolEngineFlags(t *testing.T) {
	engines := useFake(t)
	path := filepath.Join(t.TempDir(), "faces.db")
	r := run(t, environment(), "--store", path, "--preset", "buffalo_s", "--home", "/opt/likeness", "--threads", "3",
		tree(t, "anna/1.jpg", "face:anna"))
	if r.err != nil {
		t.Fatal(r.err)
	}
	if want := (likeness.Config{Preset: "buffalo_s", Home: "/opt/likeness", Threads: 3}); engines.cfg != want {
		t.Errorf("the engine was opened with %+v, want %+v", engines.cfg, want)
	}
}

func TestEnrolOrderDoesNotDependOnWorkers(t *testing.T) {
	// a batch in which what becomes of a picture depends on the ones before
	var files []string
	for i := range 30 {
		name := fmt.Sprintf("p%02d", i)
		files = append(files,
			name+"/1.jpg", "slow:"+name,
			name+"/2.jpg", "face:"+name+"#2",
			name+"/3.jpg", "slow:"+name,
		)
		switch i % 5 {
		case 0:
			files = append(files, name+"/4.jpg", fmt.Sprintf("face:stranger%02d", i))
		case 1:
			files = append(files, name+"/4.jpg", fmt.Sprintf("face:p%02d", i-1))
		case 2:
			files = append(files, name+"/4.jpg", "reject:blurry")
		case 3:
			files = append(files, name+"/4.jpg", "face:"+name+"#3", name+"/5.jpg", "face:"+name+"#4")
		case 4:
			files = append(files, name+"/4.jpg", "error:corrupt")
		}
	}
	dir := tree(t, files...)
	engines := useFake(t)
	engines.delay = 2 * time.Millisecond

	var reports, summaries [2]string
	for i, workers := range []string{"1", "8"} {
		work := t.TempDir()
		path, report := filepath.Join(work, "faces.db"), filepath.Join(work, "report.csv")
		r := run(t, environment(), "--store", path, "--report", report, "--workers", workers, "--max-templates", "3", dir)
		if r.err != nil {
			t.Fatalf("--workers %s: %v", workers, r.err)
		}
		text, err := os.ReadFile(report)
		if err != nil {
			t.Fatal(err)
		}
		reports[i], summaries[i] = string(text), r.stdout

		peak := engines.last().Peak()
		if workers == "1" && peak != 1 || workers == "8" && (peak < 2 || peak > 8) {
			t.Errorf("--workers %s: %d pictures were analysed at once", workers, peak)
		}
	}
	if reports[0] != reports[1] {
		t.Errorf("the report with one worker:\n%s\nand with eight:\n%s", reports[0], reports[1])
	}
	if summaries[0] != summaries[1] {
		t.Errorf("the summary with one worker:\n%s\nand with eight:\n%s", summaries[0], summaries[1])
	}
	for _, kind := range []outcome{enrolled, alreadyEnrolled, turnedDown, duplicate, mismatch, full, unreadable} {
		if !strings.Contains(reports[0], ","+string(kind)+",") {
			t.Errorf("the batch has no picture that ends as %q", kind)
		}
	}
}

// cliEnv is an environment with no variables set.
func cliEnv(stdout, stderr io.Writer) cli.Env {
	return cli.Env{Getenv: environment(), Stdout: stdout, Stderr: stderr}
}
