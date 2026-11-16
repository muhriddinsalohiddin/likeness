package setup

import (
	"bytes"
	"errors"
	"flag"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"

	"github.com/muhriddinsalohiddin/likeness"
	"github.com/muhriddinsalohiddin/likeness/internal/cli"
	"github.com/muhriddinsalohiddin/likeness/internal/enginetest"
	"github.com/muhriddinsalohiddin/likeness/internal/home"
	"github.com/muhriddinsalohiddin/likeness/internal/models"
)

// vars makes a getenv out of pairs of names and values.
func vars(pairs ...string) func(string) string {
	set := make(map[string]string)
	for i := 0; i < len(pairs); i += 2 {
		set[pairs[i]] = pairs[i+1]
	}
	return func(name string) string { return set[name] }
}

// source is the command as it would be if the site were the catalogue.
func (s *site) source() (source, *pauses) {
	waits := &pauses{}
	return source{
		presets: s.presets(),
		runtime: func() (models.Artifact, bool) { return s.runtime(), true },
		client:  s.Client(),
		wait:    waits.wait,
	}, waits
}

// try runs the command and returns what it printed.
func (s source) try(getenv func(string) string, args ...string) (stdout, stderr string, err error) {
	var out, complaints bytes.Buffer
	err = s.run(ctx, args, cli.Env{Getenv: getenv, Stdout: &out, Stderr: &complaints})
	return out.String(), complaints.String(), err
}

// engines stands in for the engine and notes what was asked of it.
type engines struct {
	// fail, if set, is what opening an engine comes to.
	fail   error
	asked  []likeness.Config
	opened []*enginetest.Fake
}

// useEngines puts stand-ins in place of the real engine for the length of
// a test.
func useEngines(t *testing.T) *engines {
	t.Helper()
	e := &engines{}
	real := cli.OpenEngine
	cli.OpenEngine = func(cfg likeness.Config) (cli.Engine, error) {
		e.asked = append(e.asked, cfg)
		if e.fail != nil {
			return nil, e.fail
		}
		fake := &enginetest.Fake{}
		e.opened = append(e.opened, fake)
		return fake, nil
	}
	t.Cleanup(func() { cli.OpenEngine = real })
	return e
}

func TestSetup(t *testing.T) {
	s := newSite(t)
	engine := useEngines(t)
	dir := filepath.Join(t.TempDir(), "home")
	cmd, _ := s.source()

	stdout, stderr, err := cmd.try(vars(), "--home", dir)
	if err != nil {
		t.Fatalf("setup: %v\n%s", err, stderr)
	}
	holds(t, filepath.Join(dir, "lib", home.LibraryName()), library)
	holds(t, filepath.Join(dir, "models", "detector.onnx"), detector)
	holds(t, filepath.Join(dir, "models", "embedder.onnx"), embedder)
	if got := filesIn(t, dir); len(got) != 3 {
		t.Errorf("the home holds %v, want the library and two models", got)
	}
	for _, want := range []string{
		"ONNX Runtime " + models.RuntimeVersion + " library: fetching " + s.URL + runtimeArchive,
		"detector model tiny_detector: installed as " + filepath.Join(dir, "models", "detector.onnx"),
		"embedder model tiny_embedder: installed as ",
	} {
		if !strings.Contains(stderr, want) {
			t.Errorf("the progress lacks %q:\n%s", want, stderr)
		}
	}

	// the installation was loaded once, from where it was put, and let go
	if len(engine.asked) != 1 || engine.asked[0] != (likeness.Config{Preset: "default", Home: dir}) {
		t.Errorf("the engine was opened with %+v", engine.asked)
	}
	if len(engine.opened) != 1 || !engine.opened[0].Closed() {
		t.Error("the engine that checked the installation was left open")
	}
	for _, want := range []string{"default: ready", "ONNX Runtime 0.0.0", "fake-detector", "fake-embedder", "--home " + dir, "LIKENESS_HOME"} {
		if !strings.Contains(stdout, want) {
			t.Errorf("the report lacks %q:\n%s", want, stdout)
		}
	}

	// a second run has nothing to fetch, and checks again
	before := s.requests()
	stdout, stderr, err = cmd.try(vars(), "--home", dir)
	if err != nil {
		t.Fatalf("a second setup: %v\n%s", err, stderr)
	}
	if s.requests() != before {
		t.Errorf("a second setup made %d requests", s.requests()-before)
	}
	if strings.Count(stderr, ": already installed\n") != 3 || !strings.Contains(stdout, "default: ready") || len(engine.asked) != 2 {
		t.Errorf("a second setup said:\n%s%s", stderr, stdout)
	}
}

func TestSetupHome(t *testing.T) {
	s := newSite(t)
	useEngines(t)
	cmd, _ := s.source()
	fromEnv, fromFlag := filepath.Join(t.TempDir(), "env"), filepath.Join(t.TempDir(), "flag")

	stdout, stderr, err := cmd.try(vars("LIKENESS_HOME", fromEnv))
	if err != nil {
		t.Fatalf("setup: %v\n%s", err, stderr)
	}
	if got := filesIn(t, fromEnv); len(got) != 3 {
		t.Errorf("$LIKENESS_HOME holds %v, want the library and two models", got)
	}
	if strings.Contains(stdout, "--home") {
		t.Errorf("the report explains --home to someone who did not use it:\n%s", stdout)
	}

	// the flag wins
	if _, stderr, err := cmd.try(vars("LIKENESS_HOME", fromEnv), "--home", fromFlag, "--preset", " default , default "); err != nil {
		t.Fatalf("setup: %v\n%s", err, stderr)
	}
	if got := filesIn(t, fromFlag); len(got) != 3 {
		t.Errorf("--home holds %v, want the library and two models", got)
	}
}

func TestSetupSeveralPresets(t *testing.T) {
	s := newSite(t)
	engine := useEngines(t)
	dir := t.TempDir()
	cmd, _ := s.source()

	stdout, stderr, err := cmd.try(vars("LIKENESS_HOME", dir), "--preset", "default,research", "--accept-licence")
	if err != nil {
		t.Fatalf("setup: %v\n%s", err, stderr)
	}
	holds(t, filepath.Join(dir, "models", "det.onnx"), researchDetector)
	holds(t, filepath.Join(dir, "models", "emb.onnx"), researchEmbedder)
	if got := filesIn(t, dir); len(got) != 5 {
		t.Errorf("the home holds %v, want the library and four models", got)
	}
	if got := s.requests(runtimeArchive); got != 1 {
		t.Errorf("the library was asked for %d times", got)
	}
	if len(engine.asked) != 2 || engine.asked[0].Preset != "default" || engine.asked[1].Preset != "research" {
		t.Errorf("the engine was opened with %+v", engine.asked)
	}
	if !strings.Contains(stdout, "default: ready") || !strings.Contains(stdout, "research: ready") {
		t.Errorf("the report:\n%s", stdout)
	}
	if strings.Contains(stderr, "non-commercial") {
		t.Errorf("the licence was explained again to someone who accepted it:\n%s", stderr)
	}
}

func TestSetupCheck(t *testing.T) {
	s := newSite(t)
	engine := useEngines(t)
	dir := t.TempDir()
	cmd, _ := s.source()

	engine.fail = errors.New("the library is for another processor")
	stdout, _, err := cmd.try(vars(), "--home", dir)
	if !errors.Is(err, engine.fail) {
		t.Fatalf("got %v, want what the engine said", err)
	}
	for _, want := range []string{"default preset", dir, "does not load"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("the error lacks %q: %v", want, err)
		}
	}
	if strings.Contains(stdout, "ready") {
		t.Errorf("an installation that does not load was called ready:\n%s", stdout)
	}
	if got := filesIn(t, dir); len(got) != 3 {
		t.Errorf("the home holds %v; a failed check must not undo the installation", got)
	}

	// without the check the engine is not touched
	stdout, stderr, err := cmd.try(vars(), "--home", dir, "--check=false")
	if err != nil {
		t.Fatalf("setup --check=false: %v\n%s", err, stderr)
	}
	if len(engine.asked) != 1 || !strings.Contains(stdout, "default: installed in "+dir+", not checked") {
		t.Errorf("the engine was opened %d times in all; the report:\n%s", len(engine.asked), stdout)
	}
}

func TestSetupLicence(t *testing.T) {
	s := newSite(t)
	engine := useEngines(t)
	dir := t.TempDir()
	cmd, _ := s.source()

	_, stderr, err := cmd.try(vars(), "--home", dir, "--preset", "research")
	if !errors.Is(err, cli.ErrUsage) {
		t.Fatalf("got %v, want a refusal", err)
	}
	for _, want := range []string{
		"research preset", "non-commercial research", "lab_detector", "lab_embedder", "a research group",
		s.URL + researchArchive, "--accept-licence is required for research",
	} {
		if !strings.Contains(stderr, want) {
			t.Errorf("the notice lacks %q:\n%s", want, stderr)
		}
	}
	if s.requests() != 0 || len(filesIn(t, dir)) != 0 || len(engine.asked) != 0 {
		t.Errorf("before the licence was accepted there were %d requests, the files %v and %d engines",
			s.requests(), filesIn(t, dir), len(engine.asked))
	}

	// one such preset among several holds up the whole command
	if _, stderr, err := cmd.try(vars(), "--home", dir, "--preset", "default,research"); !errors.Is(err, cli.ErrUsage) || s.requests() != 0 {
		t.Errorf("got %v after %d requests:\n%s", err, s.requests(), stderr)
	}
	if _, stderr, err := cmd.try(vars(), "--home", dir, "--preset", "research", "--accept-licence"); err != nil {
		t.Fatalf("with the licence accepted: %v\n%s", err, stderr)
	}
	// both models come out of one archive
	if got := filesIn(t, dir); len(got) != 3 {
		t.Errorf("the home holds %v, want the library and two models", got)
	}
}

func TestSetupList(t *testing.T) {
	s := newSite(t)
	engine := useEngines(t)
	dir := filepath.Join(t.TempDir(), "home")
	cmd, _ := s.source()
	lib := filepath.Join(dir, "lib", home.LibraryName())

	stdout, stderr, err := cmd.try(vars(), "--list", "--home", dir, "--preset", "default,research")
	if err != nil || stderr != "" {
		t.Fatalf("setup --list: %v\n%s", err, stderr)
	}
	for _, want := range []string{
		"likeness home: " + dir,
		"ONNX Runtime " + models.RuntimeVersion + " library: missing",
		"url      " + s.URL + runtimeArchive,
		"member   " + runtimeMember,
		"size     66000 bytes (66 kB)",
		"sha256   " + digest(library),
		"goes to  " + lib,
		"detector model tiny_detector: missing",
		"licence  MIT, from the test site",
		"goes to  " + filepath.Join(dir, "models", "detector.onnx"),
		"embedder model lab_embedder: missing",
		"member   emb.onnx",
		"licence  non-commercial research, from a research group",
		"curl -fLO " + s.URL + runtimeArchive + "\n",
		"curl -fLO " + s.URL + detectorFile + "\n",
		"--from DIRECTORY",
	} {
		if !strings.Contains(stdout, want) {
			t.Errorf("the list lacks %q:\n%s", want, stdout)
		}
	}
	if got := strings.Count(stdout, "curl -fLO "); got != 4 {
		t.Errorf("%d downloads listed, want four: the archive with two models in it once\n%s", got, stdout)
	}
	if strings.Count(stdout, ": missing\n") != 5 || strings.Contains(stdout, ": installed\n") {
		t.Errorf("an empty home was listed as:\n%s", stdout)
	}
	if _, err := os.Stat(dir); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("listing made the home directory: %v", err)
	}

	// what is there is told apart from what is not, and from what is wrong
	if _, stderr, err := cmd.try(vars(), "--home", dir); err != nil {
		t.Fatalf("setup: %v\n%s", err, stderr)
	}
	if err := os.WriteFile(lib, []byte("not a library"), 0o644); err != nil {
		t.Fatal(err)
	}
	before, opened := s.requests(), len(engine.asked)
	stdout, _, err = cmd.try(vars(), "--list", "--home", dir, "--preset", "default,research")
	if err != nil {
		t.Fatalf("setup --list: %v", err)
	}
	for _, want := range []string{
		"library: damaged, setup will replace it\n",
		"detector model tiny_detector: installed\n",
		"embedder model tiny_embedder: installed\n",
		"detector model lab_detector: missing\n",
	} {
		if !strings.Contains(stdout, want) {
			t.Errorf("the list lacks %q:\n%s", want, stdout)
		}
	}
	if s.requests() != before || len(engine.asked) != opened {
		t.Error("listing asked the site for something or opened the engine")
	}
	holds(t, lib, []byte("not a library"))

	// a home that cannot be looked into is not passed off as an empty one
	blocked := t.TempDir()
	if err := os.WriteFile(filepath.Join(blocked, "models"), nil, 0o644); err != nil {
		t.Fatal(err)
	}
	stdout, _, err = cmd.try(vars(), "--list", "--home", blocked)
	if err != nil || !strings.Contains(stdout, "detector model tiny_detector: unreadable: ") {
		t.Errorf("the list of a home with a file in the place of a directory: %v\n%s", err, stdout)
	}
}

func TestModelItems(t *testing.T) {
	presets := newSite(t).presets()
	// a third preset that shares its detector with the first
	mixed := presets[1]
	mixed.Name, mixed.Detector = "mixed", presets[0].Detector
	items := modelItems(append(presets, mixed), "home")

	got := make([]string, len(items))
	for i, it := range items {
		got[i] = it.what + " -> " + filepath.ToSlash(it.dst)
	}
	want := []string{
		"detector model tiny_detector -> home/models/detector.onnx",
		"embedder model tiny_embedder -> home/models/embedder.onnx",
		"detector model lab_detector -> home/models/det.onnx",
		"embedder model lab_embedder -> home/models/emb.onnx",
	}
	if !slices.Equal(got, want) {
		t.Errorf("the items are\n%s\nwant each model once, in the order of the presets:\n%s", strings.Join(got, "\n"), strings.Join(want, "\n"))
	}
}

func TestSetupFrom(t *testing.T) {
	s := newSite(t)
	useEngines(t)
	cmd, waits := s.source()
	from := s.mirror(t)

	dir := t.TempDir()
	stdout, stderr, err := cmd.try(vars(), "--home", dir, "--from", from, "--preset", "default,research", "--accept-licence")
	if err != nil {
		t.Fatalf("setup --from: %v\n%s", err, stderr)
	}
	holds(t, filepath.Join(dir, "lib", home.LibraryName()), library)
	holds(t, filepath.Join(dir, "models", "embedder.onnx"), embedder)
	holds(t, filepath.Join(dir, "models", "det.onnx"), researchDetector)
	if got := filesIn(t, dir); len(got) != 5 || !strings.Contains(stdout, "research: ready") {
		t.Errorf("the home holds %v; the report:\n%s", got, stdout)
	}

	// a file that was changed on the way is refused
	changed := bytes.Clone(embedder)
	changed[0] ^= 0x20
	if err := os.WriteFile(filepath.Join(from, "embedder.onnx"), changed, 0o644); err != nil {
		t.Fatal(err)
	}
	dir = t.TempDir()
	_, _, err = cmd.try(vars(), "--home", dir, "--from", from)
	if !errors.Is(err, models.ErrChecksum) || !strings.Contains(err.Error(), filepath.Join(from, "embedder.onnx")) {
		t.Fatalf("got %v, want a checksum error that names the file", err)
	}
	if got := filesIn(t, dir); len(got) != 2 {
		t.Errorf("the home holds %v, want the two files that were right and not the third", got)
	}

	if s.requests() != 0 || len(*waits) != 0 {
		t.Errorf("--from made %d requests and waited %d times", s.requests(), len(*waits))
	}
}

func TestSetupWithoutARuntimeBuild(t *testing.T) {
	s := newSite(t)
	useEngines(t)
	cmd, _ := s.source()
	cmd.runtime = func() (models.Artifact, bool) { return models.Artifact{}, false }
	platform := runtime.GOOS + "/" + runtime.GOARCH

	dir := t.TempDir()
	_, _, err := cmd.try(vars(), "--home", dir)
	if err == nil || errors.Is(err, cli.ErrUsage) {
		t.Fatalf("got %v, want an error about the platform", err)
	}
	for _, want := range []string{platform, "LIKENESS_ORT_LIBRARY"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("the error lacks %q: %v", want, err)
		}
	}
	if s.requests() != 0 || len(filesIn(t, dir)) != 0 {
		t.Errorf("%d requests and the files %v on a platform without a library", s.requests(), filesIn(t, dir))
	}

	// the list still says what else is needed
	stdout, _, err := cmd.try(vars(), "--home", dir, "--list")
	if err != nil {
		t.Fatalf("setup --list: %v", err)
	}
	for _, want := range []string{"ONNX Runtime library: the catalogue has no ONNX Runtime build for " + platform, "LIKENESS_ORT_LIBRARY", "detector model tiny_detector: missing"} {
		if !strings.Contains(stdout, want) {
			t.Errorf("the list lacks %q:\n%s", want, stdout)
		}
	}

	// with a library of the user's own, only the models are installed
	own := filepath.Join(t.TempDir(), "libonnxruntime.so")
	stdout, stderr, err := cmd.try(vars("LIKENESS_ORT_LIBRARY", own), "--home", dir)
	if err != nil {
		t.Fatalf("setup with a library of one's own: %v\n%s", err, stderr)
	}
	if got := filesIn(t, dir); len(got) != 2 || s.requests(runtimeArchive) != 0 {
		t.Errorf("the home holds %v after %d requests for the library, want two models and none", got, s.requests(runtimeArchive))
	}
	if want := "ONNX Runtime library: LIKENESS_ORT_LIBRARY names " + own; !strings.Contains(stderr, want) || !strings.Contains(stdout, "default: ready") {
		t.Errorf("the progress lacks %q, or the report the check:\n%s%s", want, stderr, stdout)
	}

	// and so it is where the catalogue does have one
	cmd, _ = s.source()
	stdout, _, err = cmd.try(vars("LIKENESS_ORT_LIBRARY", own), "--home", dir, "--list")
	if err != nil || !strings.Contains(stdout, "LIKENESS_ORT_LIBRARY names "+own) || strings.Contains(stdout, runtimeArchive) {
		t.Errorf("the list with a library of one's own: %v\n%s", err, stdout)
	}
}

func TestSetupUsage(t *testing.T) {
	s := newSite(t)
	engine := useEngines(t)
	cmd, _ := s.source()
	file := filepath.Join(t.TempDir(), "a-file")
	if err := os.WriteFile(file, nil, 0o644); err != nil {
		t.Fatal(err)
	}

	for name, c := range map[string]struct {
		args []string
		says string
	}{
		"stray argument":  {[]string{"default"}, `unexpected argument "default"`},
		"unknown flag":    {[]string{"--force"}, "flag provided but not defined"},
		"unknown preset":  {[]string{"--preset", "buffalo_xl"}, `unknown preset "buffalo_xl"; the presets are default, research`},
		"one of two":      {[]string{"--preset", "default,reserch"}, `unknown preset "reserch"`},
		"empty preset":    {[]string{"--preset", ""}, "--preset takes one name"},
		"empty name":      {[]string{"--preset", "default,,research"}, "--preset takes one name"},
		"no such --from":  {[]string{"--from", filepath.Join(file, "missing")}, "is not a directory"},
		"--from a file":   {[]string{"--from", file}, "--from " + file + " is not a directory"},
		"--check a word":  {[]string{"--check=perhaps"}, "invalid boolean value"},
		"--home no value": {[]string{"--home"}, "flag needs an argument"},
	} {
		_, stderr, err := cmd.try(vars("LIKENESS_HOME", t.TempDir()), c.args...)
		if !errors.Is(err, cli.ErrUsage) {
			t.Errorf("%s: %v", name, err)
		}
		if !strings.Contains(stderr, c.says) {
			t.Errorf("%s: explained with %q, want %q", name, stderr, c.says)
		}
	}

	stdout, stderr, err := cmd.try(vars(), "-h")
	if !errors.Is(err, flag.ErrHelp) || stdout != "" {
		t.Errorf("-h: %v, with %q on the standard output", err, stdout)
	}
	for _, want := range []string{
		"usage: likeness setup", "-preset names", "-home directory", "-accept-licence", "-from directory", "-list", "-check",
		"research   two files for research only", "LIKENESS_HOME", "LIKENESS_ORT_LIBRARY", "HTTPS_PROXY",
	} {
		if !strings.Contains(stderr, want) {
			t.Errorf("the help lacks %q:\n%s", want, stderr)
		}
	}
	if s.requests() != 0 || len(engine.asked) != 0 {
		t.Errorf("a wrong command line led to %d requests and %d engines", s.requests(), len(engine.asked))
	}
}

// TestRun goes as far as the real catalogue allows without the network.
func TestRun(t *testing.T) {
	engine := useEngines(t)
	run := func(getenv func(string) string, args ...string) (string, string, error) {
		var stdout, stderr bytes.Buffer
		err := Run(ctx, args, cli.Env{Getenv: getenv, Stdout: &stdout, Stderr: &stderr})
		return stdout.String(), stderr.String(), err
	}
	dir := filepath.Join(t.TempDir(), "home")

	stdout, stderr, err := run(vars("LIKENESS_HOME", dir), "--list")
	if err != nil || stderr != "" {
		t.Fatalf("setup --list: %v\n%s", err, stderr)
	}
	def, _ := models.LookupPreset("default")
	for _, want := range []string{
		"likeness home: " + dir,
		"detector model " + def.Detector.Name + ": missing",
		"embedder model " + def.Embedder.Name + ": missing",
		"curl -fLO " + def.Detector.Artifact.URL + "\n",
		"sha256   " + def.Embedder.Artifact.SHA256,
		"goes to  " + filepath.Join(dir, "models", def.Embedder.File),
	} {
		if !strings.Contains(stdout, want) {
			t.Errorf("the list lacks %q:\n%s", want, stdout)
		}
	}
	if library, ok := models.HostRuntime(); ok && !strings.Contains(stdout, "curl -fLO "+library.URL+"\n") {
		t.Errorf("the list lacks the runtime library of this platform:\n%s", stdout)
	}

	// with nothing said about it, the home is where the engine looks
	stdout, _, err = run(vars(), "--list", "--preset", "buffalo_s")
	if err != nil || !strings.Contains(stdout, "likeness home: "+home.Dir()+"\n") {
		t.Errorf("the default home: %v\n%s", err, stdout)
	}
	if got := strings.Count(stdout, "curl -fLO https://github.com/deepinsight/insightface/"); got != 1 {
		t.Errorf("the archive that holds both models is listed %d times:\n%s", got, stdout)
	}

	for name, c := range map[string]struct {
		args []string
		want error
		says string
	}{
		"research only":  {[]string{"--preset", "buffalo_l"}, cli.ErrUsage, "--accept-licence is required for buffalo_l"},
		"unknown preset": {[]string{"--preset", "buffalo_xl"}, cli.ErrUsage, "the presets are default, buffalo_s, buffalo_l"},
		"help":           {[]string{"-h"}, flag.ErrHelp, "buffalo_s"},
	} {
		_, stderr, err := run(vars("LIKENESS_HOME", dir), c.args...)
		if !errors.Is(err, c.want) || !strings.Contains(stderr, c.says) {
			t.Errorf("%s: %v, explained with %q, want %q", name, err, stderr, c.says)
		}
	}
	if _, err := os.Stat(dir); !errors.Is(err, os.ErrNotExist) || len(engine.asked) != 0 {
		t.Errorf("the home was made (%v) or the engine opened (%d times) by commands that install nothing", err, len(engine.asked))
	}
}
