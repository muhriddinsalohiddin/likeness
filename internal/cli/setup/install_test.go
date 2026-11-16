package setup

import (
	"archive/tar"
	"archive/zip"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"io"
	"io/fs"
	"net/http"
	"net/http/httptest"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/muhriddinsalohiddin/likeness/internal/cli"
	"github.com/muhriddinsalohiddin/likeness/internal/home"
	"github.com/muhriddinsalohiddin/likeness/internal/models"
)

var ctx = context.Background()

// What the test site hands out: a runtime library, the two models of a
// preset anyone may use, and the two of one for research only.
var (
	library          = bytes.Repeat([]byte("runtime library bytes "), 3000)
	detector         = bytes.Repeat([]byte("detector weights "), 2000)
	embedder         = bytes.Repeat([]byte("embedder weights "), 5000)
	researchDetector = bytes.Repeat([]byte("research detector weights "), 1500)
	researchEmbedder = bytes.Repeat([]byte("research embedder weights "), 4000)
)

const (
	runtimeArchive  = "/releases/v1.0/runtime-1.0.tgz"
	runtimeMember   = "runtime-1.0/lib/libruntime.so.1.0"
	detectorFile    = "/zoo/detector.onnx"
	embedderFile    = "/zoo/embedder.onnx"
	researchArchive = "/releases/research.zip"
)

func zipOf(t *testing.T, members map[string][]byte) []byte {
	t.Helper()
	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	for name, data := range members {
		w, err := zw.Create(name)
		if err != nil {
			t.Fatal(err)
		}
		w.Write(data)
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func tgzOf(t *testing.T, members map[string][]byte) []byte {
	t.Helper()
	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
	tw := tar.NewWriter(gz)
	for name, data := range members {
		if err := tw.WriteHeader(&tar.Header{Name: name, Mode: 0o644, Size: int64(len(data)), Typeflag: tar.TypeReg}); err != nil {
			t.Fatal(err)
		}
		tw.Write(data)
	}
	if err := errors.Join(tw.Close(), gz.Close()); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

// site is a web server with small artifacts on it. It counts what it is
// asked for, and can be told to make trouble.
type site struct {
	*httptest.Server
	files map[string][]byte

	mu      sync.Mutex
	asked   map[string]int
	trouble trouble
}

// trouble sees a request together with the number of times its path has
// been asked for, and reports whether it has answered the request itself.
type trouble func(w http.ResponseWriter, r *http.Request, n int) bool

func newSite(t *testing.T) *site {
	t.Helper()
	s := &site{asked: make(map[string]int), files: map[string][]byte{
		runtimeArchive: tgzOf(t, map[string][]byte{
			"runtime-1.0/README": []byte("not the library"),
			runtimeMember:        library,
		}),
		detectorFile: detector,
		embedderFile: embedder,
		researchArchive: zipOf(t, map[string][]byte{
			"det.onnx": researchDetector, "emb.onnx": researchEmbedder, "notes.txt": []byte("not a model"),
		}),
	}}
	s.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.mu.Lock()
		s.asked[r.URL.Path]++
		n, trouble := s.asked[r.URL.Path], s.trouble
		s.mu.Unlock()
		if trouble != nil && trouble(w, r, n) {
			return
		}
		b, ok := s.files[r.URL.Path]
		if !ok {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Length", strconv.Itoa(len(b)))
		w.Write(b)
	}))
	t.Cleanup(s.Close)
	return s
}

// misbehave lets f answer the requests it wants to.
func (s *site) misbehave(f trouble) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.trouble = f
}

// requests is how often a path has been asked for; with no path, how many
// requests there have been in all.
func (s *site) requests(paths ...string) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	n := 0
	for p, count := range s.asked {
		if len(paths) == 0 || slices.Contains(paths, p) {
			n += count
		}
	}
	return n
}

func digest(b []byte) string {
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}

func (s *site) artifact(file, member string, content []byte) models.Artifact {
	return models.Artifact{URL: s.URL + file, Member: member, SHA256: digest(content), Size: int64(len(content))}
}

func (s *site) runtime() models.Artifact { return s.artifact(runtimeArchive, runtimeMember, library) }

// presets is the catalogue of the site: "default" is two plain files,
// "research" two members of one archive that not everyone may use.
func (s *site) presets() []models.Preset {
	return []models.Preset{
		{
			Name: "default", Description: "two small files anyone may use",
			Detector: &models.Manifest{
				Name: "tiny_detector", File: "detector.onnx", Artifact: s.artifact(detectorFile, "", detector),
				License: "MIT", Source: "the test site",
			},
			Embedder: &models.Manifest{
				Name: "tiny_embedder", File: "embedder.onnx", Artifact: s.artifact(embedderFile, "", embedder),
				License: "Apache-2.0", Source: "the test site",
			},
		},
		{
			Name: "research", Description: "two files for research only",
			Detector: &models.Manifest{
				Name: "lab_detector", File: "det.onnx", Artifact: s.artifact(researchArchive, "det.onnx", researchDetector),
				License: "non-commercial research", NonCommercial: true, Source: "a research group",
			},
			Embedder: &models.Manifest{
				Name: "lab_embedder", File: "emb.onnx", Artifact: s.artifact(researchArchive, "emb.onnx", researchEmbedder),
				License: "non-commercial research", NonCommercial: true, Source: "a research group",
			},
		},
	}
}

// items is everything on the site, to be installed under dir: a member of
// a tgz, two plain files and two members of a zip.
func (s *site) items(dir string) []item {
	lib := item{what: "runtime library", artifact: s.runtime(), dst: filepath.Join(dir, "lib", home.LibraryName())}
	return append([]item{lib}, modelItems(s.presets(), dir)...)
}

// mirror writes the files of the site into a directory, each under the
// last element of its path: what downloading them by hand leaves behind.
func (s *site) mirror(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	for name, b := range s.files {
		if err := os.WriteFile(filepath.Join(dir, path.Base(name)), b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

// pauses notes the waits it is asked for instead of sleeping.
type pauses []time.Duration

func (p *pauses) wait(ctx context.Context, d time.Duration) error {
	*p = append(*p, d)
	return ctx.Err()
}

func newInstaller(s *site) (*installer, *bytes.Buffer, *pauses) {
	log, p := &bytes.Buffer{}, &pauses{}
	return &installer{client: s.Client(), attempts: attempts, wait: p.wait, log: log}, log, p
}

// filesIn lists the files under dir, directories left out.
func filesIn(t *testing.T, dir string) []string {
	t.Helper()
	var files []string
	err := filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		rel, err := filepath.Rel(dir, p)
		files = append(files, filepath.ToSlash(rel))
		return err
	})
	if err != nil && !errors.Is(err, fs.ErrNotExist) {
		t.Fatal(err)
	}
	return files
}

// holds checks that a file has exactly the given content.
func holds(t *testing.T, file string, want []byte) {
	t.Helper()
	got, err := os.ReadFile(file)
	if err != nil {
		t.Error(err)
	} else if !bytes.Equal(got, want) {
		t.Errorf("%s holds %d bytes that are not the %d expected", file, len(got), len(want))
	}
}

func TestInstall(t *testing.T) {
	s := newSite(t)
	dir := t.TempDir()
	items := s.items(dir)
	in, log, waits := newInstaller(s)

	if err := in.install(ctx, items); err != nil {
		t.Fatalf("a fresh install: %v\n%s", err, log)
	}
	holds(t, filepath.Join(dir, "lib", home.LibraryName()), library)
	holds(t, filepath.Join(dir, "models", "detector.onnx"), detector)
	holds(t, filepath.Join(dir, "models", "embedder.onnx"), embedder)
	holds(t, filepath.Join(dir, "models", "det.onnx"), researchDetector)
	holds(t, filepath.Join(dir, "models", "emb.onnx"), researchEmbedder)
	if got := filesIn(t, dir); len(got) != len(items) {
		t.Errorf("the directory holds %v, want the %d installed files and nothing else", got, len(items))
	}
	for _, it := range items {
		for _, want := range []string{
			it.what + ": fetching " + it.artifact.URL + "\n",
			it.what + ": installed as " + it.dst + " (",
		} {
			if !strings.Contains(log.String(), want) {
				t.Errorf("the log lacks %q:\n%s", want, log)
			}
		}
	}
	if strings.Contains(log.String(), "\r") || len(*waits) != 0 {
		t.Errorf("an install without trouble redrew a line or waited (%v):\n%q", *waits, log)
	}

	// a second run finds everything in place and asks the site for nothing
	before := s.requests()
	log.Reset()
	if err := in.install(ctx, items); err != nil {
		t.Fatalf("a second install: %v", err)
	}
	if s.requests() != before {
		t.Errorf("a second install made %d requests", s.requests()-before)
	}
	if got := strings.Count(log.String(), ": already installed\n"); got != len(items) || strings.Contains(log.String(), "fetching") {
		t.Errorf("a second install reported:\n%s", log)
	}
}

func TestInstallReplacesADamagedFile(t *testing.T) {
	s := newSite(t)
	dir := t.TempDir()
	items := s.items(dir)[1:2]
	if err := os.MkdirAll(filepath.Dir(items[0].dst), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(items[0].dst, detector[:len(detector)/2], 0o644); err != nil {
		t.Fatal(err)
	}

	in, log, _ := newInstaller(s)
	in.client = &http.Client{} // a client with no transport of its own will do as well
	if err := in.install(ctx, items); err != nil {
		t.Fatal(err)
	}
	holds(t, items[0].dst, detector)
	if !strings.Contains(log.String(), "replacing it") {
		t.Errorf("the log does not say that a file was replaced:\n%s", log)
	}
}

func TestInstallRefusesAWrongChecksum(t *testing.T) {
	s := newSite(t)
	dir := t.TempDir()
	items := s.items(dir)[1:]
	items[0].artifact.SHA256 = strings.Repeat("0", 64)

	in, _, waits := newInstaller(s)
	err := in.install(ctx, items)
	if !errors.Is(err, models.ErrChecksum) {
		t.Fatalf("got %v, want a checksum error", err)
	}
	for _, want := range []string{items[0].what, "--from", "--list"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("the error lacks %q: %v", want, err)
		}
	}
	if s.requests() != 1 || len(*waits) != 0 {
		t.Errorf("%d requests and %d waits, want one download and no second attempt", s.requests(), len(*waits))
	}
	if got := filesIn(t, dir); len(got) != 0 {
		t.Errorf("files left behind: %v", got)
	}
}

func TestInstallTriesAgain(t *testing.T) {
	s := newSite(t)
	s.misbehave(func(w http.ResponseWriter, _ *http.Request, n int) bool {
		switch n {
		case 1:
			http.Error(w, "busy", http.StatusServiceUnavailable)
		case 2:
			// the line drops half way through
			w.Header().Set("Content-Length", strconv.Itoa(len(embedder)))
			w.Write(embedder[:len(embedder)/2])
		default:
			return false
		}
		return true
	})
	dir := t.TempDir()
	items := s.items(dir)[2:3]

	in, log, waits := newInstaller(s)
	if err := in.install(ctx, items); err != nil {
		t.Fatalf("%v\n%s", err, log)
	}
	holds(t, items[0].dst, embedder)
	if got := s.requests(embedderFile); got != 3 {
		t.Errorf("%d requests, want two that fail and one that works", got)
	}
	if !slices.Equal(*waits, pauses{time.Second, 2 * time.Second}) {
		t.Errorf("waited %v, want 1s and then 2s", *waits)
	}
	for _, want := range []string{"503", "trying again in 1s", "trying again in 2s", "installed as"} {
		if !strings.Contains(log.String(), want) {
			t.Errorf("the log lacks %q:\n%s", want, log)
		}
	}
	if got := filesIn(t, dir); len(got) != 1 {
		t.Errorf("the directory holds %v, want the one model", got)
	}
}

func TestInstallGivesUp(t *testing.T) {
	s := newSite(t)
	s.misbehave(func(w http.ResponseWriter, _ *http.Request, _ int) bool {
		http.Error(w, "down", http.StatusBadGateway)
		return true
	})
	dir := t.TempDir()
	items := s.items(dir)

	in, _, waits := newInstaller(s)
	err := in.install(ctx, items)
	if err == nil {
		t.Fatal("installed from a site that only fails")
	}
	for _, want := range []string{"runtime library", "502", "gave up after 4 attempts", "--from", "likeness setup --list"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("the error lacks %q: %v", want, err)
		}
	}
	if got := s.requests(); got != attempts {
		t.Errorf("%d requests, want %d for the first item and none for the rest", got, attempts)
	}
	if !slices.Equal(*waits, pauses{time.Second, 2 * time.Second, 4 * time.Second}) {
		t.Errorf("waited %v, want pauses that grow", *waits)
	}
	if got := filesIn(t, dir); len(got) != 0 {
		t.Errorf("files left behind: %v", got)
	}
}

func TestInstallTakesNoForAnAnswer(t *testing.T) {
	for _, status := range []int{http.StatusNotFound, http.StatusForbidden, http.StatusProxyAuthRequired} {
		s := newSite(t)
		s.misbehave(func(w http.ResponseWriter, _ *http.Request, _ int) bool {
			http.Error(w, "no", status)
			return true
		})
		dir := t.TempDir()
		in, _, waits := newInstaller(s)
		err := in.install(ctx, s.items(dir))
		if err == nil {
			t.Fatalf("%d: installed all the same", status)
		}
		for _, want := range []string{"runtime library", strconv.Itoa(status), "asking again does not help", "--from"} {
			if !strings.Contains(err.Error(), want) {
				t.Errorf("%d: the error lacks %q: %v", status, want, err)
			}
		}
		if s.requests() != 1 || len(*waits) != 0 {
			t.Errorf("%d: %d requests and %d waits, want one request", status, s.requests(), len(*waits))
		}
		if got := filesIn(t, dir); len(got) != 0 {
			t.Errorf("%d: files left behind: %v", status, got)
		}
	}
}

func TestInstallStopsWhenCancelled(t *testing.T) {
	t.Run("in the middle of a download", func(t *testing.T) {
		s := newSite(t)
		started := make(chan struct{})
		s.misbehave(func(w http.ResponseWriter, r *http.Request, _ int) bool {
			w.Header().Set("Content-Length", strconv.Itoa(len(embedder)))
			w.Write(embedder[:1000])
			w.(http.Flusher).Flush()
			close(started)
			<-r.Context().Done()
			return true
		})
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		go func() {
			<-started
			cancel()
		}()
		dir := t.TempDir()
		in, _, waits := newInstaller(s)
		begun := time.Now()
		err := in.install(ctx, s.items(dir)[2:])
		if !errors.Is(err, context.Canceled) || !strings.Contains(err.Error(), "embedder model tiny_embedder") {
			t.Errorf("got %v, want the cancellation and the item it struck", err)
		}
		if took := time.Since(begun); took > 5*time.Second {
			t.Errorf("stopping took %s", took)
		}
		if s.requests() != 1 || len(*waits) != 0 {
			t.Errorf("%d requests and %d waits after a cancellation", s.requests(), len(*waits))
		}
		if got := filesIn(t, dir); len(got) != 0 {
			t.Errorf("files left behind: %v", got)
		}
	})

	t.Run("between two attempts", func(t *testing.T) {
		s := newSite(t)
		s.misbehave(func(w http.ResponseWriter, _ *http.Request, _ int) bool {
			http.Error(w, "busy", http.StatusServiceUnavailable)
			return true
		})
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		dir := t.TempDir()
		in, _, _ := newInstaller(s)
		in.wait = func(ctx context.Context, _ time.Duration) error {
			cancel()
			return ctx.Err()
		}
		if err := in.install(ctx, s.items(dir)); !errors.Is(err, context.Canceled) {
			t.Errorf("got %v, want the cancellation", err)
		}
		if s.requests() != 1 {
			t.Errorf("%d requests, want none after the cancellation", s.requests())
		}
		if got := filesIn(t, dir); len(got) != 0 {
			t.Errorf("files left behind: %v", got)
		}
	})

	t.Run("before it starts", func(t *testing.T) {
		s := newSite(t)
		ctx, cancel := context.WithCancel(ctx)
		cancel()
		dir := t.TempDir()
		in, _, _ := newInstaller(s)
		if err := in.install(ctx, s.items(dir)); !errors.Is(err, context.Canceled) {
			t.Errorf("got %v, want the cancellation", err)
		}
		if s.requests() != 0 || len(filesIn(t, dir)) != 0 {
			t.Errorf("a cancelled install made %d requests and left %v", s.requests(), filesIn(t, dir))
		}
	})
}

func TestInstallDoesNotRetryWhatTheDiskRefuses(t *testing.T) {
	s := newSite(t)
	dir := t.TempDir()
	// a file sits where the models directory has to be
	if err := os.WriteFile(filepath.Join(dir, "models"), nil, 0o644); err != nil {
		t.Fatal(err)
	}
	in, _, waits := newInstaller(s)
	err := in.install(ctx, s.items(dir)[1:])
	var refused *fs.PathError
	if !errors.As(err, &refused) || !strings.Contains(err.Error(), "detector model tiny_detector") {
		t.Fatalf("got %v, want what the file system said about the item", err)
	}
	if strings.Contains(err.Error(), "--from") {
		t.Errorf("the error sends the user to another machine for a local fault: %v", err)
	}
	if s.requests() != 0 || len(*waits) != 0 {
		t.Errorf("%d requests and %d waits for a directory that cannot be made", s.requests(), len(*waits))
	}
}

func TestInstallOffline(t *testing.T) {
	s := newSite(t)
	offline := func(from string) (*installer, *bytes.Buffer, *pauses) {
		in, log, waits := newInstaller(s)
		in.client = nil
		in.offline(from)
		return in, log, waits
	}

	t.Run("good files", func(t *testing.T) {
		from, dir := s.mirror(t), t.TempDir()
		items := s.items(dir)
		in, log, _ := offline(from)
		if err := in.install(ctx, items); err != nil {
			t.Fatalf("%v\n%s", err, log)
		}
		holds(t, filepath.Join(dir, "lib", home.LibraryName()), library)
		holds(t, filepath.Join(dir, "models", "detector.onnx"), detector)
		holds(t, filepath.Join(dir, "models", "emb.onnx"), researchEmbedder)
		if got := filesIn(t, dir); len(got) != len(items) {
			t.Errorf("the directory holds %v, want the %d installed files", got, len(items))
		}
		if want := "fetching " + filepath.Join(from, "runtime-1.0.tgz"); !strings.Contains(log.String(), want) {
			t.Errorf("the log lacks %q:\n%s", want, log)
		}
	})

	t.Run("a tampered file", func(t *testing.T) {
		from, dir := s.mirror(t), t.TempDir()
		changed := bytes.Clone(detector)
		changed[len(changed)/2] ^= 1
		if err := os.WriteFile(filepath.Join(from, "detector.onnx"), changed, 0o644); err != nil {
			t.Fatal(err)
		}
		in, _, waits := offline(from)
		err := in.install(ctx, s.items(dir)[1:])
		if !errors.Is(err, models.ErrChecksum) {
			t.Fatalf("got %v, want a checksum error", err)
		}
		for _, want := range []string{"detector model tiny_detector", filepath.Join(from, "detector.onnx"), "download it again from " + s.URL + detectorFile} {
			if !strings.Contains(err.Error(), want) {
				t.Errorf("the error lacks %q: %v", want, err)
			}
		}
		if got := filesIn(t, dir); len(got) != 0 || len(*waits) != 0 {
			t.Errorf("a tampered file left %v behind after %d waits", got, len(*waits))
		}
	})

	t.Run("a tampered archive", func(t *testing.T) {
		from, dir := s.mirror(t), t.TempDir()
		archive := filepath.Join(from, "research.zip")
		if err := os.Truncate(archive, 100); err != nil {
			t.Fatal(err)
		}
		in, _, waits := offline(from)
		err := in.install(ctx, s.items(dir)[3:])
		if err == nil {
			t.Fatal("a model was installed out of a broken archive")
		}
		for _, want := range []string{"detector model lab_detector", archive + " cannot be used as it is", "download it again from " + s.URL + researchArchive} {
			if !strings.Contains(err.Error(), want) {
				t.Errorf("the error lacks %q: %v", want, err)
			}
		}
		if got := filesIn(t, dir); len(got) != 0 || len(*waits) != 0 {
			t.Errorf("a broken archive left %v behind after %d waits", got, len(*waits))
		}
	})

	t.Run("a file that was not brought", func(t *testing.T) {
		from, dir := s.mirror(t), t.TempDir()
		if err := os.Remove(filepath.Join(from, "embedder.onnx")); err != nil {
			t.Fatal(err)
		}
		in, _, _ := offline(from)
		err := in.install(ctx, s.items(dir)[1:])
		if !errors.Is(err, fs.ErrNotExist) {
			t.Fatalf("got %v, want a missing file", err)
		}
		for _, want := range []string{"embedder model tiny_embedder", "download " + s.URL + embedderFile + " and put it in " + from} {
			if !strings.Contains(err.Error(), want) {
				t.Errorf("the error lacks %q: %v", want, err)
			}
		}
		// the detector, which was there, is installed all the same
		holds(t, filepath.Join(dir, "models", "detector.onnx"), detector)
	})

	if s.requests() != 0 {
		t.Errorf("installing from a directory made %d requests", s.requests())
	}
}

func TestDirectoryStopsWithItsRequest(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "model.onnx"), detector, 0o644); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, "https://example.com/some/where/model.onnx?download=1", nil)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := directory(dir).RoundTrip(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK || resp.ContentLength != int64(len(detector)) {
		t.Errorf("answered %d with %d bytes to come", resp.StatusCode, resp.ContentLength)
	}
	if _, err := resp.Body.Read(make([]byte, 100)); err != nil {
		t.Errorf("reading the file: %v", err)
	}
	cancel()
	if _, err := resp.Body.Read(make([]byte, 100)); !errors.Is(err, context.Canceled) {
		t.Errorf("reading after the request was cancelled: %v", err)
	}

	req.URL.Path = "/some/where/other.onnx"
	other, err := directory(dir).RoundTrip(req)
	if err == nil {
		other.Body.Close()
	}
	if !errors.Is(err, fs.ErrNotExist) {
		t.Errorf("a file that is not there: %v", err)
	}
}

func TestProgressOnATerminal(t *testing.T) {
	s := newSite(t)
	dir := t.TempDir()
	in, log, _ := newInstaller(s)
	in.redraw = true
	if err := in.install(ctx, s.items(dir)[1:2]); err != nil {
		t.Fatal(err)
	}
	out := log.String()
	if !strings.Contains(out, "\r  100%") {
		t.Errorf("no percentage was drawn:\n%q", out)
	}
	// the line of the percentage is wiped before the next message
	if !strings.Contains(out, "\rdetector model tiny_detector: installed as ") || !strings.HasSuffix(out, ")\n") {
		t.Errorf("the last line does not start on a clean one:\n%q", out)
	}

	var drawn bytes.Buffer
	m := meter{w: &drawn}
	m.update(1_500_000, -1)
	m.update(1_500_001, -1)
	m.update(2_500_000, -1)
	m.clear()
	m.clear()
	if got, want := drawn.String(), "\r  1.5 MB      \r  2.5 MB      \r"+strings.Repeat(" ", 14)+"\r"; got != want {
		t.Errorf("a download of unknown length drew %q, want %q", got, want)
	}
	var idle meter
	idle.clear() // the zero value has nowhere to write, and must not try
}

func TestTerminal(t *testing.T) {
	null, err := os.OpenFile(os.DevNull, os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer null.Close()
	file, err := os.Create(filepath.Join(t.TempDir(), "log"))
	if err != nil {
		t.Fatal(err)
	}
	if !cli.Terminal(null) {
		t.Errorf("%s, a character device, was not taken for a terminal", os.DevNull)
	}
	if cli.Terminal(file) || cli.Terminal(&bytes.Buffer{}) || cli.Terminal(io.Discard) {
		t.Error("a file or a buffer was taken for a terminal")
	}
	file.Close()
	if cli.Terminal(file) {
		t.Error("a closed file was taken for a terminal")
	}
}

func TestSize(t *testing.T) {
	for n, want := range map[int64]string{
		0: "0 kB", 232_589: "233 kB", 999_499: "999 kB", 1_000_000: "1.0 MB", 38_696_353: "38.7 MB", 174_383_860: "174.4 MB",
	} {
		if got := size(n); got != want {
			t.Errorf("%d bytes: %q, want %q", n, got, want)
		}
	}
}

func TestFileName(t *testing.T) {
	for rawURL, want := range map[string]string{
		"https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx": "face_detection_yunet_2023mar.onnx",
		"https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip":                             "buffalo_s.zip",
		"https://example.com/files/runtime.tgz?token=abc#top":                                                         "runtime.tgz",
		"runtime.tgz":                          "runtime.tgz",
		"https://example.com/a/%zz/model.onnx": "model.onnx",
	} {
		if got := fileName(rawURL); got != want {
			t.Errorf("%s: %q, want %q", rawURL, got, want)
		}
	}
}

func TestWait(t *testing.T) {
	if err := wait(ctx, time.Millisecond); err != nil {
		t.Errorf("a pause that ran its course: %v", err)
	}
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	begun := time.Now()
	if err := wait(cancelled, time.Minute); !errors.Is(err, context.Canceled) {
		t.Errorf("a pause that was cancelled: %v", err)
	}
	if took := time.Since(begun); took > 10*time.Second {
		t.Errorf("a cancelled pause took %s", took)
	}
}

func TestNewClient(t *testing.T) {
	c := newClient()
	tr, ok := c.Transport.(*http.Transport)
	if !ok {
		t.Fatalf("the transport is a %T", c.Transport)
	}
	if c.Timeout != 0 {
		t.Errorf("downloads are cut off after %s", c.Timeout)
	}
	if tr.ResponseHeaderTimeout <= 0 || tr.TLSHandshakeTimeout <= 0 {
		t.Error("a server that says nothing is waited for without end")
	}
	if tr.Proxy == nil {
		t.Error("the proxy of the environment is ignored")
	}
}
