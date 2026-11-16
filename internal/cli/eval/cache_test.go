package eval

import (
	"bytes"
	"context"
	"errors"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestCacheKeeps(t *testing.T) {
	file := filepath.Join(t.TempDir(), "vectors.cache")
	made := basis{Fingerprint: "sface:1", Detector: "yunet", Policy: "search"}
	anna := picture{name: "anna/1.jpg", size: 100, modTime: 5}
	boris := picture{name: "boris/1.jpg", size: 200, modTime: 6}
	clara := picture{name: "clara/1.jpg", size: 300, modTime: 7}

	c, err := openCache(file, made, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	if _, known := c.lookup(anna); known {
		t.Error("a new cache knows a picture")
	}
	if err := c.save(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(file); !errors.Is(err, fs.ErrNotExist) {
		t.Errorf("a cache with nothing new in it was written: %v", err)
	}

	c.keep(anna, outcome{Vector: []float32{0.6, -0.8}})
	c.keep(boris, outcome{Reject: "blurry"})
	c.keep(clara, outcome{Failure: "corrupt_image"})
	if err := c.save(); err != nil {
		t.Fatal(err)
	}
	// vectors of faces are not for everybody to read
	if at, err := os.Stat(file); err != nil || at.Mode().Perm() != 0o600 {
		t.Errorf("the cache file: %v, %v", at.Mode(), err)
	}
	if left, _ := os.ReadDir(filepath.Dir(file)); len(left) != 1 {
		t.Errorf("%d files next to the cache, want none", len(left)-1)
	}

	c, err = openCache(file, made, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []struct {
		picture picture
		outcome outcome
	}{
		{anna, outcome{Vector: []float32{0.6, -0.8}}},
		{boris, outcome{Reject: "blurry"}},
		{clara, outcome{Failure: "corrupt_image"}},
	} {
		if got, known := c.lookup(want.picture); !known || !reflect.DeepEqual(got, want.outcome) {
			t.Errorf("%s came back as %+v (%v), want %+v", want.picture.name, got, known, want.outcome)
		}
	}
	// another file under the same name is another picture
	for name, changed := range map[string]picture{
		"another size":   {name: "anna/1.jpg", size: 101, modTime: 5},
		"another time":   {name: "anna/1.jpg", size: 100, modTime: 4},
		"another name":   {name: "anna/2.jpg", size: 100, modTime: 5},
		"another person": {name: "boris/1.jpg", size: 100, modTime: 5},
	} {
		if got, known := c.lookup(changed); known {
			t.Errorf("%s: taken for the picture in the cache, %+v", name, got)
		}
	}

	// made with something else, the cache is left alone and said to be
	for name, c := range map[string]struct {
		other basis
		says  string
	}{
		"another embedder": {basis{Fingerprint: "arcface:2", Detector: "yunet", Policy: "search"}, "was made with other models (yunet, sface:1)"},
		"another detector": {basis{Fingerprint: "sface:1", Detector: "scrfd", Policy: "search"}, "was made with other models (yunet, sface:1)"},
		"another policy":   {basis{Fingerprint: "sface:1", Detector: "yunet", Policy: "enrol"}, "was made with another policy"},
	} {
		var notes bytes.Buffer
		fresh, err := openCache(file, c.other, &notes)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if _, known := fresh.lookup(anna); known || len(fresh.entries) != 0 {
			t.Errorf("%s: the cache was used all the same", name)
		}
		if got := notes.String(); !strings.HasPrefix(got, "likeness eval: the cache "+file) || !strings.Contains(got, c.says) ||
			!strings.HasSuffix(got, "it is not used and will be replaced\n") {
			t.Errorf("%s: noted as %q", name, got)
		}
	}
}

func TestCacheRefuses(t *testing.T) {
	dir := t.TempDir()
	made := basis{Fingerprint: "sface:1"}

	// no file to keep anything in: nothing is kept and nothing fails
	c, err := openCache("", made, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	c.keep(picture{name: "anna/1.jpg"}, outcome{Reject: "blurry"})
	if err := c.save(); err != nil {
		t.Errorf("saving a cache that has no file: %v", err)
	}

	// an empty file is a cache with nothing in it yet
	empty := filepath.Join(dir, "empty")
	write(t, empty, "")
	if c, err := openCache(empty, made, io.Discard); err != nil || len(c.entries) != 0 {
		t.Errorf("an empty file: %v", err)
	}

	// somebody else's file is not written over
	for name, text := range map[string]string{
		"notes":               "the pictures of the summer of 2019\n",
		"short":               "likeness",
		"another version":     "likeness eval cache 0\nwhatever it held",
		"almost the mark":     strings.ToUpper(cacheMark),
		"a picture, as it is": "\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00",
	} {
		file := filepath.Join(dir, "foreign")
		write(t, file, text)
		if _, err := openCache(file, made, io.Discard); err == nil || !strings.Contains(err.Error(), file+" is not a cache of likeness eval") ||
			!strings.Contains(err.Error(), "name another file for --cache") {
			t.Errorf("%s: %v", name, err)
		}
	}
	if _, err := openCache(dir, made, io.Discard); err == nil {
		t.Error("a directory was opened as a cache")
	}

	// a cache that was cut short
	whole := filepath.Join(dir, "whole")
	c, _ = openCache(whole, made, io.Discard)
	c.keep(picture{name: "anna/1.jpg"}, outcome{Vector: make([]float32, 64)})
	if err := c.save(); err != nil {
		t.Fatal(err)
	}
	doc, _ := os.ReadFile(whole)
	for _, keep := range []int{len(cacheMark), len(cacheMark) + 10, len(doc) - 20} {
		cut := filepath.Join(dir, "cut")
		if err := os.WriteFile(cut, doc[:keep], 0o600); err != nil {
			t.Fatal(err)
		}
		if _, err := openCache(cut, made, io.Discard); err == nil || !strings.Contains(err.Error(), "the cache is damaged; remove it") {
			t.Errorf("a cache cut off after %d of %d bytes: %v", keep, len(doc), err)
		}
	}

	// nowhere to write
	c, _ = openCache(filepath.Join(dir, "nowhere", "cache"), made, io.Discard)
	c.keep(picture{name: "anna/1.jpg"}, outcome{Reject: "blurry"})
	if err := c.save(); !errors.Is(err, fs.ErrNotExist) {
		t.Errorf("saving into a directory that is not there: %v", err)
	}
	// the name is taken by a directory: the temporary file does not stay
	taken := filepath.Join(dir, "taken")
	if err := os.Mkdir(taken, 0o755); err != nil {
		t.Fatal(err)
	}
	c = &cache{file: taken, basis: made, entries: map[string]entry{}}
	c.keep(picture{name: "anna/1.jpg"}, outcome{Reject: "blurry"})
	if err := c.save(); err == nil {
		t.Error("a cache was saved over a directory")
	}
	if left, _ := filepath.Glob(filepath.Join(dir, "*.tmp")); len(left) != 0 {
		t.Errorf("temporary files were left behind: %v", left)
	}
}

func TestEvalCache(t *testing.T) {
	fakes := useFake(t)
	dir := evaluated(t)
	cache := filepath.Join(t.TempDir(), "vectors.cache")
	ctx := context.Background()

	// analysed counts the pictures the engine is given by one run.
	analysed := func(args ...string) (n int, stdout, stderr string) {
		t.Helper()
		before := fakes.calls()
		stdout, stderr, err := eval(ctx, append(args, "--cache", cache, dir)...)
		if err != nil {
			t.Fatalf("eval %v: %v\n%s", args, err, stderr)
		}
		return fakes.calls() - before, stdout, stderr
	}

	n, first, stderr := analysed()
	if n != 85 || !strings.Contains(stderr, "85 pictures analysed in ") {
		t.Fatalf("the first run analysed %d pictures, want 85:\n%s", n, stderr)
	}
	// the second run has nothing to analyse: not the good pictures, not
	// the ones turned down, not the one that cannot be read
	n, second, stderr := analysed()
	if n != 0 || !strings.Contains(stderr, "85 pictures, all from the cache\n") {
		t.Errorf("the second run analysed %d pictures:\n%s", n, stderr)
	}
	if second != first {
		t.Errorf("the report from the cache differs.\nfirst:\n%s\nsecond:\n%s", first, second)
	}

	// a quick look at a few people leaves the rest in the cache
	if n, _, _ := analysed("--max-people", "3", "--max-per-person", "2"); n != 0 {
		t.Errorf("a run over a part of the cached pictures analysed %d", n)
	}
	if n, _, _ := analysed(); n != 0 {
		t.Errorf("the run after a partial one analysed %d pictures", n)
	}

	// only what changed is analysed again: a file touched, a file
	// rewritten, a file added
	touched := filepath.Join(dir, "p05", "1.jpg")
	if err := os.Chtimes(touched, time.Time{}, time.Now().Add(-time.Hour)); err != nil {
		t.Fatal(err)
	}
	if n, _, _ := analysed(); n != 1 {
		t.Errorf("after one picture was touched, %d were analysed", n)
	}
	rewritten := filepath.Join(dir, "p06", "1.jpg")
	at, err := os.Stat(rewritten)
	if err != nil {
		t.Fatal(err)
	}
	write(t, rewritten, "face:p06#11")
	if err := os.Chtimes(rewritten, time.Time{}, at.ModTime()); err != nil {
		t.Fatal(err)
	}
	if n, _, _ := analysed(); n != 1 {
		t.Errorf("after one picture was rewritten, %d were analysed", n)
	}
	write(t, filepath.Join(dir, "p07", "9.jpg"), "face:p07#9")
	if n, _, _ := analysed(); n != 1 {
		t.Errorf("after one picture was added, %d were analysed", n)
	}

	// another policy cannot use the vectors, says so, and replaces them
	n, _, stderr = analysed("--policy", "enrol")
	if n != 86 || !strings.Contains(stderr, "likeness eval: the cache "+cache+" was made with another policy; it is not used and will be replaced\n") {
		t.Errorf("under another policy %d pictures were analysed, want 86:\n%s", n, stderr)
	}
	if n, _, stderr := analysed("--policy", "enrol"); n != 0 || strings.Contains(stderr, "likeness eval:") {
		t.Errorf("the second run under the other policy analysed %d pictures:\n%s", n, stderr)
	}
	if n, _, _ := analysed(); n != 86 {
		t.Errorf("back under the first policy %d pictures were analysed, want 86", n)
	}

	// and so with other models
	other, _ := openCache(cache, basis{Fingerprint: "another:embedder", Detector: "another-detector"}, io.Discard)
	other.keep(picture{name: "p01/1.jpg"}, outcome{Reject: "blurry"})
	if err := other.save(); err != nil {
		t.Fatal(err)
	}
	n, _, stderr = analysed()
	if n != 86 || !strings.Contains(stderr, "was made with other models (another-detector, another:embedder)") {
		t.Errorf("after other models %d pictures were analysed, want 86:\n%s", n, stderr)
	}

	// a file that is not a cache stops the run before any work, and stays
	notes := filepath.Join(t.TempDir(), "notes.txt")
	write(t, notes, "not a cache")
	before := fakes.calls()
	_, _, err = eval(ctx, "--cache", notes, dir)
	if err == nil || !strings.Contains(err.Error(), "is not a cache of likeness eval") || fakes.calls() != before {
		t.Errorf("a file that is not a cache: %v, %d pictures analysed", err, fakes.calls()-before)
	}
	if text, _ := os.ReadFile(notes); string(text) != "not a cache" {
		t.Errorf("the file that is not a cache now holds %q", text)
	}
	if !fakes.allClosed() {
		t.Error("an engine was left open")
	}
}

func TestEvalCacheNotSaved(t *testing.T) {
	if os.Getuid() == 0 {
		t.Skip("root writes where nobody may write")
	}
	fakes := useFake(t)
	dir := crowd(t, 3, 2)
	sealed := t.TempDir()
	if err := os.Chmod(sealed, 0o555); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.Chmod(sealed, 0o755) })
	cache := filepath.Join(sealed, "cache")

	// the report is not held back by a cache that cannot be written, but
	// the run does not pass for a success either
	stdout, _, err := eval(context.Background(), "--cache", cache, dir)
	if err == nil || !strings.Contains(err.Error(), "the cache was not saved") || !errors.Is(err, fs.ErrPermission) {
		t.Errorf("a cache that cannot be written: %v", err)
	}
	if !strings.Contains(stdout, "What to run the server with") {
		t.Errorf("the report was held back:\n%s", stdout)
	}

	// a run that is stopped as well says so, and mentions the cache
	fakes.gate = make(chan struct{})
	write(t, filepath.Join(dir, "p03", "3.jpg"), "slow:p03#3")
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() {
		// six pictures go through, the seventh waits at the gate
		for before := fakes.calls(); fakes.calls() < before+7 && ctx.Err() == nil; time.Sleep(time.Millisecond) {
		}
		cancel()
	}()
	_, stderr, err := eval(ctx, "--workers", "1", "--cache", cache, dir)
	if !errors.Is(err, context.Canceled) || !strings.Contains(stderr, "likeness eval: the cache was not saved: ") {
		t.Errorf("stopped, with a cache that cannot be written: %v\n%s", err, stderr)
	}
}
