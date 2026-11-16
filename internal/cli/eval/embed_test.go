package eval

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/muhriddinsalohiddin/likeness/internal/cli"
)

func TestProgress(t *testing.T) {
	plain := plainEvery
	t.Cleanup(func() { plainEvery = plain })

	// a line whenever it is time to, and a summary
	plainEvery = 0
	var out strings.Builder
	p := newProgress(&out, 7, 2)
	p.start = p.start.Add(-2 * time.Second)
	for range 3 {
		p.step()
	}
	p.finish()
	lines := strings.Split(strings.TrimSuffix(out.String(), "\n"), "\n")
	if len(lines) != 4 || !strings.HasPrefix(lines[0], "3 of 7 pictures, 0.5 per second") ||
		!strings.HasPrefix(lines[2], "5 of 7 pictures, 1.") || lines[3] != "stopped after 5 of 7 pictures" {
		t.Errorf("progress on a file:\n%s", out.String())
	}

	// nothing until it is time to
	plainEvery = time.Hour
	out.Reset()
	p = newProgress(&out, 4, 1)
	p.start = p.start.Add(-90 * time.Second)
	for range 3 {
		p.step()
	}
	p.finish()
	if got := out.String(); got != "3 pictures analysed in 1m30s, 0.0 per second; 1 more from the cache\n" {
		t.Errorf("the summary: %q", got)
	}

	out.Reset()
	p = newProgress(&out, 2, 0)
	p.start = p.start.Add(-1500 * time.Millisecond)
	p.step()
	p.step()
	p.finish()
	if got := out.String(); !strings.HasPrefix(got, "2 pictures analysed in 1.5s, 1.3 per second\n") {
		t.Errorf("the summary without a cache: %q", got)
	}

	out.Reset()
	newProgress(&out, 9, 9).finish()
	if got := out.String(); got != "9 pictures, all from the cache\n" {
		t.Errorf("the summary of a run that analysed nothing: %q", got)
	}

	// on a terminal one line is drawn over and wiped at the end
	out.Reset()
	p = newProgress(&out, 3, 0)
	p.redraw, p.every = true, 0
	p.step()
	p.step()
	p.finish()
	got := out.String()
	if strings.Count(got, "\n") != 1 || !strings.HasPrefix(got, "\r1 of 3 pictures, ") || !strings.Contains(got, "\x1b[K\r2 of 3 pictures, ") ||
		!strings.HasSuffix(got, "\r\x1b[Kstopped after 2 of 3 pictures\n") {
		t.Errorf("progress on a terminal: %q", got)
	}
}

func TestTerminal(t *testing.T) {
	if cli.Terminal(&strings.Builder{}) {
		t.Error("a buffer is taken for a terminal")
	}
	file, err := os.Create(filepath.Join(t.TempDir(), "log"))
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	if cli.Terminal(file) {
		t.Error("a file is taken for a terminal")
	}
	// the null device is a character device, as a terminal is
	null, err := os.OpenFile(os.DevNull, os.O_WRONLY, 0)
	if err != nil {
		t.Skip(err)
	}
	defer null.Close()
	if !cli.Terminal(null) {
		t.Skip("the null device is not a character device here")
	}
	if p := newProgress(null, 1, 0); !p.redraw || p.every != redrawEvery {
		t.Errorf("progress on a character device: redraw %v every %v", p.redraw, p.every)
	}
}
