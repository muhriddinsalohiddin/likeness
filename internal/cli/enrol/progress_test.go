package enrol

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/muhriddinsalohiddin/likeness/internal/cli"
)

func TestProgress(t *testing.T) {
	// to a file or a pipe: whole lines, and not more often than asked
	var out bytes.Buffer
	p := newProgress(&out, 3)
	if p.redraw || p.every != plainEvery {
		t.Errorf("a buffer is written to as a terminal would be: %+v", p)
	}
	p.step(1)
	if out.Len() != 0 {
		t.Errorf("progress before it is time: %q", out.String())
	}
	p.every = 0
	p.step(2)
	p.step(3)
	p.finish()
	lines := strings.Split(out.String(), "\n")
	if len(lines) != 3 || !strings.HasPrefix(lines[0], "2 of 3 pictures, ") || !strings.HasPrefix(lines[1], "3 of 3 pictures, ") ||
		!strings.HasSuffix(lines[1], " per second") || lines[2] != "" {
		t.Errorf("progress in lines: %q", out.String())
	}

	// on a terminal: one line, drawn over and wiped at the end
	out.Reset()
	p = &progress{w: &out, redraw: true, total: 3, start: time.Now().Add(-time.Second)}
	p.step(1)
	p.step(2)
	p.finish()
	text := out.String()
	if strings.Contains(text, "\n") || strings.Count(text, "\r") != 3 || !strings.HasPrefix(text, "\r1 of 3 pictures, 1.0 per second\x1b[K\r2 of 3 pictures, ") ||
		!strings.HasSuffix(text, "\x1b[K\r\x1b[K") {
		t.Errorf("progress on a terminal: %q", text)
	}

	// nothing drawn, nothing to wipe
	out.Reset()
	(&progress{w: &out, redraw: true, every: time.Hour, last: time.Now()}).finish()
	if out.Len() != 0 {
		t.Errorf("finish wrote %q", out.String())
	}
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
	if !cli.Terminal(null) || cli.Terminal(file) || cli.Terminal(&bytes.Buffer{}) {
		t.Errorf("a device %v, a file %v, a buffer %v", cli.Terminal(null), cli.Terminal(file), cli.Terminal(&bytes.Buffer{}))
	}
	file.Close()
	if cli.Terminal(file) {
		t.Error("a closed file was taken for a terminal")
	}

	// a run that writes its progress to a device
	useFake(t)
	var stdout bytes.Buffer
	args := []string{"--store", filepath.Join(t.TempDir(), "faces.db"), tree(t, "anna/1.jpg", "face:anna")}
	if err := Run(ctx, args, cliEnv(&stdout, null)); err != nil || !strings.Contains(stdout.String(), "enrolled           1\n") {
		t.Errorf("%v\n%s", err, stdout.String())
	}
}
