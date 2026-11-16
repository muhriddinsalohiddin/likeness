package enrol

import (
	"io"
	"time"

	"github.com/muhriddinsalohiddin/likeness/internal/cli"
)

// plainEvery is the time between two lines of progress. It is a variable
// so that tests need not wait for one.
var plainEvery = 5 * time.Second

// redrawEvery is the time between two updates of the line on a terminal.
const redrawEvery = 200 * time.Millisecond

// progress says how far a run has come: a line every few seconds, or on a
// terminal one line that is drawn over.
type progress struct {
	w      io.Writer
	redraw bool
	every  time.Duration
	total  int
	start  time.Time
	last   time.Time
	drawn  bool
}

func newProgress(w io.Writer, total int) *progress {
	p := &progress{w: w, every: plainEvery, total: total, start: time.Now()}
	if cli.Terminal(w) {
		p.redraw, p.every = true, redrawEvery
	}
	p.last = p.start
	return p
}

// step is called after every picture and prints when it is time to.
func (p *progress) step(done int) {
	now := time.Now()
	if now.Sub(p.last) < p.every {
		return
	}
	p.last = now
	rate := 0.0
	if elapsed := now.Sub(p.start).Seconds(); elapsed > 0 {
		rate = float64(done) / elapsed
	}
	if p.redraw {
		say(p.w, "\r%d of %d pictures, %.1f per second\x1b[K", done, p.total, rate)
		p.drawn = true
		return
	}
	say(p.w, "%d of %d pictures, %.1f per second\n", done, p.total, rate)
}

// finish wipes the line a terminal was showing, to make room for what is
// printed next.
func (p *progress) finish() {
	if p.drawn {
		say(p.w, "\r\x1b[K")
	}
}
