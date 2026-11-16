package eval

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"sync"
	"sync/atomic"
	"time"

	"github.com/muhriddinsalohiddin/likeness"
	"github.com/muhriddinsalohiddin/likeness/internal/cli"
)

// The kinds of error that keep a picture from being analysed, as the
// report counts them. The first four are about the picture and are named
// as the HTTP API names them.
const (
	unsupportedImage = "unsupported_image"
	imageTooLarge    = "image_too_large"
	imageTooSmall    = "image_too_small"
	corruptImage     = "corrupt_image"
	// The file could not be read, or the engine failed on it. Either may
	// go better next time, so neither is kept in the cache.
	unreadableFile = "unreadable_file"
	engineError    = "engine_error"
)

// maxFailuresShown is the number of unreadable pictures named on stderr;
// the rest are only counted.
const maxFailuresShown = 10

func note(w io.Writer, format string, args ...any) {
	_, _ = fmt.Fprintf(w, "likeness eval: "+format+"\n", args...)
}

func failureOf(err error) string {
	switch {
	case errors.Is(err, likeness.ErrUnsupportedImage):
		return unsupportedImage
	case errors.Is(err, likeness.ErrImageTooLarge):
		return imageTooLarge
	case errors.Is(err, likeness.ErrImageTooSmall):
		return imageTooSmall
	case errors.Is(err, likeness.ErrCorruptImage):
		return corruptImage
	}
	return engineError
}

// analyse turns one picture into its outcome. The error is the cause of an
// outcome that is a failure.
func analyse(ctx context.Context, eng cli.Engine, p picture, o likeness.Options) (outcome, error) {
	data, err := os.ReadFile(p.path)
	if err != nil {
		return outcome{Failure: unreadableFile}, err
	}
	res, err := eng.Analyze(ctx, data, o)
	if err != nil {
		return outcome{Failure: failureOf(err)}, err
	}
	face, reason := res.Best()
	if reason != likeness.Accepted {
		return outcome{Reject: string(reason)}, nil
	}
	return outcome{Vector: face.Embedding}, nil
}

// embed returns the outcome of every picture, in order: from the cache
// where it has one, from the engine otherwise, workers pictures at a time.
// What the engine makes of a picture goes into the cache. When ctx is done
// it stops and returns the context's error; the cache has what was
// finished by then.
func embed(ctx context.Context, eng cli.Engine, pictures []picture, o likeness.Options, workers int, held *cache, stderr io.Writer) ([]outcome, error) {
	outcomes := make([]outcome, len(pictures))
	var pending []int
	for i, p := range pictures {
		if known, ok := held.lookup(p); ok {
			outcomes[i] = known
		} else {
			pending = append(pending, i)
		}
	}

	var (
		causes   = make([]error, len(pictures))
		analysed = make([]bool, len(pictures))
		tick     = newProgress(stderr, len(pictures), len(pictures)-len(pending))
		next     atomic.Int64
		wg       sync.WaitGroup
	)
	for range min(workers, len(pending)) {
		wg.Go(func() {
			for ctx.Err() == nil {
				n := int(next.Add(1)) - 1
				if n >= len(pending) {
					return
				}
				i := pending[n]
				out, cause := analyse(ctx, eng, pictures[i], o)
				if cause != nil && ctx.Err() != nil {
					return // cut short, which says nothing about the picture
				}
				outcomes[i], causes[i], analysed[i] = out, cause, true
				tick.step()
			}
		})
	}
	wg.Wait()
	tick.finish()

	failed := 0
	for _, i := range pending {
		if !analysed[i] {
			continue
		}
		if kind := outcomes[i].Failure; kind != unreadableFile && kind != engineError {
			held.keep(pictures[i], outcomes[i])
		}
		if causes[i] == nil {
			continue
		}
		if failed++; failed <= maxFailuresShown {
			note(stderr, "%s: %v", pictures[i].name, causes[i])
		}
	}
	if failed > maxFailuresShown {
		note(stderr, "and %d more pictures that could not be read", failed-maxFailuresShown)
	}
	return outcomes, ctx.Err()
}

// plainEvery is the time between two lines of progress. It is a variable
// so that tests need not wait for one.
var plainEvery = 5 * time.Second

// redrawEvery is the time between two updates of the line on a terminal.
const redrawEvery = 200 * time.Millisecond

// progress says how far the analysis has come: a line every few seconds,
// or on a terminal one line that is drawn over.
type progress struct {
	w      io.Writer
	redraw bool
	every  time.Duration
	// total is the number of pictures, cached how many of them needed no
	// analysis.
	total, cached int
	start         time.Time

	mu    sync.Mutex
	done  int
	last  time.Time
	drawn bool
}

func newProgress(w io.Writer, total, cached int) *progress {
	p := &progress{w: w, every: plainEvery, total: total, cached: cached, start: time.Now()}
	if cli.Terminal(w) {
		p.redraw, p.every = true, redrawEvery
	}
	p.last = p.start
	return p
}

// perSecond is the rate at which the pictures of this run were analysed.
func (p *progress) perSecond(now time.Time) float64 {
	elapsed := now.Sub(p.start).Seconds()
	if elapsed <= 0 {
		return 0
	}
	return float64(p.done) / elapsed
}

// step is called after every picture and prints when it is time to.
func (p *progress) step() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.done++
	now := time.Now()
	if now.Sub(p.last) < p.every {
		return
	}
	p.last = now
	line := fmt.Sprintf("%d of %d pictures, %.1f per second", p.cached+p.done, p.total, p.perSecond(now))
	if p.redraw {
		_, _ = fmt.Fprintf(p.w, "\r%s\x1b[K", line)
		p.drawn = true
		return
	}
	_, _ = fmt.Fprintln(p.w, line)
}

// finish wipes the line a terminal was showing and sums the analysis up.
func (p *progress) finish() {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.drawn {
		_, _ = fmt.Fprint(p.w, "\r\x1b[K")
	}
	now := time.Now()
	took := now.Sub(p.start).Round(10 * time.Millisecond)
	if took >= 10*time.Second {
		took = took.Round(time.Second)
	}
	switch left := p.total - p.cached - p.done; {
	case p.done == 0 && left == 0:
		_, _ = fmt.Fprintf(p.w, "%d pictures, all from the cache\n", p.total)
	case left > 0:
		_, _ = fmt.Fprintf(p.w, "stopped after %d of %d pictures\n", p.cached+p.done, p.total)
	case p.cached > 0:
		_, _ = fmt.Fprintf(p.w, "%d pictures analysed in %s, %.1f per second; %d more from the cache\n",
			p.done, took, p.perSecond(now), p.cached)
	default:
		_, _ = fmt.Fprintf(p.w, "%d pictures analysed in %s, %.1f per second\n", p.done, took, p.perSecond(now))
	}
}
