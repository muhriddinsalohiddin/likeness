package server

import (
	"context"
	"errors"
	"sync/atomic"
	"time"
)

var (
	errQueueFull    = errors.New("server: the queue is full")
	errQueueTimeout = errors.New("server: no free slot in time")
	errCrowded      = errors.New("server: too many pictures on their way in")
)

// limiter decides how many pictures the server works on at once.
//
// Analysing a picture takes all the cores it is given, so running more
// analyses than there are slots makes every one of them slower. A request
// therefore takes a slot for the analysis, waits for one if they are all
// taken, and is turned away when the line is already long or its wait has
// lasted too long. Turning away is cheap and immediate, and a refused
// client can retry; a client served two seconds late has usually given up.
//
// Pictures are held in memory from the moment they arrive, so the number
// of requests that are anywhere between arriving and finishing is bounded
// too, more generously: reading a picture off a slow connection costs
// memory but no processor time.
type limiter struct {
	slots    chan struct{}
	queue    int64
	patience time.Duration
	room     int64

	inside  atomic.Int64 // requests admitted and not yet finished
	waiting atomic.Int64 // of those, waiting for a slot
}

// roomPerPlace is how many requests may be inside for every one that can be
// running or queued.
const roomPerPlace = 4

func newLimiter(slots, queue int, patience time.Duration) *limiter {
	return &limiter{
		slots:    make(chan struct{}, slots),
		queue:    int64(queue),
		patience: patience,
		room:     int64(roomPerPlace * (slots + queue)),
	}
}

// enter lets a request in, before any of its picture has been read. It
// never blocks. A request that entered must leave.
func (l *limiter) enter() error {
	if l.waiting.Load() >= l.queue && len(l.slots) == cap(l.slots) {
		// It would only be read to be turned away at the queue.
		return errQueueFull
	}
	if l.inside.Add(1) > l.room {
		l.inside.Add(-1)
		return errCrowded
	}
	return nil
}

func (l *limiter) leave() { l.inside.Add(-1) }

// acquire takes a slot, waiting for one if need be, and reports how long
// the wait was. The caller must release the slot. It fails with
// errQueueFull, errQueueTimeout or the context's error.
func (l *limiter) acquire(ctx context.Context) (time.Duration, error) {
	select {
	case l.slots <- struct{}{}:
		return 0, nil
	default:
	}
	if l.waiting.Add(1) > l.queue {
		l.waiting.Add(-1)
		return 0, errQueueFull
	}
	defer l.waiting.Add(-1)

	start := time.Now()
	timer := time.NewTimer(l.patience)
	defer timer.Stop()
	// A freed slot goes to whoever has waited longest: the runtime hands it
	// over directly, so a newcomer cannot slip in ahead through the fast
	// path above.
	select {
	case l.slots <- struct{}{}:
		return time.Since(start), nil
	case <-timer.C:
		return time.Since(start), errQueueTimeout
	case <-ctx.Done():
		return time.Since(start), ctx.Err()
	}
}

func (l *limiter) release() { <-l.slots }

// running and queued are the numbers the metrics report.
func (l *limiter) running() int { return len(l.slots) }
func (l *limiter) queued() int  { return int(l.waiting.Load()) }
