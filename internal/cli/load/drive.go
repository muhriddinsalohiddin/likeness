package load

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"sync"
	"sync/atomic"
	"time"
)

// maxAnswer is how much of a search answer is read for its contents; an
// answer for 16 faces is a few kilobytes.
const maxAnswer = 1 << 20

// loader sends searches to one server.
type loader struct {
	o      *options
	client *http.Client
	target string
	auth   string
	// sent numbers the requests, to send the pictures in rotation.
	sent atomic.Uint64
}

func newLoader(o *options) *loader {
	// One connection for every request that can be in flight, and all of
	// them kept: a load that dials for each request measures the dialling.
	conns := o.concurrency
	if len(o.rates) > 0 {
		conns = o.maxPending
	}
	l := &loader{o: o, target: o.search.String()}
	if o.key != "" {
		l.auth = "Bearer " + o.key
	}
	l.client = &http.Client{
		Transport: &http.Transport{
			Proxy:               http.ProxyFromEnvironment,
			DialContext:         (&net.Dialer{Timeout: o.timeout, KeepAlive: 30 * time.Second}).DialContext,
			ForceAttemptHTTP2:   true,
			MaxIdleConns:        conns,
			MaxIdleConnsPerHost: conns,
			MaxConnsPerHost:     conns,
			IdleConnTimeout:     90 * time.Second,
			TLSHandshakeTimeout: 10 * time.Second,
		},
		// A redirect is an answer to report, not one to follow.
		CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
	}
	return l
}

// answer is the part of a search answer the report looks at.
type answer struct {
	Faces []struct {
		Match  bool   `json:"match"`
		Reject string `json:"reject"`
	} `json:"faces"`
	Reject string             `json:"reject"`
	Timing map[string]float64 `json:"timing_ms"`
}

// exchange is one request as the client saw it.
type exchange struct {
	// status is 0 and err says why when no answer arrived.
	status int
	err    error
	// took covers the whole exchange, from sending the request to the end
	// of the answer's body.
	took time.Duration
	// answer is what a 200 said, nil if that could not be read.
	answer *answer
}

// send searches with the next picture.
func (l *loader) send() exchange {
	picture := l.o.pictures[(l.sent.Add(1)-1)%uint64(len(l.o.pictures))]
	// Not the context of the command: when a run is interrupted, the
	// requests in flight are given the time to finish.
	ctx, cancel := context.WithTimeout(context.Background(), l.o.timeout)
	defer cancel()

	start := time.Now()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, l.target, bytes.NewReader(picture))
	if err != nil {
		return exchange{err: err}
	}
	req.Header.Set("Content-Type", "application/octet-stream")
	if l.auth != "" {
		req.Header.Set("Authorization", l.auth)
	}
	res, err := l.client.Do(req)
	if err != nil {
		return exchange{err: err, took: time.Since(start)}
	}
	defer res.Body.Close()

	x := exchange{status: res.StatusCode}
	if res.StatusCode == http.StatusOK {
		var a answer
		if body, err := io.ReadAll(io.LimitReader(res.Body, maxAnswer)); err == nil && json.Unmarshal(body, &a) == nil {
			x.answer = &a
		}
	}
	// To its end, so that the time is that of the whole answer and the
	// connection can be used again.
	if _, err := io.Copy(io.Discard, res.Body); err != nil {
		return exchange{err: err, took: time.Since(start)}
	}
	x.took = time.Since(start)
	return x
}

// probe sends one request ahead of the run, and stops the command when the
// answer shows that every later request would fail the same way: a table
// of ten thousand errors says less than one sentence.
func (l *loader) probe() error {
	x := l.send()
	host := l.o.search.Host
	var dial *net.OpError
	switch {
	case errors.As(x.err, &dial) && dial.Op == "dial":
		return fmt.Errorf("cannot connect to %s: %w: is the server running, and is --url right?", host, dial.Err)
	case x.status == http.StatusUnauthorized && l.auth == "":
		return fmt.Errorf("%s wants an API key: set %s to one of the server's search keys", host, keysVar)
	case x.status == http.StatusUnauthorized:
		return fmt.Errorf("%s does not know the key: the first key in %s has to be one of the server's search or admin keys", host, keysVar)
	case x.status == http.StatusForbidden:
		return fmt.Errorf("the key in %s may not search on %s: use one of the server's search or admin keys", keysVar, host)
	case x.status == http.StatusNotFound:
		return fmt.Errorf("%s answered 404: no likeness API is served there; check the path in --url", l.o.search.Redacted())
	}
	return nil
}

// sleep waits for d and reports whether it did so undisturbed.
func sleep(ctx context.Context, d time.Duration) bool {
	if d <= 0 {
		return ctx.Err() == nil
	}
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-timer.C:
		return true
	case <-ctx.Done():
		return false
	}
}

// measured is how much of the measuring a run got through: all of it,
// unless the run was cut short.
func (o *options) measured(start, stopped time.Time) time.Duration {
	return min(max(stopped.Sub(start.Add(o.warmup)), 0), o.duration)
}

// closed runs a closed loop: every worker sends a request, waits for the
// answer and sends the next, until the time is up or ctx is done. It
// returns what became of the requests begun while measuring, and how long
// the measuring lasted.
func (l *loader) closed(ctx context.Context) (*tally, time.Duration) {
	t := newTally()
	start := time.Now()
	from := start.Add(l.o.warmup)
	until := from.Add(l.o.duration)

	var workers sync.WaitGroup
	for range l.o.concurrency {
		workers.Go(func() {
			for ctx.Err() == nil {
				begun := time.Now()
				if !begun.Before(until) {
					return
				}
				x := l.send()
				if !begun.Before(from) {
					t.add(x)
				}
			}
		})
	}
	sleep(ctx, time.Until(until))
	stopped := time.Now()
	workers.Wait()
	return t, l.o.measured(start, stopped)
}

// timetable is the schedule of an open loop: request i is due i/rate
// seconds into the run, whatever became of the requests before it.
type timetable struct {
	rate           float64 // requests per second
	warmup, length time.Duration
}

// slot says how far into the run request i is due and whether it falls
// into the measured part. ok is false once the run is over.
func (t timetable) slot(i int) (due time.Duration, measured, ok bool) {
	due = time.Duration(float64(i) * float64(time.Second) / t.rate)
	return due, due >= t.warmup, due < t.warmup+t.length
}

// gate bounds the requests of an open loop that are in flight. Without it
// a server that has stopped answering would have the loop pile up requests
// until the machine sending them is the one in trouble.
type gate struct {
	inside atomic.Int64
	room   int64
}

// enter reports whether there is room for another request. A request that
// entered must leave.
func (g *gate) enter() bool {
	if g.inside.Add(1) > g.room {
		g.inside.Add(-1)
		return false
	}
	return true
}

func (g *gate) leave() { g.inside.Add(-1) }

// open runs an open loop: requests start on the timetable whether or not
// earlier ones have finished. A request that is due late, because this
// process was held up, is sent at once rather than dropped: the server is
// owed the load that was asked for. It returns what became of the requests
// due while measuring, and how long the measuring lasted.
func (l *loader) open(ctx context.Context, rate float64) (*tally, time.Duration) {
	t := newTally()
	table := timetable{rate: rate, warmup: l.o.warmup, length: l.o.duration}
	pending := &gate{room: int64(l.o.maxPending)}
	start := time.Now()
	stopped := start.Add(l.o.warmup + l.o.duration)

	var requests sync.WaitGroup
	for i := 0; ; i++ {
		due, measured, ok := table.slot(i)
		if !ok {
			break
		}
		if !sleep(ctx, time.Until(start.Add(due))) {
			stopped = time.Now()
			break
		}
		if !pending.enter() {
			if measured {
				t.skip()
			}
			continue
		}
		requests.Go(func() {
			defer pending.leave()
			x := l.send()
			if measured {
				t.add(x)
			}
		})
	}
	requests.Wait()
	return t, l.o.measured(start, stopped)
}
