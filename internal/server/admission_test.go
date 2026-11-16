package server

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestLimiter(t *testing.T) {
	l := newLimiter(1, 1, time.Minute)
	ctx := context.Background()

	if err := l.enter(); err != nil {
		t.Fatal(err)
	}
	if waited, err := l.acquire(ctx); err != nil || waited != 0 {
		t.Fatalf("a free slot: %v after %v", err, waited)
	}
	if l.running() != 1 || l.queued() != 0 {
		t.Errorf("%d running, %d queued", l.running(), l.queued())
	}

	// the second request queues
	type outcome struct {
		waited time.Duration
		err    error
	}
	second := make(chan outcome, 1)
	go func() {
		waited, err := l.acquire(ctx)
		second <- outcome{waited, err}
	}()
	eventually(t, "the second request to queue", func() bool { return l.queued() == 1 })

	// the third is turned away, at the door and at the queue
	if err := l.enter(); !errors.Is(err, errQueueFull) {
		t.Errorf("entering with a full queue: %v", err)
	}
	if _, err := l.acquire(ctx); !errors.Is(err, errQueueFull) {
		t.Errorf("queueing behind a full queue: %v", err)
	}

	time.Sleep(5 * time.Millisecond)
	l.release(0)
	if got := <-second; got.err != nil || got.waited < 5*time.Millisecond {
		t.Errorf("the queued request: %v after %v", got.err, got.waited)
	}
	if l.running() != 1 || l.queued() != 0 {
		t.Errorf("%d running, %d queued", l.running(), l.queued())
	}
	l.release(0)
	l.leave()
	if l.running() != 0 || l.inside.Load() != 0 {
		t.Errorf("%d running, %d inside at the end", l.running(), l.inside.Load())
	}
}

func TestLimiterGivesUp(t *testing.T) {
	l := newLimiter(1, 4, 20*time.Millisecond)
	if _, err := l.acquire(context.Background()); err != nil {
		t.Fatal(err)
	}

	// patience runs out
	waited, err := l.acquire(context.Background())
	if !errors.Is(err, errQueueTimeout) || waited < 20*time.Millisecond {
		t.Errorf("out of patience: %v after %v", err, waited)
	}

	// the caller goes away first
	l = newLimiter(1, 4, time.Minute)
	if _, err := l.acquire(context.Background()); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		for l.queued() == 0 {
			time.Sleep(time.Millisecond)
		}
		cancel()
	}()
	if _, err := l.acquire(ctx); !errors.Is(err, context.Canceled) {
		t.Errorf("cancelled while queued: %v", err)
	}
	if l.queued() != 0 {
		t.Errorf("%d still queued", l.queued())
	}
}

func TestLimiterRefusesWhatWouldWaitTooLong(t *testing.T) {
	l := newLimiter(2, 100, 50*time.Millisecond)
	ctx := context.Background()
	hold := func() {
		t.Helper()
		if _, err := l.acquire(ctx); err != nil {
			t.Fatal(err)
		}
	}

	// before any picture is done there is nothing to go by
	hold()
	hold()
	if l.tooLong(100) {
		t.Error("refusing on a guess")
	}
	// pictures turn out to take 40 ms: with two slots one comes free
	// every 20 ms, so two requests can expect theirs within the 50 ms
	l.release(40 * time.Millisecond)
	l.release(40 * time.Millisecond)
	hold()
	hold()
	queued := make(chan error, 2)
	for i := 0; i < 2; i++ {
		go func() {
			_, err := l.acquire(ctx)
			queued <- err
		}()
	}
	eventually(t, "two requests to queue", func() bool { return l.queued() == 2 })

	// a third would wait 60 ms: it is told so at once, at the door and
	// at the queue, though the queue has room for a hundred
	start := time.Now()
	if err := l.enter(); !errors.Is(err, errQueueFull) {
		t.Errorf("entering: %v", err)
	}
	if _, err := l.acquire(ctx); !errors.Is(err, errQueueFull) {
		t.Errorf("queueing: %v", err)
	}
	if took := time.Since(start); took > 25*time.Millisecond {
		t.Errorf("being refused took %v", took)
	}

	// the two that were let in get their slots
	l.release(40 * time.Millisecond)
	l.release(40 * time.Millisecond)
	for i := 0; i < 2; i++ {
		if err := <-queued; err != nil {
			t.Errorf("a queued request: %v", err)
		}
	}

	// the estimate follows the pictures: once they are quick again,
	// more requests may wait
	for i := 0; i < 40; i++ {
		l.typical.Store(l.typical.Load() + (int64(5*time.Millisecond)-l.typical.Load())/8)
	}
	if l.tooLong(10) || !l.tooLong(30) {
		t.Errorf("after quick pictures the typical hold is %v", time.Duration(l.typical.Load()))
	}
}

func TestLimiterWithoutAQueue(t *testing.T) {
	l := newLimiter(2, 0, time.Minute)
	for i := 0; i < 2; i++ {
		if err := l.enter(); err != nil {
			t.Fatal(err)
		}
		if _, err := l.acquire(context.Background()); err != nil {
			t.Fatal(err)
		}
	}
	if err := l.enter(); !errors.Is(err, errQueueFull) {
		t.Errorf("both slots taken and nowhere to wait: %v", err)
	}
	l.release(0)
	if err := l.enter(); err != nil {
		t.Errorf("a slot is free again: %v", err)
	}
}

func TestLimiterBoundsWhatIsInside(t *testing.T) {
	l := newLimiter(1, 1, time.Minute) // room for eight
	for i := 0; i < 8; i++ {
		if err := l.enter(); err != nil {
			t.Fatalf("request %d: %v", i, err)
		}
	}
	if err := l.enter(); !errors.Is(err, errCrowded) {
		t.Errorf("the ninth: %v", err)
	}
	l.leave()
	if err := l.enter(); err != nil {
		t.Errorf("after one left: %v", err)
	}
}

func TestLimiterNeverExceedsItsSlots(t *testing.T) {
	const slots, requests = 3, 200
	l := newLimiter(slots, requests, time.Minute)
	var (
		mu      sync.Mutex
		now     int
		highest int
		wg      sync.WaitGroup
	)
	for i := 0; i < requests; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := l.acquire(context.Background()); err != nil {
				t.Error(err)
				return
			}
			mu.Lock()
			now++
			highest = max(highest, now)
			mu.Unlock()
			time.Sleep(100 * time.Microsecond)
			mu.Lock()
			now--
			mu.Unlock()
			l.release(0)
		}()
	}
	wg.Wait()
	if highest != slots {
		t.Errorf("at most %d ran at once, want %d", highest, slots)
	}
}

// live serves the rig over a real connection, for tests that need requests
// to overlap or to break off.
func (r *rig) live() string {
	ts := httptest.NewServer(r.srv.Handler())
	r.t.Cleanup(ts.Close)
	return ts.URL
}

type reply struct {
	status int
	header http.Header
	body   string
	err    error
}

// post searches with a picture over a real connection.
func post(ctx context.Context, url string, picture io.Reader) reply {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, picture)
	if err != nil {
		return reply{err: err}
	}
	req.Header.Set("Content-Type", "image/jpeg")
	req.Header.Set("Authorization", "Bearer "+searchKey)
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		return reply{err: err}
	}
	defer res.Body.Close()
	body, err := io.ReadAll(res.Body)
	return reply{res.StatusCode, res.Header, string(body), err}
}

// later runs a request in the background.
func later(f func() reply) <-chan reply {
	ch := make(chan reply, 1)
	go func() { ch <- f() }()
	return ch
}

func (r *rig) refused(cause string) uint64 { return r.srv.metrics.refusals.get(cause, "") }

func TestOverload(t *testing.T) {
	r := newRig(t, func(c *Config) { c.MaxInflight, c.Queue, c.QueueTimeout = 1, 1, time.Minute })
	gate := make(chan struct{})
	r.fake.Gate = gate
	open := sync.OnceFunc(func() { close(gate) })
	url := r.live() + "/v1/search"
	t.Cleanup(open)
	ctx := context.Background()

	// one picture is being analysed, one waits for it to finish
	first := later(func() reply { return post(ctx, url, strings.NewReader("slow:anna")) })
	eventually(t, "the first picture to reach the engine", func() bool { return r.srv.limit.running() == 1 })
	second := later(func() reply { return post(ctx, url, strings.NewReader("slow:anna")) })
	eventually(t, "the second request to queue", func() bool { return r.srv.limit.queued() == 1 })

	// a third is told at once to come back later
	start := time.Now()
	third := post(ctx, url, strings.NewReader("face:anna"))
	if third.err != nil || third.status != http.StatusTooManyRequests || !strings.Contains(third.body, `"code":"overloaded"`) {
		t.Fatalf("the third request: %+v", third)
	}
	if third.header.Get("Retry-After") != "1" {
		t.Errorf("Retry-After %q", third.header.Get("Retry-After"))
	}
	if took := time.Since(start); took > 5*time.Second {
		t.Errorf("being refused took %v", took)
	}
	if r.fake.Calls() != 1 || r.refused("queue_full") != 1 {
		t.Errorf("%d pictures reached the engine, %d refusals counted", r.fake.Calls(), r.refused("queue_full"))
	}

	open()
	for i, pending := range []<-chan reply{first, second} {
		if got := <-pending; got.err != nil || got.status != http.StatusOK {
			t.Errorf("request %d: %+v", i+1, got)
		}
	}
	if r.fake.Peak() != 1 {
		t.Errorf("%d pictures were analysed at once", r.fake.Peak())
	}
	eventually(t, "the server to empty", func() bool {
		return r.srv.limit.running() == 0 && r.srv.limit.queued() == 0 && r.srv.limit.inside.Load() == 0
	})
}

func TestQueueTimeout(t *testing.T) {
	r := newRig(t, func(c *Config) { c.MaxInflight, c.Queue, c.QueueTimeout = 1, 4, 30*time.Millisecond })
	gate := make(chan struct{})
	r.fake.Gate = gate
	open := sync.OnceFunc(func() { close(gate) })
	url := r.live() + "/v1/search"
	t.Cleanup(open)
	ctx := context.Background()

	first := later(func() reply { return post(ctx, url, strings.NewReader("slow:anna")) })
	eventually(t, "the first picture to reach the engine", func() bool { return r.srv.limit.running() == 1 })

	// the slot does not come free in time
	second := post(ctx, url, strings.NewReader("face:anna"))
	if second.status != http.StatusTooManyRequests || second.header.Get("Retry-After") != "1" {
		t.Fatalf("the waiting request: %+v", second)
	}
	if r.refused("queue_timeout") != 1 || r.fake.Calls() != 1 {
		t.Errorf("%d timeouts counted, %d pictures analysed", r.refused("queue_timeout"), r.fake.Calls())
	}
	// its wait is on record
	var waited float64
	for _, line := range r.logs.requests(t) {
		if line["status"] == float64(http.StatusTooManyRequests) {
			if line["level"] != "WARN" {
				t.Errorf("a refusal is logged as a warning: %v", line)
			}
			waited, _ = line["stages"].(map[string]any)["queue"].(float64)
		}
	}
	if waited < 30 {
		t.Errorf("logged a wait of %v ms", waited)
	}

	open()
	if got := <-first; got.status != http.StatusOK {
		t.Errorf("the first request: %+v", got)
	}
}

func TestManyAtOnce(t *testing.T) {
	r := newRig(t, func(c *Config) { c.MaxInflight, c.Queue, c.QueueTimeout = 2, 32, time.Minute })
	r.fake.Delay = 3 * time.Millisecond
	r.enrol("anna", "face:anna")
	url := r.live() + "/v1/search"

	const requests = 24
	pending := make([]<-chan reply, requests)
	for i := range pending {
		pending[i] = later(func() reply { return post(context.Background(), url, strings.NewReader("slow:anna")) })
	}
	for i, p := range pending {
		if got := <-p; got.err != nil || got.status != http.StatusOK || !strings.Contains(got.body, `"match":true`) {
			t.Errorf("request %d: %+v", i, got)
		}
	}
	if r.fake.Peak() != 2 {
		t.Errorf("%d pictures were analysed at once, want 2", r.fake.Peak())
	}
}

func TestClientGivesUpInTheQueue(t *testing.T) {
	r := newRig(t, func(c *Config) { c.MaxInflight, c.Queue, c.QueueTimeout = 1, 4, time.Minute })
	gate := make(chan struct{})
	r.fake.Gate = gate
	open := sync.OnceFunc(func() { close(gate) })
	url := r.live() + "/v1/search"
	t.Cleanup(open)

	first := later(func() reply { return post(context.Background(), url, strings.NewReader("slow:anna")) })
	eventually(t, "the first picture to reach the engine", func() bool { return r.srv.limit.running() == 1 })

	ctx, hangUp := context.WithCancel(context.Background())
	second := later(func() reply { return post(ctx, url, strings.NewReader("face:anna")) })
	eventually(t, "the second request to queue", func() bool { return r.srv.limit.queued() == 1 })
	hangUp()
	if got := <-second; got.err == nil {
		t.Fatalf("the abandoned request was answered: %+v", got)
	}
	// its place in the queue is given up without the engine ever seeing it
	eventually(t, "the queue to empty", func() bool { return r.srv.limit.queued() == 0 && r.srv.limit.inside.Load() == 1 })

	open()
	if got := <-first; got.status != http.StatusOK {
		t.Errorf("the first request: %+v", got)
	}
	if r.fake.Calls() != 1 {
		t.Errorf("%d pictures reached the engine", r.fake.Calls())
	}
	eventually(t, "the hang-up to be logged", func() bool {
		for _, line := range r.logs.requests(t) {
			if line["status"] == float64(statusClientGone) {
				return true
			}
		}
		return false
	})
	if n := r.refused("queue_full") + r.refused("queue_timeout") + r.refused("too_many_uploads"); n != 0 {
		t.Errorf("a client that left was counted as %d refusals", n)
	}
}

func TestTooManyUploads(t *testing.T) {
	// one slot and no queue: room for four requests in all
	r := newRig(t, func(c *Config) { c.MaxInflight, c.Queue = 1, -1 })
	r.enrol("anna", "face:anna")
	url := r.live() + "/v1/search"
	ctx := context.Background()

	// four clients start sending their pictures and stall
	var (
		stalled []*io.PipeWriter
		pending []<-chan reply
	)
	for i := 0; i < 4; i++ {
		body, feed := io.Pipe()
		stalled = append(stalled, feed)
		pending = append(pending, later(func() reply { return post(ctx, url, body) }))
	}
	eventually(t, "four uploads to be under way", func() bool { return r.srv.limit.inside.Load() == 4 })

	// they take no slot, but their pictures would take memory: a fifth
	// has to wait its turn outside
	fifth := post(ctx, url, strings.NewReader("face:anna"))
	if fifth.status != http.StatusTooManyRequests || r.refused("too_many_uploads") != 1 {
		t.Fatalf("the fifth upload: %+v (%d counted)", fifth, r.refused("too_many_uploads"))
	}

	// the four finish one after another and are all served
	for i, feed := range stalled {
		io.WriteString(feed, "face:anna")
		feed.Close()
		if got := <-pending[i]; got.err != nil || got.status != http.StatusOK {
			t.Errorf("upload %d: %+v", i, got)
		}
	}
	eventually(t, "the server to empty", func() bool { return r.srv.limit.inside.Load() == 0 })
	if got := post(ctx, url, strings.NewReader("face:anna")); got.status != http.StatusOK {
		t.Errorf("after the rush: %+v", got)
	}
}
