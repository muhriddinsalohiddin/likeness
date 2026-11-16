package client

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"
)

func TestSendsAgainWhenTheServerIsBusy(t *testing.T) {
	s := play(t, busy("1"), ok(searchAnswer))
	c, waits := impatient(t, s.url, WithKey(searchKey))

	res, err := c.Search(WithRequestID(ctx, "retry-1"), []byte("face:anna"), SearchOptions{Limit: 3})
	if err != nil {
		t.Fatal(err)
	}
	if res.Reject != ReasonNoFace || res.RequestID != "ok-1" {
		t.Errorf("the answer to the second attempt: %+v", res)
	}
	if len(*waits) != 1 || (*waits)[0] != time.Second {
		t.Errorf("waited %v, want the second the server asked for", *waits)
	}
	// the second attempt is the first all over again
	got := s.requests()
	if len(got) != 2 {
		t.Fatalf("the server received %d requests, want 2", len(got))
	}
	for i, rq := range got {
		if rq.method != http.MethodPost || rq.target != "/v1/search?limit=3" || string(rq.body) != "face:anna" ||
			rq.header.Get("Authorization") != "Bearer "+searchKey || rq.header.Get("X-Request-Id") != "retry-1" ||
			rq.header.Get("Content-Type") != "application/octet-stream" {
			t.Errorf("attempt %d: %+v", i+1, rq)
		}
	}

	// every call is sent again, an enrolment included: a refused request
	// did nothing
	calls := map[string]func(c *Client) error{
		"verify": func(c *Client) error {
			_, err := c.Verify(ctx, "anna", []byte("face:anna"), VerifyOptions{})
			return err
		},
		"enrol":  func(c *Client) error { _, err := c.Enrol(ctx, "anna", []byte("face:anna"), EnrolOptions{}); return err },
		"show":   func(c *Client) error { _, err := c.Subject(ctx, "anna"); return err },
		"list":   func(c *Client) error { _, err := c.Subjects(ctx, "", 0); return err },
		"remove": func(c *Client) error { _, err := c.RemoveSubject(ctx, "anna"); return err },
		"info":   func(c *Client) error { _, err := c.Info(ctx); return err },
	}
	for name, call := range calls {
		s := play(t, busy("1"), busy("1"), ok(`{}`))
		c, waits := impatient(t, s.url)
		if err := call(c); err != nil {
			t.Errorf("%s: %v", name, err)
		}
		if got := s.requests(); len(got) != 3 || !bytes.Equal(got[0].body, got[2].body) || got[0].target != got[2].target {
			t.Errorf("%s: the server received %+v", name, got)
		}
		if len(*waits) != 2 {
			t.Errorf("%s: waited %v", name, *waits)
		}
	}
}

func TestWaitsAsLongAsTheServerAsks(t *testing.T) {
	for _, tt := range []struct {
		retryAfter string
		want       time.Duration
	}{
		{"1", time.Second},
		{"2", 2 * time.Second},
		{"3", 2 * time.Second}, // never longer than two seconds
		{"3600", 2 * time.Second},
		{time.Now().Add(time.Hour).UTC().Format(http.TimeFormat), 2 * time.Second},
		// nothing usable: the client's own short pause
		{"", 100 * time.Millisecond},
		{"0", 100 * time.Millisecond},
		{"-5", 100 * time.Millisecond},
		{"soon", 100 * time.Millisecond},
		{time.Now().Add(-time.Hour).UTC().Format(http.TimeFormat), 100 * time.Millisecond},
	} {
		s := play(t, busy(tt.retryAfter), ok(infoAnswer))
		c, waits := impatient(t, s.url)
		if _, err := c.Info(ctx); err != nil {
			t.Errorf("Retry-After %q: %v", tt.retryAfter, err)
		}
		if len(*waits) != 1 || (*waits)[0] != tt.want {
			t.Errorf("Retry-After %q: waited %v, want %v", tt.retryAfter, *waits, tt.want)
		}
	}

	// a server that names no pause is given more time with every attempt
	s := play(t, busy(""), busy(""), busy(""), busy(""), busy(""), busy(""), ok(infoAnswer))
	c, waits := impatient(t, s.url, WithRetries(6))
	if _, err := c.Info(ctx); err != nil {
		t.Fatal(err)
	}
	want := []time.Duration{100, 200, 400, 800, 1600, 2000}
	for i := range want {
		want[i] *= time.Millisecond
	}
	if len(*waits) != len(want) {
		t.Fatalf("waited %v, want %v", *waits, want)
	}
	for i := range want {
		if (*waits)[i] != want[i] {
			t.Errorf("pause %d was %v, want %v", i+1, (*waits)[i], want[i])
		}
	}

	// the pauses are spread: what the server asks for, and up to a quarter
	// more
	s = play(t, busy("1"), ok(infoAnswer))
	c = newClient(t, s.url)
	var slept time.Duration
	c.sleep = func(_ context.Context, d time.Duration) error { slept = d; return nil }
	if _, err := c.Info(ctx); err != nil {
		t.Fatal(err)
	}
	if slept < time.Second || slept > 1250*time.Millisecond {
		t.Errorf("slept %v after being asked for a second", slept)
	}
}

func TestGivesUpOnABusyServer(t *testing.T) {
	for _, tt := range []struct {
		name     string
		options  []Option
		attempts int
	}{
		{"by default", nil, 3},
		{"WithRetries(0)", []Option{WithRetries(0)}, 1},
		{"WithRetries(1)", []Option{WithRetries(1)}, 2},
		{"WithRetries(4)", []Option{WithRetries(4)}, 5},
	} {
		steps := make([]http.HandlerFunc, tt.attempts)
		for i := range steps {
			steps[i] = busy("1")
		}
		s := play(t, steps...)
		c, waits := impatient(t, s.url, tt.options...)
		_, err := c.Search(ctx, []byte("face:anna"), SearchOptions{})
		e := refused(t, err, http.StatusTooManyRequests, CodeOverloaded)
		if !IsOverloaded(err) || e.RetryAfter != time.Second || e.RequestID != "busy-1" || e.Message != "try again shortly" {
			t.Errorf("%s: the refusal handed over: %+v", tt.name, e)
		}
		s.count(tt.attempts)
		if len(*waits) != tt.attempts-1 {
			t.Errorf("%s: waited %v", tt.name, *waits)
		}
	}

	// a probe is asked once
	s := play(t, busy("1"), busy("1"))
	c, waits := impatient(t, s.url)
	if err := c.Ready(ctx); !IsOverloaded(err) {
		t.Errorf("ready: %v", err)
	}
	if err := c.Alive(ctx); !IsOverloaded(err) {
		t.Errorf("alive: %v", err)
	}
	s.count(2)
	if len(*waits) != 0 {
		t.Errorf("a probe waited %v", *waits)
	}
}

func TestOnlyOverloadIsWorthAnotherAttempt(t *testing.T) {
	for _, step := range []http.HandlerFunc{
		reply(http.StatusInternalServerError, `{"error":{"code":"internal","message":"broken"}}`),
		reply(http.StatusServiceUnavailable, `{"error":{"code":"shutting_down","message":"going"}}`, "Retry-After", "1"),
		reply(http.StatusBadGateway, "<html>"),
		reply(http.StatusNotFound, `{"error":{"code":"subject_not_found","message":"nobody"}}`),
		ok("<html>this is not the API</html>"),
	} {
		s := play(t, step)
		c, waits := impatient(t, s.url)
		if _, err := c.Search(ctx, []byte("face:anna"), SearchOptions{}); err == nil {
			t.Error("the call went through")
		}
		s.count(1)
		if len(*waits) != 0 {
			t.Errorf("waited %v", *waits)
		}
	}
}

func TestStopsWaitingWhenTheCallerIsGone(t *testing.T) {
	// the caller gives up during the pause
	s := play(t, busy("1"), ok(searchAnswer))
	c := newClient(t, s.url)
	going, cancel := context.WithCancel(ctx)
	c.sleep = func(ctx context.Context, d time.Duration) error {
		cancel()
		return sleep(ctx, d)
	}
	start := time.Now()
	_, err := c.Search(going, []byte("face:anna"), SearchOptions{})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled while waiting: %v", err)
	}
	// what the server last said is not lost
	if e := refused(t, err, http.StatusTooManyRequests, CodeOverloaded); !IsOverloaded(err) || e.RequestID != "busy-1" {
		t.Errorf("the refusal behind the cancellation: %+v", e)
	}
	if took := time.Since(start); took >= time.Second {
		t.Errorf("giving up took %v: the pause was waited out", took)
	}
	s.count(1)

	// the caller has less time left than the pause would take: the
	// refusal is handed over at once
	s = play(t, busy("1"), ok(searchAnswer))
	c, waits := impatient(t, s.url)
	hurried, cancel := context.WithTimeout(ctx, 900*time.Millisecond)
	defer cancel()
	_, err = c.Search(hurried, []byte("face:anna"), SearchOptions{})
	refused(t, err, http.StatusTooManyRequests, CodeOverloaded)
	if errors.Is(err, context.DeadlineExceeded) || len(*waits) != 0 {
		t.Errorf("with less than a second left: %v, waits %v", err, *waits)
	}
	s.count(1)

	// with time to spare the pause is taken
	s = play(t, busy("1"), ok(searchAnswer))
	c, waits = impatient(t, s.url)
	patient, cancel := context.WithTimeout(ctx, time.Minute)
	defer cancel()
	if _, err := c.Search(patient, []byte("face:anna"), SearchOptions{}); err != nil || len(*waits) != 1 {
		t.Errorf("with a minute left: %v, waits %v", err, *waits)
	}

	// a call without a context is a mistake, and is told so
	var none context.Context
	s = play(t)
	if _, err := newClient(t, s.url).Search(none, []byte("face:anna"), SearchOptions{}); err == nil || !strings.Contains(err.Error(), "nil Context") {
		t.Errorf("called without a context: %v", err)
	}
	s.count(0)

	// a caller who is gone before the call is not kept
	s = play(t)
	c, waits = impatient(t, s.url)
	gone, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := c.Search(gone, []byte("face:anna"), SearchOptions{}); !errors.Is(err, context.Canceled) {
		t.Errorf("called with a cancelled context: %v", err)
	}
	s.count(0)
	if len(*waits) != 0 {
		t.Errorf("waited %v", *waits)
	}
}

func TestSleepAndJitter(t *testing.T) {
	start := time.Now()
	if err := sleep(ctx, 20*time.Millisecond); err != nil || time.Since(start) < 20*time.Millisecond {
		t.Errorf("sleeping 20ms: %v after %v", err, time.Since(start))
	}
	gone, cancel := context.WithCancel(ctx)
	cancel()
	start = time.Now()
	if err := sleep(gone, time.Minute); !errors.Is(err, context.Canceled) || time.Since(start) > 10*time.Second {
		t.Errorf("sleeping with a cancelled context: %v after %v", err, time.Since(start))
	}
	short, cancel := context.WithTimeout(ctx, 10*time.Millisecond)
	defer cancel()
	if err := sleep(short, time.Minute); !errors.Is(err, context.DeadlineExceeded) || time.Since(start) > 10*time.Second {
		t.Errorf("sleeping past a deadline: %v after %v", err, time.Since(start))
	}

	for _, d := range []time.Duration{1, time.Millisecond, time.Second, 2 * time.Second} {
		lo, hi := d+d, time.Duration(0)
		for range 200 {
			got := jitter(d)
			lo, hi = min(lo, got), max(hi, got)
		}
		if lo < d || hi > d+d/4 {
			t.Errorf("a wait of %v became %v to %v", d, lo, hi)
		}
		if d >= time.Millisecond && lo == hi {
			t.Errorf("a wait of %v is always %v", d, lo)
		}
	}
	if got := jitter(0); got != 0 {
		t.Errorf("no wait became %v", got)
	}
	if got := jitter(-time.Second); got != 0 {
		t.Errorf("a negative wait became %v", got)
	}
}

func TestBrokenConnection(t *testing.T) {
	picture := []byte("face:anna")
	tests := []struct {
		name string
		call func(c *Client) error
		// answer is what the server says once it does answer.
		answer string
		// again says whether the call is safe to send a second time.
		again bool
	}{
		{"search", func(c *Client) error { _, err := c.Search(ctx, picture, SearchOptions{}); return err }, searchAnswer, true},
		{"verify", func(c *Client) error { _, err := c.Verify(ctx, "anna", picture, VerifyOptions{}); return err }, verifyAnswer, true},
		{"show", func(c *Client) error { _, err := c.Subject(ctx, "anna"); return err }, `{"subject":"anna"}`, true},
		{"list", func(c *Client) error { _, err := c.Subjects(ctx, "", 0); return err }, `{"subjects":[]}`, true},
		{"info", func(c *Client) error { _, err := c.Info(ctx); return err }, infoAnswer, true},
		{"remove a subject", func(c *Client) error { _, err := c.RemoveSubject(ctx, "anna"); return err }, removedAnswer, true},
		{"remove a face", func(c *Client) error { _, err := c.RemoveFace(ctx, "anna", 7); return err }, removedAnswer, true},
		{"enrol, replacing", func(c *Client) error {
			_, err := c.Enrol(ctx, "anna", picture, EnrolOptions{Replace: true})
			return err
		}, enrolAnswer, true},
		// the server may have enrolled the face before the line went dead
		{"enrol", func(c *Client) error { _, err := c.Enrol(ctx, "anna", picture, EnrolOptions{}); return err }, enrolAnswer, false},
		{"enrol, allowing a duplicate", func(c *Client) error {
			_, err := c.Enrol(ctx, "anna", picture, EnrolOptions{AllowDuplicate: true, AllowMismatch: true})
			return err
		}, enrolAnswer, false},
		// a probe wants to know how things are now
		{"ready", func(c *Client) error { return c.Ready(ctx) }, `{"status":"ready"}`, false},
		{"alive", func(c *Client) error { return c.Alive(ctx) }, `{"status":"ok"}`, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := play(t, hangUp, ok(tt.answer))
			c, waits := impatient(t, s.url, WithKey(adminKey))
			err := tt.call(c)
			if !tt.again {
				var e *Error
				if err == nil || errors.As(err, &e) || !strings.HasPrefix(err.Error(), "client: ") {
					t.Errorf("got %v, want the failure of the connection", err)
				} else {
					keyless(t, err)
				}
				s.count(1)
				return
			}
			if err != nil {
				t.Fatalf("after a broken connection: %v", err)
			}
			got := s.requests()
			if len(got) != 2 || got[0].method != got[1].method || got[0].target != got[1].target || !bytes.Equal(got[0].body, got[1].body) {
				t.Fatalf("the server received %+v", got)
			}
			if len(*waits) != 1 || (*waits)[0] != 100*time.Millisecond {
				t.Errorf("waited %v before the second attempt", *waits)
			}
		})
	}

	// a line that stays dead is given up on after the usual attempts
	s := play(t, hangUp, hangUp, hangUp)
	c, waits := impatient(t, s.url)
	_, err := c.Search(ctx, picture, SearchOptions{})
	var e *Error
	if err == nil || errors.As(err, &e) {
		t.Errorf("a dead line: %v", err)
	}
	s.count(3)
	if len(*waits) != 2 || (*waits)[1] != 200*time.Millisecond {
		t.Errorf("waited %v", *waits)
	}
	// and not tried twice by a client told not to
	s = play(t, hangUp)
	c, _ = impatient(t, s.url, WithRetries(0))
	if _, err := c.Search(ctx, picture, SearchOptions{}); err == nil {
		t.Error("a dead line gave an answer")
	}
	s.count(1)
}

func TestAnswerThatBreaksOff(t *testing.T) {
	// the server promises more than it sends
	short := func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Length", "500")
		w.WriteHeader(http.StatusOK)
		io.WriteString(w, `{"faces":[`)
	}
	s := play(t, short, ok(searchAnswer))
	c, waits := impatient(t, s.url)
	if _, err := c.Search(ctx, []byte("face:anna"), SearchOptions{}); err != nil {
		t.Errorf("a search whose first answer broke off: %v", err)
	}
	s.count(2)
	if len(*waits) != 1 {
		t.Errorf("waited %v", *waits)
	}

	// an enrolment is not repeated: the server answered, so it acted
	s = play(t, short)
	c, _ = impatient(t, s.url)
	_, err := c.Enrol(ctx, "anna", []byte("face:anna"), EnrolOptions{})
	if err == nil || !strings.Contains(err.Error(), "the answer broke off") || !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Errorf("an enrolment whose answer broke off: %v", err)
	}
	s.count(1)
}

func TestSlowServerIsNotAskedAgain(t *testing.T) {
	released := make(chan struct{})
	stall := func(_ http.ResponseWriter, r *http.Request) {
		select {
		case <-r.Context().Done():
		case <-released:
		}
	}
	s := play(t, stall)
	t.Cleanup(func() { close(released) })
	c, waits := impatient(t, s.url, WithHTTPClient(&http.Client{Timeout: 50 * time.Millisecond}))
	_, err := c.Search(ctx, []byte("face:anna"), SearchOptions{})
	if err == nil || !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("a server that does not answer in time: %v", err)
	}
	s.count(1)
	if len(*waits) != 0 {
		t.Errorf("waited %v", *waits)
	}

	// whichever of the client's timeouts it is that runs out
	s = play(t, stall)
	c, waits = impatient(t, s.url, WithHTTPClient(&http.Client{Transport: &http.Transport{ResponseHeaderTimeout: 50 * time.Millisecond}}))
	_, err = c.Search(ctx, []byte("face:anna"), SearchOptions{})
	if err == nil || !strings.Contains(err.Error(), "timeout awaiting response headers") {
		t.Errorf("a server that sends no headers in time: %v", err)
	}
	s.count(1)
	if len(*waits) != 0 {
		t.Errorf("waited %v", *waits)
	}

	// or a handshake that takes too long, which the transport reports in
	// its own way
	c, waits = impatient(t, "https://faces.invalid", WithHTTPClient(&http.Client{Transport: handshaking{}}))
	_, err = c.Search(ctx, []byte("face:anna"), SearchOptions{})
	if err == nil || !strings.Contains(err.Error(), "TLS handshake timeout") || len(*waits) != 0 {
		t.Errorf("a handshake that timed out: %v, waits %v", err, *waits)
	}

	// nor when it is the caller's deadline that passes
	s = play(t, stall)
	c, waits = impatient(t, s.url)
	hurried, cancel := context.WithTimeout(ctx, 50*time.Millisecond)
	defer cancel()
	if _, err := c.Search(hurried, []byte("face:anna"), SearchOptions{}); !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("a caller out of time: %v", err)
	}
	s.count(1)
	if len(*waits) != 0 {
		t.Errorf("waited %v", *waits)
	}
}

// handshaking is a transport whose TLS handshake never finishes in time.
type handshaking struct{}

func (handshaking) RoundTrip(*http.Request) (*http.Response, error) { return nil, slowHandshake{} }

// slowHandshake is a timeout as the transport reports one of a handshake:
// a net.Error that is not a context's deadline.
type slowHandshake struct{}

func (slowHandshake) Error() string   { return "net/http: TLS handshake timeout" }
func (slowHandshake) Timeout() bool   { return true }
func (slowHandshake) Temporary() bool { return true }

func TestAnswerTooLargeToBeOne(t *testing.T) {
	endless := func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		chunk := bytes.Repeat([]byte(" "), 1<<20)
		for range maxAnswer>>20 + 1 {
			if _, err := w.Write(chunk); err != nil {
				return
			}
		}
	}
	s := play(t, endless)
	c, _ := impatient(t, s.url)
	if _, err := c.Info(ctx); err == nil || !strings.Contains(err.Error(), "larger than") {
		t.Errorf("an answer without end: %v", err)
	}
	s.count(1)
}
