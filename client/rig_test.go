package client

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/muhriddinsalohiddin/likeness/index"
	"github.com/muhriddinsalohiddin/likeness/internal/enginetest"
	"github.com/muhriddinsalohiddin/likeness/internal/gallery"
	"github.com/muhriddinsalohiddin/likeness/internal/server"
	"github.com/muhriddinsalohiddin/likeness/store"
)

// The tests call the real server, built around a stand-in engine whose
// "pictures" are lines of text (see enginetest), so they need no models.
// Its faces resemble each other as follows, and it takes 0.7 for a match in
// a search and 0.5 in a verification:
//
//	anna ~ anna#2   0.89      anna ~ boris   -0.11
//	anna ~ anna#3   0.89      carol ~ anna    0.06
const (
	adminKey  = "admin-0123456789abcdef"
	searchKey = "search-0123456789abcdef"
)

var ctx = context.Background()

// rig is a likeness server over an empty gallery, listening on the
// loopback interface.
type rig struct {
	t    *testing.T
	fake *enginetest.Fake
	srv  *server.Server
	url  string
}

func newRig(t *testing.T, tune ...func(*server.Config)) *rig {
	t.Helper()
	fake := &enginetest.Fake{}
	info := fake.Info()
	st, err := store.Open(filepath.Join(t.TempDir(), "faces.db"), store.Options{Dim: info.Dim, Fingerprint: info.Fingerprint})
	if err != nil {
		t.Fatal(err)
	}
	g, err := gallery.Open(ctx, st, gallery.Options{Index: index.Options{OnHeap: true}})
	if err != nil {
		st.Close()
		t.Fatal(err)
	}
	t.Cleanup(func() { g.Close() })
	keys, err := server.NewKeys([]string{adminKey}, []string{searchKey}, nil)
	if err != nil {
		t.Fatal(err)
	}
	cfg := server.Config{Engine: fake, Gallery: g, Keys: keys}
	for _, f := range tune {
		f(&cfg)
	}
	srv, err := server.New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)
	return &rig{t: t, fake: fake, srv: srv, url: ts.URL}
}

// client makes a client for the rig; an empty key means none.
func (r *rig) client(key string, options ...Option) *Client {
	r.t.Helper()
	if key != "" {
		options = append([]Option{WithKey(key)}, options...)
	}
	return newClient(r.t, r.url, options...)
}

func (r *rig) admin() *Client    { r.t.Helper(); return r.client(adminKey) }
func (r *rig) searcher() *Client { r.t.Helper(); return r.client(searchKey) }

// enrol adds a face to a subject and insists that it worked.
func (r *rig) enrol(subject, picture string) *Enrolment {
	r.t.Helper()
	got, err := r.admin().Enrol(ctx, subject, []byte(picture), EnrolOptions{})
	if err != nil {
		r.t.Fatalf("enrolling %q as %s: %v", picture, subject, err)
	}
	return got
}

func newClient(t *testing.T, baseURL string, options ...Option) *Client {
	t.Helper()
	c, err := New(baseURL, options...)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

// impatient makes a client that notes the pauses it would have taken
// between attempts instead of taking them.
func impatient(t *testing.T, baseURL string, options ...Option) (*Client, *[]time.Duration) {
	t.Helper()
	c := newClient(t, baseURL, options...)
	waits := new([]time.Duration)
	c.jitter = func(d time.Duration) time.Duration { return d }
	c.sleep = func(ctx context.Context, d time.Duration) error {
		*waits = append(*waits, d)
		return ctx.Err()
	}
	return c, waits
}

// sent is a request as a script received it.
type sent struct {
	method string
	// target is the path and the query string as they came over the wire.
	target string
	header http.Header
	body   []byte
}

// script is a server that answers its requests with one prepared step
// after another and remembers what it was sent.
type script struct {
	t     *testing.T
	url   string
	steps []http.HandlerFunc

	mu   sync.Mutex
	seen []sent
}

// play starts a server that follows the steps.
func play(t *testing.T, steps ...http.HandlerFunc) *script {
	t.Helper()
	s := &script{t: t, steps: steps}
	ts := httptest.NewServer(s)
	t.Cleanup(ts.Close)
	s.url = ts.URL
	return s
}

func (s *script) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		s.t.Errorf("reading the request: %v", err)
	}
	s.mu.Lock()
	s.seen = append(s.seen, sent{r.Method, r.RequestURI, r.Header.Clone(), body})
	n := len(s.seen)
	s.mu.Unlock()
	if n > len(s.steps) {
		s.t.Errorf("request %d: the script has only %d steps", n, len(s.steps))
		http.Error(w, "off script", http.StatusInternalServerError)
		return
	}
	s.steps[n-1](w, r)
}

// requests returns what the script has received so far.
func (s *script) requests() []sent {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]sent(nil), s.seen...)
}

// count checks how many requests the script received.
func (s *script) count(want int) {
	s.t.Helper()
	if got := len(s.requests()); got != want {
		s.t.Errorf("the server received %d requests, want %d", got, want)
	}
}

// reply is a step that answers with a status and a body. Pairs of header
// names and values may follow.
func reply(status int, body string, headers ...string) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		for i := 0; i+1 < len(headers); i += 2 {
			w.Header().Set(headers[i], headers[i+1])
		}
		w.WriteHeader(status)
		io.WriteString(w, body)
	}
}

// busy is the refusal of an overloaded server; an empty retryAfter leaves
// the header out.
func busy(retryAfter string) http.HandlerFunc {
	const body = `{"error":{"code":"overloaded","message":"try again shortly"},"request_id":"busy-1"}`
	if retryAfter == "" {
		return reply(http.StatusTooManyRequests, body)
	}
	return reply(http.StatusTooManyRequests, body, "Retry-After", retryAfter)
}

// hangUp is a step that drops the connection without a word.
func hangUp(w http.ResponseWriter, _ *http.Request) {
	conn, _, err := w.(http.Hijacker).Hijack()
	if err != nil {
		panic(err)
	}
	conn.Close()
}

// Answers a likeness server could have given, for the scripts.
const (
	searchAnswer  = `{"faces":[],"faces_found":0,"reject":"no_face","threshold":0.7,"width":4,"height":3,"timing_ms":{"total":1}}`
	verifyAnswer  = `{"subject":"anna","verified":true,"score":0.9,"template":1,"threshold":0.5,"timing_ms":{"total":1}}`
	enrolAnswer   = `{"subject":"anna","template":7,"templates":1,"created":"2026-11-13T14:56:27Z","timing_ms":{"total":1}}`
	removedAnswer = `{"subject":"anna","removed":1,"templates":0}`
	infoAnswer    = `{"version":"test","limits":{"queue_timeout_ms":100}}`
)

func ok(body string) http.HandlerFunc {
	return reply(http.StatusOK, body, "X-Request-Id", "ok-1")
}

// refused insists that err is an answer of the server with the given
// status and code, and that it does not give the keys away.
func refused(t *testing.T, err error, status int, code Code) *Error {
	t.Helper()
	var e *Error
	if !errors.As(err, &e) {
		t.Fatalf("got %v, want the server's refusal %d %s", err, status, code)
	}
	if e.Status != status || e.Code != code {
		t.Fatalf("refused with %d %q, want %d %q: %v", e.Status, e.Code, status, code, err)
	}
	keyless(t, err)
	return e
}

// keyless checks that an error does not repeat an API key.
func keyless(t *testing.T, err error) {
	t.Helper()
	for _, key := range []string{adminKey, searchKey} {
		if strings.Contains(err.Error(), key) {
			t.Errorf("the error gives the API key away: %v", err)
		}
	}
}

// eventually waits for something another goroutine is about to do.
func eventually(t *testing.T, what string, done func() bool) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for !done() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(time.Millisecond)
	}
}

func near(got, want, tolerance float64) bool {
	return got >= want-tolerance && got <= want+tolerance
}
