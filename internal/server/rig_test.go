package server

import (
	"bytes"
	"context"
	"encoding/json"
	"log/slog"
	"mime/multipart"
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
	"github.com/muhriddinsalohiddin/likeness/store"
)

// The tests run against a stand-in engine whose "pictures" are lines of
// text (see enginetest), so they need no models. Its faces resemble each
// other as follows, and it takes 0.7 for a match in a search and 0.5 in a
// verification:
//
//	anna ~ anna#2   0.89      anna#2 ~ anna#3  0.77
//	anna ~ anna#3   0.89      anna ~ boris    -0.11
//	carol ~ boris   0.08      carol ~ anna     0.06
const (
	adminKey   = "admin-0123456789abcdef"
	searchKey  = "search-0123456789abcdef"
	metricsKey = "metrics-0123456789abcdef"
)

type rig struct {
	t       *testing.T
	fake    *enginetest.Fake
	gallery *gallery.Gallery
	srv     *Server
	logs    *logBuffer
}

// newRig builds a server over an empty gallery. Logging is at debug level
// unless a tune function replaces the logger.
func newRig(t *testing.T, tune ...func(*Config)) *rig {
	t.Helper()
	fake := &enginetest.Fake{}
	g := newGallery(t, fake)
	keys, err := NewKeys([]string{adminKey}, []string{searchKey}, []string{metricsKey})
	if err != nil {
		t.Fatal(err)
	}
	logs := &logBuffer{}
	cfg := Config{Engine: fake, Gallery: g, Keys: keys, Log: logs.logger(slog.LevelDebug)}
	for _, f := range tune {
		f(&cfg)
	}
	srv, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return &rig{t: t, fake: fake, gallery: g, srv: srv, logs: logs}
}

func newGallery(t *testing.T, engine Analyzer) *gallery.Gallery {
	t.Helper()
	info := engine.Info()
	st, err := store.Open(filepath.Join(t.TempDir(), "faces.db"), store.Options{Dim: info.Dim, Fingerprint: info.Fingerprint})
	if err != nil {
		t.Fatal(err)
	}
	g, err := gallery.Open(context.Background(), st, gallery.Options{Index: index.Options{OnHeap: true}})
	if err != nil {
		st.Close()
		t.Fatal(err)
	}
	t.Cleanup(func() { g.Close() })
	return g
}

// logBuffer collects the server's log lines.
type logBuffer struct {
	mu sync.Mutex
	b  bytes.Buffer
}

func (l *logBuffer) Write(p []byte) (int, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.b.Write(p)
}

func (l *logBuffer) String() string {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.b.String()
}

func (l *logBuffer) logger(level slog.Level) *slog.Logger {
	return slog.New(slog.NewJSONHandler(l, &slog.HandlerOptions{Level: level}))
}

// requests returns the "request" log lines so far.
func (l *logBuffer) requests(t *testing.T) []map[string]any {
	t.Helper()
	var lines []map[string]any
	for _, raw := range strings.Split(strings.TrimSpace(l.String()), "\n") {
		if raw == "" {
			continue
		}
		var line map[string]any
		if err := json.Unmarshal([]byte(raw), &line); err != nil {
			t.Fatalf("log line %q: %v", raw, err)
		}
		if line["msg"] == "request" {
			lines = append(lines, line)
		}
	}
	return lines
}

// answer is a response, read to its end.
type answer struct {
	t      *testing.T
	status int
	header http.Header
	body   []byte
}

func (a answer) want(status int) answer {
	a.t.Helper()
	if a.status != status {
		a.t.Fatalf("status %d, want %d; body: %s", a.status, status, a.body)
	}
	if got := a.header.Get("Content-Type"); got != "application/json" {
		a.t.Errorf("Content-Type %q", got)
	}
	return a
}

func (a answer) into(v any) {
	a.t.Helper()
	if err := json.Unmarshal(a.body, v); err != nil {
		a.t.Fatalf("decoding %s: %v", a.body, err)
	}
}

// wantError checks the status and the error code of a refusal.
func (a answer) wantError(status int, code string) errorView {
	a.t.Helper()
	a.want(status)
	var e errorView
	a.into(&e)
	if e.Error.Code != code {
		a.t.Fatalf("error code %q, want %q; body: %s", e.Error.Code, code, a.body)
	}
	if e.Error.Message == "" || e.RequestID == "" || e.RequestID != a.header.Get("X-Request-Id") {
		a.t.Errorf("an error carries a message and the request id: %s", a.body)
	}
	return e
}

func (r *rig) do(req *http.Request) answer {
	r.t.Helper()
	rec := httptest.NewRecorder()
	r.srv.Handler().ServeHTTP(rec, req)
	return answer{t: r.t, status: rec.Code, header: rec.Header(), body: rec.Body.Bytes()}
}

func (r *rig) send(method, target, key, contentType string, body []byte) answer {
	r.t.Helper()
	req := httptest.NewRequest(method, target, bytes.NewReader(body))
	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}
	if key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	return r.do(req)
}

// picture posts a picture as the request body.
func (r *rig) picture(target, key, picture string) answer {
	r.t.Helper()
	return r.send(http.MethodPost, target, key, "image/jpeg", []byte(picture))
}

func (r *rig) get(target, key string) answer {
	r.t.Helper()
	return r.send(http.MethodGet, target, key, "", nil)
}

// remove sends a DELETE as an administrator.
func (r *rig) remove(target string) answer {
	r.t.Helper()
	return r.send(http.MethodDelete, target, adminKey, "", nil)
}

// enrol adds a face to a subject and insists that it worked.
func (r *rig) enrol(subject, picture string) enrolView {
	r.t.Helper()
	var got enrolView
	r.picture("/v1/subjects/"+subject+"/faces", adminKey, picture).want(http.StatusCreated).into(&got)
	return got
}

func (r *rig) search(target, picture string) searchView {
	r.t.Helper()
	var got searchView
	r.picture(target, searchKey, picture).want(http.StatusOK).into(&got)
	return got
}

// form builds a multipart body with the picture in its "image" field.
func form(t *testing.T, fields map[string]string, picture string) (contentType string, body []byte) {
	t.Helper()
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)
	for name, value := range fields {
		if err := w.WriteField(name, value); err != nil {
			t.Fatal(err)
		}
	}
	if picture != "" {
		part, err := w.CreateFormFile(pictureField, "face.jpg")
		if err != nil {
			t.Fatal(err)
		}
		part.Write([]byte(picture))
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	return w.FormDataContentType(), buf.Bytes()
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

type matchView struct {
	Subject  string
	Score    float64
	Template uint64
}

type faceView struct {
	Box     map[string]float64
	Quality map[string]float64
	Reject  string
	Match   bool
	Matches []matchView
	Margin  *float64
}

type searchView struct {
	Faces     []faceView
	Found     int `json:"faces_found"`
	Reject    string
	Threshold float64
	Width     int
	Height    int
	Timing    map[string]float64 `json:"timing_ms"`
}

type verifyView struct {
	Subject   string
	Verified  bool
	Score     *float64
	Template  uint64
	Threshold float64
	Reject    string
	Face      *faceView
	Timing    map[string]float64 `json:"timing_ms"`
}

type enrolView struct {
	Subject   string
	Template  uint64
	Templates int
	Created   time.Time
	Face      faceView
	Timing    map[string]float64 `json:"timing_ms"`
}

type errorView struct {
	Error struct {
		Code    string
		Message string
		Reject  string
		Subject string
		Score   *float64
		Quality map[string]float64
	}
	RequestID string `json:"request_id"`
}

func near(got, want, tolerance float64) bool {
	return got >= want-tolerance && got <= want+tolerance
}
