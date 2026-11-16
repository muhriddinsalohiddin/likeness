package server

import (
	"context"
	"log/slog"
	"net"
	"net/http"
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

func TestNewChecksTheConfiguration(t *testing.T) {
	fake := &enginetest.Fake{}
	g := newGallery(t, fake)
	keys, err := NewKeys([]string{adminKey}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	// a gallery whose vectors another model made
	st, err := store.Open(filepath.Join(t.TempDir(), "other.db"), store.Options{Dim: fake.Info().Dim, Fingerprint: "another:model:0123"})
	if err != nil {
		t.Fatal(err)
	}
	foreign, err := gallery.Open(context.Background(), st, gallery.Options{Index: index.Options{OnHeap: true}})
	if err != nil {
		t.Fatal(err)
	}
	defer foreign.Close()

	for name, c := range map[string]struct {
		cfg  Config
		says string
	}{
		"no engine":          {Config{Gallery: g, Keys: keys}, "engine and a gallery"},
		"no gallery":         {Config{Engine: fake, Keys: keys}, "engine and a gallery"},
		"no keys":            {Config{Engine: fake, Gallery: g}, "no API keys"},
		"foreign gallery":    {Config{Engine: fake, Gallery: foreign, Keys: keys}, "another:model:0123"},
		"search threshold":   {Config{Engine: fake, Gallery: g, Keys: keys, SearchThreshold: 1.5}, "search threshold"},
		"verify threshold":   {Config{Engine: fake, Gallery: g, Keys: keys, VerifyThreshold: -0.2}, "verify threshold"},
		"threshold not real": {Config{Engine: fake, Gallery: g, Keys: keys, SearchThreshold: float32(nan())}, "search threshold"},
	} {
		if _, err := New(c.cfg); err == nil || !strings.Contains(err.Error(), c.says) {
			t.Errorf("%s: %v", name, err)
		}
	}

	srv, err := New(Config{Engine: fake, Gallery: g, Keys: keys})
	if err != nil {
		t.Fatal(err)
	}
	if c := srv.cfg; c.MaxInflight != 2 || c.Queue != 16 || c.QueueTimeout != 50*time.Millisecond || c.MaxBody != 10<<20 ||
		c.MaxTemplates != 10 || c.SearchThreshold != 0.7 || c.VerifyThreshold != 0.5 || c.AllowCandidates {
		t.Errorf("defaults: %+v", c)
	}
	srv, err = New(Config{Engine: fake, Gallery: g, NoAuth: true, Queue: -1, MaxInflight: 6, SearchThreshold: 0.9, VerifyThreshold: 0.8})
	if err != nil {
		t.Fatal(err)
	}
	if c := srv.cfg; c.Queue != 0 || c.MaxInflight != 6 || c.SearchThreshold != 0.9 || c.VerifyThreshold != 0.8 {
		t.Errorf("explicit settings: %+v", c)
	}
}

func nan() float64 {
	var zero float64
	return zero / zero
}

func TestInfoLogsLeaveThePeopleOut(t *testing.T) {
	logs := &logBuffer{}
	r := newRig(t, func(c *Config) { c.Log = logs.logger(slog.LevelInfo) })
	r.enrol("employee-4471", "face:anna")
	r.search("/v1/search", "face:anna")
	r.picture("/v1/verify?subject=employee-4471", searchKey, "face:boris").want(http.StatusOK)
	r.picture("/v1/subjects/employee-4471/faces", adminKey, "reject:blurry").want(http.StatusUnprocessableEntity)
	r.get("/v1/subjects/employee-4471", adminKey).want(http.StatusOK)
	r.get("/healthz", "").want(http.StatusOK)
	r.get("/metrics", metricsKey)

	lines := logs.requests(t)
	if len(lines) != 5 {
		t.Fatalf("%d request lines, want 5 (health checks and scrapes are not logged at this level):\n%s", len(lines), logs)
	}
	for i, want := range []map[string]any{
		{"route": "enrol", "method": "POST", "status": 201.0, "key": "admin-1", "result": "enrolled", "level": "INFO"},
		{"route": "search", "method": "POST", "status": 200.0, "key": "search-1", "result": "match"},
		{"route": "verify", "status": 200.0, "result": "not_verified"},
		{"route": "enrol", "status": 422.0, "result": "rejected", "reject": "blurry", "code": "face_rejected"},
		{"route": "subject", "method": "GET", "status": 200.0, "key": "admin-1"},
	} {
		for field, value := range want {
			if lines[i][field] != value {
				t.Errorf("line %d: %s is %v, want %v", i, field, lines[i][field], value)
			}
		}
		if id, _ := lines[i]["id"].(string); len(id) != 16 {
			t.Errorf("line %d has no request id: %v", i, lines[i])
		}
		if ms, _ := lines[i]["ms"].(float64); ms <= 0 {
			t.Errorf("line %d has no duration: %v", i, lines[i])
		}
	}
	if spent, _ := lines[1]["stages"].(map[string]any); spent["decode"] != 2.0 || spent["embed"] != 4.0 {
		t.Errorf("the search line says where the time went: %v", lines[1])
	}
	if lines[0]["bytes"] != float64(len("face:anna")) {
		t.Errorf("the size of the upload: %v", lines[0])
	}

	// who was looked at, with which key and in what picture stays out
	for _, private := range []string{"employee-4471", adminKey, searchKey, "face:anna"} {
		if strings.Contains(logs.String(), private) {
			t.Errorf("%q is in the log:\n%s", private, logs)
		}
	}
}

func TestDebugLogs(t *testing.T) {
	r := newRig(t) // logs at debug level
	r.enrol("employee-4471", "face:anna")
	r.get("/healthz", "").want(http.StatusOK)

	lines := r.logs.requests(t)
	if len(lines) != 2 {
		t.Fatalf("%d lines:\n%s", len(lines), r.logs)
	}
	if lines[0]["subject"] != "employee-4471" {
		t.Errorf("at debug level the subject is named: %v", lines[0])
	}
	if lines[1]["route"] != "healthz" || lines[1]["level"] != "DEBUG" {
		t.Errorf("health checks are logged at debug level: %v", lines[1])
	}
	if strings.Contains(r.logs.String(), adminKey) {
		t.Error("a key is never logged")
	}
}

func TestEngineFailures(t *testing.T) {
	r := newRig(t)
	r.enrol("anna", "face:anna")

	// an error the client can do nothing about is not explained to it
	e := r.picture("/v1/search", searchKey, "error:disk").wantError(http.StatusInternalServerError, "internal")
	if strings.Contains(e.Error.Message, "disk") || strings.Contains(e.Error.Message, "broke") {
		t.Errorf("the client was told %q", e.Error.Message)
	}
	// a panic neither, and it does not take the server down
	e = r.picture("/v1/search", searchKey, "panic").wantError(http.StatusInternalServerError, "internal")
	if strings.Contains(e.Error.Message, "blew up") {
		t.Errorf("the client was told %q", e.Error.Message)
	}
	if f := r.search("/v1/search", "face:anna").Faces[0]; !f.Match {
		t.Errorf("the server after a panic: %+v", f)
	}
	if l := r.srv.limit; l.running() != 0 || l.inside.Load() != 0 {
		t.Errorf("a failed request kept its place: %d running, %d inside", l.running(), l.inside.Load())
	}

	// the log has both in full
	text := r.logs.String()
	for _, want := range []string{"the engine broke: disk", "the engine blew up", "enginetest.(*Fake).Analyze", `"level":"ERROR"`} {
		if !strings.Contains(text, want) {
			t.Errorf("the log lacks %q:\n%s", want, text)
		}
	}
	var failed int
	for _, line := range r.logs.requests(t) {
		if line["status"] == 500.0 && line["level"] == "ERROR" {
			failed++
		}
	}
	if failed != 2 || r.srv.metrics.requests.get("search", "500") != 2 {
		t.Errorf("%d failures logged, %d counted", failed, r.srv.metrics.requests.get("search", "500"))
	}
}

// serving starts a rig on a real port and returns its address, the switch
// that stops it, and where Serve's result arrives.
func serving(t *testing.T, r *rig) (url string, stop context.CancelFunc, served <-chan error) {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- r.srv.Serve(ctx, ln) }()
	t.Cleanup(cancel)
	return "http://" + ln.Addr().String(), cancel, done
}

func TestServeStopsGracefully(t *testing.T) {
	r := newRig(t, func(c *Config) { c.DrainDelay = 20 * time.Millisecond })
	gate := make(chan struct{})
	r.fake.Gate = gate
	open := sync.OnceFunc(func() { close(gate) })
	t.Cleanup(open)
	url, stop, served := serving(t, r)

	res, err := http.Get(url + "/readyz")
	if err != nil || res.StatusCode != http.StatusOK {
		t.Fatalf("readyz: %v %v", res, err)
	}
	res.Body.Close()

	slow := later(func() reply {
		return post(context.Background(), url+"/v1/search", strings.NewReader("slow:anna"))
	})
	eventually(t, "the picture to reach the engine", func() bool { return r.srv.limit.running() == 1 })

	stop()
	eventually(t, "the server to start draining", r.srv.draining.Load)
	r.get("/readyz", "").wantError(http.StatusServiceUnavailable, "shutting_down")
	r.get("/healthz", "").want(http.StatusOK) // still alive, only not taking more

	// the request under way holds the server open
	select {
	case err := <-served:
		t.Fatalf("Serve returned while a request was running: %v", err)
	case <-time.After(100 * time.Millisecond):
	}
	open()
	if got := <-slow; got.err != nil || got.status != http.StatusOK {
		t.Errorf("the request that was under way: %+v", got)
	}
	select {
	case err := <-served:
		if err != nil {
			t.Errorf("Serve: %v", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("Serve did not return")
	}
	if res, err := http.Get(url + "/healthz"); err == nil {
		res.Body.Close()
		t.Error("the port still answers")
	}
}

func TestServeCutsOffWhatWillNotFinish(t *testing.T) {
	r := newRig(t)
	r.srv.grace = 30 * time.Millisecond
	gate := make(chan struct{})
	r.fake.Gate = gate
	open := sync.OnceFunc(func() { close(gate) })
	t.Cleanup(open)
	url, stop, served := serving(t, r)

	stuck := later(func() reply {
		return post(context.Background(), url+"/v1/search", strings.NewReader("slow:anna"))
	})
	eventually(t, "the picture to reach the engine", func() bool { return r.srv.limit.running() == 1 })
	stop()
	select {
	case err := <-served:
		if err == nil || !strings.Contains(err.Error(), "cut off") {
			t.Errorf("Serve: %v", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("Serve waited for a request that would not finish")
	}
	if got := <-stuck; got.err == nil {
		t.Errorf("the request that was cut off got an answer: %+v", got)
	}
}

func TestServeReportsABrokenListener(t *testing.T) {
	r := newRig(t)
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	ln.Close()
	if err := r.srv.Serve(context.Background(), ln); err == nil {
		t.Error("Serve on a closed listener returned without an error")
	}
}
