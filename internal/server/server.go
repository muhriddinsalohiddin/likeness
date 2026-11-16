// Package server is the HTTP side of likeness: it receives pictures, asks
// the engine what is in them and the gallery whose face it is.
//
// Requests that carry a picture pass through admission control. Only a few
// are analysed at a time, a few more wait briefly for their turn, and the
// rest are refused at once with 429: a server with more work than it can
// carry answers fewer requests, not all of them late.
package server

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"runtime/debug"
	"sort"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/muhriddinsalohiddin/likeness"
	"github.com/muhriddinsalohiddin/likeness/internal/gallery"
)

// Analyzer is the part of the engine the server uses. *likeness.Engine
// implements it.
type Analyzer interface {
	Analyze(ctx context.Context, picture []byte, o likeness.Options) (*likeness.Result, error)
	Info() likeness.Info
}

// Config describes a server. Engine and Gallery are required; any other
// field left zero takes the default its comment names.
type Config struct {
	Engine  Analyzer
	Gallery *gallery.Gallery

	// Keys are the API keys clients present. A server without any starts
	// only if NoAuth says that this is meant.
	Keys   Keys
	NoAuth bool

	// SearchThreshold and VerifyThreshold are the similarities from which
	// a face counts as a subject's: when it is looked up among everyone,
	// and when it is checked against one claimed subject. Zero means the
	// embedder's own.
	SearchThreshold, VerifyThreshold float32
	// AllowCandidates lets clients ask for the most similar subjects even
	// when none reaches the threshold. It is off unless asked for, because
	// it tells a caller who merely resembles whom.
	AllowCandidates bool

	// MaxInflight is how many pictures are analysed at the same time
	// (default 2). Queue is how many more requests may wait for their turn
	// (default 8 for each of those; negative for none) and QueueTimeout
	// how long (default 50 ms). A request that would wait longer, by how
	// long pictures have been taking, is refused with 429 on arrival
	// rather than served late.
	MaxInflight  int
	Queue        int
	QueueTimeout time.Duration

	// MaxBody bounds a request body, in bytes (default 10 MiB).
	MaxBody int64
	// MaxTemplates bounds the templates one subject may hold (default 10;
	// negative for no bound).
	MaxTemplates int

	// DrainDelay is how long the server keeps serving after it is told to
	// stop, while /readyz already says it is going: time for a load
	// balancer to notice and send new requests elsewhere.
	DrainDelay time.Duration

	// Log receives a line for every request. Nil discards them.
	Log *slog.Logger
}

const (
	defaultInflight     = 2
	queuePerSlot        = 8
	defaultQueueTimeout = 50 * time.Millisecond
	defaultMaxBody      = 10 << 20
	defaultMaxTemplates = 10

	// How long requests under way get to finish once the server stops.
	defaultGrace = 15 * time.Second
)

// Server answers HTTP requests. Make one with New.
type Server struct {
	cfg     Config
	info    likeness.Info
	log     *slog.Logger
	limit   *limiter
	metrics *metrics
	handler http.Handler
	grace   time.Duration

	draining atomic.Bool
}

// New checks the configuration and builds a server. The engine and the
// gallery stay the caller's to close, after Serve has returned.
func New(cfg Config) (*Server, error) {
	if cfg.Engine == nil || cfg.Gallery == nil {
		return nil, errors.New("server: an engine and a gallery are required")
	}
	if cfg.Keys.Len() == 0 && !cfg.NoAuth {
		return nil, errors.New("server: no API keys are set, and authentication has not been turned off")
	}
	info := cfg.Engine.Info()
	if made := cfg.Gallery.Info().Store; made.Dim != info.Dim || made.Fingerprint != info.Fingerprint {
		return nil, fmt.Errorf("server: the gallery holds vectors of %q (%d values) and the engine makes those of %q (%d)",
			made.Fingerprint, made.Dim, info.Fingerprint, info.Dim)
	}
	if cfg.SearchThreshold == 0 {
		cfg.SearchThreshold = info.SearchThreshold
	}
	if cfg.VerifyThreshold == 0 {
		cfg.VerifyThreshold = info.VerifyThreshold
	}
	for _, t := range []struct {
		name  string
		value float32
	}{{"search", cfg.SearchThreshold}, {"verify", cfg.VerifyThreshold}} {
		if !(t.value > 0 && t.value <= 1) {
			return nil, fmt.Errorf("server: the %s threshold is %v; it has to be above 0 and at most 1", t.name, t.value)
		}
	}
	if cfg.MaxInflight <= 0 {
		cfg.MaxInflight = defaultInflight
	}
	switch {
	case cfg.Queue == 0:
		cfg.Queue = queuePerSlot * cfg.MaxInflight
	case cfg.Queue < 0:
		cfg.Queue = 0
	}
	if cfg.QueueTimeout <= 0 {
		cfg.QueueTimeout = defaultQueueTimeout
	}
	if cfg.MaxBody <= 0 {
		cfg.MaxBody = defaultMaxBody
	}
	switch {
	case cfg.MaxTemplates == 0:
		cfg.MaxTemplates = defaultMaxTemplates
	case cfg.MaxTemplates < 0:
		cfg.MaxTemplates = 0
	}
	if cfg.Log == nil {
		cfg.Log = slog.New(slog.DiscardHandler)
	}

	s := &Server{
		cfg:     cfg,
		info:    info,
		log:     cfg.Log,
		limit:   newLimiter(cfg.MaxInflight, cfg.Queue, cfg.QueueTimeout),
		metrics: newMetrics(),
		grace:   defaultGrace,
	}
	s.handler = s.routes()
	return s, nil
}

// Handler is the server as an http.Handler, for mounting or testing.
func (s *Server) Handler() http.Handler { return s.handler }

// Serve answers requests arriving on the listener until the context is
// done, then stops gracefully: it reports itself as not ready, stops
// accepting connections and lets the requests under way finish. Serve
// closes the listener.
func (s *Server) Serve(ctx context.Context, ln net.Listener) error {
	srv := &http.Server{
		Handler:           s.handler,
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       30 * time.Second,
		WriteTimeout:      60 * time.Second,
		IdleTimeout:       2 * time.Minute,
		MaxHeaderBytes:    16 << 10,
		ErrorLog:          slog.NewLogLogger(s.log.Handler(), slog.LevelWarn),
	}
	stopped := make(chan error, 1)
	go func() { stopped <- srv.Serve(ln) }()

	select {
	case err := <-stopped:
		return fmt.Errorf("server: %w", err)
	case <-ctx.Done():
	}
	s.draining.Store(true)
	if s.cfg.DrainDelay > 0 {
		select {
		case err := <-stopped:
			return fmt.Errorf("server: %w", err)
		case <-time.After(s.cfg.DrainDelay):
		}
	}
	grace, cancel := context.WithTimeout(context.Background(), s.grace)
	defer cancel()
	if err := srv.Shutdown(grace); err != nil {
		_ = srv.Close()
		return fmt.Errorf("server: requests still running after %s were cut off", s.grace)
	}
	return nil
}

// route is one thing the server can be asked to do.
type route struct {
	name string
	need Scope // 0: open to anyone
	// picture routes pass through admission control.
	picture bool
	// quiet routes are health checks and scrapes: they are not counted and
	// only logged at debug level.
	quiet bool
	run   func(c *call, w http.ResponseWriter, r *http.Request) *apiError
}

func (s *Server) routes() http.Handler {
	mux := http.NewServeMux()
	methods := make(map[string][]string) // per path, what it answers to
	add := func(method, path string, rt route) {
		mux.Handle(method+" "+path, s.serve(rt))
		methods[path] = append(methods[path], method)
	}
	add("POST", "/v1/search", route{name: "search", need: ScopeSearch, picture: true, run: s.search})
	add("POST", "/v1/verify", route{name: "verify", need: ScopeSearch, picture: true, run: s.verify})
	add("POST", "/v1/subjects/{id}/faces", route{name: "enrol", need: ScopeAdmin, picture: true, run: s.enrol})
	add("GET", "/v1/subjects", route{name: "subjects", need: ScopeAdmin, run: s.listSubjects})
	add("GET", "/v1/subjects/{id}", route{name: "subject", need: ScopeAdmin, run: s.showSubject})
	add("DELETE", "/v1/subjects/{id}", route{name: "remove_subject", need: ScopeAdmin, run: s.removeSubject})
	add("DELETE", "/v1/subjects/{id}/faces/{template}", route{name: "remove_face", need: ScopeAdmin, run: s.removeFace})
	add("GET", "/v1/info", route{name: "info", need: ScopeSearch, run: s.showInfo})
	add("GET", "/metrics", route{name: "metrics", need: ScopeMetrics, quiet: true, run: s.showMetrics})
	add("GET", "/healthz", route{name: "healthz", quiet: true, run: s.healthz})
	add("GET", "/readyz", route{name: "readyz", quiet: true, run: s.readyz})

	// The mux would answer a wrong method or path in plain text; these
	// keep every answer in the one JSON shape.
	for path, allowed := range methods {
		sort.Strings(allowed)
		allow := strings.Join(allowed, ", ")
		mux.Handle(path, s.serve(route{name: "other", run: func(*call, http.ResponseWriter, *http.Request) *apiError {
			e := refuse(http.StatusMethodNotAllowed, "method_not_allowed", "this path answers to %s only", allow)
			e.allow = allow
			return e
		}}))
	}
	mux.Handle("/", s.serve(route{name: "other", run: func(*call, http.ResponseWriter, *http.Request) *apiError {
		return refuse(http.StatusNotFound, "not_found", "there is nothing at this path")
	}}))
	return mux
}

// call is what the server knows about one request while it handles it.
type call struct {
	id    string
	route string
	start time.Time
	// key is the label of the API key presented, never the key.
	key string
	// subject is logged at debug level only: it may be a personal number.
	subject string
	// result and reject say what a picture request came to.
	result string
	reject likeness.Reason
	stages stages
	final  stages
	err    *apiError
}

func (c *call) stage(name string, took time.Duration) {
	c.stages = append(c.stages, stage{name, took})
}

// timing closes the account of where the time went.
func (c *call) timing() stages {
	c.final = make(stages, len(c.stages), len(c.stages)+1)
	copy(c.final, c.stages)
	c.final = append(c.final, stage{"total", time.Since(c.start)})
	return c.final
}

// recorder remembers the status of a response for the log.
type recorder struct {
	http.ResponseWriter
	status int
}

func (r *recorder) WriteHeader(status int) {
	if r.status == 0 {
		r.status = status
	}
	r.ResponseWriter.WriteHeader(status)
}

func (r *recorder) Write(p []byte) (int, error) {
	if r.status == 0 {
		r.status = http.StatusOK
	}
	return r.ResponseWriter.Write(p)
}

func (r *recorder) Unwrap() http.ResponseWriter { return r.ResponseWriter }

// serve wraps a route in what every request gets: an id, a size limit,
// authentication, admission, a log line and a net under panics.
func (s *Server) serve(rt route) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c := &call{id: requestID(r), route: rt.name, start: time.Now()}
		// Limit the body through the real writer: it is how the connection
		// learns to close after an oversized request.
		r.Body = http.MaxBytesReader(w, r.Body, s.cfg.MaxBody)
		rec := &recorder{ResponseWriter: w}
		rec.Header().Set("X-Request-Id", c.id)
		rec.Header().Set("Cache-Control", "no-store")

		defer func() {
			if v := recover(); v != nil {
				if err, ok := v.(error); ok && errors.Is(err, http.ErrAbortHandler) {
					panic(v) // the standard way to drop a connection on purpose
				}
				s.log.LogAttrs(r.Context(), slog.LevelError, "panic",
					slog.String("id", c.id), slog.String("route", c.route),
					slog.String("panic", fmt.Sprint(v)), slog.String("stack", string(debug.Stack())))
				if rec.status == 0 {
					s.fail(rec, c, internal(nil))
				}
			}
			s.record(c, rec, r, rt)
		}()
		if e := s.handle(c, rt, rec, r); e != nil {
			s.fail(rec, c, e)
		}
	})
}

func (s *Server) handle(c *call, rt route, w http.ResponseWriter, r *http.Request) *apiError {
	if e := s.authorise(c, r, rt.need); e != nil {
		return e
	}
	if rt.picture {
		if err := s.limit.enter(); err != nil {
			return s.turnAway(err)
		}
		defer s.limit.leave()
	}
	return rt.run(c, w, r)
}

func (s *Server) authorise(c *call, r *http.Request, need Scope) *apiError {
	if need == 0 || s.cfg.NoAuth {
		return nil
	}
	token, ok := bearer(r.Header.Get("Authorization"))
	if !ok {
		s.metrics.denials.add("no_key", "")
		return refuse(http.StatusUnauthorized, "unauthorized", "send an API key in the Authorization header, as: Bearer <key>")
	}
	grants, label, ok := s.cfg.Keys.find(token)
	if !ok {
		s.metrics.denials.add("unknown_key", "")
		return refuse(http.StatusUnauthorized, "unauthorized", "the API key is not one this server knows")
	}
	c.key = label
	if grants&need == 0 {
		s.metrics.denials.add("no_right", "")
		return refuse(http.StatusForbidden, "forbidden", "this API key is not allowed to do that")
	}
	return nil
}

// turnAway refuses a request the limiter would not take.
func (s *Server) turnAway(err error) *apiError {
	cause := "queue_full"
	switch {
	case errors.Is(err, errQueueTimeout):
		cause = "queue_timeout"
	case errors.Is(err, errCrowded):
		cause = "too_many_uploads"
	case !errors.Is(err, errQueueFull):
		return gone(err) // the client hung up while waiting
	}
	s.metrics.refusals.add(cause, "")
	return overloaded(err)
}

// record counts a finished request and writes its log line.
func (s *Server) record(c *call, rec *recorder, r *http.Request, rt route) {
	status := rec.status
	if status == 0 {
		status = http.StatusOK
	}
	took := time.Since(c.start)
	level := slog.LevelInfo
	switch {
	case rt.quiet:
		level = slog.LevelDebug
	case status >= 500:
		level = slog.LevelError
	case status == http.StatusTooManyRequests:
		level = slog.LevelWarn
	}
	if !rt.quiet {
		s.metrics.requests.add(c.route, strconv.Itoa(status))
		s.metrics.durations.observe(c.route, took.Seconds())
		for _, st := range c.stages {
			s.metrics.stages.observe(st.name, st.took.Seconds())
		}
		if c.result != "" {
			s.metrics.results.add(c.route, c.result)
		}
		if c.reject != likeness.Accepted {
			s.metrics.rejects.add(c.route, string(c.reject))
		}
	}
	ctx := r.Context()
	if !s.log.Enabled(ctx, level) {
		return
	}
	attrs := make([]slog.Attr, 0, 16)
	attrs = append(attrs,
		slog.String("id", c.id), slog.String("route", c.route), slog.String("method", r.Method),
		slog.Int("status", status), slog.Float64("ms", millis(took)))
	if r.ContentLength > 0 {
		attrs = append(attrs, slog.Int64("bytes", r.ContentLength))
	}
	if host, _, err := net.SplitHostPort(r.RemoteAddr); err == nil {
		attrs = append(attrs, slog.String("remote", host))
	}
	if c.key != "" {
		attrs = append(attrs, slog.String("key", c.key))
	}
	if c.result != "" {
		attrs = append(attrs, slog.String("result", c.result))
	}
	if c.reject != likeness.Accepted {
		attrs = append(attrs, slog.String("reject", string(c.reject)))
	}
	if c.err != nil {
		attrs = append(attrs, slog.String("code", c.err.Code))
		if c.err.cause != nil {
			attrs = append(attrs, slog.String("error", c.err.cause.Error()))
		}
	}
	if len(c.stages) > 0 {
		spent := make([]any, len(c.stages))
		for i, st := range c.stages {
			spent[i] = slog.Float64(st.name, millis(st.took))
		}
		attrs = append(attrs, slog.Group("stages", spent...))
	}
	if c.subject != "" && s.log.Enabled(ctx, slog.LevelDebug) {
		attrs = append(attrs, slog.String("subject", c.subject))
	}
	s.log.LogAttrs(ctx, level, "request", attrs...)
}

// requestID takes the caller's id for the request if it sent a sane one,
// so that a request can be followed across services, and makes one up
// otherwise.
func requestID(r *http.Request) string {
	if id := r.Header.Get("X-Request-Id"); saneID(id) {
		return id
	}
	var b [8]byte
	_, _ = rand.Read(b[:]) // documented never to fail
	return hex.EncodeToString(b[:])
}

func saneID(id string) bool {
	if id == "" || len(id) > 64 {
		return false
	}
	for i := 0; i < len(id); i++ {
		c := id[i]
		letter := c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9'
		if !letter && c != '-' && c != '_' && c != '.' {
			return false
		}
	}
	return true
}
