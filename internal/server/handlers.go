package server

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/url"
	"runtime"
	"strconv"
	"time"

	"github.com/muhriddinsalohiddin/likeness"
	"github.com/muhriddinsalohiddin/likeness/internal/gallery"
	"github.com/muhriddinsalohiddin/likeness/store"
)

const (
	// maxLimit is the most subjects one search returns.
	maxLimit = 50
	// maxFaces is the most faces analysed in one picture.
	maxFaces = 16
	// pageSize and maxPage bound a page of the subject listing.
	pageSize, maxPage = 100, 1000
)

// receive reads the picture of a request and its options.
func (s *Server) receive(c *call, r *http.Request) (*upload, *params, *apiError) {
	mark := time.Now()
	up, e := readUpload(r, s.cfg.MaxBody)
	if e != nil {
		return nil, nil, e
	}
	c.stage("read", time.Since(mark))
	p, e := gather(r, up.fields)
	return up, p, e
}

// analyse runs the engine on a picture once a slot is free.
func (s *Server) analyse(ctx context.Context, c *call, picture []byte, o likeness.Options) (*likeness.Result, *apiError) {
	o.MinLiveness = s.cfg.LivenessThreshold
	waited, err := s.limit.acquire(ctx)
	c.stage("queue", waited)
	if err != nil {
		return nil, s.turnAway(err)
	}
	res, err := func() (*likeness.Result, error) {
		start := time.Now()
		defer func() { s.limit.release(time.Since(start)) }()
		return s.cfg.Engine.Analyze(ctx, picture, o)
	}()
	if err != nil {
		return nil, pictureError(err)
	}
	t := res.Timing
	c.stage("decode", t.Decode)
	c.stage("detect", t.Detect)
	c.stage("align", t.Align)
	if s.cfg.LivenessThreshold > 0 {
		c.stage("liveness", t.Live)
	}
	c.stage("embed", t.Embed)
	return res, nil
}

// threshold reads the threshold a request asks for. A request may be
// stricter than the server, not laxer: a lower threshold in a search is a
// way of asking for candidates.
func (s *Server) threshold(p *params, own float32) float32 {
	asked, ok := p.fraction("threshold")
	if !ok {
		return own
	}
	if asked < own && !s.cfg.AllowCandidates {
		p.fail("threshold", "cannot be lower than the server's, "+strconv.FormatFloat(similarity(own), 'f', -1, 64))
		return own
	}
	return asked
}

func checkSubject(id string) *apiError {
	if store.ValidSubject(id) != nil {
		return invalidSubject()
	}
	return nil
}

func invalidSubject() *apiError {
	return refuse(http.StatusBadRequest, "invalid_subject",
		"a subject id is 1 to %d bytes of UTF-8 without spaces, control characters or slashes", store.MaxSubject)
}

type matchJSON struct {
	Subject  string  `json:"subject"`
	Score    float64 `json:"score"`
	Template uint64  `json:"template"`
}

type searchFace struct {
	faceJSON
	// Match says whether the face is somebody enrolled: Matches[0].
	Match   bool        `json:"match"`
	Matches []matchJSON `json:"matches"`
	// Margin is how far the best subject is ahead of the next one.
	Margin *float64 `json:"margin,omitempty"`
}

type searchResponse struct {
	Faces     []searchFace    `json:"faces"`
	Found     int             `json:"faces_found"`
	Reject    likeness.Reason `json:"reject,omitempty"`
	Threshold float64         `json:"threshold"`
	Width     int             `json:"width"`
	Height    int             `json:"height"`
	Timing    stages          `json:"timing_ms"`
}

// search answers POST /v1/search: who is in this picture?
func (s *Server) search(c *call, w http.ResponseWriter, r *http.Request) *apiError {
	up, p, e := s.receive(c, r)
	if e != nil {
		return e
	}
	limit := p.number("limit", 1, 1, maxLimit)
	threshold := s.threshold(p, s.cfg.SearchThreshold)
	which := p.oneOf("faces", "largest", "all")
	candidates := p.flag("candidates")
	if e := p.finish(); e != nil {
		return e
	}
	if candidates && !s.cfg.AllowCandidates {
		return refuse(http.StatusForbidden, "candidates_disabled",
			"this server only names subjects that reach the threshold")
	}

	o := likeness.ForSearch()
	if which == "all" {
		o.Faces, o.MaxFaces = likeness.All, maxFaces
	}
	res, e := s.analyse(r.Context(), c, up.picture, o)
	if e != nil {
		return e
	}

	out := searchResponse{
		Faces: make([]searchFace, 0, len(res.Faces)), Found: res.Found, Reject: res.Reject,
		Threshold: similarity(threshold), Width: res.Width, Height: res.Height,
	}
	mark := time.Now()
	matched, embedded := false, false
	for i := range res.Faces {
		f := &res.Faces[i]
		face := searchFace{faceJSON: describe(f), Matches: []matchJSON{}}
		if f.Embedding != nil {
			embedded = true
			// One more than asked for, to tell how far ahead the best is.
			found, err := s.cfg.Gallery.Search(f.Embedding, max(limit, 2))
			if err != nil {
				return internal(err)
			}
			face.Match = len(found) > 0 && found[0].Score >= threshold
			for _, m := range found[:min(limit, len(found))] {
				if m.Score < threshold && !candidates {
					break
				}
				face.Matches = append(face.Matches, matchJSON{m.Subject, similarity(m.Score), m.Template})
			}
			if len(found) > 1 && (face.Match || candidates) {
				margin := similarity(found[0].Score - found[1].Score)
				face.Margin = &margin
			}
			matched = matched || face.Match
		}
		out.Faces = append(out.Faces, face)
	}
	c.stage("search", time.Since(mark))

	switch {
	case matched:
		c.result = "match"
	case embedded:
		c.result = "no_match"
	default:
		c.result = "rejected"
		_, c.reject = res.Best()
	}
	out.Timing = c.timing()
	s.reply(w, c, http.StatusOK, out)
	return nil
}

type verifyResponse struct {
	Subject  string `json:"subject"`
	Verified bool   `json:"verified"`
	// Score and Template are absent when the picture had no usable face.
	Score     *float64        `json:"score,omitempty"`
	Template  uint64          `json:"template,omitempty"`
	Threshold float64         `json:"threshold"`
	Reject    likeness.Reason `json:"reject,omitempty"`
	Face      *faceJSON       `json:"face,omitempty"`
	Timing    stages          `json:"timing_ms"`
}

// verify answers POST /v1/verify: is this a picture of that subject?
func (s *Server) verify(c *call, w http.ResponseWriter, r *http.Request) *apiError {
	up, p, e := s.receive(c, r)
	if e != nil {
		return e
	}
	subject, named := p.take("subject")
	threshold := s.threshold(p, s.cfg.VerifyThreshold)
	if e := p.finish(); e != nil {
		return e
	}
	if !named {
		return refuse(http.StatusBadRequest, "missing_subject", "name the subject to verify against in the subject parameter")
	}
	if e := checkSubject(subject); e != nil {
		return e
	}
	c.subject = subject
	if !s.cfg.Gallery.Has(subject) {
		return notEnrolled() // before any time is spent on the picture
	}

	res, e := s.analyse(r.Context(), c, up.picture, likeness.ForSearch())
	if e != nil {
		return e
	}
	out := verifyResponse{Subject: subject, Threshold: similarity(threshold)}
	face, reason := res.Best()
	if face != nil {
		seen := describe(face)
		out.Face = &seen
	}
	if reason != likeness.Accepted {
		out.Reject = reason
		c.result, c.reject = "rejected", reason
	} else {
		mark := time.Now()
		m, enrolled, err := s.cfg.Gallery.Compare(subject, face.Embedding)
		if err != nil {
			return internal(err)
		}
		if !enrolled {
			return notEnrolled() // removed while the picture was analysed
		}
		c.stage("compare", time.Since(mark))
		score := similarity(m.Score)
		out.Score, out.Template = &score, m.Template
		out.Verified = m.Score >= threshold
		c.result = "not_verified"
		if out.Verified {
			c.result = "verified"
		}
	}
	out.Timing = c.timing()
	s.reply(w, c, http.StatusOK, out)
	return nil
}

// templateMeta is what is stored next to a template.
type templateMeta struct {
	Quality qualityJSON `json:"quality"`
}

// TemplateMeta encodes what the server keeps next to a template: the
// quality measured when the face was enrolled. Anything that enrols by
// other means should store the same, so that the API can show it.
func TemplateMeta(q likeness.Quality) ([]byte, error) {
	return json.Marshal(templateMeta{Quality: qualityOf(q)})
}

type enrolResponse struct {
	Subject  string `json:"subject"`
	Template uint64 `json:"template"`
	// Templates is how many templates the subject holds now.
	Templates int       `json:"templates"`
	Created   time.Time `json:"created"`
	Face      faceJSON  `json:"face"`
	Timing    stages    `json:"timing_ms"`
}

// enrol answers POST /v1/subjects/{id}/faces: remember this face as that
// subject's.
func (s *Server) enrol(c *call, w http.ResponseWriter, r *http.Request) *apiError {
	subject := r.PathValue("id")
	if e := checkSubject(subject); e != nil {
		return e
	}
	c.subject = subject
	up, p, e := s.receive(c, r)
	if e != nil {
		return e
	}
	o := gallery.EnrolOptions{
		Replace:        p.flag("replace"),
		MaxTemplates:   s.cfg.MaxTemplates,
		DuplicateAbove: s.cfg.SearchThreshold,
		MismatchBelow:  s.cfg.VerifyThreshold,
	}
	if p.flag("allow_duplicate") {
		o.DuplicateAbove = 0 // twins exist
	}
	if p.flag("allow_mismatch") {
		o.MismatchBelow = 0 // and people change
	}
	if e := p.finish(); e != nil {
		return e
	}

	res, e := s.analyse(r.Context(), c, up.picture, likeness.ForEnrolment())
	if e != nil {
		return e
	}
	face, reason := res.Best()
	if reason != likeness.Accepted {
		c.result, c.reject = "rejected", reason
		e := refuse(http.StatusUnprocessableEntity, "face_rejected", "the picture cannot serve as a reference: %s", advice(reason))
		e.Reject = reason
		if face != nil {
			measured := qualityOf(face.Quality)
			e.Quality = &measured
		}
		return e
	}
	meta, err := TemplateMeta(face.Quality)
	if err != nil {
		return internal(err)
	}

	mark := time.Now()
	// Not the request's context: a caller that hangs up must not leave a
	// replacement half done.
	done, err := s.cfg.Gallery.Enrol(context.WithoutCancel(r.Context()), subject, face.Embedding, meta, o)
	c.stage("store", time.Since(mark))
	if err != nil {
		return s.enrolError(c, err)
	}
	c.result = "enrolled"
	w.Header().Set("Location", "/v1/subjects/"+url.PathEscape(subject))
	s.reply(w, c, http.StatusCreated, enrolResponse{
		Subject: subject, Template: done.Template.ID, Templates: done.Templates,
		Created: done.Template.Created.UTC(), Face: describe(face), Timing: c.timing(),
	})
	return nil
}

func (s *Server) enrolError(c *call, err error) *apiError {
	var (
		taken *gallery.DuplicateError
		odd   *gallery.MismatchError
	)
	switch {
	case errors.As(err, &taken):
		c.result = "duplicate"
		e := refuse(http.StatusConflict, "duplicate_face",
			"this face is already enrolled as another subject; set allow_duplicate if they really are two people")
		score := similarity(taken.Score)
		e.Subject, e.Score = taken.Subject, &score
		return e
	case errors.As(err, &odd):
		c.result = "mismatch"
		e := refuse(http.StatusConflict, "subject_mismatch",
			"the face does not look like the subject's earlier pictures; set allow_mismatch or replace to enrol it anyway")
		score := similarity(odd.Score)
		e.Score = &score
		return e
	case errors.Is(err, gallery.ErrTooManyTemplates):
		c.result = "full"
		return refuse(http.StatusConflict, "too_many_templates",
			"the subject already holds %d templates; remove one, or set replace to start afresh", s.cfg.MaxTemplates)
	case errors.Is(err, store.ErrSubject):
		return invalidSubject()
	case errors.Is(err, store.ErrReadOnly):
		return refuse(http.StatusForbidden, "read_only", "this server does not accept changes")
	}
	return internal(err)
}

type templateJSON struct {
	ID      uint64       `json:"id"`
	Created time.Time    `json:"created"`
	Quality *qualityJSON `json:"quality,omitempty"`
}

type subjectResponse struct {
	Subject   string         `json:"subject"`
	Created   time.Time      `json:"created"`
	Templates []templateJSON `json:"templates"`
}

// showSubject answers GET /v1/subjects/{id}.
func (s *Server) showSubject(c *call, w http.ResponseWriter, r *http.Request) *apiError {
	id := r.PathValue("id")
	if e := checkSubject(id); e != nil {
		return e
	}
	c.subject = id
	if e := noOptions(r); e != nil {
		return e
	}
	sub, err := s.cfg.Gallery.Subject(r.Context(), id)
	switch {
	case errors.Is(err, store.ErrNotFound):
		return notEnrolled()
	case err != nil:
		return storeError(err)
	}
	out := subjectResponse{Subject: sub.ID, Created: sub.Created.UTC(), Templates: make([]templateJSON, len(sub.Templates))}
	for i, t := range sub.Templates {
		out.Templates[i] = templateJSON{ID: t.ID, Created: t.Created.UTC()}
		// Templates enrolled by other means carry other metadata, or none.
		var meta templateMeta
		if json.Unmarshal(t.Meta, &meta) == nil && meta.Quality != (qualityJSON{}) {
			out.Templates[i].Quality = &meta.Quality
		}
	}
	s.reply(w, c, http.StatusOK, out)
	return nil
}

type summaryJSON struct {
	Subject   string    `json:"subject"`
	Created   time.Time `json:"created"`
	Templates int       `json:"templates"`
}

type listResponse struct {
	Subjects []summaryJSON `json:"subjects"`
	// Next is the value of "after" that continues the listing; absent on
	// the last page.
	Next string `json:"next,omitempty"`
}

// listSubjects answers GET /v1/subjects: the enrolled people in id order,
// a page at a time.
func (s *Server) listSubjects(c *call, w http.ResponseWriter, r *http.Request) *apiError {
	p, e := gather(r, nil)
	if e != nil {
		return e
	}
	after, _ := p.take("after")
	limit := p.number("limit", pageSize, 1, maxPage)
	if e := p.finish(); e != nil {
		return e
	}
	page, err := s.cfg.Gallery.Subjects(r.Context(), after, limit)
	if err != nil {
		return storeError(err)
	}
	out := listResponse{Subjects: make([]summaryJSON, len(page))}
	for i, sub := range page {
		out.Subjects[i] = summaryJSON{Subject: sub.ID, Created: sub.Created.UTC(), Templates: sub.Templates}
	}
	if len(page) == limit {
		out.Next = page[len(page)-1].ID
	}
	s.reply(w, c, http.StatusOK, out)
	return nil
}

type removedResponse struct {
	Subject string `json:"subject"`
	// Template is the template removed, when one was named.
	Template uint64 `json:"template,omitempty"`
	// Removed is how many templates went, Templates how many the subject
	// has left; none left means the subject is gone.
	Removed   int `json:"removed"`
	Templates int `json:"templates"`
}

// removeSubject answers DELETE /v1/subjects/{id}.
func (s *Server) removeSubject(c *call, w http.ResponseWriter, r *http.Request) *apiError {
	id := r.PathValue("id")
	if e := checkSubject(id); e != nil {
		return e
	}
	c.subject = id
	if e := noOptions(r); e != nil {
		return e
	}
	n, err := s.cfg.Gallery.RemoveSubject(context.WithoutCancel(r.Context()), id)
	if err != nil {
		return storeError(err)
	}
	if n == 0 {
		return notEnrolled()
	}
	s.reply(w, c, http.StatusOK, removedResponse{Subject: id, Removed: n})
	return nil
}

// removeFace answers DELETE /v1/subjects/{id}/faces/{template}.
func (s *Server) removeFace(c *call, w http.ResponseWriter, r *http.Request) *apiError {
	id := r.PathValue("id")
	if e := checkSubject(id); e != nil {
		return e
	}
	c.subject = id
	template, err := strconv.ParseUint(r.PathValue("template"), 10, 64)
	if err != nil {
		return invalid("template", "must be the number of a template")
	}
	if e := noOptions(r); e != nil {
		return e
	}
	found, err := s.cfg.Gallery.Remove(context.WithoutCancel(r.Context()), id, template)
	if err != nil {
		return storeError(err)
	}
	if !found {
		if !s.cfg.Gallery.Has(id) {
			return notEnrolled()
		}
		return refuse(http.StatusNotFound, "template_not_found", "the subject has no such template")
	}
	s.reply(w, c, http.StatusOK, removedResponse{
		Subject: id, Template: template, Removed: 1, Templates: s.cfg.Gallery.Templates(id),
	})
	return nil
}

// noOptions refuses a query string on a request that takes none.
func noOptions(r *http.Request) *apiError {
	p, e := gather(r, nil)
	if e != nil {
		return e
	}
	return p.finish()
}

func storeError(err error) *apiError {
	switch {
	case errors.Is(err, context.Canceled), errors.Is(err, context.DeadlineExceeded):
		return gone(err)
	case errors.Is(err, store.ErrReadOnly):
		return refuse(http.StatusForbidden, "read_only", "this server does not accept changes")
	case errors.Is(err, store.ErrSubject):
		return invalidSubject()
	}
	return internal(err)
}

type livenessJSON struct {
	Models    []string `json:"models"`
	Threshold float64  `json:"threshold"`
}

type infoResponse struct {
	Version     string `json:"version"`
	Preset      string `json:"preset"`
	Detector    string `json:"detector"`
	Embedder    string `json:"embedder"`
	Fingerprint string `json:"fingerprint"`
	Dim         int    `json:"dim"`
	Runtime     string `json:"runtime"`
	Threads     int    `json:"threads"`
	Kernel      string `json:"kernel"`
	Thresholds  struct {
		Search float64 `json:"search"`
		Verify float64 `json:"verify"`
	} `json:"thresholds"`
	// Liveness is absent when faces are not checked for being live.
	Liveness *livenessJSON `json:"liveness,omitempty"`
	Gallery  struct {
		Subjects  int       `json:"subjects"`
		Templates int       `json:"templates"`
		Encrypted bool      `json:"encrypted"`
		Created   time.Time `json:"created"`
	} `json:"gallery"`
	Limits struct {
		MaxBody        int64   `json:"max_body"`
		MaxInflight    int     `json:"max_inflight"`
		Queue          int     `json:"queue"`
		QueueTimeoutMS float64 `json:"queue_timeout_ms"`
		MaxTemplates   int     `json:"max_templates"`
		SearchLimit    int     `json:"search_limit"`
		Candidates     bool    `json:"candidates"`
	} `json:"limits"`
}

// showInfo answers GET /v1/info: what this server runs and holds.
func (s *Server) showInfo(c *call, w http.ResponseWriter, r *http.Request) *apiError {
	if e := noOptions(r); e != nil {
		return e
	}
	held := s.cfg.Gallery.Info()
	out := infoResponse{
		Version: likeness.Version, Preset: s.info.Preset, Detector: s.info.Detector, Embedder: s.info.Embedder,
		Fingerprint: s.info.Fingerprint, Dim: s.info.Dim, Runtime: s.info.Runtime, Threads: s.info.Threads,
		Kernel: held.Kernel,
	}
	if s.cfg.LivenessThreshold > 0 {
		out.Liveness = &livenessJSON{Models: s.info.Liveness, Threshold: round(float64(s.cfg.LivenessThreshold), 3)}
	}
	out.Thresholds.Search = similarity(s.cfg.SearchThreshold)
	out.Thresholds.Verify = similarity(s.cfg.VerifyThreshold)
	out.Gallery.Subjects, out.Gallery.Templates = held.Subjects, held.Templates
	out.Gallery.Encrypted, out.Gallery.Created = held.Store.Encrypted, held.Store.Created.UTC()
	out.Limits.MaxBody = s.cfg.MaxBody
	out.Limits.MaxInflight, out.Limits.Queue = s.cfg.MaxInflight, s.cfg.Queue
	out.Limits.QueueTimeoutMS = millis(s.cfg.QueueTimeout)
	out.Limits.MaxTemplates = s.cfg.MaxTemplates
	out.Limits.SearchLimit, out.Limits.Candidates = maxLimit, s.cfg.AllowCandidates
	s.reply(w, c, http.StatusOK, out)
	return nil
}

// showMetrics answers GET /metrics in the Prometheus text format.
func (s *Server) showMetrics(_ *call, w http.ResponseWriter, _ *http.Request) *apiError {
	held := s.cfg.Gallery.Info()
	w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
	err := s.metrics.write(w, gauges{
		running: s.limit.running(), queued: s.limit.queued(), inside: int(s.limit.inside.Load()),
		subjects: held.Subjects, templates: held.Templates,
		build: [][2]string{
			{"version", likeness.Version}, {"preset", s.info.Preset}, {"detector", s.info.Detector},
			{"embedder", s.info.Embedder}, {"kernel", held.Kernel}, {"go", runtime.Version()},
		},
	})
	if err != nil {
		return gone(err) // the scraper hung up; there is no one to tell
	}
	return nil
}

type statusResponse struct {
	Status string `json:"status"`
}

// healthz answers while the process is alive.
func (s *Server) healthz(c *call, w http.ResponseWriter, _ *http.Request) *apiError {
	s.reply(w, c, http.StatusOK, statusResponse{"ok"})
	return nil
}

// readyz answers 200 while the server wants requests. A server is ready
// from the moment it listens: the models are loaded and the gallery is in
// memory before it does.
func (s *Server) readyz(c *call, w http.ResponseWriter, _ *http.Request) *apiError {
	if s.draining.Load() {
		return refuse(http.StatusServiceUnavailable, "shutting_down", "the server is shutting down")
	}
	s.reply(w, c, http.StatusOK, statusResponse{"ready"})
	return nil
}
