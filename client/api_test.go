package client

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/muhriddinsalohiddin/likeness"
	"github.com/muhriddinsalohiddin/likeness/internal/server"
)

func TestSearch(t *testing.T) {
	r := newRig(t)
	r.enrol("anna", "face:anna")
	r.enrol("boris", "face:boris")
	c := r.searcher()

	res, err := c.Search(ctx, []byte("face:anna#2"), SearchOptions{})
	if err != nil {
		t.Fatal(err)
	}
	m := res.Best()
	if m == nil || m.Subject != "anna" || !near(m.Score, 0.887, 0.002) || m.Template == 0 {
		t.Fatalf("the best match: %+v", m)
	}
	if len(res.Faces) != 1 || res.FacesFound != 1 || res.Reject != "" || res.Threshold != 0.7 || res.Width != 1280 || res.Height != 720 {
		t.Errorf("the answer: %+v", res)
	}
	f := res.Faces[0]
	if !f.Match || len(f.Matches) != 1 || f.Reject != "" {
		t.Errorf("the face: %+v", f)
	}
	if f.Box != (Box{Left: 100, Top: 80, Right: 300, Bottom: 330}) {
		t.Errorf("box %+v", f.Box)
	}
	if f.Quality != (Quality{Score: 0.95, EyeDistance: 84, Yaw: 3, Pitch: -2, Roll: 1, InFrame: 1, Brightness: 140, Sharpness: 310}) {
		t.Errorf("quality %+v", f.Quality)
	}
	// boris is enrolled too, so the answer says how far ahead anna is
	if f.Margin == nil || !near(*f.Margin, 0.9, 0.2) {
		t.Errorf("margin %v", f.Margin)
	}
	// the stand-in engine always takes the same time
	tm := res.Timing
	if tm.Decode != 2*time.Millisecond || tm.Detect != 3*time.Millisecond || tm.Align != time.Millisecond || tm.Embed != 4*time.Millisecond {
		t.Errorf("stages of the engine: %+v", tm)
	}
	if tm.Total <= 0 || tm.Total > time.Minute || tm.Compare != 0 || tm.Store != 0 || len(tm.Other) != 0 {
		t.Errorf("timing: %+v", tm)
	}
	if res.RequestID == "" {
		t.Error("the answer has no request id")
	}

	// somebody who is not enrolled is an answer, not an error
	res, err = c.Search(ctx, []byte("face:carol"), SearchOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if res.Best() != nil || len(res.Faces) != 1 || res.Faces[0].Match || len(res.Faces[0].Matches) != 0 || res.Faces[0].Margin != nil {
		t.Errorf("a stranger: %+v", res)
	}
	// and so are a picture without a face and one with a poor face
	res, err = c.Search(ctx, []byte("reject:no_face"), SearchOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if res.Reject != ReasonNoFace || len(res.Faces) != 0 || res.Best() != nil {
		t.Errorf("no face: %+v", res)
	}
	res, err = c.Search(ctx, []byte("reject:blurry"), SearchOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if res.Reject != "" || len(res.Faces) != 1 || res.Faces[0].Reject != ReasonBlurry || res.Faces[0].Quality.Sharpness != 3 || res.Best() != nil {
		t.Errorf("a blurry face: %+v", res)
	}
}

func TestSearchOptions(t *testing.T) {
	r := newRig(t, func(c *server.Config) { c.AllowCandidates = true })
	r.enrol("anna", "face:anna")
	r.enrol("boris", "face:boris")
	r.enrol("carol", "face:carol")
	c := r.searcher()

	// every face in the picture
	res, err := c.Search(ctx, []byte("faces:dora,anna#2"), SearchOptions{AllFaces: true})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Faces) != 2 || res.FacesFound != 2 || res.Faces[0].Match || !res.Faces[1].Match {
		t.Fatalf("two faces: %+v", res)
	}
	// the best match is the strongest in the picture, not the largest face
	if m := res.Best(); m == nil || m.Subject != "anna" {
		t.Errorf("the best of two faces: %+v", m)
	}

	// a stricter threshold
	res, err = c.Search(ctx, []byte("face:anna#2"), SearchOptions{Threshold: 0.95})
	if err != nil {
		t.Fatal(err)
	}
	if res.Threshold != 0.95 || res.Best() != nil {
		t.Errorf("at 0.95: %+v", res)
	}

	// the nearest subjects, matched or not
	res, err = c.Search(ctx, []byte("face:anna#2"), SearchOptions{Limit: 3, Candidates: true})
	if err != nil {
		t.Fatal(err)
	}
	f := res.Faces[0]
	if len(f.Matches) != 3 || f.Matches[0].Subject != "anna" || f.Matches[1].Score >= 0.7 || f.Matches[1].Score < f.Matches[2].Score {
		t.Errorf("three candidates: %+v", f.Matches)
	}
	if m := res.Best(); m == nil || m.Subject != "anna" {
		t.Errorf("the best with candidates: %+v", m)
	}
	// a candidate below the threshold is not a match
	res, err = c.Search(ctx, []byte("face:dora"), SearchOptions{Limit: 2, Candidates: true})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Faces[0].Matches) != 2 || res.Faces[0].Match || res.Best() != nil {
		t.Errorf("candidates for a stranger: %+v", res.Faces[0])
	}
}

func TestVerify(t *testing.T) {
	r := newRig(t)
	enrolled := r.enrol("anna", "face:anna")
	c := r.searcher()

	v, err := c.Verify(ctx, "anna", []byte("face:anna#2"), VerifyOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if !v.Verified || v.Subject != "anna" || !near(v.Score, 0.887, 0.002) || v.Template != enrolled.Template || v.Threshold != 0.5 || v.Reject != "" {
		t.Errorf("anna, verified: %+v", v)
	}
	if v.Face == nil || v.Face.Box.Right != 300 || v.Face.Quality.EyeDistance != 84 {
		t.Errorf("the face compared: %+v", v.Face)
	}
	if v.Timing.Embed != 4*time.Millisecond || v.Timing.Total <= 0 || v.Timing.Search != 0 || v.RequestID == "" {
		t.Errorf("timing %+v, request id %q", v.Timing, v.RequestID)
	}

	// somebody else
	v, err = c.Verify(ctx, "anna", []byte("face:boris"), VerifyOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if v.Verified || !near(v.Score, -0.105, 0.002) {
		t.Errorf("boris as anna: %+v", v)
	}
	// a stricter threshold
	v, err = c.Verify(ctx, "anna", []byte("face:anna#2"), VerifyOptions{Threshold: 0.95})
	if err != nil {
		t.Fatal(err)
	}
	if v.Verified || v.Threshold != 0.95 {
		t.Errorf("at 0.95: %+v", v)
	}
	// a picture that cannot be compared is an answer that says why
	v, err = c.Verify(ctx, "anna", []byte("reject:too_dark"), VerifyOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if v.Verified || v.Reject != ReasonTooDark || v.Score != 0 || v.Template != 0 || v.Face == nil {
		t.Errorf("a dark picture: %+v", v)
	}
	v, err = c.Verify(ctx, "anna", []byte("reject:no_face"), VerifyOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if v.Verified || v.Reject != ReasonNoFace || v.Face != nil {
		t.Errorf("no face: %+v", v)
	}

	// someone who is not enrolled is an error, decided before the picture
	// is looked at
	calls := r.fake.Calls()
	_, err = c.Verify(ctx, "nobody", []byte("face:anna"), VerifyOptions{})
	refused(t, err, http.StatusNotFound, CodeSubjectNotFound)
	if !IsNotFound(err) || r.fake.Calls() != calls {
		t.Errorf("verifying against nobody: %v", err)
	}
	if _, err := c.Verify(ctx, "", []byte("face:anna"), VerifyOptions{}); err == nil || !strings.Contains(err.Error(), "subject id is empty") {
		t.Errorf("verifying against no subject: %v", err)
	}
}

func TestEnrol(t *testing.T) {
	r := newRig(t)
	c := r.admin()

	first, err := c.Enrol(ctx, "anna", []byte("face:anna"), EnrolOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if first.Subject != "anna" || first.Template == 0 || first.Templates != 1 || first.RequestID == "" {
		t.Errorf("the first template: %+v", first)
	}
	if age := time.Since(first.Created); age < 0 || age > time.Minute {
		t.Errorf("created %v", first.Created)
	}
	if first.Face.Box.Right != 300 || first.Face.Quality.Sharpness != 310 || first.Face.Reject != "" {
		t.Errorf("the face enrolled: %+v", first.Face)
	}
	if first.Timing.Store <= 0 || first.Timing.Detect != 3*time.Millisecond || first.Timing.Search != 0 {
		t.Errorf("timing: %+v", first.Timing)
	}

	// another picture adds a template
	second, err := c.Enrol(ctx, "anna", []byte("face:anna#2"), EnrolOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if second.Templates != 2 || second.Template == first.Template {
		t.Errorf("the second template: %+v", second)
	}
	// a replacement leaves only itself
	third, err := c.Enrol(ctx, "anna", []byte("face:anna#3"), EnrolOptions{Replace: true})
	if err != nil {
		t.Fatal(err)
	}
	if third.Templates != 1 {
		t.Errorf("after replacing: %+v", third)
	}

	// the guards of the gallery can be told to stand down
	_, err = c.Enrol(ctx, "annette", []byte("face:anna"), EnrolOptions{})
	if !IsDuplicate(err) {
		t.Fatalf("anna's face as annette: %v", err)
	}
	if _, err := c.Enrol(ctx, "annette", []byte("face:anna"), EnrolOptions{AllowDuplicate: true}); err != nil {
		t.Errorf("with AllowDuplicate: %v", err)
	}
	_, err = c.Enrol(ctx, "annette", []byte("face:boris"), EnrolOptions{})
	refused(t, err, http.StatusConflict, CodeSubjectMismatch)
	got, err := c.Enrol(ctx, "annette", []byte("face:boris"), EnrolOptions{AllowMismatch: true})
	if err != nil || got.Templates != 2 {
		t.Errorf("with AllowMismatch: %+v, %v", got, err)
	}

	if _, err := c.Enrol(ctx, "", []byte("face:anna"), EnrolOptions{}); err == nil || !strings.Contains(err.Error(), "subject id is empty") {
		t.Errorf("enrolling nobody: %v", err)
	}
}

func TestSubjects(t *testing.T) {
	r := newRig(t)
	c := r.admin()

	page, err := c.Subjects(ctx, "", 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(page.Subjects) != 0 || page.Next != "" || page.RequestID == "" {
		t.Errorf("an empty gallery: %+v", page)
	}
	for range c.AllSubjects(ctx) {
		t.Error("an empty gallery has nobody to iterate over")
	}

	for _, name := range []string{"dilnoza", "anna", "carol", "boris", "eldor"} {
		r.enrol(name, "face:"+name)
	}
	first := r.enrol("anna", "face:anna#2")

	// one subject
	sub, err := c.Subject(ctx, "anna")
	if err != nil {
		t.Fatal(err)
	}
	if sub.ID != "anna" || sub.Created.IsZero() || len(sub.Templates) != 2 || sub.RequestID == "" {
		t.Fatalf("anna: %+v", sub)
	}
	if tm := sub.Templates[1]; tm.ID != first.Template || tm.Created.IsZero() || tm.Quality == nil || tm.Quality.EyeDistance != 84 {
		t.Errorf("her second template: %+v", tm)
	}
	_, err = c.Subject(ctx, "nobody")
	if !IsNotFound(err) {
		t.Errorf("nobody: %v", err)
	}
	if _, err := c.Subject(ctx, ""); err == nil || !strings.Contains(err.Error(), "subject id is empty") {
		t.Errorf("no id: %v", err)
	}

	// everyone on one page
	page, err = c.Subjects(ctx, "", 0)
	if err != nil {
		t.Fatal(err)
	}
	if got := ids(page.Subjects); got != "anna boris carol dilnoza eldor" || page.Next != "" {
		t.Errorf("everyone: %s, next %q", got, page.Next)
	}
	if page.Subjects[0].Templates != 2 || page.Subjects[1].Templates != 1 || page.Subjects[0].Created.IsZero() {
		t.Errorf("summaries: %+v", page.Subjects)
	}

	// a page at a time
	var pages []string
	after := ""
	for {
		page, err = c.Subjects(ctx, after, 2)
		if err != nil {
			t.Fatal(err)
		}
		pages = append(pages, ids(page.Subjects))
		if page.Next == "" || len(pages) > 5 {
			break
		}
		after = page.Next
	}
	if got := strings.Join(pages, " | "); got != "anna boris | carol dilnoza | eldor" {
		t.Errorf("pages: %s", got)
	}

	// the iterator fetches the pages itself
	c.page = 2
	var seen []SubjectSummary
	for s, err := range c.AllSubjects(ctx) {
		if err != nil {
			t.Fatal(err)
		}
		seen = append(seen, s)
	}
	if got := ids(seen); got != "anna boris carol dilnoza eldor" {
		t.Errorf("iterated over %s", got)
	}
	// and stops when the caller has seen enough
	seen = nil
	for s, err := range c.AllSubjects(ctx) {
		if err != nil {
			t.Fatal(err)
		}
		if seen = append(seen, s); len(seen) == 3 {
			break
		}
	}
	if got := ids(seen); got != "anna boris carol" {
		t.Errorf("the first three: %s", got)
	}

	// a failure ends the iteration and is handed over
	turns := 0
	for s, err := range r.searcher().AllSubjects(ctx) {
		turns++
		refused(t, err, http.StatusForbidden, CodeForbidden)
		if s != (SubjectSummary{}) {
			t.Errorf("a subject came with the error: %+v", s)
		}
	}
	if turns != 1 {
		t.Errorf("a failed listing took %d turns", turns)
	}
}

func TestAllSubjectsFailsHalfWay(t *testing.T) {
	s := play(t,
		ok(`{"subjects":[{"subject":"anna","templates":1},{"subject":"boris","templates":2}],"next":"boris"}`),
		reply(http.StatusInternalServerError, `{"error":{"code":"internal","message":"the store broke"}}`),
	)
	c := newClient(t, s.url)
	var seen []string
	var failure error
	for sub, err := range c.AllSubjects(ctx) {
		if err != nil {
			failure = err
			continue
		}
		seen = append(seen, sub.ID)
	}
	refused(t, failure, http.StatusInternalServerError, CodeInternal)
	if strings.Join(seen, " ") != "anna boris" {
		t.Errorf("before the failure: %v", seen)
	}
	if got := s.requests(); len(got) != 2 || got[0].target != "/v1/subjects?limit=1000" || got[1].target != "/v1/subjects?after=boris&limit=1000" {
		t.Errorf("the pages asked for: %+v", got)
	}

	// a server that names a next page without having sent anything does
	// not keep the iterator going round
	s = play(t, ok(`{"subjects":[],"next":"anna"}`))
	for range newClient(t, s.url).AllSubjects(ctx) {
		t.Error("an empty page has nobody on it")
	}
	s.count(1)
}

func ids(subjects []SubjectSummary) string {
	names := make([]string, len(subjects))
	for i, s := range subjects {
		names[i] = s.ID
	}
	return strings.Join(names, " ")
}

func TestRemove(t *testing.T) {
	r := newRig(t)
	first := r.enrol("anna", "face:anna")
	second := r.enrol("anna", "face:anna#2")
	r.enrol("boris", "face:boris")
	r.enrol("boris", "face:boris#2")
	c := r.admin()

	// one template
	got, err := c.RemoveFace(ctx, "anna", first.Template)
	if err != nil {
		t.Fatal(err)
	}
	if got.RequestID == "" {
		t.Error("the removal has no request id")
	}
	got.RequestID = ""
	if *got != (Removal{Subject: "anna", Template: first.Template, Removed: 1, Templates: 1}) {
		t.Errorf("the first template: %+v", got)
	}
	_, err = c.RemoveFace(ctx, "anna", first.Template)
	refused(t, err, http.StatusNotFound, CodeTemplateNotFound)
	if !IsNotFound(err) {
		t.Errorf("a template that is gone: %v", err)
	}
	_, err = c.RemoveFace(ctx, "nobody", 1)
	refused(t, err, http.StatusNotFound, CodeSubjectNotFound)

	// the last template takes the subject with it
	got, err = c.RemoveFace(ctx, "anna", second.Template)
	if err != nil || got.Templates != 0 {
		t.Fatalf("the last template: %+v, %v", got, err)
	}
	if _, err := c.Subject(ctx, "anna"); !IsNotFound(err) {
		t.Errorf("anna after her last template went: %v", err)
	}

	// a whole subject
	got, err = c.RemoveSubject(ctx, "boris")
	if err != nil {
		t.Fatal(err)
	}
	if got.Subject != "boris" || got.Removed != 2 || got.Templates != 0 || got.Template != 0 || got.RequestID == "" {
		t.Errorf("boris: %+v", got)
	}
	if _, err := c.RemoveSubject(ctx, "boris"); !IsNotFound(err) {
		t.Errorf("removing boris again: %v", err)
	}
	res, err := c.Search(ctx, []byte("face:boris"), SearchOptions{})
	if err != nil || res.Best() != nil {
		t.Errorf("searching for boris afterwards: %+v, %v", res, err)
	}

	for _, remove := range []func() (*Removal, error){
		func() (*Removal, error) { return c.RemoveSubject(ctx, "") },
		func() (*Removal, error) { return c.RemoveFace(ctx, "", 1) },
	} {
		if _, err := remove(); err == nil || !strings.Contains(err.Error(), "subject id is empty") {
			t.Errorf("removing nobody: %v", err)
		}
	}
}

func TestInfo(t *testing.T) {
	// The server is told what to be, so that the test reads the client and
	// not the server's defaults.
	r := newRig(t, func(c *server.Config) {
		c.SearchThreshold, c.VerifyThreshold = 0.8, 0.6
		c.MaxInflight, c.Queue, c.QueueTimeout = 3, 5, 250*time.Millisecond
		c.MaxBody, c.MaxTemplates, c.AllowCandidates = 1<<20, 4, true
	})
	r.enrol("anna", "face:anna")
	r.enrol("anna", "face:anna#2")

	info, err := r.searcher().Info(ctx)
	if err != nil {
		t.Fatal(err)
	}
	engine := r.fake.Info()
	if info.Version != likeness.Version || info.Preset != engine.Preset || info.Detector != engine.Detector ||
		info.Embedder != engine.Embedder || info.Fingerprint != engine.Fingerprint || info.Dim != engine.Dim ||
		info.Runtime != engine.Runtime || info.Threads != engine.Threads || info.Kernel == "" {
		t.Errorf("what the server runs: %+v, with the engine %+v", info, engine)
	}
	if info.Thresholds != (Thresholds{Search: 0.8, Verify: 0.6}) {
		t.Errorf("thresholds %+v", info.Thresholds)
	}
	if g := info.Gallery; g.Subjects != 1 || g.Templates != 2 || g.Encrypted || g.Created.IsZero() || time.Since(g.Created) > time.Minute {
		t.Errorf("gallery %+v", g)
	}
	// how many subjects a search may name is not the test's to set
	if info.Limits.SearchLimit <= 0 {
		t.Errorf("a search names up to %d subjects", info.Limits.SearchLimit)
	}
	info.Limits.SearchLimit = 0
	want := Limits{MaxBody: 1 << 20, MaxInflight: 3, Queue: 5, QueueTimeout: 250 * time.Millisecond, MaxTemplates: 4, Candidates: true}
	if info.Limits != want {
		t.Errorf("limits %+v, want %+v", info.Limits, want)
	}
	if info.RequestID == "" {
		t.Error("the answer has no request id")
	}
}

func TestProbes(t *testing.T) {
	r := newRig(t, func(c *server.Config) { c.DrainDelay = 500 * time.Millisecond })
	// neither probe needs a key
	c := r.client("")
	if err := c.Ready(ctx); err != nil {
		t.Errorf("ready: %v", err)
	}
	if err := c.Alive(ctx); err != nil {
		t.Errorf("alive: %v", err)
	}

	// a server that has been told to stop stays alive for a while, and
	// says that it is going
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	running, stop := context.WithCancel(ctx)
	stopped := make(chan error, 1)
	go func() { stopped <- r.srv.Serve(running, ln) }()
	c = newClient(t, "http://"+ln.Addr().String())
	if err := c.Ready(ctx); err != nil {
		t.Fatalf("ready before the stop: %v", err)
	}
	stop()
	var going error
	eventually(t, "the server to say that it is stopping", func() bool {
		going = c.Ready(ctx)
		return going != nil
	})
	alive := c.Alive(ctx)
	e := refused(t, going, http.StatusServiceUnavailable, CodeShuttingDown)
	if e.RequestID == "" || e.Message == "" {
		t.Errorf("the refusal: %+v", e)
	}
	if alive != nil {
		t.Errorf("alive while draining: %v", alive)
	}

	// once it is gone the probes report the connection
	if err := <-stopped; err != nil {
		t.Fatal(err)
	}
	for _, probe := range []func(context.Context) error{c.Ready, c.Alive} {
		err := probe(ctx)
		var e *Error
		if err == nil || errors.As(err, &e) {
			t.Errorf("a probe of a server that is gone: %v", err)
		}
	}
}

func TestRequestIDs(t *testing.T) {
	r := newRig(t)
	c := r.searcher()

	// the caller's id is the one the server uses, for answers and refusals
	asked := WithRequestID(ctx, "login-42.a_B")
	info, err := c.Info(asked)
	if err != nil || info.RequestID != "login-42.a_B" {
		t.Errorf("info under the caller's id: %+v, %v", info, err)
	}
	_, err = c.Verify(asked, "nobody", []byte("face:anna"), VerifyOptions{})
	if e := refused(t, err, http.StatusNotFound, CodeSubjectNotFound); e.RequestID != "login-42.a_B" {
		t.Errorf("the refusal has request id %q", e.RequestID)
	}
	if !strings.Contains(err.Error(), "(request login-42.a_B)") {
		t.Errorf("the error does not name the request: %v", err)
	}

	// an id the server would not take is left out, and the server's own
	// comes back
	for _, id := range []string{"has space", "new\nline", strings.Repeat("x", 65), "ключ", ""} {
		info, err := c.Info(WithRequestID(ctx, id))
		if err != nil || info.RequestID == "" || info.RequestID == id {
			t.Errorf("asked for %q: %+v, %v", id, info, err)
		}
	}
	s := play(t, ok(infoAnswer), ok(infoAnswer), ok(infoAnswer))
	c = newClient(t, s.url)
	for _, id := range []string{"has space", strings.Repeat("x", 64), ""} {
		if _, err := c.Info(WithRequestID(ctx, id)); err != nil {
			t.Fatal(err)
		}
	}
	got := s.requests()
	if _, sent := got[0].header["X-Request-Id"]; sent {
		t.Errorf("an id with a space was sent: %q", got[0].header.Get("X-Request-Id"))
	}
	if got[1].header.Get("X-Request-Id") != strings.Repeat("x", 64) {
		t.Errorf("an id of 64 letters was not sent: %q", got[1].header.Get("X-Request-Id"))
	}
	if _, sent := got[2].header["X-Request-Id"]; sent {
		t.Error("an empty id was sent")
	}
}

func TestSubjectIDsInPaths(t *testing.T) {
	// what goes over the wire
	for id, want := range map[string]string{
		"anna":             "anna",
		"a%b?c#d":          "a%25b%3Fc%23d",
		"Алишер-1":         "%D0%90%D0%BB%D0%B8%D1%88%D0%B5%D1%80-1",
		"user@example.com": "user@example.com",
		"x&y=z+w;v":        "x&y=z+w%3Bv",
		"a/b":              "a%2Fb",
		"a b":              "a%20b",
		".":                "%2E",
		"..":               "%2E%2E",
		"...":              "...",
		"\xff":             "%FF",
	} {
		s := play(t, ok(`{}`), ok(enrolAnswer), ok(removedAnswer), ok(removedAnswer))
		c := newClient(t, s.url)
		c.Subject(ctx, id)
		c.Enrol(ctx, id, []byte("face:anna"), EnrolOptions{})
		c.RemoveSubject(ctx, id)
		c.RemoveFace(ctx, id, 18446744073709551615)
		got := s.requests()
		if len(got) != 4 {
			t.Fatalf("%q: %d requests", id, len(got))
		}
		base := "/v1/subjects/" + want
		for i, target := range []string{"GET " + base, "POST " + base + "/faces", "DELETE " + base, "DELETE " + base + "/faces/18446744073709551615"} {
			if sent := got[i].method + " " + got[i].target; sent != target {
				t.Errorf("%q: sent %s, want %s", id, sent, target)
			}
		}
	}

	// and what the server makes of it
	r := newRig(t)
	c := r.admin()
	for i, id := range []string{"a%b?c#d", "Алишер-1", "user@example.com", "30112199012345", "x&y=z+w;v", ".", "..", strings.Repeat("x", 128)} {
		picture := []byte(fmt.Sprintf("face:person-%d", i))
		enrolled, err := c.Enrol(ctx, id, picture, EnrolOptions{})
		if err != nil {
			t.Errorf("enrolling %q: %v", id, err)
			continue
		}
		if enrolled.Subject != id {
			t.Errorf("%q was enrolled as %q", id, enrolled.Subject)
		}
		if sub, err := c.Subject(ctx, id); err != nil || sub.ID != id || len(sub.Templates) != 1 {
			t.Errorf("showing %q: %+v, %v", id, sub, err)
		}
		if v, err := c.Verify(ctx, id, picture, VerifyOptions{}); err != nil || !v.Verified || v.Subject != id {
			t.Errorf("verifying %q: %+v, %v", id, v, err)
		}
		if res, err := c.Search(ctx, picture, SearchOptions{}); err != nil || res.Best() == nil || res.Best().Subject != id {
			t.Errorf("searching for %q: %+v, %v", id, res, err)
		}
		again, err := c.Enrol(ctx, id, picture, EnrolOptions{})
		if err != nil {
			t.Errorf("a second template for %q: %v", id, err)
			continue
		}
		if got, err := c.RemoveFace(ctx, id, again.Template); err != nil || got.Subject != id || got.Templates != 1 {
			t.Errorf("removing a template of %q: %+v, %v", id, got, err)
		}
		if got, err := c.RemoveSubject(ctx, id); err != nil || got.Subject != id || got.Removed != 1 {
			t.Errorf("removing %q: %+v, %v", id, got, err)
		}
	}
	if page, err := c.Subjects(ctx, "", 0); err != nil || len(page.Subjects) != 0 {
		t.Errorf("left behind: %+v, %v", page, err)
	}

	// ids the server does not take are refused by the server
	for _, id := range []string{"a b", "a/b", "a\tb", "\xff\xfe", strings.Repeat("x", 129)} {
		_, err := c.Enrol(ctx, id, []byte("face:anna"), EnrolOptions{})
		refused(t, err, http.StatusBadRequest, CodeInvalidSubject)
		_, err = c.Subject(ctx, id)
		refused(t, err, http.StatusBadRequest, CodeInvalidSubject)
		_, err = c.Verify(ctx, id, []byte("face:anna"), VerifyOptions{})
		refused(t, err, http.StatusBadRequest, CodeInvalidSubject)
	}
}

func TestBaseURLWithAPrefix(t *testing.T) {
	r := newRig(t)
	// a reverse proxy that mounts the API under a path of its own
	mux := http.NewServeMux()
	mux.Handle("/faces/api/", http.StripPrefix("/faces/api", r.srv.Handler()))
	proxy := httptest.NewServer(mux)
	t.Cleanup(proxy.Close)

	for _, base := range []string{proxy.URL + "/faces/api", proxy.URL + "/faces/api/", proxy.URL + "/faces/api//"} {
		c := newClient(t, base, WithKey(adminKey))
		if _, err := c.Info(ctx); err != nil {
			t.Errorf("%s: info: %v", base, err)
		}
		if err := c.Ready(ctx); err != nil {
			t.Errorf("%s: ready: %v", base, err)
		}
		enrolled, err := c.Enrol(ctx, "a%b?c#d", []byte("face:anna"), EnrolOptions{Replace: true})
		if err != nil || enrolled.Subject != "a%b?c#d" {
			t.Errorf("%s: enrolling: %+v, %v", base, enrolled, err)
		}
		if v, err := c.Verify(ctx, "a%b?c#d", []byte("face:anna"), VerifyOptions{}); err != nil || !v.Verified {
			t.Errorf("%s: verifying: %+v, %v", base, v, err)
		}
		if page, err := c.Subjects(ctx, "", 10); err != nil || ids(page.Subjects) != "a%b?c#d" {
			t.Errorf("%s: listing: %+v, %v", base, page, err)
		}
	}

	// without the prefix the proxy knows nothing of the request
	_, err := newClient(t, proxy.URL, WithKey(adminKey)).Info(ctx)
	var e *Error
	if !errors.As(err, &e) || e.Status != http.StatusNotFound || e.Code != "" || IsNotFound(err) {
		t.Errorf("the API at the wrong path: %v", err)
	}
	// and with a wrong one the server says so itself, which is not the
	// same as a subject that is not enrolled
	_, err = newClient(t, r.url+"/elsewhere", WithKey(adminKey)).Subject(ctx, "anna")
	refused(t, err, http.StatusNotFound, CodeNotFound)
	if IsNotFound(err) {
		t.Errorf("a wrong base URL passes for a missing subject: %v", err)
	}
}
