package server

import (
	"fmt"
	"net/http"
	"strings"
	"testing"
	"time"
)

type subjectView struct {
	Subject   string
	Created   time.Time
	Templates []struct {
		ID      uint64
		Created time.Time
		Quality map[string]float64
	}
}

type pageView struct {
	Subjects []struct {
		Subject   string
		Created   time.Time
		Templates int
	}
	Next string
}

func TestShowSubject(t *testing.T) {
	r := newRig(t)
	first := r.enrol("anna", "face:anna")
	second := r.enrol("anna", "face:anna#2")

	a := r.get("/v1/subjects/anna", adminKey).want(http.StatusOK)
	var got subjectView
	a.into(&got)
	if got.Subject != "anna" || len(got.Templates) != 2 || got.Created.IsZero() {
		t.Fatalf("anna: %s", a.body)
	}
	if got.Templates[0].ID != first.Template || got.Templates[1].ID != second.Template {
		t.Errorf("templates come oldest first: %s", a.body)
	}
	// what was measured at enrolment is kept; the vector is not handed out
	if q := got.Templates[0].Quality; q["sharpness"] != 310 || q["eye_distance"] != 84 {
		t.Errorf("quality %v", q)
	}
	if strings.Contains(string(a.body), "vector") || strings.Contains(string(a.body), "embedding") {
		t.Errorf("a subject is shown without its vectors: %s", a.body)
	}

	r.get("/v1/subjects/nobody", adminKey).wantError(http.StatusNotFound, "subject_not_found")
	r.get("/v1/subjects/anna?verbose=1", adminKey).wantError(http.StatusBadRequest, "invalid_parameter")
}

func TestListSubjects(t *testing.T) {
	r := newRig(t)

	a := r.get("/v1/subjects", adminKey).want(http.StatusOK)
	if !strings.Contains(string(a.body), `"subjects":[]`) || strings.Contains(string(a.body), "next") {
		t.Errorf("an empty listing: %s", a.body)
	}

	for _, name := range []string{"dilnoza", "anna", "carol", "boris", "eldor"} {
		r.enrol(name, "face:"+name)
	}
	r.enrol("anna", "face:anna#2")

	names := func(p pageView) string {
		ids := make([]string, len(p.Subjects))
		for i, s := range p.Subjects {
			ids[i] = s.Subject
		}
		return strings.Join(ids, " ")
	}
	var page pageView
	r.get("/v1/subjects", adminKey).want(http.StatusOK).into(&page)
	if names(page) != "anna boris carol dilnoza eldor" || page.Next != "" {
		t.Errorf("everyone, in id order: %+v", page)
	}
	if page.Subjects[0].Templates != 2 || page.Subjects[1].Templates != 1 || page.Subjects[0].Created.IsZero() {
		t.Errorf("summaries: %+v", page.Subjects)
	}

	// a page at a time
	var seen []string
	after := ""
	for pages := 0; ; pages++ {
		page = pageView{}
		r.get("/v1/subjects?limit=2&after="+after, adminKey).want(http.StatusOK).into(&page)
		seen = append(seen, names(page))
		if page.Next == "" {
			break
		}
		if pages > 5 {
			t.Fatal("the listing does not end")
		}
		after = page.Next
	}
	if got := strings.Join(seen, " | "); got != "anna boris | carol dilnoza | eldor" {
		t.Errorf("pages: %s", got)
	}

	for _, query := range []string{"limit=0", "limit=1001", "limit=all", "page=2"} {
		r.get("/v1/subjects?"+query, adminKey).wantError(http.StatusBadRequest, "invalid_parameter")
	}
}

func TestRemove(t *testing.T) {
	r := newRig(t)
	first := r.enrol("anna", "face:anna")
	second := r.enrol("anna", "face:anna#2")
	r.enrol("boris", "face:boris")

	type removed struct {
		Subject   string
		Template  uint64
		Removed   int
		Templates int
	}
	var got removed
	face := func(id uint64) string { return fmt.Sprintf("/v1/subjects/anna/faces/%d", id) }

	// one template
	r.remove(face(first.Template)).want(http.StatusOK).into(&got)
	if got != (removed{"anna", first.Template, 1, 1}) {
		t.Errorf("first template: %+v", got)
	}
	r.remove(face(first.Template)).wantError(http.StatusNotFound, "template_not_found")
	r.remove("/v1/subjects/nobody/faces/1").wantError(http.StatusNotFound, "subject_not_found")
	r.remove("/v1/subjects/anna/faces/first").wantError(http.StatusBadRequest, "invalid_parameter")
	r.remove("/v1/subjects/anna/faces/-1").wantError(http.StatusBadRequest, "invalid_parameter")
	if f := r.search("/v1/search", "face:anna").Faces[0]; !f.Match || f.Matches[0].Template != second.Template {
		t.Errorf("anna is still known by her other template: %+v", f)
	}

	// the last template takes the subject with it
	got = removed{}
	r.remove(face(second.Template)).want(http.StatusOK).into(&got)
	if got != (removed{"anna", second.Template, 1, 0}) {
		t.Errorf("last template: %+v", got)
	}
	r.get("/v1/subjects/anna", adminKey).wantError(http.StatusNotFound, "subject_not_found")
	if f := r.search("/v1/search", "face:anna").Faces[0]; f.Match {
		t.Errorf("anna is gone but still found: %+v", f)
	}
	r.picture("/v1/verify?subject=anna", searchKey, "face:anna").wantError(http.StatusNotFound, "subject_not_found")

	// a whole subject
	r.enrol("boris", "face:boris#2")
	got = removed{}
	r.remove("/v1/subjects/boris").want(http.StatusOK).into(&got)
	if got != (removed{"boris", 0, 2, 0}) {
		t.Errorf("boris: %+v", got)
	}
	r.remove("/v1/subjects/boris").wantError(http.StatusNotFound, "subject_not_found")
	r.remove("/v1/subjects/boris?force=true").wantError(http.StatusBadRequest, "invalid_parameter")
	if info := r.gallery.Info(); info.Subjects != 0 || info.Templates != 0 {
		t.Errorf("the gallery should be empty: %+v", info)
	}
}

func TestInfo(t *testing.T) {
	r := newRig(t, func(c *Config) { c.MaxInflight = 3; c.AllowCandidates = true })
	r.enrol("anna", "face:anna")
	r.enrol("anna", "face:anna#2")

	a := r.get("/v1/info", searchKey).want(http.StatusOK)
	var got struct {
		Version, Preset, Detector, Embedder, Fingerprint, Runtime, Kernel string
		Dim, Threads                                                      int
		Thresholds                                                        struct{ Search, Verify float64 }
		Gallery                                                           struct {
			Subjects, Templates int
			Encrypted           bool
			Created             time.Time
		}
		Limits struct {
			MaxBody        int64   `json:"max_body"`
			MaxInflight    int     `json:"max_inflight"`
			Queue          int     `json:"queue"`
			QueueTimeoutMS float64 `json:"queue_timeout_ms"`
			MaxTemplates   int     `json:"max_templates"`
			SearchLimit    int     `json:"search_limit"`
			Candidates     bool    `json:"candidates"`
		}
	}
	a.into(&got)
	if got.Version == "" || got.Preset != "fake" || got.Embedder != "fake-embedder" || got.Dim != 64 || got.Kernel == "" || got.Fingerprint == "" {
		t.Errorf("what runs: %s", a.body)
	}
	if got.Thresholds.Search != 0.7 || got.Thresholds.Verify != 0.5 {
		t.Errorf("thresholds: %s", a.body)
	}
	if got.Gallery.Subjects != 1 || got.Gallery.Templates != 2 || got.Gallery.Encrypted || got.Gallery.Created.IsZero() {
		t.Errorf("gallery: %s", a.body)
	}
	l := got.Limits
	if l.MaxBody != 10<<20 || l.MaxInflight != 3 || l.Queue != 24 || l.QueueTimeoutMS != 50 || l.MaxTemplates != 10 || l.SearchLimit != 50 || !l.Candidates {
		t.Errorf("limits: %s", a.body)
	}
	r.get("/v1/info?pretty=1", searchKey).wantError(http.StatusBadRequest, "invalid_parameter")
}
