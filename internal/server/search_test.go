package server

import (
	"net/http"
	"regexp"
	"strings"
	"testing"
)

func TestSearchFindsTheEnrolledPerson(t *testing.T) {
	r := newRig(t)
	anna := r.enrol("anna", "face:anna")
	r.enrol("boris", "face:boris")

	a := r.picture("/v1/search", searchKey, "face:anna#2").want(http.StatusOK)
	var got searchView
	a.into(&got)
	if got.Found != 1 || len(got.Faces) != 1 || got.Reject != "" {
		t.Fatalf("one face expected: %s", a.body)
	}
	if got.Width != 1280 || got.Height != 720 || got.Threshold != 0.7 {
		t.Errorf("picture size %dx%d, threshold %v", got.Width, got.Height, got.Threshold)
	}
	f := got.Faces[0]
	if !f.Match || len(f.Matches) != 1 {
		t.Fatalf("anna expected: %s", a.body)
	}
	if m := f.Matches[0]; m.Subject != "anna" || !near(m.Score, 0.887, 0.002) || m.Template != anna.Template {
		t.Errorf("match %+v", m)
	}
	// boris is the runner-up, at about -0.18
	if f.Margin == nil || !near(*f.Margin, 1.067, 0.01) {
		t.Errorf("margin %v", f.Margin)
	}
	if f.Reject != "" || f.Quality["score"] != 0.95 || f.Quality["eye_distance"] != 84 || f.Box["left"] != 100 || f.Box["bottom"] != 330 {
		t.Errorf("face %+v", f)
	}
	if !strings.Contains(a.header.Get("Cache-Control"), "no-store") {
		t.Errorf("answers about faces are not for caches: %v", a.header)
	}
}

func TestSearchKnowsNobody(t *testing.T) {
	r := newRig(t)

	// nobody enrolled at all
	a := r.picture("/v1/search", searchKey, "face:anna").want(http.StatusOK)
	var got searchView
	a.into(&got)
	if len(got.Faces) != 1 || got.Faces[0].Match || got.Faces[0].Margin != nil {
		t.Errorf("empty gallery: %s", a.body)
	}
	if !strings.Contains(string(a.body), `"matches":[]`) {
		t.Errorf("no matches is an empty list, not null: %s", a.body)
	}

	// somebody enrolled, but not this person; the stranger's nearest
	// neighbours stay unnamed
	r.enrol("anna", "face:anna")
	r.enrol("boris", "face:boris")
	a = r.picture("/v1/search?limit=10", searchKey, "face:carol").want(http.StatusOK)
	a.into(&got)
	if f := got.Faces[0]; f.Match || len(f.Matches) != 0 || f.Margin != nil {
		t.Errorf("a stranger: %s", a.body)
	}
	if strings.Contains(string(a.body), "anna") || strings.Contains(string(a.body), "boris") {
		t.Errorf("subjects below the threshold leaked: %s", a.body)
	}
}

func TestSearchLimitAndThreshold(t *testing.T) {
	r := newRig(t)
	r.enrol("anna", "face:anna")
	// someone who looks a lot like her
	r.picture("/v1/subjects/twin/faces?allow_duplicate=true", adminKey, "face:anna#3").want(http.StatusCreated)
	r.enrol("boris", "face:boris")

	// by default only the best subject is named, but the margin tells
	// that another is close behind
	got := r.search("/v1/search", "face:anna#2")
	if f := got.Faces[0]; len(f.Matches) != 1 || f.Matches[0].Subject != "anna" || f.Margin == nil || !near(*f.Margin, 0.122, 0.003) {
		t.Errorf("default limit: %+v", f)
	}
	// a larger limit names everyone above the threshold, best first
	got = r.search("/v1/search?limit=5", "face:anna#2")
	f := got.Faces[0]
	if len(f.Matches) != 2 || f.Matches[0].Subject != "anna" || f.Matches[1].Subject != "twin" || !near(f.Matches[1].Score, 0.765, 0.002) {
		t.Errorf("limit 5: %+v", f)
	}
	// a stricter threshold leaves the twin out
	got = r.search("/v1/search?limit=5&threshold=0.8", "face:anna#2")
	if f := got.Faces[0]; got.Threshold != 0.8 || !f.Match || len(f.Matches) != 1 {
		t.Errorf("threshold 0.8: %+v", f)
	}
	// and a stricter one still, everyone
	got = r.search("/v1/search?threshold=0.95", "face:anna#2")
	if f := got.Faces[0]; f.Match || len(f.Matches) != 0 || f.Margin != nil {
		t.Errorf("threshold 0.95: %+v", f)
	}
}

func TestSearchAllFaces(t *testing.T) {
	r := newRig(t)
	r.enrol("anna", "face:anna")
	r.enrol("boris", "face:boris")

	// the largest face by default
	got := r.search("/v1/search", "faces:anna,boris,carol")
	if got.Found != 3 || len(got.Faces) != 1 || got.Faces[0].Matches[0].Subject != "anna" {
		t.Errorf("largest: %+v", got)
	}
	got = r.search("/v1/search?faces=all", "faces:anna,boris,carol")
	if got.Found != 3 || len(got.Faces) != 3 {
		t.Fatalf("all: %+v", got)
	}
	for i, want := range []string{"anna", "boris", ""} {
		f := got.Faces[i]
		if want == "" {
			if f.Match || len(f.Matches) != 0 {
				t.Errorf("face %d is a stranger: %+v", i, f)
			}
			continue
		}
		if !f.Match || f.Matches[0].Subject != want {
			t.Errorf("face %d: %+v, want %s", i, f, want)
		}
	}
}

func TestSearchRejectedPictures(t *testing.T) {
	r := newRig(t)
	r.enrol("anna", "face:anna")

	a := r.picture("/v1/search", searchKey, "reject:no_face").want(http.StatusOK)
	var got searchView
	a.into(&got)
	if got.Reject != "no_face" || got.Found != 0 || !strings.Contains(string(a.body), `"faces":[]`) {
		t.Errorf("no face: %s", a.body)
	}

	a = r.picture("/v1/search", searchKey, "reject:blurry").want(http.StatusOK)
	got = searchView{}
	a.into(&got)
	if got.Reject != "" || len(got.Faces) != 1 {
		t.Fatalf("a blurred face: %s", a.body)
	}
	if f := got.Faces[0]; f.Reject != "blurry" || f.Match || len(f.Matches) != 0 || f.Quality["sharpness"] != 3 {
		t.Errorf("a blurred face: %+v", f)
	}
}

func TestSearchOptions(t *testing.T) {
	r := newRig(t)
	r.enrol("anna", "face:anna")
	before := r.fake.Calls()

	for _, c := range []struct{ query, code, says string }{
		{"limit=0", "invalid_parameter", "limit must be a number from 1 to 50"},
		{"limit=51", "invalid_parameter", "limit"},
		{"limit=many", "invalid_parameter", "limit"},
		{"limit=1&limit=2", "invalid_parameter", "more than once"},
		{"threshold=0.5", "invalid_parameter", "cannot be lower than the server's, 0.7"},
		{"threshold=1.5", "invalid_parameter", "above 0 and at most 1"},
		{"threshold=high", "invalid_parameter", "threshold"},
		{"threshold=NaN", "invalid_parameter", "threshold"},
		{"faces=some", "invalid_parameter", "largest, all"},
		{"candidates=maybe", "invalid_parameter", "true or false"},
		{"limt=3", "invalid_parameter", `"limt" is not an option`},
		{"subject=anna", "invalid_parameter", `"subject" is not an option`},
		{"limit=%zz", "bad_request", "query string"},
	} {
		e := r.picture("/v1/search?"+c.query, searchKey, "face:anna").wantError(http.StatusBadRequest, c.code)
		if !strings.Contains(e.Error.Message, c.says) {
			t.Errorf("%s: message %q lacks %q", c.query, e.Error.Message, c.says)
		}
	}
	r.picture("/v1/search?candidates=true", searchKey, "face:anna").wantError(http.StatusForbidden, "candidates_disabled")
	if r.fake.Calls() != before {
		t.Errorf("a request with bad options reached the engine")
	}
	// spelled the other way, and switched off explicitly, both are fine
	r.search("/v1/search?candidates=false&faces=largest&limit=50&threshold=1", "face:anna")
}

func TestSearchCandidates(t *testing.T) {
	r := newRig(t, func(c *Config) { c.AllowCandidates = true })
	r.enrol("anna", "face:anna")
	r.enrol("boris", "face:boris")

	// without asking, a server that allows candidates still gives none
	got := r.search("/v1/search?limit=5", "face:carol")
	if f := got.Faces[0]; f.Match || len(f.Matches) != 0 {
		t.Errorf("unasked: %+v", f)
	}
	got = r.search("/v1/search?limit=5&candidates=true", "face:carol")
	f := got.Faces[0]
	if f.Match || len(f.Matches) != 2 || f.Matches[0].Subject != "boris" || f.Matches[1].Subject != "anna" {
		t.Fatalf("candidates: %+v", f)
	}
	if f.Matches[0].Score < f.Matches[1].Score || f.Margin == nil || !near(*f.Margin, 0.013, 0.003) {
		t.Errorf("candidates come best first, with a margin: %+v", f)
	}
	// such a server also lets a request lower the threshold
	got = r.search("/v1/search?threshold=0.05", "face:carol")
	if f := got.Faces[0]; !f.Match || f.Matches[0].Subject != "boris" || got.Threshold != 0.05 {
		t.Errorf("lowered threshold: %+v", f)
	}
}

func TestTimings(t *testing.T) {
	r := newRig(t)
	r.enrol("anna", "face:anna")

	a := r.picture("/v1/search", searchKey, "face:anna").want(http.StatusOK)
	var got searchView
	a.into(&got)
	// the stand-in engine reports fixed times
	for stage, want := range map[string]float64{"decode": 2, "detect": 3, "align": 1, "embed": 4} {
		if got.Timing[stage] != want {
			t.Errorf("%s took %v ms, want %v", stage, got.Timing[stage], want)
		}
	}
	for _, stage := range []string{"read", "queue", "search", "total"} {
		if _, ok := got.Timing[stage]; !ok {
			t.Errorf("no time for %s: %v", stage, got.Timing)
		}
	}
	order := regexp.MustCompile(`"timing_ms":\{"read":[\d.]+,"queue":[\d.]+,"decode":2,"detect":3,"align":1,"embed":4,"search":[\d.]+,"total":[\d.]+\}`)
	if !order.Match(a.body) {
		t.Errorf("stages are listed in the order they happen: %s", a.body)
	}
	header := regexp.MustCompile(`^read;dur=[\d.]+, queue;dur=[\d.]+, decode;dur=2, detect;dur=3, align;dur=1, embed;dur=4, search;dur=[\d.]+, total;dur=[\d.]+$`)
	if got := a.header.Get("Server-Timing"); !header.MatchString(got) {
		t.Errorf("Server-Timing %q", got)
	}

	// an enrolment has its own last stage, a verification too
	enrolled := r.enrol("boris", "face:boris")
	if _, ok := enrolled.Timing["store"]; !ok {
		t.Errorf("enrolment: %v", enrolled.Timing)
	}
	var verified verifyView
	r.picture("/v1/verify?subject=anna", searchKey, "face:anna").want(http.StatusOK).into(&verified)
	if _, ok := verified.Timing["compare"]; !ok {
		t.Errorf("verification: %v", verified.Timing)
	}
	// answers without a picture have nothing to time
	if got := r.get("/v1/info", searchKey).header.Get("Server-Timing"); got != "" {
		t.Errorf("Server-Timing on info: %q", got)
	}
}
