package server

import (
	"net/http"
	"os"
	"testing"

	"github.com/muhriddinsalohiddin/likeness"
	"github.com/muhriddinsalohiddin/likeness/internal/fixtures"
	"github.com/muhriddinsalohiddin/likeness/internal/home"
	"github.com/muhriddinsalohiddin/likeness/internal/testenv"
)

// realRig serves the default models over an empty gallery. It skips the
// test on a machine where they are not installed.
func realRig(t *testing.T) *rig {
	t.Helper()
	if _, err := os.Stat(home.LibraryPath()); err != nil {
		t.Skipf("onnxruntime is not installed: %v", err)
	}
	testenv.Model(t, "yunet_2023mar")
	testenv.Model(t, "sface_2021dec")
	engine, err := likeness.Open(likeness.Config{Threads: 4})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(engine.Close)

	g := newGallery(t, engine)
	keys, err := NewKeys([]string{adminKey}, []string{searchKey}, nil)
	if err != nil {
		t.Fatal(err)
	}
	logs := &logBuffer{}
	srv, err := New(Config{Engine: engine, Gallery: g, Keys: keys, Log: logs.logger(0)})
	if err != nil {
		t.Fatal(err)
	}
	return &rig{t: t, gallery: g, srv: srv, logs: logs}
}

func closeup(t *testing.T, name string) string {
	t.Helper()
	b, err := fixtures.Closeup(name)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func wide(t *testing.T, name string) string {
	t.Helper()
	b, err := fixtures.JPEG(name)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

// The whole path with real models and real pictures: two people are
// enrolled from close-ups and looked for in other shots.
func TestWithRealModels(t *testing.T) {
	r := realRig(t)
	threshold := float64(r.srv.cfg.SearchThreshold)

	for _, name := range fixtures.Closeups() {
		got := r.enrol(name, closeup(t, name))
		if got.Templates != 1 || got.Face.Quality["eye_distance"] < 40 {
			t.Errorf("enrolling %s: %+v", name, got)
		}
		if got.Timing["detect"] <= 0 || got.Timing["embed"] <= 0 || got.Timing["decode"] <= 0 {
			t.Errorf("enrolling %s: timing %v", name, got.Timing)
		}
	}

	// each close-up finds its own subject, and by a wide margin
	for _, name := range fixtures.Closeups() {
		got := r.search("/v1/search", closeup(t, name))
		if len(got.Faces) != 1 || !got.Faces[0].Match {
			t.Fatalf("searching for %s: %+v", name, got)
		}
		f := got.Faces[0]
		if f.Matches[0].Subject != name || f.Matches[0].Score < 0.99 || f.Margin == nil || *f.Margin < 0.5 {
			t.Errorf("searching for %s: %+v (margin %v)", name, f.Matches, f.Margin)
		}
		t.Logf("%s: score %.3f, margin %.3f, %v", name, f.Matches[0].Score, *f.Margin, got.Timing)
	}

	// the wide shot the close-up was cut from: a smaller face, the same person
	got := r.search("/v1/search", wide(t, "hathaway"))
	if f := got.Faces[0]; !f.Match || f.Matches[0].Subject != "hathaway" || f.Matches[0].Score < threshold {
		t.Errorf("the wide shot of hathaway: %+v", f)
	} else {
		t.Logf("wide shot: score %.3f with %.0f px between the eyes", f.Matches[0].Score, f.Quality["eye_distance"])
	}
	// but too small a face to serve as a reference
	e := r.picture("/v1/subjects/someone/faces", adminKey, wide(t, "hathaway")).wantError(http.StatusUnprocessableEntity, "face_rejected")
	if e.Error.Reject != "face_too_small" || e.Error.Quality["eye_distance"] >= 40 {
		t.Errorf("enrolling from the wide shot: %+v", e.Error)
	}

	// one is not the other
	var verified verifyView
	r.picture("/v1/verify?subject=cassada", searchKey, closeup(t, "hathaway")).want(http.StatusOK).into(&verified)
	if verified.Verified || verified.Score == nil || *verified.Score >= float64(r.srv.cfg.VerifyThreshold) {
		t.Errorf("hathaway passed for cassada: %+v", verified)
	}
	verified = verifyView{}
	r.picture("/v1/verify?subject=cassada", searchKey, closeup(t, "cassada")).want(http.StatusOK).into(&verified)
	if !verified.Verified {
		t.Errorf("cassada did not pass for himself: %+v", verified)
	}

	// a group of strangers: several faces, nobody known
	got = r.search("/v1/search?faces=all", wide(t, "crew11"))
	if got.Found < 4 || len(got.Faces) < 4 {
		t.Errorf("the crew picture: %d faces found, %d analysed", got.Found, len(got.Faces))
	}
	for i, f := range got.Faces {
		if f.Match {
			t.Errorf("crew member %d was taken for %s (%.3f)", i, f.Matches[0].Subject, f.Matches[0].Score)
		}
	}

	// the same face under another name is caught
	e = r.picture("/v1/subjects/impostor/faces", adminKey, closeup(t, "hathaway")).wantError(http.StatusConflict, "duplicate_face")
	if e.Error.Subject != "hathaway" {
		t.Errorf("duplicate: %+v", e.Error)
	}

	// and things that are not pictures are told apart from bad pictures
	r.picture("/v1/search", searchKey, "these are not the bytes of a picture").wantError(http.StatusUnsupportedMediaType, "unsupported_image")
	whole := closeup(t, "hathaway")
	r.picture("/v1/search", searchKey, whole[:len(whole)/3]).wantError(http.StatusUnprocessableEntity, "corrupt_image")
}
