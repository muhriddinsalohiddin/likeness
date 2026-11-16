package server

import (
	"net/http"
	"strings"
	"testing"

	"github.com/muhriddinsalohiddin/likeness"
	"github.com/muhriddinsalohiddin/likeness/internal/enginetest"
)

// The stand-in engine gives a face a liveness of 0.97 and a "spoof:" picture,
// a photograph held up to the camera, one of 0.08.
func TestAPictureOfAFaceIsTurnedDown(t *testing.T) {
	r := newRig(t)
	r.enrol("anna", "face:anna")

	// her own face is found, and the answer says how live it looked
	if f := r.search("/v1/search", "face:anna").Faces[0]; !f.Match || f.Liveness != 0.97 {
		t.Errorf("a live face: %+v", f)
	}

	// her photograph is not her
	a := r.picture("/v1/search", searchKey, "spoof:anna").want(http.StatusOK)
	var got searchView
	a.into(&got)
	if f := got.Faces[0]; f.Reject != "spoof" || f.Match || len(f.Matches) != 0 || f.Liveness != 0.08 {
		t.Errorf("searching with a photograph: %s", a.body)
	}
	if strings.Contains(string(a.body), `"subject"`) {
		t.Errorf("a photograph was told whose it is: %s", a.body)
	}

	var verified verifyView
	a = r.picture("/v1/verify?subject=anna", searchKey, "spoof:anna").want(http.StatusOK)
	a.into(&verified)
	if verified.Verified || verified.Reject != "spoof" || verified.Score != nil || verified.Face == nil || verified.Face.Liveness != 0.08 {
		t.Errorf("verifying with a photograph: %s", a.body)
	}

	// nor can anybody be enrolled from one
	e := r.picture("/v1/subjects/boris/faces", adminKey, "spoof:boris").wantError(http.StatusUnprocessableEntity, "face_rejected")
	if e.Error.Reject != "spoof" || !strings.Contains(e.Error.Message, "picture of a face") || r.gallery.Has("boris") {
		t.Errorf("enrolling from a photograph: %+v", e.Error)
	}

	if n := r.srv.metrics.rejects.get("verify", "spoof"); n != 1 {
		t.Errorf("%d spoofs counted for verify", n)
	}
	var info struct {
		Liveness *struct {
			Models    []string
			Threshold float64
		}
	}
	r.get("/v1/info", searchKey).want(http.StatusOK).into(&info)
	if info.Liveness == nil || info.Liveness.Threshold != 0.55 || len(info.Liveness.Models) != 1 {
		t.Errorf("info: %+v", info.Liveness)
	}
}

func TestLivenessThreshold(t *testing.T) {
	// a stricter server turns down even the live face of the stand-in
	strict := newRig(t, func(c *Config) { c.LivenessThreshold = 0.99 })
	strict.picture("/v1/subjects/anna/faces", adminKey, "face:anna").wantError(http.StatusUnprocessableEntity, "face_rejected")

	// and one that does not check takes a photograph for the person
	lax := newRig(t, func(c *Config) { c.LivenessThreshold = -1 })
	lax.enrol("anna", "face:anna")
	a := lax.picture("/v1/verify?subject=anna", searchKey, "spoof:anna").want(http.StatusOK)
	var verified verifyView
	a.into(&verified)
	if !verified.Verified || strings.Contains(string(a.body), `"liveness":{`) {
		t.Errorf("without the check: %s", a.body)
	}
	if _, timed := verified.Timing["liveness"]; timed {
		t.Errorf("a stage that did not happen is timed: %v", verified.Timing)
	}
	if a := lax.get("/v1/info", searchKey); strings.Contains(string(a.body), "liveness") {
		t.Errorf("info of a server that does not check: %s", a.body)
	}
}

// blind is an engine opened without liveness models.
type blind struct{ *enginetest.Fake }

func (b blind) Info() likeness.Info {
	info := b.Fake.Info()
	info.Liveness = nil
	return info
}

func TestLivenessNeedsTheModels(t *testing.T) {
	engine := blind{&enginetest.Fake{}}
	g := newGallery(t, engine)
	keys, err := NewKeys([]string{adminKey}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	// without the models there is nothing to check, and that is said only
	// when a check was asked for
	srv, err := New(Config{Engine: engine, Gallery: g, Keys: keys})
	if err != nil || srv.cfg.LivenessThreshold != 0 {
		t.Errorf("an engine without liveness models: threshold %v, %v", srv.cfg.LivenessThreshold, err)
	}
	if _, err := New(Config{Engine: engine, Gallery: g, Keys: keys, LivenessThreshold: 0.5}); err == nil || !strings.Contains(err.Error(), "liveness") {
		t.Errorf("a threshold with nothing to judge by: %v", err)
	}
	if _, err := New(Config{Engine: &enginetest.Fake{}, Gallery: g, Keys: keys, LivenessThreshold: 1.5}); err == nil {
		t.Error("a liveness threshold above 1 was accepted")
	}
}
