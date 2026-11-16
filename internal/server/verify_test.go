package server

import (
	"net/http"
	"testing"
)

func TestVerify(t *testing.T) {
	r := newRig(t)
	anna := r.enrol("anna", "face:anna")
	r.enrol("boris", "face:boris")

	a := r.picture("/v1/verify?subject=anna", searchKey, "face:anna#2").want(http.StatusOK)
	var got verifyView
	a.into(&got)
	if !got.Verified || got.Subject != "anna" || got.Template != anna.Template || got.Threshold != 0.5 || got.Reject != "" {
		t.Errorf("her own picture: %s", a.body)
	}
	if got.Score == nil || !near(*got.Score, 0.887, 0.002) || got.Face == nil || got.Face.Quality["score"] != 0.95 {
		t.Errorf("her own picture: %s", a.body)
	}

	// somebody else's picture: the answer is no, and how far off
	a = r.picture("/v1/verify?subject=anna", searchKey, "face:boris").want(http.StatusOK)
	got = verifyView{}
	a.into(&got)
	if got.Verified || got.Score == nil || !near(*got.Score, -0.105, 0.002) {
		t.Errorf("someone else's picture: %s", a.body)
	}

	// a stricter threshold than the server's
	got = verifyView{}
	r.picture("/v1/verify?subject=anna&threshold=0.9", searchKey, "face:anna#2").want(http.StatusOK).into(&got)
	if got.Verified || got.Threshold != 0.9 || got.Score == nil {
		t.Errorf("threshold 0.9: %+v", got)
	}
	r.picture("/v1/verify?subject=anna&threshold=0.3", searchKey, "face:anna#2").wantError(http.StatusBadRequest, "invalid_parameter")

	// the best template counts
	r.enrol("anna", "face:anna#2")
	got = verifyView{}
	r.picture("/v1/verify?subject=anna", searchKey, "face:anna#2").want(http.StatusOK).into(&got)
	if !got.Verified || *got.Score != 1 || got.Template == anna.Template {
		t.Errorf("two templates: %+v", got)
	}
}

func TestVerifyWithoutAUsableFace(t *testing.T) {
	r := newRig(t)
	r.enrol("anna", "face:anna")

	a := r.picture("/v1/verify?subject=anna", searchKey, "reject:too_dark").want(http.StatusOK)
	var got verifyView
	a.into(&got)
	if got.Verified || got.Reject != "too_dark" || got.Score != nil || got.Template != 0 {
		t.Errorf("a dark face: %s", a.body)
	}
	if got.Face == nil || got.Face.Reject != "too_dark" {
		t.Errorf("the face that was turned down is described: %s", a.body)
	}

	a = r.picture("/v1/verify?subject=anna", searchKey, "reject:no_face").want(http.StatusOK)
	got = verifyView{}
	a.into(&got)
	if got.Verified || got.Reject != "no_face" || got.Face != nil || got.Score != nil {
		t.Errorf("no face: %s", a.body)
	}
}

func TestVerifyNeedsAnEnrolledSubject(t *testing.T) {
	r := newRig(t)
	r.enrol("anna", "face:anna")
	before := r.fake.Calls()

	r.picture("/v1/verify", searchKey, "face:anna").wantError(http.StatusBadRequest, "missing_subject")
	r.picture("/v1/verify?subject=", searchKey, "face:anna").wantError(http.StatusBadRequest, "invalid_subject")
	r.picture("/v1/verify?subject=nobody", searchKey, "face:anna").wantError(http.StatusNotFound, "subject_not_found")
	r.picture("/v1/verify?subject=anna&limit=2", searchKey, "face:anna").wantError(http.StatusBadRequest, "invalid_parameter")
	if r.fake.Calls() != before {
		t.Error("a verification that could not succeed reached the engine")
	}

	// the subject may come in the form, next to the picture
	kind, body := form(t, map[string]string{"subject": "anna"}, "face:anna")
	var got verifyView
	r.send(http.MethodPost, "/v1/verify", searchKey, kind, body).want(http.StatusOK).into(&got)
	if !got.Verified || got.Subject != "anna" {
		t.Errorf("subject in the form: %+v", got)
	}
}
