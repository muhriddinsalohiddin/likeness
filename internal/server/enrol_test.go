package server

import (
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"testing"
	"time"
)

func TestEnrol(t *testing.T) {
	r := newRig(t)

	a := r.picture("/v1/subjects/anna/faces", adminKey, "face:anna").want(http.StatusCreated)
	var first enrolView
	a.into(&first)
	if first.Subject != "anna" || first.Template == 0 || first.Templates != 1 {
		t.Errorf("first template: %s", a.body)
	}
	if age := time.Since(first.Created); age < 0 || age > time.Minute {
		t.Errorf("created %v", first.Created)
	}
	if first.Face.Quality["sharpness"] != 310 || first.Face.Box["right"] != 300 {
		t.Errorf("the face that was enrolled: %+v", first.Face)
	}
	if got := a.header.Get("Location"); got != "/v1/subjects/anna" {
		t.Errorf("Location %q", got)
	}

	// it can be found at once
	if f := r.search("/v1/search", "face:anna").Faces[0]; !f.Match || f.Matches[0].Subject != "anna" || f.Matches[0].Score != 1 {
		t.Errorf("searching right after enrolling: %+v", f)
	}

	// another picture of her adds a template
	second := r.enrol("anna", "face:anna#2")
	if second.Templates != 2 || second.Template == first.Template {
		t.Errorf("second template: %+v", second)
	}
	// and a replacement leaves only itself
	var third enrolView
	r.picture("/v1/subjects/anna/faces?replace=true", adminKey, "face:anna#3").want(http.StatusCreated).into(&third)
	if third.Templates != 1 {
		t.Errorf("after replacing: %+v", third)
	}
	if f := r.search("/v1/search", "face:anna").Faces[0]; !f.Match || f.Matches[0].Template != third.Template {
		t.Errorf("the replacement is what a search finds: %+v", f)
	}
	if info := r.gallery.Info(); info.Subjects != 1 || info.Templates != 1 {
		t.Errorf("gallery holds %d subjects, %d templates", info.Subjects, info.Templates)
	}
}

func TestEnrolTurnsDownPoorPictures(t *testing.T) {
	r := newRig(t)
	for _, reason := range []string{
		"no_face", "multiple_faces", "low_det_score", "out_of_frame", "face_too_small",
		"bad_pose", "too_dark", "too_bright", "blurry",
	} {
		e := r.picture("/v1/subjects/anna/faces", adminKey, "reject:"+reason).wantError(http.StatusUnprocessableEntity, "face_rejected")
		if e.Error.Reject != reason {
			t.Errorf("%s: reject %q", reason, e.Error.Reject)
		}
		// the measurements come along, so the caller can say what to fix
		if measured := e.Error.Quality != nil; measured != (reason != "no_face") {
			t.Errorf("%s: quality %v", reason, e.Error.Quality)
		}
		if strings.HasSuffix(e.Error.Message, reason) {
			t.Errorf("%s: the message should say it in words: %q", reason, e.Error.Message)
		}
	}
	// a reference picture shows one person
	e := r.picture("/v1/subjects/anna/faces", adminKey, "faces:anna,boris").wantError(http.StatusUnprocessableEntity, "face_rejected")
	if e.Error.Reject != "multiple_faces" {
		t.Errorf("two people: %+v", e.Error)
	}
	if info := r.gallery.Info(); info.Subjects != 0 || info.Templates != 0 {
		t.Errorf("a rejected picture was enrolled: %+v", info)
	}
}

func TestEnrolDuplicate(t *testing.T) {
	r := newRig(t)
	r.enrol("anna", "face:anna")

	e := r.picture("/v1/subjects/annette/faces", adminKey, "face:anna#2").wantError(http.StatusConflict, "duplicate_face")
	if e.Error.Subject != "anna" || e.Error.Score == nil || !near(*e.Error.Score, 0.887, 0.002) {
		t.Errorf("the conflict names the subject and the similarity: %+v", e.Error)
	}
	if r.gallery.Has("annette") {
		t.Error("the duplicate was enrolled all the same")
	}
	// more pictures of an enrolled person are not duplicates of herself
	r.enrol("anna", "face:anna#4")
	// twins, or one person with two accounts: the caller decides
	r.picture("/v1/subjects/annette/faces?allow_duplicate=true", adminKey, "face:anna#2").want(http.StatusCreated)
	// and from then on has to say so for every picture of either
	e = r.picture("/v1/subjects/anna/faces", adminKey, "face:anna#3").wantError(http.StatusConflict, "duplicate_face")
	if e.Error.Subject != "annette" {
		t.Errorf("the double is named: %+v", e.Error)
	}
	r.picture("/v1/subjects/anna/faces?allow_duplicate=true", adminKey, "face:anna#3").want(http.StatusCreated)
}

func TestEnrolMismatch(t *testing.T) {
	r := newRig(t)
	r.enrol("anna", "face:anna")

	e := r.picture("/v1/subjects/anna/faces", adminKey, "face:boris").wantError(http.StatusConflict, "subject_mismatch")
	if e.Error.Score == nil || !near(*e.Error.Score, -0.105, 0.002) || e.Error.Subject != "" {
		t.Errorf("the mismatch says how unlike: %+v", e.Error)
	}
	if n := r.gallery.Templates("anna"); n != 1 {
		t.Errorf("anna has %d templates", n)
	}
	var got enrolView
	r.picture("/v1/subjects/anna/faces?allow_mismatch=true", adminKey, "face:boris").want(http.StatusCreated).into(&got)
	if got.Templates != 2 {
		t.Errorf("allowed mismatch: %+v", got)
	}
	// replacing is saying that the old pictures no longer count
	r.picture("/v1/subjects/anna/faces?replace=true", adminKey, "face:carol").want(http.StatusCreated).into(&got)
	if got.Templates != 1 {
		t.Errorf("replacement with another face: %+v", got)
	}
}

func TestEnrolLimit(t *testing.T) {
	r := newRig(t, func(c *Config) { c.MaxTemplates = 2 })
	r.enrol("anna", "face:anna")
	r.enrol("anna", "face:anna#2")

	e := r.picture("/v1/subjects/anna/faces", adminKey, "face:anna#3").wantError(http.StatusConflict, "too_many_templates")
	if !strings.Contains(e.Error.Message, "2 templates") {
		t.Errorf("message %q", e.Error.Message)
	}
	// a full subject can still be started afresh
	var got enrolView
	r.picture("/v1/subjects/anna/faces?replace=true", adminKey, "face:anna#3").want(http.StatusCreated).into(&got)
	if got.Templates != 1 {
		t.Errorf("after replacing: %+v", got)
	}

	// no bound at all
	free := newRig(t, func(c *Config) { c.MaxTemplates = -1 })
	for i := 0; i < 14; i++ {
		free.enrol("anna", fmt.Sprintf("face:anna#%d", i))
	}
	if n := free.gallery.Templates("anna"); n != 14 {
		t.Errorf("unbounded subject has %d templates", n)
	}
}

func TestEnrolOptions(t *testing.T) {
	r := newRig(t)
	for _, query := range []string{"replace=yes", "allow_duplicate=2", "allow_mismatch=", "replcae=true", "limit=3"} {
		r.picture("/v1/subjects/anna/faces?"+query, adminKey, "face:anna").wantError(http.StatusBadRequest, "invalid_parameter")
	}
	if r.fake.Calls() != 0 || r.gallery.Has("anna") {
		t.Error("a request with bad options went through")
	}
}

func TestSubjectIDs(t *testing.T) {
	r := newRig(t)
	for _, id := range []string{"a b", "a\tb", "a/b", strings.Repeat("x", 129), "\xff\xfe"} {
		target := "/v1/subjects/" + url.PathEscape(id) + "/faces"
		r.picture(target, adminKey, "face:anna").wantError(http.StatusBadRequest, "invalid_subject")
		r.picture("/v1/verify?subject="+url.QueryEscape(id), searchKey, "face:anna").wantError(http.StatusBadRequest, "invalid_subject")
	}
	if r.fake.Calls() != 0 {
		t.Error("a request for an impossible subject reached the engine")
	}

	// anything else goes, including what needs escaping in a URL
	for i, id := range []string{"Алишер-1", "user@example.com", "30112199012345", "a%b?c#d", strings.Repeat("x", 128)} {
		target := "/v1/subjects/" + url.PathEscape(id)
		a := r.picture(target+"/faces", adminKey, fmt.Sprintf("face:person-%d", i)).want(http.StatusCreated)
		if got := a.header.Get("Location"); got != target {
			t.Errorf("%q: Location %q, want %q", id, got, target)
		}
		var got struct{ Subject string }
		r.get(target, adminKey).want(http.StatusOK).into(&got)
		if got.Subject != id {
			t.Errorf("subject %q came back as %q", id, got.Subject)
		}
		var verified verifyView
		r.picture("/v1/verify?subject="+url.QueryEscape(id), searchKey, fmt.Sprintf("face:person-%d", i)).want(http.StatusOK).into(&verified)
		if !verified.Verified || verified.Subject != id {
			t.Errorf("verifying %q: %+v", id, verified)
		}
	}
}
