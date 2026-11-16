package server

import (
	"bytes"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestPictureAsBodyOrForm(t *testing.T) {
	r := newRig(t)
	r.enrol("anna", "face:anna")

	// as the body, whatever it claims to be: the bytes decide
	for _, kind := range []string{"image/jpeg", "image/png", "image/webp", "application/octet-stream", "application/x-www-form-urlencoded", ""} {
		var got searchView
		r.send(http.MethodPost, "/v1/search", searchKey, kind, []byte("face:anna")).want(http.StatusOK).into(&got)
		if !got.Faces[0].Match {
			t.Errorf("body sent as %q: %+v", kind, got)
		}
	}

	// as a form, with the options next to the picture
	kind, body := form(t, map[string]string{"limit": "5", "faces": "all"}, "faces:anna,carol")
	var got searchView
	r.send(http.MethodPost, "/v1/search", searchKey, kind, body).want(http.StatusOK).into(&got)
	if len(got.Faces) != 2 || !got.Faces[0].Match || got.Faces[1].Match {
		t.Errorf("form: %+v", got)
	}
	// or some here and some there
	kind, body = form(t, map[string]string{"faces": "all"}, "faces:anna,carol")
	got = searchView{}
	r.send(http.MethodPost, "/v1/search?limit=5", searchKey, kind, body).want(http.StatusOK).into(&got)
	if len(got.Faces) != 2 {
		t.Errorf("options in both places: %+v", got)
	}
	// an enrolment by form
	kind, body = form(t, map[string]string{"replace": "true"}, "face:anna#2")
	var enrolled enrolView
	r.send(http.MethodPost, "/v1/subjects/anna/faces", adminKey, kind, body).want(http.StatusCreated).into(&enrolled)
	if enrolled.Templates != 1 {
		t.Errorf("replace sent in the form: %+v", enrolled)
	}
}

func TestBadForms(t *testing.T) {
	r := newRig(t)
	post := func(kind string, body []byte) answer {
		return r.send(http.MethodPost, "/v1/search", searchKey, kind, body)
	}
	// build writes a form part by part
	build := func(fill func(w *multipart.Writer)) (string, []byte) {
		var buf bytes.Buffer
		w := multipart.NewWriter(&buf)
		fill(w)
		w.Close()
		return w.FormDataContentType(), buf.Bytes()
	}
	file := func(w *multipart.Writer, field, content string) {
		part, _ := w.CreateFormFile(field, field+".jpg")
		io.WriteString(part, content)
	}

	kind, body := form(t, map[string]string{"limit": "2"}, "face:anna")
	r.send(http.MethodPost, "/v1/search?limit=3", searchKey, kind, body).wantError(http.StatusBadRequest, "invalid_parameter")

	kind, body = form(t, map[string]string{"limit": "2"}, "")
	post(kind, body).wantError(http.StatusBadRequest, "missing_image")

	kind, body = build(func(w *multipart.Writer) { file(w, "image", "") })
	post(kind, body).wantError(http.StatusBadRequest, "missing_image")

	post("image/jpeg", nil).wantError(http.StatusBadRequest, "missing_image")

	kind, body = build(func(w *multipart.Writer) { file(w, "image", "face:anna"); file(w, "image", "face:boris") })
	post(kind, body).wantError(http.StatusBadRequest, "bad_request")

	kind, body = build(func(w *multipart.Writer) { file(w, "image", "face:anna"); file(w, "photo", "face:boris") })
	post(kind, body).wantError(http.StatusBadRequest, "invalid_parameter")

	kind, body = build(func(w *multipart.Writer) {
		file(w, "image", "face:anna")
		w.WriteField("limit", "1")
		w.WriteField("limit", "2")
	})
	post(kind, body).wantError(http.StatusBadRequest, "invalid_parameter")

	kind, body = form(t, map[string]string{"faces": strings.Repeat("a", maxField+1)}, "face:anna")
	post(kind, body).wantError(http.StatusBadRequest, "invalid_parameter")

	kind, body = build(func(w *multipart.Writer) {
		for i := 0; i < maxParts; i++ {
			w.WriteField(fmt.Sprintf("field%d", i), "x")
		}
		file(w, "image", "face:anna")
	})
	post(kind, body).wantError(http.StatusBadRequest, "bad_request")

	// a form that is not one
	post("multipart/form-data", []byte("face:anna")).wantError(http.StatusBadRequest, "bad_request")
	kind, body = form(t, nil, "face:anna")
	post(kind, body[:len(body)-20]).wantError(http.StatusBadRequest, "bad_request")
	post(kind, []byte("this is no form")).wantError(http.StatusBadRequest, "missing_image")

	if r.fake.Calls() != 0 {
		t.Errorf("a broken upload reached the engine")
	}
}

func TestBodyLimit(t *testing.T) {
	r := newRig(t, func(c *Config) { c.MaxBody = 1024 })
	padded := func(n int) []byte { // a picture of exactly n bytes
		return []byte("face:anna" + strings.Repeat(" ", n-len("face:anna")))
	}

	// right at the limit
	r.send(http.MethodPost, "/v1/search", searchKey, "image/jpeg", padded(1024)).want(http.StatusOK)

	// over it, announced: refused before a byte is read
	e := r.send(http.MethodPost, "/v1/search", searchKey, "image/jpeg", padded(1025)).wantError(http.StatusRequestEntityTooLarge, "body_too_large")
	if !strings.Contains(e.Error.Message, "1024") {
		t.Errorf("message %q", e.Error.Message)
	}

	// over it, unannounced: found out while reading
	req := httptest.NewRequest(http.MethodPost, "/v1/search", struct{ io.Reader }{bytes.NewReader(padded(5000))})
	req.Header.Set("Authorization", "Bearer "+searchKey)
	if req.ContentLength != -1 {
		t.Fatal("the request was meant to hide its length")
	}
	r.do(req).wantError(http.StatusRequestEntityTooLarge, "body_too_large")

	// a form is held to the same limit, picture and fields together
	kind, body := form(t, map[string]string{"limit": "1"}, string(padded(900)))
	req = httptest.NewRequest(http.MethodPost, "/v1/search", struct{ io.Reader }{bytes.NewReader(body)})
	req.Header.Set("Content-Type", kind)
	req.Header.Set("Authorization", "Bearer "+searchKey)
	r.do(req).wantError(http.StatusRequestEntityTooLarge, "body_too_large")

	if r.fake.Calls() != 1 {
		t.Errorf("%d pictures reached the engine, want the one that fitted", r.fake.Calls())
	}
}

func TestUnreadablePictures(t *testing.T) {
	r := newRig(t)
	for picture, want := range map[string]struct {
		status int
		code   string
	}{
		"error:unsupported": {http.StatusUnsupportedMediaType, "unsupported_image"},
		"GIF89a........":    {http.StatusUnsupportedMediaType, "unsupported_image"},
		"error:corrupt":     {http.StatusUnprocessableEntity, "corrupt_image"},
		"error:small":       {http.StatusUnprocessableEntity, "image_too_small"},
		"error:large":       {http.StatusRequestEntityTooLarge, "image_too_large"},
	} {
		r.picture("/v1/search", searchKey, picture).wantError(want.status, want.code)
		r.picture("/v1/verify?subject=anna", searchKey, picture).wantError(http.StatusNotFound, "subject_not_found")
		r.picture("/v1/subjects/anna/faces", adminKey, picture).wantError(want.status, want.code)
	}
	if r.gallery.Info().Subjects != 0 {
		t.Error("an unreadable picture was enrolled")
	}
}
