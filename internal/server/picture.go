package server

import (
	"bytes"
	"errors"
	"io"
	"mime"
	"net/http"
)

// upload is the picture a request carries, and the form fields that came
// along with it.
type upload struct {
	picture []byte
	fields  map[string]string
}

const (
	// pictureField is the name of the form field that holds the picture.
	pictureField = "image"
	// maxField bounds the value of any other form field, in bytes.
	maxField = 1 << 10
	// maxParts bounds the parts of a form.
	maxParts = 32
)

// readUpload reads the picture of a request into memory: the "image" field
// of a multipart form, or else the body as it is. The body must already be
// limited in size.
//
// The form is read part by part rather than with the standard library's
// form parser, which moves large parts into temporary files: a picture of
// somebody's face has no business on the disk.
func readUpload(r *http.Request, limit int64) (*upload, *apiError) {
	if r.ContentLength > limit {
		return nil, bodyTooLarge(limit)
	}
	if kind, _, err := mime.ParseMediaType(r.Header.Get("Content-Type")); err == nil && kind == "multipart/form-data" {
		return readForm(r, limit)
	}
	picture, err := readAll(r.Body, r.ContentLength)
	if err != nil {
		return nil, bodyError(err, limit)
	}
	if len(picture) == 0 {
		return nil, refuse(http.StatusBadRequest, "missing_image",
			"send the picture as the request body, or as the %q field of a multipart form", pictureField)
	}
	return &upload{picture: picture}, nil
}

func readForm(r *http.Request, limit int64) (*upload, *apiError) {
	form, err := r.MultipartReader()
	if err != nil {
		return nil, refuse(http.StatusBadRequest, "bad_request", "the multipart form cannot be read: %v", err)
	}
	up := &upload{fields: make(map[string]string)}
	for parts := 0; ; parts++ {
		part, err := form.NextPart()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, bodyError(err, limit)
		}
		if parts == maxParts {
			return nil, refuse(http.StatusBadRequest, "bad_request", "the form has more than %d parts", maxParts)
		}
		name := part.FormName()
		switch {
		case name == pictureField:
			if up.picture != nil {
				return nil, refuse(http.StatusBadRequest, "bad_request", "the form has more than one %q field", pictureField)
			}
			if up.picture, err = readAll(part, 0); err != nil {
				return nil, bodyError(err, limit)
			}
		case name == "" || part.FileName() != "":
			return nil, refuse(http.StatusBadRequest, "invalid_parameter",
				"the form may carry one file, in the %q field; found another in %q", pictureField, name)
		default:
			value, err := io.ReadAll(io.LimitReader(part, maxField+1))
			if err != nil {
				return nil, bodyError(err, limit)
			}
			if len(value) > maxField {
				return nil, refuse(http.StatusBadRequest, "invalid_parameter", "the form field %q is too long", name)
			}
			if _, twice := up.fields[name]; twice {
				return nil, refuse(http.StatusBadRequest, "invalid_parameter", "the form field %q is given more than once", name)
			}
			up.fields[name] = string(value)
		}
	}
	if len(up.picture) == 0 {
		return nil, refuse(http.StatusBadRequest, "missing_image", "the form has no %q field with a picture in it", pictureField)
	}
	return up, nil
}

// readAll reads r to its end. A size known in advance saves the buffer
// from growing its way up to a few megabytes.
func readAll(r io.Reader, size int64) ([]byte, error) {
	var buf bytes.Buffer
	if size > 0 {
		buf.Grow(int(size) + bytes.MinRead)
	}
	_, err := buf.ReadFrom(r)
	return buf.Bytes(), err
}

func bodyTooLarge(limit int64) *apiError {
	return refuse(http.StatusRequestEntityTooLarge, "body_too_large", "the request is larger than %d bytes", limit)
}

func bodyError(err error, limit int64) *apiError {
	var tooLarge *http.MaxBytesError
	if errors.As(err, &tooLarge) {
		return bodyTooLarge(limit)
	}
	return refuse(http.StatusBadRequest, "bad_request", "the request body is incomplete or malformed")
}
