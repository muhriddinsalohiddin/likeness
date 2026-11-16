package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/muhriddinsalohiddin/likeness"
)

// apiError is a refusal the client is told about:
//
//	{"error": {"code": "duplicate_face", "message": "...", ...}, "request_id": "..."}
//
// The code is stable and meant for programs, the message for the people
// writing them.
type apiError struct {
	Code    string `json:"code"`
	Message string `json:"message"`

	// Reject and Quality say why a reference picture was turned down.
	Reject  likeness.Reason `json:"reject,omitempty"`
	Quality *qualityJSON    `json:"quality,omitempty"`
	// Subject is the person an enrolment collided with, Score how much the
	// faces resemble each other.
	Subject string   `json:"subject,omitempty"`
	Score   *float64 `json:"score,omitempty"`

	status     int
	retryAfter int    // seconds
	allow      string // the methods a path does answer to
	cause      error  // for the log, never for the client
}

// statusClientGone is what the log shows for a request whose client went
// away before it could be answered. Nothing is sent.
const statusClientGone = 499

func refuse(status int, code, format string, args ...any) *apiError {
	return &apiError{status: status, Code: code, Message: fmt.Sprintf(format, args...)}
}

// internal hides what went wrong from the client and keeps it for the log.
func internal(cause error) *apiError {
	return &apiError{
		status: http.StatusInternalServerError, Code: "internal", cause: cause,
		Message: "the server failed to handle the request; its log has the details under this request id",
	}
}

func gone(cause error) *apiError {
	return &apiError{status: statusClientGone, Code: "client_gone", cause: cause}
}

func overloaded(cause error) *apiError {
	return &apiError{
		status: http.StatusTooManyRequests, Code: "overloaded", retryAfter: 1, cause: cause,
		Message: "the server has more pictures to analyse than it can handle; try again shortly",
	}
}

func notEnrolled() *apiError {
	return refuse(http.StatusNotFound, "subject_not_found", "no such subject is enrolled")
}

// pictureError translates what the engine says about a picture it could
// not analyse.
func pictureError(err error) *apiError {
	switch {
	case errors.Is(err, likeness.ErrUnsupportedImage):
		return refuse(http.StatusUnsupportedMediaType, "unsupported_image", "the picture is not a JPEG, PNG or WebP image")
	case errors.Is(err, likeness.ErrImageTooLarge):
		return refuse(http.StatusRequestEntityTooLarge, "image_too_large", "the picture has more pixels than the server accepts")
	case errors.Is(err, likeness.ErrImageTooSmall):
		return refuse(http.StatusUnprocessableEntity, "image_too_small", "the picture is too small to hold a face")
	case errors.Is(err, likeness.ErrCorruptImage):
		return refuse(http.StatusUnprocessableEntity, "corrupt_image", "the picture is damaged or cut short")
	case errors.Is(err, context.Canceled), errors.Is(err, context.DeadlineExceeded):
		return gone(err)
	case errors.Is(err, likeness.ErrClosed):
		return refuse(http.StatusServiceUnavailable, "shutting_down", "the server is shutting down")
	}
	return internal(err)
}

// advice says, for the person in front of the camera, what a reason means.
func advice(reason likeness.Reason) string {
	switch reason {
	case likeness.NoFace:
		return "no face was found in it"
	case likeness.MultipleFaces:
		return "more than one person is in it"
	case likeness.LowScore:
		return "what is in it may not be a face"
	case likeness.OutOfFrame:
		return "part of the face is outside the frame"
	case likeness.FaceTooSmall:
		return "the face is too small; move closer to the camera"
	case likeness.BadPose:
		return "the head is turned or tilted; look straight at the camera"
	case likeness.TooDark:
		return "the face is too dark"
	case likeness.TooBright:
		return "the face is overexposed"
	case likeness.Blurry:
		return "the face is not sharp"
	case likeness.Spoof:
		return "it looks like a picture of a face rather than a face"
	}
	return string(reason)
}

type errorBody struct {
	Error     *apiError `json:"error"`
	RequestID string    `json:"request_id"`
}

// fail answers a request with an error.
func (s *Server) fail(w http.ResponseWriter, c *call, e *apiError) {
	c.err = e
	if e.status == statusClientGone {
		// Nobody is listening; the status is recorded for the log alone.
		if rec, ok := w.(*recorder); ok && rec.status == 0 {
			rec.status = statusClientGone
		}
		return
	}
	h := w.Header()
	if e.retryAfter > 0 {
		h.Set("Retry-After", strconv.Itoa(e.retryAfter))
	}
	if e.allow != "" {
		h.Set("Allow", e.allow)
	}
	if e.status == http.StatusUnauthorized {
		h.Set("WWW-Authenticate", `Bearer realm="likeness"`)
	}
	writeJSON(w, e.status, errorBody{Error: e, RequestID: c.id})
}

// reply answers a request with a JSON document, and says in a Server-Timing
// header where the time went if the request went through any stages.
func (s *Server) reply(w http.ResponseWriter, c *call, status int, body any) {
	if len(c.final) > 0 {
		w.Header().Set("Server-Timing", c.final.header())
	}
	writeJSON(w, status, body)
}

func writeJSON(w http.ResponseWriter, status int, body any) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(body); err != nil {
		// Only our own types are encoded, so this is a bug; say so plainly.
		status = http.StatusInternalServerError
		buf.Reset()
		buf.WriteString(`{"error":{"code":"internal","message":"the answer could not be encoded"}}` + "\n")
	}
	h := w.Header()
	h.Set("Content-Type", "application/json")
	h.Set("Content-Length", strconv.Itoa(buf.Len()))
	w.WriteHeader(status)
	w.Write(buf.Bytes())
}

// stage is one step of handling a request and the time it took.
type stage struct {
	name string
	took time.Duration
}

// stages is where the time of a request went, in order. In JSON it is an
// object of milliseconds: {"read": 0.41, "queue": 0, "decode": 5.2, ...}.
type stages []stage

func (s stages) MarshalJSON() ([]byte, error) {
	b := []byte{'{'}
	for i, st := range s {
		if i > 0 {
			b = append(b, ',')
		}
		b = strconv.AppendQuote(b, st.name)
		b = append(b, ':')
		b = strconv.AppendFloat(b, millis(st.took), 'f', -1, 64)
	}
	return append(b, '}'), nil
}

// header renders the stages as a Server-Timing header value.
func (s stages) header() string {
	var b strings.Builder
	for i, st := range s {
		if i > 0 {
			b.WriteString(", ")
		}
		b.WriteString(st.name + ";dur=" + strconv.FormatFloat(millis(st.took), 'f', -1, 64))
	}
	return b.String()
}

// millis keeps microseconds: a search over a few thousand faces takes no
// more than a handful of them.
func millis(d time.Duration) float64 {
	return round(float64(d)/float64(time.Millisecond), 3)
}

// round keeps the JSON readable: a similarity of 0.83421224 says no more
// than 0.8342 does.
func round(v float64, places int) float64 {
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return 0
	}
	scale := math.Pow10(places)
	return math.Round(v*scale) / scale
}

type boxJSON struct {
	Left   float64 `json:"left"`
	Top    float64 `json:"top"`
	Right  float64 `json:"right"`
	Bottom float64 `json:"bottom"`
}

type qualityJSON struct {
	Score       float64 `json:"score"`
	EyeDistance float64 `json:"eye_distance"`
	Yaw         float64 `json:"yaw"`
	Pitch       float64 `json:"pitch"`
	Roll        float64 `json:"roll"`
	InFrame     float64 `json:"in_frame"`
	// Brightness and Sharpness are measured on the aligned face, so a face
	// turned down for its size or pose before that has neither.
	Brightness float64 `json:"brightness,omitempty"`
	Sharpness  float64 `json:"sharpness,omitempty"`
}

// faceJSON is a face as it was seen in the picture.
type faceJSON struct {
	Box     boxJSON     `json:"box"`
	Quality qualityJSON `json:"quality"`
	// Liveness is absent when it was not judged.
	Liveness float64         `json:"liveness,omitempty"`
	Reject   likeness.Reason `json:"reject,omitempty"`
}

func qualityOf(q likeness.Quality) qualityJSON {
	return qualityJSON{
		Score:       round(float64(q.Score), 3),
		EyeDistance: round(float64(q.EyeDistance), 1),
		Yaw:         round(float64(q.Yaw), 1),
		Pitch:       round(float64(q.Pitch), 1),
		Roll:        round(float64(q.Roll), 1),
		InFrame:     round(float64(q.InFrame), 3),
		Brightness:  round(float64(q.Brightness), 1),
		Sharpness:   round(float64(q.Sharpness), 1),
	}
}

func describe(f *likeness.Face) faceJSON {
	return faceJSON{
		Box: boxJSON{
			Left: round(float64(f.Box.Left), 1), Top: round(float64(f.Box.Top), 1),
			Right: round(float64(f.Box.Right), 1), Bottom: round(float64(f.Box.Bottom), 1),
		},
		Quality:  qualityOf(f.Quality),
		Liveness: round(float64(f.Liveness), 3),
		Reject:   f.Reject,
	}
}

func similarity(score float32) float64 { return round(float64(score), 4) }
