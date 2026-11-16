package client

import (
	"encoding/json"
	"errors"
	"net/http"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"
)

// Code is the API's name for what went wrong. Codes are stable and meant
// for programs; a server newer than this package may send ones that are
// not listed here, and they arrive as they are.
type Code string

const (
	// CodeBadRequest: the body or the query string is malformed.
	CodeBadRequest Code = "bad_request"
	// CodeInvalidParameter: an option has a value the server does not
	// accept.
	CodeInvalidParameter Code = "invalid_parameter"
	// CodeMissingImage: the request carried no picture.
	CodeMissingImage Code = "missing_image"
	// CodeMissingSubject: a verification named no subject.
	CodeMissingSubject Code = "missing_subject"
	// CodeInvalidSubject: the subject id breaks the rules: 1 to 128 bytes
	// of UTF-8 without spaces, control characters or slashes.
	CodeInvalidSubject Code = "invalid_subject"
	// CodeUnauthorised: no API key, or one the server does not know.
	CodeUnauthorised Code = "unauthorized"
	// CodeForbidden: the key may not do this; enrolling, listing and
	// removing need an admin key.
	CodeForbidden Code = "forbidden"
	// CodeCandidatesDisabled: the search asked for candidates on a server
	// that does not name subjects below the threshold.
	CodeCandidatesDisabled Code = "candidates_disabled"
	// CodeReadOnly: the server does not accept changes.
	CodeReadOnly Code = "read_only"
	// CodeSubjectNotFound: no such subject is enrolled.
	CodeSubjectNotFound Code = "subject_not_found"
	// CodeTemplateNotFound: the subject has no such template.
	CodeTemplateNotFound Code = "template_not_found"
	// CodeNotFound: the server has nothing at the path, which points at a
	// wrong base URL rather than at a missing subject.
	CodeNotFound Code = "not_found"
	// CodeMethodNotAllowed: the path does not answer to the method.
	CodeMethodNotAllowed Code = "method_not_allowed"
	// CodeDuplicateFace: the face being enrolled already belongs to another
	// subject, named in Error.Subject with the similarity in Error.Score.
	CodeDuplicateFace Code = "duplicate_face"
	// CodeSubjectMismatch: the face being enrolled does not look like the
	// subject's earlier ones; Error.Score says how little.
	CodeSubjectMismatch Code = "subject_mismatch"
	// CodeTooManyTemplates: the subject holds as many templates as the
	// server allows.
	CodeTooManyTemplates Code = "too_many_templates"
	// CodeBodyTooLarge: the request is larger than the server accepts.
	CodeBodyTooLarge Code = "body_too_large"
	// CodeImageTooLarge: the picture has too many pixels.
	CodeImageTooLarge Code = "image_too_large"
	// CodeUnsupportedImage: the picture is not a JPEG, PNG or WebP file.
	CodeUnsupportedImage Code = "unsupported_image"
	// CodeCorruptImage: the picture is damaged or cut short.
	CodeCorruptImage Code = "corrupt_image"
	// CodeImageTooSmall: the picture is too small to hold a face.
	CodeImageTooSmall Code = "image_too_small"
	// CodeFaceRejected: the picture cannot serve as a reference;
	// Error.Reject says why and Error.Quality what was measured.
	CodeFaceRejected Code = "face_rejected"
	// CodeOverloaded: the server has more pictures than it can analyse.
	// Nothing was done; the request can be sent again.
	CodeOverloaded Code = "overloaded"
	// CodeInternal: a fault in the server; its log has the details under
	// the request id.
	CodeInternal Code = "internal"
	// CodeShuttingDown: the server is stopping.
	CodeShuttingDown Code = "shutting_down"
)

// Error is an answer of the server that is not a success. Take it out of
// an error with errors.As, or ask one of IsNotFound, IsOverloaded,
// IsRejected and IsDuplicate.
type Error struct {
	// Status is the HTTP status of the answer.
	Status int
	// Code is the API's name for what went wrong. It is empty when the
	// answer is not in the API's shape, as the error page of a proxy in
	// front of the server is not.
	Code Code
	// Message is meant for the person reading a log. For an answer that is
	// not in the API's shape it is the beginning of the body.
	Message string
	// RequestID is the id under which the server logged the request.
	RequestID string

	// Reject says why a reference picture was turned down and Quality what
	// was measured on the face, if one was found: CodeFaceRejected.
	Reject  Reason
	Quality *Quality
	// Subject is the person an enrolment collided with (CodeDuplicateFace)
	// and Score the similarity of the faces (also CodeSubjectMismatch).
	Subject string
	Score   float64
	// RetryAfter is how long the server asks to wait before the request is
	// sent again; zero if it did not say.
	RetryAfter time.Duration
}

// Error describes the answer in one line: status, code, message and
// request id. It leaves out the subject of a duplicate, which may be a
// personal number and has no place in a log.
func (e *Error) Error() string {
	var b strings.Builder
	b.WriteString("client: the server answered ")
	b.WriteString(strconv.Itoa(e.Status))
	if e.Code != "" {
		b.WriteString(" " + string(e.Code))
	} else if text := http.StatusText(e.Status); text != "" {
		b.WriteString(" " + text)
	}
	if e.Message != "" {
		b.WriteString(": " + e.Message)
	}
	if e.RequestID != "" {
		b.WriteString(" (request " + e.RequestID + ")")
	}
	return b.String()
}

// IsNotFound reports whether err says that the subject, or the template,
// is not enrolled. A path the server does not know (CodeNotFound) does not
// count: that is a wrong base URL, and must not pass for a missing person.
func IsNotFound(err error) bool {
	e, ok := errors.AsType[*Error](err)
	return ok && (e.Code == CodeSubjectNotFound || e.Code == CodeTemplateNotFound)
}

// IsOverloaded reports whether err is the server refusing a request
// because it is busy, after the client's own retries. Nothing was done
// with the request; it is safe to send again later.
func IsOverloaded(err error) bool {
	e, ok := errors.AsType[*Error](err)
	return ok && e.Status == http.StatusTooManyRequests
}

// IsRejected reports whether err is an enrolment turned down because the
// picture cannot serve as a reference. The Error says why in Reject.
func IsRejected(err error) bool {
	e, ok := errors.AsType[*Error](err)
	return ok && e.Code == CodeFaceRejected
}

// IsDuplicate reports whether err is an enrolment turned down because the
// face already belongs to another subject, named in the Error's Subject.
func IsDuplicate(err error) bool {
	e, ok := errors.AsType[*Error](err)
	return ok && e.Code == CodeDuplicateFace
}

// refusal turns an answer that is not a success into an Error.
func (c *Client) refusal(res *http.Response, body []byte) *Error {
	e := &Error{
		Status:     res.StatusCode,
		RequestID:  res.Header.Get("X-Request-Id"),
		RetryAfter: retryAfter(res.Header.Get("Retry-After")),
	}
	var wire struct {
		Error *struct {
			Code    Code     `json:"code"`
			Message string   `json:"message"`
			Reject  Reason   `json:"reject"`
			Quality *Quality `json:"quality"`
			Subject string   `json:"subject"`
			Score   *float64 `json:"score"`
		} `json:"error"`
		RequestID string `json:"request_id"`
	}
	if json.Unmarshal(body, &wire) != nil || wire.Error == nil || wire.Error.Code == "" {
		// Not from a likeness server: show what it was. The key goes first,
		// in case a proxy echoes the request.
		e.Message = excerpt(c.scrub(string(body)))
		return e
	}
	e.Code, e.Message = wire.Error.Code, c.scrub(wire.Error.Message)
	e.Reject, e.Quality, e.Subject = wire.Error.Reject, wire.Error.Quality, wire.Error.Subject
	if wire.Error.Score != nil {
		e.Score = *wire.Error.Score
	}
	if e.RequestID == "" {
		e.RequestID = wire.RequestID
	}
	return e
}

// excerptLength is how much of a body that is not an API error is quoted.
const excerptLength = 200

// excerpt is the beginning of a body on one line.
func excerpt(body string) string {
	text := strings.Join(strings.Fields(strings.ToValidUTF8(body, "")), " ")
	if len(text) <= excerptLength {
		return text
	}
	cut := excerptLength
	for cut > 0 && !utf8.RuneStart(text[cut]) {
		cut--
	}
	return text[:cut] + "..."
}
