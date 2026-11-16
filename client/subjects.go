package client

import (
	"context"
	"errors"
	"iter"
	"net/http"
	"net/url"
	"strconv"
	"time"
)

// errNoSubject is returned before anything is sent: a path with nothing
// where the subject belongs would be a different request.
var errNoSubject = errors.New("client: the subject id is empty")

// subjectPath is the path of a subject.
func subjectPath(id string) (string, error) {
	if id == "" {
		return "", errNoSubject
	}
	return "/v1/subjects/" + segment(id), nil
}

// segment escapes a subject id for use as one segment of a path. An id
// that is "." or ".." has its dots escaped as well: left alone it would be
// read as a step in the path, not as a name.
func segment(id string) string {
	switch id {
	case ".":
		return "%2E"
	case "..":
		return "%2E%2E"
	}
	return url.PathEscape(id)
}

// EnrolOptions adjust an enrolment. The zero value adds a template to the
// subject and lets the server's checks guard the gallery.
type EnrolOptions struct {
	// Replace drops the subject's earlier templates once the new one is
	// in. An enrolment that replaces can be repeated without harm, so the
	// client sends it again if the connection breaks.
	Replace bool
	// AllowDuplicate enrols the face even if it matches another subject:
	// twins, or one person who is meant to have two ids.
	AllowDuplicate bool
	// AllowMismatch enrols the face even if it does not look like the
	// subject's earlier ones: people change.
	AllowMismatch bool
}

// Enrolment is the answer to an enrolment that went through: the template
// is on disk and searchable.
type Enrolment struct {
	// Subject is the id the face was enrolled under.
	Subject string `json:"subject"`
	// Template is the id of the new template, Templates how many the
	// subject holds now and Created when the template was made.
	Template  uint64    `json:"template"`
	Templates int       `json:"templates"`
	Created   time.Time `json:"created"`
	// Face is the face that was enrolled and Timing where the server spent
	// its time.
	Face   Face   `json:"face"`
	Timing Timing `json:"timing_ms"`
	// RequestID is the id under which the server logged the request.
	RequestID string `json:"request_id,omitempty"`
}

// Enrol remembers the face in the picture as the subject's; the subject is
// created by its first face. It needs an admin key.
//
// A reference picture is held to a higher standard than a probe. One that
// does not meet it is refused with an error for which IsRejected holds.
// Three checks guard the gallery, each with a code of its own:
// CodeDuplicateFace, for which IsDuplicate holds, CodeSubjectMismatch and
// CodeTooManyTemplates.
//
// Enrolling is not idempotent: the same picture sent twice adds two
// templates. Unless Replace is set, Enrol therefore sends the request once
// and reports a broken connection instead of trying again; the caller, who
// can look the subject up, decides. A server that refuses for overload has
// done nothing, and is asked again as in every other call.
func (c *Client) Enrol(ctx context.Context, subject string, picture []byte, o EnrolOptions) (*Enrolment, error) {
	path, err := subjectPath(subject)
	if err != nil {
		return nil, err
	}
	query := url.Values{}
	if o.Replace {
		query.Set("replace", "true")
	}
	if o.AllowDuplicate {
		query.Set("allow_duplicate", "true")
	}
	if o.AllowMismatch {
		query.Set("allow_mismatch", "true")
	}
	rq := upload(path+"/faces", query, picture)
	rq.repeatable = o.Replace
	res := new(Enrolment)
	id, err := c.do(ctx, rq, res)
	if err != nil {
		return nil, err
	}
	res.RequestID = id
	return res, nil
}

// Template is one enrolled face of a subject. Its vector stays on the
// server.
type Template struct {
	// ID names the template within its subject; Created is when it was
	// enrolled.
	ID      uint64    `json:"id"`
	Created time.Time `json:"created"`
	// Quality is what was measured when the face was enrolled; nil for a
	// template that was not enrolled through the API.
	Quality *Quality `json:"quality,omitempty"`
}

// Subject is an enrolled person.
type Subject struct {
	// ID is the subject's id and Created when its first face was enrolled.
	ID      string    `json:"subject"`
	Created time.Time `json:"created"`
	// Templates are the subject's faces, oldest first.
	Templates []Template `json:"templates"`
	// RequestID is the id under which the server logged the request.
	RequestID string `json:"request_id,omitempty"`
}

// Subject returns a subject and its templates. It needs an admin key. For
// someone who is not enrolled the error satisfies IsNotFound.
func (c *Client) Subject(ctx context.Context, id string) (*Subject, error) {
	path, err := subjectPath(id)
	if err != nil {
		return nil, err
	}
	res := new(Subject)
	rid, err := c.do(ctx, &call{method: http.MethodGet, path: path, repeatable: true}, res)
	if err != nil {
		return nil, err
	}
	res.RequestID = rid
	return res, nil
}

// SubjectSummary is a subject as the listing shows it.
type SubjectSummary struct {
	// ID is the subject's id and Created when its first face was enrolled.
	ID      string    `json:"subject"`
	Created time.Time `json:"created"`
	// Templates is how many templates the subject holds.
	Templates int `json:"templates"`
}

// SubjectPage is one page of the listing of subjects.
type SubjectPage struct {
	// Subjects are the people on the page, in id order.
	Subjects []SubjectSummary `json:"subjects"`
	// Next continues the listing when passed as after; it is empty on the
	// last page.
	Next string `json:"next,omitempty"`
	// RequestID is the id under which the server logged the request.
	RequestID string `json:"request_id,omitempty"`
}

// Subjects returns one page of the enrolled people, in id order, starting
// after the id given in after; an empty one starts at the beginning. A
// limit of zero leaves the size of the page to the server; the largest the
// API allows is 1000. It needs an admin key. AllSubjects walks the pages.
func (c *Client) Subjects(ctx context.Context, after string, limit int) (*SubjectPage, error) {
	query := url.Values{}
	if after != "" {
		query.Set("after", after)
	}
	if limit != 0 {
		query.Set("limit", strconv.Itoa(limit))
	}
	res := new(SubjectPage)
	id, err := c.do(ctx, &call{method: http.MethodGet, path: "/v1/subjects", query: query, repeatable: true}, res)
	if err != nil {
		return nil, err
	}
	res.RequestID = id
	return res, nil
}

// AllSubjects iterates over everyone enrolled, in id order, fetching a
// page at a time. If a page cannot be fetched the iteration ends with that
// error:
//
//	for subject, err := range c.AllSubjects(ctx) {
//		if err != nil {
//			return err
//		}
//		// subject.ID, subject.Templates
//	}
//
// People enrolled or removed while it runs may or may not be seen.
func (c *Client) AllSubjects(ctx context.Context) iter.Seq2[SubjectSummary, error] {
	return func(yield func(SubjectSummary, error) bool) {
		after := ""
		for {
			page, err := c.Subjects(ctx, after, c.page)
			if err != nil {
				yield(SubjectSummary{}, err)
				return
			}
			for _, subject := range page.Subjects {
				if !yield(subject, nil) {
					return
				}
			}
			if page.Next == "" || len(page.Subjects) == 0 {
				return
			}
			after = page.Next
		}
	}
}

// Removal is the answer to a removal: it is on disk and out of the index.
type Removal struct {
	// Subject is the id of the subject the removal was about.
	Subject string `json:"subject"`
	// Template is the template removed, when one was named.
	Template uint64 `json:"template,omitempty"`
	// Removed is how many templates went and Templates how many the
	// subject has left; with none left the subject is gone.
	Removed   int `json:"removed"`
	Templates int `json:"templates"`
	// RequestID is the id under which the server logged the request.
	RequestID string `json:"request_id,omitempty"`
}

// RemoveSubject removes a subject and all its templates. It needs an
// admin key. For someone who is not enrolled the error satisfies
// IsNotFound; that is also what a removal answers when it had to be sent
// again after a broken connection and the first attempt had got through.
func (c *Client) RemoveSubject(ctx context.Context, id string) (*Removal, error) {
	path, err := subjectPath(id)
	if err != nil {
		return nil, err
	}
	return c.remove(ctx, path)
}

// RemoveFace removes one template of a subject, by the id an enrolment or
// Subject gave it. It needs an admin key. A subject whose last template is
// removed is gone. For a subject or a template that does not exist the
// error satisfies IsNotFound, as it does for a removal repeated after a
// broken connection when the first attempt had got through.
func (c *Client) RemoveFace(ctx context.Context, id string, template uint64) (*Removal, error) {
	path, err := subjectPath(id)
	if err != nil {
		return nil, err
	}
	return c.remove(ctx, path+"/faces/"+strconv.FormatUint(template, 10))
}

func (c *Client) remove(ctx context.Context, path string) (*Removal, error) {
	res := new(Removal)
	id, err := c.do(ctx, &call{method: http.MethodDelete, path: path, repeatable: true}, res)
	if err != nil {
		return nil, err
	}
	res.RequestID = id
	return res, nil
}
