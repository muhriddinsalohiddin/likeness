package client

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/muhriddinsalohiddin/likeness/internal/server"
)

func TestErrorsOfTheAPI(t *testing.T) {
	r := newRig(t, func(c *server.Config) { c.MaxTemplates, c.MaxBody = 2, 4<<10 })
	r.enrol("anna", "face:anna")
	r.enrol("anna", "face:anna#2")
	admin, searcher := r.admin(), r.searcher()
	picture := []byte("face:anna")

	search := func(c *Client, picture string, o SearchOptions) error {
		_, err := c.Search(ctx, []byte(picture), o)
		return err
	}
	enrol := func(subject, picture string) error {
		_, err := admin.Enrol(ctx, subject, []byte(picture), EnrolOptions{})
		return err
	}
	tests := []struct {
		name   string
		call   func() error
		status int
		code   Code
		// check looks at what else the refusal carries.
		check func(t *testing.T, e *Error)
	}{
		{"no key", func() error { return search(r.client(""), "face:anna", SearchOptions{}) },
			http.StatusUnauthorized, CodeUnauthorised, nil},
		{"a key the server does not know", func() error { return search(r.client("wrong-0123456789abcdef"), "face:anna", SearchOptions{}) },
			http.StatusUnauthorized, CodeUnauthorised, func(t *testing.T, e *Error) {
				if strings.Contains(e.Error(), "wrong-0123456789abcdef") {
					t.Errorf("the error repeats the key that was refused: %v", e)
				}
			}},
		{"a search key listing subjects", func() error { _, err := searcher.Subjects(ctx, "", 0); return err },
			http.StatusForbidden, CodeForbidden, nil},
		{"a search key enrolling", func() error { _, err := searcher.Enrol(ctx, "boris", picture, EnrolOptions{}); return err },
			http.StatusForbidden, CodeForbidden, nil},
		{"candidates on a server without them", func() error { return search(searcher, "face:anna", SearchOptions{Candidates: true}) },
			http.StatusForbidden, CodeCandidatesDisabled, nil},
		{"a subject that is not enrolled", func() error { _, err := admin.Subject(ctx, "nobody"); return err },
			http.StatusNotFound, CodeSubjectNotFound, nil},
		{"removing a subject that is not enrolled", func() error { _, err := admin.RemoveSubject(ctx, "nobody"); return err },
			http.StatusNotFound, CodeSubjectNotFound, nil},
		{"a template the subject does not have", func() error { _, err := admin.RemoveFace(ctx, "anna", 999); return err },
			http.StatusNotFound, CodeTemplateNotFound, nil},
		{"a face that is someone else's", func() error { return enrol("annette", "face:anna#3") },
			http.StatusConflict, CodeDuplicateFace, func(t *testing.T, e *Error) {
				if e.Subject != "anna" || !near(e.Score, 0.893, 0.002) {
					t.Errorf("the duplicate names %q with similarity %v", e.Subject, e.Score)
				}
				if strings.Contains(e.Error(), "anna") {
					t.Errorf("the error line names the other subject: %v", e)
				}
			}},
		{"a face unlike the subject's", func() error { return enrol("anna", "face:boris") },
			http.StatusConflict, CodeSubjectMismatch, func(t *testing.T, e *Error) {
				if e.Subject != "" || !near(e.Score, -0.105, 0.002) {
					t.Errorf("the mismatch: subject %q, similarity %v", e.Subject, e.Score)
				}
			}},
		{"a subject that is full", func() error { return enrol("anna", "face:anna#3") },
			http.StatusConflict, CodeTooManyTemplates, func(t *testing.T, e *Error) {
				if !strings.Contains(e.Message, "2 templates") {
					t.Errorf("message %q", e.Message)
				}
			}},
		{"a blurry reference picture", func() error { return enrol("boris", "reject:blurry") },
			http.StatusUnprocessableEntity, CodeFaceRejected, func(t *testing.T, e *Error) {
				if e.Reject != ReasonBlurry || e.Quality == nil || e.Quality.Sharpness != 3 || e.Quality.EyeDistance != 84 {
					t.Errorf("rejected as %q with quality %+v", e.Reject, e.Quality)
				}
			}},
		{"a reference picture without a face", func() error { return enrol("boris", "reject:no_face") },
			http.StatusUnprocessableEntity, CodeFaceRejected, func(t *testing.T, e *Error) {
				if e.Reject != ReasonNoFace || e.Quality != nil {
					t.Errorf("rejected as %q with quality %+v", e.Reject, e.Quality)
				}
			}},
		{"a reference picture of two people", func() error { return enrol("boris", "faces:boris,carol") },
			http.StatusUnprocessableEntity, CodeFaceRejected, func(t *testing.T, e *Error) {
				if e.Reject != ReasonMultipleFaces {
					t.Errorf("rejected as %q", e.Reject)
				}
			}},
		{"a request larger than the server takes", func() error { return search(searcher, "face:anna"+strings.Repeat(" ", 8<<10), SearchOptions{}) },
			http.StatusRequestEntityTooLarge, CodeBodyTooLarge, nil},
		{"a verification larger than the server takes", func() error {
			_, err := searcher.Verify(ctx, "anna", bytes.Repeat([]byte("x"), 8<<10), VerifyOptions{})
			return err
		}, http.StatusRequestEntityTooLarge, CodeBodyTooLarge, nil},
		{"a picture with too many pixels", func() error { return search(searcher, "error:large", SearchOptions{}) },
			http.StatusRequestEntityTooLarge, CodeImageTooLarge, nil},
		{"something that is not a picture", func() error { return search(searcher, "a text file", SearchOptions{}) },
			http.StatusUnsupportedMediaType, CodeUnsupportedImage, nil},
		{"a damaged picture", func() error { return enrol("boris", "error:corrupt") },
			http.StatusUnprocessableEntity, CodeCorruptImage, nil},
		{"a tiny picture", func() error { return search(searcher, "error:small", SearchOptions{}) },
			http.StatusUnprocessableEntity, CodeImageTooSmall, nil},
		{"no picture", func() error { return search(searcher, "", SearchOptions{}) },
			http.StatusBadRequest, CodeMissingImage, nil},
		{"no picture to verify", func() error { _, err := searcher.Verify(ctx, "anna", nil, VerifyOptions{}); return err },
			http.StatusBadRequest, CodeMissingImage, nil},
		{"a subject id with a space", func() error { return enrol("a b", "face:boris") },
			http.StatusBadRequest, CodeInvalidSubject, nil},
		{"verifying against an id with a slash", func() error { _, err := searcher.Verify(ctx, "a/b", picture, VerifyOptions{}); return err },
			http.StatusBadRequest, CodeInvalidSubject, nil},
		{"more subjects than a search names", func() error { return search(searcher, "face:anna", SearchOptions{Limit: 51}) },
			http.StatusBadRequest, CodeInvalidParameter, nil},
		{"a negative limit", func() error { return search(searcher, "face:anna", SearchOptions{Limit: -1}) },
			http.StatusBadRequest, CodeInvalidParameter, nil},
		{"a threshold below the server's", func() error { return search(searcher, "face:anna", SearchOptions{Threshold: 0.2}) },
			http.StatusBadRequest, CodeInvalidParameter, nil},
		{"a verification threshold above one", func() error {
			_, err := searcher.Verify(ctx, "anna", picture, VerifyOptions{Threshold: 1.5})
			return err
		}, http.StatusBadRequest, CodeInvalidParameter, nil},
		{"a page larger than the listing gives", func() error { _, err := admin.Subjects(ctx, "", 1001); return err },
			http.StatusBadRequest, CodeInvalidParameter, nil},
		{"an engine that fails", func() error { return search(searcher, "error:exploded", SearchOptions{}) },
			http.StatusInternalServerError, CodeInternal, func(t *testing.T, e *Error) {
				if strings.Contains(e.Message, "exploded") {
					t.Errorf("the message tells what broke inside the server: %q", e.Message)
				}
			}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.call()
			e := refused(t, err, tt.status, tt.code)
			if e.Message == "" || e.RequestID == "" {
				t.Errorf("a refusal carries a message and the request id: %+v", e)
			}
			if !strings.HasPrefix(err.Error(), fmt.Sprintf("client: the server answered %d %s: ", tt.status, tt.code)) ||
				!strings.HasSuffix(err.Error(), "(request "+e.RequestID+")") {
				t.Errorf("the error line: %v", err)
			}
			if e.RetryAfter != 0 {
				t.Errorf("nothing asked for a pause of %v", e.RetryAfter)
			}
			if tt.check != nil {
				tt.check(t, e)
			}
			if IsNotFound(err) != (tt.code == CodeSubjectNotFound || tt.code == CodeTemplateNotFound) ||
				IsRejected(err) != (tt.code == CodeFaceRejected) || IsDuplicate(err) != (tt.code == CodeDuplicateFace) || IsOverloaded(err) {
				t.Errorf("not found %v, rejected %v, duplicate %v, overloaded %v", IsNotFound(err), IsRejected(err), IsDuplicate(err), IsOverloaded(err))
			}
		})
	}
	// none of the refused requests changed anything
	if page, err := admin.Subjects(ctx, "", 0); err != nil || len(page.Subjects) != 1 || page.Subjects[0].Templates != 2 {
		t.Errorf("the gallery afterwards: %+v, %v", page, err)
	}
}

func TestOverloadedServer(t *testing.T) {
	// one picture at a time and nowhere to wait
	r := newRig(t, func(c *server.Config) { c.MaxInflight, c.Queue = 1, -1 })
	r.enrol("anna", "face:anna")
	gate := make(chan struct{})
	r.fake.Gate = gate
	open := sync.OnceFunc(func() { close(gate) })
	t.Cleanup(open)

	held := make(chan error, 1)
	go func() {
		_, err := r.searcher().Search(ctx, []byte("slow:anna"), SearchOptions{})
		held <- err
	}()
	eventually(t, "the first picture to reach the engine", func() bool { return r.fake.Calls() == 2 })

	// a client that does not try again sees the refusal
	_, err := r.client(searchKey, WithRetries(0)).Search(ctx, []byte("face:anna"), SearchOptions{})
	e := refused(t, err, http.StatusTooManyRequests, CodeOverloaded)
	if !IsOverloaded(err) || e.RetryAfter != time.Second || e.Message == "" || e.RequestID == "" {
		t.Errorf("the refusal: %+v", e)
	}
	// so does an enrolment, which was not begun
	_, err = r.client(adminKey, WithRetries(0)).Enrol(ctx, "boris", []byte("face:boris"), EnrolOptions{})
	refused(t, err, http.StatusTooManyRequests, CodeOverloaded)

	// one that does waits as long as the server asks, here until the slot
	// is free, and is served
	c, waits := impatient(t, r.url, WithKey(searchKey))
	pause := c.sleep
	c.sleep = func(ctx context.Context, d time.Duration) error {
		open()
		if err := <-held; err != nil {
			t.Errorf("the picture that held the slot: %v", err)
		}
		return pause(ctx, d)
	}
	res, err := c.Search(ctx, []byte("face:anna"), SearchOptions{})
	if err != nil || res.Best() == nil {
		t.Fatalf("after the wait: %+v, %v", res, err)
	}
	if len(*waits) != 1 || (*waits)[0] != time.Second {
		t.Errorf("waited %v, want the second the server asked for", *waits)
	}
	if r.fake.Calls() != 3 {
		t.Errorf("%d pictures reached the engine; the refused ones should not have", r.fake.Calls())
	}
}

func TestAnswersThatAreNotTheAPIs(t *testing.T) {
	page := "<html>\r\n<head><title>502 Bad Gateway</title></head>\r\n<body>\r\n<center><h1>502 Bad Gateway</h1></center>\r\n" +
		strings.Repeat("<!-- a padding to disable MSIE and Chrome friendly error page -->\r\n", 6) + "</body>\r\n</html>\r\n"
	tests := []struct {
		name    string
		step    http.HandlerFunc
		status  int
		message string
		line    string
	}{
		{"the error page of a proxy", reply(http.StatusBadGateway, page, "Content-Type", "text/html"), http.StatusBadGateway,
			strings.Join(strings.Fields(page), " ")[:excerptLength] + "...",
			"client: the server answered 502 Bad Gateway: <html> <head>"},
		{"plain text", reply(http.StatusServiceUnavailable, "upstream connect error\n"), http.StatusServiceUnavailable,
			"upstream connect error", "client: the server answered 503 Service Unavailable: upstream connect error"},
		{"no body at all", reply(http.StatusGatewayTimeout, ""), http.StatusGatewayTimeout,
			"", "client: the server answered 504 Gateway Timeout"},
		{"JSON of another shape", reply(http.StatusForbidden, `{"message":"blocked by policy"}`), http.StatusForbidden,
			`{"message":"blocked by policy"}`, `client: the server answered 403 Forbidden: {"message":"blocked by policy"}`},
		{"an error without a code", reply(http.StatusBadRequest, `{"error":{"message":"no"}}`), http.StatusBadRequest,
			`{"error":{"message":"no"}}`, "client: the server answered 400 Bad Request: "},
		{"a status without a name", reply(599, "odd"), 599, "odd", "client: the server answered 599: odd"},
		{"a redirect", reply(http.StatusMovedPermanently, "", "Location", "/elsewhere"), http.StatusMovedPermanently,
			"", "client: the server answered 301 Moved Permanently"},
		{"text cut inside a letter", reply(http.StatusBadGateway, "x"+strings.Repeat("я", 150)), http.StatusBadGateway,
			"x" + strings.Repeat("я", 99) + "...", "client: the server answered 502 Bad Gateway: xяя"},
		{"bytes that are not text", reply(http.StatusBadGateway, "bad \xff\xfe gateway"), http.StatusBadGateway,
			"bad gateway", "client: the server answered 502 Bad Gateway: bad gateway"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := play(t, tt.step)
			_, err := newClient(t, s.url, WithKey(searchKey)).Search(ctx, []byte("face:anna"), SearchOptions{})
			e := refused(t, err, tt.status, "")
			if e.Message != tt.message {
				t.Errorf("message %q, want %q", e.Message, tt.message)
			}
			if !strings.HasPrefix(err.Error(), tt.line) {
				t.Errorf("the error line %q does not begin with %q", err, tt.line)
			}
			if e.RequestID != "" || e.Reject != "" || e.Quality != nil || e.Subject != "" || e.Score != 0 {
				t.Errorf("fields nobody sent: %+v", e)
			}
			if IsNotFound(err) || IsRejected(err) || IsDuplicate(err) || IsOverloaded(err) {
				t.Errorf("%v passes for a refusal of the API", err)
			}
			s.count(1) // neither followed nor sent again
		})
	}
}

func TestRequestIDOfARefusal(t *testing.T) {
	const body = `{"error":{"code":"internal","message":"broken"},"request_id":"from-the-body"}`
	// the header and the body agree on a real server; the header is what
	// a proxy would also set, and it wins
	s := play(t,
		reply(http.StatusInternalServerError, body, "X-Request-Id", "from-the-header"),
		reply(http.StatusInternalServerError, body),
		reply(http.StatusBadGateway, "<html>", "X-Request-Id", "from-the-proxy"),
	)
	c := newClient(t, s.url)
	for _, want := range []string{"from-the-header", "from-the-body", "from-the-proxy"} {
		_, err := c.Info(ctx)
		var e *Error
		if !errors.As(err, &e) || e.RequestID != want {
			t.Errorf("request id of %v, want %q", err, want)
		}
	}
}

func TestValuesOfANewerServer(t *testing.T) {
	s := play(t,
		ok(`{"faces":[{"box":{"left":1,"top":2,"right":3,"bottom":4},"quality":{"score":0.9,"liveness":0.2},"reject":"wearing_a_mask","match":false,"matches":[],"mood":"calm"}],
		     "faces_found":1,"threshold":0.7,"width":4,"height":3,"model_age_days":12,
		     "timing_ms":{"read":0.5,"rerank":1.25,"search":0.004,"total":3}}`),
		reply(http.StatusUnprocessableEntity, `{"error":{"code":"liveness_failed","message":"the face is a photograph","reject":"printed","hint":"blink"},"request_id":"new-1"}`),
		reply(http.StatusPaymentRequired, `{"error":{"code":"quota_exceeded","message":"enough for today"},"request_id":"new-2"}`),
	)
	c := newClient(t, s.url)

	res, err := c.Search(ctx, []byte("face:anna"), SearchOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if got := res.Faces[0].Reject; got != Reason("wearing_a_mask") {
		t.Errorf("an unknown reason arrived as %q", got)
	}
	if res.Timing.Other["rerank"] != 1250*time.Microsecond || len(res.Timing.Other) != 1 || res.Timing.Search != 4*time.Microsecond {
		t.Errorf("an unknown stage: %+v", res.Timing)
	}

	_, err = c.Enrol(ctx, "anna", []byte("face:anna"), EnrolOptions{})
	e := refused(t, err, http.StatusUnprocessableEntity, Code("liveness_failed"))
	if e.Reject != Reason("printed") || e.Message != "the face is a photograph" || IsRejected(err) {
		t.Errorf("an unknown refusal: %+v", e)
	}
	_, err = c.Info(ctx)
	if e := refused(t, err, http.StatusPaymentRequired, Code("quota_exceeded")); e.RequestID != "new-2" {
		t.Errorf("an unknown code: %+v", e)
	}
}

func TestPredicatesSeeThroughWrapping(t *testing.T) {
	tests := []struct {
		e                                         *Error
		notFound, overloaded, rejected, duplicate bool
	}{
		{&Error{Status: 404, Code: CodeSubjectNotFound}, true, false, false, false},
		{&Error{Status: 404, Code: CodeTemplateNotFound}, true, false, false, false},
		{&Error{Status: 404, Code: CodeNotFound}, false, false, false, false},
		{&Error{Status: 404}, false, false, false, false},
		{&Error{Status: 429, Code: CodeOverloaded}, false, true, false, false},
		{&Error{Status: 429}, false, true, false, false}, // a proxy that limits requests
		{&Error{Status: 422, Code: CodeFaceRejected}, false, false, true, false},
		{&Error{Status: 422, Code: CodeCorruptImage}, false, false, false, false},
		{&Error{Status: 409, Code: CodeDuplicateFace}, false, false, false, true},
		{&Error{Status: 409, Code: CodeSubjectMismatch}, false, false, false, false},
	}
	for _, tt := range tests {
		for _, err := range []error{tt.e, fmt.Errorf("enrolling employee 1042: %w", tt.e), errors.Join(io.EOF, tt.e)} {
			if IsNotFound(err) != tt.notFound || IsOverloaded(err) != tt.overloaded || IsRejected(err) != tt.rejected || IsDuplicate(err) != tt.duplicate {
				t.Errorf("%v: not found %v, overloaded %v, rejected %v, duplicate %v",
					err, IsNotFound(err), IsOverloaded(err), IsRejected(err), IsDuplicate(err))
			}
			var e *Error
			if !errors.As(err, &e) || e != tt.e {
				t.Errorf("%v: errors.As does not find the refusal", err)
			}
		}
	}
	for _, err := range []error{nil, io.EOF, errors.New("client: 404 subject_not_found")} {
		if IsNotFound(err) || IsOverloaded(err) || IsRejected(err) || IsDuplicate(err) {
			t.Errorf("%v passes for a refusal", err)
		}
	}
}

func TestTheKeyStaysOutOfErrors(t *testing.T) {
	// a proxy that repeats the request in its error page, and a server
	// that does so in its message
	echo := func(status int, format string) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(status)
			fmt.Fprintf(w, format, r.Header.Get("Authorization"))
		}
	}
	s := play(t,
		echo(http.StatusBadGateway, "<pre>POST /v1/search\nAuthorization: %s\n</pre>"),
		echo(http.StatusBadGateway, strings.Repeat("x", excerptLength-len("Bearer ")-8)+" %s and more"),
		echo(http.StatusUnauthorized, `{"error":{"code":"unauthorized","message":"%s is not a key of this server"}}`),
		hangUp,
		ok("not JSON"),
	)
	c := newClient(t, s.url, WithKey(searchKey), WithRetries(0))
	for i, want := range []string{"Authorization: Bearer [key]", "Bearer [key] a...", "Bearer [key] is not a key", "EOF", "not the JSON"} {
		_, err := c.Search(ctx, []byte("face:anna"), SearchOptions{})
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("request %d: %v, want an error with %q", i+1, err, want)
			continue
		}
		keyless(t, err)
		// not even the part of it that the cut of a long body would leave
		if detail := fmt.Sprintf("%v %+v %#v", err, err, err); strings.Contains(detail, searchKey[:8]) {
			t.Errorf("request %d: the error gives the key away when printed in full: %s", i+1, detail)
		}
	}
	s.count(5)

	// nor is it in the error of a server that cannot be reached
	gone, _ := impatient(t, "http://faces.invalid", WithKey(searchKey), WithHTTPClient(&http.Client{Transport: failing{}}))
	_, err := gone.Search(ctx, []byte("face:anna"), SearchOptions{})
	if err == nil || !strings.Contains(err.Error(), "no route to host") {
		t.Fatalf("a server out of reach: %v", err)
	}
	keyless(t, err)
}

// failing is a transport that reaches nobody.
type failing struct{}

func (failing) RoundTrip(*http.Request) (*http.Response, error) {
	return nil, errors.New("connect: no route to host")
}
