package client

import (
	"bytes"
	"errors"
	"fmt"
	"go/parser"
	"go/token"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestNew(t *testing.T) {
	for give, want := range map[string]string{
		"http://localhost:8080":              "http://localhost:8080",
		"http://localhost:8080/":             "http://localhost:8080",
		"https://faces.internal":             "https://faces.internal",
		"https://faces.internal/api/faces":   "https://faces.internal/api/faces",
		"https://faces.internal/api/faces/":  "https://faces.internal/api/faces",
		"https://faces.internal/api/faces//": "https://faces.internal/api/faces",
		"https://faces.internal/a%20b/":      "https://faces.internal/a%20b",
		"http://[::1]:8080/":                 "http://[::1]:8080",
	} {
		c, err := New(give)
		if err != nil {
			t.Errorf("%s: %v", give, err)
			continue
		}
		if c.base != want {
			t.Errorf("%s: requests go to %s, want %s", give, c.base, want)
		}
	}

	for _, tt := range []struct{ give, want string }{
		{"", "has to look like https://host:port"},
		{"faces.internal", "has to look like https://host:port"},
		{"faces.internal:8080", "has to look like https://host:port"},
		{"localhost:8080/api", "has to look like https://host:port"},
		{"https://faces.internal/%zz", "cannot be read"},
		{"/v1", "has to look like https://host:port"},
		{"ftp://faces.internal", "has to look like https://host:port"},
		{"https://", "has to look like https://host:port"},
		{"https:///v1", "has to look like https://host:port"},
		{"https://faces.internal:port", "cannot be read"},
		{"https://faces:hunter2hunter2@faces.internal", "pass the API key with WithKey"},
		{"https://hunter2hunter2@faces.internal:port", "cannot be read"},
		{"https://faces.internal?key=hunter2hunter2", "query string"},
		{"https://faces.internal/?", "query string"},
		{"https://faces.internal/#v1", "fragment"},
	} {
		c, err := New(tt.give)
		if err == nil {
			t.Errorf("%q: made %v", tt.give, c)
			continue
		}
		if !strings.HasPrefix(err.Error(), "client: ") || !strings.Contains(err.Error(), tt.want) {
			t.Errorf("%q: %v, want an error with %q", tt.give, err, tt.want)
		}
		// whatever was in the URL may have been a secret
		if strings.Contains(err.Error(), "hunter2") {
			t.Errorf("%q: the error repeats the URL: %v", tt.give, err)
		}
	}

	for _, tt := range []struct {
		name   string
		option Option
		want   string
	}{
		{"an empty key", WithKey(""), "the API key is empty"},
		{"a key with a space", WithKey("secret 0123456789abcdef"), "outside visible ASCII"},
		{"a key with a line break", WithKey("secret-0123456789abcdef\n"), "outside visible ASCII"},
		{"a key in another script", WithKey("калит-0123456789abcdef"), "outside visible ASCII"},
		{"no http.Client", WithHTTPClient(nil), "needs an http.Client"},
		{"negative retries", WithRetries(-1), "cannot be negative"},
	} {
		_, err := New("http://localhost:8080", tt.option)
		if err == nil || !strings.Contains(err.Error(), tt.want) {
			t.Errorf("%s: %v, want an error with %q", tt.name, err, tt.want)
			continue
		}
		if strings.Contains(err.Error(), "0123456789") || strings.Contains(err.Error(), "калит") {
			t.Errorf("%s: the error repeats the key: %v", tt.name, err)
		}
	}
}

func TestDefaults(t *testing.T) {
	c := newClient(t, "http://localhost:8080")
	if c.retries != 2 || c.page != 1000 {
		t.Errorf("%d retries, pages of %d", c.retries, c.page)
	}
	if c.http.Timeout != 30*time.Second {
		t.Errorf("a call may take %v", c.http.Timeout)
	}
	tr, ok := c.http.Transport.(*http.Transport)
	if !ok {
		t.Fatalf("transport %T", c.http.Transport)
	}
	if tr.TLSHandshakeTimeout <= 0 || tr.ResponseHeaderTimeout <= 0 || tr.IdleConnTimeout <= 0 || tr.IdleConnTimeout >= 2*time.Minute || tr.Proxy == nil {
		t.Errorf("transport %+v", tr)
	}
	// two clients do not share connections
	if other := newClient(t, "http://localhost:8080"); other.http == c.http || other.http.Transport == c.http.Transport {
		t.Error("two clients share an http.Client")
	}

	// the caller's http.Client is used as it is
	own := &http.Client{Transport: failing{}}
	c = newClient(t, "http://localhost:8080", WithHTTPClient(own), WithRetries(0))
	if c.http != own {
		t.Error("WithHTTPClient was ignored")
	}
	if _, err := c.Info(ctx); err == nil || !strings.Contains(err.Error(), "no route to host") {
		t.Errorf("a call through the caller's transport: %v", err)
	}
}

func TestClientDoesNotShowItsKey(t *testing.T) {
	c := newClient(t, "https://faces.internal:8443/api/", WithKey(adminKey))
	for _, verb := range []string{"%v", "%+v", "%#v", "%s", "%q"} {
		for _, shown := range []string{
			fmt.Sprintf(verb, c), fmt.Sprintf(verb, *c), fmt.Sprintf(verb, []any{c}), fmt.Sprintf(verb, struct{ C *Client }{c}),
		} {
			if strings.Contains(shown, adminKey) || strings.Contains(shown, "0123456789") {
				t.Errorf("%s shows the key: %s", verb, shown)
			}
		}
	}
	if got, want := c.String(), "likeness client for https://faces.internal:8443/api, with an API key"; got != want {
		t.Errorf("String() = %q, want %q", got, want)
	}
	if got := fmt.Sprintf("%#v", c); got != c.String() {
		t.Errorf("%%#v prints %q", got)
	}
	bare := newClient(t, "http://localhost:8080")
	if got, want := fmt.Sprint(bare), "likeness client for http://localhost:8080, without an API key"; got != want {
		t.Errorf("without a key: %q, want %q", got, want)
	}
}

func TestWhatIsSent(t *testing.T) {
	picture := []byte("\xff\xd8\xff\xe0 a picture, with bytes \x00 of every kind \r\n--")
	tests := []struct {
		name   string
		call   func(c *Client)
		method string
		target string
		// picture says whether the request carries the picture as its body.
		picture bool
	}{
		{"a plain search", func(c *Client) { c.Search(ctx, picture, SearchOptions{}) }, "POST", "/v1/search", true},
		{"a search with every option", func(c *Client) {
			c.Search(ctx, picture, SearchOptions{Limit: 3, Threshold: 0.85, AllFaces: true, Candidates: true})
		}, "POST", "/v1/search?candidates=true&faces=all&limit=3&threshold=0.85", true},
		{"a search with a limit", func(c *Client) { c.Search(ctx, picture, SearchOptions{Limit: 1}) }, "POST", "/v1/search?limit=1", true},
		{"a plain enrolment", func(c *Client) { c.Enrol(ctx, "anna", picture, EnrolOptions{}) }, "POST", "/v1/subjects/anna/faces", true},
		{"an enrolment with every option", func(c *Client) {
			c.Enrol(ctx, "anna", picture, EnrolOptions{Replace: true, AllowDuplicate: true, AllowMismatch: true})
		}, "POST", "/v1/subjects/anna/faces?allow_duplicate=true&allow_mismatch=true&replace=true", true},
		{"a subject", func(c *Client) { c.Subject(ctx, "anna") }, "GET", "/v1/subjects/anna", false},
		{"the first page", func(c *Client) { c.Subjects(ctx, "", 0) }, "GET", "/v1/subjects", false},
		{"a later page", func(c *Client) { c.Subjects(ctx, "a&b c", 25) }, "GET", "/v1/subjects?after=a%26b+c&limit=25", false},
		{"removing a subject", func(c *Client) { c.RemoveSubject(ctx, "anna") }, "DELETE", "/v1/subjects/anna", false},
		{"removing a face", func(c *Client) { c.RemoveFace(ctx, "anna", 7) }, "DELETE", "/v1/subjects/anna/faces/7", false},
		{"info", func(c *Client) { c.Info(ctx) }, "GET", "/v1/info", false},
		{"ready", func(c *Client) { c.Ready(ctx) }, "GET", "/readyz", false},
		{"alive", func(c *Client) { c.Alive(ctx) }, "GET", "/healthz", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := play(t, ok(`{}`))
			tt.call(newClient(t, s.url, WithKey(adminKey)))
			got := s.requests()
			if len(got) != 1 {
				t.Fatalf("%d requests", len(got))
			}
			rq := got[0]
			if rq.method != tt.method || rq.target != tt.target {
				t.Errorf("sent %s %s, want %s %s", rq.method, rq.target, tt.method, tt.target)
			}
			if rq.header.Get("Authorization") != "Bearer "+adminKey || rq.header.Get("Accept") != "application/json" ||
				rq.header.Get("User-Agent") != "likeness-client-go" {
				t.Errorf("headers %v", rq.header)
			}
			if tt.picture {
				if !bytes.Equal(rq.body, picture) || rq.header.Get("Content-Type") != "application/octet-stream" ||
					rq.header.Get("Content-Length") != strconv.Itoa(len(picture)) {
					t.Errorf("the picture arrived as %d bytes of %q", len(rq.body), rq.header.Get("Content-Type"))
				}
			} else if len(rq.body) != 0 || rq.header.Get("Content-Type") != "" {
				t.Errorf("a body of %d bytes of %q", len(rq.body), rq.header.Get("Content-Type"))
			}
		})
	}

	// without a key no Authorization header is made up
	s := play(t, ok(infoAnswer))
	newClient(t, s.url).Info(ctx)
	if got, sent := s.requests()[0].header["Authorization"]; sent {
		t.Errorf("a client without a key sent Authorization: %q", got)
	}
}

func TestVerifyKeepsTheSubjectOutOfTheURL(t *testing.T) {
	const subject = "30112199012345"
	picture := []byte("\xff\xd8\xff\xe0 a picture, with bytes \x00 of every kind \r\n--")
	for _, tt := range []struct {
		options VerifyOptions
		fields  map[string]string
	}{
		{VerifyOptions{}, map[string]string{"subject": subject}},
		{VerifyOptions{Threshold: 0.625}, map[string]string{"subject": subject, "threshold": "0.625"}},
	} {
		s := play(t, busy("1"), ok(verifyAnswer))
		c, _ := impatient(t, s.url, WithKey(searchKey))
		if _, err := c.Verify(ctx, subject, picture, tt.options); err != nil {
			t.Fatal(err)
		}
		got := s.requests()
		if len(got) != 2 {
			t.Fatalf("%d requests", len(got))
		}
		// the second attempt carries the whole form again
		for i, rq := range got {
			if rq.method != http.MethodPost || rq.target != "/v1/verify" {
				t.Errorf("attempt %d: sent %s %s, want POST /v1/verify and nothing more", i+1, rq.method, rq.target)
			}
			for name, values := range rq.header {
				if strings.Contains(strings.Join(values, " "), subject) {
					t.Errorf("attempt %d: the subject is in the %s header", i+1, name)
				}
			}
			if rq.header.Get("Content-Length") != strconv.Itoa(len(rq.body)) {
				t.Errorf("attempt %d: Content-Length %s for a body of %d bytes", i+1, rq.header.Get("Content-Length"), len(rq.body))
			}
			fields, sent := readForm(t, rq)
			if len(fields) != len(tt.fields) {
				t.Errorf("attempt %d: fields %v, want %v", i+1, fields, tt.fields)
			}
			for name, want := range tt.fields {
				if fields[name] != want {
					t.Errorf("attempt %d: field %s is %q, want %q", i+1, name, fields[name], want)
				}
			}
			if !bytes.Equal(sent, picture) {
				t.Errorf("attempt %d: the picture arrived as %q", i+1, sent)
			}
		}
	}
}

// readForm takes a multipart request apart: its text fields and the
// content of its "image" field.
func readForm(t *testing.T, rq sent) (fields map[string]string, picture []byte) {
	t.Helper()
	kind, params, err := mime.ParseMediaType(rq.header.Get("Content-Type"))
	if err != nil || kind != "multipart/form-data" {
		t.Fatalf("Content-Type %q: %v", rq.header.Get("Content-Type"), err)
	}
	fields = make(map[string]string)
	form := multipart.NewReader(bytes.NewReader(rq.body), params["boundary"])
	for {
		part, err := form.NextPart()
		if errors.Is(err, io.EOF) {
			return fields, picture
		}
		if err != nil {
			t.Fatalf("reading the form: %v", err)
		}
		content, err := io.ReadAll(part)
		if err != nil {
			t.Fatalf("reading the %s field: %v", part.FormName(), err)
		}
		if part.FormName() == "image" {
			if picture != nil || part.FileName() == "" || part.Header.Get("Content-Type") != "application/octet-stream" {
				t.Errorf("the image field: file name %q, type %q", part.FileName(), part.Header.Get("Content-Type"))
			}
			picture = content
			continue
		}
		if _, twice := fields[part.FormName()]; twice || part.FileName() != "" {
			t.Errorf("the %s field is sent twice or as a file", part.FormName())
		}
		fields[part.FormName()] = string(content)
	}
}

func TestDefaultClientFollowsNoRedirects(t *testing.T) {
	s := play(t, reply(http.StatusTemporaryRedirect, "", "Location", "/elsewhere/v1/search"))
	_, err := newClient(t, s.url, WithKey(searchKey)).Search(ctx, []byte("face:anna"), SearchOptions{})
	refused(t, err, http.StatusTemporaryRedirect, "")
	s.count(1)
}

// The client must stay light: its own files may use the standard library
// and nothing else, least of all the engine, which needs cgo.
func TestImportsOnlyTheStandardLibrary(t *testing.T) {
	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	checked := 0
	for _, file := range files {
		if strings.HasSuffix(file, "_test.go") {
			continue
		}
		source, err := os.ReadFile(file)
		if err != nil {
			t.Fatal(err)
		}
		parsed, err := parser.ParseFile(token.NewFileSet(), file, source, parser.ImportsOnly)
		if err != nil {
			t.Fatal(err)
		}
		checked++
		for _, imp := range parsed.Imports {
			path, err := strconv.Unquote(imp.Path.Value)
			if err != nil {
				t.Fatalf("%s: import %s: %v", file, imp.Path.Value, err)
			}
			// A package of the standard library has no dot in the first
			// element of its path; everything else has a host name there.
			first, _, _ := strings.Cut(path, "/")
			if strings.Contains(first, ".") || path == "C" {
				t.Errorf("%s imports %s; the client may only use the standard library", file, path)
			}
		}
	}
	if checked == 0 {
		t.Error("found no source files to check")
	}
}
