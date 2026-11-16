// Package client talks to a likeness server over its HTTP API: it enrols
// people from pictures, asks who is in a picture (search) and whether a
// picture shows a given person (verify).
//
//	c, err := client.New("https://faces.internal:8080", client.WithKey(key))
//	if err != nil {
//		return err
//	}
//	res, err := c.Search(ctx, jpegBytes, client.SearchOptions{})
//	if err != nil {
//		return err
//	}
//	if m := res.Best(); m != nil {
//		// m.Subject is in the picture, with similarity m.Score
//	}
//
// A Client is safe for concurrent use: make one and share it. The package
// needs the standard library and nothing else, so importing it does not
// bring in the engine.
//
// A likeness server analyses only a few pictures at a time and refuses the
// rest at once with 429 rather than serving them late. Nothing was done
// with a refused request, so the client sends it again, up to twice unless
// WithRetries says otherwise. Before each attempt it waits as long as the
// server asks in Retry-After, though never more than two seconds, and up to
// a quarter longer so that callers refused together do not come back
// together. If the server is still busy after the last attempt, or the
// context ends before the wait does, the call fails with an *Error for
// which IsOverloaded holds.
//
// A call whose connection breaks before the answer arrives is repeated in
// the same way where a repeat does no harm: searches, verifications, reads
// and removals. An enrolment is sent once, unless it replaces the subject's
// templates: the server may already have enrolled the face, and a second
// attempt would add it again.
//
// Whatever the server refuses comes back as an *Error with the HTTP status,
// the API's error code and the id of the request, under which the server's
// log has the details. Results carry that id too, and WithRequestID lets a
// caller choose it.
package client

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"strings"
	"time"
)

const (
	// defaultRetries is how often a call is sent again unless WithRetries
	// says otherwise.
	defaultRetries = 2
	// defaultTimeout bounds a call made with the client's own http.Client,
	// from connecting to the last byte of the answer. An analysis takes
	// tens of milliseconds; the rest is room for a large upload on a slow
	// line.
	defaultTimeout = 30 * time.Second
	// listPage is the page size AllSubjects asks for: the largest the API
	// allows.
	listPage = 1000

	userAgent = "likeness-client-go"
)

// Client calls one likeness server. Make one with New.
type Client struct {
	// base is the scheme, the host and the path prefix, without a slash at
	// the end.
	base    string
	http    *http.Client
	retries int
	page    int

	// The key is held by these two functions and nowhere else, so that no
	// way of printing a Client can show it.
	keyed bool
	sign  func(http.Header)
	scrub func(string) string

	// sleep waits between two attempts and jitter spreads the waits; tests
	// replace them.
	sleep  func(ctx context.Context, d time.Duration) error
	jitter func(d time.Duration) time.Duration
}

// Option adjusts a Client under construction.
type Option func(*Client) error

// New makes a client for the server at baseURL, such as
// "https://faces.internal:8080". The URL may carry the path under which a
// reverse proxy mounts the API, with or without a slash at the end.
func New(baseURL string, options ...Option) (*Client, error) {
	u, err := url.Parse(baseURL)
	if err != nil {
		// The cause without the URL: a mistyped one may hold a password.
		if parse, ok := errors.AsType[*url.Error](err); ok {
			err = parse.Err
		}
		return nil, fmt.Errorf("client: the base URL cannot be read: %w", err)
	}
	switch {
	case u.Scheme != "http" && u.Scheme != "https", u.Host == "":
		return nil, errors.New("client: the base URL has to look like https://host:port, with the path of the API after it if there is one")
	case u.User != nil:
		return nil, errors.New("client: the base URL must not carry a user or a password; pass the API key with WithKey")
	case u.RawQuery != "" || u.ForceQuery || u.Fragment != "":
		return nil, errors.New("client: the base URL cannot have a query string or a fragment")
	}
	c := &Client{
		base:    u.Scheme + "://" + u.Host + strings.TrimRight(u.EscapedPath(), "/"),
		retries: defaultRetries,
		page:    listPage,
		sign:    func(http.Header) {},
		scrub:   func(text string) string { return text },
		sleep:   sleep,
		jitter:  jitter,
	}
	for _, option := range options {
		if err := option(c); err != nil {
			return nil, err
		}
	}
	if c.http == nil {
		c.http = defaultHTTPClient()
	}
	return c, nil
}

// WithKey sets the API key sent with every request. A search key is enough
// to search, verify and read the info; everything else needs an admin key.
// The client never logs the key and keeps it out of its errors.
func WithKey(key string) Option {
	return func(c *Client) error {
		if key == "" {
			return errors.New("client: the API key is empty")
		}
		for i := 0; i < len(key); i++ {
			if b := key[i]; b <= ' ' || b > '~' {
				return errors.New("client: the API key contains a space or a character outside visible ASCII")
			}
		}
		hide := strings.NewReplacer(key, "[key]")
		c.keyed = true
		c.sign = func(h http.Header) { h.Set("Authorization", "Bearer "+key) }
		c.scrub = hide.Replace
		return nil
	}
}

// WithHTTPClient replaces the http.Client the requests are sent with, for
// other timeouts, a proxy or TLS settings of your own. The client the
// package makes for itself follows no redirects, because a redirected
// upload arrives without its picture; one passed here keeps its own policy.
func WithHTTPClient(hc *http.Client) Option {
	return func(c *Client) error {
		if hc == nil {
			return errors.New("client: WithHTTPClient needs an http.Client")
		}
		c.http = hc
		return nil
	}
}

// WithRetries sets how many times a call is sent again after the server
// refused it as overloaded or the connection broke: 2 by default, 0 for
// never.
func WithRetries(n int) Option {
	return func(c *Client) error {
		if n < 0 {
			return errors.New("client: the number of retries cannot be negative")
		}
		c.retries = n
		return nil
	}
}

// defaultHTTPClient is the standard library's default with shorter
// patience, more idle connections to the one host it talks to, and no
// following of redirects.
func defaultHTTPClient() *http.Client {
	return &http.Client{
		Timeout: defaultTimeout,
		Transport: &http.Transport{
			Proxy:                 http.ProxyFromEnvironment,
			DialContext:           (&net.Dialer{Timeout: 5 * time.Second, KeepAlive: 30 * time.Second}).DialContext,
			ForceAttemptHTTP2:     true,
			TLSHandshakeTimeout:   5 * time.Second,
			ResponseHeaderTimeout: defaultTimeout,
			ExpectContinueTimeout: time.Second,
			MaxIdleConns:          64,
			MaxIdleConnsPerHost:   16,
			// Shorter than the two minutes after which the server drops an
			// idle connection, so the client is not caught writing to one.
			IdleConnTimeout: 90 * time.Second,
		},
		CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
	}
}

// String names the server the client calls and says whether it has a key,
// never the key.
func (c Client) String() string {
	if c.keyed {
		return "likeness client for " + c.base + ", with an API key"
	}
	return "likeness client for " + c.base + ", without an API key"
}

// GoString is String: %#v must not print the fields either.
func (c Client) GoString() string { return c.String() }

type requestIDKey struct{}

// WithRequestID returns a context whose calls carry id in X-Request-Id, so
// that a request can be followed from your service into the server's log.
// The server takes up to 64 letters, digits, '-', '_' or '.'; an id it would
// not take is not sent, and the server makes one up. Either way the result
// or the error of the call has the id the server used.
func WithRequestID(ctx context.Context, id string) context.Context {
	return context.WithValue(ctx, requestIDKey{}, id)
}

// requestID is the id the context asks for, if the server would take it.
func requestID(ctx context.Context) (string, bool) {
	id, _ := ctx.Value(requestIDKey{}).(string)
	if id == "" || len(id) > 64 {
		return "", false
	}
	for i := 0; i < len(id); i++ {
		b := id[i]
		letter := b >= 'a' && b <= 'z' || b >= 'A' && b <= 'Z' || b >= '0' && b <= '9'
		if !letter && b != '-' && b != '_' && b != '.' {
			return "", false
		}
	}
	return id, true
}
