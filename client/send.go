package client

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand/v2"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"time"
)

const (
	// maxWait caps the pause before another attempt, whatever Retry-After
	// asks for: a caller with a person at the camera is better served by a
	// refusal than by a long wait.
	maxWait = 2 * time.Second
	// firstWait is the pause before the second attempt when the server
	// named none; it doubles with every further one.
	firstWait = 100 * time.Millisecond
	// maxAnswer bounds what is read of an answer, and maxRefusal of one that
	// is not a success: no answer of the API comes near either.
	maxAnswer  = 16 << 20
	maxRefusal = 64 << 10
)

// call is one request to the API, described so that it can be sent again.
type call struct {
	method string
	// path is where the request goes below the base URL, already escaped.
	path  string
	query url.Values
	// kind is the Content-Type of the body, body makes a reader over it for
	// one attempt and size is its length. A call without a body has none of
	// them.
	kind string
	body func() io.Reader
	size int64
	// repeatable says that sending the call twice does no harm, so it may
	// be sent again when it is not known whether the server received it.
	repeatable bool
	// once marks a probe, which is never sent again: an answer that takes
	// several attempts is not the one a probe is after.
	once bool
}

// do sends a call until it is answered or not worth sending again, decodes
// the answer into the value into points to, and returns the id the server
// gave the request. A failure carries its request id itself.
func (c *Client) do(ctx context.Context, rq *call, into any) (string, error) {
	retries := c.retries
	if rq.once {
		retries = 0
	}
	for attempt := 0; ; attempt++ {
		req, err := c.request(ctx, rq)
		if err != nil {
			return "", err
		}
		id, answered, err := c.send(req, into)
		if err == nil {
			return id, nil
		}
		wait, again := pause(ctx, rq, attempt, answered, err)
		if !again || attempt >= retries {
			return "", err
		}
		wait = c.jitter(wait)
		if deadline, ok := ctx.Deadline(); ok && time.Until(deadline) < wait {
			return "", err // the caller would be gone before the next attempt
		}
		if gone := c.sleep(ctx, wait); gone != nil {
			return "", fmt.Errorf("client: %w while waiting to try again after: %w", gone, err)
		}
	}
}

// pause decides whether a failed attempt is worth another, and how long to
// wait before it.
func pause(ctx context.Context, rq *call, attempt int, answered bool, err error) (time.Duration, bool) {
	backoff := min(firstWait<<min(attempt, 8), maxWait)
	if answered {
		// Only a refusal for overload is known to have done nothing.
		e, ok := errors.AsType[*Error](err)
		if !ok || e.Status != http.StatusTooManyRequests {
			return 0, false
		}
		if e.RetryAfter > 0 {
			return min(e.RetryAfter, maxWait), true
		}
		return backoff, true
	}
	// No answer. The caller giving up or a timeout are not reasons to try
	// again: a server that is slow to answer one request is not helped by
	// a second.
	if !rq.repeatable || ctx.Err() != nil || errors.Is(err, context.DeadlineExceeded) {
		return 0, false
	}
	if slow, ok := errors.AsType[net.Error](err); ok && slow.Timeout() {
		return 0, false
	}
	return backoff, true
}

// request builds the HTTP request of one attempt.
func (c *Client) request(ctx context.Context, rq *call) (*http.Request, error) {
	target := c.base + rq.path
	if len(rq.query) > 0 {
		target += "?" + rq.query.Encode()
	}
	var body io.Reader
	if rq.body != nil {
		body = rq.body()
	}
	req, err := http.NewRequestWithContext(ctx, rq.method, target, body)
	if err != nil {
		return nil, fmt.Errorf("client: %w", err)
	}
	if rq.body != nil {
		req.ContentLength = rq.size
		// Lets the transport start over on a connection that turned out to
		// be closed before anything was written to it.
		req.GetBody = func() (io.ReadCloser, error) { return io.NopCloser(rq.body()), nil }
		req.Header.Set("Content-Type", rq.kind)
	}
	req.Header.Set("Accept", "application/json")
	req.Header.Set("User-Agent", userAgent)
	if asked, ok := requestID(ctx); ok {
		req.Header.Set("X-Request-Id", asked)
	}
	c.sign(req.Header)
	return req, nil
}

// send makes one attempt. answered says whether the server answered at
// all; if it did not, nobody knows whether it acted on the request.
func (c *Client) send(req *http.Request, into any) (id string, answered bool, err error) {
	res, err := c.http.Do(req)
	if err != nil {
		return "", false, fmt.Errorf("client: %w", err)
	}
	defer res.Body.Close()

	if res.StatusCode < 200 || res.StatusCode > 299 {
		// A refusal that breaks off half way is still a refusal.
		text, _ := io.ReadAll(io.LimitReader(res.Body, maxRefusal))
		return "", true, c.refusal(res, text)
	}
	text, err := io.ReadAll(io.LimitReader(res.Body, maxAnswer+1))
	if err != nil {
		return "", false, fmt.Errorf("client: the answer broke off: %w", err)
	}
	if len(text) > maxAnswer {
		return "", true, fmt.Errorf("client: the answer is larger than %d bytes", maxAnswer)
	}
	if into != nil {
		if err := json.Unmarshal(text, into); err != nil {
			return "", true, fmt.Errorf("client: the answer (status %d) is not the JSON the API describes: %w", res.StatusCode, err)
		}
	}
	return res.Header.Get("X-Request-Id"), true, nil
}

// retryAfter reads a Retry-After header: a number of seconds or a date.
// Zero means that it says nothing usable.
func retryAfter(value string) time.Duration {
	if value == "" {
		return 0
	}
	if seconds, err := strconv.Atoi(value); err == nil {
		return max(time.Duration(seconds)*time.Second, 0)
	}
	if at, err := http.ParseTime(value); err == nil {
		return max(time.Until(at), 0)
	}
	return 0
}

// sleep waits for d, or for less if the context ends first.
func sleep(ctx context.Context, d time.Duration) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-timer.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// jitter lengthens a wait by up to a quarter, so that callers refused at
// the same moment do not all return at the same moment.
func jitter(d time.Duration) time.Duration {
	if d <= 0 {
		return 0
	}
	return d + rand.N(d/4+1)
}
